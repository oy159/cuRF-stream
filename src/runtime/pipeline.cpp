#include "curf/cuda/pipeline.hpp"

#include "curf/cuda/memory.hpp"
#include "curf/cuda/psd.hpp"
#include "curf/runtime/host_ring.hpp"

#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace curf::cuda {
    namespace {

        using Clock = std::chrono::steady_clock;

        struct NvtxRange {
            explicit NvtxRange(const char *name) { nvtxRangePushA(name); }
            ~NvtxRange() { nvtxRangePop(); }
        };

        Percentiles percentiles(std::vector<double> v) {
            Percentiles p;
            if (v.empty()) return p;
            std::sort(v.begin(), v.end());
            auto at = [&](double q) { return v[std::min(v.size() - 1, static_cast<std::size_t>(q * static_cast<double>(v.size())))]; };
            p.p50 = at(0.5);
            p.p99 = at(0.99);
            p.p999 = at(0.999);
            p.max = v.back();
            return p;
        }

        // Per in-flight block GPU resources.
        struct Context {
            cudaStream_t stream = nullptr;
            cudaEvent_t start = nullptr, h2d_done = nullptr, done = nullptr;
            DeviceBuffer<std::uint8_t> wire;
            DeviceBuffer<cf32> planar, scratch;
            DeviceBuffer<float> psd;
            PinnedBuffer<float> h_psd;

            Context(std::size_t wire_bytes, std::size_t samples, std::size_t psd_elems)
                : wire(wire_bytes), planar(samples), scratch(samples), psd(psd_elems), h_psd(psd_elems) {
                CURF_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                CURF_CUDA_CHECK(cudaEventCreate(&start));
                CURF_CUDA_CHECK(cudaEventCreate(&h2d_done));
                CURF_CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventBlockingSync));
            }
            ~Context() {
                cudaEventDestroy(start);
                cudaEventDestroy(h2d_done);
                cudaEventDestroy(done);
                cudaStreamDestroy(stream);
            }
            Context(const Context &) = delete;
            Context &operator=(const Context &) = delete;
        };

        float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
            float ms = 0;
            CURF_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
            return ms;
        }

    } // namespace

    std::string PipelineStats::to_json() const {
        char buf[1024];
        std::snprintf(buf, sizeof buf,
                      "{\"blocks\":%llu,\"frames\":%llu,\"dropped_frames\":%llu,\"gap_frames\":%llu,\"wall_s\":%.3f,"
                      "\"wire_GBps\":%.3f,\"msps_per_channel\":%.1f,\"x_realtime\":%.2f,"
                      "\"latency_ms\":{\"p50\":%.3f,\"p99\":%.3f,\"p999\":%.3f,\"max\":%.3f},"
                      "\"h2d_ms\":{\"p50\":%.3f,\"p99\":%.3f},\"gpu_ms\":{\"p50\":%.3f,\"p99\":%.3f},"
                      "\"ring_huge_coverage\":%.3f,\"ring_registered\":%s}",
                      static_cast<unsigned long long>(blocks), static_cast<unsigned long long>(frames),
                      static_cast<unsigned long long>(dropped_frames), static_cast<unsigned long long>(gap_frames),
                      wall_s, wire_GBps, msps_per_channel, x_realtime, latency_ms.p50, latency_ms.p99, latency_ms.p999,
                      latency_ms.max, h2d_ms.p50, h2d_ms.p99, gpu_ms.p50, gpu_ms.p99, ring_huge_coverage,
                      ring_registered ? "true" : "false");
        return buf;
    }

    PipelineStats run_pipeline(ISource &source, const PipelineConfig &cfg, const PsdCallback &on_psd) {
        const StreamFormat fmt = source.format();
        const std::size_t fb = fmt.frame_bytes();
        const std::size_t block_bytes = cfg.block_frames * fb;
        const std::size_t samples = cfg.block_frames * fmt.channels;
        if (cfg.depth == 0 || cfg.ring_slots < cfg.depth) throw std::invalid_argument("pipeline: need 0 < depth <= ring_slots");

        HostRing ring(cfg.ring_slots, block_bytes);
        PipelineStats st;
        const auto hp = ring.huge_pages();
        st.ring_huge_coverage = hp.known ? hp.coverage() : -1.0;
        if (cfg.register_ring) {
            CURF_CUDA_CHECK(cudaHostRegister(ring.base(), ring.bytes(), cudaHostRegisterDefault));
            st.ring_registered = true;
        }

        PsdStage psd(fmt.channels, cfg.block_frames, cfg.nfft);
        std::vector<std::unique_ptr<Context>> ctx;
        for (std::size_t i = 0; i < cfg.depth; ++i)
            ctx.push_back(std::make_unique<Context>(block_bytes, samples, psd.psd_elems()));

        // ------------------------------------------------------------ producer
        std::atomic<bool> stop{false}, producer_done{false};
        std::atomic<std::uint64_t> dropped{0};
        std::exception_ptr producer_error;
        std::thread producer([&] {
            std::vector<std::uint8_t> trash(cfg.drop_when_full ? block_bytes : 0);
            try {
                source.start();
                while (!stop.load(std::memory_order_relaxed)) {
                    HostSlot *slot = cfg.drop_when_full ? ring.try_begin_write()
                                                        : ring.begin_write(std::chrono::milliseconds(10));
                    if (!slot) {
                        if (!cfg.drop_when_full) continue; // back-pressure: wait for the GPU
                        // Ring full: discard input, as a hardware FIFO overflow would.
                        const auto r = source.read(trash, 100);
                        if (r.status == ReadStatus::end_of_stream) break;
                        dropped.fetch_add(r.frames, std::memory_order_relaxed);
                        continue;
                    }
                    NvtxRange range("fill slot");
                    std::size_t got = 0;
                    bool first = true, eos = false;
                    while (got < cfg.block_frames && !stop.load(std::memory_order_relaxed)) {
                        const auto r = source.read({slot->data + got * fb, (cfg.block_frames - got) * fb}, 100);
                        if (r.status == ReadStatus::end_of_stream) {
                            eos = true;
                            break;
                        }
                        if (r.status != ReadStatus::ok) continue;
                        if (first) {
                            slot->first_sample = r.first_sample;
                            slot->discontinuity = r.discontinuity;
                            first = false;
                        } else if (r.discontinuity) {
                            slot->discontinuity = true;
                        }
                        got += r.frames;
                    }
                    if (got < cfg.block_frames) { // EOS or stop mid-block: drop the partial block
                        if (eos) break;
                        continue;
                    }
                    slot->frames = got;
                    ring.commit_write();
                }
                source.stop();
            } catch (...) {
                producer_error = std::current_exception();
            }
            producer_done.store(true, std::memory_order_release);
            ring.notify_all();
        });

        // ------------------------------------------------------------ consumer
        std::vector<double> lat, h2d, gpu;
        std::uint64_t submitted = 0, retired = 0, expected_sample = 0;
        bool have_expected = false;
        const auto t0 = Clock::now();
        auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - t0).count(); };

        auto retire = [&] {
            Context &c = *ctx[retired % cfg.depth];
            HostSlot &slot = ring.oldest_in_flight();
            NvtxRange range("retire");
            lat.push_back(std::chrono::duration<double, std::milli>(Clock::now() - slot.filled_at).count());
            h2d.push_back(elapsed_ms(c.start, c.h2d_done));
            gpu.push_back(elapsed_ms(c.start, c.done));
            if (have_expected && slot.first_sample != expected_sample)
                st.gap_frames += slot.first_sample - expected_sample;
            expected_sample = slot.first_sample + slot.frames;
            have_expected = true;
            st.frames += slot.frames;
            ++st.blocks;
            if (on_psd) on_psd(c.h_psd.get(), fmt.channels, cfg.nfft, slot.first_sample);
            ring.release_oldest();
            ++retired;
        };

        auto submit = [&](HostSlot &slot) {
            Context &c = *ctx[submitted % cfg.depth];
            NvtxRange range("submit");
            CURF_CUDA_CHECK(cudaEventRecord(c.start, c.stream));
            CURF_CUDA_CHECK(cudaMemcpyAsync(c.wire.get(), slot.data, slot.frames * fb, cudaMemcpyHostToDevice, c.stream));
            CURF_CUDA_CHECK(cudaEventRecord(c.h2d_done, c.stream));
            unpack_planar(c.wire.get(), fmt.wire, fmt.channels, slot.frames, c.planar.get(), c.stream, cfg.unpack);
            psd.run(c.planar.get(), c.scratch.get(), c.psd.get(), c.stream);
            CURF_CUDA_CHECK(cudaMemcpyAsync(c.h_psd.get(), c.psd.get(), c.psd.bytes(), cudaMemcpyDeviceToHost, c.stream));
            CURF_CUDA_CHECK(cudaEventRecord(c.done, c.stream));
            ring.mark_submitted();
            ++submitted;
        };

        std::exception_ptr error;
        try {
            for (;;) {
                if (!stop.load(std::memory_order_relaxed) &&
                    ((cfg.seconds > 0 && elapsed() >= cfg.seconds) || (cfg.max_blocks && submitted >= cfg.max_blocks))) {
                    stop.store(true);
                    ring.notify_all();
                }
                bool progressed = false;
                while (ring.in_flight() > 0) {
                    const cudaError_t q = cudaEventQuery(ctx[retired % cfg.depth]->done);
                    if (q == cudaErrorNotReady) break;
                    CURF_CUDA_CHECK(q);
                    retire();
                    progressed = true;
                }
                const bool may_submit = !(cfg.max_blocks && submitted >= cfg.max_blocks);
                if (may_submit && ring.in_flight() < cfg.depth) {
                    if (HostSlot *slot = ring.try_next_submit()) {
                        submit(*slot);
                        progressed = true;
                    }
                }
                if (progressed) continue;

                // Nothing retired or submitted this round.
                const bool producer_finished = producer_done.load(std::memory_order_acquire);
                const bool input_pending = ring.try_next_submit() != nullptr;
                if (ring.in_flight() == 0 && (!may_submit || (!input_pending && producer_finished))) break;

                if (ring.in_flight() >= cfg.depth || (ring.in_flight() > 0 && (!may_submit || producer_finished))) {
                    // GPU-bound (or draining): sleep until the oldest block completes.
                    CURF_CUDA_CHECK(cudaEventSynchronize(ctx[retired % cfg.depth]->done));
                } else if (ring.in_flight() > 0) {
                    ring.next_submit(std::chrono::microseconds(50)); // wait briefly for input, then poll the GPU again
                } else {
                    ring.next_submit(std::chrono::milliseconds(2));
                }
            }
        } catch (...) {
            error = std::current_exception();
        }
        st.wall_s = elapsed();

        stop.store(true);
        ring.notify_all();
        // Unblock a back-pressured producer: release everything still held.
        cudaDeviceSynchronize();
        while (ring.in_flight() > 0) ring.release_oldest();
        while (ring.try_next_submit()) {
            ring.mark_submitted();
            ring.release_oldest();
        }
        producer.join();
        if (cfg.register_ring) cudaHostUnregister(ring.base());
        if (error) std::rethrow_exception(error);
        if (producer_error) std::rethrow_exception(producer_error);

        st.dropped_frames = dropped.load();
        st.wire_GBps = static_cast<double>(st.frames * fb) / st.wall_s / 1e9;
        st.msps_per_channel = static_cast<double>(st.frames) / st.wall_s / 1e6;
        st.x_realtime = st.msps_per_channel * 1e6 / fmt.sample_rate;
        st.latency_ms = percentiles(std::move(lat));
        st.h2d_ms = percentiles(std::move(h2d));
        st.gpu_ms = percentiles(std::move(gpu));
        return st;
    }

} // namespace curf::cuda
