#include "curf/runtime/classifier.hpp"

#include "curf/cuda/channelizer.hpp"
#include "curf/cuda/memory.hpp"
#include "curf/runtime/host_ring.hpp"
#include "curf/runtime/producer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <thread>

namespace curf {

    namespace {

        using Clock = std::chrono::steady_clock;
        using cuda::DeviceBuffer;
        using cuda::PinnedBuffer;

        // One thread per classified window: argmax and its softmax probability.
        __global__ void postprocess_kernel(const float *__restrict__ logits, int classes,
                                           const std::uint32_t *__restrict__ active, const float *__restrict__ power,
                                           unsigned first, unsigned n, unsigned M, WindowResult *__restrict__ out) {
            const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;
            const float *l = logits + static_cast<std::size_t>(i) * classes;
            int best = 0;
            float mx = l[0];
            for (int c = 1; c < classes; ++c)
                if (l[c] > mx) { mx = l[c]; best = c; }
            float z = 0.f;
            for (int c = 0; c < classes; ++c) z += __expf(l[c] - mx);
            const std::uint32_t id = active[first + i];
            WindowResult r;
            r.stream = id / M;
            r.channel = id % M;
            r.cls = static_cast<std::uint32_t>(best);
            r.prob = 1.f / z;
            r.power = power[id];
            out[first + i] = r;
        }

        cuda::Percentiles percentiles(std::vector<double> v) {
            cuda::Percentiles p;
            if (v.empty()) return p;
            std::sort(v.begin(), v.end());
            auto at = [&](double q) { return v[std::min(v.size() - 1, static_cast<std::size_t>(q * static_cast<double>(v.size())))]; };
            p.p50 = at(0.5);
            p.p99 = at(0.99);
            p.p999 = at(0.999);
            p.max = v.back();
            return p;
        }

        struct Event {
            cudaEvent_t e = nullptr;
            explicit Event(unsigned flags = cudaEventDefault) { CURF_CUDA_CHECK(cudaEventCreateWithFlags(&e, flags)); }
            ~Event() { cudaEventDestroy(e); }
            Event(const Event &) = delete;
            Event &operator=(const Event &) = delete;
            bool ready() const {
                const cudaError_t q = cudaEventQuery(e);
                if (q == cudaErrorNotReady) return false;
                CURF_CUDA_CHECK(q);
                return true;
            }
        };

        float elapsed_ms(const Event &a, const Event &b) {
            float ms = 0;
            CURF_CUDA_CHECK(cudaEventElapsedTime(&ms, a.e, b.e));
            return ms;
        }

    } // namespace

    std::string ClassifierStats::to_json() const {
        std::string hist = "[";
        for (std::size_t i = 0; i < class_histogram.size(); ++i)
            hist += (i ? "," : "") + std::to_string(class_histogram[i]);
        hist += "]";
        char buf[1024];
        std::snprintf(buf, sizeof buf,
                      "{\"blocks\":%llu,\"windows\":%llu,\"classified\":%llu,\"max_active\":%u,\"dropped_frames\":%llu,"
                      "\"gap_frames\":%llu,\"wall_s\":%.3f,\"msps_per_stream\":%.1f,\"x_realtime\":%.2f,"
                      "\"windows_per_s\":%.0f,\"latency_ms\":{\"p50\":%.3f,\"p99\":%.3f,\"p999\":%.3f,\"max\":%.3f},"
                      "\"window_gpu_ms\":{\"p50\":%.3f,\"p99\":%.3f},\"channelize_ms_per_block\":%.4f,\"classes\":%s}",
                      static_cast<unsigned long long>(blocks), static_cast<unsigned long long>(windows),
                      static_cast<unsigned long long>(classified), max_active,
                      static_cast<unsigned long long>(dropped_frames), static_cast<unsigned long long>(gap_frames), wall_s,
                      msps_per_stream, x_realtime, windows_per_s, latency_ms.p50, latency_ms.p99, latency_ms.p999,
                      latency_ms.max, window_gpu_ms.p50, window_gpu_ms.p99, channelize_ms_per_block, hist.c_str());
        return buf;
    }

    ClassifierStats run_classifier(ISource &source, const engine::Model &model, const ClassifierConfig &cfg,
                                   const ResultCallback &on_results) {
        const StreamFormat fmt = source.format();
        const std::size_t fb = fmt.frame_bytes(), block_bytes = cfg.block_frames * fb;
        const std::uint32_t S = fmt.channels, M = cfg.channelizer.channels, W = cfg.window;
        const auto &in = model.tensors[model.input];
        if (in.length != W || in.channels != 2) throw std::invalid_argument("classifier: model input must be [W][2]");
        const int classes = static_cast<int>(model.tensors[model.output].channels);

        cuda::Channelizer ch(cfg.channelizer, S, cfg.block_frames);
        const std::size_t F = ch.frames();
        if (W % F != 0) throw std::invalid_argument("classifier: window must be a whole number of blocks' frames");
        const std::size_t bpw = W / F;
        if (cfg.depth == 0 || cfg.ring_slots < cfg.depth) throw std::invalid_argument("classifier: need 0 < depth <= ring_slots");

        engine::Engine eng(model, cfg.engine);
        const auto chunk = static_cast<std::uint32_t>(eng.max_batch());

        HostRing ring(cfg.ring_slots, block_bytes);
        CURF_CUDA_CHECK(cudaHostRegister(ring.base(), ring.bytes(), cudaHostRegisterDefault));

        cudaStream_t copy_s, comp_s;
        CURF_CUDA_CHECK(cudaStreamCreateWithFlags(&copy_s, cudaStreamNonBlocking));
        CURF_CUDA_CHECK(cudaStreamCreateWithFlags(&comp_s, cudaStreamNonBlocking));

        // per block in flight
        struct BlockCtx {
            DeviceBuffer<std::uint8_t> wire;
            Event h2d_done, c0, c1;
            explicit BlockCtx(std::size_t bytes) : wire(bytes) {}
        };
        std::vector<std::unique_ptr<BlockCtx>> blk;
        for (std::size_t i = 0; i < cfg.depth; ++i) blk.push_back(std::make_unique<BlockCtx>(block_bytes));

        // per window, ping-pong
        struct WinCtx {
            DeviceBuffer<cf32> buf;
            cuda::WindowStage stage;
            PinnedBuffer<std::uint32_t> count;
            DeviceBuffer<WindowResult> d_res;
            PinnedBuffer<WindowResult> h_res;
            // The host thread sleeps on count_ready / done (cudaEventSynchronize below): blocking-sync events
            // put it to sleep instead of spinning a core (default events busy-wait). On the laptop the CPU
            // and GPU share one cooling system, so a spinning core costs GPU clock under sustained load.
            Event start, count_ready{cudaEventBlockingSync}, done{cudaEventBlockingSync};
            Clock::time_point ready_at{};
            std::uint64_t first_sample = 0;
            std::uint32_t n = 0;
            WinCtx(std::uint32_t S_, std::uint32_t M_, std::uint32_t W_)
                : buf(std::size_t{S_} * W_ * M_), stage(S_, M_, W_), count(1), d_res(std::size_t{S_} * M_),
                  h_res(std::size_t{S_} * M_) {}
        };
        std::unique_ptr<WinCtx> wins[2] = {std::make_unique<WinCtx>(S, M, W), std::make_unique<WinCtx>(S, M, W)};
        auto win = [&](std::uint64_t i) -> WinCtx & { return *wins[i % 2]; };

        ProducerControl pc;
        std::thread producer = start_producer(source, ring, cfg.block_frames, fb, cfg.drop_when_full, pc);

        ClassifierStats st;
        st.class_histogram.assign(static_cast<std::size_t>(classes), 0);
        std::vector<double> lat, wgpu;
        double chan_ms = 0;
        std::uint64_t submitted = 0, retired = 0, detected = 0, gathered = 0, delivered = 0, expected = 0;
        bool have_expected = false;
        const auto t0 = Clock::now();
        auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - t0).count(); };

        std::exception_ptr error;
        try {
            for (;;) {
                if (!pc.stop.load() && ((cfg.seconds > 0 && elapsed() >= cfg.seconds) ||
                                        (cfg.max_windows && detected >= cfg.max_windows))) {
                    pc.stop.store(true);
                    ring.notify_all();
                }
                bool progressed = false;

                // 1. retire blocks whose channelizer pass is done: release the ring slot
                while (ring.in_flight() > 0 && blk[retired % cfg.depth]->c1.ready()) {
                    BlockCtx &b = *blk[retired % cfg.depth];
                    HostSlot &slot = ring.oldest_in_flight();
                    chan_ms += elapsed_ms(b.c0, b.c1);
                    if (have_expected && slot.first_sample != expected) st.gap_frames += slot.first_sample - expected;
                    expected = slot.first_sample + slot.frames;
                    have_expected = true;
                    st.frames += slot.frames;
                    ++st.blocks;
                    ring.release_oldest();
                    ++retired;
                    progressed = true;
                }

                // 2. a detected window whose active count has arrived: enqueue gather + engine + postprocess
                if (gathered < detected && win(gathered).count_ready.ready()) {
                    WinCtx &w = win(gathered);
                    const std::uint32_t n = w.count[0];
                    w.n = n;
                    for (std::uint32_t first = 0; first < n; first += chunk) {
                        const std::uint32_t c = std::min(chunk, n - first);
                        w.stage.gather(w.buf.get(), first, c, eng.input(), comp_s);
                        eng.run(static_cast<int>(c), comp_s);
                        postprocess_kernel<<<(c + 127) / 128, 128, 0, comp_s>>>(
                            eng.logits(), classes, w.stage.d_active(), w.stage.d_power(), first, c, M, w.d_res.get());
                        CURF_CUDA_CHECK(cudaGetLastError());
                    }
                    if (n)
                        CURF_CUDA_CHECK(cudaMemcpyAsync(w.h_res.get(), w.d_res.get(), n * sizeof(WindowResult),
                                                        cudaMemcpyDeviceToHost, comp_s));
                    CURF_CUDA_CHECK(cudaEventRecord(w.done.e, comp_s));
                    ++gathered;
                    progressed = true;
                }

                // 3. deliver finished windows in order
                if (delivered < gathered && win(delivered).done.ready()) {
                    WinCtx &w = win(delivered);
                    lat.push_back(std::chrono::duration<double, std::milli>(Clock::now() - w.ready_at).count());
                    wgpu.push_back(elapsed_ms(w.start, w.done));
                    for (std::uint32_t i = 0; i < w.n; ++i) ++st.class_histogram[w.h_res[i].cls];
                    if (on_results) on_results(delivered, w.first_sample, w.h_res.get(), w.n);
                    st.classified += w.n;
                    st.max_active = std::max(st.max_active, w.n);
                    ++st.windows;
                    ++delivered;
                    progressed = true;
                }

                // 4. submit the next block: H2D (copy stream), channelizer into its window buffer (compute stream);
                //    the last block of a window also enqueues detection and the count read-back.
                const std::uint64_t wi = submitted / bpw;
                const bool may_submit = !pc.stop.load() && !(cfg.max_windows && wi >= cfg.max_windows);
                if (may_submit && ring.in_flight() < cfg.depth && wi < delivered + 2) {
                    if (HostSlot *slot = ring.try_next_submit()) {
                        BlockCtx &b = *blk[submitted % cfg.depth];
                        WinCtx &w = win(wi);
                        const std::size_t part = submitted % bpw;
                        if (part == 0) w.first_sample = slot->first_sample;
                        CURF_CUDA_CHECK(cudaMemcpyAsync(b.wire.get(), slot->data, slot->frames * fb, cudaMemcpyHostToDevice, copy_s));
                        CURF_CUDA_CHECK(cudaEventRecord(b.h2d_done.e, copy_s));
                        CURF_CUDA_CHECK(cudaStreamWaitEvent(comp_s, b.h2d_done.e));
                        CURF_CUDA_CHECK(cudaEventRecord(b.c0.e, comp_s));
                        ch.run_wire(b.wire.get(), fmt.wire, w.buf.get() + part * F * M, comp_s, std::size_t{W} * M);
                        CURF_CUDA_CHECK(cudaEventRecord(b.c1.e, comp_s));
                        if (part == bpw - 1) {
                            w.ready_at = slot->filled_at;
                            CURF_CUDA_CHECK(cudaEventRecord(w.start.e, comp_s));
                            w.stage.detect(w.buf.get(), cfg.detect, comp_s);
                            CURF_CUDA_CHECK(cudaMemcpyAsync(w.count.get(), w.stage.d_count(), sizeof(std::uint32_t),
                                                            cudaMemcpyDeviceToHost, comp_s));
                            CURF_CUDA_CHECK(cudaEventRecord(w.count_ready.e, comp_s));
                            ++detected;
                        }
                        ring.mark_submitted();
                        ++submitted;
                        progressed = true;
                    }
                }
                if (progressed) continue;

                const bool outstanding = ring.in_flight() > 0 || delivered < detected;
                const bool producer_finished = pc.done.load(std::memory_order_acquire);
                if (!outstanding && (pc.stop.load() || (producer_finished && !ring.try_next_submit()))) break;
                if (outstanding) {
                    // Sleep on the GPU work that unblocks the oldest stage.
                    if (gathered < detected) CURF_CUDA_CHECK(cudaEventSynchronize(win(gathered).count_ready.e));
                    else if (delivered < gathered) CURF_CUDA_CHECK(cudaEventSynchronize(win(delivered).done.e));
                    else ring.next_submit(std::chrono::microseconds(50));
                } else {
                    ring.next_submit(std::chrono::milliseconds(2));
                }
            }
        } catch (...) {
            error = std::current_exception();
        }
        st.wall_s = elapsed();

        cudaDeviceSynchronize();
        stop_producer(ring, pc, producer);
        cudaHostUnregister(ring.base());
        cudaStreamDestroy(copy_s);
        cudaStreamDestroy(comp_s);
        if (error) std::rethrow_exception(error);
        if (pc.error) std::rethrow_exception(pc.error);

        st.dropped_frames = pc.dropped.load();
        st.msps_per_stream = static_cast<double>(st.frames) / st.wall_s / 1e6;
        st.x_realtime = st.msps_per_stream * 1e6 / fmt.sample_rate;
        st.windows_per_s = static_cast<double>(st.classified) / st.wall_s;
        st.latency_ms = percentiles(std::move(lat));
        st.window_gpu_ms = percentiles(std::move(wgpu));
        st.channelize_ms_per_block = st.blocks ? chan_ms / static_cast<double>(st.blocks) : 0;
        return st;
    }

} // namespace curf
