#include "curf/runtime/burst_classifier.hpp"
#include "curf/cuda/memory.hpp"
#include "curf/runtime/producer.hpp"

#include <cuda_fp16.h>
#include <cufft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace curf {
    namespace {
        using Clock = std::chrono::steady_clock;
        using cuda::DeviceBuffer;
        using cuda::PinnedBuffer;
        constexpr unsigned bins = burst_bins;
        constexpr unsigned frames = burst_block_frames;
        constexpr unsigned row_frames = burst_power_frames;
        constexpr unsigned rows = frames / row_frames;
        constexpr unsigned max_regions = 128;
        constexpr unsigned length = 1024;
        constexpr unsigned threads = 128;
        constexpr unsigned max_down = 48;
        constexpr float tau = 2 * std::numbers::pi_v<float>;

        struct Request {
            int center, down;
            float offset, cutoff; // fractions of input sample rate
        };

        void check_fft(cufftResult result) {
            if (result != CUFFT_SUCCESS) throw std::runtime_error("burst cuFFT failed: " + std::to_string(result));
        }

        struct ComputeStream {
            cudaStream_t stream{};
            ComputeStream() { CURF_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
            ~ComputeStream() { cudaStreamDestroy(stream); }
        };

        struct FftPlan {
            cufftHandle plan{};
            explicit FftPlan(cudaStream_t stream) {
                check_fft(cufftPlan1d(&plan, bins, CUFFT_C2C, frames / bins));
                const auto result = cufftSetStream(plan, stream);
                if (result != CUFFT_SUCCESS) { cufftDestroy(plan); check_fft(result); }
            }
            ~FftPlan() { cufftDestroy(plan); }
        };

        __device__ float sample12(unsigned value) {
            return static_cast<float>(static_cast<int>(value) - (value >= 2048 ? 4096 : 0)) / 2048.f;
        }

        __global__ void unpack_window(const std::uint8_t *bytes, float2 *raw, float2 *fft_input) {
            const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
            const auto *p = bytes + 3 * x;
            const float2 z = make_float2(sample12((p[0] << 4) | (p[1] >> 4)), sample12(((p[1] & 15) << 8) | p[2]));
            raw[x] = z;
            const float w = 0.5f - 0.5f * __cosf(tau * static_cast<float>(x % bins) / bins);
            fft_input[x] = make_float2(z.x * w, z.y * w);
        }

        __global__ void powers(const float2 *spectrum, float *power) {
            const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned k = (i + bins / 2) % bins;
            const unsigned base = 2 * (i / bins) * bins + k;
            const float2 a = spectrum[base], b = spectrum[base + bins];
            power[i] = (a.x * a.x + a.y * a.y + b.x * b.x + b.y * b.y) * (2.f / (bins * bins));
        }

        __global__ void update_floor(const float *p, float *floor, bool reset) {
            const unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
            float sum = 0;
            // The mean of the minima of four two-FFT averages approximates a
            // lower quantile without making one deep FFT null the noise floor.
            for (unsigned r = 0; r < rows; r += 4) {
                const float a = fminf(p[r * bins + k], p[(r + 1) * bins + k]);
                const float b = fminf(p[(r + 2) * bins + k], p[(r + 3) * bins + k]);
                sum += fminf(a, b);
            }
            const float value = fmaxf(sum / (rows / 4), 1e-15f);
            const float old = floor[k];
            floor[k] = reset ? value : old + (value - old) * (value < old ? .18f : .012f);
        }

        template <typename T>
        __global__ void extract(const float2 *raw, const Request *regions, T *dest) {
            const Request request = regions[blockIdx.x];
            const int h = 10 * request.down;
            __shared__ float2 taps[20 * max_down + 1];
            __shared__ float2 output[length];
            __shared__ float sum[threads];
            for (int j = threadIdx.x; j <= 2 * h; j += threads) {
                const int k = j - h;
                const float arg = tau * request.cutoff * k;
                const float f = k ? __sinf(arg) / (std::numbers::pi_v<float> * k) : 2 * request.cutoff;
                const float w = 0.42f + .5f * __cosf(tau * k / (2 * h)) + .08f * __cosf(tau * k / h);
                float s, c;
                __sincosf(-tau * request.offset * k, &s, &c);
                taps[j] = make_float2(w * f * c, w * f * s);
            }
            __syncthreads();
            float energy = 0;
            for (int j = threadIdx.x; j < length; j += threads) {
                const int pos = request.center + (j - static_cast<int>(length / 2)) * request.down;
                float i = 0, q = 0;
                for (int k = max(-h, -pos); k <= min(h, static_cast<int>(frames) - 1 - pos); ++k) {
                    const float2 z = raw[pos + k], w = taps[k + h];
                    i += z.x * w.x - z.y * w.y;
                    q += z.x * w.y + z.y * w.x;
                }
                float s, c;
                __sincosf(-tau * request.offset * (pos - request.center), &s, &c);
                output[j] = make_float2(c * i - s * q, s * i + c * q);
                energy += i * i + q * q;
            }
            sum[threadIdx.x] = energy;
            for (unsigned stride = threads / 2; stride; stride /= 2) {
                __syncthreads();
                if (threadIdx.x < stride) sum[threadIdx.x] += sum[threadIdx.x + stride];
            }
            __syncthreads();
            const float scale = rsqrtf(fmaxf(sum[0] / length, 1e-16f));
            for (unsigned j = threadIdx.x; j < length; j += threads) {
                const auto i = 2 * (blockIdx.x * length + j);
                dest[i] = static_cast<T>(output[j].x * scale);
                dest[i + 1] = static_cast<T>(output[j].y * scale);
            }
        }

        struct Span {
            int lo, hi;
            unsigned start, end, peak_row;
            float power;
        };

        float db(float value) { return 10.f * std::log10(std::max(value, 1e-15f)); }

        double p99(std::vector<double> &values) {
            if (values.empty()) return 0;
            const auto n = values.size() * 99 / 100;
            std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(n), values.end());
            return values[n];
        }

        void find_regions(const float *power, const float *floor, const std::array<bool, bins> &valid,
                          float threshold, float step, std::vector<Span> &done) {
            std::vector<Span> active, row, next;
            active.reserve(16); row.reserve(16); next.reserve(16);
            std::array<float, bins> ratio{};
            std::array<float, bins / 8> quantile{};
            const auto scan = [&](int first, int last, float level, int gap, int width, unsigned r, auto emit) {
                int left = -1, right = -1, hits = 0;
                float peak = 0, sum = 0;
                const auto finish = [&] {
                    if (left >= 0 && right + 1 - left >= width && hits * 2 >= right + 1 - left && peak >= level * 1.45f)
                        emit(Span{left, right + 1, r, r + 1, r, sum / static_cast<float>(hits)});
                };
                for (int k = first; k < last; ++k) {
                    if (ratio[k] < level) continue;
                    if (left < 0 || k - right > gap) {
                        finish(); left = k; hits = 0; peak = sum = 0;
                    }
                    right = k; ++hits;
                    peak = std::max(peak, ratio[k]);
                    sum += ratio[k];
                }
                finish();
            };
            for (unsigned r = 0; r < rows; ++r) {
                for (unsigned k = 0; k < bins; ++k)
                    ratio[k] = valid[k] ? power[r * bins + k] / floor[k] : 0;
                for (unsigned k = 0; k < quantile.size(); ++k) quantile[k] = ratio[8 * k];
                auto q = quantile.begin() + (quantile.size() * 9 / 10);
                std::nth_element(quantile.begin(), q, quantile.end());
                const float core = std::max(threshold, *q / 16);
                row.clear();
                // A strong wide signal lifts nominally idle bins between its
                // subcarriers and outside its occupied band. Do not turn
                // those skirts into hundreds of narrow classification jobs.
                const float entry = std::max(threshold, .75f * core);
                scan(0, bins, entry, 5, 4, r, [&](Span span) {
                    if ((span.hi - span.lo) * step < 6e6f) {
                        const auto side_mean = [&](int first, int last) {
                            float sum = 0;
                            for (int k = first; k < last; ++k) sum += ratio[k];
                            return sum / std::max(1, last - first);
                        };
                        const float side = std::max(
                            side_mean(std::max(0, span.lo - 12), std::max(0, span.lo - 3)),
                            side_mean(std::min(static_cast<int>(bins), span.hi + 3),
                                      std::min(static_cast<int>(bins), span.hi + 12)));
                        if (span.power >= 2.6f * side) row.push_back(span);
                        return;
                    }
                    const auto before = row.size();
                    scan(span.lo, span.hi, core, 13, std::max(4, static_cast<int>(6e6f / step)), r,
                         [&](Span candidate) { row.push_back(candidate); });
                    if (row.size() == before) row.push_back(span);
                });
                next.clear();
                for (auto s : row) {
                    auto match = std::find_if(active.begin(), active.end(), [&](const Span &p) {
                        const int overlap = std::min(p.hi, s.hi) - std::max(p.lo, s.lo);
                        return p.end == r && p.end - p.start < 3 &&
                               3 * overlap >= 2 * std::max(p.hi - p.lo, s.hi - s.lo);
                    });
                    if (match != active.end()) {
                        if (match->power > s.power) {
                            s.lo = match->lo; s.hi = match->hi; s.peak_row = match->peak_row; s.power = match->power;
                        }
                        s.start = match->start;
                        match->end = 0;
                    }
                    next.push_back(s);
                }
                for (const auto &p : active) if (p.end) done.push_back(p);
                active.swap(next);
            }
            done.insert(done.end(), active.begin(), active.end());
        }
    }

    BurstStats run_burst_classifier(ISource &source, const engine::Model &model, const BurstConfig &cfg,
                                    const BurstCallback &callback) {
        const auto fmt = source.format();
        if (fmt.channels != 1 || fmt.wire != WireFormat::packed12 ||
            fmt.sample_rate < 40e6 || fmt.sample_rate > 65e6)
            throw std::invalid_argument("burst classifier requires one packed12 receiver at 40–65 MS/s");
        if (model.tensors[model.input].length != length || model.tensors[model.input].channels != 2 ||
            model.tensors[model.output].channels != burst_classes)
            throw std::invalid_argument("burst classifier requires a 1024-point, seven-class demo model");
        if (!(cfg.band_hz > 0 && cfg.band_hz <= fmt.sample_rate) || !std::isfinite(cfg.threshold_db) ||
            cfg.threshold_db < 0 || cfg.threshold_db > 50)
            throw std::invalid_argument("invalid burst bandwidth or detection threshold");
        const auto bytes = frames * fmt.frame_bytes();
        HostRing ring(cfg.ring_slots, bytes);
        ComputeStream compute;
        const auto stream = compute.stream;
        FftPlan fft_plan(stream);
        engine::Engine inference(model, cfg.engine);
        DeviceBuffer<std::uint8_t> wire(bytes);
        DeviceBuffer<float2> raw(frames), fft_input(frames);
        DeviceBuffer<float> power(rows * bins), floor(bins);
        DeviceBuffer<Request> requests(max_regions);
        PinnedBuffer<float> host_power(rows * bins), host_floor(bins), logits(max_regions * burst_classes);
        PinnedBuffer<Request> host_requests(max_regions);
        std::array<float, bins> spectrum{}, floors{};
        std::array<bool, bins> valid{};
        const float rate = static_cast<float>(fmt.sample_rate), step = rate / bins;
        for (unsigned k = 0; k < bins; ++k) {
            const auto hz = (static_cast<int>(k) - static_cast<int>(bins / 2)) * step;
            valid[k] = std::abs(hz) <= cfg.band_hz / 2 && std::abs(hz) >= cfg.dc_guard_hz &&
                std::none_of(cfg.exclude_hz.begin(), cfg.exclude_hz.end(),
                             [&](double f) { return std::abs(cfg.tune_hz + hz - f) < 1.6 * step; });
        }
        const float threshold = std::pow(10.f, cfg.threshold_db / 10.f);
        std::vector<Span> candidates;
        std::vector<BurstRegion> regions;
        std::vector<double> latency, queued;
        ProducerControl control;
        BurstStats stats;
        std::uint64_t expected = 0;
        auto time = Clock::now();
        CURF_CUDA_CHECK(cudaHostRegister(ring.base(), ring.bytes(), cudaHostRegisterDefault));
        std::thread producer;
        try {
            producer = start_producer(source, ring, frames, fmt.frame_bytes(), cfg.drop_when_full, control);
            for (;;) {
                if (cfg.seconds > 0 && stats.frames >= cfg.seconds * fmt.sample_rate) break;
                if (cfg.max_blocks && stats.blocks >= cfg.max_blocks) break;
                auto *slot = ring.next_submit(std::chrono::milliseconds(5));
                if (!slot) {
                    if (control.done.load(std::memory_order_acquire)) break;
                    continue;
                }
                ring.mark_submitted();
                const auto begin = Clock::now();
                queued.push_back(std::chrono::duration<double, std::milli>(begin - slot->filled_at).count());
                const auto first = slot->first_sample;
                const bool gap = stats.blocks && (slot->discontinuity || first != expected);
                if (gap && first > expected) stats.gaps += first - expected;
                expected = first + frames;
                CURF_CUDA_CHECK(cudaMemcpyAsync(wire.get(), slot->data, bytes, cudaMemcpyHostToDevice, stream));
                unpack_window<<<frames / threads, threads, 0, stream>>>(wire.get(), raw.get(), fft_input.get());
                check_fft(cufftExecC2C(fft_plan.plan, fft_input.get(), fft_input.get(), CUFFT_FORWARD));
                powers<<<rows * bins / threads, threads, 0, stream>>>(fft_input.get(), power.get());
                update_floor<<<bins / threads, threads, 0, stream>>>(power.get(), floor.get(), !stats.blocks || gap);
                CURF_CUDA_CHECK(cudaGetLastError());
                CURF_CUDA_CHECK(cudaMemcpyAsync(host_power.get(), power.get(), power.bytes(), cudaMemcpyDeviceToHost, stream));
                CURF_CUDA_CHECK(cudaMemcpyAsync(host_floor.get(), floor.get(), floor.bytes(), cudaMemcpyDeviceToHost, stream));
                CURF_CUDA_CHECK(cudaStreamSynchronize(stream));

                spectrum.fill(0);
                for (unsigned row = 0; row < rows; ++row)
                    for (unsigned k = 0; k < bins; ++k)
                        spectrum[k] = std::max(spectrum[k], host_power[row * bins + k]);
                std::rotate(spectrum.begin(), spectrum.begin() + bins / 2, spectrum.end());
                for (unsigned k = 0; k < bins; ++k) {
                    spectrum[k] = db(spectrum[k]);
                    floors[k] = host_floor[k];
                }
                auto mid = floors.begin() + bins / 2;
                std::nth_element(floors.begin(), mid, floors.end());
                candidates.clear();
                find_regions(host_power.get(), host_floor.get(), valid, threshold, step, candidates);
                if (candidates.size() > max_regions) {
                    std::partial_sort(candidates.begin(), candidates.begin() + max_regions, candidates.end(),
                                      [](const Span &a, const Span &b) { return a.power > b.power; });
                    stats.skipped += candidates.size() - max_regions;
                    candidates.resize(max_regions);
                }
                regions.clear();
                for (const auto &c : candidates) {
                    const float offset = (.5f * (c.lo + c.hi - 1) - bins / 2.f) * step;
                    const float width = (c.hi - c.lo) * step;
                    const int down = std::clamp(static_cast<int>(std::lround(rate / (width * 3.6f))), 1, static_cast<int>(max_down));
                    const float cutoff = std::min(.69f * width / rate, .43f / static_cast<float>(down));
                    host_requests[regions.size()] = {static_cast<int>(c.peak_row * row_frames + row_frames / 2),
                                                     down, offset / rate, cutoff};
                    regions.push_back({c.start * row_frames, c.end * row_frames, offset, width, db(c.power), {}});
                }
                for (unsigned i = 0; i < regions.size();) {
                    const int n = std::min(inference.max_batch(), static_cast<int>(regions.size() - i));
                    CURF_CUDA_CHECK(cudaMemcpyAsync(requests.get(), host_requests.get() + i, n * sizeof(Request),
                                                   cudaMemcpyHostToDevice, stream));
                    if (cfg.engine.half_input)
                        extract<<<n, threads, 0, stream>>>(raw.get(), requests.get(), inference.input_half());
                    else
                        extract<<<n, threads, 0, stream>>>(raw.get(), requests.get(), inference.input());
                    inference.run(n, stream);
                    CURF_CUDA_CHECK(cudaMemcpyAsync(logits.get() + i * burst_classes, inference.logits(),
                                                   n * burst_classes * sizeof(float), cudaMemcpyDeviceToHost, stream));
                    i += n;
                }
                CURF_CUDA_CHECK(cudaStreamSynchronize(stream));
                for (unsigned i = 0; i < regions.size(); ++i) {
                    const auto *l = logits.get() + i * burst_classes;
                    const float high = *std::max_element(l, l + burst_classes);
                    float sum = 0;
                    for (unsigned k = 0; k < burst_classes; ++k) sum += regions[i].probabilities[k] = std::exp(l[k] - high);
                    for (auto &pr : regions[i].probabilities) pr /= sum;
                }
                if (callback) callback(first, spectrum, db(*mid), regions, {slot->data, bytes},
                                       {{host_power.get(), rows * bins}, {host_floor.get(), bins}, valid});
                ring.release_oldest();
                stats.frames += frames; ++stats.blocks; stats.regions += regions.size();
                latency.push_back(std::chrono::duration<double, std::milli>(Clock::now() - begin).count());
            }
            stop_producer(ring, control, producer);
        } catch (...) {
            cudaStreamSynchronize(stream);
            stop_producer(ring, control, producer);
            cudaHostUnregister(ring.base());
            throw;
        }
        CURF_CUDA_CHECK(cudaHostUnregister(ring.base()));
        if (control.error) std::rethrow_exception(control.error);
        stats.wall_s = std::chrono::duration<double>(Clock::now() - time).count();
        stats.dropped = control.dropped.load();
        stats.processing_p99_ms = p99(latency);
        stats.queue_p99_ms = p99(queued);
        return stats;
    }
}
