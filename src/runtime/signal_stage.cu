#include "curf/runtime/signal_stage.hpp"

#include "curf/cuda/memory.hpp"
#include "curf/cuda/extraction_bank.hpp"

#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <stdexcept>
#include <string>
#include <sstream>

namespace curf {

    using cuda::DeviceBuffer;
    using cuda::PinnedBuffer;

    namespace {

        constexpr int kSub = 512;          // sub-bands per coarse band (bandwidth estimate)
        constexpr int kCand = 64;          // candidate segments when picking the strongest 1024 samples

        struct Event {
            cudaEvent_t e = nullptr;
            Event() { CURF_CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); }
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

        // ---------------------------------------------------------------- kernels

        struct SlotList {
            const std::uint8_t *p[32]; // raw blocks of the window, in order
        };

        // wire blocks -> complex [S][NF], full scale 1
        __global__ void unpack_kernel(SlotList slots, int packed12, std::uint32_t streams, std::uint32_t block, std::uint32_t nf,
                                      float2 *__restrict__ out) {
            const std::uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= nf) return;
            const std::uint8_t *w = slots.p[k / block];
            if (!w) {
                for (unsigned s = 0; s < streams; ++s) out[std::size_t{s} * nf + k] = make_float2(0, 0);
                return;
            }
            const std::uint32_t i = k % block;
            for (std::uint32_t s = 0; s < streams; ++s) {
                float re, im;
                if (packed12) {
                    const std::uint8_t *q = w + (std::size_t{i} * streams + s) * 3;
                    const int a = (q[0] << 4) | (q[1] >> 4), b = ((q[1] & 0xF) << 8) | q[2];
                    re = static_cast<float>((a ^ 0x800) - 0x800) * (1.0f / 2048);
                    im = static_cast<float>((b ^ 0x800) - 0x800) * (1.0f / 2048);
                } else {
                    const std::uint8_t *q = w + (std::size_t{i} * streams + s) * 2;
                    re = static_cast<float>(static_cast<std::int8_t>(q[0])) * (1.0f / 128);
                    im = static_cast<float>(static_cast<std::int8_t>(q[1])) * (1.0f / 128);
                }
                out[std::size_t{s} * nf + k] = make_float2(re, im);
            }
        }

        struct BandReq {
            std::uint32_t stream;
            int k0;           // first bin, signed (negative: below the centre)
            std::uint32_t nb; // bins
        };

        // power of each request's coarse band in kSub sub-bands
        __global__ void band_power_kernel(const float2 *__restrict__ spec, const BandReq *__restrict__ req, std::uint32_t nf,
                                          float *__restrict__ sub) {
            const BandReq r = req[blockIdx.x];
            const std::uint32_t j = threadIdx.x; // sub-band
            const std::uint64_t a = std::uint64_t{r.nb} * j / kSub, b = std::uint64_t{r.nb} * (j + 1) / kSub;
            const float2 *x = spec + std::size_t{r.stream} * nf;
            float acc = 0.f;
            for (std::uint64_t i = a; i < b; ++i) {
                const int kb = r.k0 + static_cast<int>(i);
                const float2 v = x[(kb + static_cast<int>(nf)) % static_cast<int>(nf)];
                acc += v.x * v.x + v.y * v.y;
            }
            sub[std::size_t{blockIdx.x} * kSub + j] = acc;
        }

        struct WinReq {
            const float2 *src;   // extracted samples (scratch or a track buffer)
            std::uint32_t len;   // >= length
            float bw_ch;         // bandwidth input channel
            float scale;         // to full scale (1 / NF of the forward + inverse transforms)
        };

        // engine input row [L][3] per window: strongest L-sample segment of src, unit RMS, bandwidth channel;
        // also the segment's mean power (full scale)
        __global__ void window_kernel(const WinReq *__restrict__ req, std::uint32_t L, float *__restrict__ in,
                                      float *__restrict__ power, std::uint32_t *__restrict__ starts, float *__restrict__ activity) {
            const WinReq r = req[blockIdx.x];
            __shared__ float e[kCand];
            __shared__ std::uint32_t best_start;
            __shared__ float rms;
            const std::uint32_t span = r.len - L, stride = max(1u, span / (kCand - 1));
            if (threadIdx.x < kCand) {
                const std::uint32_t s0 = min(threadIdx.x * stride, span);
                float acc = 0.f;
                for (std::uint32_t i = 0; i < L; i += 4) { // every 4th sample is plenty for picking the segment
                    const float2 v = r.src[s0 + i];
                    acc += v.x * v.x + v.y * v.y;
                }
                e[threadIdx.x] = acc;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                std::uint32_t bi = 0;
                const std::uint32_t n = span == 0 ? 1u : min(static_cast<std::uint32_t>(kCand), span / stride + 1u);
                for (std::uint32_t c = 1; c < n; ++c)
                    if (e[c] > e[bi]) bi = c;
                best_start = min(bi * stride, span);
                starts[blockIdx.x] = best_start;
                unsigned active = 0;
                for (unsigned c = 0; c < n; ++c) if (e[c] >= .2f * e[bi]) ++active;
                activity[blockIdx.x] = r.len >= 2 * L && e[bi] > 0 ? static_cast<float>(active) / n : -1.f;
            }
            __syncthreads();
            float acc = 0.f;
            for (std::uint32_t i = threadIdx.x; i < L; i += blockDim.x) {
                const float2 v = r.src[best_start + i];
                acc += v.x * v.x + v.y * v.y;
            }
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
            __shared__ float part[32];
            if ((threadIdx.x & 31) == 0) part[threadIdx.x / 32] = acc;
            __syncthreads();
            if (threadIdx.x == 0) {
                float s = 0.f;
                for (std::uint32_t w = 0; w < (blockDim.x + 31) / 32; ++w) s += part[w];
                rms = sqrtf(s / static_cast<float>(L));
                power[blockIdx.x] = s / static_cast<float>(L) * r.scale * r.scale;
            }
            __syncthreads();
            const float g = 1.f / fmaxf(rms, 1e-30f);
            float *row = in + std::size_t{blockIdx.x} * L * 3;
            for (std::uint32_t i = threadIdx.x; i < L; i += blockDim.x) {
                const float2 v = r.src[best_start + i];
                row[3 * i] = v.x * g;
                row[3 * i + 1] = v.y * g;
                row[3 * i + 2] = r.bw_ch;
            }
        }

        struct AppendReq {
            const float2 *src;
            float2 *dst;
            std::uint32_t n;
        };

        __global__ void append_kernel(const AppendReq *__restrict__ req) {
            const AppendReq r = req[blockIdx.y];
            for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < r.n; i += gridDim.x * blockDim.x) r.dst[i] = r.src[i];
        }

        __global__ void softmax_kernel(const float *__restrict__ logits, std::uint32_t n, std::uint32_t c, float *__restrict__ out) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;
            const float *l = logits + std::size_t{i} * c;
            float m = l[0];
            for (std::uint32_t k = 1; k < c; ++k) m = fmaxf(m, l[k]);
            float z = 0.f;
            for (std::uint32_t k = 0; k < c; ++k) z += __expf(l[k] - m);
            for (std::uint32_t k = 0; k < c; ++k) out[std::size_t{i} * c + k] = __expf(l[k] - m) / z;
        }

        // smallest W in {2^m, 3 * 2^(m-1)} with W >= need, within [16, nf]
        std::uint32_t pick_w(double need, std::uint32_t nf) {
            std::uint32_t w = 16;
            for (;;) {
                if (w >= need || w >= nf) return std::min(w, nf);
                const std::uint32_t mid = w + w / 2;
                if (mid >= need) return mid;
                w *= 2;
            }
        }

    } // namespace

    std::string SignalStageStats::to_json() const {
        std::ostringstream o;
        auto percentiles = [&](const cuda::Percentiles &p) {
            o << "{\"p50\":" << p.p50 << ",\"p99\":" << p.p99 << ",\"p999\":" << p.p999 << ",\"max\":" << p.max << "}";
        };
        o << "{\"windows\":" << windows << ",\"signals\":" << signals << ",\"classified\":" << classified
          << ",\"skipped_busy\":" << skipped_busy << ",\"skipped_ring\":" << skipped_ring
          << ",\"skipped_capacity\":" << skipped_capacity << ",\"tracks\":" << tracks
          << ",\"skipped_signals\":" << skipped_signals << ",\"skipped_tracks\":" << skipped_tracks
          << ",\"skipped_workspace\":" << skipped_workspace
          << ",\"component_probes\":" << component_probes
          << ",\"wide\":" << wide << ",\"narrow\":" << narrow << ",\"wide_latency_ms\":";
        percentiles(wide_latency_ms); o << ",\"narrow_latency_ms\":"; percentiles(narrow_latency_ms);
        o << "}"; return o.str();
    }

    struct SignalStage::Impl {
        SignalStageConfig cfg;
        StreamFormat fmt;
        std::size_t block = 0;     // raw frames per block
        std::uint32_t nf = 0, bpw = 0, classes = 0, L = 0;
        double fs = 0, bin_hz = 0;
        Callback cb;
        engine::Engine eng;
        cudaStream_t s2 = nullptr;
        std::unique_ptr<cuda::ExtractionBank> bank;
        std::uint64_t origin = ~std::uint64_t{0};
        using Clock = std::chrono::steady_clock;
        std::vector<double> narrow_latency, wide_latency;

        // raw ring
        DeviceBuffer<std::uint8_t> ring;
        struct RingSlot {
            std::uint64_t first = ~std::uint64_t{0};
            std::size_t frames = 0;
            std::unique_ptr<Event> written, read;
            bool read_pending = false;
        };
        std::vector<RingSlot> slots;
        std::uint64_t ring_next = 0;

        // tracks
        struct Track {
            bool used = false;
            bool component_probe = false; // full coarse component, independent of fragment associations
            std::uint32_t stream = 0;
            int lo = 0, hi = 0;                 // last coarse band, signed channels
            std::uint64_t last_seen = 0, last_done = ~std::uint64_t{0}, last_class = 0;
            bool classified_once = false;
            std::uint32_t w = 0;
            int kc = 0;                         // centre bin
            double bw = 0;
            std::uint64_t estimate_window = 0;
            std::uint32_t fill = 0;             // narrow: samples collected in the buffer
            double buffer_first = 0;
            std::uint64_t fill_window = 0;      // window of the last append
        };
        std::vector<Track> tracks;
        DeviceBuffer<float2> track_buf; // [max_tracks][L]

        // per window in flight
        struct Signal {
            std::uint32_t track;
            BandReq band;
        };
        struct Pending {
            enum class State { free, a, b } state = State::free;
            std::uint64_t window = 0, first = 0;
            Clock::time_point ready_at;
            std::vector<Signal> sig;
            DeviceBuffer<float2> spec, scratch;
            DeviceBuffer<BandReq> d_band;
            PinnedBuffer<BandReq> h_band;
            DeviceBuffer<float> d_sub;
            PinnedBuffer<float> h_sub;
            DeviceBuffer<AppendReq> d_append;
            PinnedBuffer<AppendReq> h_append;
            DeviceBuffer<WinReq> d_win;
            PinnedBuffer<WinReq> h_win;
            DeviceBuffer<std::uint32_t> d_start;
            PinnedBuffer<std::uint32_t> h_start;
            DeviceBuffer<float> d_probs, d_power, d_activity;
            PinnedBuffer<float> h_probs, h_power, h_input, h_activity;
            std::unique_ptr<Event> a_done, b_done;
            std::vector<SignalResult> out; // filled in phase B, probabilities after b_done
        };
        std::vector<Pending> pend;
        std::deque<std::size_t> order; // pending indices in window order
        SignalStageStats st;

        Impl(const SignalStageConfig &c, const engine::Model &model, StreamFormat f, std::size_t block_frames, Callback cb_)
            : cfg(c), fmt(f), block(block_frames), cb(std::move(cb_)), eng(model, c.engine) {
            if (!block || !c.pending || !c.max_signals || c.max_signals > 256 || !c.max_tracks || c.channels < 2 ||
                !c.scratch_complex || c.scratch_complex > (std::size_t{1} << 26))
                throw std::invalid_argument("signal stage: invalid capacities");
            nf = c.window_frames * (c.channels / 2);
            if (nf % block) throw std::invalid_argument("signal stage: window must be whole raw blocks");
            bpw = static_cast<std::uint32_t>(nf / block);
            if (bpw > 8) throw std::invalid_argument("signal stage: at most 8 raw blocks per window");
            if (c.ring_blocks < 8 * bpw) throw std::invalid_argument("signal stage: ring too short");
            const auto &in = model.tensors[model.input];
            if (in.channels != 3 || in.length != c.length) throw std::invalid_argument("signal stage: model input must be [L][3]");
            L = c.length;
            classes = model.tensors[model.output].channels;
            fs = f.sample_rate;
            bin_hz = fs / nf;
            CURF_CUDA_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
            const std::size_t bb = block * f.frame_bytes();
            ring = DeviceBuffer<std::uint8_t>(bb * c.ring_blocks);
            slots.resize(c.ring_blocks);
            for (auto &s : slots) {
                s.written = std::make_unique<Event>();
                s.read = std::make_unique<Event>();
            }
            tracks.resize(c.max_tracks);
            track_buf = DeviceBuffer<float2>(std::size_t{c.max_tracks} * L * 2);
            bank = std::make_unique<cuda::ExtractionBank>(nf, f.channels, fs);
            pend.resize(c.pending);
            const std::size_t ms = c.max_signals;
            for (auto &p : pend) {
                p.spec = DeviceBuffer<float2>(std::size_t{f.channels} * 4 * nf);
                p.scratch = DeviceBuffer<float2>(c.scratch_complex);
                p.d_band = DeviceBuffer<BandReq>(ms);
                p.h_band = PinnedBuffer<BandReq>(ms);
                p.d_sub = DeviceBuffer<float>(ms * kSub);
                p.h_sub = PinnedBuffer<float>(ms * kSub);
                p.d_append = DeviceBuffer<AppendReq>(ms * 2);
                p.h_append = PinnedBuffer<AppendReq>(ms * 2);
                p.d_win = DeviceBuffer<WinReq>(ms);
                p.h_win = PinnedBuffer<WinReq>(ms);
                p.d_start = DeviceBuffer<std::uint32_t>(ms);
                p.h_start = PinnedBuffer<std::uint32_t>(ms);
                p.d_probs = DeviceBuffer<float>(ms * classes);
                p.d_power = DeviceBuffer<float>(ms);
                p.d_activity = DeviceBuffer<float>(ms);
                p.h_probs = PinnedBuffer<float>(ms * classes);
                p.h_power = PinnedBuffer<float>(ms);
                p.h_activity = PinnedBuffer<float>(ms);
                if (cfg.on_input) p.h_input = PinnedBuffer<float>(ms * L * 3);
                p.a_done = std::make_unique<Event>();
                p.b_done = std::make_unique<Event>();
            }
            if (static_cast<std::size_t>(eng.max_batch()) < ms) throw std::invalid_argument("signal stage: engine bucket < max_signals");
        }

        ~Impl() {
            cudaStreamSynchronize(s2);
            bank.reset();
            cudaStreamDestroy(s2);
        }

        void enqueue(const std::uint8_t *d_wire, std::size_t frames, std::uint64_t first, cudaStream_t s) {
            if (frames != block) throw std::invalid_argument("signal stage: incomplete raw block");
            if (origin == ~std::uint64_t{0}) origin = first;
            RingSlot &r = slots[ring_next % slots.size()];
            if (r.read_pending) CURF_CUDA_CHECK(cudaStreamWaitEvent(s, r.read->e)); // stage 2 still reading it
            const std::size_t bb = block * fmt.frame_bytes();
            CURF_CUDA_CHECK(cudaMemcpyAsync(ring.get() + (ring_next % slots.size()) * bb, d_wire, frames * fmt.frame_bytes(),
                                            cudaMemcpyDeviceToDevice, s));
            CURF_CUDA_CHECK(cudaEventRecord(r.written->e, s));
            r.first = first;
            r.frames = frames;
            ++ring_next;
        }

        int match_track(std::uint32_t stream, int lo, int hi, std::uint64_t w, bool component_probe = false) {
            int free_slot = -1;
            for (std::size_t i = 0; i < tracks.size(); ++i) {
                Track &t = tracks[i];
                if (t.used && w - t.last_seen > cfg.track_timeout_windows) t.used = false;
                if (!t.used) {
                    if (free_slot < 0) free_slot = static_cast<int>(i);
                    continue;
                }
                if (t.last_seen != w && t.stream == stream && t.component_probe == component_probe &&
                    lo <= t.hi + 1 && t.lo <= hi + 1) {
                    if (component_probe) {
                        const int overlap = std::max(0, std::min(hi,t.hi) - std::max(lo,t.lo) + 1);
                        const int extent = std::max(hi,t.hi) - std::min(lo,t.lo) + 1;
                        if (overlap < .7 * extent) continue;
                    }
                    return static_cast<int>(i);
                }
            }
            if (free_slot >= 0) {
                tracks[free_slot] = Track{};
                tracks[free_slot].used = true;
                tracks[free_slot].component_probe = component_probe;
                tracks[free_slot].stream = stream;
                ++st.tracks;
            }
            return free_slot;
        }

        void on_window(std::uint64_t w, std::uint64_t first, const WindowResult *r, std::size_t n, Clock::time_point ready_at) {
            ++st.windows;
            if (n == 0) return;
            const int M = static_cast<int>(cfg.channels), half = M / 2, lim = half - static_cast<int>(cfg.edge_guard);
            // group adjacent active channels per stream
            std::vector<std::vector<std::pair<int, float>>> per(fmt.channels);
            for (std::size_t i = 0; i < n; ++i) {
                const int k = static_cast<int>(r[i].channel), ks = k < half ? k : k - M;
                if (std::abs(ks) > lim) continue;
                per[r[i].stream].push_back({ks, r[i].power});
            }
            std::vector<Signal> sig;
            for (std::uint32_t stream = 0; stream < fmt.channels; ++stream) {
                auto &v = per[stream];
                std::sort(v.begin(), v.end());
                auto submit = [&](unsigned id, int lo, int hi) {
                    auto &tr = tracks[id];
                    if (tr.last_seen == w && tr.w) { tr.lo = std::min(tr.lo,lo); tr.hi = std::max(tr.hi,hi); return; }
                    tr.lo = lo; tr.hi = hi; tr.last_seen = w;
                    const bool narrow = tr.w && tr.w < L;
                    if (!narrow && tr.classified_once && w - tr.last_class < cfg.reclassify_windows) return;
                    if (sig.size() >= cfg.max_signals) { ++st.skipped_capacity; ++st.skipped_signals; return; }
                    const int bpc = static_cast<int>(4 * nf / cfg.channels);
                    int k0 = std::max((lo - 1) * bpc - bpc / 2, -static_cast<int>(2 * nf));
                    int k1 = std::min((hi + 1) * bpc + bpc / 2, static_cast<int>(2 * nf) - 1);
                    if (tr.w && tr.w < L && tr.bw > 0) {
                        // Once a narrow carrier is found, refine locally. Hundreds of kHz of guards can
                        // otherwise include the skirts of a stronger neighbour and pull the 99% OBW onto it.
                        const int radius = std::max(kSub/2, static_cast<int>(std::ceil(3.0 * tr.bw / (bin_hz/4))));
                        k0 = std::max(tr.kc - radius, -static_cast<int>(2*nf));
                        k1 = std::min(tr.kc + radius, static_cast<int>(2*nf)-1);
                    }
                    sig.push_back({id, BandReq{stream, k0, static_cast<unsigned>(k1 - k0 + 1)}});
                };
                struct Prior { unsigned id; int lo, hi; };
                std::vector<Prior> prior;
                for (unsigned id = 0; id < tracks.size(); ++id) {
                    const auto &tr = tracks[id];
                    if (tr.used && !tr.component_probe && tr.w && tr.stream == stream && w - tr.last_seen <= cfg.track_timeout_windows)
                        prior.push_back({id, tr.lo, tr.hi});
                }
                // Work on complete current coarse components. A sole known carrier owns the entire component,
                // including its skirts. If a temporary bridge connects several known carriers, partition it
                // at their previous gap instead of migrating the narrow carrier onto its strong neighbour.
                for (std::size_t a = 0; a < v.size();) {
                    std::size_t b = a;
                    while (b + 1 < v.size() && v[b+1].first <= v[b].first + 2) ++b;
                    const int lo = v[a].first, hi = v[b].first;
                    a = b + 1;
                    std::vector<Prior> matches;
                    for (const auto &tr : prior)
                        if (lo <= tr.hi && tr.lo <= hi) matches.push_back(tr);
                    std::sort(matches.begin(), matches.end(), [](const Prior &x, const Prior &y) { return x.lo < y.lo; });
                    if (matches.empty()) {
                        const int id = match_track(stream,lo,hi,w);
                        if (id >= 0) submit(static_cast<unsigned>(id),lo,hi);
                        else { ++st.skipped_capacity; ++st.skipped_tracks; }
                    } else {
                        // Early narrow fragments must not permanently partition a later wide burst.
                        // Probe the complete *currently connected* component, without joining separate
                        // detections or stealing existing carrier tracks. Established wide neighbours
                        // do not qualify: a new probe must span at least three times every prior extent.
                        int largest = 0;
                        for (const auto &m : matches) largest = std::max(largest, m.hi - m.lo + 1);
                        bool continuing_probe = false;
                        for (const auto &tr : tracks) {
                            if (!tr.used || !tr.component_probe || tr.stream != stream ||
                                w - tr.last_seen > cfg.track_timeout_windows) continue;
                            const int overlap = std::max(0, std::min(hi,tr.hi) - std::max(lo,tr.lo) + 1);
                            const int extent = std::max(hi,tr.hi) - std::min(lo,tr.lo) + 1;
                            if (overlap >= .7 * extent) { continuing_probe = true; break; }
                        }
                        if (matches.size() >= 2 && hi - lo + 1 >= 64 &&
                            (hi - lo + 1 >= 3 * largest || continuing_probe)) {
                            const int id = match_track(stream, lo, hi, w, true);
                            if (id >= 0) {
                                const auto before = sig.size();
                                submit(static_cast<unsigned>(id), lo, hi);
                                st.component_probes += sig.size() > before;
                            } else { ++st.skipped_capacity; ++st.skipped_tracks; }
                        }
                        int begin = lo;
                        for (std::size_t i = 0; i < matches.size(); ++i) {
                            const int end = i+1 == matches.size() ? hi : std::clamp((matches[i].hi + matches[i+1].lo)/2, begin, hi);
                            submit(matches[i].id,begin,end);
                            begin = end+1;
                            if (begin > hi) break;
                        }
                    }
                }
            }
            st.signals += sig.size();
            if (sig.empty()) return;
            // raw blocks of this window
            SlotList sl{};
            std::vector<std::size_t> used;
            for (std::uint32_t j = 0; j < 4 * bpw; ++j) {
                const std::int64_t delta = static_cast<std::int64_t>(j) * block - 3 * static_cast<std::int64_t>(nf);
                if (delta < 0 && first - origin < static_cast<std::uint64_t>(-delta)) continue;
                const std::uint64_t want = first + delta;
                std::size_t found = slots.size();
                for (std::size_t i = 0; i < slots.size(); ++i)
                    if (slots[i].first == want && slots[i].frames == block) found = i;
                if (found == slots.size()) {
                    ++st.skipped_ring;
                    return;
                }
                sl.p[j] = ring.get() + found * block * fmt.frame_bytes();
                used.push_back(found);
            }
            // a free pending slot
            std::size_t pi = pend.size();
            for (std::size_t i = 0; i < pend.size(); ++i)
                if (pend[i].state == Pending::State::free) { pi = i; break; }
            if (pi == pend.size()) {
                if (cfg.drop_when_busy) { ++st.skipped_busy; return; }
                poll(true); // bounded offline back-pressure; the calling thread owns raw-ring submissions
                for (std::size_t i = 0; i < pend.size(); ++i)
                    if (pend[i].state == Pending::State::free) { pi = i; break; }
            }
            Pending &p = pend[pi];
            p.window = w;
            p.first = first;
            p.ready_at = ready_at;
            p.sig = std::move(sig);
            // phase A: unpack + FFT + band powers
            for (auto u : used) CURF_CUDA_CHECK(cudaStreamWaitEvent(s2, slots[u].written->e));
            unpack_kernel<<<(4 * nf + 255) / 256, 256, 0, s2>>>(sl, fmt.wire == WireFormat::packed12, fmt.channels,
                                                            static_cast<std::uint32_t>(block), 4 * nf, p.spec.get());
            CURF_CUDA_CHECK(cudaGetLastError());
            for (auto u : used) {
                CURF_CUDA_CHECK(cudaEventRecord(slots[u].read->e, s2));
                slots[u].read_pending = true;
            }
            bank->forward(reinterpret_cast<cf32 *>(p.spec.get()), s2);
            for (std::size_t i = 0; i < p.sig.size(); ++i) p.h_band[i] = p.sig[i].band;
            CURF_CUDA_CHECK(cudaMemcpyAsync(p.d_band.get(), p.h_band.get(), p.sig.size() * sizeof(BandReq), cudaMemcpyHostToDevice, s2));
            band_power_kernel<<<static_cast<unsigned>(p.sig.size()), kSub, 0, s2>>>(p.spec.get(), p.d_band.get(), 4 * nf, p.d_sub.get());
            CURF_CUDA_CHECK(cudaGetLastError());
            CURF_CUDA_CHECK(cudaMemcpyAsync(p.h_sub.get(), p.d_sub.get(), p.sig.size() * kSub * sizeof(float), cudaMemcpyDeviceToHost, s2));
            CURF_CUDA_CHECK(cudaEventRecord(p.a_done->e, s2));
            p.state = Pending::State::a;
            order.push_back(pi);
        }

        // occupied band of one request from its sub-band powers: noise floor (20th percentile) removed, then the
        // 0.5 % / 99.5 % points of the rest; returns {centre bin, bandwidth in bins}
        std::pair<double, double> occupied(const float *sub, const BandReq &b) const {
            std::vector<float> v(sub, sub + kSub), guard;
            // Estimate noise only in the extra edge guards. A percentile of the entire band is signal-biased
            // for a wide carrier, while a small subtraction leaves noise tails dominating a narrow carrier's OBW.
            const double per = static_cast<double>(b.nb) / kSub;
            const int edge = std::clamp(static_cast<int>(48000.0 / (per * bin_hz / 4)), 1, 32);
            for (int j = 0; j < edge; ++j) { guard.push_back(v[j]); guard.push_back(v[kSub-1-j]); }
            std::nth_element(guard.begin(), guard.begin() + guard.size()/2, guard.end());
            const float floor = guard[guard.size()/2];
            double tot = 0;
            for (auto &x : v) tot += (x = std::max(0.f, x - 2.5f * floor));
            const double coarse = std::max(per, static_cast<double>(b.nb) - 8.0 * nf / cfg.channels);
            if (tot <= 0) return {b.k0 + b.nb / 2.0, coarse};
            double c = 0;
            int lo = 0, hi = kSub - 1;
            for (int j = 0; j < kSub; ++j) {
                c += v[j];
                if (c < 0.005 * tot) lo = j + 1;
                if (c < 0.995 * tot) hi = j + 1;
            }
            hi = std::min(hi, kSub - 1);
            const double a = b.k0 + lo * per, e = b.k0 + (hi + 1) * per;
            if (coarse >= 32.0 * nf / cfg.channels && e - a < .6 * coarse)
                return {b.k0 + b.nb / 2.0, coarse};
            return {(a + e) / 2, std::max(e - a, per)};
        }

        void phase_b(Pending &p) {
            struct Item { std::size_t sig; std::uint32_t w; int kc; double bw; std::size_t offset = 0; };
            std::vector<Item> items;
            for (std::size_t i = 0; i < p.sig.size(); ++i) {
                const auto [cbin, bwbin] = occupied(p.h_sub.get() + i * kSub, p.sig[i].band);
                int kc = static_cast<int>(std::lround(cbin));
                double bw = bwbin * bin_hz / 4;
                const Track &tr = tracks[p.sig[i].track];
                // Occupancy within LTE/OFDM and FM speech varies within a hop. Keep a slowly decaying
                // bandwidth envelope; recursively refining a wide carrier inside its last OBW collapses
                // it onto a resource block (or an FSK tone), changing the model's sampling rate and side input.
                if (tr.w && tr.bw > 0) {
                    if (bw < .8 * tr.bw || std::abs(kc-tr.kc) * bin_hz/4 < .25*tr.bw) kc = tr.kc;
                    const double dt = static_cast<double>(p.window-tr.estimate_window) * nf / fs;
                    bw = std::max(bw, std::exp(-dt/2.0) * tr.bw); // 2 s source-time envelope, independent of batching
                }
                auto w = pick_w(2 * bw / bin_hz, nf);
                if (tr.w && tr.w < L && tr.fill_window + 1 == p.window &&
                    2 * bw <= tr.w * bin_hz && std::abs(kc - tr.kc) <= static_cast<int>(tr.w / 2)) {
                    w = tr.w; kc = tr.kc;
                }
                items.push_back({i, w, kc, bw});
            }
            std::stable_sort(items.begin(), items.end(), [](const Item &a, const Item &b) { return a.w < b.w; });
            std::size_t off = 0;
            std::vector<Item> accepted;
            for (std::size_t begin = 0; begin < items.size();) {
                std::size_t end = begin + 1;
                while (end < items.size() && items[end].w == items[begin].w) ++end;
                const auto w = items[begin].w;
                // Keep the bank's bounded plans. A large model batch can use
                // several IFFTs at the same rate without allocating huge plans.
                while (begin < end) {
                    std::size_t capacity = cuda::ExtractionBank::max_batch(w);
                    while (capacity && capacity * 4 * w > cfg.scratch_complex - off) capacity /= 2;
                    if (!capacity) {
                        st.skipped_capacity += end - begin;
                        st.skipped_workspace += end - begin;
                        break;
                    }
                    const auto count = std::min(capacity, end - begin);
                    const auto batch = cuda::ExtractionBank::batch_size(count);
                    std::vector<cuda::ExtractionRequest> req;
                    for (std::size_t i = begin; i < begin + count; ++i) {
                        req.push_back({p.sig[items[i].sig].band.stream, items[i].kc, p.first});
                        items[i].offset = off + (i - begin) * 4 * w;
                        accepted.push_back(items[i]);
                    }
                    bank->extract(reinterpret_cast<cf32 *>(p.spec.get()), w, req,
                                  reinterpret_cast<cf32 *>(p.scratch.get() + off), s2);
                    off += std::size_t{batch} * 4 * w;
                    begin += count;
                }
                begin = end;
            }
            std::vector<AppendReq> append, carry;
            std::vector<WinReq> windows;
            p.out.clear();
            for (const auto &it : accepted) {
                Track &tr = tracks[p.sig[it.sig].track];
                const float bwc = static_cast<float>((std::log10(std::max(it.bw, 1.0)) - 5) / 2);
                SignalResult res;
                res.window = p.window; res.stream = p.sig[it.sig].band.stream; res.track = p.sig[it.sig].track;
                res.fc_hz = it.kc * bin_hz / 4; res.bw_hz = it.bw; res.fs_hz = it.w * bin_hz;
                res.narrow = it.w < L; res.observation_ms = 1000.0 * L / res.fs_hz;
                res.source_time_s = (static_cast<double>(p.first) + nf) / fs;
                double start = static_cast<double>(p.first);
                tr.bw = it.bw; tr.estimate_window = p.window;
                if (it.w >= L) {
                    windows.push_back({p.scratch.get() + it.offset, it.w, bwc, 1.f});
                    tr.fill = 0;
                } else {
                    if (tr.w != it.w || tr.kc != it.kc || tr.fill_window + 1 != p.window) tr.fill = 0;
                    if (!tr.fill) tr.buffer_first = static_cast<double>(p.first);
                    tr.w = it.w; tr.kc = it.kc; tr.fill_window = p.window;
                    auto *buf = track_buf.get() + std::size_t{p.sig[it.sig].track} * 2 * L;
                    append.push_back({p.scratch.get() + it.offset, buf + tr.fill, it.w});
                    tr.fill += it.w;
                    if (tr.fill < L) continue;
                    start = tr.buffer_first;
                    windows.push_back({buf, L, bwc, 1.f});
                    tr.fill -= L;
                    tr.buffer_first += static_cast<double>(L) * nf / it.w;
                    if (tr.fill) carry.push_back({buf + L, buf, tr.fill});
                }
                tr.w = it.w; tr.kc = it.kc; tr.bw = it.bw;
                tr.last_class = p.window; tr.classified_once = true;
                // The FIR is causal, with a group delay of 16 output samples.
                res.t0_s = (start - 16.0 * nf / it.w) / fs;
                res.t1_s = res.t0_s + static_cast<double>(L) / res.fs_hz;
                p.out.push_back(std::move(res));
            }
            for (std::size_t i = 0; i < append.size(); ++i) p.h_append[i] = append[i];
            for (std::size_t i = 0; i < carry.size(); ++i) p.h_append[append.size() + i] = carry[i];
            if (!append.empty() || !carry.empty())
                CURF_CUDA_CHECK(cudaMemcpyAsync(p.d_append.get(), p.h_append.get(), (append.size() + carry.size()) * sizeof(AppendReq), cudaMemcpyHostToDevice, s2));
            if (!append.empty()) {
                append_kernel<<<dim3(4, static_cast<unsigned>(append.size())), 256, 0, s2>>>(p.d_append.get());
                CURF_CUDA_CHECK(cudaGetLastError());
            }
            if (!windows.empty()) {
                std::copy(windows.begin(), windows.end(), p.h_win.get());
                CURF_CUDA_CHECK(cudaMemcpyAsync(p.d_win.get(), p.h_win.get(), windows.size() * sizeof(WinReq), cudaMemcpyHostToDevice, s2));
                window_kernel<<<static_cast<unsigned>(windows.size()), 256, 0, s2>>>(p.d_win.get(), L, eng.input(), p.d_power.get(), p.d_start.get(), p.d_activity.get());
                CURF_CUDA_CHECK(cudaGetLastError());
                if (!carry.empty()) {
                    append_kernel<<<dim3(4, static_cast<unsigned>(carry.size())), 256, 0, s2>>>(p.d_append.get() + append.size());
                    CURF_CUDA_CHECK(cudaGetLastError());
                }
                if (cfg.on_input)
                    CURF_CUDA_CHECK(cudaMemcpyAsync(p.h_input.get(), eng.input(), windows.size() * L * 3 * sizeof(float), cudaMemcpyDeviceToHost, s2));
                eng.run(static_cast<int>(windows.size()), s2);
                softmax_kernel<<<(static_cast<unsigned>(windows.size()) + 127) / 128, 128, 0, s2>>>(eng.logits(), static_cast<std::uint32_t>(windows.size()), classes, p.d_probs.get());
                CURF_CUDA_CHECK(cudaGetLastError());
                CURF_CUDA_CHECK(cudaMemcpyAsync(p.h_probs.get(), p.d_probs.get(), windows.size() * classes * sizeof(float), cudaMemcpyDeviceToHost, s2));
                CURF_CUDA_CHECK(cudaMemcpyAsync(p.h_power.get(), p.d_power.get(), windows.size() * sizeof(float), cudaMemcpyDeviceToHost, s2));
                CURF_CUDA_CHECK(cudaMemcpyAsync(p.h_activity.get(), p.d_activity.get(), windows.size() * sizeof(float), cudaMemcpyDeviceToHost, s2));
                CURF_CUDA_CHECK(cudaMemcpyAsync(p.h_start.get(), p.d_start.get(), windows.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, s2));
            }
            CURF_CUDA_CHECK(cudaEventRecord(p.b_done->e, s2));
            p.state = Pending::State::b;
        }

        void deliver(Pending &p) {
            for (std::size_t i = 0; i < p.out.size(); ++i) {
                SignalResult &r = p.out[i];
                const float *pr = p.h_probs.get() + i * classes;
                r.probs.assign(pr, pr + classes);
                r.cls = static_cast<std::uint32_t>(std::max_element(pr, pr + classes) - pr);
                r.prob = pr[r.cls];
                r.power_db = 10.f * std::log10(std::max(p.h_power[i], 1e-20f));
                r.activity_fraction = p.h_activity[i];
                r.t0_s += p.h_start[i] / r.fs_hz;
                r.t1_s += p.h_start[i] / r.fs_hz;
                r.processing_ms = std::chrono::duration<double, std::milli>(Clock::now() - p.ready_at).count();
                (r.narrow ? narrow_latency : wide_latency).push_back(r.processing_ms);
                if (r.narrow) ++st.narrow; else ++st.wide;
                ++st.classified;
                if (cfg.on_input) cfg.on_input(r, {p.h_input.get() + i * L * 3, std::size_t{L} * 3});
                if (cb) cb(r);
            }
            p.out.clear();
            p.state = Pending::State::free;
        }

        void poll(bool wait) {
            while (!order.empty()) {
                Pending &p = pend[order.front()];
                if (p.state == Pending::State::a) {
                    if (!wait && !p.a_done->ready()) return;
                    if (wait) CURF_CUDA_CHECK(cudaEventSynchronize(p.a_done->e));
                    phase_b(p);
                    if (p.state == Pending::State::free) { order.pop_front(); continue; }
                }
                if (p.state == Pending::State::b) {
                    if (!wait && !p.b_done->ready()) return;
                    if (wait) CURF_CUDA_CHECK(cudaEventSynchronize(p.b_done->e));
                    deliver(p);
                    order.pop_front();
                }
            }
            // ring slots whose reads are done may be overwritten without waiting
            for (auto &s : slots)
                if (s.read_pending && s.read->ready()) s.read_pending = false;
        }
    };

    SignalStage::SignalStage(const SignalStageConfig &config, const engine::Model &model, StreamFormat format,
                             std::size_t block_frames, Callback on_result)
        : impl_(std::make_unique<Impl>(config, model, format, block_frames, std::move(on_result))) {}
    SignalStage::~SignalStage() = default;

    void SignalStage::enqueue(std::size_t, const std::uint8_t *d_wire, std::size_t frames, std::uint64_t first_sample,
                              cudaStream_t stream) {
        impl_->enqueue(d_wire, frames, first_sample, stream);
    }
    void SignalStage::retire(std::size_t) { impl_->poll(false); }
    void SignalStage::on_window(std::uint64_t window, std::uint64_t first_sample, const WindowResult *r, std::size_t n, std::chrono::steady_clock::time_point ready_at) {
        impl_->poll(false);
        impl_->on_window(window, first_sample, r, n, ready_at);
        impl_->poll(false);
    }
    void SignalStage::poll() { impl_->poll(false); }
    void SignalStage::flush() { impl_->poll(true); }
    SignalStageStats SignalStage::stats() const {
        auto st = impl_->st;
        auto pct = [](std::vector<double> v) {
            cuda::Percentiles p;
            if (v.empty()) return p;
            std::sort(v.begin(), v.end());
            auto at = [&](double q) { return v[std::min(v.size()-1, std::size_t(q*v.size()))]; };
            p.p50=at(.5); p.p99=at(.99); p.p999=at(.999); p.max=v.back(); return p;
        };
        st.narrow_latency_ms = pct(impl_->narrow_latency); st.wide_latency_ms = pct(impl_->wide_latency);
        return st;
    }

} // namespace curf
