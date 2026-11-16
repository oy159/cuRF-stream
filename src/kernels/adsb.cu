#include "curf/cuda/adsb.hpp"
#include "curf/cuda/memory.hpp"

#include <cmath>
#include <cstring>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace curf::cuda {

    namespace {

        constexpr std::uint32_t kMaxTaps = 128;
        constexpr std::uint32_t kHalfSlots = 2 * (8 + 112); // half-microsecond slots: preamble + longest frame
        constexpr std::uint32_t kTail = 2048;               // decimated samples kept from the previous block
        constexpr std::uint32_t kGenerator = 0xFFF409;

        __constant__ float c_taps[kMaxTaps];
        __constant__ int c_off[kHalfSlots + 1]; // offset of half-microsecond slot h from the frame start
        __constant__ std::uint32_t c_syndrome[112];

        struct Header {
            std::uint32_t count;
            float noise_sum;
            std::uint32_t pad[2];
        };

        // Unpack the selected stream, mix by exp(-j 2 pi f n / fs) with a 32-bit phase accumulator (exact over
        // any sample time: phase0 is (first_sample * inc) mod 2^32).
        __global__ void mix_kernel(const std::uint8_t *__restrict__ wire, int packed12, std::uint32_t channels,
                                   std::uint32_t stream, std::uint32_t frames, std::uint32_t phase0, std::uint32_t inc,
                                   float2 *__restrict__ out) {
            const std::uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= frames) return;
            float i, q;
            if (packed12) {
                const std::uint8_t *p = wire + (std::size_t{k} * channels + stream) * 3;
                const int ii = (p[0] << 4) | (p[1] >> 4), qq = ((p[1] & 0xF) << 8) | p[2];
                i = static_cast<float>((ii ^ 0x800) - 0x800) * (1.0f / 2048);
                q = static_cast<float>((qq ^ 0x800) - 0x800) * (1.0f / 2048);
            } else {
                const std::uint8_t *p = wire + (std::size_t{k} * channels + stream) * 2;
                i = static_cast<float>(static_cast<std::int8_t>(p[0])) * (1.0f / 128);
                q = static_cast<float>(static_cast<std::int8_t>(p[1])) * (1.0f / 128);
            }
            const std::int32_t ph = static_cast<std::int32_t>(phase0 + k * inc); // [-pi, pi) after scaling
            float s, c;
            __sincosf(static_cast<float>(ph) * (2.0f * 3.14159265358979f / 4294967296.0f), &s, &c);
            out[k] = make_float2(i * c + q * s, q * c - i * s);
        }

        // Low-pass + decimate + magnitude; also the sum of the magnitudes (noise level of the block).
        __global__ void fir_kernel(const float2 *__restrict__ x, std::uint32_t taps, std::uint32_t decim, std::uint32_t n_out,
                                   float *__restrict__ mag, Header *hdr) {
            const std::uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
            float v = 0.f;
            if (j < n_out) {
                const float2 *p = x + std::size_t{j} * decim;
                float re = 0.f, im = 0.f;
                for (std::uint32_t k = 0; k < taps; ++k) {
                    const float2 a = p[k];
                    re = fmaf(c_taps[k], a.x, re);
                    im = fmaf(c_taps[k], a.y, im);
                }
                v = sqrtf(re * re + im * im);
                mag[j] = v;
            }
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
            if ((threadIdx.x & 31) == 0) atomicAdd(&hdr->noise_sum, v);
        }

        // 0.5 us matched filter: sum of `whole` samples plus `frac` of the next.
        __global__ void boxcar_kernel(const float *__restrict__ mag, std::uint32_t n, std::uint32_t whole, float frac,
                                      float *__restrict__ box) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i + whole >= n) return;
            float s = 0.f;
            for (std::uint32_t k = 0; k < whole; ++k) s += mag[i + k];
            box[i] = fmaf(frac, mag[i + whole], s);
        }

        __device__ std::uint32_t remainder_of(const std::uint8_t *m, int bits) {
            std::uint32_t crc = 0;
            for (int i = 0; i < bits - 24; ++i) {
                const std::uint32_t top = ((crc >> 23) ^ (m[i >> 3] >> (7 - (i & 7)))) & 1u;
                crc = (crc << 1) & 0xFFFFFFu;
                if (top) crc ^= kGenerator;
            }
            std::uint32_t parity = 0;
            for (int i = bits - 24; i < bits; ++i) parity = (parity << 1) | ((m[i >> 3] >> (7 - (i & 7))) & 1u);
            return crc ^ parity;
        }

        __device__ __forceinline__ float score_at(const float *box, std::int64_t p) {
            return box[p] + box[p + c_off[2]] + box[p + c_off[7]] + box[p + c_off[9]];
        }

        __global__ void detect_kernel(const float *__restrict__ box, std::uint32_t p0, std::uint32_t count, float box_len,
                                      std::uint32_t n_new, float threshold, std::uint32_t capacity,
                                      std::uint64_t sample0, std::uint32_t decim, std::uint32_t delay, Header *hdr,
                                      AdsbFrame *out) {
            const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= count) return;
            const std::int64_t p = std::int64_t{p0} + idx;
            const float noise_box = hdr->noise_sum / static_cast<float>(n_new) * box_len;

            const float h0 = box[p], h1 = box[p + c_off[2]], h2 = box[p + c_off[7]], h3 = box[p + c_off[9]];
            const float hmin = fminf(fminf(h0, h1), fminf(h2, h3));
            if (hmin < threshold * noise_box) return;
            constexpr int quiet[12] = {1, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15};
            float q = 0.f;
#pragma unroll
            for (int k = 0; k < 12; ++k) q += box[p + c_off[quiet[k]]];
            q *= 1.0f / 12;
            if (hmin < threshold * q) return;
            // one candidate per preamble: local maximum of the pulse sum
            const float s = h0 + h1 + h2 + h3;
            if (s < score_at(box, p - 1) || s <= score_at(box, p + 1)) return;

            std::uint8_t m[14] = {};
            auto bit = [&](int i) {
                return box[p + c_off[16 + 2 * i]] > box[p + c_off[17 + 2 * i]];
            };
            for (int i = 0; i < 5; ++i) m[0] |= static_cast<std::uint8_t>(bit(i) << (7 - i));
            const int df = m[0] >> 3;
            const bool longf = df >= 16;
            if (!(df == 0 || df == 4 || df == 5 || df == 11 || df == 16 || df == 17 || df == 18 || df == 20 || df == 21))
                return;
            const int bits = longf ? 112 : 56;
            for (int i = 5; i < bits; ++i)
                if (bit(i)) m[i >> 3] |= static_cast<std::uint8_t>(0x80u >> (i & 7));

            std::uint32_t rem = remainder_of(m, bits);
            std::uint8_t corrected = 0;
            if (df == 17 || df == 18) {
                if (rem != 0) {
                    int fix = -1;
                    for (int i = 5; i < 112; ++i)
                        if (c_syndrome[i] == rem) { fix = i; break; }
                    if (fix < 0) return;
                    m[fix >> 3] ^= static_cast<std::uint8_t>(0x80u >> (fix & 7));
                    rem = 0;
                    corrected = 1;
                }
            } else if (df == 11) {
                if (rem & 0xFFFF80u) return;
            }
            const std::uint32_t slot = atomicAdd(&hdr->count, 1u);
            if (slot >= capacity) return;
            AdsbFrame &f = out[slot];
            const std::int64_t t = static_cast<std::int64_t>(sample0) + p * decim - delay;
            f.sample = static_cast<std::uint64_t>(t < 0 ? 0 : t);
            f.signal = s * 0.25f / box_len;
            f.noise = noise_box / box_len;
            f.remainder = rem;
            for (int i = 0; i < 14; ++i) f.msg[i] = m[i];
            f.bits = static_cast<std::uint8_t>(bits);
            f.corrected = corrected;
        }

        std::vector<float> lowpass(std::uint32_t taps, double cutoff) { // cutoff: fraction of fs
            constexpr double beta = 6.0;
            auto i0 = [](double x) {
                double s = 1, t = 1;
                for (int k = 1; k < 30; ++k) {
                    t *= (x / (2 * k)) * (x / (2 * k));
                    s += t;
                }
                return s;
            };
            std::vector<double> h(taps);
            double sum = 0;
            const double mid = (taps - 1) / 2.0;
            for (std::uint32_t k = 0; k < taps; ++k) {
                const double x = k - mid, r = x / (mid + 0.5);
                const double sinc = x == 0 ? 2 * cutoff : std::sin(2 * std::numbers::pi * cutoff * x) / (std::numbers::pi * x);
                h[k] = sinc * i0(beta * std::sqrt(std::max(0.0, 1 - r * r))) / i0(beta);
                sum += h[k];
            }
            std::vector<float> out(taps);
            for (std::uint32_t k = 0; k < taps; ++k) out[k] = static_cast<float>(h[k] / sum);
            return out;
        }

        std::uint32_t syndrome(int bit, int bits) { // host copy of remainder_of on a single set bit
            std::uint8_t m[14] = {};
            m[bit >> 3] = static_cast<std::uint8_t>(0x80u >> (bit & 7));
            std::uint32_t crc = 0;
            for (int i = 0; i < bits - 24; ++i) {
                const std::uint32_t top = ((crc >> 23) ^ (m[i >> 3] >> (7 - (i & 7)))) & 1u;
                crc = (crc << 1) & 0xFFFFFFu;
                if (top) crc ^= kGenerator;
            }
            std::uint32_t parity = 0;
            for (int i = bits - 24; i < bits; ++i) parity = (parity << 1) | ((m[i >> 3] >> (7 - (i & 7))) & 1u);
            return crc ^ parity;
        }

    } // namespace

    struct AdsbDemod::Impl {
        AdsbConfig cfg;
        WireFormat wire;
        std::uint32_t channels;
        std::size_t max_frames;
        std::uint32_t inc = 0, whole = 0, span = 0;
        float frac = 0, box_len = 0;
        DeviceBuffer<float2> x;          // [taps - 1 history | block]
        DeviceBuffer<float> mag[2], box; // [kTail | decimated block]
        DeviceBuffer<std::uint8_t> out;  // Header + capacity frames
        std::vector<PinnedBuffer<std::uint8_t>> host;
        std::size_t out_bytes = 0;
        std::uint32_t cur = 0, prev_new = 0;
        bool first = true;
    };

    AdsbDemod::AdsbDemod(const AdsbConfig &config, WireFormat wire, std::uint32_t channels, std::size_t max_block_frames,
                         std::size_t slots)
        : impl_(std::make_unique<Impl>()) {
        Impl &m = *impl_;
        m.cfg = config;
        m.wire = wire;
        m.channels = channels;
        m.max_frames = max_block_frames;
        if (config.taps == 0 || config.taps > kMaxTaps) throw std::invalid_argument("adsb: 1..128 taps");
        if (config.decimation == 0 || max_block_frames % config.decimation) throw std::invalid_argument("adsb: block % decimation");
        if (config.stream >= channels) throw std::invalid_argument("adsb: stream index out of range");
        const double fs2 = config.sample_rate / config.decimation;
        m.box_len = static_cast<float>(0.5e-6 * fs2);
        if (m.box_len < 2) throw std::invalid_argument("adsb: decimated rate below 4 MS/s");
        m.whole = static_cast<std::uint32_t>(m.box_len);
        m.frac = m.box_len - static_cast<float>(m.whole);
        std::vector<int> off(kHalfSlots + 1);
        for (std::uint32_t h = 0; h <= kHalfSlots; ++h) off[h] = static_cast<int>(std::lround(h * 0.5e-6 * fs2));
        m.span = static_cast<std::uint32_t>(off[kHalfSlots - 1]) + m.whole + 3;
        if (m.span + 2 > kTail) throw std::invalid_argument("adsb: frame longer than the kept tail");
        const double cyc = config.offset_hz / config.sample_rate; // phase of the carrier; mix_kernel multiplies by its conjugate
        m.inc = static_cast<std::uint32_t>(static_cast<std::int64_t>(std::llround(cyc * 4294967296.0)));

        const auto taps = lowpass(config.taps, config.cutoff_hz / config.sample_rate);
        std::vector<std::uint32_t> syn(112);
        for (int i = 0; i < 112; ++i) syn[static_cast<std::size_t>(i)] = syndrome(i, 112);
        CURF_CUDA_CHECK(cudaMemcpyToSymbol(c_taps, taps.data(), taps.size() * sizeof(float)));
        CURF_CUDA_CHECK(cudaMemcpyToSymbol(c_off, off.data(), off.size() * sizeof(int)));
        CURF_CUDA_CHECK(cudaMemcpyToSymbol(c_syndrome, syn.data(), syn.size() * sizeof(std::uint32_t)));

        const std::size_t n_dec = max_block_frames / config.decimation;
        m.x = DeviceBuffer<float2>(max_block_frames + config.taps - 1);
        for (auto &b : m.mag) b = DeviceBuffer<float>(kTail + n_dec);
        m.box = DeviceBuffer<float>(kTail + n_dec);
        m.out_bytes = sizeof(Header) + std::size_t{config.capacity} * sizeof(AdsbFrame);
        m.out = DeviceBuffer<std::uint8_t>(m.out_bytes);
        for (std::size_t i = 0; i < slots; ++i) {
            m.host.emplace_back(m.out_bytes);
            std::memset(m.host.back().get(), 0, m.out_bytes);
        }
        CURF_CUDA_CHECK(cudaMemset(m.x.get(), 0, m.x.bytes()));
        for (auto &b : m.mag) CURF_CUDA_CHECK(cudaMemset(b.get(), 0, b.bytes()));
        CURF_CUDA_CHECK(cudaMemset(m.box.get(), 0, m.box.bytes()));
    }

    AdsbDemod::~AdsbDemod() = default;

    void AdsbDemod::enqueue(const std::uint8_t *d_wire, std::size_t frames, std::uint64_t first_sample, std::size_t slot,
                            cudaStream_t s) {
        Impl &m = *impl_;
        if (frames > m.max_frames || frames % m.cfg.decimation) throw std::invalid_argument("adsb: bad block size");
        const auto F = static_cast<std::uint32_t>(frames), T = m.cfg.taps, D = m.cfg.decimation;
        const std::uint32_t J = F / D;
        float2 *x = m.x.get();
        float *prev = m.mag[m.cur].get(), *mag = m.mag[m.cur ^ 1].get();
        Header *hdr = reinterpret_cast<Header *>(m.out.get());
        AdsbFrame *frames_out = reinterpret_cast<AdsbFrame *>(m.out.get() + sizeof(Header));

        CURF_CUDA_CHECK(cudaMemsetAsync(hdr, 0, sizeof(Header), s));
        const auto phase0 = static_cast<std::uint32_t>(first_sample * static_cast<std::uint64_t>(m.inc));
        mix_kernel<<<(F + 255) / 256, 256, 0, s>>>(d_wire, m.wire == WireFormat::packed12, m.channels, m.cfg.stream, F,
                                                   phase0, m.inc, x + (T - 1));
        CURF_CUDA_CHECK(cudaGetLastError());
        if (!m.first) // the previous block's last kTail decimated samples in front of this block's
            CURF_CUDA_CHECK(cudaMemcpyAsync(mag, prev + m.prev_new, kTail * sizeof(float), cudaMemcpyDeviceToDevice, s));
        fir_kernel<<<(J + 255) / 256, 256, 0, s>>>(x, T, D, J, mag + kTail, hdr);
        CURF_CUDA_CHECK(cudaGetLastError());
        // FIR history for the next block: the last taps - 1 mixed samples
        CURF_CUDA_CHECK(cudaMemcpyAsync(x, x + F, (T - 1) * sizeof(float2), cudaMemcpyDeviceToDevice, s));
        const std::uint32_t n = kTail + J;
        boxcar_kernel<<<(n + 255) / 256, 256, 0, s>>>(mag, n, m.whole, m.frac, m.box.get());
        CURF_CUDA_CHECK(cudaGetLastError());
        // every start position exactly once over consecutive blocks: [kTail - span, kTail + J - span)
        const std::uint32_t p0 = kTail - m.span;
        detect_kernel<<<(J + 127) / 128, 128, 0, s>>>(m.box.get(), p0, J, m.box_len, J, m.cfg.threshold, m.cfg.capacity,
                                                      first_sample - std::uint64_t{kTail} * D, D, (T - 1) / 2, hdr,
                                                      frames_out);
        CURF_CUDA_CHECK(cudaGetLastError());
        CURF_CUDA_CHECK(cudaMemcpyAsync(m.host[slot].get(), m.out.get(), m.out_bytes, cudaMemcpyDeviceToHost, s));
        m.cur ^= 1;
        m.prev_new = J;
        m.first = false;
    }

    std::span<const AdsbFrame> AdsbDemod::results(std::size_t slot) const {
        const auto *h = reinterpret_cast<const Header *>(impl_->host[slot].get());
        const auto *f = reinterpret_cast<const AdsbFrame *>(impl_->host[slot].get() + sizeof(Header));
        return {f, std::min(h->count, impl_->cfg.capacity)};
    }

    std::uint32_t AdsbDemod::found(std::size_t slot) const {
        return reinterpret_cast<const Header *>(impl_->host[slot].get())->count;
    }

    float AdsbDemod::noise(std::size_t slot) const {
        const auto *h = reinterpret_cast<const Header *>(impl_->host[slot].get());
        const std::size_t n = impl_->max_frames / impl_->cfg.decimation;
        return h->noise_sum / static_cast<float>(n);
    }

} // namespace curf::cuda
