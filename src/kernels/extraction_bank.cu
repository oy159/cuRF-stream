#include "curf/cuda/extraction_bank.hpp"
#include "curf/cuda/memory.hpp"
#include <cufft.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace curf::cuda {
    namespace {
        void ck(cufftResult r) {
            if (r != CUFFT_SUCCESS) throw std::runtime_error("extraction bank cuFFT error " + std::to_string(r));
        }
        struct Requests {
            std::uint32_t stream[64];
            int centre[64];
            double phase[64];
        };
        // Periodise the filtered, upsampled spectrum into the decimated transform. No explicit zero insertion.
        __global__ void fold(const cf32 *raw, const cf32 *response, int radius, int nr, int up, int m,
                             Requests req, unsigned count, cf32 *out) {
            const int r = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const unsigned row = blockIdx.y;
            if (r >= m) return;
            float ar = 0, ai = 0;
            if (row < count) {
                const int centre = req.centre[row];
                // Only frequencies in the retained FIR response. Each aliases to exactly one output bin.
                const int lo = centre - radius;
                int k = lo + ((r - lo) % m + m) % m;
                const auto *x = raw + std::size_t{req.stream[row]} * nr;
                for (; k <= centre + radius; k += m) {
                    const int t = k - centre;
                    if (t == radius && 2 * radius == nr * up) continue; // unique periodic Nyquist bin
                    const auto h = response[t + radius];
                    const auto v = x[(k % nr + nr) % nr];
                    ar += v.re * h.re - v.im * h.im;
                    ai += v.re * h.im + v.im * h.re;
                }
            }
            out[std::size_t{row} * m + r] = {ar, ai};
        }
        __global__ void valid(cf32 *out, int w, float scale, Requests req, unsigned count) {
            const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const unsigned row = blockIdx.y;
            if (j >= w || row >= count) return;
            auto *x = out + std::size_t{row} * (4 * w);
            const auto v = x[3 * w + j];
            const double angle = -6.2831853071795864769 * remainder(req.phase[row] +
                                  static_cast<double>(req.centre[row]) * j / (4.0 * w), 1.0);
            double s, c;
            sincos(angle, &s, &c);
            x[j] = {static_cast<float>((v.re * c - v.im * s) * scale),
                    static_cast<float>((v.re * s + v.im * c) * scale)};
        }
    }

    struct ExtractionBank::Impl {
        unsigned hop, streams, nr;
        double fs;
        struct Rate {
            unsigned up, down, radius;
            DeviceBuffer<cf32> response;
            std::map<unsigned, cufftHandle> inverse;
        };
        std::map<unsigned, Rate> rates;
        cufftHandle fwd = 0;
        DeviceBuffer<std::byte> workspace;
        std::size_t work_bytes = 0;

        void plan(cufftHandle &p, unsigned size, unsigned batch) {
            ck(cufftCreate(&p));
            ck(cufftSetAutoAllocation(p, 0));
            std::size_t bytes = 0;
            ck(cufftMakePlan1d(p, static_cast<int>(size), CUFFT_C2C, static_cast<int>(batch), &bytes));
            work_bytes = std::max(work_bytes, bytes);
        }
        Impl(unsigned n, unsigned s, double rate) : hop(n), streams(s), nr(4 * n), fs(rate) {
            if (n < 32 || (n & (n - 1)) || !s || s > 4 || n > (1u << 20) || !std::isfinite(fs) || fs <= 0)
                throw std::invalid_argument("extraction bank: invalid dimensions");
            plan(fwd, nr, s);
            for (unsigned p = 16; p <= n; p *= 2) {
                for (unsigned w : {p, p + p / 2}) {
                    if (w > n) continue;
                    auto &r = rates[w];
                    const auto ratio = extraction_ratio(n, w);
                    r.up = ratio.up; r.down = ratio.down;
                    const unsigned nh = nr * r.up;
                    // One Nyquist endpoint only: avoid double-counting a periodic bin.
                    r.radius = std::min(nh / 2, 8 * (4 * w));
                    const auto h = resampling_filter(ratio);
                    std::vector<cf32> time(nh);
                    for (std::size_t k = 0; k < h.size(); ++k) time[k] = {h[k], 0};
                    DeviceBuffer<cf32> tmp(nh);
                    CURF_CUDA_CHECK(cudaMemcpy(tmp.get(), time.data(), tmp.bytes(), cudaMemcpyHostToDevice));
                    cufftHandle hp = 0;
                    ck(cufftPlan1d(&hp, static_cast<int>(nh), CUFFT_C2C, 1));
                    ck(cufftExecC2C(hp, reinterpret_cast<cufftComplex *>(tmp.get()), reinterpret_cast<cufftComplex *>(tmp.get()), CUFFT_FORWARD));
                    CURF_CUDA_CHECK(cudaDeviceSynchronize());
                    r.response = DeviceBuffer<cf32>(2 * r.radius + 1);
                    CURF_CUDA_CHECK(cudaMemcpy(r.response.get(), tmp.get() + nh - r.radius, r.radius * sizeof(cf32), cudaMemcpyDeviceToDevice));
                    CURF_CUDA_CHECK(cudaMemcpy(r.response.get() + r.radius, tmp.get(), (r.radius + 1) * sizeof(cf32), cudaMemcpyDeviceToDevice));
                    cufftDestroy(hp);
                    for (unsigned b = 1; b <= ExtractionBank::max_batch(w); b *= 2)
                        plan(r.inverse[b], 4 * w, b);
                }
            }
            workspace = DeviceBuffer<std::byte>(work_bytes);
            ck(cufftSetWorkArea(fwd, workspace.get()));
            for (auto &[w, r] : rates)
                for (auto &[b, p] : r.inverse) ck(cufftSetWorkArea(p, workspace.get()));
        }
        ~Impl() {
            if (fwd) cufftDestroy(fwd);
            for (auto &[w, r] : rates)
                for (auto &[b, p] : r.inverse) cufftDestroy(p);
        }
    };
    ExtractionBank::ExtractionBank(unsigned hop, unsigned streams, double fs) : impl_(std::make_unique<Impl>(hop, streams, fs)) {}
    ExtractionBank::~ExtractionBank() = default;
    unsigned ExtractionBank::batch_size(std::size_t n) {
        if (!n || n > 64) throw std::invalid_argument("extraction batch outside [1,64]");
        unsigned b = 1; while (b < n) b *= 2; return b;
    }
    unsigned ExtractionBank::max_batch(unsigned w) {
        if (!w || w > (1u << 20)) throw std::invalid_argument("extraction rate outside plan budget");
        unsigned b = 64;
        while (std::size_t{b} * 4 * w > (1u << 22)) b /= 2;
        return b;
    }
    void ExtractionBank::forward(cf32 *context, cudaStream_t stream) {
        ck(cufftSetStream(impl_->fwd, stream));
        auto *p = reinterpret_cast<cufftComplex *>(context);
        ck(cufftExecC2C(impl_->fwd, p, p, CUFFT_FORWARD));
    }
    void ExtractionBank::extract(const cf32 *spectrum, unsigned w, std::span<const ExtractionRequest> requests,
                                 cf32 *out, cudaStream_t stream) {
        auto &x = *impl_;
        const auto b = batch_size(requests.size());
        const auto ri = x.rates.find(w);
        if (ri == x.rates.end() || !ri->second.inverse.count(b)) throw std::invalid_argument("extraction rate/batch unsupported");
        auto &r = ri->second;
        Requests q{};
        for (std::size_t i = 0; i < requests.size(); ++i) {
            const auto &v = requests[i];
            if (v.stream >= x.streams || std::abs(v.centre_bin) >= static_cast<int>(x.nr / 2))
                throw std::invalid_argument("extraction request outside input");
            q.stream[i] = v.stream; q.centre[i] = v.centre_bin;
            ExtractionConfig cfg; cfg.sample_rate = x.fs; cfg.centre_hz = v.centre_bin * x.fs / x.nr;
            q.phase[i] = extraction_phase_cycles(cfg, v.first_sample);
        }
        fold<<<dim3((4 * w + 255) / 256, b), 256, 0, stream>>>(spectrum, r.response.get(), r.radius, x.nr,
                     r.up, 4 * w, q, static_cast<unsigned>(requests.size()), out);
        CURF_CUDA_CHECK(cudaGetLastError());
        const auto plan = r.inverse.at(b);
        ck(cufftSetStream(plan, stream));
        auto *p = reinterpret_cast<cufftComplex *>(out);
        ck(cufftExecC2C(plan, p, p, CUFFT_INVERSE));
        valid<<<dim3((w + 255) / 256, static_cast<unsigned>(requests.size())), 256, 0, stream>>>(out, w,
                    1.f / static_cast<float>(x.nr * r.up), q, static_cast<unsigned>(requests.size()));
        CURF_CUDA_CHECK(cudaGetLastError());
    }
}
