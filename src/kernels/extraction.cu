#include "curf/cuda/extraction.hpp"
#include "curf/cuda/memory.hpp"

#include <cufft.h>

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace curf::cuda {
    namespace {
        void check_fft(cufftResult r) {
            if (r != CUFFT_SUCCESS) throw std::runtime_error("extraction cuFFT error " + std::to_string(r));
        }

        __global__ void multiply(cf32 *x, const cf32 *h, std::size_t n) {
            const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
            if (i < n) {
                const auto a = x[i], b = h[i];
                x[i] = {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
            }
        }

        __global__ void decimate(const cf32 *x, cf32 *out, std::size_t count, std::size_t history,
                                 unsigned d, float scale, double cycles, double start) {
            const std::size_t j = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
            if (j >= count) return;
            const std::size_t i = j * d;
            const auto v = x[history + i];
            // The filter was modulated to the desired RF centre. Translate its valid output to baseband.
            const double phase = -6.2831853071795864769 * remainder(start + cycles * static_cast<double>(i), 1.0);
            double s, c;
            sincos(phase, &s, &c);
            out[j] = {static_cast<float>((v.re * c - v.im * s) * scale),
                      static_cast<float>((v.re * s + v.im * c) * scale)};
        }
    }

    struct SignalExtractor::Impl {
        ExtractionConfig cfg;
        std::size_t nfft = 1;
        DeviceBuffer<cf32> history, work, response;
        cufftHandle fft = 0;
        bool have_next = false;
        std::uint64_t next = 0;

        explicit Impl(const ExtractionConfig &c) : cfg(c) {
            cfg.validate();
            while (nfft < cfg.history_frames() + cfg.block_frames) nfft *= 2;
            if (nfft > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("extraction: FFT too large");
            history = DeviceBuffer<cf32>(cfg.history_frames());
            work = DeviceBuffer<cf32>(nfft);
            response = DeviceBuffer<cf32>(nfft);
            const auto h = extraction_filter(cfg);
            std::vector<cf32> bandpass(nfft);
            const long double cycles = static_cast<long double>(cfg.centre_hz) / cfg.sample_rate;
            for (std::size_t k = 0; k < h.size(); ++k) {
                const double a = static_cast<double>(2 * std::numbers::pi_v<long double> *
                                                    std::remainder(cycles * k, 1.0L));
                bandpass[k] = {static_cast<float>(h[k] * std::cos(a)), static_cast<float>(h[k] * std::sin(a))};
            }
            // Initialisation is synchronous; hot-path copies never borrow temporary host memory.
            CURF_CUDA_CHECK(cudaMemset(history.get(), 0, history.bytes()));
            CURF_CUDA_CHECK(cudaMemcpy(response.get(), bandpass.data(), response.bytes(), cudaMemcpyHostToDevice));
            check_fft(cufftPlan1d(&fft, static_cast<int>(nfft), CUFFT_C2C, 1));
            try {
                check_fft(cufftExecC2C(fft, reinterpret_cast<cufftComplex *>(response.get()),
                                      reinterpret_cast<cufftComplex *>(response.get()), CUFFT_FORWARD));
                CURF_CUDA_CHECK(cudaDeviceSynchronize());
            } catch (...) {
                cufftDestroy(fft);
                fft = 0;
                throw;
            }
        }
        ~Impl() { if (fft) cufftDestroy(fft); }

        void run(const cf32 *input, cf32 *output, std::uint64_t first, cudaStream_t stream) {
            if (!input || !output) throw std::invalid_argument("extraction: null device buffer");
            if (have_next && first != next) throw std::invalid_argument("extraction: gap; reset history before resuming");
            if (first > std::numeric_limits<std::uint64_t>::max() - cfg.block_frames)
                throw std::invalid_argument("extraction: sample counter overflow");
            const auto hist = cfg.history_frames();
            CURF_CUDA_CHECK(cudaMemsetAsync(work.get(), 0, work.bytes(), stream));
            CURF_CUDA_CHECK(cudaMemcpyAsync(work.get(), history.get(), history.bytes(), cudaMemcpyDeviceToDevice, stream));
            CURF_CUDA_CHECK(cudaMemcpyAsync(work.get() + hist, input, cfg.block_frames * sizeof(cf32), cudaMemcpyDeviceToDevice, stream));
            // Save the last H raw samples before the FFT overwrites the workspace. This also handles H > block.
            CURF_CUDA_CHECK(cudaMemcpyAsync(history.get(), work.get() + cfg.block_frames, history.bytes(), cudaMemcpyDeviceToDevice, stream));
            check_fft(cufftSetStream(fft, stream));
            auto *p = reinterpret_cast<cufftComplex *>(work.get());
            check_fft(cufftExecC2C(fft, p, p, CUFFT_FORWARD));
            multiply<<<static_cast<unsigned>((nfft + 255) / 256), 256, 0, stream>>>(work.get(), response.get(), nfft);
            CURF_CUDA_CHECK(cudaGetLastError());
            check_fft(cufftExecC2C(fft, p, p, CUFFT_INVERSE));
            const long double cycles = static_cast<long double>(cfg.centre_hz) / cfg.sample_rate;
            const double start = extraction_phase_cycles(cfg, first);
            decimate<<<static_cast<unsigned>((cfg.output_frames() + 255) / 256), 256, 0, stream>>>(
                work.get(), output, cfg.output_frames(), hist, cfg.decimation, 1.f / static_cast<float>(nfft),
                static_cast<double>(cycles), start);
            CURF_CUDA_CHECK(cudaGetLastError());
            next = first + cfg.block_frames;
            have_next = true;
        }
    };

    SignalExtractor::SignalExtractor(const ExtractionConfig &cfg) : impl_(std::make_unique<Impl>(cfg)) {}
    SignalExtractor::~SignalExtractor() = default;
    void SignalExtractor::run(const cf32 *input, cf32 *output, std::uint64_t first, cudaStream_t stream) {
        impl_->run(input, output, first, stream);
    }
    void SignalExtractor::reset(cudaStream_t stream) {
        CURF_CUDA_CHECK(cudaMemsetAsync(impl_->history.get(), 0, impl_->history.bytes(), stream));
        impl_->have_next = false;
    }
    std::size_t SignalExtractor::fft_frames() const { return impl_->nfft; }
} // namespace curf::cuda
