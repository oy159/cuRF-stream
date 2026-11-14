#include "curf/cuda/memory.hpp"
#include "curf/cuda/psd.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace curf::cuda {
    namespace {

        void cufft_check(cufftResult r, const char *what) {
            if (r != CUFFT_SUCCESS) throw std::runtime_error(std::string(what) + " failed: cufftResult " + std::to_string(r));
        }

        // scratch[c][s][n] = planar[c][s*nfft + n] * window[n]  (same layout, so it is a
        // flat element-wise multiply with the window index = i % nfft)
        __global__ void window_kernel(const float2 *__restrict__ in, float2 *__restrict__ out,
                                      const float *__restrict__ window, std::size_t total, unsigned nfft_mask) {
            for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < total;
                 i += std::size_t{gridDim.x} * blockDim.x) {
                const float w = window[i & nfft_mask];
                const float2 v = in[i];
                out[i] = make_float2(v.x * w, v.y * w);
            }
        }

        // psd[c][shift(k)] = norm * sum_s |X[c][s][k]|^2 ; one thread per (c, k),
        // consecutive threads read consecutive k (coalesced).
        __global__ void accumulate_kernel(const float2 *__restrict__ spec, float *__restrict__ psd, unsigned channels,
                                          unsigned segments, unsigned nfft, float norm) {
            const unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned c = blockIdx.y;
            if (k >= nfft || c >= channels) return;
            const float2 *p = spec + std::size_t{c} * segments * nfft + k;
            float acc = 0.0f;
            for (unsigned s = 0; s < segments; ++s) {
                const float2 v = p[std::size_t{s} * nfft];
                acc = fmaf(v.x, v.x, fmaf(v.y, v.y, acc));
            }
            psd[std::size_t{c} * nfft + ((k + nfft / 2) & (nfft - 1))] = acc * norm;
        }

    } // namespace

    PsdStage::PsdStage(std::uint32_t channels, std::size_t frames, std::size_t nfft)
        : channels_(channels), frames_(frames), nfft_(nfft) {
        if (nfft == 0 || (nfft & (nfft - 1)) != 0) throw std::invalid_argument("PsdStage: nfft must be a power of two");
        if (frames % nfft != 0) throw std::invalid_argument("PsdStage: frames must be a multiple of nfft");

        std::vector<float> w(nfft);
        double sum = 0.0;
        for (std::size_t n = 0; n < nfft; ++n) {
            w[n] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * static_cast<double>(n) /
                                                           static_cast<double>(nfft)));
            sum += w[n];
        }
        // A tone of amplitude A at a bin centre gives |X|^2 = (A * sum w)^2 per segment.
        norm_ = static_cast<float>(1.0 / (sum * sum * static_cast<double>(segments())));

        CURF_CUDA_CHECK(cudaMalloc(&d_window_, nfft * sizeof(float)));
        CURF_CUDA_CHECK(cudaMemcpy(d_window_, w.data(), nfft * sizeof(float), cudaMemcpyHostToDevice));

        int n = static_cast<int>(nfft);
        cufft_check(cufftPlanMany(&plan_, 1, &n, nullptr, 1, n, nullptr, 1, n, CUFFT_C2C,
                                  static_cast<int>(channels * segments())),
                    "cufftPlanMany");
    }

    PsdStage::~PsdStage() {
        if (plan_) cufftDestroy(plan_);
        if (d_window_) cudaFree(d_window_);
    }

    void PsdStage::run(const cf32 *d_planar, cf32 *d_scratch, float *d_psd, cudaStream_t stream) {
        const std::size_t total = scratch_elems();
        auto *scratch = reinterpret_cast<float2 *>(d_scratch);
        const int grid = static_cast<int>(std::min<std::size_t>((total + 255) / 256, 16 * 32));
        window_kernel<<<grid, 256, 0, stream>>>(reinterpret_cast<const float2 *>(d_planar), scratch, d_window_, total,
                                                static_cast<unsigned>(nfft_ - 1));
        CURF_CUDA_CHECK(cudaGetLastError());

        cufft_check(cufftSetStream(plan_, stream), "cufftSetStream");
        cufft_check(cufftExecC2C(plan_, reinterpret_cast<cufftComplex *>(scratch),
                                 reinterpret_cast<cufftComplex *>(scratch), CUFFT_FORWARD),
                    "cufftExecC2C");

        const dim3 block(256), grid2(static_cast<unsigned>((nfft_ + 255) / 256), channels_);
        accumulate_kernel<<<grid2, block, 0, stream>>>(scratch, d_psd, channels_, static_cast<unsigned>(segments()),
                                                       static_cast<unsigned>(nfft_), norm_);
        CURF_CUDA_CHECK(cudaGetLastError());
    }

} // namespace curf::cuda
