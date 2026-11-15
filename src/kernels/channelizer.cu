#include "curf/cuda/channelizer.hpp"
#include "curf/cuda/memory.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace curf::cuda {

    namespace {

        void cufft_check(cufftResult r, const char *what) {
            if (r != CUFFT_SUCCESS) throw std::runtime_error(std::string(what) + " failed: cufftResult " + std::to_string(r));
        }

        // Wire sample decode, identical to curf::ref::unpack_sample.
        __device__ __forceinline__ float2 decode(const std::uint8_t *p, WireFormat f) {
            if (f == WireFormat::packed12) {
                const int b0 = p[0], b1 = p[1], b2 = p[2];
                int i = (b0 << 4) | (b1 >> 4), q = ((b1 & 0xF) << 8) | b2;
                i = (i ^ 0x800) - 0x800; // sign-extend 12 bits
                q = (q ^ 0x800) - 0x800;
                return make_float2(static_cast<float>(i) * (1.f / 2048.f), static_cast<float>(q) * (1.f / 2048.f));
            }
            return make_float2(static_cast<float>(static_cast<std::int8_t>(p[0])) * (1.f / 128.f),
                               static_cast<float>(static_cast<std::int8_t>(p[1])) * (1.f / 128.f));
        }

        struct Src {
            const float2 *planar;     // [streams][samples] or null
            const std::uint8_t *wire; // interleaved frames or null
            WireFormat format;
            unsigned wire_bytes, streams;
            const float2 *hist;       // [streams][L]
            long long samples;
            unsigned L;
            __device__ __forceinline__ float2 load(unsigned s, long long i) const {
                if (i < 0) return hist[static_cast<long long>(s) * L + L + i];
                if (planar) return planar[static_cast<long long>(s) * samples + i];
                return decode(wire + (i * streams + s) * wire_bytes, format);
            }
        };

        // One thread per polyphase branch m (its P taps in registers), FPB consecutive frames per block.
        //   u[m] = sum_p h[m + pM] * x[nD - m - pM];  odd global frames are stored rotated by M/2.
        // Consecutive threads read consecutive samples (descending), and a sample is re-read by L/D frames,
        // mostly from L1.
        template <int P>
        __global__ void fold_kernel(Src src, const float *__restrict__ h, float2 *__restrict__ out, unsigned M, unsigned D,
                                    unsigned frames, unsigned fpb, unsigned long long frame0, long long stride) {
            const unsigned s = blockIdx.y;
            const unsigned n0 = blockIdx.x * fpb;
            for (unsigned m = threadIdx.x; m < M; m += blockDim.x) {
                float hr[P];
#pragma unroll
                for (int p = 0; p < P; ++p) hr[p] = h[m + p * M];
                for (unsigned n = n0; n < n0 + fpb && n < frames; ++n) {
                    const long long i = static_cast<long long>(n) * D - m;
                    float2 acc = make_float2(0.f, 0.f);
#pragma unroll
                    for (int p = 0; p < P; ++p) {
                        const float2 v = src.load(s, i - static_cast<long long>(p) * M);
                        acc.x = fmaf(hr[p], v.x, acc.x);
                        acc.y = fmaf(hr[p], v.y, acc.y);
                    }
                    const unsigned mo = ((frame0 + n) & 1ull) ? ((m + M / 2) & (M - 1)) : m;
                    out[static_cast<long long>(s) * stride + static_cast<long long>(n) * M + mo] = acc;
                }
            }
        }

        // Same fold, but the input segment the block's FPB frames need, (FPB - 1) * D + L samples, is first
        // decoded once into shared memory, then folded with FMAs only. The direct version above is
        // instruction-bound (ncu: SM 73%, DRAM 31%): every input sample is decoded again by each of the L / D
        // frames that use it, plus 64-bit index arithmetic per tap.
        template <int P>
        __global__ void fold_smem_kernel(Src src, const float *__restrict__ h, float2 *__restrict__ out, unsigned M,
                                         unsigned D, unsigned frames, unsigned fpb, unsigned long long frame0,
                                         long long stride) {
            extern __shared__ float2 xs[];
            const unsigned s = blockIdx.y, n0 = blockIdx.x * fpb, L = static_cast<unsigned>(P) * M;
            const unsigned nf = min(fpb, frames - n0);
            const long long lo = static_cast<long long>(n0) * D - (L - 1);
            const unsigned nseg = (nf - 1) * D + L;
            for (unsigned j = threadIdx.x; j < nseg; j += blockDim.x) xs[j] = src.load(s, lo + j);
            __syncthreads();
            for (unsigned m = threadIdx.x; m < M; m += blockDim.x) {
                float hr[P];
#pragma unroll
                for (int p = 0; p < P; ++p) hr[p] = h[m + p * M];
                for (unsigned f = 0; f < nf; ++f) {
                    const float2 *x = xs + f * D + (L - 1) - m; // x[p] at -p*M
                    float2 acc = make_float2(0.f, 0.f);
#pragma unroll
                    for (int p = 0; p < P; ++p) {
                        const float2 v = x[-static_cast<int>(p * M)];
                        acc.x = fmaf(hr[p], v.x, acc.x);
                        acc.y = fmaf(hr[p], v.y, acc.y);
                    }
                    const unsigned n = n0 + f;
                    const unsigned mo = ((frame0 + n) & 1ull) ? ((m + M / 2) & (M - 1)) : m;
                    out[static_cast<long long>(s) * stride + static_cast<long long>(n) * M + mo] = acc;
                }
            }
        }

        // hist[s][j] = x[samples - L + j]
        __global__ void history_kernel(Src src, float2 *hist, unsigned L) {
            const unsigned s = blockIdx.y;
            for (unsigned j = blockIdx.x * blockDim.x + threadIdx.x; j < L; j += gridDim.x * blockDim.x) {
                Src cur = src;
                cur.hist = nullptr;
                hist[static_cast<long long>(s) * L + j] = cur.load(s, src.samples - L + j);
            }
        }

        constexpr unsigned kFramesPerBlock = 8;

    } // namespace

    Channelizer::Channelizer(const ChannelizerConfig &cfg, std::uint32_t streams, std::size_t block_samples)
        : cfg_(cfg), streams_(streams), block_samples_(block_samples) {
        const auto M = cfg.channels, D = cfg.decimation();
        const auto h = design_prototype(cfg);
        if (streams == 0) throw std::invalid_argument("Channelizer: no streams");
        if (block_samples % D != 0) throw std::invalid_argument("Channelizer: block_samples must be a multiple of M/2");
        if (block_samples < h.size()) throw std::invalid_argument("Channelizer: block shorter than the prototype");
        if (cfg.taps_per_phase != 4 && cfg.taps_per_phase != 8 && cfg.taps_per_phase != 16)
            throw std::invalid_argument("Channelizer: taps_per_phase must be 4, 8 or 16");
        frames_ = block_samples / D;
        CURF_CUDA_CHECK(cudaMalloc(&d_h_, h.size() * sizeof(float)));
        CURF_CUDA_CHECK(cudaMemcpy(d_h_, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
        CURF_CUDA_CHECK(cudaMalloc(&d_hist_, std::size_t{streams} * h.size() * sizeof(cf32)));
        reset();
        int n[1] = {static_cast<int>(M)};
        cufft_check(cufftPlanMany(&plan_, 1, n, nullptr, 1, static_cast<int>(M), nullptr, 1, static_cast<int>(M),
                                  CUFFT_C2C, static_cast<int>(std::size_t{streams} * frames_)),
                    "cufftPlanMany");
    }

    Channelizer::~Channelizer() {
        if (plan_) cufftDestroy(plan_);
        if (plan_stream_) cufftDestroy(plan_stream_);
        cudaFree(d_h_);
        cudaFree(d_hist_);
    }

    void Channelizer::reset() {
        CURF_CUDA_CHECK(cudaMemset(d_hist_, 0, std::size_t{streams_} * cfg_.taps() * sizeof(cf32)));
        frames_done_ = 0;
    }

    double Channelizer::flops() const noexcept {
        const double frames = static_cast<double>(streams_) * static_cast<double>(frames_);
        const double M = cfg_.channels;
        return frames * (4.0 * static_cast<double>(cfg_.taps()) + 5.0 * M * std::log2(M));
    }

    void Channelizer::fold(const void *in, bool wire, WireFormat format, cf32 *d_out, cudaStream_t stream,
                           std::size_t stride) {
        if (stride == 0) stride = frames_ * cfg_.channels;
        if (stride < frames_ * cfg_.channels) throw std::invalid_argument("Channelizer: output stride too small");
        const auto ld = static_cast<long long>(stride);
        const unsigned M = cfg_.channels, D = cfg_.decimation(), L = static_cast<unsigned>(cfg_.taps());
        Src src{wire ? nullptr : static_cast<const float2 *>(in), wire ? static_cast<const std::uint8_t *>(in) : nullptr,
                format, static_cast<unsigned>(wire_sample_bytes(format)), streams_,
                reinterpret_cast<const float2 *>(d_hist_), static_cast<long long>(block_samples_), L};
        const dim3 grid(static_cast<unsigned>((frames_ + kFramesPerBlock - 1) / kFramesPerBlock), streams_);
        const unsigned threads = M < 256 ? M : 256;
        auto *out = reinterpret_cast<float2 *>(d_out);
        const auto frames = static_cast<unsigned>(frames_);
        if (variant_ == FoldVariant::direct) {
            switch (cfg_.taps_per_phase) {
                case 4: fold_kernel<4><<<grid, threads, 0, stream>>>(src, d_h_, out, M, D, frames, kFramesPerBlock, frames_done_, ld); break;
                case 8: fold_kernel<8><<<grid, threads, 0, stream>>>(src, d_h_, out, M, D, frames, kFramesPerBlock, frames_done_, ld); break;
                default: fold_kernel<16><<<grid, threads, 0, stream>>>(src, d_h_, out, M, D, frames, kFramesPerBlock, frames_done_, ld); break;
            }
        } else {
            const std::size_t smem = ((kFramesPerBlock - 1) * D + L) * sizeof(float2);
            auto launch = [&](auto kernel) {
                CURF_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
                kernel<<<grid, threads, smem, stream>>>(src, d_h_, out, M, D, frames, kFramesPerBlock, frames_done_, ld);
            };
            switch (cfg_.taps_per_phase) {
                case 4: launch(fold_smem_kernel<4>); break;
                case 8: launch(fold_smem_kernel<8>); break;
                default: launch(fold_smem_kernel<16>); break;
            }
        }
        CURF_CUDA_CHECK(cudaGetLastError());
        history_kernel<<<dim3((L + 255) / 256, streams_), 256, 0, stream>>>(src, reinterpret_cast<float2 *>(d_hist_), L);
        CURF_CUDA_CHECK(cudaGetLastError());
        if (stride == frames_ * cfg_.channels) {
            cufft_check(cufftSetStream(plan_, stream), "cufftSetStream");
            cufft_check(cufftExecC2C(plan_, reinterpret_cast<cufftComplex *>(d_out),
                                     reinterpret_cast<cufftComplex *>(d_out), CUFFT_INVERSE),
                        "cufftExecC2C");
        } else {
            if (!plan_stream_) {
                int n[1] = {static_cast<int>(M)};
                cufft_check(cufftPlanMany(&plan_stream_, 1, n, nullptr, 1, static_cast<int>(M), nullptr, 1,
                                          static_cast<int>(M), CUFFT_C2C, static_cast<int>(frames_)),
                            "cufftPlanMany");
            }
            cufft_check(cufftSetStream(plan_stream_, stream), "cufftSetStream");
            for (std::uint32_t s = 0; s < streams_; ++s) {
                auto *p = reinterpret_cast<cufftComplex *>(d_out + s * stride);
                cufft_check(cufftExecC2C(plan_stream_, p, p, CUFFT_INVERSE), "cufftExecC2C");
            }
        }
        frames_done_ += frames_;
    }

    void Channelizer::run(const cf32 *d_planar, cf32 *d_out, cudaStream_t stream, std::size_t stride) {
        fold(d_planar, false, WireFormat::packed12, d_out, stream, stride);
    }

    void Channelizer::run_wire(const std::uint8_t *d_wire, WireFormat format, cf32 *d_out, cudaStream_t stream,
                               std::size_t stride) {
        fold(d_wire, true, format, d_out, stream, stride);
    }

} // namespace curf::cuda
