// conv1d implicit-GEMM variants (see include/curf/engine/kernels.hpp for the GEMM mapping).

#include "curf/cuda/memory.hpp"
#include "curf/engine/kernels.hpp"
#include "conv1d_internal.hpp"
#include "mma_common.cuh"

#include <mma.h>

#include <algorithm>
#include <vector>

namespace curf::engine {

    namespace {

        using namespace nvcuda;

        __device__ __forceinline__ float to_f(float v) { return v; }
        __device__ __forceinline__ float to_f(__half v) { return __half2float(v); }

        // Scalar im2col read of A[m][kk]; m < M is the caller's job.
        template <class TIn>
        __device__ __forceinline__ float load_a(const ConvParams &p, const TIn *x, long long m, int kk) {
            if (kk >= p.k * p.cin) return 0.f;
            const int b = static_cast<int>(m / p.lout), l = static_cast<int>(m - static_cast<long long>(b) * p.lout);
            const int t = kk / p.cin, ci = kk - t * p.cin;
            const int pos = l * p.stride + t - p.pad;
            if (pos < 0 || pos >= p.lin) return 0.f;
            return to_f(x[(static_cast<long long>(b) * p.lin + pos) * p.cin + ci]);
        }

        __device__ __forceinline__ float epilogue(const ConvParams &p, long long m, int n, float v) {
            v += p.bias[n];
            if (p.res) v += __half2float(p.res[m * p.cout + n]);
            return p.relu ? fmaxf(v, 0.f) : v;
        }

        // ------------------------------------------------------------------ 1. naive
        // One thread per output element, K loop straight from global memory.
        template <class TIn, bool Fused>
        __global__ void conv_naive(ConvParams p) {
            const auto *x = static_cast<const TIn *>(p.x);
            const long long total = static_cast<long long>(p.batch) * p.lout * p.cout;
            for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < total;
                 i += static_cast<long long>(gridDim.x) * blockDim.x) {
                const int n = static_cast<int>(i % p.cout);
                const long long m = i / p.cout;
                const int b = static_cast<int>(m / p.lout), l = static_cast<int>(m % p.lout);
                const __half *w = p.w + static_cast<long long>(n) * p.kpad;
                float acc = 0.f;
                for (int t = 0; t < p.k; ++t) {
                    const int pos = l * p.stride + t - p.pad;
                    if (pos < 0 || pos >= p.lin) continue;
                    const TIn *xr = x + (static_cast<long long>(b) * p.lin + pos) * p.cin;
                    for (int ci = 0; ci < p.cin; ++ci) acc += to_f(xr[ci]) * __half2float(w[t * p.cin + ci]);
                }
                p.y[i] = __float2half(Fused ? epilogue(p, m, n, acc) : acc);
            }
        }

        // ------------------------------------------------------------------ 2. shared-memory tiling
        // 64x32 output tile per block, BK=32, 256 threads each computing 4x2 outputs with fp32 FMA.
        template <class TIn, bool Fused>
        __global__ void __launch_bounds__(256) conv_smem(ConvParams p) {
            constexpr int BM = 64, BN = 32, BK = 32;
            __shared__ float As[BK][BM + 1];
            __shared__ float Bs[BK][BN + 1];
            const auto *x = static_cast<const TIn *>(p.x);
            const long long M = static_cast<long long>(p.batch) * p.lout;
            const long long m0 = static_cast<long long>(blockIdx.x) * BM;
            const int n0 = blockIdx.y * BN;
            const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
            float acc[4][2] = {};
            for (int k0 = 0; k0 < p.kpad; k0 += BK) {
                for (int e = tid; e < BM * BK; e += 256) {
                    const int r = e / BK, c = e % BK;
                    const long long m = m0 + r;
                    As[c][r] = m < M ? load_a(p, x, m, k0 + c) : 0.f;
                }
                for (int e = tid; e < BN * BK; e += 256) {
                    const int r = e / BK, c = e % BK;
                    Bs[c][r] = __half2float(p.w[static_cast<long long>(n0 + r) * p.kpad + k0 + c]);
                }
                __syncthreads();
#pragma unroll
                for (int kk = 0; kk < BK; ++kk) {
                    float a[4], b[2];
#pragma unroll
                    for (int i = 0; i < 4; ++i) a[i] = As[kk][ty * 4 + i];
#pragma unroll
                    for (int j = 0; j < 2; ++j) b[j] = Bs[kk][tx * 2 + j];
#pragma unroll
                    for (int i = 0; i < 4; ++i)
#pragma unroll
                        for (int j = 0; j < 2; ++j) acc[i][j] += a[i] * b[j];
                }
                __syncthreads();
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const long long m = m0 + ty * 4 + i;
                if (m >= M) continue;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int n = n0 + tx * 2 + j;
                    p.y[m * p.cout + n] = __float2half(Fused ? epilogue(p, m, n, acc[i][j]) : acc[i][j]);
                }
            }
        }

        // ------------------------------------------------------------------ 3/4. WMMA tensor cores
        // BMxBN block tile, BK=32, one warp per 32x32 sub-tile (2x2 16x16x16 fragments).
        // A chunks of 8 consecutive k (16-byte vectors) when cin % 8 == 0, else scalar gathers.
        // Fused: bias + residual + ReLU applied while the accumulators go back to global memory.

        // 8 consecutive A elements (k0..k0+7) of row m as 8 halves.
        template <class TIn>
        __device__ __forceinline__ uint4 load_a8(const ConvParams &p, const TIn *x, long long m, int kk) {
            uint4 out = make_uint4(0, 0, 0, 0);
            const int b = static_cast<int>(m / p.lout), l = static_cast<int>(m - static_cast<long long>(b) * p.lout);
            if (p.cin % 8 == 0) {
                if (kk >= p.k * p.cin) return out;
                const int t = kk / p.cin, ci = kk - t * p.cin;
                const int pos = l * p.stride + t - p.pad;
                if (pos < 0 || pos >= p.lin) return out;
                const TIn *src = x + (static_cast<long long>(b) * p.lin + pos) * p.cin + ci;
                if constexpr (sizeof(TIn) == 2) {
                    return *reinterpret_cast<const uint4 *>(src);
                } else {
                    const float4 lo = reinterpret_cast<const float4 *>(src)[0];
                    const float4 hi = reinterpret_cast<const float4 *>(src)[1];
                    __half2 h[4] = {__floats2half2_rn(lo.x, lo.y), __floats2half2_rn(lo.z, lo.w),
                                    __floats2half2_rn(hi.x, hi.y), __floats2half2_rn(hi.z, hi.w)};
                    return *reinterpret_cast<uint4 *>(h);
                }
            }
            __half h[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int kj = kk + j;
                float v = 0.f;
                if (kj < p.k * p.cin) {
                    const int t = kj / p.cin, ci = kj - t * p.cin;
                    const int pos = l * p.stride + t - p.pad;
                    if (pos >= 0 && pos < p.lin) v = to_f(x[(static_cast<long long>(b) * p.lin + pos) * p.cin + ci]);
                }
                h[j] = __float2half(v);
            }
            return *reinterpret_cast<uint4 *>(h);
        }

        template <class TIn, int BM, int BN, bool Fused, class AccT>
        __global__ void __launch_bounds__(BM / 32 * BN / 32 * 32) conv_wmma(ConvParams p) {
            constexpr int BK = 32, LDS = BK + 8, LDC = 24;
            constexpr int WARPS_N = BN / 32, NWARPS = BM / 32 * WARPS_N, NT = NWARPS * 32;
            __shared__ __align__(128) __half As[BM * LDS];
            __shared__ __align__(128) __half Bs[BN * LDS];
            __shared__ __align__(128) AccT Cs[NWARPS][16 * LDC];

            const auto *x = static_cast<const TIn *>(p.x);
            const long long M = static_cast<long long>(p.batch) * p.lout;
            const long long m0 = static_cast<long long>(blockIdx.x) * BM;
            const int n0 = blockIdx.y * BN;
            const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
            const int wm = warp / WARPS_N, wn = warp % WARPS_N;

            wmma::fragment<wmma::accumulator, 16, 16, 16, AccT> acc[2][2];
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], AccT(0.f));

            for (int k0 = 0; k0 < p.kpad; k0 += BK) {
                for (int c = tid; c < BM * (BK / 8); c += NT) {
                    const int r = c / (BK / 8), kc = (c % (BK / 8)) * 8;
                    const long long m = m0 + r;
                    *reinterpret_cast<uint4 *>(&As[r * LDS + kc]) =
                        m < M ? load_a8(p, x, m, k0 + kc) : make_uint4(0, 0, 0, 0);
                }
                for (int c = tid; c < BN * (BK / 8); c += NT) {
                    const int r = c / (BK / 8), kc = (c % (BK / 8)) * 8;
                    *reinterpret_cast<uint4 *>(&Bs[r * LDS + kc]) =
                        *reinterpret_cast<const uint4 *>(p.w + static_cast<long long>(n0 + r) * p.kpad + k0 + kc);
                }
                __syncthreads();
#pragma unroll
                for (int kk = 0; kk < BK; kk += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2];
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[2];
#pragma unroll
                    for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &As[(wm * 32 + i * 16) * LDS + kk], LDS);
#pragma unroll
                    for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &Bs[(wn * 32 + j * 16) * LDS + kk], LDS);
#pragma unroll
                    for (int i = 0; i < 2; ++i)
#pragma unroll
                        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
                }
                __syncthreads();
            }

            // Epilogue: one 16x16 fragment at a time through a per-warp scratch tile; each lane writes
            // 8 consecutive channels of one row (16-byte stores).
            AccT *cs = Cs[warp];
            const int r = lane / 2, c8 = (lane % 2) * 8;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::store_matrix_sync(cs, acc[i][j], LDC, wmma::mem_row_major);
                    __syncwarp();
                    const long long m = m0 + wm * 32 + i * 16 + r;
                    const int n = n0 + wn * 32 + j * 16 + c8;
                    if (m < M) {
                        float v[8];
#pragma unroll
                        for (int e = 0; e < 8; ++e) v[e] = static_cast<float>(cs[r * LDC + c8 + e]);
                        if constexpr (Fused) {
                            const float4 b0 = *reinterpret_cast<const float4 *>(p.bias + n);
                            const float4 b1 = *reinterpret_cast<const float4 *>(p.bias + n + 4);
                            v[0] += b0.x; v[1] += b0.y; v[2] += b0.z; v[3] += b0.w;
                            v[4] += b1.x; v[5] += b1.y; v[6] += b1.z; v[7] += b1.w;
                            if (p.res) {
                                const uint4 rv = *reinterpret_cast<const uint4 *>(p.res + m * p.cout + n);
                                const __half2 *rh = reinterpret_cast<const __half2 *>(&rv);
#pragma unroll
                                for (int e = 0; e < 4; ++e) {
                                    const float2 f = __half22float2(rh[e]);
                                    v[2 * e] += f.x;
                                    v[2 * e + 1] += f.y;
                                }
                            }
                            if (p.relu)
#pragma unroll
                                for (int e = 0; e < 8; ++e) v[e] = fmaxf(v[e], 0.f);
                        }
                        __half2 o[4];
#pragma unroll
                        for (int e = 0; e < 4; ++e) o[e] = __floats2half2_rn(v[2 * e], v[2 * e + 1]);
                        *reinterpret_cast<uint4 *>(p.y + m * p.cout + n) = *reinterpret_cast<uint4 *>(o);
                    }
                    __syncwarp();
                }
            }
        }

        // ------------------------------------------------------------------ 7. direct conv (small cin)
        // For the stem (cin = 2, k = 7, fp32 input): K = 14 would waste 56% of a 32-wide MMA k-tile and the
        // layer is memory bound anyway (reads 8 KB, writes 32 KB per window). One block = TL output positions
        // of one sample, all channels; the input segment and the transposed weights sit in shared memory.
        // Each thread computes 4 positions x 8 channels, so one pair of float4 weight loads feeds 32 FMAs
        // (a first version did 1 x 8 and was shared-memory bound: L1 94%, slower than the WMMA path).
        // Positions are interleaved (pos = pg + q * TL/4) so a warp's 16-byte stores cover whole rows.
        // Tile (TL positions), positions per thread (Q) and a minimum of resident blocks per SM (register cap)
        // are template parameters: the first version (TL 256, Q 4, 69 registers, 3 blocks / SM) reached only
        // 53% of DRAM bandwidth because each block's load -> compute -> store runs serially and too few blocks
        // were resident to overlap them.
        template <class TIn, int TL, int Q, int MINB>
        __global__ void __launch_bounds__(256, MINB) conv_direct(ConvParams p) {
            constexpr int PG = TL / Q;
            extern __shared__ float sm[];
            const int K = p.k * p.cin;
            const int rows = (TL - 1) * p.stride + p.k;
            float *xs = sm;                              // [rows][cin]
            float *ws = sm + ((rows * p.cin + 3) & ~3);  // [K][cout], 16-byte aligned for float4 reads
            const auto *x = static_cast<const TIn *>(p.x);
            const int b = blockIdx.y, l0 = blockIdx.x * TL;
            const int p0 = l0 * p.stride - p.pad;
            for (int i = threadIdx.x; i < rows * p.cin; i += blockDim.x) {
                const int r = i / p.cin, ci = i - r * p.cin, pos = p0 + r;
                xs[i] = pos >= 0 && pos < p.lin ? to_f(x[(static_cast<long long>(b) * p.lin + pos) * p.cin + ci]) : 0.f;
            }
            for (int i = threadIdx.x; i < K * p.cout; i += blockDim.x) {
                const int kk = i / p.cout, n = i - kk * p.cout;
                ws[i] = __half2float(p.w[static_cast<long long>(n) * p.kpad + kk]);
            }
            __syncthreads();
            const int groups = p.cout / 8;
            const int xstep = PG * p.stride * p.cin; // xs offset between the thread's positions
            for (int item = threadIdx.x; item < PG * groups; item += blockDim.x) {
                const int pg = item / groups, n = (item - pg * groups) * 8;
                float acc[Q][8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float bv = p.bias[n + e];
#pragma unroll
                    for (int q = 0; q < Q; ++q) acc[q][e] = bv;
                }
                const float *xr = xs + pg * p.stride * p.cin;
                for (int kk = 0; kk < K; ++kk) {
                    const float4 w0 = *reinterpret_cast<const float4 *>(ws + kk * p.cout + n);
                    const float4 w1 = *reinterpret_cast<const float4 *>(ws + kk * p.cout + n + 4);
#pragma unroll
                    for (int q = 0; q < Q; ++q) {
                        const float xv = xr[q * xstep + kk]; // row (pos*stride + t), channel ci: contiguous in kk
                        acc[q][0] += xv * w0.x; acc[q][1] += xv * w0.y; acc[q][2] += xv * w0.z; acc[q][3] += xv * w0.w;
                        acc[q][4] += xv * w1.x; acc[q][5] += xv * w1.y; acc[q][6] += xv * w1.z; acc[q][7] += xv * w1.w;
                    }
                }
#pragma unroll
                for (int q = 0; q < Q; ++q) {
                    const int l = l0 + pg + q * PG;
                    if (l >= p.lout) continue;
                    const long long m = static_cast<long long>(b) * p.lout + l;
                    __half2 o[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) o[e] = __floats2half2_rn(acc[q][2 * e], acc[q][2 * e + 1]);
                    uint4 v = *reinterpret_cast<uint4 *>(o);
                    if (p.res) v = detail::add_relu8(v, *reinterpret_cast<const uint4 *>(p.res + m * p.cout + n), p.relu);
                    else if (p.relu) v = detail::add_relu8(v, make_uint4(0, 0, 0, 0), true);
                    *reinterpret_cast<uint4 *>(p.y + m * p.cout + n) = v;
                }
            }
        }

        std::size_t direct_smem(const ConvParams &p, int TL = 512) {
            return (static_cast<std::size_t>(TL - 1) * p.stride + p.k) * p.cin * sizeof(float) + 16 +
                   static_cast<std::size_t>(p.k) * p.cin * p.cout * sizeof(float);
        }

        // ------------------------------------------------------------------ separate epilogue
        __global__ void conv_epilogue_kernel(ConvParams p) {
            const long long total2 = static_cast<long long>(p.batch) * p.lout * p.cout / 2;
            auto *y = reinterpret_cast<__half2 *>(p.y);
            const auto *res = reinterpret_cast<const __half2 *>(p.res);
            for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < total2;
                 i += static_cast<long long>(gridDim.x) * blockDim.x) {
                const int n = static_cast<int>((2 * i) % p.cout);
                float2 v = __half22float2(y[i]);
                v.x += p.bias[n];
                v.y += p.bias[n + 1];
                if (res) {
                    const float2 r = __half22float2(res[i]);
                    v.x += r.x;
                    v.y += r.y;
                }
                if (p.relu) {
                    v.x = fmaxf(v.x, 0.f);
                    v.y = fmaxf(v.y, 0.f);
                }
                y[i] = __floats2half2_rn(v.x, v.y);
            }
        }

        int grid_stride_blocks() {
            static const int blocks = [] {
                int dev = 0, sms = 0;
                CURF_CUDA_CHECK(cudaGetDevice(&dev));
                CURF_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
                return sms * 16;
            }();
            return blocks;
        }

        // ------------------------------------------------------------------ launchers
        template <bool Fused> void launch_naive(const ConvParams &p, cudaStream_t s) {
            const long long total = p.m() * p.cout;
            const int blocks = static_cast<int>(std::min<long long>((total + 255) / 256, grid_stride_blocks()));
            if (p.x_f32) conv_naive<float, Fused><<<blocks, 256, 0, s>>>(p);
            else conv_naive<__half, Fused><<<blocks, 256, 0, s>>>(p);
        }
        template <bool Fused> void launch_smem(const ConvParams &p, cudaStream_t s) {
            const dim3 grid(static_cast<unsigned>((p.m() + 63) / 64), static_cast<unsigned>(p.cout / 32));
            if (p.x_f32) conv_smem<float, Fused><<<grid, 256, 0, s>>>(p);
            else conv_smem<__half, Fused><<<grid, 256, 0, s>>>(p);
        }
        template <int TL, int Q, int MINB> void launch_direct(const ConvParams &p, cudaStream_t s) {
            const dim3 grid(static_cast<unsigned>((p.lout + TL - 1) / TL), static_cast<unsigned>(p.batch));
            if (p.x_f32) conv_direct<float, TL, Q, MINB><<<grid, 256, direct_smem(p, TL), s>>>(p);
            else conv_direct<__half, TL, Q, MINB><<<grid, 256, direct_smem(p, TL), s>>>(p);
        }
        // Direct conv only pays off for a handful of input channels.
        bool sup_direct(const ConvParams &p) {
            return p.cin <= 8 && p.cout % 8 == 0 && direct_smem(p) <= 48 * 1024;
        }

        template <int BM, int BN, bool Fused, class AccT> void launch_wmma(const ConvParams &p, cudaStream_t s) {
            const dim3 grid(static_cast<unsigned>((p.m() + BM - 1) / BM), static_cast<unsigned>(p.cout / BN));
            constexpr int nt = BM / 32 * BN / 32 * 32;
            if (p.x_f32) conv_wmma<float, BM, BN, Fused, AccT><<<grid, nt, 0, s>>>(p);
            else conv_wmma<__half, BM, BN, Fused, AccT><<<grid, nt, 0, s>>>(p);
        }

        bool sup_any(const ConvParams &p) { return p.cout % 2 == 0; }
        bool sup_n32(const ConvParams &p) { return p.cout % 32 == 0; }
        template <int BN> bool sup_wmma(const ConvParams &p) { return p.cout % BN == 0 && p.kpad % 32 == 0; }

        template <int BM, int BN> void add_wmma(std::vector<ConvVariant> &v) {
            const std::string tile = std::to_string(BM) + "x" + std::to_string(BN);
            v.push_back({"wmma.f32.unfused." + tile, false, &sup_wmma<BN>, &launch_wmma<BM, BN, false, float>});
            v.push_back({"wmma.f32.fused." + tile, true, &sup_wmma<BN>, &launch_wmma<BM, BN, true, float>});
            v.push_back({"wmma.f16.fused." + tile, true, &sup_wmma<BN>, &launch_wmma<BM, BN, true, __half>});
        }

        std::vector<ConvVariant> make_registry() {
            std::vector<ConvVariant> v;
            v.push_back({"naive", false, &sup_any, &launch_naive<false>});
            v.push_back({"smem", false, &sup_n32, &launch_smem<false>});
            add_wmma<64, 32>(v);
            add_wmma<128, 32>(v);
            add_wmma<64, 64>(v);
            add_wmma<128, 64>(v);
            add_wmma<64, 128>(v);
            add_wmma<128, 128>(v);
            detail::add_mma_variants(v);
            v.push_back({"direct.t256q4", true, &sup_direct, &launch_direct<256, 4, 3>});
            v.push_back({"direct.t256q4b4", true, &sup_direct, &launch_direct<256, 4, 4>});
            v.push_back({"direct.t128q2b6", true, &sup_direct, &launch_direct<128, 2, 6>});
            v.push_back({"direct.t256q2b6", true, &sup_direct, &launch_direct<256, 2, 6>});
            v.push_back({"direct.t512q4b4", true, &sup_direct, &launch_direct<512, 4, 4>});
#ifdef CURF_HAVE_CUDNN
            detail::add_cudnn_variants(v);
#endif
#ifdef CURF_HAVE_CUTLASS
            detail::add_cutlass_variants(v);
#endif
            return v;
        }

        // ------------------------------------------------------------------ head
        __global__ void gap_linear_kernel(const __half *x, const float *w, const float *bias, float *y, int length,
                                          int cin, int cout) {
            extern __shared__ float mean[];
            const int b = blockIdx.x;
            const __half *xb = x + static_cast<long long>(b) * length * cin;
            for (int c = threadIdx.x; c < cin; c += blockDim.x) {
                float s = 0.f;
                for (int l = 0; l < length; ++l) s += __half2float(xb[l * cin + c]);
                mean[c] = s / static_cast<float>(length);
            }
            __syncthreads();
            for (int n = threadIdx.x; n < cout; n += blockDim.x) {
                float s = bias[n];
                for (int c = 0; c < cin; ++c) s += w[n * cin + c] * mean[c];
                y[b * cout + n] = s;
            }
        }

        __global__ void linear_from_pool_kernel(const float *pool, const float *w, const float *bias, float *y, int cin,
                                                int cout, float inv_len) {
            extern __shared__ float mean[];
            const int b = blockIdx.x;
            for (int c = threadIdx.x; c < cin; c += blockDim.x) mean[c] = pool[b * cin + c] * inv_len;
            __syncthreads();
            for (int n = threadIdx.x; n < cout; n += blockDim.x) {
                float s = bias[n];
                for (int c = 0; c < cin; ++c) s += w[n * cin + c] * mean[c];
                y[b * cout + n] = s;
            }
        }

        __global__ void float_to_half_kernel(const float *src, __half *dst, std::size_t n) {
            for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
                 i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
                dst[i] = __float2half(src[i]);
        }

        __global__ void half_to_float_kernel(const __half *src, float *dst, std::size_t n) {
            for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
                 i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
                dst[i] = __half2float(src[i]);
        }

    } // namespace

    std::span<const ConvVariant> conv_variants() {
        static const std::vector<ConvVariant> registry = make_registry();
        return registry;
    }

    const ConvVariant *find_conv_variant(const std::string &name) {
        for (const auto &v : conv_variants())
            if (v.name == name) return &v;
        return nullptr;
    }

    void launch_conv(const ConvVariant &v, const ConvParams &p, cudaStream_t s) {
        v.launch(p, s);
        if (!v.fused) {
            const long long total2 = p.m() * p.cout / 2;
            const int blocks = static_cast<int>(std::min<long long>((total2 + 255) / 256, grid_stride_blocks()));
            conv_epilogue_kernel<<<blocks, 256, 0, s>>>(p);
        }
        CURF_CUDA_CHECK(cudaGetLastError());
    }

    void launch_gap_linear(const __half *x, const float *w, const float *bias, float *y, int batch, int length, int cin,
                           int cout, cudaStream_t s) {
        gap_linear_kernel<<<batch, 128, static_cast<std::size_t>(cin) * sizeof(float), s>>>(x, w, bias, y, length, cin,
                                                                                          cout);
        CURF_CUDA_CHECK(cudaGetLastError());
    }

    void launch_linear_from_pool(const float *pool, const float *w, const float *bias, float *y, int batch, int cin,
                                 int cout, float inv_len, cudaStream_t s) {
        linear_from_pool_kernel<<<batch, 128, static_cast<std::size_t>(cin) * sizeof(float), s>>>(pool, w, bias, y, cin,
                                                                                                cout, inv_len);
        CURF_CUDA_CHECK(cudaGetLastError());
    }

    void launch_float_to_half(const float *src, __half *dst, std::size_t n, cudaStream_t s) {
        float_to_half_kernel<<<static_cast<unsigned>(std::min<std::size_t>((n + 255) / 256, 1024)), 256, 0, s>>>(src, dst, n);
        CURF_CUDA_CHECK(cudaGetLastError());
    }

    void launch_half_to_float(const __half *src, float *dst, std::size_t n, cudaStream_t s) {
        half_to_float_kernel<<<static_cast<unsigned>(std::min<std::size_t>((n + 255) / 256, 1024)), 256, 0, s>>>(src, dst, n);
        CURF_CUDA_CHECK(cudaGetLastError());
    }

} // namespace curf::engine
