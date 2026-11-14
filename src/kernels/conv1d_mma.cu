// Ladder step 5: implicit-GEMM conv1d with a cp.async multi-stage pipeline, ldmatrix and mma.sync.
//
// Differences from the WMMA kernel (step 3/4) and what each one buys:
//   - cp.async (global -> shared without registers), STAGES-deep ring of k-tiles, one __syncthreads per
//     k-tile: global latency hides behind the MMAs of the previous tiles
//   - im2col bookkeeping hoisted out of the main loop: each thread loads the same rows and the same
//     16-byte column of every k-tile, so (batch, position) is computed once and (tap, channel) is
//     advanced incrementally; no integer division per tile
//   - ldmatrix.x4 fragment loads from padded rows (80-byte stride, conflict-free) and raw
//     mma.sync.m16n8k16 with a larger per-warp tile (up to 64x32), so more MMAs per fragment load
//   - fused epilogue straight from the accumulator registers (bias + residual + ReLU, half2 stores)
//
// Requirements: fp16 input with cin % 8 == 0 (16-byte cp.async chunks never straddle a tap),
// cout % BN == 0, kpad % 32 == 0. The fp32-input stem (cin = 2) stays on the WMMA path.

#include "conv1d_internal.hpp"
#include "mma_common.cuh"

#include "curf/cuda/memory.hpp"

#include <string>
#include <type_traits>

namespace curf::engine::detail {

    namespace {

        template <int BM, int BN, int WM, int STAGES>
        struct MmaShape {
            static constexpr int BK = 32, LDS = BK + 8, WN = 32;
            static constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN, NT = WARPS_M * WARPS_N * 32;
            static constexpr int MI = WM / 16, NI = WN / 8;
            static constexpr int A_PER = BM * (BK / 8) / NT, B_PER = BN * (BK / 8) / NT;
            static constexpr int LDC = BN + 8; // epilogue staging tile [BM][LDC] halves, reuses the ring
            static constexpr std::size_t PIPE = static_cast<std::size_t>(STAGES) * (BM + BN) * LDS * sizeof(__half);
            static constexpr std::size_t STAGE_C = static_cast<std::size_t>(BM) * LDC * sizeof(__half);
            static constexpr std::size_t SMEM = PIPE > STAGE_C ? PIPE : STAGE_C;
            static_assert(BM * (BK / 8) % NT == 0 && BN * (BK / 8) % NT == 0, "tile loads must divide evenly");
            static_assert(A_PER >= 1 && B_PER >= 1);
        };

        template <int BM, int BN, int WM, int STAGES, bool F16Acc>
        __global__ void __launch_bounds__(MmaShape<BM, BN, WM, STAGES>::NT) conv_mma(ConvParams p) {
            using S = MmaShape<BM, BN, WM, STAGES>;
            constexpr int BK = S::BK, LDS = S::LDS, NT = S::NT, MI = S::MI, NI = S::NI;
            extern __shared__ __align__(16) unsigned char smem[];
            __half *As = reinterpret_cast<__half *>(smem);   // [STAGES][BM][LDS]
            __half *Bs = As + STAGES * BM * LDS;              // [STAGES][BN][LDS]

            const auto *x = static_cast<const __half *>(p.x);
            const long long M = static_cast<long long>(p.batch) * p.lout;
            const long long m0 = static_cast<long long>(blockIdx.x) * BM;
            const int n0 = blockIdx.y * BN;
            const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
            const int wm = warp / S::WARPS_N, wn = warp % S::WARPS_N;

            // ---- loader state: thread loads rows tid/4 + i*NT/4, 16-byte column kc of every k-tile
            const int kc = (tid % 4) * 8;
            const __half *a_row[S::A_PER];
            int a_pos0[S::A_PER];
#pragma unroll
            for (int i = 0; i < S::A_PER; ++i) {
                const long long m = m0 + tid / 4 + i * (NT / 4);
                if (m < M) {
                    const int b = static_cast<int>(m / p.lout), l = static_cast<int>(m - static_cast<long long>(b) * p.lout);
                    a_row[i] = x + static_cast<long long>(b) * p.lin * p.cin;
                    a_pos0[i] = l * p.stride - p.pad;
                } else {
                    a_row[i] = x;
                    a_pos0[i] = -(1 << 30); // never in range
                }
            }
            int tap = kc / p.cin, ci = kc - (kc / p.cin) * p.cin;
            const __half *w_row = p.w + static_cast<long long>(n0 + tid / 4) * p.kpad + kc;
            const long long w_step = static_cast<long long>(NT / 4) * p.kpad;

            auto load_tile = [&](int stage, int k0) {
                __half *as = As + stage * BM * LDS;
                __half *bs = Bs + stage * BN * LDS;
#pragma unroll
                for (int i = 0; i < S::A_PER; ++i) {
                    const int pos = a_pos0[i] + tap;
                    const bool ok = tap < p.k && pos >= 0 && pos < p.lin;
                    const __half *src = ok ? a_row[i] + static_cast<long long>(pos) * p.cin + ci : x;
                    cp_async16_ca(smem_addr(as + (tid / 4 + i * (NT / 4)) * LDS + kc), src, ok);
                }
#pragma unroll
                for (int i = 0; i < S::B_PER; ++i)
                    cp_async16(smem_addr(bs + (tid / 4 + i * (NT / 4)) * LDS + kc), w_row + i * w_step + k0, true);
                ci += BK;
                while (ci >= p.cin) { ci -= p.cin; ++tap; }
            };

            using Acc = std::conditional_t<F16Acc, unsigned, float>;
            Acc acc[MI][NI][F16Acc ? 2 : 4];
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < NI; ++j)
#pragma unroll
                    for (int e = 0; e < (F16Acc ? 2 : 4); ++e) acc[i][j][e] = Acc(0);

            const int KT = p.kpad / BK;
#pragma unroll
            for (int s = 0; s < STAGES - 1; ++s) {
                if (s < KT) load_tile(s, s * BK);
                cp_async_commit();
            }

            // ldmatrix lane addressing (see PTX fragment layouts for m16n8k16)
            const int a_r = (lane % 8) + ((lane / 8) % 2) * 8, a_c = (lane / 16) * 8;
            const int b_r = (lane % 8) + (lane / 16) * 8, b_c = ((lane / 8) % 2) * 8;

            for (int kt = 0; kt < KT; ++kt) {
                cp_async_wait<STAGES - 2>();
                __syncthreads();
                if (const int nk = kt + STAGES - 1; nk < KT) load_tile(nk % STAGES, nk * BK);
                cp_async_commit();

                const __half *as = As + (kt % STAGES) * BM * LDS + (wm * WM) * LDS;
                const __half *bs = Bs + (kt % STAGES) * BN * LDS + (wn * 32) * LDS;
#pragma unroll
                for (int ks = 0; ks < BK; ks += 16) {
                    unsigned af[MI][4], bf[NI][2];
#pragma unroll
                    for (int i = 0; i < MI; ++i) ldmatrix_x4(af[i], smem_addr(as + (i * 16 + a_r) * LDS + ks + a_c));
#pragma unroll
                    for (int j = 0; j < NI / 2; ++j) {
                        unsigned r[4];
                        ldmatrix_x4(r, smem_addr(bs + (j * 16 + b_r) * LDS + ks + b_c));
                        bf[2 * j][0] = r[0];
                        bf[2 * j][1] = r[1];
                        bf[2 * j + 1][0] = r[2];
                        bf[2 * j + 1][1] = r[3];
                    }
#pragma unroll
                    for (int i = 0; i < MI; ++i)
#pragma unroll
                        for (int j = 0; j < NI; ++j) mma(acc[i][j], af[i], bf[j]);
                }
            }
            cp_async_wait<0>();
            __syncthreads(); // the ring is reused as the staging tile below

            // ---- fused epilogue, two passes through shared memory
            // pass 1 (registers -> smem): + bias, and ReLU when there is no residual; one rounding to fp16
            __half *Cs = reinterpret_cast<__half *>(smem);
            const bool relu_now = p.relu && p.res == nullptr;
#pragma unroll
            for (int j = 0; j < NI; ++j) {
                const int c = wn * 32 + j * 8 + (lane % 4) * 2;
                const float2 bias = *reinterpret_cast<const float2 *>(p.bias + n0 + c);
#pragma unroll
                for (int i = 0; i < MI; ++i) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int r = wm * WM + i * 16 + lane / 4 + h * 8;
                        float2 v = acc_pair(acc[i][j], h);
                        v.x += bias.x;
                        v.y += bias.y;
                        if (relu_now) {
                            v.x = fmaxf(v.x, 0.f);
                            v.y = fmaxf(v.y, 0.f);
                        }
                        *reinterpret_cast<__half2 *>(Cs + r * S::LDC + c) = __floats2half2_rn(v.x, v.y);
                    }
                }
            }
            __syncthreads();
            // pass 2 (smem -> global): 16-byte chunks, consecutive threads cover consecutive bytes of a row;
            // residual read with the same 16-byte pattern, then ReLU
            constexpr int CPR = BN / 8; // chunks per row
            for (int c = tid; c < BM * CPR; c += NT) {
                const int r = c / CPR, col = (c % CPR) * 8;
                const long long m = m0 + r;
                if (m >= M) continue;
                uint4 v = *reinterpret_cast<const uint4 *>(Cs + r * S::LDC + col);
                if (p.res) v = add_relu8(v, *reinterpret_cast<const uint4 *>(p.res + m * p.cout + n0 + col), p.relu);
                if (p.pool) *reinterpret_cast<uint4 *>(Cs + r * S::LDC + col) = v; // finished values, pooled below
                else *reinterpret_cast<uint4 *>(p.y + m * p.cout + n0 + col) = v;
            }
            if (p.pool) { // fused global average pool: this conv's output is never written
                __syncthreads();
                pool_tile<BM, BN, NT>(Cs, S::LDC, m0, M, p.lout, p.cout, n0, p.pool, tid);
            }
        }

        template <int BM, int BN, int WM, int STAGES, bool F16Acc>
        void launch_mma(const ConvParams &p, cudaStream_t s) {
            using S = MmaShape<BM, BN, WM, STAGES>;
            static const bool configured = [] {
                CURF_CUDA_CHECK(cudaFuncSetAttribute(conv_mma<BM, BN, WM, STAGES, F16Acc>,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     static_cast<int>(S::SMEM)));
                return true;
            }();
            (void)configured;
            const dim3 grid(static_cast<unsigned>((p.m() + BM - 1) / BM), static_cast<unsigned>(p.cout / BN));
            conv_mma<BM, BN, WM, STAGES, F16Acc><<<grid, S::NT, S::SMEM, s>>>(p);
        }

        template <int BN> bool sup_mma(const ConvParams &p) {
            return !p.x_f32 && p.cin % 8 == 0 && p.cout % BN == 0 && p.kpad % 32 == 0;
        }

        template <int BM, int BN, int WM, int STAGES> void add(std::vector<ConvVariant> &v) {
            const std::string tile = std::to_string(BM) + "x" + std::to_string(BN) + ".w" + std::to_string(WM) + ".s" +
                                     std::to_string(STAGES);
            v.push_back({"mma.f32." + tile, true, &sup_mma<BN>, &launch_mma<BM, BN, WM, STAGES, false>, true});
            v.push_back({"mma.f16." + tile, true, &sup_mma<BN>, &launch_mma<BM, BN, WM, STAGES, true>, true});
        }

    } // namespace

    void add_mma_variants(std::vector<ConvVariant> &v) {
        add<128, 32, 32, 2>(v);
        add<128, 32, 32, 3>(v);
        add<64, 64, 32, 2>(v);
        add<64, 64, 32, 3>(v);
        add<128, 64, 32, 3>(v);
        add<128, 64, 64, 3>(v);
        add<128, 64, 64, 4>(v);
        add<128, 64, 32, 2>(v);
        add<64, 128, 32, 2>(v);
        add<64, 128, 32, 3>(v);
        add<128, 128, 64, 2>(v);
        add<128, 128, 64, 3>(v);
        add<128, 128, 64, 4>(v);
        add<256, 64, 64, 3>(v);
    }

} // namespace curf::engine::detail
