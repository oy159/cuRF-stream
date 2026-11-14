// Ladder step 5: implicit-GEMM conv1d with a cp.async multi-stage pipeline, ldmatrix and mma.sync.
//
// Differences from the WMMA kernel (step 3/4) and what each one buys:
//   - cp.async (global -> shared without registers), STAGES-deep ring of k-tiles, one __syncthreads per
//     k-tile: global latency hides behind the MMAs of the previous tiles
//   - im2col bookkeeping hoisted out of the main loop: each thread loads the same rows and the same
//     16-byte column of every k-tile, so (batch, position) is computed once and (tap, channel) is
//     advanced incrementally; no integer division per tile
//   - ldmatrix.x4 fragment loads from padded rows (80-byte stride, conflict-free) and raw
//     mma.sync.m16n8k16 with a larger per-warp tile (up to 64x64), so more MMAs per fragment load
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

        // Shared-memory tile layout of a [rows][32 halves] k-tile, in units of halves.
        //   padded   (SWZ = false): row stride 40 halves (80 B, conflict-free for ldmatrix, 25% wasted)
        //   swizzled (SWZ = true):  row stride 32 halves, 16-byte chunk c of row r stored at c ^ ((r >> 1) & 3);
        //                           any 8 consecutive rows at one logical chunk hit 8 distinct bank groups
        // Swizzling was added after ncu showed the padded 3-stage 128x128 ring (61 KB) left one CTA per SM
        // (occupancy 8%), while CUTLASS's swizzled ring (48 KB) fits two.
        template <bool SWZ, int LDS> __device__ __forceinline__ int sidx(int row, int chunk) {
            return SWZ ? row * 32 + ((chunk ^ ((row >> 1) & 3)) << 3) : row * LDS + (chunk << 3);
        }

        template <int BM, int BN, int WM, int STAGES, int WN_ = 32, bool SWZ_ = false>
        struct MmaShape {
            static constexpr bool SWZ = SWZ_;
            static constexpr int BK = 32, LDS = SWZ ? BK : BK + 8, WN = WN_;
            static constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN, NT = WARPS_M * WARPS_N * 32;
            static constexpr int MI = WM / 16, NI = WN / 8;
            static constexpr int A_PER = BM * (BK / 8) / NT, B_PER = BN * (BK / 8) / NT;
            static constexpr int LDC = BN + 8; // epilogue staging tile [BM][LDC] halves, reuses the ring
            static constexpr std::size_t PIPE = static_cast<std::size_t>(STAGES) * (BM + BN) * LDS * sizeof(__half);
            static constexpr std::size_t STAGE_C = static_cast<std::size_t>(BM) * LDC * sizeof(__half);
            static constexpr std::size_t POOL = STAGE_C + static_cast<std::size_t>(NT / 32) * BN * sizeof(float); // + pool scratch
            static constexpr std::size_t SMEM = PIPE > POOL ? PIPE : POOL;
            static_assert(BM * (BK / 8) % NT == 0 && BN * (BK / 8) % NT == 0, "tile loads must divide evenly");
            static_assert(A_PER >= 1 && B_PER >= 1);
        };

        template <int BM, int BN, int WM, int STAGES, bool F16Acc, int WN = 32, bool SWZ = false>
        __global__ void __launch_bounds__(MmaShape<BM, BN, WM, STAGES, WN, SWZ>::NT) conv_mma(ConvParams p) {
            using S = MmaShape<BM, BN, WM, STAGES, WN, SWZ>;
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
                    cp_async16_ca(smem_addr(as + sidx<SWZ, LDS>(tid / 4 + i * (NT / 4), tid % 4)), src, ok);
                }
#pragma unroll
                for (int i = 0; i < S::B_PER; ++i)
                    cp_async16(smem_addr(bs + sidx<SWZ, LDS>(tid / 4 + i * (NT / 4), tid % 4)), w_row + i * w_step + k0, true);
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

            // Register double buffering of the fragments: while the MMAs of one k16 step run, the ldmatrix of
            // the next one (in this k-tile, or the first of the next k-tile right after the barrier) is in
            // flight. A first version loaded and consumed back to back and reached 60% tensor-pipe
            // utilisation where CUTLASS reached 85% on the same layer and occupancy.
            unsigned af[2][MI][4], bf[2][NI][2];
            auto load_frags = [&](int buf, int stage, int ks) {
                const __half *as = As + stage * BM * LDS;
                const __half *bs = Bs + stage * BN * LDS;
#pragma unroll
                for (int i = 0; i < MI; ++i)
                    ldmatrix_x4(af[buf][i], smem_addr(as + sidx<SWZ, LDS>(wm * WM + i * 16 + a_r, (ks + a_c) >> 3)));
#pragma unroll
                for (int j = 0; j < NI / 2; ++j) {
                    unsigned r[4];
                    ldmatrix_x4(r, smem_addr(bs + sidx<SWZ, LDS>(wn * WN + j * 16 + b_r, (ks + b_c) >> 3)));
                    bf[buf][2 * j][0] = r[0];
                    bf[buf][2 * j][1] = r[1];
                    bf[buf][2 * j + 1][0] = r[2];
                    bf[buf][2 * j + 1][1] = r[3];
                }
            };
            auto mma_all = [&](int buf) {
#pragma unroll
                for (int i = 0; i < MI; ++i)
#pragma unroll
                    for (int j = 0; j < NI; ++j) mma(acc[i][j], af[buf][i], bf[buf][j]);
            };

            if constexpr (SWZ) {
                // (swizzled variants) double-buffered fragments, see above
                cp_async_wait<STAGES - 2>(); // tile 0 has landed
                __syncthreads();
                load_frags(0, 0, 0);
                for (int kt = 0; kt < KT; ++kt) {
                    // The slot refilled here held tile kt - 1, fully read into registers before the last barrier.
                    if (const int nk = kt + STAGES - 1; nk < KT) load_tile(nk % STAGES, nk * BK);
                    cp_async_commit();
                    load_frags(1, kt % STAGES, 16);
                    mma_all(0);
                    cp_async_wait<STAGES - 2>(); // tile kt + 1 has landed
                    __syncthreads();
                    if (kt + 1 < KT) load_frags(0, (kt + 1) % STAGES, 0);
                    mma_all(1);
                }
            } else {
                // (padded variants) the original loop: load fragments and consume them back to back. Kept
                // because it is faster on single-k-tile layers (1x1 conv with K = 32: 141 vs 186 us).
                for (int kt = 0; kt < KT; ++kt) {
                    cp_async_wait<STAGES - 2>();
                    __syncthreads();
                    if (const int nk = kt + STAGES - 1; nk < KT) load_tile(nk % STAGES, nk * BK);
                    cp_async_commit();
#pragma unroll
                    for (int ks = 0; ks < BK; ks += 16) {
                        load_frags(0, kt % STAGES, ks);
                        mma_all(0);
                    }
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
                const int c = wn * WN + j * 8 + (lane % 4) * 2;
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
                pool_tile<BM, BN, NT>(Cs, S::LDC, m0, M, p.lout, p.cout, n0, p.pool, tid,
                                      reinterpret_cast<float *>(smem + S::STAGE_C));
            }
        }

        template <int BM, int BN, int WM, int STAGES, bool F16Acc, int WN = 32, bool SWZ = false>
        void launch_mma(const ConvParams &p, cudaStream_t s) {
            using S = MmaShape<BM, BN, WM, STAGES, WN, SWZ>;
            static const bool configured = [] {
                CURF_CUDA_CHECK(cudaFuncSetAttribute(conv_mma<BM, BN, WM, STAGES, F16Acc, WN, SWZ>,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     static_cast<int>(S::SMEM)));
                return true;
            }();
            (void)configured;
            const dim3 grid(static_cast<unsigned>((p.m() + BM - 1) / BM), static_cast<unsigned>(p.cout / BN));
            conv_mma<BM, BN, WM, STAGES, F16Acc, WN, SWZ><<<grid, S::NT, S::SMEM, s>>>(p);
        }

        template <int BN> bool sup_mma(const ConvParams &p) {
            return !p.x_f32 && p.cin % 8 == 0 && p.cout % BN == 0 && p.kpad % 32 == 0;
        }

        template <int BM, int BN, int WM, int STAGES, int WN = 32, bool SWZ = false> void add(std::vector<ConvVariant> &v) {
            const std::string tile = std::to_string(BM) + "x" + std::to_string(BN) + ".w" + std::to_string(WM) +
                                     (WN == 32 ? "" : "x" + std::to_string(WN)) + ".s" + std::to_string(STAGES) +
                                     (SWZ ? ".z" : "");
            v.push_back({"mma.f32." + tile, true, &sup_mma<BN>, &launch_mma<BM, BN, WM, STAGES, false, WN, SWZ>, true});
            v.push_back({"mma.f16." + tile, true, &sup_mma<BN>, &launch_mma<BM, BN, WM, STAGES, true, WN, SWZ>, true});
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
        // 64x64 warp tiles (MI = 4, NI = 8: 32 MMAs per k16 step, each A fragment reused 8x and each B
        // fragment 4x). Added after the CUTLASS baseline beat the 64x32 tiles by 18% on stage3 conv2
        // (128 -> 128 channels, K = 384): its 128x128 CTA tile runs 64x64 warps.
        add<128, 128, 64, 3, 64>(v);
        add<128, 128, 64, 2, 64>(v);
        add<128, 64, 64, 3, 64>(v);
        // swizzled (unpadded) shared memory: same tiles, 20% less smem, more CTAs per SM
        add<128, 128, 64, 3, 64, true>(v);
        add<128, 128, 64, 4, 64, true>(v);
        add<128, 128, 64, 3, 32, true>(v);
        add<128, 64, 64, 3, 64, true>(v);
        add<128, 64, 32, 3, 32, true>(v);
        add<128, 64, 64, 4, 32, true>(v);
        add<64, 64, 32, 3, 32, true>(v);
        add<64, 128, 32, 3, 32, true>(v);
        add<128, 32, 32, 3, 32, true>(v);
    }

} // namespace curf::engine::detail
