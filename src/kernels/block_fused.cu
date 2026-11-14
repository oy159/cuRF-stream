// Ladder step 8: one ResNet BasicBlock per kernel (cross-layer fusion).
//
//   y = relu(conv2(relu(conv1(x) + b1)) + b2 + shortcut(x))
//
// One CTA computes BL (64 or 128) output positions x all channels of one sample:
//   1. the x rows it needs (with halo) go to shared memory once (cp.async)
//   2. 1x1 shortcut (if any) runs first, accumulating straight into conv2's accumulators: the shortcut is
//      just extra K for conv2's GEMM, and bsc folds into the bias
//   3. conv1 for BL + 2 positions (one halo row each side for conv2's taps) -> bias + ReLU -> fp16 into
//      shared memory, overwriting x
//      (zero for positions outside [0, lout): that is conv2's zero padding)
//   4. conv2 reads its A operand from that tile: implicit im2col via per-lane ldmatrix row addresses
//   5. epilogue through shared memory, 16-byte stores (identity residual added there) or a fused
//      global-average-pool
// Weights stream through a 2-stage cp.async ring (they are re-read by every CTA, so they live in L2).
// Compared with three separate kernels this removes the write + read of the conv1 output and of the
// shortcut output, and the second read of x: per BasicBlock, DRAM traffic drops from ~5 activation
// tensors to ~2.
//
// Channel counts are compile-time (the shapes of this model); other shapes fall back to separate convs.

#include "curf/cuda/memory.hpp"
#include "curf/engine/kernels.hpp"
#include "mma_common.cuh"

#include <string>
#include <type_traits>
#include <vector>

namespace curf::engine {

    namespace {

        using namespace detail;

        // Warp jobs are 32x32 output tiles (MI = 2, NI = 4: 8 independent accumulators per k16 step). conv1
        // additionally has 2 halo rows (y1 positions l0 + BL and, via row 0, l0 - 1 is inside the main rows),
        // computed as one m16 x 32 "halo job" per 32 channels and dealt round-robin after the main jobs so the
        // warps stay balanced.
        template <int CIN, int COUT, int S, int BL_>
        struct BlockShape {
            static constexpr int BL = BL_, BK = 32;
            static constexpr int NC = COUT / 32, RP = BL / 32;                 // n32 chunks, m32 row pairs
            static constexpr int JOBS = RP * NC;                               // main jobs (both GEMMs)
            static constexpr int NW = JOBS < 4 ? 4 : (JOBS > 8 ? 8 : JOBS), NT = NW * 32;
            static constexpr int JM = (JOBS + NW - 1) / NW;                    // main jobs per warp
            static constexpr int NX = (BL + 1) * S + 3, LDX = CIN + 8;        // x rows: positions px0 ..
            static constexpr int NY = BL + 2, LDY = COUT + 8;                 // y1 rows: positions l0 - 1 ..
            static constexpr int LDW = BK + 8;
            static constexpr int REGION = NX * LDX > NY * LDY ? NX * LDX : NY * LDY; // x, then y1, then out
            static constexpr std::size_t SMEM = static_cast<std::size_t>(REGION + 2 * COUT * LDW) * sizeof(__half);
            static_assert(CIN % 32 == 0 && COUT % 32 == 0 && BL % 32 == 0, "tile shapes");
            static_assert(NC <= NW, "at most one halo job per warp");
            static_assert(2 * COUT * LDW * 2 >= NW * COUT * 4, "pool scratch must fit in the weight ring");
        };

        template <int CIN, int COUT, int S, int BL, bool SC, bool F16Acc>
        __global__ void __launch_bounds__(BlockShape<CIN, COUT, S, BL>::NT) basic_block(BlockParams p) {
            using B = BlockShape<CIN, COUT, S, BL>;
            constexpr int NT = B::NT, NW = B::NW, NC = B::NC, JM = B::JM, LDX = B::LDX, LDY = B::LDY, LDW = B::LDW;
            extern __shared__ __align__(16) unsigned char smem[];
            __half *R = reinterpret_cast<__half *>(smem); // x tile -> y1 tile -> output staging
            __half *Ws = R + B::REGION;                   // [2][COUT][LDW]

            const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
            const int b = blockIdx.y, l0 = blockIdx.x * BL;
            const int px0 = (l0 - 1) * S - 1;
            const __half *xb = p.x + static_cast<long long>(b) * p.lin * CIN;

            // ---- x tile (zero outside [0, lin): conv1's padding)
            for (int c = tid; c < B::NX * (CIN / 8); c += NT) {
                const int r = c / (CIN / 8), col = (c % (CIN / 8)) * 8, pos = px0 + r;
                const bool ok = pos >= 0 && pos < p.lin;
                cp_async16(smem_addr(R + r * LDX + col), ok ? xb + static_cast<long long>(pos) * CIN + col : p.x, ok);
            }

            // ---- weight k-tile stream: [shortcut] [conv1] [conv2]
            const int Tsc = SC ? p.kpadsc / 32 : 0, T1 = p.kpad1 / 32, T2 = p.kpad2 / 32, T = Tsc + T1 + T2;
            auto load_w = [&](int i) {
                const __half *w;
                int kp, k0;
                if (i < Tsc) { w = p.wsc; kp = p.kpadsc; k0 = i * 32; }
                else if (i < Tsc + T1) { w = p.w1; kp = p.kpad1; k0 = (i - Tsc) * 32; }
                else { w = p.w2; kp = p.kpad2; k0 = (i - Tsc - T1) * 32; }
                __half *ws = Ws + (i & 1) * COUT * LDW;
                for (int c = tid; c < COUT * 4; c += NT) {
                    const int n = c / 4, kc = (c % 4) * 8;
                    cp_async16(smem_addr(ws + n * LDW + kc), w + static_cast<long long>(n) * kp + k0 + kc, true);
                }
            };
            load_w(0);
            cp_async_commit();

            using Acc = std::conditional_t<F16Acc, unsigned, float>;
            constexpr int AR = F16Acc ? 2 : 4;
            Acc acc1[JM][2][4][AR], halo[1][4][AR], acc2[JM][2][4][AR];
            auto zero = [](auto &a) {
                Acc *f = &a[0][0][0];
#pragma unroll
                for (int i = 0; i < static_cast<int>(sizeof(a) / sizeof(Acc)); ++i) f[i] = Acc(0);
            };
#pragma unroll
            for (int j = 0; j < JM; ++j) {
                zero(acc1[j]);
                zero(acc2[j]);
            }
            zero(halo);

            const int a_r = (lane % 8) + ((lane / 8) % 2) * 8, a_c = (lane / 16) * 8;
            const int b_r = (lane % 8) + (lane / 16) * 8, b_c = ((lane / 8) % 2) * 8;
            // halo job of this warp (continues the round robin after the JOBS main jobs)
            const int my_halo = (warp - B::JOBS % NW + NW) % NW;

            // One k-tile of one warp job: rows r0 .. r0 + 16*MI - 1, channels nc*32 .. +31.
            // a_addr(row, k) = shared address of A elements (row, k .. k + 7).
            auto job = [&](auto &acc, auto mi_c, int r0, int nc, const __half *ws, int k0, auto a_addr) {
                constexpr int MI = decltype(mi_c)::value;
#pragma unroll
                for (int ks = 0; ks < 32; ks += 16) {
                    unsigned af[MI][4], bf[4][2];
#pragma unroll
                    for (int i = 0; i < MI; ++i) ldmatrix_x4(af[i], smem_addr(a_addr(r0 + i * 16 + a_r, k0 + ks + a_c)));
#pragma unroll
                    for (int jj = 0; jj < 2; ++jj) {
                        unsigned r[4];
                        ldmatrix_x4(r, smem_addr(ws + (nc * 32 + jj * 16 + b_r) * LDW + ks + b_c));
                        bf[2 * jj][0] = r[0];
                        bf[2 * jj][1] = r[1];
                        bf[2 * jj + 1][0] = r[2];
                        bf[2 * jj + 1][1] = r[3];
                    }
#pragma unroll
                    for (int i = 0; i < MI; ++i)
#pragma unroll
                        for (int j = 0; j < 4; ++j) mma(acc[i][j], af[i], bf[j]);
                }
            };
            using M2 = std::integral_constant<int, 2>;
            using M1 = std::integral_constant<int, 1>;
            auto main_jobs = [&](auto &acc, const __half *ws, int k0, auto a_addr) {
#pragma unroll
                for (int j = 0; j < JM; ++j) {
                    const int idx = warp + j * NW;
                    if (idx < B::JOBS) job(acc[j], M2{}, (idx / NC) * 32, idx % NC, ws, k0, a_addr);
                }
            };

            // conv1 A: y1 row rr (position l0 - 1 + rr), tap t -> x tile row rr*S + t
            auto a_conv1 = [&](int rr, int kk) {
                rr = rr < B::NY ? rr : B::NY - 1; // rows past the halo are discarded
                return R + (rr * S + kk / CIN) * LDX + kk % CIN;
            };
            // shortcut A: out row o reads x position S*(l0+o) -> x tile row S*o + S + 1
            auto a_sc = [&](int o, int kk) { return R + (S * o + S + 1) * LDX + kk; };
            // conv2 A: out row o, tap t -> y1 row o + t
            auto a_conv2 = [&](int o, int kk) { return R + (o + kk / COUT) * LDY + kk % COUT; };

            // y1 = relu(acc + b1), zero outside [0, lout) (conv2's padding), fp16 into the y1 tile
            auto store_y1 = [&](const auto &acc_ij, int rr, int c, int h) {
                if (rr >= B::NY) return;
                const int q = l0 - 1 + rr;
                float2 v = make_float2(0.f, 0.f);
                if (q >= 0 && q < p.lout) {
                    const float2 bias = *reinterpret_cast<const float2 *>(p.b1 + c);
                    v = acc_pair(acc_ij, h);
                    v.x = fmaxf(v.x + bias.x, 0.f);
                    v.y = fmaxf(v.y + bias.y, 0.f);
                }
                *reinterpret_cast<__half2 *>(R + rr * LDY + c) = __floats2half2_rn(v.x, v.y);
            };

            for (int i = 0; i < T; ++i) {
                cp_async_wait<0>();
                __syncthreads();
                if (i + 1 < T) load_w(i + 1);
                cp_async_commit();
                const __half *ws = Ws + (i & 1) * COUT * LDW;
                if (i < Tsc) {
                    main_jobs(acc2, ws, i * 32, a_sc);
                } else if (i < Tsc + T1) {
                    const int k0 = (i - Tsc) * 32;
                    main_jobs(acc1, ws, k0, a_conv1);
                    if (my_halo < NC) job(halo, M1{}, BL, my_halo, ws, k0, a_conv1);
                    if (i == Tsc + T1 - 1) {
                        __syncthreads(); // everyone is done with the x tile; y1 overwrites it
#pragma unroll
                        for (int j = 0; j < JM; ++j) {
                            const int idx = warp + j * NW;
                            if (idx >= B::JOBS) break;
#pragma unroll
                            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                                for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                                    for (int h = 0; h < 2; ++h)
                                        store_y1(acc1[j][mi][ni], (idx / NC) * 32 + mi * 16 + lane / 4 + h * 8,
                                                 (idx % NC) * 32 + ni * 8 + (lane % 4) * 2, h);
                        }
                        if (my_halo < NC)
#pragma unroll
                            for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                                for (int h = 0; h < 2; ++h)
                                    store_y1(halo[0][ni], BL + lane / 4 + h * 8, my_halo * 32 + ni * 8 + (lane % 4) * 2, h);
                    }
                } else {
                    main_jobs(acc2, ws, (i - Tsc - T1) * 32, a_conv2);
                }
            }
            cp_async_wait<0>();
            __syncthreads(); // done reading y1; reuse as output staging [BL][LDY]

            // ---- epilogue pass 1: + b2 (+ bsc); ReLU now when the residual is already inside the GEMM
#pragma unroll
            for (int j = 0; j < JM; ++j) {
                const int idx = warp + j * NW;
                if (idx >= B::JOBS) break;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int c = (idx % NC) * 32 + ni * 8 + (lane % 4) * 2;
                    float2 bias = *reinterpret_cast<const float2 *>(p.b2 + c);
                    if constexpr (SC) {
                        const float2 bs = *reinterpret_cast<const float2 *>(p.bsc + c);
                        bias.x += bs.x;
                        bias.y += bs.y;
                    }
#pragma unroll
                    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const int o = (idx / NC) * 32 + mi * 16 + lane / 4 + h * 8;
                            float2 v = acc_pair(acc2[j][mi][ni], h);
                            v.x += bias.x;
                            v.y += bias.y;
                            if constexpr (SC) {
                                v.x = fmaxf(v.x, 0.f);
                                v.y = fmaxf(v.y, 0.f);
                            }
                            *reinterpret_cast<__half2 *>(R + o * LDY + c) = __floats2half2_rn(v.x, v.y);
                        }
                }
            }
            __syncthreads();

            // ---- pass 2: identity residual + ReLU in 16-byte chunks; store, or keep for the fused pool
            const long long m_base = static_cast<long long>(b) * p.lout;
            for (int c = tid; c < BL * (COUT / 8); c += NT) {
                const int o = c / (COUT / 8), col = (c % (COUT / 8)) * 8, l = l0 + o;
                if (l >= p.lout) continue;
                uint4 v = *reinterpret_cast<const uint4 *>(R + o * LDY + col);
                if constexpr (!SC) v = add_relu8(v, *reinterpret_cast<const uint4 *>(xb + static_cast<long long>(l) * CIN + col), true);
                if (p.pool) {
                    if constexpr (!SC) *reinterpret_cast<uint4 *>(R + o * LDY + col) = v;
                } else {
                    *reinterpret_cast<uint4 *>(p.y + (m_base + l) * COUT + col) = v;
                }
            }
            if (p.pool) {
                __syncthreads();
                // the weight ring is idle now: reuse it as the pool scratch
                pool_tile<BL, COUT, NT>(R, LDY, m_base + l0, m_base + p.lout, p.lout, COUT, 0, p.pool, tid,
                                        reinterpret_cast<float *>(Ws));
            }
        }

        template <int CIN, int COUT, int S, int BL, bool SC, bool F16Acc>
        void launch_block(const BlockParams &p, cudaStream_t s) {
            using B = BlockShape<CIN, COUT, S, BL>;
            static const bool configured = [] {
                CURF_CUDA_CHECK(cudaFuncSetAttribute(basic_block<CIN, COUT, S, BL, SC, F16Acc>,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     static_cast<int>(B::SMEM)));
                return true;
            }();
            (void)configured;
            const dim3 grid(static_cast<unsigned>((p.lout + BL - 1) / BL), static_cast<unsigned>(p.batch));
            basic_block<CIN, COUT, S, BL, SC, F16Acc><<<grid, B::NT, B::SMEM, s>>>(p);
            CURF_CUDA_CHECK(cudaGetLastError());
        }

        template <int CIN, int COUT, int S, bool SC> bool sup_block(const BlockParams &p) {
            return p.cin == CIN && p.cout == COUT && p.stride == S && (p.wsc != nullptr) == SC &&
                   p.kpad1 == 3 * CIN && p.kpad2 == 3 * COUT && (!SC || p.kpadsc == CIN) &&
                   p.lout == (p.lin - 1) / S + 1 && (SC || p.lin == p.lout);
        }

        template <int CIN, int COUT, int S, bool SC> void add(std::vector<BlockVariant> &v) {
            const std::string shape = std::to_string(CIN) + "x" + std::to_string(COUT) + "s" + std::to_string(S);
            v.push_back({"block.f16.l128." + shape, &sup_block<CIN, COUT, S, SC>, &launch_block<CIN, COUT, S, 128, SC, true>});
            v.push_back({"block.f16.l64." + shape, &sup_block<CIN, COUT, S, SC>, &launch_block<CIN, COUT, S, 64, SC, true>});
            v.push_back({"block.f32.l128." + shape, &sup_block<CIN, COUT, S, SC>, &launch_block<CIN, COUT, S, 128, SC, false>});
            v.push_back({"block.f32.l64." + shape, &sup_block<CIN, COUT, S, SC>, &launch_block<CIN, COUT, S, 64, SC, false>});
        }

        std::vector<BlockVariant> make_block_registry() {
            std::vector<BlockVariant> v;
            add<32, 32, 1, false>(v);
            add<32, 64, 2, true>(v);
            add<64, 64, 1, false>(v);
            add<64, 128, 2, true>(v);
            return v;
        }

    } // namespace

    std::span<const BlockVariant> block_variants() {
        static const std::vector<BlockVariant> registry = make_block_registry();
        return registry;
    }

} // namespace curf::engine
