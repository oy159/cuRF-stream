// INT8 BasicBlock in one kernel (the INT8 counterpart of block_fused.cu).
//
//   y = relu(conv2(relu(conv1(x) + b1)) + b2 + shortcut(x))       all GEMMs mma.sync.m16n8k32.s8
//
// Why: the unfused INT8 engine moves ~776 KB of activations per window and at 240k windows/s that is
// ~186 GB/s, the DRAM bandwidth of the RTX 3050 Laptop - INT8 doubled the tensor rate but the layers
// became memory bound. Fusing each block keeps conv1's output (and the shortcut) on chip: per window
// the traffic drops to ~264 KB.
//
// Differences from the fp16 block kernel:
//   - weight k-tiles are 32 bytes (one m16n8k32 step) so a k-tile never straddles a tap (CIN % 32 == 0);
//     rows are padded to 48 bytes (ldmatrix conflict-free for 32-byte rows)
//   - x and y1 have separate tiles (int8 halves the footprint): x stays for the shortcut / identity
//     residual, which are read from shared memory instead of global memory
//   - conv2 and the 1x1 shortcut have different scales (s_y1 * s_w2[n] vs s_x * s_wsc[n]), so they cannot
//     share an int32 accumulator; the shortcut runs after conv2 and reuses conv1's (dead) accumulators
//   - y1 is requantised to int8 with the calibrated scale of the conv1 output tensor, as the unfused
//     engine stores it
//
// One CTA = BL output positions x all COUT channels of one sample; warp jobs are 32x32 output tiles, plus
// conv1's 2 halo rows as m16 x 32 jobs (as in block_fused.cu).

#include "conv1d_internal.hpp"
#include "mma_common.cuh"

#include "curf/cuda/memory.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace curf::engine::detail {

    namespace {

        // RES: all weights resident in shared memory (loaded once with the x tile), so the k loops run without
        // barriers; otherwise they stream through a 3-stage ring of KS 32-byte k-steps per barrier
        // (KS = 1: rows padded to 48 bytes; KS = 2: 64-byte rows, XOR-swizzled like the conv kernels' tiles).
        template <int CIN, int COUT, int S, int BL_, bool RES_ = false, int KS_ = 1>
        struct I8Block {
            static constexpr bool RES = RES_, SC = S != 1 || CIN != COUT;
            static constexpr int KS = KS_;
            static constexpr int BL = BL_, STAGES = 3, BK = 32 * KS;
            static constexpr int K1 = 3 * CIN, K2 = 3 * COUT, KSC = SC ? CIN : 0;
            static constexpr int LW1 = K1 + 16, LW2 = K2 + 16, LWSC = KSC + 16; // resident rows, bytes
            static constexpr int NC = COUT / 32, RP = BL / 32, JOBS = RP * NC;
            static constexpr int NW = JOBS < 4 ? 4 : (JOBS > 8 ? 8 : JOBS), NT = NW * 32, JM = (JOBS + NW - 1) / NW;
            static constexpr int NX = (BL + 1) * S + 3, LDX = CIN + 16;  // bytes; x rows: positions px0 ..
            static constexpr int NY = BL + 2, LDY = COUT + 16;           // bytes; y1 rows: positions l0 - 1 ..
            static constexpr int LDW = KS == 1 ? BK + 16 : BK;           // bytes
            static constexpr int LDC = COUT + 8;                         // halves, pool staging
            static constexpr int LDQ = COUT + 16;                        // bytes, int8 output staging
            static constexpr std::size_t up16(std::size_t v) { return (v + 15) / 16 * 16; }
            static constexpr std::size_t XB = up16(std::size_t{NX} * LDX), YB = up16(std::size_t{NY} * LDY);
            static constexpr std::size_t WB = RES ? std::size_t{COUT} * (LW1 + LW2 + (SC ? LWSC : 0))
                                                  : std::size_t{STAGES} * COUT * LDW;
            static constexpr std::size_t STG_POOL = up16(std::size_t{BL} * LDC * 2) + std::size_t{NW} * COUT * 4;
            static constexpr std::size_t STG_Q = std::size_t{BL} * LDQ;
            static constexpr std::size_t STG = STG_POOL > STG_Q ? STG_POOL : STG_Q;
            static constexpr std::size_t SMEM = XB + (YB + WB > STG ? YB + WB : STG);
            static_assert(CIN % 32 == 0 && COUT % 32 == 0 && BL % 32 == 0 && NC <= NW, "tile shapes");
            static_assert(KS == 1 || (KS == 2 && K1 % 64 == 0 && K2 % 64 == 0 && KSC % 64 == 0), "k-steps per tile");
            // byte offset of 16-byte chunk q of ring row n
            static __device__ __forceinline__ int wsw(int n, int q) {
                return KS == 1 ? n * LDW + q * 16 : n * LDW + ((q ^ ((n >> 1) & 3)) << 4);
            }
        };

        __device__ __forceinline__ unsigned short pack_q2(float a, float b) {
            return static_cast<unsigned short>(pack_q4(a, b, 0.f, 0.f) & 0xFFFFu);
        }
        __device__ __forceinline__ float2 i8x2(const std::int8_t *p) {
            return make_float2(static_cast<float>(p[0]), static_cast<float>(p[1]));
        }

        template <int CIN, int COUT, int S, int BL, bool SC, bool RES, int KS>
        __global__ void __launch_bounds__(I8Block<CIN, COUT, S, BL, RES, KS>::NT) block_i8(BlockParams p) {
            using B = I8Block<CIN, COUT, S, BL, RES, KS>;
            static_assert(SC == B::SC);
            constexpr int STAGES = B::STAGES, NT = B::NT, NW = B::NW, NC = B::NC, JM = B::JM;
            constexpr int LDX = B::LDX, LDY = B::LDY, LDW = B::LDW;
            extern __shared__ __align__(16) unsigned char smem[];
            std::int8_t *X = reinterpret_cast<std::int8_t *>(smem);
            std::int8_t *Y = X + B::XB;
            std::int8_t *Ws = Y + B::YB;

            const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
            const int b = blockIdx.y, l0 = blockIdx.x * BL;
            const int px0 = (l0 - 1) * S - 1;
            const auto *xg = static_cast<const std::int8_t *>(static_cast<const void *>(p.x));
            const std::int8_t *xb = xg + static_cast<long long>(b) * p.lin * CIN;

            for (int c = tid; c < B::NX * (CIN / 16); c += NT) {
                const int r = c / (CIN / 16), col = (c % (CIN / 16)) * 16, pos = px0 + r;
                const bool ok = pos >= 0 && pos < p.lin;
                cp_async16(smem_addr(X + r * LDX + col), ok ? xb + static_cast<long long>(pos) * CIN + col : xg, ok);
            }

            // weight k-tile stream: [conv1] [conv2] [shortcut], BK bytes of K per tile (RES: k-steps of 32)
            constexpr int BKT = RES ? 32 : B::BK;
            constexpr int T1 = 3 * CIN / BKT, T2 = 3 * COUT / BKT, Tsc = SC ? CIN / BKT : 0, T = T1 + T2 + Tsc;
            auto load_w = [&](int i) {
                const std::int8_t *w;
                int kp, k0;
                if (i < T1) { w = p.w1_8; kp = p.kpad1; k0 = i * BKT; }
                else if (i < T1 + T2) { w = p.w2_8; kp = p.kpad2; k0 = (i - T1) * BKT; }
                else { w = p.wsc_8; kp = p.kpadsc; k0 = (i - T1 - T2) * BKT; }
                std::int8_t *ws = Ws + (i % STAGES) * COUT * LDW;
                constexpr int CPR = BKT / 16;
                for (int c = tid; c < COUT * CPR; c += NT) {
                    const int n = c / CPR, q = c % CPR;
                    cp_async16(smem_addr(ws + B::wsw(n, q)), w + static_cast<long long>(n) * kp + k0 + q * 16, true);
                }
            };
            std::int8_t *W1r = Ws, *W2r = Ws + COUT * B::LW1, *Wscr = W2r + COUT * B::LW2; // RES layout
            if constexpr (RES) {
                auto load_all = [&](std::int8_t *dst, const std::int8_t *w, int K, int ld, int kp) {
                    const int cpr = K / 16;
                    for (int c = tid; c < COUT * cpr; c += NT) {
                        const int n = c / cpr, kc = (c % cpr) * 16;
                        cp_async16(smem_addr(dst + n * ld + kc), w + static_cast<long long>(n) * kp + kc, true);
                    }
                };
                load_all(W1r, p.w1_8, B::K1, B::LW1, p.kpad1);
                load_all(W2r, p.w2_8, B::K2, B::LW2, p.kpad2);
                if constexpr (SC) load_all(Wscr, p.wsc_8, B::KSC, B::LWSC, p.kpadsc);
                cp_async_commit();
            } else {
#pragma unroll
                for (int st = 0; st < STAGES - 1; ++st) {
                    if (st < T) load_w(st);
                    cp_async_commit();
                }
            }

            int acc1[JM][2][4][4], halo[1][4][4], acc2[JM][2][4][4];
            auto zero = [](auto &a) {
                int *f = &a[0][0][0];
#pragma unroll
                for (int i = 0; i < static_cast<int>(sizeof(a) / sizeof(int)); ++i) f[i] = 0;
            };
#pragma unroll
            for (int j = 0; j < JM; ++j) {
                zero(acc1[j]);
                zero(acc2[j]);
            }
            zero(halo);

            const int a_r = (lane % 8) + ((lane / 8) % 2) * 8, a_c = (lane / 16) * 16;
            const int b_r = (lane % 8) + (lane / 16) * 8, b_c = ((lane / 8) % 2) * 16;
            const int my_halo = (warp - B::JOBS % NW + NW) % NW;

            // b_addr(n): this lane's 16-byte B column of weight row n for the current k-step
            auto job = [&](auto &acc, auto mi_c, int r0, int nc, auto b_addr, auto a_addr) {
                constexpr int MI = decltype(mi_c)::value;
                unsigned af[MI][4], bf[4][2];
#pragma unroll
                for (int i = 0; i < MI; ++i) ldmatrix_x4(af[i], smem_addr(a_addr(r0 + i * 16 + a_r) + a_c));
#pragma unroll
                for (int jj = 0; jj < 2; ++jj) {
                    unsigned r[4];
                    ldmatrix_x4(r, smem_addr(b_addr(nc * 32 + jj * 16 + b_r)));
                    bf[2 * jj][0] = r[0];
                    bf[2 * jj][1] = r[1];
                    bf[2 * jj + 1][0] = r[2];
                    bf[2 * jj + 1][1] = r[3];
                }
#pragma unroll
                for (int i = 0; i < MI; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) mma(acc[i][j], af[i], bf[j]);
            };
            using M2 = std::integral_constant<int, 2>;
            using M1 = std::integral_constant<int, 1>;
            auto main_jobs = [&](auto &acc, auto b_addr, auto a_addr, bool with_halo) {
#pragma unroll
                for (int j = 0; j < JM; ++j) {
                    const int idx = warp + j * NW;
                    if (idx < B::JOBS) job(acc[j], M2{}, (idx / NC) * 32, idx % NC, b_addr, a_addr);
                }
                if (with_halo && my_halo < NC) job(halo, M1{}, BL, my_halo, b_addr, a_addr);
            };
            auto res_b = [&](const std::int8_t *w, int ld, int ks) {
                const std::int8_t *base = w + ks * 32 + b_c;
                return [base, ld](int n) { return base + n * ld; };
            };
            auto ring_b = [&](const std::int8_t *ws, int s) {
                const int q = 2 * s + b_c / 16;
                return [ws, q](int n) { return ws + B::wsw(n, q); };
            };

            // A row addresses (byte column 0 of this k-tile); k0 = K offset in elements (= bytes)
            auto a_conv1_at = [&](int k0) {
                const std::int8_t *base = X + (k0 / CIN) * LDX + k0 % CIN;
                return [base](int rr) {
                    rr = rr < B::NY ? rr : B::NY - 1;
                    return base + rr * (S * LDX);
                };
            };
            auto a_conv2_at = [&](int k0) {
                const std::int8_t *base = Y + (k0 / COUT) * LDY + k0 % COUT;
                return [base](int o) { return base + o * LDY; };
            };
            auto a_sc_at = [&](int k0) {
                const std::int8_t *base = X + (S + 1) * LDX + k0;
                return [base](int o) { return base + o * (S * LDX); };
            };

            // y1 = relu(acc * alpha1 + b1), zero outside [0, lout), requantised to int8
            auto store_y1 = [&](const int (&a)[4], int rr, int c, int h) {
                if (rr >= B::NY) return;
                const int q = l0 - 1 + rr;
                float2 v = make_float2(0.f, 0.f);
                if (q >= 0 && q < p.lout) {
                    const float2 bias = *reinterpret_cast<const float2 *>(p.b1 + c);
                    const float2 al = *reinterpret_cast<const float2 *>(p.alpha1 + c);
                    v.x = fmaxf(fmaf(static_cast<float>(a[2 * h]), al.x, bias.x), 0.f) * p.y1_inv_scale;
                    v.y = fmaxf(fmaf(static_cast<float>(a[2 * h + 1]), al.y, bias.y), 0.f) * p.y1_inv_scale;
                }
                *reinterpret_cast<unsigned short *>(Y + rr * LDY + c) = pack_q2(v.x, v.y);
            };

            auto finish_conv1 = [&] { // y1 tile complete in registers; conv2 reads it after the next barrier
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
#pragma unroll
                        for (int j = 0; j < JM; ++j) zero(acc1[j]); // reused for the shortcut
            };

            if constexpr (RES) {
                cp_async_wait<0>();
                __syncthreads();
#pragma unroll
                for (int ks = 0; ks < T1; ++ks) main_jobs(acc1, res_b(W1r, B::LW1, ks), a_conv1_at(ks * 32), true);
                finish_conv1();
                __syncthreads();
#pragma unroll
                for (int ks = 0; ks < T2; ++ks) main_jobs(acc2, res_b(W2r, B::LW2, ks), a_conv2_at(ks * 32), false);
                if constexpr (SC)
#pragma unroll
                    for (int ks = 0; ks < Tsc; ++ks) main_jobs(acc1, res_b(Wscr, B::LWSC, ks), a_sc_at(ks * 32), false);
            } else {
                for (int i = 0; i < T; ++i) {
                    cp_async_wait<STAGES - 2>();
                    __syncthreads();
                    if (i + STAGES - 1 < T) load_w(i + STAGES - 1);
                    cp_async_commit();
                    const std::int8_t *ws = Ws + (i % STAGES) * COUT * LDW;
#pragma unroll
                    for (int s = 0; s < KS; ++s) {
                        if (i < T1) main_jobs(acc1, ring_b(ws, s), a_conv1_at(i * BKT + 32 * s), true);
                        else if (i < T1 + T2) main_jobs(acc2, ring_b(ws, s), a_conv2_at((i - T1) * BKT + 32 * s), false);
                        else main_jobs(acc1, ring_b(ws, s), a_sc_at((i - T1 - T2) * BKT + 32 * s), false);
                    }
                    if (i == T1 - 1) finish_conv1();
                }
            }
            cp_async_wait<0>();
            __syncthreads(); // y1 and the ring are dead: the staging tile goes over them (x stays)

            __half *Cs = reinterpret_cast<__half *>(Y);  // pool staging [BL][LDC]
            std::int8_t *Qs = Y;                         // int8 output staging [BL][LDQ]
            const bool pool = p.pool != nullptr;
#pragma unroll
            for (int j = 0; j < JM; ++j) {
                const int idx = warp + j * NW;
                if (idx >= B::JOBS) break;
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int c = (idx % NC) * 32 + ni * 8 + (lane % 4) * 2;
                    float2 bias = *reinterpret_cast<const float2 *>(p.b2 + c);
                    const float2 al2 = *reinterpret_cast<const float2 *>(p.alpha2 + c);
                    float2 alsc = make_float2(0.f, 0.f);
                    if constexpr (SC) {
                        const float2 bs = *reinterpret_cast<const float2 *>(p.bsc + c);
                        alsc = *reinterpret_cast<const float2 *>(p.alphasc + c);
                        bias.x += bs.x;
                        bias.y += bs.y;
                    }
#pragma unroll
                    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const int o = (idx / NC) * 32 + mi * 16 + lane / 4 + h * 8;
                            const int *a2 = acc2[j][mi][ni];
                            float2 v = make_float2(fmaf(static_cast<float>(a2[2 * h]), al2.x, bias.x),
                                                   fmaf(static_cast<float>(a2[2 * h + 1]), al2.y, bias.y));
                            if constexpr (SC) {
                                const int *as = acc1[j][mi][ni];
                                v.x = fmaf(static_cast<float>(as[2 * h]), alsc.x, v.x);
                                v.y = fmaf(static_cast<float>(as[2 * h + 1]), alsc.y, v.y);
                            } else { // identity: x position l0 + o is x tile row o + 2
                                const float2 r = i8x2(X + (o + 2) * LDX + c);
                                v.x = fmaf(r.x, p.x_scale, v.x);
                                v.y = fmaf(r.y, p.x_scale, v.y);
                            }
                            v.x = fmaxf(v.x, 0.f);
                            v.y = fmaxf(v.y, 0.f);
                            if (pool) *reinterpret_cast<__half2 *>(Cs + o * B::LDC + c) = __floats2half2_rn(v.x, v.y);
                            else *reinterpret_cast<unsigned short *>(Qs + o * B::LDQ + c) = pack_q2(v.x * p.y_inv_scale, v.y * p.y_inv_scale);
                        }
                }
            }
            __syncthreads();
            const long long m_base = static_cast<long long>(b) * p.lout;
            if (pool) {
                pool_tile<BL, COUT, NT>(Cs, B::LDC, m_base + l0, m_base + p.lout, p.lout, COUT, 0, p.pool, tid,
                                        reinterpret_cast<float *>(smem + B::XB + B::up16(std::size_t{BL} * B::LDC * 2)));
                return;
            }
            auto *y = static_cast<std::int8_t *>(static_cast<void *>(p.y));
            for (int c = tid; c < BL * (COUT / 16); c += NT) {
                const int o = c / (COUT / 16), col = (c % (COUT / 16)) * 16, l = l0 + o;
                if (l >= p.lout) continue;
                *reinterpret_cast<uint4 *>(y + (m_base + l) * COUT + col) = *reinterpret_cast<const uint4 *>(Qs + o * B::LDQ + col);
            }
        }

        template <int CIN, int COUT, int S, int BL, bool SC, bool RES, int KS = 1>
        void launch_block_i8(const BlockParams &p, cudaStream_t s) {
            using B = I8Block<CIN, COUT, S, BL, RES, KS>;
            static const bool configured = [] {
                CURF_CUDA_CHECK(cudaFuncSetAttribute(block_i8<CIN, COUT, S, BL, SC, RES, KS>,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(B::SMEM)));
                return true;
            }();
            (void)configured;
            const dim3 grid(static_cast<unsigned>((p.lout + BL - 1) / BL), static_cast<unsigned>(p.batch));
            block_i8<CIN, COUT, S, BL, SC, RES, KS><<<grid, B::NT, B::SMEM, s>>>(p);
            CURF_CUDA_CHECK(cudaGetLastError());
        }

        template <int CIN, int COUT, int S, bool SC> bool sup_block_i8(const BlockParams &p) {
            return p.int8 && p.cin == CIN && p.cout == COUT && p.stride == S && (p.wsc_8 != nullptr) == SC &&
                   p.w1_8 && p.w2_8 && p.kpad1 >= 3 * CIN && p.kpad2 >= 3 * COUT && (!SC || p.kpadsc >= CIN) &&
                   p.lout == (p.lin - 1) / S + 1 && (SC || p.lin == p.lout);
        }

        template <int CIN, int COUT, int S, bool SC> void add_block(std::vector<BlockVariant> &v) {
            const std::string shape = std::to_string(CIN) + "x" + std::to_string(COUT) + "s" + std::to_string(S);
            auto sup = &sup_block_i8<CIN, COUT, S, SC>;
            v.push_back({"i8.block.l128." + shape, sup, &launch_block_i8<CIN, COUT, S, 128, SC, false>, true});
            v.push_back({"i8.block.l64." + shape, sup, &launch_block_i8<CIN, COUT, S, 64, SC, false>, true});
            if constexpr (I8Block<CIN, COUT, S, 64, true>::SMEM <= 99 * 1024) // resident weights, if they fit
                v.push_back({"i8.block.l64r." + shape, sup, &launch_block_i8<CIN, COUT, S, 64, SC, true>, true});
            if constexpr (I8Block<CIN, COUT, S, 128, true>::SMEM <= 99 * 1024)
                v.push_back({"i8.block.l128r." + shape, sup, &launch_block_i8<CIN, COUT, S, 128, SC, true>, true});
            if constexpr (CIN % 64 == 0) { // 64-byte swizzled ring tiles: two k-steps per barrier
                v.push_back({"i8.block.l64k2." + shape, sup, &launch_block_i8<CIN, COUT, S, 64, SC, false, 2>, true});
                v.push_back({"i8.block.l128k2." + shape, sup, &launch_block_i8<CIN, COUT, S, 128, SC, false, 2>, true});
            }
        }

    } // namespace

    void add_int8_block_variants(std::vector<BlockVariant> &v) {
        add_block<32, 32, 1, false>(v);
        add_block<32, 64, 2, true>(v);
        add_block<64, 64, 1, false>(v);
        add_block<64, 128, 2, true>(v);
    }

} // namespace curf::engine::detail
