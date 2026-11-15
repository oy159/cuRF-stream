// INT8 implicit-GEMM conv1d: mma.sync.m16n8k32.s8.s8.s32, same pipeline as conv1d_mma.cu.
//
// Why it is almost the same kernel: an m16n8k32 int8 fragment and an m16n8k16 fp16 fragment cover the same
// bytes (16 rows x 32 bytes for A, 8 columns x 32 bytes for B, each thread holding the same 4-byte words),
// so the shared-memory k-tile (64 bytes per row), its swizzle, the ldmatrix addressing and the cp.async
// loader are unchanged - only the element count per 16-byte chunk doubles (16 int8 instead of 8 halves).
// On consumer Ampere the int8 tensor-core rate is 2x the fp16 (f16-accumulate) rate and 4x fp16 with fp32
// accumulation, and every activation byte moved is one element instead of half of one.
//
// Quantisation (symmetric, zero point 0):
//   activations: per tensor, q = round(v / s_t), s_t from calibration (engine.cpp)
//   weights:     per output channel, s_w[n] = max |w[n][:]| / 127
//   y = act(int32_acc * alpha[n] + bias[n] (+ s_res * q_res)),  alpha[n] = s_w[n] * s_x,
//   then q_y = round(y / s_y), saturated to +-127.
// The epilogue stages dequantised fp16 values in shared memory (as the fp16 kernel does), adds the residual
// and requantises in the coalesced second pass; the fused global average pool reads the staged fp16 tile.
//
// Requirements: int8 input with cin % 16 == 0, cout % BN == 0, kpad % 64 == 0.

#include "conv1d_internal.hpp"
#include "mma_common.cuh"

#include "curf/cuda/memory.hpp"

#include <cstdint>
#include <string>

namespace curf::engine::detail {

    namespace {

        template <int BM, int BN, int WM, int STAGES, int WN_ = 32, bool SWZ_ = false>
        struct I8Shape {
            static constexpr bool SWZ = SWZ_;
            static constexpr int BK = 64;                                // bytes = int8 elements per k-tile
            static constexpr int LDS = SWZ ? 32 : 40;                    // row stride in halves (64 / 80 bytes)
            static constexpr int WN = WN_;
            static constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN, NT = WARPS_M * WARPS_N * 32;
            static constexpr int MI = WM / 16, NI = WN / 8;
            static constexpr int A_PER = BM * 4 / NT, B_PER = BN * 4 / NT; // 4 chunks of 16 bytes per row
            static constexpr int LDC = BN + 8;
            static constexpr std::size_t PIPE = static_cast<std::size_t>(STAGES) * (BM + BN) * LDS * 2;
            static constexpr std::size_t STAGE_C = static_cast<std::size_t>(BM) * LDC * sizeof(__half);
            static constexpr std::size_t POOL = STAGE_C + static_cast<std::size_t>(NT / 32) * BN * sizeof(float);
            static constexpr std::size_t SMEM = PIPE > POOL ? PIPE : POOL;
            static_assert(BM * 4 % NT == 0 && BN * 4 % NT == 0 && A_PER >= 1 && B_PER >= 1, "tile loads");
        };

        template <int BM, int BN, int WM, int STAGES, int WN, bool SWZ>
        __global__ void __launch_bounds__(I8Shape<BM, BN, WM, STAGES, WN, SWZ>::NT) conv_i8(ConvParams p) {
            using S = I8Shape<BM, BN, WM, STAGES, WN, SWZ>;
            constexpr int BK = S::BK, LDS = S::LDS, NT = S::NT, MI = S::MI, NI = S::NI;
            extern __shared__ __align__(16) unsigned char smem[];
            __half *As = reinterpret_cast<__half *>(smem); // raw bytes, indexed in 2-byte units like the fp16 kernel
            __half *Bs = As + STAGES * BM * LDS;

            const auto *x = static_cast<const std::int8_t *>(p.x);
            const long long M = static_cast<long long>(p.batch) * p.lout;
            const long long m0 = static_cast<long long>(blockIdx.x) * BM;
            const int n0 = blockIdx.y * BN;
            const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
            const int wm = warp / S::WARPS_N, wn = warp % S::WARPS_N;

            const int kc = (tid % 4) * 16; // this thread's 16-byte column of every k-tile, in elements
            const std::int8_t *a_row[S::A_PER];
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
                    a_pos0[i] = -(1 << 30);
                }
            }
            int tap = kc / p.cin, ci = kc - (kc / p.cin) * p.cin;
            const std::int8_t *w_row = p.w8 + static_cast<long long>(n0 + tid / 4) * p.kpad + kc;
            const long long w_step = static_cast<long long>(NT / 4) * p.kpad;

            auto load_tile = [&](int stage, int k0) {
                __half *as = As + stage * BM * LDS;
                __half *bs = Bs + stage * BN * LDS;
#pragma unroll
                for (int i = 0; i < S::A_PER; ++i) {
                    const int pos = a_pos0[i] + tap;
                    const bool ok = tap < p.k && pos >= 0 && pos < p.lin;
                    const std::int8_t *src = ok ? a_row[i] + static_cast<long long>(pos) * p.cin + ci : x;
                    cp_async16_ca(smem_addr(as + sidx<SWZ, LDS>(tid / 4 + i * (NT / 4), tid % 4)), src, ok);
                }
#pragma unroll
                for (int i = 0; i < S::B_PER; ++i)
                    cp_async16(smem_addr(bs + sidx<SWZ, LDS>(tid / 4 + i * (NT / 4), tid % 4)), w_row + i * w_step + k0, true);
                ci += BK;
                while (ci >= p.cin) { ci -= p.cin; ++tap; }
            };

            int acc[MI][NI][4];
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < NI; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[i][j][e] = 0;

            const int KT = p.kpad / BK;
#pragma unroll
            for (int s = 0; s < STAGES - 1; ++s) {
                if (s < KT) load_tile(s, s * BK);
                cp_async_commit();
            }

            // Same lane addressing as the fp16 kernel, in 2-byte units: ks = 0 / 16 selects bytes 0..31 / 32..63.
            const int a_r = (lane % 8) + ((lane / 8) % 2) * 8, a_c = (lane / 16) * 8;
            const int b_r = (lane % 8) + (lane / 16) * 8, b_c = ((lane / 8) % 2) * 8;

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
                cp_async_wait<STAGES - 2>();
                __syncthreads();
                load_frags(0, 0, 0);
                for (int kt = 0; kt < KT; ++kt) {
                    if (const int nk = kt + STAGES - 1; nk < KT) load_tile(nk % STAGES, nk * BK);
                    cp_async_commit();
                    load_frags(1, kt % STAGES, 16);
                    mma_all(0);
                    cp_async_wait<STAGES - 2>();
                    __syncthreads();
                    if (kt + 1 < KT) load_frags(0, (kt + 1) % STAGES, 0);
                    mma_all(1);
                }
            } else {
                for (int kt = 0; kt < KT; ++kt) {
                    cp_async_wait<STAGES - 2>();
                    __syncthreads();
                    if (const int nk = kt + STAGES - 1; nk < KT) load_tile(nk % STAGES, nk * BK);
                    cp_async_commit();
#pragma unroll
                    for (int ks = 0; ks < 32; ks += 16) {
                        load_frags(0, kt % STAGES, ks);
                        mma_all(0);
                    }
                }
            }
            cp_async_wait<0>();
            __syncthreads();

            // ---- epilogue pass 1: dequantise (alpha), + bias, ReLU when there is no residual -> fp16 staging tile
            __half *Cs = reinterpret_cast<__half *>(smem);
            const bool relu_now = p.relu && p.res == nullptr;
#pragma unroll
            for (int j = 0; j < NI; ++j) {
                const int c = wn * WN + j * 8 + (lane % 4) * 2;
                const float2 bias = *reinterpret_cast<const float2 *>(p.bias + n0 + c);
                const float2 al = *reinterpret_cast<const float2 *>(p.alpha + n0 + c);
#pragma unroll
                for (int i = 0; i < MI; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int r = wm * WM + i * 16 + lane / 4 + h * 8;
                        float2 v = make_float2(fmaf(static_cast<float>(acc[i][j][2 * h]), al.x, bias.x),
                                               fmaf(static_cast<float>(acc[i][j][2 * h + 1]), al.y, bias.y));
                        if (relu_now) {
                            v.x = fmaxf(v.x, 0.f);
                            v.y = fmaxf(v.y, 0.f);
                        }
                        *reinterpret_cast<__half2 *>(Cs + r * S::LDC + c) = __floats2half2_rn(v.x, v.y);
                    }
            }
            __syncthreads();
            // ---- pass 2: 8 columns per thread: + residual (int8, 8 bytes), ReLU, requantise (8-byte stores)
            const auto *res = static_cast<const std::int8_t *>(static_cast<const void *>(p.res));
            auto *y = static_cast<std::int8_t *>(static_cast<void *>(p.y));
            constexpr int CPR = BN / 8;
            for (int c = tid; c < BM * CPR; c += NT) {
                const int r = c / CPR, col = (c % CPR) * 8;
                const long long m = m0 + r;
                if (m >= M) continue;
                const uint4 hv = *reinterpret_cast<const uint4 *>(Cs + r * S::LDC + col);
                const auto *h2 = reinterpret_cast<const __half2 *>(&hv);
                float v[8];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float2 f = __half22float2(h2[e]);
                    v[2 * e] = f.x;
                    v[2 * e + 1] = f.y;
                }
                if (res) {
                    const uint2 rq = *reinterpret_cast<const uint2 *>(res + m * p.cout + n0 + col);
                    const auto *rb = reinterpret_cast<const std::int8_t *>(&rq);
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        v[e] = fmaf(static_cast<float>(rb[e]), p.res_scale, v[e]);
                        if (p.relu) v[e] = fmaxf(v[e], 0.f);
                    }
                }
                if (p.pool) {
                    __half2 o[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) o[e] = __floats2half2_rn(v[2 * e], v[2 * e + 1]);
                    *reinterpret_cast<uint4 *>(Cs + r * S::LDC + col) = *reinterpret_cast<uint4 *>(o);
                } else {
                    const float s = p.y_inv_scale;
                    *reinterpret_cast<uint2 *>(y + m * p.cout + n0 + col) =
                        make_uint2(pack_q4(v[0] * s, v[1] * s, v[2] * s, v[3] * s), pack_q4(v[4] * s, v[5] * s, v[6] * s, v[7] * s));
                }
            }
            if (p.pool) {
                __syncthreads();
                pool_tile<BM, BN, NT>(Cs, S::LDC, m0, M, p.lout, p.cout, n0, p.pool, tid,
                                      reinterpret_cast<float *>(smem + S::STAGE_C));
            }
        }

        template <int BM, int BN, int WM, int STAGES, int WN, bool SWZ>
        void launch_i8(const ConvParams &p, cudaStream_t s) {
            using S = I8Shape<BM, BN, WM, STAGES, WN, SWZ>;
            static const bool configured = [] {
                CURF_CUDA_CHECK(cudaFuncSetAttribute(conv_i8<BM, BN, WM, STAGES, WN, SWZ>,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(S::SMEM)));
                return true;
            }();
            (void)configured;
            const dim3 grid(static_cast<unsigned>((p.m() + BM - 1) / BM), static_cast<unsigned>(p.cout / BN));
            conv_i8<BM, BN, WM, STAGES, WN, SWZ><<<grid, S::NT, S::SMEM, s>>>(p);
        }

        template <int BN> bool sup_i8(const ConvParams &p) {
            return p.int8 && !p.x_f32 && p.cin % 16 == 0 && p.cout % BN == 0 && p.kpad % 64 == 0 && p.w8 && p.alpha;
        }

        template <int BM, int BN, int WM, int STAGES, int WN = 32, bool SWZ = false> void add(std::vector<ConvVariant> &v) {
            const std::string tile = std::to_string(BM) + "x" + std::to_string(BN) + ".w" + std::to_string(WM) +
                                     (WN == 32 ? "" : "x" + std::to_string(WN)) + ".s" + std::to_string(STAGES) +
                                     (SWZ ? ".z" : "");
            v.push_back({"i8.mma." + tile, true, &sup_i8<BN>, &launch_i8<BM, BN, WM, STAGES, WN, SWZ>, true, true});
        }

    } // namespace

    void add_int8_variants(std::vector<ConvVariant> &v) {
        add<128, 32, 32, 2>(v);
        add<128, 32, 32, 3>(v);
        add<64, 64, 32, 2>(v);
        add<64, 64, 32, 3>(v);
        add<128, 64, 32, 3>(v);
        add<128, 64, 64, 3>(v);
        add<64, 128, 32, 3>(v);
        add<128, 128, 64, 3>(v);
        add<128, 128, 64, 3, 64>(v);
        add<128, 64, 64, 3, 64>(v);
        add<128, 128, 64, 3, 64, true>(v);
        add<128, 128, 64, 4, 64, true>(v);
        add<128, 128, 64, 3, 32, true>(v);
        add<128, 64, 64, 3, 64, true>(v);
        add<128, 64, 32, 3, 32, true>(v);
        add<64, 64, 32, 3, 32, true>(v);
        add<64, 128, 32, 3, 32, true>(v);
        add<128, 32, 32, 3, 32, true>(v);
        add<256, 64, 64, 3, 32, true>(v);
    }

} // namespace curf::engine::detail
