#pragma once
// PTX helpers shared by the mma.sync kernels (conv1d_mma.cu, block_fused.cu).

#include <cuda_fp16.h>

namespace curf::engine::detail {

    __device__ __forceinline__ unsigned smem_addr(const void *p) {
        return static_cast<unsigned>(__cvta_generic_to_shared(p));
    }

    // 16-byte async copy bypassing L1; src_bytes = 0 zero-fills the destination (padding / out of range).
    __device__ __forceinline__ void cp_async16(unsigned dst, const void *src, bool valid) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
    }
    // Same, but allocating in L1: the k taps of a conv re-read the same input rows, so activations
    // should stay in L1 (.cg sends every tap back to L2 - measured L2 throughput 65-86%).
    __device__ __forceinline__ void cp_async16_ca(unsigned dst, const void *src, bool valid) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
    }
    __device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
    template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

    __device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], unsigned addr) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                     : "r"(addr));
    }

    __device__ __forceinline__ void mma(float (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    __device__ __forceinline__ void mma(unsigned (&c)[2], const unsigned (&a)[4], const unsigned (&b)[2]) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                     : "+r"(c[0]), "+r"(c[1])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }

    // Two accumulator values of one row (columns 2t, 2t+1) as floats; h selects row g or g + 8.
    __device__ __forceinline__ float2 acc_pair(const float (&c)[4], int h) { return make_float2(c[2 * h], c[2 * h + 1]); }
    __device__ __forceinline__ float2 acc_pair(const unsigned (&c)[2], int h) {
        return __half22float2(*reinterpret_cast<const __half2 *>(&c[h]));
    }

    // Fused global-average-pool epilogue: column sums of the staged tile Cs (activation, residual and ReLU
    // already applied) over its rows, grouped by sample (m / lout), atomically added to pool[b][n0 + c]
    // (fp32, zeroed beforehand). Only shared-memory reads in the loop: the residual must have been added
    // by a coalesced pass before (a first version read it here, 2 bytes per thread per row from global
    // memory, and was slower than not fusing at all).
    template <int ROWS, int COLS, int NT>
    __device__ __forceinline__ void pool_tile(const __half *Cs, int ldc, long long m0, long long M, int lout, int ld,
                                              int n0, float *pool, int tid) {
        static_assert(NT % COLS == 0 && ROWS % (NT / COLS) == 0, "pool_tile thread mapping");
        // Fast path (the usual case, lout a multiple of the tile height): the whole tile belongs to one
        // sample. Each thread sums 8 columns over a strided set of rows with 16-byte reads, warps reduce
        // with shuffles, warp partials go through shared memory, then one global atomic per column.
        // (Serial per-column sums, and then shared-memory float atomics, both cost 50-200 us at batch 512.)
        const long long last = (M < m0 + ROWS ? M : m0 + ROWS) - 1;
        if (m0 / lout == last / lout) { // uniform across the CTA
            constexpr int CG = COLS / 8, G = NT / CG, NWARP = NT / 32;
            static_assert(NT % CG == 0 && CG <= 32 && 32 % CG == 0, "pool_tile vector mapping");
            __shared__ float part[NWARP][COLS];
            const int c8 = (tid % CG) * 8, nrows = static_cast<int>(last - m0 + 1);
            float acc[8] = {};
            for (int r = tid / CG; r < nrows; r += G) {
                const uint4 v = *reinterpret_cast<const uint4 *>(Cs + r * ldc + c8);
                const auto *h = reinterpret_cast<const __half2 *>(&v);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float2 f = __half22float2(h[e]);
                    acc[2 * e] += f.x;
                    acc[2 * e + 1] += f.y;
                }
            }
#pragma unroll
            for (int off = CG; off < 32; off <<= 1)
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[e] += __shfl_xor_sync(0xffffffffu, acc[e], off);
            if (tid % 32 < CG)
#pragma unroll
                for (int e = 0; e < 8; ++e) part[tid / 32][c8 + e] = acc[e];
            __syncthreads();
            const long long b = m0 / lout;
            for (int i = tid; i < COLS; i += NT) {
                float sum = 0.f;
#pragma unroll
                for (int w = 0; w < NWARP; ++w) sum += part[w][i];
                atomicAdd(pool + b * ld + n0 + i, sum);
            }
            return;
        }
        constexpr int RPG = ROWS / (NT / COLS);
        const int c = tid % COLS, g = tid / COLS;
        const long long first = m0 + g * RPG;
        if (first >= M) return;
        // One division per thread; sample boundaries are then tracked incrementally (a 64-bit division per
        // row made the fused pool slower than a separate kernel).
        long long cur = first / lout, next = (cur + 1) * lout;
        float sum = 0.f;
        for (int r = g * RPG; r < (g + 1) * RPG; ++r) {
            const long long m = m0 + r;
            if (m >= M) break;
            if (m == next) {
                atomicAdd(pool + cur * ld + n0 + c, sum);
                sum = 0.f;
                ++cur;
                next += lout;
            }
            sum += __half2float(Cs[r * ldc + c]);
        }
        atomicAdd(pool + cur * ld + n0 + c, sum);
    }

    // In-place y = relu(y + r) on 8 halves.
    __device__ __forceinline__ uint4 add_relu8(uint4 v, uint4 r, bool relu) {
        auto *vh = reinterpret_cast<__half2 *>(&v);
        const auto *rh = reinterpret_cast<const __half2 *>(&r);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            float2 f = __half22float2(vh[e]);
            const float2 g = __half22float2(rh[e]);
            f.x += g.x;
            f.y += g.y;
            if (relu) {
                f.x = fmaxf(f.x, 0.f);
                f.y = fmaxf(f.y, 0.f);
            }
            vh[e] = __floats2half2_rn(f.x, f.y);
        }
        return v;
    }

} // namespace curf::engine::detail
