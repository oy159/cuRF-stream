#pragma once
// conv1d / head kernels of the cuRF engine.
//
// Every conv is an implicit GEMM over channels-last activations:
//   M = batch * lout (one row per output position), N = cout, K = k * cin (tap-major, then channel)
//   A[m][kk] = x[b][l * stride + tap - pad][ci]   (zero outside [0, lin) and for kk >= k * cin)
//   B[kk][n] = w[n][kk]                           (fp16, rows padded to kpad)
//   y[m][n]  = act(sum_kk A * B + bias[n] (+ res[m][n]))
//
// Variants form the optimisation ladder recorded in PLAN.md: naive -> shared-memory tiling ->
// WMMA tensor cores -> fused epilogue -> (later) cp.async pipelining. "Unfused" variants write the
// raw GEMM result and a separate elementwise kernel applies bias / residual / ReLU, which is what a
// framework without epilogue fusion would do.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <string>

namespace curf::engine {

    struct ConvParams {
        const void *x = nullptr;     // [batch][lin][cin], fp32 if x_f32 else fp16
        bool x_f32 = false;
        const __half *w = nullptr;   // [cout][kpad]
        const float *bias = nullptr; // [cout]
        const __half *res = nullptr; // [batch][lout][cout] or null
        __half *y = nullptr;         // [batch][lout][cout]
        float *pool = nullptr;       // if set: y is not written; sum_l act(...) is added to pool[batch][cout]
                                     // (fp32, zeroed beforehand) - global average pool fused into the epilogue
        int batch = 0, lin = 0, lout = 0, cin = 0, cout = 0, k = 1, stride = 1, pad = 0, kpad = 0;
        bool relu = false;

        long long m() const { return static_cast<long long>(batch) * lout; }
        double flops() const { return 2.0 * static_cast<double>(m()) * cout * cin * k; }
    };

    struct ConvVariant {
        std::string name;   // e.g. "naive", "smem", "wmma.f32.fused.128x64"
        bool fused = false; // epilogue inside the GEMM kernel
        bool (*supported)(const ConvParams &) = nullptr;
        void (*launch)(const ConvParams &, cudaStream_t) = nullptr;
        bool pool = false;  // supports ConvParams::pool
    };

    // All registered variants, in ladder order.
    std::span<const ConvVariant> conv_variants();
    const ConvVariant *find_conv_variant(const std::string &name);

    // Runs the variant and, if it is unfused, the separate epilogue kernel.
    void launch_conv(const ConvVariant &v, const ConvParams &p, cudaStream_t s);

    // One ResNet BasicBlock in a single kernel:
    //   y = relu(conv2(relu(conv1(x) + b1)) + b2 + shortcut(x))
    // conv1: k3 stride S pad 1, cin -> cout; conv2: k3 s1 pad1, cout -> cout;
    // shortcut: identity (S = 1, cin = cout) or 1x1 stride S conv (+ bsc).
    // The conv1 output stays in shared memory; the 1x1 shortcut is extra K of conv2's GEMM.
    struct BlockParams {
        const __half *x = nullptr;   // [batch][lin][cin]
        __half *y = nullptr;         // [batch][lout][cout]
        float *pool = nullptr;       // as ConvParams::pool
        const __half *w1 = nullptr, *w2 = nullptr, *wsc = nullptr; // [cout][kpad*]
        const float *b1 = nullptr, *b2 = nullptr, *bsc = nullptr;
        int batch = 0, lin = 0, lout = 0, cin = 0, cout = 0, stride = 1;
        int kpad1 = 0, kpad2 = 0, kpadsc = 0;
        double flops() const {
            return 2.0 * batch * lout * cout * (3.0 * cin + 3.0 * cout + (wsc ? cin : 0));
        }
    };

    struct BlockVariant {
        std::string name; // e.g. "block.f16.l128.64x128s2" (accumulator, positions per CTA, shape)
        bool (*supported)(const BlockParams &) = nullptr;
        void (*launch)(const BlockParams &, cudaStream_t) = nullptr;
    };
    std::span<const BlockVariant> block_variants();

    // y[b][n] = sum_c w[n][c] * mean_l x[b][l][c] + bias[n]; fp32 out.
    void launch_gap_linear(const __half *x, const float *w, const float *bias, float *y, int batch, int length, int cin,
                           int cout, cudaStream_t s);

    // Head after a fused pool: y[b][n] = bias[n] + inv_len * sum_c w[n][c] * pool[b][c].
    void launch_linear_from_pool(const float *pool, const float *w, const float *bias, float *y, int batch, int cin,
                                 int cout, float inv_len, cudaStream_t s);

    // Utility for tests / debugging: fp16 -> fp32 copy on the device.
    void launch_half_to_float(const __half *src, float *dst, std::size_t n, cudaStream_t s);
    void launch_float_to_half(const float *src, __half *dst, std::size_t n, cudaStream_t s);

} // namespace curf::engine
