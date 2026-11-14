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
    };

    // All registered variants, in ladder order.
    std::span<const ConvVariant> conv_variants();
    const ConvVariant *find_conv_variant(const std::string &name);

    // Runs the variant and, if it is unfused, the separate epilogue kernel.
    void launch_conv(const ConvVariant &v, const ConvParams &p, cudaStream_t s);

    // y[b][n] = sum_c w[n][c] * mean_l x[b][l][c] + bias[n]; fp32 out.
    void launch_gap_linear(const __half *x, const float *w, const float *bias, float *y, int batch, int length, int cin,
                           int cout, cudaStream_t s);

    // Utility for tests / debugging: fp16 -> fp32 copy on the device.
    void launch_half_to_float(const __half *src, float *dst, std::size_t n, cudaStream_t s);

} // namespace curf::engine
