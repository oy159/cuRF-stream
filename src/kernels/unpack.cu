#include "curf/cuda/memory.hpp"
#include "curf/cuda/unpack.hpp"

#include <algorithm>
#include <cstdint>

namespace curf::cuda {
    namespace {

        constexpr int kBlock = 256;

        __device__ __forceinline__ int sext12(unsigned v) { return static_cast<int>(v << 20) >> 20; }

        // b0 b1 b2 -> (I, Q), see WireFormat::packed12.
        __device__ __forceinline__ float2 decode12(unsigned b0, unsigned b1, unsigned b2) {
            const int i = sext12((b0 << 4) | (b1 >> 4));
            const int q = sext12(((b1 & 15u) << 8) | b2);
            // Scaling by a power of two is exact, so this equals the CPU reference's division.
            return make_float2(static_cast<float>(i) * (1.0f / 2048.0f), static_cast<float>(q) * (1.0f / 2048.0f));
        }

        __device__ __forceinline__ float2 decode8(unsigned b0, unsigned b1) {
            const int i = static_cast<signed char>(b0);
            const int q = static_cast<signed char>(b1);
            return make_float2(static_cast<float>(i) * (1.0f / 128.0f), static_cast<float>(q) * (1.0f / 128.0f));
        }

        __device__ __forceinline__ unsigned byte_of(unsigned word, int j) { return (word >> (8 * j)) & 0xffu; }

        // ---------------------------------------------------------------- naive
        // One thread per wire sample s = k * channels + c, byte loads.
        template <WireFormat F>
        __global__ void unpack_naive_kernel(const std::uint8_t *__restrict__ wire, std::uint32_t channels,
                                            std::size_t frames, float2 *__restrict__ out) {
            const std::size_t total = frames * channels;
            for (std::size_t s = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; s < total;
                 s += std::size_t{gridDim.x} * blockDim.x) {
                const std::size_t k = s / channels;
                const std::size_t c = s - k * channels;
                float2 v;
                if constexpr (F == WireFormat::packed12) {
                    const std::uint8_t *p = wire + 3 * s;
                    v = decode12(p[0], p[1], p[2]);
                } else {
                    const std::uint8_t *p = wire + 2 * s;
                    v = decode8(p[0], p[1]);
                }
                out[c * frames + k] = v;
            }
        }

        // ----------------------------------------------------------------- vec4
        // One thread per group of 4 wire samples: 3x32-bit (packed12) or 1x64-bit
        // (signed8) loads, and 16-byte stores for the 1- and 2-channel layouts.
        __device__ __forceinline__ void store4(float2 *__restrict__ out, std::uint32_t channels, std::size_t frames,
                                               std::size_t g, const float2 (&v)[4]) {
            if (channels == 1) {
                auto *o = reinterpret_cast<float4 *>(out + 4 * g);
                o[0] = make_float4(v[0].x, v[0].y, v[1].x, v[1].y);
                o[1] = make_float4(v[2].x, v[2].y, v[3].x, v[3].y);
            } else if (channels == 2) {
                // Samples are (k,c0) (k,c1) (k+1,c0) (k+1,c1) with k = 2g.
                *reinterpret_cast<float4 *>(out + 2 * g) = make_float4(v[0].x, v[0].y, v[2].x, v[2].y);
                *reinterpret_cast<float4 *>(out + frames + 2 * g) = make_float4(v[1].x, v[1].y, v[3].x, v[3].y);
            } else {
                for (int j = 0; j < 4; ++j) {
                    const std::size_t s = 4 * g + j;
                    const std::size_t k = s / channels;
                    out[(s - k * channels) * frames + k] = v[j];
                }
            }
        }

        template <WireFormat F>
        __global__ void unpack_vec4_kernel(const std::uint8_t *__restrict__ wire, std::uint32_t channels,
                                           std::size_t frames, float2 *__restrict__ out) {
            const std::size_t groups = frames * channels / 4;
            for (std::size_t g = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; g < groups;
                 g += std::size_t{gridDim.x} * blockDim.x) {
                float2 v[4];
                if constexpr (F == WireFormat::packed12) {
                    const auto *w = reinterpret_cast<const unsigned *>(wire + 12 * g);
                    const unsigned w0 = __ldg(w), w1 = __ldg(w + 1), w2 = __ldg(w + 2);
                    v[0] = decode12(byte_of(w0, 0), byte_of(w0, 1), byte_of(w0, 2));
                    v[1] = decode12(byte_of(w0, 3), byte_of(w1, 0), byte_of(w1, 1));
                    v[2] = decode12(byte_of(w1, 2), byte_of(w1, 3), byte_of(w2, 0));
                    v[3] = decode12(byte_of(w2, 1), byte_of(w2, 2), byte_of(w2, 3));
                } else {
                    const uint2 w = __ldg(reinterpret_cast<const uint2 *>(wire + 8 * g));
                    v[0] = decode8(byte_of(w.x, 0), byte_of(w.x, 1));
                    v[1] = decode8(byte_of(w.x, 2), byte_of(w.x, 3));
                    v[2] = decode8(byte_of(w.y, 0), byte_of(w.y, 1));
                    v[3] = decode8(byte_of(w.y, 2), byte_of(w.y, 3));
                }
                store4(out, channels, frames, g, v);
            }
        }

        int grid_for(std::size_t work_items) {
            static const int max_blocks = [] {
                int device = 0, sms = 0;
                CURF_CUDA_CHECK(cudaGetDevice(&device));
                CURF_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
                return sms * 32; // enough resident blocks for latency hiding; grid-stride does the rest
            }();
            const auto needed = (work_items + kBlock - 1) / kBlock;
            return static_cast<int>(std::max<std::size_t>(1, std::min<std::size_t>(needed, max_blocks)));
        }

        bool aligned16(const void *p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; }

    } // namespace

    const char *to_string(UnpackVariant v) noexcept {
        switch (v) {
        case UnpackVariant::naive:
            return "naive";
        case UnpackVariant::vec4:
            return "vec4";
        }
        return "?";
    }

    void unpack_planar(const std::uint8_t *d_wire, WireFormat format, std::uint32_t channels, std::size_t frames,
                       cf32 *d_out, cudaStream_t stream, UnpackVariant variant) {
        const std::size_t total = frames * channels;
        if (total == 0) return;
        auto *out = reinterpret_cast<float2 *>(d_out);

        const bool vec_ok = variant == UnpackVariant::vec4 && total % 4 == 0 && frames % 2 == 0 &&
                            aligned16(d_wire) && aligned16(d_out);
        if (vec_ok) {
            const int grid = grid_for(total / 4);
            if (format == WireFormat::packed12)
                unpack_vec4_kernel<WireFormat::packed12><<<grid, kBlock, 0, stream>>>(d_wire, channels, frames, out);
            else
                unpack_vec4_kernel<WireFormat::signed8><<<grid, kBlock, 0, stream>>>(d_wire, channels, frames, out);
        } else {
            const int grid = grid_for(total);
            if (format == WireFormat::packed12)
                unpack_naive_kernel<WireFormat::packed12><<<grid, kBlock, 0, stream>>>(d_wire, channels, frames, out);
            else
                unpack_naive_kernel<WireFormat::signed8><<<grid, kBlock, 0, stream>>>(d_wire, channels, frames, out);
        }
        CURF_CUDA_CHECK(cudaGetLastError());
    }

} // namespace curf::cuda
