#pragma once

#include "curf/format.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace curf::cuda {

    enum class UnpackVariant : std::uint8_t {
        naive,  // one thread per wire sample, byte loads (baseline)
        vec4,   // one thread per 4 wire samples, 32/64-bit loads, 16-byte stores
        best = vec4
    };

    const char *to_string(UnpackVariant v) noexcept;

    // Device wire frames -> device planar complex float (out[c * frames + k]),
    // matching curf::ref::unpack_planar bit for bit. Asynchronous on `stream`.
    // vec4 falls back to naive when its alignment/size preconditions do not hold
    // (wire/out 16-byte aligned, frames * channels % 4 == 0, frames even).
    void unpack_planar(const std::uint8_t *d_wire, WireFormat format, std::uint32_t channels, std::size_t frames,
                       cf32 *d_out, cudaStream_t stream, UnpackVariant variant = UnpackVariant::best);

} // namespace curf::cuda
