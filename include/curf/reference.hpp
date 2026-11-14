#pragma once

// Scalar CPU reference implementations. They define the expected results that
// every optimised CPU/GPU implementation is tested against, so they favour
// clarity over speed.

#include "curf/format.hpp"

#include <cstddef>
#include <cstdint>

namespace curf::ref {

    // Wire frames -> planar complex float, one contiguous run per channel:
    //   out[c * frames + k] = channel c, frame k
    void unpack_planar(const std::uint8_t *wire, WireFormat format, std::uint32_t channels, std::size_t frames,
                       cf32 *out) noexcept;

    // One complex sample -> wire bytes. Quantisation rounds half away from zero
    // and saturates to the 12-bit (or 8-bit) range; NaN becomes zero.
    void pack_sample(cf32 value, WireFormat format, std::uint8_t *wire) noexcept;

    // One wire sample -> complex float.
    cf32 unpack_sample(const std::uint8_t *wire, WireFormat format) noexcept;

} // namespace curf::ref
