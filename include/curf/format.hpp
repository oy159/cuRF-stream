#pragma once

#include <cstddef>
#include <cstdint>

namespace curf {

    // USB wire formats produced by OmniCore-SDR-Host (och_rx_acquire view).
    //  packed12: 3 bytes per complex sample: I[11:4], I[3:0]:Q[11:8], Q[7:0], two's complement.
    //  signed8:  2 bytes per complex sample: I[11:4], Q[11:4] (the top 8 of the 12 bits).
    // Multi-channel streams interleave whole samples in channel order per frame:
    //  frame k = [ch0 sample k][ch1 sample k]...
    enum class WireFormat : std::uint8_t { packed12, signed8 };

    constexpr std::size_t wire_sample_bytes(WireFormat f) noexcept { return f == WireFormat::packed12 ? 3 : 2; }

    // Normalisation shared by every unpack implementation: full scale is [-1, 1).
    //  packed12: value / 2048      signed8: value / 128
    constexpr float wire_scale(WireFormat f) noexcept {
        return f == WireFormat::packed12 ? 1.0f / 2048.0f : 1.0f / 128.0f;
    }

    struct StreamFormat {
        WireFormat wire = WireFormat::packed12;
        std::uint32_t channels = 1;
        double sample_rate = 61.44e6; // per channel, samples/s

        constexpr std::size_t frame_bytes() const noexcept { return wire_sample_bytes(wire) * channels; }
        constexpr double wire_bytes_per_second() const noexcept {
            return sample_rate * static_cast<double>(frame_bytes());
        }
    };

    // Complex float with the same size/alignment as CUDA's float2.
    struct alignas(8) cf32 {
        float re;
        float im;
    };
    static_assert(sizeof(cf32) == 8);

} // namespace curf
