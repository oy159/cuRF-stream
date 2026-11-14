#include "curf/reference.hpp"

#include <cmath>

namespace curf::ref {
    namespace {

        int sign_extend12(unsigned v) noexcept { return v >= 2048 ? static_cast<int>(v) - 4096 : static_cast<int>(v); }

        int quantize(float v, int lo, int hi, float scale) noexcept {
            if (std::isnan(v)) return 0;
            const float s = v * scale;
            if (s >= static_cast<float>(hi)) return hi;
            if (s <= static_cast<float>(lo)) return lo;
            return static_cast<int>(std::lround(s)); // half away from zero
        }

    } // namespace

    cf32 unpack_sample(const std::uint8_t *p, WireFormat format) noexcept {
        if (format == WireFormat::packed12) {
            const unsigned i = (static_cast<unsigned>(p[0]) << 4) | (p[1] >> 4);
            const unsigned q = ((p[1] & 15u) << 8) | p[2];
            return {static_cast<float>(sign_extend12(i)) / 2048.0f, static_cast<float>(sign_extend12(q)) / 2048.0f};
        }
        return {static_cast<float>(static_cast<std::int8_t>(p[0])) / 128.0f,
                static_cast<float>(static_cast<std::int8_t>(p[1])) / 128.0f};
    }

    void pack_sample(cf32 value, WireFormat format, std::uint8_t *p) noexcept {
        if (format == WireFormat::packed12) {
            const auto i = static_cast<unsigned>(quantize(value.re, -2048, 2047, 2048.0f)) & 4095u;
            const auto q = static_cast<unsigned>(quantize(value.im, -2048, 2047, 2048.0f)) & 4095u;
            p[0] = static_cast<std::uint8_t>(i >> 4);
            p[1] = static_cast<std::uint8_t>(((i & 15u) << 4) | (q >> 8));
            p[2] = static_cast<std::uint8_t>(q & 255u);
            return;
        }
        p[0] = static_cast<std::uint8_t>(static_cast<std::int8_t>(quantize(value.re, -128, 127, 128.0f)));
        p[1] = static_cast<std::uint8_t>(static_cast<std::int8_t>(quantize(value.im, -128, 127, 128.0f)));
    }

    void unpack_planar(const std::uint8_t *wire, WireFormat format, std::uint32_t channels, std::size_t frames,
                       cf32 *out) noexcept {
        const std::size_t bytes = wire_sample_bytes(format);
        for (std::size_t k = 0; k < frames; ++k)
            for (std::uint32_t c = 0; c < channels; ++c)
                out[c * frames + k] = unpack_sample(wire + (k * channels + c) * bytes, format);
    }

} // namespace curf::ref
