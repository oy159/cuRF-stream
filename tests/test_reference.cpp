#include "curf/reference.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

using curf::cf32;
using curf::WireFormat;

TEST_CASE("packed12 wire layout matches OmniCore", "[reference]") {
    // I = 0x7FF (+2047), Q = 0x800 (-2048): bytes I[11:4], I[3:0]:Q[11:8], Q[7:0]
    const std::array<std::uint8_t, 3> wire{0x7F, 0xF8, 0x00};
    const cf32 v = curf::ref::unpack_sample(wire.data(), WireFormat::packed12);
    CHECK(v.re == 2047.0f / 2048.0f);
    CHECK(v.im == -1.0f);

    // I = 0xFFF (-1), Q = 0x001 (+1)
    const std::array<std::uint8_t, 3> wire2{0xFF, 0xF0, 0x01};
    const cf32 w = curf::ref::unpack_sample(wire2.data(), WireFormat::packed12);
    CHECK(w.re == -1.0f / 2048.0f);
    CHECK(w.im == 1.0f / 2048.0f);
}

TEST_CASE("packed12 round-trips every code", "[reference]") {
    std::array<std::uint8_t, 3> wire{};
    for (int i = -2048; i < 2048; ++i) {
        const int q = -1 - i; // covers the whole range for Q as well
        const cf32 in{static_cast<float>(i) / 2048.0f, static_cast<float>(q) / 2048.0f};
        curf::ref::pack_sample(in, WireFormat::packed12, wire.data());
        const cf32 out = curf::ref::unpack_sample(wire.data(), WireFormat::packed12);
        REQUIRE(out.re == in.re);
        REQUIRE(out.im == in.im);
    }
}

TEST_CASE("signed8 round-trips every code", "[reference]") {
    std::array<std::uint8_t, 2> wire{};
    for (int i = -128; i < 128; ++i) {
        const cf32 in{static_cast<float>(i) / 128.0f, static_cast<float>(-1 - i) / 128.0f};
        curf::ref::pack_sample(in, WireFormat::signed8, wire.data());
        const cf32 out = curf::ref::unpack_sample(wire.data(), WireFormat::signed8);
        REQUIRE(out.re == in.re);
        REQUIRE(out.im == in.im);
    }
}

TEST_CASE("packing saturates, rounds half away from zero, maps NaN to zero", "[reference]") {
    std::array<std::uint8_t, 3> wire{};
    auto roundtrip = [&](float re) {
        curf::ref::pack_sample({re, 0.0f}, WireFormat::packed12, wire.data());
        return curf::ref::unpack_sample(wire.data(), WireFormat::packed12).re * 2048.0f;
    };
    CHECK(roundtrip(5.0f) == 2047.0f);
    CHECK(roundtrip(-5.0f) == -2048.0f);
    CHECK(roundtrip(std::numeric_limits<float>::quiet_NaN()) == 0.0f);
    CHECK(roundtrip(std::numeric_limits<float>::infinity()) == 2047.0f);
    CHECK(roundtrip(2.5f / 2048.0f) == 3.0f);
    CHECK(roundtrip(-2.5f / 2048.0f) == -3.0f);
    CHECK(roundtrip(2.49f / 2048.0f) == 2.0f);
}

TEST_CASE("unpack_planar de-interleaves channels", "[reference]") {
    constexpr std::uint32_t channels = 3;
    constexpr std::size_t frames = 5;
    std::vector<std::uint8_t> wire(frames * channels * 3);
    for (std::size_t k = 0; k < frames; ++k)
        for (std::uint32_t c = 0; c < channels; ++c) {
            const cf32 v{static_cast<float>(k) / 2048.0f, static_cast<float>(c) / 2048.0f};
            curf::ref::pack_sample(v, WireFormat::packed12, wire.data() + (k * channels + c) * 3);
        }
    std::vector<cf32> out(frames * channels);
    curf::ref::unpack_planar(wire.data(), WireFormat::packed12, channels, frames, out.data());
    for (std::uint32_t c = 0; c < channels; ++c)
        for (std::size_t k = 0; k < frames; ++k) {
            CHECK(out[c * frames + k].re * 2048.0f == static_cast<float>(k));
            CHECK(out[c * frames + k].im * 2048.0f == static_cast<float>(c));
        }
}
