#include "curf/cuda/memory.hpp"
#include "curf/cuda/unpack.hpp"
#include "curf/reference.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstring>
#include <random>
#include <vector>

using namespace curf;

namespace {

    // Runs the GPU unpack on `wire` placed `offset` bytes into a device buffer and
    // compares bitwise with the CPU reference.
    void check_against_reference(WireFormat format, std::uint32_t channels, std::size_t frames,
                                 cuda::UnpackVariant variant, std::size_t offset) {
        const std::size_t bytes = frames * channels * wire_sample_bytes(format);
        std::vector<std::uint8_t> wire(bytes);
        std::mt19937 rng(static_cast<unsigned>(frames * 31 + channels));
        for (auto &b : wire) b = static_cast<std::uint8_t>(rng());

        std::vector<cf32> expected(frames * channels);
        ref::unpack_planar(wire.data(), format, channels, frames, expected.data());

        cuda::DeviceBuffer<std::uint8_t> d_wire(bytes + offset);
        cuda::DeviceBuffer<cf32> d_out(frames * channels);
        CURF_CUDA_CHECK(cudaMemcpy(d_wire.get() + offset, wire.data(), bytes, cudaMemcpyHostToDevice));
        CURF_CUDA_CHECK(cudaMemset(d_out.get(), 0xff, d_out.bytes()));
        cuda::unpack_planar(d_wire.get() + offset, format, channels, frames, d_out.get(), nullptr, variant);
        CURF_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<cf32> got(frames * channels);
        CURF_CUDA_CHECK(cudaMemcpy(got.data(), d_out.get(), d_out.bytes(), cudaMemcpyDeviceToHost));
        REQUIRE(std::memcmp(got.data(), expected.data(), got.size() * sizeof(cf32)) == 0);
    }

} // namespace

TEST_CASE("GPU unpack matches the CPU reference bit for bit", "[cuda][unpack]") {
    const auto format = GENERATE(WireFormat::packed12, WireFormat::signed8);
    const auto variant = GENERATE(cuda::UnpackVariant::naive, cuda::UnpackVariant::vec4);
    const std::uint32_t channels = GENERATE(1u, 2u, 3u, 4u);
    const std::size_t frames = GENERATE(std::size_t{1}, std::size_t{2}, std::size_t{7}, std::size_t{1000},
                                        std::size_t{4096}, std::size_t{1} << 20);
    CAPTURE(static_cast<int>(format), cuda::to_string(variant), channels, frames);
    check_against_reference(format, channels, frames, variant, 0);
}

TEST_CASE("GPU unpack falls back correctly on misaligned input", "[cuda][unpack]") {
    const std::size_t offset = GENERATE(std::size_t{1}, std::size_t{3}, std::size_t{12});
    CAPTURE(offset);
    check_against_reference(WireFormat::packed12, 2, 4096, cuda::UnpackVariant::vec4, offset);
}
