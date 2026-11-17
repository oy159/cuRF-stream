#include "curf/cuda/extraction.hpp"
#include "curf/cuda/memory.hpp"
#include "curf/reference.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>

using namespace curf;

TEST_CASE("overlap-save extraction matches continuous FIR through bursts and block seams", "[cuda][extraction]") {
    // Non-power-of-two D, H > block, non-bin-centred LO, positive/negative frequencies and wire quantisation.
    for (const unsigned d : {1u, 3u, 16u, 64u, 256u}) {
        ExtractionConfig c;
        c.decimation = d;
        c.block_frames = d == 3 ? 3072 : 4096;
        c.centre_hz = d == 1 ? 1.123456e6 : -3.012345e6;
        constexpr std::size_t blocks = 4;
        const std::uint64_t first = d == 256 ? (std::uint64_t{1} << 60) + 12345 : 123456789;
        const std::size_t count = blocks * c.block_frames, step = c.output_frames();
        std::mt19937 rng(9);
        std::normal_distribution<float> noise(0, 0.01f);
        std::vector<cf32> x(count), expected(count / d), got(expected.size());
        for (std::size_t n = 0; n < count; ++n) {
            const double t = double(first + n) / c.sample_rate;
            const double ph = 2 * std::numbers::pi * (c.centre_hz + 0.113 * c.output_rate()) * t;
            // Burst crosses the seam; silence and noise on both sides exercise history, not just steady tones.
            const float a = n >= c.block_frames - 113 && n < 2 * c.block_frames + 257 ? 0.3f : 0.f;
            const cf32 v{a * float(std::cos(ph)) + noise(rng), a * float(std::sin(ph)) + noise(rng)};
            std::uint8_t wire[3];
            ref::pack_sample(v, WireFormat::packed12, wire);
            x[n] = ref::unpack_sample(wire, WireFormat::packed12);
        }
        ref::extract_signal(c, x, first, expected);
        cuda::SignalExtractor ex(c);
        cuda::DeviceBuffer<cf32> in(c.block_frames), out(step);
        cuda::PinnedBuffer<cf32> host_in(x.size()), host_out(got.size());
        std::copy(x.begin(), x.end(), host_in.get());
        cudaStream_t s;
        CURF_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        for (std::size_t b = 0; b < blocks; ++b) {
            CURF_CUDA_CHECK(cudaMemcpyAsync(in.get(), host_in.get() + b * c.block_frames, in.bytes(), cudaMemcpyHostToDevice, s));
            ex.run(in.get(), out.get(), first + b * c.block_frames, s);
            CURF_CUDA_CHECK(cudaMemcpyAsync(host_out.get() + b * step, out.get(), out.bytes(), cudaMemcpyDeviceToHost, s));
        }
        CURF_CUDA_CHECK(cudaStreamSynchronize(s));
        std::copy_n(host_out.get(), got.size(), got.data());
        double error = 0, seam = 0;
        for (std::size_t j = 0; j < got.size(); ++j) {
            const double e = std::hypot(got[j].re - expected[j].re, got[j].im - expected[j].im);
            error = std::max(error, e);
            if (j % step < 4 || j % step + 4 >= step) seam = std::max(seam, e);
        }
        INFO("D=" << d << " max error=" << error << " seam error=" << seam);
        CHECK(error < 3e-6);
        CHECK(seam < 3e-6);
        CHECK_THROWS_AS(ex.run(in.get(), out.get(), first + count + 1, s), std::invalid_argument);

        // Reset must clear even a history longer than a block, and allow a new timestamp.
        ex.reset(s);
        CURF_CUDA_CHECK(cudaMemsetAsync(in.get(), 0, in.bytes(), s));
        ex.run(in.get(), out.get(), 42, s);
        CURF_CUDA_CHECK(cudaMemcpyAsync(host_out.get(), out.get(), out.bytes(), cudaMemcpyDeviceToHost, s));
        CURF_CUDA_CHECK(cudaStreamSynchronize(s));
        double after_reset = 0;
        for (std::size_t j = 0; j < step; ++j)
            after_reset = std::max(after_reset, double(std::hypot(host_out[j].re, host_out[j].im)));
        CHECK(after_reset == 0);
        CURF_CUDA_CHECK(cudaStreamDestroy(s));
    }
}
