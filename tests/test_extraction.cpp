#include "curf/extraction.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <numeric>
#include <stdexcept>

using namespace curf;

TEST_CASE("signal extraction FIR defines gain, delay and rejection", "[extraction]") {
    for (const unsigned d : {1u, 3u, 16u, 256u}) {
        ExtractionConfig c;
        c.decimation = d;
        c.block_frames = 1024 * d;
        const auto h = extraction_filter(c);
        CHECK(h.size() == 32 * d + 1);
        CHECK(std::accumulate(h.begin(), h.end(), 0.0) == Catch::Approx(1.0).margin(2e-7));
        double symmetry = 0;
        for (std::size_t k = 0; k < h.size(); ++k)
            symmetry = std::max(symmetry, double(std::abs(h[k] - h[h.size() - 1 - k])));
        CHECK(symmetry < 1e-7);
        auto response = [&](double f) {
            std::complex<double> a = 0;
            for (std::size_t k = 0; k < h.size(); ++k)
                a += double(h[k]) * std::polar(1.0, -2 * std::numbers::pi * f * double(k) / d);
            return std::abs(a);
        };
        CHECK(response(0.25) == Catch::Approx(1.0).margin(5e-4));
        CHECK(response(0.5) < 5e-4);
    }
}

TEST_CASE("continuous extraction reference has causal zero history and declared delay", "[extraction]") {
    ExtractionConfig c;
    c.decimation = 3;
    c.block_frames = 192;
    c.centre_hz = 0;
    std::vector<cf32> x(2 * c.block_frames), y(x.size() / c.decimation);
    x[0] = {1, 0};
    ref::extract_signal(c, x, 0, y);
    const auto h = extraction_filter(c);
    for (std::size_t j = 0; j < y.size(); ++j) {
        CHECK(y[j].re == (j * c.decimation < h.size() ? h[j * c.decimation] : 0));
        CHECK(y[j].im == 0.f);
    }
    CHECK(std::max_element(y.begin(), y.end(), [](cf32 a, cf32 b) { return a.re < b.re; }) - y.begin() == 16);
    c.decimation = 0;
    CHECK_THROWS_AS(c.validate(), std::invalid_argument);
    c.decimation = 7;
    CHECK_THROWS_AS(c.validate(), std::invalid_argument);
}

TEST_CASE("extraction oscillator preserves increments at large hardware timestamps", "[extraction]") {
    ExtractionConfig c;
    c.centre_hz = -3012345;
    const std::uint64_t first = (std::uint64_t{1} << 60) + 12345;
    const double a = extraction_phase_cycles(c, first), b = extraction_phase_cycles(c, first + 262144);
    const double want = std::remainder(c.centre_hz / c.sample_rate * 262144, 1.0);
    CHECK(std::abs(std::remainder(b - a - want, 1.0)) < 1e-10);
}
