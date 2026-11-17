#include "curf/dsss_lock.hpp"
#include "dsss_test_signal.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <limits>
#include <stdexcept>

TEST_CASE("Barker reference locates a spread carrier with residual offset", "[dsss]") {
    const double fs=GENERATE(22e6,23.04e6,30.72e6,46.08e6,61.44e6,64e6);
    const double offset=GENERATE(-1.35e6,.55e6);
    const auto input=dsss_test::signal(fs,offset);
    const auto r=curf::ref::dsss_frequency(input,fs);
    INFO("rate "<<fs<<" offset "<<offset<<" estimate "<<r.offset_hz<<" coherence "<<r.coherence);
    CHECK(r.valid()); CHECK(std::abs(r.offset_hz-offset)<=100e3);
}
TEST_CASE("Barker reference rejects silence noise and a single tone", "[dsss]") {
    const double fs=GENERATE(23.04e6,61.44e6);
    std::vector<float> input(curf::barker11::length*3);
    SECTION("silence") {}
    SECTION("noise") {
        std::mt19937 rng(75);std::normal_distribution<float> noise;
        for (unsigned j=0;j<curf::barker11::length;++j) { input[3*j]=noise(rng);input[3*j+1]=noise(rng); }
    }
    SECTION("tone") {
        for (unsigned j=0;j<curf::barker11::length;++j) {
            const double phase=2*std::numbers::pi*.9e6*j/fs;
            input[3*j]=static_cast<float>(std::cos(phase));input[3*j+1]=static_cast<float>(std::sin(phase));
        }
    }
    const auto r=curf::ref::dsss_frequency(input,fs);
    CHECK_FALSE(r.valid()); CHECK(std::isfinite(r.coherence)); CHECK(r.coherence<=1.f);
}
TEST_CASE("Barker antialias filter rejects a code on another frequency image", "[dsss]") {
    const double offset=GENERATE(11e6,22e6);
    const auto r=curf::ref::dsss_frequency(dsss_test::signal(61.44e6,offset),61.44e6);
    CHECK_FALSE(r.valid());
}
TEST_CASE("Barker reference validates shape rate and finite values", "[dsss]") {
    auto input=dsss_test::signal(30.72e6,0);
    CHECK_THROWS_AS(curf::ref::dsss_frequency({},30.72e6),std::invalid_argument);
    CHECK_THROWS_AS(curf::ref::dsss_frequency(input,10e6),std::invalid_argument);
    input[0]=std::numeric_limits<float>::quiet_NaN();
    CHECK_THROWS_AS(curf::ref::dsss_frequency(input,30.72e6),std::invalid_argument);
}
