#include "curf/cuda/pipeline.hpp"
#include "curf/sources.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace curf;

namespace {

    constexpr double kFs = 61.44e6;
    constexpr std::size_t kNfft = 4096;

    SyntheticConfig tones_config() {
        SyntheticConfig sc;
        sc.format = {WireFormat::packed12, 2, kFs};
        // Exactly on PSD bin centres (bin = fs / nfft): +300 bins and -700 bins.
        sc.tones = {{{300 * kFs / kNfft, 0.5}}, {{-700 * kFs / kNfft, 0.25}}};
        sc.noise_rms = 0.001;
        return sc;
    }

} // namespace

TEST_CASE("pipeline PSD finds the synthetic tones with the right power", "[cuda][pipeline]") {
    SyntheticSource src(tones_config());
    cuda::PipelineConfig cfg;
    cfg.block_frames = std::size_t{1} << 16;
    cfg.ring_slots = 8;
    cfg.depth = 3;
    cfg.nfft = kNfft;
    cfg.seconds = 0;
    cfg.max_blocks = 20;

    std::vector<float> last;
    std::uint64_t callbacks = 0, last_first = 0;
    const auto st = cuda::run_pipeline(src, cfg, [&](const float *psd, std::uint32_t ch, std::size_t nfft, std::uint64_t first) {
        last.assign(psd, psd + ch * nfft);
        ++callbacks;
        last_first = first;
    });

    CHECK(st.blocks == 20);
    CHECK(callbacks == 20);
    CHECK(st.frames == 20 * cfg.block_frames);
    CHECK(st.gap_frames == 0);
    CHECK(st.dropped_frames == 0);
    CHECK(last_first == 19 * cfg.block_frames);

    auto peak = [&](std::uint32_t c) {
        const float *p = last.data() + c * kNfft;
        return static_cast<std::size_t>(std::max_element(p, p + kNfft) - p);
    };
    // fft-shifted: DC at nfft/2
    CHECK(peak(0) == kNfft / 2 + 300);
    CHECK(peak(1) == kNfft / 2 - 700);
    CHECK(last[peak(0)] == Catch::Approx(0.25).epsilon(0.02));            // 0.5^2
    CHECK(last[kNfft + peak(1)] == Catch::Approx(0.0625).epsilon(0.02));  // 0.25^2
}

TEST_CASE("pipeline back-pressure loses nothing; pageable ring still works", "[cuda][pipeline]") {
    SyntheticSource src(tones_config());
    cuda::PipelineConfig cfg;
    cfg.block_frames = std::size_t{1} << 15;
    cfg.ring_slots = 4;
    cfg.depth = 2;
    cfg.nfft = 1024;
    cfg.seconds = 0;
    cfg.max_blocks = 200;
    cfg.register_ring = false;
    const auto st = cuda::run_pipeline(src, cfg);
    CHECK(st.blocks == 200);
    CHECK(st.gap_frames == 0);
    CHECK(st.dropped_frames == 0);
}

TEST_CASE("pipeline stops on a time limit", "[cuda][pipeline]") {
    SyntheticSource src(tones_config());
    cuda::PipelineConfig cfg;
    cfg.seconds = 0.3;
    const auto st = cuda::run_pipeline(src, cfg);
    CHECK(st.blocks > 0);
    CHECK(st.wall_s < 2.0);
    CHECK(st.gap_frames == 0);
}
