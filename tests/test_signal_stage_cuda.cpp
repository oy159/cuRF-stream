// Stage 2 extraction: centre, bandwidth and extraction rate of known signals in a wideband capture.

#include "curf/cuda/memory.hpp"
#include "curf/runtime/signal_stage.hpp"
#include "curf/sources.hpp"
#include "curf/extraction.hpp"
#include "curf/reference.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <map>
#include <numbers>
#include <random>
#include <vector>

using namespace curf;

namespace {

    constexpr double kFs = 61.44e6;
    constexpr std::uint32_t kM = 512, kBlock = 1u << 17, kWindowSamples = 1u << 18;

    std::vector<std::uint8_t> pack(const std::vector<std::complex<double>> &x) {
        std::vector<std::uint8_t> w(x.size() * 3);
        auto q = [](double v) { return static_cast<int>(std::clamp(std::lround(v * 2048), -2048L, 2047L)) & 0xFFF; };
        for (std::size_t k = 0; k < x.size(); ++k) {
            const int i = q(x[k].real()), qq = q(x[k].imag());
            w[3 * k] = static_cast<std::uint8_t>(i >> 4);
            w[3 * k + 1] = static_cast<std::uint8_t>(((i & 0xF) << 4) | (qq >> 8));
            w[3 * k + 2] = static_cast<std::uint8_t>(qq & 0xFF);
        }
        return w;
    }

    // channels whose centre lies inside [f - bw/2, f + bw/2] (at least the nearest one)
    std::vector<std::uint32_t> channels_of(double f, double bw) {
        std::vector<std::uint32_t> out;
        const double sp = kFs / kM;
        for (int ks = -static_cast<int>(kM) / 2 + 1; ks < static_cast<int>(kM) / 2; ++ks)
            if (std::fabs(ks * sp - f) <= bw / 2) out.push_back(static_cast<std::uint32_t>(ks >= 0 ? ks : ks + static_cast<int>(kM)));
        if (out.empty()) {
            const int ks = static_cast<int>(std::lround(f / sp));
            out.push_back(static_cast<std::uint32_t>(ks >= 0 ? ks : ks + static_cast<int>(kM)));
        }
        return out;
    }

} // namespace

TEST_CASE("stage 2 finds centre and bandwidth of wide and narrow signals", "[cuda][stage2]") {
    const std::filesystem::path dir = CURF_S2_MODEL_DIR;
    if (!std::filesystem::exists(dir / "s2.curf")) SKIP("no stage-2 model in " + dir.string());
    const auto model = engine::Model::load(dir / "s2.curf");

    constexpr int kWindows = 40;
    const std::size_t n = std::size_t{kWindows} * kWindowSamples;
    std::vector<std::complex<double>> x(n);
    std::mt19937 rng(3);
    std::normal_distribution<double> nd(0, 0.004);
    // wide: OFDM-like, 4.5 MHz of random QPSK subcarriers (15 kHz spacing) at +8 MHz
    const double f_wide = 8e6, bw_wide = 4.5e6, f_nar = -3.0e6;
    {
        const int sc = static_cast<int>(bw_wide / 15e3);
        std::uniform_int_distribution<int> qd(0, 3);
        const std::size_t sym = static_cast<std::size_t>(kFs / 15e3); // 4096 samples
        std::vector<std::complex<double>> phase(sc);
        for (std::size_t s0 = 0; s0 < n; s0 += sym) {
            for (auto &p : phase) p = std::polar(1.0, std::numbers::pi / 4 + std::numbers::pi / 2 * qd(rng));
            for (std::size_t k = s0; k < std::min(n, s0 + sym); ++k) {
                std::complex<double> v = 0;
                for (int c = 0; c < sc; c += 8) // every 8th subcarrier: enough spread, cheap to compute
                    v += phase[c] * std::polar(1.0, 2 * std::numbers::pi * (f_wide - bw_wide / 2 + c * 15e3) * static_cast<double>(k) / kFs);
                x[k] += 0.02 * v;
            }
        }
    }
    // narrow: FM, 1 kHz tone at 3 kHz deviation (Carson ~ 8 kHz) at -3 MHz
    for (std::size_t k = 0; k < n; ++k) {
        const double t = static_cast<double>(k) / kFs;
        x[k] += 0.1 * std::polar(1.0, 2 * std::numbers::pi * f_nar * t + 3.0 * std::sin(2 * std::numbers::pi * 1e3 * t));
        x[k] += std::complex<double>(nd(rng), nd(rng));
    }
    const auto wire = pack(x);
    constexpr std::uint64_t source_origin = std::uint64_t{1} << 30;
    const double origin_s = static_cast<double>(source_origin) / kFs;

    SignalStageConfig cfg;
    cfg.engine.buckets = {1, 4, 16, 64};
    cfg.engine.tune_iters = 2;
    std::vector<SignalResult> got;
    std::vector<std::pair<SignalResult,std::vector<float>>> model_inputs;
    cfg.on_input = [&](const SignalResult &r, std::span<const float> input) {
        if (r.fc_hz < 0) model_inputs.push_back({r, {input.begin(),input.end()}});
    };
    SignalStage st(cfg, model, StreamFormat{WireFormat::packed12, 1, kFs}, kBlock, [&](const SignalResult &r) { got.push_back(r); });
    cuda::DeviceBuffer<std::uint8_t> d(kBlock * 3);
    const auto wide_ch = channels_of(f_wide, bw_wide), nar_ch = channels_of(f_nar, 1);
    for (int w = 0; w < kWindows; ++w) {
        for (int b = 0; b < 2; ++b) {
            const std::size_t first = (std::size_t{static_cast<unsigned>(w)} * 2 + b) * kBlock;
            CURF_CUDA_CHECK(cudaMemcpy(d.get(), wire.data() + first * 3, d.bytes(), cudaMemcpyHostToDevice));
            st.enqueue(0, d.get(), kBlock, source_origin + first, nullptr);
            CURF_CUDA_CHECK(cudaDeviceSynchronize());
        }
        std::vector<WindowResult> det;
        for (auto c : wide_ch) det.push_back({0, c, 0, 0, 1e-3f});
        for (auto c : nar_ch) det.push_back({0, c, 0, 0, 1e-2f});
        st.on_window(static_cast<std::uint64_t>(w), source_origin + std::uint64_t{static_cast<unsigned>(w)} * kWindowSamples, det.data(), det.size());
        st.flush();
    }
    REQUIRE(!got.empty());
    std::map<bool, std::vector<SignalResult>> by; // wide?
    for (const auto &r : got) by[r.fc_hz > 0].push_back(r);
    REQUIRE(by[true].size() >= 1);
    REQUIRE(by[false].size() >= 1);
    for (const auto &r : by[true]) {
        INFO("wide: fc " << r.fc_hz << " bw " << r.bw_hz << " fs " << r.fs_hz);
        CHECK(std::fabs(r.fc_hz - f_wide) < 0.1 * bw_wide);
        CHECK(r.bw_hz > 0.8 * bw_wide);
        CHECK(r.bw_hz < 1.25 * bw_wide);
        CHECK(r.fs_hz >= 2 * r.bw_hz * 0.99);
        CHECK(r.fs_hz < 3.01 * r.bw_hz);
        CHECK(r.activity_fraction >= 0.f);
        CHECK(r.activity_fraction <= 1.f);
    }
    for (const auto &r : by[false]) {
        INFO("narrow: fc " << r.fc_hz << " bw " << r.bw_hz << " fs " << r.fs_hz << " class " << r.cls);
        CHECK(std::fabs(r.fc_hz - f_nar) < 3e3);
        CHECK(r.bw_hz > 4e3);
        CHECK(r.bw_hz < 16e3);
        CHECK(r.fs_hz < 3.01 * r.bw_hz);
    }
    for (const auto &r : by[false]) {
        CHECK(r.narrow);
        CHECK(r.activity_fraction == -1.f);
        CHECK(std::abs(r.t1_s - r.t0_s - 1024 / r.fs_hz) < 1e-9);
        CHECK(r.processing_ms > 0);
        CHECK(std::abs(r.source_time_s - origin_s - static_cast<double>(r.window+1)*kWindowSamples/kFs) < 1e-9);
    }
    REQUIRE(by[false].size() >= 2);
    for (std::size_t i = 1; i < by[false].size(); ++i) {
        const auto &a = by[false][i-1], &b = by[false][i];
        if (a.fs_hz == b.fs_hz) CHECK(std::abs(b.t0_s - a.t1_s) < 1e-9);
    }
    // Verify the actual normalised [L][3] rows handed to the engine, including narrow-buffer carry.
    std::vector<cf32> decoded(n);
    ref::unpack_planar(wire.data(), WireFormat::packed12, 1, n, decoded.data());
    std::map<std::pair<double,double>, std::vector<cf32>> references;
    REQUIRE(!model_inputs.empty());
    for (const auto &[r, input] : model_inputs) {
        const auto key = std::pair{r.fc_hz, r.fs_hz};
        if (!references.count(key)) {
            const auto w = static_cast<unsigned>(std::lround(r.fs_hz * kWindowSamples / kFs));
            auto &gold = references[key]; gold.resize(kWindows * w);
            ref::resample_signal(kFs, r.fc_hz, extraction_ratio(kWindowSamples,w), decoded, source_origin, gold);
        }
        const auto &gold = references.at(key);
        const auto begin = std::llround((r.t0_s - origin_s) * r.fs_hz + 16);
        REQUIRE(begin >= 0); REQUIRE(std::size_t(begin) + 1024 <= gold.size());
        double energy = 0;
        for (unsigned j = 0; j < 1024; ++j) {
            const auto v = gold[std::size_t(begin)+j]; energy += double(v.re)*v.re + double(v.im)*v.im;
        }
        const double scale = std::sqrt(1024/energy);
        double err = 0;
        for (unsigned j = 0; j < 1024; ++j) {
            const auto v = gold[std::size_t(begin)+j];
            err = std::max(err, std::hypot(input[3*j] - v.re*scale, input[3*j+1] - v.im*scale));
            CHECK(std::abs(input[3*j+2] - (std::log10(r.bw_hz)-5)/2) < 1e-6);
        }
        INFO("model input error " << err); CHECK(err < 5e-5);
    }
    const auto s = st.stats();
    CHECK(s.skipped_ring == 0);
    CHECK(s.skipped_busy == 0);
}

TEST_CASE("detection-only pipeline drives stage 2 without a stage-1 engine", "[cuda][stage2][classifier]") {
    const std::filesystem::path dir = CURF_S2_MODEL_DIR;
    if (!std::filesystem::exists(dir / "s2.curf")) SKIP("no stage-2 model");
    const auto model = engine::Model::load(dir / "s2.curf");
    SignalStageConfig s2cfg;
    s2cfg.max_signals = 4; s2cfg.max_tracks = 16;
    s2cfg.drop_when_busy = false;
    s2cfg.engine.buckets = {1,4}; s2cfg.engine.autotune = false;
    std::vector<SignalResult> results;
    SignalStage stage(s2cfg, model, {WireFormat::packed12, 2, kFs}, kBlock,
                      [&](const SignalResult &r) { results.push_back(r); });
    SyntheticConfig source_cfg;
    source_cfg.format = {WireFormat::packed12, 2, kFs};
    source_cfg.tones = {{{2e6, .25}}, {{-4e6, .25}}}; source_cfg.noise_rms = .001;
    SyntheticSource source(source_cfg);
    // Only descriptors: constructing the old engine with this empty graph would fail.
    engine::Model descriptor;
    descriptor.tensors = {{2,1024,"input"}, {11,1,"output"}};
    descriptor.input = 0; descriptor.output = 1;
    ClassifierConfig cfg;
    cfg.classify_channels = false; cfg.max_windows = 132; cfg.seconds = 0;
    cfg.detect.peak_detection = true; cfg.taps = {&stage};
    const auto st = run_classifier(source, descriptor, cfg,
        [&](auto w, auto first, const WindowResult *r, std::size_t n, auto ready) { stage.on_window(w,first,r,n,ready); });
    CHECK(st.dropped_frames == 0); CHECK(st.gap_frames == 0); CHECK(st.classified == 0);
    REQUIRE(results.size() >= 4);
    CHECK(stage.stats().skipped_busy == 0); CHECK(stage.stats().skipped_ring == 0);
    for (const auto &r : results) {
        CHECK(r.narrow); REQUIRE(r.stream < 2);
        CHECK(std::abs(r.fc_hz - (r.stream == 0 ? 2e6 : -4e6)) < 3e3);
        CHECK(r.probs.size() == 13); CHECK(r.processing_ms > 0);
    }
}
