// curf_classify: the P3 end-to-end pipeline from the command line.
//
//   curf_classify [--source synthetic|file] [--file PATH] [--loop] [--format packed12|signed8]
//                 [--streams 2] [--rate 61.44e6] [--active N] [--tone-amp A] [--noise R] [--realtime]
//                 [--detect relative|absolute|all] [--threshold-db 10] [--abs-power P] [--compaction ballot|cub]
//                 [--seconds 5] [--windows N] [--buckets 1,8,32,128,512] [--drop] [--model DIR]
//                 [--ring SLOTS] [--print N] [--dump CSV] [--label NAME] [--int8] [--calib-percentile Q] [--tune-cache FILE]
//
// Synthetic source: noise plus, per stream, N tones at the centres of N channels spread over the band
// (random phases), so exactly N channels per stream carry energy - the load knob. Prints one JSON line of
// pipeline statistics; --print N also lists the results of the first N windows.

#include "curf/engine/model.hpp"
#include "curf/runtime/classifier.hpp"
#include "curf/sources.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <random>
#include <sstream>
#include <string>

using namespace curf;

int main(int argc, char **argv) {
    std::string source = "synthetic", file, label, model_dir = CURF_MODEL_DIR;
    bool loop = false, realtime = false;
    StreamFormat fmt{WireFormat::packed12, 2, 61.44e6};
    std::uint32_t active = 16;
    double tone_amp = -1, noise = 0.003;
    std::uint64_t print = 0;
    std::string dump_path; // CSV of every result: window,stream,channel,class,prob,power
    bool int8 = false;     // INT8 engine, calibrated on the model directory's golden windows
    ClassifierConfig cfg;
    cfg.engine.tune_iters = 10;
    // finer buckets than the engine default: the active count rounds up to the next bucket
    cfg.engine.buckets = {1, 8, 32, 64, 128, 192, 256, 384, 512};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--source") source = next();
        else if (a == "--file") file = next();
        else if (a == "--loop") loop = true;
        else if (a == "--format") fmt.wire = next() == "signed8" ? WireFormat::signed8 : WireFormat::packed12;
        else if (a == "--streams") fmt.channels = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--rate") fmt.sample_rate = std::stod(next());
        else if (a == "--active") active = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--tone-amp") tone_amp = std::stod(next());
        else if (a == "--noise") noise = std::stod(next());
        else if (a == "--realtime") realtime = true;
        else if (a == "--detect") {
            const auto m = next();
            cfg.detect.mode = m == "all" ? cuda::DetectMode::all : m == "absolute" ? cuda::DetectMode::absolute : cuda::DetectMode::relative;
        } else if (a == "--threshold-db") cfg.detect.threshold_db = std::stof(next());
        else if (a == "--abs-power") cfg.detect.absolute_power = std::stof(next());
        else if (a == "--compaction") cfg.detect.compaction = next() == "cub" ? cuda::Compaction::cub : cuda::Compaction::ballot;
        else if (a == "--seconds") cfg.seconds = std::stod(next());
        else if (a == "--windows") cfg.max_windows = std::stoull(next());
        else if (a == "--buckets") {
            cfg.engine.buckets.clear();
            std::stringstream ss(next());
            for (std::string x; std::getline(ss, x, ',');) cfg.engine.buckets.push_back(std::stoi(x));
        } else if (a == "--drop") cfg.drop_when_full = true;
        else if (a == "--ring") cfg.ring_slots = std::stoull(next());
        else if (a == "--dump") dump_path = next();
        else if (a == "--model") model_dir = next();
        else if (a == "--print") print = std::stoull(next());
        else if (a == "--label") label = next();
        else if (a == "--int8") int8 = true;
        else if (a == "--calib-percentile") cfg.engine.calib_percentile = std::stod(next());
        else if (a == "--tune-cache") cfg.engine.tune_cache = next();
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }

    const auto model = engine::Model::load(std::filesystem::path(model_dir) / "amc.curf");
    if (int8) {
        cfg.engine.int8 = true;
        cfg.engine.calibration = engine::Golden::load(std::filesystem::path(model_dir) / "amc.golden", model).input;
    }
    std::unique_ptr<ISource> src;
    if (source == "file") {
        src = std::make_unique<FileSource>(FileConfig{file, fmt, loop});
    } else {
        SyntheticConfig sc;
        sc.format = fmt;
        sc.noise_rms = noise;
        sc.realtime = realtime;
        const std::uint32_t M = cfg.channelizer.channels;
        const double amp = tone_amp > 0 ? tone_amp : 0.5 / std::sqrt(std::max<std::uint32_t>(active, 1));
        std::mt19937 rng(3);
        std::uniform_real_distribution<double> ph(0, 2 * std::numbers::pi);
        sc.tones.resize(fmt.channels);
        for (std::uint32_t s = 0; s < fmt.channels; ++s)
            for (std::uint32_t i = 0; i < std::min(active, M); ++i) {
                const std::uint32_t k = static_cast<std::uint32_t>((static_cast<std::uint64_t>(i) * M) / std::min(active, M));
                const long kk = k < M / 2 ? static_cast<long>(k) : static_cast<long>(k) - static_cast<long>(M);
                sc.tones[s].push_back({static_cast<double>(kk) * fmt.sample_rate / M, amp, ph(rng)});
            }
        src = std::make_unique<SyntheticSource>(sc);
    }

    std::FILE *dump = dump_path.empty() ? nullptr : std::fopen(dump_path.c_str(), "w");
    if (dump) std::fprintf(dump, "window,stream,channel,cls,prob,power\n");
    const auto st = run_classifier(*src, model, cfg, [&](std::uint64_t w, std::uint64_t first, const WindowResult *r, std::size_t n) {
        if (dump)
            for (std::size_t i = 0; i < n; ++i)
                std::fprintf(dump, "%llu,%u,%u,%u,%.4f,%.6g\n", static_cast<unsigned long long>(w), r[i].stream,
                             r[i].channel, r[i].cls, r[i].prob, r[i].power);
        if (w >= print) return;
        std::printf("window %llu (sample %llu): %zu active\n", static_cast<unsigned long long>(w),
                    static_cast<unsigned long long>(first), n);
        for (std::size_t i = 0; i < n; ++i)
            std::printf("  s%u ch%3u  class %u  p=%.3f  power %.1f dBFS\n", r[i].stream, r[i].channel, r[i].cls, r[i].prob,
                        10 * std::log10(std::max(r[i].power, 1e-20f)));
    });
    if (dump) std::fclose(dump);
    std::string js = st.to_json();
    if (!label.empty()) js.insert(1, "\"label\":\"" + label + "\",");
    if (int8) js.insert(1, "\"int8\":true,");
    std::printf("%s\n", js.c_str());
    return 0;
}
