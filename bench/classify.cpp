// curf_classify: the P3 end-to-end pipeline from the command line.
//
//   curf_classify [--source synthetic|file|omnicore] [--file PATH] [--loop] [--format packed12|signed8]
//                 [--streams 2] [--rate 61.44e6] [--active N] [--tone-amp A] [--noise R] [--realtime]
//                 [--detect relative|absolute|all] [--threshold-db 10] [--abs-power P] [--compaction ballot|cub]
//                 [--seconds 5] [--windows N] [--buckets 1,8,32,128,512] [--drop] [--model DIR]
//                 [--ring SLOTS] [--print N] [--dump CSV] [--label NAME] [--int8] [--calib-percentile Q] [--tune-cache FILE]
//                 [--fp32-input]   (INT8: gather fp32 windows; default fp16, which the fused stem rounds to anyway)
//
//                 [--ui-out PATH] [--ui-every N]   (display feed, see tools/ui_server.py)
//                 [--burst --model DIR] [--exclude-hz HZ]  (whole-signal demo model, one receiver)
//                 [--noise-shape F] [--calibrate-noise F]  (front-end noise shape: use / measure with no signal)
//                 [--dc-guard N]  (channels at DC never reported; omnicore default 1)
//                 [--neighbor-db 6] [--band-mhz B]  (detection: leakage rejection; only report +- B/2 around the
//                                                    centre - default the whole band, 52 MHz for omnicore)
//   omnicore:     [--rx 1|2|1,2] [--serial S] [--freq HZ] [--gain DB] [--agc 0..3] [--iq-bits 12|8] [--bandwidth HZ]
//                 [--rx-correction on|off]  (DC and I/Q tracking; default: preserve the hardware state)
//                 [--tx-file SC16 --tx-atten DB]  (loopback: send the file on TX1 in a loop; DB >= 40)
//                 (--streams / --rate as for the other sources; drop mode is implied: hardware cannot wait)
//   file:         [--freq HZ]  (recording centre frequency for display; defaults to baseband)
//
// Synthetic source: noise plus, per stream, N tones at the centres of N channels spread over the band
// (random phases), so exactly N channels per stream carry energy - the load knob. Prints one JSON line of
// pipeline statistics; --print N also lists the results of the first N windows.

#include "curf/engine/model.hpp"
#include "curf/runtime/classifier.hpp"
#include "curf/sources.hpp"
#include "curf/omnicore_source.hpp"
#include "burst_demo.hpp"
#include "dsss_probe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace curf;

int main(int argc, char **argv) {
    std::string source = "synthetic", file, label, model_dir = CURF_MODEL_DIR;
    bool loop = false, realtime = false;
    StreamFormat fmt{WireFormat::packed12, 2, 61.44e6};
    std::uint32_t active = 16;
    double tone_amp = -1, noise = 0.003;
    std::uint64_t print = 0;
    std::string dump_path; // CSV of every result: window,stream,channel,class,prob,power
    std::string ui_path;   // JSON lines for the display: per row max channel power (dB) + latest results
    std::uint64_t ui_every = 8; // 8 windows = 34 ms: ~29 rows/s
    // OmniCore source options (unused when built without it)
    [[maybe_unused]] std::string serial;
    [[maybe_unused]] double freq = 2.4e9, gain = 30, bandwidth = 0;
    bool freq_given = false;
    [[maybe_unused]] std::uint32_t agc = 0, iq_bits = 12;
    double band_mhz = -1; // usable band for detection; default: all (omnicore: 52 MHz, see below)
    int dc_guard = -1;    // channels at DC never reported; default 0 (omnicore: 1, the LO leakage channel)
    std::string calibrate_path, shape_path; // noise shape: write (no signal at the input) / use
    [[maybe_unused]] std::uint32_t rx_mask = 0; // omnicore --rx 1 | 2 | 1,2 (default: from --streams)
    std::optional<bool> rx_correction;
    [[maybe_unused]] std::string tx_file;
    [[maybe_unused]] double tx_atten = 89.75;
    bool int8 = false, burst = false, threshold_given = false, buckets_given = false, ring_given = false;
    std::vector<double> excluded;
    bool fp32_input = false;     // INT8 engine, calibrated on the model directory's golden windows
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
        } else if (a == "--threshold-db") {
            cfg.detect.threshold_db = std::stof(next());
            threshold_given = true;
        }
        else if (a == "--neighbor-db") cfg.detect.neighbor_reject_db = std::stof(next());
        else if (a == "--band-mhz") band_mhz = std::stod(next());
        else if (a == "--dc-guard") dc_guard = std::stoi(next());
        else if (a == "--abs-power") cfg.detect.absolute_power = std::stof(next());
        else if (a == "--compaction") cfg.detect.compaction = next() == "cub" ? cuda::Compaction::cub : cuda::Compaction::ballot;
        else if (a == "--seconds") cfg.seconds = std::stod(next());
        else if (a == "--windows") cfg.max_windows = std::stoull(next());
        else if (a == "--buckets") {
            buckets_given = true;
            cfg.engine.buckets.clear();
            std::stringstream ss(next());
            for (std::string x; std::getline(ss, x, ',');) cfg.engine.buckets.push_back(std::stoi(x));
        } else if (a == "--drop") cfg.drop_when_full = true;
        else if (a == "--ring") { cfg.ring_slots = std::stoull(next()); ring_given = true; }
        else if (a == "--dump") dump_path = next();
        else if (a == "--ui-out") ui_path = next();
        else if (a == "--ui-every") ui_every = std::max<std::uint64_t>(1, std::stoull(next()));
        else if (a == "--serial") serial = next();
        else if (a == "--freq") { freq = std::stod(next()); freq_given = true; }
        else if (a == "--gain") gain = std::stod(next());
        else if (a == "--bandwidth") bandwidth = std::stod(next());
        else if (a == "--agc") agc = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--rx-correction") {
            const auto value = next();
            if (value != "on" && value != "off") {
                std::fputs("--rx-correction expects on or off\n", stderr);
                return 2;
            }
            rx_correction = value == "on";
        }
        else if (a == "--iq-bits") iq_bits = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--tx-file") tx_file = next();
        else if (a == "--tx-atten") tx_atten = std::stod(next());
        else if (a == "--calibrate-noise") calibrate_path = next();
        else if (a == "--rx") { const auto v = next(); rx_mask = v == "2" ? 2u : v == "1,2" || v == "12" ? 3u : 1u; }
        else if (a == "--noise-shape") shape_path = next();
        else if (a == "--model") model_dir = next();
        else if (a == "--print") print = std::stoull(next());
        else if (a == "--label") label = next();
        else if (a == "--int8") int8 = true;
        else if (a == "--burst") burst = true;
        else if (a == "--exclude-hz") excluded.push_back(std::stod(next()));
        else if (a == "--fp32-input") fp32_input = true;
        else if (a == "--calib-percentile") cfg.engine.calib_percentile = std::stod(next());
        else if (a == "--tune-cache") cfg.engine.tune_cache = next();
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }

    if (freq_given && (!std::isfinite(freq) || freq < 0)) {
        std::fprintf(stderr, "--freq requires a finite, nonnegative frequency in Hz\n");
        return 2;
    }
    if (rx_correction && source != "omnicore") {
        std::fputs("--rx-correction requires --source omnicore\n", stderr);
        return 2;
    }
    const auto model = engine::Model::load(std::filesystem::path(model_dir) / "amc.curf");
    if (int8) {
        cfg.engine.int8 = true;
        cfg.engine.half_input = !fp32_input;
        cfg.engine.calibration = engine::Golden::load(std::filesystem::path(model_dir) / "amc.golden", model).input;
    }
    std::unique_ptr<ISource> src;
    const RxCorrection *correction = nullptr;
#ifdef CURF_HAVE_OMNICORE
    OmniCoreSource *och = nullptr;
#endif
    if (source == "file") {
        src = std::make_unique<FileSource>(FileConfig{file, fmt, loop});
    } else if (source == "omnicore") {
#ifdef CURF_HAVE_OMNICORE
        OmniCoreConfig oc;
        oc.serial = serial;
        oc.sample_rate = fmt.sample_rate;
        oc.channels = fmt.channels;
        oc.rx_mask = rx_mask;
        oc.frequency_hz = static_cast<std::uint64_t>(std::llround(freq));
        oc.bandwidth_hz = static_cast<std::uint32_t>(std::llround(bandwidth));
        oc.gain_db[0] = oc.gain_db[1] = gain;
        oc.agc = agc;
        oc.rx_correction = rx_correction;
        oc.iq_bits = iq_bits;
        oc.tx_file = tx_file;
        oc.tx_attenuation_db = tx_atten;
        auto o = std::make_unique<OmniCoreSource>(oc);
        och = o.get();
        if (och->rx_correction()) correction = &*och->rx_correction();
        fmt = o->format();
        std::fprintf(stderr, "omnicore: %s, %.6g MSPS x %u, %.6g MHz%s\n", o->device_name().c_str(), fmt.sample_rate / 1e6,
                     fmt.channels, freq / 1e6,
                     tx_file.empty() ? "" : (", TX1 loopback at " + std::to_string(tx_atten) + " dB attenuation").c_str());
        src = std::move(o);
        cfg.drop_when_full = true;
        // AD9361: 56 MHz analog filter and ADC noise shaping towards +- fs/2 (the loopback's edge channels read
        // up to 13 dB over the centre floor): detect within +- 26 MHz unless told otherwise
        if (band_mhz < 0) band_mhz = std::min(52e6, fmt.sample_rate) / 1e6;
        if (dc_guard < 0) dc_guard = 1;
#else
        std::fprintf(stderr, "built without OmniCore (configure with -DCURF_OMNICORE=ON)\n");
        return 2;
#endif
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

    if (burst) {
        try {
            BurstConfig bc;
            bc.engine = cfg.engine;
            if (!buckets_given) bc.engine.buckets = {1, 8, 32, 64};
            bc.seconds = cfg.seconds;
            bc.max_blocks = cfg.max_windows;
            bc.tune_hz = freq_given || source == "omnicore" ? freq : 0;
            bc.band_hz = std::min(fmt.sample_rate, band_mhz > 0 ? band_mhz * 1e6 : 50e6);
            bc.threshold_db = threshold_given ? cfg.detect.threshold_db : 12.f;
            bc.dc_guard_hz = static_cast<float>(std::max(0, dc_guard) * fmt.sample_rate / 512);
            bc.exclude_hz = excluded;
            bc.drop_when_full = cfg.drop_when_full;
            if (ring_given) bc.ring_slots = static_cast<unsigned>(cfg.ring_slots);
            const auto open = [](const std::string &path) {
                const auto close = [](std::FILE *f) { std::fclose(f); };
                std::unique_ptr<std::FILE, decltype(close)> f(path.empty() ? nullptr : std::fopen(path.c_str(), "w"), close);
                if (!path.empty() && !f) throw std::runtime_error("cannot open " + path);
                return f;
            };
            auto feed = open(ui_path), trace = open(dump_path);
            auto js = run_burst_demo(*src, model, bc, feed.get(), trace.get(), source, rx_mask == 2 ? 2 : 1,
                                     static_cast<unsigned>(ui_every), correction);
#ifdef CURF_HAVE_OMNICORE
            if (och) {
                const auto o = och->stats();
                char info[400];
                std::snprintf(info, sizeof info, "\"omnicore\":{\"samples\":%llu,\"host_dropped\":%llu,"
                    "\"device_overflows\":%llu,\"transfer_errors\":%llu,\"resyncs\":%llu,\"discontinuities\":%llu,\"tx_samples\":%llu},",
                    static_cast<unsigned long long>(o.samples), static_cast<unsigned long long>(o.host_dropped),
                    static_cast<unsigned long long>(o.device_overflows), static_cast<unsigned long long>(o.transfer_errors),
                    static_cast<unsigned long long>(o.resyncs), static_cast<unsigned long long>(o.discontinuities),
                    static_cast<unsigned long long>(o.tx_samples));
                js.insert(1, info);
            }
#endif
            std::puts(js.c_str());
            return 0;
        } catch (const std::exception &e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 2;
        }
    }
    if (!excluded.empty()) { std::fputs("--exclude-hz requires --burst\n", stderr); return 2; }
    if (band_mhz > 0) cfg.detect.band = static_cast<float>(std::min(1.0, band_mhz * 1e6 / fmt.sample_rate));
    if (dc_guard > 0) cfg.detect.dc_guard = static_cast<std::uint32_t>(dc_guard);
    if (!shape_path.empty()) {
        std::FILE *f = std::fopen(shape_path.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", shape_path.c_str()); return 2; }
        cfg.noise_shape.resize(std::size_t{fmt.channels} * cfg.channelizer.channels);
        const std::size_t got = std::fread(cfg.noise_shape.data(), sizeof(float), cfg.noise_shape.size(), f);
        std::fclose(f);
        if (got == cfg.channelizer.channels) cfg.noise_shape.resize(got); // one shape for every stream
        else if (got != cfg.noise_shape.size()) { std::fprintf(stderr, "%s: %zu values, need [streams][channels]\n", shape_path.c_str(), got); return 2; }
    }
    std::FILE *dump = dump_path.empty() ? nullptr : std::fopen(dump_path.c_str(), "w");
    if (dump) std::fprintf(dump, "window,stream,channel,cls,prob,power\n");

    // Display feed: one JSON line per ui_every windows - the max power of each channel over those windows
    // (dBFS, rounded) and, per channel detected in any of them, the probability-weighted vote over those
    // windows: [stream, channel, class, share of the vote %, windows detected]. A single window is right ~91 %
    // of the time on the cable loopback, a vote over 8 windows (34 ms) ~96 %. Channel order as produced.
    std::FILE *ui = ui_path.empty() ? nullptr : std::fopen(ui_path.c_str(), "w");
    const std::uint32_t S = fmt.channels, M = cfg.channelizer.channels;
    const double ui_frequency = source == "omnicore" || (source == "file" && freq_given) ? freq : 0.0;
    // The 5 GHz demo can confirm OFDM independently of the narrow-channel model.
    const bool probe_ofdm = ui && 5150e6 <= ui_frequency && ui_frequency < 5850e6 &&
                            40e6 <= fmt.sample_rate && fmt.sample_rate <= 65e6;
    std::unique_ptr<OfdmProbe> ofdm_probe;
    std::vector<OfdmEvidence> probe_windows(2 * S), ui_ofdm(S);
    std::vector<unsigned> probe_counts(S);
    std::vector<std::optional<double>> probe_centres(S);
    if (probe_ofdm) {
        ofdm_probe = std::make_unique<OfdmProbe>(fmt.sample_rate);
        cfg.on_input_block = [&](std::uint64_t window, bool first, std::span<const std::uint8_t> wire) {
            if (!first) return;
            for (std::uint32_t s = 0; s < S; ++s) {
                probe_windows[(window % 2) * S + s] = probe_centres[s]
                    ? ofdm_probe->observe(wire, fmt, s, *probe_centres[s]) : OfdmEvidence{};
            }
        };
    }
    const bool probe_dsss = ui && 2400e6 <= ui_frequency && ui_frequency < 2483.5e6 &&
                            40e6 <= fmt.sample_rate && fmt.sample_rate <= 65e6;
    std::unique_ptr<DsssProbe> dsss_probe;
    std::vector<std::vector<unsigned>> dsss_channels(S);
    std::vector<std::array<std::uint64_t, kBarkerChannels>> dsss_probe_until(S);
    std::vector<unsigned> dsss_windows(2 * S), ui_dsss(S);
    if (probe_dsss) {
        dsss_probe = std::make_unique<DsssProbe>(fmt.sample_rate);
        cfg.on_input_block = [&](std::uint64_t window, bool first, std::span<const std::uint8_t> wire) {
            const auto slice_bytes = DsssProbe::input_frames * fmt.frame_bytes();
            const auto slices = wire.size() / slice_bytes;
            for (std::uint32_t s = 0; s < S; ++s) {
                auto &evidence = dsss_windows[(window % 2) * S + s];
                if (first) evidence = 0;
                const auto &channels = dsss_channels[s];
                if (channels.empty() || !slices) continue;
                // Sample different parts of both blocks so periodic beacons do
                // not always land between observations.
                for (std::size_t i = 0; i < std::min<std::size_t>(2, slices); ++i) {
                    const auto channel = channels[(window + i + !first) % channels.size()];
                    if (evidence & (1u << channel)) continue;
                    const auto slice = (window + window / 17 + s + i * (slices / 2)) % slices;
                    if (dsss_probe->observes_barker(wire.subspan(slice * slice_bytes), fmt, s,
                                                    barker_frequency(channel) - ui_frequency))
                        evidence |= 1u << channel;
                }
            }
        };
    }
    std::vector<float> ui_max(std::size_t{S} * M, 0.f), ui_floor(S, 0.f);
    const auto ncls = static_cast<std::size_t>(model.tensors[model.output].channels);
    std::vector<float> ui_vote(std::size_t{S} * M * ncls, 0.f); // [S][M][classes] summed probabilities
    std::vector<std::uint8_t> ui_hits(std::size_t{S} * M, 0);
    if (ui) {
        // physical receiver of each stream, for the panel titles (omnicore --rx 2: the only stream is RX2)
        const std::string rx = rx_mask == 2 ? "[2]" : S == 2 ? "[1,2]" : "[1]";
        std::fprintf(ui, "{\"meta\":{\"streams\":%u,\"channels\":%u,\"rate\":%.17g,\"freq\":%.17g,\"window\":%u,\"every\":%llu,\"source\":\"%s\",\"rx\":%s",
                     S, M, fmt.sample_rate, ui_frequency, cfg.window,
                     static_cast<unsigned long long>(ui_every), source.c_str(), rx.c_str());
        write_rx_correction(ui, correction);
        std::fputs("}}\n", ui);
    }
    // Noise-shape calibration: per channel, the median power over (up to the first 4096) windows, divided by
    // the median over the stream's in-band channels. Run with no signal at the input.
    std::vector<float> cal; // [kept windows][S][M]
    if (ui || !calibrate_path.empty())
        cfg.on_power = [&](std::uint64_t w, const float *p, const float *fl) {
            if (ui) {
                if (w % ui_every == 0) std::fill(ui_max.begin(), ui_max.end(), 0.f);
                for (std::size_t i = 0; i < ui_max.size(); ++i) ui_max[i] = std::max(ui_max[i], p[i]);
                for (std::uint32_t s = 0; s < S; ++s) ui_floor[s] = fl[s];
            }
            if (!calibrate_path.empty() && w >= 4 && cal.size() < std::size_t{4096} * S * M) // first ~17 s
                cal.insert(cal.end(), p, p + std::size_t{S} * M);
        };
    std::string line;
    const auto st = run_classifier(*src, model, cfg, [&](std::uint64_t w, std::uint64_t first, const WindowResult *r, std::size_t n) {
        if (ui) { // per channel: vote of this row's windows
            if (w % ui_every == 0) {
                std::fill(ui_vote.begin(), ui_vote.end(), 0.f);
                std::fill(ui_hits.begin(), ui_hits.end(), std::uint8_t{0});
                std::fill(ui_ofdm.begin(), ui_ofdm.end(), OfdmEvidence{});
                std::fill(probe_counts.begin(), probe_counts.end(), 0u);
                std::fill(ui_dsss.begin(), ui_dsss.end(), 0u);
            }
            if (probe_dsss)
                for (std::uint32_t s = 0; s < S; ++s) ui_dsss[s] |= dsss_windows[(w % 2) * S + s];
            if (probe_ofdm) {
                for (std::uint32_t s = 0; s < S; ++s) {
                    const auto &result = probe_windows[(w % 2) * S + s];
                    if (result.contrast == 0) continue;
                    auto &best = ui_ofdm[s];
                    if (best.contrast > 0 && std::abs(result.offset_hz - best.offset_hz) <= 3e6) {
                        ++probe_counts[s];
                        if (result.contrast > best.contrast) best = result;
                    } else if (result.contrast > best.contrast && probe_counts[s] < 2) {
                        best = result;
                        probe_counts[s] = 1;
                    }
                }
            }
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t e = std::size_t{r[i].stream} * M + r[i].channel;
                ui_vote[e * ncls + r[i].cls] += r[i].prob;
                ui_hits[e] = static_cast<std::uint8_t>(std::min(255, ui_hits[e] + 1));
            }
        }
        if (ui && w % ui_every == ui_every - 1) {
            line.clear();
            char b[64];
            std::snprintf(b, sizeof b, "{\"w\":%llu,\"t\":%.6f,\"p\":[", static_cast<unsigned long long>(w),
                          static_cast<double>(first) / fmt.sample_rate);
            line += b;
            for (std::size_t i = 0; i < ui_max.size(); ++i) {
                const int db = static_cast<int>(std::lround(10 * std::log10(std::max(ui_max[i], 1e-15f))));
                std::snprintf(b, sizeof b, i ? ",%d" : "%d", db);
                line += b;
            }
            line += "],\"f\":[";
            for (std::uint32_t s = 0; s < S; ++s) {
                std::snprintf(b, sizeof b, s ? ",%.1f" : "%.1f", 10 * std::log10(std::max(ui_floor[s], 1e-15f)));
                line += b;
            }
            line += "],\"r\":[";
            bool comma = false;
            for (std::size_t e = 0; e < ui_hits.size(); ++e) {
                if (!ui_hits[e]) continue;
                const float *v = &ui_vote[e * ncls];
                const auto best = static_cast<std::size_t>(std::max_element(v, v + ncls) - v);
                float total = 0.f;
                for (std::size_t c = 0; c < ncls; ++c) total += v[c];
                std::snprintf(b, sizeof b, comma ? ",[%zu,%zu,%zu,%d,%u]" : "[%zu,%zu,%zu,%d,%u]", e / M, e % M, best,
                              static_cast<int>(std::lround(100 * v[best] / std::max(total, 1e-9f))), unsigned{ui_hits[e]});
                line += b;
                comma = true;
            }
            line += "]";
            if (probe_ofdm) {
                line += ",\"ofdm_cp\":[";
                for (std::uint32_t s = 0; s < S; ++s) {
                    const auto offset = std::size_t{s} * M;
                    probe_centres[s] = ofdm_probe_centre(std::span{ui_hits}.subspan(offset, M),
                                                        std::span{ui_max}.subspan(offset, M), fmt.sample_rate);
                    if (s) line += ',';
                    if (probe_counts[s] < (ui_every > 1 ? 2u : 1u)) line += "null";
                    else {
                        std::snprintf(b, sizeof b, "%.0f", ui_ofdm[s].offset_hz);
                        line += b;
                    }
                }
                line += "]";
            }
            if (probe_dsss) {
                line += ",\"dsss_barker\":[";
                for (std::uint32_t s = 0; s < S; ++s) {
                    const auto offset = std::size_t{s} * M;
                    const auto detected = barker_probe_channels(std::span{ui_hits}.subspan(offset, M),
                                                                std::span{ui_max}.subspan(offset, M),
                                                                fmt.sample_rate, ui_frequency);
                    for (const auto ch : detected)
                        dsss_probe_until[s][ch] = first + static_cast<std::uint64_t>(0.30 * fmt.sample_rate);
                    dsss_channels[s].clear();
                    for (unsigned ch = 0; ch < kBarkerChannels; ++ch)
                        if (first < dsss_probe_until[s][ch]) dsss_channels[s].push_back(ch);
                    line += s ? ",[" : "[";
                    bool next = false;
                    for (unsigned ch = 0; ch < kBarkerChannels; ++ch) {
                        if (!(ui_dsss[s] & (1u << ch))) continue;
                        std::snprintf(b, sizeof b, next ? ",%.0f" : "%.0f", barker_frequency(ch));
                        line += b;
                        next = true;
                    }
                    line += "]";
                }
                line += "]";
            }
            line += "}\n";
            std::fputs(line.c_str(), ui);
            std::fflush(ui);
        }
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
    if (!calibrate_path.empty()) {
        const std::size_t n = cal.size() / (std::size_t{S} * M);
        if (n < 16) { std::fprintf(stderr, "calibration: only %zu windows\n", n); return 1; }
        const auto h = static_cast<std::uint32_t>(cfg.detect.band * static_cast<float>(M) / 2.f);
        std::vector<float> shape(std::size_t{S} * M), col(n);
        for (std::uint32_t s = 0; s < S; ++s) {
            std::vector<float> inband;
            for (std::uint32_t k = 0; k < M; ++k) {
                for (std::size_t t = 0; t < n; ++t) col[t] = cal[(t * S + s) * M + k];
                std::nth_element(col.begin(), col.begin() + static_cast<std::ptrdiff_t>(n / 2), col.end());
                shape[std::size_t{s} * M + k] = col[n / 2];
                if (std::min(k, M - k) <= h) inband.push_back(col[n / 2]);
            }
            std::nth_element(inband.begin(), inband.begin() + static_cast<std::ptrdiff_t>(inband.size() / 2), inband.end());
            const float ref = inband[inband.size() / 2];
            for (std::uint32_t k = 0; k < M; ++k) shape[std::size_t{s} * M + k] = std::max(shape[std::size_t{s} * M + k] / ref, 1e-3f);
        }
        std::FILE *f = std::fopen(calibrate_path.c_str(), "wb");
        if (!f || std::fwrite(shape.data(), sizeof(float), shape.size(), f) != shape.size()) {
            std::fprintf(stderr, "cannot write %s\n", calibrate_path.c_str());
            return 1;
        }
        std::fclose(f);
        std::fprintf(stderr, "noise shape [%u][%u] from %zu windows -> %s\n", S, M, n, calibrate_path.c_str());
    }
    std::string js = st.to_json();
#ifdef CURF_HAVE_OMNICORE
    if (och) {
        const auto o = och->stats();
        char b[600];
        std::snprintf(b, sizeof b,
                      "\"omnicore\":{\"usb_bytes\":%llu,\"samples\":%llu,\"host_dropped\":%llu,\"device_overflows\":%llu,"
                      "\"transfer_errors\":%llu,\"resyncs\":%llu,\"queue_peak_mib\":%.1f,\"discontinuities\":%llu,"
                      "\"tx_atten_db\":%.2f,\"tx_samples\":%llu,\"tx_underflows\":%llu,\"tx_starvation\":%llu},",
                      static_cast<unsigned long long>(o.usb_bytes), static_cast<unsigned long long>(o.samples),
                      static_cast<unsigned long long>(o.host_dropped), static_cast<unsigned long long>(o.device_overflows),
                      static_cast<unsigned long long>(o.transfer_errors), static_cast<unsigned long long>(o.resyncs),
                      static_cast<double>(o.host_queue_peak_bytes) / (1 << 20), static_cast<unsigned long long>(o.discontinuities),
                      tx_file.empty() ? 0.0 : tx_atten, static_cast<unsigned long long>(o.tx_samples),
                      static_cast<unsigned long long>(o.tx_underflows), static_cast<unsigned long long>(o.tx_starvation));
        js.insert(1, b);
    }
#endif
    if (ui) {
        std::fprintf(ui, "{\"end\":%s}\n", st.to_json().c_str());
        std::fclose(ui);
    }
    if (!label.empty()) js.insert(1, "\"label\":\"" + label + "\",");
    if (int8) js.insert(1, cfg.engine.half_input ? "\"int8\":true,\"fp16_input\":true," : "\"int8\":true,");
    std::printf("%s\n", js.c_str());
    return 0;
}
