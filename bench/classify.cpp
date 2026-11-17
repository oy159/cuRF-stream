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
//                 [--noise-shape F] [--calibrate-noise F]  (front-end noise shape: use / measure with no signal)
//                 [--dc-guard N]  (channels at DC never reported; omnicore default 1)
//                 [--stage2 DIR] [--stage2-int8] [--stage1-classify]  (signal-level stage 2: adjacent detected channels -> one signal,
//                                                    extracted from the raw samples at 2-3x its bandwidth and
//                                                    classified by DIR/s2.curf; {"sig":{...}} lines in the feed)
//                 [--adsb [HZ]] [--adsb-stream N] [--adsb-threshold T]  (Mode S / ADS-B demodulator on the raw
//                                                    samples of stream N, carrier HZ (default 1090e6); decoded
//                                                    messages go to the display feed as {"adsb":{...}} lines.
//                                                    Tune off the carrier, e.g. --freq 1080e6: DC carries LO leakage)
//                 [--neighbor-db 6] [--band-mhz B]  (detection: leakage rejection; only report +- B/2 around the
//                                                    centre - default the whole band, 52 MHz for omnicore)
//   omnicore:     [--rx 1|2|1,2] [--serial S] [--freq HZ] [--gain DB] [--agc 0..3] [--iq-bits 12|8] [--bandwidth HZ]
//                 [--tx-file SC16 --tx-atten DB]  (loopback: send the file on TX1 in a loop; DB >= 40)
//                 (--streams / --rate as for the other sources; drop mode is implied: hardware cannot wait)
//
// Synthetic source: noise plus, per stream, N tones at the centres of N channels spread over the band
// (random phases), so exactly N channels per stream carry energy - the load knob. Prints one JSON line of
// pipeline statistics; --print N also lists the results of the first N windows.

#include "curf/adsb.hpp"
#include "curf/channelizer.hpp"
#include "curf/cuda/adsb.hpp"
#include "curf/engine/model.hpp"
#include "curf/runtime/classifier.hpp"
#include "curf/runtime/signal_stage.hpp"
#include "curf/sources.hpp"
#ifdef CURF_HAVE_OMNICORE
#include "curf/omnicore_source.hpp"
#endif

#include <algorithm>
#include <deque>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace curf;

namespace {

    std::string json_quote(const std::string &text) {
        std::string out = "\"";
        for (unsigned char c : text) {
            if (c == '"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
            else if (c < 32) { char b[7]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; }
            else out += static_cast<char>(c);
        }
        return out + '"';
    }

    // ADS-B on the side of the classifier: GPU demodulation per block, then on the host duplicate removal (one
    // frame can be found by neighbouring start positions or both sides of a block boundary), address checks for
    // the address/parity formats, field decoding and CPR positions.
    struct AdsbTap final : BlockTap {
        cuda::AdsbDemod demod;
        adsb::Tracker tracker;
        double fs;
        double carrier_hz;
        std::uint32_t stream;
        std::FILE *ui;
        std::deque<std::pair<std::uint64_t, std::array<std::uint8_t, 14>>> recent;
        std::uint64_t frames = 0, extended = 0, corrected = 0, short_ok = 0, unknown_addr = 0, positions = 0, dups = 0,
                      overflow = 0;
        std::vector<cuda::AdsbFrame> buf;

        AdsbTap(const cuda::AdsbConfig &c, const StreamFormat &f, std::size_t block, std::size_t slots, std::FILE *ui_, double carrier)
            : demod(c, f.wire, f.channels, block, slots), fs(f.sample_rate), carrier_hz(carrier), stream(c.stream), ui(ui_) {}

        void enqueue(std::size_t slot, const std::uint8_t *d_wire, std::size_t n, std::uint64_t first, cudaStream_t s) override {
            demod.enqueue(d_wire, n, first, slot, s);
        }

        void retire(std::size_t slot) override {
            const auto r = demod.results(slot);
            if (demod.found(slot) > r.size()) overflow += demod.found(slot) - r.size();
            buf.assign(r.begin(), r.end());
            std::sort(buf.begin(), buf.end(), [](const auto &a, const auto &b) { return a.sample < b.sample; });
            const auto window = static_cast<std::uint64_t>(4e-6 * fs);
            for (const auto &f : buf) {
                std::array<std::uint8_t, 14> key{};
                std::copy(f.msg, f.msg + 14, key.begin());
                while (!recent.empty() && recent.front().first + window < f.sample) recent.pop_front();
                if (std::any_of(recent.begin(), recent.end(), [&](const auto &e) { return e.second == key; })) {
                    ++dups;
                    continue;
                }
                recent.emplace_back(f.sample, key);
                const double t = static_cast<double>(f.sample) / fs;
                const int df = adsb::downlink_format(f.msg);
                std::uint32_t addr = 0;
                if (df == 11 || df == 17 || df == 18) {
                    if (df != 11) ++extended;
                    else ++short_ok;
                } else {
                    if (!tracker.known(f.remainder, t)) {
                        ++unknown_addr;
                        continue;
                    }
                    addr = f.remainder;
                    ++short_ok;
                }
                corrected += f.corrected;
                ++frames;
                const auto m = adsb::decode(f.msg, f.bits, addr);
                const auto pos = tracker.position(m, t);
                if (pos) ++positions;
                if (ui) {
                    const float db = 20 * std::log10(std::max(f.signal, 1e-9f));
                    const auto record = adsb::to_json(m, t, db, pos, f.corrected != 0);
                    // Demodulation has verified CRC (after the optional one-bit correction). AP formats
                    // additionally need a known address; only DF17/18 become aircraft in the display policy.
                    std::fprintf(ui, "{\"adsb\":{\"crc_ok\":true,\"s\":%u,\"freq\":%.0f,%s}\n",
                                 stream, carrier_hz, record.c_str() + 1);
                }
            }
            if (ui && !buf.empty()) std::fflush(ui);
        }

        std::string json() const {
            char b[300];
            std::snprintf(b, sizeof b,
                          "\"adsb\":{\"frames\":%llu,\"extended\":%llu,\"short\":%llu,\"corrected\":%llu,\"positions\":%llu,"
                          "\"aircraft\":%zu,\"unknown_address\":%llu,\"duplicates\":%llu,\"overflow\":%llu},",
                          static_cast<unsigned long long>(frames), static_cast<unsigned long long>(extended),
                          static_cast<unsigned long long>(short_ok), static_cast<unsigned long long>(corrected),
                          static_cast<unsigned long long>(positions), tracker.size(),
                          static_cast<unsigned long long>(unknown_addr), static_cast<unsigned long long>(dups),
                          static_cast<unsigned long long>(overflow));
            return b;
        }
    };

} // namespace

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
    [[maybe_unused]] std::string tx_file;
    [[maybe_unused]] double tx_atten = 89.75;
    bool int8 = false;
    std::string stage2_dir;      // stage-2 model directory (s2.curf, s2.golden)
    bool stage2_int8 = false;
    std::string stage2_tune_cache;
    std::uint32_t stage2_max_signals = 64, stage2_max_tracks = 256;
    std::uint32_t stage2_pending = 3;
    std::size_t stage2_scratch_mib = 32;
    bool stage1_classify = false;
    double adsb_hz = 0;          // > 0: ADS-B demodulator on the side, carrier frequency
    cuda::AdsbConfig adsb_cfg;
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
        } else if (a == "--threshold-db") cfg.detect.threshold_db = std::stof(next());
        else if (a == "--neighbor-db") cfg.detect.neighbor_reject_db = std::stof(next());
        else if (a == "--band-mhz") band_mhz = std::stod(next());
        else if (a == "--dc-guard") dc_guard = std::stoi(next());
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
        else if (a == "--ui-out") ui_path = next();
        else if (a == "--ui-every") ui_every = std::max<std::uint64_t>(1, std::stoull(next()));
        else if (a == "--serial") serial = next();
        else if (a == "--freq") { freq = std::stod(next()); freq_given = true; }
        else if (a == "--gain") gain = std::stod(next());
        else if (a == "--bandwidth") bandwidth = std::stod(next());
        else if (a == "--agc") agc = static_cast<std::uint32_t>(std::stoul(next()));
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
        else if (a == "--adsb") adsb_hz = i + 1 < argc && argv[i + 1][0] != '-' ? std::stod(next()) : 1090e6;
        else if (a == "--stage2") stage2_dir = next();
        else if (a == "--stage2-int8") stage2_int8 = true;
        else if (a == "--stage2-tune-cache") stage2_tune_cache = next();
        else if (a == "--stage2-max-signals") stage2_max_signals = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--stage2-max-tracks") stage2_max_tracks = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--stage2-scratch-mib") stage2_scratch_mib = std::stoull(next());
        else if (a == "--stage2-pending") stage2_pending = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--stage1-classify") stage1_classify = true;
        else if (a == "--adsb-stream") adsb_cfg.stream = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--adsb-threshold") adsb_cfg.threshold = std::stof(next());
        else if (a == "--fp32-input") fp32_input = true;
        else if (a == "--calib-percentile") cfg.engine.calib_percentile = std::stod(next());
        else if (a == "--tune-cache") cfg.engine.tune_cache = next();
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }

    if (!stage2_max_signals || stage2_max_signals > 256 || stage2_max_tracks < stage2_max_signals ||
        stage2_max_tracks > 4096 || stage2_scratch_mib < 8 || stage2_scratch_mib > 512 || !stage2_pending || stage2_pending > 8) {
        std::fprintf(stderr, "invalid stage-2 capacities: signals 1..256, tracks signals..4096, scratch 8..512 MiB, pending 1..8\n");
        return 2;
    }

    engine::Model model;
    if (stage2_dir.empty() || stage1_classify) {
        model = engine::Model::load(std::filesystem::path(model_dir) / "amc.curf");
    } else {
        // Detection-only stage 1 needs no trained model. Keep its legacy callback shape as a descriptor.
        model.tensors = {{2,cfg.window,"input"}, {1,1,"detection"}};
        model.input = 0; model.output = 1;
    }
    if (int8 && (stage2_dir.empty() || stage1_classify)) {
        cfg.engine.int8 = true;
        cfg.engine.half_input = !fp32_input;
        cfg.engine.calibration = engine::Golden::load(std::filesystem::path(model_dir) / "amc.golden", model).input;
    }
    std::unique_ptr<ISource> src;
    std::string receiver_name;
#ifdef CURF_HAVE_OMNICORE
    OmniCoreSource *och = nullptr;
#endif
    if (source == "file") {
        src = std::make_unique<FileSource>(FileConfig{file, fmt, loop, realtime});
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
        oc.iq_bits = iq_bits;
        oc.tx_file = tx_file;
        oc.tx_attenuation_db = tx_atten;
        auto o = std::make_unique<OmniCoreSource>(oc);
        och = o.get();
        fmt = o->format();
        receiver_name = o->device_name();
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
    double channel_noise_gain = 0;
    for (float h : design_prototype(cfg.channelizer)) channel_noise_gain += static_cast<double>(h) * h;
    std::vector<float> ui_max(std::size_t{S} * M, 0.f), ui_floor(S, 0.f);
    const auto ncls = static_cast<std::size_t>(model.tensors[model.output].channels);
    std::vector<float> ui_vote(std::size_t{S} * M * ncls, 0.f); // [S][M][classes] summed probabilities
    std::vector<std::uint8_t> ui_hits(std::size_t{S} * M, 0);
    if (ui) {
        // physical receiver of each stream, for the panel titles (omnicore --rx 2: the only stream is RX2)
        const std::string rx = rx_mask == 2 ? "[2]" : S == 2 ? "[1,2]" : "[1]";
        std::fprintf(ui, "{\"meta\":{\"version\":2,\"streams\":%u,\"channels\":%u,\"rate\":%.17g,\"freq\":%.17g,\"window\":%u,\"every\":%llu,\"source\":\"%s\",\"rx\":%s,\"stage2\":%s,\"channel_classes\":%s",
                     S, M, fmt.sample_rate, source == "omnicore" || freq_given ? freq : 0.0, cfg.window,
                     static_cast<unsigned long long>(ui_every), source.c_str(), rx.c_str(), stage2_dir.empty() ? "false" : "true",
                     stage2_dir.empty() || stage1_classify ? "true" : "false");
        if (!stage2_dir.empty())
            std::fprintf(ui, ",\"signal_classes\":[\"noise\",\"ofdm\",\"fsk\",\"4fsk\",\"psk\",\"qam\",\"dsss\",\"chirp\",\"ook\",\"am\",\"fm\",\"pulse\",\"unknown\"]");
        if (source == "omnicore")
            std::fprintf(ui, ",\"receiver\":{\"device\":%s,\"gain\":%.6g,\"agc\":%u,\"bandwidth\":%.6g,\"iq_bits\":%u,\"noise_shape\":%s}",
                         json_quote(receiver_name).c_str(), gain, agc, bandwidth > 0 ? bandwidth : std::min(fmt.sample_rate, 56e6),
                         iq_bits, json_quote(shape_path).c_str());
        if (adsb_hz > 0) std::fprintf(ui, ",\"adsb\":{\"stream\":%u,\"freq\":%.0f}", adsb_cfg.stream, adsb_hz);
        std::fprintf(ui, ",\"channel_noise_gain\":%.12g}}\n", channel_noise_gain);
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
    std::unique_ptr<AdsbTap> adsb_tap;
    if (adsb_hz > 0) {
        adsb_cfg.sample_rate = fmt.sample_rate;
        // ~15 MS/s after decimation (61.44 MS/s: 4); lower-rate recordings are used as they are
        adsb_cfg.decimation = std::max<std::uint32_t>(1, static_cast<std::uint32_t>(std::lround(fmt.sample_rate / 15.36e6)));
        adsb_cfg.offset_hz = adsb_hz - freq; // file / synthetic sources: --freq is the recording's centre
        if (std::fabs(adsb_cfg.offset_hz) > fmt.sample_rate / 2 - 2e6) {
            std::fprintf(stderr, "--adsb: %.6g MHz is outside the tuned band\n", adsb_hz / 1e6);
            return 2;
        }
        adsb_tap = std::make_unique<AdsbTap>(adsb_cfg, fmt, cfg.block_frames, cfg.depth, ui, adsb_hz);
        cfg.taps.push_back(adsb_tap.get());
        std::fprintf(stderr, "adsb: stream %u, carrier at %+.3f MHz from the centre\n", adsb_cfg.stream, adsb_cfg.offset_hz / 1e6);
    }
    // stage 2
    std::unique_ptr<engine::Model> s2_model;
    std::unique_ptr<SignalStage> stage2;
    if (!stage2_dir.empty()) {
        s2_model = std::make_unique<engine::Model>(engine::Model::load(std::filesystem::path(stage2_dir) / "s2.curf"));
        SignalStageConfig sc;
        sc.channels = cfg.channelizer.channels;
        sc.window_frames = cfg.window;
        sc.max_signals = stage2_max_signals;
        sc.max_tracks = stage2_max_tracks;
        sc.pending = stage2_pending;
        sc.scratch_complex = stage2_scratch_mib * (std::size_t{1} << 20) / sizeof(cf32);
        sc.drop_when_busy = cfg.drop_when_full;
        sc.engine.buckets = {1, 4, 16, 64};
        for (unsigned b = 128; sc.engine.buckets.back() < static_cast<int>(sc.max_signals); b *= 2)
            sc.engine.buckets.push_back(static_cast<int>(b));
        sc.engine.tune_iters = 10;
        sc.engine.tune_cache = stage2_tune_cache;
        if (stage2_int8) {
            sc.engine.int8 = true;
            sc.engine.calibration = engine::Golden::load(std::filesystem::path(stage2_dir) / "s2.golden", *s2_model).input;
        }
        cfg.classify_channels = stage1_classify;
        cfg.detect.peak_detection = true;
        stage2 = std::make_unique<SignalStage>(sc, *s2_model, fmt, cfg.block_frames, [ui, &ui_floor, &cfg, fmt, M, channel_noise_gain](const SignalResult &r) {
            if (!ui) return;
            double shape = 1;
            if (!cfg.noise_shape.empty()) {
                double sum = 0; unsigned count = 0;
                const std::size_t offset = cfg.noise_shape.size() == M ? 0 : std::size_t{r.stream} * M;
                for (unsigned k = 0; k < M; ++k) {
                    const double f = static_cast<double>(k) * fmt.sample_rate / M;
                    if (std::abs(std::remainder(f-r.fc_hz, fmt.sample_rate)) <= .375*r.fs_hz) {
                        sum += cfg.noise_shape[offset+k]; ++count;
                    }
                }
                if (count) shape = sum/static_cast<double>(count);
            }
            // Unit-gain PFB noise variance = raw variance * sum(h^2). The finite extraction FIR has
            // ENBW slightly below 0.75 * output_rate; using that upper bound is conservative.
            const double window_noise = ui_floor[r.stream] * shape / channel_noise_gain * (.75*r.fs_hz/fmt.sample_rate);
            const double snr_db = 10*std::log10(std::max(std::pow(10., r.power_db/10.)/std::max(window_noise,1e-30)-1.,1e-9));
            std::vector<std::size_t> idx(r.probs.size());
            for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + std::min<std::size_t>(3, idx.size()), idx.end(),
                              [&](std::size_t a, std::size_t b) { return r.probs[a] > r.probs[b]; });
            std::fprintf(ui, "{\"sig\":{\"w\":%llu,\"s\":%u,\"id\":%u,\"f\":%.0f,\"bw\":%.0f,\"fs\":%.0f,\"t0\":%.9f,\"t1\":%.9f,\"processing_ms\":%.4f,\"observation_ms\":%.4f,\"c\":%u,\"p\":%.6f,\"db\":%.1f,\"top\":[[%zu,%.6f],[%zu,%.6f],[%zu,%.6f]],\"t\":%.9f,\"activity_fraction\":%.6f,\"probs\":[",
                         static_cast<unsigned long long>(r.window), r.stream, r.track, r.fc_hz, r.bw_hz, r.fs_hz, r.t0_s, r.t1_s, r.processing_ms, r.observation_ms, r.cls,
                         static_cast<double>(r.prob), static_cast<double>(r.power_db), idx[0], static_cast<double>(r.probs[idx[0]]),
                         idx[1], static_cast<double>(r.probs[idx[1]]), idx[2], static_cast<double>(r.probs[idx[2]]),
                         r.source_time_s, static_cast<double>(r.activity_fraction));
            for (std::size_t i = 0; i < r.probs.size(); ++i) std::fprintf(ui, "%s%.7g", i ? "," : "", static_cast<double>(r.probs[i]));
            std::fprintf(ui, "],\"snr_db\":%.3f}}\n", snr_db);
        });
        cfg.taps.push_back(stage2.get());
        std::fprintf(stderr, "stage 2: %s%s\n", stage2_dir.c_str(), stage2_int8 ? " (INT8)" : "");
    }
    std::string line;
    const auto st = run_classifier(*src, model, cfg, [&](std::uint64_t w, std::uint64_t first, const WindowResult *r, std::size_t n, auto ready_at) {
        if (stage2) stage2->on_window(w, first, r, n, ready_at);
        if (ui) { // per channel: vote of this row's windows
            if (w % ui_every == 0) {
                std::fill(ui_vote.begin(), ui_vote.end(), 0.f);
                std::fill(ui_hits.begin(), ui_hits.end(), std::uint8_t{0});
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
            line += "]}\n";
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
    if (adsb_tap) js.insert(1, adsb_tap->json());
    if (stage2) {
        stage2->flush();
        js.insert(1, "\"stage2\":" + stage2->stats().to_json() + ",");
    }
    if (ui) {
        std::fprintf(ui, "{\"end\":%s}\n", js.c_str());
        std::fclose(ui);
    }
    if (!label.empty()) js.insert(1, "\"label\":\"" + label + "\",");
    if (int8) js.insert(1, cfg.engine.half_input ? "\"int8\":true,\"fp16_input\":true," : "\"int8\":true,");
    std::printf("%s\n", js.c_str());
    return 0;
}
