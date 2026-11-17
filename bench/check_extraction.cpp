// Known signal parameters -> wire -> GPU unpack / overlap-save -> raw IQ -> continuous CPU FIR.
// No detector, tracker, model, normalisation or confidence rules participate in this check.
#include "curf/cuda/extraction.hpp"
#include "curf/cuda/memory.hpp"
#include "curf/cuda/unpack.hpp"
#include "curf/reference.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace curf;

namespace {
    template <typename T> void dump(const std::filesystem::path &p, const std::vector<T> &v) {
        std::ofstream f(p, std::ios::binary);
        f.write(reinterpret_cast<const char *>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
        if (!f) throw std::runtime_error("cannot write " + p.string());
    }
}

int main(int argc, char **argv) try {
    ExtractionConfig cfg;
    cfg.centre_hz = -3e6;
    std::size_t blocks = 8;
    std::uint64_t first = 123456789;
    std::uint32_t streams = 2, selected = 1;
    WireFormat format = WireFormat::packed12;
    std::string signal = "burst";
    std::filesystem::path dir;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() { if (++i >= argc) throw std::invalid_argument("missing " + arg); return std::string(argv[i]); };
        if (arg == "--out") dir = value();
        else if (arg == "--decimation") cfg.decimation = static_cast<std::uint32_t>(std::stoul(value()));
        else if (arg == "--fc") cfg.centre_hz = std::stod(value());
        else if (arg == "--rate") cfg.sample_rate = std::stod(value());
        else if (arg == "--block") cfg.block_frames = std::stoull(value());
        else if (arg == "--blocks") blocks = std::stoull(value());
        else if (arg == "--first") first = std::stoull(value());
        else if (arg == "--streams") streams = static_cast<std::uint32_t>(std::stoul(value()));
        else if (arg == "--stream") selected = static_cast<std::uint32_t>(std::stoul(value()));
        else if (arg == "--signal") signal = value();
        else if (arg == "--wire") {
            const auto v = value();
            if (v == "packed12") format = WireFormat::packed12;
            else if (v == "signed8") format = WireFormat::signed8;
            else throw std::invalid_argument("wire must be packed12 or signed8");
        } else throw std::invalid_argument("unknown option: " + arg);
    }
    cfg.validate();
    if (dir.empty() || blocks < 3 || blocks > 64 || streams == 0 || streams > 4 || selected >= streams ||
        (signal != "tone" && signal != "burst"))
        throw std::invalid_argument("use --out DIR [--signal tone|burst] [--blocks 3..64] [--streams 1..4 --stream N]");
    const std::size_t count = blocks * cfg.block_frames, step = cfg.output_frames();
    const auto bytes = wire_sample_bytes(format);
    std::vector<std::uint8_t> wire(count * streams * bytes);
    std::vector<cf32> input(count), reference(blocks * step), got(reference.size());
    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0, 0.003f);
    for (std::size_t n = 0; n < count; ++n) {
        const double t = double(first + n) / cfg.sample_rate;
        const double ph = 2 * std::numbers::pi * (cfg.centre_hz + 0.113 * cfg.output_rate()) * t;
        const bool active = signal == "tone" || (n >= cfg.block_frames - cfg.block_frames / 8 &&
                                                  n < 2 * cfg.block_frames + cfg.block_frames / 5);
        const double a = active ? 0.25 : 0;
        // Equal-amplitude out-of-band interferer (well inside raw Nyquist for the default D >= 3).
        const double fi = cfg.centre_hz + 0.8 * cfg.output_rate();
        const double phi = 2 * std::numbers::pi * fi * t;
        const double ai = signal == "burst" && std::abs(fi) < cfg.sample_rate / 2 ? 0.25 : 0;
        for (unsigned s = 0; s < streams; ++s) {
            cf32 v;
            if (s == selected) {
                const float nr = signal == "burst" ? noise(rng) : 0.f;
                const float ni = signal == "burst" ? noise(rng) : 0.f;
                v = {static_cast<float>(a * std::cos(ph) + ai * std::cos(phi)) + nr,
                     static_cast<float>(a * std::sin(ph) + ai * std::sin(phi)) + ni};
            } else {
                const double other = 2 * std::numbers::pi * 0.217 * double(n);
                v = {float(0.5 * std::cos(other)), float(0.5 * std::sin(other))};
            }
            auto *p = wire.data() + (n * streams + s) * bytes;
            ref::pack_sample(v, format, p);
            if (s == selected) input[n] = ref::unpack_sample(p, format);
        }
    }
    ref::extract_signal(cfg, input, first, reference);
    cuda::SignalExtractor ex(cfg);
    cuda::DeviceBuffer<std::uint8_t> dw(cfg.block_frames * streams * bytes);
    cuda::DeviceBuffer<cf32> planar(cfg.block_frames * streams), output(step);
    cuda::PinnedBuffer<std::uint8_t> hw(wire.size());
    cuda::PinnedBuffer<cf32> hy(got.size());
    std::copy(wire.begin(), wire.end(), hw.get());
    cudaStream_t stream;
    CURF_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (std::size_t b = 0; b < blocks; ++b) {
        CURF_CUDA_CHECK(cudaMemcpyAsync(dw.get(), hw.get() + b * dw.size(), dw.bytes(), cudaMemcpyHostToDevice, stream));
        cuda::unpack_planar(dw.get(), format, streams, cfg.block_frames, planar.get(), stream);
        ex.run(planar.get() + selected * cfg.block_frames, output.get(), first + b * cfg.block_frames, stream);
        CURF_CUDA_CHECK(cudaMemcpyAsync(hy.get() + b * step, output.get(), output.bytes(), cudaMemcpyDeviceToHost, stream));
    }
    CURF_CUDA_CHECK(cudaStreamSynchronize(stream));
    CURF_CUDA_CHECK(cudaStreamDestroy(stream));
    std::copy_n(hy.get(), got.size(), got.data());
    double max_error = 0, sum = 0, seam = 0;
    for (std::size_t j = 0; j < got.size(); ++j) {
        const double e = std::hypot(got[j].re - reference[j].re, got[j].im - reference[j].im);
        max_error = std::max(max_error, e);
        sum += e * e;
        if (j % step < 4 || j % step + 4 >= step) seam = std::max(seam, e);
    }
    std::filesystem::create_directories(dir);
    dump(dir / "input.cf32", input);
    dump(dir / "cuda.cf32", got);
    dump(dir / "reference.cf32", reference);
    std::ostringstream json;
    json << std::setprecision(17)
         << "{\"sample_rate\":" << cfg.sample_rate << ",\"centre_hz\":" << cfg.centre_hz
         << ",\"decimation\":" << cfg.decimation << ",\"output_rate\":" << cfg.output_rate()
         << ",\"block_frames\":" << cfg.block_frames << ",\"blocks\":" << blocks << ",\"first_sample\":" << first
         << ",\"output_frames_per_block\":" << step << ",\"delay_input_frames\":" << cfg.delay_frames()
         << ",\"fir_taps\":" << cfg.history_frames() + 1 << ",\"fft_frames\":" << ex.fft_frames()
         << ",\"streams\":" << streams << ",\"selected_stream\":" << selected
         << ",\"wire\":\"" << (format == WireFormat::packed12 ? "packed12" : "signed8")
         << "\",\"signal\":\"" << signal << "\",\"max_abs_error\":" << max_error
         << ",\"rms_error\":" << std::sqrt(sum / double(got.size())) << ",\"seam_max_error\":" << seam << "}";
    std::ofstream meta(dir / "meta.json");
    meta << json.str() << '\n';
    if (!meta) throw std::runtime_error("cannot write metadata");
    std::cout << json.str() << '\n';
    return max_error <= 3e-6 ? 0 : 1;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 2;
}
