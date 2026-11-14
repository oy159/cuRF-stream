// curf_pipeline: run the P1 streaming pipeline and print one JSON line of stats.
//
//   curf_pipeline [--source synthetic|file] [--file PATH] [--loop]
//                 [--format packed12|signed8] [--channels N] [--rate SPS] [--realtime]
//                 [--block-frames N] [--slots N] [--depth N] [--nfft N]
//                 [--seconds S] [--blocks N] [--drop] [--no-register] [--unpack naive|vec4]
//                 [--label TEXT]

#include "curf/cuda/pipeline.hpp"
#include "curf/sources.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

using namespace curf;

int main(int argc, char **argv) try {
    std::string source = "synthetic", file, label;
    bool loop = false, realtime = false;
    StreamFormat fmt{WireFormat::packed12, 2, 61.44e6};
    cuda::PipelineConfig cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--source") source = val();
        else if (a == "--file") file = val();
        else if (a == "--loop") loop = true;
        else if (a == "--format") fmt.wire = val() == "signed8" ? WireFormat::signed8 : WireFormat::packed12;
        else if (a == "--channels") fmt.channels = static_cast<std::uint32_t>(std::stoul(val()));
        else if (a == "--rate") fmt.sample_rate = std::stod(val());
        else if (a == "--realtime") realtime = true;
        else if (a == "--block-frames") cfg.block_frames = std::stoull(val());
        else if (a == "--slots") cfg.ring_slots = std::stoull(val());
        else if (a == "--depth") cfg.depth = std::stoull(val());
        else if (a == "--nfft") cfg.nfft = std::stoull(val());
        else if (a == "--seconds") cfg.seconds = std::stod(val());
        else if (a == "--blocks") cfg.max_blocks = std::stoull(val());
        else if (a == "--drop") cfg.drop_when_full = true;
        else if (a == "--no-register") cfg.register_ring = false;
        else if (a == "--unpack") cfg.unpack = val() == "naive" ? cuda::UnpackVariant::naive : cuda::UnpackVariant::vec4;
        else if (a == "--label") label = val();
        else throw std::invalid_argument("unknown option " + a);
    }

    std::unique_ptr<ISource> src;
    if (source == "file") {
        src = std::make_unique<FileSource>(FileConfig{file, fmt, loop});
    } else {
        SyntheticConfig sc;
        sc.format = fmt;
        sc.realtime = realtime;
        sc.tones = {{{fmt.sample_rate / 16, 0.3}}, {{-fmt.sample_rate / 8, 0.2}}};
        src = std::make_unique<SyntheticSource>(sc);
    }

    const auto st = cuda::run_pipeline(*src, cfg);
    std::printf("{\"label\":\"%s\",\"source\":\"%s\",\"realtime\":%s,\"channels\":%u,\"rate\":%.0f,"
                "\"block_frames\":%zu,\"slots\":%zu,\"depth\":%zu,\"drop\":%s,\"stats\":%s}\n",
                label.c_str(), source.c_str(), realtime ? "true" : "false", fmt.channels, fmt.sample_rate,
                cfg.block_frames, cfg.ring_slots, cfg.depth, cfg.drop_when_full ? "true" : "false",
                st.to_json().c_str());
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "curf_pipeline: %s\n", e.what());
    return 1;
}
