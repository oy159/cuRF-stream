// How fast can the synthetic source feed wire data on the host?
// It must comfortably exceed the PCIe rate to be useful as a stress source.

#include "curf/sources.hpp"

#include <chrono>
#include <cstdio>
#include <vector>

int main() {
    curf::SyntheticConfig cfg;
    cfg.format = {curf::WireFormat::packed12, 2, 61.44e6};
    cfg.tones = {{{1e6, 0.3}}, {{-3e6, 0.2}}};
    curf::SyntheticSource src(cfg);

    std::vector<std::uint8_t> buf(std::size_t{8} << 20); // 8 MiB reads
    const auto t0 = std::chrono::steady_clock::now();
    std::size_t bytes = 0;
    while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2)) {
        const auto r = src.read(buf, 0);
        bytes += r.frames * cfg.format.frame_bytes();
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double gbps = static_cast<double>(bytes) / s / 1e9;
    std::printf("{\"bench\":\"synthetic_source\",\"GBps\":%.2f,\"x_usb_rate\":%.1f}\n", gbps,
                gbps * 1e9 / cfg.format.wire_bytes_per_second());
    return 0;
}
