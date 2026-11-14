// P1 baseline numbers:
//  - H2D bandwidth (pinned vs pageable): what the PCIe link delivers
//  - D2D memcpy bandwidth: practical DRAM ceiling for a read+write kernel
//  - unpack kernels: achieved bandwidth and fraction of the D2D ceiling
// Output: one JSON object per line.

#include "curf/cuda/memory.hpp"
#include "curf/cuda/unpack.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

using namespace curf;

namespace {

    // Median time in ms of `iters` runs of fn on `stream`, after warm-up.
    float time_ms(cudaStream_t stream, int iters, const std::function<void()> &fn) {
        for (int i = 0; i < 3; ++i) fn();
        CURF_CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaEvent_t a, b;
        CURF_CUDA_CHECK(cudaEventCreate(&a));
        CURF_CUDA_CHECK(cudaEventCreate(&b));
        std::vector<float> t;
        for (int i = 0; i < iters; ++i) {
            CURF_CUDA_CHECK(cudaEventRecord(a, stream));
            fn();
            CURF_CUDA_CHECK(cudaEventRecord(b, stream));
            CURF_CUDA_CHECK(cudaEventSynchronize(b));
            float ms = 0;
            CURF_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
            t.push_back(ms);
        }
        CURF_CUDA_CHECK(cudaEventDestroy(a));
        CURF_CUDA_CHECK(cudaEventDestroy(b));
        std::nth_element(t.begin(), t.begin() + static_cast<long>(t.size() / 2), t.end());
        return t[t.size() / 2];
    }

    double gbps(double bytes, float ms) { return bytes / (static_cast<double>(ms) * 1e-3) / 1e9; }

} // namespace

int main() {
    int dev = 0;
    cudaDeviceProp prop{};
    CURF_CUDA_CHECK(cudaGetDevice(&dev));
    CURF_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int mem_clock_khz = 0, bus_width = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, dev);
    const double dram_peak = 2.0 * mem_clock_khz * 1e3 * (bus_width / 8.0) / 1e9;
    std::printf("{\"device\":\"%s\",\"sm\":%d,\"cc\":\"%d.%d\",\"dram_peak_GBps\":%.1f}\n", prop.name,
                prop.multiProcessorCount, prop.major, prop.minor, dram_peak);

    cudaStream_t stream;
    CURF_CUDA_CHECK(cudaStreamCreate(&stream));
    constexpr int iters = 50;

    // ---- H2D: 64 MiB
    {
        const std::size_t bytes = std::size_t{64} << 20;
        cuda::DeviceBuffer<std::uint8_t> d(bytes);
        cuda::PinnedBuffer<std::uint8_t> pinned(bytes);
        std::vector<std::uint8_t> pageable(bytes, 1);
        std::memset(pinned.get(), 1, bytes);
        const float t_pin = time_ms(stream, iters, [&] {
            CURF_CUDA_CHECK(cudaMemcpyAsync(d.get(), pinned.get(), bytes, cudaMemcpyHostToDevice, stream));
        });
        const float t_page = time_ms(stream, 10, [&] {
            CURF_CUDA_CHECK(cudaMemcpyAsync(d.get(), pageable.data(), bytes, cudaMemcpyHostToDevice, stream));
        });
        std::printf("{\"bench\":\"h2d\",\"bytes\":%zu,\"pinned_GBps\":%.2f,\"pageable_GBps\":%.2f}\n", bytes,
                    gbps(static_cast<double>(bytes), t_pin), gbps(static_cast<double>(bytes), t_page));
    }

    // ---- unpack: 2 channels x 2^24 frames (~100 MB packed12 in, 268 MB out)
    const std::uint32_t channels = 2;
    const std::size_t frames = std::size_t{1} << 24;
    cuda::DeviceBuffer<cf32> d_out(frames * channels);

    // D2D ceiling: copy as many bytes as the unpack output (read+write traffic counted).
    double d2d = 0;
    {
        cuda::DeviceBuffer<std::uint8_t> src(d_out.bytes());
        const float t = time_ms(stream, iters, [&] {
            CURF_CUDA_CHECK(cudaMemcpyAsync(d_out.get(), src.get(), d_out.bytes(), cudaMemcpyDeviceToDevice, stream));
        });
        d2d = gbps(2.0 * static_cast<double>(d_out.bytes()), t);
        std::printf("{\"bench\":\"d2d\",\"bytes\":%zu,\"GBps\":%.1f,\"pct_dram_peak\":%.1f}\n", d_out.bytes(), d2d,
                    100.0 * d2d / dram_peak);
    }

    for (const auto format : {WireFormat::packed12, WireFormat::signed8}) {
        const std::size_t in_bytes = frames * channels * wire_sample_bytes(format);
        cuda::DeviceBuffer<std::uint8_t> d_wire(in_bytes);
        CURF_CUDA_CHECK(cudaMemset(d_wire.get(), 0x5a, in_bytes));
        for (const auto variant : {cuda::UnpackVariant::naive, cuda::UnpackVariant::vec4}) {
            const float t = time_ms(stream, iters, [&] {
                cuda::unpack_planar(d_wire.get(), format, channels, frames, d_out.get(), stream, variant);
            });
            const double traffic = static_cast<double>(in_bytes + d_out.bytes());
            const double msps = static_cast<double>(frames * channels) / (static_cast<double>(t) * 1e-3) / 1e6;
            std::printf("{\"bench\":\"unpack\",\"format\":\"%s\",\"variant\":\"%s\",\"channels\":%u,\"frames\":%zu,"
                        "\"ms\":%.3f,\"GBps\":%.1f,\"pct_d2d\":%.1f,\"Msamples_per_s\":%.0f,\"x_realtime_2x61.44\":%.0f}\n",
                        format == WireFormat::packed12 ? "packed12" : "signed8", cuda::to_string(variant), channels,
                        frames, static_cast<double>(t), gbps(traffic, t), 100.0 * gbps(traffic, t) / d2d, msps,
                        msps / (2 * 61.44));
        }
    }

    CURF_CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
