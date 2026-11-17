#include "curf/cuda/extraction_bank.hpp"
#include "curf/cuda/memory.hpp"
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

using namespace curf;

TEST_CASE("shared FFT bank matches continuous rational FIR across blocks", "[cuda][extraction][bank]") {
    constexpr unsigned hop = 4096, blocks = 5, streams = 2;
    constexpr double fs = 61.44e6;
    constexpr std::uint64_t first = 123456789;
    const std::size_t total = blocks * hop;
    const int centres[3] = {-321, 2107, 1201}; // not aligned to the hop's FFT bins
    std::vector<cf32> raw(total * streams);
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0, .1f);
    for (unsigned s = 0; s < streams; ++s)
        for (std::size_t n = 0; n < total; ++n) {
            const double ph = 2 * std::numbers::pi * (centres[s] + .37) * double(n) / (4 * hop);
            const float a = n >= hop - 23 && n < 2 * hop + 5 ? .3f : 0;
            raw[s * total + n] = {nd(rng) + a * float(std::cos(ph)), nd(rng) + a * float(std::sin(ph))};
        }
    cuda::ExtractionBank bank(hop, streams, fs);
    cuda::DeviceBuffer<cf32> spec(streams * 4 * hop), out(4 * 4 * hop);
    cuda::PinnedBuffer<cf32> context(blocks * spec.size()), got(blocks * out.size());
    for (unsigned b = 0; b < blocks; ++b)
        for (unsigned s = 0; s < streams; ++s)
            for (unsigned j = 0; j < 4 * hop; ++j) {
                const long n = long(b) * hop - 3 * hop + j;
                context[b * spec.size() + s * 4 * hop + j] = n < 0 ? cf32{0,0} : raw[s * total + std::size_t(n)];
            }
    cudaStream_t stream;
    CURF_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (unsigned w : {16u, 24u, 64u, 96u, 1024u, 1536u, 4096u}) {
        std::vector<std::vector<cf32>> expected(3, std::vector<cf32>(blocks * w));
        for (unsigned s = 0; s < 3; ++s)
            ref::resample_signal(fs, centres[s] * fs / (4 * hop), extraction_ratio(hop, w),
                                std::span(raw.data() + (s % streams) * total, total), first, expected[s]);
        for (unsigned b = 0; b < blocks; ++b) {
            CURF_CUDA_CHECK(cudaMemcpyAsync(spec.get(), context.get() + b * spec.size(), spec.bytes(), cudaMemcpyHostToDevice, stream));
            bank.forward(spec.get(), stream);
            const cuda::ExtractionRequest req[] = {{0, centres[0], first + b * hop}, {1, centres[1], first + b * hop}, {0, centres[2], first + b * hop}};
            bank.extract(spec.get(), w, req, out.get(), stream);
            CURF_CUDA_CHECK(cudaMemcpyAsync(got.get() + b * out.size(), out.get(), 4 * 4 * w * sizeof(cf32), cudaMemcpyDeviceToHost, stream));
        }
        CURF_CUDA_CHECK(cudaStreamSynchronize(stream));
        double err = 0, seam = 0;
        for (unsigned b = 0; b < blocks; ++b)
            for (unsigned s = 0; s < 3; ++s)
                for (unsigned j = 0; j < w; ++j) {
                    const auto a = got[b * out.size() + s * 4 * w + j], e = expected[s][b * w + j];
                    const double v = std::hypot(a.re - e.re, a.im - e.im);
                    err = std::max(err, v);
                    if (j < 4 || j + 4 >= w) seam = std::max(seam, v);
                }
        INFO("W=" << w << " error=" << err << " seam=" << seam);
        CHECK(err < 3e-6);
        CHECK(seam < 3e-6);
    }
    CURF_CUDA_CHECK(cudaStreamDestroy(stream));
}
