// PCIe transfer characterisation: H2D / D2H / bidirectional across sizes and
// host allocation types. Output: one JSON object per line.

#include "curf/cuda/memory.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif

using namespace curf;

namespace {

    enum class Host { pinned, pinned_wc, pageable, registered_4k, registered_thp };
    const char *name(Host h) {
        switch (h) {
        case Host::pinned: return "pinned";
        case Host::pinned_wc: return "pinned_wc";
        case Host::pageable: return "pageable";
        case Host::registered_4k: return "registered_4k";
        case Host::registered_thp: return "registered_thp";
        }
        return "?";
    }

    // 2 MiB aligned host memory, optionally backed by transparent huge pages,
    // page-locked with cudaHostRegister. Huge pages let the IOMMU map the buffer
    // with 2 MiB superpages instead of 4 KiB pages.
    struct Registered {
        std::uint8_t *ptr = nullptr;
        std::size_t bytes = 0;
        Registered(std::size_t n, bool huge) : bytes(n) {
            constexpr std::size_t align = std::size_t{2} << 20;
            ptr = static_cast<std::uint8_t *>(std::aligned_alloc(align, (n + align - 1) / align * align));
            if (!ptr) throw std::bad_alloc();
#if defined(__linux__)
            madvise(ptr, n, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
#else
            (void) huge;
#endif
            std::memset(ptr, 1, n); // fault the pages in (as huge pages where possible)
            CURF_CUDA_CHECK(cudaHostRegister(ptr, n, cudaHostRegisterDefault));
        }
        ~Registered() {
            cudaHostUnregister(ptr);
            std::free(ptr);
        }
        Registered(const Registered &) = delete;
        Registered &operator=(const Registered &) = delete;
    };

    // Fraction of the process's anonymous memory currently backed by THP (Linux).
    long anon_huge_kb() {
        long kb = -1;
#if defined(__linux__)
        if (FILE *f = std::fopen("/proc/self/smaps_rollup", "r")) {
            char line[256];
            while (std::fgets(line, sizeof line, f))
                if (std::sscanf(line, "AnonHugePages: %ld kB", &kb) == 1) break;
            std::fclose(f);
        }
#endif
        return kb;
    }

    // Median GB/s over `iters` copies timed with events.
    double copy_gbps(void *dst, const void *src, std::size_t bytes, cudaMemcpyKind kind, cudaStream_t s, int iters) {
        cudaEvent_t a, b;
        CURF_CUDA_CHECK(cudaEventCreate(&a));
        CURF_CUDA_CHECK(cudaEventCreate(&b));
        for (int i = 0; i < 3; ++i) CURF_CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, kind, s));
        std::vector<double> r;
        for (int i = 0; i < iters; ++i) {
            CURF_CUDA_CHECK(cudaEventRecord(a, s));
            CURF_CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, kind, s));
            CURF_CUDA_CHECK(cudaEventRecord(b, s));
            CURF_CUDA_CHECK(cudaEventSynchronize(b));
            float ms = 0;
            CURF_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
            r.push_back(static_cast<double>(bytes) / (ms * 1e-3) / 1e9);
        }
        CURF_CUDA_CHECK(cudaEventDestroy(a));
        CURF_CUDA_CHECK(cudaEventDestroy(b));
        std::nth_element(r.begin(), r.begin() + static_cast<long>(r.size() / 2), r.end());
        return r[r.size() / 2];
    }

} // namespace

int main() {
    constexpr std::size_t max_bytes = std::size_t{256} << 20;
    cuda::DeviceBuffer<std::uint8_t> d(max_bytes), d2(max_bytes);
    cuda::PinnedBuffer<std::uint8_t> pinned(max_bytes), pinned2(max_bytes);
    cuda::PinnedBuffer<std::uint8_t> pinned_wc(max_bytes, cudaHostAllocWriteCombined);
    std::vector<std::uint8_t> pageable(max_bytes);
    std::memset(pinned.get(), 1, max_bytes);
    std::memset(pinned2.get(), 1, max_bytes);
    std::memset(pinned_wc.get(), 1, max_bytes);
    std::memset(pageable.data(), 1, max_bytes);
    const long thp_before = anon_huge_kb();
    Registered reg_4k(max_bytes, false);
    Registered reg_thp(max_bytes, true);
    std::printf("{\"info\":\"thp\",\"anon_huge_kB_before\":%ld,\"anon_huge_kB_after\":%ld,\"buffer_kB\":%zu}\n",
                thp_before, anon_huge_kb(), max_bytes >> 10);

    cudaStream_t s1, s2;
    CURF_CUDA_CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CURF_CUDA_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));

    for (std::size_t bytes = std::size_t{64} << 10; bytes <= max_bytes; bytes *= 4) {
        for (const Host h : {Host::pinned, Host::pinned_wc, Host::pageable, Host::registered_4k, Host::registered_thp}) {
            void *host = nullptr;
            switch (h) {
            case Host::pinned: host = pinned.get(); break;
            case Host::pinned_wc: host = pinned_wc.get(); break;
            case Host::pageable: host = pageable.data(); break;
            case Host::registered_4k: host = reg_4k.ptr; break;
            case Host::registered_thp: host = reg_thp.ptr; break;
            }
            const int iters = bytes >= (std::size_t{64} << 20) ? 10 : 30;
            const double h2d = copy_gbps(d.get(), host, bytes, cudaMemcpyHostToDevice, s1, iters);
            const double d2h = copy_gbps(host, d.get(), bytes, cudaMemcpyDeviceToHost, s1, iters);
            std::printf("{\"bench\":\"pcie\",\"host\":\"%s\",\"bytes\":%zu,\"h2d_GBps\":%.2f,\"d2h_GBps\":%.2f}\n",
                        name(h), bytes, h2d, d2h);
        }
    }

    // Bidirectional: H2D on s1 and D2H on s2 at once (two copy engines).
    {
        const std::size_t bytes = std::size_t{64} << 20;
        const int iters = 10;
        for (int i = 0; i < 2; ++i) {
            CURF_CUDA_CHECK(cudaMemcpyAsync(d.get(), pinned.get(), bytes, cudaMemcpyHostToDevice, s1));
            CURF_CUDA_CHECK(cudaMemcpyAsync(pinned2.get(), d2.get(), bytes, cudaMemcpyDeviceToHost, s2));
        }
        CURF_CUDA_CHECK(cudaDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            CURF_CUDA_CHECK(cudaMemcpyAsync(d.get(), pinned.get(), bytes, cudaMemcpyHostToDevice, s1));
            CURF_CUDA_CHECK(cudaMemcpyAsync(pinned2.get(), d2.get(), bytes, cudaMemcpyDeviceToHost, s2));
        }
        CURF_CUDA_CHECK(cudaDeviceSynchronize());
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double each = static_cast<double>(bytes) * iters / sec / 1e9;
        std::printf("{\"bench\":\"pcie_bidir\",\"bytes\":%zu,\"h2d_GBps\":%.2f,\"d2h_GBps\":%.2f,\"total_GBps\":%.2f}\n",
                    bytes, each, each, 2 * each);
    }

    // Host memory bandwidth reference: memcpy pinned -> pinned.
    {
        const std::size_t bytes = std::size_t{256} << 20;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 5; ++i) std::memcpy(pinned2.get(), pinned.get(), bytes);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("{\"bench\":\"host_memcpy\",\"bytes\":%zu,\"GBps\":%.2f}\n", bytes, 5.0 * bytes / sec / 1e9);
    }

    int engines = 0;
    CURF_CUDA_CHECK(cudaDeviceGetAttribute(&engines, cudaDevAttrAsyncEngineCount, 0));
    std::printf("{\"device\":\"async_engines\",\"count\":%d}\n", engines);
    return 0;
}
