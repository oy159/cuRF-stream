#include "curf/runtime/host_ring.hpp"

#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <sys/mman.h>

namespace curf {
    namespace {
        constexpr std::size_t kHuge = std::size_t{2} << 20;
        constexpr std::size_t kPage = std::size_t{4} << 10;
        std::size_t round_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }
    } // namespace

    HostRing::HostRing(std::size_t slots, std::size_t slot_bytes) {
        if (slots == 0 || slot_bytes == 0) throw std::invalid_argument("HostRing: empty");
        slot_bytes_ = round_up(slot_bytes, kPage);
        bytes_ = round_up(slots * slot_bytes_, kHuge);

        // Over-map by one huge page and trim, leaving an exactly 2 MiB aligned mapping
        // (its own VMA, so smaps reports its huge-page coverage precisely).
        map_bytes_ = bytes_ + kHuge;
        map_ = mmap(nullptr, map_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (map_ == MAP_FAILED) throw std::bad_alloc();
        const auto raw = reinterpret_cast<std::uintptr_t>(map_);
        const auto aligned = round_up(raw, kHuge);
        const std::size_t head = aligned - raw, tail = map_bytes_ - head - bytes_;
        if (head) munmap(map_, head);
        if (tail) munmap(reinterpret_cast<void *>(aligned + bytes_), tail);
        map_ = reinterpret_cast<void *>(aligned);
        map_bytes_ = bytes_;
        base_ = static_cast<std::uint8_t *>(map_);
#if defined(MADV_HUGEPAGE)
        madvise(base_, bytes_, MADV_HUGEPAGE);
#endif
        std::memset(base_, 0, bytes_); // fault in now, as huge pages where available

        slots_.resize(slots);
        for (std::size_t i = 0; i < slots; ++i) slots_[i] = HostSlot{base_ + i * slot_bytes_, slot_bytes_};
    }

    HostRing::~HostRing() {
        if (map_) munmap(map_, map_bytes_);
    }

    HugePageInfo HostRing::huge_pages() const {
        HugePageInfo info;
#if defined(__linux__)
        FILE *f = std::fopen("/proc/self/smaps", "r");
        if (!f) return info;
        char line[512];
        bool in_vma = false;
        const auto base = reinterpret_cast<std::uintptr_t>(base_);
        while (std::fgets(line, sizeof line, f)) {
            unsigned long lo = 0, hi = 0;
            long kb = 0;
            if (std::sscanf(line, "%lx-%lx ", &lo, &hi) == 2 && std::strchr(line, ' ')) {
                if (in_vma) break; // past our mapping
                in_vma = lo <= base && base < hi;
                if (in_vma) info.mapping_bytes = hi - lo;
            } else if (in_vma && std::sscanf(line, "AnonHugePages: %ld kB", &kb) == 1) {
                info.huge_bytes = static_cast<std::size_t>(kb) * 1024;
                info.known = true;
            }
        }
        std::fclose(f);
#endif
        return info;
    }

    void HostRing::notify() {
        { std::lock_guard lk(mutex_); }
        cv_.notify_all();
    }
    void HostRing::notify_all() { notify(); }

    // ------------------------------------------------------------ producer
    HostSlot *HostRing::try_begin_write() noexcept {
        const auto w = write_.load(std::memory_order_relaxed);
        if (w - release_.load(std::memory_order_acquire) >= slots_.size()) return nullptr;
        return &slots_[w % slots_.size()];
    }

    HostSlot *HostRing::begin_write(std::chrono::microseconds timeout) {
        if (auto *s = try_begin_write()) return s;
        std::unique_lock lk(mutex_);
        HostSlot *s = nullptr;
        cv_.wait_for(lk, timeout, [&] { return (s = try_begin_write()) != nullptr; });
        return s;
    }

    void HostRing::commit_write() noexcept {
        const auto w = write_.load(std::memory_order_relaxed);
        auto &slot = slots_[w % slots_.size()];
        slot.sequence = w;
        slot.filled_at = std::chrono::steady_clock::now();
        write_.store(w + 1, std::memory_order_release);
        notify();
    }

    // ------------------------------------------------------------ consumer
    HostSlot *HostRing::try_next_submit() noexcept {
        if (submitted_ >= write_.load(std::memory_order_acquire)) return nullptr;
        return &slots_[submitted_ % slots_.size()];
    }

    HostSlot *HostRing::next_submit(std::chrono::microseconds timeout) {
        if (auto *s = try_next_submit()) return s;
        std::unique_lock lk(mutex_);
        HostSlot *s = nullptr;
        cv_.wait_for(lk, timeout, [&] { return (s = try_next_submit()) != nullptr; });
        return s;
    }

    void HostRing::mark_submitted() noexcept { ++submitted_; }

    void HostRing::release_oldest() noexcept {
        release_.store(release_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
        notify();
    }

} // namespace curf
