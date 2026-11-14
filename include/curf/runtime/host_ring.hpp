#pragma once

// Fixed-slot single-producer / single-consumer ring of host buffers that the
// GPU reads with DMA.
//
// Memory: one anonymous mapping aligned to 2 MiB and advised MADV_HUGEPAGE, so
// that (once page-locked with cudaHostRegister) the IOMMU maps it with 2 MiB
// superpages. With the IOMMU in translated mode, 4 KiB mappings cap H2D at
// ~7 GB/s on the reference laptop versus ~12.4 GB/s with huge pages
// (see PLAN.md, 2026-11-14).
//
// Slot life cycle, three monotonically increasing counters:
//   producer:  [release, write) are owned by readers; fills slot `write`, then commits
//   consumer:  submits slots in [submitted, write) to the GPU (several in flight),
//              releases them in order once the GPU has finished reading them
//   free slots = N - (write - release)
// `write` and `release` are atomics shared between the two threads; `submitted`
// is private to the consumer. A mutex/condition variable is used only to sleep.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace curf {

    struct HostSlot {
        std::uint8_t *data = nullptr;
        std::size_t capacity = 0;      // bytes
        // Filled by the producer:
        std::size_t frames = 0;
        std::uint64_t first_sample = 0;
        bool discontinuity = false;
        std::uint64_t sequence = 0;    // commit order
        std::chrono::steady_clock::time_point filled_at{};
    };

    // THP status of a memory range, read from /proc/self/smaps (Linux).
    struct HugePageInfo {
        bool known = false;            // false off Linux or if smaps is unreadable
        std::size_t mapping_bytes = 0;
        std::size_t huge_bytes = 0;    // AnonHugePages
        double coverage() const noexcept {
            return mapping_bytes ? static_cast<double>(huge_bytes) / static_cast<double>(mapping_bytes) : 0.0;
        }
    };

    class HostRing {
    public:
        // slot_bytes is rounded up to a multiple of 4 KiB.
        HostRing(std::size_t slots, std::size_t slot_bytes);
        ~HostRing();
        HostRing(const HostRing &) = delete;
        HostRing &operator=(const HostRing &) = delete;

        std::size_t slots() const noexcept { return slots_.size(); }
        std::size_t slot_bytes() const noexcept { return slot_bytes_; }
        std::uint8_t *base() const noexcept { return base_; }
        std::size_t bytes() const noexcept { return bytes_; }
        HugePageInfo huge_pages() const;

        // ---- producer
        HostSlot *try_begin_write() noexcept;
        HostSlot *begin_write(std::chrono::microseconds timeout);
        void commit_write() noexcept;

        // ---- consumer
        HostSlot *try_next_submit() noexcept; // next committed, not yet submitted slot
        HostSlot *next_submit(std::chrono::microseconds timeout);
        void mark_submitted() noexcept;
        std::size_t in_flight() const noexcept {
            return static_cast<std::size_t>(submitted_ - release_.load(std::memory_order_relaxed));
        }
        HostSlot &oldest_in_flight() noexcept { return slots_[release_.load(std::memory_order_relaxed) % slots_.size()]; }
        void release_oldest() noexcept;

        // Wake any sleeper (used for shutdown).
        void notify_all();

    private:
        void notify();

        std::uint8_t *base_ = nullptr;
        std::size_t bytes_ = 0;
        std::size_t map_bytes_ = 0;
        void *map_ = nullptr;
        std::size_t slot_bytes_ = 0;
        std::vector<HostSlot> slots_;

        alignas(64) std::atomic<std::uint64_t> write_{0};
        alignas(64) std::atomic<std::uint64_t> release_{0};
        alignas(64) std::uint64_t submitted_ = 0;

        std::mutex mutex_;
        std::condition_variable cv_;
    };

} // namespace curf
