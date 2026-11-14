#pragma once

#include "curf/format.hpp"

#include <cstdint>
#include <span>

namespace curf {

    enum class ReadStatus : std::uint8_t {
        ok,           // frames > 0 were written
        timeout,      // nothing available within the timeout
        end_of_stream // finite source exhausted
    };

    struct ReadResult {
        ReadStatus status = ReadStatus::timeout;
        std::size_t frames = 0;          // whole frames written to the destination
        std::uint64_t first_sample = 0;  // sample time (1 tick = 1 sample) of the first frame
        bool discontinuity = false;      // samples were lost before first_sample
    };

    // A stream of channel-interleaved wire frames.
    //
    // The caller owns the destination memory, so a runtime can hand in pinned
    // (page-locked) buffers that the GPU copies from directly. A source writes
    // whole frames only and never more than dst.size() / frame_bytes().
    class ISource {
    public:
        virtual ~ISource() = default;

        virtual StreamFormat format() const = 0;
        virtual void start() {}
        virtual void stop() {}
        virtual ReadResult read(std::span<std::uint8_t> dst, std::uint32_t timeout_ms) = 0;
    };

} // namespace curf
