#pragma once
// Turns per-read hardware metadata into the ISource sample timeline (first_sample, discontinuity).
//
// The timeline starts at 0 with the first delivered frame. With hardware time (OmniCore framed transport:
// 1 tick = 1 sample) a read's position is its time minus the first read's time, so every lost sample shows
// up as a gap of exactly its length. Without hardware time the position advances by the frames delivered
// plus the losses the driver reports. A time that goes backwards (the device time was reset) cannot be
// placed on the timeline: the read continues right after the previous one, flagged as a discontinuity, and
// the time origin moves so the following reads line up again.

#include <cstdint>

namespace curf {

    struct ReadMeta {
        bool has_time = false;
        std::uint64_t time = 0;        // hardware sample time of the first frame (when has_time)
        std::uint64_t dropped = 0;     // samples the driver knows were lost right before this read
        bool gap_flagged = false;      // the driver marked a discontinuity / overflow before this read
    };

    struct ReadPlacement {
        std::uint64_t first_sample = 0;
        bool discontinuity = false;
    };

    class SampleClock {
    public:
        ReadPlacement place(const ReadMeta &m, std::uint64_t frames) {
            ReadPlacement p;
            if (!started_) {
                started_ = true;
                origin_ = m.has_time ? m.time : 0;
                p.first_sample = 0;
            } else if (m.has_time) {
                // modular: the origin may lie "above" the time after a rebase
                const std::uint64_t t = m.time - origin_;
                if (static_cast<std::int64_t>(t - next_) < 0) { // time went backwards: rebase onto the timeline
                    origin_ = m.time - next_;
                    p.first_sample = next_;
                    p.discontinuity = true;
                } else {
                    p.first_sample = t;
                    p.discontinuity = t != next_ || m.gap_flagged;
                }
            } else {
                p.first_sample = next_ + m.dropped;
                p.discontinuity = m.dropped > 0 || m.gap_flagged;
            }
            next_ = p.first_sample + frames;
            return p;
        }

        std::uint64_t next() const noexcept { return next_; }

    private:
        bool started_ = false;
        std::uint64_t origin_ = 0, next_ = 0;
    };

} // namespace curf
