#include "curf/runtime/producer.hpp"

#include <chrono>
#include <vector>

namespace curf {

    std::thread start_producer(ISource &source, HostRing &ring, std::size_t block_frames, std::size_t fb,
                               bool drop_when_full, ProducerControl &ctl) {
        return std::thread([&source, &ring, block_frames, fb, drop_when_full, &ctl] {
            std::vector<std::uint8_t> trash(drop_when_full ? block_frames * fb : 0);
            try {
                source.start();
                while (!ctl.stop.load(std::memory_order_relaxed)) {
                    HostSlot *slot = drop_when_full ? ring.try_begin_write() : ring.begin_write(std::chrono::milliseconds(10));
                    if (!slot) {
                        if (!drop_when_full) continue; // back-pressure: wait for the GPU
                        const auto r = source.read(trash, 100);
                        if (r.status == ReadStatus::end_of_stream) break;
                        ctl.dropped.fetch_add(r.frames, std::memory_order_relaxed);
                        continue;
                    }
                    std::size_t got = 0;
                    bool first = true, eos = false;
                    while (got < block_frames && !ctl.stop.load(std::memory_order_relaxed)) {
                        const auto r = source.read({slot->data + got * fb, (block_frames - got) * fb}, 100);
                        if (r.status == ReadStatus::end_of_stream) {
                            eos = true;
                            break;
                        }
                        if (r.status != ReadStatus::ok) continue;
                        if (first) {
                            slot->first_sample = r.first_sample;
                            slot->discontinuity = r.discontinuity;
                            first = false;
                        } else if (r.discontinuity) {
                            slot->discontinuity = true;
                        }
                        got += r.frames;
                    }
                    if (got < block_frames) { // EOS or stop mid-block: drop the partial block
                        if (eos) break;
                        continue;
                    }
                    slot->frames = got;
                    ring.commit_write();
                }
                source.stop();
            } catch (...) {
                ctl.error = std::current_exception();
            }
            ctl.done.store(true, std::memory_order_release);
            ring.notify_all();
        });
    }

    void stop_producer(HostRing &ring, ProducerControl &ctl, std::thread &producer) {
        ctl.stop.store(true);
        ring.notify_all();
        while (ring.in_flight() > 0) ring.release_oldest();
        while (ring.try_next_submit()) {
            ring.mark_submitted();
            ring.release_oldest();
        }
        if (producer.joinable()) producer.join();
    }

} // namespace curf
