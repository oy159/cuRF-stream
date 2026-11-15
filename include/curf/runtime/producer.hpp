#pragma once
// Producer thread shared by the pipelines: reads whole blocks from an ISource into HostRing slots.
//
// Back-pressure mode waits for a free slot; drop mode discards input while the ring is full, as a hardware
// FIFO overflow would, and counts the dropped frames. A partial block at end of stream is dropped.

#include "curf/runtime/host_ring.hpp"
#include "curf/source.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <thread>

namespace curf {

    struct ProducerControl {
        std::atomic<bool> stop{false};          // set by the consumer
        std::atomic<bool> done{false};          // set by the producer when it exits
        std::atomic<std::uint64_t> dropped{0};  // frames discarded in drop mode
        std::exception_ptr error;               // valid after the thread has been joined
    };

    std::thread start_producer(ISource &source, HostRing &ring, std::size_t block_frames, std::size_t frame_bytes,
                               bool drop_when_full, ProducerControl &ctl);

    // Consumer-side teardown: stop the producer, release every slot it may be waiting on, join it.
    void stop_producer(HostRing &ring, ProducerControl &ctl, std::thread &producer);

} // namespace curf
