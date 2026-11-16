#pragma once
// GPU Mode S / ADS-B demodulator on the raw wideband stream.
//
// Per block of wire frames, asynchronously on one CUDA stream:
//   1. unpack the selected stream, mix the 1090 MHz carrier to 0 Hz (phase continuous over sample time)
//   2. low-pass FIR (+-1.5 MHz) and decimate (61.44 -> 15.36 MS/s), magnitude
//   3. 0.5 us matched filter (boxcar of the pulse length, fractional last tap)
//   4. per candidate start: preamble test (pulses at 0, 1, 3.5, 4.5 us against the quiet slots and the noise
//      level), PPM bit decisions, CRC-24; DF17/18 with one bad bit are corrected by syndrome lookup
//   5. frames that pass (DF11/17/18) or carry an address in their parity (DF0/4/5/16/20/21) go to the host
// The last 2048 decimated samples (133 us > one 120 us frame) of each block are kept for the next one, so a
// frame across a block boundary is found once, in the block where it ends.

#include "curf/format.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace curf::cuda {

    struct AdsbConfig {
        double sample_rate = 61.44e6;
        double offset_hz = 0;        // ADS-B carrier (1090 MHz) minus the tuned centre frequency
        std::uint32_t stream = 0;    // stream index within the wire frames
        std::uint32_t decimation = 4;
        double cutoff_hz = 1.5e6;    // FIR passband edge (the 0.5 us pulses' main lobe is +-2 MHz)
        std::uint32_t taps = 64;
        float threshold = 2.0f;      // weakest preamble pulse vs. mean quiet slot and vs. noise (amplitude ratio)
        std::uint32_t capacity = 256; // frames reported per block at most
    };

    struct AdsbFrame {
        std::uint64_t sample;   // approximate input sample time of the preamble start
        float signal, noise;    // mean pulse / noise amplitude (full scale = 1)
        std::uint32_t remainder; // CRC remainder after correction (0 for DF17/18, address for DF0/4/5/16/20/21)
        std::uint8_t msg[14];
        std::uint8_t bits;      // 56 or 112
        std::uint8_t corrected; // 1: one bit fixed by syndrome
    };

    class AdsbDemod {
    public:
        // slots: results buffers, used round-robin by the caller (one per block in flight)
        AdsbDemod(const AdsbConfig &config, WireFormat wire, std::uint32_t channels, std::size_t max_block_frames,
                  std::size_t slots);
        ~AdsbDemod();
        AdsbDemod(const AdsbDemod &) = delete;
        AdsbDemod &operator=(const AdsbDemod &) = delete;

        // Blocks must be enqueued in sample order on the same CUDA stream. frames % decimation == 0.
        void enqueue(const std::uint8_t *d_wire, std::size_t frames, std::uint64_t first_sample, std::size_t slot,
                     cudaStream_t stream);
        // Once the slot's work has completed: frames found (unsorted; neighbours of one frame are possible).
        std::span<const AdsbFrame> results(std::size_t slot) const;
        // Frames the block found, including any beyond `capacity` that were not reported.
        std::uint32_t found(std::size_t slot) const;
        // Mean noise amplitude of the slot's block after the filter (full scale = 1).
        float noise(std::size_t slot) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace curf::cuda
