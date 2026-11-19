#pragma once
// Live RX from an OmniCoreSDR board through OmniCore-SDR-Host's C ABI (built with -DCURF_OMNICORE=ON).
//
// read() is one och_recv in the wire format itself (packed12 for 12-bit USB IQ, sc8 for 8-bit), which is a
// plain copy from the driver's ring into the caller's pinned slot: the same single host copy the zero-copy
// lease (och_rx_acquire) would need, but recv can stop mid-block, so a read never has to hold back data.
// recv never mixes epochs in one buffer, so each read has one sample time; SampleClock places it on the
// timeline and reports every loss (host ring overflow, device overflow, reconfiguration) as a discontinuity.

#include "curf/sample_clock.hpp"
#include "curf/source.hpp"

#include <atomic>
#include <exception>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>
#include <vector>

struct och_device;
struct och_stream;

namespace curf {

    struct RxCorrection {
        bool dc_tracking, iq_tracking;
    };

    struct OmniCoreConfig {
        std::string serial;              // empty: the only connected board
        double sample_rate = 61.44e6;    // per channel
        std::uint32_t channels = 2;      // 1: RX1, 2: RX1 + RX2
        std::uint32_t rx_mask = 0;       // explicit RX selection: 1 RX1, 2 RX2, 3 both (0: from channels)
        std::uint64_t frequency_hz = 2'400'000'000;
        std::uint32_t bandwidth_hz = 0;  // 0: min(sample rate, 56 MHz) (AD9361 analog filter limit)
        double gain_db[2] = {30, 30};    // manual gain (agc = 0)
        std::uint32_t agc = 0;           // 0 manual, 1 fast, 2 slow, 3 hybrid
        std::uint32_t iq_bits = 12;      // 12: packed12 wire; 8: signed8 wire (the driver needs both TX and RX
                                         // masks = 3 for it; no TX stream is created)
        std::uint32_t host_buffer_mib = 256; // driver ring between the USB thread and read()
        std::optional<bool> rx_correction; // unset: preserve the board's DC/IQ tracking state

        // Optional loopback transmitter on TX1, same LO frequency and bandwidth as RX: the file (sc16 I/Q
        // interleaved, one channel) is sent in a loop from start() to stop(). The TX attenuation is set in the
        // same configuration call that enables TX, before anything is sent, and may never go below
        // kMinTxAttenuationDb: with TX1 cabled straight to RX1 (no external attenuator) the RX then sees
        // < -33 dBm, far below its +2.5 dBm absolute maximum.
        std::string tx_file;
        double tx_attenuation_db = 89.75;
        static constexpr double kMinTxAttenuationDb = 40.0;
    };

    struct OmniCoreStats {
        std::uint64_t usb_bytes = 0, samples = 0;
        std::uint64_t host_dropped = 0, device_overflows = 0, transfer_errors = 0, resyncs = 0;
        std::uint64_t host_queue_peak_bytes = 0;
        std::uint64_t discontinuities = 0; // reads flagged by this source
        std::uint64_t tx_samples = 0, tx_underflows = 0, tx_starvation = 0; // loopback TX only
    };

    class OmniCoreSource final : public ISource {
    public:
        explicit OmniCoreSource(OmniCoreConfig config);
        ~OmniCoreSource() override;
        OmniCoreSource(const OmniCoreSource &) = delete;
        OmniCoreSource &operator=(const OmniCoreSource &) = delete;

        StreamFormat format() const override { return format_; }
        void start() override;
        void stop() override;
        ReadResult read(std::span<std::uint8_t> dst, std::uint32_t timeout_ms) override;

        OmniCoreStats stats() const;
        std::string device_name() const { return name_; }
        const std::optional<RxCorrection> &rx_correction() const { return correction_; }

    private:
        OmniCoreConfig config_;
        StreamFormat format_{};
        och_device *dev_ = nullptr;
        och_stream *rx_ = nullptr;
        och_stream *tx_ = nullptr;
        bool running_ = false, tx_running_ = false;
        std::vector<std::int16_t> tx_buf_; // I/Q interleaved
        std::thread tx_thread_;
        std::atomic<bool> tx_stop_{false};
        std::exception_ptr tx_error_;         // written by the TX thread before it sets tx_failed_
        std::atomic<bool> tx_failed_{false};
        std::size_t tx_pos_ = 0; // next sample of tx_buf_ to send

        void start_tx();
        void stop_tx();
        int send_tx(std::uint32_t timeout_ms);
        SampleClock clock_;
        std::uint64_t discontinuities_ = 0;
        std::string name_;
        std::optional<RxCorrection> correction_;
    };

} // namespace curf
