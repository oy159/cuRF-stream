#include "curf/omnicore_source.hpp"

#include <omnicore/och.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace curf {

    namespace {
        void check(int rc, const char *what) {
            if (rc == OCH_OK) return;
            const char *e = och_last_error();
            throw std::runtime_error(std::string("OmniCore ") + what + " failed (" + std::to_string(rc) +
                                     "): " + (e ? e : ""));
        }
    } // namespace

    OmniCoreSource::OmniCoreSource(OmniCoreConfig config) : config_(std::move(config)) {
        if (config_.rx_mask) {
            if (config_.rx_mask > 3) throw std::invalid_argument("OmniCoreSource: rx_mask 1, 2 or 3");
            config_.channels = config_.rx_mask == 3 ? 2 : 1;
        }
        if (config_.channels != 1 && config_.channels != 2) throw std::invalid_argument("OmniCoreSource: 1 or 2 channels");
        if (config_.iq_bits != 12 && config_.iq_bits != 8) throw std::invalid_argument("OmniCoreSource: iq_bits 12 or 8");
        const bool tx = !config_.tx_file.empty();
        if (tx) {
            // checked before the device is even opened
            if (!(config_.tx_attenuation_db >= OmniCoreConfig::kMinTxAttenuationDb && config_.tx_attenuation_db <= 89.75))
                throw std::invalid_argument("OmniCoreSource: TX attenuation must be within [" +
                                            std::to_string(OmniCoreConfig::kMinTxAttenuationDb) + ", 89.75] dB");
            std::ifstream f(config_.tx_file, std::ios::binary | std::ios::ate);
            if (!f) throw std::runtime_error("OmniCoreSource: cannot open " + config_.tx_file);
            const auto bytes = static_cast<std::size_t>(f.tellg());
            if (bytes < 4 * 65536 || bytes % 4) throw std::runtime_error("OmniCoreSource: TX file must hold >= 64 Ki sc16 samples");
            tx_buf_.resize(bytes / 2);
            f.seekg(0);
            f.read(reinterpret_cast<char *>(tx_buf_.data()), static_cast<std::streamsize>(bytes));
        }

        check(och_open(config_.serial.empty() ? nullptr : config_.serial.c_str(), &dev_), "open");
        try {
            och_device_info info{};
            info.struct_size = sizeof info;
            check(och_get_info(dev_, &info), "get_info");
            name_ = std::string(info.name) + " " + info.serial + " (" + info.firmware + ", USB " +
                    std::to_string(info.usb_speed_mbps) + " Mb/s)";

            och_config c{};
            och_default_config(&c);
            c.sample_rate = static_cast<std::uint32_t>(std::llround(config_.sample_rate));
            c.rx_mask = config_.rx_mask ? config_.rx_mask : config_.channels == 2 ? 3u : 1u;
            c.rx_frequency = config_.frequency_hz;
            // AD9361 analog bandwidth: 200 kHz..56 MHz; default as wide as the sample rate allows
            c.rx_bandwidth = config_.bandwidth_hz ? config_.bandwidth_hz : std::min<std::uint32_t>(c.sample_rate, 56'000'000);
            c.rx_gain[0] = config_.gain_db[0];
            c.rx_gain[1] = config_.gain_db[1];
            c.rx_agc = config_.agc;
            c.usb_iq_bits = config_.iq_bits;
            // TX: never rely on the driver default (10 dB) - both channels at maximum attenuation unless the
            // loopback transmitter asks for a (bounded) lower value on TX1.
            c.tx_attenuation[0] = tx ? config_.tx_attenuation_db : 89.75;
            c.tx_attenuation[1] = 89.75;
            c.tx_frequency = config_.frequency_hz;
            c.tx_bandwidth = c.rx_bandwidth;
            if (tx) c.tx_mask = 1;
            if (config_.iq_bits == 8) c.tx_mask = 3;
            check(och_configure(dev_, &c), "configure");
            och_config applied{};
            applied.struct_size = sizeof applied;
            check(och_get_config(dev_, &applied), "get_config");

            // The AD9361 RX FIR lives in the board's RAM (loaded with `och fir rx load`, lost at power-off):
            // record whether this run had it, since it changes the band edges.
            och_fir_config fir{};
            fir.struct_size = sizeof fir;
            if (och_get_fir(dev_, OCH_RX, &fir) == OCH_OK)
                name_ += fir.enabled ? ", RX FIR on (dec " + std::to_string(fir.factor) + ", " + std::to_string(fir.tap_count) + " taps)"
                                     : ", RX FIR off";

            name_ += c.rx_mask == 3 ? ", RX1+RX2" : c.rx_mask == 2 ? ", RX2" : ", RX1";
            format_.wire = config_.iq_bits == 8 ? WireFormat::signed8 : WireFormat::packed12;
            format_.channels = config_.channels;
            format_.sample_rate = applied.sample_rate; // the rate the clock chain actually runs at

            och_stream_args a{};
            och_default_stream_args(&a, OCH_RX);
            a.format = config_.iq_bits == 8 ? OCH_SC8 : OCH_PACKED12; // the wire bytes, copied unchanged
            a.host_buffer_bytes = config_.host_buffer_mib << 20;
            check(och_create_stream(dev_, &a, &rx_), "create_stream");
            if (tx) {
                och_stream_args t{};
                och_default_stream_args(&t, OCH_TX);
                t.format = OCH_SC16;
                check(och_create_stream(dev_, &t, &tx_), "create_stream (TX)");
            }
        } catch (...) {
            if (tx_) och_destroy_stream(tx_);
            if (rx_) och_destroy_stream(rx_);
            och_close(dev_);
            throw;
        }
    }

    OmniCoreSource::~OmniCoreSource() {
        stop_tx();
        if (running_) och_stop(rx_, 1000);
        if (tx_) och_destroy_stream(tx_);
        if (rx_) och_destroy_stream(rx_);
        if (dev_) och_close(dev_);
    }

    void OmniCoreSource::start() {
        if (tx_) start_tx();
        check(och_start(rx_), "start");
        running_ = true;
    }

    void OmniCoreSource::stop() {
        stop_tx();
        if (!running_) return;
        running_ = false;
        och_stop(rx_, 1000);
    }

    // One och_send of the next part of tx_buf_ (wrapping around). OCH_TIMEOUT is not an error: the queue is full.
    int OmniCoreSource::send_tx(std::uint32_t timeout_ms) {
        const std::size_t n = tx_buf_.size() / 2;
        std::size_t accepted = 0;
        const std::size_t count = std::min<std::size_t>(n - tx_pos_, std::size_t{1} << 16);
        const int rc = och_send(tx_, tx_buf_.data() + 2 * tx_pos_, count, &accepted, timeout_ms);
        if (rc != OCH_OK && rc != OCH_TIMEOUT) return rc;
        tx_pos_ = (tx_pos_ + accepted) % n;
        return OCH_OK;
    }

    // Sends tx_buf_ in a loop: queue some samples first (the driver starts TX from its queue), then start.
    void OmniCoreSource::start_tx() {
        tx_pos_ = 0;
        for (int i = 0; i < 16; ++i) check(send_tx(100), "send (TX prefill)");
        check(och_start(tx_), "start (TX)");
        tx_running_ = true;
        tx_stop_ = false;
        tx_thread_ = std::thread([this] {
            try {
                while (!tx_stop_.load(std::memory_order_relaxed)) {
                    const int rc = send_tx(100);
                    if (rc == OCH_STOPPED) break;
                    check(rc, "send (TX)");
                }
            } catch (...) {
                tx_error_ = std::current_exception();
                tx_failed_.store(true, std::memory_order_release);
            }
        });
    }

    void OmniCoreSource::stop_tx() {
        if (!tx_running_) return;
        tx_stop_ = true;
        if (tx_thread_.joinable()) tx_thread_.join();
        och_stop(tx_, 1000);
        tx_running_ = false;
    }

    ReadResult OmniCoreSource::read(std::span<std::uint8_t> dst, std::uint32_t timeout_ms) {
        if (tx_failed_.load(std::memory_order_acquire)) { // the loopback transmitter died: stop the run
            tx_failed_ = false;
            std::rethrow_exception(tx_error_);
        }
        const std::size_t want = dst.size() / format_.frame_bytes();
        if (want == 0) return {};
        och_metadata m{};
        m.struct_size = sizeof m;
        std::size_t got = 0;
        const int rc = och_recv(rx_, dst.data(), want, &got, &m, timeout_ms);
        if (rc == OCH_TIMEOUT) return {};
        if (rc == OCH_STOPPED) return {ReadStatus::end_of_stream, 0, clock_.next(), false};
        check(rc, "recv");
        if (got == 0) return {};

        ReadMeta rm;
        rm.has_time = (m.flags & OCH_META_HAS_TIME) != 0;
        rm.time = m.time_ticks;
        rm.dropped = m.host_dropped_samples + (m.device_dropped_samples > 0 ? static_cast<std::uint64_t>(m.device_dropped_samples) : 0);
        // START opens every epoch, including the first; the clock treats the first read as the origin.
        rm.gap_flagged = (m.flags & (OCH_META_DISCONTINUITY | OCH_META_HOST_OVERFLOW | OCH_META_DEVICE_OVERFLOW |
                                     OCH_META_RECONFIGURED)) != 0;
        const ReadPlacement p = clock_.place(rm, got); // the first read is the origin, never a discontinuity
        if (p.discontinuity) ++discontinuities_;
        return {ReadStatus::ok, got, p.first_sample, p.discontinuity};
    }

    OmniCoreStats OmniCoreSource::stats() const {
        och_stats s{};
        s.struct_size = sizeof s;
        check(och_get_stats(rx_, &s), "get_stats");
        OmniCoreStats o;
        o.usb_bytes = s.usb_bytes;
        o.samples = s.samples;
        o.host_dropped = s.host_dropped_samples;
        o.device_overflows = s.device_overflows;
        o.transfer_errors = s.transfer_errors;
        o.resyncs = s.resyncs;
        o.host_queue_peak_bytes = s.host_queue_peak_bytes;
        o.discontinuities = discontinuities_;
        if (tx_) {
            och_stats t{};
            t.struct_size = sizeof t;
            check(och_get_stats(tx_, &t), "get_stats (TX)");
            o.tx_samples = t.samples;
            o.tx_underflows = t.device_underflows;
            o.tx_starvation = t.tx_starvation_events;
        }
        return o;
    }

} // namespace curf
