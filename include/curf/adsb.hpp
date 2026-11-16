#pragma once
// Mode S / ADS-B (1090 MHz) message layer, host side: CRC-24, field decoding and CPR position decoding.
// The GPU demodulator (curf/cuda/adsb.hpp) hands over raw 56 / 112 bit frames that passed its parity checks;
// everything here is plain C++ and tested on the CPU.
//
// References: ICAO Annex 10 Vol. IV; RTCA DO-260B; J. Sun, "The 1090 Megahertz Riddle" (2nd ed.).

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace curf::adsb {

    constexpr std::uint32_t kGenerator = 0xFFF409; // Mode S parity polynomial (24 bits, x^24 implicit)

    // CRC-24 remainder of the first bits - 24 bits of the frame, XORed with its last 24 bits (the parity field).
    // 0 for a valid DF17/18; for DF4/5/20/21 (address/parity) the ICAO address of the sender; for DF11 the
    // interrogator code (<= 0x7F when valid).
    std::uint32_t parity_remainder(const std::uint8_t *msg, int bits);

    // Syndrome of a single flipped bit at position `bit` of a `bits`-long frame (for 1-bit error correction).
    std::uint32_t single_bit_syndrome(int bit, int bits);

    inline int downlink_format(const std::uint8_t *msg) { return msg[0] >> 3; }
    inline int frame_bits(int df) { return df >= 16 ? 112 : 56; }

    struct Message {
        int df = -1;
        std::uint32_t icao = 0;
        int tc = 0;                        // DF17/18 type code
        std::optional<std::string> callsign;
        std::optional<int> altitude_ft;
        std::optional<int> squawk;         // DF5/21, as the 4 octal digits read in decimal (e.g. 7700)
        std::optional<double> ground_speed_kt, track_deg;
        std::optional<int> vertical_rate_fpm;
        // airborne position (tc 9-18, 20-22): raw CPR fields, decoded by Tracker
        bool has_cpr = false, cpr_odd = false;
        std::uint32_t cpr_lat = 0, cpr_lon = 0; // 17 bits each
    };

    // Field decoding of a frame whose parity has already been checked. `icao` is the address for formats that
    // carry it in the parity (DF0/4/5/16/20/21: the remainder); for DF11/17/18 it is read from the frame.
    Message decode(const std::uint8_t *msg, int bits, std::uint32_t parity_icao = 0);

    // CPR, for tests and Tracker. lat/lon CPR fields are the raw 17-bit values.
    int cpr_nl(double lat);
    std::optional<std::array<double, 2>> cpr_global(std::uint32_t lat_even, std::uint32_t lon_even, std::uint32_t lat_odd,
                                                    std::uint32_t lon_odd, bool odd_is_newer);
    std::array<double, 2> cpr_local(std::uint32_t lat_cpr, std::uint32_t lon_cpr, bool odd, double lat_ref, double lon_ref);

    // Per-aircraft state for CPR decoding and for accepting address/parity formats: an address seen in a
    // DF11/17/18 frame (parity checked) within `ttl_s` makes DF0/4/5/16/20/21 frames from it acceptable.
    class Tracker {
    public:
        explicit Tracker(double ttl_s = 60) : ttl_(ttl_s) {}

        // Updates the state; returns the position of this message if it can be decoded (global pair within 10 s,
        // else local relative to a position no older than ttl).
        std::optional<std::array<double, 2>> position(const Message &m, double t);
        void saw(std::uint32_t icao, double t) { state_[icao].seen = t; }
        bool known(std::uint32_t icao, double t) const;
        std::size_t size() const { return state_.size(); }

    private:
        struct Cpr {
            std::uint32_t lat = 0, lon = 0;
            double t = -1e30;
        };
        struct State {
            Cpr even, odd;
            double seen = -1e30, pos_t = -1e30;
            double lat = 0, lon = 0;
        };
        double ttl_;
        std::unordered_map<std::uint32_t, State> state_;
    };

    // One JSON object (no newline) with the decoded fields that are present, for the display feed.
    std::string to_json(const Message &m, double t, float signal_db, const std::optional<std::array<double, 2>> &pos,
                        bool corrected);

} // namespace curf::adsb
