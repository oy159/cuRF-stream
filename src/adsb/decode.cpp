#include "curf/adsb.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>

namespace curf::adsb {

    namespace {

        // `len` (<= 32) bits starting at bit `start` (0 = MSB of byte 0)
        std::uint32_t bits_at(const std::uint8_t *msg, int start, int len) {
            std::uint32_t v = 0;
            for (int i = start; i < start + len; ++i) v = (v << 1) | ((msg[i >> 3] >> (7 - (i & 7))) & 1u);
            return v;
        }

        // 12-bit altitude of an airborne position (Q bit = 25 ft steps; Gillham-coded Q=0 is not decoded)
        std::optional<int> altitude12(std::uint32_t a) {
            if (a == 0 || !(a & 0x10)) return std::nullopt;
            const std::uint32_t n = ((a & 0xFE0) >> 1) | (a & 0xF);
            return static_cast<int>(n) * 25 - 1000;
        }

        // 13-bit altitude code of DF0/4/16/20 (M = 0 feet, Q = 1 25 ft steps)
        std::optional<int> altitude13(std::uint32_t ac) {
            if (ac == 0 || (ac & 0x40) || !(ac & 0x10)) return std::nullopt;
            const std::uint32_t n = ((ac >> 7) << 5) | (((ac >> 5) & 1u) << 4) | (ac & 0xF);
            return static_cast<int>(n) * 25 - 1000;
        }

        // 13-bit identity (Mode A code) of DF5/21: C1 A1 C2 A2 C4 A4 X B1 D1 B2 D2 B4 D4
        int squawk13(std::uint32_t id) {
            auto b = [&](int i) { return static_cast<int>((id >> (12 - i)) & 1u); };
            const int a = b(5) * 4 + b(3) * 2 + b(1), bb = b(11) * 4 + b(9) * 2 + b(7);
            const int c = b(4) * 4 + b(2) * 2 + b(0), d = b(12) * 4 + b(10) * 2 + b(8);
            return a * 1000 + bb * 100 + c * 10 + d;
        }

        double mod(double a, double b) { return a - b * std::floor(a / b); }

        constexpr double kCprScale = 131072.0; // 2^17
    } // namespace

    std::uint32_t parity_remainder(const std::uint8_t *msg, int bits) {
        std::uint32_t crc = 0;
        for (int i = 0; i < bits - 24; ++i) {
            const std::uint32_t top = ((crc >> 23) ^ (msg[i >> 3] >> (7 - (i & 7)))) & 1u;
            crc = (crc << 1) & 0xFFFFFFu;
            if (top) crc ^= kGenerator;
        }
        return crc ^ bits_at(msg, bits - 24, 24);
    }

    std::uint32_t single_bit_syndrome(int bit, int bits) {
        std::uint8_t m[14] = {};
        m[bit >> 3] = static_cast<std::uint8_t>(0x80u >> (bit & 7));
        return parity_remainder(m, bits);
    }

    Message decode(const std::uint8_t *msg, int bits, std::uint32_t parity_icao) {
        Message m;
        m.df = downlink_format(msg);
        switch (m.df) {
        case 0: case 16: case 4: case 20:
            m.icao = parity_icao;
            m.altitude_ft = altitude13(bits_at(msg, 19, 13));
            break;
        case 5: case 21:
            m.icao = parity_icao;
            m.squawk = squawk13(bits_at(msg, 19, 13));
            break;
        case 11:
            m.icao = bits_at(msg, 8, 24);
            break;
        case 17: case 18: {
            m.icao = bits_at(msg, 8, 24);
            if (bits < 112) break;
            const int me = 32;
            m.tc = static_cast<int>(bits_at(msg, me, 5));
            if (m.tc >= 1 && m.tc <= 4) {
                static constexpr char kChars[] = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######";
                std::string s;
                for (int i = 0; i < 8; ++i) s += kChars[bits_at(msg, me + 8 + 6 * i, 6)];
                while (!s.empty() && (s.back() == ' ' || s.back() == '#')) s.pop_back();
                m.callsign = s;
            } else if ((m.tc >= 9 && m.tc <= 18) || (m.tc >= 20 && m.tc <= 22)) {
                const std::uint32_t alt = bits_at(msg, me + 8, 12);
                if (m.tc <= 18) m.altitude_ft = altitude12(alt);
                else if (alt) m.altitude_ft = static_cast<int>(std::lround(alt * 3.28084)); // GNSS height, metres
                m.has_cpr = true;
                m.cpr_odd = bits_at(msg, me + 21, 1) != 0;
                m.cpr_lat = bits_at(msg, me + 22, 17);
                m.cpr_lon = bits_at(msg, me + 39, 17);
            } else if (m.tc == 19) {
                const std::uint32_t st = bits_at(msg, me + 5, 3);
                if (st == 1 || st == 2) {
                    const std::uint32_t vew = bits_at(msg, me + 14, 10), vns = bits_at(msg, me + 25, 10);
                    if (vew && vns) {
                        const double k = st == 2 ? 4 : 1;
                        const double ew = (bits_at(msg, me + 13, 1) ? -1.0 : 1.0) * (vew - 1) * k;
                        const double ns = (bits_at(msg, me + 24, 1) ? -1.0 : 1.0) * (vns - 1) * k;
                        m.ground_speed_kt = std::hypot(ew, ns);
                        m.track_deg = mod(std::atan2(ew, ns) * 180 / std::numbers::pi, 360);
                    }
                } else if (st == 3 || st == 4) {
                    if (bits_at(msg, me + 13, 1)) m.track_deg = bits_at(msg, me + 14, 10) * 360.0 / 1024; // heading
                    const std::uint32_t as = bits_at(msg, me + 25, 10);
                    if (as) m.ground_speed_kt = (as - 1) * (st == 4 ? 4.0 : 1.0); // airspeed, not ground speed
                }
                const std::uint32_t vr = bits_at(msg, me + 37, 9);
                if (vr) m.vertical_rate_fpm = (bits_at(msg, me + 36, 1) ? -1 : 1) * static_cast<int>(vr - 1) * 64;
            }
            break;
        }
        default:
            m.icao = parity_icao;
            break;
        }
        return m;
    }

    int cpr_nl(double lat) {
        lat = std::fabs(lat);
        if (lat < 1e-9) return 59;
        if (lat > 87) return lat > 87.0 + 1e-9 ? 1 : 2;
        constexpr double nz = 15;
        const double a = 1 - std::cos(std::numbers::pi / (2 * nz));
        const double c = std::cos(std::numbers::pi / 180 * lat);
        return static_cast<int>(std::floor(2 * std::numbers::pi / std::acos(1 - a / (c * c))));
    }

    std::optional<std::array<double, 2>> cpr_global(std::uint32_t lat_even, std::uint32_t lon_even, std::uint32_t lat_odd,
                                                    std::uint32_t lon_odd, bool odd_is_newer) {
        const double le = lat_even / kCprScale, lo = lat_odd / kCprScale;
        const double ge = lon_even / kCprScale, go = lon_odd / kCprScale;
        const double j = std::floor(59 * le - 60 * lo + 0.5);
        double lat_e = 360.0 / 60 * (mod(j, 60) + le), lat_o = 360.0 / 59 * (mod(j, 59) + lo);
        if (lat_e >= 270) lat_e -= 360;
        if (lat_o >= 270) lat_o -= 360;
        if (cpr_nl(lat_e) != cpr_nl(lat_o)) return std::nullopt; // the pair straddles a longitude zone boundary
        const double lat = odd_is_newer ? lat_o : lat_e;
        const int nl = cpr_nl(lat);
        const double mm = std::floor(ge * (nl - 1) - go * nl + 0.5);
        const int ni = std::max(odd_is_newer ? nl - 1 : nl, 1);
        double lon = 360.0 / ni * (mod(mm, ni) + (odd_is_newer ? go : ge));
        if (lon >= 180) lon -= 360;
        return std::array<double, 2>{lat, lon};
    }

    std::array<double, 2> cpr_local(std::uint32_t lat_cpr, std::uint32_t lon_cpr, bool odd, double lat_ref, double lon_ref) {
        const double yz = lat_cpr / kCprScale, xz = lon_cpr / kCprScale;
        const double dlat = 360.0 / (odd ? 59 : 60);
        const double j = std::floor(lat_ref / dlat) + std::floor(mod(lat_ref, dlat) / dlat - yz + 0.5);
        const double lat = dlat * (j + yz);
        const double dlon = 360.0 / std::max(cpr_nl(lat) - (odd ? 1 : 0), 1);
        const double m = std::floor(lon_ref / dlon) + std::floor(mod(lon_ref, dlon) / dlon - xz + 0.5);
        return {lat, dlon * (m + xz)};
    }

    bool Tracker::known(std::uint32_t icao, double t) const {
        const auto it = state_.find(icao);
        return it != state_.end() && t - it->second.seen <= ttl_;
    }

    std::optional<std::array<double, 2>> Tracker::position(const Message &m, double t) {
        State &s = state_[m.icao];
        s.seen = t;
        if (!m.has_cpr) return std::nullopt;
        (m.cpr_odd ? s.odd : s.even) = Cpr{m.cpr_lat, m.cpr_lon, t};
        std::optional<std::array<double, 2>> p;
        if (t - s.pos_t <= ttl_) {
            p = cpr_local(m.cpr_lat, m.cpr_lon, m.cpr_odd, s.lat, s.lon);
        } else if (std::fabs(s.even.t - s.odd.t) <= 10) {
            p = cpr_global(s.even.lat, s.even.lon, s.odd.lat, s.odd.lon, m.cpr_odd);
        }
        if (p) {
            s.lat = (*p)[0];
            s.lon = (*p)[1];
            s.pos_t = t;
        }
        return p;
    }

    std::string to_json(const Message &m, double t, float signal_db, const std::optional<std::array<double, 2>> &pos,
                        bool corrected) {
        char b[128];
        std::snprintf(b, sizeof b, "{\"t\":%.6f,\"df\":%d,\"icao\":\"%06X\",\"sig\":%.1f", t, m.df, m.icao,
                      static_cast<double>(signal_db));
        std::string s = b;
        if (m.tc) s += ",\"tc\":" + std::to_string(m.tc);
        if (corrected) s += ",\"fix\":1";
        if (m.callsign) s += ",\"call\":\"" + *m.callsign + "\"";
        if (m.altitude_ft) s += ",\"alt\":" + std::to_string(*m.altitude_ft);
        if (m.squawk) {
            std::snprintf(b, sizeof b, ",\"sq\":\"%04d\"", *m.squawk);
            s += b;
        }
        if (pos) {
            std::snprintf(b, sizeof b, ",\"lat\":%.5f,\"lon\":%.5f", (*pos)[0], (*pos)[1]);
            s += b;
        }
        if (m.ground_speed_kt) {
            std::snprintf(b, sizeof b, ",\"gs\":%.1f", *m.ground_speed_kt);
            s += b;
        }
        if (m.track_deg) {
            std::snprintf(b, sizeof b, ",\"trk\":%.1f", *m.track_deg);
            s += b;
        }
        if (m.vertical_rate_fpm) s += ",\"vr\":" + std::to_string(*m.vertical_rate_fpm);
        return s + "}";
    }

} // namespace curf::adsb
