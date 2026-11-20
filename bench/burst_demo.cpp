#include "burst_demo.hpp"
#include "curf/omnicore_source.hpp"
#include "dsss_probe.hpp"
#include "fm_broadcast_probe.hpp"
#include "wifi_ltf_probe.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <future>
#include <map>
#include <string_view>
#include <tuple>

namespace curf {
    namespace {
        constexpr unsigned probe_frames = WifiLtfProbe::input_frames;
        static_assert(DsssProbe::input_frames == probe_frames);
        constexpr unsigned job_stride = probe_frames / 2;
        enum Label : unsigned { occupied, ofdm, dsss, gfsk, fm, am, other };
        enum Proof : unsigned { model, legacy_ltf, barker, fm_pilot };
        struct Mark {
            unsigned first, last;
            double offset_hz, width_hz;
            unsigned label;
            float score;
            Proof proof;
        };
        struct TimedMark { Mark mark; std::uint64_t base; };

        // Preserve short-window RF power for named footprints throughout their display lifetime.
        // Prefix sums let overlapping bands share the same work and exclude invalid FFT bins.
        class BandPower {
        public:
            static constexpr int gate_db = 6;

            void append(BurstPower power) {
                append_prefix(power.floor, power.valid, noise_);
                append_prefix(power.windows, power.valid, power_);
            }

            void write(std::FILE *out, std::uint64_t start, std::uint64_t end, double rate,
                       std::span<const TimedMark> marks) {
                if (start != last_end_) bands_.clear();
                last_end_ = end;
                const auto hold_frames = static_cast<std::uint64_t>(rate);
                std::erase_if(bands_, [&](const Band &b) { return end - b.last > hold_frames; });
                const double step = rate / burst_bins;
                for (const auto &[mark, base] : marks) {
                    if (!mark.label || (mark.label == gfsk && (mark.width_hz > 2e6 || mark.score < .75f))) continue;
                    const auto bin = [&](double hz) {
                        return std::clamp(static_cast<int>(std::ceil(hz / step + burst_bins / 2)), 0, int(burst_bins));
                    };
                    const int lo = bin(mark.offset_hz - mark.width_hz / 2);
                    const int hi = bin(mark.offset_hz + mark.width_hz / 2);
                    if (lo >= hi) continue;
                    const Band b{lo, hi, end, mark.proof != model, mark.score};
                    auto match = std::find_if(bands_.begin(), bands_.end(), [&](const Band &old) {
                        const int width = std::min(old.hi - old.lo, hi - lo);
                        return lo < old.hi && old.lo < hi &&
                            std::abs((old.lo + old.hi) - (lo + hi)) <= 2 * std::max(4., width * .18) &&
                            std::max(old.hi - old.lo, hi - lo) <= 2 * width;
                    });
                    if (match == bands_.end()) bands_.push_back(b);
                    else if (b.confirmed && !match->confirmed) *match = b;
                    else {
                        match->last = end;
                        match->score = std::max(match->score, b.score);
                    }
                }
                std::stable_sort(bands_.begin(), bands_.end(), [](const Band &a, const Band &b) {
                    return std::tie(a.confirmed, a.last, a.score) > std::tie(b.confirmed, b.last, b.score);
                });
                constexpr std::size_t max_bands = 24;
                if (bands_.size() > max_bands) bands_.resize(max_bands);

                std::fprintf(out, ",\"power_start\":%.8f,\"band_power\":[", static_cast<double>(start) / rate);
                bool comma = false;
                for (const auto &band : bands_) {
                    std::fprintf(out, "%s[%d,%d,\"", comma ? "," : "", band.lo, band.hi);
                    for (std::size_t offset = 0; offset < power_.size(); offset += prefix_width) {
                        const std::size_t block = offset / (prefix_width * windows_per_block);
                        const auto p = band_sum(power_, offset, band);
                        const auto floor = band_sum(noise_, block * prefix_width, band);
                        const double db = 10 * std::log10(std::max(p, 1e-15) / std::max(floor, 1e-15));
                        const auto code = static_cast<std::size_t>(std::clamp(std::floor(db) + 8, 0., 63.));
                        std::fputc(codes[code], out);
                    }
                    std::fputs("\"]", out);
                    comma = true;
                }
                std::fputc(']', out);
                power_.clear();
                noise_.clear();
            }

        private:
            struct Band { int lo, hi; std::uint64_t last; bool confirmed; float score; };
            static constexpr std::size_t prefix_width = burst_bins + 1;
            static constexpr unsigned windows_per_block = burst_block_frames / burst_power_frames;
            static constexpr std::string_view codes = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_";
            std::vector<double> power_, noise_;
            std::vector<Band> bands_;
            std::uint64_t last_end_ = 0;

            static void append_prefix(std::span<const float> bins, std::span<const bool> valid, std::vector<double> &out) {
                const auto begin = out.size();
                out.resize(begin + bins.size() / burst_bins * prefix_width);
                for (std::size_t row = 0; row < bins.size() / burst_bins; ++row) {
                    auto *prefix = out.data() + begin + row * prefix_width;
                    prefix[0] = 0;
                    for (unsigned k = 0; k < burst_bins; ++k)
                        prefix[k + 1] = prefix[k] + (valid[k] ? bins[row * burst_bins + k] : 0.f);
                }
            }

            static double band_sum(const std::vector<double> &prefix, std::size_t start, const Band &b) {
                return prefix[start + b.hi] - prefix[start + b.lo];
            }
        };

        struct Observations {
            struct Probes {
                WifiLtfProbe ltf;
                DsssProbe dsss_probe;
                explicit Probes(double rate) : ltf(rate), dsss_probe(rate) {}
            };
            std::array<Probes, 3> workers;
            const StreamFormat fmt;
            const double tune;
            const bool ism;
            std::uint64_t ltf_matches = 0, barker_matches = 0, jobs_skipped = 0, throttled_blocks = 0;

            Observations(StreamFormat f, double hz)
                : workers{Probes(f.sample_rate), Probes(f.sample_rate), Probes(f.sample_rate)},
                  fmt(f), tune(hz), ism(hz >= 2390e6 && hz <= 2490e6) {}

            std::vector<Mark> inspect(std::span<const BurstRegion> candidates, std::span<const std::uint8_t> wire,
                                      double queue_ms) {
                std::vector<Mark> evidence;
                if (!ism && !(tune >= 5100e6 && tune <= 5850e6)) return evidence;
                // Candidate regions choose a small set of channel / time pairs.
                // Overlapping search spans cover a preamble near a row boundary.
                struct Job { unsigned start, mhz; float priority; bool primary, inspect_barker; };
                std::map<std::pair<unsigned, unsigned>, Job> pending;
                for (const auto &c : candidates) {
                    const bool partial_ofdm = c.bandwidth_hz >= 4e6f && c.probabilities[ofdm] >= .55f;
                    if (c.bandwidth_hz < 6e6f && !partial_ofdm) continue;
                    const double anchor = ism ? 2412e6 : 5000e6;
                    const auto near = std::lround((tune + c.offset_hz - anchor) / 5e6);
                    for (long k = near - 1; k <= near + 1; ++k) {
                        if (ism && (k < 0 || k >= 13)) continue;
                        const auto center = anchor + static_cast<double>(k) * 5e6;
                        const auto offset = center - tune;
                        const auto distance = std::abs(offset - c.offset_hz);
                        // A faded or masked upper half can leave just one side of
                        // a 20 MHz packet in the power detector. The sequence must
                        // still match at the adjacent channel's actual center.
                        const double search = std::clamp(.55 * c.bandwidth_hz, ism ? 7e6 : 0.0, 8e6);
                        if (std::abs(offset) + 10e6 > fmt.sample_rate / 2 ||
                            distance > search) continue;
                        const bool primary = c.bandwidth_hz >= 6e6f && distance <= std::min(8e6, .55 * c.bandwidth_hz);
                        for (auto at : {c.first, c.last - 1}) {
                            const int s = static_cast<int>((at / job_stride) * job_stride) - 1024;
                            const auto start = std::clamp(s, 0, static_cast<int>(burst_block_frames - probe_frames));
                            const auto s0 = static_cast<unsigned>(start), mhz = static_cast<unsigned>(center / 1e6);
                            const float priority = c.power_db - static_cast<float>(distance / 4e6);
                            const bool barker_needed = ism && c.bandwidth_hz >= 6e6f && distance <= .55 * c.bandwidth_hz &&
                                (k == near || c.probabilities[dsss] >= .30f);
                            auto [it, inserted] = pending.try_emplace({s0, mhz}, Job{s0, mhz, priority, primary, barker_needed});
                            if (!inserted) {
                                if (primary == it->second.primary) it->second.priority = std::max(priority, it->second.priority);
                                else if (primary) it->second.priority = priority;
                                it->second.primary |= primary;
                                it->second.inspect_barker |= barker_needed;
                            }
                        }
                    }
                }
                std::vector<Job> jobs;
                jobs.reserve(pending.size());
                for (const auto &x : pending) jobs.push_back(x.second);
                // Preserve the strongest primary jobs while the receiver catches up.
                const double block_ms = 1000. * burst_block_frames / fmt.sample_rate;
                const bool catching_up = queue_ms >= 2 * block_ms;
                const std::size_t primary_budget = catching_up ? (queue_ms >= 4 * block_ms ? 24 : 48) : 96;
                const std::size_t total_budget = catching_up ? primary_budget + 12 : 128;
                if (jobs.size() > primary_budget) {
                    const auto original = jobs.size();
                    std::stable_sort(jobs.begin(), jobs.end(), [](const Job &a, const Job &b) {
                        if (a.primary != b.primary) return a.primary;
                        return a.priority > b.priority;
                    });
                    const auto secondary = std::find_if(jobs.begin(), jobs.end(), [](const Job &j) { return !j.primary; });
                    if (secondary - jobs.begin() > static_cast<std::ptrdiff_t>(primary_budget))
                        jobs.erase(jobs.begin() + primary_budget, secondary);
                    if (jobs.size() > total_budget) jobs.resize(total_budget);
                    jobs_skipped += original - jobs.size();
                    if (catching_up && jobs.size() < original) ++throttled_blocks;
                }
                const auto frame_bytes = fmt.frame_bytes();
                const auto guard = static_cast<unsigned>(std::lround(fmt.sample_rate * 8e-6));
                const auto lanes = jobs.size() >= 12 ? static_cast<unsigned>(workers.size()) : 1u;
                const auto inspect_lane = [&](unsigned lane) {
                    std::vector<Mark> found;
                    auto &worker = workers[lane];
                    for (std::size_t index = lane; index < jobs.size(); index += lanes) {
                        const auto &job = jobs[index];
                        const auto start = job.start;
                        const double offset = job.mhz * 1e6 - tune;
                        const auto slice = wire.subspan(start * frame_bytes, probe_frames * frame_bytes);
                        for (const auto &m : worker.ltf.observe(slice, fmt, offset)) {
                            const unsigned t = start + m.sample;
                            found.push_back({t > guard ? t - guard : 0, std::min(burst_block_frames, t + guard),
                                             offset, 16.5e6, ofdm, m.score, legacy_ltf});
                        }
                        if (job.inspect_barker) {
                            if (const auto at = worker.dsss_probe.barker_first_mixed(worker.ltf.mixed_input())) {
                                const auto first = start + *at;
                                const auto length = static_cast<unsigned>(std::lround(32e-6 * fmt.sample_rate));
                                found.push_back({first, std::min(burst_block_frames, first + length),
                                                 offset, 17e6, dsss, 1, barker});
                            }
                        }
                    }
                    return found;
                };
                std::array<std::future<std::vector<Mark>>, 2> futures;
                for (unsigned i = 1; i < lanes; ++i) futures[i - 1] = std::async(std::launch::async, inspect_lane, i);
                evidence = inspect_lane(0);
                for (unsigned i = 1; i < lanes; ++i) {
                    auto marks = futures[i - 1].get();
                    evidence.insert(evidence.end(), marks.begin(), marks.end());
                }
                std::sort(evidence.begin(), evidence.end(), [](const Mark &a, const Mark &b) {
                    return std::tie(a.proof, a.offset_hz, a.first) < std::tie(b.proof, b.offset_hz, b.first);
                });
                std::vector<Mark> joined;
                for (const auto &hit : evidence) {
                    if (!joined.empty() && joined.back().proof == hit.proof && joined.back().offset_hz == hit.offset_hz &&
                        hit.first + guard < joined.back().last) {
                        if (hit.proof == barker) joined.back().last = std::max(joined.back().last, hit.last);
                        else if (hit.score > joined.back().score) joined.back() = hit;
                    } else joined.push_back(hit);
                }
                for (const auto &hit : joined) {
                    if (hit.proof == legacy_ltf) ++ltf_matches;
                    else ++barker_matches;
                }
                return joined;
            }
        };

        unsigned choose_label(const BurstRegion &c, std::span<const Mark> proofs, bool ism) {
            const auto best = std::max_element(c.probabilities.begin(), c.probabilities.end());
            const auto cls = static_cast<unsigned>(best - c.probabilities.begin());
            if (*best < .50f || cls == other || cls == fm || cls == am) return occupied;
            if (cls == gfsk) {
                if (c.bandwidth_hz > 5e6f || (!ism && *best < .70f)) return occupied;
                if (std::any_of(proofs.begin(), proofs.end(), [&](const Mark &p) {
                        return std::abs(p.offset_hz - c.offset_hz) < p.width_hz / 2 &&
                            p.first < c.last && c.first < p.last;
                    })) return occupied;
                return gfsk;
            }
            if (cls != ofdm && cls != dsss) return occupied;
            if (cls == dsss && !ism) return occupied;
            if (c.bandwidth_hz < 6e6f) return occupied;
            // A confirmed DSSS packet only vetoes overlapping model output.
            if (cls == ofdm && std::any_of(proofs.begin(), proofs.end(), [&](const Mark &p) {
                    return p.proof == barker && std::abs(p.offset_hz - c.offset_hz) < 3e6 &&
                        p.first < c.last && c.first < p.last;
                })) return occupied;
            return cls;
        }
    }

    void write_rx_correction(std::FILE *feed, const RxCorrection *correction) {
        if (correction)
            std::fprintf(feed, ",\"rx_correction\":{\"dc\":%s,\"iq\":%s}",
                         correction->dc_tracking ? "true" : "false", correction->iq_tracking ? "true" : "false");
    }

    std::string run_burst_demo(ISource &source, const engine::Model &net, const BurstConfig &cfg, std::FILE *feed,
                               std::FILE *trace, std::string_view source_name, unsigned rx, unsigned every,
                               const RxCorrection *correction) {
        const auto fmt = source.format();
        every = std::max(1u, every);
        Observations probe(fmt, cfg.tune_hz);
        FmBroadcastProbe fm_probe(fmt, cfg.tune_hz, cfg.band_hz);
        std::array<float, burst_bins> peak{};
        std::array<float, burst_bins> broadcast_floor{};
        std::array<bool, burst_bins> broadcast_valid{};
        const double step = fmt.sample_rate / burst_bins;
        if (fm_probe.enabled()) {
            for (unsigned k = 0; k < burst_bins; ++k) {
                const auto offset = (static_cast<int>(k) - static_cast<int>(burst_bins / 2)) * step;
                broadcast_valid[k] = k != burst_bins / 2 && std::abs(offset) <= cfg.band_hz / 2;
            }
        }
        std::array<std::uint64_t, burst_classes> counts{};
        std::vector<TimedMark> marks;
        BandPower band_power;
        std::uint64_t first_in_row = 0, last = 0;
        unsigned blocks = 0;
        peak.fill(-150);
        if (feed) {
            std::fprintf(feed, "{\"meta\":{\"streams\":1,\"channels\":%u,\"rate\":%.0f,\"freq\":%.0f,"
            "\"window\":%u,\"every\":%u,\"source\":\"%.*s\",\"rx\":[%u],\"profile\":\"burst\","
            "\"power_window\":%u,\"power_gate_db\":%d,"
            "\"classes\":[\"noise\",\"ofdm\",\"dsss\",\"gfsk\",\"fm\",\"am\",\"other_signal\"]",
            burst_bins, fmt.sample_rate, cfg.tune_hz, burst_block_frames * 2 / burst_bins, every,
            static_cast<int>(source_name.size()), source_name.data(), rx, burst_power_frames, BandPower::gate_db);
            if (fm_probe.enabled()) std::fputs(",\"fm_broadcast\":true", feed);
            write_rx_correction(feed, correction);
            std::fputs("}}\n", feed);
        }
        if (trace) std::fputs("start,end,freq_hz,bw_hz,label,proof,noise,ofdm,dsss,gfsk,fm,am,other_signal\n", trace);
        auto flush = [&](float floor) {
            if (feed && blocks) {
                std::fprintf(feed, "{\"w\":%llu,\"t\":%.6f,\"p\":[",
                             static_cast<unsigned long long>(last / burst_block_frames),
                             static_cast<double>(last) / fmt.sample_rate);
                for (unsigned k = 0; k < burst_bins; ++k) std::fprintf(feed, "%s%.0f", k ? "," : "", peak[k]);
                std::fprintf(feed, "],\"f\":[%.1f],\"r\":[", floor);
                std::array<const TimedMark *, burst_bins> chosen{};
                for (const auto &m : marks) {
                    const auto &a = m.mark;
                    const auto lo = std::clamp(static_cast<int>(std::ceil((a.offset_hz - a.width_hz / 2) / step +
                                                                         burst_bins / 2)), 0, static_cast<int>(burst_bins));
                    const auto hi = std::clamp(static_cast<int>(std::ceil((a.offset_hz + a.width_hz / 2) / step +
                                                                         burst_bins / 2)), 0, static_cast<int>(burst_bins));
                    for (int k = lo; k < hi; ++k) {
                        const auto old = chosen[k];
                        if (!old || (!old->mark.label && a.label) || (a.proof != model && old->mark.proof == model))
                            chosen[k] = &m;
                    }
                }
                bool comma = false;
                for (unsigned k = 0; k < burst_bins; ++k) {
                    if (!chosen[k]) continue;
                    const auto &a = chosen[k]->mark;
                    std::fprintf(feed, "%s[0,%u,%u,%d,1]", comma ? "," : "", (k + burst_bins / 2) % burst_bins,
                                 a.label, static_cast<int>(std::lround(100 * a.score)));
                    comma = true;
                }
                std::fputs("],\"bursts\":[", feed);
                comma = false;
                for (const auto &m : marks) {
                    const auto &a = m.mark;
                    std::fprintf(feed, "%s[%.8f,%.8f,%.0f,%.0f,%u,%.3f,%u]", comma ? "," : "",
                                 static_cast<double>(m.base + a.first) / fmt.sample_rate,
                                 static_cast<double>(m.base + a.last) / fmt.sample_rate, cfg.tune_hz + a.offset_hz,
                                 a.width_hz, a.label, a.score, a.proof);
                    comma = true;
                }
                std::fputc(']', feed);
                band_power.write(feed, first_in_row, last + burst_block_frames, fmt.sample_rate, marks);
                std::fputs("}\n", feed);
                std::fflush(feed);
            }
            blocks = 0; marks.clear(); peak.fill(-150);
        };
        float last_floor = -100;
        const auto stats = run_burst_classifier(source, net, cfg, [&](std::uint64_t first, std::span<const float> p,
                float floor, std::span<const BurstRegion> regions, std::span<const std::uint8_t> wire,
                BurstPower power, double queue_ms) {
            if (blocks && first != first_in_row + blocks * burst_block_frames) flush(last_floor);
            if (!blocks) first_in_row = first;
            ++blocks; last = first; last_floor = floor;
            const auto fm_hits = fm_probe.observe(first, power, wire);
            if (feed && fm_probe.enabled()) {
                // Compare continuous broadcasts to neighboring spectrum noise,
                // rather than a per-bin estimate that eventually follows the carrier.
                broadcast_floor.fill(fm_probe.noise());
                band_power.append({power.windows, broadcast_floor, broadcast_valid});
            } else if (feed) band_power.append(power);
            for (unsigned k = 0; k < burst_bins; ++k) peak[k] = std::max(peak[k], p[k]);
            const auto confirmed = probe.inspect(regions, wire, cfg.drop_when_full ? queue_ms : 0.);
            const auto add = [&](const Mark &mark, const BurstRegion *result) {
                ++counts[mark.label];
                if (feed) marks.push_back({mark, first});
                if (trace) {
                    std::fprintf(trace, "%.8f,%.8f,%.0f,%.0f,%u,%u",
                                 static_cast<double>(first + mark.first) / fmt.sample_rate,
                                 static_cast<double>(first + mark.last) / fmt.sample_rate,
                                 cfg.tune_hz + mark.offset_hz, mark.width_hz, mark.label, mark.proof);
                    for (unsigned k = 0; k < burst_classes; ++k)
                        std::fprintf(trace, ",%.4f", result ? result->probabilities[k] : 0.f);
                    std::fputc('\n', trace);
                }
            };
            for (const auto &c : regions) {
                const double hz = cfg.tune_hz + c.offset_hz;
                const auto cls = fm_probe.enabled() && hz >= fm_band_low_hz && hz <= fm_band_high_hz ?
                    occupied : choose_label(c, confirmed, probe.ism);
                add({c.first, c.last, c.offset_hz, c.bandwidth_hz, cls,
                     *std::max_element(c.probabilities.begin(), c.probabilities.end()), model}, &c);
            }
            for (const auto &x : confirmed) add(x, nullptr);
            for (const auto &x : fm_hits)
                add({x.first, x.last, x.offset_hz, FmBroadcastProbe::bandwidth_hz, fm, 1.f, fm_pilot}, nullptr);
            if (blocks >= every) flush(floor);
        });
        flush(last_floor);
        char result[768];
        std::string histogram;
        for (const auto c : counts) histogram += (histogram.empty() ? "" : ",") + std::to_string(c);
        std::snprintf(result, sizeof result, "{\"profile\":\"burst\",\"frames\":%llu,\"blocks\":%llu,"
            "\"regions\":%llu,\"skipped\":%llu,\"dropped_frames\":%llu,\"gap_frames\":%llu,\"wall_s\":%.3f,"
            "\"processing_p99_ms\":%.3f,\"queue_p99_ms\":%.3f,\"wifi_ltf\":%llu,\"barker11\":%llu,"
            "\"evidence_skipped\":%llu,\"evidence_throttled_blocks\":%llu,\"classes\":[%s]}",
            static_cast<unsigned long long>(stats.frames), static_cast<unsigned long long>(stats.blocks),
            static_cast<unsigned long long>(stats.regions), static_cast<unsigned long long>(stats.skipped),
            static_cast<unsigned long long>(stats.dropped), static_cast<unsigned long long>(stats.gaps), stats.wall_s,
            stats.processing_p99_ms, stats.queue_p99_ms, static_cast<unsigned long long>(probe.ltf_matches),
            static_cast<unsigned long long>(probe.barker_matches), static_cast<unsigned long long>(probe.jobs_skipped),
            static_cast<unsigned long long>(probe.throttled_blocks), histogram.c_str());
        if (feed) std::fprintf(feed, "{\"end\":%s}\n", result);
        return result;
    }
}
