#include "burst_demo.hpp"
#include "dsss_probe.hpp"
#include "wifi_ltf_probe.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <future>
#include <map>
#include <tuple>

namespace curf {
    namespace {
        constexpr unsigned probe_frames = WifiLtfProbe::input_frames;
        static_assert(DsssProbe::input_frames == probe_frames);
        constexpr unsigned job_stride = probe_frames / 2;
        enum Label : unsigned { occupied, ofdm, dsss, gfsk, fm, am, other };
        enum Proof : unsigned { model, legacy_ltf, barker };
        struct Mark {
            unsigned first, last;
            double offset_hz, width_hz;
            unsigned label;
            float score;
            Proof proof;
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
            std::uint64_t ltf_matches = 0, barker_matches = 0, jobs_skipped = 0;

            Observations(StreamFormat f, double hz)
                : workers{Probes(f.sample_rate), Probes(f.sample_rate), Probes(f.sample_rate)},
                  fmt(f), tune(hz), ism(hz >= 2390e6 && hz <= 2490e6) {}

            std::vector<Mark> inspect(std::span<const BurstRegion> candidates, std::span<const std::uint8_t> wire) {
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
                constexpr std::size_t primary_budget = 96, total_budget = 128;
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
                            if (const auto at = worker.dsss_probe.barker_first(slice, fmt, 0, offset)) {
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

    std::string run_burst_demo(ISource &source, const engine::Model &net, const BurstConfig &cfg, std::FILE *feed,
                               std::FILE *trace, std::string_view source_name, unsigned rx, unsigned every) {
        const auto fmt = source.format();
        every = std::max(1u, every);
        Observations probe(fmt, cfg.tune_hz);
        std::array<float, burst_bins> peak{};
        const double step = fmt.sample_rate / burst_bins;
        std::array<std::uint64_t, burst_classes> counts{};
        struct TimedMark { Mark mark; std::uint64_t base; };
        std::vector<TimedMark> marks;
        std::uint64_t first_in_row = 0, last = 0;
        unsigned blocks = 0;
        peak.fill(-150);
        if (feed) std::fprintf(feed, "{\"meta\":{\"streams\":1,\"channels\":%u,\"rate\":%.0f,\"freq\":%.0f,"
            "\"window\":%u,\"every\":%u,\"source\":\"%.*s\",\"rx\":[%u],\"profile\":\"burst\","
            "\"classes\":[\"noise\",\"ofdm\",\"dsss\",\"gfsk\",\"fm\",\"am\",\"other_signal\"]}}\n",
            burst_bins, fmt.sample_rate, cfg.tune_hz, burst_block_frames * 2 / burst_bins, every,
            static_cast<int>(source_name.size()), source_name.data(), rx);
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
                std::fputs("]}\n", feed);
                std::fflush(feed);
            }
            blocks = 0; marks.clear(); peak.fill(-150);
        };
        float last_floor = -100;
        const auto stats = run_burst_classifier(source, net, cfg, [&](std::uint64_t first, std::span<const float> p,
                float floor, std::span<const BurstRegion> regions, std::span<const std::uint8_t> wire) {
            if (blocks && first != first_in_row + blocks * burst_block_frames) flush(last_floor);
            if (!blocks) first_in_row = first;
            ++blocks; last = first; last_floor = floor;
            for (unsigned k = 0; k < burst_bins; ++k) peak[k] = std::max(peak[k], p[k]);
            const auto confirmed = probe.inspect(regions, wire);
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
                const auto cls = choose_label(c, confirmed, probe.ism);
                add({c.first, c.last, c.offset_hz, c.bandwidth_hz, cls,
                     *std::max_element(c.probabilities.begin(), c.probabilities.end()), model}, &c);
            }
            for (const auto &x : confirmed) add(x, nullptr);
            if (blocks >= every) flush(floor);
        });
        flush(last_floor);
        char result[768];
        std::string histogram;
        for (const auto c : counts) histogram += (histogram.empty() ? "" : ",") + std::to_string(c);
        std::snprintf(result, sizeof result, "{\"profile\":\"burst\",\"frames\":%llu,\"blocks\":%llu,"
            "\"regions\":%llu,\"skipped\":%llu,\"dropped_frames\":%llu,\"gap_frames\":%llu,\"wall_s\":%.3f,"
            "\"processing_p99_ms\":%.3f,\"queue_p99_ms\":%.3f,\"wifi_ltf\":%llu,\"barker11\":%llu,"
            "\"evidence_skipped\":%llu,\"classes\":[%s]}",
            static_cast<unsigned long long>(stats.frames), static_cast<unsigned long long>(stats.blocks),
            static_cast<unsigned long long>(stats.regions), static_cast<unsigned long long>(stats.skipped),
            static_cast<unsigned long long>(stats.dropped), static_cast<unsigned long long>(stats.gaps), stats.wall_s,
            stats.processing_p99_ms, stats.queue_p99_ms, static_cast<unsigned long long>(probe.ltf_matches),
            static_cast<unsigned long long>(probe.barker_matches), static_cast<unsigned long long>(probe.jobs_skipped),
            histogram.c_str());
        if (feed) std::fprintf(feed, "{\"end\":%s}\n", result);
        return result;
    }
}
