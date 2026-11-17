#!/usr/bin/env python3
"""Check static-carrier identities and real-time counters on a controlled scene replay.

This checks track continuity, completeness and processing latency; it does not score modulation labels.
Use model/make_scene.py score for time-aware classification scoring of a single (unlooped) scene.
"""
import argparse
import json
from pathlib import Path
import statistics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--feed", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--min-seconds", type=float, default=0)
    ap.add_argument("--p99-ms", type=float, default=15)
    a = ap.parse_args()
    truth = json.loads(a.truth.read_text())["truth"]
    source = {}
    for e in truth:
        source.setdefault(e["source_id"], []).append(e)
    carriers = {k: (statistics.median(e["fc"] for e in es), statistics.median(e["bw"] for e in es))
                for k, es in source.items()}
    identities, seen, migrations, unmatched, geometry_errors = {}, set(), 0, 0, 0
    streams = 0
    with a.feed.open() as f:
        for line in f:
            row = json.loads(line)
            if "meta" in row:
                streams = row["meta"]["streams"]
            if "sig" not in row:
                continue
            r = row["sig"]
            matched = [(abs(r["f"] - fc), k) for k, (fc, bw) in carriers.items()
                       if abs(r["f"] - fc) <= max(5000, bw / 2)]
            key = (r["s"], r["id"])
            if not matched:
                unmatched += 1
                if key in identities:
                    migrations += 1
                continue
            _, carrier = min(matched)
            if key in identities and identities[key] != carrier:
                migrations += 1
            fc, bw = carriers[carrier]
            if abs(r["f"]-fc) > max(5000,.2*bw) or not (.25*bw <= r["bw"] <= 4*bw):
                geometry_errors += 1
            identities[key] = carrier
            seen.add((r["s"], carrier))
    stats = json.loads(a.stats.read_text())
    s2 = stats["stage2"]
    checks = {"zero_dropped_samples": stats["dropped_frames"] == 0,
              "zero_sample_gaps": stats["gap_frames"] == 0,
              "no_stage2_skips": all(s2[k] == 0 for k in ("skipped_busy", "skipped_ring", "skipped_capacity")),
              "all_carriers_reported": streams > 0 and len(seen) == streams * len(carriers),
              "stable_carrier_identities": migrations == 0,
              "stable_carrier_geometry": geometry_errors == 0,
              "both_width_classes": s2["wide"] > 0 and s2["narrow"] > 0,
              "wide_p99": s2["wide_latency_ms"]["p99"] < a.p99_ms,
              "narrow_p99": s2["narrow_latency_ms"]["p99"] < a.p99_ms,
              "duration": stats["wall_s"] >= a.min_seconds}
    print(json.dumps({"passed": all(checks.values()), "checks": checks, "reported_carriers": len(seen),
                      "expected_carriers": streams * len(carriers), "track_migrations": migrations, "geometry_errors": geometry_errors,
                      "unmatched_records": unmatched, "stage2": s2}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
