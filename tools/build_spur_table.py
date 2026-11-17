#!/usr/bin/env python3
"""Build a receiver-scoped spur table from a deliberately terminated, receive-only recording.

    python tools/build_spur_table.py --feed terminated.feed --terminated --out spurs.json

--terminated declares that the recording was made with a real 50 ohm load. This
tool cannot verify the physical connection. Never use an antenna or RX-to-TX cable
capture: persistent over-the-air transmitters would become false spur entries.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from signal_policy import calibration_context, finite


def build(feed, min_seconds=10, occupancy=.98, threshold_db=10):
    meta, end, first, last = None, None, None, None
    hits, rows = Counter(), 0
    with Path(feed).open() as f:
        for line in f:
            r = json.loads(line)
            if "meta" in r:
                if meta:
                    raise ValueError("calibration must contain one run")
                meta = r["meta"]
                if meta.get("source") != "omnicore" or not meta.get("receiver", {}).get("device"):
                    raise ValueError("calibration needs hardware identity and receiver settings")
                if meta["receiver"].get("agc") != 0:
                    raise ValueError("calibration requires fixed manual gain")
            elif "end" in r:
                end = r["end"]
            elif "p" in r:
                if not meta:
                    raise ValueError("missing metadata")
                m, streams = meta["channels"], meta["streams"]
                if len(r["p"]) != m*streams or len(r["f"]) != streams or not finite(r.get("t")):
                    raise ValueError("invalid power row")
                first = r["t"] if first is None else first
                if last is not None and r["t"] <= last:
                    raise ValueError("calibration time is not increasing")
                last = r["t"]
                rows += 1
                for s in range(streams):
                    for x in range(m):
                        k = (x + m//2) % m
                        if r["p"][s*m+k] >= r["f"][s] + threshold_db:
                            hits[s, x] += 1
    if not end or any(end.get(k, -1) != 0 for k in ("dropped_frames", "gap_frames")):
        raise ValueError("calibration must finish with zero loss and zero gaps")
    if rows < 64 or first is None or last-first < min_seconds:
        raise ValueError("calibration is too short")
    entries = []
    for s in range(meta["streams"]):
        x = 0
        while x < meta["channels"]:
            if hits[s, x] / rows < occupancy:
                x += 1
                continue
            a = x
            while x < meta["channels"] and hits[s, x] / rows >= occupancy:
                x += 1
            hz = meta["rate"] / meta["channels"]
            entries.append({"stream": s, "lo_hz": (a-meta["channels"]/2-.5)*hz,
                            "hi_hz": (x-meta["channels"]/2-.5)*hz, "label": "本机杂散"})
    return {"version": 1, "termination": "50ohm", "context": calibration_context(meta), "entries": entries,
            "measurement": {"rows": rows, "seconds": last-first, "occupancy": occupancy, "threshold_db": threshold_db}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--terminated", action="store_true", help="declare a physical 50 ohm termination")
    ap.add_argument("--min-seconds", type=float, default=10)
    ap.add_argument("--occupancy", type=float, default=.98)
    ap.add_argument("--threshold-db", type=float, default=10)
    a = ap.parse_args()
    if not a.terminated:
        ap.error("--terminated is required for a confirmed 50 ohm recording")
    if a.min_seconds < 10 or not .9 <= a.occupancy <= 1 or a.threshold_db <= 0:
        ap.error("require >=10 seconds, occupancy 0.9..1, positive threshold")
    result = build(a.feed, a.min_seconds, a.occupancy, a.threshold_db)
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(result['entries'])} calibrated intervals")


if __name__ == "__main__":
    main()
