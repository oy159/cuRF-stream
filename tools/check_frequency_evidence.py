#!/usr/bin/env python3
"""Score one complete raw feed with and without its structural frequency correction.

This isolates policy effects on the same observations. It is not a second receiver
run, an IQ replay, a protocol decoder, or a real-data accuracy measurement.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from signal_policy import SignalPolicy, load_config


class Target:
    def __init__(self, frequency, stream, seconds):
        self.frequency, self.stream, self.seconds = frequency, stream, seconds
        self.named = 0
        self.first = None
        self.last = self.gap = 0.
        self.bins = set()

    def add(self, display):
        if (display["s"] != self.stream or display["family"] != "dsss" or display["bw"] < 8e6 or
                abs(display["freq"]-self.frequency) > .5e6):
            return
        t = display["t"]
        if not 0 <= t <= self.seconds + .01:
            raise ValueError("reference time outside the declared source duration")
        self.gap = max(self.gap, t-self.last)
        self.last = t
        self.first = t if self.first is None else self.first
        self.named += 1
        self.bins.add(min(math.ceil(self.seconds/30)-1, int(t//30)))

    def result(self):
        return {"named": self.named, "first_s": self.first, "last_s": self.last if self.named else None,
                "source_30s_bins": sorted(self.bins), "maximum_gap_s": max(self.gap, self.seconds-self.last)}


def evaluate(feed, config, seconds, stream, frequencies):
    if not math.isfinite(seconds) or seconds <= 0 or stream < 0:
        raise ValueError("positive finite duration and nonnegative stream required")
    policies = {mode: SignalPolicy(config) for mode in ("corrected", "extraction_centres")}
    targets = {mode: {str(f/1e6): Target(f, stream, seconds) for f in frequencies} for mode in policies}
    reasons = {mode: Counter() for mode in policies}
    evidence, digest, complete, metadata = Counter(), hashlib.sha256(), False, None
    with Path(feed).open("rb") as source:
        for line in source:
            digest.update(line)
            row = json.loads(line)
            if complete:
                raise ValueError("records after end marker")
            if "meta" in row:
                if metadata is not None:
                    raise ValueError("multiple captures in one feed")
                metadata = row["meta"]
                if stream >= metadata["streams"]:
                    raise ValueError("reference stream is absent")
            if metadata is None:
                raise ValueError("feed must start with metadata")
            original = row
            if "sig" in row:
                sig = row["sig"]
                if "extraction_f" in sig:
                    original = {**row, "sig": {**sig, "f": sig["extraction_f"]}}
                    evidence["tested"] += 1
                    estimate = sig["dsss_frequency"]
                    evidence["valid"] += bool(estimate["valid"])
                    if sig["f"] != sig["extraction_f"]:
                        if not (estimate["valid"] and estimate["coherence"] >= .8 and estimate["contrast"] >= .1 and
                                estimate["retained_power"] >= .3 and abs(estimate["offset_hz"]) <= 1.5e6 and
                                sig["c"] == 6 and sig["p"] >= .9 and
                                abs(sig["f"]-sig["extraction_f"]-estimate["offset_hz"]) <= 1):
                            raise ValueError("frequency correction lacks its required structural evidence")
                        evidence["corrected"] += 1
            for mode, message in (("corrected", row), ("extraction_centres", original)):
                scored = policies[mode].process(message)
                if "sig" in row:
                    reasons[mode][scored["display"]["reason"]] += 1
                    for target in targets[mode].values():
                        target.add(scored["display"])
            complete = "end" in row
    if not complete:
        raise ValueError("incomplete feed: no end marker")
    return {"feed_sha256": digest.hexdigest(), "policy_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "seconds": seconds, "stream": stream, "structural_evidence": dict(evidence),
            "results": {mode: {"policy": policy.stats(), "reasons": dict(reasons[mode]),
                               "targets": {f: t.result() for f, t in targets[mode].items()}} for mode, policy in policies.items()},
            "limits": "Same complete model observations; only f is restored to extraction_f for policy rescoring. Reference label gaps are not measured packet loss or classification accuracy."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, required=True, help="source duration, with source origin zero")
    ap.add_argument("--stream", type=int, required=True)
    ap.add_argument("--target-mhz", type=float, action="append", default=[])
    ap.add_argument("--policy", type=Path)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("output already exists")
    if any(not math.isfinite(f) or f <= 0 for f in a.target_mhz):
        ap.error("target frequencies must be positive and finite")
    report = evaluate(a.feed, load_config(a.policy), a.seconds, a.stream, [f*1e6 for f in a.target_mhz])
    files = (Path(__file__), Path(__file__).with_name("signal_policy.py"))
    report["code_sha256"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    with a.out.open("x") as target:
        json.dump(report, target, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
