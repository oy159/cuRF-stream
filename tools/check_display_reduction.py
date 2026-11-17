#!/usr/bin/env python3
"""Score post-policy display reduction against full frozen M4 feeds, without rerunning IQ.

The full policy sees every model window. Both full and reduced decisions are
scored against the same truth. Counts describe observations, coverage describes
carriers; neither is a new held-out model accuracy measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from live_feed import DisplayReducer
from room_evaluation import config_hash, evaluate_case, summarize
from signal_policy import load_config


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(source, out, interval=.2):
    cases = sorted(p for p in source.iterdir() if (p/"truth.json").exists())
    if not cases:
        raise ValueError("source contains no M4 cases")
    out.mkdir(parents=True, exist_ok=False)
    config = load_config()
    full, reduced, inputs, sampling = [], [], {}, {}
    for case in cases:
        target = out/case.name
        target.mkdir()
        for name in ("truth.json", "runtime.json"):
            shutil.copy2(case/name, target/name)
        inputs[case.name] = {name: digest(case/name) for name in ("raw.feed", "truth.json", "runtime.json")}
        reducer = DisplayReducer(interval)
        with (case/"raw.feed").open() as src, (target/"display.feed").open("w") as dst:
            for line in src:
                for row in reducer.push(json.loads(line)):
                    dst.write(json.dumps(row, separators=(",", ":"))+"\n")
            for row in reducer.flush():
                dst.write(json.dumps(row, separators=(",", ":"))+"\n")
        full.append(evaluate_case(case, config, write_feed=False))
        reduced.append(evaluate_case(target, config, write_feed=False, processed=True, feed_path=target/"display.feed"))
        sampling[case.name] = reducer.stats()
    baseline = summarize(full, "full-frozen-feed")
    display = summarize(reduced, "post-policy-display-regression")
    checks = {
        "full_feed_passed": baseline["passed"], "display_passed": display["passed"],
        "carrier_coverage_preserved": all(a["high_snr_named"] == b["high_snr_named"] for a,b in zip(full,reduced)),
        "no_additional_errors": all(b["counts"].get("errors",0) <= a["counts"].get("errors",0) for a,b in zip(full,reduced)),
        "all_named_decisions_preserved": all(b["counts"].get("named",0) == a["counts"].get("named",0) for a,b in zip(full,reduced)),
        "all_observations_evaluated": all(s["input_signals"] == s["policy"]["records"] for s in sampling.values()),
        "full_policy_counters_preserved": all(all(sampling[c.name]["policy"][k] == r["counts"].get(k,0)
                                                 for k in ("named","unknown")) for c,r in zip(cases,full)),
    }
    result = {"passed": all(checks.values()), "checks": checks, "full": baseline, "display": display,
              "sampling": sampling, "inputs": inputs, "policy_sha256": config_hash(config),
              "interval_s": interval, "limits": "Regression on frozen model feeds. Reduced counts are not raw unknown-rate improvement."}
    result["code_sha256"] = {p: digest(Path(__file__).resolve().parent/p)
                             for p in ("live_feed.py","signal_policy.py","room_evaluation.py","check_display_reduction.py")}
    (out/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=.2)
    args = parser.parse_args()
    report = check(args.source, args.out, args.interval)
    print(json.dumps({k: report[k] for k in ("passed", "checks", "full", "display")}))
    raise SystemExit(0 if report["passed"] else 1)
