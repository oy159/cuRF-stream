#!/usr/bin/env python3
"""Apply the live display policy offline; optionally score named labels against version-2 truth.

    python tools/apply_signal_policy.py --feed scene_0.feed --out scene_0.display.feed \
        --truth truth.json --score display_score.json

No truth labels are passed to the policy. Errors among named modulation labels and
the unknown fraction have separate denominators. No named labels means undefined
error, never a successful zero-error evaluation. Protocol guesses need separate truth.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from signal_policy import SignalPolicy, load_config, load_spurs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from curf_model.scoring import match


class DisplayScore:
    def __init__(self, truth, scene=0):
        if truth.get("version") != 2:
            raise ValueError("truth requires version 2 emission times")
        self.truth = [e for e in truth["truth"] if e["scene"] == scene]
        self.counts = Counter()
        self.seen = set()

    def add(self, row):
        if "sig" not in row:
            return
        d = row["display"]
        status = d["status"]
        if status not in ("stable", "held"):
            self.counts["unknown" if status == "unknown" else status] += 1
            return
        self.counts["named"] += 1
        value, i = max(((match(row["sig"], e), i) for i, e in enumerate(self.truth)), default=(0, -1))
        if value <= 0:
            self.counts["unmatched"] += 1
        else:
            self.seen.add(i)
            self.counts["correct" if d["family"] == self.truth[i]["family"] else "wrong_class"] += 1

    def result(self):
        c = self.counts
        return {**{k: c[k] for k in ("named", "unknown", "correct", "wrong_class", "unmatched", "spur", "decoder_only")},
                "named_error_fraction": (c["wrong_class"] + c["unmatched"]) / c["named"] if c["named"] else None,
                "unknown_fraction": c["unknown"] / (c["named"] + c["unknown"]) if c["named"] + c["unknown"] else None,
                "transmitted_emissions": len(self.truth), "matched_named_emissions": len(self.seen),
                "metric": "modulation-family labels per model result; protocol guesses not scored"}


def apply(feed, out, policy, score=None):
    if Path(feed).resolve() == Path(out).resolve():
        raise ValueError("input and output must be different files")
    metadata, ended = False, False
    with Path(feed).open() as src, Path(out).open("w") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if "meta" in row:
                if metadata:
                    raise ValueError("offline scoring expects one finite run, not a replay loop")
                metadata = True
            elif not metadata:
                raise ValueError("feed requires metadata before results")
            if ended:
                raise ValueError("feed has records after end")
            row = policy.process(row)
            if score:
                score.add(row)
            ended = "end" in row
            dst.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    if not ended:
        raise ValueError("feed incomplete: missing end record")
    return score.result() if score else policy.stats()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy", type=Path)
    ap.add_argument("--spurs", type=Path)
    ap.add_argument("--truth", type=Path)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--score", type=Path)
    a = ap.parse_args()
    policy = SignalPolicy(load_config(a.policy), load_spurs(a.spurs))
    score = DisplayScore(json.loads(a.truth.read_text()), a.scene) if a.truth else None
    result = apply(a.feed, a.out, policy, score)
    output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if a.score:
        a.score.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
