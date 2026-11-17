#!/usr/bin/env python3
"""Wideband test scenes for the signal-level pipeline (stage 1 detection + stage 2), with ground truth.

    python model/make_scene.py make  --src ~/dataset/onair --emissions ~/dataset/ds_v5/emissions.json --out ~/dataset/scene_v1
    curf_classify --source file --file ~/dataset/scene_v1/scene_0.p12 --streams 1 --rate 61.44e6 --freq 0 \
        --stage2 model/artifacts/s2_run4/export --ui-out ~/dataset/scene_v1/scene_0.feed --band-mhz 44
    python model/make_scene.py score --out ~/dataset/scene_v1

make: per scene (0.5 s, one stream of 61.44 MS/s packed12), K sources from the recordings of tools/onair_gen.py /
tools/synth_sources.py, taken from the last 25 % of each recording (the validation side of make_dataset.py's time
split), resampled to 61.44 MS/s and placed at random non-overlapping centres within +-20 MHz. Each source's
in-band SNR (its measured 99 % bandwidth against the noise floor) is uniform in [lo, hi] dB. truth.json lists
every emission of every placed source (family, mode, centre, bandwidth, span).

score: matches the {"sig"} lines of each scene's feed to the truth (same centre within half the larger
bandwidth, overlapping in time) and reports per family: signals classified correctly / wrongly, and the
results that match no transmitted signal at all (false alarms: noise, edges, artefacts).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from curf_model.widesig import FAMILIES, ONAIR_TO_FAMILY, load_sigmf

FS = 61.44e6
CUT = 0.75


def refine_times(src, e):
    """Replace STFT-quantised edges with the clean recording's active sample edges."""
    a, b = max(0, int(e["t0"])), min(len(src), int(e["t1"]))
    power = np.abs(src[a:b]) ** 2
    if not len(power) or power.max() <= 0:
        return None
    active = np.flatnonzero(power > power.max() * 1e-4)
    return {**e, "t0": int(a + active[0]), "t1": int(a + active[-1] + 1)}


def pack12(x: np.ndarray) -> np.ndarray:
    def q(v):
        r = np.where(v >= 0, np.floor(v * 2048 + 0.5), np.ceil(v * 2048 - 0.5))
        return np.clip(r, -2048, 2047).astype(np.int32) & 0xFFF
    i, qq = q(x.real), q(x.imag)
    out = np.empty((x.size, 3), np.uint8)
    out[:, 0] = (i >> 4) & 0xFF
    out[:, 1] = ((i & 0xF) << 4) | ((qq >> 8) & 0xF)
    out[:, 2] = qq & 0xFF
    return out


def make(a):
    index = {r["id"]: r for r in json.loads((a.src / "index.json").read_text())}
    ems = json.loads(a.emissions.read_text())
    modes = [m for m in ems if m in index and ONAIR_TO_FAMILY.get(index[m]["family"]) and ems[m]]
    if a.modes:
        modes = [m for m in a.modes.split(",") if m in modes]
    rng = np.random.default_rng(a.seed)
    n = int(a.secs * FS)
    noise_rms = 0.003
    n0 = noise_rms ** 2 / FS  # noise power per Hz
    truth = []
    a.out.mkdir(parents=True, exist_ok=True)
    for sc in range(a.scenes):
        x = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64) * np.float32(noise_rms / np.sqrt(2))
        used = []  # occupied [lo, hi]
        placed = 0
        for _ in range(200):
            if placed >= a.sources:
                break
            m = modes[placed % len(modes)] if a.one_of_each else modes[rng.integers(len(modes))]
            src, fs_src = load_sigmf(a.src / f"{m}.sigmf-meta")
            # validation side of the recording, a.secs long (looped when shorter)
            start = int(CUT * len(src))
            seg_len = int(a.secs * fs_src) + 1
            seg = src[start:]
            period = len(seg)
            if len(seg) < 16:
                continue
            seg = np.resize(seg, seg_len)
            e_in = [r for e in ems[m] if e["t1"] > start
                    if (r := refine_times(src, e)) is not None and r["t1"] > start]
            if not e_in:
                continue
            lo_src = min(e["fc"] - e["bw"] / 2 for e in e_in)
            hi_src = max(e["fc"] + e["bw"] / 2 for e in e_in)
            width = hi_src - lo_src
            if width > 30e6:
                continue
            # centre of the source's occupied band at f
            for _try in range(20):
                f = rng.uniform(-20e6 + width / 2, 20e6 - width / 2) if width < 40e6 else 0.0
                lo, hi = f - width / 2 - 200e3, f + width / 2 + 200e3
                if all(hi < u0 or lo > u1 for u0, u1 in used):
                    break
            else:
                continue
            used.append((lo, hi))
            shift = f - (lo_src + hi_src) / 2  # added to every source frequency
            fr = Fraction(FS / fs_src).limit_denominator(2000)
            y = resample_poly(seg, fr.numerator, fr.denominator)[:n].astype(np.complex64)
            if len(y) < n:
                y = np.resize(y, n)
            t = np.arange(n, dtype=np.float64) / FS
            y = y * np.exp(2j * np.pi * shift * t).astype(np.complex64)
            # scale: in-band SNR of the strongest emission's bandwidth while active
            bw = max(e["bw"] for e in e_in)
            p = np.abs(y) ** 2
            act = p > p.max() * 1e-3
            ps = float(p[act].mean()) if act.any() else 1.0
            snr = rng.uniform(a.snr[0], a.snr[1])
            y *= np.float32(np.sqrt(10 ** (snr / 10) * n0 * bw / ps))
            x += y
            fam = ONAIR_TO_FAMILY[index[m]["family"]]
            for base in range(0, seg_len, period):
                for e in e_in:
                    t0 = max(0, (base + max(e["t0"], start) - start) / fs_src)
                    t1 = min(a.secs, (base + min(e["t1"], len(src)) - start) / fs_src)
                    if t1 > t0:
                        truth.append({"scene": sc, "mode": m, "family": fam, "fc": e["fc"] + shift,
                                      "bw": e["bw"], "snr_db": snr, "t0": t0, "t1": t1, "source_id": f"{sc}:{placed}"})
            placed += 1
        peak = np.abs(x).max()
        if peak > 0.9:
            x *= np.float32(0.9 / peak)
        pack12(x).tofile(a.out / f"scene_{sc}.p12")
        print(json.dumps({"scene": sc, "sources": placed, "peak": float(peak)}), flush=True)
    (a.out / "truth.json").write_text(json.dumps({"version": 2, "secs": a.secs, "scenes": a.scenes, "truth": truth}, indent=1))


def match(s, e):
    # Require overlap in both frequency and source time. Very wide merged detections cannot match a narrow carrier.
    if min(s["bw"], e["bw"]) <= 0 or max(s["bw"], e["bw"]) > 4 * min(s["bw"], e["bw"]):
        return 0.0
    flo = max(s["f"] - s["bw"]/2, e["fc"] - e["bw"]/2)
    fhi = min(s["f"] + s["bw"]/2, e["fc"] + e["bw"]/2)
    overlap = max(0.0, fhi - flo)
    if overlap < .5 * min(s["bw"], e["bw"]):
        return 0.0
    dt = max(0.0, min(s["t1"], e["t1"]) - max(s["t0"], e["t0"]))
    if dt < .25 * min(s["t1"] - s["t0"], e["t1"] - e["t0"]) or dt <= 0:
        return 0.0
    return overlap / (s["bw"] + e["bw"] - overlap) * dt / (s["t1"] - s["t0"])


def score(a):
    t = json.loads((a.out / "truth.json").read_text())
    if t.get("version") != 2:
        raise ValueError("regenerate scenes: old truth has no emission times")
    by_scene = defaultdict(list)
    for e in t["truth"]:
        by_scene[e["scene"]].append(e)
    ok, wrong, false = Counter(), Counter(), Counter()
    conf, seen = Counter(), set()
    for sc in range(t["scenes"]):
        feed = a.out / f"scene_{sc}.feed"
        lines = [json.loads(line) for line in feed.read_text().splitlines() if line.strip()]
        if not any("end" in row for row in lines):
            raise ValueError(f"{feed}: missing end record (run incomplete)")
        for row in lines:
            if "sig" not in row:
                continue
            s = row["sig"]
            if s["p"] < a.confidence:
                continue
            candidates = [(match(s, e), i) for i, e in enumerate(by_scene[sc])]
            value, best = max(candidates, default=(0, -1))
            pred = FAMILIES[s["c"]]
            if value <= 0:
                false[pred] += 1
                continue
            e = by_scene[sc][best]
            seen.add((sc, best))
            conf[(e["family"], pred)] += 1
            (ok if pred == e["family"] else wrong)[e["family"]] += 1
    correct, incorrect, spurious = sum(ok.values()), sum(wrong.values()), sum(false.values())
    total = correct + incorrect + spurious
    missed = Counter(e["family"] for sc, es in by_scene.items() for i, e in enumerate(es) if (sc, i) not in seen)
    result = {"confidence_threshold": a.confidence, "records": total, "correct": correct, "wrong_class": incorrect,
              "unmatched": spurious, "error_fraction": (incorrect + spurious) / max(total, 1),
              "transmitted_emissions": len(t["truth"]), "reported_emissions": len(seen), "missed": dict(missed),
              "per_family": {f: {"correct": ok[f], "wrong": wrong[f]} for f in FAMILIES},
              "confusion": {f"{f}->{p}": count for (f, p), count in conf.items()}}
    (a.out / "score.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["make", "score"])
    ap.add_argument("--src", type=Path)
    ap.add_argument("--emissions", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--one-of-each", action="store_true", help="place requested modes in order for controlled checks")
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--secs", type=float, default=0.5)
    ap.add_argument("--sources", type=int, default=8)
    ap.add_argument("--snr", type=float, nargs=2, default=(8.0, 30.0))
    ap.add_argument("--modes", help="comma list of modes to draw from (default: all)")
    ap.add_argument("--confidence", type=float, default=0.0, help="score only records at this raw model confidence or above")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    make(a) if a.step == "make" else score(a)


if __name__ == "__main__":
    main()
