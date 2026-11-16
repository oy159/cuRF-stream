#!/usr/bin/env python3
"""Build the stage-2 (signal-level) dataset from clean protocol recordings.

    python model/make_dataset.py --src ~/dataset/onair --out ~/dataset/ds_v1 [--per-class 10000] [--workers 10]

Input: tools/onair_gen.py output (SigMF cf32 per OnAir mode + index.json with the modulation family).
Output, per split (train / val):
  <split>.npz  x [N, 2, L] float16 (unit-RMS IQ, the model's layout), y [N] class index (FAMILIES),
               log_bw [N] log10 of the bandwidth estimate in Hz (side input), snr [N] in-band dB (nan: noise),
               mode [N] index into meta.json "modes"
  meta.json    classes, modes with their family / emission counts, parameters
  emissions.json  every emission found per mode (centre, bandwidth, span)

Splits are by time: train windows come from the first 75 % of each recording, val windows from the rest. Every
mode of a family gets the same share of that family's examples, so a family is not dominated by its widest or
busiest protocol. See curf_model/widesig.py for the extraction and impairments.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from curf_model.widesig import (FAMILIES, ONAIR_TO_FAMILY, Emission, emissions, extract, impair, load_sigmf,
                                noise_window)

CUT = 0.75  # train / val boundary, fraction of each recording


def scan(meta_path: str, mode: str, family: str) -> tuple[str, list[dict], int]:
    x, fs = load_sigmf(Path(meta_path))
    # pulses (ADS-B: 120 us frames) are shorter than one STFT frame: smoothing over time would spread them below
    # the threshold except for their centre line, so smooth over frequency only
    kw = {"smooth_t": 1, "smooth_f": 9} if family == "pulse" else {}
    return mode, [asdict(e) for e in emissions(x, fs, mode, **kw)], len(x)


def split_parts(ems: list[Emission], n: int, split: str) -> list[Emission]:
    """Train: the parts of the emissions before CUT * n, val: after. A recording whose bursts all fall on one
    side is split by burst instead (every 4th to val); a single short burst is shared by both (noted in meta)."""
    cut = int(CUT * n)
    out = []
    for e in ems:
        a, b = (e.t0, min(e.t1, cut)) if split == "train" else (max(e.t0, cut), e.t1)
        if b - a > 0:
            out.append(Emission(e.source, e.fc, e.bw, a, b, e.snr_db))
    other = any(min(e.t1, cut) > e.t0 for e in ems) if split == "val" else any(e.t1 > max(e.t0, cut) for e in ems)
    if out and other:
        return out
    if len(ems) < 2:
        return list(ems)
    if len(ems) < 4:  # the last burst for val
        return ems[-1:] if split == "val" else ems[:-1]
    return [e for i, e in enumerate(ems) if (i % 4 == 3) == (split == "val")]


def make(task: dict) -> dict:
    """Examples of one mode for one split (one worker loads the recording once)."""
    rng = np.random.default_rng(task["seed"])
    L, n = task["L"], task["n"]
    xs = np.empty((n, 2, L), np.float16)
    lbw, snr = np.empty(n, np.float32), np.empty(n, np.float32)
    if task["mode"] is None:  # noise class
        for i in range(n):
            y = noise_window(L, rng)
            xs[i] = np.stack([y.real, y.imag])
            lbw[i] = rng.uniform(np.log10(5e3), np.log10(30e6))
        snr[:] = np.nan
        return {"x": xs, "log_bw": lbw, "snr": snr, **{k: task[k] for k in ("y", "mode_idx")}}
    x, fs = load_sigmf(Path(task["path"]))
    ems = [Emission(**e) for e in task["emissions"]]
    parts = split_parts(ems, len(x), task["split"])
    w = np.array([max(p.t1 - p.t0, 1) for p in parts], float) ** 0.5  # long emissions count more, not linearly
    w /= w.sum()
    lo, hi = task["snr_range"]
    for i in range(n):
        e = parts[rng.choice(len(parts), p=w)]
        y, b_est = extract(x, fs, e, L, rng)
        s = rng.uniform(lo, hi)
        y = impair(y, max(2 * b_est, 8e3), s, rng)
        xs[i] = np.stack([y.real, y.imag])
        lbw[i], snr[i] = np.log10(b_est), s
    return {"x": xs, "log_bw": lbw, "snr": snr, **{k: task[k] for k in ("y", "mode_idx")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=10000, help="train examples per class (val: a quarter)")
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--snr", type=float, nargs=2, default=(-5.0, 30.0), help="in-band SNR range, dB")
    ap.add_argument("--min-emissions-s", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4, help="each holds one recording (up to ~0.4 GB) in memory")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scan-only", action="store_true", help="find and list the emissions, build nothing")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    index = json.loads((a.src / "index.json").read_text())
    modes = [(r["id"], ONAIR_TO_FAMILY[r["family"]]) for r in index if r.get("family") in ONAIR_TO_FAMILY]
    with ProcessPoolExecutor(a.workers) as ex:
        scanned = list(ex.map(scan, [str(a.src / f"{m}.sigmf-meta") for m, _ in modes], [m for m, _ in modes],
                              [f for _, f in modes]))
    ems = {m: (e, n) for m, e, n in scanned}
    (a.out / "emissions.json").write_text(json.dumps({m: e for m, (e, _) in ems.items()}))
    usable = []
    for m, fam in modes:
        e, n = ems[m]
        ok = all(split_parts([Emission(**x) for x in e], n, s) for s in ("train", "val"))
        bws = [x["bw"] for x in e]
        dur = [(x["t1"] - x["t0"]) / n for x in e]
        print(f"  {m:9s} {fam:8s} emissions {len(e):5d}  bw median {np.median(bws) / 1e3 if bws else 0:9.1f} kHz"
              f"  max {max(bws) / 1e3 if bws else 0:9.1f} kHz  longest {max(dur) if dur else 0:4.2f} of the file"
              f"{'' if ok else '  -- skipped: no emission in one of the splits'}")
        if ok:
            usable.append((m, fam))
    mode_names = [m for m, _ in usable]
    if a.scan_only:
        return

    for split, per in (("train", a.per_class), ("val", a.per_class // 4)):
        tasks = []
        for ci, fam in enumerate(FAMILIES):
            fm = [m for m, f in usable if f == fam]
            if fam == "noise":
                tasks.append({"mode": None, "n": per, "y": ci, "mode_idx": -1, "L": a.length,
                              "seed": zlib.crc32(f"{a.seed}/{split}/{fam}".encode())})
                continue
            if not fm:
                print(f"  class {fam}: no source, skipped")
                continue
            share = [per // len(fm) + (1 if k < per % len(fm) else 0) for k in range(len(fm))]
            for m, n in zip(fm, share):
                tasks.append({"mode": m, "path": str(a.src / f"{m}.sigmf-meta"), "emissions": ems[m][0], "split": split,
                              "n": n, "y": ci, "mode_idx": mode_names.index(m), "L": a.length, "snr_range": a.snr,
                              "seed": zlib.crc32(f"{a.seed}/{split}/{m}".encode())})
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(make, tasks))
        x = np.concatenate([r["x"] for r in res])
        y = np.concatenate([np.full(len(r["x"]), r["y"], np.uint8) for r in res])
        mode = np.concatenate([np.full(len(r["x"]), r["mode_idx"], np.int16) for r in res])
        lbw = np.concatenate([r["log_bw"] for r in res])
        snr = np.concatenate([r["snr"] for r in res])
        perm = np.random.default_rng(a.seed).permutation(len(y))
        np.savez(a.out / f"{split}.npz", x=x[perm], y=y[perm], mode=mode[perm], log_bw=lbw[perm], snr=snr[perm])
        c = Counter(y.tolist())
        print(f"{split}: {len(y)} examples, {x.nbytes / 1e9:.2f} GB; per class " +
              ", ".join(f"{FAMILIES[k]} {c[k]}" for k in sorted(c)))

    (a.out / "meta.json").write_text(json.dumps({
        "classes": FAMILIES, "modes": mode_names, "mode_family": dict(usable), "length": a.length,
        "snr_range": a.snr, "per_class": a.per_class, "train_val_cut": CUT, "source": str(a.src),
        "emissions": {m: len(ems[m][0]) for m in mode_names},
    }, indent=1))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
