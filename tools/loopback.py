#!/usr/bin/env python3
"""TX -> RX loopback on one OmniCoreSDR: a labelled transmit waveform and the scoring of what came back.

    python tools/loopback.py make  --out /tmp/lb [--windows 32] [--spacing 4] [--rms-dbfs -15]
    curf_classify --source omnicore --streams 1 --freq 5.8e9 --tx-file /tmp/lb/tx.sc16 --tx-atten 70 \
                  --seconds 10 --dump /tmp/lb/att70.csv --ui-out /tmp/lb/att70.jsonl
    python tools/loopback.py score /tmp/lb /tmp/lb/att70

Waveform (one stream, 61.44 MS/s, sc16 interleaved I/Q): every `spacing`-th channel k within the analog
passband (|offset| < 25 MHz, the AD9361 RX/TX filters stop at 56 MHz) and away from DC (|k| >= 4: LO leakage)
carries one clean signal of a random class (training synthesiser, unit RMS), all at the same level. Built in
the frequency domain over the whole period, so the waveform is circular: looping it has no seam.

Score: per labelled channel and window - detected? class right? - plus its SNR measured from the received
power over the stream's noise floor (signal channels' power in the 240 kHz channel vs. the floor estimate of
empty channels). Unlabelled channels that were detected are false alarms (image / spurs / outside signals).
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

FS, M, D, W = 61.44e6, 512, 256, 1024
EDGE_MHZ, DC_GUARD = 25.0, 4


def make(a):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
    from curf_model.synth import CLASSES, Synth, SynthConfig

    dev = torch.device("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nb = W * a.windows
    n = nb * D
    classes = [c for c in CLASSES if c != "noise"]
    rng = np.random.default_rng(a.seed)
    synth = Synth(SynthConfig(length=nb, chunk=1), device=dev, seed=a.seed)
    X = torch.zeros(n, dtype=torch.complex64, device=dev)
    labels = []
    for k in range(M):
        kk = k if k < M // 2 else k - M
        if kk % a.spacing or abs(kk) < DC_GUARD or abs(kk) * FS / M / 1e6 > EDGE_MHZ:
            continue
        name = classes[rng.integers(len(classes))]
        sig = synth._chunk(name, 1)[0]
        Sb = torch.fft.fftshift(torch.fft.fft(sig))
        idx = (kk * (nb // 2) + torch.arange(-nb // 2, nb // 2, device=dev)) % n
        X[idx] += Sb * (n / nb)
        labels.append({"channel": k, "cls": CLASSES.index(name), "name": name})
    x = torch.fft.ifft(X).cpu().numpy()
    x *= 10 ** (a.rms_dbfs / 20) / np.sqrt(np.mean(np.abs(x) ** 2))
    peak = np.abs(x).max()
    if peak >= 0.999:
        raise SystemExit(f"peak {peak:.3f} clips: lower --rms-dbfs")
    iq = np.empty(2 * n, np.int16)
    iq[0::2] = np.round(x.real * 32767)
    iq[1::2] = np.round(x.imag * 32767)
    iq.tofile(out / "tx.sc16")
    (out / "labels.json").write_text(json.dumps({"windows": a.windows, "spacing": a.spacing, "rms_dbfs": a.rms_dbfs,
                                                 "classes": CLASSES, "labels": labels}, indent=1))
    print(json.dumps({"signals": len(labels), "samples": n, "MB": round(iq.nbytes / 1e6, 1),
                      "rms_dbfs": a.rms_dbfs, "peak_dbfs": round(20 * math.log10(peak), 1),
                      "classes": Counter(l["name"] for l in labels)}))


def score(a):
    meta = json.loads((Path(a.dir) / "labels.json").read_text())
    names = meta["classes"]
    truth = {l["channel"]: l["cls"] for l in meta["labels"]}
    # floor per window from the UI feed would need the feed; use the CSV: empty channels never show up there,
    # so the floor comes from the --ui-out feed ("f", dB) when present, else the 25th percentile of detections.
    floor_db = None
    feed = Path(a.run + ".jsonl")
    if feed.exists():
        f = [json.loads(l)["f"][0] for l in open(feed) if l.startswith('{"w"')]
        floor_db = float(np.median(f))
    d = np.genfromtxt(a.run + ".csv", delimiter=",", names=True)
    d = d[d["window"] >= a.skip]
    nw = len(np.unique(d["window"]))
    ch, cl, pw = d["channel"].astype(int), d["cls"].astype(int), 10 * np.log10(np.maximum(d["power"], 1e-20))
    sig = np.isin(ch, list(truth))
    ok = np.array([truth.get(c, -1) == k for c, k in zip(ch, cl)])
    per_ch = {}
    for c, t in truth.items():
        m = ch == c
        snr = float(np.median(pw[m]) - floor_db) if floor_db is not None and m.any() else None
        per_ch[c] = {"cls": names[t], "recall": m.sum() / nw, "acc": float(ok[m].mean()) if m.any() else 0.0,
                     "snr_db": snr, "top": names[Counter(cl[m]).most_common(1)[0][0]] if m.any() else "-"}
    by_class = defaultdict(lambda: [0, 0])
    for c, k in zip(ch[sig], ok[sig]):
        e = by_class[names[truth[c]]]
        e[0] += int(k)
        e[1] += 1
    fa = Counter(ch[~sig])
    snrs = [v["snr_db"] for v in per_ch.values() if v["snr_db"] is not None]
    res = {
        "run": a.run, "windows": nw, "signals": len(truth),
        "floor_db": None if floor_db is None else round(floor_db, 1),
        "snr_db_median": round(float(np.median(snrs)), 1) if snrs else None,
        "recall": round(sig.sum() / (nw * len(truth)), 4),
        "accuracy": round(float(ok[sig].mean()), 4) if sig.any() else 0.0,
        "accuracy_by_class": {k: round(v[0] / v[1], 3) for k, v in sorted(by_class.items())},
        "false_alarms_per_window": round(len(ch[~sig]) / nw, 2),
        "false_alarm_channels": len(fa),
    }
    print(json.dumps(res))
    if a.verbose:
        for c, v in sorted(per_ch.items(), key=lambda kv: kv[1]["acc"]):
            print(f"  ch {c:3d} {v['cls']:6s} acc {v['acc']:.2f} recall {v['recall']:.2f} top {v['top']:6s} "
                  f"snr {v['snr_db'] if v['snr_db'] is None else round(v['snr_db'], 1)}")
        print("  false alarms (channel: windows):", dict(fa.most_common(12)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--out", required=True)
    m.add_argument("--windows", type=int, default=32)
    m.add_argument("--spacing", type=int, default=4)
    m.add_argument("--rms-dbfs", type=float, default=-15.0)
    m.add_argument("--seed", type=int, default=11)
    s = sub.add_parser("score")
    s.add_argument("dir")
    s.add_argument("run", help="path prefix of <run>.csv (--dump) and <run>.jsonl (--ui-out)")
    s.add_argument("--skip", type=int, default=4, help="first windows ignored (TX/RX start-up)")
    s.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    make(a) if a.cmd == "make" else score(a)


if __name__ == "__main__":
    main()
