#!/usr/bin/env python3
"""Labelled wideband capture for end-to-end accuracy, plus the model's accuracy on the same signals without
the channelizer (the reference the pipeline is compared against).

    python tools/make_scenario.py --out /tmp/scn --windows 16 --spacing 4 --snr 0 20
    curf_classify --source file --file /tmp/scn/capture.raw --detect relative --threshold-db 3 --dump /tmp/scn/results.csv
    python tools/make_scenario.py --score /tmp/scn

Capture: 2 streams x 61.44 MS/s, packed12. In each stream, every `spacing`-th channel k (centre k * 120 kHz)
carries one continuous signal of a random class (training synthesiser, clean, unit RMS) for the whole capture;
the rest is white noise. The baseband signal (240 kS/s) is placed by frequency-domain interpolation: its
spectrum (+-120 kHz) has the same bin spacing as the 256x longer wideband spectrum, so it is copied into the
bins around channel k. Per-channel SNR is uniform in [lo, hi] dB, defined like the training data: signal
power over noise power in the channel's 240 kHz.

Reference ("direct"): the same baseband signals, cut into the same 1024-sample windows, plus white noise at the
same SNR, unit RMS, through the PyTorch model - what the classifier scores on training-like inputs.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from curf_model.net import AmcResNet1D  # noqa: E402
from curf_model.synth import CLASSES, NARROW, Synth, SynthConfig  # noqa: E402

FS, M, D, W, STREAMS = 61.44e6, 512, 256, 1024, 2


def pack12(x: np.ndarray) -> np.ndarray:
    """complex [n] (full scale 1) -> packed12 bytes [n, 3], same rounding as curf::ref::pack_sample."""
    def q(v):
        r = np.where(v >= 0, np.floor(v * 2048 + 0.5), np.ceil(v * 2048 - 0.5))
        return np.clip(np.nan_to_num(r), -2048, 2047).astype(np.int32) & 0xFFF
    i, qq = q(x.real), q(x.imag)
    out = np.empty((x.size, 3), np.uint8)
    out[:, 0] = (i >> 4) & 0xFF
    out[:, 1] = ((i & 0xF) << 4) | ((qq >> 8) & 0xF)
    out[:, 2] = qq & 0xFF
    return out


def make(args):
    dev = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nb = W * args.windows                 # baseband samples per channel
    n = nb * D                            # wideband samples per stream
    classes = list(NARROW)  # one channel per signal: no noise, no wide slices
    rng = np.random.default_rng(args.seed)
    synth = Synth(SynthConfig(length=nb, chunk=1, channel_filter=False, broadcast_fm_prob=0.0), device=dev, seed=args.seed)
    ckpt = torch.load(args.ckpt, map_location=dev)
    model = AmcResNet1D(len(CLASSES)).to(dev).eval()
    model.load_state_dict(ckpt["model"])

    noise_var = args.noise_rms ** 2 * 2   # complex sample variance
    chan_noise = noise_var / D            # noise power in one channel's 240 kHz
    labels, direct_ok, direct_n = [], 0, 0
    wide = []
    for s in range(STREAMS):
        X = torch.zeros(n, dtype=torch.complex64, device=dev)
        for k in range(args.spacing // 2, M, args.spacing):
            name = classes[rng.integers(len(classes))]
            snr = rng.uniform(*args.snr)
            sig = synth._chunk(name, 1)[0]                  # [nb] clean, unit RMS, continuous
            amp = math.sqrt(chan_noise * 10 ** (snr / 10))
            Sb = torch.fft.fftshift(torch.fft.fft(sig * amp))  # bins -nb/2 .. nb/2-1
            centre = (k if k < M // 2 else k - M) * (nb // 2) # channel spacing = nb/2 bins
            idx = (centre + torch.arange(-nb // 2, nb // 2, device=dev)) % n
            X[idx] += Sb * (n / nb)
            labels.append({"stream": s, "channel": k, "cls": CLASSES.index(name), "snr_db": round(snr, 2)})
            # direct reference on the same windows (skip window 0 like the pipeline score)
            with torch.no_grad():
                win = (sig * amp).reshape(args.windows, W)[1:]
                noise = (torch.randn_like(win.real) + 1j * torch.randn_like(win.real)) * math.sqrt(chan_noise / 2)
                x = win + noise
                x = x / x.abs().pow(2).mean(dim=1, keepdim=True).sqrt()
                pred = model(torch.view_as_real(x).permute(0, 2, 1).contiguous().float()).argmax(-1)
            direct_ok += int((pred == CLASSES.index(name)).sum())
            direct_n += pred.numel()
        x = torch.fft.ifft(X).cpu().numpy()
        x += (rng.normal(size=n) + 1j * rng.normal(size=n)) * args.noise_rms
        wide.append(x)
        print(f"stream {s}: rms {np.sqrt(np.mean(np.abs(x) ** 2)):.3f}, peak {np.abs(x).max():.3f}", flush=True)
    frames = np.stack([pack12(w) for w in wide], axis=1)    # [n][stream][3] interleaved frames
    frames.tofile(out / "capture.raw")
    (out / "labels.json").write_text(json.dumps({"windows": args.windows, "labels": labels,
                                                 "direct_accuracy": direct_ok / direct_n,
                                                 "direct_windows": direct_n, "snr_db": args.snr,
                                                 "spacing": args.spacing}, indent=1))
    print(json.dumps({"capture_MB": round(frames.nbytes / 1e6, 1), "channels_per_stream": len(labels) // STREAMS,
                      "direct_accuracy": round(direct_ok / direct_n, 4)}))


def score(args):
    d = Path(args.score)
    meta = json.loads((d / "labels.json").read_text())
    truth = {(l["stream"], l["channel"]): l for l in meta["labels"]}
    import csv
    rows = list(csv.DictReader(open(d / "results.csv")))
    hits = det = false_alarm = fa_signal = fa_neighbour = 0
    by_class = {}
    for r in rows:
        w = int(r["window"])
        if w == 0 or w >= meta["windows"]:   # window 0: zero-history transient
            continue
        key = (int(r["stream"]), int(r["channel"]))
        if key not in truth:
            false_alarm += 1
            fa_signal += int(r["cls"]) != 0
            neighbour = any((key[0], (key[1] + dk) % M) in truth for dk in (-1, 1))
            fa_neighbour += neighbour
            continue
        det += 1
        t = truth[key]["cls"]
        ok = int(r["cls"]) == t
        hits += ok
        c = by_class.setdefault(CLASSES[t], [0, 0])
        c[0] += ok
        c[1] += 1
    expected = len(truth) * (meta["windows"] - 1)
    print(json.dumps({
        "pipeline_accuracy": round(hits / max(det, 1), 4),
        "direct_accuracy": round(meta["direct_accuracy"], 4),
        "detected": det, "expected": expected, "recall": round(det / expected, 4), "false_alarms": false_alarm,
        "false_alarms_next_to_a_signal": fa_neighbour, "false_alarms_classified_as_a_signal": fa_signal,
        "accuracy_by_class": {k: round(v[0] / v[1], 3) for k, v in sorted(by_class.items())}}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--score")
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--spacing", type=int, default=4)
    ap.add_argument("--snr", type=float, nargs=2, default=(5.0, 20.0))
    ap.add_argument("--noise-rms", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--ckpt", default=str(Path(__file__).resolve().parents[1] / "model/artifacts/run2/best.pt"))
    args = ap.parse_args()
    if args.score:
        score(args)
    else:
        make(args)


if __name__ == "__main__":
    main()
