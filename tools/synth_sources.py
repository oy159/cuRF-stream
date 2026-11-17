#!/usr/bin/env python3
"""Our own signal sources for the dataset, next to OnAir's (tools/onair_gen.py), where its generators have no knob
for the variety we need. Written as SigMF cf32 into the same directory, with rows merged into its index.json.

    python3 tools/synth_sources.py --out ~/dataset/onair [--count 12] [--secs 2]

Sources:
  fmbc_<k>   broadcast FM, 1 MS/s: programme (music-like or speech-like, 50 us pre-emphasis), mono or stereo
             (19 kHz pilot, L-R on 38 kHz DSB-SC), with or without RDS (57 kHz, 1187.5 bit/s biphase BPSK),
             peak deviation 40-75 kHz. Real stations differ in all of these; one clean stereo source (OnAir's) is
             why live FM was taken for DSSS.
  nfm_<k>    narrow-band FM voice, 250 kS/s: speech-like audio 300-3400 Hz with pauses, deviation 1.5-5 kHz,
             optional CTCSS tone (67-250 Hz), keyed on and off.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, resample_poly, sosfilt


def lowpass(x, fc, fs, order=8):
    return sosfilt(butter(order, fc, fs=fs, output="sos"), x)


def bandpass(x, lo, hi, fs, order=4):
    return sosfilt(butter(order, [lo, hi], btype="band", fs=fs, output="sos"), x)


def music(n, fs, rng):
    """Notes with harmonics and envelopes, plus some broadband 'drums' and noise: a crude stand-in for programme."""
    t = np.arange(n) / fs
    y = np.zeros(n)
    pos = 0
    while pos < n:
        dur = int(rng.uniform(0.1, 0.6) * fs)
        f0 = 110 * 2 ** (rng.integers(0, 36) / 12)
        seg = np.arange(min(dur, n - pos)) / fs
        env = np.exp(-seg * rng.uniform(1, 8))
        tone = sum(rng.uniform(0.2, 1) / h * np.sin(2 * np.pi * f0 * h * seg + rng.uniform(0, 6.3)) for h in range(1, 6))
        y[pos:pos + len(seg)] += env * tone
        pos += int(dur * rng.uniform(0.3, 1.0))
    y += 0.3 * lowpass(rng.normal(size=n), 8e3, fs, 4) * (np.sin(2 * np.pi * rng.uniform(1, 3) * t) > 0.7)
    return y


def speech(n, fs, rng, lo=100.0, hi=7e3):
    """Band-limited noise with a syllabic (3-6 Hz) envelope and pauses."""
    t = np.arange(n) / fs
    y = bandpass(rng.normal(size=n), lo, hi, fs)
    syl = np.maximum(0, np.sin(2 * np.pi * rng.uniform(3, 6) * t + rng.uniform(0, 6.3))) ** 2
    gate = np.ones(n)
    k = 0
    while k < n:  # talk 0.5-3 s, pause 0.1-0.8 s
        k += int(rng.uniform(0.5, 3) * fs)
        p = int(rng.uniform(0.1, 0.8) * fs)
        gate[k:k + p] = 0
        k += p
    return y * syl * gate


def norm(x):
    return x / (np.max(np.abs(x)) + 1e-12)


def compress(x, drive):
    """Broadcast-style loudness processing: soft clipping after gain, so the average modulation stays high (real
    stations run heavy multiband compression; without it the occupied bandwidth is far below a real station's)."""
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-12)
    return np.tanh(drive * x / 3) / np.tanh(drive)


def fm_broadcast(secs, rng):
    fa, fs = 192e3, 1e6  # audio / MPX rate, output rate
    n = int(secs * fa)
    prog = music if rng.random() < 0.6 else speech
    left = prog(n, fa, rng)
    right = 0.6 * left + 0.4 * prog(n, fa, rng) if rng.random() < 0.7 else left
    pre = lambda x: lfilter([1, -0.75], [1], x)  # ~50 us pre-emphasis at this rate (first-order approximation)
    drive = rng.uniform(3, 10)  # real stations: 99 % bandwidth ~150-230 kHz measured on air
    left, right = (norm(lowpass(compress(pre(c), drive), 15e3, fa)) for c in (left, right))
    t = np.arange(n) / fa
    stereo = rng.random() < 0.75
    rds = rng.random() < 0.7
    mpx = 0.9 * (left + right) / 2
    if stereo:
        mpx = 0.8 * mpx / 0.9 + 0.8 * (left - right) / 2 * np.cos(2 * np.pi * 38e3 * t) + 0.09 * np.cos(2 * np.pi * 19e3 * t)
    if rds:
        bits = rng.integers(0, 2, int(secs * 1187.5) + 2)
        sym = np.repeat(np.diff(np.concatenate([[0], bits])) % 2 * 2 - 1, 2) * np.tile([1, -1], len(bits))  # biphase
        sps = fa / 2375
        bb = lowpass(sym[(np.arange(n) / sps).astype(int) % len(sym)].astype(float), 2.4e3, fa, 4)
        mpx = mpx + 0.04 * bb * np.sin(2 * np.pi * 57e3 * t)
    dev = rng.uniform(55e3, 80e3)  # nominal 75 kHz, often exceeded
    mpx = mpx / np.max(np.abs(mpx))
    ph = 2 * np.pi * dev * np.cumsum(mpx) / fa
    y = resample_poly(np.exp(1j * ph), 125, 24)  # 192k -> 1M
    return y.astype(np.complex64), fs, {"deviation_hz": round(dev), "stereo": stereo, "rds": rds, "programme": prog.__name__, "drive": round(drive, 1)}


def fm_narrow(secs, rng):
    fs = 250e3
    n = int(secs * fs)
    a = norm(compress(speech(n, fs, rng, 300, 3400), rng.uniform(1.5, 4)))
    ctcss = rng.random() < 0.5
    if ctcss:
        a = 0.85 * a + 0.15 * np.sin(2 * np.pi * rng.uniform(67, 250) * np.arange(n) / fs)
    dev = rng.uniform(1.5e3, 5e3)
    y = np.exp(1j * 2 * np.pi * dev * np.cumsum(a) / fs)
    key = np.zeros(n)  # transmitter keyed: 2-5 s overs, gaps
    k = int(rng.uniform(0, 0.2) * fs)
    while k < n:
        on = int(rng.uniform(1.0, 4.0) * fs)
        key[k:k + on] = 1
        k += on + int(rng.uniform(0.2, 1.0) * fs)
    return (y * key).astype(np.complex64), fs, {"deviation_hz": round(dev), "ctcss": ctcss}


def write(stem: Path, y, fs, desc):
    y.astype("<c8").tofile(stem.with_suffix(".cf32"))
    stem.with_suffix(".sigmf-meta").write_text(json.dumps({
        "global": {"core:datatype": "cf32_le", "core:sample_rate": fs, "core:version": "1.0.0",
                   "core:dataset": stem.with_suffix(".cf32").name, "core:recorder": "cuRF-Stream tools/synth_sources.py",
                   "core:description": desc}, "captures": [{"core:sample_start": 0}], "annotations": []}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=12, help="recordings per source type")
    ap.add_argument("--secs", type=float, default=2)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    for kind, fn, what in (("fmbc", fm_broadcast, "broadcast FM (own generator)"), ("nfm", fm_narrow, "narrow-band FM voice (own generator)")):
        for k in range(a.count):
            y, fs, p = fn(a.secs, rng)
            name = f"{kind}_{k}"
            write(a.out / name, y, fs, f"{what}: {p}")
            rows.append({"id": name, "onair_id": None, "args": [], "family": "fm", "what": f"{what}: {p}",
                         "fs_mhz": fs / 1e6, "secs": a.secs, "obw_khz": None, "duty": None, "network_blocked": 0})
            print(json.dumps(rows[-1]), flush=True)
    idx = a.out / "index.json"
    old = json.loads(idx.read_text()) if idx.exists() else []
    ids = {r["id"] for r in rows}
    idx.write_text(json.dumps([r for r in old if r["id"] not in ids] + rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
