#!/usr/bin/env python3
"""Occupancy survey of SigMF captures (tools/capture_survey.py): which signals are on the air, how wide, how bursty.

    python3 tools/sigmf_survey.py ~/dataset/raw/20261116_2104 [--min-snr 8]

Per capture: peak level (clipping check), noise floor, and the occupied segments: contiguous 60 kHz bins whose
95th-percentile power over 1 ms frames is min-snr dB above the floor (median over the band). Each segment gets its
centre, -width, peak SNR and duty cycle (share of frames where its strongest bin is above the threshold).
"""
import argparse
import json
from pathlib import Path

import numpy as np

NFFT = 1024  # 60 kHz bins at 61.44 MS/s


def survey(meta_path: Path, min_snr: float, edge_mhz: float):
    meta = json.loads(meta_path.read_text())
    fs = meta["global"]["core:sample_rate"]
    fc = meta["captures"][0]["core:frequency"]
    x = np.fromfile(meta_path.with_suffix(".sigmf-data"), dtype="<i2").astype(np.float32).view(np.complex64) / 32768  # 12-bit << 4
    peak = 20 * np.log10(np.abs(x).max() + 1e-12)
    hop = int(fs * 1e-3)  # one FFT per ms
    nfr = len(x) // hop
    frames = np.stack([x[i * hop: i * hop + NFFT] for i in range(nfr)]) * np.hanning(NFFT)
    P = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)) ** 2 / NFFT + 1e-20)
    f = fc + np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / fs))
    keep = np.abs(f - fc) < (fs / 2 - edge_mhz * 1e6)
    keep &= np.abs(f - fc) > 2 * fs / NFFT  # LO leakage at DC
    p95 = np.percentile(P, 95, axis=0)
    floor = np.median(p95[keep])
    occ = keep & (p95 > floor + min_snr)
    segs, k = [], 0
    while k < NFFT:
        if not occ[k]:
            k += 1
            continue
        e = k
        while e + 1 < NFFT and (occ[e + 1] or (e + 2 < NFFT and occ[e + 2])):  # bridge one-bin gaps
            e += 1
        b = k + np.argmax(p95[k:e + 1])
        duty = float(np.mean(P[:, b] > floor + min_snr))
        segs.append({"mhz": round((f[k] + f[e]) / 2e6, 3), "khz": round((e - k + 1) * fs / NFFT / 1e3),
                     "snr": round(float(p95[b] - floor), 1), "duty": round(duty, 2)})
        k = e + 1
    return meta["global"].get("curf:band", meta_path.stem), fc, peak, floor, segs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--min-snr", type=float, default=8)
    ap.add_argument("--edge-mhz", type=float, default=4, help="ignore this much at each band edge (analog filter)")
    a = ap.parse_args()
    for m in sorted(a.dir.glob("*.sigmf-meta")):
        band, fc, peak, floor, segs = survey(m, a.min_snr, a.edge_mhz)
        segs.sort(key=lambda s: -s["snr"])
        gain = json.loads(m.read_text())["global"].get("curf:gain_db")
        print(f"== {band} {fc / 1e6:.2f} MHz  gain {gain} dB  peak {peak:.1f} dBFS  floor {floor:.1f} dB  segments {len(segs)}")
        for s in segs[:14]:
            print(f"   {s['mhz']:9.3f} MHz  {s['khz']:6d} kHz  SNR {s['snr']:5.1f} dB  duty {s['duty']:.2f}")


if __name__ == "__main__":
    main()
