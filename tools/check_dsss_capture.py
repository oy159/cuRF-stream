#!/usr/bin/env python3
"""Barker-11 coherent-energy evidence from IQ, independent of the classifier.

This is a PHY diagnostic, not a PLCP/FCS decoder or an accuracy label. Compare
against known DSSS and OFDM controls. No network identifiers are read or saved.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

BARKER = np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1], dtype=np.float32)


def coherent_energy(x, rate):
    """Best chip timing and sequence direction in independent 64-symbol spans.

    At 22 MS/s there are two samples/chip and 22 samples/symbol. Project each
    11-chip vector onto the unit-energy Barker sequence; divide by its input
    energy. Average 64 symbols before maximizing over timing/direction. Random
    IQ has an expected fraction near 1/11; matched DSSS approaches one.
    """
    ratio = Fraction(22e6 / rate).limit_denominator(1000)
    y = resample_poly(x, ratio.numerator, ratio.denominator).astype(np.complex64)
    count = (len(y) - 44) // (22 * 64)
    if count < 1:
        raise ValueError("need at least 66 microseconds of IQ")
    best = np.zeros(count, dtype=np.float32)
    for code in (BARKER, BARKER[::-1]):
        for phase in range(22):
            z = y[phase:phase + count * 64 * 22].reshape(count * 64, 11, 2)[:, :, 0]
            power = (np.abs(z) ** 2).sum(axis=1).reshape(count, 64).sum(axis=1)
            projected = (np.abs(z @ code) ** 2 / 11).reshape(count, 64).sum(axis=1)
            best = np.maximum(best, projected / np.maximum(power, 1e-20))
    return {"windows_64us": count, "max_coherent_fraction": float(best.max()),
            "above_0p5": int((best > .5).sum()),
            "quantiles_p50_p95_p99_p999": np.quantile(best, [.5, .95, .99, .999]).tolist()}


def inspect(meta_path, frequency=None, seconds=1):
    meta = json.loads(meta_path.read_text())
    g, capture = meta["global"], meta["captures"][0]
    rate, centre = g["core:sample_rate"], capture.get("core:frequency", 0.0)
    data = meta_path.with_suffix(".sigmf-data")
    if not data.exists():
        data = meta_path.with_suffix(".cf32") # OnAir's generator output
    n = int(rate * seconds)
    dtype = g["core:datatype"]
    if dtype == "ci16_le":
        raw = np.fromfile(data, dtype="<i2", count=2*n)
        x = raw.astype(np.float32).view(np.complex64) / 32768
    elif dtype == "cf32_le":
        x = np.fromfile(data, dtype="<c8", count=n)
    else:
        raise ValueError("supported SigMF datatypes: ci16_le, cf32_le")
    offset = 0 if frequency is None else frequency - centre
    if abs(offset) >= rate / 2:
        raise ValueError("requested centre is outside the recorded band")
    if offset:
        for a in range(0, len(x), 2**20):
            b = min(a+2**20, len(x))
            x[a:b] *= np.exp(-2j*np.pi*offset/rate*np.arange(a, b)).astype(np.complex64)
    h = hashlib.sha256()
    with data.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""): h.update(chunk)
    return {"capture": str(meta_path), "data_sha256": h.hexdigest(), "rate": rate,
            "frequency": centre + offset, "analysed_seconds": len(x)/rate,
            **coherent_energy(x, rate)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--frequency", type=float, help="RF centre to mix to DC before resampling")
    ap.add_argument("--seconds", type=float, default=1)
    ap.add_argument("--controls", type=Path, help="OnAir output directory containing wifi and wifi5")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if not .001 <= a.seconds <= 2:
        ap.error("analyse 0.001..2 seconds per input")
    report = {"real": inspect(a.capture, a.frequency, a.seconds),
              "limits": "Coherent Barker energy only; not packet decoding, CRC, or ground-truth classification accuracy."}
    if a.controls:
        report["controls"] = {name: inspect(a.controls/(name+".sigmf-meta"), seconds=a.seconds)
                              for name in ("wifi", "wifi5")}
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
