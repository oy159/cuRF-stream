#!/usr/bin/env python3
"""Convert an IQ recording (SigMF, .cf32, .cs16) to the pipeline's packed12 wire format for --source file.

    python3 tools/iq_to_wire.py in.sigmf-meta out.p12 [--peak-dbfs -6]

Prints the sample rate and centre frequency to pass to curf_classify (--rate, --freq). One stream.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path):
    rate, freq = None, 0.0
    if path.suffix in (".sigmf-meta", ".sigmf-data"):
        meta = json.loads(path.with_suffix(".sigmf-meta").read_text())
        g = meta["global"]
        rate = g["core:sample_rate"]
        freq = meta["captures"][0].get("core:frequency", 0.0)
        dt = g["core:datatype"]
        data = path.with_suffix(".sigmf-data")
    else:
        dt = {".cf32": "cf32_le", ".cs16": "ci16_le", ".sc16": "ci16_le"}[path.suffix]
        data = path
    if dt == "cf32_le":
        x = np.fromfile(data, dtype="<c8")
    elif dt == "ci16_le":
        x = np.fromfile(data, dtype="<i2").astype(np.float32).view(np.complex64)
    else:
        raise SystemExit(f"unsupported datatype {dt}")
    return x, rate, freq


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--peak-dbfs", type=float, default=-6, help="scale so the strongest sample is at this level")
    a = ap.parse_args()
    x, rate, freq = load(a.input)
    peak = np.abs(x).max()
    g = 10 ** (a.peak_dbfs / 20) / max(peak, 1e-12)
    i = np.clip(np.round(x.real * g * 2048), -2048, 2047).astype(np.int32) & 0xFFF
    q = np.clip(np.round(x.imag * g * 2048), -2048, 2047).astype(np.int32) & 0xFFF
    w = np.empty((len(x), 3), np.uint8)
    w[:, 0] = i >> 4
    w[:, 1] = ((i & 0xF) << 4) | (q >> 8)
    w[:, 2] = q & 0xFF
    w.tofile(a.output)
    print(json.dumps({"samples": len(x), "rate": rate, "freq": freq, "output": str(a.output)}))


if __name__ == "__main__":
    main()
