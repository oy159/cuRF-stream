#!/usr/bin/env python3
"""Inspect IQ exported by curf_check_extraction; optionally reproduce the old independent FFT crops.

    python3 tools/check_extraction.py /tmp/extraction --legacy

All .cf32 files are little-endian complex64, before RMS normalisation. The CPU reference is a continuous
direct FIR; the CUDA output is blocked overlap-save. --legacy additionally reconstructs signal_stage.cu's
old crop / raised-cosine / IFFT path on a steady tone with a bin-aligned centre (no model required).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def inspect(path: Path, legacy: bool = False) -> dict:
    m = json.loads((path / "meta.json").read_text())
    y = np.fromfile(path / "cuda.cf32", dtype="<c8").astype(np.complex128)
    ref = np.fromfile(path / "reference.cf32", dtype="<c8").astype(np.complex128)
    raw = np.fromfile(path / "input.cf32", dtype="<c8").astype(np.complex128)
    step, block, d = m["output_frames_per_block"], m["block_frames"], m["decimation"]
    if len(y) != m["blocks"] * step or len(ref) != len(y) or len(raw) != m["blocks"] * block:
        raise ValueError("IQ length does not match metadata")
    if not (np.isfinite(y).all() and np.isfinite(ref).all() and np.isfinite(raw).all()):
        raise ValueError("non-finite IQ")
    err = np.abs(y - ref)
    pos = np.arange(len(y)) % step
    seam = (pos < 4) | (pos >= step - 4)
    result = {"decimation": d, "signal": m["signal"], "wire": m["wire"],
              "output_rate": m["output_rate"], "group_delay_ms": 1000 * m["delay_input_frames"] / m["sample_rate"],
              "max_abs_error": float(err.max()), "rms_error": float(np.sqrt(np.mean(err ** 2))),
              "seam_max_error": float(err[seam].max()), "passed": bool(err.max() <= 3e-6)}
    if legacy:
        if m["signal"] != "tone" or step < 16 or step % 2:
            raise ValueError("--legacy needs a steady tone and an even output block >= 16")
        centre_bin = m["centre_hz"] * block / m["sample_rate"]
        if abs(centre_bin - round(centre_bin)) > 1e-8:
            raise ValueError("--legacy needs a bin-aligned centre to separate seam errors from LO quantisation")
        kc = round(centre_bin)
        t = np.arange(step)
        edge = max(1, step // 16)
        dist = np.minimum(t, step - 1 - t)
        g = np.ones(step)
        taper = dist < edge
        g[taper] = .5 - .5 * np.cos(np.pi * (dist[taper] + .5) / edge)
        spectra = np.fft.fft(raw.reshape(m["blocks"], block), axis=1)
        bands = spectra[:, (kc - step // 2 + t) % block] * g
        old = (np.fft.ifft(np.roll(bands, -step // 2, axis=1), axis=1) * step / block).reshape(-1)
        # Cropping mixes relative to each block. Remove its constant initial LO phase for a fair comparison.
        old *= np.exp(-2j * np.pi * np.remainder(m["centre_hz"] / m["sample_rate"] * m["first_sample"], 1))
        times = m["first_sample"] / d + np.arange(len(y))
        ideal_old = .25 * np.exp(2j * np.pi * .113 * times)
        ideal_new = .25 * np.exp(2j * np.pi * .113 * (times - m["delay_input_frames"] / d))
        # Exclude the FIR's explicitly defined zero-history startup, including when it spans several blocks.
        valid = np.arange(len(y)) >= (m["fir_taps"] - 1) // d
        for name, values, ideal in (("legacy", old, ideal_old), ("overlap_save", y, ideal_new)):
            e = np.abs(values - ideal)
            result[f"{name}_ideal_rms"] = float(np.sqrt(np.mean(e[valid] ** 2)))
            result[f"{name}_seam_ideal_rms"] = float(np.sqrt(np.mean(e[valid & seam] ** 2)))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--legacy", action="store_true")
    a = ap.parse_args()
    result = inspect(a.directory, a.legacy)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
