"""Locate Barker-11 intervals for Wi-Fi DSSS replay labels."""

from __future__ import annotations

import numpy as np
from scipy import fft, ndimage, signal

from .candidates import Candidate

BARKER = np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1], dtype=np.float32)
CHIP_RATE = 11_000_000


def _chip_scores(iq: np.ndarray) -> np.ndarray:
    samples_per_symbol = 22
    count = (len(iq) - samples_per_symbol) // samples_per_symbol * samples_per_symbol
    energy_taps = np.zeros(21, dtype=np.float32)
    energy_taps[::2] = 1
    energy = signal.oaconvolve(abs(iq) ** 2, energy_taps, mode="valid")[:count].reshape(-1, 22)
    usable = np.maximum(ndimage.uniform_filter1d(energy, size=32, axis=0) * 11, 1e-8)
    best = np.zeros(len(usable), dtype=np.float32)
    for code in (BARKER, BARKER[::-1]):
        taps = energy_taps.copy()
        taps[::2] = code[::-1]
        corr = signal.oaconvolve(iq, taps, mode="valid")[:count].reshape(-1, 22)
        coherence = ndimage.uniform_filter1d(abs(corr) ** 2, size=32, axis=0) / usable
        best = np.maximum(best, np.max(coherence, axis=1))
    return best


def find_barker(samples: np.ndarray, rate: float, tune: float, usable_band: float) -> list[Candidate]:
    """Return short DSSS intervals at supported 2.4 GHz channel centers."""
    if not 2390e6 <= tune <= 2495e6 or rate < 40e6:
        return []
    centers = [2412e6 + ch * 5e6 for ch in range(13)
               if abs(2412e6 + ch * 5e6 - tune) + 10e6 <= usable_band / 2]
    if not centers:
        return []
    tile = round(0.090 * rate)
    guard = round(100e-6 * rate)
    matches = []
    for first in range(0, len(samples), tile):
        last = min(first + tile, len(samples))
        lo, hi = max(0, first - guard), min(last + guard, len(samples))
        # An even FFT length keeps the positive and negative slices symmetric.
        hi -= (hi - lo) % 2
        if hi - lo < 2 * guard:
            continue
        z = samples[lo:hi].astype(np.float32) / 32768
        spectrum = fft.fft(z[:, 0] + 1j * z[:, 1])
        size = 2 * round((hi - lo) * CHIP_RATE / rate)
        signed = np.r_[np.arange(size // 2), np.arange(-size // 2, 0)]
        for center in centers:
            shift = round((center - tune) * (hi - lo) / rate)
            iq = fft.ifft(np.take(spectrum, shift + signed, mode="wrap")) * (size / (hi - lo))
            scores = _chip_scores(iq)
            edges = np.flatnonzero(np.diff(np.r_[False, scores >= .78, False])).reshape(-1, 2)
            for start, stop in edges:
                if stop - start < 16:
                    continue
                a = lo + round(start * rate / 1e6)
                b = min(len(samples), lo + round(stop * rate / 1e6))
                if a < first or a >= last or b <= a:
                    continue
                matches.append(Candidate(a, b, center, 17e6))
    return sorted(matches, key=lambda c: (c.start, c.center_hz))
