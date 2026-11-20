"""Offline detection and IQ extraction for the demo model.

The detector reports regions in source samples and absolute Hz. Extraction
keeps each region on its original timeline and passes the whole occupied band
to the model, with a bounded number of windows for long transmissions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy import fft, ndimage, signal


@dataclass(frozen=True)
class Candidate:
    start: int
    stop: int
    center_hz: float
    bandwidth_hz: float


def _components(active, ratio, frequencies, step, row_samples, sample_rate, threshold_db, analog, wide=False):
    shape = (2, 12 if wide else 4)
    padded = np.pad(active, ((1, 1), (shape[1], shape[1])))
    joined = ndimage.binary_closing(padded, structure=np.ones(shape))
    labels, _ = ndimage.label(joined[1:-1, shape[1]:-shape[1]])
    candidates = []
    for index, (times, bins) in enumerate(ndimage.find_objects(labels), start=1):
        width = (bins.stop - bins.start) * step
        duration = (times.stop - times.start) * row_samples / sample_rate
        if bins.stop - bins.start < 3 or (analog and (duration < 0.9e-3 or width > 360e3)):
            continue
        if wide and width < 6e6:
            continue
        if bins.start == 0 or bins.stop == len(frequencies):
            continue
        peak = np.max(ratio[times, bins], where=labels[times, bins] == index, initial=0)
        if peak < 10 ** ((threshold_db + 3) / 10):
            continue
        low = frequencies[bins.start] - step / 2
        high = frequencies[bins.stop - 1] + step / 2
        candidates.append(Candidate(times.start * row_samples, times.stop * row_samples, (low + high) / 2, width))
    return candidates


def extract_iq(
    samples: np.ndarray,
    sample_rate: float,
    tune_hz: float,
    candidate: Candidate,
    length: int = 1024,
    max_windows: int = 3,
) -> np.ndarray:
    """Return unit-RMS [windows, 2, length] from left-aligned SC16 samples."""
    if samples.ndim != 2 or samples.shape[1] != 2 or samples.dtype != np.dtype("<i2"):
        raise ValueError("expected little-endian SC16 samples as I/Q rows")
    if not 0 <= candidate.start < candidate.stop <= len(samples):
        raise ValueError("candidate is outside the capture")
    bandwidth = candidate.bandwidth_hz
    offset = candidate.center_hz - tune_hz
    if not np.isfinite([sample_rate, offset, bandwidth]).all() or sample_rate <= 0 or bandwidth <= 0:
        raise ValueError("invalid sample rate or candidate frequency")
    if abs(offset) + bandwidth / 2 > sample_rate / 2 or length < 1 or max_windows < 1:
        raise ValueError("candidate exceeds source bandwidth or window limits")

    # Keep useful modulation detail and transition bandwidth on either side.
    target_rate = min(sample_rate, max(192_000, bandwidth * 3.6))
    down = max(1, round(sample_rate / target_rate))
    # The guard is a multiple of the decimation stride, preserving window alignment.
    guard = 12 * down
    cutoff = min(bandwidth * 0.69, 0.43 * sample_rate / down)
    taps = signal.firwin(2 * guard + 1, cutoff, fs=sample_rate, window=("kaiser", 7.0)).astype(np.float32)
    source_width = length * down
    extent = candidate.stop - candidate.start
    count = min(max_windows, max(1, ceil(extent / source_width)))
    if count == 1:
        starts = [(candidate.start + candidate.stop - source_width) // 2]
    else:
        starts = np.rint(np.linspace(candidate.start, candidate.stop - source_width, count)).astype(int)

    result = []
    pre = guard + down
    for start in starts:
        begin = int(start) - pre
        end = int(start) + source_width + pre
        lo, hi = max(0, begin), min(len(samples), end)
        raw = samples[lo:hi].astype(np.float32) / 32768
        iq = raw[:, 0] + 1j * raw[:, 1]
        phase = (-2j * np.pi * offset / sample_rate) * np.arange(lo - begin, hi - begin)
        wave = np.zeros(end - begin, dtype=np.complex64)
        wave[lo - begin:hi - begin] = iq * np.exp(phase)
        if down == 1:
            cropped = signal.fftconvolve(wave, taps, mode="same")[pre:pre + length]
        else:
            first = pre // down
            cropped = signal.resample_poly(wave, 1, down, window=taps)[first:first + length]
        rms = max(float(np.mean(abs(cropped) ** 2)) ** 0.5, 1e-8)
        result.append(np.stack((cropped.real, cropped.imag)).astype(np.float32) / rms)
    return np.stack(result)


def detect_candidates(
    samples: np.ndarray,
    sample_rate: float,
    tune_hz: float,
    *,
    usable_band_hz: float,
    threshold_db: float = 12.0,
    exclude_hz: tuple[float, ...] = (),
    analog: bool = False,
) -> list[Candidate]:
    """Find time-frequency components in a short recorded scene."""
    size = max(64, 2 ** ceil(np.log2(sample_rate / 2500))) if analog else (1024 if sample_rate >= 10e6 else 512)
    group = 1 if analog else 2
    row_samples = size * group
    rows = len(samples) // row_samples
    if not rows:
        return []
    step = sample_rate / size
    offsets = fft.fftshift(fft.fftfreq(size, d=1 / sample_rate))
    keep = abs(offsets) <= usable_band_hz / 2
    frequencies = tune_hz + offsets[keep]
    taper = signal.windows.hann(size, sym=False).astype(np.float32)
    power = np.empty((rows, len(frequencies)), dtype=np.float32)
    for first in range(0, rows, 512):
        last = min(first + 512, rows)
        raw = samples[first * row_samples:last * row_samples].astype(np.float32)
        x = (raw[:, 0] + 1j * raw[:, 1]).reshape(-1, size)
        mag = abs(fft.fftshift(fft.fft(x * taper, axis=1), axes=1)[:, keep]) ** 2
        power[first:last] = mag.reshape(last - first, group, -1).mean(axis=1)

    floor = np.percentile(power, 20, axis=0)
    floor_span = (2 * ceil(450e3 / step) + 1) if analog else 17
    floor = np.minimum(floor, ndimage.percentile_filter(floor, 25, size=floor_span) * 2.0)
    ratio = power / np.maximum(floor, 1e-7)
    active = ratio > 10 ** (threshold_db / 10)
    for spur in exclude_hz:
        active[:, abs(frequencies - spur) < 1.6 * step] = False
    candidates = _components(active, ratio, frequencies, step, row_samples, sample_rate, threshold_db, analog)
    if not analog and usable_band_hz >= 8e6:
        # Strong wide bursts can raise most bins above the noise threshold.
        # Recover their in-band core without merging it with skirt leakage.
        row_reference = np.percentile(ratio, 90, axis=1)
        wide_mask = active & (ratio >= row_reference[:, None] / 16)
        cores = _components(wide_mask, ratio, frequencies, step, row_samples, sample_rate,
                            threshold_db, analog=False, wide=True)
        candidates = [c for c in candidates if c.bandwidth_hz < 6e6 or not any(
            c.start < k.stop and k.start < c.stop
            and abs(c.center_hz - k.center_hz) <= 0.45 * c.bandwidth_hz
            for k in cores
        )] + cores
    return sorted(candidates, key=lambda c: (c.start, c.center_hz))
