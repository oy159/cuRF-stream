"""Signal-level dataset: emissions cut from clean protocol recordings, extracted the way the runtime will.

Stage 2 of the classifier sees one detected signal at a time, not a fixed 120 kHz channel: the detector gives a
centre and a bandwidth B, the signal is mixed to 0 Hz and resampled to fs = 2 * B, and a window of L samples goes
to the model with log10(B) as a side input. So every signal fills about half of the model's band, whatever its
width (a 9 kHz DMR carrier or an 18 MHz WiFi burst), and the modulation structure is visible at every scale.

This module builds such examples from clean recordings (OnAir's generators, tools/onair_gen.py):

1. emissions(): connected components of the recording's time-frequency mask (STFT, threshold over the noise
   floor, closing): one per burst / carrier / hop, with its centre, 99 % bandwidth and time span. Multi-channel
   scenes (airband, AIS, GSM, hopping Bluetooth) give one emission per transmitter and burst.
2. extract(): what the runtime does with a detection, with its errors: bandwidth estimate x U(0.7, 1.4), centre
   error +-10 % of B, window placed anywhere that overlaps the emission.
3. impair(): channel and receiver after extraction: residual CFO, phase noise, multipath (rms delay spread up to
   1 us, which matters only for wide signals), IQ imbalance, DC offset, white noise at an in-band SNR.
Windows are normalised to unit RMS, like the channel windows of stage 1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.signal import resample

# Stage-2 classes. Families are by modulation, not protocol: the protocol (WiFi vs DTMB, DMR vs P25) is named
# afterwards by rules on bandwidth, centre frequency and burst timing.
FAMILIES = ["noise", "ofdm", "fsk", "4fsk", "psk", "qam", "dsss", "chirp", "ook", "am", "fm", "pulse", "unknown"]

# tools/onair_gen.py family -> stage-2 class
ONAIR_TO_FAMILY = {
    "ofdm": "ofdm", "gmsk": "fsk", "gfsk": "fsk", "cpfsk": "fsk", "fsk": "fsk", "4fsk": "4fsk",
    "pi4dqpsk": "psk", "d8psk": "psk", "psk": "psk", "qam": "qam", "oqpsk": "dsss", "dsss": "dsss",
    "chirp": "chirp", "ook": "ook", "am": "am", "am_msk": "am", "fm": "fm", "pulse": "pulse", "unknown": "unknown",
}


@dataclass
class Emission:
    source: str      # recording id
    fc: float        # centre, Hz from the recording's centre
    bw: float        # 99 % occupied bandwidth, Hz
    t0: int          # first / last sample
    t1: int
    snr_db: float    # in the clean recording (sanity only)


def load_sigmf(meta_path: Path) -> tuple[np.ndarray, float]:
    g = json.loads(meta_path.read_text())["global"]
    data = meta_path.with_suffix(".sigmf-data")
    if not data.exists():
        data = meta_path.with_name(g.get("core:dataset", meta_path.stem + ".cf32"))
    dt = g["core:datatype"]
    if dt == "cf32_le":
        x = np.fromfile(data, dtype="<c8")
    elif dt == "ci16_le":
        x = np.fromfile(data, dtype="<i2").astype(np.float32).view(np.complex64) / 32768
    else:
        raise ValueError(f"{meta_path}: datatype {dt}")
    return x, float(g["core:sample_rate"])


def emissions(x: np.ndarray, fs: float, source: str, nfft: int | None = None, thresh_db: float = 20.0,
              range_db: float = 45.0, min_cells: int = 4, smooth_t: int = 5, smooth_f: int = 5,
              rel_db: float = 25.0, floor_pct: float = 5.0, noise_psd: np.ndarray | None = None,
              merge_gap_s: float = 0.0, absorb_narrow: bool = True) -> list[Emission]:
    """Bursts / carriers of a clean recording as connected time-frequency components.

    A cell belongs to an emission when it is thresh_db over the noise floor (5th percentile) and within range_db
    of the strongest cell: generators that are silent between bursts have no floor at all, and window
    sidelobes of a strong carrier would otherwise join every emitter of a multi-channel scene into one."""
    if nfft is None:
        nfft = 4096 if fs <= 2.5e6 else 1024  # ~0.5 kHz bins for the narrow-band recordings
    hop = nfft
    nfr = len(x) // hop - 1
    if nfr < 2:
        return []
    win = np.hanning(nfft).astype(np.float32)
    S = np.empty((nfr, nfft), np.float32)  # [time, freq]
    for i in range(0, nfr, 2048):
        k = np.arange(i, min(nfr, i + 2048))
        seg = x[(k[:, None] * hop + np.arange(nfft)[None, :])] * win
        S[k] = np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2
    # Smooth over ~5 frames x 5 bins first: test signals of OFDM systems (LTE, NR, DAB) leave most resource
    # elements empty, and the raw cells would cut one carrier into thousands of fragments.
    if noise_psd is not None:  # whiten by the receiver's own noise spectrum (a capture with no input signal)
        S /= noise_psd[None, :]
    raw = S
    S = ndimage.uniform_filter(S, size=(smooth_t, smooth_f), mode="constant")
    floor = float(np.percentile(S, floor_pct))
    mask = S > max(floor * 10 ** (thresh_db / 10), float(S.max()) * 10 ** (-range_db / 10))
    # bridge OFDM nulls / pilots, PPM gaps and the gaps between FSK tones; one emission stays one component
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 5), bool))
    lab, n = ndimage.label(mask)
    smax = float(S.max())
    out = []
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        comp = lab[sl] == k
        if comp.sum() < min_cells or float(S[sl][comp].max()) < smax * 10 ** (-rel_db / 10):
            continue  # too small, or a sidelobe / weak fragment next to a real emission
        ts, fsl = sl
        # 99 % bandwidth of the mean spectrum over the component's time span, within the contiguous region around
        # its peak that stays above both 30 dB under that peak and the floor: the mask itself only covers the top
        # of a weak emission's spectrum (it is cut at range_db under the strongest one)
        p = S[ts].mean(0)
        pk = fsl.start + int(np.argmax(p[fsl]))
        lim = max(p[pk] * 1e-3, floor * 3)
        a, b = pk, pk + 1
        while a > 0 and p[a - 1] > lim:
            a -= 1
        while b < nfft and p[b] > lim:
            b += 1
        p = p[a:b]
        c = np.cumsum(p) / p.sum()
        lo, hi = a + np.searchsorted(c, 0.005), a + np.searchsorted(c, 0.995)
        bw = max((hi - lo + 1) * fs / nfft, fs / nfft)
        fc = float((f[lo] + f[min(hi, nfft - 1)]) / 2)
        snr = 10 * np.log10(p.max() / max(floor, 1e-30))
        # time span from the unsmoothed in-band power: the smoothing widened it by ~2 frames on each side, which
        # for short bursts (WiFi OFDM, 0.1-1 ms) would put many windows on silence labelled as the signal
        e_t = raw[ts, lo:hi + 1].sum(1)
        on = np.nonzero(e_t > 0.05 * np.percentile(e_t, 90))[0]
        f0, f1 = (ts.start + on[0], ts.start + on[-1]) if len(on) else (ts.start, ts.stop - 1)
        out.append(Emission(source, fc, float(bw), int(f0 * hop), int(f1 * hop + nfft), float(snr)))
    # pieces of one emission (same centre and width, overlapping or touching in time) become one
    out.sort(key=lambda e: (e.t0, e.fc))
    merged: list[Emission] = []
    for e in out:
        for m in merged:
            gap = nfft + int(merge_gap_s * fs)  # real captures: LTE leaves whole subframes nearly empty at low load
            if (e.t0 <= m.t1 + gap and m.t0 <= e.t1 + gap and abs(e.fc - m.fc) < 0.25 * max(e.bw, m.bw)
                    and 0.67 < e.bw / m.bw < 1.5):
                m.t0, m.t1, m.bw = min(m.t0, e.t0), max(m.t1, e.t1), max(m.bw, e.bw)
                break
        else:
            merged.append(e)
    out = merged
    # a narrow piece overlapping a much wider emission in time, inside or next to it in frequency, is part of it
    keep = []
    for e in out:
        part = absorb_narrow and any(E is not e and E.bw > 3 * e.bw and abs(e.fc - E.fc) < E.bw and e.t0 < E.t1 and E.t0 < e.t1
                   for E in out)
        if not part:
            keep.append(e)
    return keep


def noise_spectrum(x: np.ndarray, nfft: int, frames: int = 4000) -> np.ndarray:
    """Median power per bin (fftshifted) of a capture with no signal, for emissions(noise_psd=...)."""
    n = min(frames, len(x) // nfft)
    seg = x[: n * nfft].reshape(n, nfft) * np.hanning(nfft).astype(np.float32)
    return np.median(np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2, axis=0).astype(np.float32)


RUNTIME_BIN_HZ = 61.44e6 / 2 ** 18  # FFT bin of the runtime extraction (src/runtime/signal_stage.cu)


def runtime_rate(b_est: float, bin_hz: float = RUNTIME_BIN_HZ, nf: int = 2 ** 18) -> float:
    """Extraction rate the runtime picks for a bandwidth estimate: W bins, the smallest of {2^m, 1.5 * 2^m}
    (16 <= W <= nf) with W >= 2 B / bin, so fs_n = W * bin is 2..3 B (and at least 3.75 kHz)."""
    need = 2 * b_est / bin_hz
    w = 16
    while True:
        if w >= need or w >= nf:
            return min(w, nf) * bin_hz
        if w + w // 2 >= need:
            return (w + w // 2) * bin_hz
        w *= 2


def extract(x: np.ndarray, fs: float, e: Emission, L: int, rng: np.random.Generator,
            bw_err=(0.7, 1.4), fc_err: float = 0.1, min_fs: float = 0.0) -> tuple[np.ndarray, float]:
    """The runtime's extraction with its estimation errors: returns (L complex samples at fs_n, B estimate)."""
    b_est = e.bw * float(np.exp(rng.uniform(np.log(bw_err[0]), np.log(bw_err[1]))))
    fs_n = max(runtime_rate(b_est), min_fs)
    fc = e.fc + rng.uniform(-fc_err, fc_err) * e.bw
    n0 = int(np.ceil(L / fs_n * fs)) + 1
    span = e.t1 - e.t0
    if span >= n0:
        start = int(rng.integers(e.t0, e.t1 - n0 + 1))
    else:  # the whole emission inside the window, anywhere
        start = int(rng.integers(e.t1 - n0, e.t0 + 1))
    seg = np.zeros(n0, np.complex64)
    a, b = max(start, 0), min(start + n0, len(x))
    if b > a:
        seg[a - start:b - start] = x[a:b]
    t = (np.arange(n0) + start) / fs
    seg = seg * np.exp(-2j * np.pi * fc * t).astype(np.complex64)
    y = resample(seg, L)  # FFT resampling: brick-wall low-pass at +-fs_n / 2 (up or down)
    return y.astype(np.complex64), b_est


def impair(y: np.ndarray, fs_n: float, snr_db: float, rng: np.random.Generator, bw: float | None = None) -> np.ndarray:
    L = len(y)
    active = np.abs(y) ** 2
    thr = active.max() * 1e-2
    p_sig = float(active[active > thr].mean()) if (active > thr).any() else 1.0
    y = y / np.sqrt(p_sig)
    # multipath: exponential power-delay profile at the extracted rate
    tau = rng.choice([0.0, rng.uniform(20e-9, 1e-6)]) * fs_n
    if tau > 0.3:
        n = int(min(16, np.ceil(4 * tau))) + 1
        h = (rng.normal(size=n) + 1j * rng.normal(size=n)) * np.exp(-np.arange(n) / (2 * tau))
        h /= np.linalg.norm(h)
        y = np.convolve(y, h)[:L]
    # residual frequency offset, phase noise (random walk), IQ imbalance
    t = np.arange(L)
    y = y * np.exp(1j * (2 * np.pi * rng.uniform(-0.02, 0.02) * t + rng.uniform(0, 2 * np.pi)))
    lw = rng.uniform(0, 100.0)
    y = y * np.exp(1j * np.cumsum(rng.normal(0, np.sqrt(2 * np.pi * lw / fs_n), L)))
    eps, phi = rng.uniform(-0.05, 0.05), np.deg2rad(rng.uniform(-3, 3))
    mu = np.cos(phi) + 1j * eps * np.sin(phi)
    nu = eps * np.cos(phi) - 1j * np.sin(phi)
    y = mu * y + 0.5 * nu * np.conj(y)
    # white noise: SNR is in-band (over the bandwidth estimate bw; fs_n / 2 when not given), so the noise over
    # fs_n is fs_n / bw times the in-band noise
    n_tot = (fs_n / bw if bw else 2.0) * 10 ** (-snr_db / 10)
    noise = (rng.normal(size=L) + 1j * rng.normal(size=L)) * np.sqrt(n_tot / 2)
    y = y + noise + (rng.normal() + 1j * rng.normal()) * rng.uniform(0, 0.05) * np.sqrt(n_tot)
    return (y / np.sqrt(np.mean(np.abs(y) ** 2))).astype(np.complex64)


def noise_window(L: int, rng: np.random.Generator) -> np.ndarray:
    y = (rng.normal(size=L) + 1j * rng.normal(size=L)) / np.sqrt(2)
    eps, phi = rng.uniform(-0.05, 0.05), np.deg2rad(rng.uniform(-3, 3))
    y = (np.cos(phi) + 1j * eps * np.sin(phi)) * y + 0.5 * (eps * np.cos(phi) - 1j * np.sin(phi)) * np.conj(y)
    y = y + (rng.normal() + 1j * rng.normal()) * rng.uniform(0, 0.05)
    return (y / np.sqrt(np.mean(np.abs(y) ** 2))).astype(np.complex64)
