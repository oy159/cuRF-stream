"""On-the-fly GPU generator of labelled narrowband windows.

Each window imitates one output channel of the polyphase channelizer:
complex baseband at fs = 240 kS/s (512 channels x 120 kHz spacing over
61.44 MHz, 2x oversampled), 1024 samples (4.27 ms). A signal fits inside one
channel spacing: occupied bandwidth <= fs / 2 = 120 kHz.

Per window: random class, symbol rate (integer samples/symbol), roll-off,
timing offset, carrier phase, carrier frequency offset, optional short
multipath, and AWGN at a random SNR. SNR is measured over the full fs band:
signal power / noise power per complex sample.

Channel filter: every window (signal and noise) goes through the channelizer's
prototype response (Kaiser sinc, -6 dB at 90 kHz, ~75 kHz transition), so noise
is shaped like real channel output instead of white across the 240 kHz.

"wide": a slice of a signal wider than one channel (0.24-3.8 MHz GFSK, PSK/QAM,
OFDM, e.g. BLE or WiFi seen through one 120 kHz channel). Generated at R x 240
kS/s, shifted so the channel lands anywhere inside the signal band, filtered by
the prototype response at that rate and decimated. Real detections of such
signals used to be forced into a narrowband class (8psk, fm, ofdm).

"fm" includes broadcast FM (40-75 kHz deviation, stereo MPX with 19 kHz pilot),
which overfills the channel and is clipped by the channel filter.

Output windows are normalised to unit RMS, exactly as the GPU pre-processing
stage will do before inference.

Generation is batched in chunks that share the slow-to-vary parameters
(class, samples/symbol, roll-off) so everything stays vectorised on the GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

CLASSES: list[str] = ["noise", "bpsk", "qpsk", "8psk", "16qam", "64qam", "gfsk", "am", "fm", "ofdm", "wide"]
NARROW: list[str] = [c for c in CLASSES if c not in ("noise", "wide")]  # what fits in one channel
LINEAR = {"bpsk", "qpsk", "8psk", "16qam", "64qam"}


@dataclass
class SynthConfig:
    length: int = 1024
    fs: float = 240e3
    snr_db: tuple[float, float] = (-4.0, 24.0)
    sps: tuple[int, int] = (3, 24)                # samples per symbol, inclusive
    rolloff: tuple[float, float] = (0.2, 0.5)
    cfo_hz: tuple[float, float] = (-10e3, 10e3)
    multipath_prob: float = 0.5
    multipath_taps: int = 4                       # max extra taps
    multipath_delay: int = 8                      # max delay in samples
    chunk: int = 32                               # windows sharing class/sps/roll-off
    channel_filter: bool = True                   # prototype response on signal and noise
    broadcast_fm_prob: float = 0.5
    wide_rates: tuple[int, ...] = (4, 8, 16)      # generation rate for "wide", in units of fs
    classes: list[str] = field(default_factory=lambda: list(CLASSES))


def constellation(name: str, device) -> torch.Tensor:
    """Unit average power constellation points (complex64)."""
    if name == "bpsk":
        pts = torch.tensor([1.0 + 0j, -1.0 + 0j])
    elif name in ("qpsk", "8psk"):
        m = 4 if name == "qpsk" else 8
        k = torch.arange(m)
        pts = torch.exp(1j * (2 * math.pi * k / m + (math.pi / 4 if m == 4 else 0.0)))
    elif name in ("16qam", "64qam"):
        m = 4 if name == "16qam" else 8
        lv = torch.arange(m) * 2.0 - (m - 1)
        pts = (lv[:, None] + 1j * lv[None, :]).flatten()
    else:
        raise ValueError(name)
    pts = pts.to(torch.complex64)
    return (pts / pts.abs().pow(2).mean().sqrt()).to(device)


def rrc_taps(sps: int, beta: float, span: int = 8) -> torch.Tensor:
    """Root-raised-cosine taps, unit energy, length span*sps+1."""
    t = (torch.arange(span * sps + 1, dtype=torch.float64) - span * sps / 2) / sps
    taps = torch.empty_like(t)
    for i, ti in enumerate(t.tolist()):
        if abs(ti) < 1e-12:
            taps[i] = 1 - beta + 4 * beta / math.pi
        elif beta > 0 and abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            taps[i] = beta / math.sqrt(2) * ((1 + 2 / math.pi) * math.sin(math.pi / (4 * beta)) +
                                             (1 - 2 / math.pi) * math.cos(math.pi / (4 * beta)))
        else:
            num = math.sin(math.pi * ti * (1 - beta)) + 4 * beta * ti * math.cos(math.pi * ti * (1 + beta))
            den = math.pi * ti * (1 - (4 * beta * ti) ** 2)
            taps[i] = num / den
    return (taps / taps.norm()).to(torch.float32)


def gaussian_taps(sps: int, bt: float, span: int = 4) -> torch.Tensor:
    """Gaussian frequency-pulse filter for GFSK, unit DC gain."""
    t = (torch.arange(span * sps + 1, dtype=torch.float64) - span * sps / 2) / sps
    sigma = math.sqrt(math.log(2)) / (2 * math.pi * bt)
    g = torch.exp(-t ** 2 / (2 * sigma ** 2))
    return (g / g.sum()).to(torch.float32)


def channel_taps(rate: int = 1, fs: float = 240e3, cutoff_hz: float = 90e3, beta: float = 8.0) -> torch.Tensor:
    """The channelizer prototype (4096 taps, Kaiser beta 8, -6 dB at 0.75 x 120 kHz at 61.44 MHz) resampled to
    rate * fs: same duration (66.7 us), so the same frequency response. Unit DC gain."""
    n = 16 * rate + 1  # 4096 * (rate * fs) / 61.44e6
    t = torch.arange(n, dtype=torch.float64) - (n - 1) / 2
    fc = cutoff_hz / (rate * fs)
    h = torch.where(t == 0, torch.full_like(t, 2 * fc), torch.sin(2 * math.pi * fc * t) / (math.pi * t))
    r = 2 * t / (n - 1)
    w = torch.special.i0(beta * torch.sqrt((1 - r * r).clamp_min(0))) / torch.special.i0(torch.tensor(beta, dtype=torch.float64))
    h = h * w
    return (h / h.sum()).to(torch.float32)


def fir_complex(x: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """'Same'-length FIR of complex rows x [n, N] with real taps."""
    k = taps.numel()
    w = taps.flip(0).view(1, 1, k).to(x.device)
    xr = torch.view_as_real(x).permute(0, 2, 1).reshape(-1, 1, x.shape[-1])  # [2n, 1, N]
    y = F.conv1d(xr, w, padding=k // 2)[..., : x.shape[-1]]
    y = y.reshape(x.shape[0], 2, -1).permute(0, 2, 1).contiguous()
    return torch.view_as_complex(y)


class Synth:
    def __init__(self, cfg: SynthConfig | None = None, device: str | torch.device = "cuda", seed: int = 0):
        self.cfg = cfg or SynthConfig()
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.cpu_gen = torch.Generator().manual_seed(seed + 1)  # for per-chunk scalar choices
        self.const = {c: constellation(c, self.device) for c in LINEAR}
        self.class_index = {c: i for i, c in enumerate(CLASSES)}
        self.h1 = channel_taps(1).to(self.device)
        self.h_wide = {r: channel_taps(r).to(self.device) for r in self.cfg.wide_rates}

    # ------------------------------------------------------------------ utils
    def _u(self, n: int, lo: float, hi: float) -> torch.Tensor:
        return torch.rand(n, generator=self.gen, device=self.device) * (hi - lo) + lo

    def _cu(self, lo: float, hi: float) -> float:
        return float(torch.rand(1, generator=self.cpu_gen)) * (hi - lo) + lo

    def _ci(self, lo: int, hi: int) -> int:  # inclusive
        return int(torch.randint(lo, hi + 1, (1,), generator=self.cpu_gen))

    def _cn(self, n: int, length: int) -> torch.Tensor:
        z = torch.randn(n, length, 2, generator=self.gen, device=self.device)
        return torch.view_as_complex(z) * math.sqrt(0.5)

    def _message(self, n: int, length: int) -> torch.Tensor:
        """Audio-like message in [-1, 1]: 1-3 tones between 100 Hz and 5 kHz."""
        fs = self.cfg.fs
        t = torch.arange(length, device=self.device) / fs
        m = torch.zeros(n, length, device=self.device)
        for _ in range(3):
            f = self._u(n, 100.0, 5000.0)[:, None]
            a = self._u(n, 0.0, 1.0)[:, None]
            ph = self._u(n, 0.0, 2 * math.pi)[:, None]
            m += a * torch.sin(2 * math.pi * f * t + ph)
        return m / m.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)

    # ------------------------------------------------------------ modulators
    def _linear(self, name: str, n: int, length: int, sps: int, beta: float) -> torch.Tensor:
        taps = rrc_taps(sps, beta)
        nsym = length // sps + 2 * (taps.numel() // sps) + 2
        pts = self.const[name]
        idx = torch.randint(0, pts.numel(), (n, nsym), generator=self.gen, device=self.device)
        up = torch.zeros(n, nsym * sps, dtype=torch.complex64, device=self.device)
        up[:, ::sps] = pts[idx]
        return fir_complex(up, taps) * math.sqrt(sps)

    def _gfsk(self, n: int, length: int, sps: int) -> torch.Tensor:
        h = self._cu(0.5, 1.0)
        bt = self._cu(0.3, 0.5)
        nsym = length // sps + 8
        bits = torch.randint(0, 2, (n, nsym), generator=self.gen, device=self.device).float() * 2 - 1
        freq = bits.repeat_interleave(sps, dim=1)
        g = gaussian_taps(sps, bt).to(self.device)
        freq = F.conv1d(freq[:, None], g.flip(0).view(1, 1, -1), padding=g.numel() // 2)[:, 0, : freq.shape[1]]
        phase = torch.cumsum(math.pi * h / sps * freq, dim=1)
        return torch.polar(torch.ones_like(phase), phase)

    def _am(self, n: int, length: int) -> torch.Tensor:
        mu = self._u(n, 0.3, 1.0)[:, None]
        return (1 + mu * self._message(n, length)).to(torch.complex64)

    def _fm(self, n: int, length: int) -> torch.Tensor:
        if self._cu(0, 1) < self.cfg.broadcast_fm_prob:
            return self._broadcast_fm(n, length)
        dev = self._u(n, 2e3, 25e3)[:, None]
        phase = torch.cumsum(2 * math.pi * dev / self.cfg.fs * self._message(n, length), dim=1)
        return torch.polar(torch.ones_like(phase), phase)

    def _broadcast_fm(self, n: int, length: int) -> torch.Tensor:
        """Broadcast FM: MPX = 0.9 (mono L+R, stereo L-R on 38 kHz DSB-SC) + 0.1 pilot at 19 kHz, peak deviation
        40-75 kHz. Instantaneous frequency reaches past the channel edge; the channel filter clips it."""
        fs = self.cfg.fs
        t = torch.arange(length, device=self.device)[None, :] / fs
        a, b = self._message(n, length), self._message(n, length)
        stereo = (torch.rand(n, 1, generator=self.gen, device=self.device) < 0.7).float()
        p0 = self._u(n, 0, 2 * math.pi)[:, None]
        mpx = 0.45 * (a + b) + stereo * 0.45 * (a - b) * torch.cos(2 * (2 * math.pi * 19e3 * t + p0)) \
            + stereo * 0.1 * torch.cos(2 * math.pi * 19e3 * t + p0)
        mpx = mpx / mpx.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
        dev = self._u(n, 40e3, 75e3)[:, None]
        phase = torch.cumsum(2 * math.pi * dev / fs * mpx, dim=1)
        return torch.polar(torch.ones_like(phase), phase)

    def _wide(self, n: int, length: int) -> torch.Tensor:
        """Slice of a signal wider than the channel, as the channelizer would output it (already channel-filtered)."""
        R = self.cfg.wide_rates[self._ci(0, len(self.cfg.wide_rates) - 1)]
        fs_hi, N = self.cfg.fs * R, length * R
        kind = self._ci(0, 3)
        if kind == 0:  # GFSK, 2..R samples/symbol at fs_hi: 0.24 .. 1.9 Msym/s at R=16 (BLE, proprietary 2.4 GHz)
            sps = self._ci(2, max(2, R))
            h, bt = self._cu(0.3, 1.0), self._cu(0.3, 0.6)
            nsym = N // sps + 8
            bits = torch.randint(0, 2, (n, nsym), generator=self.gen, device=self.device).float() * 2 - 1
            freq = bits.repeat_interleave(sps, dim=1)
            g = gaussian_taps(sps, bt).to(self.device)
            freq = F.conv1d(freq[:, None], g.flip(0).view(1, 1, -1), padding=g.numel() // 2)[:, 0, : freq.shape[1]]
            phase = torch.cumsum(math.pi * h / sps * freq, dim=1)
            s = torch.polar(torch.ones_like(phase), phase)
            bw = (1 + h) * fs_hi / sps
        elif kind in (1, 2):  # PSK/QAM with symbol rate above the channel
            sps = self._ci(2, max(2, R // 2))
            beta = self._cu(0.2, 0.5)
            s = self._linear(["bpsk", "qpsk", "8psk", "16qam", "64qam"][self._ci(0, 4)], n, N, sps, beta)
            bw = (1 + beta) * fs_hi / sps
        else:  # wide OFDM (WiFi / LTE-like), 50-90 % of fs_hi occupied
            nfft = 64 * (2 ** self._ci(0, 2))
            used = int(nfft * self._cu(0.5, 0.9)) // 2 * 2
            cp = nfft // 4
            pts = self.const["qpsk" if self._ci(0, 1) == 0 else "16qam"]
            nsym = N // (nfft + cp) + 2
            grid = torch.zeros(n, nsym, nfft, dtype=torch.complex64, device=self.device)
            bins = torch.cat([torch.arange(1, used // 2 + 1), torch.arange(nfft - used // 2, nfft)]).to(self.device)
            idx = torch.randint(0, pts.numel(), (n, nsym, bins.numel()), generator=self.gen, device=self.device)
            grid[:, :, bins] = pts[idx]
            sym = torch.fft.ifft(grid, dim=-1) * math.sqrt(nfft)
            s = torch.cat([sym[..., -cp:], sym], dim=-1).reshape(n, -1)
            bw = fs_hi * used / nfft
        s = s[:, :N]
        # put the channel centre anywhere inside the signal band (at least ~one channel spacing wide)
        half = max(bw / 2 - 30e3, 0.0)
        f0 = self._u(n, -half, half)[:, None]
        t = torch.arange(N, device=self.device)[None, :]
        s = s * torch.polar(torch.ones(n, N, device=self.device), -2 * math.pi * f0 / fs_hi * t)
        s = fir_complex(s, self.h_wide[R])
        ph = int(torch.randint(0, R, (1,), generator=self.cpu_gen))
        return s[:, ph::R][:, :length]

    def _ofdm(self, n: int, length: int) -> torch.Tensor:
        nfft = 64 if self._ci(0, 1) == 0 else 128
        used = int(nfft * self._cu(0.2, 0.45)) // 2 * 2
        cp = nfft // (4 if self._ci(0, 1) == 0 else 8)
        pts = self.const["qpsk" if self._ci(0, 1) == 0 else "16qam"]
        nsym = length // (nfft + cp) + 2
        grid = torch.zeros(n, nsym, nfft, dtype=torch.complex64, device=self.device)
        bins = torch.cat([torch.arange(1, used // 2 + 1), torch.arange(nfft - used // 2, nfft)]).to(self.device)
        idx = torch.randint(0, pts.numel(), (n, nsym, bins.numel()), generator=self.gen, device=self.device)
        grid[:, :, bins] = pts[idx]
        sym = torch.fft.ifft(grid, dim=-1) * math.sqrt(nfft)
        sym = torch.cat([sym[..., -cp:], sym], dim=-1)
        return sym.reshape(n, -1)

    # ---------------------------------------------------------------- chunks
    def _chunk(self, name: str, n: int) -> torch.Tensor:
        cfg, L = self.cfg, self.cfg.length
        guard = 256
        if name == "noise":
            return torch.zeros(n, L, dtype=torch.complex64, device=self.device)
        sps = self._ci(*cfg.sps)
        if name in LINEAR:
            s = self._linear(name, n, L + guard, sps, self._cu(*cfg.rolloff))
        elif name == "gfsk":
            s = self._gfsk(n, L + guard, sps)
        elif name == "am":
            s = self._am(n, L + guard)
        elif name == "fm":
            s = self._fm(n, L + guard)
        elif name == "ofdm":
            s = self._ofdm(n, L + guard)
        elif name == "wide":
            s = self._wide(n, L + guard)
        else:
            raise ValueError(name)

        # Random timing: crop at a random offset past the filter transient.
        start = torch.randint(guard // 2, s.shape[1] - L + 1, (n,), generator=self.gen, device=self.device)
        s = torch.gather(s, 1, start[:, None] + torch.arange(L, device=self.device)[None, :])

        # Multipath: direct path plus up to `multipath_taps` weaker echoes.
        mp = torch.rand(n, generator=self.gen, device=self.device) < cfg.multipath_prob
        if mp.any():
            h = torch.zeros(n, cfg.multipath_delay + 1, dtype=torch.complex64, device=self.device)
            h[:, 0] = 1
            for _ in range(cfg.multipath_taps):
                d = torch.randint(1, cfg.multipath_delay + 1, (n,), generator=self.gen, device=self.device)
                a = self._u(n, 0.0, 0.5) * torch.exp(1j * self._u(n, 0, 2 * math.pi))
                h[torch.arange(n, device=self.device), d] += a.to(torch.complex64)
            h = torch.where(mp[:, None], h, torch.nn.functional.one_hot(
                torch.zeros(n, dtype=torch.long, device=self.device), h.shape[1]).to(torch.complex64))
            pad = torch.cat([torch.zeros(n, cfg.multipath_delay, dtype=s.dtype, device=self.device), s], dim=1)
            s = sum(h[:, k:k + 1] * pad[:, cfg.multipath_delay - k: cfg.multipath_delay - k + L]
                    for k in range(cfg.multipath_delay + 1))

        # Carrier phase and frequency offset ("wide" is already placed at a random point of its band).
        lo, hi = cfg.cfo_hz if name != "wide" else (0.0, 0.0)
        if name == "fm":  # a broadcast station is up to half a channel spacing off the nearest channel centre
            lo, hi = min(lo, -60e3), max(hi, 60e3)
        cfo = self._u(n, lo, hi)[:, None]
        ph = self._u(n, 0, 2 * math.pi)[:, None]
        t = torch.arange(L, device=self.device)[None, :]
        s = s * torch.polar(torch.ones(n, L, device=self.device), 2 * math.pi * cfo / cfg.fs * t + ph)
        if cfg.channel_filter and name != "wide":
            s = fir_complex(s, self.h1)
        return s / s.abs().pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-12)

    def batch(self, n: int):
        """Returns x [n, 2, L] float32 (unit RMS), y [n] int64, snr_db [n] (NaN for noise)."""
        cfg = self.cfg
        xs, ys = [], []
        for _ in range((n + cfg.chunk - 1) // cfg.chunk):
            name = cfg.classes[self._ci(0, len(cfg.classes) - 1)]
            xs.append(self._chunk(name, cfg.chunk))
            ys.append(torch.full((cfg.chunk,), self.class_index[name], dtype=torch.long, device=self.device))
        s = torch.cat(xs)[:n]
        y = torch.cat(ys)[:n]
        snr = self._u(n, *cfg.snr_db)
        noise = self._cn(n, cfg.length)
        if cfg.channel_filter:  # wideband input noise seen through the channel; SNR stays referred to the input
            noise = fir_complex(noise, self.h1)
        noise = noise * torch.pow(10.0, -snr / 20)[:, None]
        is_noise = y == self.class_index["noise"]
        x = s + noise
        x = x / x.abs().pow(2).mean(dim=1, keepdim=True).sqrt()
        perm = torch.randperm(n, generator=self.gen, device=self.device)
        x, y, snr = x[perm], y[perm], torch.where(is_noise, torch.nan, snr)[perm]
        return torch.view_as_real(x).permute(0, 2, 1).contiguous(), y, snr
