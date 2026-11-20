"""Training labels and synthetic examples for whole-signal candidate windows."""

from __future__ import annotations

import numpy as np
import torch

from .synth import CLASSES, Synth, SynthConfig, fir_complex

LABELS = ["noise", "ofdm", "dsss", "gfsk", "fm", "am", "other_signal"]
INPUT_LENGTH = 1024


class DemoSynth(Synth):
    classes = LABELS

    def __init__(self, device, seed: int):
        # Frequency offsets are residual errors after centering the RF candidate.
        cfg = SynthConfig(classes=list(LABELS), cfo_hz=(-6000, 6000), chunk=16)
        super().__init__(cfg, device=device, seed=seed)

    def _cn(self, n: int, length: int) -> torch.Tensor:
        noise = super()._cn(n, length)
        # A real candidate is filtered and decimated before amplitude normalization.
        if self._ci(0, 2):
            t = torch.arange(-12, 13, device=self.device)
            taps = torch.exp(-0.5 * (t / self._cu(0.8, 3.5)) ** 2)
            noise = fir_complex(noise, taps / taps.norm())
        return noise

    def _barker(self, n: int, length: int, sps: int) -> torch.Tensor:
        code = torch.tensor([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1], device=self.device)
        nsym = length // (11 * sps) + 2
        bits = torch.randint(0, 2, (n, nsym), device=self.device, generator=self.gen) * 2 - 1
        chips = (bits[:, :, None] * code).flatten(1).repeat_interleave(sps, dim=1)
        return chips.to(torch.complex64)

    def _ofdm(self, n: int, length: int) -> torch.Tensor:
        nfft = (64, 128, 256)[self._ci(0, 2)]
        half = int(nfft * self._cu(0.19, 0.35) / 2)
        cp = nfft // (4, 8, 16)[self._ci(0, 2)]
        used = torch.cat([torch.arange(1, half + 1), torch.arange(nfft - half, nfft)]).to(self.device)
        pts = self.const["qpsk" if self._ci(0, 1) == 0 else "16qam"]
        count = length // (nfft + cp) + 2
        grid = torch.zeros(n, count, nfft, dtype=torch.complex64, device=self.device)
        idx = torch.randint(pts.numel(), (n, count, len(used)), generator=self.gen, device=self.device)
        grid[:, :, used] = pts[idx]
        sym = torch.fft.ifft(grid, dim=-1) * nfft**0.5
        return torch.cat((sym[..., -cp:], sym), dim=-1).flatten(1)

    def _chip_windows(self, name: str, n: int) -> torch.Tensor:
        length = self.cfg.length
        sps = self._ci(2, 5) if name == "dsss" else self._ci(3, 10)
        wave = self._barker(n, length + 128, sps) if name == "dsss" else self._gfsk(n, length + 128, sps)
        at = torch.arange(length, device=self.device)[None, :]
        shift = torch.randint(16, 96, (n, 1), device=self.device, generator=self.gen)
        wave = torch.gather(wave, 1, at + shift)
        if self._ci(0, 4) != 0:
            duration = torch.randint(length // 9, length + 1, (n, 1), device=self.device, generator=self.gen)
            start = ((length - duration) * self._u(n, 0.0, 1.0)[:, None]).long()
            ramp = torch.minimum(at - start, start + duration - at).clamp(0, 8) / 8
            wave = wave * ramp
        phase = self._u(n, 0, 2 * torch.pi)[:, None] + self._u(n, -.025, .025)[:, None] * (2 * torch.pi * at)
        return wave * torch.exp(1j * phase)

    def _chunk(self, name: str, n: int) -> torch.Tensor:
        if name in ("dsss", "gfsk"):
            return self._chip_windows(name, n)
        if name == "other_signal":
            if self._ci(0, 3) == 0:
                t = torch.arange(self.cfg.length, device=self.device)[None, :]
                f = self._u(n, -0.12, 0.12)[:, None]
                return torch.exp(1j * 2 * torch.pi * f * t)
            name = ("bpsk", "qpsk", "8psk", "16qam", "64qam")[self._ci(0, 4)]
        return super()._chunk(name, n)


def make_synth(profile: str, device, seed: int) -> Synth:
    if profile == "demo":
        return DemoSynth(device, seed)
    if profile == "run1":
        return Synth(SynthConfig(), device=device, seed=seed)
    raise ValueError(f"unknown training profile: {profile}")


def profile_classes(profile: str) -> list[str]:
    if profile == "demo":
        return LABELS
    if profile == "run1":
        return CLASSES
    raise ValueError(f"unknown training profile: {profile}")


def load_examples(path, classes: list[str], device):
    """Read complete, already-normalized windows from an explicitly selected split."""
    with np.load(path, allow_pickle=False) as data:
        iq = data["iq"]
        labels = data["labels"]
        if data["classes"].tolist() != classes:
            raise ValueError(f"{path}: capture labels do not match the training profile")
        if labels.ndim != 1 or not len(labels) or iq.shape != (len(labels), 2, INPUT_LENGTH):
            raise ValueError(f"{path}: expected nonempty [N, 2, {INPUT_LENGTH}] IQ and N labels")
        if not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() >= len(classes):
            raise ValueError(f"{path}: capture labels contain an invalid class index")
        if not np.isfinite(iq).all():
            raise ValueError(f"{path}: capture IQ contains non-finite values")
    return torch.tensor(iq, dtype=torch.float32, device=device), torch.tensor(labels, dtype=torch.long, device=device)
