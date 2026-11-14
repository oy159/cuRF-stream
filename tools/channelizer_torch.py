#!/usr/bin/env python3
"""PyTorch baseline of the polyphase channelizer (same definition as include/curf/channelizer.hpp).

    python tools/channelizer_torch.py [--streams 2] [--block 131072] [--channels 512] [--taps 8]

Idiomatic framework version: unfold the (zero-history) input into overlapping L-sample segments,
multiply by the reversed prototype, fold the P phases with a reshape-sum, rotate odd frames by M/2,
inverse FFT. Checked against a direct numpy evaluation on a small case, then timed with CUDA events.
Prints one JSON line.
"""

import argparse
import json

import numpy as np
import torch


def bessel_i0(x):
    return np.i0(x)


def prototype(M, P, cutoff=0.75, beta=8.0):
    """Same design as curf::design_prototype (Kaiser-windowed sinc, unit DC gain)."""
    L = M * P
    t = np.arange(L) - 0.5 * (L - 1)
    fc = cutoff / M
    sinc = np.where(t == 0, 2 * fc, np.sin(2 * np.pi * fc * t) / (np.pi * np.where(t == 0, 1, t)))
    r = 2 * t / (L - 1)
    w = bessel_i0(beta * np.sqrt(np.clip(1 - r * r, 0, None))) / bessel_i0(beta)
    h = sinc * w
    return (h / h.sum()).astype(np.float32)


def channelize_torch(x, h, M):
    """x: [S, N] complex64 (one block, zero history), h: [L] float32 -> [S, N/D, M] complex64."""
    L, D = h.numel(), M // 2
    S, N = x.shape
    xpad = torch.cat([torch.zeros(S, L - 1, dtype=x.dtype, device=x.device), x], dim=1)
    seg = xpad.unfold(1, L, D)[:, : N // D]             # [S, F, L]: seg[n][j] = x[nD - (L-1) + j]
    v = seg.flip(-1) * h                                 # v[n][l] = h[l] x[nD - l]
    u = v.reshape(S, -1, L // M, M).sum(dim=2)          # u[n][m] = sum_p h[m + pM] x[nD - m - pM]
    u[:, 1::2] = torch.roll(u[:, 1::2], M // 2, dims=-1)  # (-1)^{kn}
    return torch.fft.ifft(u, dim=-1) * M


def channelize_direct(x, h, M, frames):
    L, D = len(h), M // 2
    k = np.arange(M)
    out = np.zeros((frames, M), np.complex128)
    for n in range(frames):
        for l in range(L):
            s = n * D - l
            if 0 <= s < len(x):
                out[n] += h[l] * x[s] * np.exp(-2j * np.pi * k * s / M)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, default=2)
    ap.add_argument("--block", type=int, default=1 << 17)
    ap.add_argument("--channels", type=int, default=512)
    ap.add_argument("--taps", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    dev = torch.device("cuda")
    M, P = args.channels, args.taps

    # correctness on a small case against the direct definition
    h_small = prototype(64, 4)
    rng = np.random.default_rng(0)
    xs = (rng.normal(size=1024) + 1j * rng.normal(size=1024)).astype(np.complex64)
    want = channelize_direct(xs, h_small, 64, 1024 // 32)
    got = channelize_torch(torch.from_numpy(xs)[None].to(dev), torch.from_numpy(h_small).to(dev), 64)[0].cpu().numpy()
    err = float(np.abs(got - want).max() / np.abs(want).max())

    h = torch.from_numpy(prototype(M, P)).to(dev)
    x = torch.randn(args.streams, args.block, dtype=torch.complex64, device=dev) * 0.2
    with torch.no_grad():
        for _ in range(10):
            channelize_torch(x, h, M)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record()
        for _ in range(args.iters):
            channelize_torch(x, h, M)
        b.record()
        torch.cuda.synchronize()
    ms = a.elapsed_time(b) / args.iters
    msps = args.block / (ms * 1e3)
    print(json.dumps({"path": "torch", "streams": args.streams, "block": args.block, "channels": M, "taps": P,
                      "ms_per_block": round(ms, 4), "msps_per_stream": round(msps, 1),
                      "x_realtime": round(msps / 61.44, 2), "rel_err_vs_direct": err,
                      "torch": torch.__version__}))


if __name__ == "__main__":
    main()
