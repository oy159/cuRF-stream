#!/usr/bin/env python3
"""AD9361 RX FIR for 61.44 MS/s output, decimating by 2 (input 122.88 MHz, HB1 bypassed), as an FTR file.

    python tools/fir/design_rx_fir.py [--taps 64] [--pass-mhz 24] [--stop-mhz 30.72] > rx.ftr
    och fir rx load rx.ftr && och fir rx on     # the SDK then moves the chain to ADC 491.52 -> 245.76 -> 122.88

At 61.44 MS/s the SDK accepts 64 taps with decimation 2 (128 is rejected: fir_rate_invalid). Coefficients are
equiripple (Parks-McClellan), scaled to sum 32768 (unity DC gain at GAIN 0). The checked-in
rx_dec2_61m44_64tap.ftr is this script's default output: passband ripple 0.08 dB, stopband -78 dB (float).
"""
import argparse
import sys

import numpy as np
from scipy import signal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taps", type=int, default=64)
    ap.add_argument("--pass-mhz", type=float, default=24.0)
    ap.add_argument("--stop-mhz", type=float, default=30.72)
    ap.add_argument("--fs-mhz", type=float, default=122.88, help="FIR input rate")
    a = ap.parse_args()
    if a.taps % 16 or not 16 <= a.taps <= 128:
        ap.error("taps: 16..128, multiple of 16")
    best = None
    for w in (5, 10, 20, 40):
        h = signal.remez(a.taps, [0, a.pass_mhz, a.stop_mhz, a.fs_mhz / 2], [1, 0], weight=[1, w], fs=a.fs_mhz)
        f, H = signal.freqz(h, worN=8192, fs=a.fs_mhz)
        A = 20 * np.log10(np.abs(H) + 1e-12)
        ripple, stop = np.ptp(A[f <= a.pass_mhz]), A[f >= a.stop_mhz].max()
        if ripple < 0.5 and (best is None or stop < best[1]):
            best = (h, stop, ripple)
    h, stop, ripple = best
    q = np.round(h / h.sum() * 32768).astype(int)
    print(f"ripple {ripple:.2f} dB, stopband {stop:.1f} dB, max coefficient {abs(q).max()}", file=sys.stderr)
    print(f"# cuRF: RX lowpass for {a.fs_mhz / 2:g} MSPS output, input {a.fs_mhz:g} MHz, "
          f"pass +-{a.pass_mhz:g} MHz, stop >= {a.stop_mhz:g} MHz")
    print("RX 3 GAIN 0 DEC 2")
    print("\n".join(str(int(c)) for c in q))


if __name__ == "__main__":
    main()
