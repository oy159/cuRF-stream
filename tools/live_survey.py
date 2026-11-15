#!/usr/bin/env python3
"""Summary of a live capture: per-stream spectrum profile and the strongest channels with their class mix.

    curf_classify --source omnicore --freq 950e6 --seconds 5 --ui-out /tmp/sv.jsonl --dump /tmp/sv.csv
    python tools/live_survey.py /tmp/sv 950
"""
import json
import sys
from collections import Counter

import numpy as np

base, fc = sys.argv[1], float(sys.argv[2])
names = ["noise", "bpsk", "qpsk", "8psk", "16qam", "64qam", "gfsk", "am", "fm", "ofdm"]
M, df = 512, 0.12
rows = [json.loads(l) for l in open(base + ".jsonl") if l.startswith('{"w"')]
P = np.array([r["p"] for r in rows]).reshape(len(rows), 2, M)
F = np.array([r["f"] for r in rows])
k = np.arange(M)
off = np.where(k < M // 2, k, k - M) * df
order = np.argsort(off)
d = np.genfromtxt(base + ".csv", delimiter=",", names=True)
nw = len(np.unique(d["window"])) if d.size else 1
for s in range(2):
    med = np.median(P[:, s], 0)
    print(f"== stream {s}: floor {F[:, s].mean():.1f} dB, peak {P[:, s].max():.0f} dBFS, "
          f"channels > floor+10dB (median): {(med > F[:, s].mean() + 10).sum()}")
    prof = med[order]
    print("   profile every 16 ch: " + " ".join(f"{fc + off[order][i]:.1f}:{prof[i]:.0f}" for i in range(0, M, 16)))
    m = d["stream"] == s
    print(f"   active/window {m.sum() / nw:.0f}; classes: " +
          ", ".join(f"{names[int(c)]} {n}" for c, n in Counter(d['cls'][m]).most_common()))
    ch, cl, pw = d["channel"][m].astype(int), d["cls"][m].astype(int), 10 * np.log10(d["power"][m])
    st = []
    for c in np.unique(ch):
        sel = ch == c
        cnt = Counter(cl[sel])
        top, n = cnt.most_common(1)[0]
        st.append((pw[sel].mean(), c, fc + (c if c < M // 2 else c - M) * df, sel.sum() / nw, names[top], n / sel.sum()))
    st.sort(reverse=True)
    print("   strongest: dB  ch   MHz   duty class frac")
    for t in st[:15]:
        print("     %.1f %3d %8.2f %.2f %-5s %.2f" % t)
