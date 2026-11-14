#!/usr/bin/env python3
"""Quantify copy/compute overlap in an Nsight Systems report.

    nsys stats -q -f csv --report cuda_gpu_trace REPORT.nsys-rep > trace.csv
    python3 tools/nsys_overlap.py trace.csv

Prints, over the span of GPU activity: how busy the H2D engine, the D2H engine
and the SMs (kernels) were, and what fraction of H2D time ran concurrently
with kernels.
"""

import csv
import sys


def union(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total(u):
    return sum(e - s for s, e in u)


def intersect(u, v):
    i = j = t = 0
    while i < len(u) and j < len(v):
        s, e = max(u[i][0], v[j][0]), min(u[i][1], v[j][1])
        if e > s:
            t += e - s
        if u[i][1] < v[j][1]:
            i += 1
        else:
            j += 1
    return t


def main(path):
    kinds = {"h2d": [], "d2h": [], "kernel": []}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            s, d, n = int(r["Start (ns)"]), int(r["Duration (ns)"]), r["Name"]
            k = "h2d" if "HtoD" in n or "Host-to-Device" in n else "d2h" if "DtoH" in n or "Device-to-Host" in n else "kernel"
            kinds[k].append((s, s + d))
    u = {k: union(v) for k, v in kinds.items()}
    every = union(kinds["h2d"] + kinds["d2h"] + kinds["kernel"])
    span = every[-1][1] - every[0][0]
    pct = lambda x: 100.0 * x / span
    print(f"span {span / 1e6:.1f} ms")
    print(f"busy: H2D {pct(total(u['h2d'])):.1f}%  D2H {pct(total(u['d2h'])):.1f}%  "
          f"kernels {pct(total(u['kernel'])):.1f}%  any {pct(total(every)):.1f}%")
    print(f"H2D time overlapped with kernels: {100.0 * intersect(u['h2d'], u['kernel']) / total(u['h2d']):.1f}%")


if __name__ == "__main__":
    main(sys.argv[1])
