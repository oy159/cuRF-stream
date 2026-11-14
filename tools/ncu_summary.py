#!/usr/bin/env python3
"""Condense an Nsight Compute report into one row per kernel.

    ncu --import REPORT.ncu-rep --csv --page raw > raw.csv
    python3 tools/ncu_summary.py raw.csv [--markdown]

Columns: duration, SM / memory / DRAM throughput (% of peak), tensor pipe utilisation, DRAM traffic,
achieved occupancy, registers, dynamic+static shared memory, top warp stall reasons.
"""

import csv
import re
import sys

COLS = [
    ("us", "gpu__time_duration.sum", 1e-3),
    ("SM%", "sm__throughput.avg.pct_of_peak_sustained_elapsed", 1),
    ("Mem%", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", 1),
    ("DRAM%", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 1),
    ("L1%", "l1tex__throughput.avg.pct_of_peak_sustained_active", 1),
    ("L2%", "lts__throughput.avg.pct_of_peak_sustained_elapsed", 1),
    ("TC%", "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active", 1),
    ("DRAM_MB", ("dram__bytes_read.sum", "dram__bytes_write.sum"), 1e-6),
    ("occ%", "sm__warps_active.avg.pct_of_peak_sustained_active", 1),
    ("regs", "launch__registers_per_thread", 1),
    ("smemKB", ("launch__shared_mem_per_block_dynamic", "launch__shared_mem_per_block_static"), 1 / 1024),
]
STALLS = {
    "long_sb": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "barrier": "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "mio": "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "lg_thr": "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
}
# Units ncu may attach in the second header row, normalised to the base unit used above.
SCALE = {"msecond": 1e6, "usecond": 1e3, "nsecond": 1, "ms": 1e6, "us": 1e3, "ns": 1,
         "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "byte": 1, "Kibyte": 1024, "Mibyte": 1024 ** 2,
         "Kbyte/block": 1e3, "byte/block": 1, "Mbyte/block": 1e6}


def short(name: str) -> str:
    m = re.search(r"(conv_\w+|gap_linear_kernel|\w+_kernel)(<[^(]*>)?", name)
    if not m:
        return name[:60]
    tmpl = (m.group(2) or "").replace("__half", "h").replace("float", "f").replace(" ", "")
    tmpl = re.sub(r"\(int\)|\(bool\)", "", tmpl)
    return (m.group(1) + tmpl)[:60]


def main(path: str, markdown: bool) -> None:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, units, data = rows[0], rows[1], rows[2:]
    idx = {h: i for i, h in enumerate(header)}

    def val(r, key):
        keys = key if isinstance(key, tuple) else (key,)
        total = 0.0
        for k in keys:
            if k not in idx or not r[idx[k]]:
                return None
            v = float(r[idx[k]].replace(",", ""))
            total += v * SCALE.get(units[idx[k]], 1)
        return total

    names = ["kernel"] + [c[0] for c in COLS] + ["top stalls (% of active)"]
    out = []
    for r in data:
        row = [short(r[idx["Kernel Name"]])]
        for _, key, scale in COLS:
            v = val(r, key)
            row.append("-" if v is None else f"{v * scale:.1f}" if v * scale < 1000 else f"{v * scale:.0f}")
        st = sorted(((val(r, k) or 0, n) for n, k in STALLS.items()), reverse=True)[:2]
        row.append(" ".join(f"{n}={v:.0f}" for v, n in st))
        out.append(row)
    if markdown:
        print("| " + " | ".join(names) + " |")
        print("|" + "---|" * len(names))
        for row in out:
            print("| " + " | ".join(row) + " |")
    else:
        w = [max(len(x[i]) for x in out + [names]) for i in range(len(names))]
        for row in [names] + out:
            print("  ".join(c.ljust(w[i]) for i, c in enumerate(row)))


if __name__ == "__main__":
    main(sys.argv[1], "--markdown" in sys.argv[2:])
