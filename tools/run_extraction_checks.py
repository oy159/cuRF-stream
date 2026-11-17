#!/usr/bin/env python3
"""Run the known-signal IQ checks on a CUDA host and write one JSON record per case.

    python tools/run_extraction_checks.py --bin build/cuda-release/bench/curf_check_extraction \
        --out /tmp/curf-m2-iq --results docs/results/m2_extraction.jsonl
"""
import argparse
import json
from pathlib import Path
import subprocess

from check_extraction import inspect


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    a = ap.parse_args()
    a.results.parent.mkdir(parents=True, exist_ok=True)
    # Standard-rate cases exercise the actual 61.44 MS/s -> 2^18-sample runtime window sizes.
    cases = [
        ("wide_tone", ["--decimation", "4", "--signal", "tone", "--first", "0"], True),
        ("wide_burst", ["--decimation", "4"], False),
        ("channel_tone", ["--decimation", "256", "--signal", "tone", "--first", "0"], True),
        ("channel_burst", ["--decimation", "256"], False),
        ("narrow_tone", ["--decimation", "4096", "--signal", "tone", "--first", "0"], True),
        ("narrow_burst", ["--decimation", "4096"], False),
        ("signed8_burst", ["--decimation", "4096", "--wire", "signed8"], False),
        ("nonbinary_burst", ["--decimation", "3", "--block", "262143", "--fc", "3012345"], False),
        ("long_history", ["--decimation", "1024", "--block", "8192"], False),
        ("near_nyquist", ["--decimation", "4096", "--fc", "30600000"], False),
    ]
    with a.results.open("w") as result_file:
        for name, opts, legacy in cases:
            directory = a.out / name
            run = subprocess.run([str(a.bin.resolve()), "--out", str(directory), *opts],
                                 text=True, capture_output=True)
            if run.returncode:
                raise RuntimeError(f"{name}: exit {run.returncode}\n{run.stdout}\n{run.stderr}")
            result = {"case": name, "capture": json.loads(run.stdout), **inspect(directory, legacy)}
            line = json.dumps(result, allow_nan=False)
            result_file.write(line + "\n")
            result_file.flush()
            print(line, flush=True)
            if not result["passed"]:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
