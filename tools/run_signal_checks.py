#!/usr/bin/env python3
"""Reproduce M2 with WiFi + BLE + LTE + narrow FM, then a paced dual-RX replay.

    python tools/run_signal_checks.py --bin build/cuda-release/bench/curf_classify \
      --model model/artifacts/s2_run4/export --src ~/dataset/onair \
      --emissions ~/dataset/ds_v5/emissions.json --out /tmp/curf-m2 --seconds 300

The classification score is reported separately; M2 acceptance checks zero loss/skips, carrier identities,
wide/narrow processing p99 <15 ms and the requested duration. This is synthetic file replay, not SDR capture.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("bin", "model", "src", "emissions", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--int8", action="store_true")
    a = ap.parse_args()
    if a.seconds < 1:
        ap.error("--seconds must be >= 1")
    root = Path(__file__).resolve().parents[1]
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    scene = out / "scene"
    commands = []

    def run(args, log):
        commands.append([str(x) for x in args])
        print(f"running {log.name}", flush=True)
        with log.open("w") as f:
            subprocess.run(commands[-1], check=True, stdout=f)

    run([sys.executable, root / "model/make_scene.py", "make", "--src", a.src.resolve(),
         "--emissions", a.emissions.resolve(), "--out", scene, "--scenes", "1", "--secs", ".5",
         "--sources", "4", "--modes", "wifi5,ble,cell5,nfm_0", "--one-of-each", "--snr", "25", "30"],
        out / "generate.log")
    raw = np.memmap(scene / "scene_0.p12", dtype=np.uint8, mode="r").reshape(-1, 3)
    dual = scene / "scene_0_2rx.p12"
    with dual.open("wb") as f:
        for begin in range(0, len(raw), 65536):
            np.repeat(raw[begin:begin+65536], 2, axis=0).tofile(f)
    base = [a.bin.resolve(), "--source", "file", "--stage2", a.model.resolve(), "--realtime", "--drop", "--band-mhz", "44"]
    if a.int8:
        base.append("--stage2-int8")
    run([*base, "--file", scene / "scene_0.p12", "--streams", "1", "--seconds", "0", "--ui-out", scene / "scene_0.feed"],
        scene / "scene_0.json")
    run([sys.executable, root / "tools/check_signal_tracks.py", "--truth", scene / "truth.json", "--feed", scene / "scene_0.feed",
         "--stats", scene / "scene_0.json"], out / "controlled_check.json")
    run([sys.executable, root / "model/make_scene.py", "score", "--out", scene], out / "classification_score.json")
    run([*base, "--file", dual, "--streams", "2", "--seconds", str(a.seconds), "--loop", "--ui-out", out / "realtime.feed"],
        out / "realtime.json")
    run([sys.executable, root / "tools/check_signal_tracks.py", "--truth", scene / "truth.json", "--feed", out / "realtime.feed",
         "--stats", out / "realtime.json", "--min-seconds", str(a.seconds)], out / "acceptance.json")
    (out / "commands.json").write_text(json.dumps(commands, indent=2))
    print((out / "acceptance.json").read_text())


if __name__ == "__main__":
    main()
