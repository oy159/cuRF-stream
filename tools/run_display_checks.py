#!/usr/bin/env python3
"""M3 smoke checks in three separate RF bands, with real stage-2 model output.

    python tools/run_display_checks.py --bin build/cuda-release/bench/curf_classify \
      --model model/artifacts/s2_run4/export --src ~/dataset/onair \
      --emissions ~/dataset/ds_v5/emissions.json --out ~/dataset/m3_checks

These finite, fixed-seed scenes check policy/feed integration. They do not replace
M4's broader randomized display-error acceptance or M5's hardware captures.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def make_adsb(path):
    """Receive-only IQ fixture using the same published DF17 frames as test_adsb_cuda.cpp."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"model"))
    from make_scene import pack12
    fs, duration = 61.44e6, .25
    n = int(fs*duration)
    rng = np.random.default_rng(17)
    x = (rng.normal(size=(n,2)) + 1j*rng.normal(size=(n,2))).astype(np.complex64) * np.float32(.003/np.sqrt(2))
    messages = [bytes.fromhex(m) for m in ("8D40621D58C382D690C8AC2863A7", "8D40621D58C386435CC412692AD6")]
    for i, start in enumerate(range(10000, n-8000, int(.01*fs))):
        def pulse(us):
            a, b = round(start+us*1e-6*fs), round(start+(us+.5)*1e-6*fs)
            t = np.arange(a,b)/fs
            x[a:b,1] += .05*np.exp(2j*np.pi*10e6*t+.3j)
        for us in (0,1,3.5,4.5):
            pulse(us)
        for k in range(112):
            bit = (messages[i%2][k//8] >> (7-k%8)) & 1
            pulse(8+k+(0 if bit else .5))
    pack12(x.reshape(-1)).tofile(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("bin", "model", "src", "emissions", "out"):
        ap.add_argument("--"+k, required=True, type=Path)
    ap.add_argument("--seconds", default=.5, type=float)
    a = ap.parse_args()
    if not .5 <= a.seconds <= 10:
        ap.error("--seconds must be 0.5..10")
    root = Path(__file__).resolve().parents[1]
    a.out.mkdir(parents=True, exist_ok=True)
    commands, results = [], {}

    def run(args, log):
        commands.append([str(x) for x in args])
        with log.open("w") as f:
            subprocess.run(commands[-1], check=True, stdout=f)

    for name, freq, modes in (("ism24", 2440e6, "wifi5,ble"), ("wlan5", 5800e6, "wifi5"), ("cellular", 2135e6, "cell5")):
        out = a.out/name
        out.mkdir(exist_ok=True)
        print(f"checking {name}", flush=True)
        run([sys.executable, root/"model/make_scene.py", "make", "--src", a.src, "--emissions", a.emissions,
             "--out", out, "--scenes", 1, "--secs", a.seconds, "--sources", len(modes.split(',')),
             "--modes", modes, "--one-of-each", "--snr", 20, 30], out/"generate.log")
        # This older fixture chooses arbitrary baseband centres. Give its WiFi
        # carrier a physical channel centre for the current RF-grid prior.
        truth = json.loads((out/"truth.json").read_text())["truth"]
        wifi = [e for e in truth if e["mode"].startswith("wifi")]
        if wifi:
            centre = (min(e["fc"]-e["bw"]/2 for e in wifi) + max(e["fc"]+e["bw"]/2 for e in wifi))/2
            origin = 2412e6 if name == "ism24" else 5000e6
            freq = origin + round((freq+centre-origin)/5e6)*5e6 - centre
        run([a.bin.resolve(), "--source", "file", "--file", out/"scene_0.p12", "--streams", 1,
             "--stage2", a.model.resolve(), "--freq", freq, "--realtime", "--drop", "--seconds", 0,
             "--band-mhz", 44, "--ui-out", out/"scene_0.feed"], out/"runtime.json")
        run([sys.executable, root/"tools/apply_signal_policy.py", "--feed", out/"scene_0.feed", "--out", out/"display.feed",
             "--truth", out/"truth.json", "--score", out/"score.json"], out/"policy.log")
        rows = [json.loads(l) for l in (out/"display.feed").read_text().splitlines()]
        stats = json.loads((out/"runtime.json").read_text())
        score = json.loads((out/"score.json").read_text())
        signals = [r for r in rows if "sig" in r]
        assert rows[0]["meta"]["freq"] == freq, "explicit replay RF centre lost"
        assert signals and all(len(r["sig"]["probs"]) == 13 for r in signals), "missing full model output"
        assert all(r["display"]["family"] == "unknown" or r["display"]["family"] in r["display"]["allowed"] for r in signals), "impossible displayed family"
        assert score["named"] > 0, "no model labels passed policy"
        assert stats["dropped_frames"] == stats["gap_frames"] == 0, "sample loss"
        assert all(stats["stage2"][k] == 0 for k in ("skipped_busy", "skipped_ring", "skipped_capacity")), "stage-2 skips"
        results[name] = {"score": score, "stage2": stats["stage2"], "rf_center_hz": freq, "passed": True}
    out = a.out/"adsb"
    out.mkdir(exist_ok=True)
    make_adsb(out/"adsb.p12")
    run([a.bin.resolve(), "--source", "file", "--file", out/"adsb.p12", "--streams", 2,
         "--stage2", a.model.resolve(), "--freq", 1080e6, "--realtime", "--drop", "--seconds", 0,
         "--adsb", 1090e6, "--adsb-stream", 1, "--band-mhz", 44, "--ui-out", out/"raw.feed"], out/"runtime.json")
    run([sys.executable, root/"tools/apply_signal_policy.py", "--feed", out/"raw.feed", "--out", out/"display.feed"], out/"policy.log")
    rows = [json.loads(l) for l in (out/"display.feed").read_text().splitlines()]
    decoded = [r for r in rows if r.get("display", {}).get("status") == "decoded"]
    assert len(decoded) >= 10, "missing verified ADS-B messages"
    assert all(r["adsb"]["crc_ok"] and r["display"]["s"] == 1 and r["display"]["freq"] == 1090e6 for r in decoded), "incorrect ADS-B RF/receiver linkage"
    stats = json.loads((out/"runtime.json").read_text())
    assert stats["dropped_frames"] == stats["gap_frames"] == 0, "ADS-B fixture sample loss"
    results["adsb"] = {"decoded": len(decoded), "position_messages": sum("lat" in r["adsb"] for r in decoded), "passed": True}
    (a.out/"commands.json").write_text(json.dumps(commands, indent=2))
    (a.out/"checks.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
