#!/usr/bin/env python3
"""M5 receive-only live trials, shared display server, and bounded diagnostic capture.

Record display feeds and runtime counters rather than full-rate IQ. Live labels
are observations, not ground truth or a measured classification accuracy.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

from signal_policy import SignalPolicy, load_config
from ui_server import Handler, Hub
from live_feed import capture_file_feed

ROOT = Path(__file__).resolve().parents[1]
# Site presets, separate from classifier defaults. M5 v2 checks full 2.4 GHz
# candidates at the ordinary 6 dB detection threshold after component recovery.
PROFILES = {"ism24": (2440e6, 20, 6), "wlan5": (5770e6, 40, 18),
            "cell900": (939e6, 40, 6), "adsb": (1100e6, 45, 6)}


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def runtime_checks(runtime, meta, case, seconds, ended):
    driver, stage = runtime.get("omnicore", {}), runtime.get("stage2", {})
    return {
        "complete_feed": ended,
        "physical_receiver": meta.get("source") == "omnicore" and meta.get("rx") == case["rx"],
        "rf_and_rate": meta.get("freq") == case["freq"] and meta.get("rate") == 61.44e6,
        "duration": runtime.get("wall_s", 0) >= seconds-.02,
        "zero_sample_loss": all(runtime.get(k, -1) == 0 for k in ("dropped_frames", "gap_frames")),
        "zero_driver_loss": all(driver.get(k, -1) == 0 for k in
                                ("host_dropped", "device_overflows", "transfer_errors", "resyncs", "discontinuities")),
        "zero_stage2_skips": all(stage.get(k, -1) == 0 for k in ("skipped_busy", "skipped_ring", "skipped_capacity")),
        "no_transmission": driver.get("tx_samples", -1) == 0,
        "decoder_capacity": runtime.get("adsb", {}).get("overflow", 0) == 0,
    }


def summarize_feed(path):
    labels, reasons, statuses = Counter(), Counter(), Counter()
    meta, ended, policy = {}, False, {}
    receivers = set()
    for line in path.open():
        row = json.loads(line)
        if "meta" in row:
            meta = row["meta"]
        if "display" in row:
            d = row["display"]
            statuses[d["status"]] += 1
            reasons[d.get("reason", "decoded")] += 1
            if d["status"] in ("stable", "held"):
                # A frequency bucket is a presentation summary, not an emitter ID.
                labels[(d["s"], d["family"], d.get("protocol"), round(d["freq"]/1e6, 1))] += 1
                receivers.add(d["s"])
        if "end" in row:
            policy = row["end"].get("policy", {})
            ended = True
    return {"meta": meta, "ended": ended, "policy": policy, "statuses": dict(statuses),
            "reasons": dict(reasons), "receivers_with_named": sorted(receivers),
            "labels": [{"stream": k[0], "family": k[1], "protocol": k[2], "freq_mhz_bucket": k[3], "records": n}
                       for k, n in labels.most_common(30)]}


def record_case(binary, model, out, case, seconds, hub, max_bytes=2**30, capacities=(64, 256, 32), tune_cache=None, display_interval=0, pending=3):
    out.mkdir()
    cmd = [str(binary.resolve()), "--source", "omnicore", "--rx", ",".join(map(str, case["rx"])),
           "--freq", str(case["freq"]), "--gain", str(case["gain"]), "--threshold-db", str(case["threshold"]),
           "--neighbor-db", str(case.get("neighbor_db", 6)),
           "--stage2-max-signals", str(capacities[0]), "--stage2-max-tracks", str(capacities[1]),
           "--stage2-scratch-mib", str(capacities[2]),
           "--stage2-pending", str(pending),
           "--bandwidth", "56e6", "--band-mhz", "44", "--stage2", str(model.resolve()), "--seconds", str(seconds)]
    if case["name"] == "adsb":
        cmd += ["--adsb", "1090e6", "--adsb-stream", "0", "--adsb-threshold", "1.6"]
    if tune_cache:
        cmd += ["--stage2-tune-cache", str(tune_cache.resolve())]
    write_json(out/"command.json", {"argv": cmd, "feed": "display.feed",
                                  "transport": "bounded file tail" if display_interval else "inherited pipe fd"})
    started = utc()
    if display_interval:
        sampling = capture_file_feed(cmd, out, hub, max_bytes, display_interval, capacities[1]*2+1)
        result = finish_case(out, case, seconds, started)
        result.update(raw_feed_sha256=digest(out/"raw.feed"), display_sampling=sampling)
        write_json(out/"result.json", result)
        return result
    r, w = os.pipe()
    proc = None
    try:
        with (out/"runtime.json").open("w") as stats, (out/"stderr.log").open("w") as err:
            try:
                proc = subprocess.Popen(cmd+["--ui-out", f"/dev/fd/{w}"], pass_fds=(w,), stdout=stats, stderr=err)
            finally:
                os.close(w)
            with os.fdopen(r) as source, (out/"display.feed").open("w", buffering=1) as feed:
                r = None
                written = 0
                for line in source:
                    published = hub.publish(line.strip())+"\n"
                    written += len(published.encode("utf-8"))
                    if written > max_bytes:
                        raise RuntimeError("diagnostic feed limit exceeded; incomplete run retained")
                    feed.write(published)
            if proc.wait() != 0:
                raise RuntimeError(f"receiver exited {proc.returncode}; see {out/'stderr.log'}")
    finally:
        if r is not None:
            os.close(r)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    return finish_case(out, case, seconds, started)


def finish_case(out, case, seconds, started):
    summary = summarize_feed(out/"display.feed")
    runtime = json.loads((out/"runtime.json").read_text())
    checks = runtime_checks(runtime, summary["meta"], case, seconds, summary["ended"])
    result = {"case": case, "started_utc": started, "finished_utc": utc(), "checks": checks,
              "runtime_passed": all(checks.values()), "long_run": seconds >= 300,
              "observations": summary, "runtime": runtime, "feed_sha256": digest(out/"display.feed"),
              "limits": "Receive-only qualitative trial. Named labels are not ground truth; no real-data accuracy is inferred. ADC clipping is not counted continuously."}
    write_json(out/"result.json", result)
    return result


def monitor(path, stop):
    with path.open("w", buffering=1) as f:
        while not stop.is_set():
            try:
                row = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                f.write(json.dumps({"utc": utc(), "gpu_csv": row.stdout.strip(), "exit_code": row.returncode})+"\n")
            except (OSError, subprocess.TimeoutExpired) as e:
                f.write(json.dumps({"utc": utc(), "error": str(e)})+"\n")
            stop.wait(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("bin", "model", "out"):
        ap.add_argument("--"+key, type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--bands", default=",".join(PROFILES))
    ap.add_argument("--dual", action="store_true", help="also run a full-rate dual-RX 2.4 GHz case")
    ap.add_argument("--dual-only", action="store_true", help="run only the dual-RX 2.4 GHz case")
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--revision", help="declared source revision; actual files are independently hashed")
    ap.add_argument("--neighbor-db", type=float, default=6, help="spectral neighbour rejection; 0 disables it")
    ap.add_argument("--ism-center-mhz", type=float, default=2440, help="2.4 GHz trial centre; keep target carriers inside the usable 44 MHz span")
    ap.add_argument("--max-signals", type=int, default=64)
    ap.add_argument("--max-tracks", type=int, default=256)
    ap.add_argument("--scratch-mib", type=int, default=32)
    ap.add_argument("--pending", type=int, default=3, help="bounded stage-2 windows in flight (1..8)")
    ap.add_argument("--tune-cache", type=Path, help="stage-2 kernel tuning cache; preserves the engine's cache validation")
    ap.add_argument("--max-feed-mib", type=int, default=1024, help="hard per-case capture limit; exceeding it stops reception")
    ap.add_argument("--display-interval-ms", type=int, default=0, help="0: complete pipe feed; otherwise file-backed display coalescing, full model records retained")
    a = ap.parse_args()
    names = a.bands.split(",")
    if not 5 <= a.seconds <= 3600 or not names or len(set(names)) != len(names) or any(n not in PROFILES for n in names):
        ap.error("require 5..3600 seconds and unique known band names")
    if (a.out/"manifest.json").exists():
        ap.error("output already contains a run; choose a new directory")
    if not 64 <= a.max_feed_mib <= 4096:
        ap.error("feed limit must be in 64..4096 MiB")
    if a.display_interval_ms and not 50 <= a.display_interval_ms <= 1000:
        ap.error("display interval must be 0 or 50..1000 ms")
    if not 0 <= a.neighbor_db <= 30:
        ap.error("neighbour rejection must be in 0..30 dB")
    if not 2400 <= a.ism_center_mhz <= 2483.5:
        ap.error("ISM trial centre must be in 2400..2483.5 MHz")
    if not 1 <= a.max_signals <= 512 or not a.max_signals <= a.max_tracks <= 4096 or not 8 <= a.scratch_mib <= 512:
        ap.error("invalid stage-2 capacity")
    if not 1 <= a.pending <= 8 or a.dual and a.dual_only:
        ap.error("require pending 1..8 and at most one dual mode")
    a.out.mkdir(parents=True, exist_ok=True)
    profiles = {**PROFILES, "ism24": (a.ism_center_mhz*1e6, *PROFILES["ism24"][1:])}
    plan = [{"name": n, "freq": profiles[n][0], "gain": profiles[n][1], "threshold": profiles[n][2],
             "neighbor_db": a.neighbor_db, "rx": [2]} for n in names]
    if a.dual or a.dual_only:
        plan.append({**plan[0], "name": "dual_ism24", "freq": profiles["ism24"][0], "gain": profiles["ism24"][1],
                     "threshold": profiles["ism24"][2], "rx": [1, 2]})
    if a.dual_only:
        plan = plan[-1:]
    if shutil.disk_usage(a.out).free < 3*2**30+len(plan)*a.max_feed_mib*2**20:
        ap.error("need the planned maximum feed sizes plus 3 GiB of free space")
    config = load_config()
    files = ["tools/run_live_checks.py", "tools/signal_policy.py", "tools/ui_server.py", "tools/policy/default.json",
             "tools/live_feed.py",
             "tools/ui/app.js", "tools/ui/index.html", "bench/classify.cpp", "src/source/omnicore_source.cpp",
             "src/runtime/signal_stage.cu", "src/kernels/extraction_bank.cu", "include/curf/runtime/signal_stage.hpp",
             "include/curf/cuda/extraction_bank.hpp"]
    manifest = {"version": 1, "created_utc": utc(), "revision": a.revision, "seconds": a.seconds, "cases": plan,
                "policy": config, "max_feed_bytes": a.max_feed_mib*2**20,
                "capacities": [a.max_signals, a.max_tracks, a.scratch_mib],
                "pending": a.pending,
                "display_interval_ms": a.display_interval_ms,
                "tune_cache": str(a.tune_cache) if a.tune_cache else None,
                "initial_tune_cache_sha256": digest(a.tune_cache) if a.tune_cache and a.tune_cache.exists() else None,
                "binary_sha256": digest(a.bin), "model_sha256": digest(a.model/"s2.curf"),
                "code_sha256": {p: digest(ROOT/p) for p in files}, "command": sys.argv,
                "scope": "Live qualitative reception; no IQ dataset collection, no noise/spur calibration, no transmission."}
    write_json(a.out/"manifest.json", manifest)
    for name in files:
        target = a.out/"source_snapshot"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    frozen = a.out/"artifacts"
    frozen.mkdir()
    shutil.copy2(a.bin, frozen/"curf_classify")
    shutil.copy2(a.model/"s2.curf", frozen/"s2.curf")
    if a.tune_cache and a.tune_cache.exists():
        shutil.copy2(a.tune_cache, frozen/"s2.tune")
    hub = Hub(SignalPolicy(config))
    Handler.hub, Handler.classes = hub, []
    server = ThreadingHTTPServer((a.bind, a.port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"serving live trials at http://{a.bind}:{a.port}/", flush=True)
    stop = threading.Event()
    watcher = threading.Thread(target=monitor, args=(a.out/"gpu.jsonl", stop), daemon=True)
    watcher.start()
    results = []
    try:
        for case in plan:
            print(f"starting {case['name']} RX{case['rx']} for {a.seconds:g}s", flush=True)
            result = record_case(a.bin, a.model, a.out/case["name"], case, a.seconds, hub, a.max_feed_mib*2**20,
                                 (a.max_signals, a.max_tracks, a.scratch_mib), a.tune_cache, a.display_interval_ms/1000, a.pending)
            results.append(result)
            write_json(a.out/"progress.json", results)
            print(json.dumps({"case": case["name"], "runtime_passed": result["runtime_passed"],
                              "policy": result["observations"]["policy"]}, ensure_ascii=False), flush=True)
        report = {"runtime_passed": all(r["runtime_passed"] for r in results), "long_runs": a.seconds >= 300,
                  "cases": len(results), "received_adsb_crc": any(r["runtime"].get("adsb", {}).get("extended", 0)>0 for r in results),
                  "results": results, "classification_accuracy": None,
                  "limits": "Runtime success is separate from qualitative demonstration readiness and classification accuracy."}
        write_json(a.out/"result.json", report)
        return 0 if report["runtime_passed"] else 1
    except BaseException as e:
        write_json(a.out/"failure.json", {"utc": utc(), "error": str(e), "completed_cases": len(results)})
        raise
    finally:
        stop.set()
        watcher.join(timeout=6)
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
