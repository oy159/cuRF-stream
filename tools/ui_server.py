#!/usr/bin/env python3
"""Minimal live display for curf_classify: waterfall + recognition labels in the browser.

Live: runs curf_classify with --ui-out on a pipe and streams its rows to every connected browser.

    python3 tools/ui_server.py --port 8080 -- build/cuda-release/bench/curf_classify --int8 \
        --tune-cache /tmp/i8h.tune --source omnicore --freq 2.44e9 --seconds 300

Replay of a recorded feed (curf_classify --ui-out FEED.jsonl), at the recorded pace:

    python3 tools/ui_server.py --port 8080 --replay FEED.jsonl [--speed 1] [--loop]

Then open http://<host>:8080/. Standard library only. Rows are JSON lines (see bench/classify.cpp):
{"meta": {...}} once, then {"w", "t", "p": [S*M dBFS], "f": [S floor dB], "r": [[stream, channel, cls, prob%]]},
and {"end": {stats}} at the end.
"""

import argparse
from collections import OrderedDict, deque
import json
import os
import queue
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from signal_policy import SignalPolicy, finite, load_config, load_spurs

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = ROOT / "model/artifacts/run2/export/metrics.json"


class Hub:
    """Fan-out of feed lines to the connected browsers; keeps meta / last end for late joiners."""

    def __init__(self, policy=None, reprocess=False):
        self.lock = threading.Lock()
        self.clients: list[queue.Queue] = []
        self.meta: str | None = None
        self.end: str | None = None
        self.rows = 0
        self.policy = policy or SignalPolicy()
        self.processed = False
        self.reprocess = reprocess
        self.last_row = None
        self.power_history = deque(maxlen=250)
        self.signals = OrderedDict()
        self.aircraft = OrderedDict()
        self.error = None
        self.now = 0
        self.latest_stats = None

    def publish(self, line: str):
        with self.lock:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("feed record must be an object")
            is_meta = "meta" in message
            if is_meta:
                self.processed = "policy" in message["meta"] and not self.reprocess
                self.end = self.last_row = self.error = None
                self.signals.clear()
                self.power_history.clear()
                self.aircraft.clear()
                self.now = 0
                self.latest_stats = None
            if not self.processed:
                message = self.policy.process(message)
            line = json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            if "policy_stats" in message:
                self.latest_stats = json.dumps({"policy_stats": message["policy_stats"]}, separators=(",", ":"))
            if is_meta:
                self.meta = line
            elif "end" in message:
                self.end = line
            elif "error" in message:
                self.error = line
            else:
                self.rows += 1
                self.end = None
                if "p" in message:
                    self.last_row = line
                    self.power_history.append(line)
                    self.now = max(self.now, message.get("t", 0))
                d = message.get("display", {})
                if "uid" in d:
                    self.now = max(self.now, d["t"])
                    self.signals[d["uid"]] = (line, d["t"], d.get("ttl_s", 3))
                    self.signals.move_to_end(d["uid"])
                    while len(self.signals) > 512:
                        self.signals.popitem(last=False)
                a = message.get("adsb")
                if a and not message.get("adsb_rejected") and a.get("crc_ok") is True:
                    self.aircraft[a["icao"]] = line
                    self.aircraft.move_to_end(a["icao"])
                    while len(self.aircraft) > 128:
                        self.aircraft.popitem(last=False)
            for q in self.clients:
                if is_meta:
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break
                if not q.full():  # preserve atomic raw + policy records, with a bounded backlog
                    q.put_nowait(line)
        return line

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1024)
        with self.lock:
            seen = set()

            def add(line):
                if line not in seen:
                    q.put_nowait(line)
                    seen.add(line)

            if self.meta:
                add(self.meta)
            for line in self.power_history:
                add(line)
            for line, t, ttl in self.signals.values():
                if self.now - t <= ttl:
                    add(line)
            for line in self.aircraft.values():
                add(line)
            if self.latest_stats:
                add(self.latest_stats)
            if self.end:
                add(self.end)
            if self.error:
                add(self.error)
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.clients.remove(q)


def run_live(hub: Hub, cmd: list[str], tee: Path | None = None):
    r, w = os.pipe()
    proc = subprocess.Popen(cmd + ["--ui-out", f"/dev/fd/{w}"], pass_fds=(w,))
    os.close(w)
    out = open(tee, "w", buffering=1) if tee else None
    with os.fdopen(r, "r") as f:
        for line in f:
            published = hub.publish(line.strip())
            if out:
                out.write(published + "\n")
    if out:
        out.close()
    rc = proc.wait()
    if rc:
        hub.publish(json.dumps({"error": {"message": f"采集程序退出，状态码 {rc}；请检查终端日志。"}}))
    print(f"curf_classify exited with {rc}; still serving the last state", file=sys.stderr)


def run_replay(hub: Hub, path: Path, speed: float, loop: bool):
    while True:
        t0 = w0 = None
        meta = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "meta" in row:
                    meta = row["meta"]
                t = row.get("t")
                if "sig" in row:
                    sig = row["sig"]
                    hop = meta.get("window", 1024)*meta.get("channels", 512)/2/meta.get("rate", 61.44e6)
                    t = sig.get("t", (sig.get("w", 0)+1)*hop)
                elif "adsb" in row:
                    t = row["adsb"].get("t")
                if finite(t):
                    if t0 is None:
                        t0, w0 = time.monotonic(), t
                    delay = t0 + (t - w0) / speed - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                hub.publish(line)
        if not loop:
            return
        time.sleep(1)


class Handler(BaseHTTPRequestHandler):
    hub: Hub
    classes: list[str]

    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send((ROOT / "tools/ui/index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path.split("?", 1)[0] in ("/", "/index.html"):
            self._send((ROOT / "tools/ui/index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/app.js":
            self._send((ROOT / "tools/ui/app.js").read_bytes(), "text/javascript; charset=utf-8")
        elif self.path == "/classes":
            self._send(json.dumps(self.classes).encode(), "application/json")
        elif self.path == "/feed":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            q = self.hub.subscribe()
            try:
                while True:
                    try:
                        line = q.get(timeout=15)
                        self.wfile.write(b"data: " + line.encode() + b"\n\n")
                    except queue.Empty:
                        self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                self.hub.unsubscribe(q)
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--replay", type=Path, help="recorded --ui-out feed instead of a live command")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--tee", type=Path, help="live mode: also save the feed to this file (replayable)")
    ap.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="model metrics.json with the class names")
    ap.add_argument("--policy", type=Path, help="JSON overrides for the signal display policy")
    ap.add_argument("--spurs", type=Path, help="50 ohm spur table; used only with matching receiver settings")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- curf_classify ... (live mode)")
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if bool(a.replay) == bool(cmd):
        ap.error("give either --replay FEED or -- COMMAND")
    if not finite(a.speed) or a.speed <= 0:
        ap.error("--speed must be positive and finite")

    hub = Hub(SignalPolicy(load_config(a.policy), load_spurs(a.spurs)), reprocess=bool(a.policy or a.spurs))
    Handler.hub = hub
    try:
        Handler.classes = json.loads(a.metrics.read_text())["classes"]
    except OSError:
        Handler.classes = []
    def target():
        try:
            if a.replay:
                run_replay(hub, a.replay, a.speed, a.loop)
            else:
                run_live(hub, cmd, a.tee)
        except (OSError, ValueError, KeyError) as e:
            print(f"feed failed: {e}", file=sys.stderr)
            hub.publish(json.dumps({"error": {"message": "数据流读取失败；请检查 feed 格式与终端日志。"}}))
    threading.Thread(target=target, daemon=True).start()
    srv = ThreadingHTTPServer((a.bind, a.port), Handler)
    srv.daemon_threads = True
    print(f"serving on http://{a.bind}:{a.port}/", file=sys.stderr)
    srv.serve_forever()


if __name__ == "__main__":
    main()
