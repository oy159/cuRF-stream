#!/usr/bin/env python3
"""Minimal live display for curf_classify: waterfall + recognition labels in the browser.

Live: runs curf_classify with --ui-out on a pipe and streams its rows to every connected browser.

    python3 tools/ui_server.py --port 8080 -- build/cuda-release/bench/curf_classify --int8 \
        --tune-cache /tmp/i8h.tune --source omnicore --freq 2.44e9 --seconds 300

Replay of a recorded feed (curf_classify --ui-out FEED.jsonl), at the recorded pace:

    python3 tools/ui_server.py --port 8080 --replay FEED.jsonl [--speed 1] [--loop]

Then open http://<host>:8080/. Standard library only. Rows are JSON lines (see bench/classify.cpp):
{"meta": {...}} once, then {"w", "t", "p": [S*M dBFS], "f": [S floor dB], "r": [[stream, channel, cls, prob%]]},
with optional "ofdm_cp": [estimated offset in Hz or null per receiver], and {"end": {stats}} at the end.
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = ROOT / "model/artifacts/run1/export/metrics.json"


class Hub:
    """Fan-out of feed lines to the connected browsers; keeps meta / last end for late joiners."""

    def __init__(self):
        self.lock = threading.Lock()
        self.clients: list[queue.Queue] = []
        self.meta: str | None = None
        self.end: str | None = None
        self.rows = 0

    def publish(self, line: str):
        if line.startswith('{"meta"'):
            if line == self.meta:  # replay loop: same stream again, keep the browsers' history
                return
            self.meta, self.end = line, None
        elif line.startswith('{"end"'):
            self.end = line
        else:
            self.rows += 1
            self.end = None
        with self.lock:
            for q in self.clients:
                if q.qsize() < 64:  # a slow browser skips rows instead of growing a backlog
                    q.put(line)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self.lock:
            if self.meta:
                q.put(self.meta)
            if self.end:
                q.put(self.end)
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.clients.remove(q)


def run_live(hub: Hub, cmd: list[str]):
    r, w = os.pipe()
    proc = subprocess.Popen(cmd + ["--ui-out", f"/dev/fd/{w}"], pass_fds=(w,))
    os.close(w)
    with os.fdopen(r, "r") as f:
        for line in f:
            hub.publish(line.strip())
    rc = proc.wait()
    print(f"curf_classify exited with {rc}; still serving the last state", file=sys.stderr)


def run_replay(hub: Hub, path: Path, speed: float, loop: bool):
    while True:
        t0 = w0 = None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('{"meta"'):
                    record = json.loads(line)
                    record["meta"].update(replay=True, loop=loop)
                    line = json.dumps(record, separators=(",", ":"))
                elif line.startswith('{"w"'):
                    # pace by the sample time of the row: a file source's feed is not produced in real time
                    head = line[: line.index(',"p"')] + "}"
                    t = json.loads(head)["t"]
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
    ap.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="model metrics.json with the class names")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- curf_classify ... (live mode)")
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if bool(a.replay) == bool(cmd):
        ap.error("give either --replay FEED or -- COMMAND")

    hub = Hub()
    Handler.hub = hub
    try:
        Handler.classes = json.loads(a.metrics.read_text())["classes"]
    except OSError:
        Handler.classes = []
    target = (lambda: run_replay(hub, a.replay, a.speed, a.loop)) if a.replay else (lambda: run_live(hub, cmd))
    threading.Thread(target=target, daemon=True).start()
    srv = ThreadingHTTPServer((a.bind, a.port), Handler)
    srv.daemon_threads = True
    print(f"serving on http://{a.bind}:{a.port}/", file=sys.stderr)
    srv.serve_forever()


if __name__ == "__main__":
    main()
