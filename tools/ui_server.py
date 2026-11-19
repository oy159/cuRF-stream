#!/usr/bin/env python3
"""Minimal live display for curf_classify: waterfall + recognition labels in the browser.

Live: runs curf_classify with --ui-out on a pipe and streams its rows to every connected browser.

    python3 tools/ui_server.py --port 8080 -- build/cuda-release/bench/curf_classify --int8 \
        --tune-cache /tmp/i8h.tune --source omnicore --freq 2.44e9 --seconds 300

Browser-controlled RX (with optional --remote user@host for an SSH-connected receiver):

    python3 tools/ui_server.py --control --auto-start --port 8080 -- \
        build/cuda-release/bench/curf_classify --burst --source omnicore --streams 1 --rx 2

Control mode listens on loopback and starts with 2441 MHz, 18 dB and no time limit.

Replay of a recorded feed (curf_classify --ui-out FEED.jsonl), at the recorded pace:

    python3 tools/ui_server.py --port 8080 --replay FEED.jsonl [--speed 1] [--loop]

Then open http://<host>:8080/. Standard library only. Rows are JSON lines (see bench/classify.cpp):
{"meta": {...}} once, then {"w", "t", "p": [S*M dBFS], "f": [S floor dB], "r": [[stream, channel, cls, prob%]]},
with optional "ofdm_cp": [estimated offset in Hz or null per receiver] and "dsss_barker": [matched RF centres in Hz per receiver].
The feed ends with {"end": {stats}}.
"""

import argparse
import json
import math
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = ROOT / "model/artifacts/run1/export/metrics.json"
RECEIVER_DEFAULTS = {"frequency_mhz": 2441, "gain_db": 18, "seconds": 0}
RECEIVER_LIMITS = {"frequency_mhz": (70, 6000), "gain_db": (-3, 76), "seconds": (0, None)}

# Keep the SSH input open while the child owns the receiver. EOF stops that
# particular child, including when the local server exits or loses the connection.
REMOTE_RECEIVER = """\
exec 3>&1 4<&0
"$@" --ui-out /dev/fd/3 1>&2 </dev/null 4<&- &
receiver=$!
(IFS= read -r ignored <&4; kill "$receiver" 2>/dev/null) 3>&- &
watcher=$!
trap 'kill "$receiver" "$watcher" 2>/dev/null' HUP INT TERM
wait "$receiver"
result=$?
kill "$watcher" 2>/dev/null
wait "$watcher" 2>/dev/null
exit "$result"
"""


class Hub:
    """Fan-out of feed lines to the connected browsers; keeps meta / last end for late joiners."""

    def __init__(self):
        self.lock = threading.Lock()
        self.clients: list[queue.Queue] = []
        self.meta: str | None = None
        self.end: str | None = None
        self.receiver: str | None = None

    def publish(self, line: str):
        with self.lock:
            important = True
            if line.startswith('{"meta"'):
                if line == self.meta:  # replay loop: keep the browsers' history
                    return
                self.meta, self.end = line, None
            elif line.startswith('{"end"'):
                self.end = line
            else:
                self.end = None
                important = False
            for q in self.clients:
                if important or q.qsize() < 64:
                    q.put(line)

    def publish_receiver(self, status: dict, reset: bool = False):
        with self.lock:
            self.receiver = json.dumps({"receiver": status})
            if reset:
                self.meta = self.end = None
            for q in self.clients:
                if reset:
                    try:
                        while True:
                            q.get_nowait()
                    except queue.Empty:
                        pass
                q.put(self.receiver)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self.lock:
            if self.receiver:
                q.put(self.receiver)
            if self.meta:
                q.put(self.meta)
            if self.end:
                q.put(self.end)
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.clients.remove(q)


def receiver_settings(data: dict) -> dict:
    settings = {}
    for key, (minimum, maximum) in RECEIVER_LIMITS.items():
        value = data.get(key)
        if (type(value) not in (int, float) or not math.isfinite(value) or value < minimum
                or (maximum is not None and value > maximum)):
            raise ValueError(key)
        if key != "frequency_mhz" and value != int(value):
            raise ValueError(key)
        settings[key] = round(value * 1e6) / 1e6 if key == "frequency_mhz" else int(value)
    return settings


class LiveReceiver:
    def __init__(self, hub: Hub, command: list[str], remote: str | None):
        self.hub, self.command, self.remote = hub, command, remote
        self.operation = threading.Lock()
        self.lock = threading.Lock()
        self.settings = RECEIVER_DEFAULTS.copy()
        self.child = None
        self.state = "idle"
        self.run_id = 0
        self._publish("idle")

    def _publish(self, state, *, reset=False, error=None):
        self.state = state
        self.hub.publish_receiver({
            "state": state, "settings": self.settings, "run": self.run_id,
            "limits": RECEIVER_LIMITS, "error": error,
        }, reset)

    def apply(self, settings: dict):
        with self.operation:
            with self.lock:
                if settings == self.settings and self.state in ("starting", "running"):
                    return
            self._stop()
            command = self.command + [
                "--freq", str(round(settings["frequency_mhz"] * 1e6)),
                "--gain", str(settings["gain_db"]), "--seconds", str(settings["seconds"]),
            ]
            if self.remote:
                command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                           "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                           "--", self.remote, shlex.join(["sh", "-c", REMOTE_RECEIVER, "curf-rx", *command])]
            else:
                command = ["sh", "-c", 'exec "$@" --ui-out /dev/fd/3 3>&1 1>&2', "curf-rx", *command]
            with self.lock:
                self.settings = settings
                self.run_id += 1
                self._publish("starting", reset=True)
                try:
                    child = self.child = subprocess.Popen(
                        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, errors="replace", start_new_session=True,
                    )
                except OSError as exc:
                    self._publish("error", error=str(exc))
                    return
            timer = threading.Timer(20, self._startup_timeout, args=(child,))
            timer.daemon = True
            timer.start()
            threading.Thread(target=self._read, args=(child, timer), daemon=True).start()

    def _read(self, child, timer):
        log = deque(maxlen=4)

        def read_errors():
            with child.stderr:
                for message in child.stderr:
                    print(message, end="", file=sys.stderr)
                    if message.strip():
                        log.append(message.strip()[:240])

        stderr = threading.Thread(target=read_errors, daemon=True)
        stderr.start()
        received = False
        with child.stdout:
            for line in child.stdout:
                line = line.strip()
                if not line.startswith(('{"meta"', '{"w"', '{"end"')):
                    continue
                with self.lock:
                    if self.child is not child or self.state == "stopping":
                        continue
                    self.hub.publish(line)
                    if not received and line.startswith('{"w"'):
                        received = True
                        timer.cancel()
                        self._publish("running")
        status = child.wait()
        timer.cancel()
        stderr.join(timeout=1)
        with self.lock:
            child.stdin.close()
            if self.child is not child or self.state == "stopping":
                return
            self.child = None
            if status == 0 and received:
                self._publish("ended")
            else:
                self._publish("error", error=log[-1] if log else f"接收程序退出（{status}）")

    def _startup_timeout(self, child):
        with self.operation:
            if self._stop(startup_child=child):
                with self.lock:
                    self._publish("error", error="20 秒内未收到采样，请重新开始接收。")

    def stop(self):
        with self.operation:
            self._stop()

    def _stop(self, *, startup_child=None):
        with self.lock:
            child = self.child
            if child is None or (startup_child is not None and
                                 (child is not startup_child or self.state != "starting")):
                return False
            self._publish("stopping")
            child.stdin.close()
            if not self.remote:
                child.terminate()
        try:
            child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        with self.lock:
            self.child = None
            self._publish("idle")
        return True


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
    receiver: LiveReceiver | None = None

    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict, status: int):
        self._send(json.dumps(data).encode(), "application/json", status)

    def do_POST(self):
        if self.path != "/receiver" or not self.receiver:
            self.send_error(404)
            return
        port = self.server.server_port
        hosts = {"127.0.0.1", "localhost"}
        allowed = {f"{host}:{port}" for host in hosts}
        if port == 80:
            allowed |= hosts
        host = self.headers.get("Host", "").lower()
        origin = self.headers.get("Origin")
        if host not in allowed or (origin and origin.lower() != f"http://{host}"):
            self._json({"error": "请从本机可视化页面控制接收。"}, 403)
            return
        if self.headers.get_content_type() != "application/json":
            self._json({"error": "接收设置需要 JSON 数据。"}, 415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2048:
                raise ValueError
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError
        except (ValueError, UnicodeError):
            self._json({"error": "无法读取接收设置。"}, 400)
            return
        action = data.get("action")
        if action == "apply":
            try:
                settings = receiver_settings(data)
            except (ValueError, OverflowError):
                self._json({"error": "中心频率、增益或时长超出可用范围。"}, 422)
                return
            self.receiver.apply(settings)
        elif action == "stop":
            self.receiver.stop()
        else:
            self._json({"error": "无法识别接收操作。"}, 400)
            return
        self._json({}, 202)

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
    ap.add_argument("--bind", help="listen address (loopback by default with --control)")
    ap.add_argument("--replay", type=Path, help="recorded --ui-out feed instead of a live command")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--control", action="store_true", help="enable browser control of the live receiver")
    ap.add_argument("--auto-start", action="store_true", help="start controlled reception when the server opens")
    ap.add_argument("--remote", help="SSH destination for the controlled receiver")
    ap.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="model metrics.json with the class names")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- curf_classify ... (live mode)")
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if bool(a.replay) == bool(cmd):
        ap.error("give either --replay FEED or -- COMMAND")
    if (a.remote or a.auto_start) and not a.control:
        ap.error("--remote and --auto-start require --control")
    bind = a.bind or ("127.0.0.1" if a.control else "0.0.0.0")
    if a.control and (not cmd or bind not in ("127.0.0.1", "localhost")):
        ap.error("--control requires a live command and a loopback bind address")
    if a.control and any(option in cmd for option in ("--tx-file", "--ui-out")):
        ap.error("--control requires a receive command without --tx-file or --ui-out")

    hub = Hub()
    Handler.hub = hub
    try:
        Handler.classes = json.loads(a.metrics.read_text())["classes"]
    except OSError:
        Handler.classes = []
    receiver = LiveReceiver(hub, cmd, a.remote) if a.control else None
    Handler.receiver = receiver
    with ThreadingHTTPServer((bind, a.port), Handler) as srv:
        try:
            if receiver:
                def stop_server(_signum, _frame):
                    raise KeyboardInterrupt

                signal.signal(signal.SIGTERM, stop_server)
                if a.auto_start:
                    receiver.apply(RECEIVER_DEFAULTS)
            else:
                target = (lambda: run_replay(hub, a.replay, a.speed, a.loop)) if a.replay else (lambda: run_live(hub, cmd))
                threading.Thread(target=target, daemon=True).start()
            print(f"serving on http://{bind}:{srv.server_port}/", file=sys.stderr)
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if receiver:
                receiver.stop()


if __name__ == "__main__":
    main()
