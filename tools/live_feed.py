"""Bounded display reduction over a complete, independently written model feed."""
import json
from pathlib import Path
import shutil
import subprocess
import time


class DisplayReducer:
    def __init__(self, interval, max_pending=8192):
        if interval <= 0 or max_pending < 1:
            raise ValueError("positive display interval and capacity required")
        self.interval, self.max_pending = interval, max_pending
        self.bucket, self.pending, self.sequence = None, {}, 0
        self.input_signals = self.output_signals = self.peak_pending = 0

    def flush(self):
        rows = [r for _, _, r in sorted(self.pending.values(), key=lambda x: x[:2])]
        self.output_signals += sum("sig" in r for r in rows)
        self.pending.clear()
        return rows

    def push(self, row):
        if "meta" in row:
            output = self.flush()
            row["meta"]["display_sampling"] = {"mode": "latest_per_track", "interval_s": self.interval,
                                                "raw_file": "raw.feed"}
            self.bucket = None
            return output + [row]
        if "sig" in row:
            s = row["sig"]
            key, stamp = ("sig", s["s"], s["id"]), s["t"]
            self.input_signals += 1
        elif "p" in row and "t" in row:
            # Power rows are already only ~30 Hz. Keep the measured waterfall
            # cadence while flushing overdue signal updates at bucket boundaries.
            bucket = int(row["t"] / self.interval)
            output = []
            if self.bucket is None or bucket > self.bucket:
                output = self.flush()
                self.bucket = bucket
            return output + [row]
        else:
            # In particular, preserve every decoded ADS-B message and the end record.
            return self.flush() + [row]
        bucket = int(stamp / self.interval)
        output = []
        if self.bucket is None or bucket > self.bucket:
            output = self.flush()
            self.bucket = bucket
        if key not in self.pending and len(self.pending) >= self.max_pending:
            output += self.flush()
        self.pending[key] = stamp, self.sequence, row
        self.sequence += 1
        self.peak_pending = max(self.peak_pending, len(self.pending))
        return output

    def stats(self):
        return {"interval_s": self.interval, "input_signals": self.input_signals,
                "output_signals": self.output_signals,
                "coalesced_signals": self.input_signals-self.output_signals,
                "peak_pending": self.peak_pending, "capacity": self.max_pending}


def capture_file_feed(cmd, out, hub, max_bytes, interval, max_pending):
    """Child writes raw JSON directly; display processing cannot block its pipe.

    Linux prlimit bounds the raw file to 3/4 of the per-case budget. The display
    writer checks its 1/4 quota before each write. Both originals remain on error.
    """
    prlimit = shutil.which("prlimit")
    if not prlimit:
        raise RuntimeError("file-backed live display requires Linux prlimit")
    raw, display = out/"raw.feed", out/"display.feed"
    raw.touch(exist_ok=False)
    raw_limit = max_bytes * 3 // 4
    display_limit = max_bytes - raw_limit
    reducer = DisplayReducer(interval, max_pending)
    written = unread_peak = lines = 0
    proc = None
    try:
        with (out/"runtime.json").open("w") as stats, (out/"stderr.log").open("w") as err:
            proc = subprocess.Popen([prlimit, f"--fsize={raw_limit}:{raw_limit}", "--", *cmd,
                                     "--ui-out", str(raw)], stdout=stats, stderr=err)
            with raw.open() as source, display.open("w", buffering=1) as target:
                partial = ""

                def emit(rows):
                    nonlocal written
                    for row in rows:
                        value = hub.publish(json.dumps(row, separators=(",", ":")))+"\n"
                        n = len(value.encode("utf-8"))
                        if written+n > display_limit:
                            raise RuntimeError("diagnostic display feed limit exceeded; incomplete run retained")
                        target.write(value)
                        written += n

                while True:
                    line = source.readline()
                    if line:
                        partial += line
                        if len(partial) > 2**20:
                            raise RuntimeError("raw feed line exceeds bounded parser capacity")
                        if partial.endswith("\n"):
                            emit(reducer.push(json.loads(partial)))
                            partial = ""
                            lines += 1
                            if lines % 128 == 0:
                                unread_peak = max(unread_peak, raw.stat().st_size-source.tell())
                    elif proc.poll() is not None:
                        if source.tell() < raw.stat().st_size:
                            continue # the child may have appended between EOF and poll
                        if partial:
                            raise RuntimeError("incomplete raw feed line")
                        break
                    else:
                        unread_peak = max(unread_peak, raw.stat().st_size-source.tell())
                        time.sleep(.005)
                emit(reducer.flush())
            if proc.wait() != 0:
                raise RuntimeError(f"receiver exited {proc.returncode}; raw/display evidence retained")
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    sampling = {**reducer.stats(), "raw_limit_bytes": raw_limit, "display_limit_bytes": display_limit,
                "raw_bytes": raw.stat().st_size, "display_bytes": written,
                "peak_unread_bytes": unread_peak}
    (out/"display_sampling.json").write_text(json.dumps(sampling, indent=2)+"\n")
    return sampling
