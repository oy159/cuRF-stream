"""Signal display policy shared by live SSE and offline replay. Standard library only.

Raw model probabilities are evidence, never rewritten. The allowed-set posterior is
reported alongside its original mass and raw confidence; a one-class prior cannot
turn a weak model output into a trusted label. All state uses source sample time.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re

FAMILIES = ["noise", "ofdm", "fsk", "4fsk", "psk", "qam", "dsss", "chirp", "ook", "am", "fm", "pulse", "unknown"]
NAMED = set(FAMILIES) - {"noise", "unknown"}
DEFAULT_CONFIG = Path(__file__).with_name("policy") / "default.json"


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_config(path=None):
    config = json.loads(DEFAULT_CONFIG.read_text())
    if path:
        override = json.loads(Path(path).read_text())
        if set(override) - set(config):
            raise ValueError("unknown policy configuration key")
        config.update(override)
    if config["version"] != 1:
        raise ValueError("unsupported policy version")
    for key in ("confidence", "raw_confidence", "allowed_mass", "margin", "other_confidence", "hold_confidence"):
        if not finite(config[key]) or not 0 < config[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    for key in ("confirmations", "release_observations", "max_carriers"):
        if type(config[key]) is not int or not 1 <= config[key] <= 4096:
            raise ValueError(f"invalid {key}")
    for key in ("vote_seconds", "hold_seconds"):
        if not finite(config[key]) or not 0 < config[key] <= 30:
            raise ValueError(f"invalid {key}")
    if config["hold_confidence"] > config["confidence"] or config["other_confidence"] < config["confidence"]:
        raise ValueError("invalid confidence hysteresis")
    if config["spur_action"] not in ("hide", "mark"):
        raise ValueError("spur_action must be hide or mark")
    for b in config["bands"]:
        if not finite(b["lo_mhz"]) or not finite(b["hi_mhz"]) or not 0 < b["lo_mhz"] < b["hi_mhz"]:
            raise ValueError("invalid band edges")
        if not set(b["allow"]) <= NAMED:
            raise ValueError("invalid band classes")
    return config


def calibration_context(meta):
    """Exact receiver settings: do not apply a spur recording to another RF setup."""
    keys = ("source", "freq", "rate", "streams", "rx", "receiver")
    return {k: meta.get(k) for k in keys}


def load_spurs(path):
    if not path:
        return None
    table = json.loads(Path(path).read_text())
    if table.get("version") != 1 or table.get("termination") != "50ohm":
        raise ValueError("spur table requires a version-1 50ohm calibration")
    context = table.get("context", {})
    if context.get("source") != "omnicore" or not context.get("receiver"):
        raise ValueError("spur table lacks receiver settings")
    if not isinstance(table.get("entries"), list):
        raise ValueError("spur table requires an entries list")
    for e in table["entries"]:
        if type(e.get("stream")) is not int or e["stream"] < 0 or not all(finite(e.get(k)) for k in ("lo_hz", "hi_hz")) or e["lo_hz"] >= e["hi_hz"]:
            raise ValueError("invalid spur interval")
    return table


@dataclass
class Carrier:
    uid: int
    stream: int
    fc: float
    bw: float
    first: float
    last: float
    raw_id: int
    votes: deque = field(default_factory=deque)
    stable: str | None = None
    bad: int = 0
    accepted_at: float = -math.inf
    last_window: int = -1
    duty: float | None = None


class SignalPolicy:
    def __init__(self, config=None, spurs=None, classes=None):
        self.config = config or load_config()
        self.spurs = spurs
        self.fallback_classes = classes
        self.reset({})

    def reset(self, meta):
        if not isinstance(meta, dict):
            raise ValueError("metadata must be an object")
        if meta:
            if not finite(meta.get("rate")) or meta["rate"] <= 0:
                raise ValueError("metadata requires a positive sample rate")
            if type(meta.get("streams")) is not int or not 1 <= meta["streams"] <= 4:
                raise ValueError("metadata requires 1..4 streams")
            if not finite(meta.get("freq", 0)):
                raise ValueError("invalid RF centre")
            if "signal_classes" in meta and (not isinstance(meta["signal_classes"], list) or not all(isinstance(c, str) for c in meta["signal_classes"])):
                raise ValueError("invalid class names")
        self.meta = dict(meta)
        self.classes = list(meta.get("signal_classes") or self.fallback_classes or FAMILIES)
        self.carriers = {}
        self.next_uid = 1
        self.now = 0.0
        self.counts = Counter()
        self.spur_status = "not_loaded" if not self.spurs else (
            "active" if self.spurs["context"] == calibration_context(meta) else "context_mismatch")

    def stats(self):
        n, u = self.counts["named"], self.counts["unknown"]
        return {**{k: self.counts[k] for k in ("records", "named", "unknown", "spur", "decoder_only", "decoded", "invalid_adsb")},
                "unknown_fraction": u / (n + u) if n + u else None,
                "denominator": "named + unknown signal records", "active_carriers": len(self.carriers)}

    def process(self, message):
        """Return a new envelope; raw sig/adsb/row data remains byte-value equivalent."""
        if not isinstance(message, dict):
            raise ValueError("feed records must be JSON objects")
        if "meta" in message:
            self.reset(message["meta"])
            meta = {**message["meta"], "signal_classes": self.classes,
                    "policy": {"version": 1, "config": self.config, "spur_status": self.spur_status}}
            return {**message, "meta": meta}
        if "sig" in message:
            if not isinstance(message["sig"], dict):
                raise ValueError("signal must be an object")
            display = self.signal(message["sig"])
            return {**message, "display": display, "policy_stats": self.stats()}
        if "adsb" in message:
            a = message["adsb"]
            if not isinstance(a, dict):
                raise ValueError("ADS-B record must be an object")
            stream = a.get("s", self.meta.get("adsb", {}).get("stream", 0))
            freq = a.get("freq", self.meta.get("adsb", {}).get("freq", 1090e6))
            valid = (a.get("crc_ok") is True and a.get("df") in (17, 18) and finite(a.get("t")) and
                     isinstance(a.get("icao"), str) and re.fullmatch(r"[0-9A-Fa-f]{6}", a["icao"]) and
                     type(stream) is int and 0 <= stream < self.meta.get("streams", 1) and finite(freq) and abs(freq-1090e6) <= 1e6)
            if valid:
                self.counts["decoded"] += 1
                display = {"uid": f"adsb:{stream}", "s": stream, "f": freq - self.meta.get("freq", 0), "freq": freq,
                           "bw": 2e6, "t": a["t"], "t0": a["t"], "t1": a["t"], "family": "pulse", "status": "decoded",
                           "label": "ADS-B · CRC 通过", "protocol": "ADS-B", "protocol_evidence": "crc", "confidence": None,
                           "visible": True, "icao": a["icao"], "db": a.get("sig"), "ttl_s": 2.0}
                return {**message, "display": display, "policy_stats": self.stats()}
            self.counts["invalid_adsb"] += 1
            return {**message, "adsb_rejected": True, "policy_stats": self.stats()}
        if "end" in message:
            return {**message, "end": {**message["end"], "policy": self.stats()}}
        if finite(message.get("t")):
            self.now = max(self.now, message["t"])
            self.expire(self.now)
        return message

    def expire(self, now):
        for uid in [k for k, v in self.carriers.items() if now - v.last > self.config["vote_seconds"]]:
            del self.carriers[uid]

    def carrier(self, r, now):
        self.expire(now)
        candidates = []
        for c in self.carriers.values():
            tolerance = max(5000, .20 * min(c.bw, r["bw"]))
            if c.stream == r["s"] and abs(c.fc - r["f"]) <= tolerance and max(c.bw, r["bw"]) <= 2.5 * min(c.bw, r["bw"]):
                candidates.append((abs(c.fc - r["f"]) / tolerance, c.uid, c))
        if candidates:
            c = min(candidates)[2]
        else:
            if len(self.carriers) >= self.config["max_carriers"]:
                del self.carriers[min(self.carriers, key=lambda k: self.carriers[k].last)]
            c = Carrier(self.next_uid, r["s"], r["f"], r["bw"], now, now, r.get("id", -1))
            self.carriers[c.uid] = c
            self.next_uid += 1
        c.fc, c.bw, c.last, c.raw_id = r["f"], r["bw"], now, r.get("id", -1)
        activity = r.get("activity_fraction")
        if finite(activity) and 0 <= activity <= 1:
            c.duty = activity if c.duty is None else .65 * c.duty + .35 * activity
        return c

    def band(self, freq):
        if self.meta.get("freq", 0) <= 0:
            return None
        return next((b for b in self.config["bands"] if b["lo_mhz"] * 1e6 <= freq < b["hi_mhz"] * 1e6), None)

    def is_spur(self, r):
        if self.spur_status != "active":
            return False
        lo, hi = r["f"] - r["bw"] / 2, r["f"] + r["bw"] / 2
        return any(e["stream"] == r["s"] and max(0, min(hi, e["hi_hz"]) - max(lo, e["lo_hz"])) >= .8 * r["bw"]
                   for e in self.spurs["entries"])

    def protocol(self, family, band, r, c, now):
        kind = band.get("protocol") if band else None
        burst = c.duty is not None and c.duty < .65
        continuous = c.duty is not None and c.duty >= .8 and now - c.first >= .3
        bw = r["bw"]
        if kind in ("ism24", "wlan5") and family == "ofdm" and 16e6 <= bw <= 22e6 and burst:
            return "WiFi（推测）", "band + OFDM + 16–22 MHz + burst activity"
        if kind == "ism24" and family == "dsss" and 8e6 <= bw <= 24e6 and burst:
            return "WiFi DSSS（推测）", "band + DSSS + bandwidth + burst activity"
        if kind == "ism24" and family == "fsk" and .5e6 <= bw <= 2.5e6 and burst:
            return "蓝牙 / BLE（推测）", "band + FSK + bandwidth + burst activity"
        if kind == "ism24" and family == "dsss" and 1e6 <= bw <= 3e6 and burst:
            return "Zigbee（推测）", "band + DSSS + bandwidth + burst activity"
        if kind in ("cellular", "lte_nr", "nr") and family == "ofdm" and .8e6 <= bw <= 22e6 and continuous:
            return {"cellular": "4G LTE（推测）", "lte_nr": "LTE / NR（推测）", "nr": "5G NR（推测）"}[kind], "band + OFDM + bandwidth + sustained activity"
        if kind == "cellular" and band.get("gsm") and family == "fsk" and .1e6 <= bw <= .35e6 and continuous:
            return "GSM（推测）", "band + FSK + bandwidth + sustained activity"
        return None, None

    def signal(self, r):
        cfg = self.config
        self.counts["records"] += 1
        if not all(finite(r.get(k)) for k in ("f", "bw", "t0", "t1")) or r["bw"] <= 0 or r["t1"] < r["t0"] or type(r.get("s")) is not int or not 0 <= r["s"] < self.meta.get("streams", 1):
            self.counts["unknown"] += 1
            return {"status": "unknown", "family": "unknown", "label": "未识别信号", "reason": "invalid_record", "visible": False}
        hop = self.meta.get("window", 1024) * self.meta.get("channels", 512) / 2 / self.meta.get("rate", 61.44e6)
        now = r.get("t", (r.get("w", 0) + 1) * hop)
        if not finite(now):
            now = r["t1"]
        self.now = max(self.now, now)
        c = self.carrier(r, now)
        freq = self.meta.get("freq", 0) + r["f"]
        band = self.band(freq)
        base = {"uid": c.uid, "s": r["s"], "id": r.get("id"), "w": r.get("w"), "f": r["f"], "freq": freq,
                "bw": r["bw"], "db": r.get("db"), "t": now, "t0": r["t0"], "t1": r["t1"], "age_s": now-c.first,
                "processing_ms": r.get("processing_ms"), "observation_ms": r.get("observation_ms"),
                "band": band["name"] if band else "其他 / 无射频先验", "activity_fraction": c.duty,
                "family": "unknown", "status": "unknown", "label": "未识别信号", "protocol": None, "visible": True,
                "ttl_s": cfg["vote_seconds"]}
        if band and band.get("decoder_only"):
            self.counts["decoder_only"] += 1
            return {**base, "status": "decoder_only", "reason": "decoder_only", "visible": False}
        if self.is_spur(r):
            self.counts["spur"] += 1
            return {**base, "status": "spur", "label": "本机杂散", "reason": "calibrated_spur", "visible": cfg["spur_action"] == "mark"}
        probs = r.get("probs")
        valid = (isinstance(probs, list) and len(probs) == len(self.classes) and all(finite(p) and 0 <= p <= 1 for p in probs)
                 and abs(sum(probs) - 1) < .02 and len(set(self.classes)) == len(self.classes))
        allowed = set(band["allow"]) if band else NAMED
        base["allowed"] = sorted(allowed)
        family, reason, admissible = None, "missing_probabilities", False
        if valid:
            raw = dict(zip(self.classes, probs))
            mass = sum(raw.get(k, 0) for k in allowed)
            adjusted = {k: raw.get(k, 0) / mass if mass > 0 else 0 for k in sorted(allowed)}
            ranking = sorted(adjusted, key=lambda k: (-adjusted[k], k))
            family = ranking[0] if ranking else None
            posterior = adjusted.get(family, 0)
            second = adjusted[ranking[1]] if len(ranking) > 1 else 0
            raw_p = raw.get(family, 0)
            threshold = cfg["confidence"] if band else cfg["other_confidence"]
            base.update(raw=raw, adjusted=adjusted, allowed_mass=mass, confidence=raw_p, posterior=posterior, margin=posterior-second, candidate=family)
            reason = ("allowed_mass" if mass < cfg["allowed_mass"] else "raw_confidence" if raw_p < (cfg["raw_confidence"] if band else threshold)
                      else "low_confidence" if posterior < threshold else "small_margin" if posterior-second < cfg["margin"] else "confirming")
            admissible = reason == "confirming"
        while c.votes and now - c.votes[0][0] > cfg["vote_seconds"]:
            c.votes.popleft()
        window = r.get("w", now)
        fresh = window != c.last_window
        if fresh:
            c.last_window = window
            if admissible:
                if c.votes and c.votes[-1][1] != family:
                    c.votes.clear()
                c.votes.append((now, family))
                # Bound retained evidence even for high-rate indefinitely active carriers.
                while len(c.votes) > cfg["confirmations"]:
                    c.votes.popleft()
            else:
                c.votes.clear()
        base["confirmations"] = len(c.votes)
        if admissible and len(c.votes) >= cfg["confirmations"]:
            c.stable, c.bad, c.accepted_at = family, 0, now
            base.update(family=family, status="stable", label=family.upper(), reason="confirmed")
        elif (valid and family == c.stable and now - c.accepted_at <= cfg["hold_seconds"] and
              base["allowed_mass"] >= cfg["hold_confidence"] and base["confidence"] >= cfg["hold_confidence"] and
              base["posterior"] >= cfg["hold_confidence"] and base["margin"] >= cfg["margin"] / 2):
            base.update(family=c.stable, status="held", label=c.stable.upper(), reason="hysteresis")
        else:
            if fresh:
                c.bad += 1
                if c.bad >= cfg["release_observations"]:
                    c.stable = None
            base["reason"] = reason
        if base["family"] != "unknown":
            proto, evidence = self.protocol(base["family"], band, r, c, now)
            base.update(protocol=proto, protocol_evidence=evidence)
            self.counts["named"] += 1
        else:
            self.counts["unknown"] += 1
        return base
