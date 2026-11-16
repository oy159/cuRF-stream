#!/usr/bin/env python3
"""Record real over-the-air IQ for the dataset: one short capture per band, saved as SigMF.

    python3 tools/capture_survey.py --out ~/dataset/raw/$(date +%Y%m%d_%H%M) [--bands fm,ism433,...] \
        [--seconds 1] [--gain 40] [--rx 2]

Each band becomes <name>.sigmf-data (ci16_le: the 12-bit samples shifted left by 4, full scale = 32768) +
<name>.sigmf-meta, recorded with `och rx` at 61.44 MS/s (the pipeline's rate) on one receiver. A capture with
clipped samples is recorded again 10 dB lower (strong nearby transmitters, e.g. the host's own WiFi). Annotations are left empty: labelling is a
separate step (decoders such as OnAir run on the files as an external tool, plus our own detector).
Bands are chosen for China; 56 MHz instantaneous bandwidth around each centre.
"""

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

OCH = Path.home() / "opt/omnicore/0.8.0/bin/och"
RATE = 61.44e6

# name: (centre Hz, what is expected in +-28 MHz)
BANDS = {
    "fm": (98.0e6, "FM broadcast (WBFM stereo + RDS)"),
    "air": (125.0e6, "FM broadcast top, VOR/ILS, airband AM voice, ACARS/VDL2, 137 MHz weather satellites"),
    "uhf400": (400.0e6, "PDT/DMR/TETRA trunked radio, 433 MHz ISM lower edge, radiosondes"),
    "ism433": (433.92e6, "433 MHz ISM: OOK/FSK remotes and sensors, LoRa"),
    "gsm900": (935.0e6, "920-925 MHz LoRa/ISM, GSM/LTE/NR 900 MHz downlink"),
    "adsb": (1100.0e6, "Mode S / ADS-B 1090 MHz, DME"),
    "lte1800": (1840.0e6, "LTE/NR 1800 MHz downlink"),
    "lte2100": (2140.0e6, "UMTS/LTE/NR 2.1 GHz downlink"),
    "ism2440": (2440.0e6, "WiFi 2.4 GHz, BLE, Bluetooth, Zigbee, drones"),
    "nr3500": (3500.0e6, "5G NR n78"),
    "wifi5800": (5785.0e6, "WiFi 5.8 GHz, drone video"),
}


def record(raw: Path, freq: float, gain: float, a) -> float:
    """One och capture; returns the share of clipped samples."""
    cmd = [str(OCH), "rx", "--file", str(raw), "--overwrite", "--sample-rate", f"{RATE / 1e6:g}M", "--channels", str(a.rx),
           "--format", "sc16", "--lo-frequency", f"{freq / 1e6:g}M", "--gain", f"{gain:g}", "--seconds", f"{a.seconds:g}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"och rx failed:\n{r.stdout}\n{r.stderr}")
    x = np.fromfile(raw, dtype="<i2")
    return float(np.mean(np.abs(x) >= 2047 * 16))


def capture(name: str, freq: float, note: str, a) -> dict:
    out = a.out / name
    raw = out.with_suffix(".sc16")
    gain = a.gain
    t0 = dt.datetime.now(dt.timezone.utc)
    clip = record(raw, freq, gain, a)
    while clip > 1e-6 and gain - 10 >= 0:
        gain -= 10
        t0 = dt.datetime.now(dt.timezone.utc)
        clip = record(raw, freq, gain, a)
    data = out.with_suffix(".sigmf-data")
    raw.rename(data)
    extra = Path(str(raw) + ".metadata.jsonl")
    och_meta = [json.loads(l) for l in extra.read_text().splitlines() if l.strip()] if extra.exists() else []
    if extra.exists():
        extra.rename(out.with_suffix(".och.jsonl"))
    meta = {
        "global": {
            "core:datatype": "ci16_le",
            "core:sample_rate": RATE,
            "core:version": "1.0.0",
            "core:hw": f"OmniCoreSDR (AD9361) RX{a.rx}, gain {gain:g} dB, 12-bit samples << 4 (full scale 32768)",
            "core:recorder": "cuRF-Stream tools/capture_survey.py (och rx)",
            "core:description": note,
            "curf:band": name,
            "curf:gain_db": gain,
            "curf:clipped": clip,
            "curf:rx": a.rx,
            "curf:och": och_meta[:4],
        },
        "captures": [{"core:sample_start": 0, "core:frequency": freq, "core:datetime": t0.isoformat().replace("+00:00", "Z")}],
        "annotations": [],
    }
    out.with_suffix(".sigmf-meta").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return {"band": name, "freq_mhz": freq / 1e6, "gain_db": gain, "clipped": clip, "bytes": data.stat().st_size}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bands", default=",".join(BANDS), help="comma list of: " + ", ".join(BANDS))
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--gain", type=float, default=40)
    ap.add_argument("--rx", type=int, choices=(1, 2), default=2)
    ap.add_argument("--min-free-gb", type=float, default=20, help="refuse to record below this much free disk")
    a = ap.parse_args()
    names = [b.strip() for b in a.bands.split(",") if b.strip()]
    for n in names:
        if n not in BANDS:
            ap.error(f"unknown band {n}")
    a.out.mkdir(parents=True, exist_ok=True)
    need = len(names) * a.seconds * RATE * 4 / 1e9
    free = shutil.disk_usage(a.out).free / 1e9
    if free - need < a.min_free_gb:
        sys.exit(f"{need:.1f} GB to record but only {free:.1f} GB free (keeping {a.min_free_gb:g} GB)")
    for n in names:
        freq, note = BANDS[n]
        print(json.dumps(capture(n, freq, note, a), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
