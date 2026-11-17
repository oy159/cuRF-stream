#!/usr/bin/env python3
"""Record real over-the-air IQ for the dataset: one short capture per band, saved as SigMF.

    python3 tools/capture_survey.py --out ~/dataset/raw/$(date +%Y%m%d_%H%M) [--bands fm,ism433,...] \
        [--seconds 1] [--gain 40] [--rx 2]

Each band becomes <name>.sigmf-data (ci16_le: the 12-bit samples shifted left by 4, full scale = 32768) +
<name>.sigmf-meta, recorded with `och rx` at 61.44 MS/s (the pipeline's rate) on one receiver. A capture with
clipped samples is recorded again 10 dB lower (strong nearby transmitters, e.g. the host's own WiFi). Annotations are left empty: labelling is a
separate step (decoders such as OnAir run on the files as an external tool, plus our own detector).
Bands are chosen for China; 56 MHz instantaneous bandwidth around each centre.

After recording, an occupancy summary of every capture is printed (which signals are on the air, how wide, how
bursty); --survey-only prints it for an existing directory without recording:

    python3 tools/capture_survey.py --out ~/dataset/raw/20261116_2104 --survey-only [--min-snr 8]

Survey: per capture, the peak level (clipping check), noise floor, and the occupied segments: contiguous 60 kHz
bins whose 95th-percentile power over 1 ms frames is min-snr dB above the floor (median over the band), each
with its centre, width, peak SNR and duty cycle (share of frames where its strongest bin is above the threshold).
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


NFFT = 1024  # 60 kHz bins at 61.44 MS/s


def survey(meta_path: Path, min_snr: float, edge_mhz: float):
    meta = json.loads(meta_path.read_text())
    fs = meta["global"]["core:sample_rate"]
    fc = meta["captures"][0]["core:frequency"]
    x = np.fromfile(meta_path.with_suffix(".sigmf-data"), dtype="<i2").astype(np.float32).view(np.complex64) / 32768  # 12-bit << 4
    peak = 20 * np.log10(np.abs(x).max() + 1e-12)
    hop = int(fs * 1e-3)  # one FFT per ms
    nfr = len(x) // hop
    frames = np.stack([x[i * hop: i * hop + NFFT] for i in range(nfr)]) * np.hanning(NFFT)
    P = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)) ** 2 / NFFT + 1e-20)
    f = fc + np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / fs))
    keep = np.abs(f - fc) < (fs / 2 - edge_mhz * 1e6)
    keep &= np.abs(f - fc) > 2 * fs / NFFT  # LO leakage at DC
    p95 = np.percentile(P, 95, axis=0)
    floor = np.median(p95[keep])
    occ = keep & (p95 > floor + min_snr)
    segs, k = [], 0
    while k < NFFT:
        if not occ[k]:
            k += 1
            continue
        e = k
        while e + 1 < NFFT and (occ[e + 1] or (e + 2 < NFFT and occ[e + 2])):  # bridge one-bin gaps
            e += 1
        b = k + np.argmax(p95[k:e + 1])
        duty = float(np.mean(P[:, b] > floor + min_snr))
        segs.append({"mhz": round((f[k] + f[e]) / 2e6, 3), "khz": round((e - k + 1) * fs / NFFT / 1e3),
                     "snr": round(float(p95[b] - floor), 1), "duty": round(duty, 2)})
        k = e + 1
    return meta["global"].get("curf:band", meta_path.stem), fc, peak, floor, segs




def print_survey(d: Path, min_snr: float, edge_mhz: float):
    for m in sorted(d.glob("*.sigmf-meta")):
        band, fc, peak, floor, segs = survey(m, min_snr, edge_mhz)
        segs.sort(key=lambda s: -s["snr"])
        gain = json.loads(m.read_text())["global"].get("curf:gain_db")
        print(f"== {band} {fc / 1e6:.2f} MHz  gain {gain} dB  peak {peak:.1f} dBFS  floor {floor:.1f} dB  segments {len(segs)}")
        for s in segs[:14]:
            print(f"   {s['mhz']:9.3f} MHz  {s['khz']:6d} kHz  SNR {s['snr']:5.1f} dB  duty {s['duty']:.2f}")


def record(raw: Path, freq: float, gain: float, a) -> float:
    """One och capture; returns the share of clipped samples."""
    cmd = [str(OCH), "rx", "--file", str(raw), "--overwrite", "--sample-rate", f"{RATE / 1e6:g}M", "--channels", str(a.rx),
           "--format", "sc16", "--lo-frequency", f"{freq / 1e6:g}M", "--gain", f"{gain:g}", "--seconds", f"{a.seconds:g}",
           # always explicit: och keeps the device's last analogue bandwidth otherwise (it was 18 MHz, which cut
           # everything beyond +-9 MHz of the 61.44 MS/s band and looked like a "hump" around the LO)
           "--bandwidth", "56M"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"och rx failed:\n{r.stdout}\n{r.stderr}")
    x = np.fromfile(raw, dtype="<i2")
    return float(np.mean(np.abs(x) >= 2047 * 16))


def capture(name: str, freq: float, note: str, a) -> dict:
    out = a.out / name
    raw = out.with_suffix(".sc16")
    gain = a.gain
    if a.gains_from:
        gain = json.loads((a.gains_from / f"{name}.sigmf-meta").read_text())["global"]["curf:gain_db"]
    t0 = dt.datetime.now(dt.timezone.utc)
    clip = record(raw, freq, gain, a)
    while clip > 1e-6 and gain - 10 >= 0 and not a.gains_from:
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
            "curf:rf_bandwidth_hz": 56e6,
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
    ap.add_argument("--gains-from", type=Path, help="use each band's gain from these captures (e.g. noise references "
                    "on the terminated RX1 for a set of RX2 captures); no automatic gain reduction")
    ap.add_argument("--survey-only", action="store_true", help="only print the occupancy summary of --out")
    ap.add_argument("--min-snr", type=float, default=8, help="survey: segment threshold over the floor, dB")
    ap.add_argument("--edge-mhz", type=float, default=4, help="survey: ignore this much at each band edge")
    a = ap.parse_args()
    if a.survey_only:
        print_survey(a.out, a.min_snr, a.edge_mhz)
        return
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
    print_survey(a.out, a.min_snr, a.edge_mhz)


if __name__ == "__main__":
    main()
