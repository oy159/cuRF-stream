#!/usr/bin/env python3
"""Clean protocol signals for the dataset from OnAir's built-in test-signal generators (external tool).

    python3 tools/onair_gen.py --out ~/dataset/onair [--ids wifi,ble,...] [--secs 2] [--snr 60]

OnAir (https://github.com/adswill/OnAir, GPL-3.0-or-later) is run as a separate program, never linked or copied:
`onair-cli --synthetic --standard <id> --snr S --record <id>.cf32 --secs N` records the mode's own test signal
at the mode's own sample rate as SigMF (cf32). It runs through `onair-offline` (an LD_PRELOAD shim that allows
loopback only), so nothing leaves the machine. The recordings are generator output, not OnAir code; our own
pipeline (tools/make_dataset.py) resamples them, places them in the wideband scene and adds channel and
receiver impairments.

Each id is labelled with a modulation family (FAMILY below) and checked: 99 % occupied bandwidth (Welch over
the whole file) and duty cycle (share of 100 us frames more than 10 dB over the quietest decile).
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

OFFLINE = Path.home() / "opt/onair-4.2.0/onair-offline"

# OnAir mode id -> (modulation family, what it is). An id OnAir does not know falls back silently to a default
# 9.143 MS/s test signal (seen for cell_lte, cdma, chv, is95, pmr): check the index for that rate / bandwidth.
FAMILY = {
    "wifi": ("dsss", "802.11b DSSS beacons, 1 Mbit/s DBPSK Barker-11 (OnAir's 2.4 GHz default; not OFDM)"),
    "cell": ("ofdm", "LTE downlink 10 MHz"),
    "nr": ("ofdm", "5G NR downlink"),
    "nbiot": ("ofdm", "NB-IoT 180 kHz"),
    "dab": ("ofdm", "DAB 1.5 MHz"),
    "dtmb": ("ofdm", "DTMB 8 MHz (China DTV, multi-carrier mode)"),
    "halow": ("ofdm", "802.11ah"),
    "gsmr": ("gmsk", "GSM(-R) 270.833 kbit/s"),
    "ais": ("gmsk", "AIS 9.6 kbit/s"),
    "ble": ("gfsk", "Bluetooth LE 1 Mbit/s"),
    "btc": ("gfsk", "Bluetooth BR/EDR"),
    "nrf24": ("gfsk", "nRF24L01"),
    "zwave": ("gfsk", "Z-Wave"),
    "ant": ("gfsk", "ANT+"),
    "dmr": ("4fsk", "DMR 12.5 kHz"),
    "dpmr": ("4fsk", "dPMR 6.25 kHz"),
    "nxdn": ("4fsk", "NXDN"),
    "p25": ("4fsk", "P25 phase 1 C4FM"),
    "m17": ("4fsk", "M17"),
    "tetra": ("pi4dqpsk", "TETRA 25 kHz"),
    "vdl2": ("d8psk", "VDL mode 2"),
    "zigbee": ("oqpsk", "802.15.4 O-QPSK DSSS 2 Mchip/s"),
    "umts": ("dsss", "UMTS W-CDMA 3.84 Mchip/s"),
    "cdma2000": ("dsss", "CDMA2000 1.2288 Mchip/s"),
    "loraraw": ("chirp", "LoRa"),
    "lorawan": ("chirp", "LoRaWAN"),
    "sensors": ("ook", "433 MHz sensors (rtl_433 style)"),
    "ism": ("ook", "ISM remotes / sensors"),
    "airband": ("am", "airband AM voice"),
    "fm": ("fm", "broadcast FM stereo + RDS"),
    "frs": ("fm", "narrow-band FM voice (FRS)"),
    "adsb": ("pulse", "Mode S / ADS-B 1090"),
    "uat": ("cpfsk", "UAT 978"),
    "modeac": ("pulse", "SSR Mode A/C replies"),
    "dvbc": ("qam", "DVB-C single-carrier QAM"),
    "dvbs": ("psk", "DVB-S QPSK"),
    "iridium": ("psk", "Iridium DE-QPSK bursts"),
    "inmc": ("psk", "Inmarsat-C BPSK"),
    "orbcomm": ("psk", "Orbcomm SDPSK"),
    # outlier exposure: modulations outside every family above (labelled "unknown" in training)
    "atv": ("unknown", "analogue TV, VSB AM video"),
    "horus": ("unknown", "Horus MFSK telemetry"),
    "js8": ("unknown", "JS8 8-GFSK"),
    "fsq": ("unknown", "FSQ incremental MFSK"),
    "ifkp": ("unknown", "IFKP incremental MFSK"),
    "hfdig": ("unknown", "FT8 8-GFSK"),
    "rpr": ("unknown", "Robust Packet Radio"),
    "tones": ("unknown", "selective-call / CTCSS tones"),
    "pager": ("fsk", "POCSAG / FLEX pagers"),
    "acars": ("am_msk", "ACARS AM-MSK 2400 bit/s"),
}


# Variants: one OnAir mode with generator options (SynthConfig::modeOpt / modeVal, see its <mode>_gen.h, or the
# mode's own CLI flags) -> name: (OnAir id, extra arguments, family, what)
VARIANTS = {
    "wifi5": ("wifi", ["--sopt", "0", "1"], "ofdm", "802.11a non-HT OFDM 6 Mbit/s, 5 GHz beacons"),
    "wifi6": ("wifi", ["--sopt", "0", "2"], "ofdm", "802.11ax-band non-HT OFDM, 6 GHz discovery frames"),
    "cell5": ("cell", ["--sopt", "0", "1"], "ofdm", "LTE downlink 5 MHz"),
    "cell1p4": ("cell", ["--sopt", "0", "2"], "ofdm", "LTE downlink 1.4 MHz"),
    "cell20": ("cell", ["--sopt", "0", "3"], "ofdm", "LTE downlink 20 MHz"),
    "dtmb16": ("dtmb", ["--sopt", "1", "2"], "ofdm", "DTMB 16QAM"),
    "nr28": ("nr", ["--sopt", "0", "1"], "ofdm", "5G NR n28 cell, 15 kHz SCS"),
    "gsmr1": ("gsmr", ["--sopt", "0", "1"], "gmsk", "GSM one cell"),
    "gsmr3": ("gsmr", ["--sopt", "0", "2"], "gmsk", "GSM three cells, one 12 dB weaker"),
    "dvbs1": ("dvbs", ["--sopt", "0", "1", "--sval", "0", "1e6"], "psk", "DVB-S QPSK 1 Msym/s"),
    "dvbs8psk": ("dvbs", ["--sopt", "1", "1", "--sval", "0", "2e6"], "psk", "DVB-S2 8PSK 2 Msym/s"),
    "dvbs16apsk": ("dvbs", ["--sopt", "1", "2", "--sval", "0", "3e6"], "qam", "DVB-S2 16APSK 3 Msym/s"),
    "dvbs32apsk": ("dvbs", ["--sopt", "1", "3", "--sopt", "2", "9"], "qam", "DVB-S2 32APSK 5/6, 5 Msym/s"),
    # (--dvbc-qam / --dvbc-sr set the DVB-C receiver only: the test signal stays the default one)
}


def measure(stem: Path):
    g = json.loads(stem.with_suffix(".sigmf-meta").read_text())["global"]
    fs = g["core:sample_rate"]
    x = np.fromfile(stem.with_suffix(".cf32"), dtype="<c8")
    if len(x) < 4096:
        return fs, len(x) / fs, None, None, None
    n = 4096
    seg = x[: len(x) // n * n].reshape(-1, n) * np.hanning(n)
    psd = np.fft.fftshift((np.abs(np.fft.fft(seg, axis=1)) ** 2).mean(0))
    c = np.cumsum(psd) / psd.sum()
    lo, hi = np.searchsorted(c, 0.005), np.searchsorted(c, 0.995)
    obw = (hi - lo + 1) * fs / n
    hop = max(1, int(fs * 100e-6))
    p = (np.abs(x[: len(x) // hop * hop]) ** 2).reshape(-1, hop).mean(1)
    quiet = np.percentile(p, 10) + 1e-20
    duty = float(np.mean(p > quiet * 10))
    if p.max() <= 0:
        return fs, len(x) / fs, None, None, None  # silent
    return fs, len(x) / fs, obw, duty, 10 * np.log10(p.max() / quiet)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ids", default=",".join(list(FAMILY) + list(VARIANTS)))
    ap.add_argument("--merge", action="store_true", help="update the rows of these ids in an existing index.json")
    ap.add_argument("--secs", type=float, default=2)
    ap.add_argument("--snr", type=float, default=60, help="OnAir's synthetic noise level (high: clean signal)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mid in [i.strip() for i in a.ids.split(",") if i.strip()]:
        stem = a.out / mid
        for f in (stem.with_suffix(".cf32"), stem.with_suffix(".sigmf-meta"), stem.with_suffix(".sigmf-data")):
            f.unlink(missing_ok=True)
        oid, extra, fam, what = VARIANTS.get(mid, (mid, [], *FAMILY.get(mid, ("unknown", ""))))
        r = subprocess.run([str(OFFLINE), "onair-cli", "--synthetic", "--standard", oid, *extra, "--snr", f"{a.snr:g}",
                            "--secs", f"{a.secs:g}", "--record", str(stem.with_suffix(".cf32"))],
                           capture_output=True, text=True, timeout=120 + 10 * a.secs)
        net = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("[nonet]")]
        if not stem.with_suffix(".sigmf-meta").exists():
            print(json.dumps({"id": mid, "error": "no recording"}))
            continue
        fs, dur, obw, duty, pdr = measure(stem)
        row = {"id": mid, "onair_id": oid, "args": extra, "family": fam, "what": what, "fs_mhz": float(fs) / 1e6, "secs": round(dur, 2),
               "obw_khz": None if obw is None else round(float(obw) / 1e3, 1), "duty": None if duty is None else round(float(duty), 3),
               "peak_to_quiet_db": None if pdr is None else round(float(pdr), 1), "network_blocked": len(net)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    idx = a.out / "index.json"
    if a.merge and idx.exists():
        ids = {r["id"] for r in rows}
        rows = [r for r in json.loads(idx.read_text()) if r["id"] not in ids] + rows
    idx.write_text(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
