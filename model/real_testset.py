#!/usr/bin/env python3
"""Real over-the-air test set for the stage-2 classifier (RX2 captures from tools/capture_survey.py).

    python model/real_testset.py scan  --raw ~/dataset/raw/20261116_2104 --out ~/dataset/real_v1
    python model/real_testset.py label --out ~/dataset/real_v1          (candidate labels -> labels.json)
    python model/real_testset.py cut   --out ~/dataset/real_v1          (per-emission SigMF for OnAir decoders)
    python model/real_testset.py build --out ~/dataset/real_v1          (windows of the labelled emissions)
    python model/real_testset.py eval  --out ~/dataset/real_v1 --ckpt model/artifacts/s2_run2/best.pt

scan: emissions of each 61.44 MS/s capture in two passes, after whitening by the receiver's noise spectrum (a
capture of the same band and gain with no input: the AD9361 floor is far from flat, up to ~14 dB of hump around
the LO at high gain, which a single threshold takes for 16 MHz "signals"). The wide pass (30 kHz bins) finds everything; each
emission narrower than 400 kHz is then mixed to 0 Hz, decimated to 960 kS/s and scanned again (234 Hz bins),
which separates and measures the narrow-band transmitters (AM / FM voice, DMR, GSM carriers). The DC channel
(LO leakage) and |f| > 20 MHz (analogue filter edge, raised quantisation noise) are left out. Each emission gets
a candidate label from the band plan and its measurements (labels.json, editable), with a confidence and the
reason; only "high" labels, or labels confirmed by a decoder, count in the evaluation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

import re
import subprocess
from collections import Counter
from fractions import Fraction

from curf_model.widesig import FAMILIES, Emission, emissions, extract, load_sigmf, noise_spectrum

FS = 61.44e6
NARROW_FS = 960e3  # 61.44 MS/s / 64
GRID = 240e3       # narrow-pass stream spacing


def narrow_stream(x: np.ndarray, fc: float) -> np.ndarray:
    n = np.arange(len(x))
    y = x * np.exp(-2j * np.pi * fc / FS * n).astype(np.complex64)
    return resample_poly(y, 1, 64).astype(np.complex64)


def mean_spectrum(x: np.ndarray, nfft: int) -> np.ndarray:
    n = len(x) // nfft
    acc = np.zeros(nfft, np.float64)
    win = np.hanning(nfft).astype(np.float32)
    for i in range(0, n, 4096):
        seg = x[i * nfft: min(n, i + 4096) * nfft].reshape(-1, nfft) * win
        acc += (np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2).sum(0)
    return (acc / n).astype(np.float32)


def stationary(psd: np.ndarray, fs: float, source: str, n: int, thresh_db: float = 4.0, min_bins: int = 2) -> list[Emission]:
    """Carriers present over the whole capture, from its mean whitened spectrum (averaging over ~30k frames makes
    a few dB significant). One emission per contiguous run of bins (gaps of up to 2 bins bridged), spanning the
    whole capture: FM wanders with its audio and LTE / NR leave subframes nearly empty, so the time-frequency
    pass cuts them into pieces that each show only part of the signal."""
    nfft = len(psd)
    floor = np.median(psd)
    on = psd > floor * 10 ** (thresh_db / 10)
    on = np.convolve(on, np.ones(5), "same") > 0  # bridge 2-bin gaps
    on &= np.convolve(psd > floor * 10 ** (thresh_db / 10), np.ones(5), "same") > 0
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    out, k = [], 0
    while k < nfft:
        if not on[k]:
            k += 1
            continue
        e = k
        while e + 1 < nfft and on[e + 1]:
            e += 1
        if e - k + 1 >= min_bins:
            p = psd[k:e + 1]
            c = np.cumsum(p) / p.sum()
            lo, hi = k + np.searchsorted(c, 0.005), k + np.searchsorted(c, 0.995)
            out.append(Emission(source, float((f[lo] + f[hi]) / 2), float((hi - lo + 1) * fs / nfft), 0, n,
                                float(10 * np.log10(p.max() / floor))))
        k = e + 1
    return out


def covered(e: Emission, st: list[Emission]) -> bool:
    """Most of e's band inside a stationary carrier's band."""
    for s_ in st:
        lo, hi = max(e.fc - e.bw / 2, s_.fc - s_.bw / 2), min(e.fc + e.bw / 2, s_.fc + s_.bw / 2)
        if hi - lo > 0.5 * e.bw:
            return True
    return False


def scan_capture(meta: Path, noise_dir: Path | None) -> list[dict]:
    x, fs = load_sigmf(meta)
    m = json.loads(meta.read_text())
    band, f0 = m["global"].get("curf:band", meta.stem), m["captures"][0]["core:frequency"]
    psd = None
    if noise_dir and (noise_dir / meta.name).exists():  # same band and gain, terminated input
        psd = noise_spectrum(load_sigmf(noise_dir / meta.name)[0], 2048)
    wide = emissions(x, fs, band, nfft=2048, thresh_db=10, range_db=80, rel_db=70, floor_pct=40, noise_psd=psd,
                     merge_gap_s=12e-3, absorb_narrow=False)  # VHF: wideband impulses overlap FM / AM carriers
    mpsd = mean_spectrum(x, 2048)
    st = stationary(mpsd / psd if psd is not None else mpsd, fs, band, len(x))
    out = []
    for e in st:
        if abs(e.fc) > 20e6 or (abs(e.fc) < 60e3 and e.bw < 200e3):
            continue
        if e.bw >= 400e3:
            out.append({**asdict(e), "f_mhz": (f0 + e.fc) / 1e6, "pass": "steady", "fs": fs})
    wide = [e for e in wide if not covered(e, st)]
    keys = set()
    for e in st:
        if e.bw < 400e3 and abs(e.fc) <= 20e6 and not (abs(e.fc) < 60e3 and e.bw < 200e3):
            keys.add(round(e.fc / GRID) * GRID)
    for e in wide:
        if abs(e.fc) > 20e6 or (abs(e.fc) < 60e3 and e.bw < 200e3):
            continue
        if e.bw >= 400e3:
            out.append({**asdict(e), "f_mhz": (f0 + e.fc) / 1e6, "pass": "wide", "fs": fs})
        else:
            keys.add(round(e.fc / GRID) * GRID)
    # narrow pass: one 960 kS/s stream per 240 kHz grid point that has narrow emissions; each stream keeps what
    # lies within half a grid step of its centre, so every frequency is scanned once
    for key in sorted(keys):
        y = narrow_stream(x, key)
        nst = [n for n in stationary(mean_spectrum(y, 4096), NARROW_FS, band, len(y)) if n.bw <= 300e3]
        for n in nst:
            if abs(n.fc) > GRID / 2 or abs(key + n.fc) < 30e3:
                continue
            out.append({**asdict(n), "fc": key + n.fc, "f_mhz": (f0 + key + n.fc) / 1e6, "pass": "steady",
                        "fs": NARROW_FS, "wide_fc": key})
        for n in emissions(y, NARROW_FS, band, nfft=4096, thresh_db=12, range_db=80, rel_db=50, floor_pct=40,
                           smooth_t=3, smooth_f=5, merge_gap_s=20e-3, absorb_narrow=False):
            if covered(n, nst):
                continue
            if abs(n.fc) > GRID / 2 or n.bw > 300e3 or abs(key + n.fc) < 30e3:
                continue
            out.append({**asdict(n), "fc": key + n.fc, "f_mhz": (f0 + key + n.fc) / 1e6, "pass": "narrow",
                        "fs": NARROW_FS, "wide_fc": key})
    dur = len(x) / fs
    for o in out:
        o["band"] = band
        o["duty"] = round((o["t1"] - o["t0"]) / o["fs"] / dur, 3)
    return out


def wifi_shape_db(x: np.ndarray, e: dict) -> float:
    """Inner (|f| < 4 MHz) over shoulder (5-8 MHz) power of a 2.4 GHz WiFi burst: ~0 dB for OFDM (flat top, steep
    edges); 802.11b DSSS peaks in the middle: +5 dB for a bare sinc^2 of 11 Mchip/s, measured +8..+13 dB with the
    transmit filters of real access points."""
    a, b = max(e["t0"], 0), min(e["t1"], len(x))
    seg = x[a:b]
    n = 1024
    k = max(1, len(seg) // n)
    seg = seg[: k * n].reshape(k, n) * np.exp(-2j * np.pi * e["fc"] / FS * (np.arange(k * n).reshape(k, n) + a))
    P = (np.abs(np.fft.fftshift(np.fft.fft(seg * np.hanning(n), axis=1), axes=1)) ** 2).mean(0)
    f = np.abs(np.fft.fftshift(np.fft.fftfreq(n, 1 / FS)))
    return float(10 * np.log10(P[f < 4e6].mean() / P[(f > 5e6) & (f < 8e6)].mean()))


def rule_label(e: dict) -> tuple[str | None, str, str]:
    """Candidate class, confidence (high / check / low) and the reason, from the band plan (China) and measurements."""
    f, bw, band = e["f_mhz"], e["bw"], e["band"]
    if 87.5 <= f <= 108 and 80e3 <= bw <= 300e3:
        return "fm", "high", "FM broadcast band, 80-300 kHz wide"
    if 118 <= f <= 137 and 4e3 <= bw <= 25e3 and e["duty"] < 0.9:
        return "am", "high", "airband (AM voice / ACARS): 4-25 kHz, intermittent"
    if 118 <= f <= 137 and bw < 4e3 and e["duty"] >= 0.9:
        return None, "low", "steady narrow carrier in the airband (spur / unmodulated): not AM voice"
    if 108 <= f < 118 and bw <= 50e3:
        return "am", "low", "VOR / ILS band"
    if 1805 <= f <= 1880 and bw >= 1.2e6:
        return "ofdm", "high", "LTE/NR band 3 downlink (no UMTS there in China), >= 1.2 MHz"
    if 3300 <= f <= 3800 and bw >= 5e6:
        return "ofdm", "high", "NR n78, >= 5 MHz"
    if (925 <= f <= 960 or 2110 <= f <= 2170) and bw >= 1.2e6:
        return "ofdm", "check", "band 8 / band 1 downlink: LTE/NR or UMTS, decoder decides"
    if (930 <= f <= 960 or 1805 <= f <= 1830) and 120e3 <= bw <= 320e3:
        return "fsk", "low", "GSM (GMSK) carrier width, but NB-IoT (OFDM) fits too"
    if 5150 <= f <= 5850 and bw >= 10e6:
        return "ofdm", "high", "5 GHz WiFi (OFDM only: 802.11a/n/ac/ax), >= 10 MHz"
    if 2400 <= f <= 2483.5 and bw >= 10e6:
        return "wifi?", "check", "2.4 GHz WiFi burst: DSSS or OFDM by spectral shape"
    if 2400 <= f <= 2483.5 and 0.6e6 <= bw <= 2.5e6 and e["duty"] < 0.05:
        return "fsk", "low", "1-2 MHz short bursts at 2.4 GHz: BLE / Bluetooth / nRF (GFSK), or Zigbee (O-QPSK)"
    return None, "low", "no rule"


OFFLINE = Path.home() / "opt/onair-4.2.0/onair-offline"
# OnAir decoders that confirm a class: (mode, sample rate, class, pattern whose first group > 0 means decoded)
DECODERS = [("cell", 20e6, "ofdm", r"(\d+) identified"), ("umts", 7.68e6, "dsss", r"UMTS: (\d+) cells")]


def cut(x: np.ndarray, fc: float, rate: float) -> np.ndarray:
    fr = Fraction(rate / FS).limit_denominator(1000)
    y = x * np.exp(-2j * np.pi * fc / FS * np.arange(len(x))).astype(np.complex64)
    return resample_poly(y, fr.numerator, fr.denominator).astype(np.complex64)


def write_sigmf(stem: Path, y: np.ndarray, rate: float, freq: float):
    y.astype("<c8").tofile(stem.with_suffix(".sigmf-data"))
    stem.with_suffix(".sigmf-meta").write_text(json.dumps({
        "global": {"core:datatype": "cf32_le", "core:sample_rate": rate, "core:version": "1.0.0"},
        "captures": [{"core:sample_start": 0, "core:frequency": freq}], "annotations": []}))


def decode(stem: Path, mode: str, pattern: str) -> int:
    """Runs one OnAir decoder (offline) on a cut; returns the count it reports. Only that number is kept: the
    decoders also print network names and addresses, which are not stored."""
    r = subprocess.run([str(OFFLINE), "onair-cli", "--file", str(stem.with_suffix(".sigmf-meta")), "--standard", mode,
                        "--secs", "2"], capture_output=True, text=True, timeout=300)
    counts = [int(m.group(1)) for m in re.finditer(pattern, r.stdout + r.stderr)]
    return max(counts) if counts else 0


def windows_of(x: np.ndarray, fs: float, e: dict, L: int, n: int, rng) -> tuple[list[np.ndarray], float]:
    """Windows of one real emission, extracted as the runtime will (measured bandwidth, no added impairments)."""
    em = Emission(e["source"], e["fc"], e["bw"], e["t0"], e["t1"], e["snr_db"])
    out = []
    for _ in range(n):
        y, b = extract(x, fs, em, L, rng, bw_err=(1.0, 1.0), fc_err=0.0)
        out.append((y / np.sqrt(np.mean(np.abs(y) ** 2) + 1e-20)).astype(np.complex64))
    return out, e["bw"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["scan", "label", "cut", "build", "eval"])
    ap.add_argument("--raw", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--noise", type=Path, help="noise references (same bands and gains, no signal): whiten first")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.step == "scan":
        allem = {}
        for meta in sorted(a.raw.glob("*.sigmf-meta")):
            em = scan_capture(meta, a.noise)
            allem[meta.stem] = em
            print(f"== {meta.stem}: {len(em)} emissions")
            for e in sorted(em, key=lambda e: e["f_mhz"]):
                print(f"   {e['f_mhz']:10.4f} MHz  {e['bw'] / 1e3:9.1f} kHz  SNR {e['snr_db']:5.1f} dB  span {e['duty']:.2f}  {e['pass']}")
        (a.out / "emissions.json").write_text(json.dumps({"raw": str(a.raw), "captures": allem}, indent=1))
    if a.step == "label":
        d = json.loads((a.out / "emissions.json").read_text())
        a.raw, allem = Path(d["raw"]), d["captures"]
        # candidate labels (labels.json is meant to be reviewed / edited by hand before build)
        labels = []
        for cap, em in allem.items():
            x = None
            for i, e in enumerate(em):
                cls, conf, why = rule_label(e)
                q = 100 if "wide_fc" in e else 4  # transmitter: 10 kHz (narrow streams) / 250 kHz (wide) grid
                row = {"capture": cap, "i": i, "group": f"{cap}@{round(e['f_mhz'] * q) / q:g}", "f_mhz": round(e["f_mhz"], 4), "bw_khz": round(e["bw"] / 1e3, 1),
                       "snr_db": round(e["snr_db"], 1), "span": e["duty"], "class": cls, "conf": conf, "why": why}
                if cls == "wifi?":
                    if x is None:
                        x, _ = load_sigmf(a.raw / f"{cap}.sigmf-meta")
                    r = wifi_shape_db(x, e)
                    row.update(shape_db=round(r, 1))
                    row["class"], row["conf"] = (("ofdm", "high") if abs(r) < 2.0 else ("dsss", "high") if r > 6.0
                                                 else (None, "low"))
                    row["why"] += f": inner/shoulder {r:+.1f} dB"
                labels.append(row)
        (a.out / "labels.json").write_text(json.dumps(labels, indent=1, ensure_ascii=False))
        print("candidate labels:", Counter((r["class"], r["conf"]) for r in labels).most_common())

    if a.step == "cut":  # confirm the "check" labels with OnAir decoders
        d = json.loads((a.out / "emissions.json").read_text())
        labels = json.loads((a.out / "labels.json").read_text())
        cuts = a.out / "cuts"
        cuts.mkdir(exist_ok=True)
        for cap in sorted({r["capture"] for r in labels if r["conf"] == "check"}):
            x, _ = load_sigmf(Path(d["raw"]) / f"{cap}.sigmf-meta")
            f0 = json.loads((Path(d["raw"]) / f"{cap}.sigmf-meta").read_text())["captures"][0]["core:frequency"]
            groups = {}
            for r in labels:
                if r["capture"] == cap and r["conf"] == "check":
                    groups.setdefault(r["group"], []).append(r)
            for g, rs in groups.items():
                e = max((d["captures"][cap][r["i"]] for r in rs), key=lambda e: e["t1"] - e["t0"])
                found = {}
                for mode, rate, cls, pat in DECODERS:
                    stem = cuts / f"{g.replace('@', '_')}_{mode}"
                    write_sigmf(stem, cut(x, e["fc"], rate), rate, f0 + e["fc"])
                    found[mode] = decode(stem, mode, pat)
                    stem.with_suffix(".sigmf-data").unlink()
                hits = [cls for mode, _, cls, _ in DECODERS if found[mode] > 0]
                for r in rs:
                    r["decoded"] = found
                    r["class"], r["conf"] = (hits[0], "confirmed") if len(set(hits)) == 1 else (r["class"], "low")
                print(f"  {g} ({len(rs)} emissions, {e['bw'] / 1e3:.0f} kHz): {found} -> {rs[0]['class']} ({rs[0]['conf']})",
                      flush=True)
        (a.out / "labels.json").write_text(json.dumps(labels, indent=1, ensure_ascii=False))

    if a.step == "build":
        d = json.loads((a.out / "emissions.json").read_text())
        labels = json.loads((a.out / "labels.json").read_text())
        rng = np.random.default_rng(0)
        L, per_group = 1024, 48
        gsize = Counter(r["group"] for r in labels)
        X, Y, B, E, C = [], [], [], [], []
        for cap, em in d["captures"].items():
            x, fs = load_sigmf(Path(d["raw"]) / f"{cap}.sigmf-meta")
            rows = {r["i"]: r for r in labels if r["capture"] == cap}
            narrow_cache = {}
            # narrow emissions grouped by their 960 kS/s stream, so each stream is made once
            order = sorted(range(len(em)), key=lambda i: ("wide_fc" in em[i], em[i].get("wide_fc", 0.0)))
            for i in order:
                e = em[i]
                r = rows[i]
                y_cls = FAMILIES.index(r["class"]) if r["class"] in FAMILIES and r["conf"] in ("high", "confirmed") else -1
                if "wide_fc" in e:  # found in a 960 kS/s stream
                    key = e["wide_fc"]
                    if key not in narrow_cache:
                        narrow_cache = {key: narrow_stream(x, key)}
                    src, sfs, ee = narrow_cache[key], NARROW_FS, {**e, "fc": e["fc"] - key}
                else:
                    src, sfs, ee = x, fs, e
                budget = per_group if y_cls >= 0 else 8  # unlabelled transmitters: only predictions to look at
                n = max(1, round(budget / gsize[r["group"]]))  # every transmitter about the same weight
                ws, bw = windows_of(src, sfs, ee, L, n, rng)
                X += ws; Y += [y_cls] * len(ws); B += [np.log10(bw)] * len(ws); E += [r["group"]] * len(ws); C += [cap] * len(ws)
            # noise: random places with no emission (in frequency, over the whole capture)
            occ = [(e["fc"] - e["bw"], e["fc"] + e["bw"]) for e in em]
            got = tries = 0
            while got < 48 and tries < 5000:  # VHF captures can be nearly covered by wideband impulse interference
                tries += 1
                bw = float(10 ** rng.uniform(4.5, 6.7))
                fc = float(rng.uniform(-19e6 + bw, 19e6 - bw))
                if abs(fc) < bw or any(lo < fc + bw and fc - bw < hi for lo, hi in occ):
                    continue
                ee = {"source": cap, "fc": fc, "bw": bw, "t0": 0, "t1": len(x), "snr_db": 0.0}
                ws, _ = windows_of(x, fs, ee, L, 1, rng)
                X += ws; Y.append(0); B.append(np.log10(bw)); E.append(f"{cap}/noise"); C.append(cap)
                got += 1
            print(f"  {cap}: {len(em)} emissions, {got} noise windows", flush=True)
        xs = np.stack([np.stack([w.real, w.imag]) for w in X]).astype(np.float16)
        np.savez(a.out / "real.npz", x=xs, y=np.array(Y, np.int16), log_bw=np.array(B, np.float32),
                 emission=np.array(E), capture=np.array(C))
        y = np.array(Y)
        print(f"real windows: {len(y)}, labelled {int((y >= 0).sum())}: " +
              ", ".join(f"{FAMILIES[k]} {v}" for k, v in sorted(Counter(y[y >= 0].tolist()).items())))

    if a.step == "eval":
        import torch
        from curf_model.net import SignalResNet1D
        z = np.load(a.out / "real.npz")
        ck = torch.load(a.ckpt, map_location="cuda")
        model = SignalResNet1D(len(ck["classes"]), ck["width"]).cuda().eval()
        model.load_state_dict(ck["model"])
        x = torch.from_numpy(z["x"]).cuda().float()
        bw = torch.from_numpy(z["log_bw"]).cuda()
        with torch.no_grad():
            inp = torch.cat([x, ((bw - 5) / 2)[:, None, None].expand(-1, 1, x.shape[-1])], 1)
            logits = torch.cat([model(inp[i:i + 1024]) for i in range(0, len(inp), 1024)])
        pred = logits.argmax(-1).cpu().numpy()
        y, em, cap = z["y"], z["emission"], z["capture"]
        lab = y >= 0
        print(f"labelled windows {lab.sum()}: accuracy {np.mean(pred[lab] == y[lab]):.3f}")
        for k in sorted(set(y[lab].tolist())):
            m = y == k
            print(f"   {FAMILIES[k]:8s} n={m.sum():5d}  acc {np.mean(pred[m] == k):.3f}  predicted as " +
                  ", ".join(f"{FAMILIES[c]} {v}" for c, v in Counter(pred[m].tolist()).most_common(4)))
        # per emission: majority vote over its windows
        votes = {}
        for e in sorted(set(em[lab].tolist())):
            m = (em == e) & lab  # a transmitter's group can also hold unlabelled pieces
            votes[e] = (int(y[m][0]), Counter(pred[m].tolist()).most_common(1)[0][0])
        ok = [t == p for t, p in votes.values()]
        print(f"labelled transmitters {len(ok)}: majority-vote accuracy {np.mean(ok):.3f}")
        for e, (t, p) in sorted(votes.items()):
            if e.endswith("/noise"):
                continue
            print(f"   {e:22s} label {FAMILIES[t]:6s} vote {FAMILIES[p]:7s} {'ok' if t == p else 'WRONG'}")
        print("unlabelled emissions, predicted class by capture:")
        for c in sorted(set(cap.tolist())):
            m = (cap == c) & ~lab
            if m.any():
                print(f"   {c:9s} " + ", ".join(f"{FAMILIES[k]} {v}" for k, v in Counter(pred[m].tolist()).most_common(6)))


if __name__ == "__main__":
    main()
