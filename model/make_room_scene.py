#!/usr/bin/env python3
"""Seeded M4 RF-band scenes. Stream packed12 in bounded chunks; keep independent RX truth.

Sources come only from the clean recordings' validation tail. Dev uses 75..87.5%,
acceptance 87.5..100%. This separates scene tuning; it is not a new modulation dataset.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from curf_model.widesig import ONAIR_TO_FAMILY
from make_scene import pack12, refine_times

FS = 61.44e6
HOP = 2**18
PROFILES = {"ism24": (2440e6, 2418e6, 2462e6), "wlan5": (5795e6, 5773e6, 5817e6),
            "cell900": (939e6, 925e6, 960e6), "cell2100": (2135e6, 2113e6, 2157e6)}
PROTOCOLS = {"wifi": "wifi", "wifi5": "wifi", "wifi6": "wifi", "ble": "ble", "zigbee": "zigbee",
             "cell": "lte", "cell5": "lte", "cell1p4": "lte", "cell20": "lte"}
SOURCE_MODES = (*PROTOCOLS, "fmbc_0", "dvbs8psk")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def epochs(seconds, rng, continuous=False):
    """Carrier lifetimes; the recording still contains its original packet-level bursts."""
    start = float(rng.uniform(.01, .08))
    if continuous:
        end = seconds - float(rng.uniform(.01, .06))
        if rng.random() < .4 and end-start > .4:
            pause = float(rng.uniform(.28, .38))
            return [(start, pause), (pause+float(rng.uniform(.02,.07)), end)]
        return [(start, end)]
    out = []
    while start < seconds-.07:
        end = min(seconds-.01, start+float(rng.uniform(.10,.26)))
        out.append((start,end))
        start = end+float(rng.uniform(.025,.09))
    return out


def map_emissions(emissions, period, offset, begin, end, fs=FS):
    """Crop repeated clean-source emissions to a randomly phased, gated carrier epoch."""
    base = begin-offset
    out = []
    for k in range(-1, (end-begin+offset)//period+2):
        origin = base+k*period
        for e in emissions:
            a = max(begin, origin+e["a"])
            b = min(end, origin+e["b"])
            if b > a:
                out.append({**{q:v for q,v in e.items() if q not in ("a","b")}, "t0": a/fs, "t1": b/fs})
    return out


class Catalog:
    def __init__(self, root, emission_file, split):
        self.root, self.split = Path(root), split
        self.index = {r["id"]:r for r in json.loads((self.root/"index.json").read_text())}
        self.emissions = json.loads(Path(emission_file).read_text())
        self.cache, self.provenance = {}, {}

    def load(self, mode):
        if mode in self.cache:
            return self.cache[mode]
        meta_path = self.root/(mode+".sigmf-meta")
        g = json.loads(meta_path.read_text())["global"]
        if g["core:datatype"] != "cf32_le":
            raise ValueError("M4 requires clean cf32_le sources")
        path = meta_path.with_name(g.get("core:dataset", mode+".cf32"))
        raw = np.memmap(path, dtype="<c8", mode="r")
        fs = float(g["core:sample_rate"])
        lo, hi = (.75,.875) if self.split == "dev" else (.875,1.)
        a, b = int(len(raw)*lo), int(len(raw)*hi)
        source = np.array(raw[a:b], dtype=np.complex64)
        if len(source) < 16:
            raise ValueError(f"{mode}: validation slice too short")
        es = []
        for e in self.emissions[mode]:
            if e["t1"] <= a or e["t0"] >= b:
                continue
            clipped = {**e, "t0": max(a,e["t0"]), "t1": min(b,e["t1"])}
            e = refine_times(raw, clipped)
            if e:
                es.append({"a":round((e["t0"]-a)*FS/fs), "b":round((e["t1"]-a)*FS/fs),
                           "fc":e["fc"], "bw":e["bw"]})
        if not es:
            raise ValueError(f"{mode}: no emissions in {self.split} validation slice")
        low, high = min(e["fc"]-e["bw"]/2 for e in es), max(e["fc"]+e["bw"]/2 for e in es)
        family = ONAIR_TO_FAMILY[self.index[mode]["family"]]
        self.provenance[mode] = {"data_sha256":file_hash(path), "meta_sha256":file_hash(meta_path),
                                 "samples":[a,b], "sample_rate":fs}
        result = source, fs, es, (low+high)/2, high-low, family
        self.cache[mode] = result
        return result


@dataclass
class Transmitter:
    stream: int
    mode: str
    family: str
    protocol: str
    centre: float
    width: float
    snr_db: float
    wave: np.ndarray
    windows: list
    emissions: list
    source_id: str
    expected_named: bool


def generate(catalog, out, seed, profile, seconds=.8, streams=1, kind="positive", snr=(10,30)):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tune, rf_lo, rf_hi = PROFILES[profile]
    n = int(np.ceil(seconds*FS/HOP))*HOP
    seconds = n/FS
    txs, truth, carriers = [], [], []
    for s in range(streams):
        used = []
        if kind == "noise":
            modes = []
        elif kind == "ood":
            modes = ["fmbc_0", "dvbs8psk"]
        elif profile == "ism24":
            modes = [str(rng.choice(["wifi5","wifi6","wifi"])), "ble"]
            if rng.random() < .35:
                modes += ["zigbee"]
        elif profile == "wlan5":
            modes = ["wifi5", "wifi6"] if rng.random() < .5 else [str(rng.choice(["wifi5","wifi6"]))]
        else:
            modes = [str(rng.choice(["cell5","cell1p4","cell","cell20"]))]
        for j, mode in enumerate(modes):
            source, fs, es, centre, width, family = catalog.load(mode)
            # A weak delayed path and small CFO vary without reusing training-window impairments.
            delay = max(1, round(float(rng.uniform(.03e-6,.3e-6))*fs))
            alpha = rng.uniform(0,.3)*np.exp(1j*rng.uniform(-np.pi,np.pi))
            source = source.copy()
            source[delay:] += alpha*source[:-delay].copy()
            frac = Fraction(FS/fs).limit_denominator(2000)
            wave = resample_poly(source, frac.numerator, frac.denominator).astype(np.complex64)
            p = np.abs(wave)**2
            active = p > p.max()*1e-3
            power = float(p[active].mean())
            level = float(rng.uniform(*snr))
            target_power = .003**2/FS*max(e["bw"] for e in es)*10**(level/10)
            wave *= np.float32(np.sqrt(target_power/power))
            # Non-overlapping measured occupied spans; every requested source must be placed.
            margin = 250e3
            low, high = rf_lo+width/2+margin, rf_hi-width/2-margin
            choices = None
            if profile == "ism24" and mode.startswith("wifi"):
                choices = [f*1e6 for f in (2427,2432,2437,2442,2447,2452) if low <= f*1e6 <= high]
            if profile == "wlan5" and mode.startswith("wifi"):
                choices = [f*1e6 for f in (5785,5805) if low <= f*1e6 <= high]
            for attempt in range(300):
                if low >= high:
                    break
                fc = float(rng.choice(choices)) if choices else float(rng.uniform(low,high))
                a, b = fc-width/2-margin, fc+width/2+margin
                if all(b <= u0 or a >= u1 for u0,u1 in used):
                    break
            else:
                raise ValueError(f"{profile}: unable to place {mode}; seed {seed}")
            if low >= high:
                raise ValueError(f"{profile}: source {mode} too wide")
            used.append((a,b))
            cfo = float(rng.uniform(-min(300,width*.002),min(300,width*.002)))
            shift = fc-tune-centre+cfo
            win = epochs(seconds,rng,mode.startswith("cell"))
            windows = [(round(a*FS),round(b*FS),int(rng.integers(0,len(wave)))) for a,b in win]
            source_id = f"{seed}:{s}:{j}"
            transformed = [{**e,"fc":e["fc"]+shift} for e in es]
            named = kind == "positive"
            t = Transmitter(s,mode,family,PROTOCOLS.get(mode,"unknown"),shift,width,level,wave,windows,transformed,source_id,named)
            txs.append(t)
            for a,b,offset in windows:
                for e in map_emissions(transformed,len(wave),offset,a,b):
                    truth.append({**e,"scene":0,"stream":s,"mode":mode,"family":family,"protocol":t.protocol,
                                  "snr_db":level,"source_id":source_id,"expected_named":named,"carrier_fc":fc-tune+cfo,"carrier_bw":width})
            carriers.append({"source_id":source_id,"stream":s,"mode":mode,"family":family,"protocol":t.protocol,
                             "snr_db":level,"expected_named":named,"fc":fc-tune+cfo,"bw":width,"epochs":win})
    # Chunked generation bounds memory even for independent dual-RX captures.
    peak, clipped, count = 0., 0, 0
    imbalance = [(float(rng.uniform(-.025,.025)),float(rng.uniform(-.02,.02))) for _ in range(streams)]
    dc = [complex(rng.normal(),rng.normal())*.0001 for _ in range(streams)]
    raw_hash=hashlib.sha256()
    with (out/"scene_0.p12").open("wb") as f:
        for begin in range(0,n,2**18):
            end = min(n,begin+2**18)
            chunk = (rng.normal(size=(end-begin,streams))+1j*rng.normal(size=(end-begin,streams))).astype(np.complex64)*np.float32(.003/np.sqrt(2))
            for t in txs:
                for a,b,offset in t.windows:
                    lo,hi=max(begin,a),min(end,b)
                    if hi <= lo:
                        continue
                    k=np.arange(lo,hi,dtype=np.int64)
                    phase=2*np.pi*np.remainder(t.centre*k/FS,1.)
                    chunk[lo-begin:hi-begin,t.stream] += t.wave[(k-a+offset)%len(t.wave)]*np.exp(1j*phase).astype(np.complex64)
            for s,(gain,phase) in enumerate(imbalance):
                v=chunk[:,s].copy()
                chunk[:,s] = v + (gain+1j*phase)*v.conj()+dc[s]
            peak=max(peak,float(np.abs(chunk).max()))
            clipped+=int(np.count_nonzero((np.abs(chunk.real)>=1)|(np.abs(chunk.imag)>=1)))
            count+=chunk.size
            wire=pack12(chunk.reshape(-1))
            raw_hash.update(wire)
            wire.tofile(f)
    payload={"version":2,"scenes":1,"secs":seconds,"rate":FS,"freq":tune,"streams":streams,"seed":seed,
             "split":catalog.split,"profile":profile,"kind":kind,"nominal_snr_db":list(snr),"truth":truth,"carriers":carriers,
             "generation":{"noise_rms":.003,"multipath_delay_s":[.03e-6,.3e-6],"iq_imbalance":imbalance,"dc":[[v.real,v.imag] for v in dc],
                           "peak":peak,"clipped_fraction":clipped/max(count,1),"raw_sha256":raw_hash.hexdigest()},"sources":{t.mode:catalog.provenance[t.mode] for t in txs}}
    (out/"truth.json").write_text(json.dumps(payload,indent=2))
    return payload


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src",type=Path,required=True)
    ap.add_argument("--emissions",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--split",choices=("dev","acceptance"),required=True)
    ap.add_argument("--profile",choices=PROFILES,required=True)
    ap.add_argument("--kind",choices=("positive","noise","ood"),default="positive")
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--seconds",type=float,default=.8)
    ap.add_argument("--streams",type=int,choices=(1,2),default=1)
    ap.add_argument("--snr",type=float,nargs=2,default=(10,30))
    a=ap.parse_args()
    if not .4 <= a.seconds <= 10 or not -5 <= a.snr[0] <= a.snr[1] <= 35:
        ap.error("require duration 0.4..10 and SNR -5..35 dB")
    result=generate(Catalog(a.src,a.emissions,a.split),a.out,a.seed,a.profile,a.seconds,a.streams,a.kind,a.snr)
    print(json.dumps({k:result[k] for k in ("seed","profile","kind","streams","generation")}))


if __name__ == "__main__":
    main()
