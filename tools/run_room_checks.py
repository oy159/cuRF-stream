#!/usr/bin/env python3
"""Generate and replay the M4 suite, freezing policy and provenance before evaluation.

Run dev first; tune only on dev raw feeds. Acceptance uses disjoint scene seeds and
clean-source time slices, includes low-SNR, dual RX, noise and out-of-prior controls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import platform
import shutil

import numpy as np
import scipy

from signal_policy import load_config
from room_evaluation import config_hash, evaluate_case, summarize

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"model"))
from make_room_scene import Catalog, PROFILES, SOURCE_MODES, file_hash, generate


def cases(split,count,seed_base=None):
    base=seed_base if seed_base is not None else (13001 if split=="dev" else 23001)
    levels=[(0,8),(8,14),(14,22),(22,30)]
    out=[]
    for i in range(count):
        out.append({"seed":base+i,"profile":list(PROFILES)[i%4],"kind":"positive",
                    "streams":2 if (i//16+i//4)%4==3 else 1,"snr":levels[(i//4)%4]})
    for i in range(8):
        out.append({"seed":base+1000+i,"profile":list(PROFILES)[i%4],"kind":"noise" if i<4 else "ood",
                    "streams":2 if i%2 else 1,"snr":(15,30)})
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("bin","model","src","emissions","out"):
        ap.add_argument("--"+k,type=Path,required=True)
    ap.add_argument("--split",choices=("dev","acceptance"),required=True)
    ap.add_argument("--policy",type=Path)
    ap.add_argument("--bandwidth-estimator",choices=("current","history"),default="current")
    ap.add_argument("--count",type=int,help="positive scenes; default dev 32, acceptance 96")
    ap.add_argument("--seed-base",type=int,help="new scene seed range for a separately frozen run")
    ap.add_argument("--seconds",type=float,default=.8)
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--rescore",action="store_true",help="dev only: reuse raw feeds to evaluate new policy")
    a=ap.parse_args()
    count=a.count if a.count is not None else (32 if a.split=="dev" else 96)
    if not 4<=count<=256 or count%4 or not .4<=a.seconds<=5:
        ap.error("count must be a multiple of four in 4..256; duration 0.4..5")
    if a.seed_base is not None and not 1<=a.seed_base<=2**31-2000:
        ap.error("seed base must be in 1..2^31-2000")
    if a.rescore and a.split!="dev":
        ap.error("acceptance policy is frozen; rescore is for dev only")
    config=load_config(a.policy)
    plan=cases(a.split,count,a.seed_base)
    a.out.mkdir(parents=True,exist_ok=True)
    manifest_path=a.out/"manifest.json"
    source_files=["tools/signal_policy.py","tools/room_evaluation.py","tools/run_room_checks.py","tools/policy/default.json",
                  "model/make_room_scene.py","model/make_scene.py","model/curf_model/widesig.py","model/curf_model/scoring.py","bench/classify.cpp",
                  "src/runtime/signal_stage.cu","include/curf/runtime/signal_stage.hpp","src/kernels/extraction_bank.cu",
                  "include/curf/cuda/extraction_bank.hpp"]
    manifest={"version":1,"split":a.split,"seconds":a.seconds,"cases":plan,"policy":config,"policy_sha256":config_hash(config),
              "bandwidth_estimator":a.bandwidth_estimator,
              "binary_sha256":file_hash(a.bin),"model_sha256":file_hash(a.model/"s2.curf"),"emissions_sha256":file_hash(a.emissions),
              "code_sha256":{p:file_hash(ROOT/p) for p in source_files},"command":sys.argv,
              "software":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
              "metric":"named family and any displayed protocol guess, errors counted once; unknown separate"}
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if not (a.resume or a.rescore):
            ap.error("output already has a frozen manifest; use --resume or a new directory")
        if old["cases"]!=json.loads(json.dumps(plan)) or old["seconds"]!=a.seconds:
            ap.error("case plan differs from frozen manifest")
        if not a.rescore and any(old.get(k)!=manifest[k] for k in ("policy_sha256","binary_sha256","model_sha256","emissions_sha256","code_sha256","bandwidth_estimator")):
            ap.error("resume requires identical policy, code, binary, input index and bandwidth estimator")
    else:
        if a.rescore:
            ap.error("dev rescore requires an existing run")
        manifest_path.write_text(json.dumps(manifest,indent=2))
        for name in source_files:
            target=a.out/"source_snapshot"/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
        frozen=a.out/"artifacts"
        frozen.mkdir(exist_ok=True)
        shutil.copy2(a.bin,frozen/"curf_classify")
        shutil.copy2(a.model/"s2.curf",frozen/"s2.curf")
        (frozen/"policy.json").write_text(json.dumps(config,indent=2))
    catalog=Catalog(a.src,a.emissions,a.split)
    if not a.rescore:
        # Check every possible source before generating/replaying a long suite.
        # A silent PTT validation tail must not invalidate the final controls.
        for mode in SOURCE_MODES:
            catalog.load(mode)
    results=[]
    for i,c in enumerate(plan):
        out=a.out/f"{i:02d}_{c['profile']}_{c['kind']}"
        if not a.rescore and not (a.resume and (out/"runtime.json").exists() and (out/"raw.feed").exists()):
            generate(catalog,out,c["seed"],c["profile"],a.seconds,c["streams"],c["kind"],c["snr"])
            truth=json.loads((out/"truth.json").read_text())
            cmd=[str(a.bin.resolve()),"--source","file","--file",str(out/"scene_0.p12"),"--streams",str(c["streams"]),
                 "--stage2",str(a.model.resolve()),"--freq",str(truth["freq"]),"--seconds","0","--realtime","--drop",
                 "--stage2-bandwidth",a.bandwidth_estimator,
                 "--band-mhz","44","--ui-out",str(out/"raw.feed")]
            (out/"command.json").write_text(json.dumps(cmd,indent=2))
            with (out/"runtime.json").open("w") as f:
                subprocess.run(cmd,check=True,stdout=f)
        r=evaluate_case(out,config,write_feed=not a.rescore)
        results.append(r)
        print(f"{i+1}/{len(plan)} {c['profile']} {c['kind']}: named={r['counts'].get('named',0)} errors={r['counts'].get('errors',0)} unknown={r['counts'].get('unknown',0)}",flush=True)
        (a.out/"progress.json").write_text(json.dumps(results,indent=2,ensure_ascii=False))
    report=summarize(results,a.split,min_named=20 if a.split=="dev" else 100)
    report["policy_sha256"]=config_hash(config)
    tag="trial_"+config_hash(config)[:12] if a.rescore else "result"
    (a.out/(tag+".json")).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
