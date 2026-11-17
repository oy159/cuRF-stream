#!/usr/bin/env python3
"""Export the stage-2 (signal-level) classifier for the engine: ONNX + reference batch, then model/convert.py.

    python model/export_stage2.py --ckpt model/artifacts/s2_run3/best.pt --data ~/dataset/ds_v4 \
        --out model/artifacts/s2_run3/export
    python model/convert.py --onnx model/artifacts/s2_run3/export/s2_fp32.onnx \
        --ref model/artifacts/s2_run3/export/reference.npz --out model/artifacts/s2_run3/export --name s2

Writes s2_fp32.onnx (input "x" [batch, 3, 1024]: I, Q, (log10(B / Hz) - 5) / 2 constant over the window),
reference.npz (256 validation windows, 20 per class where available, with float32 / float16 PyTorch logits)
and metrics.json (classes, the training metrics, fp16 agreement on the whole validation split), which
ui_server.py and the classifier read for the class names.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from curf_model.net import SignalResNet1D
from train_stage2 import make_input


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="dataset directory (val.npz) for the reference batch")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location=dev)
    classes = ck["classes"]
    model = SignalResNet1D(len(classes), ck["width"]).to(dev).eval()
    model.load_state_dict(ck["model"])

    z = np.load(a.data / "val.npz")
    x = torch.from_numpy(z["x"]).to(dev)
    bw = torch.from_numpy(z["log_bw"]).to(dev)
    y = z["y"]
    with torch.no_grad():
        p32 = torch.cat([model(make_input(x[i:i + 2048], bw[i:i + 2048])).argmax(-1) for i in range(0, len(y), 2048)])
        m16 = SignalResNet1D(len(classes), ck["width"]).to(dev).eval().half()
        m16.load_state_dict(ck["model"])
        p16 = torch.cat([m16(make_input(x[i:i + 2048], bw[i:i + 2048]).half()).argmax(-1) for i in range(0, len(y), 2048)])
    agree = float((p32 == p16).float().mean())

    # reference batch: up to 20 windows of every class
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.nonzero(y == k)[0], min(20, int((y == k).sum())), replace=False)
                          for k in range(len(classes)) if (y == k).any()])[:256]
    rx = make_input(x[idx], bw[idx]).float()
    with torch.no_grad():
        l32 = model(rx).float().cpu().numpy()
        l16 = m16(rx.half()).float().cpu().numpy()
    np.savez(a.out / "reference.npz", input=rx.cpu().numpy(), label=y[idx], snr=z["snr"][idx], logits_fp32=l32,
             logits_fp16=l16)

    cpu = SignalResNet1D(len(classes), ck["width"]).eval()
    cpu.load_state_dict({k: v.cpu() for k, v in ck["model"].items()})
    torch.onnx.export(cpu, torch.zeros(1, 3, ck["length"]), a.out / "s2_fp32.onnx", input_names=["x"],
                      output_names=["logits"], dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(a.out / "s2_fp32.onnx"), providers=["CPUExecutionProvider"])
    lo = sess.run(None, {"x": rx.cpu().numpy()})[0]

    train_metrics = json.loads((a.ckpt.parent / "metrics.json").read_text()) if (a.ckpt.parent / "metrics.json").exists() else {}
    metrics = {"classes": classes, "stage": 2, "length": ck["length"], "in_ch": 3, "bw_channel": ck["bw_channel"],
               "checkpoint": str(a.ckpt), "data": str(a.data), "fp16_top1_agreement_val": round(agree, 5),
               **{k: v for k, v in train_metrics.items() if k.startswith("acc") or k == "per_class"}}
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({"fp16_top1_agreement_val": agree, "onnxruntime_max_abs_diff": float(np.abs(lo - l32).max()),
                      "fp16_max_abs_diff": float(np.abs(l16 - l32).max()), "reference_windows": int(len(idx))}))


if __name__ == "__main__":
    main()
