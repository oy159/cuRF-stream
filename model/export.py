"""Evaluate a trained checkpoint and export the frozen model.

    python model/export.py --ckpt model/artifacts/run2/best.pt --out model/artifacts/run2/export

Writes:
  metrics.json          accuracy overall / per SNR (2 dB bins) / per class, confusion matrix (SNR >= 10 dB)
  amc_fp32.onnx         float32 graph (BatchNorm not folded; the engine converter folds it)
  amc_fp16.onnx         float16 graph, used as the TensorRT 11 strongly-typed FP16 baseline
  weights.npz           raw state_dict tensors (float32)
  reference.npz         fixed input batch with float32 and float16 PyTorch logits, for engine checks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from curf_model.net import AmcResNet1D
from curf_model.synth import CLASSES, Synth, SynthConfig


@torch.no_grad()
def predict(model, x, batch=4096):
    return torch.cat([model(x[i:i + batch]) for i in range(0, x.shape[0], batch)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--eval-size", type=int, default=131072)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = AmcResNet1D(len(CLASSES)).to(device).eval()
    model.load_state_dict(ckpt["model"])

    # ---------------------------------------------------------------- metrics
    synth = Synth(SynthConfig(), device=device, seed=777)  # disjoint from training/validation seeds
    xs, ys, snrs = zip(*[synth.batch(4096) for _ in range(args.eval_size // 4096)])
    x, y, snr = torch.cat(xs), torch.cat(ys), torch.cat(snrs)
    pred = predict(model, x).argmax(-1)
    ok = (pred == y).float()
    sig = ~torch.isnan(snr)

    by_snr = {}
    for lo in range(-4, 24, 2):
        m = sig & (snr >= lo) & (snr < lo + 2)
        by_snr[lo] = round(ok[m].mean().item(), 4)
    per_class = {c: round(ok[y == i].mean().item(), 4) for i, c in enumerate(CLASSES)}
    hi = (y == 0) | (sig & (snr >= 10))
    conf = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
    for t, p in zip(y[hi].tolist(), pred[hi].tolist()):
        conf[t, p] += 1

    # fp16 agreement on the same data (what the engines will run)
    model16 = AmcResNet1D(len(CLASSES)).to(device).eval().half()
    model16.load_state_dict(ckpt["model"])
    pred16 = predict(model16, x.half()).argmax(-1)

    metrics = {
        "checkpoint": str(args.ckpt), "step": ckpt.get("step"), "params": ckpt.get("params"),
        "mflop_per_window": ckpt.get("flops", 0) / 1e6, "classes": CLASSES, "eval_size": int(x.shape[0]),
        "acc": round(ok.mean().item(), 4), "acc_snr_ge_10dB": round(ok[sig & (snr >= 10)].mean().item(), 4),
        "acc_by_snr_2dB_bins": by_snr, "acc_by_class": per_class,
        "confusion_snr_ge_10dB": conf.tolist(),
        "fp16_top1_agreement": round((pred16 == pred).float().mean().item(), 5),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("acc", "acc_snr_ge_10dB", "fp16_top1_agreement")}))
    print("acc by SNR:", by_snr)
    print("acc by class:", per_class)

    # ----------------------------------------------------------------- export
    model_cpu = AmcResNet1D(len(CLASSES)).eval()
    model_cpu.load_state_dict({k: v.cpu() for k, v in ckpt["model"].items()})
    dummy = torch.zeros(1, 2, 1024)
    common = dict(input_names=["iq"], output_names=["logits"], dynamic_axes={"iq": {0: "batch"}, "logits": {0: "batch"}},
                  opset_version=17, dynamo=False)
    torch.onnx.export(model_cpu, dummy, args.out / "amc_fp32.onnx", **common)
    model_gpu16 = AmcResNet1D(len(CLASSES)).eval().half().to(device)
    model_gpu16.load_state_dict(ckpt["model"])
    torch.onnx.export(model_gpu16, dummy.half().to(device), args.out / "amc_fp16.onnx", **common)

    np.savez(args.out / "weights.npz", **{k: v.detach().float().cpu().numpy() for k, v in ckpt["model"].items()
                                          if v.dtype.is_floating_point})

    ref_synth = Synth(SynthConfig(), device=device, seed=4242)
    rx, ry, rsnr = ref_synth.batch(256)
    with torch.no_grad():
        l32 = model(rx).float().cpu().numpy()
        l16 = model16(rx.half()).float().cpu().numpy()
    np.savez(args.out / "reference.npz", input=rx.cpu().numpy(), label=ry.cpu().numpy(), snr=rsnr.cpu().numpy(),
             logits_fp32=l32, logits_fp16=l16)

    # ONNX Runtime cross-check of the fp32 graph
    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.out / "amc_fp32.onnx"), providers=["CPUExecutionProvider"])
    lo = sess.run(None, {"iq": rx.cpu().numpy()})[0]
    print(json.dumps({"onnxruntime_max_abs_diff": float(np.abs(lo - l32).max()),
                      "fp16_max_abs_diff": float(np.abs(l16 - l32).max())}))


if __name__ == "__main__":
    main()
