#!/usr/bin/env python3
"""Train the stage-2 (signal-level) classifier on a dataset from make_dataset.py.

    python model/train_stage2.py --data ~/dataset/ds_v2 --out model/artifacts/s2_run1 [--epochs 40]

The whole dataset sits on the GPU as float16 (0.7 GB for 130k + 32k windows). Augmentation per batch: a random
carrier phase and random spectral inversion (complex conjugate), both of which keep every class. The bandwidth
estimate goes in as a third input channel, constant over the window: (log10(B / Hz) - 5) / 2.

Writes best.pt (best val accuracy), metrics.json (overall, per SNR bin, per class, per mode, the confusion at
>= 10 dB) and the training log.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from curf_model.net import SignalResNet1D, count_flops


def bw_channel(log_bw: torch.Tensor, L: int) -> torch.Tensor:
    return ((log_bw - 5.0) / 2.0)[:, None, None].expand(-1, 1, L)


def make_input(x: torch.Tensor, log_bw: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.float(), bw_channel(log_bw, x.shape[-1])], dim=1)


def augment(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    ph = torch.rand(x.shape[0], 1, device=x.device) * (2 * torch.pi)
    c, s = torch.cos(ph), torch.sin(ph)
    i, q = x[:, 0], x[:, 1]
    i, q = i * c - q * s, i * s + q * c
    flip = torch.rand(x.shape[0], 1, device=x.device) < 0.5
    q = torch.where(flip, -q, q)
    return torch.stack([i, q], 1)


@torch.no_grad()
def predict(model, x, lbw, batch=2048):
    model.eval()
    out = []
    for i in range(0, x.shape[0], batch):
        with torch.autocast("cuda", torch.float16):
            out.append(model(make_input(x[i:i + batch], lbw[i:i + batch])).argmax(-1))
    model.train()
    return torch.cat(out).cpu().numpy()


def report(pred, y, snr, mode, classes, modes) -> dict:
    ok = pred == y
    hi = (snr >= 10) | np.isnan(snr)
    out = {"acc": float(ok.mean()), "acc_ge10db": float(ok[hi].mean())}
    for lo, h in [(-5, 0), (0, 10), (10, 20), (20, 30)]:
        m = (snr >= lo) & (snr < h)
        out[f"acc_snr[{lo},{h})"] = float(ok[m].mean())
    out["per_class"] = {c: float(ok[y == k].mean()) for k, c in enumerate(classes)}
    out["per_mode_ge10db"] = {m: float(ok[(mode == k) & hi].mean()) for k, m in enumerate(modes) if ((mode == k) & hi).any()}
    cm = np.zeros((len(classes), len(classes)), np.int64)
    np.add.at(cm, (y[hi], pred[hi]), 1)
    out["confusion_ge10db"] = cm.tolist()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = "cuda"
    meta = json.loads((a.data / "meta.json").read_text())
    classes, modes = meta["classes"], meta["modes"]

    def load(split):
        z = np.load(a.data / f"{split}.npz")
        return (torch.from_numpy(z["x"]).to(dev), torch.from_numpy(z["y"].astype(np.int64)).to(dev),
                torch.from_numpy(z["log_bw"]).to(dev), z["snr"], z["mode"])

    xt, yt, bt, _, _ = load("train")
    xv, yv, bv, sv, mv = load("val")
    yv_np = yv.cpu().numpy()
    L = xt.shape[-1]
    model = SignalResNet1D(len(classes), a.width).to(dev)
    params = sum(p.numel() for p in model.parameters())
    flops = count_flops(model, L, in_ch=3)
    print(f"model: {params} parameters, {flops / 1e6:.1f} MFLOP/window; train {len(yt)}, val {len(yv)}")

    steps = a.epochs * len(yt) // a.batch
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler()
    log, best, t0 = [], -1.0, time.time()
    every = len(yt) // a.batch  # one epoch
    for step in range(1, steps + 1):
        i = torch.randint(0, len(yt), (a.batch,), device=dev)
        with torch.autocast("cuda", torch.float16):
            loss = F.cross_entropy(model(make_input(augment(xt[i]), bt[i])), yt[i], label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % every == 0 or step == steps:
            pred = predict(model, xv, bv)
            acc = float((pred == yv_np).mean())
            log.append({"step": step, "epoch": step / every, "loss": float(loss), "val_acc": acc, "s": time.time() - t0})
            print(f"epoch {step / every:5.1f}  loss {float(loss):.3f}  val {acc:.4f}  ({time.time() - t0:.0f} s)", flush=True)
            if acc > best:
                best = acc
                torch.save({"model": model.state_dict(), "classes": classes, "width": a.width, "length": L,
                            "in_ch": 3, "bw_channel": "(log10(B) - 5) / 2", "data": str(a.data)}, a.out / "best.pt")
    model.load_state_dict(torch.load(a.out / "best.pt", map_location=dev)["model"])
    m = report(predict(model, xv, bv), yv_np, sv, mv, classes, modes)
    m.update({"params": params, "mflop_per_window": flops / 1e6, "epochs": a.epochs, "train_s": time.time() - t0,
              "classes": classes, "data": str(a.data)})
    (a.out / "metrics.json").write_text(json.dumps(m, indent=1))
    (a.out / "log.json").write_text(json.dumps(log))
    print(json.dumps({k: v for k, v in m.items() if k not in ("per_mode_ge10db", "confusion_ge10db")}, indent=1))
    cm = np.array(m["confusion_ge10db"])
    conf = [f"{classes[r]}->{classes[c]} {cm[r, c] / cm[r].sum():.0%}" for r in range(len(classes)) for c in range(len(classes))
            if r != c and cm[r].sum() and cm[r, c] / cm[r].sum() >= 0.03]
    print("confusions >= 3 % at >= 10 dB:", ", ".join(conf))
    worst = sorted(m["per_mode_ge10db"].items(), key=lambda kv: kv[1])[:8]
    print("weakest modes at >= 10 dB:", ", ".join(f"{k} {v:.2f}" for k, v in worst))


if __name__ == "__main__":
    main()
