"""Train the ResNet-1D modulation classifier on on-the-fly synthetic data.

    python model/train.py --out model/artifacts/run2 --steps 30000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from curf_model.net import AmcResNet1D, count_flops
from curf_model.synth import CLASSES, Synth, SynthConfig


def make_val(n: int, seed: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    synth = Synth(SynthConfig(), device=device, seed=seed)
    parts = [synth.batch(4096) for _ in range(n // 4096)]
    return tuple(torch.cat(p) for p in zip(*parts))  # type: ignore[return-value]


@torch.no_grad()
def evaluate(model, x, y, snr, batch: int = 4096) -> dict:
    model.eval()
    pred = torch.cat([model(x[i:i + batch]).argmax(-1) for i in range(0, x.shape[0], batch)])
    model.train()
    ok = (pred == y).float()
    out = {"acc": ok.mean().item()}
    sig = ~torch.isnan(snr)
    for lo, hi in [(-4, 0), (0, 6), (6, 12), (12, 24)]:
        m = sig & (snr >= lo) & (snr < hi)
        out[f"acc_snr[{lo},{hi})"] = ok[m].mean().item()
    out["acc_noise"] = ok[~sig].mean().item()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-size", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    model = AmcResNet1D(len(CLASSES)).to(device)
    flops = count_flops(model)
    params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"params": params, "mflop_per_window": flops / 1e6}), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    train_synth = Synth(SynthConfig(), device=device, seed=args.seed + 1000)
    val = make_val(args.val_size, seed=12345, device=device)

    log = open(args.out / "train_log.jsonl", "w")
    best, t0, running = -1.0, time.time(), 0.0
    for step in range(1, args.steps + 1):
        x, y, _ = train_synth.batch(args.batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x), y, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()

        if step % args.val_every == 0 or step == args.steps:
            metrics = evaluate(model, *val)
            rec = {"step": step, "loss": running, "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - t0, 1), **metrics}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if metrics["acc"] > best:
                best = metrics["acc"]
                torch.save({"model": model.state_dict(), "classes": CLASSES, "step": step, "metrics": metrics,
                            "params": params, "flops": flops}, args.out / "best.pt")
    torch.save({"model": model.state_dict(), "classes": CLASSES, "step": args.steps, "params": params,
                "flops": flops}, args.out / "last.pt")
    print(json.dumps({"done": True, "best_acc": best}), flush=True)


if __name__ == "__main__":
    main()
