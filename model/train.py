"""Train the ResNet-1D modulation classifier on on-the-fly synthetic data.

    python model/train.py --out model/artifacts/run1 --steps 30000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from curf_model.demo import load_examples, make_synth, profile_classes
from curf_model.net import AmcResNet1D, count_flops


def make_val(n: int, seed: int, device, profile: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    synth = make_synth(profile, device, seed)
    parts = [synth.batch(min(4096, n - start)) for start in range(0, n, 4096)]
    return tuple(torch.cat(p) for p in zip(*parts))  # type: ignore[return-value]


@torch.no_grad()
def evaluate(model, x, y, snr, classes, batch: int = 1024) -> dict:
    model.eval()
    pred = torch.cat([model(x[i:i + batch]).argmax(-1) for i in range(0, x.shape[0], batch)])
    model.train()
    ok = (pred == y).float()
    out = {"acc": ok.mean().item(),
           "acc_by_class": {c: ok[y == i].mean().item() for i, c in enumerate(classes) if (y == i).any()}}
    sig = ~torch.isnan(snr)
    for lo, hi in [(-4, 0), (0, 6), (6, 12), (12, 24)]:
        m = sig & (snr >= lo) & (snr < hi)
        if m.any():
            out[f"acc_snr[{lo},{hi})"] = ok[m].mean().item()
    if (~sig).any():
        out["acc_noise"] = ok[~sig].mean().item()
    return out


@torch.no_grad()
def evaluate_captures(model, captures, classes):
    model.eval()
    x, y = captures
    pred = torch.cat([model(part).argmax(-1) for part in x.split(512)])
    per_class = {c: (pred[y == i] == i).float().mean().item()
                 for i, c in enumerate(classes) if (y == i).any()}
    model.train()
    return per_class, sum(per_class.values()) / len(per_class)


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
    ap.add_argument("--profile", choices=("run1", "demo"), default="run1")
    ap.add_argument("--train-iq", type=Path)
    ap.add_argument("--validation-iq", type=Path)
    ap.add_argument("--capture-fraction", type=float, default=0.2)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    if min(args.val_size, args.steps, args.warmup, args.val_every) < 1 or args.batch < 2:
        ap.error("training and validation counts must be positive and batch must be at least two")
    if not 0 < args.capture_fraction < 1:
        ap.error("capture fraction must be in (0, 1)")
    if args.train_iq and args.validation_iq and args.train_iq.resolve() == args.validation_iq.resolve():
        ap.error("training and capture validation must use separate files")
    classes = profile_classes(args.profile)
    model = AmcResNet1D(len(classes)).to(device)
    flops = count_flops(model)
    params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"params": params, "mflop_per_window": flops / 1e6}), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    train_synth = make_synth(args.profile, device, args.seed + 1000)
    val = make_val(args.val_size, seed=12345, device=device, profile=args.profile)
    captures = load_examples(args.train_iq, classes, device) if args.train_iq else None
    capture_val = load_examples(args.validation_iq, classes, device) if args.validation_iq else None
    n_capture = min(args.batch - 1, max(1, round(args.batch * args.capture_fraction))) if captures else 0

    log = open(args.out / "train_log.jsonl", "w")
    best, t0, running = -1.0, time.time(), 0.0
    for step in range(1, args.steps + 1):
        x, y, _ = train_synth.batch(args.batch - n_capture)
        if captures:
            rx, ry = captures
            pick = torch.randint(len(ry), (n_capture,), device=device)
            x, y = torch.cat((x, rx[pick])), torch.cat((y, ry[pick]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x), y, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()

        if step % args.val_every == 0 or step == args.steps:
            metrics = evaluate(model, *val, classes)
            ranking = metrics["acc"]
            if capture_val:
                by_class, macro = evaluate_captures(model, capture_val, classes)
                metrics.update(capture_acc_by_class=by_class, capture_macro_acc=macro)
                ranking = (ranking + macro) / 2
            rec = {"step": step, "loss": running, "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - t0, 1), **metrics}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if ranking > best:
                best = ranking
                torch.save({"model": model.state_dict(), "classes": classes, "profile": args.profile,
                            "step": step, "metrics": metrics, "params": params, "flops": flops}, args.out / "best.pt")
    torch.save({"model": model.state_dict(), "classes": classes, "profile": args.profile, "step": args.steps, "params": params,
                "flops": flops}, args.out / "last.pt")
    print(json.dumps({"done": True, "best_score" if capture_val else "best_acc": best}), flush=True)


if __name__ == "__main__":
    main()
