"""Classify time-frequency regions in a short single-receiver SC16 recording.

    python model/replay.py capture.sc16 --checkpoint model/artifacts/demo/best.pt --out capture.predictions.json

The capture JSON beside the file supplies the tuned frequency, sample rate,
analog bandwidth and sample count. Output times use the original sample clock.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from curf_model.barker import find_barker
from curf_model.candidates import Candidate, detect_candidates, extract_iq
from curf_model.demo import LABELS
from curf_model.net import AmcResNet1D


def visible_label(candidate: Candidate, sample_rate: float, probabilities, confidence: float) -> str:
    index = int(np.argmax(probabilities))
    label = LABELS[index]
    if probabilities[index] < confidence or label in ("noise", "other_signal"):
        return "occupied"
    duration = (candidate.stop - candidate.start) / sample_rate
    bandwidth = candidate.bandwidth_hz
    # Frequency span and duration reject short/wide artifacts of analog labels.
    analog_limits = {"am": (60e3, 6e-3), "fm": (320e3, 4e-3)}
    if label in analog_limits:
        max_band, min_time = analog_limits[label]
        if bandwidth > max_band or duration < min_time:
            return "occupied"
    if label in ("dsss", "ofdm") and bandwidth < 6e6:
        return "occupied"
    if label == "gfsk" and bandwidth > 5e6:
        return "occupied"
    return label


@torch.inference_mode()
def classify(model, samples, rate, tune, candidates, confidence, device):
    rows = []
    for offset in range(0, len(candidates), 128):
        batch = candidates[offset:offset + 128]
        windows = [extract_iq(samples, rate, tune, candidate, max_windows=4) for candidate in batch]
        ends = np.cumsum([len(w) for w in windows])
        iq = torch.from_numpy(np.concatenate(windows)).to(device)
        predictions = model(iq).softmax(-1).cpu().numpy()
        for candidate, p in zip(batch, np.split(predictions, ends[:-1])):
            scores = p.mean(axis=0)
            rows.append(dict(
                start_ms=1000 * candidate.start / rate,
                stop_ms=1000 * candidate.stop / rate,
                center_hz=candidate.center_hz,
                bandwidth_hz=candidate.bandwidth_hz,
                label=visible_label(candidate, rate, scores, confidence),
                probabilities=dict(zip(LABELS, map(float, scores))),
            ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold-db", type=float, default=12.0)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--band-mhz", type=float)
    parser.add_argument("--analog", action="store_true", help="use fine resolution for sustained narrowband signals")
    parser.add_argument("--barker", action="store_true", help="add verified Barker-11 evidence in the 2.4 GHz band")
    parser.add_argument("--exclude-hz", type=float, action="append", default=[])
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("confidence must be between 0 and 1")
    capture_meta = json.loads((args.metadata or args.capture.with_suffix(".json")).read_text())
    config, stats = capture_meta["config"], capture_meta["rx"]
    if capture_meta["format"] != "sc16" or stats["channels"] != 1:
        parser.error("expected a single-channel SC16 recording")
    if stats["samples_per_channel"] * 4 != args.capture.stat().st_size:
        parser.error("capture length differs from the recorded sample count")
    rate, tune = config["sample_rate"], config["rx_frequency"]
    band = args.band_mhz * 1e6 if args.band_mhz is not None else config["rx_bandwidth"] * 0.90
    if not 0 < band <= rate:
        parser.error("usable band must be positive and within the capture sample rate")
    samples = np.memmap(args.capture, mode="r", dtype="<i2", shape=(stats["samples_per_channel"], 2))
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if checkpoint.get("profile") != "demo" or checkpoint["classes"] != LABELS:
        parser.error("expected a demo-profile checkpoint")
    model = AmcResNet1D(len(LABELS)).to(args.device).eval()
    model.load_state_dict(checkpoint["model"])
    t0 = perf_counter()
    regions = detect_candidates(samples, rate, tune, usable_band_hz=band,
                                threshold_db=args.threshold_db, exclude_hz=tuple(args.exclude_hz), analog=args.analog)
    rows = classify(model, samples, rate, tune, regions, args.confidence, args.device)
    if args.barker:
        for observed in find_barker(samples, rate, tune, band):
            start, stop = 1000 * observed.start / rate, 1000 * observed.stop / rate
            for row in rows:
                if (row["label"] == "ofdm" and abs(row["center_hz"] - observed.center_hz) < 3e6
                        and row["start_ms"] <= stop and start <= row["stop_ms"]):
                    row["label"] = "occupied"
            rows.append(dict(start_ms=start, stop_ms=stop, center_hz=observed.center_hz,
                             bandwidth_hz=observed.bandwidth_hz, label="dsss", evidence="barker11"))
        rows.sort(key=lambda row: (row["start_ms"], row["center_hz"]))
    report = dict(capture=str(args.capture), sample_rate=rate, tune_hz=tune, usable_band_hz=band,
                  seconds=len(samples) / rate, elapsed_s=perf_counter() - t0, classes=LABELS,
                  checkpoint_step=checkpoint.get("step"),
                  settings=dict(threshold_db=args.threshold_db, confidence=args.confidence,
                                exclude_hz=args.exclude_hz, analog=args.analog, barker=args.barker),
                  regions=rows)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "regions"} | {"regions": len(rows)}))


if __name__ == "__main__":
    main()
