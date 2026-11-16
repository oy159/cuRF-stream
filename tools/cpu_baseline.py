#!/usr/bin/env python3
"""CPU baselines for the P3 load table (run on the CUDA host, i5-11400H, AVX-512).

    python tools/cpu_baseline.py --onnx model/artifacts/run2/export/amc_fp32.onnx

1. Inference: ONNX Runtime CPU (MLAS kernels, all cores) on the same fp32 graph, windows/s per batch size.
2. Channelizer: the PyTorch implementation of tools/channelizer_torch.py on the CPU (MKL FFT), as x real time
   for 2 streams at 61.44 MS/s.
Prints JSON lines.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def bench(fn, min_s=3.0):
    fn()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < min_s:
        fn()
        n += 1
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--batches", default="1,8,64,256")
    args = ap.parse_args()

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = os.cpu_count()
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for b in [int(x) for x in args.batches.split(",")]:
        x = np.random.default_rng(0).normal(size=(b, 2, 1024)).astype(np.float32)
        t = bench(lambda: sess.run(None, {name: x}))
        print(json.dumps({"baseline": "onnxruntime_cpu", "threads": os.cpu_count(), "batch": b,
                          "ms_per_batch": round(t * 1e3, 3), "windows_per_s": round(b / t),
                          "ort": ort.__version__}), flush=True)

    import torch
    from channelizer_torch import channelize_torch, prototype
    torch.set_num_threads(os.cpu_count())
    block, M = 1 << 17, 512
    h = torch.from_numpy(prototype(M, 8))
    xs = (torch.randn(2, block, dtype=torch.complex64) * 0.2)
    with torch.no_grad():
        t = bench(lambda: channelize_torch(xs, h, M))
    print(json.dumps({"baseline": "torch_cpu_channelizer", "threads": torch.get_num_threads(), "streams": 2,
                      "block": block, "ms_per_block": round(t * 1e3, 3),
                      "x_realtime": round(block / 61.44e6 / t, 3)}), flush=True)


if __name__ == "__main__":
    main()
