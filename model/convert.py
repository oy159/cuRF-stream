"""Offline converter: ONNX -> cuRF engine model (.curf) + golden activations (.golden).

    python model/convert.py --onnx export/amc_fp32.onnx --ref export/reference.npz --out export/
    python model/convert.py --selftest

Passes
  1. constant/BN folding: BatchNormalization after Conv is folded into the conv weights and bias
     (PyTorch's eval-mode exporter usually folds it already; the pass handles graphs where it did not).
  2. epilogue fusion: Conv [-> BN] [-> Add(residual)] [-> Relu] becomes one `conv` op whose
     epilogue does bias + residual + ReLU; ReduceMean(last axis) -> Gemm becomes one `gap_linear` op.
  3. layout: activations are channels-last (NLC, [batch][length][channels]), so the implicit-GEMM
     K dimension (tap-major, then input channel) is contiguous in memory; weights become
     [cout][kpad] fp16 with kpad = round_up(k * cin, 32) and zero padding.

The engine only implements the two fused ops; anything else in the graph is a conversion error.

File format (little endian), mirrored by include/curf/engine/model.hpp:
  "CURFNET1"  u32 version, n_tensors, n_ops, input, output, 3x reserved
  tensors     n x { u32 channels, length; char name[56] }
  ops         n x { u32 kind, in, res, out, cin, cout, k, stride, pad, relu, kpad, reserved;
                    u64 w_off, b_off; char name[48] }
  u64 blob_bytes, blob   (every array starts on a 64-byte boundary)
  conv:       w = fp16 [cout][kpad], b = fp32 [cout]
  gap_linear: w = fp32 [cout][cin],  b = fp32 [cout]   (in [L][cin] -> mean over L -> out [cout])

Golden file: "CURFGLD1" u32 n_full, n_layer, n_tensors, reserved;
  input fp32 [n_full][L][2], labels i32 [n_full], logits fp32 [n_full][classes] (PyTorch fp32),
  then per tensor { u32 channels, length; fp32 [n_layer][length][channels] } from the numpy executor.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NO_TENSOR = 0xFFFFFFFF
KIND_CONV, KIND_GAP_LINEAR = 0, 1
KPAD = 32


@dataclass
class Tensor:
    name: str
    channels: int
    length: int


@dataclass
class Op:
    kind: int
    name: str
    inp: int
    out: int
    res: int = NO_TENSOR
    cin: int = 0
    cout: int = 0
    k: int = 1
    stride: int = 1
    pad: int = 0
    relu: bool = False
    w: np.ndarray = field(default=None, repr=False)   # conv: [cout, cin, k] fp32; linear: [cout, cin]
    b: np.ndarray = field(default=None, repr=False)

    @property
    def kpad(self) -> int:
        return -(-self.k * self.cin // KPAD) * KPAD if self.kind == KIND_CONV else 0


# ----------------------------------------------------------------------------- ONNX -> fused graph

def attr(node, name, default=None):
    import onnx
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


def convert_graph(model) -> tuple[list[Tensor], list[Op], int, int]:
    import onnx.numpy_helper as nh
    g = model.graph
    init = {i.name: nh.to_array(i).astype(np.float32) for i in g.initializer}
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    done: set[int] = set()

    def only_consumer(name, op_type):
        c = consumers.get(name, [])
        return c[0] if len(c) == 1 and c[0].op_type == op_type else None

    inp = g.input[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if len(dims) != 3 or dims[1] <= 0 or dims[2] <= 0:
        raise ValueError(f"expected input [batch, C, L], got {dims}")
    tensors = [Tensor(inp.name, dims[1], dims[2])]
    index = {inp.name: 0}
    ops: list[Op] = []

    def new_tensor(name, c, length):
        index[name] = len(tensors)
        tensors.append(Tensor(name, c, length))
        return index[name]

    for n in g.node:
        if id(n) in done:
            continue
        if n.op_type == "Conv":
            if attr(n, "group", 1) != 1 or any(d != 1 for d in attr(n, "dilations", [1])):
                raise ValueError(f"{n.name}: grouped/dilated conv not supported")
            pads = attr(n, "pads", [0, 0])
            if pads[0] != pads[1]:
                raise ValueError(f"{n.name}: asymmetric padding not supported")
            w = init[n.input[1]].copy()
            b = init[n.input[2]].copy() if len(n.input) > 2 and n.input[2] else np.zeros(w.shape[0], np.float32)
            src = tensors[index[n.input[0]]]
            op = Op(KIND_CONV, n.name, index[n.input[0]], NO_TENSOR, cin=w.shape[1], cout=w.shape[0], k=w.shape[2],
                    stride=attr(n, "strides", [1])[0], pad=pads[0], w=w, b=b)
            if op.cin != src.channels:
                raise ValueError(f"{n.name}: channel mismatch {op.cin} vs {src.channels}")
            out = n.output[0]
            if (bn := only_consumer(out, "BatchNormalization")) is not None:          # pass 1: BN fold
                gamma, beta, mean, var = (init[x] for x in bn.input[1:5])
                s = gamma / np.sqrt(var + attr(bn, "epsilon", 1e-5))
                op.w = op.w * s[:, None, None]
                op.b = (op.b - mean) * s + beta
                done.add(id(bn))
                out = bn.output[0]
            add = only_consumer(out, "Add")
            other = None if add is None else (add.input[1] if add.input[0] == out else add.input[0])
            # Fuse the Add into whichever conv runs second (e.g. conv2, not the shortcut conv before it).
            if add is not None and other in index:                                      # pass 2: residual
                op.res = index[other]
                done.add(id(add))
                out = add.output[0]
            if (relu := only_consumer(out, "Relu")) is not None:                     # pass 2: activation
                op.relu = True
                done.add(id(relu))
                out = relu.output[0]
            lout = (src.length + 2 * op.pad - op.k) // op.stride + 1
            op.out = new_tensor(out, op.cout, lout)
            if op.res != NO_TENSOR and (tensors[op.res].channels, tensors[op.res].length) != (op.cout, lout):
                raise ValueError(f"{n.name}: residual shape mismatch")
            ops.append(op)
        elif n.op_type == "ReduceMean":
            axes = attr(n, "axes", None)
            if axes not in ([-1], [2]) or attr(n, "keepdims", 1) != 0:
                raise ValueError(f"{n.name}: only mean over the length axis is supported")
            gemm = only_consumer(n.output[0], "Gemm")
            if gemm is None or attr(gemm, "transB", 0) != 1 or attr(gemm, "transA", 0) != 0:
                raise ValueError(f"{n.name}: ReduceMean must feed a Gemm(transB=1)")
            w = init[gemm.input[1]] * attr(gemm, "alpha", 1.0)
            b = init[gemm.input[2]] * attr(gemm, "beta", 1.0)
            src = tensors[index[n.input[0]]]
            op = Op(KIND_GAP_LINEAR, gemm.name, index[n.input[0]], NO_TENSOR, cin=src.channels, cout=w.shape[0],
                    w=w.astype(np.float32), b=b.astype(np.float32))
            op.out = new_tensor(gemm.output[0], op.cout, 1)
            done.add(id(gemm))
            ops.append(op)
        else:
            raise ValueError(f"unsupported op {n.op_type} ({n.name})")
        done.add(id(n))

    output = index[g.output[0].name]
    return tensors, ops, 0, output


# ----------------------------------------------------------------------------- numpy executor (NLC, fp32)

def conv_nlc(x: np.ndarray, op: Op) -> np.ndarray:
    b, lin, cin = x.shape
    lout = (lin + 2 * op.pad - op.k) // op.stride + 1
    xp = np.pad(x, ((0, 0), (op.pad, op.pad), (0, 0)))
    cols = np.stack([xp[:, t:t + op.stride * (lout - 1) + 1:op.stride, :] for t in range(op.k)], axis=2)
    wk = op.w.transpose(0, 2, 1).reshape(op.cout, op.k * cin)          # [cout][t][ci]
    return cols.reshape(b, lout, op.k * cin) @ wk.T + op.b


def execute(tensors, ops, x_ncl: np.ndarray) -> list[np.ndarray]:
    acts: list = [None] * len(tensors)
    acts[0] = np.ascontiguousarray(x_ncl.transpose(0, 2, 1), dtype=np.float32)   # NCL -> NLC
    for op in ops:
        x = acts[op.inp]
        if op.kind == KIND_CONV:
            y = conv_nlc(x, op)
            if op.res != NO_TENSOR:
                y = y + acts[op.res]
            if op.relu:
                y = np.maximum(y, 0)
        else:
            y = (x.mean(axis=1) @ op.w.T + op.b)[:, None, :]
        acts[op.out] = y.astype(np.float32)
    return acts


# ----------------------------------------------------------------------------- serialisation

def name_bytes(s: str, n: int) -> bytes:
    raw = s.encode()[: n - 1]
    return raw + b"\0" * (n - len(raw))


def write_model(path: Path, tensors, ops, inp, out) -> int:
    blob = bytearray()

    def put(arr: np.ndarray) -> int:
        blob.extend(b"\0" * (-len(blob) % 64))
        off = len(blob)
        blob.extend(np.ascontiguousarray(arr).tobytes())
        return off

    recs = []
    for op in ops:
        if op.kind == KIND_CONV:
            wk = op.w.transpose(0, 2, 1).reshape(op.cout, op.k * op.cin)
            wp = np.zeros((op.cout, op.kpad), np.float16)
            wp[:, : op.k * op.cin] = wk.astype(np.float16)
            w_off = put(wp)
        else:
            w_off = put(op.w.astype(np.float32))
        b_off = put(op.b.astype(np.float32))
        recs.append(struct.pack("<12I2Q48s", op.kind, op.inp, op.res, op.out, op.cin, op.cout, op.k, op.stride,
                                op.pad, int(op.relu), op.kpad, 0, w_off, b_off, name_bytes(op.name, 48)))
    with open(path, "wb") as f:
        f.write(b"CURFNET1")
        f.write(struct.pack("<8I", 1, len(tensors), len(ops), inp, out, 0, 0, 0))
        for t in tensors:
            f.write(struct.pack("<2I56s", t.channels, t.length, name_bytes(t.name, 56)))
        for r in recs:
            f.write(r)
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
    return len(blob)


def write_golden(path: Path, x_ncl, labels, logits, acts, n_layer):
    with open(path, "wb") as f:
        f.write(b"CURFGLD1")
        f.write(struct.pack("<4I", x_ncl.shape[0], n_layer, len(acts), 0))
        f.write(np.ascontiguousarray(x_ncl.transpose(0, 2, 1), dtype=np.float32).tobytes())
        f.write(labels.astype(np.int32).tobytes())
        f.write(logits.astype(np.float32).tobytes())
        for a in acts:
            f.write(struct.pack("<2I", a.shape[2], a.shape[1]))
            f.write(np.ascontiguousarray(a[:n_layer], dtype=np.float32).tobytes())


def summary(tensors, ops) -> dict:
    rows, total = [], 0
    for op in ops:
        lout = tensors[op.out].length
        if op.kind == KIND_CONV:
            flops = 2 * lout * op.cout * op.cin * op.k
            gemm = {"M_per_window": lout, "N": op.cout, "K": op.k * op.cin, "Kpad": op.kpad}
        else:
            flops = 2 * op.cin * op.cout
            gemm = {}
        total += flops
        bytes_io = 2 * (tensors[op.inp].channels * tensors[op.inp].length + tensors[op.out].channels * lout
                        + (tensors[op.res].channels * tensors[op.res].length if op.res != NO_TENSOR else 0))
        rows.append({"name": op.name, "kind": ["conv", "gap_linear"][op.kind], "in": op.inp, "out": op.out,
                     "res": None if op.res == NO_TENSOR else op.res, "k": op.k, "stride": op.stride,
                     "relu": op.relu, "flops_per_window": flops, "fp16_bytes_per_window": bytes_io, **gemm})
    return {"tensors": [vars(t) for t in tensors], "ops": rows, "mflop_per_window": total / 1e6}


# ----------------------------------------------------------------------------- self test (BN folding path)

def selftest() -> None:
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    c, l = 8, 64
    inits = {
        "w0": rng.normal(size=(c, 2, 5)).astype(np.float32), "b0": rng.normal(size=c).astype(np.float32),
        "g": rng.uniform(0.5, 2, c).astype(np.float32), "be": rng.normal(size=c).astype(np.float32),
        "mu": rng.normal(size=c).astype(np.float32), "var": rng.uniform(0.5, 2, c).astype(np.float32),
        "w1": rng.normal(size=(c, c, 3)).astype(np.float32),
        "g1": rng.uniform(0.5, 2, c).astype(np.float32), "be1": rng.normal(size=c).astype(np.float32),
        "mu1": rng.normal(size=c).astype(np.float32), "var1": rng.uniform(0.5, 2, c).astype(np.float32),
        "fw": rng.normal(size=(3, c)).astype(np.float32), "fb": rng.normal(size=3).astype(np.float32),
    }
    nodes = [
        helper.make_node("Conv", ["x", "w0", "b0"], ["c0"], kernel_shape=[5], pads=[2, 2], strides=[2]),
        helper.make_node("BatchNormalization", ["c0", "g", "be", "mu", "var"], ["n0"], epsilon=1e-5),
        helper.make_node("Relu", ["n0"], ["r0"]),
        helper.make_node("Conv", ["r0", "w1"], ["c1"], kernel_shape=[3], pads=[1, 1]),
        helper.make_node("BatchNormalization", ["c1", "g1", "be1", "mu1", "var1"], ["n1"], epsilon=1e-5),
        helper.make_node("Add", ["r0", "n1"], ["a1"]),
        helper.make_node("Relu", ["a1"], ["r1"]),
        helper.make_node("ReduceMean", ["r1"], ["m"], axes=[-1], keepdims=0),
        helper.make_node("Gemm", ["m", "fw", "fb"], ["y"], transB=1),
    ]
    graph = helper.make_graph(nodes, "selftest", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2, l])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 3])],
                              [numpy_helper.from_array(v, k) for k, v in inits.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    onnx.checker.check_model(model)
    x = rng.normal(size=(4, 2, l)).astype(np.float32)
    want = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    tensors, ops, _, out = convert_graph(model)
    assert [op.kind for op in ops] == [KIND_CONV, KIND_CONV, KIND_GAP_LINEAR], ops
    assert ops[1].res == ops[0].out and ops[1].relu
    got = execute(tensors, ops, x)[out][:, 0, :]
    err = float(np.abs(got - want).max())
    print(json.dumps({"selftest_max_abs_diff": err, "ops": len(ops)}))
    assert err < 1e-4, err


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path)
    ap.add_argument("--ref", type=Path, help="reference.npz from export.py")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--name", default="amc")
    ap.add_argument("--layer-windows", type=int, default=32, help="windows with per-layer golden activations")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    import onnx
    model = onnx.load(str(args.onnx))
    tensors, ops, inp, out = convert_graph(model)
    args.out.mkdir(parents=True, exist_ok=True)
    blob = write_model(args.out / f"{args.name}.curf", tensors, ops, inp, out)

    ref = np.load(args.ref)
    acts = execute(tensors, ops, ref["input"])
    logits = acts[out][:, 0, :]
    # Strict fp32 check against onnxruntime (CPU). PyTorch's "fp32" logits were computed on the GPU where
    # cuDNN uses TF32 by default, so they are only reported, not used as the golden.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    ort_logits = sess.run(None, {sess.get_inputs()[0].name: ref["input"]})[0]
    err = float(np.abs(logits - ort_logits).max())
    err_pt = float(np.abs(logits - ref["logits_fp32"]).max())
    if err > 1e-4:
        raise SystemExit(f"fused graph disagrees with onnxruntime: max |diff| = {err}")
    write_golden(args.out / f"{args.name}.golden", ref["input"], ref["label"], ort_logits, acts,
                 min(args.layer_windows, ref["input"].shape[0]))
    s = summary(tensors, ops)
    s["executor_vs_onnxruntime_max_abs_diff"] = err
    s["executor_vs_pytorch_gpu_tf32_max_abs_diff"] = err_pt
    s["blob_bytes"] = blob
    (args.out / f"{args.name}.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({"ops": len(ops), "tensors": len(tensors), "blob_bytes": blob,
                      "mflop_per_window": round(s["mflop_per_window"], 3), "vs_ort": err, "vs_pytorch_tf32": err_pt}))
    for r in s["ops"]:
        print(f"  {r['kind']:10s} {r['name'][:40]:40s} in={r['in']:2d} res={str(r['res']):4s} out={r['out']:2d} "
              f"k={r['k']} s={r['stride']} relu={int(r['relu'])} N={r.get('N', '-')} K={r.get('K', '-')}")


if __name__ == "__main__":
    main()
