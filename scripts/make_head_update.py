#!/usr/bin/env python3
"""Locate the final FC weight in models/large.onnx and write a perturbed copy.

Finds the last Gemm/MatMul node's weight initializer with the onnx package,
searches the file for that tensor's exact raw bytes (must occur exactly once),
and writes a same-size copy where those float32 values get +1e-3 * N(0,1)
noise (numpy seed 12345). Metadata goes to .cache/head/head_meta.txt.
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parent.parent
src = ROOT / "models" / "large.onnx"
outdir = ROOT / ".cache" / "head"
outdir.mkdir(parents=True, exist_ok=True)
dst = outdir / "large_head_perturbed.onnx"

raw_file = src.read_bytes()
model = onnx.load(str(src))
inits = {t.name: t for t in model.graph.initializer}

fc = None
for node in reversed(model.graph.node):
    if node.op_type in ("Gemm", "MatMul"):
        for inp in node.input:
            if inp in inits and len(inits[inp].dims) == 2:
                fc = (node, inits[inp])
                break
        if fc:
            break
if fc is None:
    sys.exit("no final FC weight initializer found")
node, t = fc
arr = numpy_helper.to_array(t)
assert arr.dtype == np.float32, arr.dtype
raw = arr.astype("<f4").tobytes()
stored = "raw_data" if t.raw_data else "float_data"
if t.raw_data:
    assert t.raw_data == raw

off = raw_file.find(raw)
if off < 0:
    sys.exit("tensor bytes not found in file")
if raw_file.find(raw, off + 1) >= 0:
    sys.exit("tensor bytes occur more than once in file")

rng = np.random.default_rng(12345)
noise = (rng.standard_normal(arr.size).astype("<f4") * np.float32(1e-3)).reshape(arr.shape)
new = (arr + noise).astype("<f4")
new_raw = new.tobytes()
assert len(new_raw) == len(raw)
out = raw_file[:off] + new_raw + raw_file[off + len(raw):]
assert len(out) == len(raw_file)
dst.write_bytes(out)

# The perturbed file must still parse and carry the new weights.
m2 = onnx.load(str(dst))
t2 = next(x for x in m2.graph.initializer if x.name == t.name)
assert np.array_equal(numpy_helper.to_array(t2), new)

meta = {
    "tensor_name": t.name,
    "node_op": node.op_type,
    "dims": "x".join(map(str, t.dims)),
    "stored_as": stored,
    "byte_offset": off,
    "byte_len": len(raw),
    "file_len": len(raw_file),
    "perturbed_path": str(dst),
    "noise": "1e-3 * standard_normal, numpy default_rng(12345)",
}
with open(outdir / "head_meta.txt", "w") as f:
    for k, v in meta.items():
        f.write(f"{k}={v}\n")
print(meta)
