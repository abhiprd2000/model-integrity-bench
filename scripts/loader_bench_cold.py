#!/usr/bin/env python3
"""COLD-cache variant of loader_bench.py (run as root: sudo .venv/bin/python ...).

Before EVERY timed run (and the tamper detection run): sync + write 3 to
/proc/sys/vm/drop_caches. 5 timed runs per (model, runtime, mode), median.
Verified bytes still come from `mib verify-read`; its sidecar read is cold too.
Original description follows.

Part D: verified loader vs plain read, into real runtimes.

Plain   : open(path).read() -> runtime from bytes.
Verified: `mib verify-read` (signature + leaf file -> root, then every 4 KiB
          block checked before it is emitted) captured from stdout -> runtime
          from bytes. Includes process spawn, sidecar parse, signature check.
Per (model, runtime, mode): 1 untimed warm-up run + N timed runs, each run a
fresh session; reported value = median over the 3 runs. total_ms is the
per-run sum (median of totals, not a sum of medians). Pinned by the caller
(taskset -c 2), runtimes limited to 1 thread.
Everything printed/written comes straight from the timers below.
"""
import csv, hashlib, os, shutil, statistics, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MIB = ROOT / "target" / "release" / "mib"
SEAL = ROOT / ".cache" / "seal"
PUB = SEAL / "pub.key"
OUT = ROOT / "results" / "r_loader_cold.csv"

import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter

MODELS = [
    ("large", "ResNet-50", "onnxruntime", ROOT / "models" / "large.onnx"),
    ("mid", "MobileNetV2", "ai-edge-litert", ROOT / "models" / "mid.tflite"),
]
rows = []


def add(model, runtime, mode, metric, value, unit=""):
    rows.append([model, runtime, mode, metric, value, unit])


def plain_read(path):
    with open(path, "rb") as f:
        return f.read()


def verified_read(path, sidecar):
    p = subprocess.run(
        [str(MIB), "verify-read", str(path), "--pubkey", str(PUB),
         "--sidecar", str(sidecar), "--min-version", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout, p.stderr.decode().strip()


def make_session(runtime, blob):
    """Returns (session_object, run_first_inference_callable)."""
    if runtime == "onnxruntime":
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        sess = ort.InferenceSession(blob, so, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        x = np.zeros(shape, dtype=np.float32)
        return sess, lambda: sess.run(None, {inp.name: x})
    interp = Interpreter(model_content=blob, num_threads=1)
    interp.allocate_tensors()
    d = interp.get_input_details()[0]
    x = np.zeros(d["shape"], dtype=d["dtype"])

    def go():
        interp.set_tensor(d["index"], x)
        interp.invoke()
        return interp.get_tensor(interp.get_output_details()[0]["index"])
    return interp, go


def drop_caches():
    subprocess.run(["sync"], check=True)
    with open("/proc/sys/vm/drop_caches", "w") as f:
        f.write("3")


def one_run(runtime, path, sidecar, mode):
    drop_caches()
    t0 = time.perf_counter()
    if mode == "plain":
        blob = plain_read(path)
    else:
        rc, blob, err = verified_read(path, sidecar)
        assert rc == 0, (rc, err)
    t1 = time.perf_counter()
    sess, go = make_session(runtime, blob)
    t2 = time.perf_counter()
    go()
    t3 = time.perf_counter()
    return [(t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3], blob


add("-", "onnxruntime", "-", "runtime_version", ort.__version__)
try:
    import ai_edge_litert
    add("-", "ai-edge-litert", "-", "runtime_version", getattr(ai_edge_litert, "__version__", "unknown"))
except Exception as e:  # noqa
    add("-", "ai-edge-litert", "-", "runtime_version", f"unknown({e})")

for label, pretty, runtime, path in MODELS:
    sidecar = SEAL / f"{path.name}.mib"
    assert sidecar.exists(), f"seal first: {sidecar}"
    # byte-identity of the two paths (checked once, untimed)
    rc, vb, err = verified_read(path, sidecar)
    assert rc == 0, err
    assert hashlib.sha256(vb).digest() == hashlib.sha256(plain_read(path)).digest()
    add(label, runtime, "verified", "bytes_identical_to_plain", 1)
    del vb

    for mode in ["plain", "verified"]:
        one_run(runtime, path, sidecar, mode)           # warm-up
        runs = [one_run(runtime, path, sidecar, mode)[0] for _ in range(5)]
        for k, metric in enumerate(["read_ms", "session_create_ms", "first_inference_ms", "total_ms"]):
            med = statistics.median(r[k] for r in runs)
            add(label, runtime, mode, metric, f"{med:.3f}", "ms")
            print(f"[{pretty}/{runtime}] {mode:<8} {metric:<19} {med:10.2f} ms   runs={[round(r[k],1) for r in runs]}")

    # ---- 1-byte tampered copy: must be refused before the runtime sees it
    tdir = ROOT / ".cache" / "tamper"
    tdir.mkdir(parents=True, exist_ok=True)
    tpath = tdir / path.name
    blob = bytearray(plain_read(path))
    off = len(blob) // 2 + 12345
    blob[off] ^= 0x01
    tpath.write_bytes(blob)
    drop_caches()
    t0 = time.perf_counter()
    rc, out, err = verified_read(tpath, sidecar)       # tampered file, ORIGINAL sidecar
    detect_ms = (time.perf_counter() - t0) * 1e3
    refused = rc != 0
    runtime_constructed = False
    if not refused:                                    # only now would the runtime see bytes
        make_session(runtime, out)
        runtime_constructed = True
    print(f"[{pretty}] tamper at byte {off}: exit={rc} refused={refused} "
          f"emitted_before_refusal={len(out)} B detect={detect_ms:.1f} ms :: {err}")
    add(label, runtime, "verified", "tamper_offset", off, "byte")
    add(label, runtime, "verified", "tamper_exit_code", rc)
    add(label, runtime, "verified", "tamper_refused", int(refused))
    add(label, runtime, "verified", "tamper_runtime_constructed", int(runtime_constructed))
    add(label, runtime, "verified", "tamper_detect_ms", f"{detect_ms:.3f}", "ms")
    add(label, runtime, "verified", "tamper_verified_bytes_emitted_before_stop", len(out), "B")
    add(label, runtime, "verified", "tamper_stderr", err)
    # control: the SAME tampered bytes through the plain path
    try:
        _, go = make_session(runtime, plain_read(tpath))
        go()
        ctl = "runtime_loaded_and_ran_without_complaint"
    except Exception as e:  # noqa
        ctl = f"runtime_error:{type(e).__name__}"
    print(f"[{pretty}] control (plain read of tampered copy): {ctl}")
    add(label, runtime, "plain", "tamper_control", ctl)
    tpath.unlink()

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "runtime", "mode", "metric", "value", "unit"])
    w.writerows(rows)
if os.environ.get("SUDO_UID"):
    os.chown(OUT, int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"]))
print(f"wrote {OUT}")
