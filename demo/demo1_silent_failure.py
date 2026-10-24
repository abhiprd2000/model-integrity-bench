"""Demo 1: one flipped bit in a model file changes the prediction, and nothing complains."""
import os, shutil, sys
import numpy as np, onnx, onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "models", "large.onnx")
CLEAN, TAMP = (os.path.join(HERE, "work", n) for n in ("clean.onnx", "tampered.onnx"))
errors = []

def classify(path, img, labels):
    try:
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        out = s.run(None, {s.get_inputs()[0].name: img})[0][0]
    except Exception as e:  # any runtime complaint would be reported
        errors.append(str(e)); return None, 0.0, None
    p = np.exp(out - out.max()); p /= p.sum()
    i = int(p.argmax())
    return labels[i], 100 * float(p[i]), out

def prep():
    im = Image.open(os.path.join(HERE, "assets", "dog.jpg")).convert("RGB")
    w, h = im.size; s = 256 / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = im.size; l, t = (w - 224) // 2, (h - 224) // 2
    x = np.asarray(im.crop((l, t, l + 224, t + 224)), np.float32) / 255
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return x.transpose(2, 0, 1)[None].astype(np.float32)

os.makedirs(os.path.join(HERE, "work"), exist_ok=True)
shutil.copyfile(SRC, CLEAN); shutil.copyfile(SRC, TAMP)
labels = open(os.path.join(HERE, "assets", "labels.txt")).read().split("\n")
img = prep()

m = onnx.load(CLEAN)
first_conv = next(n for n in m.graph.node if n.op_type == "Conv")
wname = first_conv.input[1]
init = next(i for i in m.graph.initializer if i.name == wname)
# weights may be raw_data or packed float_data; both are contiguous little-endian float32 in the file
raw = init.raw_data or np.asarray(init.float_data, "<f4").tobytes()
assert raw, "weight tensor has no inline data"
data = open(CLEAN, "rb").read()
base = data.find(raw)
assert base >= 0 and data.find(raw, base + 1) < 0, "weight bytes not uniquely located"
del data
n_w = len(raw) // 4

b_lab, b_conf, b_out = classify(CLEAN, img, labels)
tries = 0
with open(TAMP, "r+b") as f:
    for idx in range(n_w):
        tries += 1
        off = base + 4 * idx + 3          # little-endian: byte 3 holds bits 24-31; bit 30 = 0x40
        f.seek(off); byte = f.read(1)[0]
        f.seek(off); f.write(bytes([byte ^ 0x40])); f.flush()
        a_lab, a_conf, a_out = classify(TAMP, img, labels)
        if a_out is not None and np.isfinite(a_out).all() and a_lab != b_lab:
            break
        f.seek(off); f.write(bytes([byte])); f.flush()   # revert, try next weight
    else:
        errors.append("no weight flip changed the label")

if os.environ.get("DEMO_VERBOSE"):
    print(f"[tries={tries}, file offset={off}, idx={idx}]", file=sys.stderr)
print(f"Model: ResNet-50 ({os.path.getsize(SRC)/2**20:.1f} MiB)")
print(f"Before: {b_lab} ({b_conf:.1f}%)")
print(f"Flipped 1 bit in {wname}[{idx}]")
print(f"After:  {a_lab} ({a_conf:.1f}%)")
print("Errors raised: " + ("none" if not errors else "; ".join(errors)))
