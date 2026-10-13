#!/usr/bin/env python3
"""F1 with ARM added: per-block verify (one 4 KiB block) and whole-file hash vs
model size, for x86 (results/e1_blocksize.csv) and ARM (results_arm_pinned/, phone run pinned to one core).

Same rows as the original F1 (block_size == 4096 of e1_blocksize.csv; size =
nblocks * 4 KiB from e4_writepath.csv) and the same style rules as
plot_results.py: 3.4 in wide, 8 pt serif, markers + linestyles, grayscale safe,
vector PDF, model names (DTLN, MobileNetV2, ResNet-50), never tiny/mid/large.
Only figures/F1.pdf is written. Nothing is typed in; every point is a CSV value.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
plt.rcParams.update({
    "figure.figsize": (3.4, 2.5), "font.size": 8, "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Liberation Serif", "Times New Roman",
                   "Times", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "grid.alpha": 0.3, "axes.grid": True, "axes.grid.which": "both",
    "lines.linewidth": 1.2, "lines.markersize": 3.5, "pdf.fonttype": 42,
})
NAME = {"tiny": "DTLN", "mid": "MobileNetV2", "large": "ResNet-50"}
MODELS = ["tiny", "mid", "large"]
ARM_DIR = "results_arm_pinned"   # pinned run; results_arm/ was unpinned and bimodal
rd = lambda p: list(csv.DictReader(open(ROOT / p, newline="")))


def series(root):
    e1 = rd(f"{root}/e1_blocksize.csv")
    e4 = rd(f"{root}/e4_writepath.csv")
    nblk = {r["model"]: int(r["nblocks"]) for r in e4}
    row = {m: next(r for r in e1 if r["model"] == m and r["block_size"] == "4096") for m in MODELS}
    mb = {m: nblk[m] * 4096 / 1e6 for m in MODELS}
    pb = [(mb[m], float(row[m]["verify_us"])) for m in MODELS]
    wf = [(mb[m], float(row[m]["wholefile_us"])) for m in MODELS]
    return mb, pb, wf


mb, pb_x, wf_x = series("results")
mb_a, pb_a, wf_a = series(ARM_DIR)
assert mb == mb_a, "x86 and ARM must cover the same model bytes"

fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
spec = [  # (points, label, color, marker, linestyle, hollow)
    (pb_x, "x86 per-block verify", "0.05", "o", "-", False),
    (wf_x, "x86 whole-file hash", "0.35", "s", "--", True),
    (pb_a, "ARM per-block verify", "0.20", "^", "-.", False),
    (wf_a, "ARM whole-file hash", "0.55", "D", ":", True),
]
for pts, lab, col, mk, ls, hollow in spec:
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, linestyle=ls,
            markerfacecolor="none" if hollow else col, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
xs = [p[0] for pts, *_ in spec for p in pts]
ys = [p[1] for pts, *_ in spec for p in pts]
ax.set_xlim(min(xs) / 2.2, max(xs) * 2.2)
ax.set_ylim(min(ys) / 2.2, max(ys) * 8.0)
ax.set_xlabel("Model size (MB, log scale)")
ax.set_ylabel("Time (µs, log scale)")
# model names once, just above the highest series at each size (x is shared)
for i, m in enumerate(MODELS):
    top = max(wf_x[i][1], wf_a[i][1])
    ax.annotate(NAME[m], (mb[m], top), xytext=(0, 6), textcoords="offset points",
                ha=("left" if i == 0 else "center"), va="bottom", fontsize=6.5)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False, ncol=2,
          borderaxespad=0.0, columnspacing=0.8, handlelength=2.2, fontsize=7.5)
fig.savefig(ROOT / "figures" / "F1.pdf")
for (pts, lab, *_ ) in spec:
    print(f"{lab:22}", [f"{x:.2f}MB:{y:.2f}us" for x, y in pts])
