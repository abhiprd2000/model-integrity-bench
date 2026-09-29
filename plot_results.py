#!/usr/bin/env python3
"""Regenerate the paper figures F1-F7 from the measured results in results/.

Every plotted number comes from results/e1..e7_*.csv (direct timings produced by
`cargo run --release --bin experiments`, or exact arithmetic labeled
basis=computed); the break-even points marked in F5 are the floor values written
by the experiment to results/e7_summary.txt. No placeholder or hand-entered
data. Plain csv module; no pandas required.

Style: 3.4 in wide, 8 pt serif text matching an IEEE Times body, axis labels in
words with units, markers + line styles (grayscale safe), light grid, vector PDF
out. The CSV labels tiny/mid/large are never printed: they are the DTLN,
MobileNetV2 and ResNet-50 models (see the model sources listed in
model-integrity-results-x86_64.txt).
"""
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

plt.rcParams.update({
    "figure.figsize": (3.4, 2.5),   # 3.4 in wide, single-column IEEE
    "font.size": 8,
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Liberation Serif", "Times New Roman",
                   "Times", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "grid.alpha": 0.3,              # light grid
    "axes.grid": True,
    "axes.grid.which": "both",
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "pdf.fonttype": 42,             # embed TrueType, keeps text as text
})

# CSV model label -> the model actually measured (paper names, never tiny/mid/large).
NAME = {"tiny": "DTLN", "mid": "MobileNetV2", "large": "ResNet-50"}
MODELS = ["tiny", "mid", "large"]

# Grayscale-safe series styles (marker + linestyle + gray level).
STYLE = {
    "tiny":  dict(marker="o", linestyle="-",  color="0.05"),
    "mid":   dict(marker="s", linestyle="--", color="0.40"),
    "large": dict(marker="^", linestyle=":",  color="0.65"),
}
BLOCK4K = "4096"        # the per-block unit measured in e1/e4 (one 4 KiB block)
TINY_PT = 6.5           # "small text" for the in-plot model-name labels
MICRO = "\u00b5s"       # µs


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)
    print(f"wrote figures/{name}.pdf")


def label_points(ax, pts, dy):
    """Model name beside each (x, y, name) point: small text, edge labels pulled in."""
    ordered = sorted(pts)           # left to right, so the leftmost is first
    for i, (x, y, name) in enumerate(ordered):
        # Leftmost label starts at its point; the rest are centred so that the two
        # right-hand points (13.98 MB and 102.6 MB) cannot run into each other.
        ha = "left" if i == 0 else "center"
        ax.annotate(name, (x, y), xytext=(0, dy), textcoords="offset points",
                    ha=ha, va=("bottom" if dy > 0 else "top"), fontsize=TINY_PT)


def read_break_even(path):
    """Break-even W* per model, as reported in results/e7_summary.txt.

    Each line reads e.g. '[mid] break-even W* = floor(5882.3392 us / 4.3643 us) =
    1347 dirty blocks/cycle = 39.5% of nblocks (3413)'. The figure prints that
    floor() value and its percentage, i.e. the published break-even point, rather
    than a freshly recomputed ratio that rounds to a different integer.
    """
    pat = re.compile(r"\[(\w+)\] break-even W\* =.*?= (\d+) dirty blocks/cycle"
                     r" = ([\d.]+)% of nblocks")
    found = {}
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                found[m.group(1)] = (int(m.group(2)), float(m.group(3)))
    missing = [m for m in MODELS if m not in found]
    if missing:
        raise SystemExit(f"no W* line in {path} for: {missing}")
    return found


# ---- F1: flat per-block verify vs whole-file hash vs model size (log-log) ----
e1 = read_csv(RESULTS / "e1_blocksize.csv")
e4 = read_csv(RESULTS / "e4_writepath.csv")
nblk = {r["model"]: int(r["nblocks"]) for r in e4}
row4k = {m: next(r for r in e1 if r["model"] == m and r["block_size"] == BLOCK4K)
         for m in MODELS}
# Model size = bytes actually covered by the 4 KiB grid (nblocks * 4 KiB) -> MB.
size_mb = {m: nblk[m] * int(BLOCK4K) / 1e6 for m in MODELS}
perblock = sorted((size_mb[m], float(row4k[m]["verify_us"]), NAME[m]) for m in MODELS)
wholefile = sorted((size_mb[m], float(row4k[m]["wholefile_us"]), NAME[m]) for m in MODELS)

fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
ax.plot([p[0] for p in perblock], [p[1] for p in perblock], color="0.05",
        marker="o", linestyle="-", label="Per-block verify (one 4 KiB block)")
ax.plot([p[0] for p in wholefile], [p[1] for p in wholefile], color="0.35",
        marker="s", linestyle="--", markerfacecolor="none", label="Whole-file hash")
ax.set_xscale("log")
ax.set_yscale("log")
xs = [p[0] for p in perblock + wholefile]
ys = [p[1] for p in perblock + wholefile]
ax.set_xlim(min(xs) / 2.2, max(xs) * 2.2)      # headroom for the point labels
ax.set_ylim(min(ys) / 2.2, max(ys) * 8.0)      # headroom for the point labels
ax.set_xlabel("Model size (MB, log scale)")
ax.set_ylabel(f"Time ({MICRO}, log scale)")
label_points(ax, perblock, 7)                  # just above the flat per-block line
label_points(ax, wholefile, 8)                 # over the whole-file line
# Legend above the axes: inside there is no band that is free for two rows
# without touching the whole-file line or its point labels.
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False,
          borderaxespad=0.0)
save(fig, "F1")

# ---- F2: block size vs write-update time and tree bytes % (two panels) ----
e2 = read_csv(RESULTS / "e2_metadata.csv")
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(3.4, 4.2), sharex=True, constrained_layout=True)
for m in MODELS:
    rs = sorted((r for r in e1 if r["model"] == m), key=lambda r: int(r["block_size"]))
    ax1.plot([int(r["block_size"]) for r in rs],
             [float(r["write_update_us"]) for r in rs], label=NAME[m], **STYLE[m])
    rs2 = sorted((r for r in e2 if r["model"] == m),
                 key=lambda r: int(r["block_size"]))
    ax2.plot([int(r["block_size"]) for r in rs2],
             [float(r["tree_pct_of_model"]) for r in rs2], **STYLE[m])
ax1.set_xscale("log", base=2)
ax1.set_yscale("log")
ax1.set_ylabel(f"Write-update time ({MICRO}, log scale)")
ax2.set_yscale("log")
ax2.set_ylabel("Tree bytes (% of model, log scale)")
ax2.set_xlabel("Block size (bytes, log scale)")
ax1.set_title("(a)", loc="left")   # tags above the axes: they cannot touch a data point
ax2.set_title("(b)", loc="left")
ax1.legend(loc="upper left")       # upper-left of (a) is empty; curves climb to the right
save(fig, "F2")

# ---- F3: BLAKE3 vs SHA-256, grouped bars, 3 operations, log y ----
e3 = read_csv(RESULTS / "e3_blake3_vs_sha256.csv")
ops = ["whole_file", "one_block", "write_update"]
op_labels = ["Whole file", "One block (4 KiB)", "Write update"]
model = "large"                     # ResNet-50: largest model, widest hash margin
b3 = [float(next(r["time_us"] for r in e3 if r["model"] == model
                 and r["operation"] == op and r["algorithm"] == "BLAKE3")) for op in ops]
sha = [float(next(r["time_us"] for r in e3 if r["model"] == model
                  and r["operation"] == op and r["algorithm"] == "SHA-256")) for op in ops]
fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
x = range(len(ops))
w = 0.38
ax.bar([i - w / 2 for i in x], b3, w, label="BLAKE3", color="0.15")
ax.bar([i + w / 2 for i in x], sha, w, label="SHA-256",
       color="1.0", edgecolor="0.15", linewidth=0.8)
ax.set_yscale("log")
ax.set_xticks(list(x), op_labels)
ax.set_ylabel(f"Time ({MICRO}, log scale)")
ax.legend(loc="upper right")
save(fig, "F3")

# ---- F4: write-path breakdown, stacked bars (leaf rehash + path update) ----
leaf = [float(r["leaf_rehash_us"]) for r in e4]
full = [float(r["leaf_plus_path_us"]) for r in e4]
path = [f - l for f, l in zip(full, leaf)]
fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
xb = range(len(e4))
ax.bar(xb, leaf, 0.5, label="Leaf rehash", color="0.2")
ax.bar(xb, path, 0.5, bottom=leaf, label="Merkle path update",
       color="0.75", edgecolor="0.2", linewidth=0.5)
ax.set_xticks(list(xb), [f"{NAME[r['model']]}\n(depth {r['tree_depth']})" for r in e4])
ax.set_ylabel(f"Write-update time ({MICRO})")
# Legend above the axes: outside the bars, so it cannot touch any bar.
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
          frameon=False, borderaxespad=0.0)
save(fig, "F4")

# ---- F5: E7 crossover, cost vs W_pct, both policies, W* marked ----
xover = read_csv(RESULTS / "e7_crossover.csv")
w_star = read_break_even(RESULTS / "e7_summary.txt")   # floor value + %, per model
fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.4), sharex=True,
                         constrained_layout=True)
for ax, m in zip(axes, ["mid", "large"]):
    rs = sorted((r for r in xover if r["model"] == m), key=lambda r: float(r["W_pct"]))
    wp = [float(r["W_pct"]) for r in rs]
    ax.plot(wp, [float(r["perblock_ms"]) for r in rs], color="0.05", marker="o",
            linestyle="-", label="Per-block policy")
    # Measured whole-file cost of this model, flat across the full x range.
    # wholefile_ms arrives from the CSV as text: convert it, then draw it (the
    # raw string was silently not rendered before, so the line was missing).
    ax.axhline(float(rs[0]["wholefile_ms"]), linestyle="--", color="0.4",
               label="Whole-file policy")
    blocks, pct = w_star[m]
    ax.axvline(pct, linestyle=":", color="0.1")       # dotted line at W* stays
    ax.annotate(f"W* = {blocks:,} ({pct:.1f}%)", xy=(pct, 0.88),
                xycoords=("data", "axes fraction"), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=TINY_PT)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Time (ms, log scale)")
    # Model name as a panel tag in the upper left, clear of W*, the curves and
    # the legend (which sits in the empty lower right, below the crossover).
    ax.text(0.02, 0.97, NAME[m], transform=ax.transAxes, ha="left", va="top")
    ax.legend(loc="lower right")
axes[-1].set_xlabel("Dirty blocks per cycle W (% of blocks, log scale)")
save(fig, "F5")

# ---- F6: E7 predicted vs measured with y = x line ----
e7 = read_csv(RESULTS / "e7_predicted_vs_measured.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
lo, hi = float("inf"), 0.0
for m in MODELS:
    rs = [r for r in e7 if r["model"] == m]
    px = [float(r["predicted_ms"]) for r in rs]
    py = [float(r["measured_ms"]) for r in rs]
    lo = min(lo, min(px + py))
    hi = max(hi, max(px + py))
    ax.plot(px, py, linestyle="none", label=NAME[m],
            **{k: v for k, v in STYLE[m].items() if k != "linestyle"})
lim = (lo * 0.5, hi * 1.5)
ax.plot(lim, lim, linestyle="--", color="0.4", label="y = x")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Predicted time (ms, log scale)")
ax.set_ylabel("Measured time (ms, log scale)")
ax.legend(loc="upper left")        # empty above the diagonal; points lie on it
save(fig, "F6")

# ---- F7: E5 load overhead, grouped bars (warm / cold panels) ----
e5 = read_csv(RESULTS / "e5_load.csv")
modes = ["wholefile_hash", "perblock_verify"]
mode_labels = ["Whole-file hash", "Per-block verify"]
fig, axes = plt.subplots(
    2, 1, figsize=(3.4, 4.2), sharex=True, constrained_layout=True)
for ax, cache, tag in zip(axes, ["warm", "cold"], ["(a) warm cache", "(b) cold cache"]):
    top = 0.0
    for j, (mo, ml) in enumerate(zip(modes, mode_labels)):
        vals = [float(next(r["overhead_pct_vs_plain"] for r in e5
                           if r["model"] == m and r["mode"] == mo and r["cache"] == cache))
                for m in MODELS]
        top = max(top, max(vals))
        ax.bar([i + (j - 0.5) * 0.38 for i in range(len(MODELS))], vals, 0.38,
               label=ml, color=("0.15" if j == 0 else "0.7"),
               edgecolor="0.15", linewidth=0.5)
    ax.set_ylim(0, top * 1.35)     # headroom: tag and legend stay clear of every bar
    ax.set_xticks(range(len(MODELS)), [NAME[m] for m in MODELS])
    ax.set_ylabel("Overhead vs plain read (%)")
    ax.text(0.02, 0.95, tag, transform=ax.transAxes, ha="left", va="top")
    ax.legend(loc="upper right")
axes[-1].set_xlabel("Model")
save(fig, "F7")

print("All figures F1-F7 written to figures/ from results/ measurements only.")

