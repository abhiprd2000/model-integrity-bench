#!/usr/bin/env python3
"""F8: update-cost crossover for ResNet-50, random vs contiguous dirty sets,
unbatched vs batched, against whole-file re-verification.

Reads only results/r_workloads.csv (measured) and r_workloads_crossover.csv
(interpolated crossovers written by the benchmark). Same style rules as
plot_results.py: 3.4 in wide, 8 pt serif, markers + linestyles, grayscale-safe,
vector PDF; the model is named ResNet-50, never the CSV label.
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
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 6.5,
    "grid.alpha": 0.3, "axes.grid": True, "axes.grid.which": "both",
    "lines.linewidth": 1.2, "lines.markersize": 3.5, "pdf.fonttype": 42,
})
rd = lambda p: list(csv.DictReader(open(ROOT / "results" / p, newline="")))
rows = [r for r in rd("r_workloads.csv") if r["model"] == "large"]
cross = {(r["pattern"], r["policy"]): float(r["crossover_pct"])
         for r in rd("r_workloads_crossover.csv") if r["model"] == "large"}

series = [  # (pattern, column, policy, label, marker, linestyle, gray)
    ("random", "unbatched_ms", "unbatched", "random, per-block", "o", "-", "0.05"),
    ("contiguous", "unbatched_ms", "unbatched", "contiguous, per-block", "s", "--", "0.35"),
    ("random", "batched_ms", "batched", "random, batched", "^", "-.", "0.15"),
    ("contiguous", "batched_ms", "batched", "contiguous, batched", "v", ":", "0.55"),
]
fig, ax = plt.subplots()
for pat, col, pol, lab, mk, ls, g in series:
    rs = [r for r in rows if r["pattern"] == pat]
    x = [float(r["W_pct"]) for r in rs]
    y = [float(r[col]) for r in rs]
    ax.plot(x, y, marker=mk, linestyle=ls, color=g, label=lab)
    ax.axvline(cross[(pat, pol)], color=g, linestyle=ls, linewidth=0.6, alpha=0.7)
wf = [float(r["wholefile_ms"]) for r in rows if r["pattern"] == "random"]
xs = [float(r["W_pct"]) for r in rows if r["pattern"] == "random"]
ax.plot(xs, wf, color="0.0", linestyle="-", linewidth=2.0, alpha=0.35,
        label="whole-file re-verify")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([1, 5, 10, 25, 50, 100]); ax.set_xticklabels(["1", "5", "10", "25", "50", "100"])
ax.set_xlabel("Dirty blocks (% of file)")
ax.set_ylabel("Update cost (ms)")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "F8.pdf")
print("wrote figures/F8.pdf; crossovers:", cross)
