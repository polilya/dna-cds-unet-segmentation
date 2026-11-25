"""Figures for E8 and P1: test MCC per seed of each architecture (docs/baselines_mcc.png) and each network with
and without HMM decoding (docs/hmm_decoding.png). Reads runs/*/results.json and runs/*/hmm.json.

    python experiments/plot_baselines_hmm.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
RUNS, OUT = ROOT / "runs", ROOT / "docs"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def res(run):
    return json.loads((RUNS / run / "results.json").read_text())


# --- Figure 1: test MCC per seed -------------------------------------------------------------------
models = [  # label, params, runs
    ("UNet2 k11", "1.54M", [f"v2_res_k11_s{i}" for i in range(3)]),
    ("UNet2 k11 + dilated refine", "1.63M", ["v2_hybrid_k11_s0"]),
    ("CNN + BiLSTM (Helixer-style)", "1.54M", ["base_bilstm_1.5M_s0"]),
    ("UNet2 main (reference)", "5.57M", [f"v2_residual_s{i}" for i in range(3)]),
    ("DilatedResNet, equal compute", "0.15M", ["base_dilated_w24_s0"]),
    ("DilatedResNet (SpliceAI-style)", "1.53M", [f"base_dilated_1.5M_s{i}" for i in range(3)]),
]
fig, ax = plt.subplots(figsize=(8.6, 3.9))
for y, (label, params, runs) in enumerate(models):
    r = [res(x) for x in runs]
    mcc = np.array([x["test_mcc"] for x in r])
    mins = np.mean([x["train_seconds"] for x in r]) / 60
    ax.scatter(mcc, [y] * len(mcc), s=46, color=BLUE, alpha=0.55, edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.plot([mcc.mean()] * 2, [y - 0.28, y + 0.28], color=INK, lw=2, zorder=4)
    ax.text(mcc.mean(), y + 0.36, f"{mcc.mean():.3f}" + (f" ± {mcc.std(ddof=1):.3f}" if len(mcc) > 1 else " (1 seed)"),
            ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.text(0.835, y, f"{params}   {mins:.0f} min", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(models)), [m[0] for m in models], color=INK)
ax.set_xlim(0.62, 0.83)
ax.set_ylim(-0.6, len(models) - 0.1)
ax.text(0.835, len(models) - 0.35, "params   training", fontsize=8.5, color=INK2, weight="bold")
ax.set_xlabel("Test MCC per base (chr20 + chr21)   ·   dots = seeds, bar = mean")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.subplots_adjust(right=0.80)
fig.text(0.01, 0.97, "DilatedResNet beats UNet2 at equal parameters and at equal compute", va="top", color=INK,
         fontsize=11.5, weight="bold")
fig.savefig(OUT / "baselines_mcc.png", dpi=200)

# --- Figure 2: networks ± HMM, small multiples -------------------------------------------------------
nets = [("UNet2 k11", "v2_res_k11_s0"), ("UNet2, 32k steps", "v2_res_32k_s0"),
        ("CNN + BiLSTM", "base_bilstm_1.5M_s0"), ("DilatedResNet", "base_dilated_1.5M_s0")]
steps = [("raw", "network only", BLUE, "o"), ("smooth_by_mcc", "+ smooth HMM", AQUA, "s"),
         ("gene_by_mcc", "+ gene HMM", ORANGE, "D")]
pts = {run: [json.loads((RUNS / run / "hmm.json").read_text())[k]["test"] for k, *_ in steps] for _, run in nets}
fig, axes = plt.subplots(1, 4, figsize=(11, 3.6), sharex=True, sharey=True)
for ax, (label, run) in zip(axes, nets):
    for other in pts:  # every network's points in faint grey for context
        if other != run:
            ax.scatter([t["mcc"] for t in pts[other]], [t["seg_f1_tol10"] for t in pts[other]], s=14, color=GRID,
                       zorder=1)
    p = pts[run]
    for a, b in ((0, 1), (0, 2)):
        ax.annotate("", (p[b]["mcc"], p[b]["seg_f1_tol10"]), (p[a]["mcc"], p[a]["seg_f1_tol10"]),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1, shrinkA=6, shrinkB=6), zorder=2)
    for t, (_, name, c, m) in zip(p, steps):
        ax.scatter(t["mcc"], t["seg_f1_tol10"], s=70, color=c, marker=m, edgecolor=SURF, linewidth=2, zorder=3)
        ax.annotate(f"{t['mcc']:.3f} / {t['seg_f1_tol10']:.3f}", (t["mcc"], t["seg_f1_tol10"]), xytext=(7, -3),
                    textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_title(label, loc="left", color=INK, fontsize=10, weight="bold")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Test MCC per base")
axes[0].set_ylabel("Segment F1 (both ends ±10 bp)")
axes[0].set_xlim(0.655, 0.81)
axes[0].set_ylim(0.08, 0.68)
handles = [plt.Line2D([], [], marker=m, color=c, ls="", markersize=8, label=n) for _, n, c, m in steps]
fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.005, 0.935))
fig.suptitle("HMM decoding lifts segment F1 for every network (test chr20 + chr21, seed 0; HMM tuned on chr22 by MCC)", x=0.01, ha="left",
             color=INK, fontsize=11.5, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.87))
fig.savefig(OUT / "hmm_decoding.png", dpi=200)
print("saved")
