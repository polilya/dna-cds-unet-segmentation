"""Summary tables and figures for the README and docs/RESULTS.md, read from runs/*/results.json.

    python experiments/report_figures.py      # writes results/*.csv and figures/v2_*.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS, FIG, OUT = Path("runs"), Path("figures"), Path("results")

# (experiment, label) in the order the model was developed
STAGES = [
    ("e0_old_protocol", "2023 protocol"),
    ("e1_unet2023_bce", "2023 model, honest test"),
    ("e3_channels32", "v1: 32-wide"),
    ("best_ch32_bn_bs32", "v1: + BatchNorm, batch 32"),
    ("arch_fullres", "v1: + 1-bp convolution"),
    ("best_ch64_bn_bs32", "v1: 64-wide"),
    ("v2_decoder", "UNet2: new decoder"),
    ("v2_residual", "UNet2: residual blocks"),
    ("v2_full", "UNet2: + per-strand outputs"),
    ("v2_twohead", "UNet2: two-head (gate weight 1)"),
    ("v2_twohead_w0.1", "UNet2: two-head (gate weight 0.1)"),
    ("v2_twohead_detach", "UNet2: two-head (detached gate)"),
    ("v2_res_32k", "UNet2: residual, 32k steps"),
]
V1_SIZE = ["e3_channels4", "e3_channels8", "e1_unet2023_bce", "e3_channels32", "e3_channels64",
           "best_ch32_bn_bs32", "arch_fullres", "best_ch64_bn_bs32"]
V2_SIZE = ["v2_res_w8", "v2_res_w16_k21", "v2_res_w16", "v2_res_k11", "v2_res_k21", "v2_residual"]
CURVES = [("best_ch32_bn_bs32_s0", "v1: 32-wide + BN"), ("arch_fullres_s0", "v1: + 1-bp convolution"),
          ("best_ch64_bn_bs32_s0", "v1: 64-wide"), ("v2_decoder_s0", "UNet2: new decoder"),
          ("v2_residual_s0", "UNet2: residual blocks"), ("v2_res_32k_s0", "UNet2: residual, 32k steps")]
METRICS = ["test_mcc", "test_auprc", "test_f1", "test_seg_f1_tol10", "plain_test_mcc", "nogate_test_mcc",
           "test_gate_kept_fraction", "test_seconds", "nogate_test_seconds", "train_seconds"]


def load():
    rows = []
    for f in sorted(RUNS.glob("*_s[0-9]/results.json")):
        res = json.loads(f.read_text())
        cfg_f = f.parent / "config.json"
        cfg = json.loads(cfg_f.read_text()) if cfg_f.exists() else {}
        exp = f.parent.name.rsplit("_s", 1)[0]
        rows.append({"experiment": exp, "seed": f.parent.name.rsplit("_s", 1)[1],
                     "n_params": cfg.get("n_params", res.get("n_params")),
                     **{m: res.get(m) for m in METRICS}})
    return pd.DataFrame(rows)


def summarise(df):
    g = df.groupby("experiment")
    out = g[["n_params"]].first().join(g.size().rename("seeds"))
    for m in METRICS:
        out[m + "_mean"] = g[m].mean()
        out[m + "_sd"] = g[m].std(ddof=0)
    return out


def fig_progression(s):
    names = [n for n, _ in STAGES if n in s.index]
    labels = [dict(STAGES)[n] + (" *" if s.loc[n, "seeds"] < 3 else "") for n in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(names) + 1))
    ax.barh(y - 0.2, s.loc[names, "test_mcc_mean"], 0.38, xerr=s.loc[names, "test_mcc_sd"], color="#2a78d6",
            label="MCC (per base)")
    ax.barh(y + 0.2, s.loc[names, "test_seg_f1_tol10_mean"].fillna(0), 0.38, color="#1baf7a",
            label="segment F1 (both ends ±10 bp)")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("test, whole chr20 + chr21 (mean ± sd over seeds; * = 1 seed)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "v2_progression.png", dpi=150)


def fig_size(s):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for names, color, marker, label in ((V1_SIZE, "#898781", "o", "2023 layout (v1)"),
                                        (V2_SIZE, "#6250d6", "s", "UNet2, residual blocks")):
        names = [n for n in names if n in s.index]
        ax.scatter(s.loc[names, "n_params"] / 1e6, s.loc[names, "test_mcc_mean"], c=color, marker=marker, s=55,
                   label=label, edgecolors="white", linewidths=1, zorder=3)
    if "v2_res_32k" in s.index:
        ax.scatter([s.loc["v2_res_32k", "n_params"] / 1e6], [s.loc["v2_res_32k", "test_mcc_mean"]], c="#eb6834",
                   marker="^", s=70, label="UNet2, 32k steps", edgecolors="white", linewidths=1, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (millions, log scale)")
    ax.set_ylabel("test MCC")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "v2_size_vs_mcc.png", dpi=150)


def fig_curves():
    fig, ax = plt.subplots(figsize=(7, 4))
    for run, label in CURVES:
        f = RUNS / run / "log.csv"
        if not f.exists():
            continue
        log = pd.read_csv(f)
        bs = json.loads((RUNS / run / "config.json").read_text())["batch_size"]
        ax.plot(log.step * bs / 1e3, log.val_mcc, label=label, lw=1.8)
    ax.set_xlabel("training windows seen (thousands)")
    ax.set_ylabel("validation MCC (chr22)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "v2_learning_curves.png", dpi=150)


def main():
    OUT.mkdir(exist_ok=True)
    df = load()
    s = summarise(df)
    df.to_csv(OUT / "runs.csv", index=False)
    s.round(4).to_csv(OUT / "experiments.csv")
    fig_progression(s)
    fig_size(s)
    fig_curves()
    print(s[["n_params", "seeds", "test_mcc_mean", "test_mcc_sd", "test_seg_f1_tol10_mean"]].round(3).to_string())


if __name__ == "__main__":
    main()
