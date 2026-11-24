"""How the composition of the test set changes the scores of the one-head and two-head models.

Each model scans whole chr20 + chr21 once (its own inference settings: RC averaging, edge cropping, and for
two-head models the window gate, which is also scored switched off). The predictions are then cut into
non-overlapping 1-kb windows (more than 10 % N dropped) and scored on test sets of different composition:

  bases 1:1        all CDS bases plus an equal random number of non-CDS bases (base-level balance)
  windows 1:k      every window with at least one CDS base plus k times as many CDS-free windows, k = 1, 5, 10
  natural          all windows (about 1 : 14 windows, 1 : 89 bases)

Thresholds are the ones each run chose on validation; nothing is tuned on the test set. Sampled sets are
drawn 5 times; the tables report mean and sd.

    python experiments/ratio_study.py --runs runs/v2_residual_s0 runs/v2_twohead_w0.1_s0
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cdsseg import metrics
from cdsseg.data.dataset import Genome
from cdsseg.models import build
from cdsseg.predict import chromosome_target, scan_chromosome

TEST = ["chr20", "chr21"]
W = 1000
REPEATS = 5
SETS = ["bases 1:1", "windows 1:1", "windows 1:5", "windows 1:10", "natural"]


def load(run_dir, device):
    cfg = json.loads((run_dir / "config.json").read_text())
    res = json.loads((run_dir / "results.json").read_text())
    model = build(cfg["model"], **cfg["model_kwargs"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.eval()
    return model, cfg, res


def windows(genome, probs):
    """Predictions and labels as (n_windows, W) arrays of non-overlapping windows with at most 10 % N."""
    P, T = [], []
    for c in TEST:
        n = genome.length[c] // W * W
        keep = (np.asarray(genome.seq[c][:n]).reshape(-1, W) == 4).mean(1) <= 0.1
        P.append(probs[c][:n].reshape(-1, W)[keep])
        T.append(chromosome_target(genome, c)[:n].reshape(-1, W)[keep])
    return np.concatenate(P), np.concatenate(T)


def scores(p, t, thr):
    m = metrics.base_metrics(p.ravel(), t.ravel(), thr)
    out = {k: m[k] for k in ("mcc", "f1", "precision", "recall", "auprc")}
    if p.ndim == 2:
        out.update(metrics.window_dice(p, t, thr))
    out["cds_base_fraction"] = float(t.mean())
    return out


def evaluate(P, T, thr, rng):
    has = T.any(1)
    pos, neg = np.flatnonzero(has), np.flatnonzero(~has)
    rows = []
    for rep in range(REPEATS):
        p, t = P.ravel(), T.ravel()
        cds = np.flatnonzero(t)
        other = rng.choice(np.flatnonzero(~t), len(cds), replace=False)
        idx = np.concatenate([cds, other])
        rows.append({"set": "bases 1:1", "rep": rep, **scores(p[idx], t[idx], thr)})
        for k in (1, 5, 10):
            sel = np.concatenate([pos, rng.choice(neg, min(len(neg), k * len(pos)), replace=False)])
            rows.append({"set": f"windows 1:{k}", "rep": rep, **scores(P[sel], T[sel], thr)})
    rows.append({"set": "natural", "rep": 0, **scores(P, T, thr)})
    return rows


def plot(df):
    """Gate on and gate off give the same scores, so each two-head model is drawn once (gate on)."""
    df = df[~df.model.str.endswith("(gate off)")]
    mean = df.groupby(["model", "set"], sort=False).mean(numeric_only=True)
    names = {"v2_residual_s0": "one-head (twin)", "v2_twohead_s0 (gate on)": "two-head, gate weight 1",
             "v2_twohead_w0.1_s0 (gate on)": "two-head, gate weight 0.1",
             "v2_twohead_detach_s0 (gate on)": "two-head, detached gate"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for model in df.model.unique():
        m = mean.loc[model].reindex(SETS)
        for ax, key, title in zip(axes, ("mcc", "precision", "window_dice_all_windows"),
                                  ("base-level MCC", "base-level precision (recall is constant per model)",
                                   "per-window Dice (2023 metric)")):
            ax.plot(SETS, m[key], marker="o", label=names.get(model, model))
            ax.set_title(title, fontsize=10)
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig("docs/ratio_study.png", dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default="runs/ratio_study")
    ap.add_argument("--chroms", default=",".join(TEST), help="for a quick check on a smaller test set")
    ap.add_argument("--plot-only", action="store_true", help="redraw docs/ratio_study.png from <out>/all.csv")
    args = ap.parse_args()
    if args.plot_only:
        plot(pd.read_csv(Path(args.out) / "all.csv"))
        return
    TEST[:] = args.chroms.split(",")
    genome = Genome("data/processed", TEST)
    rows = []
    for r in args.runs:
        run = Path(r)
        model, cfg, res = load(run, args.device)
        modes = [("", res["threshold"])]
        if getattr(model, "gate_head", None) is not None:
            modes = [("gate on", res["threshold"]), ("gate off", res["nogate_threshold"])]
        for mode, thr in modes:
            model.gate_threshold = res.get("gate_threshold") if mode == "gate on" else None
            probs = {c: scan_chromosome(model, genome, c, window=cfg["window"], stride=cfg["window"] // 2,
                                        device=args.device, rc=cfg.get("rc_average", False),
                                        crop=cfg.get("crop", 0)) for c in TEST}
            P, T = windows(genome, probs)
            name = run.name + (f" ({mode})" if mode else "")
            for row in evaluate(P, T, thr, np.random.default_rng(0)):
                rows.append({"model": name, **row})
            print(name, "done", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "all.csv", index=False)
    agg = df.groupby(["model", "set"], sort=False).agg(["mean", "std"]).drop(columns="rep").round(4)
    agg.to_csv(out / "summary.csv")
    pd.set_option("display.width", 250)
    print(df.groupby(["model", "set"], sort=False)[["mcc", "f1", "precision", "recall", "window_dice_all_windows",
                                                     "cds_base_fraction"]].mean().round(3).to_string())

    plot(df)


if __name__ == "__main__":
    main()
