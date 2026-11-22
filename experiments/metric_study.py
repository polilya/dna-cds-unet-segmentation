"""Which metric can be trusted? Score real models and two reference predictors under each metric.

Models: E0 (2023 protocol) and E1 (honest training), plus two reference predictors that carry
no information: 'empty' (never predicts CDS) and 'all' (predicts CDS everywhere).
Test data: whole chr20 + chr21, tiled into non-overlapping 1000-bp windows. Every window that
contains CDS is kept; CDS-free windows are subsampled to class ratios 1:1, 1:10, 1:60 and the
natural ratio. A trustworthy metric must (1) give the reference predictors a low score at every
ratio and (2) rank E1 above E0.

    python experiments/metric_study.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cdsseg import metrics
from cdsseg.data.dataset import Genome
from cdsseg.models import build
from cdsseg.predict import scan_chromosome

W = 1000
OUT = Path("runs/metric_study")


def window_dice_all(pred, target, smooth):
    """Mean per-window Dice over all windows with smoothing term `smooth` (the draft's k)."""
    inter = (pred * target).sum(1)
    return float(((2 * inter + smooth) / (pred.sum(1) + target.sum(1) + smooth)).mean())


def score(prob, target, thr=0.5):
    pred = (prob >= thr).astype(np.float32)
    t = target.astype(np.float32)
    has = t.sum(1) > 0
    inter = (pred * t).sum(1)
    d_cds = (2 * inter + 1e-3) / (pred.sum(1) + t.sum(1) + 1e-3)
    b = metrics.base_metrics(prob.ravel(), target.ravel(), thr)
    gdice = 2 * inter.sum() / (pred.sum() + t.sum()) if pred.sum() + t.sum() else 0.0
    return {
        "dice_per_window_all (k=0.001)": window_dice_all(pred, t, 1e-3),
        "dice_per_window_all (k=3)": window_dice_all(pred, t, 3.0),
        "dice_per_window_cds_only": float(d_cds[has].mean()),
        "dice_global (= F1)": float(gdice),
        "f1": b["f1"], "mcc": b["mcc"], "auprc": b["auprc"], "auroc": b["auroc"],
        "recall": b["recall"], "precision": b["precision"],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = Genome("data/processed", ["chr20", "chr21"])
    models = {"E1 (honest, BCE)": ("unet", "runs/e1_unet2023_bce_s0/best.pt"),
              "E0 (2023 protocol)": ("unet", "runs/e0ckpt_s0/best.pt")}
    probs = {}
    for name, (arch, ckpt) in models.items():
        m = build(arch).to("mps")
        m.load_state_dict(torch.load(ckpt, map_location="mps"))
        probs[name] = {c: scan_chromosome(m, g, c, window=W, stride=W // 2) for c in g.chroms}

    # non-overlapping windows, drop windows with assembly gaps
    X = {}
    T = []
    for c in g.chroms:
        n = g.length[c] // W * W
        seq = np.asarray(g.seq[c][:n]).reshape(-1, W)
        lab = (np.asarray(g.cds[c][:n]) > 0).reshape(-1, W)
        keep = (seq == 4).mean(1) < 0.1
        T.append(lab[keep])
        for name, p in probs.items():
            X.setdefault(name, []).append(p[c][:n].reshape(-1, W)[keep])
    T = np.concatenate(T)
    X = {k: np.concatenate(v) for k, v in X.items()}
    X["reference: empty"] = np.zeros_like(T, dtype=np.float32)
    X["reference: all CDS"] = np.ones_like(T, dtype=np.float32)

    has = T.any(1)
    pos_idx, neg_idx = np.flatnonzero(has), np.flatnonzero(~has)
    rng = np.random.default_rng(0)
    print(f"windows: {len(T):,}  with CDS: {len(pos_idx):,} ({has.mean():.1%})")
    rows = []
    for label, k in (("1:1", 1), ("1:10", 10), ("1:60", 60), ("natural", None)):
        neg = neg_idx if k is None else rng.choice(neg_idx, size=min(len(neg_idx), k * len(pos_idx)), replace=False)
        idx = np.concatenate([pos_idx, neg])
        for name, P in X.items():
            rows.append({"ratio": label, "model": name, **score(P[idx], T[idx])})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics_by_ratio.csv", index=False)
    pd.set_option("display.width", 250)
    for metric in [c for c in df.columns if c not in ("ratio", "model")]:
        print(f"\n{metric}")
        print(df.pivot(index="model", columns="ratio", values=metric)[["1:1", "1:10", "1:60", "natural"]].round(3))
    (OUT / "summary.json").write_text(json.dumps({"n_windows": int(len(T)), "cds_windows": int(has.sum())}))


if __name__ == "__main__":
    main()
