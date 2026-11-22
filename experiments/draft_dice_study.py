"""Evaluate the draft's Dice variants as evaluation metrics, with the same sanity test as metric_study.py.

Variants (per window on soft probabilities p and true mask t, then averaged over windows):
  standard      (2 sum(pt) + k) / (sum p + sum t + k), k = 1e-3
  dice_k        same with k = 0.3 and k = 3
  truncated     empty window: 1 if max p <= 0.2 else 0; otherwise standard
  weighted      2TP / (2TP + Wp*FP + Wt*FN), Wp + Wt = 2, from the CDS : non-CDS ratio 1:5
  symmetric     (D(t, p) + w D(1-t, 1-p)) / (1 + w), w = 1/5
  combined      symmetric, with k = 3 and the weights above in both terms

    python experiments/draft_dice_study.py
"""
import numpy as np
import pandas as pd
import torch

from cdsseg.data.dataset import Genome
from cdsseg.models import build
from cdsseg.predict import scan_chromosome

W = 1000


def soft_parts(p, t):
    tp = (p * t).sum(1)
    fp = (p * (1 - t)).sum(1)
    fn = ((1 - p) * t).sum(1)
    return tp, fp, fn


def dice(p, t, k=1e-3, wp=1.0, wt=1.0):
    tp, fp, fn = soft_parts(p, t)
    return (2 * tp + k) / (2 * tp + wp * fp + wt * fn + k)


def variants(p, t):
    empty = t.sum(1) == 0
    trunc = dice(p, t).copy()
    trunc[empty] = (p[empty].max(1) <= 0.2).astype(float)
    sym = lambda k=1e-3, wp=1.0, wt=1.0: (dice(p, t, k, wp, wt) + 0.2 * dice(1 - p, 1 - t, k, wp, wt)) / 1.2
    return {"standard (k=0.001)": dice(p, t), "dice_k (k=0.3)": dice(p, t, 0.3), "dice_k (k=3)": dice(p, t, 3.0),
            "truncated": trunc, "weighted (Wp=1/3, Wt=5/3)": dice(p, t, wp=1 / 3, wt=5 / 3),
            "weighted (Wp=5/3, Wt=1/3)": dice(p, t, wp=5 / 3, wt=1 / 3), "symmetric": sym(),
            "combined (sym + k=3 + weights)": sym(3.0, 1 / 3, 5 / 3)}


def main():
    g = Genome("data/processed", ["chr20", "chr21"])
    probs = {}
    for name, ckpt in (("E1", "runs/e1_unet2023_bce_s0/best.pt"), ("E0", "runs/e0ckpt_s0/best.pt")):
        m = build("unet").to("mps")
        m.load_state_dict(torch.load(ckpt, map_location="mps"))
        probs[name] = {c: scan_chromosome(m, g, c) for c in g.chroms}
    P, T = {k: [] for k in probs}, []
    for c in g.chroms:
        n = g.length[c] // W * W
        keep = (np.asarray(g.seq[c][:n]).reshape(-1, W) == 4).mean(1) < 0.1
        T.append((np.asarray(g.cds[c][:n]) > 0).reshape(-1, W)[keep])
        for k in probs:
            P[k].append(probs[k][c][:n].reshape(-1, W)[keep])
    T = np.concatenate(T).astype(np.float32)
    P = {k: np.concatenate(v) for k, v in P.items()}
    P["empty"], P["all CDS"] = np.zeros_like(T), np.ones_like(T)
    has = T.sum(1) > 0
    rng = np.random.default_rng(0)
    rows = []
    for ratio, k in (("1:1", 1), ("natural (1:14)", None)):
        neg = np.flatnonzero(~has) if k is None else rng.choice(np.flatnonzero(~has), k * has.sum(), replace=False)
        idx = np.concatenate([np.flatnonzero(has), neg])
        for model, p in P.items():
            for name, v in variants(p[idx], T[idx]).items():
                rows.append({"ratio": ratio, "metric": name, "model": model, "score": float(v.mean())})
    df = pd.DataFrame(rows)
    df.to_csv("runs/metric_study/draft_dice_variants.csv", index=False)
    pd.set_option("display.width", 200)
    t = df.pivot_table(index="metric", columns=["ratio", "model"], values="score").round(3)
    print(t[[(r, m) for r in ("1:1", "natural (1:14)") for m in ("E1", "E0", "empty", "all CDS")]])


if __name__ == "__main__":
    main()
