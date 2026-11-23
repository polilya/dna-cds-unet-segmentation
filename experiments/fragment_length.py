"""How much does the model lose when only isolated fragments of the genome are available?

Draft assemblies and metagenomes give contigs or reads, not whole chromosomes. Here chr20 and chr21 are
cut into non-overlapping fragments of length L, every fragment is predicted on its own (no flanking
context, no overlap averaging), and base-level metrics are computed over all fragments together.
The reference is the usual whole-chromosome scan (overlapping windows). The threshold is the one the
run chose on validation, so nothing is tuned on the test set.

A second table bins positions by their distance to the nearest fragment edge, which shows how much
flanking context the model needs.

    python experiments/fragment_length.py --runs runs/best_ch32_bn_s0 runs/e1_unet2023_bce_s0
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
from cdsseg.predict import chromosome_target, n_mask, predict_batch, scan_chromosome

TEST = ["chr20", "chr21"]
LENGTHS = (100, 150, 300, 500, 1000, 2000, 5000, 20000, 100000)
EDGE_BINS = (0, 5, 20, 50, 100, 250, 500, 1000, 10**9)
POSITIONS_PER_BATCH = 2**19


def load(run_dir, device):
    cfg = json.loads((run_dir / "config.json").read_text())
    model = build(cfg["model"], **cfg["model_kwargs"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.eval()
    thr = json.loads((run_dir / "results.json").read_text())["threshold"]
    return model, cfg, thr


@torch.no_grad()
def predict_fragments(model, seq, L, device):
    """Non-overlapping fragments of length L (the tail shorter than L is dropped), each predicted alone."""
    n = len(seq) // L * L
    frags = np.asarray(seq[:n]).reshape(-1, L)
    keep = (frags == 4).mean(1) < 0.1
    prob = np.zeros(frags.shape, dtype=np.float32)
    idx = np.flatnonzero(keep)
    bs = max(1, POSITIONS_PER_BATCH // L)
    for i in range(0, len(idx), bs):
        rows = idx[i:i + bs]
        x = torch.from_numpy(frags[rows]).to(device)
        prob[rows] = predict_batch(model, x)
    return prob.ravel(), np.repeat(keep, L), n


def mcc_at(prob, target, thr):
    pred = prob >= thr
    tp = np.sum(pred & target, dtype=np.float64)
    fp = np.sum(pred & ~target, dtype=np.float64)
    fn = np.sum(~pred & target, dtype=np.float64)
    tn = len(target) - tp - fp - fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den else 0.0


def evaluate_run(run_dir, genome, device):
    model, cfg, thr = load(run_dir, device)
    rows, edge_rows = [], []
    targets = {c: chromosome_target(genome, c) for c in TEST}
    valid = {c: ~n_mask(genome, c) for c in TEST}

    probs = {c: scan_chromosome(model, genome, c, window=cfg["window"], stride=cfg["window"] // 2,
                                device=device) for c in TEST}
    p = np.concatenate([probs[c][valid[c]] for c in TEST])
    t = np.concatenate([targets[c][valid[c]] for c in TEST])
    rows.append({"L": "whole", **metrics.base_metrics(p, t, thr)})

    for L in LENGTHS:
        ps, ts, ds, inside = [], [], [], []
        for c in TEST:
            prob, keep, n = predict_fragments(model, genome.seq[c], L, device)
            m = keep & valid[c][:n]
            ps.append(prob[m])
            ts.append(targets[c][:n][m])
            pos = np.arange(L)
            ds.append(np.tile(np.minimum(pos, L - 1 - pos), n // L)[m])
            # fragments lying entirely inside a CDS: no boundary and no non-coding context at all
            full = keep[::L] & targets[c][:n].reshape(-1, L).all(1)
            inside.append(prob.reshape(-1, L)[full])
        p, t, d = np.concatenate(ps), np.concatenate(ts), np.concatenate(ds)
        inside = np.concatenate(inside)
        rows.append({"L": L, **metrics.base_metrics(p, t, thr), "n_inside_cds": len(inside),
                     "recall_inside_cds": float((inside >= thr).mean()) if len(inside) else np.nan})
        if L >= 2000:
            for lo, hi in zip(EDGE_BINS[:-1], EDGE_BINS[1:]):
                sel = (d >= lo) & (d < hi)
                if sel.sum() > 10**5:
                    edge_rows.append({"L": L, "dist_lo": lo, "dist_hi": hi, "n": int(sel.sum()),
                                      "mcc": mcc_at(p[sel], t[sel], thr)})
        print(f"{run_dir.name}  L={L:>6}  MCC {rows[-1]['mcc']:.3f}  AUPRC {rows[-1]['auprc']:.3f}", flush=True)
    for r in rows + edge_rows:
        r["run"] = run_dir.name
    return rows, edge_rows


def plot(df, edges, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for run, g in df.groupby("run"):
        frag = g[g.L != "whole"]
        line, = a1.plot(frag.L.astype(int), frag.mcc, marker="o", label=run)
        a1.axhline(g[g.L == "whole"].mcc.iloc[0], color=line.get_color(), ls="--", lw=1)
    a1.set_xscale("log")
    a1.set_xlabel("fragment length L, bp (each fragment predicted alone)")
    a1.set_ylabel("test MCC (chr20+21)")
    a1.set_title("Isolated fragments vs whole chromosome (dashed)")
    a1.legend(fontsize=8)
    e = edges[edges.L == edges.L.max()]
    for run, g in e.groupby("run"):
        a2.plot([f"{lo}-{hi}" if hi < 10**9 else f"{lo}+" for lo, hi in zip(g.dist_lo, g.dist_hi)], g.mcc,
                marker="o", label=run)
    a2.set_xlabel("distance to the nearest fragment edge, bp")
    a2.set_ylabel("MCC")
    a2.set_title(f"Edge effect (L = {int(e.L.max())})")
    a2.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default="runs/fragment_length")
    args = ap.parse_args()
    genome = Genome("data/processed", TEST)
    rows, edge_rows = [], []
    for r in args.runs:
        a, b = evaluate_run(Path(r), genome, args.device)
        rows += a
        edge_rows += b
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, edges = pd.DataFrame(rows), pd.DataFrame(edge_rows)
    df.to_csv(out / "by_length.csv", index=False)
    edges.to_csv(out / "by_edge_distance.csv", index=False)
    plot(df, edges, "docs/fragment_length.png")
    print(df[["run", "L", "mcc", "f1", "auprc", "precision", "recall", "n_inside_cds", "recall_inside_cds"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
