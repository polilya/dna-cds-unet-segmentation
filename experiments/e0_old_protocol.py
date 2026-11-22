"""E0: reproduce the 2023 evaluation protocol, then score the same model the honest way.

2023 protocol: chromosome 1 only; positive windows built around CDS of 9-300 bp placed at a
random offset; the same number of random CDS-free windows; a random 80/10/10 split of windows;
batch Dice loss; Adam; batch 32; per-window Dice on the test windows.

Honest evaluation of the same trained model: scan whole chr20 and chr21 (never seen) at the
natural class ratio with base-level and segment-level metrics.

    python experiments/e0_old_protocol.py --n-pos 6000 --epochs 10 --seed 0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cdsseg import losses, metrics
from cdsseg.data.dataset import Genome
from cdsseg.models import build, n_params
from cdsseg.train import evaluate_test, predict_windows


def build_2023_dataset(genome, n_pos, window, rng):
    seq, lab = genome.seq["chr1"], genome.cds["chr1"]
    n = genome.length["chr1"]
    iv = pd.read_csv(Path(genome._dir) / "cds_intervals.tsv", sep="\t")
    iv = iv[iv.chrom == "chr1"].drop_duplicates(["start", "end", "strand"])
    iv = iv[(iv.end - iv.start >= 9) & (iv.end - iv.start <= 300)]
    pick = iv.sample(n=n_pos, random_state=int(rng.integers(1 << 31)))
    X, Y = [], []
    for s, e in zip(pick.start, pick.end):
        st = int(rng.integers(max(0, e - window), min(s, n - window) + 1))
        X.append(seq[st:st + window]); Y.append(lab[st:st + window] > 0)
    neg = 0
    while neg < n_pos:
        st = int(rng.integers(0, n - window))
        w, y = seq[st:st + window], lab[st:st + window]
        if (w == 4).mean() > 0.1 or y.any():
            continue
        X.append(w); Y.append(np.zeros(window, bool)); neg += 1
    X, Y = np.stack(X), np.stack(Y)
    idx = rng.permutation(len(X))
    a, b = int(0.8 * len(X)), int(0.9 * len(X))
    return (X[idx[:a]], Y[idx[:a]]), (X[idx[a:b]], Y[idx[a:b]]), (X[idx[b:]], Y[idx[b:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/e0_old_protocol")
    args = ap.parse_args()
    out = Path(args.out + f"_s{args.seed}"); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    dev = "mps"

    genome = Genome("data/processed", ["chr1", "chr20", "chr21"])
    (xtr, ytr), (xva, yva), (xte, yte) = build_2023_dataset(genome, args.n_pos, 1000, rng)
    model = build("unet").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = losses.get("dice_batch")
    best, state = -1, None
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(xtr))
        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            loss = loss_fn(model(torch.from_numpy(xtr[b]).to(dev)), torch.from_numpy(ytr[b]).float().to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        vd = metrics.window_dice(predict_windows(model, xva, dev, False), yva)["window_dice_all_windows"]
        print(f"epoch {ep + 1}: loss {loss.item():.3f}  val window Dice {vd:.3f}", flush=True)
        if vd > best:
            best, state = vd, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)

    pte = predict_windows(model, xte, dev, False)
    res = {"protocol": "2023", "n_params": n_params(model), "train_windows": len(xtr), "test_windows": len(xte),
           **{f"old_{k}": v for k, v in metrics.window_dice(pte, yte).items()},
           **{f"old_{k}": v for k, v in metrics.base_metrics(pte, yte).items()}}
    res.update(evaluate_test(model, genome, ["chr20", "chr21"], 0.5, 1000, dev, False))
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
