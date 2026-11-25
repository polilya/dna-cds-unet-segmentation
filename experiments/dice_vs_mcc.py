"""The 2023 draft's per-window Dice variants against base-level MCC, on the current models.

Predictors: two trivial ones (empty, all CDS), the 2023 model trained with the 2023 protocol (E0) and with the
honest protocol (E1), and the four current networks (soft probabilities) with their HMM decodings (binary masks,
settings chosen on chr22 by MCC, as in hmm_eval.py). Test chromosomes 20 + 21 are cut into non-overlapping 1-kb
windows (more than 10 % N dropped) and scored on four compositions:

  CDS windows only   every window with at least one CDS base
  windows 1:1        those plus as many CDS-free windows (the 2023 draft's balanced test)
  windows 1:10       ... ten times as many
  natural            all windows (about 1 : 14)

Every metric of the draft (see draft_dice_study.py) is computed per window and averaged, and base-level MCC is
computed on the pooled bases of the same windows. The reference is base-level MCC at the natural ratio: for
each metric and composition, Kendall's tau of its ranking of the predictors against the reference, and the rank
of the empty predictor.

With --binary, the networks' probabilities are thresholded (at each run's validation threshold) before the Dice
variants are computed, so every predictor is a 0/1 mask; by default they are scored on soft probabilities, as in
the 2023 draft. Writes runs/metric_study/dice_vs_mcc[_binary].csv and docs/dice_vs_mcc[_binary].png.

    python experiments/dice_vs_mcc.py [--binary]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau

from cdsseg import hmm, metrics
from cdsseg.data.dataset import Genome
from cdsseg.models import build
from cdsseg.predict import chromosome_target, scan_chromosome
from experiments.draft_dice_study import variants

TEST, W = ["chr20", "chr21"], 1000
NETS = [("UNet2 k11", "v2_res_k11_s0"), ("UNet2 32k", "v2_res_32k_s0"), ("BiLSTM", "base_bilstm_1.5M_s0"),
        ("Dilated", "base_dilated_1.5M_s0")]
OLD = [("2023 model, 2023 protocol", "e0ckpt_s0"), ("2023 model, honest protocol", "e1_unet2023_bce_s0")]
SETS = ["CDS windows only", "windows 1:1", "windows 1:10", "natural"]


def old_probs(genome, run):
    """Whole-chromosome probabilities of a 2023-layout run (plain scan, as in the 2023 setting), cached."""
    out = Path("runs") / run / "probs"
    out.mkdir(exist_ok=True)
    model = None
    probs = {}
    for c in TEST:
        f = out / f"{c}.npy"
        if not f.exists():
            if model is None:
                model = build("unet").to("mps")
                model.load_state_dict(torch.load(Path("runs") / run / "best.pt", map_location="mps"))
            np.save(f, scan_chromosome(model, genome, c).astype(np.float16))
        probs[c] = np.load(f).astype(np.float32)
    return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", action="store_true")
    args = ap.parse_args()
    tag = "_binary" if args.binary else ""
    genome = Genome("data/processed", TEST)
    preds, thr = {}, {}
    for label, run in OLD:
        preds[label] = old_probs(genome, run)
        thr[label] = json.loads((Path("runs") / run / "results.json").read_text()).get("threshold", 0.5)
    for label, run in NETS:
        r = Path("runs") / run
        p = {c: np.load(r / "probs" / f"{c}.npy").astype(np.float32) for c in TEST}
        preds[label] = p
        thr[label] = json.loads((r / "results.json").read_text())["threshold"]
        cfg = json.loads((r / "hmm.json").read_text())
        for model in ("smooth", "gene"):
            c_ = cfg[f"{model}_by_mcc"]
            preds[f"{label} + {model} HMM"] = {
                c: hmm.decode(p[c], np.asarray(genome.seq[c]), model=model, alpha=c_["alpha"],
                              beta=c_["beta"]).astype(np.float32) for c in TEST}
            thr[f"{label} + {model} HMM"] = 0.5
            print(f"decoded {label} {model}", flush=True)

    P, T = {k: [] for k in preds}, []
    for c in TEST:
        n = genome.length[c] // W * W
        keep = (np.asarray(genome.seq[c][:n]).reshape(-1, W) == 4).mean(1) <= 0.1
        T.append(chromosome_target(genome, c)[:n].reshape(-1, W)[keep].astype(np.float32))
        for k in preds:
            P[k].append(preds[k][c][:n].reshape(-1, W)[keep])
    T = np.concatenate(T)
    P = {k: np.concatenate(v) for k, v in P.items()}
    P["empty"], P["all CDS"] = np.zeros_like(T), np.ones_like(T)
    thr["empty"] = thr["all CDS"] = 0.5
    has = T.sum(1) > 0
    pos, neg = np.flatnonzero(has), np.flatnonzero(~has)
    rng = np.random.default_rng(0)
    idx = {"CDS windows only": pos, "windows 1:1": np.concatenate([pos, rng.choice(neg, len(pos), replace=False)]),
           "windows 1:10": np.concatenate([pos, rng.choice(neg, min(len(neg), 10 * len(pos)), replace=False)]),
           "natural": np.arange(len(T))}
    print(f"{len(T)} windows, {len(pos)} with CDS (1 : {len(neg) / len(pos):.1f})")

    rows = []
    for s, ix in idx.items():
        t = T[ix]
        for k, p in P.items():
            q = p[ix]
            row = {"set": s, "predictor": k,
                   "base MCC": metrics.base_metrics(q.ravel(), t.ravel(), thr[k])["mcc"]}
            d = (q >= thr[k]).astype(np.float32) if args.binary else q
            row.update({m: float(v.mean()) for m, v in variants(d, t).items()})
            rows.append(row)
    df = pd.DataFrame(rows)
    Path("runs/metric_study").mkdir(exist_ok=True)
    df.to_csv(f"runs/metric_study/dice_vs_mcc{tag}.csv", index=False)

    ref = df[df["set"] == "natural"].set_index("predictor")["base MCC"]
    mets = [c for c in df.columns if c not in ("set", "predictor")]
    agree, empty_rank = pd.DataFrame(index=mets, columns=SETS, dtype=float), pd.DataFrame(index=mets, columns=SETS)
    for s in SETS:
        d = df[df["set"] == s].set_index("predictor").loc[ref.index]
        for m in mets:
            agree.loc[m, s] = kendalltau(d[m], ref).statistic
            empty_rank.loc[m, s] = int((d[m] > d.loc["empty", m]).sum() + 1)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\nKendall tau vs base MCC at the natural ratio\n", agree.round(2))
    print(f"\nRank of the empty predictor among {len(ref)} (1 = best)\n", empty_rank)
    nat = df[df["set"] == "natural"].set_index("predictor")[mets].sort_values("base MCC", ascending=False)
    print("\nNatural ratio, all predictors\n", nat.round(3))
    print("\nCDS windows only\n", df[df["set"] == "CDS windows only"].set_index("predictor")[mets].loc[nat.index].round(3))

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    surf, ink, ink2 = "#fcfcfb", "#0b0b0b", "#52514e"
    fig.patch.set_facecolor(surf)
    im = ax.imshow(agree.to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(mets)):
        for j in range(len(SETS)):
            v, r = agree.iloc[i, j], empty_rank.iloc[i, j]
            ax.text(j, i, f"{v:.2f}\nempty #{r}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.6 else ink)
    ax.set_xticks(range(len(SETS)), [s.replace(" ", "\n", 1) for s in SETS], color=ink)
    ax.set_yticks(range(len(mets)), mets, color=ink)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Kendall τ vs base MCC (natural ratio)", color=ink2)
    cb.outline.set_visible(False)
    fig.text(0.01, 0.97, f"Which metric ranks {len(ref)} predictors like base MCC on whole chromosomes?",
             color=ink, fontsize=10.5, weight="bold", va="top")
    inputs = "0/1 masks (networks thresholded)" if args.binary else "soft probabilities, as in 2023"
    fig.text(0.01, 0.925, f"Dice on {inputs}. τ = 1: same ranking; 'empty #k': rank of the never-CDS predictor",
             color=ink2, fontsize=8, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(f"docs/dice_vs_mcc{tag}.png", dpi=180, facecolor=surf)
    print(f"saved docs/dice_vs_mcc{tag}.png")


if __name__ == "__main__":
    main()
