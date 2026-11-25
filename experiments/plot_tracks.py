"""Genome-browser view of each network's predictions on the test chromosome chr20, at two scales: one region of
~150 kb with several genes (docs/tracks_region.png) and two neighbouring CDS of one gene inside it
(docs/tracks_cds_pair.png). For every network: the CDS probability, the thresholded output and both HMM decodings
(settings chosen by validation MCC, as in hmm_eval.py).

Regions are picked by rule, not by eye: among 150-kb windows with 3-8 proteins, the one whose mean MCC over the
four networks is the median; inside it, the exon pair (intron 200-1500 bp) with the median mean MCC.

    python experiments/plot_tracks.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cdsseg import hmm
from cdsseg.data.dataset import Genome
from cdsseg.predict import chromosome_target

CHROM, WIDTH, PAD = "chr20", 150_000, 100_000
NETS = [("UNet2 k11", "v2_res_k11_s0"), ("UNet2, 32k steps", "v2_res_32k_s0"),
        ("CNN + BiLSTM", "base_bilstm_1.5M_s0"), ("DilatedResNet", "base_dilated_1.5M_s0")]
SURF, INK, INK2, GRID, TRUTH = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#0b0b0b"
DECODINGS = [("network ≥ threshold", "#2a78d6"), ("+ smooth HMM", "#1baf7a"), ("+ gene HMM", "#eb6834")]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def mcc(pred, target):
    tp = float(np.sum(pred & target)); tn = float(np.sum(~pred & ~target))
    fp = float(np.sum(pred & ~target)); fn = float(np.sum(~pred & target))
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0


def segments(mask):
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(m))
    return list(zip(edges[::2], edges[1::2]))


def main():
    genome = Genome("data/processed", [CHROM])
    seq, target = np.asarray(genome.seq[CHROM]), chromosome_target(genome, CHROM).astype(bool)
    n = len(seq)
    runs = {run: Path("runs") / run for _, run in NETS}
    prob = {run: np.load(r / "probs" / f"{CHROM}.npy").astype(np.float32) for run, r in runs.items()}
    thr = {run: json.loads((r / "results.json").read_text())["threshold"] for run, r in runs.items()}
    hmm_cfg = {run: json.loads((r / "hmm.json").read_text()) for run, r in runs.items()}
    cds = pd.read_csv("data/processed/cds_intervals.tsv", sep="\t")
    cds = cds[cds["chrom"] == CHROM]

    # region: median of the 4-network mean MCC among 150-kb windows holding 3-8 proteins
    cands = []
    for s in range(0, n - WIDTH, WIDTH // 3):
        inside = cds[(cds["start"] >= s) & (cds["end"] <= s + WIDTH)]
        k = inside["protein_id"].str.split(".").str[0].nunique()
        if 3 <= k <= 8 and target[s:s + WIDTH].sum() > 2000:
            cands.append((np.mean([mcc(prob[r][s:s + WIDTH] >= thr[r], target[s:s + WIDTH]) for r in prob]), s))
    cands.sort()
    score, r0 = cands[len(cands) // 2]
    r1 = r0 + WIDTH
    print(f"region {CHROM}:{r0}-{r1}, mean MCC {score:.3f} (median of {len(cands)} candidate windows)")

    # decode the region with padding on both sides, then crop
    lo, hi = max(0, r0 - PAD), min(n, r1 + PAD)
    dec = {}
    for run in prob:
        p = prob[run][lo:hi]
        out = [(p >= thr[run])]
        for model in ("smooth", "gene"):
            c = hmm_cfg[run][f"{model}_by_mcc"]
            out.append(hmm.decode(p, seq[lo:hi], model=model, alpha=c["alpha"], beta=c["beta"]).astype(bool))
        dec[run] = [o[r0 - lo:r1 - lo] for o in out]
    tgt = target[r0:r1]

    # exon pair: consecutive CDS of one protein, intron 200-1500 bp, median mean MCC over the pair view
    pairs = []
    inside = cds[(cds["start"] >= r0) & (cds["end"] <= r1)].sort_values(["protein_id", "start"])
    for _, g in inside.groupby("protein_id"):
        a = g[["start", "end"]].to_numpy()
        for (s1, e1), (s2, e2) in zip(a[:-1], a[1:]):
            if 200 <= s2 - e1 <= 1500 and e1 - s1 >= 60 and e2 - s2 >= 60:
                v0, v1 = s1 - 300 - r0, e2 + 300 - r0
                pairs.append((np.mean([mcc(dec[r][0][v0:v1], tgt[v0:v1]) for r in dec]), v0, v1))
    pairs = sorted(set(pairs))
    _, p0, p1 = pairs[len(pairs) // 2]
    print(f"pair view {CHROM}:{r0 + p0}-{r0 + p1} (median of {len(pairs)} exon pairs)")

    for name, (v0, v1) in (("tracks_region", (0, WIDTH)), ("tracks_cds_pair", (p0, p1))):
        plot(name, v0, v1, r0, tgt, prob, dec, thr, lo)


def plot(name, v0, v1, r0, tgt, prob, dec, thr, lo):
    length = v1 - v0
    kb = length > 20_000
    x = (np.arange(v0, v1) + r0) / (1e3 if kb else 1)
    fig, axes = plt.subplots(1 + len(NETS), 1, figsize=(11, 1.0 + 1.55 * len(NETS)), sharex=True,
                             gridspec_kw={"height_ratios": [0.35] + [1] * len(NETS)})
    unit = 1e3 if kb else 1

    def bars(ax, mask, y, h, color):
        segs = segments(mask[v0:v1])
        ax.broken_barh([((s + v0 + r0) / unit, max((e - s) / unit, length / unit / 1500)) for s, e in segs],
                       (y, h), color=color, lw=0)

    ax = axes[0]
    bars(ax, tgt, 0, 1, TRUTH)
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0.5], ["true CDS"], color=INK)
    n_true = len(segments(tgt[v0:v1]))
    for ax, (label, run) in zip(axes[1:], NETS):
        p = prob[run][lo:][r0 - lo:][v0:v1]
        if kb:  # one value per pixel column: the maximum keeps narrow peaks visible
            k = max(1, length // 2000)
            m = len(p) // k * k
            xs, ps = x[:m].reshape(-1, k).mean(1), p[:m].reshape(-1, k).max(1)
        else:
            xs, ps = x, p
        ax.fill_between(xs, 1.55, 1.55 + ps, color=GRID, lw=0)
        ax.plot(xs, 1.55 + ps, color=INK2, lw=0.7)
        ax.axhline(1.55 + thr[run], color=INK2, lw=0.6, ls=(0, (3, 3)))
        for i, (d, (_, color)) in enumerate(zip(dec[run], DECODINGS)):
            bars(ax, np.pad(d, (0, 0)), 1.0 - 0.5 * i, 0.38, color)
            k = len(segments(d[v0:v1]))
            ax.text(1.002, (1.19 - 0.5 * i) / 2.65, f"{k} seg", transform=ax.transAxes, fontsize=7.5, color=INK2,
                    va="center")
        ax.set_ylim(-0.05, 2.6)
        ax.set_yticks([0.19, 0.69, 1.19, 2.05], ["gene HMM", "smooth HMM", "≥ threshold", "probability"])
        ax.set_title(label, loc="left", color=INK, fontsize=9.5, weight="bold", pad=2)
    axes[0].text(1.002, 0.5, f"{n_true} seg", transform=axes[0].transAxes, fontsize=7.5, color=INK2, va="center")
    for ax in axes:
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel(f"{CHROM} position, {'kb' if kb else 'bp'}", color=INK2)
    axes[-1].ticklabel_format(axis="x", style="plain", useOffset=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TRUTH, label="true CDS")] + \
              [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for l, c in DECODINGS] + \
              [plt.Line2D([], [], color=INK2, ls=(0, (3, 3)), lw=0.8, label="threshold (chosen on chr22)")]
    fig.legend(handles=handles, loc="upper left", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.005, 0.955))
    scale = f"{length / 1e3:.0f} kb with several genes" if kb else f"two neighbouring CDS of one gene, {length:,} bp"
    fig.suptitle(f"Predicted CDS on test chromosome {CHROM}: {scale}", x=0.01, ha="left", color=INK, fontsize=11,
                 weight="bold")
    fig.tight_layout(rect=(0, 0, 0.955, 0.93))
    fig.savefig(f"docs/{name}.png", dpi=170)
    print(f"saved docs/{name}.png")


if __name__ == "__main__":
    main()
