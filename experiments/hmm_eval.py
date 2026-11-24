"""Network + HMM: decode saved probabilities (experiments/save_probs.py) with cdsseg.hmm and score them.

For each run, alpha (emission weight) and beta (emission bias) of each HMM are chosen on the validation chromosome
(chr22), once by base MCC and once by segment F1 (both boundaries within 10 bp); the chosen settings are then
applied unchanged to the test chromosomes. The raw network output (thresholded at the run's validation threshold)
is scored the same way. Results go to runs/<run>/hmm.json.

    python experiments/hmm_eval.py --runs runs/v2_res_32k_s0 runs/base_dilated_1.5M_s0
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from cdsseg import hmm, metrics
from cdsseg.data.dataset import Genome
from cdsseg.predict import chromosome_target, n_mask

VAL, TEST = ["chr22"], ["chr20", "chr21"]
ALPHAS = (0.125, 0.25, 0.5, 1.0, 2.0)
BETAS = (-2.25, -1.5, -0.75, 0.0, 0.75)


def score(pred, target):
    b = metrics.base_metrics(pred.astype(np.float32), target, 0.5)
    s = metrics.segment_metrics(pred, target, 0.5)
    return {k: float(v) for k, v in {**b, **s}.items() if k not in ("auprc", "auroc")}


def scored(genome, chroms, run, decoder):
    """Concatenate decoded predictions over chromosomes (N positions dropped) and score them."""
    preds, targets = [], []
    for c in chroms:
        prob = np.load(run / "probs" / f"{c}.npy").astype(np.float32)
        keep = ~n_mask(genome, c)
        preds.append(decoder(prob, np.asarray(genome.seq[c]))[keep])
        targets.append(chromosome_target(genome, c)[keep])
    return score(np.concatenate(preds), np.concatenate(targets))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()
    genome = Genome("data/processed", VAL + TEST)
    for r in args.runs:
        run = Path(r)
        if not all((run / "probs" / f"{c}.npy").exists() for c in VAL + TEST):
            print(f"{run.name}: probabilities missing, run save_probs.py first")
            continue
        thr = json.loads((run / "results.json").read_text())["threshold"]
        raw = lambda p, s: (p >= thr).astype(np.uint8)  # noqa: E731
        out = {"raw": {"val": scored(genome, VAL, run, raw), "test": scored(genome, TEST, run, raw)}}
        for model in ("smooth", "gene"):
            grid = []
            for a, b in itertools.product(ALPHAS, BETAS):
                dec = lambda p, s, a=a, b=b: hmm.decode(p, s, model=model, alpha=a, beta=b)  # noqa: E731
                grid.append({"alpha": a, "beta": b, "val": scored(genome, VAL, run, dec)})
                print(f"{run.name} {model} alpha {a} beta {b}: val MCC {grid[-1]['val']['mcc']:.3f} "
                      f"seg F1 {grid[-1]['val']['seg_f1_tol10']:.3f}", flush=True)
            for crit in ("mcc", "seg_f1_tol10"):
                best = max(grid, key=lambda g: g["val"][crit])
                dec = lambda p, s, a=best["alpha"], b=best["beta"]: hmm.decode(p, s, model=model, alpha=a, beta=b)  # noqa: E731
                out[f"{model}_by_{crit}"] = {**best, "test": scored(genome, TEST, run, dec)}
            out[f"{model}_grid"] = grid
        (run / "hmm.json").write_text(json.dumps(out, indent=1))
        print(f"\n{run.name} (test chr20+21)")
        for k, v in out.items():
            if "grid" in k:
                continue
            t = v["test"]
            print(f"  {k:22s} MCC {t['mcc']:.3f}  P {t['precision']:.3f}  R {t['recall']:.3f}  "
                  f"seg F1 ±10 {t['seg_f1_tol10']:.3f}  ±0 {t['seg_f1_tol0']:.3f}  "
                  f"segments {t['n_pred_segments']}/{t['n_true_segments']}", flush=True)


if __name__ == "__main__":
    main()
