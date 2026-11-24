"""Save whole-chromosome CDS probabilities of trained runs, for post-processing (HMM decoding) without a GPU.

Each run scans the validation and test chromosomes once with its own inference settings (RC averaging, edge
cropping) and writes runs/<run>/probs/<chrom>.npy (float16). Chromosomes already saved are skipped.

    python experiments/save_probs.py --runs runs/v2_res_32k_s0 runs/base_dilated_1.5M_s0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cdsseg.data.dataset import Genome
from cdsseg.models import build
from cdsseg.predict import scan_chromosome

CHROMS = ["chr22", "chr20", "chr21"]  # validation first: the HMM is tuned there


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--chroms", default=",".join(CHROMS))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    chroms = args.chroms.split(",")
    genome = Genome("data/processed", chroms)
    for r in args.runs:
        run = Path(r)
        cfg = json.loads((run / "config.json").read_text())
        model = build(cfg["model"], **cfg["model_kwargs"]).to(args.device)
        model.load_state_dict(torch.load(run / "best.pt", map_location=args.device))
        out = run / "probs"
        out.mkdir(exist_ok=True)
        for c in chroms:
            f = out / f"{c}.npy"
            if f.exists():
                continue
            print(f"{run.name}: scanning {c}", flush=True)
            p = scan_chromosome(model, genome, c, window=cfg["window"], stride=cfg["window"] // 2,
                                batch_size=cfg.get("infer_batch", 512), device=args.device,
                                rc=cfg.get("rc_average", False), crop=cfg.get("crop", 0))
            np.save(f, p.astype(np.float16))


if __name__ == "__main__":
    main()
