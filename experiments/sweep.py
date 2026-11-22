"""Overnight sweep: learning curve (E2), model size (E3), one-at-a-time hyperparameters (E4), losses (E5).

Every run is a cdsseg.train run with 3 seeds. Finished runs (results.json present) are skipped,
so the sweep can be stopped and resumed. A summary table is written to runs/summary.csv.

    python experiments/sweep.py --groups e2,e3,e4,e5          # everything (~8-10 h on an M2 Max)
    python experiments/sweep.py --groups e2 --dry-run         # list the runs only
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from cdsseg.train import TRAIN, default_config, run

SEEDS = (0, 1, 2)
WINDOWS_SEEN = 4000 * 128  # every run sees the same number of training windows


def experiments():
    """(group, name, overrides) for every configuration; the baseline is E1 (2023 U-Net, BCE)."""
    exps = []
    for k in (1, 2, 4, 8, 19):
        exps.append(("e2", f"e2_chroms{k}", {"train": TRAIN[:k]}))
    for c in ((4, 8, 16), (8, 16, 32), (32, 64, 128), (64, 128, 256)):
        exps.append(("e3", f"e3_channels{c[0]}", {"model_kwargs": {"channels": list(c)}}))
    for k in (5, 9, 15, 21, 31, 61):
        exps.append(("e4", f"e4_kernel{k}", {"model_kwargs": {"kernel": k}}))
    for depth, ch in ((2, (16, 32)), (4, (16, 32, 64, 128))):
        exps.append(("e4", f"e4_depth{depth}", {"model_kwargs": {"channels": list(ch), "pool": 4 if depth == 4 else 5}}))
    for n in (2, 3):
        exps.append(("e4", f"e4_convs{n}", {"model_kwargs": {"convs_per_level": n}}))
    exps.append(("e4", "e4_batchnorm", {"model_kwargs": {"norm": True}}))
    for d in (0.0, 0.1, 0.4):
        exps.append(("e4", f"e4_dropout{d}", {"model_kwargs": {"dropout": d}}))
    for bs in (32, 64, 256, 512):
        exps.append(("e4", f"e4_batch{bs}", {"batch_size": bs, "steps": WINDOWS_SEEN // bs,
                                             "eval_every": max(50, WINDOWS_SEEN // bs // 16)}))
    for lr in (3e-4, 1e-3, 5e-3):
        exps.append(("e4", f"e4_lr{lr}", {"lr": lr}))
    for w in (2000, 4000):
        exps.append(("e4", f"e4_window{w}", {"window": w, "batch_size": 128 * 1000 // w,
                                             "steps": 4000}))
    for pf in (0.2, 0.0):
        exps.append(("e4", f"e4_posfrac{pf}", {"pos_fraction": pf or None}))
    for loss, kw in (("dice_batch", {}), ("dice_window", {"smooth": 1.0}), ("dice_window", {"smooth": 3.0}),
                     ("bce_dice", {}), ("focal", {}), ("tversky", {}), ("bce", {"pos_weight": 5.0})):
        tag = loss + "".join(f"_{k}{v}" for k, v in kw.items())
        exps.append(("e5", f"e5_{tag}", {"loss": loss, "loss_kwargs": kw}))
    return exps


def summarise(runs_dir="runs"):
    rows = []
    for f in Path(runs_dir).glob("*/results.json"):
        if not (f.parent / "config.json").exists():  # e.g. E0, which has its own script
            continue
        cfg = json.loads((f.parent / "config.json").read_text())
        res = json.loads(f.read_text())
        name = f.parent.name
        rows.append({"run": name, "experiment": name.rsplit("_s", 1)[0], "seed": cfg["seed"],
                     "n_params": cfg.get("n_params"), **{k: v for k, v in res.items() if not isinstance(v, dict)}})
    df = pd.DataFrame(rows).sort_values("run")
    df.to_csv(Path(runs_dir) / "summary.csv", index=False)
    cols = ["test_f1", "test_mcc", "test_auprc", "test_seg_f1_tol10", "train_seconds"]
    agg = df.groupby("experiment")[cols].agg(["mean", "std", "count"]).round(4)
    agg.to_csv(Path(runs_dir) / "summary_by_experiment.csv")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="e2,e3,e4,e5")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    groups = set(args.groups.split(","))
    seeds = [int(s) for s in args.seeds.split(",")]
    todo = []
    for group, name, over in experiments():
        if group not in groups:
            continue
        for s in seeds:
            if not (Path("runs") / f"{name}_s{s}" / "results.json").exists():
                todo.append((name, s, over))
    print(f"{len(todo)} runs to do")
    if args.dry_run:
        for name, s, over in todo:
            print(f"  {name}_s{s}  {over}")
        return
    t0 = time.time()
    for i, (name, s, over) in enumerate(todo, 1):
        cfg = default_config()
        cfg.update(over)
        cfg.update({"name": f"{name}_s{s}", "seed": s})
        print(f"\n=== [{i}/{len(todo)}] {cfg['name']}  ({(time.time() - t0) / 3600:.1f} h elapsed)", flush=True)
        try:
            run(cfg)
        except Exception as e:  # one failing configuration must not stop the night
            print(f"FAILED {cfg['name']}: {e!r}", flush=True)
        summarise()
    print(summarise())


if __name__ == "__main__":
    main()
