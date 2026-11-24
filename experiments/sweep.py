"""Overnight sweep: learning curve (E2), model size (E3), one-at-a-time hyperparameters (E4), losses (E5).

Every run is a cdsseg.train run with 3 seeds. Finished runs (results.json present) are skipped,
so the sweep can be stopped and resumed. A summary table is written to runs/summary.csv.

    python experiments/sweep.py --groups e2,e3,e4,e5          # everything (~8-10 h on an M2 Max)
    python experiments/sweep.py --groups e2 --dry-run         # list the runs only
"""
import argparse
import json
import re
import subprocess
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
    # best settings from E3/E4 combined, selected by validation MCC; checks whether the gains add up
    bs32 = {"batch_size": 32, "steps": WINDOWS_SEEN // 32, "eval_every": WINDOWS_SEEN // 32 // 16}
    for name, ch, extra in (("best_ch32_bn", (32, 64, 128), {}),
                            ("best_ch32_bn_bs32", (32, 64, 128), bs32),
                            ("best_ch64_bn", (64, 128, 256), {})):
        exps.append(("best", name, {"model_kwargs": {"channels": list(ch), "norm": True}, "select": "mcc", **extra}))
    # follow-ups on the best recipe (wide + BatchNorm + batch 32): a 1-bp convolution before the head, and the
    # 64-wide model, which crashed on MPS at batch 128
    for name, kw in (("arch_fullres", {"channels": [32, 64, 128], "norm": True, "full_res_conv": True}),
                     ("best_ch64_bn_bs32", {"channels": [64, 128, 256], "norm": True})):
        exps.append(("arch", name, {"model_kwargs": kw, "select": "mcc", **bs32}))
    # UNet2: the 2023 layout's weak points fixed one at a time on top of the best recipe; every run is
    # tested with reverse-complement averaging and 250-bp edge cropping and also without (plain_* keys)
    v2 = {"model": "unet2", "select": "mcc", "rc_average": True, "crop": 250, **bs32}
    for name, kw in (("v2_decoder", {"residual": False, "strands": 1}),
                     ("v2_residual", {"residual": True, "strands": 1}),
                     ("v2_full", {"residual": True, "strands": 2})):
        exps.append(("v2", name, {"model_kwargs": {"channels": [32, 64, 128], **kw}, **v2}))
    # re-check on UNet2 (residual, 1 strand) the conclusions first drawn on the 2023 layout:
    # v2b, 3 seeds: width; v2c, run with --seeds 0: learning rate, kernel, training length, amount of data
    res = {"channels": [32, 64, 128], "residual": True, "strands": 1}
    # 40-80-160 (8.7M params) matches the best 2023-layout model (64-wide, 7.7M); 64-wide UNet2 would be 22M
    exps.append(("v2b", "v2_res_ch40", {"model_kwargs": {**res, "channels": [40, 80, 160]}, **v2}))
    for lr in (1e-3, 4e-3):
        exps.append(("v2c", f"v2_res_lr{lr}", {"model_kwargs": res, **v2, "lr": lr}))
    for k in (11, 21):
        exps.append(("v2c", f"v2_res_k{k}", {"model_kwargs": {**res, "kernel": k}, **v2}))
    exps.append(("v2c", "v2_res_32k", {"model_kwargs": res, **v2, "steps": 2 * bs32["steps"]}))
    for k in (1, 4):
        exps.append(("v2c", f"v2_res_chroms{k}", {"model_kwargs": res, **v2, "train": TRAIN[:k]}))
    # night of 2026-11-23: (1) two-head model = v2_residual + a window gate head (+257 params); its one-head twin
    # trained separately is v2_residual itself; (2) smaller UNet2 models, one seed each
    exps.append(("night", "v2_twohead", {"model_kwargs": {**res, "gate": True}, **v2}))
    for name, ch, k in (("v2_res_w16", [16, 32, 64], 41), ("v2_res_w16_k21", [16, 32, 64], 21),
                        ("v2_res_w8", [8, 16, 32], 41)):
        exps.append(("night", name, {"model_kwargs": {**res, "channels": ch, "kernel": k}, **v2, "_seeds": [0]}))
    # the joint gate loss (weight 1) cost the two-head model ~0.08 MCC: a lighter gate loss, and a gate trained on
    # detached features (the segmenter then trains exactly like the one-head twin)
    exps.append(("gate2", "v2_twohead_w0.1", {"model_kwargs": {**res, "gate": True}, **v2, "gate_weight": 0.1,
                                               "_seeds": [0]}))
    exps.append(("gate2", "v2_twohead_detach", {"model_kwargs": {**res, "gate": True, "gate_detach": True}, **v2,
                                                 "_seeds": [0]}))
    # E8, standard architectures at the size of UNet2 (1.5M ~ v2_res_k11, 5.6M = v2_residual), same data, labels,
    # training budget, selection and inference (RC averaging, 250-bp crop). SpliceAI-style dilated ResNet: 12 residual
    # blocks of two kernel-11 convolutions, dilations 1/4/10 as in SpliceAI-2k, receptive field 1201 bp, no
    # down-sampling. Helixer-style CNN + BiLSTM: two kernel-9 convolutions, then a 2-layer bidirectional LSTM.
    # Ordered by priority: the 1.5M pair (~2 h each), then the 5.6M pair (~4-5 h each; full resolution is costly).
    spliceai = [1] * 4 + [4] * 4 + [10] * 4
    base = {"select": "mcc", "rc_average": True, "crop": 250, **bs32}
    for name, model, kw in (("base_dilated_1.5M", "dilated", {"channels": 76, "kernel": 11, "dilations": spliceai}),
                            ("base_bilstm_1.5M", "bilstm", {"channels": 128, "kernel": 9, "hidden": 192}),
                            ("base_bilstm_5.6M", "bilstm", {"channels": 256, "kernel": 9, "hidden": 364}),
                            ("base_dilated_5.6M", "dilated", {"channels": 146, "kernel": 11, "dilations": spliceai})):
        exps.append(("base", name, {"model": model, "model_kwargs": kw, **base, "_seeds": [0],
                                    **({"infer_batch": 128} if model == "bilstm" else {})}))
    return exps


MIN_BATTERY = 15  # percent; below this no new run is started (a run in progress is never interrupted)


def battery_percent():
    """Battery charge in percent on a Mac (pmset), or None where it cannot be read."""
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"(\d+)%", out)
        return int(m.group(1)) if m else None
    except (OSError, subprocess.SubprocessError):
        return None


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
    ap.add_argument("--reverse", action="store_true", help="walk the list from the end (for a second parallel worker)")
    ap.add_argument("--take-deferred", action="store_true",
                    help="also run runs held back with a `deferred` marker (claimed on purpose, not failed)")
    args = ap.parse_args()
    groups = set(args.groups.split(","))
    seeds = [int(s) for s in args.seeds.split(",")]
    todo = []
    for group, name, over in experiments():
        if group not in groups:
            continue
        for s in over.get("_seeds", seeds):  # an experiment may fix its own seeds
            if not (Path("runs") / f"{name}_s{s}" / "results.json").exists():
                todo.append((name, s, over))
    print(f"{len(todo)} runs to do")
    if args.dry_run:
        for name, s, over in todo:
            print(f"  {name}_s{s}  {over}")
        return
    if args.reverse:
        todo = todo[::-1]
    t0 = time.time()
    for i, (name, s, over) in enumerate(todo, 1):
        charge = battery_percent()
        if charge is not None and charge < MIN_BATTERY:
            print(f"battery at {charge}% (< {MIN_BATTERY}%): not starting {name}_s{s}, stopping", flush=True)
            break
        run_dir = Path("runs") / f"{name}_s{s}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.take_deferred and (run_dir / "deferred").exists():  # a run held back earlier: take it over now
            (run_dir / "deferred").unlink()
            (run_dir / "claimed").unlink(missing_ok=True)
        try:  # claim the run so parallel workers never train the same configuration
            (run_dir / "claimed").touch(exist_ok=False)
        except FileExistsError:
            continue
        if (run_dir / "results.json").exists():
            continue
        cfg = default_config()
        cfg.update({k: v for k, v in over.items() if not k.startswith("_")})
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
