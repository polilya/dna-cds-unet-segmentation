"""Train a segmentation model with a chromosome hold-out and evaluate it on whole test chromosomes.

    python -m cdsseg.train --name e1_unet2023 --model unet --steps 4000
    python -m cdsseg.train --name k15 --model unet --model-kwargs '{"kernel": 15}' --seed 1

Writes runs/<name>/: config.json, log.csv (training curve), best.pt (best validation metric, --select auprc|mcc|f1),
results.json (validation and test metrics).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from cdsseg import losses, metrics
from cdsseg.data.dataset import Genome, RandomWindows
from cdsseg.models import build, n_params
from cdsseg.predict import chromosome_target, n_mask, scan_chromosome

TEST = ["chr20", "chr21"]
VAL = ["chr22"]
TRAIN = [f"chr{i}" for i in range(1, 20)]


def fixed_windows(genome, chroms, n, window, seed):
    """A fixed sample of uniform windows (natural class ratio) for cheap validation during training."""
    rs = RandomWindows(genome, window=window, batch_size=n, pos_fraction=None, rc_augment=False, seed=seed)
    return next(rs)


@torch.no_grad()
def predict_windows(model, seq, device, amp, batch_size=512):
    model.eval()
    out = []
    for i in range(0, len(seq), batch_size):
        x = torch.from_numpy(seq[i:i + batch_size]).to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=amp):
            out.append(torch.sigmoid(model(x).float()).cpu().numpy())
    return np.concatenate(out)


def evaluate_test(model, genome, chroms, thr, window, device, amp):
    probs, targets = [], []
    for c in chroms:
        keep = ~n_mask(genome, c)
        p = scan_chromosome(model, genome, c, window=window, stride=window // 2, device=device, amp=amp)
        probs.append(p[keep])
        targets.append(chromosome_target(genome, c)[keep])
    prob, target = np.concatenate(probs), np.concatenate(targets)
    res = {f"test_{k}": v for k, v in metrics.base_metrics(prob, target, thr).items()}
    res.update({f"test_{k}": v for k, v in metrics.segment_metrics(prob, target, thr).items()})
    return res


def run(cfg):
    out = Path(cfg["runs"]) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev, amp = cfg["device"], cfg["amp"]

    genome = Genome(cfg["data"], cfg["train"] + cfg["val"] + cfg["test"])
    train_g = Genome(cfg["data"], cfg["train"])
    val_g = Genome(cfg["data"], cfg["val"])
    sampler = RandomWindows(train_g, window=cfg["window"], batch_size=cfg["batch_size"],
                            pos_fraction=cfg["pos_fraction"], seed=cfg["seed"])
    vseq, vlab = fixed_windows(val_g, cfg["val"], cfg["val_windows"], cfg["window"], seed=12345)
    vtarget = vlab > 0

    model = build(cfg["model"], **cfg["model_kwargs"]).to(dev)
    loss_fn = losses.get(cfg["loss"], **cfg["loss_kwargs"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"], pct_start=0.05)
    cfg["n_params"] = n_params(model)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    best, best_step, bad, t0 = -1.0, 0, 0, time.time()
    log = open(out / "log.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["step", "seconds", "train_loss", "val_loss", "val_auprc", "val_f1", "val_mcc"])
    running = []
    for step in range(1, cfg["steps"] + 1):
        model.train()
        s, l = next(sampler)
        x = torch.from_numpy(s).to(dev)
        y = torch.from_numpy(l > 0).float().to(dev)
        with torch.autocast(dev, dtype=torch.float16, enabled=amp):
            logits = model(x)
        loss = loss_fn(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running.append(loss.item())
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            vp = predict_windows(model, vseq, dev, amp)
            vl = float(loss_fn(torch.logit(torch.from_numpy(vp).clamp(1e-6, 1 - 1e-6)),
                               torch.from_numpy(vtarget).float()))
            m = metrics.base_metrics(vp, vtarget)
            w.writerow([step, round(time.time() - t0, 1), np.mean(running), vl, m["auprc"], m["f1"], m["mcc"]])
            log.flush()
            print(f"step {step:6d}  {time.time() - t0:6.0f}s  train {np.mean(running):.4f}  val {vl:.4f}  "
                  f"AUPRC {m['auprc']:.3f}  F1 {m['f1']:.3f}  MCC {m['mcc']:.3f}", flush=True)
            running = []
            if m[cfg["select"]] > best:
                best, best_step, bad = m[cfg["select"]], step, 0
                torch.save(model.state_dict(), out / "best.pt")
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    print(f"early stop at step {step}")
                    break
    log.close()

    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    vp = predict_windows(model, vseq, dev, amp)
    thr = metrics.best_threshold(vp, vtarget)
    res = {"best_step": best_step, "train_seconds": round(time.time() - t0, 1), "threshold": thr,
           **{f"val_{k}": v for k, v in metrics.base_metrics(vp, vtarget, thr).items()}}
    if cfg["test"]:
        res.update(evaluate_test(model, genome, cfg["test"], thr, cfg["window"], dev, amp))
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))
    return res


def default_config():
    return {"name": "run", "runs": "runs", "data": "data/processed", "train": TRAIN, "val": VAL,
            "test": TEST, "model": "unet", "model_kwargs": {}, "loss": "bce", "loss_kwargs": {},
            "window": 1000, "batch_size": 128, "pos_fraction": 0.5, "lr": 2e-3, "weight_decay": 1e-4,
            "steps": 4000, "eval_every": 250, "patience": 6, "select": "auprc", "val_windows": 4096, "seed": 0,
            "device": "mps" if torch.backends.mps.is_available() else "cpu", "amp": False}


def main():
    cfg = default_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in cfg.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        elif isinstance(v, (dict, list)):
            ap.add_argument(flag, type=json.loads, default=v)
        elif v is None:
            ap.add_argument(flag, default=v)
        else:
            ap.add_argument(flag, type=type(v), default=v)
    args = vars(ap.parse_args())
    if args["pos_fraction"] is not None and args["pos_fraction"] <= 0:
        args["pos_fraction"] = None
    run(args)


if __name__ == "__main__":
    main()
