"""E6 (first pass): an independent window classifier as stage 1, the E1-type U-Net as stage 2.

Stage 1: CNN with global max-pooling, trained to say whether a 1000-bp window contains any CDS
(training chromosomes, 1:1 sampling). Its threshold is set on chr22 for a target window recall.
Stage 2: the trained U-Net (runs/mcc_study_bce_s0). Predictions in windows rejected by stage 1 are
set to 0. Compared on non-overlapping windows of chr20 + chr21 at the natural ratio.

    python experiments/two_stage.py
"""
import json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from cdsseg import metrics
from cdsseg.data.dataset import Genome, RandomWindows
from cdsseg.models import build, conv_block, one_hot
from cdsseg.predict import scan_chromosome
from cdsseg.train import TRAIN

W, DEV = 1000, "mps"
OUT = Path("runs/e6_two_stage"); OUT.mkdir(parents=True, exist_ok=True)


class WindowClassifier(nn.Module):
    def __init__(self, channels=(32, 64, 128), kernel=9, pool=4):
        super().__init__()
        layers, c_prev = [], 4
        for c in channels:
            layers += [conv_block(c_prev, c, kernel, 2, 0, True), nn.MaxPool1d(pool)]
            c_prev = c
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(c_prev, 1)

    def forward(self, codes):
        return self.head(self.body(one_hot(codes)).amax(-1))[:, 0]


def windows(g, model_fn):
    P, T, X = [], [], []
    for c in g.chroms:
        n = g.length[c] // W * W
        s = np.asarray(g.seq[c][:n]).reshape(-1, W)
        keep = (s == 4).mean(1) < 0.1
        X.append(s[keep]); T.append((np.asarray(g.cds[c][:n]) > 0).reshape(-1, W)[keep])
        if model_fn: P.append(model_fn(g, c)[:n].reshape(-1, W)[keep])
    return np.concatenate(X), np.concatenate(T), (np.concatenate(P) if P else None)


@torch.no_grad()
def classify(model, X, bs=1024):
    model.eval()
    return np.concatenate([torch.sigmoid(model(torch.from_numpy(X[i:i + bs]).to(DEV))).cpu().numpy() for i in range(0, len(X), bs)])


torch.manual_seed(0)
cls = WindowClassifier().to(DEV)
opt = torch.optim.AdamW(cls.parameters(), lr=2e-3)
steps = 3000
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.05)
sampler = RandomWindows(Genome("data/processed", TRAIN), window=W, batch_size=128, pos_fraction=0.5, seed=0)
t0 = time.time()
for step in range(1, steps + 1):
    cls.train()
    s, l = next(sampler)
    y = torch.from_numpy((l > 0).any(1)).float().to(DEV)
    loss = F.binary_cross_entropy_with_logits(cls(torch.from_numpy(s).to(DEV)), y)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 500 == 0: print(f"classifier step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)

seg = build("unet").to(DEV); seg.load_state_dict(torch.load("runs/mcc_study_bce_s0/best.pt", map_location=DEV))
segfn = lambda g, c: scan_chromosome(seg, g, c)
Xv, Tv, _ = windows(Genome("data/processed", ["chr22"]), None)
Xt, Tt, Pt = windows(Genome("data/processed", ["chr20", "chr21"]), segfn)
cv, ct = classify(cls, Xv), classify(cls, Xt)
yv, yt = Tv.any(1), Tt.any(1)
wm = metrics.base_metrics(ct, yt)
res = {"classifier_window_auprc": wm["auprc"], "classifier_window_auroc": wm["auroc"]}
one = metrics.base_metrics(Pt.ravel(), Tt.ravel(), 0.6)
res["one_stage"] = {k: one[k] for k in ("mcc", "f1", "precision", "recall")}
print(f"classifier window AUPRC {wm['auprc']:.3f} AUROC {wm['auroc']:.3f}")
print(f"one-stage            MCC {one['mcc']:.3f} F1 {one['f1']:.3f} P {one['precision']:.3f} R {one['recall']:.3f}")
for target in (0.90, 0.95, 0.99):
    thr = np.quantile(cv[yv], 1 - target)
    passed = ct >= thr
    g = metrics.base_metrics((Pt * passed[:, None]).ravel(), Tt.ravel(), 0.6)
    res[f"gate_recall_{target}"] = {"pass_fraction": float(passed.mean()), "window_recall": float(passed[yt].mean()),
                                    "cds_bases_lost": float(Tt[~passed].sum() / Tt.sum()),
                                    **{k: g[k] for k in ("mcc", "f1", "precision", "recall")}}
    print(f"two-stage, gate recall {target}: passes {passed.mean():5.1%}, CDS bases lost {res[f'gate_recall_{target}']['cds_bases_lost']:.1%} | "
          f"MCC {g['mcc']:.3f} F1 {g['f1']:.3f} P {g['precision']:.3f} R {g['recall']:.3f}")
x = torch.randint(0, 5, (1024, W), dtype=torch.uint8, device=DEV)
for name, net in (("segmenter", seg), ("classifier", cls)):
    with torch.no_grad():
        net.eval(); net(x); torch.mps.synchronize(); t = time.time()
        for _ in range(10): net(x)
        torch.mps.synchronize()
    res[f"{name}_windows_per_s"] = 10 * 1024 / (time.time() - t)
    print(f"{name} inference {res[f'{name}_windows_per_s']:,.0f} windows/s")
(OUT / "results.json").write_text(json.dumps(res, indent=2))
