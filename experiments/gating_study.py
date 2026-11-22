"""How much could a 'find windows with CDS first, segment only those' cascade save, and what does it cost?

Stage 1 is approximated by the E1 model itself: window score = max per-base probability.
Thresholds are set on chr22 (validation) to reach a target window recall; effects are measured
on chr20 + chr21. Also times one-stage inference vs a small classifier.

    python experiments/gating_study.py
"""
import time
import numpy as np
import torch
from cdsseg import metrics
from cdsseg.data.dataset import Genome
from cdsseg.models import build
from cdsseg.predict import scan_chromosome

W = 1000


def windows(g, probs):
    S, T, P = [], [], []
    for c in g.chroms:
        n = g.length[c] // W * W
        keep = (np.asarray(g.seq[c][:n]).reshape(-1, W) == 4).mean(1) < 0.1
        p = probs[c][:n].reshape(-1, W)[keep]
        S.append(p.max(1)); P.append(p); T.append((np.asarray(g.cds[c][:n]) > 0).reshape(-1, W)[keep])
    return np.concatenate(S), np.concatenate(P), np.concatenate(T)


m = build("unet").to("mps"); m.load_state_dict(torch.load("runs/mcc_study_bce_s0/best.pt", map_location="mps"))
val, test = Genome("data/processed", ["chr22"]), Genome("data/processed", ["chr20", "chr21"])
sv, pv, tv = windows(val, {c: scan_chromosome(m, val, c) for c in val.chroms})
st, pt, tt = windows(test, {c: scan_chromosome(m, test, c) for c in test.chroms})
yv, yt = tv.any(1), tt.any(1)
thr_base = 0.6
print(f"test windows {len(yt):,}, with CDS {yt.mean():.1%}")
wm = metrics.base_metrics(st, yt)
print(f"stage-1 proxy window AUPRC {wm['auprc']:.3f}  AUROC {wm['auroc']:.3f}")
base = metrics.base_metrics(pt.ravel(), tt.ravel(), thr_base)
print(f"one-stage:   MCC {base['mcc']:.3f}  precision {base['precision']:.3f}  recall {base['recall']:.3f}  windows segmented 100%")
for target in (0.90, 0.95, 0.99):
    thr = np.quantile(sv[yv], 1 - target)            # window threshold reaching the target recall on chr22
    passed = st >= thr
    gated = pt * passed[:, None]
    g = metrics.base_metrics(gated.ravel(), tt.ravel(), thr_base)
    lost = tt[~passed].sum() / tt.sum()
    print(f"gate recall {target:.2f}: passes {passed.mean():5.1%} of windows, window recall on test {passed[yt].mean():.3f}, "
          f"CDS bases lost {lost:.1%} | MCC {g['mcc']:.3f}  precision {g['precision']:.3f}  recall {g['recall']:.3f}")

# timing: one-stage U-Net vs a small classifier-sized CNN on the same windows
x = torch.randint(0, 5, (1024, W), dtype=torch.uint8, device="mps")
for name, net in (("U-Net 2023 (segmenter)", build("unet")), ("small CNN (classifier-sized)", build("cnn", channels=(16, 32, 64), kernels=(9, 9, 9)))):
    net = net.to("mps").eval()
    with torch.no_grad():
        net(x); torch.mps.synchronize(); t = time.time()
        for _ in range(10): net(x)
        torch.mps.synchronize()
    print(f"{name:30s} {10 * 1024 / (time.time() - t):,.0f} windows/s inference")
