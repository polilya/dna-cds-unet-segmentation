"""Evaluation metrics for per-base CDS segmentation.

Base-level metrics treat every base of the evaluated region as one sample at the natural
class ratio. Segment-level metrics follow the gene-prediction convention (Burset & Guigo 1996):
a predicted segment is correct only if both of its boundaries match a true segment.
"""
import numpy as np

N_BINS = 2000


def _confusion(prob, target, thr):
    pred = prob >= thr
    tp = np.count_nonzero(pred & target)
    fp = np.count_nonzero(pred & ~target)
    fn = np.count_nonzero(~pred & target)
    tn = target.size - tp - fp - fn
    return tp, fp, fn, tn


def base_metrics(prob, target, thr=0.5):
    """Precision, recall, F1 and MCC at threshold `thr`; AUPRC and AUROC over all thresholds.

    prob: float array of probabilities; target: bool array, same shape. Curves are computed
    from a 2000-bin histogram of probabilities, which is exact up to the bin width and works
    on whole chromosomes.
    """
    prob = np.asarray(prob, dtype=np.float32).ravel()
    target = np.asarray(target, dtype=bool).ravel()
    tp, fp, fn, tn = _confusion(prob, target, thr)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0

    bins = np.minimum((prob * N_BINS).astype(np.int64), N_BINS - 1)
    pos = np.bincount(bins[target], minlength=N_BINS)[::-1].cumsum()   # TP when thresholding at each bin, high to low
    neg = np.bincount(bins[~target], minlength=N_BINS)[::-1].cumsum()  # FP
    P, Nn = pos[-1], neg[-1]
    rec = pos / P if P else np.zeros_like(pos, dtype=float)
    prec = np.where(pos + neg > 0, pos / np.maximum(pos + neg, 1), 1.0)
    auprc = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    fpr = neg / Nn if Nn else np.zeros_like(neg, dtype=float)
    auroc = float(np.trapezoid(np.concatenate([[0.0], rec]), np.concatenate([[0.0], fpr])))
    return {"precision": precision, "recall": recall, "f1": f1, "mcc": float(mcc),
            "auprc": auprc, "auroc": auroc, "pos_fraction": float(target.mean())}


def best_threshold(prob, target, grid=np.linspace(0.05, 0.95, 19)):
    """Threshold maximising base-level F1 (choose on validation, apply to test)."""
    scores = [base_metrics(prob, target, t)["f1"] for t in grid]
    return float(grid[int(np.argmax(scores))])


def segments(mask):
    """Runs of True in a 1D bool array as (start, end) pairs, end exclusive."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.diff(m.astype(np.int8))
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)], axis=1)


def _matched(a, b, tol):
    """For each segment in a, whether some segment in b has both boundaries within tol."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), dtype=bool)
    order = np.argsort(b[:, 0])
    bs, be = b[order, 0], b[order, 1]
    out = np.zeros(len(a), dtype=bool)
    lo = np.searchsorted(bs, a[:, 0] - tol, side="left")
    hi = np.searchsorted(bs, a[:, 0] + tol, side="right")
    for i in np.flatnonzero(hi > lo):
        out[i] = np.any(np.abs(be[lo[i]:hi[i]] - a[i, 1]) <= tol)
    return out


def segment_metrics(prob, target, thr=0.5, tols=(0, 3, 10)):
    """Segment-level recall and precision with exact and tolerant boundary matching."""
    true_seg = segments(np.asarray(target, dtype=bool))
    pred_seg = segments(np.asarray(prob) >= thr)
    out = {"n_true_segments": int(len(true_seg)), "n_pred_segments": int(len(pred_seg))}
    for t in tols:
        r = _matched(true_seg, pred_seg, t).mean() if len(true_seg) else 0.0
        p = _matched(pred_seg, true_seg, t).mean() if len(pred_seg) else 0.0
        out[f"seg_recall_tol{t}"] = float(r)
        out[f"seg_precision_tol{t}"] = float(p)
        out[f"seg_f1_tol{t}"] = float(2 * p * r / (p + r)) if p + r else 0.0
    return out


def window_dice(prob, target, thr=0.5, smooth=1e-3):
    """Mean per-window Dice over windows that contain CDS (the draft's reporting convention)."""
    pred = (np.asarray(prob) >= thr).astype(np.float32)
    t = np.asarray(target, dtype=np.float32)
    has = t.sum(1) > 0
    inter = (pred * t).sum(1)
    dice = (2 * inter + smooth) / (pred.sum(1) + t.sum(1) + smooth)
    return {"window_dice_cds_windows": float(dice[has].mean()) if has.any() else float("nan"),
            "window_dice_all_windows": float(dice.mean())}
