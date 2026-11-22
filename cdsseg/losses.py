"""Training losses. All take logits and float targets of shape (B, L)."""
import torch
import torch.nn.functional as F


def bce(logits, target, pos_weight=None):
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)


def dice_batch(logits, target, smooth=1e-6):
    """Dice over the whole batch at once, as in the 2023 code (reduce_sum without an axis)."""
    p = torch.sigmoid(logits)
    return 1 - (2 * (p * target).sum() + smooth) / (p.sum() + target.sum() + smooth)


def dice_window(logits, target, smooth=1.0):
    """Mean of per-window soft Dice; `smooth` is the draft's k (empty windows score 1 only if nothing is predicted)."""
    p = torch.sigmoid(logits)
    d = (2 * (p * target).sum(1) + smooth) / (p.sum(1) + target.sum(1) + smooth)
    return 1 - d.mean()


def focal(logits, target, gamma=2.0, alpha=0.25):
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    w = alpha * target + (1 - alpha) * (1 - target)
    return (w * (1 - pt) ** gamma * ce).mean()


def tversky(logits, target, alpha=0.3, beta=0.7, smooth=1.0):
    """Tversky loss (Salehi et al. 2017): alpha weights false positives, beta false negatives."""
    p = torch.sigmoid(logits)
    tp = (p * target).sum(1)
    fp = (p * (1 - target)).sum(1)
    fn = ((1 - p) * target).sum(1)
    return 1 - ((tp + smooth) / (tp + alpha * fp + beta * fn + smooth)).mean()


def bce_dice(logits, target):
    return bce(logits, target) + dice_window(logits, target)


LOSSES = {"bce": bce, "dice_batch": dice_batch, "dice_window": dice_window,
          "focal": focal, "tversky": tversky, "bce_dice": bce_dice}


def get(name, **kwargs):
    fn = LOSSES[name]
    return lambda logits, target: fn(logits, target, **kwargs)
