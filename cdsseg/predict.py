"""Whole-chromosome inference: overlapping windows, probabilities averaged where windows overlap."""
import numpy as np
import torch

from cdsseg.data.dataset import tiled_windows

_COMPLEMENT = torch.tensor([3, 2, 1, 0, 4], dtype=torch.uint8)


def cds_prob(logits):
    """Per-base CDS probability: sigmoid for a merged label (B, L); P(+ or -) for per-strand logits (B, 2, L)."""
    p = torch.sigmoid(logits.float())
    return 1 - (1 - p[:, 0]) * (1 - p[:, 1]) if p.ndim == 3 else p


@torch.no_grad()
def predict_batch(model, x, rc=False, amp=False):
    """CDS probabilities (B, L) as a numpy array for a batch of base codes on the model's device.

    rc=True also predicts the reverse complement, maps it back (flip; per-strand outputs swap
    strands) and averages the two, so the prediction does not depend on strand orientation.
    """
    with torch.autocast(x.device.type, dtype=torch.float16, enabled=amp):
        p = cds_prob(model(x))
        if rc:
            xr = _COMPLEMENT.to(x.device)[x.long()].flip(-1)
            lr = model(xr).flip(-1)
            if lr.ndim == 3:
                lr = lr[:, [1, 0]]
            p = (p + cds_prob(lr)) / 2
    return p.float().cpu().numpy()


@torch.no_grad()
def scan_chromosome(model, genome, chrom, window=1000, stride=500, batch_size=512, device="mps",
                    amp=False, rc=False, crop=0):
    """Tile a chromosome with windows and average overlapping predictions.

    crop > 0 drops `crop` bases at both ends of every window (except at the chromosome ends),
    where the model sees little context; keep stride <= window - 2 * crop for full coverage.
    """
    model.eval()
    seq = genome.seq[chrom]
    n = genome.length[chrom]
    acc = np.zeros(n, dtype=np.float32)
    cnt = np.zeros(n, dtype=np.float32)
    starts = tiled_windows(genome, chrom, window, stride)
    for i in range(0, len(starts), batch_size):
        st = starts[i:i + batch_size]
        x = torch.from_numpy(np.stack([seq[s:s + window] for s in st])).to(device)
        p = predict_batch(model, x, rc=rc, amp=amp)
        for s, row in zip(st, p):
            lo = 0 if s == 0 else crop
            hi = window if s + window == n else window - crop
            acc[s + lo:s + hi] += row[lo:hi]
            cnt[s + lo:s + hi] += 1
    return acc / np.maximum(cnt, 1)


def chromosome_target(genome, chrom):
    """True CDS mask on either strand."""
    return np.asarray(genome.cds[chrom]) > 0


def n_mask(genome, chrom):
    """Positions inside assembly gaps (N), excluded from evaluation."""
    return np.asarray(genome.seq[chrom]) == 4
