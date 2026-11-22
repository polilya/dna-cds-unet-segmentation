"""Whole-chromosome inference: overlapping windows, probabilities averaged where windows overlap."""
import numpy as np
import torch

from cdsseg.data.dataset import tiled_windows


@torch.no_grad()
def scan_chromosome(model, genome, chrom, window=1000, stride=500, batch_size=512, device="mps",
                    amp=False):
    model.eval()
    seq = genome.seq[chrom]
    n = genome.length[chrom]
    acc = np.zeros(n, dtype=np.float32)
    cnt = np.zeros(n, dtype=np.float32)
    starts = tiled_windows(genome, chrom, window, stride)
    for i in range(0, len(starts), batch_size):
        st = starts[i:i + batch_size]
        x = torch.from_numpy(np.stack([seq[s:s + window] for s in st])).to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=amp):
            p = torch.sigmoid(model(x).float()).cpu().numpy()
        for s, row in zip(st, p):
            acc[s:s + window] += row
            cnt[s:s + window] += 1
    return acc / np.maximum(cnt, 1)


def chromosome_target(genome, chrom):
    """True CDS mask on either strand."""
    return np.asarray(genome.cds[chrom]) > 0


def n_mask(genome, chrom):
    """Positions inside assembly gaps (N), excluded from evaluation."""
    return np.asarray(genome.seq[chrom]) == 4
