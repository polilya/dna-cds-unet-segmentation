"""Windows cut on the fly from the memory-mapped arrays written by prepare.py.

Training batches are sampled at random positions (optionally enriched for windows that
contain a CDS); evaluation walks a whole chromosome with a fixed stride so the class
ratio is the natural one.
"""
from pathlib import Path

import numpy as np

# reverse complement of the base codes A=0 C=1 G=2 T=3 N=4
COMPLEMENT = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
# swapping strands swaps the two label bits: +strand (1) <-> -strand (2)
SWAP_STRAND = np.array([0, 2, 1, 3], dtype=np.uint8)


class Genome:
    """Read-only view of the processed chromosomes."""

    def __init__(self, processed_dir, chroms):
        d = Path(processed_dir)
        self.chroms = [c if c.startswith("chr") else f"chr{c}" for c in chroms]
        self.seq = {c: np.load(d / f"{c}.seq.npy", mmap_mode="r") for c in self.chroms}
        self.cds = {c: np.load(d / f"{c}.cds.npy", mmap_mode="r") for c in self.chroms}
        self.length = {c: len(self.seq[c]) for c in self.chroms}
        self._intervals = None
        self._dir = d

    def intervals(self):
        """CDS intervals (start, end) per chromosome, loaded lazily."""
        if self._intervals is None:
            import pandas as pd
            df = pd.read_csv(self._dir / "cds_intervals.tsv", sep="\t")
            df = df[df["chrom"].isin(self.chroms)]
            self._intervals = {c: g[["start", "end"]].to_numpy() for c, g in df.groupby("chrom")}
        return self._intervals


def reverse_complement(seq, label):
    """Reverse-complement a batch of windows (B, L) and swap the strand bits of the labels."""
    return COMPLEMENT[seq[:, ::-1]], SWAP_STRAND[label[:, ::-1]]


class RandomWindows:
    """Endless random training batches.

    pos_fraction: share of windows centred on a randomly chosen CDS (the CDS lands at a random
    offset inside the window); the rest are uniform over the chromosomes. pos_fraction=None
    samples uniformly, which keeps the natural class ratio.
    """

    def __init__(self, genome, window=1000, batch_size=128, pos_fraction=0.5,
                 max_n_fraction=0.1, rc_augment=True, seed=0):
        self.g, self.window, self.batch_size = genome, window, batch_size
        self.pos_fraction, self.max_n_fraction, self.rc_augment = pos_fraction, max_n_fraction, rc_augment
        self.rng = np.random.default_rng(seed)
        lengths = np.array([genome.length[c] for c in genome.chroms], dtype=float)
        self.chrom_p = lengths / lengths.sum()
        if pos_fraction:
            iv = genome.intervals()
            counts = np.array([len(iv.get(c, [])) for c in genome.chroms], dtype=float)
            self.cds_p = counts / counts.sum()

    def _uniform_start(self, chrom):
        return int(self.rng.integers(0, self.g.length[chrom] - self.window))

    def _cds_start(self, chrom):
        s, e = self.g.intervals()[chrom][self.rng.integers(len(self.g.intervals()[chrom]))]
        # place the CDS start anywhere in the window (long CDS simply run past its end)
        lo, hi = max(0, s - self.window + 1), min(s, self.g.length[chrom] - self.window)
        return int(self.rng.integers(lo, hi + 1))

    def __iter__(self):
        return self

    def __next__(self):
        B, L = self.batch_size, self.window
        seq = np.empty((B, L), dtype=np.uint8)
        lab = np.empty((B, L), dtype=np.uint8)
        i = 0
        while i < B:
            positive = self.pos_fraction and self.rng.random() < self.pos_fraction
            if positive:
                chrom = self.g.chroms[self.rng.choice(len(self.g.chroms), p=self.cds_p)]
                start = self._cds_start(chrom)
            else:
                chrom = self.g.chroms[self.rng.choice(len(self.g.chroms), p=self.chrom_p)]
                start = self._uniform_start(chrom)
            s = self.g.seq[chrom][start:start + L]
            if (s == 4).mean() > self.max_n_fraction:
                continue
            seq[i], lab[i] = s, self.g.cds[chrom][start:start + L]
            i += 1
        if self.rc_augment:
            flip = self.rng.random(B) < 0.5
            if flip.any():
                seq[flip], lab[flip] = reverse_complement(seq[flip], lab[flip])
        return seq, lab


def tiled_windows(genome, chrom, window=1000, stride=500):
    """Start positions that tile a whole chromosome; the last window is aligned to the end."""
    n = genome.length[chrom]
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] != n - window:
        starts.append(n - window)
    return np.array(starts)
