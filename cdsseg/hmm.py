"""HMM decoding of per-base CDS probabilities into gene-consistent CDS segments (hybrid NN/HMM).

The network's probability p enters as a scaled emission: CDS states score alpha * (logit p - beta) per base,
non-CDS states score 0; only the difference matters for Viterbi. Two state graphs:

  smooth   two states (non-CDS, CDS) with geometric durations: plain segmentation smoothing, no biology
  gene     14 states on both strands: intergenic; exon states that track the codon phase; intron states that
           carry the phase across the intron. Transitions are allowed only at the sequence signals they need:
           + strand  start ATG, stop codon in frame (and no other in-frame stop), donor GT/GC, acceptor AG
           - strand  the same read on the reverse complement (left end TTA/CTA/TCA, right end CAT, CT...AC)

Signal positions were checked on chr1-3 (cds_intervals.tsv, half-open, stop codon inside the CDS): ATG starts
and in-frame stops for 99.9 % of proteins, GT...AG for 99 % of introns, GC donors for most of the rest.
Transition rates come from mean lengths on the training chromosomes (exon 160 bp, intron 6.4 kb, 11.7 exons per
protein), converted to rates per signal occurrence.
"""
import ctypes
import subprocess
from pathlib import Path

import numpy as np

_SRC = Path(__file__).with_name("viterbi.c")
_LIB = Path(__file__).with_name("_viterbi.so")
_lib = None


def _load():
    global _lib
    if _lib is None:
        if not _LIB.exists() or _LIB.stat().st_mtime < _SRC.stat().st_mtime:
            subprocess.run(["cc", "-O3", "-shared", "-fPIC", "-o", str(_LIB), str(_SRC)], check=True)
        _lib = ctypes.CDLL(str(_LIB))
        p = np.ctypeslib.ndpointer
        _lib.viterbi.argtypes = [ctypes.c_int64, ctypes.c_int, p(np.uint8), ctypes.c_int, p(np.int32), p(np.int32),
                                 p(np.float64), p(np.int32), p(np.uint8), p(np.float32), p(np.float64), p(np.uint8)]
        _lib.viterbi.restype = ctypes.c_int
    return _lib


BASES = "ACGT"


def match(seq, patterns, offset):
    """mask[t] = True where seq[t + offset : t + offset + len] equals one of the patterns (False off the ends)."""
    n = len(seq)
    out = np.zeros(n, dtype=bool)
    for pat in patterns:
        k = len(pat)
        ok = np.ones(n - k + 1, dtype=bool)
        for i, ch in enumerate(pat):
            ok &= seq[i:n - k + 1 + i] == BASES.index(ch)
        lo, hi = max(0, -offset), min(n, n - k + 1 - offset)  # t range with 0 <= t + offset <= n - k
        out[lo:hi] |= ok[lo + offset:hi + offset]
    return out


STOPS, RSTOPS = ["TAA", "TAG", "TGA"], ["TTA", "CTA", "TCA"]
MOTIFS = {  # name -> (patterns, offset of the pattern relative to the position entered)
    "atg": (["ATG"], 0), "donor": (["GT", "GC"], 0), "acceptor": (["AG"], -2), "stop3": (STOPS, -2),
    "rstop": (RSTOPS, 0), "cat_before": (["CAT"], -3), "rdonor": (["CT"], 0), "racceptor": (["AC", "GC"], -2),
}


def motif_masks(seq):
    masks = {k: match(seq, pats, off) for k, (pats, off) in MOTIFS.items()}
    masks["notstop3"] = ~masks["stop3"]
    masks["notrstop"] = ~masks["rstop"]
    return masks


def gene_graph(freq, exon=160.0, intron=6361.0, gene_gap=2e5, r_end=0.1):
    """States, CDS flags and edges (src, dst, log p, motif name or None) of the two-strand gene model."""
    d = min(0.5, 1 / exon / freq["donor"])      # leave an exon, per donor site
    r = min(0.5, 1 / intron / freq["acceptor"])  # leave an intron, per acceptor site
    a = 1 / gene_gap / freq["atg"]               # start a gene, per start signal
    IG, E, E2S, I, RE, RI = 0, [1, 2, 3], 4, [5, 6, 7], [8, 9, 10], [11, 12, 13]
    cds = np.zeros(14, dtype=np.uint8)
    cds[E + [E2S] + RE] = 1
    lg = np.log
    edges = [(IG, IG, lg(1 - 2 * a), None), (IG, E[0], lg(a), "atg"), (IG, RE[0], lg(a), "rstop"),
             (E[0], E[1], lg(1 - d), None), (E[1], E[2], lg(1 - d), "notstop3"), (E[1], E2S, lg(1 - d), "stop3"),
             (E[2], E[0], lg(1 - d), None), (E2S, IG, 0.0, None),
             (RE[0], RE[1], lg(1 - d), None), (RE[1], RE[2], lg(1 - d), None),
             (RE[2], RE[0], lg(1 - d) + lg(1 - r_end), "notrstop"), (RE[2], IG, lg(r_end), "cat_before")]
    for j in range(3):
        k = (j + 1) % 3  # an exon ending at codon position j resumes at position k after the intron
        edges += [(E[j], I[k], lg(d), "donor"), (I[k], I[k], lg(1 - r), None), (I[k], E[k], lg(r), "acceptor"),
                  (RE[j], RI[k], lg(d), "rdonor"), (RI[k], RI[k], lg(1 - r), None),
                  (RI[k], RE[k], lg(r), "racceptor")]
    return cds, edges


def smooth_graph(cds_len=160.0, gap=5000.0):
    lg = np.log
    a, b = 1 / gap, 1 / cds_len
    return np.array([0, 1], dtype=np.uint8), [(0, 0, lg(1 - a), None), (0, 1, lg(a), None),
                                               (1, 1, lg(1 - b), None), (1, 0, lg(b), None)]


def decode(prob, seq, model="gene", alpha=1.0, beta=-2.25, **graph_kwargs):
    """Binary CDS path (uint8, one per base) for a whole chromosome."""
    seq = np.asarray(seq)
    if model == "gene":
        masks = motif_masks(seq)
        cds, edges = gene_graph({k: max(m.mean(), 1e-6) for k, m in masks.items()}, **graph_kwargs)
    else:
        masks, (cds, edges) = {}, smooth_graph(**graph_kwargs)
    names = sorted({m for *_, m in edges if m is not None})
    stack = np.ascontiguousarray(np.stack([masks[m] for m in names]).astype(np.uint8)) if names \
        else np.zeros((1, 1), dtype=np.uint8)
    src, dst, lp, mot = (np.array(x) for x in zip(*edges))
    mot = np.array([-1 if m is None else names.index(m) for m in mot], dtype=np.int32)
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-4, 1 - 1e-4)
    em = (alpha * (np.log(p / (1 - p)) - beta)).astype(np.float32)
    init = np.full(len(cds), -np.inf)
    init[0] = 0.0
    path = np.empty(len(p), dtype=np.uint8)
    rc = _load().viterbi(len(p), len(cds), cds, len(edges), src.astype(np.int32), dst.astype(np.int32),
                         lp.astype(np.float64), mot, stack, em, init, path)
    if rc != 0:
        raise MemoryError("viterbi: out of memory")
    return cds[path]
