"""Build per-chromosome sequence and CDS label arrays from hg38 FASTA and the RefSeq GFF.

Outputs, one set per chromosome in --out:
    chrN.seq.npy   uint8, one code per base: A=0 C=1 G=2 T=3 N=4 (soft-masking ignored)
    chrN.cds.npy   uint8 bit mask per base: bit 0 = CDS on the + strand, bit 1 = CDS on the - strand
plus
    cds_intervals.tsv   unique CDS intervals (0-based, half-open) with strand and protein id
    meta.json           lengths, CDS counts and CDS base fraction per chromosome

Arrays are plain .npy files so training code can open them with np.load(..., mmap_mode="r")
and cut windows on the fly instead of materialising a dataset.

Usage:
    python -m cdsseg.data.prepare --raw data/raw --out data/processed --chroms 1-22
"""
import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

import numpy as np

GFF_NAME = "GCF_000001405.40_GRCh38.p14_genomic.gff.gz"

# A/C/G/T in either case -> 0..3, everything else (N, IUPAC codes) -> 4
CODE = np.full(256, 4, dtype=np.uint8)
for i, base in enumerate("ACGT"):
    CODE[ord(base)] = i
    CODE[ord(base.lower())] = i


def parse_chroms(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += [str(i) for i in range(int(a), int(b) + 1)]
        else:
            out.append(part)
    return out


def read_fasta(path):
    """Return the single sequence in a gzipped FASTA as a uint8 code array."""
    with gzip.open(path, "rb") as f:
        header = f.readline()
        assert header.startswith(b">"), path
        raw = f.read().replace(b"\n", b"")
    return CODE[np.frombuffer(raw, dtype=np.uint8)]


def read_gff(path, chroms, source):
    """Map RefSeq accessions to chromosome names and collect CDS intervals from one annotation source."""
    accession_to_chrom = {}
    cds = defaultdict(set)  # chrom -> {(start0, end, strand, protein_id)}
    wanted = set(chroms)
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            seqid, src, ftype, start, end, _, strand, _, attrs = cols
            if ftype == "region":
                m = re.search(r"(?:^|;)chromosome=([^;]+)", attrs)
                if m and "genome=chromosome" in attrs and seqid.startswith("NC_"):
                    accession_to_chrom[seqid] = m.group(1)
                continue
            if ftype != "CDS" or src != source:
                continue
            chrom = accession_to_chrom.get(seqid)
            if chrom not in wanted:
                continue
            m = re.search(r"(?:^|;)protein_id=([^;]+)", attrs)
            pid = unquote(m.group(1)) if m else "."
            cds[chrom].add((int(start) - 1, int(end), strand, pid))
    return accession_to_chrom, cds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--chroms", default="1-22")
    ap.add_argument("--source", default="BestRefSeq", help="GFF column 2 value to keep (exact match)")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chroms = parse_chroms(args.chroms)

    acc2chrom, cds = read_gff(raw / GFF_NAME, chroms, args.source)
    meta = {"source": args.source, "gff": GFF_NAME, "accessions": {}, "chromosomes": {}}

    rows = []
    for chrom in chroms:
        seq = read_fasta(raw / f"chr{chrom}.fa.gz")
        label = np.zeros(len(seq), dtype=np.uint8)
        intervals = sorted(cds[chrom])
        for start, end, strand, pid in intervals:
            assert end <= len(seq), (chrom, start, end, len(seq))
            label[start:end] |= 1 if strand == "+" else 2
            rows.append(f"chr{chrom}\t{start}\t{end}\t{strand}\t{pid}")
        np.save(out / f"chr{chrom}.seq.npy", seq)
        np.save(out / f"chr{chrom}.cds.npy", label)
        acc = next(a for a, c in acc2chrom.items() if c == chrom)
        meta["accessions"][f"chr{chrom}"] = acc
        meta["chromosomes"][f"chr{chrom}"] = {
            "length": int(len(seq)),
            "n_fraction": round(float((seq == 4).mean()), 4),
            "cds_intervals": len(intervals),
            "unique_cds_coordinates": len({(s, e, st) for s, e, st, _ in intervals}),
            "cds_base_fraction": round(float((label > 0).mean()), 5),
        }
        print(f"chr{chrom}: {len(seq):,} bp, {len(intervals):,} CDS intervals, "
              f"CDS bases {100 * (label > 0).mean():.2f} %")

    (out / "cds_intervals.tsv").write_text("chrom\tstart\tend\tstrand\tprotein_id\n" + "\n".join(rows) + "\n")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
