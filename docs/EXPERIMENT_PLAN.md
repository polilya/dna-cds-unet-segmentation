# Experiment plan: CDS segmentation with a 1D U-Net, redone properly

Written 2026-11-22. The goal is to turn the 2023 student project into a study whose results a reviewer would accept: a held-out test that cannot leak, standard metrics, honest baselines, and hyperparameters chosen by measurement rather than by hand. Everything must run on one MacBook Pro (M2 Max, 38-core GPU, 32 GB RAM, PyTorch MPS).

## 1. What the 2023 work actually did

- **Task.** Binary per-base segmentation: for every base in a 1,000-bp window, is it part of a CDS. Input is one-hot DNA (1000 × 4).
- **Data.** Human chromosome 1 only, RefSeq `BestRefSeq` CDS; CDS shorter than 9 or longer than 300 bp dropped (7.8 %). Positive windows are built *around* a CDS, negatives are random CDS-free windows; the class ratio is then set by hand (1:1, 1:10, 1:60).
- **Split.** `train_test_split` over windows of the same chromosome (random 80/10/10). Windows from the same gene, overlapping windows and paralogs end up on both sides: the test score is optimistic.
- **Model.** Small Keras encoder–decoder (3 levels, ~65k–350k parameters). The latest notebook (Feb 2025) trains a plain 4-layer CNN, not the U-Net described in the draft.
- **Loss / metric.** Dice computed over the whole batch (`reduce_sum` without an axis), plus hand-made variants (truncated, weighted, symmetric, large smoothing k).
- **Training.** Adam with default settings, batch 32, 10–30 epochs, one run per configuration, hyperparameters varied one at a time for 15 epochs.
- **Results.** Dice 0.53–0.65 on balanced test windows; 0.12–0.22 at 1:10 and 0.02–0.05 at 1:60. The two-stage design (classifier + U-Net), the main hypothesis, was never evaluated end to end.

## 2. How this compares with the field

| Work | Approach | Relevance |
|---|---|---|
| SegmentNT (Dalla-Torre et al., Nature Methods 2025; bioRxiv 2024) | Pretrained Nucleotide Transformer + **1D U-Net head**, per-nucleotide masks for 14 element types incl. exons, inputs up to 30–50 kb; train on all chromosomes except 20, 21 (test) and 22 (validation); metric MCC per element | Closest prior work: same idea of nucleotide-level segmentation with a 1D U-Net. Their ablation reports that a 1D U-Net trained **without** the pretrained encoder performs poorly (MCC ≈ 0.07). |
| Helixer (Stiehler et al. 2020; Holst et al., Nature Methods 2025) | CNN + BiLSTM base-wise classifier (intergenic / UTR / CDS / intron) + HMM post-processing | Standard deep-learning gene annotator; reports genic and subgenic (CDS) F1 |
| Tiberius (Gabriel et al., Bioinformatics 2024) | CNN + LSTM + differentiable HMM, end to end | State of the art for ab initio mammalian gene prediction (gene-level F1 ≈ 62 % on human) |
| SpliceAI (Jaganathan et al. 2019) | Deep dilated residual CNN, 10 kb context | Shows how much context nucleotide-level tasks need |
| AUGUSTUS | Classical HMM gene finder | Classical baseline used in all the papers above |

**Novelty, honestly.** A plain 1D U-Net that segments coding sequence from one-hot DNA is not new as a method, and SegmentNT already shows that without pretraining it is weak. The draft predates the SegmentNT preprint (June 2023 vs March 2024) but was never published, so it carries no priority. What can still be a contribution:

1. A careful, reproducible measurement of how far a **small from-scratch CNN/U-Net** gets on CDS segmentation under a proper chromosome hold-out, with learning curves and a hyperparameter study, on laptop-scale compute.
2. The **metric analysis**: why Dice is misleading for sparse 1D segmentation with empty windows, which metric to use instead, and how the choice of loss changes the result. This is the most original part of the draft, but it needs to be connected to Tversky / generalised Dice / focal loss and backed by experiments.
3. A direct comparison with SegmentNT's exon predictions on the same held-out chromosomes, to quantify the gap a foundation model closes.

**Verdict on the 2023 results.** Not impressive as they stand: the test leaks, the evaluation windows are built around CDS (much easier than scanning a genome), and at realistic class ratios the Dice collapses to 0.02–0.2. They are a reasonable student exploration, not a result. The plan below fixes the evaluation first; only then do the numbers mean anything.

## 3. Evaluation protocol (fixed before any training)

- **Genome and labels.** GRCh38.p14, NCBI RefSeq annotation (`GCF_000001405.40`), source `BestRefSeq`, feature type `CDS`, **both strands**. Label track: 1 per base inside any CDS. Keep all CDS lengths (no 9–300 filter); report long-CDS performance separately.
- **Split by chromosome** (same as SegmentNT so numbers are comparable): test = chr20 + chr21, validation = chr22, train = the remaining autosomes (chrX/Y excluded). Smaller runs train on a subset of training chromosomes but always evaluate on the same test set.
- **Evaluation = scanning the whole test chromosome** with overlapping windows and averaging overlaps, so the class ratio is the natural one (~1–2 % of bases are CDS). No hand-made 1:1 test sets.
- **Metrics** (all per base unless stated):
  - Precision, recall, **F1** and **MCC** for the CDS class (MCC is SegmentNT's headline metric; F1 matches Helixer's subgenic CDS F1).
  - **AUPRC** (threshold-free; the right curve under heavy imbalance) and AUROC for reference.
  - **Exon-level** recall and precision (Burset & Guigó convention): a predicted CDS segment counts as correct only if both boundaries match exactly; also reported with ±3 bp and ±10 bp tolerance.
  - Per-window Dice (mean over windows that contain CDS) only for continuity with the draft.
- **Statistics.** Every reported configuration: 3 seeds, mean ± SD. Model selection on validation only; the test set is scored once per final model.

## 4. Experiments

| # | Question | Design | Output |
|---|---|---|---|
| E0 | Does the new pipeline reproduce the 2023 numbers? | Original data protocol (chr1, windows around CDS, random split) with the original model | Table showing the old protocol inflates scores vs E1 |
| E1 | Honest baseline | Original architecture, chromosome split, whole-chromosome scan, BCE loss | First real F1 / MCC / AUPRC |
| E2 | Is the data enough? (learning curve) | Train on 1, 2, 4, 8, all training chromosomes (fixed steps, early stopping on chr22) | Validation metric vs amount of data. Still rising → data-limited; flat → model-limited |
| E3 | Under- or overfitting? | Train/validation loss and metrics per epoch for E1–E2; model size sweep (channels ×0.25 … ×4) | Diagnosis per model size: gap → overfitting, both low → underfitting |
| E4 | Hyperparameters | One-at-a-time sensitivity around the E1 baseline, then a small Bayesian search (Optuna, ~60 trials, short budget): kernel size 3–41, depth 2–5, base channels 8–64, convs per level 1–3, batch size 32–512, learning rate 1e-4–3e-3, window length 1–8 kb | Sensitivity curves with seed error bars showing where the optimum is; best config re-run with 3 seeds |
| E5 | Which loss? (the draft's metric study, done properly) | Same model trained with BCE, weighted BCE, per-window Dice, batch Dice, Dice + BCE, focal, Tversky, and the draft's symmetric / large-k Dice | Effect of the loss on F1, MCC, AUPRC, exon-level metrics |
| E6 | One stage vs two stages (the draft's hypothesis) | Window classifier + U-Net vs U-Net alone, on the whole-chromosome scan; accuracy and runtime | Answer to the original hypothesis |
| E7 | Context length | Windows of 1, 2, 4, 8 kb (and dilated convolutions as a cheap way to widen the receptive field) | Whether more context helps, as SpliceAI suggests |
| E8 | Baselines | (a) per-base logistic regression on k-mer counts; (b) dilated residual CNN (SpliceAI-style); (c) CNN + BiLSTM (Helixer-style), same data and budget | Is the U-Net better than simple and established architectures? |
| E9 | External reference | SegmentNT (pretrained, PyTorch, Hugging Face) inference on chr20/21 exon class; optional AUGUSTUS ab initio (human model) on the same chromosomes | Where the small model stands against published tools |

E9 compares exons (SegmentNT) with CDS (this work); the difference (UTR parts of exons) is reported explicitly or the comparison is restricted to coding exons.

## 5. Code: from notebooks to a pipeline

```
cdsseg/
├── data/prepare.py     download FASTA + GFF, build per-chromosome uint8 sequence and label arrays (.npy, memory-mapped)
├── data/dataset.py     on-the-fly windows from the memmaps; sampling by chromosome; reverse-complement augmentation
├── models.py           configurable UNet1D, plain CNN, dilated ResNet, CNN-BiLSTM, window classifier
├── losses.py           BCE, focal, Dice (per-window / batch), Tversky, draft variants
├── metrics.py          base-level P/R/F1/MCC, AUPRC/AUROC, exon-level matching, per-window Dice
├── train.py            PyTorch on MPS, YAML config, early stopping, checkpoint, CSV + TensorBoard logs
├── predict.py          whole-chromosome sliding-window inference with overlap averaging
└── experiments/        one YAML per run; sweep runner; results collected into one CSV
```

Why: the notebooks keep the whole dataset in pandas CSVs (up to 340 MB each, strings of bases), mix TensorFlow and PyTorch, and cannot be re-run unattended. Memory-mapped uint8 arrays take 1 byte per base (~3 GB for all autosomes plus labels), windows are cut on the fly, and every run is a command with a config, so sweeps can run overnight.

## 6. Compute and disk budget

- Model size 50k–2M parameters; a 1 kb window costs microseconds on the M2 Max GPU. A typical run (≈200k windows × 10 epochs) is expected to take minutes; E4 (~60 short trials) and E2 (5 data sizes × 3 seeds) should each fit in one night. E9 (SegmentNT, ~500M parameters) is the only heavy job and is optional.
- Disk: ~1 GB downloads (GFF 75 MB, per-chromosome FASTA ~0.9 GB total), ~3–4 GB processed arrays, results and checkpoints < 2 GB. Start heavy steps only after disk space is freed.

## 7. Order of work

1. Environment (conda env with PyTorch MPS, pyfaidx, pandas, scikit-learn, optuna, tensorboard) and data download.
2. `prepare.py`; sanity checks (CDS count per chromosome, fraction of CDS bases, comparison with the 2023 `cds.csv` for chr1).
3. `dataset.py`, `models.py`, `metrics.py`, `train.py`, `predict.py`; unit tests on tiny synthetic data.
4. E0 and E1, then E2–E3.
5. E4 and E5.
6. E6–E8, then E9 if time allows.
7. Rewrite the paper around the results; update the README.
