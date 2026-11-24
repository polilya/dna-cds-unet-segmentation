# Research questions and experiments — status

Version 2 of the CDS segmentation paper. Last updated 2026-11-23.
Test set everywhere: whole chr20 + chr21 (104 Mbp), train chr1–19, validation chr22; base-level
MCC unless stated otherwise; mean ± sd over 3 seeds.

Status: ✅ done · 🔄 running / script ready · ⏳ planned · ❓ needs a decision

## Priorities (set by the user, 2026-11-23)

1. Smaller models: how small can UNet2 get without losing much (widths 8/16/32, kernels 11/21/41).
2. SegmentNT run on our data and labels (same chr20 + chr21 test, CDS derived from its outputs) — needs the 2.2 GB weights download, waiting for approval.
3. Two-head model: a window head first decides "any CDS in this window?", the decoder segments only windows that pass; compared with its one-head twin of the same size trained separately (v2_residual, 257 fewer params).
4. The 2023 layout (v1) is not revisited: UNet2 supersedes it.

5. Later: justify kernel size per level and network depth — first from sources (papers, but also forums and practitioner write-ups: SpliceAI/Basenji/Enformer design notes, U-Net design discussions), then test if needed. Known issue: with a 1-kb window the bottleneck is 8 positions long and level 3 is 40, so a kernel of 41 there mostly sees padding.
6. Rule: parts of a model known to do nothing (e.g. kernel taps that only ever see padding, unused heads) are not carried into new experiments; each new configuration drops them.

## A. Research questions

| # | Question | Origin | What answering it gives | Status | Result |
|---|---|---|---|---|---|
| Q1 | Can a small 1D U-Net segment CDS directly from raw DNA? | 2023 draft | The core claim of the paper | ✅ | Yes, with an honest protocol: MCC 0.425 (2023 architecture), 0.479 (wider U-Net, 1.9M params); AUROC 0.94–0.96 |
| Q2 | Is per-window Dice a valid metric? Do the draft's "complex Dice" variants fix it? | 2023 draft | Decides how every result in the paper is reported | ✅ | No. An empty predictor scores 0.935 and beats trained models; all draft variants (k-smoothing, truncated, weighted, symmetric, combined) still rank the empty predictor first. MCC chosen as the main metric, with AUPRC, F1 and segment metrics |
| Q3 | How imbalanced is the test data? | 2023 draft | Justifies the metric choice; corrects the draft | ✅ | 1.12 % of test bases are CDS (1 : 89), same as train (1 : 83). Window ratio depends on window length: 1 : 47 at 150 bp, 1 : 14 at 1 kb, 1 : 1.2 at 50 kb — the imbalance moves between levels, it does not disappear. The draft's "1 : 60 windows" was a base ratio on chr1 |
| Q4 | How does the test window length affect the result? | 2023 draft | Splits into (a) a sanity check for whole-chromosome scans and (b) a real-world question: how well does the model work on isolated fragments (draft assemblies, metagenomic contigs, reads)? | 🔄 | Script `experiments/fragment_length.py` ready; runs after the best-config training |
| Q5 | Does a two-stage scheme (classify window → segment only positive windows) help? | User hypothesis | Accuracy and compute trade-off; motivation for a two-head model | 🔄 | First pass done (see E6). Two-head version agreed, not built |
| Q6 | Which single question does paper v2 answer? | Open | The paper's framing | 🔄 | Decision (2026-11-23): run the experiments for all three options first, choose the framing from the results. Options: (1) small task-specific model vs genomic foundation model (SegmentNT) — recommended; (2) metric pitfalls in imbalanced genomic segmentation; (3) does gating pay off |

## B. Experiments

| ID | Experiment | What it gives | Status | Result |
|---|---|---|---|---|
| — | Notebooks → pipeline (`cdsseg/`) | Reproducible, testable code | ✅ | Data prep, models, metrics, losses, training, inference, tests |
| E0 | Reproduce the 2023 protocol (chr1, 1 : 1 windows, random split, Dice loss) | Shows how much the old protocol overstated quality | ✅ | Old-style window Dice 0.58 looked good; honest whole-chromosome test MCC 0.274, AUPRC 0.106 |
| E1 | Honest baseline: 2023 U-Net, BCE, chromosome hold-out | Reference point for everything | ✅ | MCC 0.425 ± 0.006, F1 0.428, AUPRC 0.397, AUROC 0.94 |
| M1 | Metric study (Dice variants vs F1 / MCC / AUPRC at 1 : 1, 1 : 10, natural ratio) | Answers Q2 | ✅ | See Q2; figures `docs/metric_study.png`, `docs/mcc_toy_example.png` |
| E2 | Learning curve: 1, 2, 4, 8, 19 training chromosomes | Is data the bottleneck? | ✅ | No: 1 chromosome 0.415, 19 chromosomes 0.425. Holds for this model size and step budget; worth re-checking for the wider model |
| E3 | Model width: channels 4 → 64 (31k → 7.7M params) | Is capacity the bottleneck? | ✅ | Yes: 0.341 → 0.392 → 0.425 → 0.479 → 0.482. Saturates near 2M params |
| E4 | One-at-a-time hyperparameters | Optimal settings | ✅ | BatchNorm +0.047; batch 32 best (512 worst: 0.357); kernel ≥ 31 needed (5 → 0.305, 61 → 0.428); lr 2e-3 optimal; depth, dropout, window 1–4 kb, CDS enrichment 0.2–0.5 matter little; 3 convs/level unstable |
| E5 | Losses: BCE, focal, pos-weighted BCE, Dice variants, Tversky, BCE + Dice | Which loss to train with | ✅ | BCE ≈ focal (0.425). BCE + Dice 0.311. Pure Dice and Tversky collapsed to a constant output (MCC 0) at lr 2e-3 |
| E5b | Soft-MCC loss and model selection by MCC | Train on the metric we report? | ✅ | Selection by MCC helps slightly (0.435); soft-MCC loss worse (0.380, poorly calibrated). Recommendation: train BCE, select by MCC |
| E5c | Re-check Dice losses at lr 1e-3 | Is the Dice collapse a learning-rate artefact? | ⏳ | — |
| B1 | Best settings combined (wide + BatchNorm, + batch 32, 64-wide + BatchNorm), select by MCC | Do the gains add up? Picks the main model | ✅ | **Wide + BatchNorm + batch 32: MCC 0.568 ± 0.008, AUPRC 0.576, segment F1 ±10 bp 0.099** (10 min training). Same without batch 32: 0.453 ± 0.034 (unstable). 64-wide + BatchNorm: all 3 seeds hit an MPS internal error, not finished |
| B2 | 2023 layout + one 1-bp convolution before the head (wide + BatchNorm + batch 32) | Fixes the missing full-resolution layer | ✅ | MCC 0.594 ± 0.009, AUPRC 0.621, segment F1 ±10 bp 0.145 (from 0.099), predicted segments 18k (from 27k); 14 min |
| B3 | 64-wide + BatchNorm + batch 32 | Is capacity above 2M params useful once training is stable? | ✅ | Yes: MCC 0.625 ± 0.003, AUPRC 0.652, segment F1 0.158; 7.7M params, 21 min. Batch 32 also avoided the MPS crash |
| V2 | UNet2: standard decoder, residual blocks, per-strand outputs; RC averaging and edge cropping at inference | Fixes the remaining weak points of the 2023 layout | ✅ | 3 seeds, test with RC + crop / plain: new decoder only 0.596 ± 0.010 / 0.583, seg F1 0.167; **+ residual blocks 0.704 ± 0.001 / 0.678, AUPRC 0.757, seg F1 0.296** (5.6M params); + two strand outputs 0.704 ± 0.005 / 0.673, seg F1 0.292 (no gain). Residual blocks are the main gain; RC + crop add ~+0.025 MCC and +0.07 seg F1 for free |
| TH | Two-head UNet2: window gate head on the bottleneck (+257 params) trained jointly (gate loss weight 1), decoder runs only on windows the gate passes (threshold keeps 99% of CDS windows on validation) | User's two-stage hypothesis; twin = v2_residual trained separately | ✅ | 3 seeds: MCC 0.627 ± 0.006 with the gate = 0.627 without it, seg F1 0.185; gate AUROC 0.916, decodes 64% of test windows, test scan 575 s vs 766 s (−25%). The gate itself costs no accuracy, but joint training costs 0.08 MCC vs the one-head twin (0.704). **Gate weight 0.1 (1 seed): MCC 0.706 with the gate = 0.706 without, seg F1 0.310 — equal to the one-head twin, while decoding 60 % of windows (test 605 s vs 846 s, −29 %); gate AUROC 0.934.** Detached gate (1 seed): MCC 0.694 gated = ungated, seg F1 0.305, 58 % of windows decoded, test 567 s vs 849 s (−33 %) |
| RS | Test-set composition: bases 1:1; windows with CDS : empty 1:1, 1:5, 1:10; natural (one-head twin and the three two-head variants, gate on/off) | How the class balance of the test set changes scores; does the gate reduce false positives? | ✅ | Recall is fixed per model; precision falls as empty windows are added (one-head: 0.996 → 0.729), so MCC falls (0.79 at windows 1:1 → 0.70 natural). Gate on = gate off on every set: the windows the gate drops were already predicted empty, so the gate saves compute but does not remove false positives. **Model ranking depends on composition**: the detached-gate model is best on balanced sets (0.83) and worst-but-one on natural data (0.69, lower precision). Per-window Dice rises as empty windows are added (0.82 → 0.95), the opposite of the real difficulty. Training used ~53 % CDS windows, 1 : 9.5 by base |
| SM | Smaller UNet2 (residual, 1 seed) | How small can the model get | ✅ | 32-wide kernel 11: 1.54M params, MCC 0.670, seg F1 0.331; 32-wide kernel 21: 2.88M, 0.692, 0.311; 16-wide kernel 21: 0.72M, 0.626, 0.245; 16-wide kernel 41: 1.39M, 0.622, 0.197; 8-wide kernel 41: 0.35M, 0.504, 0.082 (reference 32-wide kernel 41: 5.6M, 0.704, 0.296). Width matters more than kernel size |
| V2C | Re-checks on UNet2 (residual, 1 seed) | Do conclusions from the 2023 layout hold? | ✅ | lr 1e-3 0.681, lr 4e-3 0.714 (seg F1 0.326) vs 2e-3 0.705; **32k steps 0.740, AUPRC 0.805, seg F1 0.370** (val MCC still rising at 16k: undertrained); training data 1 / 4 / 19 chromosomes: 0.680 / 0.686 / 0.705 (seg F1 0.249 / 0.275 / 0.296) — data helps mildly |
| E6 | Two-stage: window classifier gates the U-Net | Answers Q5 | 🔄 | Classifier window AUROC 0.90. Gate at recall 0.90: 28.5 % of windows segmented, 9.3 % of CDS lost, MCC 0.450 vs 0.435 one-stage, precision 0.555 vs 0.516, about 30 % less compute |
| E6b | Two-head model (shared encoder, window head + segmentation head), hard negatives, 3 seeds | Cleaner, cheaper version of E6 | ⏳ | Agreed with the user |
| F1 | Isolated fragments of length 150 bp – 100 kb vs whole chromosome; accuracy vs distance to fragment edge | Answers Q4; practical relevance to draft assemblies and metagenomes; how much context the model needs | ✅ | Best model (s0): whole-chromosome scan MCC 0.577; isolated fragments 1 kb 0.560, 500 bp 0.528, 300 bp 0.486, 150 bp 0.402, 100 bp 0.354; ≥2 kb 0.547–0.554 (precision drops when input is longer than the 1 kb training window). MCC near a fragment edge 0.37–0.42 (<5–20 bp), full accuracy from ~100–250 bp: effective context ≈ ±250 bp. Recall on fragments fully inside a CDS rises with length (100 bp 0.27 → 1 kb 0.67) for the best model, not for the 2023 model (≈0.2–0.3). Train-window transfer: each model is best near its training length; 2 kb / 4 kb-trained models are worse on short fragments (100 bp: 0.22 / 0.21 vs 0.33 for 1 kb) |
| E7 | Context: dilated convolutions, larger receptive field | Does more context help? | ⏳ | — |
| E8 | Baselines: logistic regression on k-mers, DilatedResNet (SpliceAI-style), CNN-BiLSTM (Helixer-style) | Puts the U-Net in context | ⏳ ready to run: `--groups base`, size-matched to UNet2 (1.5M and 5.6M), 1 seed each, ~13 h in total | — |
| E9 | SegmentNT on chr20 + 21 with the same metrics | The comparison for Q6 option 1 | ⏳ | SegmentNT has no CDS class (14 elements; exon includes UTRs). Its reported scratch U-Net MCC 0.07 is the mean over 14 elements at lr 5e-5, threshold 0.5 — not comparable with our CDS MCC. Plan: derive CDS from its exon/UTR outputs and/or train our U-Net on its exon label |
| P1 | Boundary post-processing: smoothing, minimum length, HMM | Fixes the main weakness: segment F1 at ±10 bp is only 0.02–0.09 (fragmented predictions) | ⏳ | — |
| W1 | Rewrite the paper (v2) with the new protocol and results | The deliverable | ⏳ | Includes correcting the 1 : 60 claim |
| G1 | Push the sweep summary, figures and this file to GitHub | Results visible in the repo | ⏳ | Code is pushed; results are local only |

## C. Main findings so far

1. The 2023 evaluation protocol overstated quality; under an honest chromosome hold-out the original
   model reaches MCC 0.425 (0.274 when trained with the old protocol).
2. Per-window Dice is not a valid metric for this task; MCC on whole chromosomes is.
3. The model is capacity-limited, not data-limited: wider U-Net and BatchNorm give +0.05 MCC each.
4. Dice-type losses fail; BCE with selection by MCC is the recommended recipe.
5. Base-level accuracy is decent, but exact CDS boundaries are not yet recovered.
