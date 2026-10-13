# Current Project State

**Updated:** 2026-10-13

This document is the concise, public-facing checkpoint for the current state of the project. It is intended to help collaborators and future sessions resume work without relying on private local-machine or chat history.

When this document conflicts with newer committed code, manifests, tests, or decision records, the newer repository state is authoritative.

## Current research question

How well does a tuberculosis chest X-ray classifier trained on one source dataset generalize to independent external datasets collected from different clinical sources?

The current proposal-aligned core experiment is:

1. develop a **3-class** classifier on a cleaned TBX11K development pool using **Healthy / Sick non-TB / TB**;
2. evaluate the 3-class model on a held-out TBX11K internal test split;
3. freeze the selected model and protocol;
4. harmonize Healthy + Sick non-TB into **non-TB**, leaving TB as the positive class;
5. evaluate the frozen binary-harmonized output externally on Shenzhen;
6. evaluate externally on Montgomery County as an additional external stress test.

This 3-class internal design was approved on 2026-09-23 after the dataset audit showed that the released train/validation material does not expose trustworthy active/latent/active+latent TB subtype labels separately. Those subtype labels are not inferred.

The project remains provenance-first and leakage-aware. External datasets must not influence training, early stopping, hyperparameter selection, threshold tuning, or checkpoint selection.

## Phase 2 evidence already established

### Dataset acquisition and audit

The following datasets have been acquired and audited:

- TBX11K
- Shenzhen
- Montgomery County

The repository contains versioned manifests, integrity/audit evidence, and cross-dataset overlap reports. Raw radiographs remain local-only and ignored by Git.

### Primary labeled TBX11K development pool

The current primary labeled pool contains:

- **8,400 rows total**
- **3,800 Healthy**
- **3,800 Sick non-TB**
- **800 TB**

The pool is drawn from the labeled TBX11K `health/`, `sick/`, and `tb/` content. The existing manifest's canonical binary label remains useful for cross-dataset harmonization, while the internal training task uses a separate 3-class mapping derived from the released folder/list labels.

No trustworthy patient/group identifiers are currently available. The project therefore must not claim patient-level splitting unless genuine grouping metadata are later established.

### Final duplicate-controlled pool and split

Within the 8,400-row primary labeled TBX11K pool:

- **126 exact-duplicate groups**
- **252 rows participating in those groups**
- **8,274 unique SHA-256 image checksums before perceptual-candidate adjudication**
- **17 dHash candidate pairs wholly inside the primary pool, all Sick non-TB**

The audit contains 27 internal dHash candidate pairs in total. The other 10 connect a primary Sick non-TB image to unreleased TBX11K `test/` content and therefore do not affect the project split.

All 17 primary-pool dHash candidate pairs were visually adjudicated using the local raw radiographs and confirmed as transformed/re-exported copies of the same underlying radiograph. The adjudication decisions and rationales are versioned in `data/audit/tbx11k_primary_dhash_adjudication.csv`; no raw radiographs or review contact sheets are committed.

After unioning exact SHA-256 duplicates with the confirmed transformed/re-exported copies and retaining one deterministic representative per family, the final development pool contains **8,257 unique images**:

- **8,114 singleton families**;
- **143 duplicate families of size 2**.

The final split is frozen at **70/15/15 with seed 42**, stratified by the three-class internal label:

| Split | Healthy | Sick non-TB | TB | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 2,660 | 2,560 | 560 | 5,780 |
| Validation | 570 | 549 | 120 | 1,239 |
| Internal test | 570 | 548 | 120 | 1,238 |

Zero duplicate families cross splits. Image-level duplicate control is established. Patient-level independence is **not** claimed because trustworthy patient/group identifiers are unavailable.

### TBX11K overlap with Shenzhen and Montgomery

The audit identified **800 perceptual TBX11K ↔ Shenzhen/Montgomery candidates**.

Current evidence shows:

- the candidates occur in TBX11K `extra/` or `test/`;
- none occur in the primary labeled `health/`, `sick/`, or `tb/` development pool.

Therefore the intended primary TBX11K development pool is not currently affected by those perceptual external-overlap candidates.

The expensive full-dataset audit should not be rerun unless a concrete inconsistency, new data release, corrupted local copy, or methodological reason appears.

## Phase 2 finalized

The duplicate-safe split policy is implemented and frozen. It:

- produces deterministic output;
- combines exact SHA-256 duplicates and the 17 visually confirmed transformed/re-exported pairs into duplicate families;
- retains the lexicographically smallest sample ID as the deterministic representative of each family;
- hard-fails on conflicting three-class labels within a family;
- ensures no exact or confirmed transformed duplicate family appears in more than one split;
- preserves class stratification as closely as practical;
- excludes unknown-label rows from the labeled experiment;
- requires any data-derived preprocessing parameters to be fitted on training only;
- applies stochastic augmentation only to training;
- prevents the held-out internal test and Shenzhen/Montgomery from influencing model-development decisions;
- distinguishes training, tuning/validation, held-out internal testing, and external testing in reporting;
- supports only an image-level split-independence claim after duplicate controls; patient-level independence is not claimed without genuine patient/group metadata.

The finalized artifacts are `data/manifests/tbx11k_split.csv` and `data/audit/tbx11k_split_summary.json`. Validation on 2026-09-26 reported:

- targeted split tests: **5 passed**;
- full test suite: **23 passed**;
- Ruff: **all checks passed**;
- duplicate-family cross-split violations: **0**;
- no raw radiographs or machine-local paths committed.

## Phase 3 objective

M1 implements the strict frozen-manifest adapter, fixed preprocessing with
training-only augmentation, ImageNet DenseNet-121 with three outputs, and a bounded
one-step smoke CLI. Actual verification and reproducible commands are recorded in
[m1-implementation.md](m1-implementation.md). The frozen split and Phase 2 audit
artifacts are unchanged. Baseline training and held-out evaluations remain pending.

Local M1 verification on 2026-10-13 passed 64 tests and real-image CUDA optimizer
steps in FP32 and FP16. Peak allocated memory was 348.5 MiB and 235.3 MiB,
respectively. This is execution evidence only, not scientific performance evidence.

With Phase 2 closed, Phase 3 began with the compute environment and real-data
GPU smoke pathway described above.

The goal is to establish a reproducible CUDA-capable environment and prove that the project can execute correctly on real TBX11K data before any large training run.

A successful smoke test should demonstrate:

1. NVIDIA GPU availability;
2. CUDA visibility from PyTorch;
3. loading of a small batch of real TBX11K images;
4. valid preprocessing/tensor construction;
5. at least one GPU forward pass;
6. preferably one backward/optimizer step;
7. no path, dtype, label, or memory errors;
8. reproducible environment documentation.

Large baseline training should wait until this smoke test succeeds.

M2's bounded CUDA pilot is now verified: three short epochs, 81 optimizer steps,
54 training and 54 independent validation images, selected-checkpoint reload with
matching validation results, 72.97 seconds total. Peak Torch memory was 402.28 MiB
allocated / 426 MiB reserved; sampled temperature was 57–62 C. Class weighting
was disabled. The frozen 8,257-image assignments and held-out boundaries remain
unchanged. See [m2-pilot.md](m2-pilot.md) and the sanitized execution receipt in
[m2-verification.json](m2-verification.json). This is engineering verification;
subset metrics do not estimate final model performance or establish convergence.

## Baseline model and evaluation

M3A's bounded full-data preflight passed on 2026-10-13: all 5,780 training images
once, all 1,239 validation images once, 2,890 finite AdamW updates, 30.59 minutes
including interpreter startup, and exact checkpoint/continuation-state reload.
Peak Torch memory was 402.28 MiB allocated / 436 MiB reserved; sampled temperature
was 58–65 C with no thermal slowdown. CPU and actual CUDA stochastic fixtures
establish epoch-boundary resume equivalence. The full-data validation macro-F1
0.908306 and accuracy 0.947538 are **one-epoch preflight diagnostics**, not final
research results. See [m3a-preflight.md](m3a-preflight.md) and
[m3a-verification.json](m3a-verification.json). The M3A checkpoint is not a final
baseline, and M3B remains gated on review and separate scientific approval.

Approved M3A decisions preserve unweighted cross-entropy, the existing conservative
training-only augmentation without flipping, and validation macro-F1 selection
with lower-loss/earlier-epoch tie breaks. No imbalance intervention was executed.

The locked baseline is an **ImageNet-pretrained DenseNet-121 adapted for 3-class internal classification**.

Internal TBX11K reporting should include accuracy, macro-F1, per-class recall, and a 3x3 confusion matrix.

For Shenzhen/Montgomery, predictions are harmonized to **TB vs non-TB** and reported with AUROC, sensitivity/recall, specificity, precision, F1, and a binary confusion matrix. Any threshold must be selected using TBX11K development data only and frozen before external evaluation. Where feasible, uncertainty should be reported using bootstrap confidence intervals.

Optional architecture comparisons, stronger augmentation, additional datasets, acquisition-shift analysis, qualitative error analysis, or Grad-CAM should wait until the baseline pipeline is complete and reproducible.

## Immediate next actions

Resume work in this order:

1. review the M3A implementation, full-data evidence and CPU/CUDA resume fixtures;
2. approve M3B's epoch cap, early-stopping patience and any scheduler/threshold
   policy; retain the already approved primary loss, augmentation and selection;
3. implement and execute separately bounded full baseline training from fresh
   ImageNet weights only after M3B authorization; do not promote the M3A checkpoint;
4. implement held-out evaluation after model and threshold decisions are frozen.

## Related documentation

Read these alongside this checkpoint when deeper context is needed:

- `docs/experimental-protocol.md`
- `docs/phase2-evidence-hardening.md`
- `docs/working-decisions.md`
- `docs/tbx11k-overlap-audit.md`
- `docs/phase2-audit-status.md`
- `docs/dataset-audit.md`
- `data/README.md`
- `data/manifests/tbx11k.csv`
- `data/manifests/shenzhen.csv`
- `data/manifests/montgomery.csv`
