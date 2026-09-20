# Current Project State

**Updated:** 2026-09-20

This document is the concise, public-facing checkpoint for the current state of the project. It is intended to help collaborators and future sessions resume work without relying on private local-machine or chat history.

When this document conflicts with newer committed code, manifests, tests, or decision records, the newer repository state is authoritative.

## Current research question

How well does a tuberculosis chest X-ray classifier trained on one source dataset generalize to independent external datasets collected from different clinical sources?

The current proposal-aligned core experiment is:

1. develop a binary TB classifier on a cleaned TBX11K development pool;
2. evaluate on a held-out TBX11K internal test split;
3. freeze the selected model and protocol;
4. evaluate externally on Shenzhen;
5. evaluate externally on Montgomery County as an additional external stress test.

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
- **7,600 canonical label 0**
- **800 canonical label 1**

The pool is drawn from the labeled TBX11K `health/`, `sick/`, and `tb/` content used for the planned binary experiment.

No trustworthy patient/group identifiers are currently available. The project therefore must not claim patient-level splitting unless genuine grouping metadata are later established.

### Internal exact duplicates

Within the 8,400-row primary labeled TBX11K pool:

- **126 exact-duplicate groups**
- **252 rows participating in those groups**
- **8,274 unique SHA-256 image checksums**

These exact duplicates must not cross train/validation/internal-test boundaries.

### TBX11K overlap with Shenzhen and Montgomery

The audit identified **800 perceptual TBX11K ↔ Shenzhen/Montgomery candidates**.

Current evidence shows:

- the candidates occur in TBX11K `extra/` or `test/`;
- none occur in the primary labeled `health/`, `sick/`, or `tb/` development pool.

Therefore the intended primary TBX11K development pool is not currently affected by those perceptual external-overlap candidates.

The expensive full-dataset audit should not be rerun unless a concrete inconsistency, new data release, corrupted local copy, or methodological reason appears.

## Final Phase 2 decision still pending

PR #10 merged the reconciled Phase 2 implementation, manifests, audit evidence, tests, and documentation into `main`, but intentionally left the exact-duplicate handling and final TBX11K split policy unresolved.

The remaining Phase 2 task is to implement and formally freeze that policy.

### Intended exact-duplicate policy

The current intended policy is:

1. group primary labeled TBX11K rows by SHA-256;
2. verify that all rows sharing a checksum have the same canonical label;
3. hard-fail and require manual investigation if one checksum maps to conflicting labels;
4. otherwise retain one deterministic representative per checksum;
5. construct the project split from the resulting unique-image pool.

This is preferred over simply keeping duplicate copies in the same partition because it avoids giving repeated images extra statistical weight during training or evaluation.

### Intended split policy

The current proposed split is:

- **70% train**
- **15% validation**
- **15% internal test**
- stratified by the binary canonical label
- fixed random seed: **42**

This remains a proposed policy until it is implemented, tested, documented, and versioned.

The final split implementation must ensure:

- deterministic output;
- no exact checksum appears in more than one split;
- class stratification is preserved as closely as practical;
- unknown-label rows are excluded from the labeled experiment;
- no Shenzhen or Montgomery sample influences model-development decisions;
- no patient-level separation claim is made without genuine patient/group metadata.

## Phase 2 closure criteria

Phase 2 should be considered formally closed only after:

- [ ] deterministic exact-deduplication logic is implemented;
- [ ] conflicting labels inside an exact-duplicate group hard-fail;
- [ ] the final stratified train/validation/internal-test split is implemented;
- [ ] split manifest(s) are saved/versioned where redistribution permits;
- [ ] tests cover duplicate isolation, determinism, label conflicts, and split integrity;
- [ ] the relevant protocol/decision/status documentation is updated;
- [ ] the normal targeted test suite passes;
- [ ] Ruff passes;
- [ ] Git contains no raw radiographs or machine-local paths;
- [ ] Phase 2 closure is explicitly recorded.

Latest reported validation before this checkpoint:

- **13 tests passed**
- **Ruff passed**

## Phase 3 objective

Immediately after Phase 2 closes, begin the compute-environment and real-data GPU smoke-test phase.

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

## Baseline model and evaluation

The current baseline in the experimental protocol is an **ImageNet-pretrained DenseNet-121 adapted for binary classification**.

Primary metrics:

- AUROC
- sensitivity / recall
- specificity
- precision
- F1
- confusion matrix

Accuracy may be reported as a supplementary metric. Where feasible, uncertainty should be reported using bootstrap confidence intervals.

Optional architecture comparisons, stronger augmentation, additional datasets, acquisition-shift analysis, qualitative error analysis, or Grad-CAM should wait until the baseline pipeline is complete and reproducible.

## Immediate next actions

Resume work in this order:

1. implement the duplicate-safe TBX11K split policy;
2. add/update tests for split integrity and failure cases;
3. run targeted tests and Ruff;
4. update the relevant research documentation and formally close Phase 2;
5. set up the CUDA-capable development environment for Phase 3;
6. transfer/locate ignored raw data without committing it;
7. run the tiny real-data GPU smoke test;
8. proceed to reproducible baseline training only after the smoke test passes.

## Related documentation

Read these alongside this checkpoint when deeper context is needed:

- `docs/experimental-protocol.md`
- `docs/working-decisions.md`
- `docs/tbx11k-overlap-audit.md`
- `docs/phase2-audit-status.md`
- `docs/dataset-audit.md`
- `data/README.md`
- `data/manifests/tbx11k.csv`
- `data/manifests/shenzhen.csv`
- `data/manifests/montgomery.csv`
