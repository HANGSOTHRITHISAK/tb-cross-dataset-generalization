# Current Project State

**Updated:** 2026-09-26

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

With Phase 2 closed, begin the compute-environment and real-data GPU smoke-test phase.

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

The locked baseline is an **ImageNet-pretrained DenseNet-121 adapted for 3-class internal classification**.

Internal TBX11K reporting should include accuracy, macro-F1, per-class recall, and a 3x3 confusion matrix.

For Shenzhen/Montgomery, predictions are harmonized to **TB vs non-TB** and reported with AUROC, sensitivity/recall, specificity, precision, F1, and a binary confusion matrix. Any threshold must be selected using TBX11K development data only and frozen before external evaluation. Where feasible, uncertainty should be reported using bootstrap confidence intervals.

Optional architecture comparisons, stronger augmentation, additional datasets, acquisition-shift analysis, qualitative error analysis, or Grad-CAM should wait until the baseline pipeline is complete and reproducible.

## Immediate next actions

Resume work in this order:

1. set up the CUDA-capable development environment for Phase 3;
2. locate the ignored raw data without committing it;
3. run the tiny real-data GPU smoke test against the frozen split;
4. verify training-only preprocessing fitting and training-only stochastic augmentation in the executable pipeline;
5. proceed to reproducible baseline training only after the smoke test passes.

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
