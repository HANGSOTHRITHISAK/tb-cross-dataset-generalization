# Phase 2 audit status

**Status:** open; evidence hardened on 2026-09-24. Split implementation is staged, but the final split is intentionally not frozen while the 17 primary-pool dHash decisions remain pending.

Phase 2 has reproducible file-level audit evidence for TBX11K, Shenzhen, and Montgomery. The duplicate-safe split machinery now exists, but it hard-fails until every required visual adjudication row is resolved.

The versioned manifests and reports contain metadata, repository-relative paths, checksums, image statistics, and audit findings. They do not contain raw radiographs. Raw datasets remain ignored under `data/raw/`.

## Verified primary-pool state

The source of truth is `data/audit/tbx11k_audit.json`:

- 8,400 primary labeled TBX11K rows;
- 126 exact SHA-256 duplicate groups;
- 252 rows participating in those groups;
- 8,274 unique SHA-256 checksums before perceptual-candidate adjudication;
- 27 internal dHash candidate pairs;
- 17 dHash candidate pairs wholly within the primary labeled pool, all Sick non-TB;
- 10 dHash candidate pairs connecting a primary Sick non-TB image to unreleased TBX11K test content;
- no trustworthy patient/group identifiers.

dHash candidates are not confirmed duplicates. GitHub does not contain the raw radiographs, so visual adjudication cannot be completed from repository metadata alone.

## Frozen dataset roles

- TBX11K is the development domain for Healthy / Sick non-TB / TB classification.
- DenseNet-121 with ImageNet initialization and a 3-class head remains the baseline.
- Shenzhen and Montgomery remain untouched external-test datasets.
- Healthy + Sick non-TB are harmonized to non-TB for external testing.
- External data must not influence training, preprocessing choices, early stopping, hyperparameter selection, checkpoint selection, or threshold selection.

## Split implementation staged on 2026-09-26

The staged implementation:

- versions the 17 candidate rows in `data/audit/tbx11k_primary_dhash_adjudication.csv`;
- refuses to build a final split while any decision is `pending` or lacks a rationale;
- unions exact SHA-256 duplicates with only visually confirmed transformed/re-exported copies;
- hard-fails if any resulting duplicate family has conflicting 3-class labels;
- keeps one deterministic representative per duplicate family;
- assigns the retained unique-image pool with a deterministic, 3-class-stratified 70/15/15 split using seed 42;
- writes a metadata-only split manifest and summary, with no raw images or machine-local paths.

This deliberately allows implementation work to proceed without prematurely declaring the split final.

## Required sequence before Phase 2 closure

1. Manually adjudicate the 17 primary-pool dHash candidate pairs using the original images.
2. Record each pair as a confirmed transformed/re-exported copy or a similar-but-distinct radiograph.
3. Treat confirmed copies as one duplicate family; otherwise retain both samples.
4. Verify every exact or confirmed duplicate family has a consistent 3-class label.
5. Run the staged builder to deduplicate deterministically and generate the proposed stratified 70/15/15 split with seed 42.
6. Save/version the resolved adjudication, split manifest, and split summary without raw images or machine-local paths.
7. Run tests covering determinism, class stratification, duplicate isolation, label-conflict failure, and split integrity.
8. Verify that data-derived preprocessing is fitted on training only and stochastic augmentation is training-only.
9. Run the targeted test suite and Ruff.
10. Update status/decision documentation and explicitly record Phase 2 closure.

After duplicate controls, reporting may claim image-level split independence. It must not claim patient-level independence because trustworthy patient/group IDs are unavailable.

## Evidence boundary and stop rule

The targeted literature check supports the bounded adjudication gate and the explicit leakage controls above. It does not support changing the project direction, dataset roles, label design, baseline, proposed split ratio, or seed.

See `docs/phase2-evidence-hardening.md` for references and rationale. No broader dataset-audit research is warranted unless adjudication reveals a systematic issue, the dataset release changes, or integrity checks fail.
