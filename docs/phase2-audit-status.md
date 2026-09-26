# Phase 2 audit status

**Status:** closed on 2026-09-26. The duplicate-controlled TBX11K split is finalized and frozen.

Phase 2 has reproducible file-level audit evidence for TBX11K, Shenzhen, and Montgomery. All required visual adjudication rows are resolved, and the final metadata-only split manifest and summary are versioned.

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

All 17 primary-pool dHash candidate pairs were manually reviewed using local raw radiographs and confirmed as transformed/re-exported copies with matching anatomy and positioning. Only the decisions and concise rationales are versioned; no raw radiographs or contact sheets are committed.

## Frozen dataset roles

- TBX11K is the development domain for Healthy / Sick non-TB / TB classification.
- DenseNet-121 with ImageNet initialization and a 3-class head remains the baseline.
- Shenzhen and Montgomery remain untouched external-test datasets.
- Healthy + Sick non-TB are harmonized to non-TB for external testing.
- External data must not influence training, preprocessing choices, early stopping, hyperparameter selection, checkpoint selection, or threshold selection.

## Final split frozen on 2026-09-26

The finalized implementation:

- versions the 17 candidate rows in `data/audit/tbx11k_primary_dhash_adjudication.csv`;
- refuses to build a final split while any decision is `pending` or lacks a rationale;
- unions exact SHA-256 duplicates with only visually confirmed transformed/re-exported copies;
- hard-fails if any resulting duplicate family has conflicting 3-class labels;
- keeps one deterministic representative per duplicate family;
- assigns the retained unique-image pool with a deterministic, 3-class-stratified 70/15/15 split using seed 42;
- writes a metadata-only split manifest and summary, with no raw images or machine-local paths.

The 8,400 primary rows reduce to **8,257 retained unique images**, comprising **8,114 singleton families** and **143 duplicate families of size 2**.

The frozen 70/15/15, seed-42 split is:

| Split | Healthy | Sick non-TB | TB | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 2,660 | 2,560 | 560 | 5,780 |
| Validation | 570 | 549 | 120 | 1,239 |
| Internal test | 570 | 548 | 120 | 1,238 |

Zero duplicate families cross splits. Image-level duplicate control is established. Patient-level independence is **not** claimed because trustworthy patient/group IDs are unavailable.

## Closure verification

- targeted split tests: **5 passed**;
- full pytest suite: **23 passed**;
- Ruff: **all checks passed**;
- adjudication CSV, split manifest, and split summary contain metadata only;
- raw radiographs and local review sheets remain uncommitted.

Data-derived preprocessing must still be fitted on training only, and stochastic augmentation must remain training-only during Phase 3 implementation.

## Evidence boundary and stop rule

The targeted literature check supports the completed adjudication gate and the explicit leakage controls above. It does not support changing the project direction, dataset roles, label design, baseline, frozen split ratio, or seed.

See `docs/phase2-evidence-hardening.md` for references and rationale. No broader dataset-audit research is warranted unless adjudication reveals a systematic issue, the dataset release changes, or integrity checks fail.
