# Dataset Audit

> **Status update — 2026-09-23:** this file preserves the earlier dataset-selection investigation. The current project decision is now locked to audited **TBX11K** for development with **Shenzhen** and **Montgomery County** as external evaluation domains. The internal TBX11K label space is **Healthy / Sick non-TB / TB**; external evaluation is harmonized to **TB vs non-TB**. See [current-project-state.md](current-project-state.md) and [experimental-protocol.md](experimental-protocol.md) for the current source of truth.

> **Working draft — 16 September 2026.** This is a provenance-first audit, not a final dataset selection. Counts and access conditions should be rechecked against the original distribution when the data are downloaded.

This document records the provenance and suitability of candidate tuberculosis chest X-ray datasets before any training begins. The main goal is to prevent a superficially strong cross-dataset experiment from being invalidated by duplicated images, repackaged source collections, patient leakage, or class-conditional acquisition artifacts.

## Current working decision

No dataset is currently locked as the primary training domain.

The Kiran/Jabeen Pakistani cohort remains a high-priority candidate because it is large (3,008 images: 2,494 TB and 514 normal), publicly accessible, and described as coming from a local hospital in Pakistan. However, newly inspected 2026 external audit evidence makes it inappropriate to call this a clean primary training source before our own file-level audit. The public dataset record does not document per-class acquisition provenance, equipment, diagnostic reference standard, or verified patient grouping, and an independent preprint reports unusually strong class separability from low-level/acquisition-sensitive representations.

**Shenzhen** remains especially useful because its published provenance places TB-positive and normal images in the same hospital/routine acquisition stream. **Montgomery County** remains a useful small external stress test. **TBX11K and the Rahman/Qatar-Dhaka composite must not be treated as clean independent external domains relative to Shenzhen/Montgomery without source filtering and deduplication.**

Current conservative direction:

1. Audit Kiran/Jabeen Mendeley **V2** first; do not yet assign it permanently to train or external test.
2. Keep Shenzhen as a likely core domain.
3. Keep Montgomery County primarily as a small untouched external stress test.
4. Keep TBX11K/Rahman outside the core independence claim unless source filtering/deduplication supports their use.
5. Treat TB Portals and other cleaner clinical cohorts as later extensions rather than course-project dependencies.

## Candidate table

| Dataset | Source / institution | Images / class information | Access | Independence / acquisition assessment | Working role | Status |
|---|---|---|---|---|---|---|
| Kiran/Jabeen Pakistani TB CXR | Local hospital, Pakistan (hospital unnamed publicly) | 3,008 total: 2,494 TB, 514 normal | Mendeley Data; CC BY 4.0 | No overlap with classic training corpora was reported by one external V2 hash/pHash audit, but per-class acquisition provenance and true patient grouping remain undocumented. External confound audit reports very strong low-level class separability. | High-priority domain; role not yet locked | **Priority audit** |
| Shenzhen Hospital CXR | Shenzhen No. 3 People's Hospital, China | 662 total: 336 TB-consistent, 326 normal | NLM/LHNCBC public distribution | Original NLM clinical collection. Published provenance indicates both classes came from the same hospital/routine stream and approximately the same one-month window. Later composites reuse its images. | Likely core domain / possible reciprocal experiment | **Keep** |
| Montgomery County CXR | Montgomery County TB program, Maryland, USA | 138 total: 58 TB-consistent, 80 normal | NLM/LHNCBC public distribution | Original NLM clinical collection with both classes from the same screening program. Images are reused in later composites. | Small untouched external stress test | **Keep** |
| TBX11K | Curated TB benchmark | 11,200 total; 5,000 healthy, 5,000 sick/non-TB, 1,200 TB-category | Public research distribution | Per-class acquisition/site breakdown is not sufficiently documented for our purposes; known source relationships/overlap with older TB collections make naive independence claims unsafe. | Optional secondary analysis | **Do not use naively** |
| Rahman / Qatar-Dhaka TB CXR database | Composite dataset assembled by Qatar University / University of Dhaka collaborators | Common public release: 700 TB + 3,500 normal | Kaggle/public research distribution | Explicit composite provenance: TB and normal classes draw heavily from different upstream sources; includes NLM material. | Exclude from primary cross-domain claim | **Exclude from core** |
| TB Portals | NIAID-supported multi-country/multi-institution collection | Large longitudinal TB resource; multiple images may exist per patient | Data-access process required for full imaging | Potentially strong independent extension; patient grouping mandatory. Access friction makes it unsuitable as a course-project dependency. | Future external validation | **Extension** |
| RadAI | Nam Dinh Hospital, Vietnam | 10,569 patients in post-deployment dataset; includes TB among reported findings | Public data record | Distinct hospital/population and potentially valuable, but usable image pixels and suitable TB ground truth still require verification. | Candidate only if image/label audit succeeds | **Investigate later** |

## Kiran/Jabeen Pakistani cohort — evidence status

### Original public record

Mendeley V2 (`10.17632/8j2g3csprk.2`, May 2024) and V3 (`10.17632/8j2g3csprk.3`, March 2026) publish the same headline description: 2,494 TB and 514 normal chest X-rays collected from a local hospital in Pakistan, with all images processed/resized to a uniform dimension. Both are CC BY 4.0. The public version-comparison metadata does not establish whether the underlying image files are byte-identical.

For the first local audit, **pin V2** because the recent 2026 confound study explicitly used V2 and publishes reproducibility artifacts against that version. V3 can be compared later by file inventory/hashes.

### External evidence already available — not our result

A 2026 medRxiv preprint, *Auditing Class-Conditional Acquisition Confounding Across Five Open Tuberculosis Chest X-ray Corpora*, audits Montgomery, Shenzhen, Rahman, TBX11K and the Pakistani Mendeley V2 cohort. It is recent and not peer reviewed, so its results are supporting evidence rather than ground truth.

Its public repository reports:

- Two holdout-vs-training overlap checks (May and August 2026) over 3,008 Pakistani images versus 13,260 training rows found **zero cross-set exact/near-duplicate matches** using MD5 plus a 256-bit perceptual-hash Hamming threshold. This supports cross-dataset independence from that specific training corpus, but we will still hash our own downloads.
- For Mendeley-PK, public provenance is classified as **undocumented/unclear** at the per-class acquisition level: the public description says the images came from a local hospital but does not separately document where TB and normal images came from, the device, dates, reference standard, or per-class pipeline.
- Released robustness results report class AUROC **0.9881 from image-statistics features** and near-perfect class separability from several frozen/low-spatial-content representations on the Pakistani cohort. Border intensity and sharpness are among the most important simple image-statistics features in its released analysis. These are warning signals for shortcut learning, not evidence that TB pathology itself is easy to classify.
- A released manifest uses `patient_id = mendeley_pk_<filename stem>`. Inspection of the associated index builder shows that this identifier is constructed from each filename; it is **not evidence of a real patient identifier**. Until original metadata proves otherwise, patient-level independence is unverified.

### What we must verify ourselves

- exact V2 file count and folder structure
- SHA-256 hashes and near-duplicate candidates within the cohort
- overlap against any other datasets we actually use
- whether real patient/study metadata exist
- dimensions, channels, format, compression/file size
- border statistics, sharpness, intensity/histogram characteristics and obvious text/markers by class
- whether TB/normal acquisition pipelines are comparable
- diagnostic/reference-standard description for TB labels
- whether V3 materially differs from V2

Until these checks are complete, the Pakistani cohort is a **high-priority domain, not a cleared training source**.

## Acquisition-confounding rule

Dataset independence alone is insufficient. A classifier can appear to detect TB while learning acquisition/source cues when positive and negative classes were collected or processed differently. For every candidate dataset we therefore audit both:

1. **cross-dataset independence** — whether source images/patients overlap other datasets; and
2. **within-dataset class comparability** — whether TB-positive and TB-negative images share a comparable acquisition and processing pipeline.

A useful result from the 2026 external audit is methodological rather than dataset-specific: within-corpus performance can stay extremely high while transfer to an unseen cohort collapses. Our project should therefore report the cross-domain matrix as the central result and treat in-domain scores cautiously.

## Proposed experiment — working draft v2

The exact train/test assignment remains contingent on the audits.

### Stage 1: lock audited domains

Download candidate originals, create manifests, test overlap, inspect class-conditional acquisition features, and document whether patient-level grouping is possible. Select the cleanest defensible development domain only after this gate.

### Stage 2: fixed baseline and cross-dataset evaluation

Use one fixed ImageNet-pretrained baseline (currently DenseNet-121). Tune only on the designated development domain. Freeze the selected checkpoint/protocol, then evaluate on one or more untouched external domains.

Preferred minimum matrix after dataset lock:

- development-domain train -> development-domain test
- development-domain train -> external domain 1
- development-domain train -> external domain 2 when feasible

A reciprocal experiment can be added only if the second domain is large enough to support a defensible development split. Montgomery is likely too small to serve as the main training source.

Report at minimum AUROC, sensitivity/recall, specificity, precision, F1, confusion matrices, and the in-domain-to-external performance change. Bootstrap confidence intervals are desirable where sample size permits.

### Stage 3: lightweight confound/shift diagnostic

If time permits, add one compact diagnostic rather than a second giant project. Examples:

- compare simple class-conditional image statistics across domains
- test whether source/domain is predictable from non-pathology image characteristics
- inspect errors and Grad-CAM qualitatively
- compare one controlled augmentation/domain-generalization intervention

The objective is to help interpret the generalization gap, **not** to reproduce the 2026 confound paper.

## Leakage and split integrity

- Multiple images from one patient must stay in one split whenever real patient grouping is available.
- A filename-derived synthetic identifier must never be presented as verified patient metadata.
- External datasets are not used for hyperparameter selection, threshold optimization, early stopping, or model selection.
- Exact-file hashes and perceptual hashes should be computed before training to identify duplicates/near-duplicates within and across downloaded datasets.
- Composite datasets are never assumed independent merely because they have a different repository name.
- Dataset-specific preprocessing must not leak class labels through directory conventions, overlays, borders, dimensions, compression, or filenames.

## Download-time audit checklist

For each downloaded source, generate a lightweight manifest containing only redistributable metadata:

- relative image identifier/path
- class label after harmonization
- patient/group identifier only when genuinely supplied or recoverable
- original image dimensions and channels
- file format / file size
- SHA-256 hash
- perceptual hash
- intensity and border/sharpness summary features
- dataset/source label

Then inspect class-conditional distributions before training. Any feature that strongly predicts class despite carrying little plausible pathology information is a reason to investigate acquisition/processing provenance before interpreting model accuracy.

## Sources consulted for this working draft

- Kiran, S. & Jabeen, I. *Dataset of Tuberculosis Chest X-rays Images*. Mendeley Data, V2 DOI `10.17632/8j2g3csprk.2`; V3 DOI `10.17632/8j2g3csprk.3`.
- NLM/LHNCBC public Tuberculosis Chest X-ray Datasets: Shenzhen Hospital and Montgomery County collections.
- Liu et al. TBX11K / *Rethinking Computer-Aided Tuberculosis Diagnosis*.
- Rahman et al. public TB Chest X-ray Database and its documented composite provenance.
- NIAID TB Portals data-access documentation.
- Bilal, A. *Auditing Class-Conditional Acquisition Confounding Across Five Open Tuberculosis Chest X-ray Corpora*, medRxiv, posted 17 August 2026, DOI `10.64898/2026.08.14.26360390` (preprint).
- Public reproducibility repository accompanying the Bilal preprint, inspected for provenance notes, overlap logs, released result JSONs, manifests and index-building code.

## Open questions before dataset lock

- Obtain and locally audit Mendeley V2.
- Verify whether V2 and V3 image files actually differ.
- Verify real patient/group identifiers and diagnostic reference standard for Pakistan.
- Hash candidate datasets against each other directly.
- Extend our audit tooling with border/sharpness/histogram features useful for class-conditional shortcut checks.
- Decide the final development and external-test roles only after those checks.

## Decision gate

**Do not begin final model training until the primary development dataset is explicitly cleared after file-level and class-conditional acquisition auditing.** Generic tooling, reproducibility scaffolding and source-independent code may proceed before then.
