# Dataset Audit

> **Working draft — 16 September 2026.** This is a provenance-first audit, not a final dataset selection. Counts and access conditions should be rechecked against the original distribution when the data are downloaded.

This document records the provenance and suitability of candidate tuberculosis chest X-ray datasets before any training begins. The main goal is to prevent a superficially strong cross-dataset experiment from being invalidated by duplicated images, repackaged source collections, patient leakage, or class-conditional acquisition artifacts.

## Current working decision

The strongest immediately accessible candidate for a **primary training domain** is the Kiran/Jabeen Pakistani hospital dataset (Mendeley DOI `10.17632/8j2g3csprk`). It contains 2,494 TB images and 514 normal images described as collected from a local hospital in Pakistan and is released under CC BY 4.0. Its class imbalance and pre-resizing require explicit handling, and patient-level identifiers/grouping have not yet been verified.

**Shenzhen** remains a useful independent clinical domain and **Montgomery County** is especially attractive as a small untouched external validation set. **TBX11K and the Rahman/Qatar-Dhaka composite must not be treated as clean independent external domains relative to Shenzhen/Montgomery without source filtering and deduplication**, because they incorporate/repackage older public collections.

This gives the project a conservative working direction:

1. Primary development/training: Pakistani hospital dataset, pending file-level audit.
2. External domain: Shenzhen, with no hyperparameter tuning against its external-test portion.
3. Secondary external stress test: Montgomery County, preferably kept untouched until late evaluation.
4. TBX11K/Rahman: provenance/overlap controls or optional secondary analyses only.
5. TB Portals: optional research extension if access can be obtained without delaying the course project.

## Candidate table

| Dataset | Source / institution | Images / class information | Access | Independence / overlap assessment | Working role | Status |
|---|---|---|---|---|---|---|
| Kiran/Jabeen Pakistani TB CXR | Local hospital, Pakistan | 3,008 total: 2,494 TB, 514 normal | Mendeley Data; CC BY 4.0 | Described as locally collected; no NLM/TBX11K ancestry identified in current audit. File-level duplicate and patient-ID audit still required. | Primary training/development candidate | **Priority audit** |
| Shenzhen Hospital CXR | Shenzhen No. 3 People's Hospital, China | 662 total: 336 TB-consistent, 326 normal | NLM/LHNCBC public distribution | Original NLM clinical collection; independent collection from Montgomery. Images also appear in later composite datasets, so do not pair naively with those composites. | External domain / possible reciprocal experiment | **Keep** |
| Montgomery County CXR | Montgomery County TB program, Maryland, USA | 138 total: 58 TB-consistent, 80 normal | NLM/LHNCBC public distribution | Original NLM clinical collection; independent collection from Shenzhen. Images also appear in later composite datasets. | Small untouched external stress test | **Keep** |
| TBX11K | Curated TB benchmark | 11,200 total; 5,000 healthy, 5,000 sick/non-TB, 1,200 TB-category | Public research distribution | Known source relationship/overlap with older TB collections including Shenzhen/Montgomery; unsafe as an independent domain against those collections without filtering/deduplication. | Optional secondary analysis | **Do not use naively** |
| Rahman / Qatar-Dhaka TB CXR database | Composite dataset assembled by Qatar University / University of Dhaka collaborators | Common public release: 700 TB + 3,500 normal; larger variants/derived subsets also circulate | Kaggle/public research distribution | Composite/repackaged data drawing on older sources including NLM material; not a clean independent hospital domain. | Avoid for primary cross-domain claim | **Exclude from core** |
| TB Portals | NIAID-supported multi-country/multi-institution collection | Large longitudinal TB resource; multiple images may exist per patient | Data-access process required for full imaging | Potentially strong independent extension; patient grouping mandatory. Access friction makes it unsuitable as a course-project dependency. | Future external validation | **Extension** |
| RadAI | Nam Dinh Hospital, Vietnam | 10,569 patients in post-deployment dataset; includes TB among 21 findings | Mendeley Data; CC BY 4.0 | Distinct hospital/population and potentially valuable, but the release centers on predictions/reports/demographics; availability of usable CXR image pixels and TB ground truth must be verified before inclusion. | Candidate only if image/label audit succeeds | **Investigate later** |

## Why the Pakistani dataset is currently attractive

The public record describes 3,008 images collected from a local hospital in Pakistan: 2,494 TB and 514 normal. This is substantially larger than Shenzhen and Montgomery and, unlike TBX11K or the Rahman composite, is presented as a locally collected cohort rather than an aggregation of the classic NLM datasets. It therefore gives us a plausible training domain while preserving Shenzhen/Montgomery as external domains.

However, it is **not yet cleared for training**. Before use we need to inspect the actual downloaded files and answer:

- Are filenames or metadata sufficient to group multiple images from the same patient?
- Are all 3,008 images unique? Compute cryptographic hashes plus perceptual hashes.
- Are TB and normal images processed identically, or are there class-specific borders, dimensions, compression signatures, annotations, or other acquisition artifacts?
- What does “TB patient” mean diagnostically (microbiological confirmation, radiological diagnosis, treatment record, etc.)?
- Are the 514 normal controls from the same hospital/acquisition period and pipeline as the TB cases?
- What resizing/preprocessing was performed before public release, and has diagnostically relevant metadata been removed?

Until those questions are answered, the dataset is a **leading candidate**, not a finalized source.

## Acquisition-confounding rule

Dataset independence alone is insufficient. A classifier can appear to detect TB while learning acquisition/source cues when positive and negative classes were collected or processed differently. For every candidate dataset we therefore audit both:

1. **cross-dataset independence** — whether source images/patients overlap other datasets; and
2. **within-dataset class comparability** — whether TB-positive and TB-negative images share a comparable acquisition and processing pipeline.

A recent 2026 audit of five open TB CXR corpora explicitly studies this class-conditional acquisition-confounding problem and includes the Pakistani Mendeley dataset alongside Montgomery, Shenzhen, Rahman, and TBX11K. This makes acquisition auditing part of the core methodology rather than an optional caveat.

## Proposed experiment — working draft

The exact split depends on patient metadata discovered during download.

### Stage 1: development baseline

Train a fixed ImageNet-pretrained baseline (currently DenseNet-121) on the Pakistani dataset. Use patient-level train/validation/internal-test partitions if patient grouping is recoverable. If patient grouping is unavailable, document that limitation explicitly and use deduplication plus conservative image-level splits rather than implying patient independence.

### Stage 2: external evaluation

Freeze model selection before external evaluation. Evaluate the selected model on harmonized Shenzhen and Montgomery cohorts. Report at minimum AUROC, sensitivity, specificity, precision, F1, confusion matrices, and the in-domain-to-external performance change.

### Stage 3: robustness extension

If time permits, compare one controlled intervention such as stronger augmentation, pooled-source training, or a second architecture. Do not expand into a large architecture leaderboard.

## Leakage and split integrity

- Multiple images from one patient must stay in one split whenever patient grouping is available.
- External datasets are not used for hyperparameter selection, threshold optimization, early stopping, or model selection.
- Exact-file hashes and perceptual hashes should be computed before training to identify duplicates/near-duplicates within and across downloaded datasets.
- Composite datasets are never assumed independent merely because they have a different repository name.
- Dataset-specific preprocessing must not leak class labels through directory conventions, overlays, borders, dimensions, or filenames.

## Download-time audit checklist

For each downloaded source, generate a lightweight manifest containing only redistributable metadata:

- relative image identifier/path
- class label after harmonization
- patient/group identifier if available
- original image dimensions and channels
- file format
- SHA-256 hash
- perceptual hash
- dataset/source label

Then inspect class-conditional distributions of dimensions, aspect ratio, intensity, compression/file type, and obvious borders/text markers before training.

## Sources consulted for this working draft

- Kiran, S. & Jabeen, I. *Dataset of Tuberculosis Chest X-rays Images*. Mendeley Data, DOI `10.17632/8j2g3csprk` (V2 released 2024; V3 released 2026). Public description: 2,494 TB + 514 normal CXRs from a local hospital in Pakistan; CC BY 4.0.
- NLM/LHNCBC public Tuberculosis Chest X-ray Datasets: Shenzhen Hospital and Montgomery County collections.
- Liu et al. TBX11K project / *Rethinking Computer-Aided Tuberculosis Diagnosis*.
- Rahman et al. public TB Chest X-ray Database and its documented composite provenance.
- NIAID TB Portals data-access documentation.
- *Auditing Class-Conditional Acquisition Confounding Across Five Open Tuberculosis Chest X-ray Corpora*, medRxiv preprint posted 17 August 2026. This is a preprint and should be treated as recent supporting methodology rather than settled evidence.

## Open questions before dataset lock

- Verify patient/group identifiers and diagnostic reference standard for the Pakistani dataset.
- Download and hash candidate datasets to test exact and near-duplicate overlap directly.
- Quantify class-conditional image-size/intensity/artifact differences before training.
- Decide whether Shenzhen is used entirely as external test data or whether a reciprocal Shenzhen-trained experiment remains worthwhile.
- Reassess TB Portals only if access is straightforward enough not to threaten the course timeline.

## Decision gate

**Do not begin final model training until the Pakistani dataset passes the file-level audit.** Pipeline scaffolding and manifest/audit utilities may be implemented before then because those components do not depend on final dataset selection.
