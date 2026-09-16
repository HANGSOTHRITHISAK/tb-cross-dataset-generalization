# Dataset Audit

This document is the decision record for selecting datasets for the cross-dataset TB experiment.

**Do not choose datasets solely because they are easy to download.** A dataset is useful as an external domain only if its provenance is sufficiently independent from the training source.

## Audit fields

For every candidate dataset, record:

| Field | What to verify |
|---|---|
| Dataset name | Canonical/original name |
| Original source | First-party host or canonical publication |
| Institution / geography | Where the images originated |
| Population | Relevant inclusion/population information |
| Images / patients | Counts, preferably by class |
| Task labels | Exact meaning of TB/normal/other labels |
| Image format | PNG/JPEG/DICOM/etc. |
| View information | PA/AP/lateral and whether identifiable |
| Patient ID | Whether patient-level grouping is possible |
| Metadata | Age, sex, acquisition data, etc. if available |
| License / terms | Access and redistribution constraints |
| Known derivatives | Kaggle/repackaged versions and other mirrors |
| Overlap risk | Potential shared images/patients with other candidates |
| Known quirks | Artifacts, preprocessing, label noise, imbalance |
| Proposed role | Train / validation / external test / reject |

## Candidate table

Dataset candidates will be added here only after provenance is verified from authoritative sources.

| Dataset | Provenance verified | Patient IDs | Licensing checked | Overlap risk checked | Proposed role |
|---|---:|---:|---:|---:|---|
| TBD | No | TBD | No | No | Candidate |

## Selection criteria

The final combination should favor:

1. genuinely independent clinical sources,
2. compatible TB classification labels,
3. enough samples for a meaningful experiment,
4. patient-level identifiers or defensible grouping where possible,
5. accessible and clearly documented usage terms,
6. differences in source/population/acquisition that make external validation meaningful,
7. manageable download and compute requirements for the course timeline.

## Important warning: repackaged datasets

Public chest X-ray collections are frequently mirrored, combined, renamed, or redistributed. Two downloads with different dataset names are not automatically independent domains.

Before the experimental matrix is locked, trace candidate datasets back to their original sources and investigate whether one collection contains images originating from another.
