# Dataset Audit

This document records the provenance and suitability of candidate tuberculosis chest X-ray datasets before any training begins.

## Audit checklist

For every candidate dataset, record:

- Official dataset name and original source
- Citation / paper
- Institution and country / population where known
- Number of patients and images
- TB-positive and TB-negative counts
- Label definition and diagnostic reference standard
- Image format, resolution, and view position
- Whether patient identifiers or grouping information are available
- Licensing, access, and redistribution terms
- Known duplicates, derivatives, or overlap with other datasets
- Important acquisition or demographic differences
- Whether the dataset is suitable as a training domain, internal test domain, or external test domain

## Candidate table

| Dataset | Source / Institution | Patients | Images | TB+ | TB- | Patient IDs | License / Access | Known overlap | Intended role | Status |
|---|---|---:|---:|---:|---:|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | To audit |

## Independence rule

A dataset counts as a distinct external domain only when its provenance is meaningfully independent. Repackaged copies, subsets, or mirrors of the same original images must not be treated as separate datasets.

## Split integrity

Whenever multiple images can belong to the same patient, all images from one patient must stay in the same split. External test datasets must remain untouched during model and hyperparameter selection.
