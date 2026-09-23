# Cross-Dataset Generalization for Tuberculosis Detection in Chest X-Rays

This repository contains the code, experiment configuration, documentation, and results for a computer-vision project studying **cross-dataset generalization in deep-learning-based tuberculosis (TB) detection from chest X-rays**.

The central question is simple but important:

> How well does a model trained on one chest X-ray dataset generalize to an independent dataset collected from a different clinical source?

Rather than optimizing only for in-domain performance, this project focuses on the **generalization gap** between performance on a model's source dataset and performance on external datasets.

## Project status

**Current stage:** experimental design and dataset audit.

The repository is being structured so that the course project can remain manageable while also supporting a stronger research-oriented extension later.

## Planned experimental design

The locked course-project baseline uses **TBX11K** as the development domain and **Shenzhen** plus **Montgomery County** as untouched external evaluation domains.

The internal TBX11K task is **3-class classification**:

- Healthy
- Sick non-TB
- TB

The audited release does not expose trustworthy active/latent/active+latent subtype labels separately for the labeled train/validation TB images, so those subtypes are not inferred.

For external evaluation, the label space is harmonized to **TB vs non-TB** by combining Healthy + Sick non-TB into non-TB. The selected DenseNet-121 model is frozen before evaluation on Shenzhen and Montgomery.

Optional extensions, only after the baseline is reproducible:

- architecture comparison
- augmentation / robustness experiments
- dataset-shift or acquisition-statistics analysis
- qualitative error analysis and Grad-CAM visualization

## Research principles

This project will prioritize:

- **Independent dataset provenance** — avoid treating repackaged or overlapping data as separate domains.
- **Patient/group separation when verified metadata exist** — do not claim patient-level splitting when genuine grouping identifiers are unavailable.
- **Reproducibility** — fixed seeds, explicit configs, saved split manifests, and documented environments.
- **External validation** — evaluate models outside the dataset on which they were trained.
- **Clinically meaningful metrics** — AUROC, sensitivity, specificity, precision, F1, and confusion matrices rather than accuracy alone.
- **Transparent limitations** — this is retrospective research and is not intended for clinical deployment.

## Repository structure

```text
tb-cross-dataset-generalization/
├── configs/              # Experiment and training configurations
├── data/                 # Local data location (raw datasets are NOT committed)
│   ├── manifests/        # Dataset/split CSVs or metadata that may be safely versioned
│   └── README.md         # Dataset setup and provenance notes
├── docs/                 # Research notes, protocol, dataset audit, project documentation
├── notebooks/            # Exploratory analysis only; production logic belongs in src/
├── results/              # Generated metrics/plots (large outputs ignored by Git)
├── src/                  # Reusable Python source code
│   ├── data/             # Dataset loading and preprocessing
│   ├── models/           # Model construction
│   ├── training/         # Training utilities
│   └── evaluation/       # Metrics and cross-dataset evaluation
├── .gitignore
├── CONTRIBUTING.md
├── requirements.txt
└── README.md
```

## Data policy

**Do not commit chest X-ray images or downloaded datasets to this repository.**

Raw data should remain local and must follow the licensing and usage requirements of each source dataset. Only lightweight metadata or split manifests should be versioned when redistribution is permitted.

Before using a dataset, record:

1. original source and citation,
2. institution / population where known,
3. label definition,
4. number of patients and images,
5. patient identifiers or grouping information,
6. licensing / redistribution terms,
7. known duplicates, derivatives, or overlap with other datasets.

## Reproducibility

Experiment-specific choices should be stored in `configs/` rather than hard-coded into notebooks. Random seeds, dataset splits, model settings, preprocessing, and evaluation procedures should be documented so that another team member can reproduce the same experiment.

## Team

- **HANG Sothrithisak**
- **Jothimuni Vinod Avishka Mendis**

Course project: **AIN 320 — Computer Vision**

## Disclaimer

This repository is an academic computer-vision project. Models developed here are **not medical devices and must not be used for clinical diagnosis or patient care**.
