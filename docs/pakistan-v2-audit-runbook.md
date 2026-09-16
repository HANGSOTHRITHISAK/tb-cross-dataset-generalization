# Pakistan Mendeley V2 Audit Runbook

> Working runbook for the Kiran/Jabeen Pakistani TB chest X-ray cohort. This does **not** clear the dataset for training; it defines the reproducible steps required to decide its role.

## Pinned source

- Dataset: **Dataset of Tuberculosis Chest X-rays Images**
- Authors: Saira Kiran, Ishrat Jabeen
- Version: **V2**
- DOI: `10.17632/8j2g3csprk.2`
- Published: 28 May 2024
- License: CC BY 4.0
- Published counts: 3,008 images total
  - 2,494 TB
  - 514 normal
- Public description: images collected from a local hospital in Pakistan and resized to a uniform dimension.

V2 is pinned for the first audit because the 2026 acquisition-confounding preprint explicitly used this version and released reproducibility artifacts against it. V3 should be treated as a separate version until file identity is verified.

## Local placement

Do not commit the images.

Preferred local layout after extracting the original V2 download:

```text
data/raw/mendeley_pk_v2/
├── TB Chest X-rays/
└── Normal Chest X-rays/
```

If the archive ships inside an extra top-level directory such as `Kiran:Jabeen`, either point the audit command at that directory directly or move the two class directories under `data/raw/mendeley_pk_v2/`. Do not rename individual images before the initial audit.

## First-pass command

From the repository root:

```bash
python -m src.data.audit data/raw/mendeley_pk_v2 \
  --manifest results/audits/mendeley_pk_v2_manifest.csv \
  --summary results/audits/mendeley_pk_v2_summary.md
```

Expected first-pass checks:

- exactly 3,008 decodable images unless the original distribution has changed
- class-folder counts consistent with 2,494 TB / 514 normal
- no unexpected image formats or decode failures
- exact SHA-256 duplicate groups inspected
- identical dHash groups inspected as near-duplicate candidates
- class-conditional aspect ratio, file size, intensity, border, sharpness and histogram differences reviewed

## Patient / study identity gate

Do **not** assume one image equals one patient.

The public reproducibility repository accompanying the 2026 confound audit constructs a `patient_id` from each image filename stem. That is useful as a row identifier but is not evidence of real patient-level metadata. Search the original V2 download for metadata files, manifests, README files, filenames or directory conventions that establish genuine patient/study grouping.

If true grouping cannot be recovered:

1. document that limitation explicitly;
2. do not call an image-level split a patient-level split;
3. prefer using this cohort as an untouched external domain rather than relying on an internal split for strong generalization claims.

## Acquisition-confound gate

The public dataset description does not separately document the acquisition source of TB and normal images. Therefore inspect whether low-level characteristics differ strongly by class.

Priority signals from our audit manifest:

- `aspect_ratio`
- `file_bytes`
- `mean_intensity` / `std_intensity`
- `border_mean`
- `sharpness_laplacian_var`
- `hist_bin_00` ... `hist_bin_15`

Large standardized differences are **screening signals**, not proof of confounding. If they appear, inspect representative images and source documentation before deciding whether the dataset can serve as a development domain.

## Cross-dataset overlap gate

Our own downloaded copy must eventually be compared against every dataset used in the experiment. The external 2026 audit reports zero overlap between its Pakistani V2 holdout and its training corpus under its own exact/perceptual-hash policy, but that result does not replace hashing the precise files used in this project.

## Clearance outcomes

After the audit, assign one of these outcomes in `docs/dataset-audit.md`:

- **Clear for development** — provenance/metadata are adequate and no material shortcut/leakage problem is found.
- **Conditional** — usable with explicit limitations or preprocessing controls.
- **External-only** — useful as a held-out domain but not defensible for internal development/splitting.
- **Reject from core** — leakage, unverifiable structure, or acquisition confounding makes the core claim unreliable.

## Do not do yet

Until the audit is complete:

- do not tune DenseNet-121 on this cohort;
- do not create a final train/validation/test split;
- do not rebalance or preprocess away suspicious artifacts before recording the original distributions;
- do not use external test performance for model selection.
