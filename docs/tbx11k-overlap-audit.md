# TBX11K -> Shenzhen / Montgomery Overlap Audit

> Working runbook. This preserves the proposal-aligned experiment while adding the leakage controls required by the dataset provenance.

## Working experiment

The main course-project path remains:

1. develop the TB classifier on **TBX11K**;
2. evaluate internally on a held-out TBX11K split;
3. freeze the selected model/protocol;
4. evaluate externally on **Shenzhen** and **Montgomery County**.

The proposal is a direction rather than an immutable contract, so additional datasets or analyses may be added later. The core experiment should not be changed without an empirical reason.

## Why an overlap audit is mandatory

Current public TorchXRayVision documentation explicitly warns that TBX11K overlaps its NLM TB dataset containing Montgomery and Shenzhen images. It also states that TBX11K incorporates images from four older TB collections, including Montgomery County and Shenzhen.

Therefore, a naive TBX11K-train -> Shenzhen/Montgomery-test experiment can leak external-test images into training. The solution is not to abandon the experiment; it is to construct a **cleaned TBX11K development pool** after direct file-level comparison.

## Required local inputs

Keep raw images outside Git. Suggested layout:

```text
data/raw/
├── tbx11k/
├── shenzhen/
└── montgomery/
```

Generate one manifest per dataset with `src.data.audit` before comparing datasets.

Example:

```bash
python -m src.data.audit data/raw/tbx11k \
  --manifest data/interim/tbx11k-manifest.csv \
  --summary data/interim/tbx11k-audit.md

python -m src.data.audit data/raw/shenzhen \
  --manifest data/interim/shenzhen-manifest.csv \
  --summary data/interim/shenzhen-audit.md

python -m src.data.audit data/raw/montgomery \
  --manifest data/interim/montgomery-manifest.csv \
  --summary data/interim/montgomery-audit.md
```

## Cross-dataset comparison

Run the exact/perceptual overlap checker twice:

```bash
python -m src.data.overlap \
  data/interim/tbx11k-manifest.csv \
  data/interim/shenzhen-manifest.csv \
  --left-name TBX11K \
  --right-name Shenzhen \
  --matches data/interim/tbx11k-vs-shenzhen.csv \
  --summary data/interim/tbx11k-vs-shenzhen.md

python -m src.data.overlap \
  data/interim/tbx11k-manifest.csv \
  data/interim/montgomery-manifest.csv \
  --left-name TBX11K \
  --right-name Montgomery \
  --matches data/interim/tbx11k-vs-montgomery.csv \
  --summary data/interim/tbx11k-vs-montgomery.md
```

The default dHash Hamming threshold is deliberately only a screening threshold. Exact SHA-256 matches are strong duplicate evidence; perceptual-hash matches must be visually inspected before exclusion.

## Cleaning rule

Create an exclusion manifest rather than deleting raw data. For each confirmed overlap, record:

- TBX11K relative path / identifier
- matched external dataset and path
- exact or perceptual evidence
- perceptual Hamming distance when relevant
- manual-review decision
- exclusion reason

The training/index builder should exclude confirmed overlaps from **every TBX11K development split**. Shenzhen and Montgomery themselves remain untouched external sets.

## Split rule after cleaning

Only after overlap removal:

- create TBX11K train / validation / internal-test partitions;
- preserve patient/group integrity if genuine grouping metadata are available;
- never use Shenzhen or Montgomery for early stopping, hyperparameter selection, threshold optimization, or checkpoint selection;
- keep the external-test preprocessing fixed before observing external performance.

## Decision outcomes

**Proceed as proposed:** overlap can be identified and removed while leaving a useful TBX11K development cohort.

**Proceed with caveat:** some provenance cannot be fully reconstructed, but direct duplicate controls substantially reduce known leakage risk; document the residual limitation.

**Redesign core domains:** only if the downloaded data make a defensible TBX11K -> Shenzhen/Montgomery evaluation impossible. This is an empirical fallback, not the current plan.

## Optional extensions

The Pakistani Mendeley cohort remains useful as a possible additional independent-domain stress test or acquisition-confounding analysis after its own audit. It does not currently replace TBX11K in the course-project core.
