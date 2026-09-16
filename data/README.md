# Data

Raw chest X-ray datasets are intentionally **not stored in this repository**.

Use this directory for lightweight, redistributable metadata and manifests only.

Recommended local layout:

```text
data/
├── raw/          # Original downloaded datasets (ignored by Git)
├── interim/      # Temporary conversions / cleaned metadata (ignored)
├── processed/    # Model-ready local data (ignored)
├── external/     # External validation datasets if kept separately (ignored)
└── manifests/    # Versioned split/index CSVs when redistribution is permitted
```

Before adding any manifest or metadata file, confirm that its redistribution is allowed by the source dataset's terms.

Dataset provenance and suitability decisions belong in `docs/dataset-audit.md`.
