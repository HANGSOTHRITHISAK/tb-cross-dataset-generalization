# Data Directory

Raw chest X-ray datasets are intentionally excluded from Git.

Recommended local layout:

```text
data/
├── raw/          # Original downloaded datasets; never modify in place
├── interim/      # Intermediate conversions / temporary preprocessing
├── processed/    # Model-ready derived data when needed
├── external/     # External-domain data if kept separately
└── manifests/    # Versionable split/index metadata when redistribution permits
```

## Rules

- Never commit raw medical images.
- Preserve original downloads unchanged under `raw/`.
- Keep dataset-specific provenance and setup instructions in `docs/dataset_audit.md`.
- Prefer reproducible preprocessing over manually edited image folders.
- Split at patient level whenever patient identifiers are available.
- Do not tune models against the external test set.
- Before committing manifests, confirm that their contents can legally be redistributed.

Dataset download instructions will be added after the dataset audit is complete.
