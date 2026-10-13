# Source code

Reusable implementation lives here.

Implemented modules:

- `data/` — audit/manifest tools, strict frozen-split join, dataset loading and transforms
- `models/` — ImageNet DenseNet-121 with a three-class head
- `training/` — bounded one-step smoke execution and monitoring

Full training/checkpoint selection and `evaluation/` metrics, thresholding,
cross-dataset evaluation and bootstrap intervals remain pending.

Keep experiment-specific settings in `configs/` rather than hard-coding them in source files.
