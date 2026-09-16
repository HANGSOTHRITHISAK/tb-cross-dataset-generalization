# Source code

Reusable implementation lives here.

Planned modules:

- `data/` — dataset adapters, manifests, transforms, split utilities
- `models/` — model construction and checkpoint loading
- `training/` — training loops, losses, optimization, reproducibility helpers
- `evaluation/` — metrics, thresholding, cross-dataset evaluation, bootstrap intervals

Keep experiment-specific settings in `configs/` rather than hard-coding them in source files.
