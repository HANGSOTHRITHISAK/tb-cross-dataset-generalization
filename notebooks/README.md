# Notebooks

Use notebooks for exploration, sanity checks, visualization, and analysis.

Keep reusable project logic in `src/` so experiments do not depend on hidden notebook state.

Suggested progression:

- `00_dataset_audit.ipynb` — inspect candidate datasets and metadata
- `01_data_sanity_checks.ipynb` — class balance, dimensions, duplicates, view positions
- `02_baseline_training.ipynb` — optional thin front end for the training pipeline
- `03_cross_dataset_evaluation.ipynb` — compare in-domain and external performance
- `04_error_analysis.ipynb` — qualitative analysis / Grad-CAM if pursued

Notebook names are suggestions only; do not create empty notebooks just to fill the directory.
