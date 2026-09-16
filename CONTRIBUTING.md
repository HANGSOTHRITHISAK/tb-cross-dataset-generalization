# Contributing

This is a small collaborative research project. The goal of the workflow is to keep experiments reproducible without creating unnecessary process overhead.

## Workflow

1. Pull the latest `main` before starting work.
2. Create a short-lived branch for meaningful changes, for example:
   - `data/dataset-audit`
   - `feat/dataloader`
   - `exp/densenet-baseline`
   - `docs/methodology`
3. Keep commits focused and descriptive.
4. Open a pull request into `main` for substantial code, experiment, or protocol changes.
5. Have the other team member review changes that affect dataset splits, preprocessing, evaluation, or the experimental protocol.

Small documentation fixes may be committed directly to `main` while the project is in its early setup stage.

## Commit style

Prefer concise prefixes:

- `feat:` new functionality
- `fix:` bug fix
- `data:` dataset metadata / split work
- `exp:` experiment configuration or experiment-specific work
- `docs:` documentation
- `refactor:` internal code improvement
- `chore:` tooling or repository maintenance

## Data safety

Never commit:

- raw chest X-ray datasets,
- patient-identifiable information,
- credentials or API keys,
- large model checkpoints,
- outputs whose redistribution terms are unclear.

Before committing manifests or metadata, confirm that the source dataset permits redistribution of that information.

## Experiment integrity

Changes to train/validation/test splits must be explicit and reviewable. Patient-level grouping should be used whenever patient identifiers are available. Do not tune hyperparameters against the external test set.

Notebooks are intended for exploration and visualization. Reusable data loading, preprocessing, training, and evaluation logic should live under `src/`.
