# Experimental Protocol

> Status: draft. Decisions will be locked after the dataset audit.

## Research question

How much does a tuberculosis classifier trained on one chest X-ray dataset degrade when evaluated on an independent dataset collected from a different clinical source under a standardized preprocessing and evaluation pipeline?

## Primary study design

For two independent datasets, `A` and `B`:

1. Train on the training partition of `A`.
2. Select model/hyperparameters using only the validation partition of `A`.
3. Evaluate the final model on the held-out test partition of `A` (in-domain).
4. Without retraining or tuning on `B`, evaluate the same model on the designated external test data from `B`.
5. Repeat symmetrically with `B` as the training source and `A` as the external domain.

Core matrix:

| Training domain | In-domain evaluation | External evaluation |
|---|---|---|
| A | A-test | B-external |
| B | B-test | A-external |

## Potential extension

If time and dataset provenance permit, add a third independent dataset `C` and evaluate models trained on `A`, `B`, and/or `A+B` against `C` as a held-out external domain.

## Leakage controls

- Split by patient rather than image whenever patient identifiers are available.
- Investigate duplicates and derivative/repackaged datasets before defining domains.
- Do not use the external test domain for hyperparameter selection.
- Keep final test sets untouched until the training procedure is established.

## Baseline model

A pretrained convolutional neural network will be selected after the dataset audit. DenseNet-121 is a candidate baseline because of its established use in chest X-ray research, but architecture choice is not the primary research question.

The initial project should avoid an unnecessary architecture race. A second architecture can be added later to test whether the observed generalization gap is architecture-specific.

## Preprocessing decisions to lock

- image resolution,
- grayscale handling / channel conversion,
- intensity scaling and normalization,
- augmentation policy,
- class imbalance strategy,
- image-view inclusion criteria,
- patient-level split procedure.

These choices must be applied consistently enough that preprocessing differences do not accidentally dominate the cross-dataset comparison.

## Primary evaluation

Planned metrics:

- AUROC,
- sensitivity (recall),
- specificity,
- precision,
- F1 score,
- confusion matrix.

Where feasible, uncertainty estimates such as bootstrap confidence intervals will be added to the final evaluation.

The primary quantity of interest is the **generalization gap** between in-domain and external-domain performance.

## Optional analysis

After the baseline experiment is complete:

- pooled-domain training,
- stronger augmentation / robustness intervention,
- dataset-level image statistics and shift visualization,
- subgroup analysis when reliable metadata permits,
- error analysis,
- Grad-CAM as qualitative visualization only.

## Reproducibility

Each experiment should record:

- random seed,
- dataset/split manifest version,
- model architecture and initialization,
- optimizer and learning-rate settings,
- preprocessing and augmentation,
- epoch / early-stopping policy,
- software environment,
- final metrics and checkpoint identifier.

This document should be updated when design decisions are finalized rather than allowing the implementation to become the only record of the protocol.
