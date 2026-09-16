# Experimental Protocol

## Primary research question

How much does a tuberculosis chest X-ray classifier's performance change when it is evaluated on an independent clinical dataset rather than the dataset on which it was trained?

## Minimum viable experiment

With two independent datasets, `A` and `B`:

| Training domain | Evaluation domain | Role |
|---|---|---|
| A | A | In-domain baseline |
| A | B | External validation |
| B | B | In-domain baseline |
| B | A | External validation |

The same model family, preprocessing policy, training procedure, and evaluation code should be used so that the comparison primarily reflects dataset shift rather than pipeline changes.

## Optional extensions

- Add a third independent dataset `C` as an untouched external test domain.
- Train on pooled `A + B` and evaluate on `C`.
- Compare a small number of model architectures.
- Compare standard and stronger augmentation.
- Analyze image acquisition / intensity / resolution differences across datasets.
- Perform qualitative error analysis and Grad-CAM visualization.

## Evaluation

Primary metrics:

- AUROC
- sensitivity / recall
- specificity
- precision
- F1 score
- confusion matrix

Accuracy may be reported as a supplementary metric but should not be the main measure when class prevalence differs across datasets.

Where feasible, report uncertainty with bootstrap confidence intervals.

## Leakage controls

- Split by patient rather than image whenever patient grouping is available.
- Do not use external test data for hyperparameter tuning or threshold selection.
- Audit candidate datasets for duplicate or repackaged images before calling them independent domains.
- Preserve split manifests so experiments can be reproduced exactly.

## Baseline model

The planned baseline is an ImageNet-pretrained DenseNet-121 adapted for binary classification. This is a starting point, not a commitment; the final model choice should be locked only after the dataset audit and implementation constraints are clear.

## Reproducibility

Each experiment should record:

- config file
- random seed
- dataset and split manifest versions
- model architecture and initialization
- preprocessing and augmentation
- optimizer / learning rate / batch size / epoch count
- checkpoint identifier
- evaluation output

## Scope control

The two-dataset cross-validation design is the required core. All additional experiments are optional and should only be added after the baseline pipeline is complete and reproducible.
