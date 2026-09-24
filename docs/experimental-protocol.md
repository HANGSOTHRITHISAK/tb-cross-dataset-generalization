# Experimental Protocol

**Status:** approved project design, updated 2026-09-23.

## Primary research question

How well does a DenseNet-121 tuberculosis chest X-ray classifier trained on TBX11K retain performance when evaluated on independent external datasets collected from different clinical sources?

The project also asks a related internal question: how well can the model distinguish **healthy**, **sick non-TB**, and **TB** images within the released TBX11K training/validation label structure?

## Label design

### Internal TBX11K task

The released TBX11K train/validation material used by this project exposes three recoverable diagnostic groups:

1. **Healthy**
2. **Sick non-TB**
3. **TB**

The proposal originally anticipated finer TB subtype labels such as active and latent TB. The audited release path/list files expose the labeled TB images under a single `tb/` category, so subtype labels are **not inferred**.

The internal model is therefore a **3-class classifier**.

### External harmonization

Shenzhen and Montgomery do not expose the same 3-class structure. External evaluation therefore uses a harmonized binary label space:

- Healthy -> non-TB
- Sick non-TB -> non-TB
- TB -> TB

Shenzhen and Montgomery remain untouched external test datasets and are never used for training, early stopping, hyperparameter selection, checkpoint selection, or threshold tuning.

## Core experiment

1. Audit and clean the labeled TBX11K development pool.
2. Construct deterministic TBX11K train/validation/internal-test partitions after exact-duplicate handling.
3. Fine-tune an ImageNet-pretrained DenseNet-121 for 3-class classification.
4. Evaluate 3-class performance on the held-out TBX11K internal test set.
5. Freeze the selected model and all development decisions.
6. Harmonize the output to TB vs non-TB.
7. Evaluate externally on Shenzhen.
8. Evaluate externally on Montgomery County as an additional stress test.
9. Compare internal and external performance and analyze the generalization gap.

## Baseline model

The locked baseline architecture is **DenseNet-121** with ImageNet initialization and a 3-output classification head for the internal TBX11K task.

Architecture comparisons are optional and should not delay completion of the reproducible baseline.

## Internal evaluation

Report at minimum:

- accuracy,
- macro-F1,
- per-class recall/sensitivity,
- 3x3 confusion matrix.

Per-class results are important because the project specifically distinguishes healthy negatives from sick non-TB negatives.

## External binary evaluation

Report at minimum:

- AUROC,
- sensitivity / recall,
- specificity,
- precision,
- F1,
- confusion matrix.

Any decision threshold used for external classification must be selected using TBX11K development data only and then frozen before Shenzhen/Montgomery evaluation.

Where feasible, report bootstrap confidence intervals.

## Partition roles and terminology

- **Training:** fits model parameters and any data-derived preprocessing parameters.
- **Tuning/validation:** supports early stopping, hyperparameter selection, checkpoint selection, and any threshold selection.
- **Held-out internal testing:** estimates final in-domain performance after development decisions are frozen.
- **External testing:** Shenzhen and Montgomery estimate cross-dataset performance after the model and development protocol are frozen.

The held-out internal test set and both external test sets must not influence training, preprocessing choices, early stopping, hyperparameter selection, checkpoint selection, or threshold selection.

## Leakage and provenance controls

- Do not use Shenzhen or Montgomery for model-development decisions.
- Audit candidate datasets for exact and perceptual overlap before calling them independent domains.
- Deduplicate the primary labeled TBX11K pool by exact SHA-256 before splitting, after hard-failing on any checksum group with conflicting 3-class labels.
- Manually adjudicate the 17 primary-pool dHash candidate pairs listed in `data/audit/tbx11k_audit.json` before the split is frozen. The candidates are not confirmed duplicates. Confirmed transformed/re-exported copies are handled as one duplicate family; visually similar but distinct radiographs remain separate.
- Exact or confirmed transformed duplicate copies must not cross TBX11K training/tuning/internal-test partitions.
- Fit any data-derived preprocessing parameters on the training partition only, then apply them unchanged to all other partitions and datasets.
- Apply stochastic augmentation only to the training partition.
- Preserve split and adjudication manifests so experiments can be reproduced exactly.
- Claim only **image-level split independence after duplicate controls**. Do not claim patient-level independence because trustworthy patient/group identifiers are not available in the released TBX11K material currently used by the project.
- Do not infer unreleased TB subtype labels.

See `docs/phase2-evidence-hardening.md` for the verified counts, evidence basis, and bounded stop decision.

## Optional analysis after the baseline

If time permits:

- compare errors involving healthy versus sick non-TB negatives,
- analyze acquisition/intensity/resolution differences across datasets,
- perform qualitative error analysis or Grad-CAM,
- test one controlled augmentation or robustness intervention,
- compare one additional architecture.

## Reproducibility

Each experiment should record:

- config file,
- random seed,
- dataset and split manifest versions,
- label-space version,
- model architecture and initialization,
- preprocessing and augmentation,
- optimizer / learning rate / batch size / epoch count,
- checkpoint identifier,
- evaluation output.

## Scope control

The required core is **TBX11K 3-class internal classification plus frozen TB/non-TB external evaluation on Shenzhen and Montgomery**. Optional analyses should only be added after the baseline pipeline is complete and reproducible.
