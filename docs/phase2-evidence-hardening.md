# Phase 2 evidence-hardening review

**Date:** 2026-09-24  
**Scope:** bounded methodological review before the TBX11K split is frozen

## Question

Does the current Phase 2 protocol require a concrete methodological adjustment before the proposed 70/15/15, seed-42 split is frozen?

## Verified repository evidence

The versioned audit artifacts report:

- 8,400 images in the primary labeled TBX11K development pool;
- 126 exact SHA-256 duplicate groups involving 252 rows;
- 8,274 unique SHA-256 checksums before any perceptual-candidate adjudication;
- 27 internal dHash candidate pairs produced by shared 32-bit dHash-band screening with Hamming distance at most 8;
- 17 candidate pairs with both images in the primary labeled pool, all within `sick/` (Sick non-TB);
- 10 candidate pairs joining a primary-pool Sick non-TB image to unreleased `test/` content.

The source of truth for these counts is [`data/audit/tbx11k_audit.json`](../data/audit/tbx11k_audit.json). dHash candidates are screening hits, not confirmed duplicates. The repository does not contain the raw radiographs, so this review did **not** visually adjudicate any pair.

## Targeted evidence conclusion

The current design is methodologically sound and does not require a change to the research question, dataset roles, 3-class label design, DenseNet-121 baseline, proposed split ratio, or seed.

One small hardening is required before the split can be frozen:

1. manually compare the 17 primary-pool dHash candidate pairs using the original radiographs;
2. classify each pair as either a confirmed transformed/re-exported copy or a similar-but-distinct radiograph;
3. treat confirmed copies as one duplicate family and retain one deterministic representative before splitting;
4. leave similar-but-distinct radiographs as separate samples;
5. preserve the adjudication result as a versioned, non-image evidence artifact.

The 10 candidate pairs involving unreleased TBX11K test content do not affect the primary-pool split because that test content is outside the development pool. They should remain documented but do not block this split freeze.

This is deliberately narrower than automatic perceptual deduplication. Perceptual hashing can surface plausible candidates, but it cannot by itself establish image identity.

## Explicit protocol controls

The final split/training protocol should state:

- all data-derived preprocessing parameters are fitted on the training partition only and then applied unchanged to tuning/validation, held-out internal-test, and external-test data;
- stochastic augmentation is applied only while loading the training partition;
- after exact-duplicate handling and adjudication of the 17 primary-pool perceptual candidates, the project may claim **image-level split independence after duplicate controls**;
- the project must not claim patient-level independence because trustworthy patient/group identifiers are unavailable;
- dataset roles are named **training**, **tuning/validation**, **held-out internal testing**, and **external testing**;
- neither the held-out internal test set nor Shenzhen/Montgomery may influence fitting, early stopping, hyperparameter selection, checkpoint selection, threshold selection, or preprocessing choices.

## Evidence basis

- CLAIM 2024 recommends reporting the level at which partitions are disjoint and states that medical-image partitions generally should be disjoint at patient level or higher. It also recommends the terms internal testing and external testing because “validation” is ambiguous.  
  Tejani AS et al. *Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update.* Radiology: Artificial Intelligence. 2024. https://doi.org/10.1148/ryai.240300
- A radiology data-handling review explains that image-level splitting can leak correlated examinations and recommends patient-level splitting when verified patient identifiers exist.  
  Rouzrokh P et al. *Mitigating Bias in Radiology Machine Learning: 1. Data Handling.* Radiology: Artificial Intelligence. 2022. https://doi.org/10.1148/ryai.210290
- Empirical medical-imaging evidence shows that non-independent splitting can materially inflate reported performance.  
  Tampu IE et al. *Inflation of test accuracy due to data leakage in deep learning-based classification of OCT images.* Scientific Data. 2022. https://doi.org/10.1038/s41597-022-01618-6
- Cross-hospital chest-radiograph performance can vary substantially, and source hospital is itself highly predictable from images, supporting untouched external testing and careful interpretation of acquisition shift.  
  Zech JR et al. *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs.* PLOS Medicine. 2018. https://doi.org/10.1371/journal.pmed.1002683
- Chest-radiograph models can learn acquisition/source shortcuts instead of pathology, reinforcing the need to keep external datasets out of development decisions.  
  DeGrave AJ et al. *AI for radiographic COVID-19 detection selects shortcuts over signal.* Nature Machine Intelligence. 2021. https://doi.org/10.1038/s42256-021-00338-7
- The original TBX11K publication establishes the dataset and its intended TB computer-aided-diagnosis setting.  
  Liu Y et al. *Rethinking Computer-Aided Tuberculosis Diagnosis.* CVPR. 2020. https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Rethinking_Computer-Aided_Tuberculosis_Diagnosis_CVPR_2020_paper.html
- The original NLM publication documents Shenzhen and Montgomery as distinct public TB chest-radiograph collections.  
  Jaeger S et al. *Two public chest X-ray datasets for computer-aided screening of pulmonary diseases.* Quantitative Imaging in Medicine and Surgery. 2014. https://doi.org/10.3978/j.issn.2223-4292.2014.11.20
- Recent TB-specific acquisition-confounding evidence reaches the same caution but is a **preprint** and is treated as supporting, not controlling, evidence.  
  Bilal A et al. *Auditing Class-Conditional Acquisition Confounding in Open Tuberculosis Chest X-Ray Benchmarks.* medRxiv preprint. 2026. https://doi.org/10.64898/2026.08.14.26360390

## Stop decision

No broader dataset-audit research is warranted now. The remaining evidence task is the bounded visual adjudication of the 17 primary-pool candidate pairs, followed by the already planned duplicate-safe split implementation and verification.
