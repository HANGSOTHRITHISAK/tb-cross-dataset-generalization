# Working Decisions

> This file records provisional research decisions so that the project can move forward without pretending that exploratory choices are final.

## 2026-09-16 — Dataset strategy v1

### Decision

Proceed with a provenance-first design. The current leading primary training source is the Kiran/Jabeen Pakistani hospital TB CXR dataset (`10.17632/8j2g3csprk`), subject to file-level auditing. Preserve Shenzhen and Montgomery County as external domains. Do not use TBX11K or the Rahman composite as independent external datasets against Shenzhen/Montgomery unless source filtering and deduplication demonstrate independence.

### Why

The project studies cross-dataset generalization, so dataset independence is part of the experimental validity rather than merely a data-engineering detail. Larger composite datasets are attractive computationally but can silently contain images from the same source collections used for external testing.

### Confidence

**Medium.** Public provenance is sufficient to justify this as the working direction, but the Pakistani dataset has not yet passed patient-ID, duplicate, diagnostic-reference, and class-conditional acquisition audits.

### What can proceed now

- data-manifest schema
- duplicate/hash utilities
- image-statistics audit tooling
- generic PyTorch dataset interfaces
- reproducible configuration/logging scaffolding

### What should wait

- final train/validation/test split
- final dataset declaration
- final preprocessing choices
- full baseline training
- claims about external generalization

### Next decision gate

Download/inspect the candidate source files and determine whether the Pakistani dataset provides adequate patient grouping and whether TB/normal classes show obvious acquisition or preprocessing confounding.

---

## 2026-09-16 — Dataset strategy v2

### Trigger

A newly posted 2026 medRxiv preprint and its public reproducibility repository provide substantially stronger evidence about acquisition confounding in the same five open TB CXR corpora we are considering. This is external evidence, not a result produced by this project, and the preprint is not yet peer reviewed.

### Revised decision

Do **not** pre-commit the Kiran/Jabeen Pakistani dataset as the primary training domain. Keep it as a high-priority independent domain, but decide its role only after our own file-level audit.

For the course MVP, preserve a simple cross-dataset design built from the cleanest defensible pair available after auditing. Shenzhen remains attractive because its public provenance documents both classes coming from the same hospital/routine acquisition window. Montgomery remains useful as a small external stress test rather than a primary training source. The Pakistani cohort may become an external domain, a training domain with explicit confound caveats, or a bias-analysis case depending on our own findings.

### Evidence that changed the decision

- The external audit used Mendeley **V2** (`10.17632/8j2g3csprk.2`) and reports zero exact/near-duplicate overlap between the 3,008 Pakistani images and its 13,260-image training corpus under its published MD5 + pHash policy.
- That same work reports extremely strong class separability from features that should be treated as shortcut/confound evidence rather than proof of pathology recognition. In its released robustness results, the Pakistani cohort reaches AUROC 0.988 using low-dimensional image statistics and approximately 1.0 using several frozen/low-spatial-content representations.
- The Pakistani dataset's public documentation does not provide a per-class acquisition breakdown, device information, diagnostic reference standard, or verified patient grouping. A third-party manifest creates `patient_id` values directly from image stems; these are not evidence of true patient identifiers.
- Mendeley V2 and V3 have the same published class counts and description, but file equality has not been established. For reproducibility, use **V2 first** because the 2026 audit explicitly names that version; compare V3 only after hashes/file manifests can establish whether it materially differs.

### What this means for our contribution

We should not turn the course project into a reproduction of the confound-audit preprint. Our core contribution remains a transparent, reproducible **cross-dataset generalization experiment** with a fixed baseline and carefully audited domain boundaries. A lightweight shortcut/acquisition diagnostic can strengthen the interpretation of any performance drop, but it stays secondary to the cross-dataset evaluation matrix.

### Confidence

**High** that the project must remain confound-aware and provenance-first. **Medium** on final dataset roles until we inspect the original image distributions ourselves.

### Next decision gate

1. Acquire the original Kiran/Jabeen **V2** files.
2. Run our own manifest/hash/image-statistics tooling.
3. Verify whether any real patient/study identifiers exist rather than assuming one image equals one patient.
4. Compare class-conditional border, sharpness, histogram/intensity, dimensions, format, and compression characteristics.
5. Only then lock the training and external-test domains.

---

## 2026-09-16 — Dataset strategy v3: proposal-aligned core, flexible extensions

### Trigger

The accepted course proposal already defines a useful experimental spine around TBX11K with Shenzhen and Montgomery as cross-dataset tests. The proposal is not immutable, but there is no empirical reason to replace that core before first testing whether the known overlap can be controlled directly.

### Revised working decision

Return **TBX11K to the main development/training role**, subject to a mandatory overlap-cleaning gate. Keep **Shenzhen** and **Montgomery County** as the primary external domains from the proposal.

The core experiment is therefore:

- cleaned TBX11K -> held-out TBX11K internal test
- cleaned TBX11K -> Shenzhen external test
- cleaned TBX11K -> Montgomery external stress test

The Pakistani Mendeley cohort remains an optional additional independent domain or confound-analysis extension after its own audit; it does not currently replace TBX11K.

### Why this is defensible

Public documentation explicitly warns that TBX11K overlaps the NLM TB datasets containing Shenzhen and Montgomery. That makes a naive train/test pairing invalid, but it does **not** automatically make TBX11K unusable. The correct next question is whether direct hashing/perceptual comparison can identify the overlapping rows and allow us to construct a cleaned TBX11K development pool while leaving the external sets untouched.

This keeps the project close to the accepted proposal while improving its methodology. If the overlap cannot be cleanly controlled, redesign remains available as a documented fallback rather than a premature pivot.

### What can proceed now

- download/audit TBX11K, Shenzhen and Montgomery
- generate per-dataset manifests
- run exact SHA-256 and perceptual-hash cross-dataset comparisons
- create a versioned exclusion manifest for confirmed TBX11K overlaps
- inspect TBX11K class balance after cleaning
- then lock the train/validation/internal-test split

### What still waits

- final baseline training
- final external evaluation
- any claim that Shenzhen/Montgomery are unseen relative to TBX11K
- any expansion to Pakistan as a core domain

### Next decision gate

Complete Issue #8: quantify TBX11K overlap with Shenzhen/Montgomery, manually review near-duplicate candidates, and determine whether the cleaned TBX11K pool remains suitable for the core experiment.


---

## 2026-09-23 — Label design v4: three-class internal, binary external

### Trigger

The accepted proposal called for multi-class TBX11K classification followed by broader TB-vs-non-TB external evaluation. The completed dataset audit showed that the released TBX11K train/validation path/list files expose labeled content as `health/`, `sick/`, and a single `tb/` category. The exact Active/Latent/Active+Latent subtype for the 800 labeled TB train/validation images is not recoverable from the release and will not be inferred.

The lecturer approved adjusting the internal label space to match the labels that are actually recoverable from the audited release.

### Locked decision

The project will use:

- **Internal TBX11K task:** Healthy / Sick non-TB / TB
- **Internal model:** ImageNet-pretrained DenseNet-121 with a 3-class head
- **External harmonization:** Healthy + Sick non-TB -> non-TB; TB -> TB
- **External datasets:** Shenzhen and Montgomery County, held out from all model-development decisions

The existing binary canonical labels in the audit manifests remain valid as the cross-dataset harmonized label space. A separate explicit TBX11K internal mapping is used for the 3-class task.

### Why

This preserves the multi-class intent of the accepted proposal without fabricating unavailable TB subtype labels. It also makes the healthy-versus-sick-negative distinction directly measurable while retaining a compatible TB/non-TB endpoint for Shenzhen and Montgomery.

### Scope boundary

Do not redesign the project around unreleased subtype labels. Do not use external datasets for tuning. Finish the duplicate-safe TBX11K split and reproducible baseline before optional acquisition-shift, Grad-CAM, augmentation, or architecture-comparison extensions.
