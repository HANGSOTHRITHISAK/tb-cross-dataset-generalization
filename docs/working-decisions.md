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
