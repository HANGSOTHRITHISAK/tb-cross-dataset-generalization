# Phase 2 local audit status

This branch imports reproducible, file-level audit evidence from the local
implementation. It does not alter the canonical experimental protocol or make
an overlap-adjudication, duplicate-handling, or split decision.

The versioned manifests and reports contain metadata, paths relative to the
repository data layout, checksums, image statistics, and audit findings. They
do not contain raw radiographs. Raw datasets remain ignored under `data/raw/`.

## Scope of the imported evidence

- Shenzhen and Montgomery acquisition provenance, per-image manifests, and
  image-integrity audits.
- TBX11K archive provenance, manifest, annotation inventory, integrity audit,
  and cross-dataset exact/perceptual candidate reports.
- Reproducible exclusion and pending-review lists. These lists are evidence
  artifacts, not a finalized cleaning policy.

## Required review before methodological use

- Manually adjudicate the TBX11K perceptual candidates against Shenzhen and
  Montgomery.
- Decide how to handle internal TBX11K duplicate groups.
- Decide the split policy only after the overlap decision is documented.
- Do not use Shenzhen or Montgomery to tune any development choice.

The imported TBX11K manifest represents the release's unreleased-test labels
as blank canonical labels. The typed manifest contract permits this explicit
unknown state and dataset consumers must reject it if a labeled sample is
required.