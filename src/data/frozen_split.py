"""Read frozen assignments without deduplicating or generating new splits."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path, PurePosixPath

from .labels import TBX11K_INTERNAL_CLASS_NAMES, normalize_tbx11k_internal_label
from .manifest import file_sha256, load_manifest
from .schema import SampleRecord
from .tbx11k_split import SPLIT_NAMES, is_primary_tbx11k_record


def load_frozen_tbx11k(
    manifest: str | Path,
    assignments: str | Path,
    *,
    image_root: str | Path | None = None,
    summary: str | Path | None = None,
    expected_split_sha256: str | None = None,
) -> list[SampleRecord]:
    """Strict ID join; extra audited rows are intentionally outside the retained pool.

    image_root, when supplied, is the release's imgs directory. Relative release
    paths come from audited metadata, never from a newly inferred split.
    No image bytes are opened here (including held-out images).
    """
    manifest, assignments = Path(manifest), Path(assignments)
    if expected_split_sha256 and file_sha256(assignments) != expected_split_sha256:
        raise ValueError("Frozen split file hash mismatch")
    records = {record.sample_id: record for record in load_manifest(manifest)}
    with assignments.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "original_label",
            "internal_label",
            "internal_class",
            "canonical_binary_label",
            "split",
            "checksum_sha256",
            "duplicate_family_id",
            "duplicate_family_size",
            "duplicate_member_ids",
        }
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("Ambiguous duplicate split columns")
        if not required.issubset(fields):
            raise ValueError("Frozen split missing required columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Frozen split is empty")
    root = Path(image_root).resolve() if image_root is not None else None
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    seen_families: set[str] = set()
    seen_members: set[str] = set()
    counts: Counter[tuple[str, str]] = Counter()
    joined = []
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed frozen split row")
        sample_id = row["sample_id"]
        if not sample_id or sample_id != sample_id.strip() or sample_id in seen_ids:
            raise ValueError("Missing or duplicate split sample_id")
        seen_ids.add(sample_id)
        if sample_id not in records:
            raise ValueError(
                f"Split sample ID missing from audited manifest: {sample_id}"
            )
        record = records[sample_id]
        if not is_primary_tbx11k_record(record):
            raise ValueError("Split references a non-primary TBX11K record")
        label = normalize_tbx11k_internal_label(record.original_label)
        expected_folder = {0: "health", 1: "sick", 2: "tb"}[label]
        if not sample_id.startswith(f"tbx11k:{expected_folder}/"):
            raise ValueError("Sample ID contradicts primary released label")
        binary = int(label == 2)
        if (
            row["original_label"] != record.original_label
            or int(row["internal_label"]) != label
            or row["internal_class"] != TBX11K_INTERNAL_CLASS_NAMES[label]
            or int(row["canonical_binary_label"]) != binary
            or record.canonical_label != binary
        ):
            raise ValueError(f"Inconsistent labels for {sample_id}")
        checksum = row["checksum_sha256"]
        if (
            not re.fullmatch(r"[0-9a-f]{64}", checksum)
            or checksum != record.checksum_sha256
        ):
            raise ValueError(f"Image checksum mismatch for {sample_id}")
        if checksum in seen_hashes:
            raise ValueError("Duplicate retained image checksum")
        seen_hashes.add(checksum)
        if row["split"] not in SPLIT_NAMES:
            raise ValueError("Invalid frozen split assignment")
        members = json.loads(row["duplicate_member_ids"])
        if (
            not isinstance(members, list)
            or not members
            or any(not isinstance(member, str) or not member for member in members)
            or len(members) != len(set(members))
            or len(members) != int(row["duplicate_family_size"])
            or sample_id != min(members)
            or sample_id not in members
        ):
            raise ValueError("Invalid duplicate family membership")
        family = (
            "dup-"
            + hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()[:16]
        )
        if row["duplicate_family_id"] != family or family in seen_families:
            raise ValueError("Invalid or duplicate retained family")
        if seen_members.intersection(members):
            raise ValueError("Duplicate member appears in multiple retained families")
        seen_members.update(members)
        seen_families.add(family)
        for member in members:
            other = records.get(member)
            if other is None or not is_primary_tbx11k_record(other):
                raise ValueError(
                    "Duplicate member missing from primary audited manifest"
                )
            if (
                normalize_tbx11k_internal_label(other.original_label) != label
                or other.canonical_label != binary
            ):
                raise ValueError("Conflicting duplicate member labels")
        path = record.image_path
        if root is not None:
            relative = PurePosixPath(
                str(record.metadata.get("relative_release_path", ""))
            )
            if (
                not relative.parts
                or relative.is_absolute()
                or ".." in relative.parts
                or "\\" in str(relative)
                or ":" in str(relative)
                or sample_id != f"tbx11k:{relative.as_posix()}"
            ):
                raise ValueError("Unsafe or inconsistent relative_release_path")
            path = root.joinpath(*relative.parts).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Image path escapes image_root")
        counts[(row["split"], row["internal_class"])] += 1
        joined.append(replace(record, image_path=path, split=row["split"]))
    if summary is not None:
        evidence = json.loads(Path(summary).read_text(encoding="utf-8"))
        if evidence["inputs"]["manifest_sha256"] != file_sha256(manifest):
            raise ValueError("Audited manifest hash mismatch with frozen summary")
        if evidence["retained_unique_images"] != len(joined):
            raise ValueError("Frozen split count mismatch")
        actual = {
            split: {
                name: counts[(split, name)]
                for name in TBX11K_INTERNAL_CLASS_NAMES.values()
            }
            for split in SPLIT_NAMES
        }
        if evidence["split_class_counts"] != actual:
            raise ValueError("Frozen split class counts mismatch")
    return joined
