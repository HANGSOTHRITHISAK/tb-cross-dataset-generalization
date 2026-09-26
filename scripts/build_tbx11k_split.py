from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from src.data.labels import TBX11K_INTERNAL_CLASS_NAMES
from src.data.manifest import file_sha256, load_manifest
from src.data.tbx11k_split import (
    deduplicate_primary_tbx11k,
    load_adjudication,
    split_counts,
    stratified_split,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/tbx11k.csv"
ADJUDICATION = ROOT / "data/audit/tbx11k_primary_dhash_adjudication.csv"
EXCLUSIONS = ROOT / "data/audit/tbx11k_exclusions.csv"
OUT_MANIFEST = ROOT / "data/manifests/tbx11k_split.csv"
OUT_SUMMARY = ROOT / "data/audit/tbx11k_split_summary.json"
SEED = 42
RATIOS = (0.70, 0.15, 0.15)


def load_excluded_sample_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            (row.get("sample_id") or "").strip()
            for row in csv.DictReader(handle)
            if (row.get("sample_id") or "").strip()
        }


def write_split_manifest(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            writer.writerow(
                {
                    "sample_id": item.record.sample_id,
                    "original_label": item.record.original_label,
                    "internal_label": item.internal_label,
                    "internal_class": TBX11K_INTERNAL_CLASS_NAMES[item.internal_label],
                    "canonical_binary_label": item.record.canonical_label,
                    "split": item.record.split,
                    "checksum_sha256": item.record.checksum_sha256 or "",
                    "duplicate_family_id": item.duplicate_family_id,
                    "duplicate_family_size": item.duplicate_family_size,
                    "duplicate_member_ids": json.dumps(item.duplicate_member_ids),
                }
            )


def main() -> None:
    records = load_manifest(MANIFEST)
    confirmed_pairs = load_adjudication(ADJUDICATION, expected_rows=17)
    excluded = load_excluded_sample_ids(EXCLUSIONS)
    deduplicated = deduplicate_primary_tbx11k(
        records,
        confirmed_copy_pairs=confirmed_pairs,
        excluded_sample_ids=excluded,
    )
    assigned = stratified_split(deduplicated, seed=SEED, ratios=RATIOS)
    write_split_manifest(assigned, OUT_MANIFEST)

    family_sizes = Counter(item.duplicate_family_size for item in deduplicated)
    summary = {
        "dataset": "tbx11k",
        "split_policy": {
            "seed": SEED,
            "ratios": {
                "train": RATIOS[0],
                "validation": RATIOS[1],
                "internal_test": RATIOS[2],
            },
            "stratification": "released TBX11K 3-class internal label",
            "representative_policy": "lexicographically smallest sample_id per duplicate family",
        },
        "inputs": {
            "manifest": MANIFEST.relative_to(ROOT).as_posix(),
            "manifest_sha256": file_sha256(MANIFEST),
            "adjudication": ADJUDICATION.relative_to(ROOT).as_posix(),
            "adjudication_sha256": file_sha256(ADJUDICATION),
            "exclusions": EXCLUSIONS.relative_to(ROOT).as_posix(),
            "exclusions_sha256": file_sha256(EXCLUSIONS) if EXCLUSIONS.is_file() else None,
        },
        "confirmed_perceptual_pairs": len(confirmed_pairs),
        "retained_unique_images": len(deduplicated),
        "duplicate_family_size_histogram": {
            str(size): count for size, count in sorted(family_sizes.items())
        },
        "split_class_counts": split_counts(assigned),
        "patient_level_independence_claimed": False,
        "image_level_duplicate_control": True,
    }
    OUT_SUMMARY.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
