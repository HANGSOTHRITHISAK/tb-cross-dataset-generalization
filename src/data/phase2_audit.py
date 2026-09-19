from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .manifest import save_manifest
from .schema import SampleRecord

NLM_PATTERNS = {
    "shenzhen": re.compile(r"^CHNCXR_(\d{4})_([01])\.png$", re.IGNORECASE),
    "montgomery": re.compile(r"^MCUCXR_(\d{4,5})_([01])\.png$", re.IGNORECASE),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path, size: int = 16) -> str:
    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
        )
    bits = pixels[:, 1:] > pixels[:, :-1]
    return f"{int(''.join('1' if value else '0' for value in bits.flat), 2):0{size * size // 4}x}"


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def audit_nlm_dataset(
    raw_root: str | Path,
    dataset: str,
    manifest_path: str | Path,
    audit_dir: str | Path,
    *,
    source_url: str,
    retrieval_date: str,
) -> dict[str, Any]:
    raw_root = Path(raw_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    audit_dir = Path(audit_dir).resolve()
    if dataset not in NLM_PATTERNS:
        raise ValueError(f"Unsupported NLM dataset: {dataset}")
    image_dir = raw_root / "CXR_png"
    reading_dir = raw_root / "ClinicalReadings"
    pattern = NLM_PATTERNS[dataset]
    records: list[SampleRecord] = []
    corrupt: list[dict[str, str]] = []
    unexpected: list[str] = []
    perceptual: dict[str, str] = {}
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    clinical_readings_present = 0
    for path in sorted(image_dir.glob("*")):
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            unexpected.append(path.name)
            continue
        label_code = match.group(2)
        checksum = sha256_file(path)
        flags: list[str] = []
        width = height = bit_depth = None
        mode = None
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
                extrema = image.getextrema()
                bit_depth = (
                    16
                    if mode.startswith("I;16")
                    else 8
                    if mode in {"L", "RGB", "RGBA"}
                    else None
                )
                if (
                    isinstance(extrema, tuple)
                    and extrema
                    and isinstance(extrema[0], tuple)
                ):
                    minimum = min(channel[0] for channel in extrema)
                    maximum = max(channel[1] for channel in extrema)
                else:
                    minimum, maximum = extrema
            dimensions[f"{width}x{height}"] += 1
            modes[str(mode)] += 1
            perceptual[path.name] = difference_hash(path)
        except (OSError, ValueError, UnidentifiedImageError) as error:
            flags.append("corrupt_or_unreadable")
            corrupt.append({"file": path.name, "error": str(error)})
            minimum = maximum = None
        reading = reading_dir / f"{path.stem}.txt"
        if not reading.is_file():
            flags.append("missing_clinical_reading")
        else:
            clinical_readings_present += 1
        annotation_candidates = sorted(raw_root.rglob(f"{path.stem}*.json"))
        if dataset == "montgomery":
            annotation_candidates = sorted(raw_root.glob(f"ManualMask/**/*{path.name}"))
        else:
            annotation_candidates.sort(
                key=lambda candidate: (
                    0
                    if candidate.relative_to(raw_root).parts[0] == "Annotations"
                    else 1,
                    candidate.as_posix(),
                )
            )
        record = SampleRecord(
            sample_id=f"{dataset}:{path.stem}",
            image_path=Path("..") / "raw" / dataset / "CXR_png" / path.name,
            dataset=dataset,
            original_label="normal" if label_code == "0" else "abnormal_tb_consistent",
            canonical_label=int(label_code),
            patient_id=None,
            group_id=None,
            split="external_test",
            source_partition="official_external_release",
            annotation_path=(
                Path("..")
                / "raw"
                / dataset
                / annotation_candidates[0].relative_to(raw_root)
                if annotation_candidates
                else None
            ),
            clinical_reading_path=None,

            width=width,
            height=height,
            image_mode=mode,
            bit_depth=bit_depth,
            checksum_sha256=checksum,
            audit_flags=tuple(flags),
            provenance={"source_url": source_url, "retrieval_date": retrieval_date},
            metadata={
                "observed_min": minimum,
                "observed_max": maximum,
                "dhash_256": perceptual.get(path.name),
                "all_annotation_paths": [
                    (
                        Path("..") / "raw" / dataset / candidate.relative_to(raw_root)
                    ).as_posix()
                    for candidate in annotation_candidates
                ],
            },
        )
        records.append(record)
    hashes: dict[str, list[str]] = defaultdict(list)
    for record in records:
        hashes[str(record.checksum_sha256)].append(record.sample_id)
    exact_duplicates = [ids for ids in hashes.values() if len(ids) > 1]
    near_duplicates: list[dict[str, Any]] = []
    names = sorted(perceptual)
    for left, right in itertools.combinations(names, 2):
        distance = hamming_distance(perceptual[left], perceptual[right])
        if distance <= 8:
            near_duplicates.append(
                {"left": left, "right": right, "dhash_distance": distance}
            )
    audit_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(records, manifest_path)
    all_files = [
        path
        for path in raw_root.rglob("*")
        if path.is_file() and not path.name.endswith(".part")
    ]
    basenames: dict[str, list[str]] = defaultdict(list)
    image_hashes = {
        (
            raw_root / "CXR_png" / Path(record.image_path).name
        ).resolve(): record.checksum_sha256
        for record in records
    }
    raw_hashes: dict[str, list[str]] = defaultdict(list)
    for path in all_files:
        basenames[path.name].append(path.relative_to(raw_root).as_posix())
        checksum = image_hashes.get(path.resolve()) or sha256_file(path)
        raw_hashes[str(checksum)].append(path.relative_to(raw_root).as_posix())
    raw_exact_duplicate_groups = [
        {"checksum_sha256": checksum, "paths": paths}
        for checksum, paths in sorted(raw_hashes.items())
        if len(paths) > 1
    ]
    summary = {
        "dataset": dataset,
        "source_url": source_url,
        "retrieval_date": retrieval_date,
        "image_count": len(records),
        "label_counts": dict(
            sorted(Counter(r.original_label for r in records).items())
        ),
        "canonical_label_counts": {
            str(k): v
            for k, v in sorted(Counter(r.canonical_label for r in records).items())
        },
        "image_formats": dict(
            Counter(
                path.suffix.lower() for path in image_dir.glob("*") if path.is_file()
            )
        ),
        "raw_file_count": len(all_files),
        "raw_bytes": sum(path.stat().st_size for path in all_files),
        "all_file_extensions": dict(
            sorted(Counter(path.suffix.lower() for path in all_files).items())
        ),
        "duplicate_basenames": {
            name: paths for name, paths in sorted(basenames.items()) if len(paths) > 1
        },
        "raw_exact_duplicate_groups": raw_exact_duplicate_groups,
        "dimensions": dict(dimensions.most_common()),
        "modes": dict(modes),
        "global_observed_min": min(
            (
                r.metadata["observed_min"]
                for r in records
                if r.metadata["observed_min"] is not None
            ),
            default=None,
        ),
        "global_observed_max": max(
            (
                r.metadata["observed_max"]
                for r in records
                if r.metadata["observed_max"] is not None
            ),
            default=None,
        ),
        "clinical_readings_present": clinical_readings_present,

        "annotations_linked": sum(r.annotation_path is not None for r in records),
        "patient_group_ids_available": False,
        "unexpected_image_files": unexpected,
        "corrupt_images": corrupt,
        "exact_duplicate_groups": exact_duplicates,
        "near_duplicate_candidates": near_duplicates,
        "near_duplicate_method": "256-bit difference hash on grayscale 17x16 thumbnail; candidates have Hamming distance <= 8 and require human review",
    }
    (audit_dir / f"{dataset}_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def write_cross_dataset_duplicates(
    manifest_paths: list[str | Path], output_path: str | Path
) -> list[dict[str, str]]:
    from .manifest import load_manifest

    by_hash: dict[str, list[SampleRecord]] = defaultdict(list)
    for path in manifest_paths:
        for record in load_manifest(path):
            if record.checksum_sha256:
                by_hash[record.checksum_sha256].append(record)
    rows: list[dict[str, str]] = []
    for checksum, records in by_hash.items():
        if len({record.dataset for record in records}) > 1:
            rows.append(
                {
                    "checksum_sha256": checksum,
                    "datasets": ";".join(
                        sorted({record.dataset for record in records})
                    ),
                    "sample_ids": ";".join(
                        sorted(record.sample_id for record in records)
                    ),
                }
            )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["checksum_sha256", "datasets", "sample_ids"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_cross_dataset_near_duplicates(
    manifest_paths: list[str | Path], output_path: str | Path, *, max_distance: int = 8
) -> list[dict[str, str | int]]:
    from .manifest import load_manifest

    records = [record for path in manifest_paths for record in load_manifest(path)]
    rows: list[dict[str, str | int]] = []
    for left, right in itertools.combinations(records, 2):
        if left.dataset == right.dataset:
            continue
        left_hash = left.metadata.get("dhash_256")
        right_hash = right.metadata.get("dhash_256")
        if not left_hash or not right_hash:
            continue
        distance = hamming_distance(str(left_hash), str(right_hash))
        if distance <= max_distance:
            rows.append(
                {
                    "left_sample_id": left.sample_id,
                    "right_sample_id": right.sample_id,
                    "dhash_distance": distance,
                }
            )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["left_sample_id", "right_sample_id", "dhash_distance"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows
