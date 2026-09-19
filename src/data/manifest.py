from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from .schema import SampleRecord


def load_manifest(path: str | Path) -> list[SampleRecord]:
    path = Path(path)
    if path.suffix.casefold() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    elif path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("Manifest must be .csv or .jsonl")
    records = [SampleRecord.from_dict(row, base_dir=path.parent) for row in rows]
    ids = [record.sample_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate sample_id values")
    return records


def save_manifest(records: Iterable[SampleRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.to_dict() for record in records]
    fields = [
        "sample_id",
        "image_path",
        "dataset",
        "original_label",
        "canonical_label",
        "patient_id",
        "group_id",
        "split",
        "source_partition",
        "annotation_path",
        "clinical_reading_path",
        "width",
        "height",
        "image_mode",
        "bit_depth",
        "checksum_sha256",
        "audit_flags",
        "provenance",
        "bounding_boxes",
        "metadata",
    ]
    if path.suffix.casefold() == ".jsonl":
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    elif path.suffix.casefold() == ".csv":
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError("Manifest must be .csv or .jsonl")
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
