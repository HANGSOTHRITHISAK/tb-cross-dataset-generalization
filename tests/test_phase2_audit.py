from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.data.manifest import load_manifest
from src.data.phase2_audit import audit_nlm_dataset, write_cross_dataset_duplicates


def _write_fixture(root: Path, prefix: str) -> None:
    (root / "CXR_png").mkdir(parents=True)
    (root / "ClinicalReadings").mkdir()
    for index, label in ((1, 0), (2, 1)):
        stem = f"{prefix}_{index:04d}_{label}"
        Image.fromarray(np.full((12, 16), index * 40, dtype=np.uint16)).save(
            root / "CXR_png" / f"{stem}.png"
        )
        (root / "ClinicalReadings" / f"{stem}.txt").write_text(
            "male 40yrs\nnormal" if label == 0 else "female 50yrs\nPTB",
            encoding="utf-8",
        )


def test_real_manifest_audit_contract(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw" / "shenzhen"
    _write_fixture(raw, "CHNCXR")
    manifest = tmp_path / "data" / "manifests" / "shenzhen.csv"
    summary = audit_nlm_dataset(
        raw,
        "shenzhen",
        manifest,
        tmp_path / "data" / "audit",
        source_url="https://example.invalid/authoritative",
        retrieval_date="2026-09-19",
    )
    assert summary["image_count"] == 2
    assert summary["corrupt_images"] == []
    assert summary["clinical_readings_present"] == 2
    records = load_manifest(manifest)
    assert records[0].width == 16
    assert records[0].height == 12
    assert records[0].checksum_sha256
    assert records[0].patient_id is None
    assert records[0].split == "external_test"
    assert records[0].image_path.is_file()


def test_cross_dataset_exact_duplicate_report(sample_records, tmp_path: Path) -> None:
    from dataclasses import replace

    from src.data.manifest import save_manifest

    checksum = "a" * 64
    left = replace(sample_records[0], dataset="shenzhen", checksum_sha256=checksum)
    right = replace(sample_records[1], dataset="montgomery", checksum_sha256=checksum)
    first = save_manifest([left], tmp_path / "shenzhen.csv")
    second = save_manifest([right], tmp_path / "montgomery.csv")
    rows = write_cross_dataset_duplicates([first, second], tmp_path / "duplicates.csv")
    assert len(rows) == 1
    assert rows[0]["datasets"] == "montgomery;shenzhen"


def test_blank_tbx11k_label_round_trips_as_none(tmp_path: Path) -> None:
    from src.data.manifest import save_manifest
    from src.data.schema import SampleRecord

    path = tmp_path / "tbx11k.csv"
    save_manifest(
        [
            SampleRecord(
                sample_id="tbx11k:unreleased/sample.png",
                image_path=Path("unreleased/sample.png"),
                dataset="tbx11k",
                original_label="unreleased",
                canonical_label=None,
            )
        ],
        path,
    )
    assert load_manifest(path)[0].canonical_label is None


def test_supervised_dataset_rejects_unlabeled_sample(tmp_path: Path) -> None:
    import pytest
    import torch

    from src.data.datasets import ManifestDataset
    from src.data.schema import SampleRecord

    image_path = tmp_path / "unreleased.png"
    Image.new("L", (2, 2)).save(image_path)
    dataset = ManifestDataset(
        [
            SampleRecord(
                sample_id="tbx11k:unreleased.png",
                image_path=image_path,
                dataset="tbx11k",
                original_label="unreleased",
                canonical_label=None,
            )
        ],
        transform=lambda image: torch.zeros(1),
    )
    with pytest.raises(ValueError, match="no released canonical label"):
        dataset[0]
