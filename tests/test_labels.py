from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.datasets import ManifestDataset
from src.data.labels import (
    TBX11K_INTERNAL_CLASS_NAMES,
    harmonize_tbx11k_internal_to_binary,
    normalize_label,
    normalize_tbx11k_internal_label,
)
from src.data.schema import SampleRecord


def test_tbx11k_internal_three_class_mapping() -> None:
    assert TBX11K_INTERNAL_CLASS_NAMES == {
        0: "healthy",
        1: "sick_non_tb",
        2: "tb",
    }
    assert normalize_tbx11k_internal_label("Healthy") == 0
    assert normalize_tbx11k_internal_label("Sick & Non-TB") == 1
    assert normalize_tbx11k_internal_label(
        "TB (subtype unspecified in released path)"
    ) == 2


def test_tbx11k_internal_mapping_does_not_infer_subtypes() -> None:
    with pytest.raises(ValueError, match="do not infer"):
        normalize_tbx11k_internal_label("latent TB")


def test_internal_classes_harmonize_to_external_binary_space() -> None:
    assert harmonize_tbx11k_internal_to_binary(0) == 0
    assert harmonize_tbx11k_internal_to_binary(1) == 0
    assert harmonize_tbx11k_internal_to_binary(2) == 1


def test_existing_binary_harmonization_remains_available() -> None:
    assert normalize_label("tbx11k", "Healthy") == 0
    assert normalize_label("tbx11k", "Sick & Non-TB") == 0
    assert normalize_label(
        "tbx11k", "TB (subtype unspecified in released path)"
    ) == 1


def test_manifest_dataset_can_emit_internal_three_class_target(tmp_path: Path) -> None:
    image_path = tmp_path / "tb.png"
    Image.new("L", (2, 2)).save(image_path)
    record = SampleRecord(
        sample_id="tbx11k:tb/tb0001.png",
        image_path=image_path,
        dataset="tbx11k",
        original_label="TB (subtype unspecified in released path)",
        canonical_label=1,
    )
    dataset = ManifestDataset(
        [record],
        transform=lambda image: torch.zeros(1),
        label_space="tbx11k_internal_3class",
    )
    item = dataset[0]
    assert item["label"].dtype == torch.long
    assert item["label"].item() == 2
