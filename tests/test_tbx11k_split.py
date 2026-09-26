from pathlib import Path

import pytest

from src.data.schema import SampleRecord
from src.data.tbx11k_split import (
    deduplicate_primary_tbx11k,
    load_adjudication,
    split_counts,
    stratified_split,
)


def rec(sample_id: str, label: str, checksum: str) -> SampleRecord:
    canonical = 1 if label.startswith("TB") else 0
    return SampleRecord(
        sample_id=sample_id,
        image_path=Path(sample_id.split(":", 1)[-1]),
        dataset="tbx11k",
        original_label=label,
        canonical_label=canonical,
        source_partition="official_trainval_release",
        checksum_sha256=checksum,
    )


def test_exact_and_confirmed_duplicates_keep_one_representative() -> None:
    records = [
        rec("tbx11k:health/h2.png", "Healthy", "same"),
        rec("tbx11k:health/h1.png", "Healthy", "same"),
        rec("tbx11k:sick/s2.png", "Sick & Non-TB", "s2"),
        rec("tbx11k:sick/s1.png", "Sick & Non-TB", "s1"),
        rec("tbx11k:tb/t1.png", "TB (subtype unspecified in released path)", "t1"),
    ]
    result = deduplicate_primary_tbx11k(
        records,
        confirmed_copy_pairs=[("tbx11k:sick/s2.png", "tbx11k:sick/s1.png")],
    )
    assert [item.record.sample_id for item in result] == [
        "tbx11k:health/h1.png",
        "tbx11k:sick/s1.png",
        "tbx11k:tb/t1.png",
    ]
    assert [item.duplicate_family_size for item in result] == [2, 2, 1]


def test_conflicting_duplicate_labels_fail() -> None:
    records = [
        rec("tbx11k:health/h1.png", "Healthy", "same"),
        rec("tbx11k:sick/s1.png", "Sick & Non-TB", "same"),
    ]
    with pytest.raises(ValueError, match="conflicting 3-class labels"):
        deduplicate_primary_tbx11k(records)


def test_split_is_deterministic_and_stratified() -> None:
    records = []
    labels = [
        ("health", "Healthy"),
        ("sick", "Sick & Non-TB"),
        ("tb", "TB (subtype unspecified in released path)"),
    ]
    for folder, label in labels:
        for index in range(20):
            records.append(
                rec(
                    f"tbx11k:{folder}/{index:02d}.png",
                    label,
                    f"{folder}-{index}",
                )
            )
    deduped = deduplicate_primary_tbx11k(records)
    first = stratified_split(deduped, seed=42)
    second = stratified_split(deduped, seed=42)
    assert [(x.record.sample_id, x.record.split) for x in first] == [
        (x.record.sample_id, x.record.split) for x in second
    ]
    counts = split_counts(first)
    assert counts == {
        "train": {"healthy": 14, "sick_non_tb": 14, "tb": 14},
        "validation": {"healthy": 3, "sick_non_tb": 3, "tb": 3},
        "internal_test": {"healthy": 3, "sick_non_tb": 3, "tb": 3},
    }


def test_pending_adjudication_blocks_split(tmp_path: Path) -> None:
    path = tmp_path / "adjudication.csv"
    path.write_text(
        "left_sample_id,right_sample_id,dhash_distance,decision,rationale,duplicate_family_id\n"
        "tbx11k:sick/a.png,tbx11k:sick/b.png,1,pending,,dhash-candidate-001\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete"):
        load_adjudication(path)


def test_confirmed_adjudication_requires_family_and_rationale(tmp_path: Path) -> None:
    path = tmp_path / "adjudication.csv"
    path.write_text(
        "left_sample_id,right_sample_id,dhash_distance,decision,rationale,duplicate_family_id\n"
        "tbx11k:sick/a.png,tbx11k:sick/b.png,1,confirmed_copy,"
        "same radiograph,dhash-candidate-001\n",
        encoding="utf-8",
    )
    assert load_adjudication(path) == [
        ("tbx11k:sick/a.png", "tbx11k:sick/b.png")
    ]
