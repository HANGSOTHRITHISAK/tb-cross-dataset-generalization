from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace

from .schema import SampleRecord


def _stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"split-v1:{seed}:{value}".encode()).hexdigest()


def deterministic_group_split(
    records: Iterable[SampleRecord],
    *,
    seed: int = 2026,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    require_groups: bool = False,
) -> list[SampleRecord]:
    records = list(records)
    if not records:
        return []
    if abs(sum(ratios) - 1.0) > 1e-9 or any(r <= 0 for r in ratios):
        raise ValueError("split ratios must be positive and sum to 1")
    if require_groups and any(record.effective_group is None for record in records):
        raise ValueError("Verified patient/group identifiers are required but missing")

    grouped: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        group = record.effective_group or f"sample:{record.sample_id}"
        grouped[group].append(record)
    by_label: dict[tuple[int, int], list[str]] = defaultdict(list)
    for group, items in grouped.items():
        label_signature = (
            sum(item.canonical_label == 0 for item in items),
            sum(item.canonical_label == 1 for item in items),
        )
        by_label[label_signature].append(group)
    assignment: dict[str, str] = {}
    names = ("train", "validation", "internal_test")
    for label_signature, groups in by_label.items():
        ordered = sorted(
            groups, key=lambda group: _stable_key(f"{label_signature}:{group}", seed)
        )
        count = len(ordered)
        n_train = round(count * ratios[0])
        n_val = round(count * ratios[1])
        if count >= 3:
            n_train = min(max(1, n_train), count - 2)
            n_val = min(max(1, n_val), count - n_train - 1)
        boundaries = (n_train, n_train + n_val)
        for index, group in enumerate(ordered):
            split = (
                names[0]
                if index < boundaries[0]
                else names[1]
                if index < boundaries[1]
                else names[2]
            )
            assignment[group] = split
    result = []
    for record in records:
        group = record.effective_group or f"sample:{record.sample_id}"
        result.append(replace(record, split=assignment[group]))
    assert_disjoint(result)
    return result


def assert_disjoint(records: Iterable[SampleRecord]) -> None:
    sample_splits: dict[str, str] = {}
    group_splits: dict[str, str] = {}
    for record in records:
        if record.split is None:
            raise ValueError(f"Sample {record.sample_id!r} has no split")
        previous = sample_splits.setdefault(record.sample_id, record.split)
        if previous != record.split:
            raise ValueError(f"Sample overlap: {record.sample_id!r}")
        if record.effective_group is not None:
            prior_group = group_splits.setdefault(record.effective_group, record.split)
            if prior_group != record.split:
                raise ValueError(f"Group overlap: {record.effective_group!r}")
