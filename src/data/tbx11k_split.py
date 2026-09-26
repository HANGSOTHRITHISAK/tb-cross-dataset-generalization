from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .labels import TBX11K_INTERNAL_CLASS_NAMES, normalize_tbx11k_internal_label
from .schema import SampleRecord

SPLIT_NAMES = ("train", "validation", "internal_test")
VALID_ADJUDICATION_DECISIONS = {"confirmed_copy", "distinct"}


@dataclass(frozen=True, slots=True)
class DeduplicatedRecord:
    record: SampleRecord
    internal_label: int
    duplicate_family_id: str
    duplicate_family_size: int
    duplicate_member_ids: tuple[str, ...]


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def _stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"tbx11k-split-v1:{seed}:{value}".encode()).hexdigest()


def _family_id(member_ids: Sequence[str]) -> str:
    joined = "\n".join(sorted(member_ids))
    digest = hashlib.sha256(joined.encode()).hexdigest()[:16]
    return f"dup-{digest}"


def is_primary_tbx11k_record(record: SampleRecord) -> bool:
    if record.dataset != "tbx11k" or record.canonical_label is None:
        return False
    if record.source_partition == "official_trainval_release":
        return True
    # Backward-compatible fallback for older manifests that predate source_partition.
    tail = record.sample_id.split(":", 1)[-1]
    top = tail.split("/", 1)[0]
    return top in {"health", "sick", "tb"}


def load_adjudication(
    path: str | Path, *, expected_rows: int | None = None
) -> list[tuple[str, str]]:
    """Load resolved primary-pool dHash decisions and return confirmed-copy pairs.

    The split pipeline intentionally refuses to continue while any decision is pending
    or any required rationale is blank. Raw images are never read or written here.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Adjudication file is empty")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} adjudication rows, found {len(rows)}"
        )

    confirmed: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=2):
        left = (row.get("left_sample_id") or "").strip()
        right = (row.get("right_sample_id") or "").strip()
        decision = (row.get("decision") or "").strip().casefold()
        rationale = (row.get("rationale") or "").strip()
        family_id = (row.get("duplicate_family_id") or "").strip()
        if not left or not right:
            raise ValueError(f"Adjudication row {index} is missing a sample ID")
        pair = tuple(sorted((left, right)))
        if pair in seen_pairs:
            raise ValueError(f"Duplicate adjudication pair at row {index}: {pair}")
        seen_pairs.add(pair)
        if decision in {"", "pending"}:
            raise ValueError(
                f"Adjudication is incomplete at row {index}: {left} <-> {right}"
            )
        if decision not in VALID_ADJUDICATION_DECISIONS:
            raise ValueError(
                f"Unknown adjudication decision {decision!r} at row {index}; "
                f"expected one of {sorted(VALID_ADJUDICATION_DECISIONS)}"
            )
        if not rationale:
            raise ValueError(f"Adjudication row {index} requires a short rationale")
        if decision == "confirmed_copy":
            if not family_id:
                raise ValueError(
                    f"Confirmed-copy row {index} requires duplicate_family_id"
                )
            confirmed.append((left, right))
    return confirmed


def deduplicate_primary_tbx11k(
    records: Iterable[SampleRecord],
    *,
    confirmed_copy_pairs: Iterable[tuple[str, str]] = (),
    excluded_sample_ids: Iterable[str] = (),
) -> list[DeduplicatedRecord]:
    """Return one deterministic representative per exact/confirmed duplicate family."""
    excluded = set(excluded_sample_ids)
    primary = [
        record
        for record in records
        if is_primary_tbx11k_record(record) and record.sample_id not in excluded
    ]
    by_id = {record.sample_id: record for record in primary}
    if len(by_id) != len(primary):
        raise ValueError("Primary TBX11K records contain duplicate sample_id values")

    uf = _UnionFind(by_id)
    by_checksum: dict[str, list[str]] = defaultdict(list)
    for record in primary:
        if record.checksum_sha256:
            by_checksum[record.checksum_sha256].append(record.sample_id)
    for member_ids in by_checksum.values():
        if len(member_ids) > 1:
            first = member_ids[0]
            for other in member_ids[1:]:
                uf.union(first, other)

    for left, right in confirmed_copy_pairs:
        missing = [sample_id for sample_id in (left, right) if sample_id not in by_id]
        if missing:
            raise ValueError(
                "Confirmed-copy adjudication references sample IDs outside the eligible "
                f"primary pool: {missing}"
            )
        uf.union(left, right)

    components: dict[str, list[str]] = defaultdict(list)
    for sample_id in by_id:
        components[uf.find(sample_id)].append(sample_id)

    deduplicated: list[DeduplicatedRecord] = []
    for member_ids in components.values():
        members = tuple(sorted(member_ids))
        labels = {
            normalize_tbx11k_internal_label(by_id[sample_id].original_label)
            for sample_id in members
        }
        if len(labels) != 1:
            detail = {
                sample_id: by_id[sample_id].original_label for sample_id in members
            }
            raise ValueError(
                "Duplicate family has conflicting 3-class labels and requires manual "
                f"investigation: {detail}"
            )
        internal_label = next(iter(labels))
        representative_id = members[0]
        representative = by_id[representative_id]
        family_id = _family_id(members)
        deduplicated.append(
            DeduplicatedRecord(
                record=representative,
                internal_label=internal_label,
                duplicate_family_id=family_id,
                duplicate_family_size=len(members),
                duplicate_member_ids=members,
            )
        )
    return sorted(deduplicated, key=lambda item: item.record.sample_id)


def _allocate_counts(count: int, ratios: Sequence[float]) -> tuple[int, ...]:
    raw = [count * ratio for ratio in ratios]
    base = [math.floor(value) for value in raw]
    remainder = count - sum(base)
    order = sorted(range(len(ratios)), key=lambda i: (-(raw[i] - base[i]), i))
    for index in order[:remainder]:
        base[index] += 1
    if count >= len(ratios):
        for empty_index, value in enumerate(base):
            if value > 0:
                continue
            donor = max(range(len(base)), key=lambda i: (base[i], -i))
            if base[donor] <= 1:
                break
            base[donor] -= 1
            base[empty_index] += 1
    return tuple(base)


def stratified_split(
    records: Iterable[DeduplicatedRecord],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> list[DeduplicatedRecord]:
    """Assign a deterministic 3-class-stratified TBX11K split."""
    records = list(records)
    if abs(sum(ratios) - 1.0) > 1e-9 or any(ratio <= 0 for ratio in ratios):
        raise ValueError("split ratios must be positive and sum to 1")
    by_label: dict[int, list[DeduplicatedRecord]] = defaultdict(list)
    for item in records:
        if item.internal_label not in TBX11K_INTERNAL_CLASS_NAMES:
            raise ValueError(f"Unknown TBX11K internal label: {item.internal_label}")
        by_label[item.internal_label].append(item)

    assigned: list[DeduplicatedRecord] = []
    for internal_label in sorted(by_label):
        ordered = sorted(
            by_label[internal_label],
            key=lambda item: _stable_key(item.record.sample_id, seed),
        )
        counts = _allocate_counts(len(ordered), ratios)
        boundaries = (counts[0], counts[0] + counts[1])
        for index, item in enumerate(ordered):
            split = (
                SPLIT_NAMES[0]
                if index < boundaries[0]
                else SPLIT_NAMES[1]
                if index < boundaries[1]
                else SPLIT_NAMES[2]
            )
            assigned.append(
                replace(item, record=replace(item.record, split=split))
            )
    return sorted(assigned, key=lambda item: item.record.sample_id)


def split_counts(
    records: Iterable[DeduplicatedRecord],
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {
        name: {class_name: 0 for class_name in TBX11K_INTERNAL_CLASS_NAMES.values()}
        for name in SPLIT_NAMES
    }
    for item in records:
        if item.record.split not in SPLIT_NAMES:
            raise ValueError(f"Record {item.record.sample_id!r} has no valid split")
        class_name = TBX11K_INTERNAL_CLASS_NAMES[item.internal_label]
        out[item.record.split][class_name] += 1
    return out
