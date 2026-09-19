from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

VALID_DATASETS = {"tbx11k", "shenzhen", "montgomery", "synthetic"}


@dataclass(frozen=True, slots=True)
class SampleRecord:
    sample_id: str
    image_path: Path
    dataset: str
    original_label: str
    canonical_label: int | None
    patient_id: str | None = None
    group_id: str | None = None
    split: str | None = None
    source_partition: str | None = None
    annotation_path: Path | None = None
    clinical_reading_path: Path | None = None
    width: int | None = None
    height: int | None = None
    image_mode: str | None = None
    bit_depth: int | None = None
    checksum_sha256: str | None = None
    audit_flags: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    bounding_boxes: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        if self.dataset not in VALID_DATASETS:
            raise ValueError(f"Unsupported dataset: {self.dataset!r}")
        if self.canonical_label not in (None, 0, 1):
            raise ValueError("canonical_label must be None, 0, or 1")

    @property
    def effective_group(self) -> str | None:
        return self.group_id or self.patient_id

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["image_path"] = str(self.image_path)
        value["annotation_path"] = (
            str(self.annotation_path) if self.annotation_path is not None else ""
        )
        value["clinical_reading_path"] = (
            str(self.clinical_reading_path)
            if self.clinical_reading_path is not None
            else ""
        )
        value["audit_flags"] = json.dumps(list(self.audit_flags), sort_keys=True)
        value["provenance"] = json.dumps(self.provenance, sort_keys=True)
        value["bounding_boxes"] = json.dumps(list(self.bounding_boxes), sort_keys=True)
        value["metadata"] = json.dumps(self.metadata, sort_keys=True)
        return value

    @classmethod
    def from_dict(
        cls, row: dict[str, Any], *, base_dir: Path | None = None
    ) -> SampleRecord:
        required = {
            "sample_id",
            "image_path",
            "dataset",
            "original_label",
            "canonical_label",
        }
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Manifest row missing fields: {sorted(missing)}")
        image_path = Path(str(row["image_path"]))
        if base_dir is not None and not image_path.is_absolute():
            image_path = (base_dir / image_path).resolve()

        def optional(name: str) -> str | None:
            value = row.get(name)
            return None if value in (None, "") else str(value)

        boxes_raw = row.get("bbox_json", row.get("bounding_boxes", "[]")) or "[]"
        meta_raw = row.get("metadata_json", row.get("metadata", "{}")) or "{}"
        flags_raw = row.get("audit_flags", "[]") or "[]"
        provenance_raw = row.get("provenance", "{}") or "{}"
        boxes = json.loads(boxes_raw) if isinstance(boxes_raw, str) else boxes_raw
        metadata = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw

        def optional_path(name: str) -> Path | None:
            value = row.get(name)
            if value in (None, ""):
                return None
            path = Path(str(value))
            return (
                (base_dir / path).resolve()
                if base_dir is not None and not path.is_absolute()
                else path
            )

        return cls(
            sample_id=str(row["sample_id"]),
            image_path=image_path,
            dataset=str(row["dataset"]).strip().lower(),
            original_label=str(row["original_label"]),
            canonical_label=(
                None
                if row.get("canonical_label") in (None, "")
                else int(row["canonical_label"])
            ),
            patient_id=optional("patient_id"),
            group_id=optional("group_id"),
            split=optional("split"),
            source_partition=optional("source_partition"),
            annotation_path=optional_path("annotation_path"),
            clinical_reading_path=optional_path("clinical_reading_path"),
            width=int(row["width"]) if row.get("width") not in (None, "") else None,
            height=int(row["height"]) if row.get("height") not in (None, "") else None,
            image_mode=optional("image_mode"),
            bit_depth=int(row["bit_depth"])
            if row.get("bit_depth") not in (None, "")
            else None,
            checksum_sha256=optional("checksum_sha256"),
            audit_flags=tuple(
                json.loads(flags_raw) if isinstance(flags_raw, str) else flags_raw
            ),
            provenance=(
                json.loads(provenance_raw)
                if isinstance(provenance_raw, str)
                else dict(provenance_raw)
            ),
            bounding_boxes=tuple(boxes),
            metadata=dict(metadata),
        )
