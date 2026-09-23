from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset

from .labels import normalize_tbx11k_internal_label
from .schema import SampleRecord

LabelSpace = Literal["canonical_binary", "tbx11k_internal_3class"]


class ManifestDataset(Dataset[dict[str, object]]):
    """Minimal supervised dataset backed by audited manifest records."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        transform: Callable[[Image.Image], torch.Tensor],
        *,
        require_files: bool = True,
        label_space: LabelSpace = "canonical_binary",
    ) -> None:
        self.records = list(records)
        self.transform = transform
        self.label_space = label_space
        if require_files:
            missing = [
                str(record.image_path)
                for record in self.records
                if not record.image_path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Missing images referenced by manifest: {missing[:5]}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]

        if self.label_space == "tbx11k_internal_3class":
            if record.dataset != "tbx11k":
                raise ValueError(
                    "tbx11k_internal_3class label space is valid only for TBX11K records"
                )
            label = torch.tensor(
                normalize_tbx11k_internal_label(record.original_label),
                dtype=torch.long,
            )
        else:
            if record.canonical_label is None:
                raise ValueError(
                    f"Sample {record.sample_id!r} has no released canonical label"
                )
            label = torch.tensor(record.canonical_label, dtype=torch.float32)

        with Image.open(record.image_path) as image:
            tensor = self.transform(image.copy())

        return {
            "image": tensor,
            "label": label,
            "sample_id": record.sample_id,
        }
