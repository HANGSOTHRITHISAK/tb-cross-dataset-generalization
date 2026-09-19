from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .schema import SampleRecord


class ManifestDataset(Dataset[dict[str, object]]):
    """Minimal supervised dataset backed by audited manifest records."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        transform: Callable[[Image.Image], torch.Tensor],
        *,
        require_files: bool = True,
    ) -> None:
        self.records = list(records)
        self.transform = transform
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
        if record.canonical_label is None:
            raise ValueError(
                f"Sample {record.sample_id!r} has no released canonical label"
            )
        with Image.open(record.image_path) as image:
            tensor = self.transform(image.copy())
        return {
            "image": tensor,
            "label": torch.tensor(record.canonical_label, dtype=torch.float32),
            "sample_id": record.sample_id,
        }