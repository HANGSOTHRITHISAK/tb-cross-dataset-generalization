from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.schema import SampleRecord


@pytest.fixture
def sample_records(tmp_path: Path) -> list[SampleRecord]:
    records = []
    for index in range(12):
        path = tmp_path / f"image_{index}.png"
        Image.fromarray(np.full((20, 24), index * 10, dtype=np.uint8), mode="L").save(
            path
        )
        records.append(
            SampleRecord(
                sample_id=f"s{index}",
                image_path=path,
                dataset="tbx11k",
                original_label="Active TB" if index % 2 else "Healthy",
                canonical_label=index % 2,
                patient_id=f"p{index // 2}",
            )
        )
    return records
