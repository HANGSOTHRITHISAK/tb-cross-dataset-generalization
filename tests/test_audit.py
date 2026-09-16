from pathlib import Path

import numpy as np
from PIL import Image

from src.data.audit import (
    border_mean,
    build_manifest,
    difference_hash,
    laplacian_variance,
    sha256_file,
    standardized_mean_differences,
    summarize,
)


def test_hashes_are_stable(tmp_path: Path) -> None:
    path = tmp_path / "tb" / "sample.png"
    path.parent.mkdir()
    Image.new("L", (16, 12), color=128).save(path)
    assert sha256_file(path) == sha256_file(path)
    with Image.open(path) as image:
        assert difference_hash(image) == difference_hash(image)


def test_low_level_feature_helpers() -> None:
    gray = np.full((10, 20), 100, dtype=np.float32)
    assert border_mean(gray) == 100.0
    assert laplacian_variance(gray) == 0.0

    edged = gray.copy()
    edged[:, 10:] = 200
    assert laplacian_variance(edged) > 0.0


def test_manifest_and_summary(tmp_path: Path) -> None:
    for label, value in (("normal", 40), ("tb", 200)):
        directory = tmp_path / label
        directory.mkdir()
        for index in range(3):
            image = np.full((10, 20), value + index, dtype=np.uint8)
            Image.fromarray(image, mode="L").save(directory / f"{label}-{index}.png")

    manifest = build_manifest(tmp_path)
    assert len(manifest) == 6
    assert set(manifest["label"]) == {"normal", "tb"}
    assert set(manifest["width"]) == {20}
    assert set(manifest["height"]) == {10}
    assert set(manifest["aspect_ratio"]) == {2.0}
    assert "border_mean" in manifest.columns
    assert "sharpness_laplacian_var" in manifest.columns
    assert "hist_bin_00" in manifest.columns
    assert "hist_bin_15" in manifest.columns

    effects = standardized_mean_differences(manifest)
    assert effects
    assert any(feature == "mean_intensity" for feature, _ in effects)

    text = summarize(manifest)
    assert "Files discovered: **6**" in text
    assert "`normal`: 3" in text
    assert "`tb`: 3" in text
    assert "Largest simple class-conditional differences" in text
