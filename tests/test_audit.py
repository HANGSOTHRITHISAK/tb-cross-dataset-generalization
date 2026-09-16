from pathlib import Path

from PIL import Image

from src.data.audit import build_manifest, difference_hash, sha256_file, summarize


def test_hashes_are_stable(tmp_path: Path) -> None:
    path = tmp_path / "tb" / "sample.png"
    path.parent.mkdir()
    Image.new("L", (16, 12), color=128).save(path)
    assert sha256_file(path) == sha256_file(path)
    with Image.open(path) as image:
        assert difference_hash(image) == difference_hash(image)


def test_manifest_and_summary(tmp_path: Path) -> None:
    for label, value in (("normal", 40), ("tb", 200)):
        directory = tmp_path / label
        directory.mkdir()
        Image.new("L", (20, 10), color=value).save(directory / f"{label}.png")

    manifest = build_manifest(tmp_path)
    assert len(manifest) == 2
    assert set(manifest["label"]) == {"normal", "tb"}
    assert set(manifest["width"]) == {20}
    assert set(manifest["height"]) == {10}
    text = summarize(manifest)
    assert "Files discovered: **2**" in text
    assert "`normal`: 1" in text
    assert "`tb`: 1" in text
