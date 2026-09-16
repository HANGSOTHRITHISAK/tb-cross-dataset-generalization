"""Dataset-agnostic image audit utilities for TB chest X-ray datasets."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size + 1, hash_size))
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"

def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path

def infer_label(path: Path, root: Path) -> str | None:
    relative = path.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else None

def inspect_image(path: Path, root: Path) -> dict[str, object]:
    row: dict[str, object] = {"path": path.relative_to(root).as_posix(), "label": infer_label(path, root), "sha256": sha256_file(path), "error": None}
    try:
        with Image.open(path) as image:
            image.load()
            gray = np.asarray(image.convert("L"), dtype=np.float32)
            row.update({"width": image.width, "height": image.height, "mode": image.mode, "format": image.format, "dhash": difference_hash(image), "mean_intensity": float(gray.mean()), "std_intensity": float(gray.std()), "min_intensity": float(gray.min()), "max_intensity": float(gray.max()), "file_bytes": path.stat().st_size})
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row

def build_manifest(root: Path) -> pd.DataFrame:
    paths = list(iter_images(root))
    if not paths:
        raise ValueError(f"No supported image files found under {root}")
    return pd.DataFrame(inspect_image(path, root) for path in paths)

def summarize(manifest: pd.DataFrame) -> str:
    valid = manifest[manifest["error"].isna()].copy()
    lines = ["# Dataset audit summary", "", f"- Files discovered: **{len(manifest)}**", f"- Successfully decoded: **{len(valid)}**", f"- Decode failures: **{manifest['error'].notna().sum()}**", f"- Rows participating in exact-hash duplicate groups: **{int(valid.duplicated('sha256', keep=False).sum())}**", f"- Rows sharing an identical dHash: **{int(valid.duplicated('dhash', keep=False).sum())}**", "", "## Class counts", ""]
    counts = manifest["label"].fillna("<unknown>").value_counts()
    lines.extend(f"- `{label}`: {count}" for label, count in counts.items())
    lines.extend(["", "## Interpretation notes", "", "Identical perceptual hashes are only candidates for visual duplication; inspect them before exclusion. This generic audit cannot infer patient identity or diagnostic validity. Dataset-specific metadata must be joined before patient-level splitting or final clearance.", ""])
    return "\n".join(lines)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory and audit a local image dataset.")
    parser.add_argument("root", type=Path, help="Root directory containing dataset images")
    parser.add_argument("--manifest", type=Path, default=Path("audit-manifest.csv"))
    parser.add_argument("--summary", type=Path, default=Path("audit-summary.md"))
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist or is not a directory: {root}")
    manifest = build_manifest(root)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    args.summary.write_text(summarize(manifest), encoding="utf-8")
    print(f"Wrote {args.manifest} and {args.summary}")

if __name__ == "__main__":
    main()
