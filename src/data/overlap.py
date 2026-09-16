"""Compare audit manifests for exact and perceptual image overlap.

Exact SHA-256 matches are strong evidence of file duplication. Perceptual-hash
matches are screening candidates only and must be visually reviewed before any
sample is excluded from an experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"path", "sha256", "dhash"}


def _clean_hash(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    return text or None


def hamming_distance_hex(left: str, right: str) -> int:
    """Return Hamming distance between equal-length hexadecimal hashes."""
    left = left.strip().lower()
    right = right.strip().lower()
    if len(left) != len(right):
        raise ValueError("Perceptual hashes must have equal hexadecimal length")
    return (int(left, 16) ^ int(right, 16)).bit_count()


def validate_manifest(frame: pd.DataFrame, name: str) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{name} manifest is missing columns: {sorted(missing)}")


def exact_matches(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Return cross-manifest SHA-256 matches."""
    ldf = left.copy()
    rdf = right.copy()
    ldf["sha256"] = ldf["sha256"].map(_clean_hash)
    rdf["sha256"] = rdf["sha256"].map(_clean_hash)
    ldf = ldf[ldf["sha256"].notna()]
    rdf = rdf[rdf["sha256"].notna()]

    columns_left = ["path", "sha256"] + (["label"] if "label" in ldf.columns else [])
    columns_right = ["path", "sha256"] + (["label"] if "label" in rdf.columns else [])
    merged = ldf[columns_left].merge(
        rdf[columns_right], on="sha256", suffixes=("_left", "_right"), how="inner"
    )
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "path_left",
                "path_right",
                "label_left",
                "label_right",
                "match_type",
                "hash_distance",
                "sha256",
                "dhash_left",
                "dhash_right",
            ]
        )

    if "label_left" not in merged:
        merged["label_left"] = None
    if "label_right" not in merged:
        merged["label_right"] = None
    merged["match_type"] = "exact_sha256"
    merged["hash_distance"] = 0
    merged["dhash_left"] = None
    merged["dhash_right"] = None
    return merged[
        [
            "path_left",
            "path_right",
            "label_left",
            "label_right",
            "match_type",
            "hash_distance",
            "sha256",
            "dhash_left",
            "dhash_right",
        ]
    ]


def perceptual_candidates(
    left: pd.DataFrame,
    right: pd.DataFrame,
    max_distance: int = 6,
    exact: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return cross-manifest dHash candidates within a Hamming threshold."""
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative")

    exact_pairs: set[tuple[str, str]] = set()
    if exact is not None and not exact.empty:
        exact_pairs = set(zip(exact["path_left"], exact["path_right"]))

    left_rows: list[tuple[str, object, str]] = []
    for row in left.itertuples(index=False):
        dhash = _clean_hash(getattr(row, "dhash"))
        if dhash is not None:
            left_rows.append((getattr(row, "path"), getattr(row, "label", None), dhash))

    right_rows: list[tuple[str, object, str]] = []
    for row in right.itertuples(index=False):
        dhash = _clean_hash(getattr(row, "dhash"))
        if dhash is not None:
            right_rows.append((getattr(row, "path"), getattr(row, "label", None), dhash))

    matches: list[dict[str, object]] = []
    for left_path, left_label, left_hash in left_rows:
        for right_path, right_label, right_hash in right_rows:
            if (left_path, right_path) in exact_pairs:
                continue
            if len(left_hash) != len(right_hash):
                continue
            distance = hamming_distance_hex(left_hash, right_hash)
            if distance <= max_distance:
                matches.append(
                    {
                        "path_left": left_path,
                        "path_right": right_path,
                        "label_left": left_label,
                        "label_right": right_label,
                        "match_type": "dhash_candidate",
                        "hash_distance": distance,
                        "sha256": None,
                        "dhash_left": left_hash,
                        "dhash_right": right_hash,
                    }
                )

    return pd.DataFrame(
        matches,
        columns=[
            "path_left",
            "path_right",
            "label_left",
            "label_right",
            "match_type",
            "hash_distance",
            "sha256",
            "dhash_left",
            "dhash_right",
        ],
    )


def compare_manifests(
    left: pd.DataFrame, right: pd.DataFrame, max_dhash_distance: int = 6
) -> pd.DataFrame:
    validate_manifest(left, "left")
    validate_manifest(right, "right")
    exact = exact_matches(left, right)
    near = perceptual_candidates(left, right, max_dhash_distance, exact)
    result = pd.concat([exact, near], ignore_index=True)
    if not result.empty:
        result = result.sort_values(
            ["match_type", "hash_distance", "path_left", "path_right"]
        ).reset_index(drop=True)
    return result


def summarize_matches(
    matches: pd.DataFrame,
    left_name: str,
    right_name: str,
    max_dhash_distance: int,
) -> str:
    exact = matches[matches["match_type"] == "exact_sha256"] if not matches.empty else matches
    near = matches[matches["match_type"] == "dhash_candidate"] if not matches.empty else matches
    unique_left = matches["path_left"].nunique() if not matches.empty else 0
    unique_right = matches["path_right"].nunique() if not matches.empty else 0
    return "\n".join(
        [
            "# Cross-dataset overlap summary",
            "",
            f"- Left manifest: **{left_name}**",
            f"- Right manifest: **{right_name}**",
            f"- Exact SHA-256 pairs: **{len(exact)}**",
            f"- dHash candidate pairs (distance <= {max_dhash_distance}): **{len(near)}**",
            f"- Unique left-side files flagged: **{unique_left}**",
            f"- Unique right-side files flagged: **{unique_right}**",
            "",
            "## Interpretation",
            "",
            "Exact SHA-256 matches are strong evidence of duplicate files. dHash matches are only candidate visual duplicates; resized, cropped, or visually similar radiographs can collide, so every perceptual candidate must be reviewed before exclusion. Absence of a dHash match also does not prove independence.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two dataset audit manifests for overlap.")
    parser.add_argument("left_manifest", type=Path)
    parser.add_argument("right_manifest", type=Path)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--max-dhash-distance", type=int, default=6)
    parser.add_argument("--matches", type=Path, default=Path("overlap-matches.csv"))
    parser.add_argument("--summary", type=Path, default=Path("overlap-summary.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left = pd.read_csv(args.left_manifest)
    right = pd.read_csv(args.right_manifest)
    matches = compare_manifests(left, right, args.max_dhash_distance)

    args.matches.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.matches, index=False)
    args.summary.write_text(
        summarize_matches(matches, args.left_name, args.right_name, args.max_dhash_distance),
        encoding="utf-8",
    )
    print(f"Wrote {args.matches} and {args.summary}")


if __name__ == "__main__":
    main()
