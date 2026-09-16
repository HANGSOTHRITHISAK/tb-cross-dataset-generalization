import pandas as pd
import pytest

from src.data.overlap import compare_manifests, hamming_distance_hex


def _manifest(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["path", "label", "sha256", "dhash"])


def test_hamming_distance_hex() -> None:
    assert hamming_distance_hex("0000", "0000") == 0
    assert hamming_distance_hex("0000", "0001") == 1
    assert hamming_distance_hex("0000", "000f") == 4
    with pytest.raises(ValueError):
        hamming_distance_hex("00", "0000")


def test_compare_manifests_finds_exact_and_near_candidates() -> None:
    left = _manifest(
        [
            {"path": "tbx/exact.png", "label": "tb", "sha256": "abc", "dhash": "0000000000000000"},
            {"path": "tbx/near.png", "label": "normal", "sha256": "left-near", "dhash": "0000000000000000"},
            {"path": "tbx/far.png", "label": "normal", "sha256": "left-far", "dhash": "ffffffffffffffff"},
        ]
    )
    right = _manifest(
        [
            {"path": "sz/exact.png", "label": "tb", "sha256": "abc", "dhash": "0000000000000000"},
            {"path": "sz/near.png", "label": "normal", "sha256": "right-near", "dhash": "0000000000000003"},
        ]
    )

    matches = compare_manifests(left, right, max_dhash_distance=2)

    exact = matches[matches["match_type"] == "exact_sha256"]
    assert len(exact) == 1
    assert exact.iloc[0]["path_left"] == "tbx/exact.png"
    assert exact.iloc[0]["path_right"] == "sz/exact.png"

    near = matches[matches["match_type"] == "dhash_candidate"]
    assert ((near["path_left"] == "tbx/near.png") & (near["path_right"] == "sz/near.png")).any()
    assert not ((near["path_left"] == "tbx/far.png") & (near["path_right"] == "sz/near.png")).any()


def test_missing_required_columns_fail_cleanly() -> None:
    left = pd.DataFrame({"path": ["a.png"], "sha256": ["abc"]})
    right = _manifest([{"path": "b.png", "label": "tb", "sha256": "def", "dhash": "0" * 16}])
    with pytest.raises(ValueError, match="missing columns"):
        compare_manifests(left, right)
