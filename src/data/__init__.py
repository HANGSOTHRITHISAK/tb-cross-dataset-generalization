"""Generic and Phase 2 dataset-audit utilities."""

from .datasets import ManifestDataset
from .labels import normalize_label
from .manifest import load_manifest, save_manifest
from .schema import SampleRecord
from .splits import assert_disjoint, deterministic_group_split

__all__ = [
    "ManifestDataset",
    "SampleRecord",
    "assert_disjoint",
    "deterministic_group_split",
    "load_manifest",
    "normalize_label",
    "save_manifest",
]