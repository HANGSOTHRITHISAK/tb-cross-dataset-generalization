from __future__ import annotations

import re


def _key(value: object) -> str:
    return re.sub(r"[\s_&+/-]+", " ", str(value).strip().casefold()).strip()


# Canonical binary label space used for cross-dataset harmonization:
# 0 = non-TB, 1 = TB.
LABEL_MAPS: dict[str, dict[str, int]] = {
    "tbx11k": {
        "healthy": 0,
        "sick non tb": 0,
        "sick but non tb": 0,
        "tb": 1,
        "tb (subtype unspecified in released path)": 1,
        "active tb": 1,
        "latent tb": 1,
        "active latent tb": 1,
        "uncertain tb": 1,
    },
    "shenzhen": {
        "normal": 0,
        "0": 0,
        "tb": 1,
        "tuberculosis": 1,
        "abnormal": 1,
        "abnormal tb consistent": 1,
        "tb consistent abnormal": 1,
        "1": 1,
    },
    "montgomery": {
        "normal": 0,
        "0": 0,
        "tb": 1,
        "tuberculosis": 1,
        "abnormal": 1,
        "abnormal tb consistent": 1,
        "tb consistent abnormal": 1,
        "1": 1,
    },
    "synthetic": {"normal": 0, "tb": 1, "0": 0, "1": 1},
}


# Approved internal TBX11K training/evaluation label space:
# 0 = healthy, 1 = sick non-TB, 2 = TB.
#
# The released train/validation material exposes TB under one tb/ category.
# Active/latent/active+latent subtype labels are not inferred.
TBX11K_INTERNAL_CLASS_NAMES: dict[int, str] = {
    0: "healthy",
    1: "sick_non_tb",
    2: "tb",
}

TBX11K_INTERNAL_LABELS: dict[str, int] = {
    "healthy": 0,
    "sick non tb": 1,
    "sick but non tb": 1,
    "tb": 2,
    "tb (subtype unspecified in released path)": 2,
}


def normalize_label(dataset: str, original_label: object) -> int:
    """Map an audited dataset label into the shared binary TB/non-TB space."""
    dataset_key = dataset.strip().casefold()
    try:
        return LABEL_MAPS[dataset_key][_key(original_label)]
    except KeyError as error:
        raise ValueError(
            f"Unverified label {original_label!r} for {dataset!r}; add an audited explicit mapping"
        ) from error


def normalize_tbx11k_internal_label(original_label: object) -> int:
    """Map a released TBX11K train/validation label into the 3-class internal space."""
    key = _key(original_label)
    try:
        return TBX11K_INTERNAL_LABELS[key]
    except KeyError as error:
        raise ValueError(
            f"Unverified TBX11K internal label {original_label!r}; "
            "do not infer unreleased TB subtype labels"
        ) from error


def harmonize_tbx11k_internal_to_binary(internal_label: int) -> int:
    """Collapse healthy/sick-non-TB/TB into non-TB/TB for external evaluation."""
    if internal_label in (0, 1):
        return 0
    if internal_label == 2:
        return 1
    raise ValueError(f"Unknown TBX11K internal class: {internal_label!r}")
