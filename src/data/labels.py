from __future__ import annotations

import re


def _key(value: object) -> str:
    return re.sub(r"[\s_&+/-]+", " ", str(value).strip().casefold()).strip()


LABEL_MAPS: dict[str, dict[str, int]] = {
    "tbx11k": {
        "healthy": 0,
        "sick non tb": 0,
        "sick but non tb": 0,
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


def normalize_label(dataset: str, original_label: object) -> int:
    dataset_key = dataset.strip().casefold()
    try:
        return LABEL_MAPS[dataset_key][_key(original_label)]
    except KeyError as error:
        raise ValueError(
            f"Unverified label {original_label!r} for {dataset!r}; add an audited explicit mapping"
        ) from error
