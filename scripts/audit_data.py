from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from src.data.phase2_audit import (
    audit_nlm_dataset,
    write_cross_dataset_duplicates,
    write_cross_dataset_near_duplicates,
)

SOURCES = {
    "shenzhen": "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html",
    "montgomery": "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit acquired NLM TB datasets")
    parser.add_argument("datasets", nargs="+", choices=SOURCES)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    manifests = []
    for dataset in args.datasets:
        manifest = args.data_root / "manifests" / f"{dataset}.csv"
        summary = audit_nlm_dataset(
            args.data_root / "raw" / dataset,
            dataset,
            manifest,
            args.data_root / "audit",
            source_url=SOURCES[dataset],
            retrieval_date=datetime.now(UTC).date().isoformat(),
        )
        manifests.append(manifest)
        print(json.dumps(summary, indent=2))
    if len(manifests) > 1:
        rows = write_cross_dataset_duplicates(
            manifests, args.data_root / "audit" / "cross_dataset_exact_duplicates.csv"
        )
        print(f"Cross-dataset exact duplicate groups: {len(rows)}")
        near_rows = write_cross_dataset_near_duplicates(
            manifests, args.data_root / "audit" / "cross_dataset_near_duplicates.csv"
        )
        print(f"Cross-dataset near-duplicate candidates: {len(near_rows)}")


if __name__ == "__main__":
    main()
