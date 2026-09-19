from __future__ import annotations

import csv
import hashlib
import itertools
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from src.data.manifest import load_manifest
from src.data.phase2_audit import hamming_distance

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/tbx11k.csv"
RAW = ROOT / "data/raw/tbx11k/TBX11K"
ARCH = ROOT / "data/raw/TBX11K.zip"
AUD = ROOT / "data/audit"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def main():
    recs = load_manifest(MANIFEST)
    lists = {
        p.stem: [
            x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()
        ]
        for p in (RAW / "lists").glob("*.txt")
    }
    archive_sha = sha(ARCH)
    with zipfile.ZipFile(ARCH) as z:
        fs = [i for i in z.infolist() if not i.is_dir()]
        zi = {
            "archive_path": ARCH.relative_to(ROOT).as_posix(),
            "archive_sha256": archive_sha,
            "archive_bytes": ARCH.stat().st_size,
            "member_count": len(z.infolist()),
            "file_count": len(fs),
            "compressed_bytes": sum(i.compress_size for i in fs),
            "uncompressed_bytes": sum(i.file_size for i in fs),
            "bad_member": z.testzip(),
            "top_level_counts": dict(
                sorted(
                    Counter(
                        i.filename.replace("\\", "/").split("/")[0] for i in fs
                    ).items()
                )
            ),
            "extensions": dict(
                sorted(Counter(Path(i.filename).suffix.lower() for i in fs).items())
            ),
        }
    (AUD / "tbx11k_zip_inventory.json").write_text(
        json.dumps(zi, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checks = defaultdict(list)
    for r in recs:
        checks[r.checksum_sha256].append(r.sample_id)
    exact = [
        {"checksum_sha256": c, "sample_ids": v}
        for c, v in sorted(checks.items())
        if c and len(v) > 1
    ]
    dh = {
        r.sample_id: r.metadata.get("dhash_256")
        for r in recs
        if r.metadata.get("dhash_256")
    }
    bands = defaultdict(list)
    for sid, d in dh.items():
        for b in range(8):
            bands[(b, d[b * 8 : (b + 1) * 8])].append(sid)
    near = []
    seen = set()
    for vals in bands.values():
        if len(vals) > 50:
            continue
        for a, b in itertools.combinations(sorted(vals), 2):
            if (a, b) in seen:
                continue
            seen.add((a, b))
            dist = hamming_distance(dh[a], dh[b])
            if 0 < dist <= 8:
                near.append(
                    {
                        "left": a,
                        "right": b,
                        "dhash_distance": dist,
                        "candidate_method": "shared_32bit_dhash_band",
                    }
                )
    exact_cross = {}
    near_cross = {}
    exact_ids = set()
    for dataset in ("shenzhen", "montgomery"):
        ext = load_manifest(ROOT / f"data/manifests/{dataset}.csv")
        by = {r.checksum_sha256: r for r in ext if r.checksum_sha256}
        er = []
        nr = []
        for r in recs:
            if r.checksum_sha256 in by:
                e = by[r.checksum_sha256]
                er.append(
                    {
                        "tbx_sample_id": r.sample_id,
                        "tbx_image_path": str(r.image_path),
                        "external_sample_id": e.sample_id,
                        "external_image_path": str(e.image_path),
                        "dataset": dataset,
                        "checksum_sha256": r.checksum_sha256,
                    }
                )
                exact_ids.add(r.sample_id)
            d = r.metadata.get("dhash_256")
            if d:
                for e in ext:
                    ed = e.metadata.get("dhash_256")
                    if ed:
                        dist = hamming_distance(str(d), str(ed))
                        if dist <= 8:
                            nr.append(
                                {
                                    "tbx_sample_id": r.sample_id,
                                    "external_sample_id": e.sample_id,
                                    "dataset": dataset,
                                    "dhash_distance": dist,
                                }
                            )
        exact_cross[dataset] = er
        near_cross[dataset] = nr
        for name, rows in (
            (f"tbx11k_vs_{dataset}_exact_duplicates.csv", er),
            (f"tbx11k_vs_{dataset}_near_duplicates.csv", nr),
        ):
            fields = (
                sorted({k for x in rows for k in x})
                if rows
                else ["tbx_sample_id", "external_sample_id", "dataset"]
            )
            with (AUD / name).open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
    exclusions = []
    for r in recs:
        reasons = []
        if r.sample_id in exact_ids:
            reasons.append("confirmed_exact_overlap_with_audited_external_dataset")
        if "bundled_external_source" in r.audit_flags:
            reasons.append("bundled_external_subset_not_independent_of_source_dataset")
        if reasons:
            exclusions.append(
                {
                    "sample_id": r.sample_id,
                    "image_path": str(r.image_path),
                    "canonical_label": ""
                    if r.canonical_label is None
                    else r.canonical_label,
                    "exclusion_reason": ";".join(reasons),
                }
            )
    with (AUD / "tbx11k_exclusions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "image_path",
                "canonical_label",
                "exclusion_reason",
            ],
        )
        w.writeheader()
        w.writerows(exclusions)
    primary = [
        r
        for r in recs
        if r.sample_id.split(":", 1)[1].split("/", 1)[0] in {"health", "sick", "tb"}
        and r.canonical_label is not None
    ]
    ex = {x["sample_id"] for x in exclusions}
    candidate = [r for r in primary if r.sample_id not in ex]
    dims = Counter(f"{r.width}x{r.height}" for r in recs)
    modes = Counter(r.image_mode for r in recs)
    corrupt = [r.sample_id for r in recs if "corrupt_or_unreadable" in r.audit_flags]
    boxes = Counter()
    anns = 0
    for p in (RAW / "annotations/xml").glob("*.xml"):
        import xml.etree.ElementTree as ET

        root = ET.parse(p).getroot()
        names = [o.findtext("name") or "" for o in root.findall(".//object")]
        boxes.update(names)
        anns += 1
    summary = {
        "dataset": "tbx11k",
        "archive": zi,
        "release_root": RAW.relative_to(ROOT).as_posix(),
        "image_count_all_extracted": len(recs),
        "nominal_official_tbx11k_image_count": 11200,
        "official_trainval_labeled_image_count": 8400,
        "official_test_list_image_count": len(lists.get("all_test", [])),
        "official_test_nominal_tbx11k_count": 2800,
        "release_path_image_count_including_mixed_test": sum(
            r.sample_id.split(":", 1)[1].split("/", 1)[0]
            in {"health", "sick", "tb", "test"}
            for r in recs
        ),
        "primary_tbx11k_image_count": 11200,
        "bundled_external_extra_image_count": sum(
            "bundled_external_source" in r.audit_flags for r in recs
        ),
        "raw_file_count": sum(p.is_file() for p in RAW.rglob("*")),
        "raw_bytes": sum(p.stat().st_size for p in RAW.rglob("*") if p.is_file()),
        "official_list_counts": {k: len(v) for k, v in sorted(lists.items())},
        "partition_counts": dict(
            sorted(Counter(r.split or "unassigned" for r in recs).items())
        ),
        "original_label_counts": dict(
            sorted(Counter(r.original_label for r in recs).items())
        ),
        "canonical_label_counts_including_unknown": dict(
            sorted(
                Counter(
                    "unknown" if r.canonical_label is None else str(r.canonical_label)
                    for r in recs
                ).items()
            )
        ),
        "dimensions": dict(dims.most_common()),
        "modes": dict(modes),
        "global_observed_min": min(
            (
                r.metadata.get("observed_min")
                for r in recs
                if r.metadata.get("observed_min") is not None
            ),
            default=None,
        ),
        "global_observed_max": max(
            (
                r.metadata.get("observed_max")
                for r in recs
                if r.metadata.get("observed_max") is not None
            ),
            default=None,
        ),
        "corrupt_images": corrupt,
        "exact_duplicate_groups": exact,
        "near_duplicate_candidates": near,
        "near_duplicate_method": "shared 32-bit dHash band candidate generation, Hamming distance <= 8; candidates require review and are not auto-excluded",
        "annotation_xml_count": anns,
        "annotation_json_files": sorted(
            p.name for p in (RAW / "annotations/json").glob("*.json")
        ),
        "annotation_box_label_counts": dict(sorted(boxes.items())),
        "annotation_images_linked": sum(bool(r.annotation_path) for r in recs),
        "patient_group_ids_available": False,
        "test_ground_truth_available": False,
        "exact_cross_overlap_counts": {k: len(v) for k, v in exact_cross.items()},
        "perceptual_cross_candidate_counts": {k: len(v) for k, v in near_cross.items()},
        "cleaned_candidate_pool": {
            "definition": "primary labeled health/sick/tb rows after confirmed external exact overlaps and bundled external rows are excluded; no split is frozen",
            "count": len(candidate),
            "canonical_label_counts": dict(
                sorted(Counter(r.canonical_label for r in candidate).items())
            ),
        },
        "pending_review": [
            "TBX11K test ground truth is not released; test rows retain canonical_label=None.",
            "The release path/list files expose a single tb/ folder; exact Active/Latent/Active+Latent/Uncertain subtype for those 800 labeled trainval rows is not recoverable from the archive and is not inferred.",
            "No trustworthy patient/group identifiers are present in the release.",
            "Perceptual candidates are flags for review, not automatic exclusions.",
        ],
    }
    (AUD / "tbx11k_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "images": len(recs),
                "corrupt": len(corrupt),
                "exact_groups": len(exact),
                "near_candidates": len(near),
                "candidate_pool": len(candidate),
                "candidate_labels": summary["cleaned_candidate_pool"][
                    "canonical_label_counts"
                ],
                "cross_exact": summary["exact_cross_overlap_counts"],
                "cross_near": summary["perceptual_cross_candidate_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
