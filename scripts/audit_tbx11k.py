from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.data.manifest import load_manifest, save_manifest
from src.data.phase2_audit import difference_hash, hamming_distance
from src.data.schema import SampleRecord

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/tbx11k/TBX11K"
ARCHIVE = ROOT / "data/raw/TBX11K.zip"
MANIFEST = ROOT / "data/manifests/tbx11k.csv"
AUDIT = ROOT / "data/audit"


def sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def read_lists():
    return {
        p.stem: [
            x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()
        ]
        for p in sorted((RAW / "lists").glob("*.txt"))
    }


def parse_boxes(p):
    if not p.is_file():
        return ()
    out = []
    for o in ET.parse(p).getroot().findall(".//object"):
        b = o.find("bndbox")
        name = o.findtext("name") or ""
        if b is None:
            continue
        try:
            vals = [float(b.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax")]
        except ValueError:
            continue
        out.append(
            {
                "label": name,
                "xmin": vals[0],
                "ymin": vals[1],
                "xmax": vals[2],
                "ymax": vals[3],
            }
        )
    return tuple(out)


def info(p):
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            ex = im.getextrema()
            lo = min(x[0] for x in ex) if isinstance(ex[0], tuple) else ex[0]
            hi = max(x[1] for x in ex) if isinstance(ex[0], tuple) else ex[1]
            return (
                im.width,
                im.height,
                im.mode,
                16 if im.mode.startswith("I;16") else 8,
                lo,
                hi,
                None,
            )
    except (OSError, ValueError, UnidentifiedImageError) as e:
        return None, None, None, None, None, None, str(e)


def zip_inventory():
    with zipfile.ZipFile(ARCHIVE) as z:
        fs = [i for i in z.infolist() if not i.is_dir()]
        return {
            "archive_path": str(ARCHIVE),
            "archive_sha256": sha256(ARCHIVE),
            "archive_bytes": ARCHIVE.stat().st_size,
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


def main():
    lists = read_lists()
    memberships = defaultdict(list)
    for n, rows in lists.items():
        for r in rows:
            memberships[r].append(n)
    paths = sorted(
        p
        for p in (RAW / "imgs").rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    recs = []
    corrupt = []
    hashes = defaultdict(list)
    dhashes = {}
    dims = Counter()
    modes = Counter()
    labels = Counter()
    orig = Counter()
    parts = Counter()
    boxlabels = Counter()
    boxcounts = Counter()
    no_list = []
    archive_hash = sha256(ARCHIVE)
    for p in paths:
        rel = p.relative_to(RAW / "imgs").as_posix()
        top = rel.split("/")[0]
        mem = sorted(memberships.get(rel, []))
        if not mem:
            no_list.append(rel)
        ext_source = None
        if top == "health":
            original = "Healthy"
            canonical = 0
        elif top == "sick":
            original = "Sick & Non-TB"
            canonical = 0
        elif top == "tb":
            original = "TB (subtype unspecified in released path)"
            canonical = 1
        elif top == "test":
            original = "Official test (ground truth not released)"
            canonical = None
        elif top == "extra":
            ext_source = rel.split("/")[1]
            original = "Bundled external image (source label encoded in filename)"
            m = re.search(r"_([01])\.[^.]+$", p.name)
            canonical = int(m.group(1)) if m else None
        else:
            original = "Unknown release path"
            canonical = None
        if top == "extra":
            split = (
                "supplemental_train"
                if "/train/" in rel
                else "supplemental_validation"
                if "/val/" in rel
                else "supplemental_unknown"
            )
            source = "bundled_external_subset"
        elif top == "test":
            split = "official_test"
            source = "official_test_list"
        elif top in {"health", "sick", "tb"}:
            split = (
                "official_train"
                if "TBX11K_train" in mem and "TBX11K_val" not in mem
                else "official_validation"
                if "TBX11K_val" in mem and "TBX11K_train" not in mem
                else "official_trainval"
            )
            source = "official_trainval_release"
        else:
            split = None
            source = None
        w, h, mode, depth, lo, hi, error = info(p)
        flags = []
        if w is None:
            flags.append("corrupt_or_unreadable")
            corrupt.append({"path": rel, "error": error})
        else:
            dims[f"{w}x{h}"] += 1
            modes[mode] += 1
            dhashes[rel] = difference_hash(p)
        if canonical is None:
            flags.append("ground_truth_not_released")
        if not mem:
            flags.append("not_in_official_list")
        if ext_source:
            flags.append("bundled_external_source")
        xml = RAW / "annotations/xml" / f"{p.stem}.xml"
        boxes = parse_boxes(xml)
        for b in boxes:
            boxlabels[b["label"]] += 1
        boxcounts[len(boxes)] += 1 if boxes else 0
        checksum = sha256(p)
        hashes[checksum].append(rel)
        labels["unknown" if canonical is None else str(canonical)] += 1
        orig[original] += 1
        parts[split or "unassigned"] += 1
        recs.append(
            SampleRecord(
                sample_id=f"tbx11k:{rel}",
                image_path=Path("..")
                / "raw"
                / "tbx11k"
                / "TBX11K"
                / "imgs"
                / Path(rel),
                dataset="tbx11k",
                original_label=original,
                canonical_label=canonical,
                split=split,
                source_partition=source,
                annotation_path=Path("..")
                / "raw"
                / "tbx11k"
                / "TBX11K"
                / "annotations"
                / "xml"
                / xml.name
                if xml.is_file()
                else None,
                width=w,
                height=h,
                image_mode=mode,
                bit_depth=depth,
                checksum_sha256=checksum,
                audit_flags=tuple(flags),
                provenance={
                    "source_url": "https://github.com/yun-liu/Tuberculosis",
                    "retrieval_date": "2026-09-19",
                    "archive_sha256": archive_hash,
                },
                bounding_boxes=boxes,
                metadata={
                    "relative_release_path": rel,
                    "official_list_memberships": mem,
                    "bundled_external_source": ext_source,
                    "patient_group_identifier_available": False,
                    "observed_min": lo,
                    "observed_max": hi,
                    "dhash_256": dhashes.get(rel),
                    "label_basis": "directory label for health/sick/tb; official test label intentionally blank; subtype of tb folder not exposed in release lists",
                    "ground_truth_released": canonical is not None,
                },
            )
        )
    AUDIT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    save_manifest(recs, MANIFEST)
    exact = [
        {"checksum_sha256": c, "paths": ps}
        for c, ps in sorted(hashes.items())
        if len(ps) > 1
    ]
    near = []
    bands = defaultdict(list)
    for r, d in dhashes.items():
        for b in range(8):
            bands[(b, d[b * 8 : (b + 1) * 8])].append(r)
    seen = set()
    for bucket in bands.values():
        if len(bucket) > 50:
            continue
        for a, b in itertools.combinations(sorted(bucket), 2):
            if (a, b) in seen:
                continue
            seen.add((a, b))
            dist = hamming_distance(dhashes[a], dhashes[b])
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
    external_exact_ids = defaultdict(set)
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
                external_exact_ids[dataset].add(r.sample_id)
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
                sorted({k for row in rows for k in row})
                if rows
                else ["tbx_sample_id", "external_sample_id", "dataset"]
            )
            with (AUDIT / name).open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
    exclusions = []
    for r in recs:
        reasons = []
        if any(r.sample_id in ids for ids in external_exact_ids.values()):
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
    with (AUDIT / "tbx11k_exclusions.csv").open("w", newline="", encoding="utf-8") as f:
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
    exids = {x["sample_id"] for x in exclusions}
    candidate = [r for r in primary if r.sample_id not in exids]
    summary = {
        "dataset": "tbx11k",
        "archive": zip_inventory(),
        "release_root": str(RAW),
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
        "official_list_counts": {n: len(v) for n, v in sorted(lists.items())},
        "partition_counts": dict(sorted(parts.items())),
        "original_label_counts": dict(sorted(orig.items())),
        "canonical_label_counts_including_unknown": dict(sorted(labels.items())),
        "dimensions": dict(dims.most_common()),
        "modes": dict(modes),
        "global_observed_min": min(
            (
                r.metadata["observed_min"]
                for r in recs
                if r.metadata["observed_min"] is not None
            ),
            default=None,
        ),
        "global_observed_max": max(
            (
                r.metadata["observed_max"]
                for r in recs
                if r.metadata["observed_max"] is not None
            ),
            default=None,
        ),
        "corrupt_images": corrupt,
        "exact_duplicate_groups": exact,
        "near_duplicate_candidates": near,
        "near_duplicate_method": "shared 32-bit dHash band candidate generation, Hamming distance <= 8; candidates require review and are not auto-excluded",
        "annotation_xml_count": len(list((RAW / "annotations/xml").glob("*.xml"))),
        "annotation_json_files": sorted(
            p.name for p in (RAW / "annotations/json").glob("*.json")
        ),
        "annotation_box_label_counts": dict(sorted(boxlabels.items())),
        "annotation_images_linked": sum(bool(r.annotation_path) for r in recs),
        "patient_group_ids_available": False,
        "test_ground_truth_available": False,
        "no_list_membership": no_list,
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
    (AUDIT / "tbx11k_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (AUDIT / "tbx11k_zip_inventory.json").write_text(
        json.dumps(summary["archive"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(MANIFEST),
                "audit": str(AUDIT / "tbx11k_audit.json"),
                "images": len(recs),
                "corrupt": len(corrupt),
                "exact_duplicates": len(exact),
                "near_candidates": len(near),
                "candidate_pool": len(candidate),
                "label_counts": summary["cleaned_candidate_pool"][
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
