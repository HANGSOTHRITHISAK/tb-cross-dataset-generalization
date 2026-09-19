from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE2_ARTIFACTS = (ROOT / "data" / "manifests", ROOT / "data" / "audit")
MACHINE_LOCAL_PATH = re.compile(
    r"(?i)(?:\b[a-z]:[\\/]+(?:projects|users)[\\/]+|"
    r"(?<![:/])/(?:home|users|mnt|tmp|workspaces?)/)"
)
RESTRICTED_NLM_METADATA = re.compile(
    r'(?i)"(?:age_years|sex_source|clinical_reading_sha256)"'
)


def _versioned_artifacts() -> list[Path]:
    return [path for directory in PHASE2_ARTIFACTS for path in directory.rglob("*") if path.is_file()]


def test_phase2_artifacts_do_not_contain_machine_local_paths() -> None:
    offenders = [
        str(path.relative_to(ROOT))
        for path in _versioned_artifacts()
        if MACHINE_LOCAL_PATH.search(path.read_text(encoding="utf-8-sig"))
    ]
    assert not offenders, f"Machine-local paths in Phase 2 artifacts: {offenders}"


def test_nlm_public_artifacts_omit_derived_clinical_metadata() -> None:
    artifacts = [
        ROOT / "data" / "manifests" / "shenzhen.csv",
        ROOT / "data" / "manifests" / "montgomery.csv",
        ROOT / "data" / "audit" / "shenzhen_audit.json",
        ROOT / "data" / "audit" / "montgomery_audit.json",
    ]
    offenders = [
        str(path.relative_to(ROOT))
        for path in artifacts
        if RESTRICTED_NLM_METADATA.search(path.read_text(encoding="utf-8-sig"))
    ]
    assert not offenders, f"Restricted clinical metadata in public artifacts: {offenders}"


def test_machine_local_path_detector_covers_windows_and_posix() -> None:
    local_examples = [
        r"C:\Projects\CV-Generalization\data\audit.json",
        r"C:\\Projects\\CV-Generalization\\data\\audit.json",
        "/home/alice/project/data/audit.json",
        "/Users/alice/project/data/audit.json",
        "/mnt/data/project/data/audit.json",
        "/tmp/project/data/audit.json",
        "/workspace/project/data/audit.json",
    ]
    assert all(MACHINE_LOCAL_PATH.search(value) for value in local_examples)
    assert not MACHINE_LOCAL_PATH.search("../raw/tbx11k/TBX11K")
    assert not MACHINE_LOCAL_PATH.search(
        "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/"
    )
