from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE2_ARTIFACTS = (ROOT / "data" / "manifests", ROOT / "data" / "audit")
MACHINE_LOCAL_PATH = re.compile(r"(?i)\b[a-z]:[\\/](?:projects|users)[\\/]")
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