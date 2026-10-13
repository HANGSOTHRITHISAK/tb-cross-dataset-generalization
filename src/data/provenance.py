"""Execution evidence bound to the approved metadata and selected image bytes."""

from src.data.frozen_split import metadata_sha256
from src.data.manifest import file_sha256

APPROVED_MANIFEST = "85e6e6fe4a56ac8b7abeb29a68ac2a256ba1cc46164fd8dd6f1ab6f5187437cd"
APPROVED_SPLIT = "e4e69370c933f2b12b4b00df62b43ee5b7dee31e415ad67c75adfc84926c2227"


def validate_image_evidence(manifest, assignments, selected):
    """Only open selected development images; callers cannot assert real provenance.

    A fixture remains usable but cannot claim approved real TBX11K execution.
    Frozen join validation must precede this function.
    """
    for record in selected:
        if record.split not in {"train", "validation"}:
            raise ValueError("Image evidence requires a development partition")
        if file_sha256(record.image_path) != record.checksum_sha256:
            raise ValueError(f"Raw image checksum mismatch: {record.sample_id}")
    approved = (
        metadata_sha256(manifest) == APPROVED_MANIFEST
        and metadata_sha256(assignments) == APPROVED_SPLIT
    )
    return "verified_real_tbx11k" if approved else "unverified_or_fixture_inputs"
