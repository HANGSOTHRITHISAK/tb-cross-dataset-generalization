from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.data.datasets import ManifestDataset
from src.data.frozen_split import load_frozen_tbx11k
from src.data.manifest import file_sha256, save_manifest
from src.data.schema import SampleRecord
from src.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transform,
    seed_everything,
)
from src.models import build_densenet121
from src.training import smoke

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frozen_fixture(tmp_path):
    root = tmp_path / "imgs"
    root.mkdir()
    records, rows = [], []
    for index, partition in enumerate(
        ["train", "train", "train", "validation", "internal_test"]
    ):
        relative = f"health/h{index}.png"
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        # Distinct PNG bytes for strict retained-hash uniqueness.
        Image.fromarray(np.full((64, 64), 40 + index * 20, dtype=np.uint8)).save(path)
        sample_id = f"tbx11k:{relative}"
        record = SampleRecord(
            sample_id=sample_id,
            image_path=path,
            dataset="tbx11k",
            original_label="Healthy",
            canonical_label=0,
            source_partition="official_trainval_release",
            checksum_sha256=file_sha256(path),
            metadata={"relative_release_path": relative},
        )
        records.append(record)
        rows.append(
            {
                "sample_id": sample_id,
                "original_label": "Healthy",
                "internal_label": "0",
                "internal_class": "healthy",
                "canonical_binary_label": "0",
                "split": partition,
                "checksum_sha256": record.checksum_sha256,
                "duplicate_family_id": "dup-"
                + hashlib.sha256(sample_id.encode()).hexdigest()[:16],
                "duplicate_family_size": "1",
                "duplicate_member_ids": json.dumps([sample_id]),
            }
        )
    manifest, assignments = tmp_path / "manifest.csv", tmp_path / "split.csv"
    save_manifest(records, manifest)
    _write_rows(assignments, rows)
    return manifest, assignments, rows, records, root


def _write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_repository_frozen_join_preserves_every_assignment():
    assignments = ROOT / "data/manifests/tbx11k_split.csv"
    before = file_sha256(assignments)
    records = load_frozen_tbx11k(
        ROOT / "data/manifests/tbx11k.csv",
        assignments,
        summary=ROOT / "data/audit/tbx11k_split_summary.json",
        expected_split_sha256="4638fb21c180a3e05231d9ff682319bf4ddf36bf2b24643fa43e20776ddabf4e",
    )
    with assignments.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(records) == 8257
    assert [(r.sample_id, r.split, r.checksum_sha256) for r in records] == [
        (r["sample_id"], r["split"], r["checksum_sha256"]) for r in rows
    ]
    assert file_sha256(assignments) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_id", ""),
        ("sample_id", "missing"),
        ("original_label", "TB"),
        ("internal_label", "2"),
        ("internal_class", "tb"),
        ("canonical_binary_label", "1"),
        ("checksum_sha256", "0" * 64),
        ("split", "test"),
        ("duplicate_family_id", "bad"),
        ("duplicate_family_size", "2"),
        ("duplicate_member_ids", "[]"),
        ("duplicate_member_ids", '["missing"]'),
    ],
)
def test_invalid_assignment_rejected(frozen_fixture, field, value):
    manifest, assignments, rows, _, _ = frozen_fixture
    rows[0][field] = value
    _write_rows(assignments, rows)
    with pytest.raises(ValueError):
        load_frozen_tbx11k(manifest, assignments)


def test_duplicate_split_and_manifest_ids_rejected(frozen_fixture):
    manifest, assignments, rows, records, _ = frozen_fixture
    _write_rows(assignments, rows + [rows[0]])
    with pytest.raises(ValueError, match="duplicate"):
        load_frozen_tbx11k(manifest, assignments)
    _write_rows(assignments, rows)
    save_manifest(records + [records[0]], manifest)
    with pytest.raises(ValueError, match="duplicate"):
        load_frozen_tbx11k(manifest, assignments)


def test_duplicate_hash_rejected(frozen_fixture):
    manifest, assignments, rows, records, _ = frozen_fixture
    records[1] = replace(records[1], checksum_sha256=records[0].checksum_sha256)
    rows[1]["checksum_sha256"] = records[0].checksum_sha256
    save_manifest(records, manifest)
    _write_rows(assignments, rows)
    with pytest.raises(ValueError, match="Duplicate retained"):
        load_frozen_tbx11k(manifest, assignments)


def test_relocation_and_unsafe_paths(frozen_fixture, tmp_path):
    manifest, assignments, _, records, root = frozen_fixture
    # Audited paths may be stale; relocation uses the audited release-relative path.
    save_manifest(
        [replace(r, image_path=Path("old") / r.image_path.name) for r in records],
        manifest,
    )
    loaded = load_frozen_tbx11k(manifest, assignments, image_root=root)
    assert loaded[0].image_path == root / "health/h0.png"
    for bad in [
        "../escape.png",
        "C:/private.png",
        "/absolute.png",
        "health\\h0.png",
        "health/other.png",
    ]:
        save_manifest(
            [replace(records[0], metadata={"relative_release_path": bad})]
            + records[1:],
            manifest,
        )
        with pytest.raises(ValueError, match="relative_release_path"):
            load_frozen_tbx11k(manifest, assignments, image_root=root)


def test_missing_member_and_conflicting_member_rejected(frozen_fixture):
    manifest, assignments, rows, records, _ = frozen_fixture
    members = [records[0].sample_id, "tbx11k:health/z.png"]
    rows[0].update(
        duplicate_member_ids=json.dumps(members),
        duplicate_family_size="2",
        duplicate_family_id="dup-"
        + hashlib.sha256("\n".join(members).encode()).hexdigest()[:16],
    )
    _write_rows(assignments, rows)
    with pytest.raises(ValueError, match="member missing"):
        load_frozen_tbx11k(manifest, assignments)
    other = replace(
        records[0], sample_id=members[1], original_label="TB", canonical_label=1
    )
    save_manifest(records + [other], manifest)
    with pytest.raises(ValueError, match="Conflicting"):
        load_frozen_tbx11k(manifest, assignments)


def test_split_lock_and_summary_binding(frozen_fixture, tmp_path):
    manifest, assignments, _, _, _ = frozen_fixture
    with pytest.raises(ValueError, match="split file hash"):
        load_frozen_tbx11k(manifest, assignments, expected_split_sha256="0" * 64)
    summary = tmp_path / "summary.json"
    evidence = {
        "inputs": {"manifest_sha256": "bad"},
        "retained_unique_images": 5,
        "split_class_counts": {},
    }
    summary.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="manifest hash"):
        load_frozen_tbx11k(manifest, assignments, summary=summary)
    evidence["inputs"]["manifest_sha256"] = file_sha256(manifest)
    summary.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="class counts"):
        load_frozen_tbx11k(manifest, assignments, summary=summary)
    evidence["retained_unique_images"] = 4
    summary.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="count mismatch"):
        load_frozen_tbx11k(manifest, assignments, summary=summary)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P"])
def test_eval_transform_channels_dtype_and_determinism(mode):
    image = Image.new(mode, (45, 38))
    transform = build_transform()
    a, b = transform(image), transform(image)
    assert a.shape == (3, 224, 224) and a.dtype == torch.float32
    assert torch.isfinite(a).all() and torch.equal(a, b)
    assert torch.allclose(
        a[:, 0, 0],
        torch.tensor([-mean / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)]),
    )


def test_train_seed_and_evaluation_has_no_random_affine():
    image = Image.fromarray(np.arange(64 * 64, dtype=np.uint8).reshape(64, 64))
    train, evaluation = build_transform(training=True), build_transform()
    seed_everything(42)
    a = train(image)
    seed_everything(42)
    assert torch.equal(a, train(image))
    assert not torch.equal(a, train(image))
    assert not any(
        isinstance(t, smoke.torchvision.transforms.RandomAffine)
        for t in evaluation.transforms
    )
    with pytest.raises(ValueError, match="image mode"):
        evaluation(Image.fromarray(np.zeros((32, 32), dtype=np.uint16)))


def test_dataset_internal_label_dtype_missing_files(frozen_fixture):
    _, _, _, records, _ = frozen_fixture
    dataset = ManifestDataset(
        records, build_transform(), label_space="tbx11k_internal_3class"
    )
    assert dataset[0]["label"].dtype == torch.int64
    assert dataset[0]["label"].item() == 0
    assert dataset[0]["image"].shape == (3, 224, 224)
    for original, internal, binary in [("Sick non-TB", 1, 0), ("TB", 2, 1)]:
        other = replace(records[0], original_label=original, canonical_label=binary)
        dataset = ManifestDataset(
            [other], build_transform(), label_space="tbx11k_internal_3class"
        )
        assert dataset[0]["label"].item() == internal
    with pytest.raises(ValueError, match="Unsupported label"):
        ManifestDataset(records, build_transform(), label_space="typo")
    records[0].image_path.unlink()
    with pytest.raises(FileNotFoundError):
        ManifestDataset(records, build_transform())


def test_densenet_finite_full_model_optimizer_step():
    # Offline CPU fixture: explicitly random weights, smaller spatial size.
    torch.set_num_threads(2)
    seed_everything(42)
    model = build_densenet121(pretrained=False)
    before = model.features.conv0.weight.detach().clone()
    result = smoke.optimizer_step(
        model,
        torch.rand(2, 3, 32, 32),
        torch.tensor([0, 2]),
        device=torch.device("cpu"),
        learning_rate=0.0001,
        weight_decay=0.0001,
    )
    assert result["output_shape"] == [2, 3] and result["classifier_updated"]
    assert np.isfinite(result["loss"])
    assert not torch.equal(before, model.features.conv0.weight.detach())


def test_pretrained_weights_fail_clearly_when_uncached(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    with pytest.raises(FileNotFoundError, match="explicitly allow"):
        build_densenet121()
    cache = tmp_path / "checkpoints/densenet121-a639ec97.pth"
    cache.parent.mkdir()
    cache.write_bytes(b"invalid cached weights")
    with pytest.raises(ValueError, match="published hash"):
        build_densenet121()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(3, 3)

    def forward(self, images):
        return self.classifier(images.mean(dim=(2, 3)))


def test_smoke_receipt_training_only_and_failure(frozen_fixture, tmp_path, monkeypatch):
    manifest, assignments, _, records, _ = frozen_fixture
    # Make held-out images unavailable: a training smoke must still succeed.
    for record in records[3:]:
        record.image_path.unlink()
    monkeypatch.setattr(smoke, "build_densenet121", lambda **kwargs: TinyModel())
    path = tmp_path / "receipt.json"
    result = smoke.run_smoke(
        manifest=manifest,
        assignments=assignments,
        receipt_path=path,
        pretrained=False,
        image_evidence="synthetic_fixture",
    )
    receipt = json.loads(path.read_text())
    assert result == receipt and receipt["status"] == "passed"
    assert receipt["optimizer_steps"] == 1 and receipt["partition"] == "train"
    assert not receipt["real_image_batch_processed"]
    assert set(receipt["sample_ids"]).issubset({r.sample_id for r in records[:3]})
    assert str(tmp_path) not in path.read_text()
    records[0].image_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        smoke.run_smoke(
            manifest=manifest,
            assignments=assignments,
            receipt_path=path,
            pretrained=False,
            batch_size=3,
        )
    assert json.loads(path.read_text())["status"] == "failed"
    assert json.loads(path.read_text())["optimizer_steps"] == 0


def test_nonfinite_loss_rejected():
    with pytest.raises(ValueError, match="Non-finite"):
        smoke.optimizer_step(
            TinyModel(),
            torch.full((2, 3, 32, 32), float("nan")),
            torch.tensor([0, 2]),
            device=torch.device("cpu"),
            learning_rate=0.0001,
            weight_decay=0,
        )


def test_monitor_unavailable_and_thermal_stop(monkeypatch):
    monkeypatch.setattr(
        subprocess := smoke.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a, 0, stdout="GPU-test,RTX test,86,50,100,Not Active\n"
        ),
    )
    monitor = smoke.GPUMonitor()
    monitor.sample()
    with pytest.raises(RuntimeError, match="temperature"):
        monitor.check()

    def unavailable(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess, "run", unavailable)
    monitor = smoke.GPUMonitor()
    monitor.sample()
    assert monitor.unavailable_reason == "FileNotFoundError"


@pytest.mark.parametrize(
    "settings",
    [
        {"batch_size": 1},
        {"batch_size": 5},
        {"image_size": 225},
        {"learning_rate": float("nan")},
        {"weight_decay": -1},
        {"amp": True},
        {"device": "mps"},
    ],
)
def test_bounded_settings_write_failed_receipt(frozen_fixture, tmp_path, settings):
    manifest, assignments, _, _, _ = frozen_fixture
    receipt = tmp_path / "failed.json"
    with pytest.raises(ValueError):
        smoke.run_smoke(
            manifest=manifest, assignments=assignments, receipt_path=receipt, **settings
        )
    assert json.loads(receipt.read_text())["status"] == "failed"
    assert json.loads(receipt.read_text())["optimizer_steps"] == 0


def test_unavailable_cuda_has_no_fallback(frozen_fixture, tmp_path, monkeypatch):
    manifest, assignments, _, _, _ = frozen_fixture
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    receipt = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        smoke.run_smoke(
            manifest=manifest,
            assignments=assignments,
            receipt_path=receipt,
            device="cuda",
        )
    assert json.loads(receipt.read_text())["optimizer_steps"] == 0


def test_receipt_cannot_overwrite_inputs(frozen_fixture):
    manifest, assignments, _, _, _ = frozen_fixture
    before = assignments.read_bytes()
    with pytest.raises(ValueError, match="inputs cannot be overwritten"):
        smoke.run_smoke(
            manifest=manifest, assignments=assignments, receipt_path=assignments
        )
    assert assignments.read_bytes() == before


def test_ambiguous_split_columns_and_empty_rows(frozen_fixture):
    manifest, assignments, _, _, _ = frozen_fixture
    assignments.write_text("sample_id,sample_id\na,b\n")
    with pytest.raises(ValueError, match="Ambiguous"):
        load_frozen_tbx11k(manifest, assignments)
    assignments.write_text("sample_id\na\n")
    with pytest.raises(ValueError, match="required columns"):
        load_frozen_tbx11k(manifest, assignments)
