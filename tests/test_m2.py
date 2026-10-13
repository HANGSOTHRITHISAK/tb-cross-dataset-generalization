from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from src.data.datasets import ManifestDataset
from src.data.frozen_split import load_frozen_tbx11k
from src.data.manifest import file_sha256, save_manifest
from src.data.provenance import validate_image_evidence
from src.data.schema import SampleRecord
from src.data.transforms import build_transform
from src.training import pilot, smoke


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 4, 1)
        self.classifier = nn.Linear(4, 3)

    def forward(self, x):
        return self.classifier(self.features(x).mean((2, 3)))


@pytest.fixture
def development_fixture(tmp_path):
    records, rows = [], []
    for partition, per_class in [("train", 2), ("validation", 1), ("internal_test", 1)]:
        for label, (folder, name) in enumerate(
            zip(["health", "sick", "tb"], ["Healthy", "Sick non-TB", "TB"])
        ):
            for i in range(per_class):
                relative = f"{folder}/{partition}{i}.png"
                path = tmp_path / "imgs" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                # Held-out image bytes do not exist: they cannot be opened.
                if partition != "internal_test":
                    Image.fromarray(
                        np.full((32, 32), 20 + len(records) * 15, dtype=np.uint8)
                    ).save(path)
                checksum = (
                    file_sha256(path)
                    if path.exists()
                    else hashlib.sha256(relative.encode()).hexdigest()
                )
                sample_id = f"tbx11k:{relative}"
                records.append(
                    SampleRecord(
                        sample_id=sample_id,
                        image_path=path,
                        dataset="tbx11k",
                        original_label=name,
                        canonical_label=int(label == 2),
                        source_partition="official_trainval_release",
                        checksum_sha256=checksum,
                        metadata={"relative_release_path": relative},
                    )
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "original_label": name,
                        "internal_label": label,
                        "internal_class": pilot.CLASS_NAMES[label],
                        "canonical_binary_label": int(label == 2),
                        "split": partition,
                        "checksum_sha256": checksum,
                        "duplicate_family_id": "dup-"
                        + hashlib.sha256(sample_id.encode()).hexdigest()[:16],
                        "duplicate_family_size": 1,
                        "duplicate_member_ids": json.dumps([sample_id]),
                    }
                )
    manifest, assignments = tmp_path / "manifest.csv", tmp_path / "split.csv"
    save_manifest(records, manifest)
    with assignments.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return manifest, assignments


@pytest.fixture
def config():
    return pilot.PilotConfig(
        epochs=2,
        image_size=32,
        train_per_class=(2, 2, 2),
        validation_per_class=(1, 1, 1),
        max_steps=6,
        max_seconds=60,
    )


def run_fixture(development_fixture, tmp_path, config, monkeypatch, name="run"):
    monkeypatch.setattr(pilot, "build_densenet121", lambda **kw: TinyModel())
    manifest, assignments = development_fixture
    return pilot.run_pilot(
        manifest=manifest,
        assignments=assignments,
        output_dir=tmp_path / name,
        config=config,
        device="cpu",
        pretrained=False,
    )


def test_multistep_update_repeatability_and_checkpoint(
    development_fixture, tmp_path, config, monkeypatch
):
    first = run_fixture(development_fixture, tmp_path, config, monkeypatch)
    second = run_fixture(development_fixture, tmp_path, config, monkeypatch, "repeat")
    assert first["status"] == "passed" and first["optimizer_steps"] == 6
    assert first["classifier_updated"] and first["checkpoint_usable"]
    assert (
        first["model"]["initial_state_sha256"]
        == second["model"]["initial_state_sha256"]
    )
    assert first["final_state_sha256"] == second["final_state_sha256"]
    assert [s["sample_ids"] for s in first["steps"]] == [
        s["sample_ids"] for s in second["steps"]
    ]
    assert [s["loss"] for s in first["steps"]] == [s["loss"] for s in second["steps"]]
    checkpoint = torch.load(tmp_path / "run/best.pt", weights_only=True)
    best = max(
        first["history"],
        key=lambda h: (
            h["validation"]["macro_f1"],
            -h["validation"]["loss"],
            -h["epoch"],
        ),
    )
    assert checkpoint["metadata"]["epoch"] == best["epoch"]
    assert first["best_checkpoint"]["sha256"] == file_sha256(tmp_path / "run/best.pt")
    assert checkpoint["optimizer_state"]["state"][0]["step"].item() == best["epoch"] * 3
    assert first["image_evidence"] == "unverified_or_fixture_inputs"
    assert str(tmp_path) not in (tmp_path / "run/receipt.json").read_text()
    assert {sid for s in first["steps"] for sid in s["sample_ids"]} == set(
        first["subsets"]["train"]["sample_ids"]
    )
    assert not set(first["subsets"]["train"]["sample_ids"]) & set(
        first["subsets"]["validation"]["sample_ids"]
    )


def test_densenet_multistep_cpu_fixture(development_fixture, tmp_path, config):
    manifest, assignments = development_fixture
    result = pilot.run_pilot(
        manifest=manifest,
        assignments=assignments,
        output_dir=tmp_path / "dense",
        config=config,
        device="cpu",
        pretrained=False,
    )
    assert result["optimizer_steps"] == 6 and result["classifier_updated"]


def test_validation_does_not_update_and_has_no_augmentation(development_fixture):
    manifest, assignments = development_fixture
    records = [
        r for r in load_frozen_tbx11k(manifest, assignments) if r.split == "validation"
    ]
    dataset = ManifestDataset(
        records, build_transform(image_size=32), label_space="tbx11k_internal_3class"
    )
    assert torch.equal(dataset[0]["image"], dataset[0]["image"])
    model = TinyModel()
    before = pilot.state_sha256(model)
    result = pilot.evaluate(
        model,
        DataLoader(dataset, batch_size=2),
        device=torch.device("cpu"),
        guard=lambda: None,
    )
    assert result["sample_count"] == 3 and result["class_counts"] == [1, 1, 1]
    assert before == pilot.state_sha256(model) and all(
        p.grad is None for p in model.parameters()
    )
    assert result["partition"] == "validation"
    dataset.records[0] = replace(records[0], split="internal_test")
    with pytest.raises(ValueError, match="exclusively validation"):
        pilot.evaluate(
            model, DataLoader(dataset), device=torch.device("cpu"), guard=lambda: None
        )


def test_metrics_zero_division_and_class_order():
    result = pilot.classification_metrics([[2, 1, 0], [0, 0, 0], [1, 0, 0]])
    assert result["accuracy"] == 0.5
    assert result["macro_f1"] == pytest.approx(2 / 9)
    assert result["per_class_recall"] == [2 / 3, 0, 0]
    assert result["class_names"] == ["healthy", "sick_non_tb", "tb"]
    assert result["zero_division"] == 0


def test_selection_ties_and_validation_only():
    best = {"partition": "validation", "macro_f1": 0.5, "loss": 1.0, "epoch": 1}
    assert pilot.is_better(best, None)
    assert not pilot.is_better({**best, "epoch": 2}, best)
    assert pilot.is_better({**best, "loss": 0.9}, best)
    assert not pilot.is_better({**best, "macro_f1": 0.4, "loss": 0.01}, best)
    for partition in ["train", "internal_test", "shenzhen", "montgomery"]:
        with pytest.raises(ValueError, match="validation"):
            pilot.is_better({**best, "partition": partition}, best)
    with pytest.raises(ValueError, match="Non-finite"):
        pilot.is_better({**best, "loss": float("nan")}, best)


def test_weights_use_complete_train_only(
    development_fixture, tmp_path, config, monkeypatch
):
    manifest, assignments = development_fixture
    records = load_frozen_tbx11k(manifest, assignments)
    skewed = records + [records[0]] * 4
    assert pilot.training_weights(skewed) == pytest.approx([10 / 18, 10 / 6, 10 / 6])
    heldout = [replace(r, original_label="TB") for r in records if r.split != "train"]
    assert pilot.training_weights(skewed + heldout * 100) == pilot.training_weights(
        skewed
    )
    with pytest.raises(ValueError, match="All training classes"):
        pilot.training_weights([r for r in records if r.original_label != "TB"])
    result = run_fixture(
        development_fixture, tmp_path, replace(config, class_weighted=True), monkeypatch
    )
    assert result["imbalance"]["enabled"] and not result["imbalance"]["oversampling"]


def test_subset_membership_reproducible_and_heldout_rejected(development_fixture):
    records = load_frozen_tbx11k(*development_fixture)
    assert pilot.select_subset(records, "train", (1, 1, 1), 42) == pilot.select_subset(
        list(reversed(records)), "train", (1, 1, 1), 42
    )
    with pytest.raises(ValueError, match="permitted"):
        pilot.select_subset(records, "internal_test", (1, 1, 1), 42)
    with pytest.raises(ValueError, match="too small"):
        pilot.select_subset(records, "train", (3, 1, 1), 42)


@pytest.mark.parametrize(
    "exception,reason",
    [
        (
            ValueError("Non-finite gradient"),
            "invalid_or_nonfinite_input_state_or_configuration",
        ),
        (torch.cuda.OutOfMemoryError("fixture"), "cuda_out_of_memory"),
        (pilot.ResourceStop("temperature fixture"), "resource_guard"),
    ],
)
def test_failure_receipt(
    development_fixture, tmp_path, config, monkeypatch, exception, reason
):
    def fail(*a, **kw):
        raise exception

    monkeypatch.setattr(pilot, "train_step", fail)
    with pytest.raises(type(exception)):
        run_fixture(development_fixture, tmp_path, config, monkeypatch)
    receipt = json.loads((tmp_path / "run/receipt.json").read_text())
    assert receipt["status"] == "failed" and not receipt["checkpoint_usable"]
    assert receipt["optimizer_steps"] == 0 and receipt["phase"] == "training"
    assert receipt["failure_reason"] == reason


def test_nonfinite_gradient_prevents_optimizer_update():
    model = TinyModel()
    before = pilot.state_sha256(model)
    model.classifier.weight.register_hook(lambda grad: grad * float("nan"))
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="Non-finite gradient"):
        pilot.train_step(
            model,
            {"image": torch.ones(2, 3, 32, 32), "label": torch.tensor([0, 2])},
            optimizer,
            torch.amp.GradScaler("cuda", enabled=False),
            device=torch.device("cpu"),
            amp=False,
            weights=None,
            guard=lambda: None,
        )
    assert pilot.state_sha256(model) == before and not optimizer.state


def test_early_stopping_and_no_training_metric_selection(
    development_fixture, tmp_path, config, monkeypatch
):
    def fixed(model, loader, **kw):
        return {"partition": "validation", "loss": 1.0, "macro_f1": 0.1}

    monkeypatch.setattr(pilot, "evaluate", fixed)
    result = run_fixture(
        development_fixture,
        tmp_path,
        replace(config, epochs=3, max_steps=9, patience=1),
        monkeypatch,
    )
    assert result["stop_reason"] == "early_stopping" and result["optimizer_steps"] == 6
    assert result["best_checkpoint"]["epoch"] == 1
    monkeypatch.setattr(
        pilot,
        "evaluate",
        lambda *a, **k: {"partition": "train", "loss": 0, "macro_f1": 1},
    )
    with pytest.raises(ValueError, match="validation"):
        run_fixture(development_fixture, tmp_path, config, monkeypatch, "invalid")
    assert not (tmp_path / "invalid/best.pt").exists()


@pytest.mark.parametrize("override", [{"max_steps": 1}, {"max_seconds": 0.00001}])
def test_hard_budgets(development_fixture, tmp_path, config, monkeypatch, override):
    with pytest.raises(pilot.BudgetStop):
        run_fixture(
            development_fixture, tmp_path, replace(config, **override), monkeypatch
        )
    result = json.loads((tmp_path / "run/receipt.json").read_text())
    assert (
        not result["checkpoint_usable"]
        and result["failure_reason"] == "duration_or_step_budget"
    )
    assert result["optimizer_steps"] <= 1


def test_fixture_cannot_claim_real_data_by_label(
    development_fixture, tmp_path, monkeypatch
):
    manifest, assignments = development_fixture
    records = load_frozen_tbx11k(manifest, assignments)
    assert (
        validate_image_evidence(manifest, assignments, records[:2])
        == "unverified_or_fixture_inputs"
    )
    monkeypatch.setattr(smoke, "build_densenet121", lambda **kw: TinyModel())
    receipt = smoke.run_smoke(
        manifest=manifest,
        assignments=assignments,
        receipt_path=tmp_path / "smoke.json",
        pretrained=False,
        image_size=32,
        image_evidence="real_tbx11k",
    )
    assert not receipt["real_image_batch_processed"]
    assert receipt["image_evidence"] == "unverified_or_fixture_inputs"


@pytest.mark.parametrize(
    "override",
    [
        {"epochs": 4},
        {"precision": "bf16"},
        {"train_per_class": (1, 1, 1)},
        {"max_seconds": float("inf")},
        {"patience": 0},
        {"stop_temperature_c": 90},
        {"max_cuda_reserved_mib": 6144},
        {"learning_rate": float("nan")},
    ],
)
def test_invalid_config(override):
    with pytest.raises(ValueError):
        replace(pilot.PilotConfig(), **override).validate()


def test_no_overwrite_or_input_output_collision(
    development_fixture, tmp_path, config, monkeypatch
):
    result = run_fixture(development_fixture, tmp_path, config, monkeypatch)
    with pytest.raises(FileExistsError):
        run_fixture(development_fixture, tmp_path, config, monkeypatch)
    assert result == json.loads((tmp_path / "run/receipt.json").read_text())
    with pytest.raises(ValueError, match="input metadata"):
        pilot.run_pilot(
            manifest=development_fixture[0],
            assignments=development_fixture[1],
            output_dir=tmp_path,
        )


def test_memory_temperature_guards_and_final_thermal_sample(monkeypatch):
    import time

    monitor = smoke.GPUMonitor(max_temperature=80)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 5121 * 2**20)
    with pytest.raises(pilot.ResourceStop, match="memory budget"):
        pilot.check_resources(
            monitor, torch.device("cuda"), pilot.PilotConfig(), time.perf_counter()
        )
    monitor.overheated.set()
    with pytest.raises(RuntimeError, match="temperature"):
        pilot.check_resources(
            monitor, torch.device("cuda"), pilot.PilotConfig(), time.perf_counter()
        )
    receipt = {"status": "passed", "checkpoint_usable": True}
    monkeypatch.setattr(monitor, "stop", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 2)
    pilot.finalize_monitor(receipt, monitor, torch.device("cuda"))
    assert receipt["status"] == "failed" and not receipt["checkpoint_usable"]
    assert receipt["error_type"] == "ThermalStop"


def test_unavailable_cuda_no_fallback(
    development_fixture, tmp_path, config, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        pilot.run_pilot(
            manifest=development_fixture[0],
            assignments=development_fixture[1],
            output_dir=tmp_path / "failed",
            config=config,
        )
    receipt = json.loads((tmp_path / "failed/receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["optimizer_steps"] == 0


def test_invalid_config_has_failure_receipt(
    development_fixture, tmp_path, config, monkeypatch
):
    with pytest.raises(ValueError):
        run_fixture(
            development_fixture,
            tmp_path,
            replace(config, learning_rate=float("nan")),
            monkeypatch,
        )
    receipt = json.loads((tmp_path / "run/receipt.json").read_text())
    assert receipt["config"]["learning_rate"] == "nan" and receipt["status"] == "failed"


def test_weighted_loss_changes_with_training_weights():
    logits, labels = (
        torch.tensor([[3.0, 1.0, 0.0], [2.0, 1.0, 0.0]]),
        torch.tensor([0, 2]),
    )
    plain = nn.functional.cross_entropy(logits, labels)
    weighted = nn.functional.cross_entropy(
        logits, labels, weight=torch.tensor([0.7, 0.75, 3.4])
    )
    assert weighted > plain
