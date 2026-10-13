from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.data.frozen_split import load_frozen_tbx11k
from src.data.transforms import seed_everything
from src.training import full_data, pilot, preflight
from src.training.epoch import EpochTrainer
from tests import test_m2

development_fixture = test_m2.development_fixture

ROOT = Path(__file__).resolve().parents[1]


class StochasticTinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Dropout(0.2))
        self.classifier = nn.Linear(4, 3)

    def forward(self, x):
        features = self.features(x).mean((2, 3))
        if self.training:
            features = (
                features + random.random() * 0.01 + float(np.random.rand()) * 0.01
            )
        return self.classifier(features)


def make_trainer(fixture, device="cpu", binding=None):
    seed_everything(42)
    config = replace(full_data.FullDataConfig(), image_size=32)
    partitions = full_data.full_partitions(
        load_frozen_tbx11k(*fixture), require_approved_counts=False
    )
    loaders, generators = full_data.build_loaders(partitions, config)
    return EpochTrainer(
        StochasticTinyModel(),
        loaders,
        generators,
        device=torch.device(device),
        config=config,
        binding=binding
        or {"config": {"seed": 42}, "inputs": "fixture", "source": "fixture"},
    )


def epoch(trainer):
    receipt = {"steps": []}
    training = trainer.train_epoch(guard=lambda: None, receipt=receipt)
    metrics, seconds = trainer.validate(guard=lambda: None, receipt=receipt)
    trainer.finish_epoch(training, metrics, seconds)
    return receipt


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_interrupted_resume_matches_uninterrupted(
    development_fixture, tmp_path, device
):
    if device == "cuda" and (
        os.environ.get("M3A_TEST_CUDA_FIXTURES") != "1" or not torch.cuda.is_available()
    ):
        pytest.skip("Opt-in actual CUDA tiny fixture; no real dataset")
    torch.set_num_threads(2)
    uninterrupted = make_trainer(development_fixture, device)
    epoch(uninterrupted)
    expected = epoch(uninterrupted)
    expected_model = pilot.state_sha256(uninterrupted.model)
    expected_optimizer = full_data.digest_state(uninterrupted.optimizer.state_dict())
    expected_rng = full_data.digest_state(
        full_data.rng_snapshot(uninterrupted.generators)
    )
    interrupted = make_trainer(development_fixture, device)
    epoch(interrupted)
    checkpoint = tmp_path / "boundary.pt"
    interrupted.save(checkpoint)
    resumed = make_trainer(development_fixture, device)
    resumed.restore(checkpoint)
    actual = epoch(resumed)
    assert [s["sample_ids"] for s in expected["steps"]] == [
        s["sample_ids"] for s in actual["steps"]
    ]
    assert [s["loss"] for s in expected["steps"]] == [
        s["loss"] for s in actual["steps"]
    ]
    assert pilot.state_sha256(resumed.model) == expected_model
    assert full_data.digest_state(resumed.optimizer.state_dict()) == expected_optimizer
    assert (
        full_data.digest_state(full_data.rng_snapshot(resumed.generators))
        == expected_rng
    )
    assert resumed.best == uninterrupted.best or {
        k: v for k, v in resumed.best.items() if k != "batch_seconds_including_fetch"
    } == {
        k: v
        for k, v in uninterrupted.best.items()
        if k != "batch_seconds_including_fetch"
    }
    assert resumed.completed_epochs == 2 and resumed.optimizer_steps == 6


def test_complete_frozen_metadata_counts():
    records = load_frozen_tbx11k(
        ROOT / "data/manifests/tbx11k.csv",
        ROOT / "data/manifests/tbx11k_split.csv",
        summary=ROOT / "data/audit/tbx11k_split_summary.json",
        expected_split_sha256="4638fb21c180a3e05231d9ff682319bf4ddf36bf2b24643fa43e20776ddabf4e",
    )
    partitions = full_data.full_partitions(records)
    assert {k: len(v) for k, v in partitions.items()} == {
        "train": 5780,
        "validation": 1239,
    }
    assert {
        k: pilot.class_counts(v, k) for k, v in partitions.items()
    } == full_data.EXPECTED_COUNTS
    with pytest.raises(ValueError, match="counts"):
        full_data.full_partitions(records[1:])


def test_fixture_runner_heldout_isolation_and_reload(
    development_fixture, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        full_data, "build_densenet121", lambda **kw: StochasticTinyModel()
    )
    # The fixture's internal-test files do not exist. Both hashing and decoding
    # must still succeed on all development samples, including odd validation size.
    result = full_data.run_preflight(
        manifest=development_fixture[0],
        assignments=development_fixture[1],
        output_dir=tmp_path / "run",
        fixture=True,
        pretrained=False,
        device="cpu",
    )
    assert result["status"] == "passed" and result["completed_epochs"] == 1
    assert result["optimizer_steps"] == 3
    assert result["training_samples_this_epoch"] == 6
    assert result["validation_samples_this_epoch"] == 3
    assert result["best_validation"]["sample_count"] == 3
    assert result["checkpoint"]["reload_all_state_exact"]
    assert result["checkpoint"]["extra_validation_passes"] == 0
    assert "gpu_uuid" not in json.dumps(result)
    assert result["image_evidence"] == "unverified_or_fixture_inputs"
    with pytest.raises(FileExistsError):
        full_data.run_preflight(
            manifest=development_fixture[0],
            assignments=development_fixture[1],
            output_dir=tmp_path / "run",
        )
    reloaded = full_data.run_preflight(
        manifest=development_fixture[0],
        assignments=development_fixture[1],
        output_dir=tmp_path / "reload",
        fixture=True,
        pretrained=False,
        device="cpu",
        resume=tmp_path / "run/latest.pt",
    )
    assert reloaded["status"] == "passed" and reloaded["steps"] == []
    assert reloaded["optimizer_steps"] == 3


@pytest.mark.parametrize("key", ["config", "inputs", "source"])
def test_changed_binding_rejected(development_fixture, tmp_path, key):
    trainer = make_trainer(development_fixture)
    epoch(trainer)
    trainer.save(tmp_path / "checkpoint.pt")
    modified = {**trainer.binding, key: "changed"}
    with pytest.raises(ValueError, match="Changed configuration"):
        make_trainer(development_fixture, binding=modified).restore(
            tmp_path / "checkpoint.pt"
        )


@pytest.mark.parametrize("change", ["version", "counters", "model", "nonfinite"])
def test_incompatible_checkpoint_rejected(development_fixture, tmp_path, change):
    trainer = make_trainer(development_fixture)
    epoch(trainer)
    path = tmp_path / "checkpoint.pt"
    trainer.save(path)
    state = torch.load(path, weights_only=True)
    if change == "version":
        state["format_version"] = -1
    elif change == "counters":
        state["optimizer_steps"] += 1
    elif change == "model":
        state["model_state"]["classifier.bias"] += 1
    else:
        state["model_state"]["classifier.bias"][0] = float("nan")
    torch.save(state, path)
    with pytest.raises(ValueError):
        trainer.restore(path)


def test_partial_or_train_selection_cannot_checkpoint(development_fixture, tmp_path):
    trainer = make_trainer(development_fixture)
    with pytest.raises(ValueError, match="completed"):
        trainer.save(tmp_path / "bad.pt")
    receipt = {"steps": []}
    training = trainer.train_epoch(guard=lambda: None, receipt=receipt)
    metrics, seconds = trainer.validate(guard=lambda: None, receipt=receipt)
    with pytest.raises(ValueError, match="Partial validation"):
        trainer.finish_epoch(training, {**metrics, "sample_count": 1}, seconds)
    with pytest.raises(ValueError, match="validation metrics"):
        trainer.finish_epoch(training, {**metrics, "partition": "train"}, seconds)
    with pytest.raises(ValueError, match="completed"):
        trainer.save(tmp_path / "bad.pt")


@pytest.mark.parametrize(
    "override",
    [
        {"epochs": 2},
        {"precision": "fp16"},
        {"batch_size": 4},
        {"max_seconds": 3601},
        {"max_seconds": float("nan")},
        {"stop_temperature_c": 81},
        {"max_cuda_reserved_mib": 5121},
        {"learning_rate": 0.01},
    ],
)
def test_scientific_and_resource_contract(override):
    with pytest.raises(ValueError):
        replace(full_data.FullDataConfig(), **override).validate()


def test_hard_supervisor_kills_worker(tmp_path):
    result = preflight.supervise(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        tmp_path / "timeout",
        max_seconds=0.2,
    )
    assert result["status"] == "failed" and result["supervisor"]["forced_timeout"]
    assert result["failure_reason"] == "hard_duration_budget"
    assert not result["checkpoint_usable"]
    assert result["supervisor"]["duration_seconds_including_imports"] < 5


def test_failure_receipt_and_operation_budget(
    development_fixture, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        full_data, "build_densenet121", lambda **kw: StochasticTinyModel()
    )
    with pytest.raises(pilot.BudgetStop):
        full_data.run_preflight(
            manifest=development_fixture[0],
            assignments=development_fixture[1],
            output_dir=tmp_path / "failure",
            fixture=True,
            pretrained=False,
            device="cpu",
            config=replace(full_data.FullDataConfig(), max_seconds=1e-9),
        )
    result = json.loads((tmp_path / "failure/receipt.json").read_text())
    assert (
        result["failure_reason"] == "hard_duration_budget"
        and result["optimizer_steps"] == 0
    )


@pytest.mark.parametrize("fault", ["oom", "numeric", "resource"])
def test_runner_stops_without_retry_on_failure(
    development_fixture, tmp_path, monkeypatch, fault
):
    from src.training import epoch as engine

    monkeypatch.setattr(
        full_data, "build_densenet121", lambda **kw: StochasticTinyModel()
    )
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        if fault == "oom":
            raise torch.cuda.OutOfMemoryError("fixture")
        if fault == "resource":
            raise pilot.ResourceStop("fixture")
        raise ValueError("Non-finite gradient")

    monkeypatch.setattr(engine, "train_step", fail)
    with pytest.raises((ValueError, RuntimeError)):
        full_data.run_preflight(
            manifest=development_fixture[0],
            assignments=development_fixture[1],
            output_dir=tmp_path / "failed",
            fixture=True,
            pretrained=False,
            device="cpu",
        )
    receipt = json.loads((tmp_path / "failed/receipt.json").read_text())
    assert len(calls) == 1
    assert receipt["status"] == "failed" and receipt["optimizer_steps"] == 0
    assert (
        not receipt["checkpoint_usable"]
        and not (tmp_path / "failed/latest.pt").exists()
    )


def test_heldout_evaluation_rejected(development_fixture):
    trainer = make_trainer(development_fixture)
    trainer.loaders["validation"].dataset.records[0] = replace(
        trainer.loaders["validation"].dataset.records[0], split="internal_test"
    )
    with pytest.raises(ValueError, match="validation records"):
        trainer.validate(guard=lambda: None, receipt={})


def test_worker_crash_has_failure_receipt(tmp_path):
    result = preflight.supervise(
        [sys.executable, "-c", "raise SystemExit(7)"],
        tmp_path / "crash",
        max_seconds=10,
    )
    assert result["status"] == "failed" and result["failure_reason"] == "worker_failed"
    assert result["supervisor"]["worker_exit_code"] == 7


def test_worker_start_failure_has_receipt(tmp_path):
    result = preflight.supervise(
        [str(tmp_path / "missing-executable")],
        tmp_path / "startup-failed",
        max_seconds=10,
    )
    assert (
        result["status"] == "failed"
        and result["failure_reason"] == "supervisor_failure"
    )


@pytest.mark.parametrize(
    "fault", ["no_samples", "no_temperature", "no_thermal", "command_failed", "hot"]
)
def test_required_thermal_telemetry(fault):
    monitor = full_data.GPUMonitor(80)
    monitor.samples = [{"temperature_c": 60, "thermal_slowdown": "Not Active"}]
    if fault == "no_samples":
        monitor.samples = []
    elif fault == "no_temperature":
        monitor.samples[0]["temperature_c"] = None
    elif fault == "no_thermal":
        monitor.samples[0]["thermal_slowdown"] = "[N/A]"
    elif fault == "command_failed":
        monitor.unavailable_reason = "TimeoutExpired"
    else:
        monitor.overheated.set()
    with pytest.raises(RuntimeError):
        full_data.check_telemetry(monitor)


def test_memory_guard_and_telemetry_sanitization(monkeypatch):
    monitor = full_data.GPUMonitor(80)
    monitor.samples = [{"gpu_uuid": "private", "temperature_c": 60}]
    assert full_data.sanitized_samples(monitor) == [{"temperature_c": 60}]
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 5121 * 2**20)
    with pytest.raises(pilot.ResourceStop):
        pilot.check_resources(
            monitor,
            torch.device("cuda"),
            full_data.FullDataConfig(),
            time.perf_counter(),
        )
    monitor.overheated.set()
    with pytest.raises(RuntimeError, match="temperature"):
        pilot.check_resources(
            monitor,
            torch.device("cuda"),
            full_data.FullDataConfig(),
            time.perf_counter(),
        )
