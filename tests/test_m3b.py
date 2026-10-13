from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from src.data.frozen_split import load_frozen_tbx11k
from src.data.manifest import file_sha256
from src.data.transforms import seed_everything
from src.training import (
    baseline,
    baseline_session,
    cpu_temperature,
    epoch,
    full_data,
    pilot,
)
from tests import test_m2, test_m3a

development_fixture = test_m2.development_fixture
ROOT = Path(__file__).resolve().parents[1]


def make_trainer(fixture, device="cpu", binding=None):
    seed_everything(42)
    config = baseline.ExperimentConfig()
    partitions = full_data.full_partitions(
        load_frozen_tbx11k(*fixture), require_approved_counts=False
    )
    loaders, generators = full_data.build_loaders(partitions, config)
    return baseline.BaselineTrainer(
        test_m3a.StochasticTinyModel(),
        loaders,
        generators,
        device=torch.device(device),
        config=config,
        binding=binding
        or {
            "purpose": "M3B",
            "scientific": asdict(config),
            "inputs": "fixture",
            "source": "fixture",
            "environment": device,
        },
    )


def fixture_epoch(trainer, *, metrics=None):
    receipt = {"steps": []}
    training = trainer.train_epoch(guard=lambda: None, receipt=receipt)
    observed, seconds = trainer.validate(guard=lambda: None, receipt=receipt)
    if metrics:
        observed.update(metrics)
    trainer.finish_epoch(training, observed, seconds)
    return receipt


def fixture_session(fixture, output, **kwargs):
    return baseline.run_session(
        manifest=fixture[0],
        assignments=fixture[1],
        output_dir=output,
        fixture=True,
        model_factory=test_m3a.StochasticTinyModel,
        device=kwargs.pop("device", "cpu"),
        **kwargs,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_multisession_exact_equivalence(development_fixture, tmp_path, device):
    if device == "cuda" and (
        os.environ.get("M3B_TEST_CUDA_FIXTURES") != "1" or not torch.cuda.is_available()
    ):
        pytest.skip("Optional bounded CUDA fixture; no real dataset")
    torch.set_num_threads(2)
    continuous = make_trainer(development_fixture, device)
    fixture_epoch(continuous)
    expected = fixture_epoch(continuous)
    expected_model = pilot.state_sha256(continuous.model)
    expected_optimizer = full_data.digest_state(continuous.optimizer.state_dict())
    expected_rng = full_data.digest_state(epoch.rng_snapshot(continuous.generators))
    first = fixture_session(
        development_fixture,
        tmp_path / "session1",
        device=device,
        config_file_sha256="configuration-copy-a",
    )
    second = fixture_session(
        development_fixture,
        tmp_path / "session2",
        device=device,
        resume=tmp_path / "session1/latest.pt",
        session=baseline.SessionConfig(max_seconds=90),
        config_file_sha256="configuration-copy-b",
    )
    assert first["session_number"] == 1 and second["session_number"] == 2
    assert (
        first["experiment_id"] == second["experiment_id"]
        and first["run_id"] != second["run_id"]
    )
    assert second["completed_epoch"] == 2 and second["optimizer_steps"] == 6
    assert second["optimizer_steps_this_session"] == 3
    assert first["session_outcome"] == "session_complete_experiment_active"
    assert first["session_completed"] and not first["experiment_finished"]
    assert (
        first["training_samples_this_epoch"]
        == second["training_samples_this_epoch"]
        == 6
    )
    assert (
        first["validation_samples_this_epoch"]
        == second["validation_samples_this_epoch"]
        == 3
    )
    assert [s["sample_ids"] for s in second["steps"]] == [
        s["sample_ids"] for s in expected["steps"]
    ]
    assert [s["loss"] for s in second["steps"]] == [
        s["loss"] for s in expected["steps"]
    ]
    state = torch.load(
        tmp_path / "session2/latest.pt", map_location="cpu", weights_only=True
    )
    assert state["model_sha256"] == expected_model
    assert full_data.digest_state(state["optimizer_state"]) == expected_optimizer
    assert full_data.digest_state(state["rng"]) == expected_rng
    assert second["checkpoint_restoration"]["all_state_exact"]
    assert second["checkpoint_restoration"]["additional_validation_passes"] == 0
    assert state["m3b_controller"]["stale_epochs"] == continuous.stale_epochs
    assert (
        state["m3b_controller"]["termination_reason"] == continuous.termination_reason
    )
    for key in ("macro_f1", "loss", "epoch", "confusion_matrix"):
        assert state["best_validation"][key] == continuous.best[key]


def test_early_stopping_and_best_across_real_fixture_sessions(
    development_fixture, tmp_path, monkeypatch
):
    original = baseline.BaselineTrainer.validate

    # Actual fixture training and validation still execute; only the selection
    # metrics are controlled to exercise exact ties and persistent stopping.
    def controlled(self, **kwargs):
        metrics, seconds = original(self, **kwargs)
        metrics.update(macro_f1=0.5, loss=1.0)
        return metrics, seconds

    monkeypatch.setattr(baseline.BaselineTrainer, "validate", controlled)
    previous = None
    first_best = None
    for index in range(1, 5):
        output = tmp_path / f"session{index}"
        result = fixture_session(development_fixture, output, resume=previous)
        state = torch.load(output / "latest.pt", weights_only=True)
        best_hash = full_data.digest_state(state["best_model_state"])
        first_best = first_best or best_hash
        assert best_hash == first_best and result["best_epoch"] == 1
        assert result["stale_epochs"] == index - 1
        assert result["experiment_finished"] == (index == 4)
        previous = output / "latest.pt"
    assert result["termination_reason"] == "early_stopping"
    assert (
        result["session_completed"]
        and result["session_outcome"] == "experiment_finished"
    )
    with pytest.raises(baseline.ExperimentFinished):
        fixture_session(development_fixture, tmp_path / "refused", resume=previous)
    receipt = json.loads((tmp_path / "refused/receipt.json").read_text())
    assert receipt["failure_reason"] == "experiment_already_finished"
    assert receipt["experiment_finished"] and not receipt["session_completed"]
    assert receipt["steps"] == [] and receipt["training_samples_this_epoch"] == 0


def test_lexicographic_improvement_and_earliest_tie(development_fixture, tmp_path):
    trainer = make_trainer(development_fixture)
    sequence = [
        (0.6, 1.0, True, 1, 0),
        (0.6, 0.9, True, 2, 0),
        (0.5, 0.1, False, 2, 1),
        (0.6, 0.9, False, 2, 2),
        (0.7, 1.2, True, 5, 0),
    ]
    previous_best = None
    for f1, loss, improved, best_epoch, stale in sequence:
        fixture_epoch(trainer, metrics={"macro_f1": f1, "loss": loss})
        assert trainer.history[-1]["checkpoint_selected"] == improved
        assert trainer.best["epoch"] == best_epoch and trainer.stale_epochs == stale
        if not improved:
            assert full_data.digest_state(trainer.best_model) == previous_best
        previous_best = full_data.digest_state(trainer.best_model)
        path = tmp_path / f"epoch{trainer.completed_epochs}.pt"
        trainer.save(path)
        trainer = make_trainer(development_fixture)
        trainer.restore(path)
    assert trainer.stale_epochs == 0 and trainer.termination_reason is None


def test_global_ten_epoch_cap_across_sessions(development_fixture, tmp_path):
    trainer = make_trainer(development_fixture)
    for index in range(1, 11):
        fixture_epoch(trainer, metrics={"macro_f1": index / 10, "loss": 1.0})
        path = tmp_path / f"session{index}.pt"
        trainer.save(path)
        trainer = make_trainer(development_fixture)
        trainer.restore(path)
        assert (
            trainer.completed_epochs == index and trainer.optimizer_steps == index * 3
        )
    assert trainer.termination_reason == "max_epochs"
    with pytest.raises(baseline.ExperimentFinished):
        trainer.train_epoch(guard=lambda: None, receipt={"steps": []})


@pytest.mark.parametrize("key", ["scientific", "inputs", "source", "environment"])
def test_resume_binding_rejection(development_fixture, tmp_path, key):
    trainer = make_trainer(development_fixture)
    fixture_epoch(trainer)
    trainer.save(tmp_path / "saved.pt")
    with pytest.raises(ValueError, match="Changed scientific"):
        make_trainer(
            development_fixture, binding={**trainer.binding, key: "changed"}
        ).restore(tmp_path / "saved.pt")


def test_changed_input_rejected_in_session(development_fixture, tmp_path):
    fixture_session(development_fixture, tmp_path / "first")
    manifest = development_fixture[0]
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="frozen inputs"):
        fixture_session(
            development_fixture,
            tmp_path / "changed",
            resume=tmp_path / "first/latest.pt",
        )


def test_m3a_checkpoint_cannot_seed_m3b(development_fixture, tmp_path):
    trainer = make_trainer(development_fixture)
    fixture_epoch(trainer)
    epoch.EpochTrainer.save(trainer, tmp_path / "m3a.pt")
    with pytest.raises(ValueError, match="M3A weights"):
        trainer.restore(tmp_path / "m3a.pt")


@pytest.mark.parametrize("change", ["optimizer", "rng", "stale", "best_model"])
def test_checkpoint_payload_integrity(development_fixture, tmp_path, change):
    trainer = make_trainer(development_fixture)
    fixture_epoch(trainer)
    path = tmp_path / "latest.pt"
    trainer.save(path)
    state = torch.load(path, weights_only=True)
    if change == "optimizer":
        state["optimizer_state"]["param_groups"][0]["lr"] = 0.01
    elif change == "rng":
        state["rng"]["torch_cpu"][0] ^= 1
    elif change == "stale":
        state["m3b_controller"]["stale_epochs"] += 1
    else:
        state["best_model_state"]["classifier.bias"] += 1
    torch.save(state, path)
    with pytest.raises(ValueError, match="payload hash"):
        trainer.restore(path)


def test_mutated_optimizer_configuration_rejected(development_fixture, tmp_path):
    trainer = make_trainer(development_fixture)
    fixture_epoch(trainer)
    trainer.optimizer.param_groups[0]["lr"] = 0.01
    with pytest.raises(ValueError, match="Optimizer settings changed"):
        trainer.save(tmp_path / "invalid.pt")


def test_failed_save_preserves_previous_checkpoint(
    development_fixture, tmp_path, monkeypatch
):
    trainer = make_trainer(development_fixture)
    fixture_epoch(trainer)
    path = tmp_path / "latest.pt"
    trainer.save(path)
    previous_hash = file_sha256(path)
    original = torch.save

    def interrupted(state, target, *args, **kwargs):
        if "m3b_controller" in state:
            Path(target).write_bytes(b"incomplete staging bytes")
            raise OSError("fixture interrupted checkpoint write")
        return original(state, target, *args, **kwargs)

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(OSError):
        trainer.save(path)
    assert file_sha256(path) == previous_hash
    trainer.restore(path)


@pytest.mark.parametrize("fault", ["numeric", "optimizer", "resource", "interrupted"])
def test_partial_session_failure_is_truthful_and_prior_checkpoint_untouched(
    development_fixture, tmp_path, monkeypatch, fault
):
    fixture_session(development_fixture, tmp_path / "prior")
    previous = tmp_path / "prior/latest.pt"
    before = file_sha256(previous)
    original = epoch.train_step
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            result = original(*args, **kwargs)
            if fault == "optimizer":
                next(iter(args[2].state.values()))["exp_avg"].fill_(float("nan"))
            return result
        if fault == "numeric":
            raise ValueError("Non-finite training loss")
        if fault == "resource":
            raise pilot.ResourceStop("fixture thermal guard")
        raise KeyboardInterrupt()

    monkeypatch.setattr(epoch, "train_step", fail)
    with pytest.raises((ValueError, RuntimeError, KeyboardInterrupt)):
        fixture_session(development_fixture, tmp_path / "failed", resume=previous)
    receipt = json.loads((tmp_path / "failed/receipt.json").read_text())
    assert receipt["status"] == "failed" and not receipt["session_completed"]
    assert (
        receipt["completed_epoch"] == 1 and receipt["training_samples_this_epoch"] == 2
    )
    assert not (tmp_path / "failed/latest.pt").exists()
    assert file_sha256(previous) == before
    assert len(calls) == (1 if fault == "optimizer" else 2)


def test_fixture_guard_refuses_real_inputs_before_any_image_read(tmp_path, monkeypatch):
    monkeypatch.setattr(
        baseline,
        "validate_image_evidence",
        lambda *args: pytest.fail("Real bytes must not be read"),
    )
    with pytest.raises(ValueError, match="cannot use approved real"):
        baseline.run_session(
            manifest=ROOT / "data/manifests/tbx11k.csv",
            assignments=ROOT / "data/manifests/tbx11k_split.csv",
            output_dir=tmp_path / "refused",
            fixture=True,
            device="cpu",
            model_factory=test_m3a.StochasticTinyModel,
        )


@pytest.mark.parametrize(
    "override",
    [
        {"max_epochs": 11},
        {"patience": 4},
        {"precision": "fp16"},
        {"learning_rate": 0.01},
        {"loss": "weighted"},
        {"seed": 1},
        {"num_threads": 4},
        {"scheduler": "cosine"},
    ],
)
def test_immutable_approved_protocol(override):
    with pytest.raises(ValueError, match="approved protocol"):
        replace(baseline.ExperimentConfig(), **override).validate()


@pytest.mark.parametrize(
    "override",
    [
        {"max_seconds": 3601},
        {"max_seconds": float("nan")},
        {"stop_temperature_c": 81},
        {"max_cuda_reserved_mib": 5121},
    ],
)
def test_session_safety_contract(override):
    with pytest.raises(ValueError):
        replace(baseline.SessionConfig(), **override).validate()


def test_supervisor_cli_uses_hard_deadline_without_training(tmp_path, monkeypatch):
    calls = []

    def stopped(command, output, **kwargs):
        calls.append((command, kwargs))
        output.mkdir()
        return {
            "status": "failed",
            "session_completed": True,
            "failure_reason": "hard_duration_budget",
        }

    monkeypatch.setattr(baseline_session, "supervise", stopped)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "baseline_session",
            "--config",
            str(ROOT / "configs/m3b.yaml"),
            "--output-dir",
            str(tmp_path / "session"),
            "--max-seconds",
            "15",
        ],
    )
    assert baseline_session.main() == 1
    assert (
        calls[0][0][2] == "src.training.baseline" and calls[0][1]["max_seconds"] == 15
    )
    receipt = json.loads((tmp_path / "session/receipt.json").read_text())
    assert (
        not receipt["session_completed"]
        and receipt["session_outcome"] == "failed_or_interrupted"
    )


def test_cli_invalid_config_has_sanitized_failure_receipt(tmp_path, monkeypatch):
    bad = tmp_path / "invalid.yaml"
    bad.write_text("invalid: [yaml\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "baseline_session",
            "--config",
            str(bad),
            "--output-dir",
            str(tmp_path / "failed"),
        ],
    )
    assert baseline_session.main() == 1
    receipt = json.loads((tmp_path / "failed/receipt.json").read_text())
    assert receipt["status"] == "failed" and not receipt["session_completed"]
    assert str(tmp_path) not in json.dumps(receipt)


def test_session_operation_budget_and_no_partial_checkpoint(
    development_fixture, tmp_path
):
    with pytest.raises(pilot.BudgetStop):
        fixture_session(
            development_fixture,
            tmp_path / "timeout",
            session=baseline.SessionConfig(max_seconds=1e-9),
        )
    receipt = json.loads((tmp_path / "timeout/receipt.json").read_text())
    assert receipt["failure_reason"] == "hard_duration_budget"
    assert receipt["completed_epoch"] == 0 and receipt["optimizer_steps"] == 0
    assert not (tmp_path / "timeout/latest.pt").exists()


def test_failed_receipt_cannot_authorize_resume(development_fixture, tmp_path):
    fixture_session(development_fixture, tmp_path / "prior")
    path = tmp_path / "prior/receipt.json"
    receipt = json.loads(path.read_text())
    receipt.update(status="failed", checkpoint_usable=False)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="successful session receipt"):
        fixture_session(
            development_fixture,
            tmp_path / "refused",
            resume=tmp_path / "prior/latest.pt",
        )


def test_cpu_monitor_unavailable_and_no_gpu_substitution(monkeypatch):
    monkeypatch.setattr(cpu_temperature.platform, "system", lambda: "Windows")
    monkeypatch.setattr(cpu_temperature, "read_cpu_temperatures", lambda provider: [])
    monitor = cpu_temperature.CPUTemperatureMonitor()
    monitor.start()
    result = monitor.receipt()
    assert not result["temperature_available"] and result["samples"] == []
    assert result["unavailable_reason"] == "no_readable_cpu_temperature_sensor"
    assert monitor.thread is None


def test_cpu_provider_association_filter_and_null_measurements(monkeypatch):
    calls = []

    def query(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                [
                    {"SensorType": "Temperature", "Value": 65.0, "Parent": "/amdcpu/0"},
                    {"SensorType": "Temperature", "Value": None, "Parent": "/amdcpu/0"},
                    {"SensorType": "Load", "Value": 90, "Parent": "/amdcpu/0"},
                ]
            ),
        )

    monkeypatch.setattr(cpu_temperature.subprocess, "run", query)
    assert cpu_temperature.read_cpu_temperatures(cpu_temperature.PROVIDERS[0]) == [65.0]
    assert "Where-Object HardwareType -eq 'Cpu'" in calls[0][0][-1]
    assert "$hardware.Identifier -contains $_.Parent" in calls[0][0][-1]
    assert calls[0][1]["timeout"] == 8


def test_cpu_provider_loss_is_explicit(monkeypatch):
    monkeypatch.setattr(cpu_temperature.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        cpu_temperature, "read_cpu_temperatures", lambda provider: [60.0, 64.0]
    )
    monitor = cpu_temperature.CPUTemperatureMonitor()
    monitor.sample()
    assert monitor.receipt()["temperature_available"]
    assert monitor.samples[-1]["temperature_c"] == 64.0
    monkeypatch.setattr(cpu_temperature, "read_cpu_temperatures", lambda provider: [])
    monitor.sample()
    assert not monitor.receipt()["temperature_available"]
    assert monitor.receipt()["unavailable_reason"] is not None
