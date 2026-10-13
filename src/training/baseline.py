"""M3B: one explicit session, one additional epoch; no automatic continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml

from src.data.frozen_split import load_frozen_tbx11k, metadata_sha256
from src.data.manifest import file_sha256
from src.data.provenance import (
    APPROVED_MANIFEST,
    APPROVED_SPLIT,
    validate_image_evidence,
)
from src.data.transforms import seed_everything
from src.models import build_densenet121
from src.training.cpu_temperature import CPUTemperatureMonitor
from src.training.epoch import EpochTrainer, finite_state
from src.training.full_data import (
    ROOT,
    build_loaders,
    check_telemetry,
    continuation_digest,
    digest_state,
    full_partitions,
    monitor_receipt,
)
from src.training.pilot import (
    BudgetStop,
    ResourceStop,
    atomic_json,
    check_resources,
    class_counts,
    is_better,
    state_sha256,
)
from src.training.smoke import GPUMonitor


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable approved scientific settings and deterministic execution settings."""

    seed: int = 42
    max_epochs: int = 10
    patience: int = 3
    batch_size: int = 2
    image_size: int = 224
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    precision: str = "fp32"
    optimizer: str = "adamw"
    architecture: str = "imagenet_densenet121_3class"
    loss: str = "unweighted_cross_entropy"
    scheduler: str = "none"
    augmentation: str = "train_affine_5_degrees_translate_0.02_no_flip"
    normalization: str = "fixed_imagenet_rgb"
    selection: str = "validation_macro_f1_lower_loss_earlier_epoch"
    num_threads: int = 2
    num_workers: int = 0

    def validate(self):
        expected = asdict(ExperimentConfig())
        actual = asdict(self)
        if actual != expected or any(
            type(actual[k]) is not type(v) for k, v in expected.items()
        ):
            raise ValueError(
                "M3B scientific and deterministic settings must match the approved protocol"
            )


@dataclass(frozen=True)
class SessionConfig:
    """Session-only safety parameters never enter scientific compatibility binding."""

    max_seconds: float = 3600
    stop_temperature_c: int = 80
    max_cuda_reserved_mib: int = 5120

    def validate(self):
        if not math.isfinite(self.max_seconds) or not 0 < self.max_seconds <= 3600:
            raise ValueError(
                "Session time limit must be positive and at most 3600 seconds"
            )
        if (
            type(self.stop_temperature_c) is not int
            or not 50 <= self.stop_temperature_c <= 80
        ):
            raise ValueError("GPU temperature stop must not exceed 80 C")
        if (
            type(self.max_cuda_reserved_mib) is not int
            or not 256 <= self.max_cuda_reserved_mib <= 5120
        ):
            raise ValueError("CUDA reserved memory must not exceed 5120 MiB")


class ExperimentFinished(RuntimeError):
    pass


def stopping_state(history, config):
    best, stale = None, 0
    for index, entry in enumerate(history, 1):
        metrics = entry["validation"]
        if entry["epoch"] != index or metrics["epoch"] != index:
            raise ValueError("Non-contiguous completed epoch history")
        improved = is_better(metrics, best)
        if entry["checkpoint_selected"] != improved:
            raise ValueError("Checkpoint selection history is inconsistent")
        if improved:
            best, stale = metrics, 0
        else:
            stale += 1
        if index < len(history) and (
            stale >= config.patience or index >= config.max_epochs
        ):
            raise ValueError("History continued beyond experiment termination")
    reason = (
        "early_stopping"
        if stale >= config.patience
        else "max_epochs"
        if len(history) >= config.max_epochs
        else None
    )
    return stale, reason, best


class BaselineTrainer(EpochTrainer):
    """M3A engine plus persistent M3B controller and checkpoint integrity."""

    def __init__(self, *args, config, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        self.config = config
        self.optimizer_settings = [
            {k: v for k, v in group.items() if k != "params"}
            for group in self.optimizer.param_groups
        ]
        self.experiment_id = str(uuid.uuid4())
        self.stale_epochs = 0
        self.termination_reason = None

    def train_epoch(self, **kwargs):
        if self.termination_reason:
            raise ExperimentFinished(
                "Experiment is already terminated; further training is refused"
            )
        progress = kwargs.pop("progress", lambda: None)

        def checked_progress():
            finite_state(self.optimizer.state)
            progress()

        return super().train_epoch(progress=checked_progress, **kwargs)

    def finish_epoch(self, *args):
        if self.termination_reason:
            raise ExperimentFinished("Experiment is already terminated")
        super().finish_epoch(*args)
        self.stale_epochs, self.termination_reason, _ = stopping_state(
            self.history, self.config
        )

    def controller_state(self):
        return {
            "version": 1,
            "experiment_id": self.experiment_id,
            "completed_sessions": self.completed_epochs,
            "stale_epochs": self.stale_epochs,
            "termination_reason": self.termination_reason,
            "experiment_config": asdict(self.config),
        }

    def save(self, path):
        self.check_optimizer_settings(self.optimizer.state_dict())
        path = Path(path)
        staged = path.with_name(f".{path.stem}-engine.pt")
        if staged.exists():
            raise FileExistsError("Checkpoint staging file already exists")
        try:
            # Preserve M3A's engine serialization instead of duplicating it.
            super().save(staged)
            state = torch.load(staged, map_location="cpu", weights_only=True)
            state["m3b_controller"] = self.controller_state()
            state["m3b_payload_sha256"] = digest_state(state)
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(state, temporary)
            temporary.replace(path)
        finally:
            staged.unlink(missing_ok=True)

    def restore(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        controller = state.get("m3b_controller")
        if not isinstance(controller, dict) or controller.get("version") != 1:
            raise ValueError(
                "M3B requires an M3B continuation checkpoint; M3A weights are not a baseline"
            )
        integrity = state.pop("m3b_payload_sha256", None)
        if integrity != digest_state(state):
            raise ValueError("Checkpoint continuation payload hash mismatch")
        if controller["experiment_config"] != asdict(self.config):
            raise ValueError("Changed immutable experiment configuration")
        if state.get("binding") != self.binding:
            raise ValueError(
                "Changed scientific settings, frozen inputs, source or environment"
            )
        self.check_optimizer_settings(state["optimizer_state"])
        stale, reason, best = stopping_state(state["history"], self.config)
        if (
            controller["stale_epochs"] != stale
            or controller["termination_reason"] != reason
            or controller["completed_sessions"] != state["completed_epochs"]
            or state["best_validation"] != best
            or not 1 <= state["completed_epochs"] <= self.config.max_epochs
        ):
            raise ValueError("Inconsistent persistent stopping or best-selection state")
        try:
            uuid.UUID(controller["experiment_id"])
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError("Invalid experiment identifier") from error
        model_hash = super().restore(path)
        self.experiment_id = controller["experiment_id"]
        self.stale_epochs, self.termination_reason = stale, reason
        return model_hash

    def check_optimizer_settings(self, state):
        actual = [
            {k: v for k, v in group.items() if k != "params"}
            for group in state["param_groups"]
        ]
        if actual != self.optimizer_settings:
            raise ValueError(
                "Optimizer settings changed from the immutable approved experiment"
            )

    def save_best(self, path):
        if not self.at_boundary or self.best_model is None:
            raise ValueError("A complete validation-selected best model is required")
        state = {
            "purpose": "M3B_validation_selected_model_not_external_threshold",
            "experiment_id": self.experiment_id,
            "binding": self.binding,
            "selected_epoch": self.best["epoch"],
            "validation": self.best,
            "model_state": self.best_model,
            "model_state_sha256": digest_state(self.best_model),
        }
        finite_state(state)
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)


def controller_digest(trainer):
    return digest_state(
        {
            "engine": continuation_digest(trainer),
            "controller": trainer.controller_state(),
        }
    )


def run_session(
    *,
    manifest,
    assignments,
    output_dir,
    experiment=None,
    session=None,
    summary=None,
    expected_split_sha256=None,
    image_root=None,
    resume=None,
    device="cuda",
    fixture=False,
    model_factory=None,
    config_file_sha256=None,
):
    """One epoch only. Fixture mode cannot open approved real dataset images."""
    experiment, session = experiment or ExperimentConfig(), session or SessionConfig()
    output = Path(output_dir).resolve()
    inputs = [Path(p).resolve() for p in (manifest, assignments, summary, resume) if p]
    if any(p.is_relative_to(output) for p in inputs):
        raise ValueError(
            "Output cannot contain input metadata or the previous checkpoint"
        )
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    old_threads = torch.get_num_threads()
    monitor = cpu_monitor = selected_device = None
    receipt = {
        "status": "failed",
        "purpose": "M3B_fixture_verification" if fixture else "M3B_baseline_session",
        "run_id": str(uuid.uuid4()),
        "started_utc": datetime.now(UTC).isoformat(),
        "experiment_config": asdict(experiment),
        "session_config": asdict(session),
        "session_number": None if resume else 1,
        "phase": "configuration",
        "optimizer_steps": None if resume else 0,
        "completed_epoch": None if resume else 0,
        "training_samples_this_epoch": 0,
        "validation_samples_this_epoch": 0,
        "session_completed": False,
        "experiment_finished": False,
        "checkpoint_usable": False,
        "steps": [],
        "history": [],
        "initialization": ("fresh_fixture" if fixture else "fresh_imagenet")
        if not resume
        else "resume_m3b",
        "previous_checkpoint_sha256": None,
        "image_evidence": "not_validated",
        "fixture": fixture,
        "config_file_sha256": config_file_sha256,
    }
    try:
        experiment.validate()
        session.validate()
        selected_device = torch.device(device)
        if selected_device.type not in {"cpu", "cuda"}:
            raise ValueError("Supported devices are CPU and CUDA")
        if not fixture and (
            selected_device.type != "cuda" or model_factory is not None
        ):
            raise ValueError("Real M3B requires CUDA and the approved ImageNet model")
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no CPU fallback")
        if not fixture and (summary is None or expected_split_sha256 is None):
            raise ValueError("Real M3B requires the frozen summary and split hash")
        torch.set_num_threads(experiment.num_threads)
        seed_everything(experiment.seed)
        environment = {
            "python": platform.python_version(),
            "system": platform.system(),
            "cuda_runtime": torch.version.cuda,
            "device": str(selected_device),
            "packages": {d.metadata["Name"]: d.version for d in distributions()},
            "num_threads": experiment.num_threads,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        }
        if selected_device.type == "cuda":
            props = torch.cuda.get_device_properties(selected_device)
            environment.update(gpu_name=props.name, gpu_memory_bytes=props.total_memory)
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                timeout=3,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            environment.update(
                driver_versions=driver.stdout.strip().splitlines(),
                gpu_capability=list(torch.cuda.get_device_capability(selected_device)),
            )
            torch.cuda.reset_peak_memory_stats(selected_device)
            monitor = GPUMonitor(session.stop_temperature_c)
            monitor.start()
        cpu_monitor = CPUTemperatureMonitor(enabled=not fixture)
        cpu_monitor.start()
        receipt["environment"] = environment

        def guard():
            check_resources(monitor, selected_device, session, started)
            if monitor:
                check_telemetry(monitor)

        guard()
        receipt["phase"] = "input_validation"
        records = load_frozen_tbx11k(
            manifest,
            assignments,
            summary=summary,
            expected_split_sha256=expected_split_sha256,
            image_root=image_root,
        )
        # Refuse approved real inputs in fixture mode before reading image bytes.
        if fixture and (
            metadata_sha256(manifest) == APPROVED_MANIFEST
            or metadata_sha256(assignments) == APPROVED_SPLIT
        ):
            raise ValueError(
                "Fixture verification cannot use approved real TBX11K images"
            )
        partitions = full_partitions(records, require_approved_counts=not fixture)
        receipt["image_evidence"] = validate_image_evidence(
            manifest, assignments, partitions["train"] + partitions["validation"]
        )
        if not fixture and receipt["image_evidence"] != "verified_real_tbx11k":
            raise ValueError("Real M3B requires approved frozen dataset provenance")
        guard()
        receipt["inputs"] = {
            "manifest_sha256": file_sha256(manifest),
            "split_sha256": file_sha256(assignments),
            "manifest_lf_sha256": metadata_sha256(manifest),
            "split_lf_sha256": metadata_sha256(assignments),
            "summary_sha256": file_sha256(summary) if summary else None,
        }
        receipt["partitions"] = {
            name: {
                "sample_count": len(subset),
                "class_counts": class_counts(subset, name),
                "membership_sha256": hashlib.sha256(
                    "\n".join(r.sample_id for r in subset).encode()
                ).hexdigest(),
            }
            for name, subset in partitions.items()
        }
        source = {
            p.relative_to(ROOT).as_posix(): file_sha256(p)
            for p in sorted((ROOT / "src").rglob("*.py"))
        }
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        receipt["implementation"] = {
            "git_commit": commit.stdout.strip(),
            "source_sha256": source,
        }
        receipt["phase"] = "model_initialization"
        model = (
            model_factory()
            if fixture and model_factory
            else build_densenet121(pretrained=not fixture, allow_download=False)
        )
        initialization = {
            "architecture": type(model).__name__
            if fixture
            else experiment.architecture,
            "weights": "random_fixture" if fixture else "IMAGENET1K_V1",
            "initial_state_sha256": state_sha256(model),
            "weights_sha256": None
            if fixture
            else file_sha256(
                Path(torch.hub.get_dir()) / "checkpoints/densenet121-a639ec97.pth"
            ),
        }
        receipt["model_initialization"] = initialization
        binding = {
            "purpose": "M3B",
            "experiment": asdict(experiment),
            "inputs": receipt["inputs"],
            "partitions": receipt["partitions"],
            "source_sha256": source,
            "environment": environment,
            "model_initialization": initialization,
        }
        loaders, generators = build_loaders(partitions, experiment)
        trainer = BaselineTrainer(
            model,
            loaders,
            generators,
            device=selected_device,
            config=experiment,
            binding=binding,
        )
        if resume:
            receipt["previous_checkpoint_sha256"] = file_sha256(resume)
            previous_receipt = json.loads(
                (Path(resume).parent / "receipt.json").read_text(encoding="utf-8")
            )
            if not (
                previous_receipt.get("status") == "passed"
                and previous_receipt.get("session_completed")
                and previous_receipt.get("checkpoint_usable")
                and previous_receipt.get("checkpoints", {})
                .get("latest", {})
                .get("sha256")
                == receipt["previous_checkpoint_sha256"]
            ):
                raise ValueError(
                    "Resume requires the successful session receipt and matching latest checkpoint"
                )
            trainer.restore(resume)
            if (
                previous_receipt.get("experiment_id") != trainer.experiment_id
                or previous_receipt.get("completed_epoch") != trainer.completed_epochs
            ):
                raise ValueError("Previous receipt does not match experiment or epoch")
            receipt.update(
                session_number=trainer.completed_epochs + 1,
                completed_epoch=trainer.completed_epochs,
                optimizer_steps=trainer.optimizer_steps,
                experiment_id=trainer.experiment_id,
                history=trainer.history,
                experiment_finished=bool(trainer.termination_reason),
            )
            if trainer.termination_reason:
                receipt["termination_reason"] = trainer.termination_reason
                raise ExperimentFinished(
                    "Experiment is finished; no additional epoch is permitted"
                )
        else:
            receipt["experiment_id"] = trainer.experiment_id
        previous_steps = trainer.optimizer_steps
        last_write = 0.0

        def progress(force=False):
            nonlocal last_write
            now = time.perf_counter()
            if force or now - last_write >= 5:
                monitor_receipt(receipt, monitor, selected_device)
                receipt["cpu_monitor"] = cpu_monitor.receipt()
                receipt["duration_seconds"] = now - started
                atomic_json(output / "receipt.json", receipt)
                print(
                    json.dumps(
                        {
                            k: receipt[k]
                            for k in (
                                "phase",
                                "session_number",
                                "optimizer_steps",
                                "training_samples_this_epoch",
                                "validation_samples_this_epoch",
                            )
                        }
                    ),
                    flush=True,
                )
                last_write = now

        progress(force=True)
        epoch_started = time.perf_counter()
        training = trainer.train_epoch(guard=guard, receipt=receipt, progress=progress)
        metrics, seconds = trainer.validate(
            guard=guard, receipt=receipt, progress=progress
        )
        trainer.finish_epoch(training, metrics, seconds)
        receipt.update(
            phase="checkpoint",
            completed_epoch=trainer.completed_epochs,
            history=trainer.history,
            optimizer_steps_this_session=trainer.optimizer_steps - previous_steps,
            stale_epochs=trainer.stale_epochs,
            best_epoch=trainer.best["epoch"],
            best_validation=trainer.best,
            termination_reason=trainer.termination_reason,
            experiment_finished=bool(trainer.termination_reason),
            throughput_images_per_second={
                "train": training["training_sample_count"]
                / training["training_seconds"],
                "validation": metrics["sample_count"] / seconds,
            },
        )
        guard()
        trainer.save_best(output / "best.pt")
        trainer.save(output / "latest.pt")
        guard()
        receipt["epoch_seconds"] = time.perf_counter() - epoch_started
        receipt["phase"] = "checkpoint_verification"
        before_model, before_controller = (
            state_sha256(model),
            controller_digest(trainer),
        )
        restored = trainer.restore(output / "latest.pt")
        best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        if restored != before_model or controller_digest(trainer) != before_controller:
            raise ValueError("Reloaded continuation state mismatch")
        if (
            best["binding"] != binding
            or best["validation"] != trainer.best
            or digest_state(best["model_state"]) != digest_state(trainer.best_model)
        ):
            raise ValueError("Best model export mismatch")
        receipt["checkpoints"] = {
            name: {
                "filename": name + ".pt",
                "sha256": file_sha256(output / (name + ".pt")),
                "bytes": (output / (name + ".pt")).stat().st_size,
            }
            for name in ("latest", "best")
        }
        receipt["checkpoint_restoration"] = {
            "model_state_sha256": restored,
            "continuation_sha256": before_controller,
            "all_state_exact": True,
            "best_model_exact": True,
            "additional_validation_passes": 0,
        }
        guard()
        receipt.update(
            status="passed",
            session_completed=True,
            checkpoint_usable=True,
            session_outcome="experiment_finished"
            if trainer.termination_reason
            else "session_complete_experiment_active",
        )
    except (Exception, KeyboardInterrupt) as error:
        receipt.update(
            status="failed",
            session_completed=False,
            session_outcome="failed_or_interrupted",
            checkpoint_usable=False,
            error_type=type(error).__name__,
            failure_reason="experiment_already_finished"
            if isinstance(error, ExperimentFinished)
            else "hard_duration_budget"
            if isinstance(error, BudgetStop)
            else "cuda_out_of_memory"
            if isinstance(error, torch.cuda.OutOfMemoryError)
            else "resource_guard"
            if isinstance(error, ResourceStop)
            or (monitor and monitor.overheated.is_set())
            else "invalid_or_nonfinite_input_state_or_configuration",
        )
        raise
    finally:
        if monitor:
            monitor.stop()
            monitor_receipt(receipt, monitor, selected_device)
            try:
                check_telemetry(monitor)
            except RuntimeError:
                receipt.update(
                    status="failed",
                    checkpoint_usable=False,
                    session_completed=False,
                    session_outcome="failed_or_interrupted",
                    error_type="ResourceStop",
                    failure_reason="resource_guard",
                )
        if cpu_monitor:
            cpu_monitor.stop()
            receipt["cpu_monitor"] = cpu_monitor.receipt()
        receipt["duration_seconds"] = time.perf_counter() - started
        torch.set_num_threads(old_threads)
        atomic_json(output / "receipt.json", receipt)
    if receipt["status"] != "passed":
        raise ResourceStop("Final resource sample invalidated the session")
    return receipt


def read_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return raw, ExperimentConfig(**raw["experiment"])


def main():
    # Reuse the M3A supervisor's existing internal worker marker.
    if os.environ.get("M3A_SUPERVISED") != "1":
        raise RuntimeError("Launch via python -m src.training.baseline_session")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--max-seconds", type=float, default=3600)
    args = parser.parse_args()
    try:
        raw, experiment = read_config(args.config)
        base = args.config.resolve().parent.parent
        result = run_session(
            manifest=base / raw["manifest"],
            assignments=base / raw["assignments"],
            summary=base / raw["summary"],
            expected_split_sha256=raw["split_sha256"],
            output_dir=args.output_dir,
            experiment=experiment,
            session=SessionConfig(max_seconds=args.max_seconds),
            resume=args.resume,
            image_root=args.image_root,
            config_file_sha256=file_sha256(args.config),
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(
            f"M3B session failed ({type(error).__name__}); inspect the local receipt",
            flush=True,
        )
        return 1
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "session_outcome",
                    "completed_epoch",
                    "experiment_finished",
                    "stale_epochs",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
