"""M3A: complete frozen development partitions, at most one epoch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.datasets import ManifestDataset
from src.data.frozen_split import load_frozen_tbx11k, metadata_sha256
from src.data.manifest import file_sha256
from src.data.provenance import validate_image_evidence
from src.data.transforms import build_transform, seed_everything
from src.models import build_densenet121
from src.training.epoch import EpochTrainer, rng_snapshot
from src.training.pilot import (
    BudgetStop,
    ResourceStop,
    atomic_json,
    check_resources,
    class_counts,
    config_metadata,
    state_sha256,
)
from src.training.smoke import GPUMonitor

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COUNTS = {"train": [2660, 2560, 560], "validation": [570, 549, 120]}


@dataclass(frozen=True)
class FullDataConfig:
    seed: int = 42
    epochs: int = 1
    batch_size: int = 2
    image_size: int = 224
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    precision: str = "fp32"
    max_seconds: float = 3600
    stop_temperature_c: int = 80
    max_cuda_reserved_mib: int = 5120
    num_threads: int = 2

    def validate(self):
        # The reusable engine has no research schedule. This entry point only
        # authorizes the single M3A epoch; future schedules need scientific review.
        approved = {
            "seed": 42,
            "epochs": 1,
            "batch_size": 2,
            "image_size": 224,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "precision": "fp32",
        }
        if any(getattr(self, k) != v for k, v in approved.items()):
            raise ValueError(
                "M3A requires the approved one-epoch FP32 starting settings"
            )
        if not math.isfinite(self.max_seconds) or not 0 < self.max_seconds <= 3600:
            raise ValueError(
                "M3A hard duration must be positive and at most 3600 seconds"
            )
        if not 50 <= self.stop_temperature_c <= 80:
            raise ValueError("Temperature stop must not exceed 80 C")
        if not 256 <= self.max_cuda_reserved_mib <= 5120:
            raise ValueError("CUDA reserved memory must not exceed 5 GiB")
        if type(self.num_threads) is not int or not 1 <= self.num_threads <= 8:
            raise ValueError("Invalid thread count")


def full_partitions(records, *, require_approved_counts=True):
    selected = {
        name: sorted((r for r in records if r.split == name), key=lambda r: r.sample_id)
        for name in EXPECTED_COUNTS
    }
    for name, subset in selected.items():
        ids = [r.sample_id for r in subset]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Empty partition or duplicate sample IDs")
        if (
            require_approved_counts
            and class_counts(subset, name) != EXPECTED_COUNTS[name]
        ):
            raise ValueError("Complete frozen partition counts required")
    if {r.sample_id for r in selected["train"]} & {
        r.sample_id for r in selected["validation"]
    }:
        raise ValueError("Training and validation IDs overlap")
    return selected


def build_loaders(partitions, config):
    generators = {
        name: torch.Generator().manual_seed(config.seed + offset)
        for offset, name in enumerate(partitions)
    }
    loaders = {
        name: DataLoader(
            ManifestDataset(
                subset,
                build_transform(training=name == "train", image_size=config.image_size),
                label_space="tbx11k_internal_3class",
            ),
            batch_size=config.batch_size,
            shuffle=name == "train",
            drop_last=False,
            num_workers=0,
            generator=generators[name],
        )
        for name, subset in partitions.items()
    }
    return loaders, generators


def digest_state(value):
    """Stable comparison of all restored optimizer/RNG/scaler/selection state."""
    digest = hashlib.sha256()

    def visit(item):
        digest.update(type(item).__name__.encode())
        if isinstance(item, torch.Tensor):
            digest.update(str((item.dtype, list(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(str(len(item)).encode())
            for entry in item:
                visit(entry)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def continuation_digest(trainer):
    return digest_state(
        {
            "optimizer": trainer.optimizer.state_dict(),
            "scaler": trainer.scaler.state_dict(),
            "rng": rng_snapshot(trainer.generators),
            "completed_epochs": trainer.completed_epochs,
            "optimizer_steps": trainer.optimizer_steps,
            "best": trainer.best,
            "best_model": trainer.best_model,
            "history": trainer.history,
        }
    )


def sanitized_samples(monitor):
    return [{k: v for k, v in s.items() if k != "gpu_uuid"} for s in monitor.samples]


def check_telemetry(monitor):
    monitor.check()
    if (
        monitor.unavailable_reason
        or not monitor.samples
        or monitor.samples[-1]["temperature_c"] is None
        or monitor.samples[-1]["thermal_slowdown"].lower()
        not in {"active", "not active"}
    ):
        raise ResourceStop("Required temperature/thermal telemetry unavailable")


def monitor_receipt(receipt, monitor, device):
    if monitor:
        receipt["gpu_monitor"] = {
            "samples": sanitized_samples(monitor),
            "unavailable_reason": monitor.unavailable_reason,
            "overheated": monitor.overheated.is_set(),
            "stop_temperature_c": monitor.max_temperature,
        }
        receipt["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        receipt["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)


def run_preflight(
    *,
    manifest,
    assignments,
    output_dir,
    config=None,
    summary=None,
    expected_split_sha256=None,
    image_root=None,
    device="cuda",
    pretrained=True,
    resume=None,
    config_sha256=None,
    fixture=False,
):
    """Fixture mode is Python-only and cannot claim verified real execution."""
    config = config or FullDataConfig()
    output = Path(output_dir).resolve()
    inputs = [Path(p).resolve() for p in (manifest, assignments, summary, resume) if p]
    if any(p.is_relative_to(output) for p in inputs):
        raise ValueError("Output directory cannot contain input metadata or checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    monitor = selected_device = None
    previous_threads = torch.get_num_threads()
    receipt = {
        "status": "failed",
        "purpose": "M3A_one_epoch_preflight_diagnostics_not_final_baseline",
        "started_utc": datetime.now(UTC).isoformat(),
        "config": config_metadata(config),
        "optimizer_steps": 0,
        "training_samples_this_epoch": 0,
        "validation_samples_this_epoch": 0,
        "completed_epochs": 0,
        "phase": "configuration",
        "steps": [],
        "history": [],
        "checkpoint_usable": False,
        "selection": {
            "partition": "validation",
            "metric": "macro_f1",
            "tie_breaking": "lower_validation_loss_then_earliest_epoch",
        },
        "loss": "unweighted_cross_entropy",
        "image_evidence": "not_validated",
        "fixture": fixture,
    }
    try:
        config.validate()
        selected_device = torch.device(device)
        if selected_device.type not in {"cpu", "cuda"}:
            raise ValueError("Supported devices: cpu, cuda")
        if not fixture and (selected_device.type != "cuda" or not pretrained):
            raise ValueError("Real M3A requires CUDA and fresh ImageNet initialization")
        if not fixture and (summary is None or expected_split_sha256 is None):
            raise ValueError("Real M3A requires frozen summary and split hash")
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        torch.set_num_threads(config.num_threads)
        seed_everything(config.seed)
        environment = {
            "python": platform.python_version(),
            "system": platform.system(),
            "cuda_runtime": torch.version.cuda,
            "packages": {d.metadata["Name"]: d.version for d in distributions()},
            "device": str(selected_device),
        }
        receipt["environment"] = environment
        if selected_device.type == "cuda":
            props = torch.cuda.get_device_properties(selected_device)
            environment.update(gpu_name=props.name, gpu_memory_bytes=props.total_memory)
            torch.cuda.reset_peak_memory_stats(selected_device)
            monitor = GPUMonitor(config.stop_temperature_c)
            monitor.start()

        def guard():
            check_resources(monitor, selected_device, config, start)
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
        partitions = full_partitions(records, require_approved_counts=not fixture)
        guard()
        receipt["image_evidence"] = validate_image_evidence(
            manifest, assignments, partitions["train"] + partitions["validation"]
        )
        if fixture and receipt["image_evidence"] == "verified_real_tbx11k":
            raise ValueError("Fixture mode cannot run the approved real dataset")
        if not fixture and receipt["image_evidence"] != "verified_real_tbx11k":
            raise ValueError(
                "Real M3A requires the approved frozen metadata provenance"
            )
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
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        source = {
            p.relative_to(ROOT).as_posix(): file_sha256(p)
            for p in sorted((ROOT / "src").rglob("*.py"))
        }
        receipt["implementation"] = {
            "git_commit": commit.stdout.strip(),
            "source_sha256": source,
        }
        receipt["config_file_sha256"] = config_sha256
        receipt["determinism"] = {
            "algorithms": True,
            "num_workers": 0,
            "train_loader_seed": config.seed,
            "validation_loader_seed": config.seed + 1,
            "num_threads": config.num_threads,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "augmentation": "train_only_affine_5_degrees_translate_0.02_no_flip",
        }
        loaders, generators = build_loaders(partitions, config)
        receipt["phase"] = "model_initialization"
        model = build_densenet121(pretrained=pretrained, allow_download=False)
        receipt["model"] = {
            "architecture": "densenet121_three_outputs",
            "initialization": "IMAGENET1K_V1" if pretrained else "random_fixture",
            "initial_state_sha256": state_sha256(model),
            "weights_sha256": file_sha256(
                Path(torch.hub.get_dir()) / "checkpoints/densenet121-a639ec97.pth"
            )
            if pretrained
            else None,
        }
        binding = {
            "config": receipt["config"],
            "config_file_sha256": config_sha256,
            "inputs": receipt["inputs"],
            "partitions": receipt["partitions"],
            "source_sha256": source,
            "environment": environment,
            "model": receipt["model"],
            "determinism": receipt["determinism"],
            "selection": receipt["selection"],
            "loss": receipt["loss"],
        }
        trainer = EpochTrainer(
            model,
            loaders,
            generators,
            device=selected_device,
            config=config,
            binding=binding,
        )
        if resume:
            trainer.restore(resume)
            receipt.update(
                optimizer_steps=trainer.optimizer_steps,
                completed_epochs=trainer.completed_epochs,
                resumed_checkpoint_sha256=file_sha256(resume),
            )
            # M3A's total epoch cap includes completed epochs; reload-only is allowed.
            if trainer.completed_epochs > config.epochs:
                raise ValueError("Checkpoint exceeds the M3A total epoch budget")
        last_write = 0.0

        def progress(force=False):
            nonlocal last_write
            now = time.perf_counter()
            if force or now - last_write >= 5:
                monitor_receipt(receipt, monitor, selected_device)
                receipt["duration_seconds"] = now - start
                atomic_json(output / "receipt.json", receipt)
                print(
                    json.dumps(
                        {
                            k: receipt[k]
                            for k in (
                                "phase",
                                "optimizer_steps",
                                "training_samples_this_epoch",
                                "validation_samples_this_epoch",
                                "duration_seconds",
                            )
                        }
                    ),
                    flush=True,
                )
                last_write = now

        progress(force=True)
        for _ in range(trainer.completed_epochs, config.epochs):
            epoch_start = time.perf_counter()
            training = trainer.train_epoch(
                guard=guard, receipt=receipt, progress=progress
            )
            metrics, seconds = trainer.validate(
                guard=guard, receipt=receipt, progress=progress
            )
            trainer.finish_epoch(training, metrics, seconds)
            receipt["phase"] = "checkpoint"
            guard()
            trainer.save(output / "latest.pt")
            receipt["history"] = trainer.history
            receipt["completed_epochs"] = trainer.completed_epochs
            receipt["epoch_seconds"] = time.perf_counter() - epoch_start
            receipt["throughput_images_per_second"] = {
                "train": training["training_sample_count"]
                / training["training_seconds"],
                "validation": metrics["sample_count"] / seconds,
            }
            progress(force=True)
        receipt["phase"] = "checkpoint_verification"
        guard()
        # Reload all continuation state on the existing model without another
        # real validation pass or an additional optimizer step.
        before_model, before_continuation = (
            state_sha256(model),
            continuation_digest(trainer),
        )
        if (
            resume
            and trainer.completed_epochs == config.epochs
            and not (output / "latest.pt").exists()
        ):
            trainer.save(output / "latest.pt")
        restored = trainer.restore(output / "latest.pt")
        if (
            restored != before_model
            or continuation_digest(trainer) != before_continuation
        ):
            raise ValueError("Reloaded model or continuation state mismatch")
        receipt["checkpoint"] = {
            "filename": "latest.pt",
            "sha256": file_sha256(output / "latest.pt"),
            "bytes": (output / "latest.pt").stat().st_size,
            "model_state_sha256": restored,
            "continuation_state_sha256": before_continuation,
            "reload_all_state_exact": True,
            "extra_validation_passes": 0,
            "preflight_only_not_final_research_checkpoint": True,
        }
        receipt["history"] = trainer.history
        receipt["best_validation"] = trainer.best
        guard()
        receipt.update(
            status="passed",
            checkpoint_usable=True,
            stop_reason="one_epoch_cap_completed",
        )
    except Exception as error:
        receipt.update(
            status="failed",
            checkpoint_usable=False,
            error_type=type(error).__name__,
            failure_reason=(
                "hard_duration_budget"
                if isinstance(error, BudgetStop)
                else "cuda_out_of_memory"
                if isinstance(error, torch.cuda.OutOfMemoryError)
                else "resource_guard"
                if isinstance(error, ResourceStop)
                or (monitor and monitor.overheated.is_set())
                else "invalid_or_nonfinite_input_state_or_configuration"
            ),
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
                    error_type="ResourceStop",
                    failure_reason="resource_guard",
                )
        receipt["duration_seconds"] = time.perf_counter() - start
        torch.set_num_threads(previous_threads)
        atomic_json(output / "receipt.json", receipt)
    if receipt["status"] != "passed":
        raise ResourceStop("Final resource sample invalidated execution")
    return receipt


def main():
    if os.environ.get("M3A_SUPERVISED") != "1":
        raise RuntimeError(
            "Launch via python -m src.training.preflight for a hard deadline"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/preflight.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    try:
        raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        base = args.config.resolve().parent.parent
        result = run_preflight(
            manifest=base / raw["manifest"],
            assignments=base / raw["assignments"],
            summary=base / raw["summary"],
            expected_split_sha256=raw["split_sha256"],
            config=FullDataConfig(**raw["preflight"]),
            output_dir=args.output_dir,
            image_root=args.image_root,
            resume=args.resume,
            config_sha256=file_sha256(args.config),
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(
            f"M3A failed ({type(error).__name__}); inspect the local receipt",
            flush=True,
        )
        return 1
    print(
        json.dumps(
            {k: result[k] for k in ("status", "optimizer_steps", "duration_seconds")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
