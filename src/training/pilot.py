"""Bounded development pilot; never opens held-out image bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.datasets import ManifestDataset
from src.data.frozen_split import load_frozen_tbx11k, metadata_sha256
from src.data.labels import TBX11K_INTERNAL_CLASS_NAMES, normalize_tbx11k_internal_label
from src.data.manifest import file_sha256
from src.data.provenance import validate_image_evidence
from src.data.transforms import build_transform, seed_everything
from src.models import build_densenet121
from src.training.smoke import AMP_INITIAL_SCALE, GPUMonitor, _finite

CLASS_NAMES = list(TBX11K_INTERNAL_CLASS_NAMES.values())


@dataclass(frozen=True)
class PilotConfig:
    seed: int = 42
    epochs: int = 3
    batch_size: int = 2
    image_size: int = 224
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    precision: str = "fp32"
    train_per_class: tuple[int, int, int] = (24, 24, 6)
    validation_per_class: tuple[int, int, int] = (24, 24, 6)
    max_steps: int = 81
    max_seconds: float = 180
    patience: int | None = None
    class_weighted: bool = False
    stop_temperature_c: int = 80
    max_cuda_reserved_mib: int = 5120
    num_threads: int = 2

    def validate(self):
        if not 1 <= self.epochs <= 3:
            raise ValueError("Pilot epochs must be 1..3; full training requires M3")
        if not 2 <= self.batch_size <= 4 or not 32 <= self.image_size <= 224:
            raise ValueError("Pilot batch_size 2..4 and image_size 32..224 required")
        if self.precision not in {"fp32", "fp16"}:
            raise ValueError("Precision must be fp32 or fp16")
        for counts in (self.train_per_class, self.validation_per_class):
            if len(counts) != 3 or any(
                type(n) is not int or not 1 <= n <= 128 for n in counts
            ):
                raise ValueError("Specify 1..128 samples for each of three classes")
        if sum(self.train_per_class) % self.batch_size:
            raise ValueError(
                "Training subset must have complete batches (no dropped images)"
            )
        if not 1 <= self.max_steps <= 1000 or not 0 < self.max_seconds <= 600:
            raise ValueError("Pilot requires at most 1000 steps and 600 seconds")
        if not math.isfinite(self.max_seconds):
            raise ValueError("Invalid duration budget")
        if self.patience is not None and not 1 <= self.patience <= 100:
            raise ValueError("Patience must be positive or disabled")
        if not 1 <= self.num_threads <= 8 or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid thread count or seed")
        if not 50 <= self.stop_temperature_c <= 85:
            raise ValueError("Temperature guard must be 50..85 C")
        if not 256 <= self.max_cuda_reserved_mib <= 5120:
            raise ValueError("CUDA reserved memory budget must be 256..5120 MiB")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Invalid learning rate")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Invalid weight decay")


def class_counts(records, partition):
    counts = [0, 0, 0]
    for record in records:
        if record.split == partition:
            counts[normalize_tbx11k_internal_label(record.original_label)] += 1
    return counts


def training_weights(records):
    """N/(3*n_c), from the complete frozen training partition exclusively."""
    counts = class_counts(records, "train")
    if not all(counts):
        raise ValueError("All training classes required for inverse-frequency weights")
    return [sum(counts) / (3 * n) for n in counts]


def select_subset(records, partition, counts, seed):
    if partition not in {"train", "validation"}:
        raise ValueError("Only training and validation subsets are permitted")
    rng = random.Random(seed)
    selected = []
    for label, count in enumerate(counts):
        candidates = sorted(
            (
                r
                for r in records
                if r.split == partition
                and normalize_tbx11k_internal_label(r.original_label) == label
            ),
            key=lambda r: r.sample_id,
        )
        if len(candidates) < count:
            raise ValueError("Partition class too small for requested subset")
        selected.extend(rng.sample(candidates, count))
    # Stable membership, independently shuffled training order by epoch loader.
    return sorted(selected, key=lambda r: r.sample_id)


def classification_metrics(matrix):
    """Rows=true, columns=predicted; all three classes, zero division -> 0."""
    matrix = torch.as_tensor(matrix, dtype=torch.int64)
    if matrix.shape != (3, 3) or (matrix < 0).any() or matrix.sum() == 0:
        raise ValueError("Expected nonempty 3x3 confusion matrix")
    m = matrix.double()
    tp, actual, predicted = m.diag(), m.sum(1), m.sum(0)
    recall = torch.where(actual > 0, tp / actual, 0)
    denominator = actual + predicted
    f1 = torch.where(denominator > 0, 2 * tp / denominator, 0)
    return {
        "accuracy": (tp.sum() / m.sum()).item(),
        "macro_f1": f1.mean().item(),
        "per_class_recall": recall.tolist(),
        "confusion_matrix": matrix.tolist(),
        "class_names": CLASS_NAMES,
        "sample_count": int(matrix.sum()),
        "class_counts": actual.long().tolist(),
        "zero_division": 0,
    }


def is_better(candidate, best):
    if candidate["partition"] != "validation":
        raise ValueError("Checkpoint selection requires validation metrics")
    for key in ("macro_f1", "loss"):
        if not math.isfinite(candidate[key]):
            raise ValueError("Non-finite selection metric")
    if best is None:
        return True
    return (candidate["macro_f1"], -candidate["loss"]) > (
        best["macro_f1"],
        -best["loss"],
    )


class BudgetStop(RuntimeError):
    pass


class ResourceStop(RuntimeError):
    pass


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def config_metadata(config):
    """JSON-native settings; invalid non-finite config still permits a receipt."""
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, float) and not math.isfinite(value):
            data[key] = str(value)
    return json.loads(json.dumps(data, allow_nan=False))


def check_resources(monitor, device, config, started):
    if monitor:
        monitor.check()
        if torch.cuda.memory_reserved(device) > config.max_cuda_reserved_mib * 2**20:
            raise ResourceStop("CUDA reserved memory budget exceeded")
    if time.perf_counter() - started >= config.max_seconds:
        raise BudgetStop("Duration budget reached at an operation boundary")


def finalize_monitor(receipt, monitor, device):
    monitor.stop()
    receipt["gpu_monitor"] = {
        "samples": monitor.samples,
        "unavailable_reason": monitor.unavailable_reason,
        "overheated": monitor.overheated.is_set(),
        "stop_temperature_c": monitor.max_temperature,
    }
    if monitor.overheated.is_set():
        receipt.update(
            status="failed",
            checkpoint_usable=False,
            error_type="ThermalStop",
            failure_reason="resource_guard",
        )
    receipt["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    receipt["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)


def state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_step(model, batch, optimizer, scaler, *, device, amp, weights, guard):
    model.train()
    guard()
    images, labels = batch["image"].to(device), batch["label"].to(device)
    _finite(images, "training inputs")
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        logits = model(images)
        if logits.shape != (len(labels), 3):
            raise ValueError("Expected three-class logits")
        _finite(logits, "training logits")
        loss = nn.functional.cross_entropy(logits, labels, weight=weights)
    _finite(loss, "training loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients:
        raise ValueError("No gradients produced")
    for gradient in gradients:
        _finite(gradient, "gradient")
    guard()
    scaler.step(optimizer)
    scaler.update()
    for parameter in model.parameters():
        _finite(parameter, "updated parameter")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return loss.detach().float().item()


def evaluate(model, loader, *, device, guard):
    """Unweighted validation CE is comparable between imbalance configurations."""
    if any(r.split != "validation" for r in loader.dataset.records):
        raise ValueError("Evaluation requires exclusively validation records")
    model.eval()
    matrix = torch.zeros((3, 3), dtype=torch.int64)
    total_loss, count, runtimes = 0.0, 0, []
    with torch.inference_mode():
        for batch in loader:
            start = time.perf_counter()
            guard()
            images, labels = batch["image"].to(device), batch["label"].to(device)
            _finite(images, "validation inputs")
            # Always FP32 evaluation, deterministic and independent of train AMP.
            logits = model(images)
            if logits.shape != (len(labels), 3):
                raise ValueError("Expected three-class validation logits")
            _finite(logits, "validation logits")
            loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
            _finite(loss, "validation loss")
            total_loss += loss.item()
            count += len(labels)
            matrix += torch.bincount(
                (labels * 3 + logits.argmax(1)).cpu(), minlength=9
            ).reshape(3, 3)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            guard()
            runtimes.append(time.perf_counter() - start)
    result = classification_metrics(matrix)
    result.update(
        loss=total_loss / count, partition="validation", batch_seconds=runtimes
    )
    return result


def run_pilot(
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
    allow_download=False,
    config_sha256=None,
):
    config = config if config is not None else PilotConfig()
    output = Path(output_dir).resolve()
    inputs = [
        Path(p).resolve() for p in (manifest, assignments, summary) if p is not None
    ]
    if any(p.is_relative_to(output) for p in inputs):
        raise ValueError("Output directory cannot contain input metadata")
    # Never overwrite an existing run/checkpoint, even if a previous run failed.
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    receipt = {
        "status": "failed",
        "purpose": "bounded_engineering_pilot_not_performance_estimate",
        "started_utc": datetime.now(UTC).isoformat(),
        "config": config_metadata(config),
        "config_file_sha256": config_sha256,
        "optimizer_steps": 0,
        "phase": "configuration",
        "history": [],
        "steps": [],
        "image_evidence": "not_validated",
        "checkpoint_usable": False,
        "selection": {
            "metric": "macro_f1",
            "partition": "validation",
            "tie_breaking": "lower_validation_loss_then_earliest_epoch",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "cuda_runtime": torch.version.cuda,
            "packages": {d.metadata["Name"]: d.version for d in distributions()},
        },
    }
    monitor, selected_device = None, None
    old_threads = torch.get_num_threads()
    try:
        config.validate()
        source_root = Path(__file__).resolve().parents[2]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            cwd=source_root,
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        receipt["implementation"] = {
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "source_sha256": {
                path.relative_to(source_root).as_posix(): file_sha256(path)
                for path in sorted((source_root / "src").rglob("*.py"))
            },
        }
        selected_device = torch.device(device)
        receipt["device"] = str(selected_device)
        if selected_device.type not in {"cpu", "cuda"}:
            raise ValueError("Supported devices: cpu, cuda")
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        amp = config.precision == "fp16"
        if amp and selected_device.type != "cuda":
            raise ValueError("FP16 requires CUDA")
        torch.set_num_threads(config.num_threads)
        seed_everything(config.seed)
        receipt["determinism"] = {
            "algorithms": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "num_workers": 0,
            "shuffle_seed": config.seed,
            "amp_initial_scale": AMP_INITIAL_SCALE if amp else None,
        }
        receipt["phase"] = "input_validation"
        records = load_frozen_tbx11k(
            manifest,
            assignments,
            summary=summary,
            expected_split_sha256=expected_split_sha256,
            image_root=image_root,
        )
        train = select_subset(records, "train", config.train_per_class, config.seed)
        validation = select_subset(
            records, "validation", config.validation_per_class, config.seed + 1
        )
        receipt["image_evidence"] = validate_image_evidence(
            manifest, assignments, train + validation
        )
        receipt["inputs"] = {
            "manifest_sha256": file_sha256(manifest),
            "split_sha256": file_sha256(assignments),
            "manifest_lf_sha256": metadata_sha256(manifest),
            "split_lf_sha256": metadata_sha256(assignments),
            "summary_sha256": file_sha256(summary) if summary else None,
        }
        receipt["subsets"] = {
            partition: {
                "sample_ids": [r.sample_id for r in subset],
                "class_counts": class_counts(subset, partition),
                "sample_count": len(subset),
            }
            for partition, subset in [("train", train), ("validation", validation)]
        }
        receipt["training_class_counts"] = class_counts(records, "train")
        candidate_weights = training_weights(records)
        receipt["imbalance"] = {
            "enabled": config.class_weighted,
            "candidate_weights": candidate_weights,
            "frequency_partition": "complete_frozen_train",
            "oversampling": False,
        }
        receipt["preprocessing"] = {
            "image_size": config.image_size,
            "normalization": "fixed_imagenet",
            "train_affine_degrees": 5,
            "train_translate": [0.02, 0.02],
            "validation_stochastic_augmentation": False,
        }
        loaders = [
            DataLoader(
                ManifestDataset(
                    subset,
                    build_transform(training=training, image_size=config.image_size),
                    label_space="tbx11k_internal_3class",
                ),
                batch_size=config.batch_size,
                shuffle=training,
                num_workers=0,
                generator=torch.Generator().manual_seed(config.seed),
            )
            for subset, training in [(train, True), (validation, False)]
        ]
        receipt["phase"] = "model_initialization"
        model = build_densenet121(pretrained=pretrained, allow_download=allow_download)
        initial_classifier = model.classifier.weight.detach().cpu().clone()
        receipt["model"] = {
            "architecture": "densenet121",
            "initialization": "IMAGENET1K_V1" if pretrained else "random_fixture",
            "initial_state_sha256": state_sha256(model),
            "class_names": CLASS_NAMES,
        }
        if pretrained:
            receipt["model"]["weights_sha256"] = file_sha256(
                Path(torch.hub.get_dir()) / "checkpoints/densenet121-a639ec97.pth"
            )
        if selected_device.type == "cuda":
            properties = torch.cuda.get_device_properties(selected_device)
            receipt["environment"].update(
                gpu_name=properties.name,
                gpu_total_memory_bytes=properties.total_memory,
                gpu_capability=list(torch.cuda.get_device_capability(selected_device)),
            )
            torch.cuda.reset_peak_memory_stats(selected_device)
            monitor = GPUMonitor(config.stop_temperature_c)
            monitor.start()

        def guard():
            check_resources(monitor, selected_device, config, start)

        guard()
        model.to(selected_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=AMP_INITIAL_SCALE)
        weights = (
            torch.tensor(candidate_weights, device=selected_device)
            if config.class_weighted
            else None
        )
        receipt["optimizer"] = "AdamW"
        best, stale = None, 0
        for epoch in range(1, config.epochs + 1):
            epoch_start = time.perf_counter()
            receipt["phase"], receipt["epoch"] = "training", epoch
            losses, observed = [], []
            for batch in loaders[0]:
                if receipt["optimizer_steps"] >= config.max_steps:
                    raise BudgetStop("Maximum optimizer steps reached")
                step_start = time.perf_counter()
                loss = train_step(
                    model,
                    batch,
                    optimizer,
                    scaler,
                    device=selected_device,
                    amp=amp,
                    weights=weights,
                    guard=guard,
                )
                receipt["optimizer_steps"] += 1
                observed.extend(batch["sample_id"])
                denominator = (
                    sum(candidate_weights[label] for label in batch["label"].tolist())
                    if config.class_weighted
                    else len(batch["label"])
                )
                losses.append((loss, denominator))
                receipt["steps"].append(
                    {
                        "epoch": epoch,
                        "step": receipt["optimizer_steps"],
                        "loss": loss,
                        "seconds": time.perf_counter() - step_start,
                        "sample_ids": batch["sample_id"],
                    }
                )
                guard()
            receipt["phase"] = "validation"
            val_start = time.perf_counter()
            metrics = evaluate(model, loaders[1], device=selected_device, guard=guard)
            metrics["epoch"] = epoch
            entry = {
                "epoch": epoch,
                "training_loss": sum(v * n for v, n in losses)
                / sum(n for _, n in losses),
                "training_sample_count": len(observed),
                "validation": metrics,
                "training_seconds": val_start - epoch_start,
                "validation_seconds": time.perf_counter() - val_start,
            }
            improved = is_better(metrics, best)
            if improved:
                receipt["phase"] = "checkpoint"
                metadata = {
                    "epoch": epoch,
                    "optimizer_steps": receipt["optimizer_steps"],
                    "validation": metrics,
                    "inputs": receipt["inputs"],
                    "config": receipt["config"],
                    "model": receipt["model"],
                    "selection": receipt["selection"],
                    "image_evidence": receipt["image_evidence"],
                    "state_sha256": state_sha256(model),
                }
                temporary = output / "best.pt.tmp"
                torch.save(
                    {
                        "metadata": metadata,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                    temporary,
                )
                temporary.replace(output / "best.pt")
                best, stale = metrics, 0
                receipt["best_checkpoint"] = {
                    "filename": "best.pt",
                    "sha256": file_sha256(output / "best.pt"),
                    **metadata,
                }
            else:
                stale += 1
            entry.update(
                checkpoint_selected=improved,
                epoch_seconds=time.perf_counter() - epoch_start,
            )
            if selected_device.type == "cuda":
                entry["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
                    selected_device
                )
                entry["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(
                    selected_device
                )
            receipt["history"].append(entry)
            atomic_json(output / "receipt.json", receipt)
            guard()
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": entry["training_loss"],
                        "validation_loss": metrics["loss"],
                        "macro_f1": metrics["macro_f1"],
                        "checkpoint_selected": improved,
                    }
                ),
                flush=True,
            )
            if config.patience is not None and stale >= config.patience:
                receipt["stop_reason"] = "early_stopping"
                break
        receipt["final_state_sha256"] = state_sha256(model)
        receipt["classifier_updated"] = not torch.equal(
            initial_classifier, model.classifier.weight.detach().cpu()
        )
        if not receipt["classifier_updated"]:
            raise ValueError("Optimizer did not update classifier weights")
        # Exercise the saved best checkpoint, independently of the final epoch.
        receipt["phase"] = "checkpoint_verification"
        guard()
        checkpoint = torch.load(
            output / "best.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        if state_sha256(model) != receipt["best_checkpoint"]["state_sha256"]:
            raise ValueError("Reloaded checkpoint state mismatch")
        del checkpoint
        verification = evaluate(model, loaders[1], device=selected_device, guard=guard)
        for key in ("loss", "macro_f1"):
            if verification[key] != best[key]:
                raise ValueError("Reloaded checkpoint validation mismatch")
        receipt["checkpoint_validation"] = verification
        guard()
        receipt["status"], receipt["checkpoint_usable"] = "passed", True
        receipt.setdefault("stop_reason", "configured_epochs_completed")
    except Exception as error:
        # Avoid disclosing paths carried by PIL/OS exceptions in public receipts.
        receipt["error_type"] = type(error).__name__
        receipt["failure_reason"] = (
            "duration_or_step_budget"
            if isinstance(error, BudgetStop)
            else "cuda_out_of_memory"
            if isinstance(error, torch.cuda.OutOfMemoryError)
            else "resource_guard"
            if isinstance(error, ResourceStop)
            or (monitor and monitor.overheated.is_set())
            else "invalid_or_nonfinite_input_state_or_configuration"
        )
        if isinstance(error, ValueError) and str(error).startswith("Non-finite "):
            receipt["failure_detail"] = str(error)[:80]
        raise
    finally:
        if monitor:
            finalize_monitor(receipt, monitor, selected_device)
        receipt["duration_seconds"] = time.perf_counter() - start
        torch.set_num_threads(old_threads)
        atomic_json(output / "receipt.json", receipt)
    if receipt["status"] != "passed":
        raise ResourceStop("Final thermal sample invalidated execution")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-weight-download", action="store_true")
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = args.config.resolve().parent.parent
    try:
        result = run_pilot(
            manifest=base / raw["manifest"],
            assignments=base / raw["assignments"],
            summary=base / raw["summary"],
            expected_split_sha256=raw["split_sha256"],
            config=PilotConfig(**raw["pilot"]),
            output_dir=args.output_dir,
            image_root=args.image_root,
            device=args.device,
            allow_download=args.allow_weight_download,
            config_sha256=file_sha256(args.config),
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(
            f"Pilot failed ({type(error).__name__}); inspect the local failure receipt"
        )
        return 1
    print(
        json.dumps(
            {k: result[k] for k in ["status", "optimizer_steps", "duration_seconds"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
