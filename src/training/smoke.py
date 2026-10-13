"""A bounded one-step engineering check, never a baseline training runner."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

# Required by CUDA's deterministic matrix operations; set before CUDA is used.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torchvision
import yaml
from torch import nn

from src.data.datasets import ManifestDataset
from src.data.frozen_split import load_frozen_tbx11k
from src.data.manifest import file_sha256
from src.data.transforms import build_transform, seed_everything
from src.models import build_densenet121

AMP_INITIAL_SCALE = 1024.0


class GPUMonitor:
    """Best-effort nvidia-smi snapshots. Records all GPUs with their UUIDs."""

    def __init__(self, max_temperature: int = 85):
        self.max_temperature = max_temperature
        self.samples: list[dict] = []
        self.unavailable_reason: str | None = None
        self.overheated = threading.Event()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def sample(self) -> None:
        try:
            output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,name,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.hw_thermal_slowdown",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=3,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
            for line in output.strip().splitlines():
                uuid, name, temperature, utilization, memory, thermal = [
                    item.strip() for item in line.split(",")
                ]

                def numeric(value):
                    try:
                        return int(value)
                    except ValueError:
                        return None

                temp = numeric(temperature)
                self.samples.append(
                    {
                        "gpu_uuid": uuid,
                        "name": name,
                        "temperature_c": temp,
                        "utilization_percent": numeric(utilization),
                        "device_memory_used_mib": numeric(memory),
                        "thermal_slowdown": thermal,
                        "time_utc": datetime.now(UTC).isoformat(),
                    }
                )
                if (
                    temp is not None and temp >= self.max_temperature
                ) or thermal.lower() == "active":
                    self.overheated.set()
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            self.unavailable_reason = type(error).__name__

    def start(self) -> None:
        self.sample()

        def poll():
            while not self.stop_event.wait(1):
                self.sample()

        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()

    def check(self) -> None:
        if self.overheated.is_set():
            raise RuntimeError(
                "GPU temperature limit or hardware thermal slowdown observed; stopping smoke"
            )

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=4)
        self.sample()


def _finite(tensor: torch.Tensor, description: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"Non-finite {description}")


def optimizer_step(
    model,
    images,
    labels,
    *,
    device,
    learning_rate,
    weight_decay,
    amp=False,
    monitor=None,
):
    """Full-model AdamW step with explicit finite and parameter-update checks."""
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # A one-step check cannot rely on several skipped updates to calibrate the
    # default 65536 scale. Start conservatively; still reject every non-finite grad.
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=AMP_INITIAL_SCALE)
    before = model.classifier.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    if monitor:
        monitor.check()
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        logits = model(images)
        if logits.shape != (len(labels), 3):
            raise ValueError("Expected a three-class output per image")
        _finite(logits, "logits")
        loss = nn.functional.cross_entropy(logits, labels)
    _finite(loss, "loss")
    if monitor:
        monitor.check()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    if not gradients:
        raise ValueError("No gradients produced")
    for gradient in gradients:
        _finite(gradient, "gradient")
    if monitor:
        monitor.check()
    scaler.step(optimizer)
    scaler.update()
    for parameter in model.parameters():
        _finite(parameter, "updated parameter")
    changed = not torch.equal(before, model.classifier.weight.detach())
    if not changed:
        raise ValueError("Optimizer did not update classifier weights")
    return {
        "loss": loss.detach().float().item(),
        "output_shape": list(logits.shape),
        "optimizer_steps": 1,
        "classifier_updated": changed,
    }


def run_smoke(
    *,
    manifest,
    assignments,
    receipt_path,
    summary=None,
    image_root=None,
    expected_split_sha256=None,
    device="cpu",
    batch_size=2,
    image_size=224,
    seed=42,
    learning_rate=0.0001,
    weight_decay=0.0001,
    pretrained=True,
    allow_download=False,
    amp=False,
    image_evidence="real_tbx11k",
):
    output_path = Path(receipt_path).resolve()
    input_paths = {Path(manifest).resolve(), Path(assignments).resolve()}
    if summary is not None:
        input_paths.add(Path(summary).resolve())
    if output_path in input_paths or output_path.suffix.lower() != ".json":
        raise ValueError(
            "Receipt must be a separate JSON file; inputs cannot be overwritten"
        )
    started = time.perf_counter()
    receipt = {
        "status": "failed",
        "purpose": "engineering_smoke_not_performance_evidence",
        "started_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "settings": {
            "device": str(device),
            "batch_size": batch_size,
            "image_size": image_size,
            "seed": seed,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "amp_fp16": amp,
            "amp_initial_scale": AMP_INITIAL_SCALE if amp else None,
            "pretrained": pretrained,
            "weights": "IMAGENET1K_V1" if pretrained else None,
        },
        "optimizer_steps": 0,
        "real_image_batch_processed": False,
        "image_evidence": image_evidence,
    }
    monitor = None
    selected_device = None
    try:
        if not 2 <= batch_size <= 4 or not 32 <= image_size <= 224:
            raise ValueError("Smoke requires batch_size 2..4 and image_size 32..224")
        if (
            not math.isfinite(learning_rate)
            or learning_rate <= 0
            or not math.isfinite(weight_decay)
            or weight_decay < 0
        ):
            raise ValueError("Invalid optimizer settings")
        selected_device = torch.device(device)
        if selected_device.type not in {"cpu", "cuda"}:
            raise ValueError("Smoke supports cpu or cuda devices")
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        if amp and selected_device.type != "cuda":
            raise ValueError("FP16 smoke requires CUDA")
        seed_everything(seed)
        records = load_frozen_tbx11k(
            manifest,
            assignments,
            image_root=image_root,
            summary=summary,
            expected_split_sha256=expected_split_sha256,
        )
        train = [record for record in records if record.split == "train"]
        if len(train) < batch_size:
            raise ValueError("Training partition too small for requested batch")
        selected = random.Random(seed).sample(train, batch_size)
        for record in selected:
            if file_sha256(record.image_path) != record.checksum_sha256:
                raise ValueError(f"Raw image checksum mismatch: {record.sample_id}")
        receipt["inputs"] = {
            "manifest_sha256": file_sha256(manifest),
            "split_sha256": file_sha256(assignments),
            "summary_sha256": file_sha256(summary) if summary else None,
        }
        receipt["sample_ids"] = [record.sample_id for record in selected]
        receipt["partition"] = "train"
        dataset = ManifestDataset(
            selected,
            build_transform(training=True, image_size=image_size),
            label_space="tbx11k_internal_3class",
        )
        batch = [dataset[index] for index in range(len(dataset))]
        images = torch.stack([item["image"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        if (
            images.dtype != torch.float32
            or images.shape != (batch_size, 3, image_size, image_size)
            or labels.dtype != torch.int64
        ):
            raise ValueError("Unexpected batch shape or dtype")
        _finite(images, "input images")
        receipt["real_image_batch_processed"] = image_evidence == "real_tbx11k"
        receipt["batch"] = {
            "shape": list(images.shape),
            "dtype": str(images.dtype),
            "labels": labels.tolist(),
        }
        # Weight availability/download is resolved before the monitored GPU step.
        model = build_densenet121(pretrained=pretrained, allow_download=allow_download)
        if pretrained:
            receipt["pretrained_weights_sha256"] = file_sha256(
                Path(torch.hub.get_dir()) / "checkpoints" / "densenet121-a639ec97.pth"
            )
        if selected_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(selected_device)
            receipt["environment"]["gpu_name"] = torch.cuda.get_device_name(
                selected_device
            )
            receipt["environment"]["gpu_capability"] = list(
                torch.cuda.get_device_capability(selected_device)
            )
            monitor = GPUMonitor()
            monitor.start()
            monitor.check()
        step_started = time.perf_counter()
        model = model.to(selected_device)
        images, labels = images.to(selected_device), labels.to(selected_device)
        receipt.update(
            optimizer_step(
                model,
                images,
                labels,
                device=selected_device,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                amp=amp,
                monitor=monitor,
            )
        )
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)
        receipt["step_duration_seconds"] = time.perf_counter() - step_started
        if monitor:
            monitor.check()
        receipt["status"] = "passed"
    except Exception as error:
        receipt["error_type"] = type(error).__name__
        raise
    finally:
        if monitor:
            monitor.stop()
            receipt["gpu_monitor"] = {
                "samples": monitor.samples,
                "unavailable_reason": monitor.unavailable_reason,
                "overheated": monitor.overheated.is_set(),
                "stop_temperature_c": monitor.max_temperature,
            }
            if monitor.overheated.is_set():
                receipt["status"] = "failed"
                receipt["error_type"] = "ThermalStop"
            receipt["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
                selected_device
            )
            receipt["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(
                selected_device
            )
        receipt["duration_seconds"] = time.perf_counter() - started
        path = Path(receipt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    if receipt["status"] != "passed":
        raise RuntimeError("GPU thermal stop recorded; smoke did not pass")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Local TBX11K imgs directory; omitted uses audited relative paths",
    )
    parser.add_argument("--receipt", type=Path, default=Path("results/m1-smoke.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--amp", action="store_true", help="Explicit CUDA FP16; never BF16"
    )
    parser.add_argument("--allow-weight-download", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = args.config.resolve().parent.parent
    try:
        result = run_smoke(
            manifest=base / config["manifest"],
            assignments=base / config["assignments"],
            summary=base / config["summary"],
            expected_split_sha256=config["split_sha256"],
            image_root=args.image_root,
            receipt_path=args.receipt,
            device=args.device,
            batch_size=args.batch_size,
            amp=args.amp,
            allow_download=args.allow_weight_download,
            seed=config["seed"],
            image_size=config["image_size"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"Smoke failed ({type(error).__name__}): {error}")
        return 1
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "loss", "optimizer_steps", "duration_seconds")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
