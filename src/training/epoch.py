"""Reusable epoch infrastructure; no entry point or research schedule."""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.training.pilot import (
    classification_metrics,
    is_better,
    state_sha256,
    train_step,
)
from src.training.smoke import _finite

CHECKPOINT_VERSION = 1


def rng_snapshot(generators):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            *numpy_state[2:],
        ],
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
        "loaders": {name: gen.get_state() for name, gen in generators.items()},
    }


def restore_rng(state, generators):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    for name, generator in generators.items():
        generator.set_state(state["loaders"][name])


def finite_state(value):
    if isinstance(value, torch.Tensor):
        _finite(value, "checkpoint state")
    elif isinstance(value, dict):
        for item in value.values():
            finite_state(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_state(item)


class EpochTrainer:
    """Persistent AdamW and tested epoch-boundary state, FP32 only for M3A."""

    def __init__(self, model, loaders, generators, *, device, config, binding):
        self.model = model.to(device)
        self.loaders, self.generators = loaders, generators
        self.device, self.binding = device, binding
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.completed_epochs = self.optimizer_steps = 0
        self.best = self.best_model = None
        self.history = []
        self.at_boundary = True

    def train_epoch(self, *, guard, receipt, progress=lambda: None):
        if not self.at_boundary:
            raise ValueError(
                "Cannot restart a partial epoch; restore a boundary checkpoint"
            )
        self.at_boundary = False
        start = time.perf_counter()
        expected = [r.sample_id for r in self.loaders["train"].dataset.records]
        observed, losses = [], []
        receipt.update(phase="training", training_samples_this_epoch=0)
        iterator = iter(self.loaders["train"])
        for _ in range(len(self.loaders["train"])):
            step_start = time.perf_counter()
            guard()
            batch = next(iterator)
            loss = train_step(
                self.model,
                batch,
                self.optimizer,
                self.scaler,
                device=self.device,
                amp=False,
                weights=None,
                guard=guard,
            )
            self.optimizer_steps += 1
            observed.extend(batch["sample_id"])
            losses.append((loss, len(batch["label"])))
            receipt.update(
                optimizer_steps=self.optimizer_steps,
                training_samples_this_epoch=len(observed),
            )
            entry = {
                "epoch": self.completed_epochs + 1,
                "step": self.optimizer_steps,
                "loss": loss,
                "seconds_including_fetch": time.perf_counter() - step_start,
                "sample_ids": batch["sample_id"],
            }
            if self.device.type == "cuda":
                entry.update(
                    cuda_allocated_bytes=torch.cuda.memory_allocated(self.device),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(self.device),
                )
            receipt["steps"].append(entry)
            progress()
            guard()
        if sorted(observed) != sorted(expected) or len(set(observed)) != len(expected):
            raise ValueError(
                "Epoch did not consume every assigned training sample once"
            )
        return {
            "training_loss": sum(v * n for v, n in losses) / len(observed),
            "training_sample_count": len(observed),
            "training_seconds": time.perf_counter() - start,
            "training_order": observed,
        }

    def validate(self, *, guard, receipt, progress=lambda: None):
        loader = self.loaders["validation"]
        if any(r.split != "validation" for r in loader.dataset.records):
            raise ValueError("Evaluation requires exclusively validation records")
        start = time.perf_counter()
        receipt.update(phase="validation", validation_samples_this_epoch=0)
        self.model.eval()
        matrix = torch.zeros((3, 3), dtype=torch.int64)
        observed, timings = [], []
        total_loss = 0.0
        with torch.inference_mode():
            iterator = iter(loader)
            for _ in range(len(loader)):
                step_start = time.perf_counter()
                guard()
                batch = next(iterator)
                images, labels = (
                    batch["image"].to(self.device),
                    batch["label"].to(self.device),
                )
                _finite(images, "validation inputs")
                logits = self.model(images)
                if logits.shape != (len(labels), 3):
                    raise ValueError("Expected three-class validation logits")
                _finite(logits, "validation logits")
                loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
                _finite(loss, "validation loss")
                total_loss += loss.item()
                matrix += torch.bincount(
                    (labels * 3 + logits.argmax(1)).cpu(), minlength=9
                ).reshape(3, 3)
                observed.extend(batch["sample_id"])
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                timings.append(time.perf_counter() - step_start)
                receipt["validation_samples_this_epoch"] = len(observed)
                progress()
                guard()
        expected = [r.sample_id for r in loader.dataset.records]
        if observed != expected or len(set(observed)) != len(expected):
            raise ValueError(
                "Validation must consume every assigned sample once in order"
            )
        metrics = classification_metrics(matrix)
        metrics.update(
            loss=total_loss / len(observed),
            partition="validation",
            epoch=self.completed_epochs + 1,
            batch_seconds_including_fetch=timings,
        )
        return metrics, time.perf_counter() - start

    def finish_epoch(self, training, metrics, validation_seconds):
        if self.at_boundary or metrics["epoch"] != self.completed_epochs + 1:
            raise ValueError("Invalid epoch boundary")
        if metrics["sample_count"] != len(self.loaders["validation"].dataset):
            raise ValueError("Partial validation cannot select a checkpoint")
        improved = is_better(metrics, self.best)
        if improved:
            self.best = copy.deepcopy(metrics)
            self.best_model = {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }
        self.completed_epochs += 1
        self.history.append(
            {
                "epoch": self.completed_epochs,
                **training,
                "validation": metrics,
                "validation_seconds": validation_seconds,
                "checkpoint_selected": improved,
            }
        )
        self.at_boundary = True

    def save(self, path):
        if not self.at_boundary or self.completed_epochs < 1 or self.best is None:
            raise ValueError(
                "Only completed training/validation boundaries are resumable"
            )
        state = {
            "format_version": CHECKPOINT_VERSION,
            "binding": self.binding,
            "completed_epochs": self.completed_epochs,
            "optimizer_steps": self.optimizer_steps,
            "model_state": self.model.state_dict(),
            "model_sha256": state_sha256(self.model),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "rng": rng_snapshot(self.generators),
            "best_validation": self.best,
            "best_model_state": self.best_model,
            "history": self.history,
        }
        finite_state(state)
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def restore(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format_version") != CHECKPOINT_VERSION:
            raise ValueError("Incompatible checkpoint format")
        if state.get("binding") != self.binding:
            raise ValueError(
                "Changed configuration, frozen inputs, source or environment"
            )
        epochs, steps = state["completed_epochs"], state["optimizer_steps"]
        if (
            type(epochs) is not int
            or epochs < 1
            or type(steps) is not int
            or steps != epochs * len(self.loaders["train"])
            or len(state["history"]) != epochs
            or state["best_validation"]["partition"] != "validation"
            or not 1 <= state["best_validation"]["epoch"] <= epochs
            or set(state["rng"]["loaders"]) != set(self.generators)
            or len(state["rng"]["torch_cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError("Invalid checkpoint counters, selection or RNG topology")
        finite_state(state)
        self.model.load_state_dict(state["model_state"], strict=True)
        if state_sha256(self.model) != state["model_sha256"]:
            raise ValueError("Checkpoint model hash mismatch")
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scaler.load_state_dict(state["scaler_state"])
        self.completed_epochs, self.optimizer_steps = epochs, steps
        self.best, self.best_model = state["best_validation"], state["best_model_state"]
        self.history = state["history"]
        restore_rng(state["rng"], self.generators)
        self.at_boundary = True
        return state["model_sha256"]
