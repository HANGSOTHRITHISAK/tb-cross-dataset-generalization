"""Launch M3A in a worker that is forcibly stopped at its hard wall-time budget."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def write_receipt(path, receipt):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def supervise(command, output, *, max_seconds, cwd=None):
    """Separate process deadline covers imports, I/O, kernels and worker cleanup."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 3600:
        raise ValueError("Hard budget must be positive and at most 3600 seconds")
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".log")
    started = time.perf_counter()
    timed_out = False
    supervisor_error = None
    process = None
    try:
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env={**os.environ, "M3A_SUPERVISED": "1"},
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                process.wait(
                    timeout=max(0.001, max_seconds - (time.perf_counter() - started))
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait(timeout=5)
    except (OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        supervisor_error = type(error).__name__
    finally:
        # KeyboardInterrupt or unexpected supervisor failure also stops training.
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "receipt.json"
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        receipt = {
            "status": "failed",
            "checkpoint_usable": False,
            "phase": "worker_startup",
        }
    receipt["supervisor"] = {
        "hard_budget_seconds": max_seconds,
        "duration_seconds_including_imports": time.perf_counter() - started,
        "worker_exit_code": process.returncode if process else None,
        "forced_timeout": timed_out,
        "log_filename": log.name,
    }
    if timed_out or supervisor_error or process is None or process.returncode != 0:
        receipt.update(status="failed", checkpoint_usable=False)
        if timed_out:
            receipt.update(
                error_type="HardTimeout",
                failure_reason="hard_duration_budget",
                progress_is_last_receipt_lower_bound=True,
            )
        elif supervisor_error:
            receipt.update(
                error_type=supervisor_error, failure_reason="supervisor_failure"
            )
        else:
            receipt.setdefault("failure_reason", "worker_failed")
    write_receipt(path, receipt)
    return receipt


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/preflight.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    base = args.config.resolve().parent.parent
    inputs = [
        args.config,
        *(base / raw[k] for k in ("manifest", "assignments", "summary")),
    ]
    if args.resume:
        inputs.append(args.resume)
    if any(p.resolve().is_relative_to(output) for p in inputs):
        raise ValueError("Output directory cannot contain input metadata or checkpoint")
    command = [
        sys.executable,
        "-m",
        "src.training.full_data",
        "--config",
        str(args.config.resolve()),
        "--output-dir",
        str(output),
    ]
    if args.image_root:
        command.extend(["--image-root", str(args.image_root.resolve())])
    if args.resume:
        command.extend(["--resume", str(args.resume.resolve())])
    receipt = supervise(
        command,
        output,
        max_seconds=raw["preflight"]["max_seconds"],
        cwd=Path(__file__).resolve().parents[2],
    )
    print(
        json.dumps(
            {
                k: receipt.get(k)
                for k in (
                    "status",
                    "optimizer_steps",
                    "duration_seconds",
                    "failure_reason",
                    "supervisor",
                )
            },
            indent=2,
        )
    )
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
