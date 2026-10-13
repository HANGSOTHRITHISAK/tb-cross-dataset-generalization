"""Supervised M3B CLI: exactly one explicitly requested session, then stop."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from src.training.preflight import supervise, write_receipt


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m3b.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--max-seconds", type=float, default=3600)
    args = parser.parse_args()
    base, output = args.config.resolve().parent.parent, args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh session output directory")
    command = [
        sys.executable,
        "-m",
        "src.training.baseline",
        "--config",
        str(args.config.resolve()),
        "--output-dir",
        str(output),
        "--max-seconds",
        str(args.max_seconds),
    ]
    if args.resume:
        command.extend(["--resume", str(args.resume.resolve())])
    if args.image_root:
        command.extend(["--image-root", str(args.image_root.resolve())])
    try:
        raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        inputs = [
            args.config,
            *(base / raw[k] for k in ("manifest", "assignments", "summary")),
        ]
        if args.resume:
            inputs.append(args.resume)
        if any(p.resolve().is_relative_to(output) for p in inputs):
            raise ValueError("Output cannot contain inputs or a previous checkpoint")
        receipt = supervise(
            command,
            output,
            max_seconds=args.max_seconds,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        KeyboardInterrupt,
    ) as error:
        output.mkdir(parents=True, exist_ok=True)
        receipt = {
            "status": "failed",
            "checkpoint_usable": False,
            "run_id": str(uuid.uuid4()),
            "purpose": "M3B_baseline_session",
            "phase": "supervisor_configuration",
            "error_type": type(error).__name__,
            "failure_reason": "invalid_session_configuration",
        }
    if receipt["status"] != "passed":
        receipt.update(session_completed=False, session_outcome="failed_or_interrupted")
        write_receipt(output / "receipt.json", receipt)
    print(
        json.dumps(
            {
                k: receipt.get(k)
                for k in (
                    "status",
                    "session_outcome",
                    "completed_epoch",
                    "experiment_finished",
                    "stale_epochs",
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
