#!/usr/bin/env python3
"""Terminal-only MLX smoke runner.

This script is intentionally outside normal CI. It first proves MLX can execute
in the current Terminal session, then runs the bounded MLX smoke coverage that
ships with the repo.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from halo_forge.backend.mlx_readiness import check_mlx_readiness


def _run(label: str, command: list[str]) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    ended = datetime.now(timezone.utc)
    return {
        "label": label,
        "status": "passed" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "command": command,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "stdout_tail": proc.stdout.splitlines()[-40:],
        "stderr_tail": proc.stderr.splitlines()[-40:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded MLX terminal smoke checks")
    parser.add_argument("--output-dir", required=True, help="Directory for mlx_smoke_summary.json")
    parser.add_argument("--skip-live", action="store_true", help="Only run readiness and unit-level checks")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "mlx_smoke_summary.json"

    readiness = check_mlx_readiness()
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "readiness": readiness.to_dict(),
        "checks": [],
    }

    if not readiness.executable:
        summary["status"] = "skipped"
        summary["reason"] = "MLX is not executable in this process environment."
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(summary_path)
        return 2

    if not args.skip_live:
        summary["checks"].append(
            _run(
                "mlx_sft_raft_live_smoke",
                [sys.executable, "-m", "pytest", "tests/test_mlx_smoke.py", "-q"],
            )
        )
        summary["checks"].append(
            _run(
                "mlx_dpo_reference_free_live_smoke",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_mlx_live_training.py::test_mlx_dpo_live_reference_free_sigmoid_runs_one_cycle",
                    "-q",
                ],
            )
        )
        summary["checks"].append(
            _run(
                "mlx_dpo_reference_model_live_smoke",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_mlx_live_training.py::test_mlx_dpo_live_reference_model_sigmoid_runs_one_cycle",
                    "-q",
                ],
            )
        )
        summary["checks"].append(
            _run(
                "mlx_dpo_non_sigmoid_variants",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_mlx_live_training.py::test_mlx_dpo_live_non_sigmoid_variants_run_one_cycle",
                    "-q",
                ],
            )
        )
        summary["checks"].append(
            _run(
                "mlx_grpo_reference_free_live_smoke",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_mlx_live_training.py::test_mlx_grpo_live_reference_free_runs_one_cycle",
                    "-q",
                ],
            )
        )
        summary["checks"].append(
            _run(
                "mlx_grpo_reference_model_live_smoke",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_mlx_live_training.py::test_mlx_grpo_live_reference_model_runs_one_cycle",
                    "-q",
                ],
            )
        )

    summary["checks"].append(
        _run(
            "mlx_dpo_loss_unit",
            [sys.executable, "-m", "pytest", "tests/test_mlx_dpo_loss.py", "-q"],
        )
    )
    summary["checks"].append(
        _run(
            "mlx_dpo_reference_model_terminal",
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_mlx_terminal_smoke.py::test_mlx_dpo_reference_model_terminal",
                "-q",
            ],
        )
    )
    summary["checks"].append(
        _run(
            "mlx_grpo_terminal",
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_mlx_terminal_smoke.py::test_mlx_grpo_terminal",
                "-q",
            ],
        )
    )

    failed = [check for check in summary["checks"] if check.get("status") == "failed"]
    summary["status"] = "failed" if failed else "passed"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(summary_path)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
