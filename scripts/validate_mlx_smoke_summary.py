#!/usr/bin/env python3
"""Validate the shape and expected labels of an MLX smoke summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


EXPECTED_PASSED_LABELS = {
    "mlx_sft_raft_live_smoke",
    "mlx_dpo_reference_free_live_smoke",
    "mlx_dpo_reference_model_live_smoke",
    "mlx_grpo_reference_free_live_smoke",
    "mlx_dpo_loss_unit",
    "mlx_dpo_reference_model_terminal",
    "mlx_grpo_terminal",
}
EXPECTED_SKIPPED_LABELS = {"mlx_dpo_non_sigmoid_variants"}


class SummaryValidationError(ValueError):
    """Raised when an MLX smoke summary does not match the release contract."""


def validate_summary(summary: dict[str, Any]) -> None:
    if summary.get("status") != "passed":
        raise SummaryValidationError("summary.status must be 'passed'")
    readiness = summary.get("readiness")
    if not isinstance(readiness, dict):
        raise SummaryValidationError("summary.readiness must be an object")
    if readiness.get("status") != "ready":
        raise SummaryValidationError("readiness.status must be 'ready'")
    if readiness.get("executable") is not True:
        raise SummaryValidationError("readiness.executable must be true")
    probe = readiness.get("probe")
    if not isinstance(probe, dict) or "gpu" not in str(probe.get("default_device", "")).lower():
        raise SummaryValidationError("readiness.probe.default_device must report an MLX GPU")

    checks = summary.get("checks")
    if not isinstance(checks, list):
        raise SummaryValidationError("summary.checks must be a list")
    by_label = {
        check.get("label"): check
        for check in checks
        if isinstance(check, dict) and isinstance(check.get("label"), str)
    }
    missing = sorted((EXPECTED_PASSED_LABELS | EXPECTED_SKIPPED_LABELS) - set(by_label))
    if missing:
        raise SummaryValidationError(f"summary.checks missing labels: {', '.join(missing)}")

    for label in sorted(EXPECTED_PASSED_LABELS):
        check = by_label[label]
        if check.get("status") != "passed":
            raise SummaryValidationError(f"{label} must have status 'passed'")
        if check.get("returncode") != 0:
            raise SummaryValidationError(f"{label} must have returncode 0")

    for label in sorted(EXPECTED_SKIPPED_LABELS):
        check = by_label[label]
        if check.get("status") != "skipped":
            raise SummaryValidationError(f"{label} must have status 'skipped'")
        if not check.get("reason"):
            raise SummaryValidationError(f"{label} must include a skip reason")


def load_summary(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SummaryValidationError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SummaryValidationError("summary root must be an object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Path to mlx_smoke_summary.json")
    args = parser.parse_args(argv)

    try:
        validate_summary(load_summary(args.summary))
    except SummaryValidationError as exc:
        print(f"MLX smoke summary invalid: {exc}", file=sys.stderr)
        return 1
    print("MLX smoke summary valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
