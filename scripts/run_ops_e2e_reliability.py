#!/usr/bin/env python3
"""
Run deterministic non-code ops E2E launch reliability checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from halo_forge.ops_e2e_reliability import (
    DEFAULT_OPS_E2E_REPORT_FILE,
    OPS_E2E_STATUSES,
    build_ops_e2e_report,
    compute_ops_e2e_reliability,
    validate_ops_e2e_module,
    write_ops_e2e_report,
)
from halo_forge.ops_module_readiness import OPS_MODULES
from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED, normalize_seed


def _parse_validation_targets(values: Iterable[str]) -> list[Tuple[str, Path]]:
    targets: list[Tuple[str, Path]] = []
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(
                f"Invalid validation target '{text}'. Expected format: module=/path/to/output"
            )
        module, path_text = text.split("=", 1)
        module_key = module.strip().lower()
        if module_key not in OPS_MODULES:
            raise ValueError(f"Unsupported module in validation target: {module_key}")
        output_path = Path(path_text.strip())
        targets.append((module_key, output_path))
    return targets


def _parse_fixture_pack(pack: str) -> list[Tuple[str, Path]]:
    pack_text = str(pack or "").strip()
    if not pack_text:
        return []

    if "/" in pack_text or pack_text.startswith("."):
        pack_root = Path(pack_text).expanduser()
        if not pack_root.is_absolute():
            pack_root = (REPO_ROOT / pack_root).resolve()
    else:
        pack_root = (REPO_ROOT / "tests" / "fixtures" / "ops_e2e" / pack_text).resolve()

    if not pack_root.exists() or not pack_root.is_dir():
        raise ValueError(f"Fixture pack directory not found: {pack_root}")

    targets: list[Tuple[str, Path]] = []
    for module in OPS_MODULES:
        if module == "ui_ops":
            targets.append((module, REPO_ROOT))
            continue
        module_dir = pack_root / module
        if not module_dir.exists() or not module_dir.is_dir():
            raise ValueError(f"Fixture pack missing module directory: {module_dir}")
        targets.append((module, module_dir))
    return targets


def _matrix(seed: int) -> Dict[str, list[dict]]:
    return {
        "vlm": [
            {
                "scenario": "launch_stop_relaunch_resume",
                "command": [
                    "halo-forge",
                    "vlm",
                    "train",
                    "--dataset",
                    "textvqa",
                    "--seed",
                    str(seed),
                ],
            }
        ],
        "audio": [
            {
                "scenario": "launch_stop_relaunch_resume",
                "command": [
                    "halo-forge",
                    "audio",
                    "train",
                    "--dataset",
                    "librispeech",
                    "--seed",
                    str(seed),
                ],
            }
        ],
        "reasoning": [
            {
                "scenario": "launch_stop_relaunch_resume",
                "command": [
                    "halo-forge",
                    "reasoning",
                    "train",
                    "--dataset",
                    "gsm8k",
                    "--seed",
                    str(seed),
                ],
            }
        ],
        "agentic": [
            {
                "scenario": "launch_stop_relaunch_resume",
                "command": [
                    "halo-forge",
                    "agentic",
                    "train",
                    "--dataset",
                    "xlam",
                    "--seed",
                    str(seed),
                ],
            }
        ],
        "inference": [
            {
                "scenario": "launch_stop_relaunch",
                "command": [
                    "halo-forge",
                    "inference",
                    "optimize",
                    "--target-precision",
                    "int4",
                ],
            }
        ],
        "benchmark": [
            {
                "scenario": "launch_stop_relaunch",
                "command": [
                    "halo-forge",
                    "benchmark",
                    "eval",
                    "--benchmark",
                    "humaneval",
                ],
            }
        ],
        "ui_ops": [
            {
                "scenario": "route_surface_contract",
                "command": ["halo-forge", "ui", "--no-browser"],
            }
        ],
    }


def _print_report_lines(modules: Dict[str, Any]) -> None:
    for module in OPS_MODULES:
        entry = modules[module]
        resume_ok = bool(entry.resume_latest_ok) if entry.resume_latest_ok is not None else False
        print(
            "OPS_E2E "
            f"module={module} status={entry.status} "
            f"launch={1 if entry.launch_ok else 0} "
            f"stop={1 if entry.stop_ok else 0} "
            f"relaunch={1 if entry.relaunch_ok else 0} "
            f"resume={1 if resume_ok else 0}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Non-code ops E2E reliability runner")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAINING_SEED,
        help="Deterministic seed used for report generation",
    )
    parser.add_argument(
        "--print-matrix",
        action="store_true",
        help="Print deterministic scenario matrix",
    )
    parser.add_argument(
        "--matrix-format",
        choices=("json", "markdown"),
        default="markdown",
        help="Matrix output format",
    )
    parser.add_argument(
        "--validate-module",
        action="append",
        default=[],
        help="Validate module target (format: module=/path/to/output)",
    )
    parser.add_argument(
        "--fixture-pack",
        default="",
        help=(
            "Use fixture pack for module validation. "
            "Examples: v1 or tests/fixtures/ops_e2e/v1"
        ),
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write canonical E2E reliability report JSON",
    )
    parser.add_argument(
        "--report-file",
        default=str(DEFAULT_OPS_E2E_REPORT_FILE),
        help="E2E reliability report output path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any module status is fail",
    )
    args = parser.parse_args()

    seed = normalize_seed(args.seed)
    matrix = _matrix(seed)

    if args.print_matrix:
        if args.matrix_format == "json":
            print(json.dumps(matrix, indent=2, sort_keys=True))
        else:
            print("# Ops E2E Reliability Matrix\n")
            for module in OPS_MODULES:
                print(f"## {module.upper()}")
                for scenario in matrix[module]:
                    command = " ".join(str(part) for part in scenario["command"])
                    print(f"- `{scenario['scenario']}`")
                    print(f"  - Command: `{command}`")
                print()

    try:
        fixture_targets = _parse_fixture_pack(args.fixture_pack)
        explicit_targets = _parse_validation_targets(args.validate_module)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    targets_by_module: Dict[str, Path] = {}
    for module, path in fixture_targets:
        targets_by_module[module] = path
    for module, path in explicit_targets:
        targets_by_module[module] = path
    targets = [(module, path) for module, path in targets_by_module.items()]

    if targets:
        entries = {
            module: validate_ops_e2e_module(module=module, output_dir=path, seed=seed)
            for module, path in targets
        }
        report = build_ops_e2e_report(module_entries=entries, seed=seed, source="script")
    else:
        report = compute_ops_e2e_reliability(seed=seed, source="script")

    _print_report_lines(report.modules)

    if args.write_report:
        report_path = Path(args.report_file)
        write_ops_e2e_report(report_path, report)
        print(f"Wrote ops E2E reliability report: {report_path}")

    if args.strict:
        failing = [module for module in OPS_MODULES if report.modules[module].status == "fail"]
        if failing:
            print("ERROR: failing modules: " + ", ".join(failing))
            return 1

    invalid_statuses = [
        module
        for module in OPS_MODULES
        if report.modules[module].status not in OPS_E2E_STATUSES
    ]
    if invalid_statuses:
        print("ERROR: invalid statuses for modules: " + ", ".join(invalid_statuses))
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
