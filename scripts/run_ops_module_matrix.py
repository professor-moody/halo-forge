#!/usr/bin/env python3
"""
Build and validate cross-module operations matrix artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from halo_forge.ops_module_readiness import (
    DEFAULT_OPS_READINESS_REPORT_FILE,
    OPS_MODULES,
    OpsModuleReadiness,
    build_ops_readiness_report,
    compute_ops_module_readiness,
    default_output_map,
    validate_ops_module,
    write_ops_readiness_report,
)
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
        pack_root = (REPO_ROOT / "tests" / "fixtures" / "ops_readiness" / pack_text).resolve()

    if not pack_root.exists() or not pack_root.is_dir():
        raise ValueError(f"Fixture pack directory not found: {pack_root}")

    targets: list[Tuple[str, Path]] = []
    for module in OPS_MODULES:
        if module == "ui_ops":
            # ui_ops validator inspects actual route/service wiring in the repository.
            targets.append((module, REPO_ROOT))
            continue
        module_dir = pack_root / module
        if not module_dir.exists() or not module_dir.is_dir():
            raise ValueError(f"Fixture pack missing module directory: {module_dir}")
        targets.append((module, module_dir))
    return targets


def _matrix(seed: int) -> Dict[str, list[dict]]:
    base = default_output_map()
    return {
        "vlm": [
            {
                "scenario": "train_contract_validation",
                "command": [
                    "halo-forge",
                    "vlm",
                    "train",
                    "--dataset",
                    "textvqa",
                    "--seed",
                    str(seed),
                ],
                "expected_output_dir": base["vlm"],
            },
        ],
        "audio": [
            {
                "scenario": "train_contract_validation",
                "command": [
                    "halo-forge",
                    "audio",
                    "train",
                    "--dataset",
                    "librispeech",
                    "--seed",
                    str(seed),
                ],
                "expected_output_dir": base["audio"],
            },
        ],
        "reasoning": [
            {
                "scenario": "train_contract_validation",
                "command": [
                    "halo-forge",
                    "reasoning",
                    "train",
                    "--dataset",
                    "gsm8k",
                    "--seed",
                    str(seed),
                ],
                "expected_output_dir": base["reasoning"],
            },
        ],
        "agentic": [
            {
                "scenario": "train_contract_validation",
                "command": [
                    "halo-forge",
                    "agentic",
                    "train",
                    "--dataset",
                    "xlam",
                    "--seed",
                    str(seed),
                ],
                "expected_output_dir": base["agentic"],
            },
        ],
        "inference": [
            {
                "scenario": "optimize_contract_validation",
                "command": [
                    "halo-forge",
                    "inference",
                    "optimize",
                    "--target-precision",
                    "int4",
                ],
                "expected_output_dir": base["inference"],
            },
        ],
        "benchmark": [
            {
                "scenario": "results_contract_validation",
                "command": [
                    "halo-forge",
                    "benchmark",
                    "eval",
                    "--benchmark",
                    "humaneval",
                ],
                "expected_output_dir": base["benchmark"],
            },
        ],
        "ui_ops": [
            {
                "scenario": "route_service_contract_validation",
                "command": ["halo-forge", "ui"],
                "expected_output_dir": base["ui_ops"],
            },
        ],
    }


def _print_report_lines(report_modules: Dict[str, OpsModuleReadiness]) -> None:
    for module in OPS_MODULES:
        entry = report_modules[module]
        print(
            "OPS_READY "
            f"module={module} status={entry.status} "
            f"errors={len(entry.errors)} warnings={len(entry.warnings)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-module operations matrix helper",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAINING_SEED,
        help="Deterministic seed used for matrix/report generation",
    )
    parser.add_argument(
        "--print-matrix",
        action="store_true",
        help="Print deterministic matrix scenarios",
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
        help="Validate module output target (format: module=/path/to/output)",
    )
    parser.add_argument(
        "--fixture-pack",
        default="",
        help=(
            "Use fixture pack for module validation. "
            "Examples: v1 or tests/fixtures/ops_readiness/v1"
        ),
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write canonical ops readiness report",
    )
    parser.add_argument(
        "--report-file",
        default=str(DEFAULT_OPS_READINESS_REPORT_FILE),
        help="Ops readiness report output path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any module status is fail",
    )
    args = parser.parse_args()

    seed = normalize_seed(args.seed)
    matrix = _matrix(seed)

    if args.print_matrix or (not args.validate_module and not args.write_report):
        if args.matrix_format == "json":
            print(json.dumps(matrix, indent=2, sort_keys=True))
        else:
            print("# Ops Module Matrix\n")
            for module in OPS_MODULES:
                print(f"## {module.upper()}")
                for scenario in matrix[module]:
                    cmd = " ".join(str(part) for part in scenario["command"])
                    print(f"- `{scenario['scenario']}`")
                    print(f"  - Command: `{cmd}`")
                    print(f"  - Expected output: `{scenario['expected_output_dir']}`")
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
        # Explicit module targets override fixture pack mapping for that module.
        targets_by_module[module] = path
    targets = [(module, path) for module, path in targets_by_module.items()]

    if targets:
        entries: Dict[str, OpsModuleReadiness] = {}
        for module, path in targets:
            entries[module] = validate_ops_module(module=module, output_dir=path, seed=seed)
        report = build_ops_readiness_report(
            module_entries=entries,
            seed=seed,
            source="script",
        )
    else:
        report = compute_ops_module_readiness(seed=seed, source="script")

    _print_report_lines(report.modules)

    if args.write_report:
        report_path = Path(args.report_file)
        write_ops_readiness_report(report_path, report)
        print(f"Wrote ops readiness report: {report_path}")

    if args.strict:
        failing = [module for module in OPS_MODULES if report.modules[module].status == "fail"]
        if failing:
            print("ERROR: failing modules: " + ", ".join(failing))
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
