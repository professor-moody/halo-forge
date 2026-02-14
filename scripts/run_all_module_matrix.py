#!/usr/bin/env python3
"""
Build and validate all-module (coding + non-coding) readiness artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from halo_forge.all_module_readiness import (
    ALL_MODULES,
    DEFAULT_ALL_MODULE_READINESS_REPORT_FILE,
    AllModuleReadiness,
    build_all_module_readiness_report,
    compute_all_module_readiness,
    default_output_map,
    validate_all_module,
    write_all_module_readiness_report,
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
        if module_key not in ALL_MODULES:
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
        pack_root = (REPO_ROOT / "tests" / "fixtures" / "all_modules" / pack_text).resolve()

    if not pack_root.exists() or not pack_root.is_dir():
        raise ValueError(f"Fixture pack directory not found: {pack_root}")

    targets: list[Tuple[str, Path]] = []
    for module in ALL_MODULES:
        if module == "ui_ops":
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
        "config": [
            {
                "scenario": "config_contract_validation",
                "command": ["halo-forge", "config", "validate", "configs/raft.yaml"],
                "expected_output_dir": base["config"],
            },
        ],
        "data": [
            {
                "scenario": "data_contract_validation",
                "command": ["halo-forge", "data", "validate", "data/sample.jsonl"],
                "expected_output_dir": base["data"],
            },
        ],
        "info": [
            {
                "scenario": "info_contract_validation",
                "command": ["halo-forge", "info"],
                "expected_output_dir": base["info"],
            },
        ],
        "plot": [
            {
                "scenario": "plot_contract_validation",
                "command": ["halo-forge", "plot", "benchmarks", "results/benchmarks"],
                "expected_output_dir": base["plot"],
            },
        ],
        "sft": [
            {
                "scenario": "sft_contract_validation",
                "command": ["halo-forge", "sft", "train", "--model", "Qwen/Qwen2.5-Coder-3B"],
                "expected_output_dir": base["sft"],
            },
        ],
        "raft": [
            {
                "scenario": "raft_contract_validation",
                "command": ["halo-forge", "raft", "train", "--model", "Qwen/Qwen2.5-Coder-3B"],
                "expected_output_dir": base["raft"],
            },
        ],
        "benchmark_code": [
            {
                "scenario": "benchmark_code_contract_validation",
                "command": [
                    "halo-forge",
                    "benchmark",
                    "eval",
                    "--benchmark",
                    "humaneval",
                    "--limit",
                    "5",
                ],
                "expected_output_dir": base["benchmark_code"],
            },
        ],
        "benchmark_non_code": [
            {
                "scenario": "benchmark_non_code_contract_validation",
                "command": [
                    "halo-forge",
                    "reasoning",
                    "benchmark",
                    "--dataset",
                    "gsm8k",
                    "--limit",
                    "20",
                ],
                "expected_output_dir": base["benchmark_non_code"],
            },
        ],
        "inference": [
            {
                "scenario": "inference_contract_validation",
                "command": [
                    "halo-forge",
                    "inference",
                    "optimize",
                    "--target-precision",
                    "int4",
                    "--dry-run",
                ],
                "expected_output_dir": base["inference"],
            },
        ],
        "vlm": [
            {
                "scenario": "vlm_contract_validation",
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
                "scenario": "audio_contract_validation",
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
                "scenario": "reasoning_contract_validation",
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
                "scenario": "agentic_contract_validation",
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
        "ui_ops": [
            {
                "scenario": "ui_ops_contract_validation",
                "command": ["halo-forge", "ui", "--no-browser"],
                "expected_output_dir": base["ui_ops"],
            },
        ],
    }


def _print_report_lines(report_modules: Dict[str, AllModuleReadiness]) -> None:
    for module in sorted(report_modules.keys()):
        entry = report_modules[module]
        print(
            "ALL_READY "
            f"module={module} status={entry.status} "
            f"errors={len(entry.errors)} warnings={len(entry.warnings)}"
        )


def _select_modules(args_modules: List[str]) -> List[str]:
    requested = []
    for module in args_modules:
        key = str(module or "").strip().lower()
        if not key:
            continue
        if key not in ALL_MODULES:
            raise ValueError(f"Unsupported module selection: {key}")
        if key not in requested:
            requested.append(key)
    return requested


def main() -> int:
    parser = argparse.ArgumentParser(description="All-module readiness matrix helper")
    parser.add_argument(
        "--profile",
        default="bounded-v1",
        help="Readiness profile name (default: bounded-v1)",
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
        "--module",
        action="append",
        default=[],
        help="Module filter (repeatable). Defaults to all modules.",
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
        help="Use fixture pack (e.g., v1 or tests/fixtures/all_modules/v1)",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write canonical all-module readiness report",
    )
    parser.add_argument(
        "--report-file",
        default=str(DEFAULT_ALL_MODULE_READINESS_REPORT_FILE),
        help="All-module readiness report output path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any selected module has status fail",
    )
    args = parser.parse_args()

    _ = args.profile  # reserved for future profile branching
    seed = normalize_seed(args.seed)
    matrix = _matrix(seed)

    try:
        selected_modules = _select_modules(args.module)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    if not selected_modules:
        selected_modules = list(ALL_MODULES)

    if args.print_matrix or (not args.validate_module and not args.write_report):
        filtered = {module: matrix[module] for module in selected_modules}
        if args.matrix_format == "json":
            print(json.dumps(filtered, indent=2, sort_keys=True))
        else:
            print("# All Module Matrix\n")
            for module in selected_modules:
                print(f"## {module.upper()}")
                for scenario in filtered[module]:
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
        if module in selected_modules:
            targets_by_module[module] = path
    for module, path in explicit_targets:
        if module in selected_modules:
            targets_by_module[module] = path
    targets = [(module, path) for module, path in targets_by_module.items()]

    if targets:
        entries: Dict[str, AllModuleReadiness] = {}
        for module, path in targets:
            entries[module] = validate_all_module(
                module=module,
                output_dir=path,
                seed=seed,
                require_artifacts=True,
            )
        report = build_all_module_readiness_report(
            module_entries=entries,
            seed=seed,
            source="script",
        )
    else:
        output_map = default_output_map()
        output_map = {module: output_map[module] for module in selected_modules}
        report = compute_all_module_readiness(
            output_map=output_map,
            seed=seed,
            source="script",
            require_artifacts=False,
        )

    filtered_modules = {module: report.modules[module] for module in selected_modules}
    _print_report_lines(filtered_modules)

    if args.write_report:
        report_path = Path(args.report_file)
        write_all_module_readiness_report(report_path, report)
        print(f"Wrote all-module readiness report: {report_path}")

    if args.strict:
        failing = [
            module
            for module in selected_modules
            if report.modules[module].status == "fail"
        ]
        if failing:
            print("ERROR: failing modules: " + ", ".join(failing))
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
