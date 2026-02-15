#!/usr/bin/env python3
"""Run all-module bootstrap evidence generation and write canonical report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from halo_forge.all_module_readiness import ALL_MODULES
from halo_forge.all_module_bootstrap import (
    DEFAULT_ALL_MODULE_BOOTSTRAP_OUTPUT_ROOT,
    DEFAULT_ALL_MODULE_BOOTSTRAP_REPORT_FILE,
    compute_all_module_bootstrap,
    write_all_module_bootstrap_report,
)
from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED, normalize_seed


def _selected_modules(values: list[str]) -> List[str]:
    selected: List[str] = []
    for module in values:
        key = str(module or "").strip().lower()
        if not key:
            continue
        if key not in ALL_MODULES:
            raise ValueError(f"Unsupported module selection: {key}")
        if key not in selected:
            selected.append(key)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all-module bootstrap evidence generation")
    parser.add_argument(
        "--bootstrap-profile",
        default="contract-v1",
        choices=["contract-v1", "live-local"],
        help="Bootstrap profile (default: contract-v1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAINING_SEED,
        help="Deterministic seed (default: 42)",
    )
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        help="Filter module(s) for bootstrap (repeatable)",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_ALL_MODULE_BOOTSTRAP_OUTPUT_ROOT),
        help="Evidence output root used for bootstrap artifact generation",
    )
    parser.add_argument(
        "--report-file",
        default=str(DEFAULT_ALL_MODULE_BOOTSTRAP_REPORT_FILE),
        help="Bootstrap report JSON path",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Retained for parity; report is always written",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail non-zero on module fail status",
    )
    args = parser.parse_args()

    try:
        selected = _selected_modules(args.module)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    seed = normalize_seed(args.seed)

    try:
        report = compute_all_module_bootstrap(
            bootstrap_profile=args.bootstrap_profile,
            seed=seed,
            source="script",
            output_root=Path(args.output_root),
            module_filters=selected,
            strict=args.strict,
        )
    except Exception as exc:
        print(f"ERROR: failed to compute all-module bootstrap report: {exc}")
        return 2

    modules_to_print = selected or list(ALL_MODULES)
    for module in modules_to_print:
        entry = report.modules[module]
        print(
            "ALL_BOOTSTRAP "
            f"module={module} status={entry.status} "
            f"created={len(entry.artifacts_created)} errors={len(entry.errors)} warnings={len(entry.warnings)}"
        )

    report_path = Path(args.report_file)
    try:
        write_all_module_bootstrap_report(report_path, report)
    except Exception as exc:
        print(f"ERROR: failed to write bootstrap report: {exc}")
        return 2
    print(f"Wrote all-module bootstrap report: {report_path}")

    if args.strict:
        failing = [module for module in modules_to_print if report.modules[module].status == "fail"]
        if failing:
            print("ERROR: failing modules: " + ", ".join(failing))
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
