#!/usr/bin/env python3
"""Run all-module qualification checks and optional baseline drift comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from halo_forge.all_module_readiness import ALL_MODULES
from halo_forge.all_module_qualification import (
    ALL_MODULE_QUALIFICATION_STATUSES,
    DEFAULT_ALL_MODULE_QUALIFICATION_BASELINE_FILE,
    DEFAULT_ALL_MODULE_QUALIFICATION_REPORT_FILE,
    build_qualification_baseline_payload,
    compare_qualification_baselines,
    compute_all_module_qualification,
    format_qualification_drift_lines,
    load_qualification_baseline_file,
    validate_qualification_baseline_payload,
    write_all_module_qualification_report,
    write_qualification_baseline_file,
)
from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED, normalize_seed


def _parse_fixture_pack(pack: str) -> Dict[str, str]:
    text = str(pack or "").strip()
    if not text:
        return {}

    if "/" in text or text.startswith("."):
        pack_root = Path(text).expanduser()
        if not pack_root.is_absolute():
            pack_root = (REPO_ROOT / pack_root).resolve()
    else:
        pack_root = (REPO_ROOT / "tests" / "fixtures" / "all_modules" / text).resolve()

    if not pack_root.exists() or not pack_root.is_dir():
        raise ValueError(f"Fixture pack directory not found: {pack_root}")

    output_map: Dict[str, str] = {}
    for module in ALL_MODULES:
        if module == "ui_ops":
            output_map[module] = str(REPO_ROOT)
            continue
        module_dir = pack_root / module
        if not module_dir.exists() or not module_dir.is_dir():
            raise ValueError(f"Fixture pack missing module directory: {module_dir}")
        output_map[module] = str(module_dir)
    return output_map


def _selected_modules(values: Iterable[str]) -> List[str]:
    selected: List[str] = []
    for module in values:
        key = str(module or "").strip().lower()
        if not key:
            continue
        if key not in ALL_MODULES:
            raise ValueError(f"Unsupported module selection: {key}")
        if key not in selected:
            selected.append(key)
    if not selected:
        return list(ALL_MODULES)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all-module qualification orchestration checks")
    parser.add_argument(
        "--qualification-profile",
        default="contract-v1",
        choices=["contract-v1", "fixture-v1", "live-local"],
        help="Qualification profile (default: contract-v1)",
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
        help="Filter module(s) for qualification (repeatable)",
    )
    parser.add_argument(
        "--fixture-pack",
        default="",
        help="Fixture pack for fixture profile (e.g., v1 or tests/fixtures/all_modules/v1)",
    )
    parser.add_argument(
        "--report-file",
        default=str(DEFAULT_ALL_MODULE_QUALIFICATION_REPORT_FILE),
        help="Qualification report JSON path",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write qualification report JSON",
    )
    parser.add_argument(
        "--baseline-file",
        default=str(DEFAULT_ALL_MODULE_QUALIFICATION_BASELINE_FILE),
        help="Baseline JSON path for qualification drift checks",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write/overwrite qualification baseline snapshot",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare qualification output against baseline",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail non-zero on module fail status or hard drift",
    )
    args = parser.parse_args()

    try:
        selected = _selected_modules(args.module)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    seed = normalize_seed(args.seed)

    output_map: Dict[str, str] = {}
    if args.qualification_profile == "fixture-v1":
        fixture_pack = args.fixture_pack or "v1"
        try:
            output_map = _parse_fixture_pack(fixture_pack)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 2
    elif args.fixture_pack:
        try:
            output_map = _parse_fixture_pack(args.fixture_pack)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 2

    try:
        report = compute_all_module_qualification(
            output_map=output_map or None,
            seed=seed,
            profile=args.qualification_profile,
            source="script",
            module_filters=selected,
        )
    except Exception as exc:
        print(f"ERROR: failed to compute all-module qualification report: {exc}")
        return 2

    for module in selected:
        entry = report.modules[module]
        print(
            "ALL_QUAL "
            f"module={module} status={entry.status} "
            f"errors={len(entry.errors)} warnings={len(entry.warnings)}"
        )
        if entry.status not in ALL_MODULE_QUALIFICATION_STATUSES:
            print(f"ERROR: invalid status for module={module}: {entry.status}")
            return 2

    report_path = Path(args.report_file)
    if args.write_report or args.write_baseline or args.compare_baseline:
        try:
            write_all_module_qualification_report(report_path, report)
        except Exception as exc:
            print(f"ERROR: failed to write qualification report: {exc}")
            return 2
        print(f"Wrote all-module qualification report: {report_path}")

    current_baseline = build_qualification_baseline_payload(report)
    baseline_path = Path(args.baseline_file)

    if args.write_baseline:
        try:
            write_qualification_baseline_file(baseline_path, current_baseline)
        except Exception as exc:
            print(f"ERROR: failed to write baseline: {exc}")
            return 2
        print(f"Wrote all-module qualification baseline: {baseline_path}")

    hard_drifts = []
    if args.compare_baseline:
        if not baseline_path.exists():
            print(f"ERROR: baseline file not found: {baseline_path}")
            return 2
        try:
            expected = load_qualification_baseline_file(baseline_path)
        except Exception as exc:
            print(f"ERROR: failed to load baseline: {exc}")
            return 2

        schema_errors = validate_qualification_baseline_payload(expected)
        if schema_errors:
            print("ERROR: invalid baseline schema")
            for error in schema_errors:
                print(f"  - {error}")
            return 2

        drifts = compare_qualification_baselines(expected=expected, current=current_baseline)
        for line in format_qualification_drift_lines(drifts):
            print(line)
        hard_drifts = [drift for drift in drifts if drift.get("severity") == "hard"]

    if hard_drifts:
        print("ERROR: hard qualification drift detected")
        return 1

    if args.strict:
        failing = [module for module in selected if report.modules[module].status == "fail"]
        if failing:
            print("ERROR: failing modules: " + ", ".join(failing))
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
