#!/usr/bin/env python3
"""
Run bounded dataset-backed burn-in checks for non-code operational modules.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from halo_forge.ops_dataset_burnin import (
    DEFAULT_BURNIN_PROFILE,
    DEFAULT_OPS_BURNIN_BASELINE_FILE,
    DEFAULT_OPS_BURNIN_REPORT_FILE,
    OPS_BURNIN_STATUSES,
    OPS_MODULES,
    build_burnin_baseline_payload,
    compare_burnin_baselines,
    compute_ops_dataset_burnin,
    format_burnin_drift_lines,
    load_burnin_baseline_file,
    validate_burnin_baseline_payload,
    write_burnin_baseline_file,
    write_ops_burnin_report,
)
from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED, normalize_seed


def _print_report_lines(report) -> None:
    for module in OPS_MODULES:
        entry = report.modules[module]
        print(
            "OPS_BURNIN "
            f"module={module} status={entry.status} "
            f"errors={len(entry.errors)} warnings={len(entry.warnings)}"
        )


def _fail_modules(report) -> List[str]:
    return [module for module in OPS_MODULES if report.modules[module].status == "fail"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ops dataset burn-in and baseline drift checks")
    parser.add_argument(
        "--burnin-profile",
        default=DEFAULT_BURNIN_PROFILE,
        help="Burn-in profile to run (default: tiny-v1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAINING_SEED,
        help="Deterministic seed for burn-in profile execution",
    )
    parser.add_argument(
        "--report-file",
        default=str(DEFAULT_OPS_BURNIN_REPORT_FILE),
        help="Report JSON output path",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write burn-in report JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when modules have status=fail (warn does not fail)",
    )
    parser.add_argument(
        "--execute-commands",
        action="store_true",
        help="Attempt bounded command execution for profile scenarios",
    )
    parser.add_argument(
        "--command-timeout-sec",
        type=float,
        default=45.0,
        help="Per-command timeout when --execute-commands is set",
    )
    parser.add_argument(
        "--fixture-pack",
        default="v1",
        help="Fixture pack root name/path for deterministic output contract checks",
    )
    parser.add_argument(
        "--baseline-file",
        default=str(DEFAULT_OPS_BURNIN_BASELINE_FILE),
        help="Baseline JSON path for burn-in drift checks",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write/overwrite baseline from current burn-in report",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare current burn-in report against baseline",
    )
    args = parser.parse_args()

    seed = normalize_seed(args.seed)
    try:
        report = compute_ops_dataset_burnin(
            profile=args.burnin_profile,
            seed=seed,
            source="script",
            execute_commands=bool(args.execute_commands),
            command_timeout_sec=float(args.command_timeout_sec),
            fixture_pack=args.fixture_pack,
        )
    except Exception as exc:
        print(f"ERROR: failed to compute burn-in report: {exc}")
        return 2

    _print_report_lines(report)
    for module in OPS_MODULES:
        if report.modules[module].status not in OPS_BURNIN_STATUSES:
            print(f"ERROR: invalid status for module={module}: {report.modules[module].status}")
            return 2

    report_path = Path(args.report_file)
    if args.write_report:
        try:
            write_ops_burnin_report(report_path, report)
        except Exception as exc:
            print(f"ERROR: failed to write report: {exc}")
            return 2
        print(f"Wrote ops dataset burn-in report: {report_path}")

    baseline_path = Path(args.baseline_file)
    current_baseline = build_burnin_baseline_payload(report)

    if args.write_baseline:
        try:
            write_burnin_baseline_file(baseline_path, current_baseline)
        except Exception as exc:
            print(f"ERROR: failed to write baseline: {exc}")
            return 2
        print(f"Wrote ops dataset burn-in baseline: {baseline_path}")

    hard_drifts = []
    warn_drifts = []
    if args.compare_baseline:
        if not baseline_path.exists():
            print(f"ERROR: baseline file not found: {baseline_path}")
            return 2
        try:
            expected = load_burnin_baseline_file(baseline_path)
        except Exception as exc:
            print(f"ERROR: failed to load baseline: {exc}")
            return 2

        schema_errors = validate_burnin_baseline_payload(expected)
        if schema_errors:
            print("ERROR: invalid baseline schema")
            for error in schema_errors:
                print(f"  - {error}")
            return 2

        drifts = compare_burnin_baselines(expected=expected, current=current_baseline)
        for line in format_burnin_drift_lines(drifts):
            print(line)
        hard_drifts = [drift for drift in drifts if drift.get("severity") == "hard"]
        warn_drifts = [drift for drift in drifts if drift.get("severity") != "hard"]

    if hard_drifts:
        print("ERROR: hard burn-in contract drift detected")
        return 1

    if args.strict:
        failing = _fail_modules(report)
        if failing:
            print("ERROR: failing modules: " + ", ".join(failing))
            return 1

    if warn_drifts:
        print(f"WARN: burn-in drift detected ({len(warn_drifts)} warn drift(s))")

    return 0


if __name__ == "__main__":
    sys.exit(main())
