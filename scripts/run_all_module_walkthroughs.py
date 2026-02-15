#!/usr/bin/env python3
"""Run all-module walkthrough contract checks and emit a canonical report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from halo_forge.all_module_readiness import ALL_MODULES
from halo_forge.all_module_walkthroughs import (
    DEFAULT_WALKTHROUGH_REPORT_FILE,
    WALKTHROUGH_PROFILES,
    WalkthroughReportV1,
    compute_walkthroughs,
    normalized_walkthrough_payload,
    write_walkthrough_report,
)
from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED, normalize_seed


def _selected_modules(values: List[str]) -> List[str]:
    selected: List[str] = []
    for value in values:
        key = str(value or "").strip().lower()
        if not key:
            continue
        if key not in ALL_MODULES:
            raise ValueError(f"Unsupported module selection: {key}")
        if key not in selected:
            selected.append(key)
    if selected:
        return selected
    return list(ALL_MODULES)


def _print_lines(report: WalkthroughReportV1, modules: List[str]) -> None:
    for module in modules:
        entry = report.modules[module]
        print(
            "WALKTHROUGH "
            f"module={module} status={entry.status} "
            f"steps={len(entry.steps)} errors={len(entry.errors)} warnings={len(entry.warnings)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all-module walkthrough checks")
    parser.add_argument(
        "--profile",
        choices=list(WALKTHROUGH_PROFILES),
        default="contract-v1",
        help="Walkthrough profile: contract-v1 (default) or live-local",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAINING_SEED,
        help="Deterministic seed used for command templates (default: 42)",
    )
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        help="Filter module(s) for walkthrough execution (repeatable)",
    )
    parser.add_argument(
        "--report-file",
        default=str(DEFAULT_WALKTHROUGH_REPORT_FILE),
        help="Output report JSON path",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute bounded command probes (used with profile=live-local)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any selected module status is fail",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=30.0,
        help="Probe timeout in seconds for live-local execute mode",
    )
    parser.add_argument(
        "--print-normalized",
        action="store_true",
        help="Print normalized deterministic payload JSON",
    )
    args = parser.parse_args()

    seed = normalize_seed(args.seed)
    modules = _selected_modules(args.module)

    report = compute_walkthroughs(
        modules=modules,
        seed=seed,
        profile=args.profile,
        execute=args.execute,
        command_timeout_sec=args.timeout_sec,
    )

    _print_lines(report, modules)

    report_path = Path(args.report_file)
    write_walkthrough_report(report_path, report)
    print(f"WALKTHROUGH_REPORT file={report_path}")

    if args.print_normalized:
        print(json.dumps(normalized_walkthrough_payload(report), indent=2, sort_keys=True))

    if args.strict:
        failing = [module for module in modules if report.modules[module].status == "fail"]
        if failing:
            print("ERROR: failing modules: " + ", ".join(failing))
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
