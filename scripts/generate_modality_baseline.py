#!/usr/bin/env python3
"""
Generate or validate deterministic modality runtime baseline snapshots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from halo_forge.cli import TestRunner
from halo_forge.modality_baseline import (
    DEFAULT_MODALITY_BASELINE_FILE,
    build_baseline_payload,
    compare_baseline_payloads,
    compute_fixture_pack_fingerprint,
    format_drift_lines,
    load_baseline_file,
    validate_baseline_payload,
    write_baseline_file,
)
from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED


def _has_torch_runtime() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _write_mode(runner: TestRunner, baseline_path: Path) -> int:
    if not _has_torch_runtime():
        print("ERROR: torch runtime is required for --write baseline generation")
        return 1

    runner.test_modality_fixtures()
    modality_entries = runner.test_modality_train_smoke()
    payload = build_baseline_payload(
        modality_entries=modality_entries,
        seed=DEFAULT_TRAINING_SEED,
    )
    write_baseline_file(baseline_path, payload)
    print(f"Wrote modality baseline: {baseline_path}")
    print(f"fixture_pack={payload['fixture_pack']}")
    return 0


def _check_mode(runner: TestRunner, baseline_path: Path) -> int:
    if not baseline_path.exists():
        print(f"ERROR: baseline file not found: {baseline_path}")
        return 1

    runner.test_modality_fixtures()
    expected = load_baseline_file(baseline_path)

    expected_errors = validate_baseline_payload(expected)
    if expected_errors:
        print("ERROR: baseline schema invalid")
        for error in expected_errors:
            print(f"  - {error}")
        return 1

    if not _has_torch_runtime():
        current_fingerprint = compute_fixture_pack_fingerprint()
        if str(expected.get("fixture_pack", "")) != current_fingerprint:
            print(
                "ERROR: fixture fingerprint mismatch without torch runtime "
                f"expected={expected.get('fixture_pack')} actual={current_fingerprint}"
            )
            return 1
        print("PASS: torch runtime unavailable; validated baseline schema + fixture fingerprint")
        return 0

    modality_entries = runner.test_modality_train_smoke()
    current = build_baseline_payload(
        modality_entries=modality_entries,
        seed=DEFAULT_TRAINING_SEED,
    )
    drifts = compare_baseline_payloads(expected, current)
    if drifts:
        print("ERROR: modality baseline drift detected")
        for line in format_drift_lines(drifts):
            print(line)
        return 1
    print("PASS: modality baseline matches current deterministic runtime contract")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate/check modality baseline snapshots")
    parser.add_argument(
        "--baseline-file",
        default=str(DEFAULT_MODALITY_BASELINE_FILE),
        help="Path to modality baseline JSON file",
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--write", action="store_true", help="Write/overwrite baseline snapshot")
    mode_group.add_argument("--check", action="store_true", help="Check current runtime against baseline")
    parser.add_argument("--verbose", action="store_true", help="Verbose runner output")
    args = parser.parse_args()

    baseline_path = Path(args.baseline_file)
    runner = TestRunner(verbose=args.verbose)

    try:
        if args.write:
            return _write_mode(runner, baseline_path)
        return _check_mode(runner, baseline_path)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
