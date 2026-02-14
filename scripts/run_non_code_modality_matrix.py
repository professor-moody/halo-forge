#!/usr/bin/env python3
"""
Build and validate non-code modality research/testing matrix artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from halo_forge.modality_research import (
    build_matrix_markdown,
    build_non_code_modality_matrix,
    build_validation_report_markdown,
    matrix_as_json_serializable,
    parse_validation_targets,
    validate_modality_training_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Non-code modality research/testing matrix helper",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed used for matrix generation",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=2,
        help="Default positive-case training cycles",
    )
    parser.add_argument(
        "--train-output-root",
        default="models/phase7d",
        help="Base directory for generated training output paths in matrix",
    )
    parser.add_argument(
        "--benchmark-output-root",
        default="results/phase7d",
        help="Base directory for generated benchmark output paths in matrix",
    )
    parser.add_argument(
        "--print-matrix",
        action="store_true",
        help="Print matrix scenarios (defaults to enabled when no other action is requested)",
    )
    parser.add_argument(
        "--matrix-format",
        choices=("json", "markdown"),
        default="markdown",
        help="Matrix output format",
    )
    parser.add_argument(
        "--validate-training",
        action="append",
        default=[],
        help="Validate completed modality output directory (format: modality=/path/to/output)",
    )
    parser.add_argument(
        "--report-file",
        default="",
        help="Optional markdown report output path for validation results",
    )
    args = parser.parse_args()

    matrix = build_non_code_modality_matrix(
        seed=args.seed,
        train_output_root=args.train_output_root,
        benchmark_output_root=args.benchmark_output_root,
        cycles=args.cycles,
    )

    should_print_matrix = args.print_matrix or not args.validate_training
    if should_print_matrix:
        if args.matrix_format == "json":
            print(json.dumps(matrix_as_json_serializable(matrix), indent=2, sort_keys=True))
        else:
            print(build_matrix_markdown(matrix))

    try:
        targets = parse_validation_targets(args.validate_training)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    if not targets:
        return 0

    results = []
    for modality, output_dir in targets:
        result = validate_modality_training_artifacts(
            modality=modality,
            output_dir=output_dir,
            expected_seed=args.seed,
        )
        results.append(result)

    report = build_validation_report_markdown(results)
    if args.report_file:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"Wrote validation report: {report_path}")
    else:
        print(report)

    failures = [r for r in results if not r.ok]
    if failures:
        print(f"ERROR: validation failed for {len(failures)} modality output(s)")
        return 1

    print("PASS: all modality output validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
