#!/usr/bin/env python3
"""Run Halo Forge release-confidence checks.

The default path is CI-safe: it validates tracked source state, serving, MLX
readiness contracts, and frontend build health when dependencies are already
installed. ``--include-live-mlx`` validates an existing generated smoke summary
without creating or committing run artifacts.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Check:
    label: str
    command: list[str]
    cwd: Path = REPO_ROOT
    required: bool = True


def _run(check: Check) -> bool:
    print(f"\n==> {check.label}")
    print("$ " + " ".join(check.command))
    proc = subprocess.run(check.command, cwd=check.cwd, check=False)
    if proc.returncode == 0:
        print(f"PASS {check.label}")
        return True
    status = "FAIL" if check.required else "WARN"
    print(f"{status} {check.label} exited {proc.returncode}")
    return not check.required


def _frontend_checks(skip_frontend: bool) -> list[Check]:
    if skip_frontend:
        return []
    public_app = REPO_ROOT / "public_app"
    if not (public_app / "package.json").exists():
        print("\nSKIP frontend checks: public_app/package.json not found")
        return []
    if not (public_app / "node_modules").exists():
        print("\nSKIP frontend checks: public_app/node_modules not installed")
        return []
    contract_tests = sorted(
        path.relative_to(public_app).as_posix()
        for path in (public_app / "scripts").glob("*contract.test.mjs")
    )
    return [
        Check("frontend typecheck", ["npm", "run", "lint"], cwd=public_app),
        *(
            [
                Check(
                    "frontend contract tests",
                    ["node", "--test", *contract_tests],
                    cwd=public_app,
                )
            ]
            if contract_tests
            else []
        ),
        Check("frontend build", ["npm", "run", "build"], cwd=public_app),
    ]


def build_checks(args: argparse.Namespace) -> list[Check]:
    py = sys.executable
    ruff = shutil.which("ruff") or "ruff"
    checks = [
        Check("git diff whitespace check", ["git", "diff", "--check"]),
        Check("frozen dependency lock", [py, "scripts/check_dependency_contract.py"]),
        Check(
            "static undefined-name and syntax gate",
            [ruff, "check", "--select", "E9,F63,F7,F82", "."],
        ),
        Check(
            "syntax compile",
            [py, "-m", "compileall", "-q", "halo_forge", "ui", "tests", "scripts"],
        ),
        Check(
            "release interface, docs, and scenario contracts",
            [py, "scripts/check_release_interfaces.py"],
        ),
        Check(
            "modality baseline",
            [py, "scripts/generate_modality_baseline.py", "--check"],
        ),
        Check(
            "strict operational module matrix",
            [
                py,
                "scripts/run_ops_module_matrix.py",
                "--fixture-pack",
                "v1",
                "--strict",
            ],
        ),
        Check(
            "full Python test suite",
            [py, "-m", "pytest", "tests", "-q"],
        ),
    ]
    checks.extend(_frontend_checks(args.skip_frontend))
    if args.include_live_mlx:
        checks.append(
            Check(
                "existing live MLX smoke summary",
                [
                    py,
                    "scripts/validate_mlx_smoke_summary.py",
                    str(args.mlx_smoke_summary),
                ],
            )
        )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-live-mlx",
        action="store_true",
        help="Validate an existing runs/mlx-smoke/mlx_smoke_summary.json file.",
    )
    parser.add_argument(
        "--mlx-smoke-summary",
        type=Path,
        default=Path("runs/mlx-smoke/mlx_smoke_summary.json"),
        help="Existing MLX smoke summary to validate when --include-live-mlx is set.",
    )
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="Skip frontend lint/build even if public_app/node_modules exists.",
    )
    args = parser.parse_args()

    if args.include_live_mlx and not args.mlx_smoke_summary.exists():
        print(f"Missing MLX smoke summary: {args.mlx_smoke_summary}", file=sys.stderr)
        return 2

    ok = True
    for check in build_checks(args):
        ok = _run(check) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
