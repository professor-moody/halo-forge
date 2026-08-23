#!/usr/bin/env python3
"""Verify that the frozen dependency exports match ``uv.lock`` exactly."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTS = (
    ("constraints/release.txt", ("dev",)),
    ("constraints/release-mlx.txt", ("dev", "mlx")),
)


def _run(command: list[str], *, env: dict[str, str]) -> None:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        print(result.stdout, end="", file=sys.stderr)
        raise SystemExit(result.returncode)


def main() -> int:
    uv = shutil.which("uv")
    if uv is None:
        print("uv is required to verify the frozen dependency contract", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="halo-forge-dependency-check-") as temp_dir:
        temp_root = Path(temp_dir)
        env = os.environ.copy()
        env["UV_CACHE_DIR"] = str(temp_root / "uv-cache")
        _run([uv, "lock", "--check"], env=env)

        mismatches: list[str] = []
        for relative_path, extras in EXPORTS:
            generated = temp_root / Path(relative_path).name
            command = [uv, "export"]
            for extra in extras:
                command.extend(("--extra", extra))
            command.extend(
                (
                    "--no-emit-project",
                    "--no-hashes",
                    "--no-annotate",
                    "--no-header",
                    "--frozen",
                    "--output-file",
                    str(generated),
                )
            )
            _run(command, env=env)
            committed = REPO_ROOT / relative_path
            if not committed.is_file() or committed.read_bytes() != generated.read_bytes():
                mismatches.append(relative_path)

        if mismatches:
            print(
                "Frozen dependency exports are stale: " + ", ".join(mismatches),
                file=sys.stderr,
            )
            print(
                "Regenerate them with the documented uv export commands in "
                "docs/RELEASE_CHECKLIST.md.",
                file=sys.stderr,
            )
            return 1

    print("PASS: uv.lock and frozen release constraints are synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
