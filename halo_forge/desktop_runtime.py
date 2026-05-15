"""Desktop runtime entrypoint for the Halo Forge Tauri app.

This module is intentionally thin. The Tauri shell owns process lifecycle;
this wrapper gives PyInstaller a stable entrypoint and gives the shell a
fast self-check before starting the dashboard service.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


def _check_import(name: str) -> tuple[bool, str | None]:
    try:
        __import__(name)
    except Exception as exc:
        return False, str(exc)
    return True, None


def _self_check() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    if not ((3, 10) <= sys.version_info < (3, 14)):
        errors.append(f"Python >=3.10,<3.14 required; found {sys.version.split()[0]}")

    frontend_dist = os.environ.get("HALO_FORGE_FRONTEND_DIST")
    if not frontend_dist:
        errors.append("HALO_FORGE_FRONTEND_DIST is not set")
    elif not (Path(frontend_dist) / "index.html").is_file():
        errors.append(f"Dashboard index.html not found under {frontend_dist}")

    for module_name in ("fastapi", "uvicorn", "halo_forge.cli", "halo_forge.public_api.app"):
        ok, detail = _check_import(module_name)
        if not ok:
            errors.append(f"Could not import {module_name}: {detail}")

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        for module_name in ("mlx", "mlx_lm", "mlx.nn"):
            ok, detail = _check_import(module_name)
            if not ok:
                errors.append(f"MLX support import failed for {module_name}: {detail}")

    payload: dict[str, Any] = {
        "ok": not errors,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "frontend_dist": frontend_dist,
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if not errors else 78


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--desktop-self-check", action="store_true")
    known, remaining = parser.parse_known_args(argv)

    if known.desktop_self_check:
        return _self_check()

    if remaining[:2] == ["-m", "halo_forge.cli"]:
        remaining = remaining[2:]

    from halo_forge.cli import main as cli_main

    sys.argv = [sys.argv[0], *remaining]
    cli_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
