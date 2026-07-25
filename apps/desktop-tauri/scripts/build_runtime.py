#!/usr/bin/env python3
"""Build the bundled Halo Forge desktop runtime with PyInstaller."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def venv_python(venv: Path) -> Path:
    if platform.system() == "Windows":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def compatible_python(executable: Path | str) -> bool:
    try:
        result = subprocess.run(
            [
                str(executable),
                "-c",
                "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info < (3, 14) else 1)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def select_seed_python(repo_root: Path) -> str:
    candidates: list[str] = []
    if configured := os.environ.get("HALO_FORGE_RUNTIME_PYTHON"):
        candidates.append(configured)
    candidates.append(str(repo_root / ".venv/bin/python"))
    candidates.append(sys.executable)
    for name in ("python3.13", "python3.12", "python3.11", "python3.10", "python3"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(resolved)

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if compatible_python(candidate):
            return candidate
    raise SystemExit(
        "No compatible Python found for desktop runtime build. "
        "Set HALO_FORGE_RUNTIME_PYTHON to Python >=3.10,<3.14."
    )


def runtime_executable(dist_dir: Path) -> Path:
    name = "halo-forge-runtime.exe" if platform.system() == "Windows" else "halo-forge-runtime"
    return dist_dir / "halo-forge-runtime" / name


def install_runtime_deps(py: Path, repo_root: Path, *, profile: str) -> None:
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools<82"], cwd=repo_root)
    run([str(py), "-m", "pip", "install", "pyinstaller>=6.0"], cwd=repo_root)

    editable = ".[mlx]" if profile == "macos-mlx" else "."
    run([str(py), "-m", "pip", "install", editable], cwd=repo_root)


def build_pyinstaller(py: Path, repo_root: Path, runtime_dir: Path) -> Path:
    dist_dir = runtime_dir / "dist"
    build_dir = runtime_dir / "build"
    entry = runtime_dir / "desktop_runtime_entry.py"
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    if build_dir.exists():
        shutil.rmtree(build_dir)

    cmd = [
        str(py),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        "halo-forge-runtime",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(build_dir),
        "--specpath",
        str(build_dir),
        "--collect-submodules",
        "halo_forge",
        "--collect-submodules",
        "ui",
        "--collect-submodules",
        "mlx_lm",
        "--collect-submodules",
        "keyring",
        "--hidden-import",
        "uvicorn.loops.auto",
        "--hidden-import",
        "uvicorn.protocols.http.auto",
        "--hidden-import",
        "uvicorn.protocols.websockets.auto",
        "--hidden-import",
        "mlx._reprlib_fix",
        *mlx_binary_args(py),
        str(entry),
    ]
    run(cmd, cwd=repo_root)
    return runtime_executable(dist_dir)


def mlx_binary_args(py: Path) -> list[str]:
    probe = (
        "from pathlib import Path\n"
        "import mlx.core as mx\n"
        "root = Path(mx.__file__).resolve().parent\n"
        "print(root / 'lib' / 'libjaccl.dylib')\n"
        "print(root / 'lib' / 'mlx.metallib')\n"
    )
    try:
        result = subprocess.run(
            [str(py), "-c", probe],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return []
    libjaccl, metallib = (result.stdout.splitlines() + ["", ""])[:2]
    separator = ";" if platform.system() == "Windows" else ":"
    args: list[str] = []
    if libjaccl and Path(libjaccl).is_file():
        args.extend(["--add-binary", f"{libjaccl}{separator}."])
    if metallib and Path(metallib).is_file():
        args.extend(["--add-data", f"{metallib}{separator}."])
        args.extend(["--add-data", f"{metallib}{separator}mlx/lib"])
    return args


def self_check(exe: Path, frontend_dist: Path) -> None:
    env = os.environ.copy()
    env["HALO_FORGE_FRONTEND_DIST"] = str(frontend_dist)
    run([str(exe), "--desktop-self-check"], cwd=exe.parent, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-venv", action="store_true", help="Do not recreate the runtime build venv")
    parser.add_argument(
        "--profile",
        choices=("auto", "macos-mlx", "linux-dashboard", "windows-dashboard"),
        default="auto",
        help="Runtime dependency profile",
    )
    args = parser.parse_args()

    runtime_dir = Path(__file__).resolve().parents[1] / "runtime"
    repo_root = Path(__file__).resolve().parents[3]
    frontend_dist = repo_root / "public_app" / "dist"
    if not (frontend_dist / "index.html").is_file():
        raise SystemExit("public_app/dist/index.html is missing; run `npm run build` in public_app first")

    profile = args.profile
    if profile == "auto":
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            profile = "macos-mlx"
        elif platform.system() == "Windows":
            profile = "windows-dashboard"
        else:
            profile = "linux-dashboard"

    venv = runtime_dir / ".venv-build"
    if venv.exists() and not args.reuse_venv:
        shutil.rmtree(venv)
    if not venv.exists():
        run([select_seed_python(repo_root), "-m", "venv", str(venv)], cwd=repo_root)

    py = venv_python(venv)
    install_runtime_deps(py, repo_root, profile=profile)
    exe = build_pyinstaller(py, repo_root, runtime_dir)
    self_check(exe, frontend_dist)
    print(f"Built desktop runtime: {exe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
