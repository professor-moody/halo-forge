#!/usr/bin/env python3
"""Smoke-test the packaged desktop runtime with a tiny SFT run.

This intentionally exercises the PyInstaller runtime directly from the built
app resources. It does not require the repo-local .venv once the app/runtime
bundle has been built.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
APP_BUNDLE = (
    REPO_ROOT
    / "apps/desktop-tauri/src-tauri/target/release/bundle/macos/Halo Forge.app"
)
APP_RUNTIME = (
    APP_BUNDLE
    / "Contents/Resources/runtime/halo-forge-runtime/halo-forge-runtime"
)
APP_FRONTEND = APP_BUNDLE / "Contents/Resources/frontend"
DIST_RUNTIME = (
    REPO_ROOT
    / "apps/desktop-tauri/runtime/dist/halo-forge-runtime"
    / ("halo-forge-runtime.exe" if sys.platform == "win32" else "halo-forge-runtime")
)
DIST_FRONTEND = REPO_ROOT / "public_app/dist"
ROUTES = (
    "/",
    "/start",
    "/runs",
    "/results",
    "/diagnostics",
    "/models",
    "/playground",
    "/connect",
)
ACCELERATOR_CHOICES = (
    "auto",
    "rocm",
    "rocm_gfx1151",
    "cuda",
    "mps",
    "mlx",
    "cpu",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, help="Path to halo-forge-runtime executable.")
    parser.add_argument("--frontend", type=Path, help="Path to built public_app frontend directory.")
    parser.add_argument("--port", type=int, default=8765, help="Dashboard smoke port.")
    parser.add_argument("--keep-workdir", action="store_true", help="Keep temporary smoke artifacts.")
    parser.add_argument(
        "--model",
        default="hf-internal-testing/tiny-random-gpt2",
        help="Tiny open model used for packaged SFT smoke.",
    )
    parser.add_argument(
        "--accelerator",
        choices=ACCELERATOR_CHOICES,
        default="auto",
        help="Accelerator passed to the packaged Halo Forge CLI (default: auto).",
    )
    return parser.parse_args()


def resolve_runtime(explicit: Path | None) -> Path:
    candidates = [explicit, APP_RUNTIME, DIST_RUNTIME]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise SystemExit(
        "No packaged runtime found. Build it with `cd apps/desktop-tauri && npm run build:runtime`."
    )


def resolve_frontend(explicit: Path | None) -> Path:
    candidates = [explicit, APP_FRONTEND, DIST_FRONTEND]
    for candidate in candidates:
        if candidate and (candidate / "index.html").exists():
            return candidate
    raise SystemExit("No built frontend found. Build it with `cd public_app && npm run build`.")


def assert_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(f"Port {port} is already in use. Stop that service or pass --port.")


def run_checked(cmd: list[str], env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a command, surfacing its captured output when it fails.

    This used to pass ``check=True`` and let ``CalledProcessError`` propagate.
    The top-level handler prints ``str(exc)``, which for that exception is only
    "Command [...] returned non-zero exit status 1" -- the stdout/stderr this
    function captured was discarded, so a red packaged-runtime gate said that
    training failed but never why. This is the only CI step that executes a real
    optimizer step, so it is the step whose failures most need to be readable.
    """
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        captured = (result.stdout or "").strip()
        # Tail-limit: a failing trainer can emit a lot, and the end is the part
        # that carries the traceback.
        if len(captured) > 8000:
            captured = "...(truncated)...\n" + captured[-8000:]
        raise RuntimeError(
            f"command failed (exit {result.returncode}): {' '.join(cmd)}\n"
            f"--- captured output (stdout+stderr) ---\n{captured or '(no output)'}"
        )
    return result


def fetch(url: str, timeout: float = 2.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def log_tail(path: Path, limit: int = 8000) -> str:
    try:
        captured = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        return f"(could not read {path}: {exc})"
    if len(captured) > limit:
        captured = "...(truncated)...\n" + captured[-limit:]
    return captured or "(no output)"


def wait_for_health(
    port: int,
    process: subprocess.Popen[str],
    log_path: Path | None = None,
) -> dict[str, object]:
    deadline = time.time() + 45
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            detail = f"Dashboard exited early with code {process.returncode}"
            if log_path is not None:
                detail += f"\n--- dashboard output ---\n{log_tail(log_path)}"
            raise RuntimeError(detail)
        try:
            payload = json.loads(fetch(f"http://127.0.0.1:{port}/api/public/health"))
            if payload.get("ok") is True:
                return payload
        except Exception as exc:  # pragma: no cover - message is diagnostic only
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"Dashboard health timed out: {last_error}")


def write_tiny_dataset(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps({"text": "### Instruction\nSay hello.\n\n### Response\nHello."}),
                json.dumps({"text": "### Instruction\nAdd one and one.\n\n### Response\n2."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_sft_smoke(
    runtime: Path,
    frontend: Path,
    workdir: Path,
    model: str,
    accelerator: str,
) -> dict[str, object]:
    data_dir = workdir / "data"
    output_dir = workdir / "sft-output"
    log_dir = workdir / "logs"
    app_dir = workdir / "app"
    data_dir.mkdir(parents=True)
    train_file = data_dir / "train.jsonl"
    write_tiny_dataset(train_file)

    env = os.environ.copy()
    env.update(
        {
            "HALO_FORGE_FRONTEND_DIST": str(frontend),
            "HALO_FORGE_APP_DIR": str(app_dir),
            "HALO_FORGE_LOG_DIR": str(log_dir),
        }
    )
    run_checked([str(runtime), "--desktop-self-check"], env=env, cwd=REPO_ROOT)
    cli_command = [str(runtime), "-m", "halo_forge.cli"]
    if accelerator != "auto":
        cli_command.extend(["--accelerator", accelerator])
    cli_command.extend(
        [
            "sft",
            "train",
            "--model",
            model,
            "--data",
            str(train_file),
            "--output",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--gradient-accumulation",
            "1",
            "--learning-rate",
            "0.0002",
            "--max-seq-length",
            "64",
            "--validation-split",
            "0",
            "--save-steps",
            "999",
            "--eval-steps",
            "999",
            "--early-stopping-patience",
            "1",
            "--max-samples",
            "2",
            "--no-lora",
            "--no-caffeinate",
        ]
    )
    result = run_checked(
        cli_command,
        env=env,
        cwd=REPO_ROOT,
    )
    summary_path = output_dir / "training_summary.json"
    final_model = output_dir / "final_model"
    if not summary_path.exists():
        raise RuntimeError(f"Missing training summary: {summary_path}")
    if not final_model.exists():
        raise RuntimeError(f"Missing final model directory: {final_model}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    steps = int(summary.get("total_train_steps_executed") or 0)
    if steps <= 0:
        raise RuntimeError(f"Expected optimizer steps > 0, got {steps}")
    return {
        "output_dir": str(output_dir),
        "final_model": str(final_model),
        "summary": str(summary_path),
        "steps": steps,
        "stdout_tail": result.stdout.splitlines()[-12:],
    }


def run_dashboard_route_smoke(runtime: Path, frontend: Path, workdir: Path, port: int) -> dict[str, object]:
    env = os.environ.copy()
    env.update(
        {
            "HALO_FORGE_FRONTEND_DIST": str(frontend),
            "HALO_FORGE_APP_DIR": str(workdir / "dashboard-app"),
            "HALO_FORGE_LOG_DIR": str(workdir / "dashboard-logs"),
        }
    )
    log_path = workdir / "dashboard.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [
                str(runtime),
                "-m",
                "halo_forge.cli",
                "dashboard",
                "--no-build",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            health = wait_for_health(port, proc, log_path)
            version = json.loads(fetch(f"http://127.0.0.1:{port}/api/public/version"))
            missing_routes: list[str] = []
            for route in ROUTES:
                html = fetch(f"http://127.0.0.1:{port}{route}")
                if "<!doctype html>" not in html.lower():
                    missing_routes.append(route)
            if missing_routes:
                raise RuntimeError(f"Routes did not return dashboard HTML: {missing_routes}")
            return {"health": health, "version": version, "routes": list(ROUTES), "log": str(log_path)}
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def main() -> int:
    args = parse_args()
    runtime = resolve_runtime(args.runtime)
    frontend = resolve_frontend(args.frontend)
    assert_port_free(args.port)

    workdir = Path(tempfile.mkdtemp(prefix="halo-forge-packaged-smoke-"))
    try:
        sft = run_sft_smoke(runtime, frontend, workdir, args.model, args.accelerator)
        dashboard = run_dashboard_route_smoke(runtime, frontend, workdir, args.port)
        payload = {
            "ok": True,
            "runtime": str(runtime),
            "frontend": str(frontend),
            "sft": sft,
            "dashboard": dashboard,
        }
        print(json.dumps(payload, indent=2))
        if args.keep_workdir:
            keep = REPO_ROOT / ".tmp-packaged-sft-smoke"
            if keep.exists():
                raise RuntimeError(f"Refusing to overwrite existing {keep}")
            workdir.rename(keep)
            print(f"Kept smoke artifacts at {keep}")
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "workdir": str(workdir)}, indent=2))
        return 1
    finally:
        if workdir.exists() and not args.keep_workdir:
            shutil.rmtree(workdir)


if __name__ == "__main__":
    raise SystemExit(main())
