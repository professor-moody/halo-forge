"""Durable subprocess launcher for Artifact Studio serving profiles."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from halo_forge.huggingface_access import inject_huggingface_token
from halo_forge.workstation_jobs import process_start_time


class SubprocessServingStarter:
    """Start the existing OpenAI-compatible server with real process identity.

    The returned process is intentionally supervised by its durable PID/start
    identity rather than an in-memory ``Popen`` handle. This lets a restarted
    dashboard adopt and monitor the server without losing its workstation lease.
    """

    def __init__(
        self,
        catalog: Any,
        *,
        base_path: Path | str | None = None,
        log_dir: Path | str | None = None,
        python_executable: Optional[str] = None,
        popen: Callable[..., Any] = subprocess.Popen,
        process_identity: Callable[[int], Optional[float]] = process_start_time,
    ):
        self.catalog = catalog
        self.base_path = Path(base_path or Path.cwd()).expanduser().resolve()
        self.log_dir = Path(log_dir or (Path.home() / ".halo-forge" / "serve")).expanduser()
        self.python_executable = python_executable or sys.executable
        self.popen = popen
        self.process_identity = process_identity

    def _model_path(self, occurrence: Any) -> str:
        locations = [
            item
            for item in self.catalog.list_locations(occurrence.blob_id)
            if item.state == "available" and Path(item.path).expanduser().exists()
        ]
        location = next(
            (item for item in locations if item.storage_mode == "managed"),
            locations[0] if locations else None,
        )
        if location is None:
            raise ValueError(f"artifact {occurrence.id} has no available model location")
        return str(Path(location.path).expanduser().resolve())

    def __call__(
        self,
        profile: Mapping[str, Any],
        occurrence: Any,
        launch_spec: Mapping[str, Any],
    ) -> dict[str, Any]:
        endpoint = dict(profile.get("endpoint_settings") or {})
        host = str(endpoint.get("host") or "127.0.0.1")
        port = int(endpoint.get("port") or 8001)
        backend = str(profile.get("backend") or "local").strip().lower()
        model_path = self._model_path(occurrence)
        serving_id = str(launch_spec.get("serving_id") or profile.get("id") or "serve")

        self.log_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in serving_id)
        log_path = self.log_dir / f"{safe_id}-{int(time.time())}.log"
        command = [
            self.python_executable,
            "-m",
            "halo_forge.cli",
            "serve",
            "--model",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
        ]
        if backend not in {"", "auto", "local"}:
            command.extend(["--backend", backend])
        if bool(endpoint.get("trust_remote_code", False)):
            command.append("--trust-remote-code")

        environment = os.environ.copy()
        environment.setdefault("PYTHONUNBUFFERED", "1")
        inject_huggingface_token(environment)
        log_handle = log_path.open("ab")
        try:
            process = self.popen(
                command,
                cwd=self.base_path,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
        except Exception:
            log_handle.close()
            raise
        # The child owns the duplicated descriptor. Closing the parent's copy
        # avoids leaking one handle for every serving restart.
        log_handle.close()
        if process.poll() is not None:
            raise RuntimeError(f"serving process exited immediately with code {process.returncode}")
        started_at = self.process_identity(int(process.pid))
        if started_at is None:
            try:
                process.terminate()
            finally:
                raise RuntimeError("serving process start identity could not be resolved")
        return {
            "state": "running",
            "process_id": int(process.pid),
            "process_started_at": float(started_at),
            "model": model_path,
            "backend": backend,
            "host": host,
            "port": port,
            "url": f"http://{host}:{port}/v1",
            "log_path": str(log_path),
            "command": command,
        }


__all__ = ["SubprocessServingStarter"]
