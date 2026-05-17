"""Managed local `halo-forge serve` process for the dashboard."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from halo_forge.huggingface_access import inject_huggingface_token


@dataclass
class ServeStartRequest:
    model: str
    backend: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = 8001
    trust_remote_code: bool = False


class ManagedServeProcess:
    """Own one local OpenAI-compatible serve process."""

    STARTING_GRACE_SECONDS = 20.0

    def __init__(self, *, base_path: Path | None = None, log_dir: Path | None = None) -> None:
        self.base_path = (base_path or Path.cwd()).resolve()
        self.log_dir = (log_dir or Path.home() / ".halo-forge" / "serve").expanduser()
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_handle: Any | None = None
        self._model: str | None = None
        self._backend: str | None = None
        self._host = "127.0.0.1"
        self._port = 8001
        self._started_at: float | None = None
        self._log_path: Path | None = None
        self._last_error: str | None = None

    def status(self) -> dict[str, Any]:
        proc = self._proc
        running = proc is not None and proc.poll() is None
        exit_code = None if proc is None else proc.poll()
        health_payload = self._health_payload() if running else None
        load_error = (
            health_payload.get("load_error")
            if isinstance(health_payload, dict) and isinstance(health_payload.get("load_error"), dict)
            else None
        )
        healthy = bool(health_payload and health_payload.get("ok"))
        model_ready = bool(health_payload and health_payload.get("adapter_loaded"))
        state = self._state(
            running=running,
            healthy=healthy,
            exit_code=exit_code,
            load_error=load_error,
        )
        return {
            "running": running,
            "state": state,
            "active_action": self._active_action(state),
            "pid": proc.pid if proc is not None and running else None,
            "model": self._model,
            "backend": self._backend,
            "host": self._host,
            "port": self._port,
            "url": self.base_url,
            "started_at": self._started_at,
            "exit_code": exit_code,
            "log_path": str(self._log_path) if self._log_path else None,
            "logs_available": bool(self._log_path and self._log_path.exists()),
            "last_error": self._last_error,
            "error_hint": self._error_hint(state=state, exit_code=exit_code, load_error=load_error),
            "healthy": healthy,
            "model_ready": model_ready,
            "adapter_loaded": model_ready,
            "load_error": load_error,
            "load_error_kind": load_error.get("error_kind") if load_error else None,
            "load_error_message": load_error.get("message") if load_error else None,
            "message": self._message(state=state, exit_code=exit_code, model_ready=model_ready, load_error=load_error),
        }

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}/v1"

    def start(self, request: ServeStartRequest) -> dict[str, Any]:
        current = self.status()
        if current["running"]:
            self._last_error = "A local model is already being served."
            raise ValueError("A local model is already being served. Stop it before starting another.")
        model = str(request.model or "").strip()
        if not model:
            raise ValueError("model is required")
        if not self._port_available(request.host, int(request.port)):
            self._last_error = f"{request.host}:{int(request.port)} is already in use."
            raise ValueError(
                f"{request.host}:{int(request.port)} is already in use. "
                "Stop the process on that port or choose another local serve port."
            )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self._log_path = self.log_dir / f"serve_{stamp}.log"
        self._log_handle = self._log_path.open("ab")
        cmd = [
            sys.executable,
            "-m",
            "halo_forge.cli",
            "serve",
            "--model",
            model,
            "--host",
            request.host,
            "--port",
            str(int(request.port)),
        ]
        if request.backend:
            cmd.extend(["--backend", str(request.backend)])
        if request.trust_remote_code:
            cmd.append("--trust-remote-code")

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        inject_huggingface_token(env)
        self._proc = subprocess.Popen(
            cmd,
            cwd=self.base_path,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        self._model = model
        self._backend = request.backend
        self._host = request.host
        self._port = int(request.port)
        self._started_at = time.time()
        self._last_error = None
        return self.status()

    def stop(self, *, timeout_s: float = 8.0) -> dict[str, Any]:
        proc = self._proc
        if proc is None:
            return self.status()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        self._close_log_handle()
        self._proc = None
        self._started_at = None
        self._last_error = None
        return self.status()

    def logs(self, *, tail: int = 200) -> dict[str, Any]:
        path = self._log_path
        if path is None or not path.exists():
            return {"available": False, "lines": [], "path": str(path) if path else None}
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as exc:
            return {"available": False, "lines": [], "path": str(path), "reason": str(exc)}
        return {"available": True, "lines": lines[-max(1, int(tail)):], "path": str(path)}

    def health(self) -> dict[str, Any]:
        healthy = self.is_healthy()
        return {**self.status(), "healthy": healthy}

    def is_healthy(self) -> bool:
        payload = self._health_payload()
        return bool(payload and payload.get("ok"))

    def _health_payload(self) -> dict[str, Any] | None:
        if self._proc is None or self._proc.poll() is not None:
            return None
        try:
            with urllib.request.urlopen(f"http://{self._host}:{self._port}/health", timeout=0.75) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                if not (200 <= int(resp.status) < 300):
                    return None
                payload = json.loads(raw)
                return payload if isinstance(payload, dict) else None
        except (OSError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            return None

    def _state(
        self,
        *,
        running: bool,
        healthy: bool,
        exit_code: int | None,
        load_error: dict[str, Any] | None = None,
    ) -> str:
        if self._proc is None:
            return "idle"
        if not running:
            return "exited"
        if load_error:
            return "unhealthy"
        if healthy:
            return "running"
        if self._started_at and time.time() - self._started_at <= self.STARTING_GRACE_SECONDS:
            return "starting"
        return "unhealthy"

    def _message(
        self,
        *,
        state: str,
        exit_code: int | None,
        model_ready: bool = False,
        load_error: dict[str, Any] | None = None,
    ) -> str:
        if state == "idle":
            return "No local model is being served."
        if state == "starting":
            return "Starting local model server. This can take a minute while the model loads."
        if state == "running":
            if model_ready:
                return "Local model is loaded and ready to chat."
            return "Local server is ready. The model will load on the first message."
        if state == "unhealthy":
            if load_error and load_error.get("message"):
                return str(load_error["message"])
            return "Local model server is running but health checks are failing."
        if state == "exited":
            return f"Local model server exited with code {exit_code}."
        return "Local serving status is unknown."

    def _active_action(self, state: str) -> str | None:
        if state == "starting":
            return "loading_model"
        if state == "running":
            return "serving"
        if state == "unhealthy":
            return "check_logs"
        if state == "exited":
            return "review_logs"
        return None

    def _error_hint(
        self,
        *,
        state: str,
        exit_code: int | None,
        load_error: dict[str, Any] | None = None,
    ) -> str | None:
        if state == "idle":
            return None
        if self._last_error:
            return self._last_error
        if load_error and load_error.get("hint"):
            return str(load_error["hint"])
        if state == "unhealthy":
            return "The process is still running, but the model did not load. Review the serving log."
        if state == "exited":
            return f"The local serve process exited with code {exit_code}. Review the serving log before retrying."
        return None

    def _port_available(self, host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex((host, port)) != 0

    def _close_log_handle(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            finally:
                self._log_handle = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.stop(timeout_s=1.0)
        except Exception:
            pass
