from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

from halo_forge.huggingface_access import (
    HF_TOKEN_ENV,
    HuggingFaceAccessManager,
    _classify_model_error,
)


class FakeKeyring:
    def __init__(self) -> None:
        self.value: str | None = None

    def get_password(self, service: str, username: str) -> str | None:
        return self.value

    def set_password(self, service: str, username: str, password: str) -> None:
        self.value = password

    def delete_password(self, service: str, username: str) -> None:
        self.value = None


def test_huggingface_token_precedence_env_keyring_file(tmp_path):
    keyring = FakeKeyring()
    token_file = tmp_path / "hf_token"
    token_file.write_text("hf_file\n", encoding="utf-8")
    keyring.value = "hf_keyring"

    manager = HuggingFaceAccessManager(
        token_file=token_file,
        env={HF_TOKEN_ENV: "hf_env"},
        keyring_module=keyring,
    )
    assert manager.resolve(include_token=True).source == "env"
    assert manager.resolve(include_token=True).token == "hf_env"

    manager = HuggingFaceAccessManager(token_file=token_file, env={}, keyring_module=keyring)
    assert manager.resolve(include_token=True).source == "keyring"
    assert manager.resolve(include_token=True).token == "hf_keyring"

    keyring.value = None
    manager = HuggingFaceAccessManager(token_file=token_file, env={}, keyring_module=keyring)
    assert manager.resolve(include_token=True).source == "file"
    assert manager.resolve(include_token=True).token == "hf_file"


def test_huggingface_save_clear_file_fallback_is_sanitized(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "halo_forge.huggingface_access.verify_huggingface_token",
        lambda token: {"ok": True, "username": "tester", "message": "Connected as tester."},
    )
    token_file = tmp_path / "secrets" / "huggingface_token"
    manager = HuggingFaceAccessManager(token_file=token_file, env={}, keyring_module=None)

    status = manager.save("hf_secret")

    assert status["present"] is True
    assert status["source"] == "file"
    assert status["username"] == "tester"
    assert "hf_secret" not in str(status)
    assert token_file.read_text(encoding="utf-8").strip() == "hf_secret"
    if os.name != "nt":
        assert oct(token_file.stat().st_mode & 0o777) == "0o600"

    cleared = manager.clear()
    assert cleared["present"] is False
    assert not token_file.exists()
    assert "hf_secret" not in str(cleared)


def test_huggingface_clear_env_reports_external_credential(monkeypatch):
    monkeypatch.setattr(
        "halo_forge.huggingface_access.verify_huggingface_token",
        lambda token: {"ok": True, "username": "env-user", "message": "Connected."},
    )
    manager = HuggingFaceAccessManager(env={HF_TOKEN_ENV: "hf_env_secret"}, keyring_module=None)

    status = manager.clear()

    assert status["present"] is True
    assert status["source"] == "env"
    assert status["can_clear"] is False
    assert "cannot be cleared from the dashboard" in status["message"]
    assert "hf_env_secret" not in str(status)


def test_huggingface_model_error_classification():
    assert _classify_model_error("401 Client Error: Cannot access gated repo", has_token=False) == "auth_required"
    assert _classify_model_error("401 Client Error: Cannot access gated repo", has_token=True) == "gated"
    assert _classify_model_error("Repository Not Found for url", has_token=False) == "missing"
    assert _classify_model_error("Name resolution failed", has_token=False) == "network_error"


def test_managed_serve_injects_hf_token(monkeypatch, tmp_path):
    from halo_forge.public_api.serve_manager import ManagedServeProcess, ServeStartRequest

    captured: dict[str, object] = {}

    class FakePopen:
        pid = 12345

        def __init__(self, cmd, cwd=None, stdout=None, stderr=None, env=None):
            captured["cmd"] = cmd
            captured["env"] = env

        def poll(self):
            return None

    monkeypatch.setenv(HF_TOKEN_ENV, "hf_process_secret")
    monkeypatch.setattr("halo_forge.public_api.serve_manager.subprocess.Popen", FakePopen)
    monkeypatch.setattr(ManagedServeProcess, "_port_available", lambda self, host, port: True)
    monkeypatch.setattr(ManagedServeProcess, "_health_payload", lambda self: None)

    manager = ManagedServeProcess(base_path=tmp_path, log_dir=tmp_path / "logs")
    status = manager.start(ServeStartRequest(model="Qwen/Qwen2.5-0.5B-Instruct"))

    assert status["state"] == "starting"
    env = captured["env"]
    assert isinstance(env, dict)
    assert env[HF_TOKEN_ENV] == "hf_process_secret"


def test_training_subprocess_injects_hf_token(monkeypatch, tmp_path):
    from ui.services.training_service import TrainingService
    from ui.state import AppState

    state = AppState()
    job = state.create_job("sft", "test", output_dir=tmp_path)
    service = TrainingService(state)
    captured: dict[str, object] = {}

    async def fake_exec(*cmd, stdout=None, stderr=None, env=None, cwd=None):
        captured["cmd"] = cmd
        captured["env"] = env
        return SimpleNamespace(stdout=None)

    async def fake_stream(job_id: str):
        return None

    monkeypatch.setenv(HF_TOKEN_ENV, "hf_train_secret")
    monkeypatch.setattr("ui.services.training_service.asyncio.create_subprocess_exec", fake_exec)
    monkeypatch.setattr(service, "_stream_logs", fake_stream)

    asyncio.run(service._launch_process(job.id, ["python", "-c", "print('ok')"], no_caffeinate=True))

    env = captured["env"]
    assert isinstance(env, dict)
    assert env[HF_TOKEN_ENV] == "hf_train_secret"
