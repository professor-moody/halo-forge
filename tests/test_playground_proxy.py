"""Playground proxy tests (Track F-S).

The proxy at /api/public/playground/chat forwards to a configurable
serve URL. Tests inject a fake httpx client to validate the request
shaping + upstream-error pass-through without standing up a real
serve endpoint.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("HALOFORGE_RUN_DB_PATH", str(tmp_path / "runs.db"))
    from halo_forge.run_db import db as db_mod
    db_mod._GLOBAL_DB.clear()
    from halo_forge.auth.dependency import reset_store_for_tests
    reset_store_for_tests(None)

    from fastapi.testclient import TestClient
    from halo_forge.public_api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    db_mod._GLOBAL_DB.clear()


def _patch_httpx(monkeypatch, *, status_code=200, body=None):
    """Replace httpx.Client with a stub that records the call."""
    captured = {}
    fake_response = SimpleNamespace(
        status_code=status_code,
        text="" if body is None else "stub",
        json=lambda: body or {
            "id": "test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        },
    )

    class _FakeClient:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["body"] = json
            return fake_response

    import halo_forge.public_api.service as service_mod
    # The service does `import httpx` inside the function; we monkeypatch
    # the sys.modules entry so the `client = httpx.Client(...)` call
    # resolves to our fake.
    import sys
    fake_module = SimpleNamespace(Client=_FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_module)
    return captured


def test_playground_chat_forwards_messages(client, monkeypatch):
    captured = _patch_httpx(monkeypatch)
    r = client.post(
        "/api/public/playground/chat",
        json={
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
            "model": "fake-model",
            "temperature": 0.5,
            "max_tokens": 64,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "ok"
    # Forwarded payload preserved structure.
    assert captured["body"]["model"] == "fake-model"
    assert captured["body"]["temperature"] == 0.5
    assert captured["body"]["max_tokens"] == 64
    assert len(captured["body"]["messages"]) == 2


def test_playground_chat_default_serve_url(client, monkeypatch):
    captured = _patch_httpx(monkeypatch)
    client.post(
        "/api/public/playground/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    # Default targets the local serve endpoint at :8001/v1.
    assert "127.0.0.1:8001" in captured["url"]
    assert captured["url"].endswith("/chat/completions")


def test_playground_chat_custom_serve_url(client, monkeypatch):
    captured = _patch_httpx(monkeypatch)
    client.post(
        "/api/public/playground/chat",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "serve_url": "https://my-host.example.com/v1",
            "api_key": "sk-custom",
        },
    )
    assert captured["url"] == "https://my-host.example.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-custom"


def test_playground_chat_env_var_serve_url(client, monkeypatch):
    monkeypatch.setenv("HALOFORGE_PLAYGROUND_BASE_URL", "https://env-host/v1")
    captured = _patch_httpx(monkeypatch)
    client.post(
        "/api/public/playground/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert "env-host" in captured["url"]


def test_playground_chat_explicit_serve_url_overrides_env(client, monkeypatch):
    monkeypatch.setenv("HALOFORGE_PLAYGROUND_BASE_URL", "https://env-host/v1")
    captured = _patch_httpx(monkeypatch)
    client.post(
        "/api/public/playground/chat",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "serve_url": "https://explicit/v1",
        },
    )
    assert "explicit" in captured["url"]
    assert "env-host" not in captured["url"]


def test_playground_chat_upstream_error_passthrough(client, monkeypatch):
    _patch_httpx(
        monkeypatch,
        status_code=500,
        body={"detail": "model not loaded"},
    )
    r = client.post(
        "/api/public/playground/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    # Public API still returns 200 — the upstream_error envelope lets
    # the UI render the actual problem inline.
    assert r.status_code == 200
    body = r.json()
    assert body["upstream_error"] is True
    assert body["status"] == 500


def test_playground_chat_gated_model_error_is_friendly(client, monkeypatch):
    raw = (
        "failed to load serving adapter: You are trying to access a gated repo. "
        "401 Client Error for url https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct. "
        "Cannot access gated repo. Please log in."
    )
    _patch_httpx(
        monkeypatch,
        status_code=403,
        body={"detail": raw},
    )
    r = client.post(
        "/api/public/playground/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["upstream_error"] is True
    assert body["error_kind"] == "gated_model"
    assert body["message"] == (
        "This model requires Hugging Face access. Connect Hugging Face, accept the model license, "
        "or choose an open model."
    )
    assert body["action"] == "connect_huggingface"
    assert "Cannot access gated repo" in body["detail"]["detail"]


def test_playground_chat_empty_messages_400(client):
    r = client.post(
        "/api/public/playground/chat",
        json={"messages": []},
    )
    assert r.status_code == 400


def test_playground_chat_missing_messages_field_400(client):
    r = client.post(
        "/api/public/playground/chat",
        json={"model": "x"},
    )
    assert r.status_code == 400


def test_playground_chat_default_model_when_omitted(client, monkeypatch):
    captured = _patch_httpx(monkeypatch)
    client.post(
        "/api/public/playground/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert captured["body"]["model"] == "halo-forge"
