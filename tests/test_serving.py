"""OpenAI-compatible serving tests (Track I1).

Inject a fake adapter so we exercise the FastAPI surface without
loading a real model. Validates request/response shape, sampling-knob
plumbing, the chat-template / ChatML fallback path, and error mapping.
"""

from __future__ import annotations

from typing import Optional

import pytest


class _FakeAdapter:
    """Captures the prompt + sampling args for assertion."""

    backend_name = "fake"

    def __init__(self, *, response: str = "ok", model_name: str = "fake/model"):
        self.model_name = model_name
        self.response = response
        self.last_prompt: Optional[str] = None
        self.last_kwargs: Optional[dict] = None

    def generate(self, prompt: str, **kwargs) -> str:
        self.last_prompt = prompt
        self.last_kwargs = kwargs
        return self.response


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from halo_forge.serving.app import create_serving_app

    fake = _FakeAdapter(response="hello there")
    app = create_serving_app(model_name=fake.model_name, adapter=fake)
    with TestClient(app) as c:
        yield c, fake


def test_models_endpoint_returns_served_model(client):
    c, fake = client
    r = c.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert any(m["id"] == fake.model_name for m in body["data"])


def test_chat_completions_basic_round_trip(client):
    c, fake = client
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": fake.model_name,
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Say hi."},
            ],
            "max_tokens": 32,
            "temperature": 0.5,
            "top_p": 0.95,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == fake.model_name
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "hello there"
    assert choice["finish_reason"] == "stop"
    # Sampling knobs landed in adapter.generate.
    assert fake.last_kwargs["max_tokens"] == 32
    assert fake.last_kwargs["temperature"] == 0.5
    assert fake.last_kwargs["top_p"] == 0.95
    # Prompt embeds both messages in ChatML fallback.
    assert "Say hi." in fake.last_prompt
    assert "Be terse." in fake.last_prompt


def test_chat_completions_empty_messages_400(client):
    c, _ = client
    r = c.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": []},
    )
    assert r.status_code == 400


def test_chat_completions_streaming_sends_openai_sse_chunks(client):
    c, fake = client
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": fake.model_name,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert '"object":"chat.completion.chunk"' in body
    assert '"role":"assistant"' in body
    assert '"content":"hello there"' in body
    assert "data: [DONE]" in body


def test_completions_endpoint_round_trip(client):
    c, fake = client
    r = c.post(
        "/v1/completions",
        json={
            "model": fake.model_name,
            "prompt": "Once upon a time",
            "max_tokens": 16,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "hello there"
    # Plain prompt passes through unmodified.
    assert fake.last_prompt == "Once upon a time"


def test_completions_streaming_sends_openai_sse_chunks(client):
    c, fake = client
    r = c.post(
        "/v1/completions",
        json={
            "model": fake.model_name,
            "prompt": "Once upon a time",
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert '"object":"text_completion"' in body
    assert '"text":"hello there"' in body
    assert "data: [DONE]" in body


def test_completions_stop_sequence_truncates(client):
    """Stop sequences post-truncate the adapter's output to honor the
    OpenAI semantics even on backends that don't natively support stop."""
    from fastapi.testclient import TestClient

    from halo_forge.serving.app import create_serving_app

    fake = _FakeAdapter(response="hello world. Goodbye world.")
    app = create_serving_app(model_name="x", adapter=fake)
    with TestClient(app) as c:
        r = c.post(
            "/v1/completions",
            json={"model": "x", "prompt": "x", "stop": ["Goodbye"]},
        )
    assert r.status_code == 200
    text = r.json()["choices"][0]["text"]
    assert text == "hello world. "  # cut at "Goodbye"


def test_health_endpoint(client):
    c, fake = client
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["model"] == fake.model_name


def test_validation_errors_for_out_of_range_temperature():
    """Pydantic validators bound `temperature` to the OpenAI-spec range."""
    from fastapi.testclient import TestClient

    from halo_forge.serving.app import create_serving_app

    app = create_serving_app(model_name="x", adapter=_FakeAdapter())
    with TestClient(app) as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 5.0,
            },
        )
    assert r.status_code == 422  # FastAPI validation error


def test_truncate_at_stop_handles_multiple_and_empty():
    from halo_forge.serving.adapter import _truncate_at_stop

    assert _truncate_at_stop("abc<end>def<end>", ["<end>"]) == "abc"
    # First-occurring stop wins, not the leftmost in the stop list.
    assert _truncate_at_stop("zzAxxBzz", ["B", "A"]) == "zz"
    # Empty stops are ignored, not treated as zero-length matches.
    assert _truncate_at_stop("hello", ["", "bye"]) == "hello"


def test_cli_serve_help_registers(monkeypatch, capsys):
    """`halo-forge serve --help` must show the new subcommand without
    blowing up imports."""
    import sys
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "serve", "--help"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "OpenAI-compatible" in out or "serve" in out.lower()
