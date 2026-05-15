"""FastAPI app implementing the OpenAI v1 surface (Track I1).

Three endpoints:
  - POST /v1/chat/completions
  - POST /v1/completions
  - GET  /v1/models

The chat endpoint builds a prompt from the messages array using the
tokenizer's chat template when available, with a generic ChatML-style
fallback. Both completion endpoints support OpenAI-shaped
``text/event-stream`` responses for clients that set ``stream: true``.
The current streaming path emits the adapter's completed result through
the OpenAI wire format; backend-native token streaming is deliberately
deferred.

The app is built lazily by `create_serving_app(model_name)` so importing
this module doesn't pull torch / mlx into the import graph (relevant for
the public_api app, which may import `halo_forge.serving` for type
references but not actually serve).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
import uuid
from typing import Any, Iterator, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from halo_forge.serving.adapter import (
    ServingAdapter,
    _truncate_at_stop,
    build_serving_adapter,
)

logger = logging.getLogger(__name__)

GATED_MODEL_MESSAGE = (
    "This model requires Hugging Face access. Choose an open model, log in with a token, "
    "or use a local artifact."
)


# ----- request / response models -------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=256, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, gt=0.0, le=1.0)
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = Field(default=256, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, gt=0.0, le=1.0)
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


class ModelEntry(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "halo-forge"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelEntry]


# ----- helpers --------------------------------------------------------------


def _build_chat_prompt(adapter: ServingAdapter, messages: List[ChatMessage]) -> str:
    """Render ``messages`` to a single prompt string the adapter can consume.

    Prefers the tokenizer's chat template (the model's authoritative format).
    Falls back to a generic ChatML rendering for adapters that don't
    expose a tokenizer (the MLX path keeps it private; ChatML works for
    most modern instruct models).
    """
    tokenizer = getattr(adapter, "_tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [m.model_dump() for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )

    chunks = []
    for msg in messages:
        chunks.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
    chunks.append("<|im_start|>assistant\n")
    return "\n".join(chunks)


def _approximate_tokens(text: str) -> int:
    """Rough char-per-token estimate. Good enough for usage reporting; the
    real per-tokenizer count requires running through the tokenizer which
    we keep off the hot path."""
    return max(0, len(text) // 4)


def _sse(payload: dict[str, Any] | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n"


def _chat_stream(
    *,
    chunk_id: str,
    created: int,
    model: str,
    text: str,
) -> Iterator[str]:
    base = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }
    yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
    if text:
        yield _sse({**base, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})
    yield _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    yield _sse("[DONE]")


def _completion_stream(
    *,
    chunk_id: str,
    created: int,
    model: str,
    text: str,
) -> Iterator[str]:
    base = {
        "id": chunk_id,
        "object": "text_completion",
        "created": created,
        "model": model,
    }
    yield _sse({**base, "choices": [{"index": 0, "text": text, "finish_reason": None}]})
    yield _sse({**base, "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]})
    yield _sse("[DONE]")


def _looks_like_local_path(model_name: str) -> bool:
    text = str(model_name or "").strip()
    return text.startswith(("./", "../", "/", "~"))


def _missing_local_model_detail(model_name: str) -> Optional[str]:
    if not _looks_like_local_path(model_name):
        return None
    path = Path(model_name).expanduser()
    if path.exists():
        return None
    return f"local model path does not exist: {model_name}"


def _load_error_to_http_exception(exc: Exception, *, model_name: str) -> HTTPException:
    text = str(exc)
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return HTTPException(
            status_code=503,
            detail=f"serving backend dependency unavailable: {text}",
        )
    if "No Metal device available" in text or "metal::load_device" in text:
        return HTTPException(
            status_code=503,
            detail=(
                "MLX is installed, but this process cannot access a Metal GPU. "
                "Run from a normal Apple Silicon Terminal session or choose another backend."
            ),
        )
    missing_path = _missing_local_model_detail(model_name)
    if missing_path is not None:
        return HTTPException(status_code=400, detail=missing_path)
    if _looks_like_gated_hf_error(text):
        return HTTPException(
            status_code=403,
            detail={
                "error_kind": "gated_model",
                "message": GATED_MODEL_MESSAGE,
                "model": model_name,
                "hint": "Open an ungated Qwen/MLX model, run `huggingface-cli login`, set HF_TOKEN, or serve a local artifact.",
            },
        )
    return HTTPException(status_code=500, detail=f"failed to load serving adapter: {text}")


def _looks_like_gated_hf_error(text: str) -> bool:
    lower = str(text or "").lower()
    gated_markers = (
        "gated repo",
        "cannot access gated repo",
        "restricted",
        "please log in",
        "401 client error",
        "repository not found",
        "private repository",
    )
    return any(marker in lower for marker in gated_markers) and (
        "huggingface.co" in lower
        or "hugging face" in lower
        or "hf.co" in lower
        or "repo" in lower
    )


# ----- app factory ---------------------------------------------------------


def create_serving_app(
    *,
    model_name: str,
    backend_name: Optional[str] = None,
    trust_remote_code: bool = False,
    adapter: Optional[ServingAdapter] = None,
) -> FastAPI:
    """Build a FastAPI app serving ``model_name`` on the OpenAI v1 surface.

    Args:
        model_name: HuggingFace id, mlx-community id, or local path.
        backend_name: Optional backend override (forwarded to
            ``build_serving_adapter``).
        adapter: Pre-built adapter (used by tests to inject a fake without
            loading a real model).
    """
    app = FastAPI(title="halo-forge serving", version="0.1.0")
    started_at = int(time.time())

    state: dict[str, Any] = {"adapter": adapter, "model_name": model_name}

    def _get_adapter() -> ServingAdapter:
        if state["adapter"] is None:
            missing_path = _missing_local_model_detail(model_name)
            if missing_path is not None:
                raise HTTPException(status_code=400, detail=missing_path)
            try:
                state["adapter"] = build_serving_adapter(
                    model_name,
                    backend_name=backend_name,
                    trust_remote_code=trust_remote_code,
                )
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("Serving adapter load failed for model=%s", model_name)
                raise _load_error_to_http_exception(exc, model_name=model_name) from exc
        return state["adapter"]

    @app.get("/v1/models", response_model=ModelList)
    def list_models() -> ModelList:
        return ModelList(
            data=[ModelEntry(id=model_name, created=started_at)]
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(req: ChatCompletionRequest):
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must be non-empty")

        adapter = _get_adapter()
        prompt = _build_chat_prompt(adapter, req.messages)
        text = adapter.generate(
            prompt,
            max_tokens=256 if req.max_tokens is None else req.max_tokens,
            temperature=0.7 if req.temperature is None else req.temperature,
            top_p=1.0 if req.top_p is None else req.top_p,
            stop=req.stop,
        )
        if req.stop:
            text = _truncate_at_stop(text, req.stop)
        created = int(time.time())
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        served_model = req.model or model_name
        if req.stream:
            return StreamingResponse(
                _chat_stream(
                    chunk_id=response_id,
                    created=created,
                    model=served_model,
                    text=text,
                ),
                media_type="text/event-stream",
            )

        return ChatCompletionResponse(
            id=response_id,
            created=created,
            model=served_model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=_approximate_tokens(prompt),
                completion_tokens=_approximate_tokens(text),
                total_tokens=_approximate_tokens(prompt) + _approximate_tokens(text),
            ),
        )

    @app.post("/v1/completions", response_model=CompletionResponse)
    def completions(req: CompletionRequest):
        adapter = _get_adapter()
        text = adapter.generate(
            req.prompt,
            max_tokens=256 if req.max_tokens is None else req.max_tokens,
            temperature=0.7 if req.temperature is None else req.temperature,
            top_p=1.0 if req.top_p is None else req.top_p,
            stop=req.stop,
        )
        if req.stop:
            text = _truncate_at_stop(text, req.stop)
        created = int(time.time())
        response_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        served_model = req.model or model_name
        if req.stream:
            return StreamingResponse(
                _completion_stream(
                    chunk_id=response_id,
                    created=created,
                    model=served_model,
                    text=text,
                ),
                media_type="text/event-stream",
            )
        return CompletionResponse(
            id=response_id,
            created=created,
            model=served_model,
            choices=[CompletionChoice(index=0, text=text, finish_reason="stop")],
            usage=Usage(
                prompt_tokens=_approximate_tokens(req.prompt),
                completion_tokens=_approximate_tokens(text),
                total_tokens=_approximate_tokens(req.prompt)
                + _approximate_tokens(text),
            ),
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        adapter = state.get("adapter")
        return {
            "ok": True,
            "model": model_name,
            "backend": getattr(adapter, "backend_name", None) or backend_name or "auto",
            "adapter_loaded": adapter is not None,
            "started_at": started_at,
            "streaming_supported": True,
        }

    return app


__all__ = [
    "create_serving_app",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "CompletionRequest",
    "CompletionResponse",
    "ModelList",
]
