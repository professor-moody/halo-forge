"""FastAPI app implementing the OpenAI v1 surface (Track I1).

Three endpoints:
  - POST /v1/chat/completions
  - POST /v1/completions
  - GET  /v1/models

The chat endpoint builds a prompt from the messages array using the
tokenizer's chat template when available, with a generic ChatML-style
fallback. Both completion endpoints are non-streaming in v1; streaming
arrives with Track I3 alongside speculative decoding.

The app is built lazily by `create_serving_app(model_name)` so importing
this module doesn't pull torch / mlx into the import graph (relevant for
the public_api app, which may import `halo_forge.serving` for type
references but not actually serve).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from halo_forge.serving.adapter import (
    ServingAdapter,
    _truncate_at_stop,
    build_serving_adapter,
)

logger = logging.getLogger(__name__)


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
            state["adapter"] = build_serving_adapter(
                model_name,
                backend_name=backend_name,
                trust_remote_code=trust_remote_code,
            )
        return state["adapter"]

    @app.get("/v1/models", response_model=ModelList)
    def list_models() -> ModelList:
        return ModelList(
            data=[ModelEntry(id=model_name, created=started_at)]
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
        if req.stream:
            raise HTTPException(
                status_code=501,
                detail="Streaming not implemented in v1; arrives with Track I3.",
            )
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

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=req.model or model_name,
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
    def completions(req: CompletionRequest) -> CompletionResponse:
        if req.stream:
            raise HTTPException(
                status_code=501,
                detail="Streaming not implemented in v1; arrives with Track I3.",
            )

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
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=req.model or model_name,
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
        return {"ok": True, "model": model_name}

    return app


__all__ = [
    "create_serving_app",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "CompletionRequest",
    "CompletionResponse",
    "ModelList",
]
