"""Lazy runtime integrations for Dataset Lab recipes.

Nothing in this module imports heavyweight optional dependencies until the
corresponding recipe step is executed.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import threading
from pathlib import Path
from typing import Any, Mapping

from .errors import RecipeError
from .profiling import record_text


def _api_key(params: Mapping[str, Any]) -> str | None:
    env_name = str(params.get("api_key_env") or "HALOFORGE_TEACHER_API_KEY")
    return (
        str(params["api_key"])
        if params.get("api_key") and str(params.get("api_key")) != "<redacted>"
        else os.environ.get(env_name) or os.environ.get("OPENAI_API_KEY")
    )


def _media_value(row: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    field = str(params.get("media_field") or "")
    if field:
        return row.get(field)
    return row.get("image") if row.get("image") is not None else row.get("audio")


def _media_data_url(value: Any, base_dir: Path | None = None) -> str | None:
    if isinstance(value, Mapping):
        value = value.get("path") or value.get("url") or value.get("reference")
    if not isinstance(value, str) or not value:
        return None
    if value.startswith(("http://", "https://", "data:")):
        return value
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir:
        path = base_dir / path
    if not path.is_file():
        return None
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def configured_teacher(
    prompt: str,
    params: Mapping[str, Any] | None = None,
    row: Mapping[str, Any] | None = None,
) -> str:
    """Generate text from Ollama or an OpenAI-compatible local/hosted endpoint."""
    import requests

    config = dict(params or {})
    endpoint_type = str(config.get("endpoint_type") or "openai_compatible").lower()
    model = str(config.get("teacher_model") or config.get("model") or "default")
    sampling = dict(config.get("sampling") or {})
    max_tokens = int(sampling.get("max_tokens", config.get("max_tokens", 512)))
    temperature = float(sampling.get("temperature", config.get("temperature", 0.8)))
    top_p = float(sampling.get("top_p", config.get("top_p", 1.0)))
    seed = sampling.get("seed", config.get("seed"))
    timeout = float(config.get("timeout_seconds", 120))
    base_dir = Path(str(config["base_dir"])) if config.get("base_dir") else None
    media = _media_data_url(_media_value(row or {}, config), base_dir)

    if endpoint_type == "ollama":
        base_url = str(config.get("base_url") or "http://127.0.0.1:11434").rstrip("/")
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                **({"seed": int(seed)} if seed is not None else {}),
            },
        }
        if media and ";base64," in media:
            payload["images"] = [media.split(";base64,", 1)[1]]
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
        response.raise_for_status()
        return str(response.json().get("response") or "")

    base_url = str(
        config.get("base_url")
        or os.environ.get("HALOFORGE_TEACHER_BASE_URL")
        or "http://127.0.0.1:8001/v1"
    ).rstrip("/")
    url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    content: Any = prompt
    if media:
        if media.startswith("data:audio/"):
            encoded = media.split(";base64,", 1)[1] if ";base64," in media else media
            audio_format = media.split("/", 1)[1].split(";", 1)[0]
            content = [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": encoded, "format": audio_format}},
            ]
        else:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": media}},
            ]
    headers = {"Content-Type": "application/json"}
    key = _api_key(config)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    messages = []
    if config.get("system_prompt"):
        messages.append({"role": "system", "content": str(config["system_prompt"])})
    messages.append({"role": "user", "content": content})
    response = requests.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **({"seed": int(seed)} if seed is not None else {}),
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    try:
        return str(payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RecipeError("Teacher endpoint returned no chat completion") from exc


def configured_verifier(
    record: Mapping[str, Any], params: Mapping[str, Any] | None = None
) -> float:
    """Resolve a registered verifier and return its normalized reward."""
    from halo_forge.rlvr.verifiers import get_verifier

    config = dict(params or {})
    name = str(config.get("verifier") or config.get("verifier_name") or "json_structure")
    kwargs = dict(config.get("verifier_kwargs") or {})
    verifier_cls = get_verifier(name)
    verifier = verifier_cls(**kwargs)
    output_field = str(config.get("output_field") or "")
    if output_field and output_field in record:
        candidate = record[output_field]
    else:
        candidate = (
            record.get("response")
            or record.get("completion")
            or record.get("transcript")
            or record.get("label")
            or record_text(record)
        )
    result = verifier.verify(str(candidate or ""))
    return float(getattr(result, "reward", result))


def configured_judge(record: Mapping[str, Any], params: Mapping[str, Any] | None = None) -> float:
    config = dict(params or {})
    config.setdefault("verifier", "llm_judge")
    kwargs = dict(config.get("verifier_kwargs") or {})
    for source, target in (
        ("rubric", "rubric"),
        ("judge_model", "judge_model"),
        ("base_url", "base_url"),
        ("timeout_seconds", "timeout_s"),
    ):
        if source in config:
            kwargs[target] = config[source]
    kwargs.setdefault("prompt", str(record.get("prompt") or ""))
    config["verifier_kwargs"] = kwargs
    return configured_verifier(record, config)


_SEMANTIC_LOCK = threading.Lock()
_SEMANTIC_MODEL: Any = None
_SEMANTIC_CACHE: dict[str, Any] = {}


def configured_semantic_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Cosine similarity using a lazily loaded local sentence-transformer."""
    global _SEMANTIC_MODEL
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RecipeError(
            "Semantic dedup requires `pip install sentence-transformers` "
            "or `pip install halo-forge[data-lab]`."
        ) from exc

    with _SEMANTIC_LOCK:
        if _SEMANTIC_MODEL is None:
            model_name = os.environ.get(
                "HALOFORGE_SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
            _SEMANTIC_MODEL = SentenceTransformer(model_name)

    def embedding(record: Mapping[str, Any]):
        text = record_text(record)
        cached = _SEMANTIC_CACHE.get(text)
        if cached is None:
            cached = _SEMANTIC_MODEL.encode(text, normalize_embeddings=True)
            _SEMANTIC_CACHE[text] = cached
        return cached

    return float(np.dot(embedding(left), embedding(right)))


__all__ = [
    "configured_judge",
    "configured_semantic_similarity",
    "configured_teacher",
    "configured_verifier",
]
