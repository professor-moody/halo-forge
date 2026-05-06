"""Backend-aware text generation adapter for the serving endpoint.

Wraps either MLX (`mlx_lm.generate`) or PyTorch (`transformers.AutoModelForCausalLM`)
behind one ``generate(prompt, **kwargs) -> str`` interface so the FastAPI
layer doesn't have to know which backend it's talking to.

This is deliberately *lighter* than `inference.optimizer.InferenceOptimizer` —
that class is the right tool for quantize-and-export pipelines but it
validates a full optimization config every constructor call. For serving
we just want "load model, generate, repeat".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ServingAdapter:
    """Generic interface every backend-specific adapter implements."""

    model_name: str
    backend_name: str

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
    ) -> str:
        raise NotImplementedError


class _MLXServingAdapter(ServingAdapter):
    """Routes generation through ``halo_forge.backend.mlx.MLXInferenceAdapter``."""

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name, backend_name="mlx")
        from halo_forge.backend.mlx import MLXInferenceAdapter

        self._inner = MLXInferenceAdapter(model_name)
        self._inner.load()

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
    ) -> str:
        # Stop strings are honored by the FastAPI layer (post-trim) so
        # both adapters share one implementation; we ignore `stop` here.
        return self._inner.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )


class _TorchServingAdapter(ServingAdapter):
    """PyTorch path: transformers AutoModelForCausalLM + tokenizer.

    Reuses the accelerator helpers so we land on the right device on
    rocm / cuda / mps / cpu without per-host conditional code.
    """

    def __init__(self, model_name: str):
        from halo_forge.backend import get_backend

        backend = get_backend()
        super().__init__(model_name=model_name, backend_name=backend.name)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from halo_forge.utils.accelerator import (
            get_device_map,
            recommended_attn_impl,
            recommended_dtype,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=recommended_dtype(),
            device_map=get_device_map(),
            trust_remote_code=True,
            attn_implementation=recommended_attn_impl(),
        )
        self._model.eval()

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
    ) -> str:
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)

        with torch.no_grad():
            output = self._model.generate(**inputs, **gen_kwargs)
        # Slice off the prompt tokens so we return only the continuation.
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][prompt_len:]
        # Stop strings honored at the FastAPI layer; see _MLXServingAdapter.
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


def _truncate_at_stop(text: str, stops: list[str]) -> str:
    """Cut ``text`` at the first occurrence of any string in ``stops``."""
    cut = len(text)
    for s in stops:
        if not s:
            continue
        idx = text.find(s)
        if idx >= 0 and idx < cut:
            cut = idx
    return text[:cut]


def build_serving_adapter(model_name: str, *, backend_name: Optional[str] = None) -> ServingAdapter:
    """Construct a ``ServingAdapter`` matching the active or requested backend.

    Args:
        model_name: HuggingFace id, mlx-community id, or local path.
        backend_name: Override the active backend ("mlx" forces the MLX
            path even on a host with both available). Defaults to the
            value `halo_forge.backend.get_backend()` returns.

    Returns:
        A loaded adapter ready for `.generate(...)`.

    Raises:
        ImportError if MLX path is requested without `mlx-lm` installed.
    """
    if backend_name is None:
        try:
            from halo_forge.backend import get_backend

            backend_name = get_backend().name
        except Exception:
            backend_name = "cpu"

    if backend_name == "mlx":
        return _MLXServingAdapter(model_name)
    return _TorchServingAdapter(model_name)


__all__ = ["ServingAdapter", "build_serving_adapter"]
