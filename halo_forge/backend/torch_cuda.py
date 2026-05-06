"""NVIDIA CUDA backend.

Falls out for free from the ROCm work — only differs in attention defaults
(SDPA on CUDA, eager on ROCm) and the lack of HIP-specific env vars.
"""

from __future__ import annotations

from typing import Any, Dict

from halo_forge.backend._torch_common import _TorchBackendBase
from halo_forge.backend.base import BackendCapabilities


_CAPABILITIES = BackendCapabilities(
    name="cuda",
    supports_bf16=True,
    supports_fp16=True,
    preferred_dtype_str="bfloat16",
    supports_4bit=True,
    supports_8bit=True,
    # Modern CUDA + transformers ship FA2 by default but we conservatively
    # leave that to opt-in via spec.attn_implementation; SDPA is the safe
    # default that doesn't require flash-attn package.
    supports_flash_attn=True,
    preferred_attn_impl="sdpa",
    supports_training=True,
    supports_peft=True,
)


class CUDABackend(_TorchBackendBase):
    def __init__(self) -> None:
        super().__init__(name="cuda", capabilities=_CAPABILITIES)

    def device(self) -> str:
        return "cuda"

    def empty_cache(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def training_defaults(self) -> Dict[str, Any]:
        return {
            "bf16": True,
            "fp16": False,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "optim": "adamw_torch",
            "attn_implementation": self.capabilities.preferred_attn_impl,
        }

    def inference_defaults(self) -> Dict[str, Any]:
        return {
            "low_cpu_mem_usage": True,
            "attn_implementation": self.capabilities.preferred_attn_impl,
        }
