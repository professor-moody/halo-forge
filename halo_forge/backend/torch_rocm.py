"""ROCm backend (gfx1151 / Strix Halo). Also the fallback for other ROCm cards
since the runtime is the same — Strix-specific env vars are only applied on
hosts where `_is_rocm_host()` returns True.
"""

from __future__ import annotations

from typing import Any, Dict

from halo_forge.backend._torch_common import _TorchBackendBase
from halo_forge.backend.base import BackendCapabilities


_CAPABILITIES = BackendCapabilities(
    name="rocm",
    supports_bf16=True,
    supports_fp16=True,
    preferred_dtype_str="bfloat16",
    supports_4bit=True,  # via community ROCm bitsandbytes wheels
    supports_8bit=True,
    # gfx1151 lacks working FA2; SDPA needs the AOTriton flag. Eager is the
    # only safe default for Strix Halo.
    supports_flash_attn=False,
    preferred_attn_impl="eager",
    supports_training=True,
    supports_peft=True,
)


class ROCmBackend(_TorchBackendBase):
    def __init__(self) -> None:
        super().__init__(name="rocm_gfx1151", capabilities=_CAPABILITIES)

    def device(self) -> str:
        return "cuda"  # ROCm exposes the cuda namespace via HIP

    def setup_environment(self) -> Dict[str, str]:
        from halo_forge.utils.strix_halo import setup_strix_halo_env

        return setup_strix_halo_env()

    def empty_cache(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def training_defaults(self) -> Dict[str, Any]:
        # Mirrors get_strix_halo_training_config() but lives here so trainers
        # can consult the backend object instead of branching on a Strix flag.
        return {
            "bf16": True,
            "fp16": False,
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": False,
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
