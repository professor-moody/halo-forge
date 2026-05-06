"""Apple Silicon PyTorch MPS backend.

Notable differences from CUDA/ROCm:
- bfloat16 is patchy pre-macOS 14 — default to float16 instead.
- bitsandbytes has no MPS kernels; quantization is unsupported.
- accelerate's auto device_map placer has historically split modules to CPU
  on MPS hosts (transformers <4.45). We pin to explicit `{"": "mps:0"}` to
  dodge that — happens via the shared `get_device_map()` helper.
- macOS multiprocessing fork issues: dataloader_num_workers=0 / pin_memory=False.
"""

from __future__ import annotations

from typing import Any, Dict

from halo_forge.backend._torch_common import _TorchBackendBase
from halo_forge.backend.base import BackendCapabilities


_CAPABILITIES = BackendCapabilities(
    name="mps",
    # bf16 is technically supported on macOS 14+ but unreliable on older
    # versions; we leave the flag True so callers can opt-in via spec.dtype
    # but make fp16 the *preferred* default.
    supports_bf16=True,
    supports_fp16=True,
    preferred_dtype_str="float16",
    supports_4bit=False,
    supports_8bit=False,
    supports_flash_attn=False,
    preferred_attn_impl="sdpa",
    supports_training=True,  # via PyTorch MPS; small models only in practice
    supports_peft=True,
)


class MPSBackend(_TorchBackendBase):
    def __init__(self) -> None:
        super().__init__(name="mps", capabilities=_CAPABILITIES)

    def device(self) -> str:
        return "mps"

    def empty_cache(self) -> None:
        import torch

        mps_mod = getattr(torch, "mps", None)
        if mps_mod is not None and hasattr(mps_mod, "empty_cache"):
            mps_mod.empty_cache()

    def training_defaults(self) -> Dict[str, Any]:
        return {
            # Default to fp16 on MPS — bf16 is patchy. Explicit user override
            # via cfg.bf16=True still wins.
            "bf16": False,
            "fp16": True,
            "dataloader_num_workers": 0,  # macOS fork issues
            "dataloader_pin_memory": False,  # MPS can't pin host memory
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
