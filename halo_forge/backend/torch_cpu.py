"""CPU fallback backend.

Last-resort path for tests, tiny smoke runs, and any host without an
accelerator. Training is technically supported but unrealistic for the model
sizes halo-forge targets — trainers should warn and proceed only when
explicitly opted in.
"""

from __future__ import annotations

from typing import Any, Dict

from halo_forge.backend._torch_common import _TorchBackendBase
from halo_forge.backend.base import BackendCapabilities


_CAPABILITIES = BackendCapabilities(
    name="cpu",
    # bf16/fp16 on CPU are technically available but slow and patchy; default
    # to fp32 and let callers opt in if they know what they're doing.
    supports_bf16=False,
    supports_fp16=False,
    preferred_dtype_str="float32",
    supports_4bit=False,
    supports_8bit=False,
    supports_flash_attn=False,
    preferred_attn_impl="eager",
    supports_training=True,
    supports_peft=True,
)


class CPUBackend(_TorchBackendBase):
    def __init__(self) -> None:
        super().__init__(name="cpu", capabilities=_CAPABILITIES)

    def device(self) -> str:
        return "cpu"

    def training_defaults(self) -> Dict[str, Any]:
        return {
            "bf16": False,
            "fp16": False,
            "gradient_checkpointing": True,
            "optim": "adamw_torch",
            "attn_implementation": self.capabilities.preferred_attn_impl,
        }

    def inference_defaults(self) -> Dict[str, Any]:
        return {
            "low_cpu_mem_usage": True,
            "attn_implementation": self.capabilities.preferred_attn_impl,
        }
