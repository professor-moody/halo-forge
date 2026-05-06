"""Shared implementation for PyTorch-flavored backends.

ROCm / CUDA / MPS / CPU all use the same HuggingFace transformers loader and
the same `BitsAndBytesConfig` translation; they only differ in:
  - dtype defaults (bf16 vs fp16 vs fp32)
  - device_map ({"": "mps:0"} vs "auto" vs {"": "cpu"})
  - attn implementation defaults (eager vs sdpa)
  - quantization availability (bitsandbytes only on CUDA/ROCm)
  - env vars at startup (HSA_* on ROCm; nothing on the others by default)

This base class collapses the common bits so the four concrete classes are
each <30 lines.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from halo_forge.backend.base import (
    BackendCapabilities,
    BackendStrategy,
    BackendUnsupportedError,
    LoadSpec,
)


class _TorchBackendBase(BackendStrategy):
    """Common HF-transformers + BitsAndBytesConfig handling for torch backends."""

    name: str
    capabilities: BackendCapabilities

    def __init__(self, name: str, capabilities: BackendCapabilities) -> None:
        self.name = name
        self.capabilities = capabilities

    # ------------------------------------------------------------------
    # Model loading via HF transformers
    # ------------------------------------------------------------------

    def load_causal_lm(self, spec: LoadSpec) -> Any:
        from transformers import AutoModelForCausalLM

        kwargs: Dict[str, Any] = {
            "dtype": self.resolve_dtype(spec.dtype),
            "device_map": spec.device_map if spec.device_map is not None else self._default_device_map(),
            "trust_remote_code": spec.trust_remote_code,
            "attn_implementation": spec.attn_implementation or self.capabilities.preferred_attn_impl,
        }
        quant_cfg = self.resolve_quantization(spec.quantization)
        if quant_cfg is not None:
            kwargs["quantization_config"] = quant_cfg
        kwargs.update(spec.extra)
        return AutoModelForCausalLM.from_pretrained(spec.model_name, **kwargs)

    def load_tokenizer(self, spec: LoadSpec) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            spec.model_name,
            trust_remote_code=spec.trust_remote_code,
        )

    # ------------------------------------------------------------------
    # dtype / quantization resolution
    # ------------------------------------------------------------------

    def resolve_dtype(self, requested: Optional[str]) -> Any:
        import torch

        if requested is None:
            requested = self.capabilities.preferred_dtype_str

        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if requested not in mapping:
            raise BackendUnsupportedError(
                f"Backend {self.name!r} cannot resolve dtype {requested!r}; "
                f"choose one of {sorted(mapping)}."
            )

        if requested in {"bfloat16", "bf16"} and not self.capabilities.supports_bf16:
            raise BackendUnsupportedError(
                f"Backend {self.name!r} does not support bfloat16."
            )
        if requested in {"float16", "fp16"} and not self.capabilities.supports_fp16:
            raise BackendUnsupportedError(
                f"Backend {self.name!r} does not support float16."
            )
        return mapping[requested]

    def resolve_quantization(self, requested: Optional[str]) -> Any:
        if requested in (None, "", "none"):
            return None

        normalized = str(requested).lower().strip()
        if normalized == "4bit" and not self.capabilities.supports_4bit:
            raise BackendUnsupportedError(
                f"Backend {self.name!r} does not support 4-bit quantization "
                f"(bitsandbytes is unavailable)."
            )
        if normalized == "8bit" and not self.capabilities.supports_8bit:
            raise BackendUnsupportedError(
                f"Backend {self.name!r} does not support 8-bit quantization."
            )

        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise BackendUnsupportedError(
                "transformers is required for quantization config."
            ) from exc

        import torch

        if normalized == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        if normalized == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)

        raise BackendUnsupportedError(
            f"Unknown quantization {requested!r}; supported: '4bit', '8bit', None."
        )

    # ------------------------------------------------------------------
    # Resource discipline
    # ------------------------------------------------------------------

    def seed_all(self, seed: int) -> None:
        from halo_forge.utils.accelerator import seed_accelerator

        import torch

        torch.manual_seed(seed)
        seed_accelerator(seed)

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _default_device_map(self) -> Any:
        """Subclasses override to provide accelerate's `device_map` value."""
        from halo_forge.utils.accelerator import get_device_map

        return get_device_map()
