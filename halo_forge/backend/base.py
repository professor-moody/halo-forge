"""BackendStrategy ABC plus the contract types every backend speaks.

A backend is a single strategy object that knows how to load a model, place
tensors, pick an attention implementation, free memory, and seed RNGs for one
specific accelerator family. The registry in `halo_forge.backend.registry`
wires names ("rocm_gfx1151", "cuda", "mps", "mlx", "cpu") to lazy factories;
callers either pass `--backend NAME` explicitly or let `detect_backend()` pick.

Design notes:
- The ABC is intentionally thin. We do not unify training across backends —
  the PyTorch SFT/RAFT trainers stay backend-aware (cuda/rocm/mps share one
  path; MLX has sibling MLX-native trainers in later phases). Backends are
  responsible for *loading* and *resource discipline*, not for orchestrating
  training loops.
- `BackendCapabilities` is a frozen dataclass so trainers can compare against
  it cheaply and surface "feature X is unavailable on this backend" errors at
  config time instead of mid-load.
- `LoadSpec` is the cross-backend contract. Backends ignore fields they don't
  understand; the dispatcher's job is to translate `LoadSpec.dtype` etc. into
  whatever each backend's underlying loader (HF transformers, mlx_lm) expects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class BackendUnsupportedError(RuntimeError):
    """Raised when a LoadSpec asks for something this backend cannot honor.

    Examples: 4-bit quantization on MPS (no bitsandbytes); training on MLX
    backend before Phase 4 ships; bf16 on a backend whose capabilities flag
    it as unavailable.
    """


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend can and cannot do.

    Trainers consult this before trying to construct a config that the backend
    would refuse — e.g. SFT trainer downgrades load_in_4bit to bf16 on MPS by
    checking `supports_4bit` first.
    """

    name: str

    # Numeric precision
    supports_bf16: bool
    supports_fp16: bool
    preferred_dtype_str: str  # "bfloat16" | "float16" | "float32"

    # Quantization
    supports_4bit: bool
    supports_8bit: bool

    # Attention
    supports_flash_attn: bool
    preferred_attn_impl: str  # "sdpa" | "eager" | "flash_attention_2"

    # Training capability
    supports_training: bool
    # PEFT (HF peft library) — only PyTorch backends. MLX has its own LoRA path.
    supports_peft: bool
    # Apple M5+ experimental capability flag. Surfacing only; no auto-tuning.
    supports_neural_accelerators: bool = False


@dataclass
class LoadSpec:
    """Cross-backend description of a model-load request.

    Fields are advisory: backends fall back to capability defaults when a
    field is None. `extra` is an escape hatch for backend-specific kwargs
    (e.g. mlx-lm's tokenizer config) that don't deserve first-class fields.
    """

    model_name: str
    dtype: Optional[str] = None  # "bfloat16" | "float16" | "float32" | None
    quantization: Optional[str] = None  # "4bit" | "8bit" | None
    attn_implementation: Optional[str] = None  # None -> backend default
    trust_remote_code: bool = True
    device_map: Optional[Any] = None  # None -> backend default
    extra: Dict[str, Any] = field(default_factory=dict)


class BackendStrategy(ABC):
    """One accelerator family's strategy for loading and managing models.

    Implementations live in `halo_forge.backend.{torch_rocm,torch_mps,...}`.
    Concrete backends are stateless beyond the dataclass fields; that lets
    the registry hand out shared singletons cheaply.
    """

    name: str
    capabilities: BackendCapabilities

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup_environment(self) -> Dict[str, str]:
        """Apply backend-specific env vars at process start. Returns the dict
        of vars that were *newly* set so callers can log them. Default no-op.
        """
        return {}

    @abstractmethod
    def device(self) -> str:
        """Logical device string callers can compare against ("cuda", "mps",
        "cpu", "mlx"). Used for reporting; placement should go through
        `load_causal_lm` and friends.
        """

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @abstractmethod
    def load_causal_lm(self, spec: LoadSpec) -> Any:
        """Load a causal-LM model for this backend.

        Returns whatever the backend's runtime understands as a model
        (torch.nn.Module for PyTorch backends, mlx_lm's loaded model for MLX).
        """

    @abstractmethod
    def load_tokenizer(self, spec: LoadSpec) -> Any:
        """Load a tokenizer compatible with `load_causal_lm`'s output."""

    # ------------------------------------------------------------------
    # Config resolution
    # ------------------------------------------------------------------

    @abstractmethod
    def resolve_dtype(self, requested: Optional[str]) -> Any:
        """Map a requested dtype string ("bfloat16"/"float16"/...) to the
        backend's actual dtype object. Returns the *backend default* when
        `requested` is None. Raises BackendUnsupportedError if the request
        is incompatible with this backend's capabilities.
        """

    @abstractmethod
    def resolve_quantization(self, requested: Optional[str]) -> Any:
        """Translate a quantization string ("4bit"/"8bit"/None) into the
        backend's quantization config object (e.g. BitsAndBytesConfig for
        PyTorch backends, None for MLX which embeds quantization into the
        model name). Raises BackendUnsupportedError on unsupported requests.
        """

    # ------------------------------------------------------------------
    # Resource discipline
    # ------------------------------------------------------------------

    def empty_cache(self) -> None:
        """Release transient device-side memory. Default no-op; CUDA/ROCm
        and MPS backends override.
        """

    def seed_all(self, seed: int) -> None:
        """Seed the backend's RNGs deterministically."""

    # ------------------------------------------------------------------
    # Defaults for trainers
    # ------------------------------------------------------------------

    def training_defaults(self) -> Dict[str, Any]:
        """Backend-tuned defaults for HuggingFace TrainingArguments-style
        configs. Trainers merge these with user config.
        """
        return {}

    def inference_defaults(self) -> Dict[str, Any]:
        """Backend-tuned defaults for inference-only loads (low_cpu_mem_usage,
        attn_implementation, etc.).
        """
        return {}


__all__ = [
    "BackendCapabilities",
    "BackendStrategy",
    "BackendUnsupportedError",
    "LoadSpec",
]
