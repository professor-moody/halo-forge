"""Apple MLX backend.

MLX is Apple's native ML framework — it does not share a runtime with PyTorch.
Models, weights, tokenizers, and the inference loop are all MLX-native. The
HuggingFace transformers loader does not work here; we use `mlx_lm.load` /
`mlx_lm.generate` instead.

Phase 3 ships *inference only*. Training (LoRA SFT, RLVR) lands in Phases 4-5
and will flip `supports_training=True` plus add MLX-native trainer modules.

Weight format: MLX needs MLX-format safetensors. The `mlx-community/...`
namespace on HuggingFace publishes pre-converted variants of common models.
For other models, use `python -m mlx_lm.convert` — see docs/MLX.md.
"""

from __future__ import annotations

import platform
from dataclasses import replace
from typing import Any, Dict, Optional

from halo_forge.backend.base import (
    BackendCapabilities,
    BackendStrategy,
    BackendUnsupportedError,
    LoadSpec,
)


_CAPABILITIES = BackendCapabilities(
    name="mlx",
    # MLX dtypes: float32 and float16/bfloat16 are all supported on Apple
    # Silicon. We default to float16 to mirror MPS (and because MLX's bf16
    # support is solid but not a meaningful win over fp16 on M-series GPUs).
    supports_bf16=True,
    supports_fp16=True,
    preferred_dtype_str="float16",
    # MLX has its own quantization scheme (group-wise, not bitsandbytes).
    # Quantized weights are baked into the model artifact (e.g.
    # `mlx-community/Qwen2.5-7B-Instruct-4bit`), not requested at load time —
    # so we report no support for the BitsAndBytes-style runtime quantization.
    supports_4bit=False,
    supports_8bit=False,
    supports_flash_attn=False,
    preferred_attn_impl="sdpa",  # advisory only; MLX has its own attention
    # Phase 4 enables LoRA SFT via MLXSFTTrainer (halo_forge.sft.mlx_trainer).
    # RAFT/RLVR is still pending — Phase 5 stages 5a/5b/5c.
    supports_training=True,
    supports_peft=False,  # MLX has its own LoRA, not HF peft
)


class MLXBackend(BackendStrategy):
    """MLX inference backend. Phase 3 surface."""

    def __init__(self) -> None:
        self.name = "mlx"
        self.capabilities = _mlx_capabilities()

    def device(self) -> str:
        return "mlx"

    # ------------------------------------------------------------------
    # Model loading via mlx_lm
    # ------------------------------------------------------------------

    def load_causal_lm(self, spec: LoadSpec) -> Any:
        """Load an MLX causal-LM via `mlx_lm.load`.

        Returns the bare model object; the tokenizer is fetched separately
        through `load_tokenizer` so callers match the BackendStrategy contract.
        Internally `mlx_lm.load` returns `(model, tokenizer)` — we cache the
        tokenizer on the model so `load_tokenizer` doesn't repeat the work.
        """
        try:
            from mlx_lm import load as mlx_load
        except ImportError as exc:
            raise BackendUnsupportedError(
                "MLX backend requires `mlx-lm`. Install with `pip install '.[mlx]'` "
                "(or `pip install mlx-lm`)."
            ) from exc

        # mlx_lm.load takes the tokenizer config via kwargs; spec.extra is the
        # escape hatch for callers that need to pass adapter_path / lazy load.
        kwargs: Dict[str, Any] = {}
        kwargs.update(spec.extra)
        model, tokenizer = mlx_load(spec.model_name, **kwargs)
        # Stash so load_tokenizer doesn't reload — mlx_lm.load does both halves
        # in one shot, and re-running it would download/parse weights twice.
        try:
            setattr(model, "_halo_forge_tokenizer", tokenizer)
        except (AttributeError, TypeError):
            pass
        return model

    def load_tokenizer(self, spec: LoadSpec) -> Any:
        """Load the MLX tokenizer for `spec.model_name`.

        If the caller already invoked `load_causal_lm`, they should pass that
        model back via `spec.extra["model"]` to reuse the cached tokenizer
        rather than re-downloading.
        """
        cached_model = spec.extra.get("model") if spec.extra else None
        if cached_model is not None:
            cached = getattr(cached_model, "_halo_forge_tokenizer", None)
            if cached is not None:
                return cached

        try:
            from mlx_lm import load as mlx_load
        except ImportError as exc:
            raise BackendUnsupportedError(
                "MLX backend requires `mlx-lm`. Install with `pip install '.[mlx]'`."
            ) from exc

        _, tokenizer = mlx_load(spec.model_name)
        return tokenizer

    # ------------------------------------------------------------------
    # dtype / quantization
    # ------------------------------------------------------------------

    def resolve_dtype(self, requested: Optional[str]) -> Any:
        """Map requested dtype string to mlx.core.dtype.

        MLX exposes `mlx.core.float16`, `mlx.core.bfloat16`, `mlx.core.float32`.
        Note: most loaders ignore this and pull the dtype from the safetensors
        header, but we still resolve it for trainer/inference defaults.
        """
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise BackendUnsupportedError(
                "MLX backend requires `mlx`. Install with `pip install '.[mlx]'`."
            ) from exc

        if requested is None:
            requested = self.capabilities.preferred_dtype_str

        mapping = {
            "bfloat16": mx.bfloat16,
            "bf16": mx.bfloat16,
            "float16": mx.float16,
            "fp16": mx.float16,
            "float32": mx.float32,
            "fp32": mx.float32,
        }
        if requested not in mapping:
            raise BackendUnsupportedError(
                f"MLX backend cannot resolve dtype {requested!r}; "
                f"choose one of {sorted(mapping)}."
            )
        return mapping[requested]

    def resolve_quantization(self, requested: Optional[str]) -> Any:
        """MLX quantization is baked into the model artifact, not configured
        at load time. We accept None and reject anything else with a hint.
        """
        if requested in (None, "", "none"):
            return None
        raise BackendUnsupportedError(
            f"MLX does not honor runtime quantization={requested!r}. Use a "
            f"pre-quantized model (e.g. `mlx-community/Qwen2.5-7B-Instruct-4bit`) "
            f"or convert with `python -m mlx_lm.convert`."
        )

    # ------------------------------------------------------------------
    # Resource discipline
    # ------------------------------------------------------------------

    def empty_cache(self) -> None:
        try:
            import mlx.core as mx
        except ImportError:
            return
        # mlx 0.20+ moved clear_cache to the top-level `mx.clear_cache`;
        # older versions had it under `mx.metal.clear_cache`. Try the new
        # location first, fall back to the legacy one, give up silently if
        # neither exists (the cache will still be reclaimed eventually).
        clear = getattr(mx, "clear_cache", None)
        if clear is None:
            clear = getattr(getattr(mx, "metal", None), "clear_cache", None)
        if clear is not None:
            clear()

    def seed_all(self, seed: int) -> None:
        try:
            import mlx.core as mx
        except ImportError:
            return
        mx.random.seed(seed)

    # ------------------------------------------------------------------
    # Defaults (inference only in Phase 3)
    # ------------------------------------------------------------------

    def inference_defaults(self) -> Dict[str, Any]:
        return {
            # Surfaced for parity with the torch backends; mlx_lm.generate has
            # its own kwargs (`max_tokens`, `temp`, etc.) that callers pass
            # through `MLXInferenceAdapter.generate` directly.
        }


def _mlx_capabilities() -> BackendCapabilities:
    try:
        from halo_forge.telemetry.apple_silicon import AppleSiliconTelemetry
        from halo_forge.utils.apple_chip import parse_chip_brand

        chip = parse_chip_brand(AppleSiliconTelemetry._detect_device_name())
        macos_ok = _macos_version_tuple() >= (26, 2)
        supports_na = bool(chip and chip.generation >= 5 and macos_ok)
    except Exception:
        supports_na = False
    return replace(_CAPABILITIES, supports_neural_accelerators=supports_na)


def _macos_version_tuple() -> tuple[int, int]:
    version = platform.mac_ver()[0]
    parts: list[int] = []
    for item in version.split(".")[:2]:
        try:
            parts.append(int(item))
        except ValueError:
            parts.append(0)
    while len(parts) < 2:
        parts.append(0)
    return parts[0], parts[1]


class MLXInferenceAdapter:
    """Inference-time wrapper around an MLX model + tokenizer pair.

    Lives next to the backend so the inference dispatcher can hold a single
    object instead of juggling (model, tokenizer, sampler) tuples. Designed
    to be parallel to the existing PyTorch inference path in
    `halo_forge.inference.optimizer.InferenceOptimizer` so the public API
    can route on backend type without the rest of the system caring.
    """

    def __init__(self, model_name: str, *, backend: Optional[MLXBackend] = None) -> None:
        self.model_name = model_name
        self.backend = backend or MLXBackend()
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        spec = LoadSpec(model_name=self.model_name)
        self._model = self.backend.load_causal_lm(spec)
        self._tokenizer = self.backend.load_tokenizer(
            LoadSpec(model_name=self.model_name, extra={"model": self._model})
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> str:
        """Generate text. Lazily loads on first call.

        Sampling knobs cross an mlx-lm API boundary that moved in 0.21+:
        the old top-level `temp=` kwarg was replaced by a `sampler=` built
        via `mlx_lm.sample_utils.make_sampler`. We construct the sampler
        here so callers see a stable interface; extra kwargs (e.g.
        `repetition_penalty`) flow through to mlx_lm.generate verbatim.
        """
        if self._model is None or self._tokenizer is None:
            self.load()

        try:
            from mlx_lm import generate as mlx_generate
        except ImportError as exc:
            raise BackendUnsupportedError(
                "MLX backend requires `mlx-lm`. Install with `pip install '.[mlx]'`."
            ) from exc

        # Build the sampler the new way; if make_sampler isn't importable
        # (mlx-lm <0.21) fall back to the legacy `temp=` kwarg path. Both
        # paths produce the same generated string for the caller.
        try:
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=float(temperature), top_p=float(top_p))
            return mlx_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                **kwargs,
            )
        except ImportError:
            return mlx_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                **kwargs,
            )

    def cleanup(self) -> None:
        self._model = None
        self._tokenizer = None
        self.backend.empty_cache()
