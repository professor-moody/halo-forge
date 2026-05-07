"""vLLM-backed rollout generator (Track I6).

vLLM is the throughput frontier for batched LLM generation: continuous
batching, paged-attention KV cache, and CUDA-graph capture combine to
beat HF ``model.generate`` by 5-10× on the same hardware. RAFT spends
most of its wall-clock generating samples, so swapping the rollout
engine for vLLM is the single largest speedup available without
changing the algorithm.

This module is the third sibling alongside ``TorchRolloutGenerator``
(rollout.py) and ``MLXRolloutGenerator`` (mlx_rollout.py). It implements
the same ``RolloutGenerator`` protocol so the trainer drops it in
without algorithmic change.

Backend gating:
  - **CUDA**: native, recommended path.
  - **ROCm**: vLLM has community ROCm support (gfx9 / gfx10 / gfx11);
    works on MI300X / 7900 XTX. Strix Halo (gfx1151) is iffier —
    treat it as experimental.
  - **MLX / MPS / CPU**: vLLM doesn't support these. Constructor
    raises a typed error pointing at the right alternative.

Sampling parity with TorchRolloutGenerator:
  - ``num_samples`` per prompt → SamplingParams(n=num_samples)
  - ``temperature``, ``top_p`` map directly
  - ``max_new_tokens`` → ``max_tokens``

Streaming-cache resume mirrors the torch path so a partially completed
run can pick up where it left off without regenerating.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

from halo_forge.rlvr.rollout import resume_from_cache


class VLLMUnavailableError(RuntimeError):
    """Raised when vLLM is requested on a backend it doesn't support."""


class VLLMRolloutGenerator:
    """Rollout generator backed by ``vllm.LLM``.

    The trainer instantiates one of these and the RAFT loop calls
    ``generate_samples`` per cycle. The vllm ``LLM`` instance is held
    on the generator (not the trainer) because vLLM's KV-cache
    initialization is *expensive* — re-creating it per cycle would
    eat the throughput win. Instead we reload weights via
    ``load_weights`` between cycles when the policy updates (commit-2
    of this track will wire that path).

    Args:
        model_name: HuggingFace id, mlx-community id, or local path.
            For HF ids, vLLM downloads through the HF cache.
        backend_name: optional accelerator-kind override; defaults to
            the active backend. Used to gate against MLX / MPS / CPU.
        dtype: vllm dtype string ("auto", "bfloat16", "float16",
            "float32"). "auto" lets vllm pick based on the model card.
        gpu_memory_utilization: fraction of VRAM vllm reserves for KV
            cache. Default 0.9 matches vllm's default; lower if you're
            sharing the GPU with the trainer's policy update step.
        max_model_len: cap on prompt+completion tokens. None lets vllm
            read it from the model config.
        seed: deterministic generation seed; defaults to 42.
        llm: pre-built ``vllm.LLM`` instance. Used by tests to inject a
            fake without importing vllm.
    """

    SUPPORTED_BACKENDS: tuple[str, ...] = ("cuda", "rocm", "rocm_gfx1151")

    def __init__(
        self,
        model_name: str,
        *,
        backend_name: Optional[str] = None,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        seed: int = 42,
        llm: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.seed = seed
        self._llm = llm

        if llm is None:
            self._validate_backend(backend_name)

        self.backend_name = backend_name or self._detect_backend()

        # Strix Halo vLLM support is community / experimental. Don't
        # block — let users try on their own machine — but make the
        # status loud so failures aren't mysterious.
        from halo_forge.utils.backend_config import warn_experimental_vllm_backend

        warn_experimental_vllm_backend(self.backend_name)

    # ------------------------------------------------------------------
    # Backend gating
    # ------------------------------------------------------------------

    @classmethod
    def _detect_backend(cls) -> str:
        try:
            from halo_forge.backend import get_backend

            return get_backend().name
        except Exception:
            return "unknown"

    @classmethod
    def _validate_backend(cls, override: Optional[str]) -> None:
        name = override or cls._detect_backend()
        if name in cls.SUPPORTED_BACKENDS or name.startswith("rocm"):
            return
        if name == "mlx":
            raise VLLMUnavailableError(
                "vLLM is not supported on Apple Silicon (MLX). Use "
                "--rollout-engine mlx for MLX-native rollouts; on the same "
                "hardware MLX is competitive with vLLM."
            )
        if name in {"mps", "cpu", "unknown"}:
            raise VLLMUnavailableError(
                f"vLLM rollout requires a CUDA or ROCm backend; got {name!r}. "
                f"Use --rollout-engine torch (HF generate) or run on a CUDA host."
            )

    # ------------------------------------------------------------------
    # vllm bootstrap
    # ------------------------------------------------------------------

    def _ensure_llm(self):
        if self._llm is not None:
            return self._llm
        try:
            from vllm import LLM
        except ImportError as exc:
            raise VLLMUnavailableError(
                "vLLM is not installed. Install with `pip install '.[vllm]'` "
                "(CUDA / ROCm only)."
            ) from exc

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "seed": self.seed,
        }
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len

        self._llm = LLM(**kwargs)
        return self._llm

    # ------------------------------------------------------------------
    # Rollout — matches the RolloutGenerator protocol exactly
    # ------------------------------------------------------------------

    def generate_samples(
        self,
        prompts: List[str],
        *,
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
        batch_size: int,
        system_prompt: str,
        cache_path: Optional[Path] = None,
        log: Optional[Any] = None,
    ) -> List[Tuple[str, str]]:
        from vllm import SamplingParams

        all_samples, _start_batch = resume_from_cache(
            cache_path, num_samples=num_samples, batch_size=batch_size
        )
        target = len(prompts) * num_samples
        if log and all_samples:
            log(
                f"Resumed {len(all_samples)} cached samples; need {target - len(all_samples)} more",
                "dim",
            )
        if len(all_samples) >= target:
            return all_samples

        # Filter prompts that already have a full set in the cache.
        already_done: dict[str, int] = {}
        for prompt, _completion in all_samples:
            already_done[prompt] = already_done.get(prompt, 0) + 1
        remaining_prompts: List[str] = []
        remaining_n_per_prompt: List[int] = []
        for prompt in prompts:
            need = num_samples - already_done.get(prompt, 0)
            if need > 0:
                remaining_prompts.append(prompt)
                remaining_n_per_prompt.append(need)

        if not remaining_prompts:
            return all_samples

        formatted_prompts = [
            self._format_chat(p, system_prompt) for p in remaining_prompts
        ]

        # vLLM accepts a per-request SamplingParams (allows different `n`
        # per prompt). When all prompts need the same `n` we pass one
        # SamplingParams for the whole batch — slightly less Python
        # overhead and exactly what the user expects from a uniform call.
        uniform = all(n == num_samples for n in remaining_n_per_prompt)

        sampling_template = SamplingParams(
            n=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            seed=self.seed,
        )

        llm = self._ensure_llm()
        start_time = time.time()
        if log:
            log(
                f"vLLM generating {sum(remaining_n_per_prompt)} samples "
                f"across {len(remaining_prompts)} prompts (continuous batching)",
                "info",
            )

        if uniform:
            outputs = llm.generate(formatted_prompts, sampling_template)
        else:
            per_request = [
                SamplingParams(
                    n=n,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=1.0,
                    seed=self.seed,
                )
                for n in remaining_n_per_prompt
            ]
            outputs = llm.generate(formatted_prompts, per_request)

        cache_file = open(cache_path, "a") if cache_path else None
        try:
            for prompt, request_output in zip(remaining_prompts, outputs):
                # vllm RequestOutput exposes `.outputs` as a list of
                # CompletionOutput; we read `.text` from each one.
                for completion_output in request_output.outputs:
                    sample = (prompt, completion_output.text)
                    all_samples.append(sample)
                    if cache_file:
                        cache_file.write(
                            json.dumps(
                                {"prompt": prompt, "completion": completion_output.text}
                            )
                            + "\n"
                        )
            if cache_file:
                cache_file.flush()
        finally:
            if cache_file:
                cache_file.close()

        elapsed = time.time() - start_time
        if log:
            log(
                f"vLLM produced {len(all_samples)} samples in {elapsed/60:.1f} min "
                f"({len(all_samples)/max(elapsed,1e-3):.1f} samples/s)",
                "success",
            )
        return all_samples

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_chat(self, prompt: str, system_prompt: str) -> str:
        """Render the prompt to a single chat-formatted string.

        Mirrors `TorchRolloutGenerator.generate_samples` — system + user
        messages applied through the model's tokenizer chat template
        when available. vLLM owns its own tokenizer internally; we get
        a handle on it via ``llm.get_tokenizer()`` lazily.
        """
        try:
            llm = self._ensure_llm()
            tokenizer = llm.get_tokenizer()
            if getattr(tokenizer, "chat_template", None):
                return tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            # Fall through to the ChatML default if the tokenizer surface
            # doesn't expose what we need.
            pass

        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


__all__ = ["VLLMRolloutGenerator", "VLLMUnavailableError"]
