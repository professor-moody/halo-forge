"""MLX-backed rollout generator for the RAFT loop.

Phase 5a goal: generate rollouts on MLX (fast on Apple Silicon) but feed the
accepted samples back into the existing PyTorch policy-update step. The
PyTorch trainer never sees an mlx tensor — it only consumes the
`(prompt, completion)` text tuples this generator returns, identical in shape
to `TorchRolloutGenerator`'s output. That makes 5a useful even if 5b/5c
(full MLX-native RAFT) never lands.

mlx_lm's generate API is single-prompt + single-sample, so we loop. The
naive cost is `n_prompts * n_samples_per_prompt * generate(...)`. On Apple
Silicon this is still markedly faster than torch+MPS for >0.5B models because
mlx_lm's KV cache and quantized matmuls are tuned for the platform.

Caches resume the same way as the torch path so an interrupted RAFT cycle
can be picked back up after a `--accelerator` change.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

from halo_forge.rlvr.rollout import append_to_cache, resume_from_cache


class MLXRolloutUnavailable(RuntimeError):
    """Raised when MLX rollout is requested but mlx-lm isn't installed."""


def _require_mlx_lm():
    """Import mlx_lm.generate / load. Same defensive pattern as MLXSFTTrainer.

    Kept as a function so this module imports cleanly on Linux/CUDA hosts;
    only paying the import cost when the rollout actually runs.
    """
    try:
        from mlx_lm import generate as mlx_generate
        from mlx_lm import load as mlx_load
    except ImportError as exc:
        raise MLXRolloutUnavailable(
            "MLX rollout requires the `[mlx]` extra. Install with: "
            "`pip install -e '.[mlx]'` (Apple Silicon only)."
        ) from exc
    return mlx_load, mlx_generate


class MLXRolloutGenerator:
    """Rollout generator using mlx_lm.

    The generator owns its own MLX model + tokenizer (loaded lazily on first
    `generate_samples`). The trainer's PyTorch model and tokenizer are only
    used by the policy-update step; the two stay decoupled, which is what
    makes the 5a hybrid work.

    Args:
        model_name: HuggingFace ID or local path to MLX-format weights.
            For best results use a `mlx-community/...` variant — see
            `docs/MLX.md` for conversion.
        chat_template: optional override applied via the MLX tokenizer's
            `apply_chat_template`. None means use the tokenizer's default.
    """

    def __init__(self, model_name: str, *, chat_template: Optional[str] = None) -> None:
        self.model_name = model_name
        self.chat_template = chat_template
        self._model: Any = None
        self._tokenizer: Any = None
        self._mlx_load: Any = None
        self._mlx_generate: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        self._mlx_load, self._mlx_generate = _require_mlx_lm()
        self._model, self._tokenizer = self._mlx_load(self.model_name)

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
        # batch_size is accepted for protocol parity but ignored — mlx_lm
        # generates one prompt at a time. We still group prompts into
        # "logical batches" for cache-resume + progress reporting.
        all_samples, start_batch = resume_from_cache(
            cache_path, num_samples=num_samples, batch_size=batch_size
        )
        total = len(prompts) * num_samples
        if log and start_batch:
            log(
                f"[mlx] Resuming from batch {start_batch + 1} "
                f"({len(all_samples)} samples already cached)",
                "dim",
            )
        if len(all_samples) >= total:
            if log:
                log(
                    f"[mlx] Generation already complete ({len(all_samples)} samples cached)",
                    "success",
                )
            return all_samples

        self._ensure_loaded()
        # Imported lazily after _ensure_loaded so MLXRolloutUnavailable surfaces
        # before tqdm imports on hosts that have tqdm but no mlx.
        from tqdm import tqdm

        start_time = time.time()
        batch_iter = list(range(0, len(prompts), max(1, batch_size)))
        try:
            pbar = tqdm(
                enumerate(batch_iter),
                total=len(batch_iter),
                desc="Generating (mlx)",
                initial=start_batch,
                unit="batch",
            )
            for batch_idx, i in pbar:
                if batch_idx < start_batch:
                    continue
                batch_prompts = prompts[i : i + batch_size]
                batch_samples: List[Tuple[str, str]] = []
                # Build a sampler once per cycle; mlx-lm 0.21+ requires the
                # sampler= kwarg, older versions accept temp= directly. We
                # pick whichever path the installed version supports.
                sampler = None
                try:
                    from mlx_lm.sample_utils import make_sampler

                    sampler = make_sampler(temp=float(temperature))
                except ImportError:
                    pass

                for prompt in batch_prompts:
                    formatted = self._format_prompt(prompt, system_prompt)
                    for _ in range(num_samples):
                        if sampler is not None:
                            completion = self._mlx_generate(
                                self._model,
                                self._tokenizer,
                                prompt=formatted,
                                max_tokens=max_new_tokens,
                                sampler=sampler,
                            )
                        else:
                            completion = self._mlx_generate(
                                self._model,
                                self._tokenizer,
                                prompt=formatted,
                                max_tokens=max_new_tokens,
                                temp=temperature,
                            )
                        # mlx_lm.generate returns the *generated* tokens
                        # decoded — no prompt prefix to strip. Keep parity
                        # with the torch path's "decode only new tokens"
                        # behavior automatically.
                        batch_samples.append((prompt, completion))
                        all_samples.append((prompt, completion))
                append_to_cache(cache_path, batch_samples)
                pbar.set_postfix(samples=len(all_samples), refresh=True)
            pbar.close()
        finally:
            pass

        elapsed = time.time() - start_time
        if log:
            log(
                f"[mlx] Generated {len(all_samples)} samples in {elapsed/60:.1f} minutes",
                "success",
            )
        return all_samples

    def _format_prompt(self, prompt: str, system_prompt: str) -> str:
        """Apply the MLX tokenizer's chat template to a (system, user) pair.

        Falls back to a manual ChatML rendering if the tokenizer doesn't
        support `apply_chat_template` (rare for MLX-converted models, but
        possible for raw bare tokenizers).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            sys_part = f"<|im_start|>system\n{system_prompt}<|im_end|>\n" if system_prompt else ""
            return (
                f"{sys_part}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            )

    def cleanup(self) -> None:
        """Drop the model + tokenizer. The trainer reloads its own torch
        model after each cycle; the MLX rollout state stays warm across
        cycles unless the caller explicitly cleans up.
        """
        self._model = None
        self._tokenizer = None


__all__ = [
    "MLXRolloutGenerator",
    "MLXRolloutUnavailable",
]
