"""Pluggable rollout generators for RAFT.

The RAFT loop has three orthogonal stages: rollout (generate samples),
verify (run verifiers), filter+train (SFT on accepted samples). Phase 5a
makes rollout pluggable so MLX-flavored generation can substitute for the
default PyTorch path on Apple Silicon — typically the dominant cost in a
RAFT cycle, and the place where MLX is most clearly faster than torch+MPS.

The protocol intentionally mirrors `RAFTTrainer.generate_samples`'s public
contract: same args, same return type, same cache-resume semantics. That
keeps the trainer-side wiring trivial — `self._rollout_generator.generate`
is a drop-in for the inline torch loop.

Verifier dispatch and the policy-update step are *not* swapped here. The
PyTorch trainer still owns the SFT-on-accepted-samples step, the
fresh-base-model reload (RLVR-correctness invariant in raft_trainer.py),
and verifier orchestration. Phase 5b lifts those into MLX-native code.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple


class RolloutGenerator(Protocol):
    """Anything that can sample completions from a causal LM.

    Implementations live in this module (`TorchRolloutGenerator`) and in
    `halo_forge.rlvr.mlx_rollout` (`MLXRolloutGenerator`). The trainer holds
    one and delegates.
    """

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
        """Return `[(prompt, completion), ...]`.

        Args mirror the trainer's existing signature so the integration is a
        drop-in. `log` is an optional callable `(msg, level)` matching
        RAFTTrainer._log; implementations may ignore it.
        """
        ...


def resume_from_cache(
    cache_path: Optional[Path],
    *,
    num_samples: int,
    batch_size: int,
) -> Tuple[List[Tuple[str, str]], int]:
    """Load any pre-generated samples from a streaming JSONL cache.

    Returns `(samples, start_batch)`. `start_batch` is the next batch index
    to process given that `len(samples) // num_samples` prompts are already
    complete. Identical resume semantics to `RAFTTrainer.generate_samples`
    so any rollout impl can opt into checkpointing for free.
    """
    if not cache_path or not cache_path.exists():
        return [], 0

    samples: List[Tuple[str, str]] = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            samples.append((data["prompt"], data["completion"]))

    completed_prompts = len(samples) // max(1, num_samples)
    start_batch = completed_prompts // max(1, batch_size)
    return samples, start_batch


def append_to_cache(cache_path: Optional[Path], samples: List[Tuple[str, str]]) -> None:
    """Append a batch of (prompt, completion) tuples to the JSONL cache."""
    if not cache_path:
        return
    with open(cache_path, "a") as f:
        for prompt, completion in samples:
            f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")


class TorchRolloutGenerator:
    """Rollout generator backed by HuggingFace `model.generate`.

    Used as the default rollout when `RAFTTrainer` is initialized without an
    explicit generator (preserves the historical behavior). Holds references
    to the trainer's loaded model and tokenizer; does not load anything itself
    so the existing fresh-base-model-reload invariant in
    `raft_trainer._reload_model` continues to work as written.
    """

    def __init__(self, model: Any, tokenizer: Any, *, clear_cache_every_n_batches: int = 4):
        self.model = model
        self.tokenizer = tokenizer
        self.clear_cache_every_n_batches = max(1, clear_cache_every_n_batches)

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
        # Lazy import torch + tqdm so this module loads cleanly on MLX-only
        # hosts that won't ever invoke this path.
        import torch
        from tqdm import tqdm

        from halo_forge.utils.accelerator import empty_accelerator_cache

        all_samples, start_batch = resume_from_cache(
            cache_path, num_samples=num_samples, batch_size=batch_size
        )
        total = len(prompts) * num_samples
        if log and start_batch:
            log(
                f"Resuming from batch {start_batch + 1} ({len(all_samples)} samples already cached)",
                "dim",
            )
        if len(all_samples) >= total:
            if log:
                log(f"Generation already complete ({len(all_samples)} samples cached)", "success")
            return all_samples

        self.model.eval()
        start_time = time.time()
        batch_iter = list(range(0, len(prompts), batch_size))
        cache_file = open(cache_path, "a") if cache_path else None

        try:
            pbar = tqdm(
                enumerate(batch_iter),
                total=len(batch_iter),
                desc="Generating",
                initial=start_batch,
                unit="batch",
            )
            for batch_idx, i in pbar:
                if batch_idx < start_batch:
                    continue
                batch_prompts = prompts[i : i + batch_size]

                formatted = []
                for prompt in batch_prompts:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                    formatted.append(
                        self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    )

                inputs = self.tokenizer(
                    formatted,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ).to(self.model.device)
                input_len = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=num_samples,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                completions = []
                for output in outputs:
                    new_tokens = output[input_len:]
                    completions.append(
                        self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    )

                batch_samples = []
                for j, prompt in enumerate(batch_prompts):
                    start_idx = j * num_samples
                    end_idx = (j + 1) * num_samples
                    for completion in completions[start_idx:end_idx]:
                        sample = (prompt, completion)
                        all_samples.append(sample)
                        batch_samples.append(sample)
                        if cache_file:
                            cache_file.write(
                                json.dumps({"prompt": prompt, "completion": completion}) + "\n"
                            )
                if cache_file:
                    cache_file.flush()

                if (batch_idx + 1) % self.clear_cache_every_n_batches == 0:
                    empty_accelerator_cache()
                    gc.collect()
                pbar.set_postfix(samples=len(all_samples), refresh=True)
            pbar.close()
        finally:
            if cache_file:
                cache_file.close()

        elapsed = time.time() - start_time
        if log:
            log(
                f"Generated {len(all_samples)} samples in {elapsed/60:.1f} minutes",
                "success",
            )
        return all_samples


__all__ = [
    "RolloutGenerator",
    "TorchRolloutGenerator",
    "append_to_cache",
    "resume_from_cache",
]
