"""Backend-agnostic RAFT building blocks.

Lifted out of `halo_forge.rlvr.raft_trainer` so the MLX-native RAFT loop
(`mlx_raft_trainer.MLXRAFTTrainer`) can reuse verifier dispatch + reward
shaping + filtering without dragging the PyTorch trainer's torch / peft /
transformers dependencies.

Nothing here imports torch or mlx. The PyTorch RAFT trainer keeps its
public method signatures (`verify_and_filter`, etc.) and now delegates
internals here; the MLX trainer calls these helpers directly.

Design notes:
- `verify_and_filter_samples` is a free function — takes `(samples,
  verifier, config)` and returns `(filtered, stats, representative_examples)`.
  The trainer holds `representative_examples` as state for the recovery
  guidance hook; the function returns a fresh list each call so the trainer
  decides when to overwrite.
- `RAFTCycleSpec` is the minimal subset of `RAFTConfig` this module needs.
  We pass the full RAFTConfig through duck-typing rather than introducing a
  new dataclass — config attribute names match across backends.
"""

from __future__ import annotations

import gc
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple


class _VerifierLike(Protocol):
    """Minimum surface a verifier must expose for this module."""

    def verify_batch(self, completions: List[str], prompts: List[str]) -> List[Any]:
        ...


def verify_and_filter_samples(
    samples: List[Tuple[str, str]],
    verifier: _VerifierLike,
    *,
    chunk_size: int,
    reward_threshold: float,
    keep_top_percent: float,
    min_samples_per_cycle: Optional[int] = None,
    progress_logger: Optional[Callable[[str, str], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """Verify completions, sort by reward, and filter.

    Mirrors the PyTorch trainer's `verify_and_filter` exactly so loss/reward
    curves are comparable between backends. Chunked verification prevents
    memory exhaustion on large batches; the chunk size knob lives on
    `RAFTConfig.verification_chunk_size`.

    Args:
        samples: `[(prompt, completion), ...]`.
        verifier: any object exposing `verify_batch(completions, prompts) ->
            list of result objects with `.reward`, `.success`, `.details`.
        chunk_size: how many completions to verify per chunk.
        reward_threshold: drop everything below this reward.
        keep_top_percent: keep this fraction of the above-threshold set.
        min_samples_per_cycle: if set and the kept set is smaller, lower
            the effective threshold to grab the top-N highest-reward
            samples regardless of threshold (auto-adjust).
        progress_logger: optional `(msg, level)` callable for per-step
            warnings (e.g. threshold adjustment notes).

    Returns:
        `(filtered, stats, representative_examples)`. `filtered` is the
        kept subset; `stats` is the printable telemetry dict; `representative_examples`
        is a small set of failure / drop reasons useful for recovery
        guidance — empty list if nothing notable.
    """
    from tqdm import tqdm

    log = progress_logger or (lambda msg, level="info": None)

    print(f"\nVerifying {len(samples)} samples...")
    prompts = [s[0] for s in samples]
    completions = [s[1] for s in samples]

    results = []
    num_chunks = (len(completions) + chunk_size - 1) // chunk_size
    start_time = time.time()
    for i in tqdm(
        range(0, len(completions), chunk_size),
        desc="Verifying",
        unit="chunk",
        total=num_chunks,
    ):
        chunk_end = min(i + chunk_size, len(completions))
        chunk_prompts = prompts[i:chunk_end]
        chunk_completions = completions[i:chunk_end]
        chunk_results = verifier.verify_batch(chunk_completions, chunk_prompts)
        results.extend(chunk_results)
        gc.collect()
    elapsed = time.time() - start_time
    print(f"[OK] Verification completed in {elapsed:.1f}s")

    # Combine + sort
    all_data: List[Dict[str, Any]] = []
    for (prompt, completion), result in zip(samples, results):
        all_data.append({
            "prompt": prompt,
            "completion": completion,
            "reward": result.reward,
            "success": result.success,
            "details": result.details,
        })
    all_data.sort(key=lambda x: x["reward"], reverse=True)

    # Filter
    effective_threshold = reward_threshold
    above_threshold = [d for d in all_data if d["reward"] >= effective_threshold]
    keep_count = max(1, int(len(above_threshold) * keep_top_percent))
    filtered = above_threshold[:keep_count]

    threshold_adjusted = False
    if min_samples_per_cycle and len(filtered) < min_samples_per_cycle:
        log(
            f"Only {len(filtered)} samples kept, below minimum {min_samples_per_cycle}",
            "warning",
        )
        if len(all_data) >= min_samples_per_cycle:
            filtered = all_data[:min_samples_per_cycle]
            new_threshold = filtered[-1]["reward"]
            log(
                f"Auto-adjusted: taking top {min_samples_per_cycle} samples "
                f"(lowest reward: {new_threshold:.2f})",
                "dim",
            )
            effective_threshold = new_threshold
            above_threshold = filtered
            threshold_adjusted = True
        else:
            filtered = all_data
            log(f"Not enough samples, using all {len(all_data)}", "warning")

    stats = {
        "total_samples": len(samples),
        "above_threshold": len(above_threshold),
        "kept": len(filtered),
        "avg_reward": sum(d["reward"] for d in all_data) / len(all_data) if all_data else 0,
        "avg_kept_reward": (
            sum(d["reward"] for d in filtered) / len(filtered) if filtered else 0
        ),
        "success_rate": (
            sum(1 for d in all_data if d["success"]) / len(all_data) if all_data else 0
        ),
        "reward_distribution": {
            "0.0": sum(1 for d in all_data if d["reward"] < 0.2),
            "0.5": sum(1 for d in all_data if 0.4 <= d["reward"] < 0.6),
            "0.7": sum(1 for d in all_data if 0.6 <= d["reward"] < 0.9),
            "1.0": sum(1 for d in all_data if d["reward"] >= 0.9),
        },
        "threshold_adjusted": threshold_adjusted,
        "effective_threshold": effective_threshold,
    }

    representative_examples = _build_representative_examples(
        all_data=all_data,
        above_threshold=above_threshold,
        keep_count=keep_count,
        effective_threshold=effective_threshold,
    )

    return filtered, stats, representative_examples


def _build_representative_examples(
    *,
    all_data: List[Dict[str, Any]],
    above_threshold: List[Dict[str, Any]],
    keep_count: int,
    effective_threshold: float,
) -> List[Dict[str, Any]]:
    """Return a small set of failure / drop examples for recovery guidance.

    Three buckets, in priority order: verification failures, samples below
    threshold, and samples dropped by keep_top_percent. Caps at 3 per bucket
    and returns the first non-empty bucket — enough signal to debug, not so
    much that the recovery view becomes a wall of text.
    """
    failed_examples = [
        {
            "reason": "verification_failed",
            "label": "Verifier failure",
            "preview": d.get("completion", ""),
            "context": d.get("prompt", ""),
            "reward": d.get("reward"),
        }
        for d in all_data
        if not d.get("success")
    ][:3]
    threshold_examples = [
        {
            "reason": "below_reward_threshold",
            "label": "Below threshold",
            "preview": d.get("completion", ""),
            "context": d.get("prompt", ""),
            "reward": d.get("reward"),
        }
        for d in all_data
        if d.get("success") and d.get("reward", 0.0) < effective_threshold
    ][:3]
    keep_drop_examples = [
        {
            "reason": "dropped_by_keep_percent",
            "label": "Dropped by keep percent",
            "preview": d.get("completion", ""),
            "context": d.get("prompt", ""),
            "reward": d.get("reward"),
        }
        for d in above_threshold[keep_count:]
    ][:3]
    return failed_examples or threshold_examples or keep_drop_examples


def print_filtering_summary(stats: Dict[str, Any]) -> None:
    """Plain-text print so it works through pipes / log files."""
    print("\nFiltering results:")
    print(f"  Total: {stats['total_samples']}")
    print(f"  Above threshold ({stats['effective_threshold']:.2f}): {stats['above_threshold']}")
    print(f"  Kept (top {stats['kept']}): {stats['kept']}")
    print(f"  Avg reward: {stats['avg_reward']:.3f}")
    print(f"  Avg kept reward: {stats['avg_kept_reward']:.3f}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    if stats.get("threshold_adjusted"):
        print(f"  Threshold auto-adjusted to {stats['effective_threshold']:.2f}")


def filtered_to_jsonl_records(
    filtered: List[Dict[str, Any]],
    *,
    system_prompt: str = "",
) -> List[Dict[str, Any]]:
    """Convert filtered RAFT samples into chat-format SFT records.

    The MLX SFT trainer (Phase 4) consumes JSONL with `{"text": "..."}` or
    `{"messages": [...]}` records. We emit the messages form so the
    tokenizer's chat template renders correctly per model — same approach
    the PyTorch path uses internally.
    """
    records: List[Dict[str, Any]] = []
    for d in filtered:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": d["prompt"]})
        messages.append({"role": "assistant", "content": d["completion"]})
        records.append({"messages": messages})
    return records


__all__ = [
    "filtered_to_jsonl_records",
    "print_filtering_summary",
    "verify_and_filter_samples",
]
