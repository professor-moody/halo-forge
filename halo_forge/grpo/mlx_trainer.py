"""MLX-native GRPO trainer.

Track T2 + T17 v1.

Same scope discipline as MLX DPO: ship the reference-free variant
first; full reference-model GRPO needs dual-model memory and stays
roadmap. The algorithm path:

    1. Sample N completions per prompt via MLXRolloutGenerator.
    2. Score each via the requested verifier (Track V1 plugin registry).
    3. Compute group-relative advantages: a_i = (r_i - mean(group)) / std(group).
    4. Forward through policy to get response-token log-probs.
    5. Loss = -E[a · log π(completion|prompt)]; KL term skipped in
       reference-free mode (no π_ref to compute against).
    6. mlx grad + AdamW step.

Reuses building blocks from MLX DPO (`_response_logprobs`) so the
loss math stays in one place.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from halo_forge.grpo.config import GRPOConfig
from halo_forge.runtime_determinism import build_run_id, set_global_seed
from halo_forge.training_contracts import (
    attach_effectiveness_contract,
    build_cycle_summary,
    build_training_summary,
    write_json_atomic,
)
from halo_forge.training_recovery import attach_recovery_guidance
from halo_forge.utils.backend_config import warn_unsupported_for_mlx

logger = logging.getLogger(__name__)


def _require_mlx_lm() -> Dict[str, Any]:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        from mlx_lm import load as mlx_load
        from mlx_lm.tuner.utils import linear_to_lora_layers
    except ImportError as exc:
        raise ImportError(
            "MLX GRPO requires the `[mlx]` extra. Install with `pip install '.[mlx]'`."
        ) from exc

    return {
        "mx": mx, "nn": nn, "optim": optim,
        "tree_flatten": tree_flatten,
        "mlx_load": mlx_load,
        "linear_to_lora_layers": linear_to_lora_layers,
    }


def _group_advantages(rewards: List[float], *, scale_by_std: bool = True) -> List[float]:
    """Compute group-relative advantages.

    a_i = (r_i - mean) / std   when scale_by_std (canonical GRPO)
    a_i = (r_i - mean)         otherwise (RLOO-flavored)

    Group of size 1 → advantage=0 (no signal). Constant rewards within
    a group → std=0 → advantage=0 (avoid division-by-zero noise).
    """
    if not rewards:
        return []
    if len(rewards) == 1:
        return [0.0]
    mean = sum(rewards) / len(rewards)
    if not scale_by_std:
        return [r - mean for r in rewards]
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var)
    if std < 1e-8:
        return [0.0 for _ in rewards]
    return [(r - mean) / std for r in rewards]


class MLXGRPOTrainer:
    """Reference-free GRPO trainer on MLX (Apple Silicon).

    v1 contract:
      - reference_free=True required (no dual-model memory)
      - rollout via MLXRolloutGenerator (mlx_lm.generate)
      - verifier via the V1 plugin registry
      - single-cycle: rollout once, train once, save adapter

    Multi-cycle GRPO (alternating rollout / update phases) is the
    natural follow-up; this v1 ships single-cycle so the loss math +
    integration land before we layer on the cycle controller.
    """

    def __init__(self, config: Optional[GRPOConfig] = None):
        self.config = config or GRPOConfig()
        warn_unsupported_for_mlx(self.config, trainer_label="MLX GRPO")

        if not self.config.reference_free:
            raise NotImplementedError(
                "MLX GRPO v1 supports --reference-free only. Full reference-"
                "model GRPO on MLX is roadmap. Pass --reference-free, or run "
                "on a PyTorch backend."
            )

        self.model: Any = None
        self.tokenizer: Any = None
        self.run_id: str = ""
        self.training_summary: Dict[str, Any] = {}

    def train(
        self,
        train_file: Optional[str] = None,
        dataset: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        deps = _require_mlx_lm()
        mx = deps["mx"]
        nn = deps["nn"]
        cfg = self.config

        if dataset is not None:
            cfg.dataset = dataset
        if train_file is not None:
            cfg.train_file = train_file
        if resume_from_checkpoint is not None:
            logger.warning("MLX GRPO v1 ignores resume_from_checkpoint — fresh run.")

        cfg.seed = set_global_seed(cfg.seed)
        self.run_id = build_run_id("grpo_mlx")

        prompts = self._load_prompts(cfg.train_file, cfg.dataset, cfg.max_samples)
        if not prompts:
            raise ValueError("No prompts loaded — check --data / --dataset paths.")

        # 1. Rollouts via MLXRolloutGenerator.
        from halo_forge.rlvr.mlx_rollout import MLXRolloutGenerator

        rollout = MLXRolloutGenerator(cfg.model_name)
        logger.info("MLX GRPO: generating %d×%d completions", len(prompts), cfg.num_generations)
        samples = rollout.generate_samples(
            prompts,
            num_samples=cfg.num_generations,
            max_new_tokens=cfg.max_completion_length,
            temperature=cfg.temperature,
            batch_size=max(1, cfg.batch_size),
            system_prompt="",
        )

        # Group samples by prompt to compute group-relative advantages.
        # `samples` is [(prompt, completion), ...] in prompt-major order.
        groups: Dict[str, List[str]] = {}
        for prompt, completion in samples:
            groups.setdefault(prompt, []).append(completion)

        # 2. Score via the requested verifier.
        from halo_forge.rlvr.verifiers import get_verifier

        verifier = get_verifier(cfg.verifier_name)()

        scored: List[Tuple[str, str, float, float]] = []  # (prompt, completion, reward, advantage)
        for prompt, completions in groups.items():
            rewards: List[float] = []
            for c in completions:
                try:
                    result = verifier.verify(c)
                    rewards.append(float(result.reward))
                except Exception as exc:
                    logger.warning("Verifier raised on completion: %s", exc)
                    rewards.append(0.0)
            advantages = _group_advantages(rewards, scale_by_std=cfg.scale_rewards)
            for completion, reward, advantage in zip(completions, rewards, advantages):
                if reward < cfg.reward_threshold:
                    advantage = 0.0
                scored.append((prompt, completion, reward, advantage))

        # 3. Load model + apply LoRA for the policy update step.
        logger.info("Loading MLX model: %s", cfg.model_name)
        self.model, self.tokenizer = deps["mlx_load"](cfg.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.freeze()
        lora_cfg = {
            "rank": cfg.lora_r,
            "alpha": cfg.lora_alpha,
            "dropout": cfg.lora_dropout,
            "scale": float(cfg.lora_alpha) / float(cfg.lora_r) if cfg.lora_r else 1.0,
        }
        deps["linear_to_lora_layers"](
            self.model, num_layers=self._count_blocks() or 16, config=lora_cfg
        )

        optimizer = deps["optim"].AdamW(
            learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        # 4. Training loop.
        from halo_forge.dpo.mlx_trainer import _response_logprobs

        def loss_fn(model, prompt_tokens, completion_tokens, advantage):
            # GRPO policy gradient: -advantage * sum(log π(completion|prompt))
            logp = _response_logprobs(
                mx=mx, nn=nn, model=model,
                prompt_tokens=prompt_tokens, response_tokens=completion_tokens,
            )
            return -advantage * logp

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)

        loss_history: List[float] = []
        reward_history: List[float] = [r for _, _, r, _ in scored]
        advantage_history: List[float] = [a for _, _, _, a in scored]

        logger.info(
            "GRPO update: %d (prompt, completion) pairs, mean reward=%.3f",
            len(scored),
            sum(reward_history) / max(1, len(reward_history)),
        )

        t0 = time.time()
        step = 0
        for epoch in range(cfg.num_epochs):
            for prompt, completion, _reward, advantage in scored:
                if abs(advantage) < 1e-9:
                    continue  # skip zero-advantage samples — no signal

                prompt_ids = mx.array(self.tokenizer.encode(prompt))
                completion_ids = mx.array(self.tokenizer.encode(completion))
                if completion_ids.shape[0] == 0:
                    continue
                if (
                    prompt_ids.shape[0] + completion_ids.shape[0]
                    > cfg.max_prompt_length + cfg.max_completion_length
                ):
                    completion_ids = completion_ids[: cfg.max_completion_length]

                loss_value, grads = loss_and_grad(
                    self.model, prompt_ids, completion_ids, advantage
                )
                optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), optimizer.state)
                loss_history.append(float(loss_value))

                step += 1
                if step % cfg.logging_steps == 0:
                    avg_loss = sum(loss_history[-cfg.logging_steps:]) / max(
                        1, len(loss_history[-cfg.logging_steps:])
                    )
                    print(f"  step {step:>5}  loss={avg_loss:.4f}  advantage={advantage:.3f}")

        elapsed = time.time() - t0

        # 5. Save adapter.
        output_dir = Path(cfg.output_dir)
        adapter_dir = output_dir / "final_model"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        try:
            from mlx_lm.tuner.utils import save_adapter

            save_adapter(self.model, str(adapter_dir / "adapters.safetensors"))
        except (ImportError, AttributeError):
            tflat = deps["tree_flatten"](self.model.trainable_parameters())
            mx.save_safetensors(
                str(adapter_dir / "adapters.safetensors"),
                {k: v for k, v in tflat},
            )
        adapter_config = {
            "fine_tune_type": "lora",
            "num_layers": self._count_blocks() or 16,
            "lora_parameters": {
                "rank": cfg.lora_r,
                "alpha": cfg.lora_alpha,
                "dropout": cfg.lora_dropout,
                "scale": float(cfg.lora_alpha) / float(cfg.lora_r) if cfg.lora_r else 1.0,
            },
        }
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(adapter_config, indent=2)
        )

        summary = self._build_summary(
            loss_history=loss_history,
            reward_history=reward_history,
            advantage_history=advantage_history,
            scored_pairs=scored,
            duration_seconds=elapsed,
            n_prompts=len(prompts),
            train_file=cfg.train_file,
            dataset=cfg.dataset,
            final_output=adapter_dir,
        )
        write_json_atomic(output_dir / "training_summary.json", summary)
        self.training_summary = summary
        return summary

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_prompts(
        train_file: Optional[str],
        dataset: Optional[str],
        max_samples: Optional[int],
    ) -> List[str]:
        if not train_file and not dataset:
            raise ValueError("MLX GRPO requires --data (JSONL) or --dataset")
        if train_file:
            with open(train_file) as f:
                rows = [
                    json.loads(line)
                    for line in f
                    if line.strip()
                ]
            prompts = [
                str(row.get("prompt") or row.get("text") or row.get("question") or "")
                for row in rows
            ]
        else:
            from datasets import load_dataset

            ds = load_dataset(dataset, split="train")  # type: ignore[arg-type]
            for col in ("prompt", "text", "question", "instruction"):
                if col in ds.column_names:
                    prompts = [str(row[col]) for row in ds]
                    break
            else:
                raise ValueError(
                    f"Dataset {dataset!r} has no prompt-shaped column. "
                    "Expected one of: prompt / text / question / instruction"
                )
        prompts = [p for p in prompts if p]
        if max_samples:
            prompts = prompts[:max_samples]
        return prompts

    def _count_blocks(self) -> int:
        layers = getattr(self.model, "layers", None)
        if layers is None:
            inner = getattr(self.model, "model", None)
            if inner is not None:
                layers = getattr(inner, "layers", None)
        try:
            return len(layers) if layers is not None else 0
        except TypeError:
            return 0

    def _build_summary(
        self,
        *,
        loss_history,
        reward_history,
        advantage_history,
        scored_pairs,
        duration_seconds,
        n_prompts,
        train_file,
        dataset,
        final_output,
    ):
        cfg = self.config
        n_completions = len(scored_pairs)
        final_loss = loss_history[-1] if loss_history else None
        initial_loss = loss_history[0] if loss_history else final_loss
        avg_reward = (
            sum(reward_history) / max(1, len(reward_history)) if reward_history else None
        )

        cycle = build_cycle_summary(
            cycle=0,
            learning_rate=cfg.learning_rate,
            samples_seen=n_completions,
            samples_kept=n_completions,
            cycle_duration_seconds=duration_seconds,
            update_metrics={
                "train_steps_executed": len(loss_history),
                "train_loss": final_loss,
                "initial_train_loss": initial_loss,
                "weights_updated": len(loss_history) > 0,
                "update_reason": "updated" if loss_history else "no_optimizer_steps",
                "optimizer_steps": len(loss_history),
                "skipped_batches_non_finite": 0,
            },
            yield_diagnostics={
                "stage_counts": None, "rates": None,
                "minimums": {"minimum_samples_target": max(1, n_prompts)},
                "rejection_reasons": None, "summary": None,
            },
            extra={
                "n_prompts": n_prompts,
                "n_completions": n_completions,
                "num_generations": cfg.num_generations,
                "avg_reward": avg_reward,
                "beta": cfg.beta,
                "verifier": cfg.verifier_name,
                "reference_free": True,
            },
        )
        summary = build_training_summary(
            modality="grpo",
            model_name=cfg.model_name,
            total_cycles_planned=1,
            cycles=[cycle],
            run_id=self.run_id,
            seed=cfg.seed,
            base_model_name=cfg.model_name,
            active_model_name=cfg.model_name,
            yield_diagnostics={
                "stage_counts": None, "rates": None,
                "minimums": {"minimum_samples_target": max(1, n_prompts)},
                "rejection_reasons": None, "summary": None,
            },
            extra={
                "dataset": dataset or "",
                "train_file": train_file or "",
                "n_prompts": n_prompts,
                "n_completions": n_completions,
                "num_generations": cfg.num_generations,
                "beta": cfg.beta,
                "epsilon": cfg.epsilon,
                "temperature": cfg.temperature,
                "verifier": cfg.verifier_name,
                "reference_free": True,
                "backend": "mlx",
                "reward_history": reward_history,
                "advantage_history": advantage_history,
            },
        )
        summary["final_model_path"] = str(final_output)
        attach_effectiveness_contract(
            summary,
            minimum_samples_kept=max(1, n_prompts),
            minimum_optimizer_steps=1,
            evaluation={
                "metric_name": "avg_reward",
                "baseline_value": None,
                "final_value": avg_reward,
                "higher_is_better": True,
                "tolerance": 0.0,
            },
            evaluation_required=False,
            checkpoint_written=True,
            final_model_path=str(final_output),
            training_summary_path=Path(cfg.output_dir) / "training_summary.json",
        )
        attach_recovery_guidance(
            summary,
            modality="grpo",
            launch_args={
                "model": cfg.model_name,
                "dataset": dataset or "",
                "output_dir": cfg.output_dir,
                "num_generations": cfg.num_generations,
                "beta": cfg.beta,
                "verifier": cfg.verifier_name,
                "reference_free": True,
            },
            representative_examples=[],
        )
        return summary


__all__ = ["MLXGRPOTrainer", "_group_advantages"]
