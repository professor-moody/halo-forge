"""MLX-native GRPO trainer.

Track T2 + T17.

Same scope discipline as MLX DPO: keep the reference-free path small and
explicit, and route the reference-model path through the existing
``reference_free=False`` config. The algorithm path:

    1. Sample N completions per prompt via MLXRolloutGenerator.
    2. Score each via the requested verifier (Track V1 plugin registry).
    3. Compute group-relative advantages: a_i = (r_i - mean(group)) / std(group).
    4. Forward through the freshly-adapted policy once per kept sample to record
       the rollout-time log-probs (log π_old) used by the importance ratio.
    5. Forward through policy to get response-token log-probs, then
       loss = -min(ρ·a, clip(ρ, 1-ε, 1+ε)·a) with ρ = exp(log π - log π_old).
       Reference-model mode adds β · KL_k3, the unbiased low-variance estimator
       KL ≈ exp(log π_ref - log π) - (log π_ref - log π) - 1, which is ≥ 0,
       zero only at log π = log π_ref, and convex in the log-ratio.
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
from halo_forge.training_signal import complete_signal_boundary
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


def _group_prompt_occurrences(
    prompts: List[str],
    samples: List[Tuple[str, str]],
    *,
    num_generations: int,
) -> List[Tuple[int, str, List[str]]]:
    """Group prompt-major rollouts by source occurrence, not prompt text.

    Equal prompt strings can represent different dataset records.  A dict
    keyed by prompt silently merged those records and changed GRPO advantage
    groups.  The rollout contract is prompt-major and fixed-width, so preserve
    that occurrence boundary and fail truthfully if a backend violates it.
    """

    width = int(num_generations)
    if width < 1:
        raise ValueError("num_generations must be positive")
    expected = len(prompts) * width
    if len(samples) != expected:
        raise ValueError(
            f"MLX GRPO rollout returned {len(samples)} samples; expected {expected} "
            f"({len(prompts)} prompts x {width} generations)"
        )
    groups: List[Tuple[int, str, List[str]]] = []
    for source_index, prompt in enumerate(prompts):
        start = source_index * width
        observed = samples[start : start + width]
        mismatches = [value for value, _ in observed if value != prompt]
        if mismatches:
            raise ValueError(
                "MLX GRPO rollout order did not preserve prompt occurrences; "
                "cannot compute truthful group-relative advantages"
            )
        groups.append((source_index, prompt, [completion for _, completion in observed]))
    return groups


class _ScalarOps:
    """Array-op namespace for plain Python floats.

    The loss math below is written against an ``ops`` namespace (``exp``,
    ``minimum``, ``clip``) so the same code runs on ``mlx.core`` arrays inside
    the trainer and on floats / numpy arrays inside tests, without importing
    mlx on hosts that do not have it.
    """

    exp = staticmethod(math.exp)
    minimum = staticmethod(min)

    @staticmethod
    def clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))


_SCALAR_OPS = _ScalarOps()


def _default_ops(value: Any) -> Any:
    """Pick the array namespace matching ``value`` when the caller omits ``ops``.

    Python floats (and numpy scalars, which subclass ``float``) use plain math;
    ``mlx.core`` / numpy arrays route back to their own module so callers that
    never think about ``ops`` — e.g. the MLX measurement scripts — still work.
    """
    if isinstance(value, (int, float)):
        return _SCALAR_OPS
    module_root = type(value).__module__.split(".")[0]
    if module_root == "mlx":
        import mlx.core as mx

        return mx
    if module_root == "numpy":
        import numpy as np

        return np
    return _SCALAR_OPS


def _grpo_reference_kl(policy_logp: Any, reference_logp: Any, *, ops: Any = None) -> Any:
    """Return the k3 estimator of KL(π ‖ π_ref) for one sequence.

    ``k3`` is the estimator used by DeepSeek's GRPO and by TRL:

        r  = log π_ref - log π
        kl = exp(r) - r - 1

    It is non-negative, exactly zero at ``log π == log π_ref``, and convex in
    ``r``, so it supplies a genuine restoring gradient toward the reference.
    The previous signed log-ratio (``log π - log π_ref``) was linear: it gave a
    constant gradient with no restoring force and went negative — i.e. it *paid*
    the policy — whenever the policy fell below the reference.
    """
    ops = ops if ops is not None else _default_ops(policy_logp)
    log_ratio = reference_logp - policy_logp
    return ops.exp(log_ratio) - log_ratio - 1.0


def _clip_binds(ratio: float, advantage: float, epsilon: Optional[float]) -> bool:
    """Return True when the PPO clip actually changed the surrogate.

    ``loss = -min(ρ·A, clip(ρ, 1-ε, 1+ε)·A)`` takes the clipped branch only when
    the clipped term is the pessimistic one: for ``A > 0`` that needs
    ``ρ > 1+ε``, and for ``A < 0`` it needs ``ρ < 1-ε``.  The other two
    quadrants leave ``ρ`` outside the trust region yet still select the
    unclipped branch, so ``|ρ - 1| > ε`` would overstate how often clipping bit.
    """
    if epsilon is None or epsilon <= 0.0 or advantage == 0.0:
        return False
    if advantage > 0.0:
        return ratio > 1.0 + epsilon
    return ratio < 1.0 - epsilon


def _grpo_policy_loss(
    policy_logp: Any,
    advantage: float,
    beta: float,
    reference_logp: Any = None,
    old_policy_logp: Any = None,
    epsilon: Optional[float] = None,
    ops: Any = None,
) -> Any:
    """Return the scalar GRPO policy loss for one prompt/completion pair.

    With ``old_policy_logp`` (the log-prob under the policy that produced the
    rollout) the surrogate is the PPO-style clipped objective

        ρ    = exp(log π - log π_old)
        loss = -min(ρ·A, clip(ρ, 1-ε, 1+ε)·A)

    which bounds how far one update can move the policy per rollout batch.
    Without it — the single-sample / measurement path — the surrogate degrades
    to the plain policy gradient ``-A · log π`` (equivalent to ρ ≡ 1).

    ``reference_logp`` adds ``beta ·`` the k3 KL penalty (see
    :func:`_grpo_reference_kl`). ``ops`` selects the array namespace and
    defaults to the namespace matching ``policy_logp``; the trainer passes
    ``mlx.core`` explicitly.
    """
    ops = ops if ops is not None else _default_ops(policy_logp)
    if old_policy_logp is None:
        loss = -advantage * policy_logp
    else:
        ratio = ops.exp(policy_logp - old_policy_logp)
        if epsilon is None or epsilon <= 0.0:
            loss = -(ratio * advantage)
        else:
            clipped = ops.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
            loss = -ops.minimum(ratio * advantage, clipped * advantage)
    if reference_logp is not None:
        loss = loss + beta * _grpo_reference_kl(policy_logp, reference_logp, ops=ops)
    return loss


class MLXGRPOTrainer:
    """GRPO trainer on MLX (Apple Silicon).

    Contract:
      - reference_free=True skips the KL/reference-model term
      - reference_free=False loads a frozen reference model and applies
        β · KL_k3(π ‖ π_ref) — non-negative and convex in the log-ratio
      - epsilon clips the importance ratio against the rollout-time policy,
        matching the TRL torch path's clipped surrogate
      - rollout via MLXRolloutGenerator (mlx_lm.generate)
      - verifier via the V1 plugin registry
      - single-cycle: rollout once, train once, save adapter

    Multi-cycle GRPO (alternating rollout / update phases) is the
    natural follow-up; this v1 ships single-cycle so the loss math +
    integration land before we layer on the cycle controller.
    """

    def __init__(self, config: Optional[GRPOConfig] = None, signal_sink: Optional[Any] = None):
        self.config = config or GRPOConfig()
        self.signal_sink = signal_sink
        warn_unsupported_for_mlx(self.config, trainer_label="MLX GRPO")
        from halo_forge.utils.neural_accelerators import validate_neural_accelerator_opt_in

        validate_neural_accelerator_opt_in(self.config, logger=logger, label="MLX GRPO")

        self.model: Any = None
        self.reference_model: Any = None
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
        self.reference_model = None

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
        rollout.cleanup()

        # Group samples by prompt to compute group-relative advantages.
        # `samples` is [(prompt, completion), ...] in prompt-major order.
        groups = _group_prompt_occurrences(
            prompts,
            samples,
            num_generations=cfg.num_generations,
        )

        # 2. Score via the requested verifier.
        from halo_forge.rlvr.verifiers import get_verifier

        verifier = get_verifier(cfg.verifier_name)()

        scored: List[Tuple[str, str, float, float]] = []  # (prompt, completion, reward, advantage)
        for source_index, prompt, completions in groups:
            rewards: List[float] = []
            observations: List[Any] = []
            for c in completions:
                try:
                    result = verifier.verify(c)
                    rewards.append(float(result.reward))
                    observations.append(result)
                except Exception as exc:
                    logger.warning("Verifier raised on completion: %s", exc)
                    rewards.append(0.0)
                    observations.append(
                        {
                            "reward": 0.0,
                            "success": False,
                            "error": str(exc),
                            "details": "verification_exception",
                        }
                    )
            advantages = _group_advantages(rewards, scale_by_std=cfg.scale_rewards)
            for candidate_ordinal, (completion, reward, advantage, observation) in enumerate(
                zip(completions, rewards, advantages, observations)
            ):
                if reward < cfg.reward_threshold:
                    advantage = 0.0
                if self.signal_sink is not None:
                    selected = reward >= cfg.reward_threshold and abs(advantage) >= 1e-9
                    self.signal_sink.capture(
                        record=None,
                        source_index=source_index,
                        source={"trainer": "grpo", "backend": "mlx"},
                        candidate_ordinal=candidate_ordinal,
                        occurrence_id=(
                            f"rollout:0:source:{source_index}:"
                            f"candidate:{candidate_ordinal}"
                        ),
                        prompt=prompt,
                        context={"advantage": advantage},
                        output=completion,
                        expected=None,
                        training_observation=observation,
                        selected=selected,
                        selection_reason=(
                            "used_for_policy_update"
                            if selected
                            else (
                                "below_reward_threshold"
                                if reward < cfg.reward_threshold
                                else "zero_group_advantage"
                            )
                        ),
                        generation_settings={
                            "temperature": cfg.temperature,
                            "max_new_tokens": cfg.max_completion_length,
                            "num_generations": cfg.num_generations,
                        },
                        producer_model_hash=cfg.model_name,
                    )
                scored.append((prompt, completion, reward, advantage))

        # 3. Load model + apply LoRA for the policy update step.
        logger.info("Loading MLX model: %s", cfg.model_name)
        self.model, self.tokenizer = deps["mlx_load"](cfg.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if not cfg.reference_free:
            logger.info("Loading frozen MLX reference model: %s", cfg.model_name)
            self.reference_model, _ = deps["mlx_load"](cfg.model_name)
            self.reference_model.freeze()

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

        clip_epsilon = float(cfg.epsilon) if cfg.epsilon and cfg.epsilon > 0 else None

        def loss_fn(model, prompt_tokens, completion_tokens, advantage, old_logp):
            policy_logp = _response_logprobs(
                mx=mx, nn=nn, model=model,
                prompt_tokens=prompt_tokens, response_tokens=completion_tokens,
            )
            reference_logp = None
            kl = mx.array(0.0)
            if self.reference_model is not None:
                reference_logp = _response_logprobs(
                    mx=mx, nn=nn, model=self.reference_model,
                    prompt_tokens=prompt_tokens, response_tokens=completion_tokens,
                )
                kl = _grpo_reference_kl(policy_logp, reference_logp, ops=mx)
            loss = _grpo_policy_loss(
                policy_logp,
                advantage=float(advantage),
                beta=cfg.beta,
                reference_logp=reference_logp,
                old_policy_logp=old_logp,
                epsilon=clip_epsilon,
                ops=mx,
            )
            return loss, (
                policy_logp,
                reference_logp if reference_logp is not None else policy_logp,
                kl,
                mx.exp(policy_logp - old_logp),
            )

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)

        loss_history: List[float] = []
        kl_history: List[float] = []
        clipped_updates = 0
        reward_history: List[float] = [r for _, _, r, _ in scored]
        advantage_history: List[float] = [a for _, _, _, a in scored]

        logger.info(
            "GRPO update: %d (prompt, completion) pairs, mean reward=%.3f",
            len(scored),
            sum(reward_history) / max(1, len(reward_history)),
        )

        # Record log π_old once, before any optimizer step. LoRA is zero-init,
        # so the freshly adapted policy is still numerically the rollout policy;
        # every later update is scored against these fixed rollout log-probs, so
        # the importance ratio is real (it is only identically 1 for the very
        # first update of the batch).
        batch: List[Tuple[Any, Any, float, Any]] = []
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

            old_logp = _response_logprobs(
                mx=mx, nn=nn, model=self.model,
                prompt_tokens=prompt_ids, response_tokens=completion_ids,
            )
            mx.eval(old_logp)
            batch.append(
                (prompt_ids, completion_ids, float(advantage), mx.stop_gradient(old_logp))
            )

        t0 = time.time()
        step = 0
        for epoch in range(cfg.num_epochs):
            for prompt_ids, completion_ids, advantage, old_logp in batch:
                (loss_value, aux), grads = loss_and_grad(
                    self.model, prompt_ids, completion_ids, advantage, old_logp
                )
                optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), optimizer.state)
                loss_history.append(float(loss_value))
                _policy_logp, _reference_logp, kl_value, ratio_value = aux
                kl_history.append(float(kl_value))
                if _clip_binds(float(ratio_value), advantage, clip_epsilon):
                    clipped_updates += 1

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
        complete_signal_boundary(
            self.signal_sink,
            boundary_value="final",
            checkpoint_path=adapter_dir,
        )

        summary = self._build_summary(
            loss_history=loss_history,
            kl_history=kl_history,
            reward_history=reward_history,
            advantage_history=advantage_history,
            scored_pairs=scored,
            duration_seconds=elapsed,
            n_prompts=len(prompts),
            train_file=cfg.train_file,
            dataset=cfg.dataset,
            final_output=adapter_dir,
            clipped_updates=clipped_updates,
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
        kl_history,
        reward_history,
        advantage_history,
        scored_pairs,
        duration_seconds,
        n_prompts,
        train_file,
        dataset,
        final_output,
        clipped_updates=0,
    ):
        cfg = self.config
        n_completions = len(scored_pairs)
        n_updates = len(loss_history)
        clip_enabled = bool(cfg.epsilon and cfg.epsilon > 0)
        final_loss = loss_history[-1] if loss_history else None
        initial_loss = loss_history[0] if loss_history else final_loss
        avg_reward = (
            sum(reward_history) / max(1, len(reward_history)) if reward_history else None
        )
        avg_kl = sum(kl_history) / max(1, len(kl_history)) if kl_history else None
        final_kl = kl_history[-1] if kl_history else None

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
                "avg_kl_final": final_kl,
                "kl_estimator": "k3",
                "beta": cfg.beta,
                "verifier": cfg.verifier_name,
                "reference_free": cfg.reference_free,
                "reference_model_loaded": not cfg.reference_free,
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
                "ratio_clipping_applied": clip_enabled,
                "clipped_updates": int(clipped_updates),
                "clipped_update_fraction": (
                    int(clipped_updates) / n_updates if n_updates else None
                ),
                "temperature": cfg.temperature,
                "verifier": cfg.verifier_name,
                "reference_free": cfg.reference_free,
                "reference_model_loaded": not cfg.reference_free,
                "backend": "mlx",
                "reward_history": reward_history,
                "advantage_history": advantage_history,
                "kl_history": kl_history,
                "kl_estimator": "k3",
                "avg_kl_final": final_kl,
                "avg_kl": avg_kl,
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
                "reference_free": cfg.reference_free,
            },
            representative_examples=[],
        )
        return summary


__all__ = [
    "MLXGRPOTrainer",
    "_group_advantages",
    "_grpo_policy_loss",
    "_grpo_reference_kl",
    "_clip_binds",
    "_group_prompt_occurrences",
]
