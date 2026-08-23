"""MLX-native DPO trainer (Track T1 / T17).

Closes the silent-failure gap from the May 2026 audit: until now
``halo-forge dpo train --accelerator mlx`` raised NotImplementedError.
This ships **reference-free DPO** (`--reference-free`) and the canonical
reference-model sigmoid variant. The reference-model path loads a second
frozen copy of the base model, so it is explicit and still limited to
``loss_type="sigmoid"`` until memory measurements justify more variants.

Why a custom loop instead of wrapping ``mlx_lm.tuner.train``: DPO's
loss is structurally different from SFT's cross-entropy. SFT computes
a per-token loss over the whole sequence; DPO computes per-pair
logp(chosen) - logp(rejected) over only the *response* tokens (prompt
tokens are masked) and combines them through DPO-family preference losses.
``mlx_lm.tuner``
isn't shaped for that — we own the loop here so the loss math is
explicit and inspectable.

Algorithm (reference-free DPO):

    pi_chosen   = sum log p_θ(yw | x)   over response tokens
    pi_rejected = sum log p_θ(yl | x)   over response tokens
    loss        = -log σ(β · (pi_chosen − pi_rejected))

Pairs map directly to TRL's ``loss_type`` choices for ``sigmoid``, ``ipo``,
``hinge``, and ``kto_pair``. cDPO label smoothing is supported for sigmoid.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from halo_forge.dpo.config import DPOConfig
from halo_forge.runtime_determinism import build_run_id, set_global_seed
from halo_forge.training_contracts import (
    attach_effectiveness_contract,
    build_cycle_summary,
    build_training_summary,
    write_json_atomic,
)
from halo_forge.training_recovery import attach_recovery_guidance
from halo_forge.utils.backend_config import (
    build_mlx_lr_schedule,
    build_mlx_optimizer,
    detect_mlx_capabilities,
    mlx_training_knob_gaps,
    resolve_mlx_step_budget,
    resolve_mlx_warmup_steps,
    warn_unsupported_for_mlx,
)

logger = logging.getLogger(__name__)

SUPPORTED_MLX_DPO_LOSS_TYPES = {"sigmoid", "ipo", "hinge", "kto_pair"}


def _require_mlx_lm() -> Dict[str, Any]:
    """Lazy MLX import so this module loads on non-MLX hosts."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        from mlx_lm import load as mlx_load
        from mlx_lm.tuner.utils import linear_to_lora_layers
    except ImportError as exc:
        raise ImportError(
            "MLX DPO requires the `[mlx]` extra. Install with `pip install '.[mlx]'`."
        ) from exc

    return {
        "mx": mx,
        "nn": nn,
        "optim": optim,
        "tree_flatten": tree_flatten,
        "mlx_load": mlx_load,
        "linear_to_lora_layers": linear_to_lora_layers,
    }


def _response_logprobs(
    *,
    mx: Any,
    nn: Any,
    model: Any,
    prompt_tokens: Any,
    response_tokens: Any,
) -> Any:
    """Sum log-probs of ``response_tokens`` under ``model`` given
    ``prompt_tokens`` as the conditioning context.

    Implements the DPO per-token logp computation. Returns a scalar
    (per-batch-element) suitable for direct subtraction in the loss.

    We concatenate prompt + response, run a single forward pass, and
    gather the log-probs at the response positions only — this matches
    TRL's `concatenated_forward` shape on a one-row batch and is the
    simplest correct implementation.
    """
    # Concatenate along sequence axis. Both tensors have shape (T,).
    full = mx.concatenate([prompt_tokens, response_tokens], axis=0)
    # Shift: model predicts token t+1 from positions 0..t, so the
    # logits at position t correspond to token at position t+1.
    inputs = full[None, :-1]  # (1, T-1)
    targets = full[1:]  # (T-1,)
    logits = model(inputs)  # (1, T-1, V)
    # Restrict to the slice covering response tokens. The first response
    # token is predicted at position len(prompt) - 1 in the *input* sequence.
    response_start = prompt_tokens.shape[0] - 1
    response_logits = logits[0, response_start:, :]  # (R, V)
    response_targets = targets[response_start:]  # (R,)
    log_probs = nn.log_softmax(response_logits, axis=-1)
    # Gather the log-prob of the actual target token at each position.
    gathered = mx.take_along_axis(log_probs, response_targets[:, None], axis=-1)
    return gathered.sum()


def _sigmoid_dpo_loss(
    *,
    mx: Any,
    nn: Any,
    chosen_logp: Any,
    rejected_logp: Any,
    beta: float,
    label_smoothing: float = 0.0,
    reference_chosen_logp: Any = None,
    reference_rejected_logp: Any = None,
) -> Any:
    """Standard sigmoid DPO loss with optional cDPO label smoothing.

    In reference-free mode π_ref drops out. When reference log-probs are
    provided, the margin is:

    ``(π_chosen − π_rejected) − (ref_chosen − ref_rejected)``.
    """
    logits = beta * _dpo_margin(
        chosen_logp=chosen_logp,
        rejected_logp=rejected_logp,
        reference_chosen_logp=reference_chosen_logp,
        reference_rejected_logp=reference_rejected_logp,
    )
    if label_smoothing > 0:
        # cDPO: assume label is correct with prob (1-eps), wrong with
        # prob eps. Loss is the convex combination.
        return (
            -(1 - label_smoothing) * nn.log_sigmoid(logits)
            - label_smoothing * nn.log_sigmoid(-logits)
        )
    return -nn.log_sigmoid(logits)


def _mlx_sigmoid(mx: Any, value: Any) -> Any:
    return 1.0 / (1.0 + mx.exp(-value))


def _mlx_relu(mx: Any, value: Any) -> Any:
    return mx.maximum(mx.array(0.0), value)


def _dpo_logratios(
    *,
    chosen_logp: Any,
    rejected_logp: Any,
    reference_chosen_logp: Any = None,
    reference_rejected_logp: Any = None,
) -> tuple[Any, Any]:
    if reference_chosen_logp is None and reference_rejected_logp is None:
        return chosen_logp, rejected_logp
    if reference_chosen_logp is None or reference_rejected_logp is None:
        raise ValueError("both reference_chosen_logp and reference_rejected_logp are required")
    return chosen_logp - reference_chosen_logp, rejected_logp - reference_rejected_logp


def _dpo_preference_loss(
    *,
    mx: Any,
    nn: Any,
    loss_type: str,
    chosen_logp: Any,
    rejected_logp: Any,
    beta: float,
    label_smoothing: float = 0.0,
    reference_chosen_logp: Any = None,
    reference_rejected_logp: Any = None,
    chosen_length: int | None = None,
    rejected_length: int | None = None,
) -> Any:
    """Return a scalar DPO-family loss for a single chosen/rejected pair."""
    if loss_type == "sigmoid":
        return _sigmoid_dpo_loss(
            mx=mx,
            nn=nn,
            chosen_logp=chosen_logp,
            rejected_logp=rejected_logp,
            beta=beta,
            label_smoothing=label_smoothing,
            reference_chosen_logp=reference_chosen_logp,
            reference_rejected_logp=reference_rejected_logp,
        )

    chosen_logratio, rejected_logratio = _dpo_logratios(
        chosen_logp=chosen_logp,
        rejected_logp=rejected_logp,
        reference_chosen_logp=reference_chosen_logp,
        reference_rejected_logp=reference_rejected_logp,
    )
    delta = chosen_logratio - rejected_logratio

    if loss_type == "hinge":
        return _mlx_relu(mx, 1.0 - beta * delta)

    if loss_type == "ipo":
        if chosen_length is not None:
            chosen_logratio = chosen_logratio / max(1, int(chosen_length))
        if rejected_length is not None:
            rejected_logratio = rejected_logratio / max(1, int(rejected_length))
        ipo_delta = chosen_logratio - rejected_logratio
        target = 1.0 / (2.0 * beta)
        centered = ipo_delta - target
        return centered * centered

    if loss_type == "kto_pair":
        kl = _mlx_relu(mx, (chosen_logratio + rejected_logratio) / 2.0)
        chosen_loss = 1.0 - _mlx_sigmoid(mx, beta * (chosen_logratio - kl))
        rejected_loss = 1.0 - _mlx_sigmoid(mx, beta * (kl - rejected_logratio))
        return (chosen_loss + rejected_loss) / 2.0

    raise NotImplementedError(
        f"MLX DPO supports loss_type values {sorted(SUPPORTED_MLX_DPO_LOSS_TYPES)}; got {loss_type!r}."
    )


def _accumulate_gradients(accumulated: Any, gradients: Any) -> Any:
    """Add ``gradients`` into ``accumulated``, elementwise over the tree.

    Deliberately written against ``+`` rather than ``mlx.utils.tree_map`` so
    the accumulation logic is testable on plain numbers without importing
    mlx. ``accumulated=None`` (first micro batch) returns ``gradients``.
    """
    if accumulated is None:
        return gradients
    if isinstance(gradients, dict):
        return {
            key: _accumulate_gradients(accumulated[key], value)
            for key, value in gradients.items()
        }
    if isinstance(gradients, (list, tuple)):
        combined = [
            _accumulate_gradients(prev, value)
            for prev, value in zip(accumulated, gradients)
        ]
        return type(gradients)(combined)
    return accumulated + gradients


def _scale_gradients(gradients: Any, factor: float) -> Any:
    """Multiply every leaf of a gradient tree by ``factor``."""
    if isinstance(gradients, dict):
        return {key: _scale_gradients(value, factor) for key, value in gradients.items()}
    if isinstance(gradients, (list, tuple)):
        return type(gradients)([_scale_gradients(value, factor) for value in gradients])
    return gradients * factor


def _dpo_margin(
    *,
    chosen_logp: Any,
    rejected_logp: Any,
    reference_chosen_logp: Any = None,
    reference_rejected_logp: Any = None,
) -> Any:
    """Return the policy-vs-reference preference margin.

    The helper is intentionally scalar/array-polymorphic so tests can
    exercise the math without importing MLX.
    """
    policy_margin = chosen_logp - rejected_logp
    if reference_chosen_logp is None and reference_rejected_logp is None:
        return policy_margin
    if reference_chosen_logp is None or reference_rejected_logp is None:
        raise ValueError("both reference_chosen_logp and reference_rejected_logp are required")
    return policy_margin - (reference_chosen_logp - reference_rejected_logp)


class MLXDPOTrainer:
    """DPO trainer on MLX (Apple Silicon).

    Currently supports ``loss_type`` values in
    ``SUPPORTED_MLX_DPO_LOSS_TYPES`` in reference-free mode or with a
    frozen reference model.
    """

    def __init__(self, config: Optional[DPOConfig] = None):
        self.config = config or DPOConfig()
        warn_unsupported_for_mlx(self.config, trainer_label="MLX DPO")
        from halo_forge.utils.neural_accelerators import validate_neural_accelerator_opt_in

        validate_neural_accelerator_opt_in(self.config, logger=logger, label="MLX DPO")

        if self.config.loss_type not in SUPPORTED_MLX_DPO_LOSS_TYPES:
            raise NotImplementedError(
                f"MLX DPO supports loss_type values {sorted(SUPPORTED_MLX_DPO_LOSS_TYPES)}; "
                f"got {self.config.loss_type!r}."
            )
        if self.config.loss_type != "sigmoid" and self.config.label_smoothing:
            raise NotImplementedError(
                "MLX DPO label_smoothing is supported for loss_type='sigmoid' only; "
                f"got loss_type={self.config.loss_type!r} with label_smoothing={self.config.label_smoothing}."
            )

        self.model: Any = None
        self.reference_model: Any = None
        self.tokenizer: Any = None
        self.run_id: str = ""
        self.training_summary: Dict[str, Any] = {}
        # Training-critical knobs as actually applied — see MLXSFTTrainer.
        self.backend_config_applied: Dict[str, Any] = {}

    def train(
        self,
        train_file: Optional[str] = None,
        dataset: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        from halo_forge.dpo.datasets import load_preference_dataset

        deps = _require_mlx_lm()
        mx = deps["mx"]
        nn = deps["nn"]
        cfg = self.config

        if dataset is not None:
            cfg.dataset = dataset
        if train_file is not None:
            cfg.train_file = train_file
        if resume_from_checkpoint is not None:
            logger.warning("MLX DPO v1 ignores resume_from_checkpoint — fresh run.")

        cfg.seed = set_global_seed(cfg.seed)
        self.run_id = build_run_id("dpo_mlx")

        train_ds, val_ds = load_preference_dataset(
            train_file=cfg.train_file,
            validation_file=cfg.validation_file,
            dataset=cfg.dataset,
            split=cfg.dataset_split,
            max_samples=cfg.max_samples,
            validation_split=cfg.validation_split,
            seed=cfg.seed,
        )

        print(f"Loading MLX policy model: {cfg.model_name}")
        self.model, self.tokenizer = deps["mlx_load"](cfg.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if not cfg.reference_free:
            print(f"Loading frozen MLX reference model: {cfg.model_name}")
            self.reference_model, _ = deps["mlx_load"](cfg.model_name)
            self.reference_model.freeze()

        # Apply LoRA (mlx_lm freeze-then-swap convention).
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

        # Iterate by example: DPO batches are tricky to pad due to the
        # variable prompt+response lengths; v1 keeps the loop sequential
        # for clarity. Batched loss is a follow-up perf win. Gradient
        # accumulation is what recovers the configured *effective* batch
        # size on top of that one-pair-at-a-time loop.
        rows = list(train_ds)

        # This loop is ours, so accumulation is always available; the mlx
        # optimizer module still decides whether warmup / clipping are.
        capabilities = replace(
            detect_mlx_capabilities(optim_module=deps["optim"]),
            grad_accumulation=True,
        )
        knob_gaps = mlx_training_knob_gaps(capabilities)
        if knob_gaps:
            warn_unsupported_for_mlx(cfg, trainer_label="MLX DPO", fields=knob_gaps)

        grad_accum = max(1, int(cfg.gradient_accumulation_steps or 1))
        micro_budget, planned_optimizer_steps = resolve_mlx_step_budget(
            micro_batches_per_epoch=max(1, len(rows)),
            num_epochs=cfg.num_epochs,
            gradient_accumulation_steps=grad_accum,
            max_steps=cfg.max_steps,
        )
        warmup_steps = resolve_mlx_warmup_steps(
            cfg, capabilities, optimizer_steps=planned_optimizer_steps
        )

        optimizer = build_mlx_optimizer(
            deps["optim"],
            learning_rate=build_mlx_lr_schedule(
                deps["optim"],
                learning_rate=cfg.learning_rate,
                warmup_steps=warmup_steps,
                total_steps=planned_optimizer_steps,
            ),
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm,
        )
        # The loop below scores one pair at a time, so `batch_size` is not
        # applied at all: the effective batch is the accumulation factor
        # alone. Say so rather than echoing the requested value back.
        requested_batch_size = int(getattr(cfg, "batch_size", 1) or 1)
        if requested_batch_size > 1:
            line = (
                f"[MLX DPO] Config field 'batch_size'={requested_batch_size!r} is not "
                "honored on the MLX backend. The MLX DPO loop scores one preference "
                f"pair at a time, so the effective batch is {grad_accum} pair(s) "
                f"(gradient_accumulation_steps), not {requested_batch_size * grad_accum}. "
                "Raise --gradient-accumulation to grow the effective batch."
            )
            logger.warning(line)
            print(f"WARNING: {line}")

        self.backend_config_applied = {
            "backend": "mlx",
            "batch_size": 1,
            "batch_size_requested": requested_batch_size,
            "effective_batch_size": grad_accum,
            "gradient_accumulation_steps": grad_accum,
            "gradient_accumulation_steps_requested": int(cfg.gradient_accumulation_steps or 1),
            "max_steps": cfg.max_steps,
            "optimizer_steps_planned": planned_optimizer_steps,
            "micro_batch_budget": micro_budget,
            "warmup_steps": warmup_steps,
            "warmup_ratio": float(cfg.warmup_ratio or 0.0) if warmup_steps else 0.0,
            "warmup_ratio_requested": float(cfg.warmup_ratio or 0.0),
            "max_grad_norm": (
                float(cfg.max_grad_norm or 0.0) if capabilities.grad_clipping else 0.0
            ),
            "max_grad_norm_requested": float(cfg.max_grad_norm or 0.0),
        }

        loss_history: list[float] = []
        accuracy_history: list[float] = []
        margin_history: list[float] = []

        def loss_fn(model, prompt_tokens, chosen_tokens, rejected_tokens):
            chosen_logp = _response_logprobs(
                mx=mx, nn=nn, model=model,
                prompt_tokens=prompt_tokens, response_tokens=chosen_tokens,
            )
            rejected_logp = _response_logprobs(
                mx=mx, nn=nn, model=model,
                prompt_tokens=prompt_tokens, response_tokens=rejected_tokens,
            )
            reference_chosen_logp = None
            reference_rejected_logp = None
            if self.reference_model is not None:
                reference_chosen_logp = _response_logprobs(
                    mx=mx, nn=nn, model=self.reference_model,
                    prompt_tokens=prompt_tokens, response_tokens=chosen_tokens,
                )
                reference_rejected_logp = _response_logprobs(
                    mx=mx, nn=nn, model=self.reference_model,
                    prompt_tokens=prompt_tokens, response_tokens=rejected_tokens,
                )
            loss = _dpo_preference_loss(
                mx=mx, nn=nn,
                loss_type=cfg.loss_type,
                chosen_logp=chosen_logp,
                rejected_logp=rejected_logp,
                beta=cfg.beta,
                label_smoothing=cfg.label_smoothing,
                reference_chosen_logp=reference_chosen_logp,
                reference_rejected_logp=reference_rejected_logp,
                chosen_length=int(chosen_tokens.shape[0]),
                rejected_length=int(rejected_tokens.shape[0]),
            )
            # Auxiliary metrics for the training_summary; same shape TRL's
            # rewards/accuracies / rewards/margins surface.
            margin = cfg.beta * _dpo_margin(
                chosen_logp=chosen_logp,
                rejected_logp=rejected_logp,
                reference_chosen_logp=reference_chosen_logp,
                reference_rejected_logp=reference_rejected_logp,
            )
            return loss, (chosen_logp, rejected_logp, margin)

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)

        print(
            f"=" * 70
            + f"\nMLX DPO ({'reference-free' if cfg.reference_free else 'reference-model'}, {cfg.loss_type}) on {len(rows)} pairs\n"
            f"  beta={cfg.beta} learning_rate={cfg.learning_rate} epochs={cfg.num_epochs}\n"
            f"  grad_accum={grad_accum} optimizer_steps={planned_optimizer_steps} "
            f"warmup_steps={warmup_steps}\n"
            + "=" * 70
        )
        t0 = time.time()
        step = 0
        optimizer_steps = 0
        accumulated_grads: Any = None
        budget_exhausted = False
        for epoch in range(cfg.num_epochs):
            if budget_exhausted:
                break
            for row in rows:
                # `max_steps` (via micro_budget) is a hard ceiling, not a hint.
                if step >= micro_budget:
                    budget_exhausted = True
                    break
                prompt_ids = mx.array(self.tokenizer.encode(row["prompt"]))
                chosen_ids = mx.array(self.tokenizer.encode(row["chosen"]))
                rejected_ids = mx.array(self.tokenizer.encode(row["rejected"]))

                if chosen_ids.shape[0] == 0 or rejected_ids.shape[0] == 0:
                    continue
                if (
                    prompt_ids.shape[0] + max(chosen_ids.shape[0], rejected_ids.shape[0])
                    > cfg.max_seq_length
                ):
                    # Trim chosen/rejected to fit. DPO is length-sensitive
                    # so we prefer to drop the row, but trimming preserves
                    # the gradient signal for borderline-long examples.
                    budget = cfg.max_seq_length - prompt_ids.shape[0]
                    if budget < 4:
                        continue
                    chosen_ids = chosen_ids[: budget // 2]
                    rejected_ids = rejected_ids[: budget // 2]

                (loss_value, aux), grads = loss_and_grad(
                    self.model, prompt_ids, chosen_ids, rejected_ids
                )
                step += 1
                accumulated_grads = _accumulate_gradients(accumulated_grads, grads)
                if step % grad_accum == 0:
                    self._apply_update(mx, optimizer, accumulated_grads, grad_accum)
                    accumulated_grads = None
                    optimizer_steps += 1
                else:
                    # Keep the accumulator materialized so the compute graph
                    # doesn't grow across the whole accumulation window.
                    mx.eval(accumulated_grads)

                loss_scalar = float(loss_value)
                chosen_logp, rejected_logp, margin = aux
                loss_history.append(loss_scalar)
                margin_history.append(float(margin))
                accuracy_history.append(1.0 if float(chosen_logp) > float(rejected_logp) else 0.0)

                if step % cfg.logging_steps == 0:
                    avg_loss = sum(loss_history[-cfg.logging_steps:]) / max(
                        1, len(loss_history[-cfg.logging_steps:])
                    )
                    avg_acc = sum(accuracy_history[-cfg.logging_steps:]) / max(
                        1, len(accuracy_history[-cfg.logging_steps:])
                    )
                    print(
                        f"  step {step:>5}  loss={avg_loss:.4f}  "
                        f"acc={avg_acc:.2f}  margin={float(margin):.3f}"
                    )

        pending = step % grad_accum
        if pending:
            # Don't discard a partial accumulation window (small datasets can
            # end mid-window); apply it scaled by what actually accumulated.
            self._apply_update(mx, optimizer, accumulated_grads, pending)
            optimizer_steps += 1

        elapsed = time.time() - t0
        # Save adapter
        output_dir = Path(cfg.output_dir)
        adapter_dir = output_dir / "final_model"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        try:
            from mlx_lm.tuner.utils import save_adapter

            save_adapter(self.model, str(adapter_dir / "adapters.safetensors"))
        except (ImportError, AttributeError):
            # Older mlx-lm versions had a different save path; fall back to
            # writing trainable params manually.
            tflat = deps["tree_flatten"](self.model.trainable_parameters())
            mx.save_safetensors(
                str(adapter_dir / "adapters.safetensors"),
                {k: v for k, v in tflat},
            )

        # Adapter config so mlx_lm.load(adapter_path=...) can rebuild
        # LoRA layers — same convention as MLXSFTTrainer.
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
            accuracy_history=accuracy_history,
            margin_history=margin_history,
            optimizer_steps=optimizer_steps,
            duration_seconds=elapsed,
            train_dataset=train_ds,
            val_dataset=val_ds,
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

    def _apply_update(
        self, mx: Any, optimizer: Any, gradients: Any, accumulated: int
    ) -> None:
        """Average the accumulated gradients and take one optimizer step."""
        if gradients is None:
            return
        if accumulated > 1:
            gradients = _scale_gradients(gradients, 1.0 / float(accumulated))
        optimizer.update(self.model, gradients)
        mx.eval(self.model.parameters(), optimizer.state)

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
        accuracy_history,
        margin_history,
        optimizer_steps,
        duration_seconds,
        train_dataset,
        val_dataset,
        train_file,
        dataset,
        final_output,
    ):
        cfg = self.config
        n_train = len(train_dataset)
        n_val = len(val_dataset)
        final_loss = loss_history[-1] if loss_history else None
        initial_loss = loss_history[0] if loss_history else final_loss

        cycle = build_cycle_summary(
            cycle=0,
            learning_rate=cfg.learning_rate,
            samples_seen=n_train,
            samples_kept=n_train,
            cycle_duration_seconds=duration_seconds,
            update_metrics={
                # Optimizer steps, not micro batches: with accumulation the
                # two differ by `gradient_accumulation_steps`.
                "train_steps_executed": int(optimizer_steps),
                "train_loss": final_loss,
                "initial_train_loss": initial_loss,
                "weights_updated": int(optimizer_steps) > 0,
                "update_reason": "updated" if optimizer_steps else "no_optimizer_steps",
                "optimizer_steps": int(optimizer_steps),
                "skipped_batches_non_finite": 0,
            },
            yield_diagnostics={
                "stage_counts": None,
                "rates": None,
                "minimums": {"minimum_samples_target": max(1, n_train)},
                "rejection_reasons": None,
                "summary": None,
            },
            extra={
                "train_examples": n_train,
                "validation_examples": n_val,
                "reward_accuracy_final": (
                    sum(accuracy_history[-50:]) / max(1, len(accuracy_history[-50:]))
                    if accuracy_history else None
                ),
                "reward_margin_final": margin_history[-1] if margin_history else None,
                "micro_batches_executed": len(loss_history),
                "gradient_accumulation_steps": self.backend_config_applied.get(
                    "gradient_accumulation_steps", 1
                ),
                "beta": cfg.beta,
                "loss_type": cfg.loss_type,
                "reference_free": cfg.reference_free,
                "reference_model_loaded": not cfg.reference_free,
            },
        )
        summary = build_training_summary(
            modality="dpo",
            model_name=cfg.model_name,
            total_cycles_planned=1,
            cycles=[cycle],
            run_id=self.run_id,
            seed=cfg.seed,
            base_model_name=cfg.model_name,
            active_model_name=cfg.model_name,
            yield_diagnostics={
                "stage_counts": None,
                "rates": None,
                "minimums": {"minimum_samples_target": max(1, n_train)},
                "rejection_reasons": None,
                "summary": None,
            },
            extra={
                "dataset": dataset or "",
                "train_file": train_file or "",
                "train_examples": n_train,
                "validation_examples": n_val,
                "beta": cfg.beta,
                "loss_type": cfg.loss_type,
                "label_smoothing": cfg.label_smoothing,
                "reference_free": cfg.reference_free,
                "reference_model_loaded": not cfg.reference_free,
                "backend": "mlx",
            },
        )
        summary["final_model_path"] = str(final_output)
        # Report what the backend applied, not what was requested.
        summary["backend_config_applied"] = dict(self.backend_config_applied)
        attach_effectiveness_contract(
            summary,
            minimum_samples_kept=max(1, n_train),
            minimum_optimizer_steps=1,
            evaluation={
                "metric_name": "dpo_loss",
                "baseline_value": initial_loss,
                "final_value": final_loss,
                "higher_is_better": False,
                "tolerance": 0.0,
            },
            evaluation_required=False,
            checkpoint_written=True,
            final_model_path=str(final_output),
            training_summary_path=Path(cfg.output_dir) / "training_summary.json",
        )
        attach_recovery_guidance(
            summary,
            modality="dpo",
            launch_args={
                "model": cfg.model_name,
                "dataset": dataset or "",
                "output_dir": cfg.output_dir,
                "epochs": cfg.num_epochs,
                # Effective values, so a relaunch reproduces this run.
                "batch_size": self.backend_config_applied.get("batch_size", 1),
                "gradient_accumulation_steps": self.backend_config_applied.get(
                    "gradient_accumulation_steps", 1
                ),
                "max_steps": cfg.max_steps,
                "beta": cfg.beta,
                "loss_type": cfg.loss_type,
                "reference_free": cfg.reference_free,
            },
            representative_examples=[],
        )
        return summary


__all__ = ["MLXDPOTrainer"]
