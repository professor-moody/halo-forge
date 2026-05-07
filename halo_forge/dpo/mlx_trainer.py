"""MLX-native DPO trainer (Track T1 / T17).

Closes the silent-failure gap from the May 2026 audit: until now
``halo-forge dpo train --accelerator mlx`` raised NotImplementedError.
This v1 ships **reference-free DPO** (`--reference-free`), which is the
narrower-contract variant that Tülu 3 et al. publish and which
sidesteps the dual-model memory question entirely. Full reference-model
DPO on MLX (loading two copies of the base) lands in a follow-up.

Why a custom loop instead of wrapping ``mlx_lm.tuner.train``: DPO's
loss is structurally different from SFT's cross-entropy. SFT computes
a per-token loss over the whole sequence; DPO computes per-pair
logp(chosen) - logp(rejected) over only the *response* tokens (prompt
tokens are masked) and combines them through a sigmoid. ``mlx_lm.tuner``
isn't shaped for that — we own the loop here so the loss math is
explicit and inspectable.

Algorithm (reference-free DPO):

    pi_chosen   = sum log p_θ(yw | x)   over response tokens
    pi_rejected = sum log p_θ(yl | x)   over response tokens
    loss        = -log σ(β · (pi_chosen − pi_rejected))

Pairs map directly to TRL's ``loss_type="sigmoid"`` with
``reference_free=True``. cDPO label smoothing is supported; IPO /
hinge / kto_pair are not (they need the reference model).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from halo_forge.dpo.config import DPOConfig
from halo_forge.dpo.datasets import load_preference_dataset
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
) -> Any:
    """Standard sigmoid DPO loss with optional cDPO label smoothing.

    Reference-free variant: π_ref drops out, so this is just
    ``-log σ(β (logp_chosen − logp_rejected))``.
    """
    logits = beta * (chosen_logp - rejected_logp)
    if label_smoothing > 0:
        # cDPO: assume label is correct with prob (1-eps), wrong with
        # prob eps. Loss is the convex combination.
        return (
            -(1 - label_smoothing) * nn.log_sigmoid(logits)
            - label_smoothing * nn.log_sigmoid(-logits)
        )
    return -nn.log_sigmoid(logits)


class MLXDPOTrainer:
    """Reference-free DPO trainer on MLX (Apple Silicon).

    Currently supports ``loss_type="sigmoid"`` with
    ``reference_free=True``. Other loss types and reference-model
    support are roadmap work (T17 follow-up).
    """

    def __init__(self, config: Optional[DPOConfig] = None):
        self.config = config or DPOConfig()
        warn_unsupported_for_mlx(self.config, trainer_label="MLX DPO")

        if not self.config.reference_free:
            raise NotImplementedError(
                "MLX DPO v1 supports --reference-free only. Full reference-"
                "model DPO on MLX (load two copies of the base) is on the "
                "roadmap. For now: pass --reference-free, or run on a "
                "PyTorch backend."
            )
        if self.config.loss_type != "sigmoid":
            raise NotImplementedError(
                f"MLX DPO v1 supports loss_type='sigmoid' only; got {self.config.loss_type!r}. "
                "IPO / hinge / kto_pair require the reference model."
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
            logger.warning("MLX DPO v1 ignores resume_from_checkpoint — fresh run.")

        cfg.seed = set_global_seed(cfg.seed)
        self.run_id = build_run_id("dpo_mlx")

        train_ds, val_ds = load_preference_dataset(
            train_file=cfg.train_file,
            dataset=cfg.dataset,
            split=cfg.dataset_split,
            max_samples=cfg.max_samples,
            validation_split=cfg.validation_split,
            seed=cfg.seed,
        )

        print(f"Loading MLX model: {cfg.model_name}")
        self.model, self.tokenizer = deps["mlx_load"](cfg.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

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

        optimizer = deps["optim"].AdamW(
            learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

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
            loss = _sigmoid_dpo_loss(
                mx=mx, nn=nn,
                chosen_logp=chosen_logp,
                rejected_logp=rejected_logp,
                beta=cfg.beta,
                label_smoothing=cfg.label_smoothing,
            )
            # Auxiliary metrics for the training_summary; same shape TRL's
            # rewards/accuracies / rewards/margins surface.
            margin = cfg.beta * (chosen_logp - rejected_logp)
            return loss, (chosen_logp, rejected_logp, margin)

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)

        # Iterate by example: DPO batches are tricky to pad due to the
        # variable prompt+response lengths; v1 keeps the loop sequential
        # for clarity. Batched loss is a follow-up perf win.
        rows = list(train_ds)
        print(
            f"=" * 70 + f"\nMLX DPO (reference-free, sigmoid) on {len(rows)} pairs\n"
            f"  beta={cfg.beta} learning_rate={cfg.learning_rate} epochs={cfg.num_epochs}\n"
            + "=" * 70
        )
        t0 = time.time()
        step = 0
        for epoch in range(cfg.num_epochs):
            for row in rows:
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
                optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), optimizer.state)

                loss_scalar = float(loss_value)
                chosen_logp, rejected_logp, margin = aux
                loss_history.append(loss_scalar)
                margin_history.append(float(margin))
                accuracy_history.append(1.0 if float(chosen_logp) > float(rejected_logp) else 0.0)

                step += 1
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
                "train_steps_executed": len(loss_history),
                "train_loss": final_loss,
                "initial_train_loss": initial_loss,
                "weights_updated": len(loss_history) > 0,
                "update_reason": "updated" if loss_history else "no_optimizer_steps",
                "optimizer_steps": len(loss_history),
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
                "beta": cfg.beta,
                "loss_type": cfg.loss_type,
                "reference_free": True,
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
                "reference_free": True,
                "backend": "mlx",
            },
        )
        summary["final_model_path"] = str(final_output)
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
                "batch_size": cfg.batch_size,
                "beta": cfg.beta,
                "loss_type": cfg.loss_type,
                "reference_free": True,
            },
            representative_examples=[],
        )
        return summary


__all__ = ["MLXDPOTrainer"]
