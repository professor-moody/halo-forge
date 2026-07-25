"""Native MLX continued causal-pretraining trainer."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence

from halo_forge.cpt.config import CPTConfig
from halo_forge.cpt.packing import (
    CorpusPackingPlan,
    PackedCorpusSequence,
    build_corpus_packing_plan,
    model_identity_hash,
    pack_corpus_records,
    tokenizer_identity_hash,
)
from halo_forge.cpt.trainer import (
    deterministic_corpus_validation_split,
    load_corpus_jsonl,
    resolve_cpt_run_id,
    verify_cpt_packing_plan,
    verify_cpt_training_artifact,
)
from halo_forge.runtime_determinism import set_global_seed
from halo_forge.training_contracts import (
    attach_effectiveness_contract,
    build_cycle_summary,
    build_training_summary,
    write_json_atomic,
)


class MLXCPTUnavailable(RuntimeError):
    """Raised when native MLX CPT dependencies are unavailable."""


def _require_mlx() -> Dict[str, Any]:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        from mlx_lm import load as mlx_load
        from mlx_lm.tuner.utils import linear_to_lora_layers
    except ImportError as exc:
        raise MLXCPTUnavailable("MLX CPT requires the `[mlx]` extra on Apple Silicon.") from exc
    return {
        "mx": mx,
        "nn": nn,
        "optim": optim,
        "tree_flatten": tree_flatten,
        "mlx_load": mlx_load,
        "linear_to_lora_layers": linear_to_lora_layers,
    }


def causal_next_token_loss(
    *,
    mx: Any,
    nn: Any,
    model: Any,
    input_ids: Any,
    attention_mask: Any = None,
) -> Any:
    """Return masked causal next-token cross entropy.

    Inputs at position ``t`` predict the token at ``t + 1``.  Padding targets
    are excluded through the shifted attention mask.
    """

    if len(input_ids.shape) == 1:
        input_ids = input_ids[None, :]
        if attention_mask is not None and len(attention_mask.shape) == 1:
            attention_mask = attention_mask[None, :]
    if input_ids.shape[-1] < 2:
        raise ValueError("causal next-token loss requires at least two tokens")
    model_inputs = input_ids[..., :-1]
    targets = input_ids[..., 1:]
    outputs = model(model_inputs)
    logits = getattr(outputs, "logits", outputs[0] if isinstance(outputs, tuple) else outputs)
    losses = nn.losses.cross_entropy(logits, targets, reduction="none")
    if attention_mask is None:
        return mx.mean(losses)
    target_mask = attention_mask[..., 1:].astype(losses.dtype)
    denominator = mx.maximum(mx.sum(target_mask), mx.array(1.0))
    return mx.sum(losses * target_mask) / denominator


def _batches(
    sequences: Sequence[PackedCorpusSequence],
    *,
    target_tokens: int,
    effective_batch_size: int,
) -> Iterator[list[PackedCorpusSequence]]:
    """Cycle deterministically through one-pass blocks until the budget is met."""

    if not sequences:
        return
    consumed = 0
    pending: list[PackedCorpusSequence] = []
    index = 0
    while consumed < target_tokens:
        source = sequences[index % len(sequences)]
        remaining = target_tokens - consumed
        ids = source.input_ids
        if remaining < len(ids):
            if remaining < 2:
                break
            ids = ids[:remaining]
            source = PackedCorpusSequence(ids, source.source_spans)
        pending.append(source)
        consumed += len(source.input_ids)
        index += 1
        if len(pending) >= effective_batch_size:
            yield pending
            pending = []
    if pending:
        yield pending


def _pad_batch(mx: Any, sequences: Sequence[PackedCorpusSequence], pad_token_id: int):
    width = max(len(value.input_ids) for value in sequences)
    ids = []
    masks = []
    for sequence in sequences:
        values = list(sequence.input_ids)
        padding = width - len(values)
        ids.append(values + [int(pad_token_id)] * padding)
        masks.append([1] * len(values) + [0] * padding)
    return mx.array(ids), mx.array(masks)


class MLXCPTTrainer:
    """Native MLX CPT with LoRA or full-weight adaptation."""

    def __init__(self, config: CPTConfig):
        if not isinstance(config, CPTConfig):
            raise TypeError("MLXCPTTrainer requires an explicit CPTConfig")
        self.config = config
        self.model: Any = None
        self.tokenizer: Any = None
        self.run_id = ""
        self.packing_plan: Optional[CorpusPackingPlan] = None
        self.packing_plan_hash: Optional[str] = None
        self.training_summary: Dict[str, Any] = {}

    def _count_blocks(self) -> int:
        layers = getattr(self.model, "layers", None)
        if layers is None:
            layers = getattr(getattr(self.model, "model", None), "layers", None)
        try:
            return len(layers) if layers is not None else 0
        except TypeError:
            return 0

    def _configure_adaptation(self, deps: Dict[str, Any]) -> None:
        cfg = self.config
        if cfg.adaptation == "full":
            unfreeze = getattr(self.model, "unfreeze", None)
            if callable(unfreeze):
                unfreeze()
            return
        freeze = getattr(self.model, "freeze", None)
        if callable(freeze):
            freeze()
        lora = {
            "rank": cfg.lora_r,
            "alpha": cfg.lora_alpha,
            "dropout": cfg.lora_dropout,
            "scale": float(cfg.lora_alpha) / float(cfg.lora_r),
        }
        if cfg.target_modules and any("." in value for value in cfg.target_modules):
            lora["keys"] = list(cfg.target_modules)
        deps["linear_to_lora_layers"](
            self.model,
            num_layers=self._count_blocks() or 16,
            config=lora,
        )

    def _prepare(
        self,
        train_file: str,
        validation_file: Optional[str],
    ):
        cfg = self.config
        verify_cpt_training_artifact(
            cfg,
            train_file=train_file,
            validation_file=validation_file,
        )
        records = load_corpus_jsonl(train_file)
        if validation_file:
            train_records = records
            validation_records = load_corpus_jsonl(validation_file)
        else:
            train_records, validation_records = deterministic_corpus_validation_split(
                records,
                fraction=cfg.validation_fraction,
                seed=cfg.seed,
            )
        actual_tokenizer_hash = tokenizer_identity_hash(
            self.tokenizer,
            tokenizer_id=getattr(self.tokenizer, "name_or_path", None) or cfg.model_name,
            revision=cfg.tokenizer_revision,
            explicit_hash=cfg.tokenizer_hash,
        )
        train_split = pack_corpus_records(
            train_records,
            self.tokenizer,
            max_sequence_length=cfg.max_sequence_length,
        )
        validation_split = (
            pack_corpus_records(
                validation_records,
                self.tokenizer,
                max_sequence_length=cfg.max_sequence_length,
            )
            if validation_records
            else None
        )
        plan = build_corpus_packing_plan(
            train=train_split,
            validation=validation_split,
            tokenizer_id=getattr(self.tokenizer, "name_or_path", None) or cfg.model_name,
            tokenizer_revision=cfg.tokenizer_revision,
            tokenizer_hash=actual_tokenizer_hash,
            max_sequence_length=cfg.max_sequence_length,
            budget_mode=cfg.budget_mode,
            target_tokens=cfg.target_tokens,
            corpus_passes=cfg.corpus_passes,
            effective_batch_size=cfg.effective_batch_size,
        )
        self.packing_plan = plan
        self.packing_plan_hash = verify_cpt_packing_plan(cfg, plan)
        return train_split, validation_split

    def train(
        self,
        train_file: Optional[str] = None,
        validation_file: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        if resume_from_checkpoint:
            raise ValueError(
                "MLX CPT is final/full-trial only and does not support checkpoint resume"
            )
        deps = _require_mlx()
        mx = deps["mx"]
        nn = deps["nn"]
        cfg = self.config
        cfg.seed = set_global_seed(cfg.seed)
        mx.random.seed(cfg.seed)
        self.run_id = resolve_cpt_run_id("cpt_mlx")
        source = train_file or cfg.train_file
        if not source:
            raise ValueError("CPT requires train_file from a corpus training artifact")

        started = time.time()
        self.model, self.tokenizer = deps["mlx_load"](cfg.model_name)
        if getattr(self.tokenizer, "eos_token_id", None) is None:
            raise ValueError("CPT requires a tokenizer with eos_token_id")
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        train_split, validation_split = self._prepare(
            source,
            validation_file or cfg.validation_file,
        )
        assert self.packing_plan is not None
        self._configure_adaptation(deps)

        optimizer = deps["optim"].AdamW(
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        def loss_fn(model, input_ids, attention_mask):
            return causal_next_token_loss(
                mx=mx,
                nn=nn,
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)
        scheduled_tokens = (
            int(self.packing_plan.target_tokens)
            if self.packing_plan.budget_mode == "tokens"
            else max(
                1,
                math.ceil(
                    self.packing_plan.train_tokens * float(self.packing_plan.corpus_passes or 1.0)
                ),
            )
        )
        train_losses: list[float] = []
        optimizer_steps = 0
        for batch in _batches(
            train_split.sequences,
            target_tokens=scheduled_tokens,
            effective_batch_size=cfg.effective_batch_size,
        ):
            ids, mask = _pad_batch(mx, batch, int(self.tokenizer.pad_token_id))
            loss, grads = loss_and_grad(self.model, ids, mask)
            optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), optimizer.state)
            train_losses.append(float(loss))
            optimizer_steps += 1
            if cfg.max_steps is not None and optimizer_steps >= cfg.max_steps:
                break

        eval_losses: list[float] = []
        if validation_split is not None:
            for sequence in validation_split.sequences:
                ids, mask = _pad_batch(mx, [sequence], int(self.tokenizer.pad_token_id))
                loss = loss_fn(self.model, ids, mask)
                mx.eval(loss)
                eval_losses.append(float(loss))

        final_path = Path(cfg.output_dir) / "final_model"
        final_path.mkdir(parents=True, exist_ok=True)
        if cfg.adaptation == "lora":
            try:
                from mlx_lm.tuner.utils import save_adapter

                save_adapter(self.model, str(final_path / "adapters.safetensors"))
            except (ImportError, AttributeError):
                values = {
                    key: value
                    for key, value in deps["tree_flatten"](self.model.trainable_parameters())
                }
                mx.save_safetensors(str(final_path / "adapters.safetensors"), values)
            (final_path / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "fine_tune_type": "lora",
                        "num_layers": self._count_blocks() or 16,
                        "lora_parameters": {
                            "rank": cfg.lora_r,
                            "alpha": cfg.lora_alpha,
                            "dropout": cfg.lora_dropout,
                            "scale": float(cfg.lora_alpha) / float(cfg.lora_r),
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            save_weights = getattr(self.model, "save_weights", None)
            if callable(save_weights):
                save_weights(str(final_path / "model.safetensors"))
            else:
                values = {
                    key: value for key, value in deps["tree_flatten"](self.model.parameters())
                }
                mx.save_safetensors(str(final_path / "model.safetensors"), values)
            model_config = getattr(self.model, "config", None) or getattr(self.model, "args", None)
            if hasattr(model_config, "to_dict"):
                model_config = model_config.to_dict()
            elif hasattr(model_config, "__dict__"):
                model_config = dict(vars(model_config))
            if not isinstance(model_config, dict):
                model_config = {}
            (final_path / "config.json").write_text(
                json.dumps(
                    {
                        **model_config,
                        "_name_or_path": cfg.model_name,
                        "base_model_revision": cfg.model_revision,
                        "halo_forge_adaptation": "full",
                        "halo_forge_objective": "causal_next_token",
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
        try:
            self.tokenizer.save_pretrained(str(final_path))
        except Exception:
            pass

        duration = time.time() - started
        summary = self._build_summary(
            duration=duration,
            optimizer_steps=optimizer_steps,
            train_losses=train_losses,
            eval_losses=eval_losses,
            final_path=final_path,
        )
        write_json_atomic(Path(cfg.output_dir) / "training_summary.json", summary)
        self.training_summary = summary
        return summary

    def _build_summary(
        self,
        *,
        duration: float,
        optimizer_steps: int,
        train_losses: Sequence[float],
        eval_losses: Sequence[float],
        final_path: Path,
    ) -> Dict[str, Any]:
        cfg = self.config
        assert self.packing_plan is not None
        final_train_loss = train_losses[-1] if train_losses else None
        final_eval_loss = eval_losses[-1] if eval_losses else None
        cycle = build_cycle_summary(
            cycle=0,
            learning_rate=cfg.learning_rate,
            samples_seen=self.packing_plan.train_blocks,
            samples_kept=self.packing_plan.train_blocks,
            cycle_duration_seconds=duration,
            update_metrics={
                "train_steps_executed": optimizer_steps,
                "optimizer_steps": optimizer_steps,
                "train_loss": final_train_loss,
                "initial_train_loss": train_losses[0] if train_losses else None,
                "weights_updated": optimizer_steps > 0,
                "update_reason": "updated" if optimizer_steps > 0 else "no_optimizer_steps",
                "skipped_batches_non_finite": 0,
            },
            extra={
                "backend": "mlx",
                "objective": "causal_next_token",
                "adaptation": cfg.adaptation,
                "eval_loss": final_eval_loss,
                "packing_plan": self.packing_plan.to_dict(),
                "packing_plan_hash": self.packing_plan_hash,
            },
        )
        summary = build_training_summary(
            modality="cpt",
            model_name=cfg.model_name,
            total_cycles_planned=1,
            cycles=[cycle],
            run_id=self.run_id,
            seed=cfg.seed,
            base_model_name=cfg.model_name,
            active_model_name=cfg.model_name,
            extra={
                "backend": "mlx",
                "objective": "causal_next_token",
                "adaptation": cfg.adaptation,
                "packing_plan": self.packing_plan.to_dict(),
                "packing_plan_hash": self.packing_plan_hash,
                "model_hash": model_identity_hash(
                    cfg.model_name,
                    revision=cfg.model_revision,
                    explicit_hash=cfg.model_hash,
                ),
                "tokenizer_hash": self.packing_plan.tokenizer_hash,
                "eval_loss": final_eval_loss,
                "resume_from_checkpoint": "",
            },
        )
        summary["final_model_path"] = str(final_path)
        attach_effectiveness_contract(
            summary,
            minimum_samples_kept=1,
            minimum_optimizer_steps=1,
            evaluation={
                "metric_name": "eval_loss",
                "baseline_value": eval_losses[0] if eval_losses else None,
                "final_value": final_eval_loss,
                "higher_is_better": False,
                "tolerance": 0.0,
            },
            evaluation_required=False,
            checkpoint_written=final_path.exists(),
            final_model_path=str(final_path),
            training_summary_path=Path(cfg.output_dir) / "training_summary.json",
        )
        return summary


__all__ = [
    "MLXCPTTrainer",
    "MLXCPTUnavailable",
    "causal_next_token_loss",
]
