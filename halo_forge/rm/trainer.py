"""PyTorch reward-model trainer (Track T3).

Wraps `trl.RewardTrainer` so we get TRL's tested Bradley-Terry loss
math + center-rewards regularization for free, and own only the parts
halo-forge cares about: backend dispatch, run-id minting, output_dir
layout, training_summary contract.

Output is a `AutoModelForSequenceClassification` checkpoint with one
output unit. `predict_reward(prompt, response)` is a small helper for
turning the trained RM into a halo-forge verifier.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import RewardTrainer as _TRLRewardTrainer
from trl import RewardConfig as _TRLRewardConfig

from halo_forge.dpo.datasets import load_preference_dataset
from halo_forge.rm.config import RMConfig
from halo_forge.runtime_determinism import build_run_id, set_global_seed
from halo_forge.sft.trainer import _parse_init_lora_weights
from halo_forge.training_contracts import (
    attach_effectiveness_contract,
    build_cycle_summary,
    build_training_summary,
    write_json_atomic,
)
from halo_forge.training_recovery import attach_recovery_guidance
from halo_forge.utils.accelerator import (
    empty_accelerator_cache,
    get_device_map,
    recommended_attn_impl,
    recommended_dtype,
    supports_4bit_quantization,
)

logger = logging.getLogger(__name__)


def _format_pair(row, tokenizer, max_length: int):
    """TRL's RewardTrainer expects ``input_ids_chosen``, ``attention_mask_chosen``,
    ``input_ids_rejected``, ``attention_mask_rejected`` columns. We
    tokenize prompt+chosen and prompt+rejected here so the trainer can
    iterate without a custom collator."""
    chosen_text = (row.get("prompt") or "") + (row.get("chosen") or "")
    rejected_text = (row.get("prompt") or "") + (row.get("rejected") or "")
    chosen = tokenizer(
        chosen_text, truncation=True, max_length=max_length, padding=False,
    )
    rejected = tokenizer(
        rejected_text, truncation=True, max_length=max_length, padding=False,
    )
    return {
        "input_ids_chosen": chosen["input_ids"],
        "attention_mask_chosen": chosen["attention_mask"],
        "input_ids_rejected": rejected["input_ids"],
        "attention_mask_rejected": rejected["attention_mask"],
    }


class RewardModelTrainer:
    """Halo-forge RM trainer (PyTorch path)."""

    def __init__(self, config: Optional[RMConfig] = None):
        self.config = config or RMConfig()
        from halo_forge.utils.neural_accelerators import validate_neural_accelerator_opt_in

        validate_neural_accelerator_opt_in(self.config, logger=logger, label="Reward Model")
        self.model = None
        self.tokenizer = None
        self.training_summary: Dict[str, Union[str, int, float, dict, list, None]] = {}
        self.run_id: str = ""

    def _load_tokenizer(self) -> None:
        if self.tokenizer is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        cfg = self.config
        if not cfg.load_in_4bit:
            return None
        if not supports_4bit_quantization():
            if cfg.allow_quantization_fallback:
                logger.warning(
                    "load_in_4bit unavailable on this backend; falling back to bf16."
                )
                return None
            raise RuntimeError(
                "load_in_4bit requires CUDA/ROCm with bitsandbytes."
            )
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )

    def setup_model(self) -> None:
        cfg = self.config
        self._load_tokenizer()

        bnb_config = self._build_quantization_config()
        attn_impl = cfg.attn_implementation or recommended_attn_impl()

        logger.info("Loading base model with classification head: %s", cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=1,  # scalar reward
            quantization_config=bnb_config,
            dtype=recommended_dtype(),
            device_map=get_device_map(),
            trust_remote_code=cfg.trust_remote_code,
            attn_implementation=attn_impl,
        )

        # Some base models ship without a pad token id on the config;
        # the classification head needs one to compute attention masks.
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if bnb_config is not None:
            self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",  # not CAUSAL_LM — the head matters
            use_dora=cfg.use_dora,
            use_rslora=cfg.use_rslora,
            init_lora_weights=_parse_init_lora_weights(cfg.init_lora_weights),
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        empty_accelerator_cache()

    def train(
        self,
        train_file: Optional[str] = None,
        dataset: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        cfg = self.config
        cfg.seed = set_global_seed(cfg.seed)
        self.run_id = build_run_id("rm")

        train_file = train_file or cfg.train_file
        dataset = dataset or cfg.dataset

        train_ds, val_ds = load_preference_dataset(
            train_file=train_file,
            dataset=dataset,
            split=cfg.dataset_split,
            max_samples=cfg.max_samples,
            validation_split=cfg.validation_split,
            seed=cfg.seed,
        )

        self.setup_model()

        # Pre-tokenize chosen+rejected pairs into the columns TRL's
        # RewardTrainer expects. We do this map step here so the dataset
        # iteration inside the trainer is dataloader-clean.
        train_ds = train_ds.map(
            lambda row: _format_pair(row, self.tokenizer, cfg.max_length),
            remove_columns=[c for c in train_ds.column_names
                            if c not in ("input_ids_chosen", "attention_mask_chosen",
                                         "input_ids_rejected", "attention_mask_rejected")],
        )
        val_ds = val_ds.map(
            lambda row: _format_pair(row, self.tokenizer, cfg.max_length),
            remove_columns=[c for c in val_ds.column_names
                            if c not in ("input_ids_chosen", "attention_mask_chosen",
                                         "input_ids_rejected", "attention_mask_rejected")],
        )

        trl_args = _TRLRewardConfig(
            output_dir=cfg.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm,
            lr_scheduler_type="cosine",
            optim=cfg.optim,
            bf16=cfg.bf16,
            gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=cfg.logging_steps,
            save_strategy="steps",
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            eval_strategy="steps",
            eval_steps=cfg.eval_steps,
            seed=cfg.seed,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            report_to=[],
            max_length=cfg.max_length,
            center_rewards_coefficient=cfg.center_rewards_coefficient,
        )

        trainer = _TRLRewardTrainer(
            model=self.model,
            args=trl_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.tokenizer,
        )

        logger.info(
            "RM training: model=%s n_train=%d n_val=%d",
            cfg.model_name, len(train_ds), len(val_ds),
        )
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        final_output = Path(cfg.output_dir) / "final_model"
        trainer.save_model(str(final_output))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(final_output))

        summary = self._build_summary(
            trainer=trainer,
            train_result=train_result,
            train_dataset=train_ds,
            val_dataset=val_ds,
            train_file=train_file,
            dataset=dataset,
            final_output=final_output,
        )
        write_json_atomic(Path(cfg.output_dir) / "training_summary.json", summary)
        self.training_summary = summary
        return summary

    def _build_summary(self, *, trainer, train_result, train_dataset, val_dataset,
                       train_file, dataset, final_output):
        cfg = self.config
        log_history = list(getattr(trainer.state, "log_history", []))
        loss_points = [
            float(e["loss"]) for e in log_history
            if isinstance(e, dict) and isinstance(e.get("loss"), (int, float))
        ]
        # TRL's RewardTrainer logs `accuracy` (margin sign agreement with
        # the preference label) — surfaces "is the RM learning anything".
        accuracy_points = [
            float(e["accuracy"]) for e in log_history
            if isinstance(e, dict) and isinstance(e.get("accuracy"), (int, float))
        ]
        eval_loss_points = [
            float(e["eval_loss"]) for e in log_history
            if isinstance(e, dict) and isinstance(e.get("eval_loss"), (int, float))
        ]

        total_steps = int(
            getattr(train_result, "global_step", 0)
            or getattr(train_result, "metrics", {}).get("global_step", 0)
            or 0
        )
        final_loss = (
            loss_points[-1] if loss_points
            else (
                float(train_result.training_loss)
                if isinstance(getattr(train_result, "training_loss", None), (int, float))
                else None
            )
        )
        initial_loss = loss_points[0] if loss_points else final_loss

        cycle = build_cycle_summary(
            cycle=0,
            learning_rate=cfg.learning_rate,
            samples_seen=len(train_dataset),
            samples_kept=len(train_dataset),
            cycle_duration_seconds=float(
                getattr(train_result, "metrics", {}).get("train_runtime", 0.0) or 0.0
            ),
            update_metrics={
                "train_steps_executed": total_steps,
                "train_loss": final_loss,
                "initial_train_loss": initial_loss,
                "weights_updated": total_steps > 0,
                "update_reason": "updated" if total_steps > 0 else "no_optimizer_steps",
                "optimizer_steps": total_steps,
                "skipped_batches_non_finite": 0,
            },
            yield_diagnostics={
                "stage_counts": None, "rates": None,
                "minimums": {"minimum_samples_target": max(1, len(train_dataset))},
                "rejection_reasons": None, "summary": None,
            },
            extra={
                "train_examples": len(train_dataset),
                "validation_examples": len(val_dataset),
                "eval_loss": eval_loss_points[-1] if eval_loss_points else None,
                "accuracy_final": accuracy_points[-1] if accuracy_points else None,
                "center_rewards_coefficient": cfg.center_rewards_coefficient,
            },
        )
        summary = build_training_summary(
            modality="rm",
            model_name=cfg.model_name,
            total_cycles_planned=1,
            cycles=[cycle],
            run_id=self.run_id,
            seed=cfg.seed,
            base_model_name=cfg.model_name,
            active_model_name=cfg.model_name,
            yield_diagnostics={
                "stage_counts": None, "rates": None,
                "minimums": {"minimum_samples_target": max(1, len(train_dataset))},
                "rejection_reasons": None, "summary": None,
            },
            extra={
                "dataset": dataset or "",
                "train_file": train_file or "",
                "train_examples": len(train_dataset),
                "validation_examples": len(val_dataset),
                "eval_loss": eval_loss_points[-1] if eval_loss_points else None,
                "accuracy_history": accuracy_points,
            },
        )
        summary["final_model_path"] = str(final_output)
        attach_effectiveness_contract(
            summary,
            minimum_samples_kept=max(1, len(train_dataset)),
            minimum_optimizer_steps=1,
            evaluation={
                "metric_name": "preference_accuracy",
                "baseline_value": accuracy_points[0] if accuracy_points else None,
                "final_value": accuracy_points[-1] if accuracy_points else None,
                "higher_is_better": True,
                "tolerance": 0.0,
            },
            evaluation_required=False,
            checkpoint_written=any(Path(cfg.output_dir).glob("checkpoint-*")),
            final_model_path=str(final_output),
            training_summary_path=Path(cfg.output_dir) / "training_summary.json",
        )
        attach_recovery_guidance(
            summary,
            modality="rm",
            launch_args={
                "model": cfg.model_name,
                "dataset": dataset or "",
                "output_dir": cfg.output_dir,
                "epochs": cfg.num_epochs,
                "batch_size": cfg.batch_size,
                "max_samples": cfg.max_samples,
            },
            representative_examples=[],
        )
        return summary


__all__ = ["RewardModelTrainer"]
