"""PyTorch DPO trainer.

Wraps `trl.DPOTrainer` so we get the algorithm coverage and tested loss math
that TRL ships (sigmoid / IPO / hinge / KTO-pair / RPO / cDPO label
smoothing) without rewriting any of it. Halo-forge owns the parts that
matter to *us*: backend dispatch, run-id minting, output_dir layout,
training_summary contract, recovery guidance, and CLI surface — all
shared with SFT and RAFT so the public API + frontend treat DPO
identically.

Output shape: `<output_dir>/training_summary.json` with the same schema as
SFT, so the public API's `_resolve_run_source` / `_project_cycles_for_charts`
handle it without modification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOTrainer as _TRLDPOTrainer
from trl import DPOConfig as _TRLDPOConfig

from halo_forge.dpo.config import DPOConfig
from halo_forge.dpo.datasets import load_preference_dataset
from halo_forge.sft.trainer import _parse_init_lora_weights
from halo_forge.runtime_determinism import build_run_id, set_global_seed
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


class DPOTrainer:
    """Halo-forge DPO trainer (PyTorch path).

    Example:
        config = DPOConfig(model_name="Qwen/Qwen2.5-3B-Instruct", beta=0.1)
        trainer = DPOTrainer(config)
        trainer.train(dataset="ultrafeedback")
    """

    def __init__(self, config: Optional[DPOConfig] = None):
        self.config = config or DPOConfig()
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
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _build_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        cfg = self.config
        if not cfg.load_in_4bit:
            return None
        if not supports_4bit_quantization():
            if cfg.allow_quantization_fallback:
                print(
                    "WARNING: load_in_4bit requested but bitsandbytes is unavailable on "
                    "this backend (Apple Silicon MPS / CPU). Falling back to bf16."
                )
                return None
            raise RuntimeError(
                "load_in_4bit requires CUDA/ROCm with bitsandbytes; set "
                "allow_quantization_fallback=True to fall back to bf16."
            )
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )

    def setup_model(self) -> None:
        """Load policy model + tokenizer + LoRA adapters."""
        cfg = self.config
        self._load_tokenizer()

        bnb_config = self._build_quantization_config()
        attn_impl = cfg.attn_implementation or recommended_attn_impl()

        print(f"Loading base model: {cfg.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_config,
            dtype=recommended_dtype(),
            device_map=get_device_map(),
            trust_remote_code=cfg.trust_remote_code,
            attn_implementation=attn_impl,
        )

        if bnb_config is not None:
            self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
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
        """Run DPO training.

        Args:
            train_file: Local JSONL file with prompt/chosen/rejected rows.
            dataset: HuggingFace dataset id or short name (see
                `PREFERENCE_DATASETS`).
            resume_from_checkpoint: Optional checkpoint path.
        """
        cfg = self.config
        cfg.seed = set_global_seed(cfg.seed)
        self.run_id = build_run_id("dpo")

        train_file = train_file or cfg.train_file
        dataset = dataset or cfg.dataset
        if not train_file and not dataset:
            raise ValueError(
                "DPO training requires either --data (JSONL) or --dataset (HF id / short name)"
            )

        train_ds, val_ds = load_preference_dataset(
            train_file=train_file,
            dataset=dataset,
            split=cfg.dataset_split,
            max_samples=cfg.max_samples,
            validation_split=cfg.validation_split,
            seed=cfg.seed,
        )

        self.setup_model()

        # TRL's DPOConfig is a TrainingArguments subclass; we forward the
        # subset of fields halo-forge exposes, leaving the rest at TRL
        # defaults (TRL refines those each release; we'd rather inherit
        # the upstream defaults than freeze a possibly-stale snapshot).
        trl_args = _TRLDPOConfig(
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
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=cfg.seed,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            report_to=[],
            # DPO-specific
            beta=cfg.beta,
            loss_type=cfg.loss_type,
            label_smoothing=cfg.label_smoothing,
            max_length=cfg.max_seq_length,
            max_prompt_length=cfg.max_prompt_length,
            reference_free=cfg.reference_free,
        )

        trainer = _TRLDPOTrainer(
            model=self.model,
            ref_model=None,  # PEFT path — TRL uses the base model as ref automatically
            args=trl_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.tokenizer,
        )

        print("=" * 70)
        print(f"DPO TRAINING (beta={cfg.beta}, loss={cfg.loss_type})")
        print("=" * 70)
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final adapter
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
            resume_from_checkpoint=resume_from_checkpoint,
            final_output=final_output,
        )
        write_json_atomic(Path(cfg.output_dir) / "training_summary.json", summary)
        self.training_summary = summary
        return summary

    def _build_summary(
        self,
        *,
        trainer: _TRLDPOTrainer,
        train_result: Any,
        train_dataset,
        val_dataset,
        train_file: Optional[str],
        dataset: Optional[str],
        resume_from_checkpoint: Optional[str],
        final_output: Path,
    ) -> Dict[str, Any]:
        cfg = self.config
        log_history = list(getattr(trainer.state, "log_history", []))
        loss_points = [
            float(entry["loss"])
            for entry in log_history
            if isinstance(entry, dict) and isinstance(entry.get("loss"), (int, float))
        ]
        eval_loss_points = [
            float(entry["eval_loss"])
            for entry in log_history
            if isinstance(entry, dict) and isinstance(entry.get("eval_loss"), (int, float))
        ]
        # DPO-specific reward telemetry from TRL — chosen reward, rejected
        # reward, accuracy (chosen > rejected), margin. These show up in
        # log_history alongside loss; we surface them in `extra` so the
        # frontend can chart them later (Phase Q1 of F track).
        reward_acc = [
            float(entry["rewards/accuracies"])
            for entry in log_history
            if isinstance(entry, dict)
            and isinstance(entry.get("rewards/accuracies"), (int, float))
        ]
        reward_margin = [
            float(entry["rewards/margins"])
            for entry in log_history
            if isinstance(entry, dict)
            and isinstance(entry.get("rewards/margins"), (int, float))
        ]

        total_train_steps = int(
            getattr(train_result, "global_step", 0)
            or getattr(train_result, "metrics", {}).get("global_step", 0)
            or 0
        )
        final_loss = (
            loss_points[-1]
            if loss_points
            else (
                float(train_result.training_loss)
                if isinstance(getattr(train_result, "training_loss", None), (int, float))
                else None
            )
        )
        initial_loss = loss_points[0] if loss_points else final_loss

        cycle_summary = build_cycle_summary(
            cycle=0,
            learning_rate=cfg.learning_rate,
            samples_seen=len(train_dataset),
            samples_kept=len(train_dataset),
            cycle_duration_seconds=float(
                getattr(train_result, "metrics", {}).get("train_runtime", 0.0) or 0.0
            ),
            update_metrics={
                "train_steps_executed": total_train_steps,
                "train_loss": final_loss,
                "initial_train_loss": initial_loss,
                "weights_updated": total_train_steps > 0,
                "update_reason": "updated" if total_train_steps > 0 else "no_optimizer_steps",
                "optimizer_steps": total_train_steps,
                "skipped_batches_non_finite": 0,
            },
            yield_diagnostics={
                "stage_counts": None,
                "rates": None,
                "minimums": {"minimum_samples_target": max(1, len(train_dataset))},
                "rejection_reasons": None,
                "summary": None,
            },
            extra={
                "train_examples": len(train_dataset),
                "validation_examples": len(val_dataset),
                "eval_loss": eval_loss_points[-1] if eval_loss_points else None,
                "reward_accuracy_final": reward_acc[-1] if reward_acc else None,
                "reward_margin_final": reward_margin[-1] if reward_margin else None,
                "beta": cfg.beta,
                "loss_type": cfg.loss_type,
            },
        )
        summary = build_training_summary(
            modality="dpo",
            model_name=cfg.model_name,
            total_cycles_planned=1,
            cycles=[cycle_summary],
            run_id=self.run_id,
            seed=cfg.seed,
            base_model_name=cfg.model_name,
            active_model_name=cfg.model_name,
            yield_diagnostics={
                "stage_counts": None,
                "rates": None,
                "minimums": {"minimum_samples_target": max(1, len(train_dataset))},
                "rejection_reasons": None,
                "summary": None,
            },
            extra={
                "dataset": dataset or "",
                "train_file": train_file or "",
                "train_examples": len(train_dataset),
                "validation_examples": len(val_dataset),
                "eval_loss": eval_loss_points[-1] if eval_loss_points else None,
                "resume_from_checkpoint": resume_from_checkpoint or "",
                "beta": cfg.beta,
                "loss_type": cfg.loss_type,
                "label_smoothing": cfg.label_smoothing,
                "reference_free": cfg.reference_free,
                "reward_accuracy_history": reward_acc,
                "reward_margin_history": reward_margin,
            },
        )
        summary["final_model_path"] = str(final_output)
        attach_effectiveness_contract(
            summary,
            minimum_samples_kept=max(1, len(train_dataset)),
            minimum_optimizer_steps=1,
            evaluation={
                "metric_name": "eval_loss",
                "baseline_value": eval_loss_points[0] if eval_loss_points else None,
                "final_value": eval_loss_points[-1] if eval_loss_points else None,
                "higher_is_better": False,
                "tolerance": 0.0,
            },
            evaluation_required=False,
            checkpoint_written=any(Path(cfg.output_dir).glob("checkpoint-*")),
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
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "max_samples": cfg.max_samples,
                "beta": cfg.beta,
                "loss_type": cfg.loss_type,
            },
            representative_examples=[],
        )
        return summary


__all__ = ["DPOTrainer"]
