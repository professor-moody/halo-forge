"""PyTorch GRPO trainer.

Wraps `trl.GRPOTrainer` so we get TRL's tested loss math (clipped
policy ratio, KL term, group-relative advantage) and the vLLM rollout
integration TRL ships in v1.0+. Halo-forge owns: backend dispatch,
verifier resolution via the plugin registry (V1), output_dir layout,
training_summary contract.

The verifier integration is the interesting part. TRL takes
``reward_funcs`` — callables ``(prompts, completions, **kwargs) -> List[float]``
that score each candidate. We wrap halo-forge's `Verifier` instances so
the rich verifier ecosystem (code execution, schema, LLM-as-judge, etc.)
plugs straight in without each user re-implementing the bridge.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import GRPOTrainer as _TRLGRPOTrainer
from trl import GRPOConfig as _TRLGRPOConfig

from halo_forge.grpo.config import GRPOConfig
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


def _build_reward_func(verifier_name: str, reward_threshold: float):
    """Resolve a halo-forge verifier and adapt it to TRL's reward API.

    TRL expects ``reward_func(prompts, completions, **kwargs) -> List[float]``.
    Halo-forge verifiers expose ``verify(code: str) -> VerifyResult``.
    We bridge: instantiate the verifier from the plugin registry, call
    ``verify`` on each completion, return the ``reward`` field.
    """
    from halo_forge.rlvr.verifiers import get_verifier

    verifier_cls = get_verifier(verifier_name)
    verifier = verifier_cls()

    def reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        try:
            if hasattr(verifier, "verify_batch"):
                results = verifier.verify_batch(completions, prompts)
            else:
                results = []
                for prompt, completion in zip(prompts, completions):
                    if hasattr(verifier, "verify_with_prompt"):
                        try:
                            results.append(verifier.verify_with_prompt(completion, prompt=prompt))
                        except TypeError:
                            results.append(verifier.verify_with_prompt(completion, prompt))
                    else:
                        results.append(verifier.verify(completion))
        except Exception as exc:
            logger.warning("Verifier %s raised during batch scoring: %s", verifier_name, exc)
            return [0.0 for _ in completions]

        for result in results:
            try:
                reward = float(getattr(result, "reward", 0.0) or 0.0)
                success = bool(getattr(result, "success", False))
                rewards.append(reward if success and reward >= reward_threshold else 0.0)
            except Exception as exc:
                logger.warning("Verifier %s returned an invalid result: %s", verifier_name, exc)
                rewards.append(0.0)
        return rewards

    reward_func.__name__ = f"halo_forge_{verifier_name}_reward"
    return reward_func


def _load_prompts_dataset(
    *,
    train_file: Optional[str],
    dataset: Optional[str],
    max_samples: Optional[int],
):
    """GRPO dataset is just prompts; the trainer generates completions."""
    from datasets import Dataset, load_dataset

    if not train_file and not dataset:
        raise ValueError("GRPO requires --data (JSONL prompts) or --dataset")

    if train_file:
        import json
        rows = []
        with open(train_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = row.get("prompt") or row.get("text") or row.get("question")
                if prompt:
                    rows.append({"prompt": str(prompt)})
        ds = Dataset.from_list(rows)
    else:
        ds = load_dataset(dataset, split="train")
        if "prompt" not in ds.column_names:
            # Try the common alternates.
            for alt in ("text", "question", "instruction"):
                if alt in ds.column_names:
                    ds = ds.rename_column(alt, "prompt")
                    break

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


class GRPOTrainer:
    """Halo-forge GRPO trainer (PyTorch path)."""

    def __init__(self, config: Optional[GRPOConfig] = None):
        self.config = config or GRPOConfig()
        from halo_forge.utils.neural_accelerators import validate_neural_accelerator_opt_in

        validate_neural_accelerator_opt_in(self.config, logger=logger, label="GRPO")
        self.model = None
        self.tokenizer = None
        self.training_summary: Dict[str, Any] = {}
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
                    "load_in_4bit requested but bitsandbytes is unavailable; falling back to bf16."
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
        cfg = self.config
        self._load_tokenizer()

        bnb_config = self._build_quantization_config()
        attn_impl = cfg.attn_implementation or recommended_attn_impl()

        logger.info("Loading base model: %s", cfg.model_name)
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
        cfg = self.config
        cfg.seed = set_global_seed(cfg.seed)
        self.run_id = build_run_id("grpo")

        train_file = train_file or cfg.train_file
        dataset = dataset or cfg.dataset

        train_ds = _load_prompts_dataset(
            train_file=train_file,
            dataset=dataset,
            max_samples=cfg.max_samples,
        )

        self.setup_model()

        reward_func = _build_reward_func(cfg.verifier_name, cfg.reward_threshold)

        trl_args = _TRLGRPOConfig(
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
            seed=cfg.seed,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            report_to=[],
            # GRPO-specific
            num_generations=cfg.num_generations,
            beta=cfg.beta,
            epsilon=cfg.epsilon,
            temperature=cfg.temperature,
            scale_rewards=cfg.scale_rewards,
            max_prompt_length=cfg.max_prompt_length,
            max_completion_length=cfg.max_completion_length,
        )

        trainer = _TRLGRPOTrainer(
            model=self.model,
            reward_funcs=[reward_func],
            args=trl_args,
            train_dataset=train_ds,
            processing_class=self.tokenizer,
        )

        logger.info(
            "GRPO training: model=%s verifier=%s num_generations=%d beta=%.3f",
            cfg.model_name, cfg.verifier_name, cfg.num_generations, cfg.beta,
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
            train_file=train_file,
            dataset=dataset,
            final_output=final_output,
        )
        write_json_atomic(Path(cfg.output_dir) / "training_summary.json", summary)
        self.training_summary = summary
        return summary

    def _build_summary(self, *, trainer, train_result, train_dataset,
                       train_file, dataset, final_output):
        cfg = self.config
        log_history = list(getattr(trainer.state, "log_history", []))
        loss_points = [
            float(e["loss"]) for e in log_history
            if isinstance(e, dict) and isinstance(e.get("loss"), (int, float))
        ]
        reward_points = [
            float(e["reward"]) for e in log_history
            if isinstance(e, dict) and isinstance(e.get("reward"), (int, float))
        ]
        kl_points = [
            float(e["kl"]) for e in log_history
            if isinstance(e, dict) and isinstance(e.get("kl"), (int, float))
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
            samples_seen=len(train_dataset) * cfg.num_generations,
            samples_kept=len(train_dataset) * cfg.num_generations,
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
                "num_generations": cfg.num_generations,
                "avg_reward_final": reward_points[-1] if reward_points else None,
                "avg_kl_final": kl_points[-1] if kl_points else None,
                "beta": cfg.beta,
                "epsilon": cfg.epsilon,
                "temperature": cfg.temperature,
                "verifier": cfg.verifier_name,
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
                "minimums": {"minimum_samples_target": max(1, len(train_dataset))},
                "rejection_reasons": None, "summary": None,
            },
            extra={
                "dataset": dataset or "",
                "train_file": train_file or "",
                "train_examples": len(train_dataset),
                "num_generations": cfg.num_generations,
                "beta": cfg.beta,
                "epsilon": cfg.epsilon,
                "temperature": cfg.temperature,
                "verifier": cfg.verifier_name,
                "rollout_engine": cfg.rollout_engine,
                "reward_history": reward_points,
                "kl_history": kl_points,
            },
        )
        summary["final_model_path"] = str(final_output)
        attach_effectiveness_contract(
            summary,
            minimum_samples_kept=max(1, len(train_dataset)),
            minimum_optimizer_steps=1,
            evaluation={
                "metric_name": "avg_reward",
                "baseline_value": reward_points[0] if reward_points else None,
                "final_value": reward_points[-1] if reward_points else None,
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
            modality="grpo",
            launch_args={
                "model": cfg.model_name,
                "dataset": dataset or "",
                "output_dir": cfg.output_dir,
                "num_generations": cfg.num_generations,
                "beta": cfg.beta,
                "verifier": cfg.verifier_name,
            },
            representative_examples=[],
        )
        return summary


__all__ = ["GRPOTrainer"]
