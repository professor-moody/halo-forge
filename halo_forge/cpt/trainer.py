"""Hugging Face/PyTorch continued causal-pretraining trainer."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from halo_forge.cpt.config import CPTConfig
from halo_forge.cpt.packing import (
    CorpusPackingPlan,
    PackedCorpusSplit,
    build_corpus_packing_plan,
    model_identity_hash,
    pack_corpus_records,
    packing_plan_hash,
    tokenizer_identity_hash,
)
from halo_forge.runtime_determinism import RUN_ID_ENV, build_run_id, set_global_seed
from halo_forge.training_contracts import (
    attach_effectiveness_contract,
    build_cycle_summary,
    build_training_summary,
    write_json_atomic,
)


def _parse_init_lora_weights(value: Any) -> Any:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return value


def resolve_cpt_run_id(modality: str = "cpt") -> str:
    """Resolve and publish the canonical managed/direct CPT run identity."""

    run_id = build_run_id(modality)
    os.environ.setdefault(RUN_ID_ENV, run_id)
    return run_id


def load_corpus_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"corpus file not found: {source}")
    records: list[Dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid corpus JSONL row {line_number}: {exc}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"corpus JSONL row {line_number} must be an object")
            text = value.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"corpus JSONL row {line_number} requires non-empty text")
            records.append(dict(value))
    if not records:
        raise ValueError(f"corpus file contains no usable records: {source}")
    return records


def deterministic_corpus_validation_split(
    records: Sequence[Mapping[str, Any]],
    *,
    fraction: float,
    seed: int,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Split whole documents by stable identity without cross-split fragments."""

    count = len(records)
    if count < 2 or fraction <= 0:
        return [dict(value) for value in records], []
    validation_count = min(count - 1, max(1, round(count * float(fraction))))

    def rank(item: tuple[int, Mapping[str, Any]]) -> tuple[str, int]:
        index, record = item
        identity = (
            record.get("document_id")
            or record.get("document_hash")
            or hashlib.sha256(
                json.dumps(record, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
        )
        digest = hashlib.sha256(f"{int(seed)}:{identity}".encode("utf-8")).hexdigest()
        return digest, index

    selected = {index for index, _record in sorted(enumerate(records), key=rank)[:validation_count]}
    train = [dict(record) for index, record in enumerate(records) if index not in selected]
    validation = [dict(record) for index, record in enumerate(records) if index in selected]
    return train, validation


def verify_cpt_training_artifact(
    config: CPTConfig,
    *,
    train_file: str | Path,
    validation_file: Optional[str | Path] = None,
) -> Optional[Dict[str, Any]]:
    """Verify and bind a managed CPT artifact before tokenization or updates.

    Managed launches carry the content-addressed artifact identity into the
    trainer process.  Re-verifying the manifest here closes the gap between
    API preflight and execution: a split cannot be changed after scheduling
    and still reach the optimizer.
    """

    configured_identity = any(
        (
            config.training_artifact_id,
            config.training_artifact_hash,
            config.expected_packing_plan_hash,
        )
    )
    if not configured_identity:
        return None
    if not config.training_artifact_id or not config.training_artifact_hash:
        raise ValueError(
            "managed CPT requires both training_artifact_id and training_artifact_hash"
        )

    source = Path(train_file).expanduser().resolve()
    roots = (source.parent, source.parent.parent)
    artifact_root = next((root for root in roots if (root / "manifest.json").is_file()), None)
    if artifact_root is None:
        raise ValueError("managed CPT train_file is not inside a training artifact")
    try:
        manifest = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"managed CPT training artifact manifest is unreadable: {exc}") from exc

    from halo_forge.data_lab.training_artifacts import TrainingArtifactRenderer

    TrainingArtifactRenderer._verify_manifest_files(artifact_root, manifest)
    artifact_id = str(manifest.get("artifact_id") or "")
    artifact_hash = str(manifest.get("artifact_hash") or "")
    if artifact_id != config.training_artifact_id:
        raise ValueError(
            "managed CPT training artifact ID changed between preflight and execution"
        )
    if artifact_hash != config.training_artifact_hash:
        raise ValueError(
            "managed CPT training artifact hash changed between preflight and execution"
        )
    if str(manifest.get("trainer_mode") or "").lower() != "cpt":
        raise ValueError("managed CPT received a non-CPT training artifact")

    split_paths = dict(manifest.get("split_paths") or {})
    expected_train = artifact_root / str(split_paths.get("train") or "")
    if not split_paths.get("train") or expected_train.resolve() != source:
        raise ValueError("managed CPT train_file does not match the artifact train split")
    expected_validation = split_paths.get("validation")
    if validation_file is not None:
        if not expected_validation:
            raise ValueError(
                "managed CPT validation_file is not declared by the training artifact"
            )
        if (artifact_root / str(expected_validation)).resolve() != Path(
            validation_file
        ).expanduser().resolve():
            raise ValueError(
                "managed CPT validation_file does not match the artifact validation split"
            )
    elif expected_validation:
        raise ValueError("managed CPT omitted the artifact's preserved validation split")

    pinned_values = {
        "model_name": manifest.get("model"),
        "model_revision": manifest.get("model_revision"),
        "model_hash": manifest.get("model_hash"),
        "tokenizer_revision": manifest.get("tokenizer_revision"),
        "tokenizer_hash": manifest.get("tokenizer_hash"),
    }
    for field, pinned in pinned_values.items():
        if pinned is None:
            continue
        current = getattr(config, field)
        if current is not None and str(current) != str(pinned):
            raise ValueError(
                f"managed CPT {field} conflicts with the verified training artifact"
            )
        setattr(config, field, pinned)

    pinned_plan_hash = str(manifest.get("packing_plan_hash") or "").strip()
    if not pinned_plan_hash:
        raise ValueError("managed CPT training artifact has no packing-plan identity")
    if (
        config.expected_packing_plan_hash
        and config.expected_packing_plan_hash != pinned_plan_hash
    ):
        raise ValueError(
            "managed CPT packing-plan identity changed between preflight and execution"
        )
    config.expected_packing_plan_hash = pinned_plan_hash
    return manifest


def verify_cpt_packing_plan(config: CPTConfig, plan: CorpusPackingPlan) -> str:
    """Return the observed plan hash, rejecting managed render/execution drift."""

    observed = packing_plan_hash(plan)
    expected = config.expected_packing_plan_hash
    if expected and observed != expected:
        raise ValueError(
            "CPT packing plan drifted from the verified training artifact "
            f"(expected {expected}, observed {observed})"
        )
    return observed


def _plan_and_splits(
    config: CPTConfig,
    tokenizer: Any,
    *,
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
) -> tuple[CorpusPackingPlan, PackedCorpusSplit, Optional[PackedCorpusSplit], str, Optional[str]]:
    actual_tokenizer_hash = tokenizer_identity_hash(
        tokenizer,
        tokenizer_id=getattr(tokenizer, "name_or_path", None) or config.model_name,
        revision=config.tokenizer_revision,
        explicit_hash=config.tokenizer_hash,
    )
    resolved_model_hash = model_identity_hash(
        config.model_name,
        revision=config.model_revision,
        explicit_hash=config.model_hash,
    )
    train_split = pack_corpus_records(
        train_records,
        tokenizer,
        max_sequence_length=config.max_sequence_length,
    )
    validation_split = (
        pack_corpus_records(
            validation_records,
            tokenizer,
            max_sequence_length=config.max_sequence_length,
        )
        if validation_records
        else None
    )
    plan = build_corpus_packing_plan(
        train=train_split,
        validation=validation_split,
        tokenizer_id=getattr(tokenizer, "name_or_path", None) or config.model_name,
        tokenizer_revision=config.tokenizer_revision,
        tokenizer_hash=actual_tokenizer_hash,
        max_sequence_length=config.max_sequence_length,
        budget_mode=config.budget_mode,
        target_tokens=config.target_tokens,
        corpus_passes=config.corpus_passes,
        effective_batch_size=config.effective_batch_size,
    )
    return plan, train_split, validation_split, actual_tokenizer_hash, resolved_model_hash


class CausalLMCollator:
    """Pad packed blocks while masking padding from next-token loss."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        import torch

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("CPT tokenizer requires pad_token_id or eos_token_id")
        width = max(len(value["input_ids"]) for value in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            ids = [int(value) for value in feature["input_ids"]]
            padding = width - len(ids)
            input_ids.append(ids + [int(pad_token_id)] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
            labels.append(ids + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class CPTTrainer:
    """PyTorch/HF causal next-token trainer for immutable corpus artifacts."""

    def __init__(self, config: CPTConfig, *, backend: Any = None):
        if not isinstance(config, CPTConfig):
            raise TypeError("CPTTrainer requires an explicit CPTConfig")
        self.config = config
        self.backend = backend
        self.model: Any = None
        self.tokenizer: Any = None
        self.run_id = ""
        self.packing_plan: Optional[CorpusPackingPlan] = None
        self.packing_plan_hash: Optional[str] = None
        self.training_summary: Dict[str, Any] = {}

    def _load_tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            revision=self.config.tokenizer_revision or self.config.model_revision,
            trust_remote_code=self.config.trust_remote_code,
        )
        if getattr(tokenizer, "eos_token_id", None) is None:
            raise ValueError("CPT requires a tokenizer with eos_token_id")
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        return tokenizer

    def _load_model(self) -> Any:
        from halo_forge.backend import LoadSpec, get_backend

        backend = self.backend or get_backend(require_training=True)
        quantization = "4bit" if self.config.load_in_4bit else None
        try:
            model = backend.load_causal_lm(
                LoadSpec(
                    model_name=self.config.model_name,
                    dtype="bfloat16" if self.config.bf16 else None,
                    quantization=quantization,
                    attn_implementation=self.config.attn_implementation,
                    trust_remote_code=self.config.trust_remote_code,
                    extra={
                        **(
                            {"revision": self.config.model_revision}
                            if self.config.model_revision
                            else {}
                        )
                    },
                )
            )
        except Exception:
            if not (quantization and self.config.allow_quantization_fallback):
                raise
            model = backend.load_causal_lm(
                LoadSpec(
                    model_name=self.config.model_name,
                    dtype="bfloat16" if self.config.bf16 else None,
                    attn_implementation=self.config.attn_implementation,
                    trust_remote_code=self.config.trust_remote_code,
                    extra={
                        **(
                            {"revision": self.config.model_revision}
                            if self.config.model_revision
                            else {}
                        )
                    },
                )
            )

        if self.config.adaptation == "lora":
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            if self.config.load_in_4bit:
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(
                model,
                LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=self.config.target_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                    use_dora=self.config.use_dora,
                    use_rslora=self.config.use_rslora,
                    init_lora_weights=_parse_init_lora_weights(self.config.init_lora_weights),
                ),
            )
        self.model = model
        return model

    def prepare_packed_corpus(
        self,
        *,
        train_file: Optional[str] = None,
        validation_file: Optional[str] = None,
    ) -> tuple[Any, Optional[Any]]:
        """Load document JSONL and materialize exact packed HF datasets."""

        from datasets import Dataset

        cfg = self.config
        source = train_file or cfg.train_file
        if not source:
            raise ValueError("CPT requires train_file from a corpus training artifact")
        records = load_corpus_jsonl(source)
        validation_source = validation_file or cfg.validation_file
        if validation_source:
            train_records = records
            validation_records = load_corpus_jsonl(validation_source)
        else:
            train_records, validation_records = deterministic_corpus_validation_split(
                records,
                fraction=cfg.validation_fraction,
                seed=cfg.seed,
            )
        tokenizer = self.tokenizer or self._load_tokenizer()
        verify_cpt_training_artifact(
            cfg,
            train_file=source,
            validation_file=validation_source,
        )
        (
            plan,
            train_split,
            validation_split,
            _tokenizer_hash,
            _model_hash,
        ) = _plan_and_splits(
            cfg,
            tokenizer,
            train_records=train_records,
            validation_records=validation_records,
        )
        self.packing_plan = plan
        self.packing_plan_hash = verify_cpt_packing_plan(cfg, plan)
        train_dataset = Dataset.from_list(
            [sequence.to_dict() for sequence in train_split.sequences]
        )
        validation_dataset = (
            Dataset.from_list([sequence.to_dict() for sequence in validation_split.sequences])
            if validation_split is not None
            else None
        )
        return train_dataset, validation_dataset

    def train(
        self,
        train_file: Optional[str] = None,
        validation_file: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run causal next-token CPT and return the canonical summary."""

        from transformers import Trainer, TrainingArguments

        cfg = self.config
        cfg.seed = set_global_seed(cfg.seed)
        self.run_id = resolve_cpt_run_id("cpt")
        started = time.time()

        self._load_tokenizer()
        train_dataset, validation_dataset = self.prepare_packed_corpus(
            train_file=train_file,
            validation_file=validation_file,
        )
        self._load_model()
        assert self.packing_plan is not None

        values: Dict[str, Any] = {
            "output_dir": cfg.output_dir,
            "per_device_train_batch_size": cfg.batch_size,
            "per_device_eval_batch_size": cfg.batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "max_steps": cfg.max_steps or self.packing_plan.estimated_steps,
            "learning_rate": cfg.learning_rate,
            "warmup_ratio": cfg.warmup_ratio,
            "lr_scheduler_type": "cosine",
            "optim": cfg.optim,
            "weight_decay": cfg.weight_decay,
            "max_grad_norm": cfg.max_grad_norm,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "bf16": cfg.bf16,
            "logging_steps": cfg.logging_steps,
            "save_strategy": "steps",
            "save_steps": cfg.save_steps,
            "save_total_limit": cfg.save_total_limit,
            "eval_strategy": "steps" if validation_dataset is not None else "no",
            "eval_steps": cfg.eval_steps,
            "do_eval": validation_dataset is not None,
            "seed": cfg.seed,
            "remove_unused_columns": True,
            "report_to": "tensorboard",
        }
        signature = inspect.signature(TrainingArguments.__init__)
        arguments = TrainingArguments(
            **{
                key: value
                for key, value in values.items()
                if key in signature.parameters and value is not None
            }
        )
        trainer = Trainer(
            model=self.model,
            args=arguments,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            processing_class=self.tokenizer,
            data_collator=CausalLMCollator(self.tokenizer),
        )
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        final_path = Path(cfg.output_dir) / "final_model"
        trainer.save_model(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        duration = time.time() - started

        log_history = list(getattr(trainer.state, "log_history", ()))
        train_losses = [
            float(value["loss"])
            for value in log_history
            if isinstance(value.get("loss"), (int, float))
        ]
        eval_losses = [
            float(value["eval_loss"])
            for value in log_history
            if isinstance(value.get("eval_loss"), (int, float))
        ]
        steps = int(getattr(trainer.state, "global_step", 0))
        summary = self._build_summary(
            duration=duration,
            optimizer_steps=steps,
            train_loss=(
                train_losses[-1] if train_losses else float(getattr(result, "training_loss", 0.0))
            ),
            initial_train_loss=train_losses[0] if train_losses else None,
            eval_loss=eval_losses[-1] if eval_losses else None,
            initial_eval_loss=eval_losses[0] if eval_losses else None,
            final_path=final_path,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        write_json_atomic(Path(cfg.output_dir) / "training_summary.json", summary)
        self.training_summary = summary
        return summary

    def _build_summary(
        self,
        *,
        duration: float,
        optimizer_steps: int,
        train_loss: Optional[float],
        initial_train_loss: Optional[float],
        eval_loss: Optional[float],
        initial_eval_loss: Optional[float],
        final_path: Path,
        resume_from_checkpoint: Optional[str],
    ) -> Dict[str, Any]:
        cfg = self.config
        assert self.packing_plan is not None
        plan = self.packing_plan.to_dict()
        cycle = build_cycle_summary(
            cycle=0,
            learning_rate=cfg.learning_rate,
            samples_seen=self.packing_plan.train_blocks,
            samples_kept=self.packing_plan.train_blocks,
            cycle_duration_seconds=duration,
            update_metrics={
                "train_steps_executed": optimizer_steps,
                "optimizer_steps": optimizer_steps,
                "train_loss": train_loss,
                "initial_train_loss": initial_train_loss,
                "weights_updated": optimizer_steps > 0,
                "update_reason": "updated" if optimizer_steps > 0 else "no_optimizer_steps",
                "skipped_batches_non_finite": 0,
            },
            extra={
                "backend": "hf",
                "objective": "causal_next_token",
                "adaptation": cfg.adaptation,
                "packing_plan": plan,
                "packing_plan_hash": self.packing_plan_hash,
                "eval_loss": eval_loss,
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
                "backend": "hf",
                "objective": "causal_next_token",
                "adaptation": cfg.adaptation,
                "train_file": cfg.train_file or "",
                "validation_file": cfg.validation_file or "",
                "resume_from_checkpoint": resume_from_checkpoint or "",
                "packing_plan": plan,
                "packing_plan_hash": self.packing_plan_hash,
                "model_hash": model_identity_hash(
                    cfg.model_name,
                    revision=cfg.model_revision,
                    explicit_hash=cfg.model_hash,
                ),
                "tokenizer_hash": self.packing_plan.tokenizer_hash,
                "eval_loss": eval_loss,
            },
        )
        summary["final_model_path"] = str(final_path)
        attach_effectiveness_contract(
            summary,
            minimum_samples_kept=1,
            minimum_optimizer_steps=1,
            evaluation={
                "metric_name": "eval_loss",
                "baseline_value": initial_eval_loss,
                "final_value": eval_loss,
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
    "CPTTrainer",
    "CausalLMCollator",
    "deterministic_corpus_validation_split",
    "load_corpus_jsonl",
    "resolve_cpt_run_id",
    "verify_cpt_packing_plan",
    "verify_cpt_training_artifact",
]
