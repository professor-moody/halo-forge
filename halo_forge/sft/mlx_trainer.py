"""MLX LoRA SFT trainer.

Sibling of `halo_forge.sft.trainer.SFTTrainer` for the Apple MLX backend.
Same `SFTConfig`; same `train(resume_from_checkpoint=...)` -> summary contract
so the CLI / public API don't need to know which backend is running.

Architecture:
    1. Load model + tokenizer via `mlx_lm.load` (handles MLX-format weights)
    2. Apply LoRA layers via `mlx_lm.tuner.utils.linear_to_lora_layers`
    3. Reuse halo_forge.sft.datasets to fetch + format the training data —
       same dataset registry as the PyTorch path so users can swap backends
       without editing data configs.
    4. Materialize formatted records as JSONL in a temp dir (mlx_lm's tuner
       wants a `data` directory containing `train.jsonl` / `valid.jsonl`).
    5. Run `mlx_lm.tuner.trainer.train` against the temp dir.
    6. Save adapter weights to `cfg.output_dir/final_model/adapters.safetensors`.
    7. Build canonical summary via `halo_forge.training_contracts` so
       downstream code (effectiveness gates, public API, recovery guidance)
       sees the same shape as the PyTorch SFT path.

Limitations vs. PyTorch SFT (intentional, scoped to Phase 4):
    - No QLoRA / bitsandbytes — MLX bakes quantization into the model
      artifact (see docs/MLX.md). load_in_4bit raises with a clear redirect.
    - No HF `Trainer` callbacks (early stopping is implemented inline).
    - No DeepSpeed / FSDP / accelerate. MLX is single-device on Apple Silicon.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from halo_forge.runtime_determinism import (
    build_run_id,
    set_global_seed,
)
from halo_forge.sft.config import SFTConfig
from halo_forge.training_contracts import (
    attach_effectiveness_contract,
    build_cycle_summary,
    build_training_summary,
    build_yield_diagnostics,
    write_json_atomic,
)
from halo_forge.training_recovery import attach_recovery_guidance


class MLXTrainerUnavailable(RuntimeError):
    """Raised when the MLX backend is requested but mlx-lm isn't installed."""


def _require_mlx_lm():
    """Import mlx_lm + mlx, raising a clean error if missing.

    Kept as a function (not a top-level import) so this module imports cleanly
    on Linux/CUDA hosts where mlx is unavailable. The dispatcher in
    halo_forge.sft._dispatch only routes MLX-flavored configs here.
    """
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        from mlx_lm import load as mlx_load
        from mlx_lm.tuner.datasets import CacheDataset, TextDataset
        from mlx_lm.tuner.trainer import TrainingArgs, train as mlx_tuner_train
        from mlx_lm.tuner.utils import linear_to_lora_layers
    except ImportError as exc:
        raise MLXTrainerUnavailable(
            "MLX SFT requires the `[mlx]` extra. Install with: "
            "`pip install -e '.[mlx]'` (Apple Silicon only)."
        ) from exc
    return {
        "mx": mx,
        "nn": nn,
        "optim": optim,
        "tree_flatten": tree_flatten,
        "mlx_load": mlx_load,
        "TextDataset": TextDataset,
        "CacheDataset": CacheDataset,
        "TrainingArgs": TrainingArgs,
        "mlx_tuner_train": mlx_tuner_train,
        "linear_to_lora_layers": linear_to_lora_layers,
    }


class MLXSFTTrainer:
    """LoRA SFT trainer over MLX. Mirrors the public surface of `SFTTrainer`.

    Expected usage from CLI / dispatcher:
        config = SFTConfig(model_name="mlx-community/Qwen2.5-3B-Instruct-bf16")
        trainer = MLXSFTTrainer(config)
        summary = trainer.train(resume_from_checkpoint=None)
    """

    def __init__(self, config: Optional[SFTConfig] = None) -> None:
        self.config = config or SFTConfig()
        # Tracks T4/T5 silent-failure fix: warn loudly when the user
        # opts into PEFT / optimizer features that MLX can't honor
        # (DoRA/rsLoRA/PiSSA/bnb-optimizer). Without this, MLX runs
        # quietly fall back to vanilla LoRA / mlx.optimizers.AdamW.
        from halo_forge.utils.backend_config import warn_unsupported_for_mlx
        from halo_forge.utils.neural_accelerators import validate_neural_accelerator_opt_in

        warn_unsupported_for_mlx(self.config, trainer_label="MLX SFT")
        validate_neural_accelerator_opt_in(self.config, label="MLX SFT")
        self.model: Any = None
        self.tokenizer: Any = None
        self.run_id: str = ""
        self.training_summary: Dict[str, Any] = {}
        self.dataset_yield_diagnostics: Dict[str, Any] = {}
        self.dataset_representative_examples: list[dict[str, Any]] = []
        # Phase 5c: when set, the next `train()` will load the base model
        # *with* this adapter pre-applied via mlx_lm's `adapter_path` kwarg.
        # The MLX RAFT trainer sets this between cycles so SFT chains across
        # cycles instead of starting from base weights every time.
        self.resume_adapter_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.load_in_4bit:
            # bitsandbytes-style quantization is not honored on MLX; the
            # backend already documents this in docs/MLX.md.
            if cfg.allow_quantization_fallback:
                print(
                    "WARNING: load_in_4bit is not honored on the MLX backend; "
                    "use a pre-quantized model artifact (e.g. "
                    "`mlx-community/...-4bit`) or run `python -m mlx_lm.convert -q`. "
                    "Continuing with the model as loaded."
                )
            else:
                raise MLXTrainerUnavailable(
                    "load_in_4bit is unavailable on MLX. Set "
                    "SFTConfig.allow_quantization_fallback=True or use a "
                    "pre-quantized model artifact."
                )

    # ------------------------------------------------------------------
    # Dataset preparation: HF Dataset -> mlx_lm-style JSONL
    # ------------------------------------------------------------------

    def _load_and_materialize(self, data_dir: Path) -> tuple[int, int]:
        """Fetch + format the training data, write `train.jsonl` and
        `valid.jsonl` into `data_dir`. Returns (n_train, n_val).

        Reuses the registry in `halo_forge.sft.datasets` so the same
        `--dataset codealpaca` flag works on both backends.
        """
        cfg = self.config
        dataset_name = cfg.dataset
        train_file = cfg.train_file

        records: list[Dict[str, Any]] = []
        if dataset_name:
            from halo_forge.sft.datasets import (
                get_sft_dataset_spec,
                load_sft_dataset,
            )

            spec = get_sft_dataset_spec(dataset_name)
            display = spec.huggingface_id if spec else dataset_name
            print(f"Loading HF dataset for MLX: {display}")
            ds = load_sft_dataset(
                dataset_name,
                max_samples=cfg.max_samples,
                split="train",
                tokenizer=self.tokenizer,
            )
            for row in ds:
                text = self._row_to_text(row)
                if text:
                    records.append({"text": text})
        elif train_file:
            print(f"Loading local JSONL: {train_file}")
            path = Path(train_file)
            if not path.exists():
                raise FileNotFoundError(f"train_file not found: {train_file}")
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = self._row_to_text(obj)
                    if text:
                        records.append({"text": text})
        else:
            raise ValueError("SFTConfig requires either `dataset` or `train_file`.")

        if not records:
            raise ValueError("No usable training records produced; check dataset config.")

        # Apply max_samples limit if set (mirrors PyTorch path).
        if cfg.max_samples is not None and len(records) > cfg.max_samples:
            records = records[: cfg.max_samples]

        # Validation split — mirror the PyTorch trainer's behavior so loss
        # curves are comparable across backends.
        n_val = max(1, int(len(records) * cfg.validation_split)) if cfg.validation_split else 0
        n_val = min(n_val, len(records) - 1)  # always keep ≥1 train sample
        train_records = records[:-n_val] if n_val > 0 else records
        val_records = records[-n_val:] if n_val > 0 else []

        data_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(data_dir / "train.jsonl", train_records)
        _write_jsonl(data_dir / "valid.jsonl", val_records or train_records[:1])  # mlx_lm wants both

        # Stash a few representative samples for recovery guidance.
        self.dataset_representative_examples = train_records[:3]
        self.dataset_yield_diagnostics = build_yield_diagnostics(
            stage_counts={
                "generated": len(records),
                "verified": len(records),
                "filtered": len(records),
                "kept": len(train_records),
                "dropped": 0,
            },
            minimums={"minimum_samples_target": 1},
            summary={
                "status": "healthy" if train_records else "no_signal",
                "text": (
                    "Most samples were usable for SFT."
                    if train_records
                    else "No usable records were found in the dataset."
                ),
            },
        )

        print(f"  train={len(train_records)} val={len(val_records)}")
        return len(train_records), len(val_records)

    def _row_to_text(self, row: Any) -> str:
        """Coerce a dataset row into a plain `text` string for mlx_lm.

        load_sft_dataset emits HF Dataset rows that have already been chat-
        templated by the tokenizer (the `text` column is the formatted
        prompt+response). We accept either a `text` field or fall back to
        common alternatives so locally-loaded JSONL still works.
        """
        if isinstance(row, dict):
            for key in ("text", "messages", "content", "completion", "output"):
                if key in row and row[key]:
                    val = row[key]
                    if isinstance(val, str):
                        return val
                    # If `messages` is a list of dicts, render via tokenizer
                    # chat template when available.
                    if isinstance(val, list) and self.tokenizer is not None:
                        try:
                            return self.tokenizer.apply_chat_template(
                                val, tokenize=False, add_generation_prompt=False
                            )
                        except Exception:
                            return json.dumps(val)
        return ""

    # ------------------------------------------------------------------
    # LoRA setup
    # ------------------------------------------------------------------

    def _apply_lora(self, deps: Dict[str, Any]) -> None:
        cfg = self.config
        if cfg.lora_r <= 0:
            print("LoRA rank=0; running full fine-tune (not recommended on MLX)")
            return

        # mlx_lm's lora.py reference order: freeze the base FIRST, then
        # `linear_to_lora_layers` swaps in trainable LoRA modules. Reverse
        # order produces 0 trainable params because `freeze()` sweeps all
        # params including the just-added LoRA tensors.
        try:
            self.model.freeze()
        except Exception:
            pass

        # mlx_lm's `keys` field wants fully-qualified module paths (e.g.
        # "self_attn.q_proj") not bare attribute names like SFTConfig.target_modules.
        # When omitted, linear_to_lora_layers auto-discovers every Linear /
        # QuantizedLinear / Embedding under the targeted blocks, which is the
        # behavior we want by default. Custom keys can still be threaded
        # through SFTConfig.target_modules if the user provides them in the
        # FQN form.
        lora_config = {
            "rank": cfg.lora_r,
            "alpha": cfg.lora_alpha,
            "dropout": cfg.lora_dropout,
            "scale": float(cfg.lora_alpha) / float(cfg.lora_r) if cfg.lora_r else 1.0,
        }
        if cfg.target_modules and any("." in m for m in cfg.target_modules):
            lora_config["keys"] = list(cfg.target_modules)

        # `num_layers` controls how many *transformer blocks* (counted from
        # the top) get LoRA applied. mlx_lm doesn't accept -1 as "all" the
        # way the PyTorch path does — we count the model's blocks and pass
        # an explicit number. Falls back to a sensible default for models
        # that don't expose `.layers`.
        num_blocks = self._count_transformer_blocks() or 16
        deps["linear_to_lora_layers"](
            self.model, num_layers=num_blocks, config=lora_config
        )
        # Stashed for adapter_config.json — see train() for why mlx_lm
        # needs the layer count baked into the artifact.
        self._lora_num_layers = num_blocks

        trainable_params = sum(
            v.size for _, v in deps["tree_flatten"](self.model.trainable_parameters())
        )
        print(f"LoRA applied: rank={cfg.lora_r} alpha={cfg.lora_alpha} "
              f"layers={num_blocks} trainable_params={trainable_params:,}")

    def _count_transformer_blocks(self) -> int:
        """Best-effort count of transformer blocks in `self.model`.

        mlx_lm's models expose either `model.layers` (Qwen, Llama) or
        `model.model.layers` (Gemma sometimes). Returns 0 if neither
        attribute is present so the caller can fall back to a default.
        """
        layers = getattr(self.model, "layers", None)
        if layers is None:
            inner = getattr(self.model, "model", None)
            if inner is not None:
                layers = getattr(inner, "layers", None)
        try:
            return len(layers) if layers is not None else 0
        except TypeError:
            return 0

    # ------------------------------------------------------------------
    # Training entry point
    # ------------------------------------------------------------------

    def train(
        self,
        train_file: Optional[str] = None,
        dataset: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run MLX LoRA SFT and return the canonical summary dict.

        Args mirror SFTTrainer.train so the dispatcher can call either
        trainer with the same kwargs.
        """
        deps = _require_mlx_lm()
        cfg = self.config
        if dataset is not None:
            cfg.dataset = dataset
        if train_file is not None:
            cfg.train_file = train_file

        self._validate_config()
        self.run_id = build_run_id("sft_mlx")
        seed = set_global_seed(cfg.seed)

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load model + tokenizer from MLX-format weights. If a previous
        #    adapter was set (Phase 5c RAFT chaining), pass it to mlx_lm.load
        #    so the base + adapter are fused before LoRA layers go on top.
        print(f"Loading MLX model: {cfg.model_name}")
        load_kwargs: Dict[str, Any] = {}
        if self.resume_adapter_path:
            print(f"Resuming from adapter: {self.resume_adapter_path}")
            load_kwargs["adapter_path"] = self.resume_adapter_path
        try:
            self.model, self.tokenizer = deps["mlx_load"](cfg.model_name, **load_kwargs)
        except TypeError:
            # Older mlx_lm versions don't accept adapter_path; fall back to
            # loading without it and warn so the user knows chaining was
            # skipped this run.
            if load_kwargs:
                print(
                    "WARNING: installed mlx_lm.load doesn't accept adapter_path; "
                    "loading base only. Upgrade mlx-lm to chain adapters across cycles."
                )
            self.model, self.tokenizer = deps["mlx_load"](cfg.model_name)

        # 2. Materialize formatted training data into a temp dir.
        with tempfile.TemporaryDirectory(prefix="halo_forge_mlx_") as tmpdir:
            data_dir = Path(tmpdir)
            n_train, n_val = self._load_and_materialize(data_dir)

            # 3. Apply LoRA.
            self._apply_lora(deps)

            # 4. Build mlx_lm TrainingArgs from SFTConfig.
            adapter_dir = output_dir / "final_model"
            adapter_dir.mkdir(parents=True, exist_ok=True)

            iters = self._estimate_iters(n_train)
            steps_per_eval = max(1, min(cfg.eval_steps, max(1, iters // 4)))
            steps_per_save = max(1, min(cfg.save_steps, max(1, iters // 2)))

            args = deps["TrainingArgs"](
                batch_size=cfg.batch_size,
                iters=iters,
                val_batches=max(1, min(25, n_val // max(1, cfg.batch_size))),
                steps_per_report=10,
                steps_per_eval=steps_per_eval,
                steps_per_save=steps_per_save,
                max_seq_length=cfg.max_seq_length,
                grad_checkpoint=cfg.gradient_checkpointing,
                adapter_file=str(adapter_dir / "adapters.safetensors"),
            )

            optimizer = deps["optim"].AdamW(
                learning_rate=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )

            # 5. mlx_lm.tuner.train wants dataset *objects*, not file paths,
            #    and the tokenizer is baked into the datasets rather than
            #    passed alongside. We re-read our temp JSONL into mlx_lm's
            #    TextDataset wrapper — same records, the right shape.
            train_records = _read_jsonl(data_dir / "train.jsonl")
            val_records = _read_jsonl(data_dir / "valid.jsonl")
            # `CacheDataset` lazily tokenizes via the inner dataset's
            # .process() method on first __getitem__. mlx_lm's trainer
            # iterates `dataset[idx][0]` expecting a token-id list, which
            # is exactly what the cached pair `(token_ids, 0)` provides.
            train_ds = deps["CacheDataset"](
                deps["TextDataset"](train_records, self.tokenizer)
            )
            val_ds = deps["CacheDataset"](
                deps["TextDataset"](val_records, self.tokenizer)
            )

            print("=" * 70)
            print("STARTING MLX LoRA TRAINING")
            print("=" * 70)
            t0 = time.time()
            train_loss_history: list[float] = []
            eval_loss_history: list[float] = []

            class _LossCollector:
                """Capture mlx_lm's per-step metrics into our history lists."""

                def on_train_loss_report(self, info: Dict[str, Any]) -> None:
                    loss = info.get("train_loss")
                    if isinstance(loss, (int, float)):
                        train_loss_history.append(float(loss))

                def on_val_loss_report(self, info: Dict[str, Any]) -> None:
                    loss = info.get("val_loss")
                    if isinstance(loss, (int, float)):
                        eval_loss_history.append(float(loss))

            deps["mlx_tuner_train"](
                model=self.model,
                optimizer=optimizer,
                train_dataset=train_ds,
                val_dataset=val_ds,
                args=args,
                training_callback=_LossCollector(),
            )

            # mlx_lm.load(adapter_path=...) reads adapter_config.json to
            # rebuild LoRA layers on the base model before loading weights.
            # mlx_lm.tuner.trainer.train doesn't write that file — we have
            # to ourselves so the next RAFT cycle (or a manual reload) can
            # chain adapters via the documented mechanism.
            adapter_config = {
                "fine_tune_type": "lora",
                "num_layers": getattr(self, "_lora_num_layers", 16),
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

            duration = time.time() - t0

        # 6. Save tokenizer alongside the adapter for inference parity.
        try:
            self.tokenizer.save_pretrained(str(adapter_dir))
        except Exception:
            pass

        # 7. Build canonical summary dict so downstream consumers (effectiveness
        # gates, public API run detail, recovery flow) see the same shape as
        # the PyTorch SFT trainer's output.
        summary = self._build_summary(
            n_train=n_train,
            n_val=n_val,
            duration=duration,
            train_loss_history=train_loss_history,
            eval_loss_history=eval_loss_history,
            iters=iters,
            adapter_path=adapter_dir,
            seed=seed,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        write_json_atomic(output_dir / "training_summary.json", summary)
        self.training_summary = summary
        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_iters(self, n_train: int) -> int:
        """Convert epochs → iter count the way mlx_lm.tuner expects."""
        cfg = self.config
        steps_per_epoch = max(1, n_train // max(1, cfg.batch_size))
        return max(1, steps_per_epoch * cfg.num_epochs)

    def _build_summary(
        self,
        *,
        n_train: int,
        n_val: int,
        duration: float,
        train_loss_history: list[float],
        eval_loss_history: list[float],
        iters: int,
        adapter_path: Path,
        seed: int,
        resume_from_checkpoint: Optional[str],
    ) -> Dict[str, Any]:
        cfg = self.config
        final_train_loss = train_loss_history[-1] if train_loss_history else None
        initial_train_loss = train_loss_history[0] if train_loss_history else final_train_loss
        final_eval_loss = eval_loss_history[-1] if eval_loss_history else None
        initial_eval_loss = eval_loss_history[0] if eval_loss_history else final_eval_loss

        weights_updated = bool(train_loss_history) and iters > 0
        cycle_summary = build_cycle_summary(
            cycle=0,
            learning_rate=cfg.learning_rate,
            samples_seen=n_train,
            samples_kept=n_train,
            cycle_duration_seconds=duration,
            update_metrics={
                "train_steps_executed": iters,
                "train_loss": final_train_loss,
                "initial_train_loss": initial_train_loss,
                "weights_updated": weights_updated,
                "update_reason": "updated" if weights_updated else "no_optimizer_steps",
                "optimizer_steps": iters,
                "skipped_batches_non_finite": 0,
            },
            yield_diagnostics={
                "stage_counts": self.dataset_yield_diagnostics.get("stage_counts"),
                "rates": self.dataset_yield_diagnostics.get("rates"),
                "minimums": {"minimum_samples_target": max(1, n_train)},
                "rejection_reasons": self.dataset_yield_diagnostics.get("rejection_reasons"),
                "summary": self.dataset_yield_diagnostics.get("summary"),
            },
            extra={
                "train_examples": n_train,
                "validation_examples": n_val,
                "eval_loss": final_eval_loss,
                "backend": "mlx",
            },
        )

        summary = build_training_summary(
            modality="sft",
            model_name=cfg.model_name,
            total_cycles_planned=1,
            cycles=[cycle_summary],
            run_id=self.run_id,
            seed=seed,
            base_model_name=cfg.model_name,
            active_model_name=cfg.model_name,
            yield_diagnostics={
                "stage_counts": self.dataset_yield_diagnostics.get("stage_counts"),
                "rates": self.dataset_yield_diagnostics.get("rates"),
                "minimums": {"minimum_samples_target": max(1, n_train)},
                "rejection_reasons": self.dataset_yield_diagnostics.get("rejection_reasons"),
                "summary": self.dataset_yield_diagnostics.get("summary"),
            },
            extra={
                "dataset": cfg.dataset or "",
                "train_file": cfg.train_file or "",
                "train_examples": n_train,
                "validation_examples": n_val,
                "eval_loss": final_eval_loss,
                "resume_from_checkpoint": resume_from_checkpoint or "",
                "backend": "mlx",
            },
        )
        summary["final_model_path"] = str(adapter_path)
        attach_effectiveness_contract(
            summary,
            minimum_samples_kept=max(1, n_train),
            minimum_optimizer_steps=1,
            evaluation={
                "metric_name": "eval_loss",
                "baseline_value": initial_eval_loss,
                "final_value": final_eval_loss,
                "higher_is_better": False,
                "tolerance": 0.0,
            },
            evaluation_required=False,
            checkpoint_written=adapter_path.exists(),
            final_model_path=str(adapter_path),
            training_summary_path=Path(cfg.output_dir) / "training_summary.json",
        )
        attach_recovery_guidance(
            summary,
            modality="sft",
            launch_args={
                "model": cfg.model_name,
                "dataset": cfg.dataset or "",
                "output_dir": cfg.output_dir,
                "epochs": cfg.num_epochs,
                "batch_size": cfg.batch_size,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "max_samples": cfg.max_samples,
                "backend": "mlx",
            },
            representative_examples=self.dataset_representative_examples,
        )
        return summary


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Compact JSONL writer; one record per line."""
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    """Counterpart of `_write_jsonl`. Skips blank lines."""
    records: list[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


__all__ = [
    "MLXSFTTrainer",
    "MLXTrainerUnavailable",
]
