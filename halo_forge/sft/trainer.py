#!/usr/bin/env python3
"""
SFT Training Module

QLoRA-based supervised fine-tuning optimized for AMD Strix Halo.
Supports Qwen, Llama, and other transformer models.
"""

import os
import sys
import inspect
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

import torch
import yaml
import jsonlines
from datasets import Dataset
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED, build_run_id, set_global_seed
from halo_forge.training_contracts import (
    attach_effectiveness_contract,
    build_cycle_summary,
    build_yield_diagnostics,
    build_training_summary,
    emit_yield_log_line,
    write_json_atomic,
)
from halo_forge.training_recovery import attach_recovery_guidance
from halo_forge.utils.accelerator import (
    detect_gpu_kind,
    empty_accelerator_cache,
    get_device_map,
    get_torch_device,
    is_accelerator_available,
    recommended_attn_impl,
    recommended_dtype,
    supports_4bit_quantization,
)

# SFTConfig moved to halo_forge.sft.config so MLX-only hosts (no torch) can
# import the config without dragging in this module's torch dependencies.
from halo_forge.sft.config import SFTConfig


def _parse_init_lora_weights(value: str):
    """Translate halo-forge's stringly-typed ``init_lora_weights`` to the
    bool / string value PEFT's ``LoraConfig`` accepts.

    PEFT permits ``True`` (default Kaiming init), ``False`` (zero init for
    LoRA-A — fast convergence in some settings), ``"gaussian"``, ``"pissa"``,
    ``"pissa_niter_[N]"``, ``"loftq"``, ``"olora"``, etc. We accept the same
    strings plus the case-insensitive booleans ``"true"`` / ``"false"`` so
    the CLI/YAML surface stays uniform.
    """
    if value is None:
        return True
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return value


class SFTTrainer:
    """
    SFT trainer optimized for AMD Strix Halo.
    
    Features:
    - QLoRA for memory efficiency
    - Smoke test before training
    - Early stopping
    - Resume from checkpoint
    
    Example:
        config = SFTConfig(model_name="Qwen/Qwen2.5-Coder-7B")
        trainer = SFTTrainer(config)
        trainer.train("data/train.jsonl")
    """
    
    def __init__(self, config: Optional[SFTConfig] = None):
        """
        Initialize SFT trainer.
        
        Args:
            config: SFT configuration (uses defaults if None)
        """
        self.config = config or SFTConfig()
        from halo_forge.utils.neural_accelerators import validate_neural_accelerator_opt_in

        validate_neural_accelerator_opt_in(self.config, label="SFT")
        self.model = None
        self.tokenizer = None
        self.training_summary: Dict[str, Union[str, int, float, dict, list, None]] = {}
        self.run_id: str = ""
        self.dataset_yield_diagnostics: Dict[str, Any] = {}
        self.dataset_representative_examples: list[dict[str, Any]] = []
        self._trainable_parameter_hash_before: Optional[str] = None

    def _hash_trainable_parameters(self) -> str:
        """Hash exact trainable tensors for V21 real-path evidence."""

        if self.model is None:
            raise RuntimeError("The model must be prepared before hashing parameters")
        digest = hashlib.sha256()
        count = 0
        for name, parameter in sorted(self.model.named_parameters(), key=lambda item: item[0]):
            if not bool(parameter.requires_grad):
                continue
            count += 1
            encoded = name.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(str(tuple(parameter.shape)).encode("ascii"))
            digest.update(str(parameter.dtype).encode("ascii"))
            raw = parameter.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        if count == 0:
            raise RuntimeError("The trainer has no trainable parameters to certify")
        return digest.hexdigest()
    
    def check_environment(self):
        """Verify the active accelerator and surface what we'll run on.

        Cross-backend (Phase 1 multi-backend port). The trainer used to
        hard-exit when ``torch.cuda.is_available()`` was False, which
        was wrong on Apple Silicon (MPS) and any other non-CUDA host
        the rest of halo-forge supports. We now accept any accelerator
        the kernel/torch combo reports — including CPU as a loud-warning
        fallback so a misconfigured CUDA box still surfaces what it's
        about to do."""
        print("=" * 70)
        print("ENVIRONMENT CHECK")
        print("=" * 70)
        print()

        kind = detect_gpu_kind()
        print(f"PyTorch version: {torch.__version__}")
        print(f"Accelerator: {kind}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")

        if not is_accelerator_available():
            print()
            print("WARNING: no GPU accelerator detected — falling back to CPU.")
            print("Training will be very slow on anything beyond a tiny model.")
            print()
            return

        if kind == "cuda":
            device_name = torch.cuda.get_device_name(0)
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Device: {device_name}")
            print(f"Total memory: {total_memory_gb:.1f} GB")
            if total_memory_gb < 25:
                print(f"\nWARNING: Only {total_memory_gb:.1f} GB detected!")
                print("Expected: ~128GB for Strix Halo")
            else:
                print("Memory check passed")
        elif kind in ("mps", "mlx"):
            # Apple Silicon — torch can't directly query free VRAM, so
            # we just report the device handle and trust the user's
            # box is sized for the model they picked.
            print(f"Device: {get_torch_device()}")
            print("(Apple Silicon / MPS — VRAM is unified system memory)")
        else:
            print(f"Device: {get_torch_device()}")

        print()
    
    def load_dataset(
        self,
        file_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        validation_file: Optional[str] = None,
    ) -> tuple:
        """
        Load dataset from JSONL file or HuggingFace.
        
        Args:
            file_path: Path to local JSONL file
            dataset_name: HuggingFace dataset ID or short name
            
        Returns:
            (train_dataset, val_dataset)
        """
        print("=" * 70)
        print("LOADING DATASET")
        print("=" * 70)
        print()
        
        # Determine source
        dataset_name = dataset_name or self.config.dataset
        file_path = file_path or self.config.train_file
        validation_file = validation_file or self.config.validation_file
        
        if dataset_name:
            # Load from HuggingFace
            from halo_forge.sft.datasets import load_sft_dataset, get_sft_dataset_spec
            
            spec = get_sft_dataset_spec(dataset_name)
            if spec:
                print(f"Loading HuggingFace dataset: {spec.name}")
                print(f"  Source: {spec.huggingface_id}")
                print(f"  Domain: {spec.domain}")
                print(f"  Size: {spec.size_hint}")
            else:
                print(f"Loading HuggingFace dataset: {dataset_name}")
            
            # Pass tokenizer for proper chat template formatting
            dataset = load_sft_dataset(
                dataset_name,
                max_samples=self.config.max_samples,
                split="train",
                tokenizer=self.tokenizer  # Ensures correct BOS tokens
            )
            
            print(f"Loaded {len(dataset)} examples")
            self.dataset_yield_diagnostics = build_yield_diagnostics(
                stage_counts={
                    "generated": len(dataset),
                    "verified": len(dataset),
                    "filtered": len(dataset),
                    "kept": 0,
                    "dropped": 0,
                },
                minimums={"minimum_samples_target": 1},
                summary={
                    "status": "healthy" if len(dataset) > 0 else "no_signal",
                    "text": (
                        "Most samples were usable for SFT."
                        if len(dataset) > 0
                        else "No usable records were found in the dataset."
                    ),
                },
            )
            
        elif file_path:
            # Load from local file
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Dataset not found: {file_path}")
            
            print(f"Loading from local file: {file_path}")
            
            examples = []
            raw_records = 0
            missing_text = 0
            format_invalid = 0
            with jsonlines.open(file_path) as reader:
                for obj in reader:
                    raw_records += 1
                    if not isinstance(obj, dict):
                        format_invalid += 1
                        if len(self.dataset_representative_examples) < 3:
                            self.dataset_representative_examples.append(
                                {
                                    "reason": "format_invalid",
                                    "label": "Malformed dataset row",
                                    "preview": str(obj),
                                    "context": str(file_path),
                                }
                            )
                        continue
                    text = obj.get("text")
                    if not isinstance(text, str):
                        prompt = obj.get("prompt")
                        response = obj.get("response")
                        if isinstance(prompt, str) and isinstance(response, str):
                            text = f"{prompt.strip()}\n{response.strip()}".strip()
                    if not isinstance(text, str):
                        missing_text += 1
                        if len(self.dataset_representative_examples) < 3:
                            self.dataset_representative_examples.append(
                                {
                                    "reason": "missing_text",
                                    "label": "Missing text field",
                                    "preview": str({k: v for k, v in obj.items() if k != "text"}),
                                    "context": str(file_path),
                                }
                            )
                        continue
                    if not text.strip():
                        missing_text += 1
                        if len(self.dataset_representative_examples) < 3:
                            self.dataset_representative_examples.append(
                                {
                                    "reason": "missing_text",
                                    "label": "Empty text field",
                                    "preview": str({k: v for k, v in obj.items() if k != "text"}),
                                    "context": str(file_path),
                                }
                            )
                        continue
                    examples.append({"text": text})

            print(f"Loaded {len(examples)} examples")
            
            dataset = Dataset.from_list(examples)
            
            # Apply max_samples limit
            truncated_count = 0
            if self.config.max_samples and len(dataset) > self.config.max_samples:
                truncated_count = len(dataset) - self.config.max_samples
                dataset = dataset.shuffle(seed=self.config.seed).select(range(self.config.max_samples))
                print(f"Limited to {self.config.max_samples} samples")
            self.dataset_yield_diagnostics = build_yield_diagnostics(
                stage_counts={
                    "generated": raw_records,
                    "verified": len(examples),
                    "filtered": len(dataset),
                    "kept": 0,
                    "dropped": max(0, raw_records - len(dataset)),
                },
                minimums={"minimum_samples_target": 1},
                rejection_reasons={
                    "missing_text": missing_text,
                    "format_invalid": format_invalid,
                    "truncated_or_skipped": truncated_count,
                },
                summary={
                    "status": "healthy" if len(dataset) > 0 else "no_signal",
                    "text": (
                        "Most records were usable for SFT."
                        if len(dataset) > 0 and (missing_text + format_invalid) == 0
                        else (
                            "Many records were skipped because text was missing."
                            if missing_text > format_invalid and missing_text > 0
                            else (
                                "Some records were skipped because they were not valid training rows."
                                if (missing_text + format_invalid + truncated_count) > 0
                                else "No usable records were found in the dataset."
                            )
                        )
                    ),
                },
            )
        else:
            raise ValueError("Either dataset or train_file must be specified")
        
        # Shuffle the training rows deterministically.  A Dataset Lab
        # validation binding is loaded verbatim and never re-split from train.
        dataset = dataset.shuffle(seed=self.config.seed)
        if validation_file:
            validation_path = Path(validation_file).expanduser()
            if not validation_path.is_file():
                raise FileNotFoundError(f"Validation dataset not found: {validation_path}")
            validation_examples = []
            with jsonlines.open(validation_path) as reader:
                for index, obj in enumerate(reader, start=1):
                    if not isinstance(obj, dict):
                        raise ValueError(f"Validation row {index} must be an object")
                    text = obj.get("text")
                    if not isinstance(text, str):
                        prompt = obj.get("prompt")
                        response = obj.get("response")
                        if isinstance(prompt, str) and isinstance(response, str):
                            text = f"{prompt.strip()}\n{response.strip()}".strip()
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"Validation row {index} has no usable text")
                    validation_examples.append({"text": text})
            if not validation_examples:
                raise ValueError("Validation dataset contains no usable rows")
            train_dataset = dataset
            val_dataset = Dataset.from_list(validation_examples)
        else:
            split_idx = int(len(dataset) * (1 - self.config.validation_split))
            if len(dataset) > 0 and split_idx <= 0:
                split_idx = 1
            train_dataset = dataset.select(range(split_idx))
            val_dataset = (
                dataset.select(range(split_idx, len(dataset)))
                if split_idx < len(dataset)
                else dataset.select([])
            )
        
        print(f"  Train: {len(train_dataset)} examples")
        print(f"  Validation: {len(val_dataset)} examples")
        print()

        stage_counts = dict(self.dataset_yield_diagnostics.get("stage_counts") or {})
        if stage_counts:
            stage_counts["kept"] = len(train_dataset)
            # Validation holdout is not rejected data; keep rejection counts tied to filtering.
            stage_counts["dropped"] = max(
                0,
                stage_counts.get("generated", 0) - stage_counts.get("filtered", 0),
            )
            self.dataset_yield_diagnostics = build_yield_diagnostics(
                stage_counts=stage_counts,
                minimums={"minimum_samples_target": 1},
                rejection_reasons=self.dataset_yield_diagnostics.get("rejection_reasons"),
                summary=self.dataset_yield_diagnostics.get("summary"),
            )
            emit_yield_log_line(self.dataset_yield_diagnostics)
        
        return train_dataset, val_dataset
    
    def setup_model(self):
        """Load model with QLoRA and tokenizer."""
        print("=" * 70)
        print("LOADING MODEL & TOKENIZER")
        print("=" * 70)
        print()
        
        cfg = self.config
        
        # Tokenizer
        # Load tokenizer if not already loaded (may be loaded early for dataset formatting)
        if self.tokenizer is None:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_name,
                trust_remote_code=cfg.trust_remote_code
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            print("Tokenizer already loaded (used for dataset formatting)")
        
        print(f"Tokenizer loaded: {len(self.tokenizer)} tokens")
        print()
        
        # Quantization config — gated on backend capability. Apple Silicon
        # MPS and CPU have no bitsandbytes kernels; we either warn and fall
        # back to unquantized or fail loudly per cfg.allow_quantization_fallback.
        use_4bit = bool(cfg.load_in_4bit)
        if use_4bit and not supports_4bit_quantization():
            if cfg.allow_quantization_fallback:
                print(
                    "WARNING: load_in_4bit requested but bitsandbytes is unavailable on "
                    "this backend (Apple Silicon MPS / CPU). Falling back to bf16."
                )
                use_4bit = False
            else:
                raise RuntimeError(
                    "load_in_4bit requires a CUDA/ROCm host with bitsandbytes; "
                    "set SFTConfig.allow_quantization_fallback=True to fall back to bf16."
                )

        if use_4bit:
            print("Configuring 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant
            )
        else:
            print("Loading model in full bf16 precision...")
            bnb_config = None

        # Load model. device_map and attn_implementation route through the
        # accelerator helpers so MPS and CPU hosts pick correct defaults.
        print("Loading base model...")
        attn_impl = cfg.attn_implementation or recommended_attn_impl()
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_config,
            dtype=recommended_dtype(),
            device_map=get_device_map(),
            trust_remote_code=cfg.trust_remote_code,
            attn_implementation=attn_impl
        )

        print("Base model loaded")

        # Prepare for QLoRA
        if use_4bit:
            print("Preparing model for QLoRA...")
            self.model = prepare_model_for_kbit_training(self.model)

        if cfg.lora_r <= 0:
            print("LoRA disabled; using full fine-tuning")
            if cfg.capture_parameter_hashes:
                self._trainable_parameter_hash_before = self._hash_trainable_parameters()
            empty_accelerator_cache()
            print("Model ready")
            print()
            return
        
        # Apply LoRA. PEFT additions (use_dora / use_rslora /
        # init_lora_weights) come from cfg; they default to vanilla LoRA so
        # no behavior change unless the user opts in.
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
        
        print("Applying LoRA adapters...")
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        if cfg.capture_parameter_hashes:
            self._trainable_parameter_hash_before = self._hash_trainable_parameters()
        
        # Model is already placed by device_map; just free transient buffers.
        empty_accelerator_cache()
        print("Model ready")
        
        print()
        
        # Memory check
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"Model loaded: {mem_allocated:.2f} GB allocated")
            print()
    
    def run_smoke_test(self):
        """Quick forward/backward pass to validate setup."""
        print("=" * 70)
        print("SMOKE TEST")
        print("=" * 70)
        print()
        print("Running quick forward/backward pass...")
        
        test_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nTest<|im_end|>\n<|im_start|>assistant\nOK<|im_end|>"
        
        try:
            inputs = self.tokenizer(
                test_text,
                return_tensors="pt",
                max_length=self.config.max_seq_length,
                truncation=True
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            loss.backward()
            
            print(f"Smoke test PASSED (loss: {loss.item():.4f})")
            print()
            
            self.model.zero_grad()
            empty_accelerator_cache()
                
        except Exception as e:
            print(f"Smoke test FAILED: {e}")
            print("\nFix the issue before starting training.")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def tokenize_function(self, examples):
        """Tokenize examples."""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=False,
            return_tensors=None
        )
    
    def _load_tokenizer(self):
        """Load tokenizer early for dataset formatting."""
        if self.tokenizer is not None:
            return  # Already loaded
            
        cfg = self.config
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            trust_remote_code=cfg.trust_remote_code
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def train(
        self,
        train_file: Optional[str] = None,
        validation_file: Optional[str] = None,
        dataset: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None
    ):
        """
        Run SFT training.
        
        Args:
            train_file: Path to training data (overrides config)
            dataset: HuggingFace dataset ID or short name (overrides config)
            resume_from_checkpoint: Checkpoint to resume from
        """
        print("=" * 70)
        print("SFT TRAINING")
        print("=" * 70)
        print()
        
        cfg = self.config
        cfg.seed = set_global_seed(cfg.seed)
        self.run_id = build_run_id("sft")
        
        # Environment check
        self.check_environment()
        
        # Load tokenizer early for proper dataset formatting
        # This ensures correct BOS tokens and chat template are used
        self._load_tokenizer()
        print(f"Tokenizer loaded for dataset formatting: {cfg.model_name}")
        
        # Load data (dataset takes precedence over file)
        # Pass tokenizer so formatters use correct chat template
        train_dataset, val_dataset = self.load_dataset(
            file_path=train_file,
            dataset_name=dataset,
            validation_file=validation_file,
        )
        
        # Setup model (tokenizer already loaded, will be reused)
        self.setup_model()
        
        # Tokenize
        print("=" * 70)
        print("TOKENIZING DATASET")
        print("=" * 70)
        print()
        
        print("Tokenizing training set...")
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['text'],
            desc="Tokenizing train"
        )
        
        print("Tokenizing validation set...")
        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['text'],
            desc="Tokenizing val"
        )
        
        print("Tokenization complete")
        print()
        
        # Smoke test
        self.run_smoke_test()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Training arguments. Transformers 5 removed a few older kwargs
        # (notably overwrite_output_dir), so filter against the installed
        # signature instead of making dashboard training brittle.
        has_eval_data = len(val_dataset) > 0
        model_device = str(getattr(self.model, "device", "")).lower()
        training_arg_values: Dict[str, Any] = {
            "output_dir": cfg.output_dir,
            "overwrite_output_dir": True,
            "per_device_train_batch_size": cfg.batch_size,
            "per_device_eval_batch_size": cfg.batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "num_train_epochs": cfg.num_epochs,
            "max_steps": cfg.max_steps,
            "learning_rate": cfg.learning_rate,
            "warmup_ratio": cfg.warmup_ratio,
            "lr_scheduler_type": "cosine",
            "optim": cfg.optim,
            "weight_decay": cfg.weight_decay,
            "max_grad_norm": cfg.max_grad_norm,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "gradient_checkpointing_kwargs": {'use_reentrant': False},
            "bf16": cfg.bf16,
            "logging_steps": 10,
            "logging_dir": f"{cfg.output_dir}/logs",
            "report_to": "tensorboard",
            "save_strategy": "steps",
            "save_steps": cfg.save_steps,
            "save_total_limit": cfg.save_total_limit,
            "eval_strategy": "steps" if has_eval_data else "no",
            "eval_steps": cfg.eval_steps,
            "do_eval": has_eval_data,
            "load_best_model_at_end": has_eval_data,
            "metric_for_best_model": "eval_loss" if has_eval_data else None,
            "greater_is_better": False if has_eval_data else None,
            "seed": cfg.seed,
            # ROCm optimizations
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": False,
            # Transformers 5 validates bf16 against the selected execution
            # device.  Without this flag a CPU model is treated as a missing
            # GPU and rejected even though CPU bf16 is a supported path.
            "use_cpu": model_device == "cpu" or model_device.startswith("cpu:"),
        }
        signature = inspect.signature(TrainingArguments.__init__)
        training_args = TrainingArguments(
            **{
                key: value
                for key, value in training_arg_values.items()
                if key in signature.parameters and value is not None
            }
        )
        
        print("=" * 70)
        print("CONFIGURATION")
        print("=" * 70)
        print()
        print(f"Batch size: {cfg.batch_size}")
        print(f"Gradient accumulation: {cfg.gradient_accumulation_steps}")
        print(f"Effective batch size: {cfg.batch_size * cfg.gradient_accumulation_steps}")
        print(f"Epochs: {cfg.num_epochs}")
        print(f"Learning rate: {cfg.learning_rate}")
        print(f"Max sequence length: {cfg.max_seq_length}")
        print()
        
        callbacks = []
        if has_eval_data:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=cfg.early_stopping_patience,
                    early_stopping_threshold=cfg.early_stopping_threshold,
                )
            )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if has_eval_data else None,
            processing_class=self.tokenizer,  # Updated from deprecated 'tokenizer'
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        # Train
        print("=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print()
        
        if resume_from_checkpoint:
            print(f"Resuming from: {resume_from_checkpoint}")
        
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        print()
        print("=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print()
        
        # Save final model
        print("Saving final model...")
        final_output = Path(cfg.output_dir) / "final_model"
        trainer.save_model(str(final_output))
        self.tokenizer.save_pretrained(str(final_output))
        
        print(f"Model saved to: {final_output}")
        print()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated(0) / 1e9
            print(f"Peak memory usage: {peak_memory:.2f} GB")
            print()

        log_history = list(getattr(trainer.state, "log_history", []))
        train_loss_points = [
            float(entry["loss"])
            for entry in log_history
            if isinstance(entry, dict) and isinstance(entry.get("loss"), (int, float))
        ]
        eval_loss_points = [
            float(entry["eval_loss"])
            for entry in log_history
            if isinstance(entry, dict) and isinstance(entry.get("eval_loss"), (int, float))
        ]
        total_train_steps = int(
            getattr(train_result, "global_step", 0)
            or getattr(train_result, "metrics", {}).get("global_step", 0)
            or 0
        )
        final_train_loss = (
            train_loss_points[-1]
            if train_loss_points
            else (
                float(train_result.training_loss)
                if isinstance(getattr(train_result, "training_loss", None), (int, float))
                else None
            )
        )
        initial_train_loss = train_loss_points[0] if train_loss_points else final_train_loss
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
                "train_loss": final_train_loss,
                "initial_train_loss": initial_train_loss,
                "weights_updated": total_train_steps > 0,
                "update_reason": "updated" if total_train_steps > 0 else "no_optimizer_steps",
                "optimizer_steps": total_train_steps,
                "skipped_batches_non_finite": 0,
            },
            yield_diagnostics={
                "stage_counts": self.dataset_yield_diagnostics.get("stage_counts"),
                "rates": self.dataset_yield_diagnostics.get("rates"),
                "minimums": {"minimum_samples_target": max(1, len(train_dataset))},
                "rejection_reasons": self.dataset_yield_diagnostics.get("rejection_reasons"),
                "summary": self.dataset_yield_diagnostics.get("summary"),
            },
            extra={
                "train_examples": len(train_dataset),
                "validation_examples": len(val_dataset),
                "eval_loss": eval_loss_points[-1] if eval_loss_points else None,
            },
        )
        summary = build_training_summary(
            modality="sft",
            model_name=cfg.model_name,
            total_cycles_planned=1,
            cycles=[cycle_summary],
            run_id=self.run_id,
            seed=cfg.seed,
            base_model_name=cfg.model_name,
            active_model_name=cfg.model_name,
            yield_diagnostics={
                "stage_counts": self.dataset_yield_diagnostics.get("stage_counts"),
                "rates": self.dataset_yield_diagnostics.get("rates"),
                "minimums": {"minimum_samples_target": max(1, len(train_dataset))},
                "rejection_reasons": self.dataset_yield_diagnostics.get("rejection_reasons"),
                "summary": self.dataset_yield_diagnostics.get("summary"),
            },
            extra={
                "dataset": dataset or cfg.dataset or "",
                "train_file": train_file or cfg.train_file or "",
                "train_examples": len(train_dataset),
                "validation_examples": len(val_dataset),
                "eval_loss": eval_loss_points[-1] if eval_loss_points else None,
                "resume_from_checkpoint": resume_from_checkpoint or "",
            },
        )
        summary["final_model_path"] = str(final_output)
        if cfg.capture_parameter_hashes:
            parameter_hash_after = self._hash_trainable_parameters()
            summary["parameter_evidence"] = {
                "algorithm": "sha256-trainable-tensors-v1",
                "before": self._trainable_parameter_hash_before,
                "after": parameter_hash_after,
                "changed": bool(
                    self._trainable_parameter_hash_before
                    and self._trainable_parameter_hash_before != parameter_hash_after
                ),
            }
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
            modality="sft",
            launch_args={
                "model": cfg.model_name,
                "dataset": dataset or cfg.dataset or "",
                "output_dir": cfg.output_dir,
                "epochs": cfg.num_epochs,
                "batch_size": cfg.batch_size,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "max_samples": cfg.max_samples,
            },
            representative_examples=self.dataset_representative_examples,
        )
        write_json_atomic(Path(cfg.output_dir) / "training_summary.json", summary)
        self.training_summary = summary

        return summary
