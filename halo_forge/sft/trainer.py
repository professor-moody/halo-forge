#!/usr/bin/env python3
"""
SFT Training Module

QLoRA-based supervised fine-tuning optimized for AMD Strix Halo.
Supports Qwen, Llama, and other transformer models.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
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


@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    
    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-7B"
    trust_remote_code: bool = True
    attn_implementation: str = "eager"  # Required for ROCm
    
    # Data - supports both local files and HuggingFace datasets
    train_file: Optional[str] = None  # Local JSONL file
    dataset: Optional[str] = None  # HuggingFace dataset ID or short name
    max_samples: Optional[int] = None  # Limit number of samples
    validation_split: float = 0.05
    max_seq_length: int = 2048
    
    # QLoRA (4-bit is slower on Strix Halo - use BF16 by default)
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # Training
    output_dir: str = "models/sft"
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    
    # Optimization
    bf16: bool = True
    gradient_checkpointing: bool = True
    
    # Saving
    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 250
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default for Qwen models
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"]
    
    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        # Flatten nested config
        flat = {}
        for section in ['model', 'data', 'lora', 'qlora', 'training']:
            if section in data:
                flat.update(data[section])
        
        # Map config keys
        key_map = {
            'name': 'model_name',
            'train_file': 'train_file',
            'per_device_train_batch_size': 'batch_size',
            'num_train_epochs': 'num_epochs',
            'r': 'lora_r',
            'alpha': 'lora_alpha',
            'dropout': 'lora_dropout',
        }
        
        mapped = {}
        for k, v in flat.items():
            mapped_key = key_map.get(k, k)
            if hasattr(cls, '__dataclass_fields__') and mapped_key in cls.__dataclass_fields__:
                mapped[mapped_key] = v
        
        return cls(**mapped)


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
        self.model = None
        self.tokenizer = None
    
    def check_environment(self):
        """Verify ROCm/PyTorch environment."""
        print("=" * 70)
        print("ENVIRONMENT CHECK")
        print("=" * 70)
        print()
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if not torch.cuda.is_available():
            print("\nERROR: CUDA/ROCm not available!")
            print("Make sure you're inside the halo-forge toolbox.")
            sys.exit(1)
        
        device_name = torch.cuda.get_device_name(0)
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"Device: {device_name}")
        print(f"Total memory: {total_memory_gb:.1f} GB")
        
        if total_memory_gb < 25:
            print(f"\nWARNING: Only {total_memory_gb:.1f} GB detected!")
            print("Expected: ~128GB for Strix Halo")
        else:
            print("Memory check passed")
        
        print()
    
    def load_dataset(self, file_path: Optional[str] = None, dataset_name: Optional[str] = None) -> tuple:
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
            
        elif file_path:
            # Load from local file
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Dataset not found: {file_path}")
            
            print(f"Loading from local file: {file_path}")
            
            examples = []
            with jsonlines.open(file_path) as reader:
                for obj in reader:
                    if 'text' in obj:
                        examples.append({'text': obj['text']})
            
            print(f"Loaded {len(examples)} examples")
            
            dataset = Dataset.from_list(examples)
            
            # Apply max_samples limit
            if self.config.max_samples and len(dataset) > self.config.max_samples:
                dataset = dataset.shuffle(seed=42).select(range(self.config.max_samples))
                print(f"Limited to {self.config.max_samples} samples")
        else:
            raise ValueError("Either dataset or train_file must be specified")
        
        # Shuffle and split
        dataset = dataset.shuffle(seed=42)
        
        split_idx = int(len(dataset) * (1 - self.config.validation_split))
        train_dataset = dataset.select(range(split_idx))
        val_dataset = dataset.select(range(split_idx, len(dataset)))
        
        print(f"  Train: {len(train_dataset)} examples")
        print(f"  Validation: {len(val_dataset)} examples")
        print()
        
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
        
        # Quantization config
        if cfg.load_in_4bit:
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
        
        # Load model directly on GPU (unified memory handles this automatically)
        print("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # Unified memory on Strix Halo makes this optimal
            trust_remote_code=cfg.trust_remote_code,
            attn_implementation=cfg.attn_implementation
        )
        
        print("Base model loaded")
        
        # Prepare for QLoRA
        if cfg.load_in_4bit:
            print("Preparing model for QLoRA...")
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        print("Applying LoRA adapters...")
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Note: With device_map="auto", model is already on GPU
        # Just clear cache for memory efficiency
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Model ready on GPU")
        
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
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
            dataset_name=dataset
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
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            overwrite_output_dir=True,
            
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            
            num_train_epochs=cfg.num_epochs,
            
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type="cosine",
            
            optim="adamw_torch",
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm,
            
            gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            bf16=cfg.bf16,
            
            logging_steps=10,
            logging_dir=f"{cfg.output_dir}/logs",
            report_to="tensorboard",
            
            save_strategy="steps",
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            
            eval_strategy="steps",
            eval_steps=cfg.eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            seed=42,
            
            # ROCm optimizations
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
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
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,  # Updated from deprecated 'tokenizer'
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=cfg.early_stopping_patience,
                    early_stopping_threshold=cfg.early_stopping_threshold
                )
            ]
        )
        
        # Train
        print("=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print()
        
        if resume_from_checkpoint:
            print(f"Resuming from: {resume_from_checkpoint}")
        
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
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
        
        return str(final_output)

