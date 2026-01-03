---
title: "Configuration"
description: "Complete configuration reference"
weight: 1
---

## SFT Configuration

```yaml
# configs/sft.yaml
model:
  name: Qwen/Qwen2.5-Coder-7B       # HuggingFace model name
  trust_remote_code: true           # For custom architectures
  
data:
  train_file: data/train.jsonl      # Training data path
  max_seq_length: 2048              # Maximum sequence length
  validation_split: 0.05            # Validation set proportion
  
training:
  num_train_epochs: 3               # Training epochs
  per_device_train_batch_size: 2    # Batch size per GPU
  gradient_accumulation_steps: 16   # Gradient accumulation
  learning_rate: 2e-5               # Learning rate
  warmup_ratio: 0.03                # Warmup proportion
  weight_decay: 0.01                # L2 regularization
  max_grad_norm: 0.3                # Gradient clipping
  bf16: true                        # Use BF16 precision
  gradient_checkpointing: true      # Save memory
  dataloader_num_workers: 0         # REQUIRED for Strix Halo
  dataloader_pin_memory: false      # REQUIRED for Strix Halo

lora:
  r: 64                             # LoRA rank
  alpha: 128                        # LoRA alpha (scaling)
  dropout: 0.05                     # LoRA dropout
  target_modules:                   # Modules to adapt
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

output:
  dir: models/sft                   # Output directory
  save_steps: 500                   # Checkpoint frequency
  save_total_limit: 3               # Max checkpoints to keep
  logging_steps: 10                 # Log frequency

early_stopping:
  patience: 5                       # Epochs without improvement
  threshold: 0.001                  # Minimum improvement
```

## RAFT Configuration

```yaml
# configs/raft.yaml
sft_checkpoint: models/sft/final_model  # Starting checkpoint
output_dir: models/raft                  # Output directory
prompts: data/prompts.jsonl              # Training prompts

raft:
  num_cycles: 5                     # Number of RAFT cycles
  samples_per_prompt: 8             # Samples to generate
  reward_threshold: 0.5             # Minimum reward to keep
  keep_top_percent: 0.5             # Top % of samples to keep

generation:
  max_new_tokens: 1024              # Max tokens to generate
  temperature: 0.7                  # Sampling temperature
  top_p: 0.95                       # Nucleus sampling
  batch_size: 4                     # Generation batch size

training:
  epochs: 1                         # Epochs per cycle
  batch_size: 2                     # Training batch size
  gradient_accumulation_steps: 16   # Gradient accumulation
  learning_rate: 5e-5               # Learning rate

verifier:
  type: gcc                         # Verifier type
  max_workers: 8                    # Parallel verifications
  timeout: 30                       # Verification timeout
  run_after_compile: false          # Run compiled binary
```

## Verifier Configuration

### GCC/Clang

```yaml
verifier:
  type: gcc  # or clang
  flags:
    - "-w"
    - "-O2"
  timeout: 30
  max_workers: 8
  run_after_compile: false
  run_timeout: 5
  memory_limit_mb: 256
  warn_as_error: false
```

### MinGW (Windows cross-compile)

```yaml
verifier:
  type: mingw
  flags:
    - "-static"
    - "-Wl,--subsystem,console"
    - "-lntdll"
    - "-w"
    - "-O2"
  timeout: 30
  max_workers: 8
```

### Pytest

```yaml
verifier:
  type: pytest
  test_file: tests/test_generated.py
  timeout: 60
  extra_args:
    - "-v"
    - "--tb=short"
```

## Environment Variables

```bash
# ROCm/PyTorch
export HSA_ENABLE_SDMA=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES=0

# HuggingFace
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers

# Training
export TOKENIZERS_PARALLELISM=false
```

## CLI Flags

### SFT

```bash
halo-forge sft train \
  --config configs/sft.yaml \      # Config file
  --data data/train.jsonl \        # Override data path
  --output models/sft \            # Override output
  --model Qwen/Qwen2.5-Coder-7B \  # Override model
  --epochs 3 \                     # Override epochs
  --resume checkpoint-1000         # Resume from checkpoint
```

### RAFT

```bash
halo-forge raft train \
  --config configs/raft.yaml \     # Config file
  --checkpoint models/sft/final \  # SFT checkpoint
  --prompts data/prompts.jsonl \   # Prompts file
  --verifier gcc \                 # Verifier type
  --cycles 5 \                     # Number of cycles
  --samples-per-prompt 8 \         # Samples per prompt
  --output models/raft \           # Output directory
  --resume                         # Resume from last cycle
```

### Benchmark

```bash
halo-forge benchmark run \
  --model models/raft/cycle_5 \    # Model path
  --prompts data/test.jsonl \      # Test prompts
  --verifier gcc \                 # Verifier
  --samples 10 \                   # Samples per prompt
  --k 1,5,10 \                     # k values for pass@k
  --output results/benchmark.json  # Results file
```
