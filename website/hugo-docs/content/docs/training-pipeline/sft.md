---
title: "SFT Training"
description: "Supervised fine-tuning to establish baseline capability"
weight: 2
---

Supervised Fine-Tuning establishes baseline capabilities before RAFT.

## Basic Usage

```bash
halo-forge sft train \
  --data data/train.jsonl \
  --output models/sft \
  --model Qwen/Qwen2.5-Coder-7B \
  --epochs 3
```

## Configuration

```yaml
# configs/sft.yaml
model:
  name: Qwen/Qwen2.5-Coder-7B
  
data:
  train_file: data/train.jsonl
  max_seq_length: 2048

training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 2e-5
  warmup_ratio: 0.03
  bf16: true
  gradient_checkpointing: true
  dataloader_num_workers: 0      # Required for Strix Halo
  dataloader_pin_memory: false   # Required for Strix Halo

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

output:
  dir: models/sft
  save_steps: 500
  logging_steps: 10
```

## LoRA Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| `r` | 64 | Rank, higher = more capacity |
| `alpha` | 128 | Scaling, typically 2x rank |
| `dropout` | 0.05 | Regularization |
| `target_modules` | all linear | Full coverage |

## Learning Rate

- Start with `2e-5` for 7B models
- Use `5e-5` for smaller models (0.5B-3B)
- Warmup helps stability

## Batch Size

Effective batch size = `per_device_batch_size × gradient_accumulation_steps`

For 7B on Strix Halo:
- `per_device_train_batch_size: 2`
- `gradient_accumulation_steps: 16`
- Effective: 32

## Early Stopping

Watch the loss curve. If validation loss stops decreasing:

```yaml
training:
  early_stopping: true
  early_stopping_patience: 3
```

## Output Structure

```
models/sft/
├── checkpoint-500/
├── checkpoint-1000/
├── final_model/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer/
└── training_log.json
```

## Resuming Training

```bash
halo-forge sft train \
  --config configs/sft.yaml \
  --resume models/sft/checkpoint-1000
```

## Why SFT First?

RAFT works by filtering model outputs. If the base model can't produce any valid code, there's nothing to filter.

| Stage | Compile Rate |
|-------|--------------|
| Base Qwen 7B | ~5% |
| After SFT | ~15-25% |
| After RAFT | ~45-55% |

SFT creates the foundation that RAFT refines.
