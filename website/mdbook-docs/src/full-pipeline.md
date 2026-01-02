# Full Pipeline

Complete guide to training a code generation model with halo-forge.

## Overview

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐
│   Data   │ → │   SFT    │ → │   RAFT   │ → │ Benchmark │
└──────────┘    └──────────┘    └──────────┘    └───────────┘
```

1. **Data** — Gather training examples
2. **SFT** — Supervised fine-tuning for baseline
3. **RAFT** — Iterative verification loop
4. **Benchmark** — Evaluate with pass@k

## Step 1: Data Generation

### Public Datasets

```bash
# CodeForces C++ solutions
halo-forge data prepare --dataset codeforces_cpp --output data/train.jsonl

# MBPP Python problems
halo-forge data prepare --dataset mbpp --output data/train.jsonl

# HumanEval
halo-forge data prepare --dataset humaneval --output data/train.jsonl
```

### LLM Generation

```bash
# Using local Ollama
halo-forge data generate \
  --prompts data/prompts.txt \
  --backend ollama \
  --model deepseek-coder:6.7b \
  --output data/generated.jsonl

# Using Claude API
halo-forge data generate \
  --prompts data/prompts.txt \
  --backend anthropic \
  --model claude-3-sonnet \
  --output data/generated.jsonl
```

## Step 2: SFT Training

```bash
halo-forge sft train \
  --data data/train.jsonl \
  --output models/sft \
  --model Qwen/Qwen2.5-Coder-7B \
  --epochs 3
```

Or with a config file:

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
  bf16: true
  gradient_checkpointing: true

lora:
  r: 64
  alpha: 128
  dropout: 0.05
```

```bash
halo-forge sft train --config configs/sft.yaml
```

## Step 3: RAFT Training

```bash
halo-forge raft train \
  --checkpoint models/sft/final_model \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --cycles 5 \
  --output models/raft
```

### RAFT Configuration

```yaml
# configs/raft.yaml
sft_checkpoint: models/sft/final_model
output_dir: models/raft
prompts: data/prompts.jsonl

raft:
  num_cycles: 5
  samples_per_prompt: 8
  reward_threshold: 0.5
  keep_top_percent: 0.5

generation:
  max_new_tokens: 1024
  temperature: 0.7
  batch_size: 4

training:
  epochs: 1
  batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 5e-5

verifier:
  type: gcc
```

## Step 4: Benchmarking

```bash
# Compare all stages
for stage in sft raft/cycle_1 raft/cycle_3 raft/cycle_5; do
  halo-forge benchmark run \
    --model models/$stage/final_model \
    --prompts data/test.jsonl \
    --verifier gcc \
    --samples 20 \
    --output results/${stage}_benchmark.json
done
```

## Expected Timeline

| Stage | Duration | Notes |
|-------|----------|-------|
| Data prep | 10-30 min | Depends on source |
| SFT (3 epochs) | 2-4 hours | 7B model, 1000 examples |
| RAFT (5 cycles) | 4-8 hours | 500 prompts, 8 samples each |
| Benchmark | 30-60 min | 100 prompts, 20 samples |

## Tips

- **Start small**: Use 0.5B model and 2 cycles for initial testing
- **Monitor loss**: Watch for degradation after cycle 5-6
- **Diverse prompts**: Quality matters more than quantity
- **Checkpoint often**: Each cycle saves a checkpoint
