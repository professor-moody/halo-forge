---
title: "Full Pipeline"
description: "Complete guide to training a code generation model"
weight: 1
---

## Overview

<div class="pipeline-diagram">
  <div class="pipeline-step">
    <div class="step-number">1</div>
    <div class="step-content">
      <div class="step-title">Data Generation</div>
      <div class="step-desc">Public datasets (MBPP, CodeForces) or LLM-generated examples</div>
    </div>
  </div>
  <div class="pipeline-arrow">↓</div>
  <div class="pipeline-step">
    <div class="step-number">2</div>
    <div class="step-content">
      <div class="step-title">SFT Training</div>
      <div class="step-desc">LoRA fine-tuning to establish baseline capability (~15-25% compile rate)</div>
    </div>
  </div>
  <div class="pipeline-arrow">↓</div>
  <div class="pipeline-step raft-step">
    <div class="step-number">3</div>
    <div class="step-content">
      <div class="step-title">RAFT Training</div>
      <div class="step-desc">Generate → Verify → Filter → Train → Repeat</div>
      <div class="raft-loop">5-6 cycles, ~2 hours each</div>
    </div>
  </div>
  <div class="pipeline-arrow">↓</div>
  <div class="pipeline-step">
    <div class="step-number">4</div>
    <div class="step-content">
      <div class="step-title">Benchmark</div>
      <div class="step-desc">pass@k evaluation (~45-55% compile rate after RAFT)</div>
    </div>
  </div>
</div>

<style>
.pipeline-diagram { max-width: 500px; margin: 2rem 0; }
.pipeline-step { display: flex; align-items: flex-start; gap: 1rem; padding: 1rem; background: var(--bg-elevated, #1a1714); border: 1px solid var(--border, #2a2520); border-radius: 8px; }
.pipeline-step.raft-step { border-color: var(--accent, #f97316); }
.step-number { width: 28px; height: 28px; background: var(--accent, #f97316); color: var(--bg-void, #0a0908); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem; flex-shrink: 0; }
.step-content { flex: 1; }
.step-title { font-weight: 600; margin-bottom: 0.25rem; }
.step-desc { font-size: 0.9rem; color: var(--text-secondary, #a8a198); }
.raft-loop { font-size: 0.8rem; color: var(--accent, #f97316); margin-top: 0.5rem; font-style: italic; }
.pipeline-arrow { text-align: center; font-size: 1.5rem; color: var(--text-tertiary, #6b635a); margin: 0.5rem 0; }
</style>

## Step 1: Data Generation

### Option A: Public Datasets

```bash
# List available datasets
halo-forge data prepare --list

# Download CodeForces C++ examples
halo-forge data prepare --dataset codeforces_cpp --output data/train.jsonl
```

### Option B: LLM Generation

```bash
export DEEPSEEK_API_KEY=your_key_here

# Generate Rust async examples
halo-forge data generate \
  --topic rust_async \
  --backend deepseek \
  --output data/rust.jsonl
```

### Data Format

```json
{
  "text": "<|im_start|>system\nYou are an expert programmer.<|im_end|>\n<|im_start|>user\nWrite a function to...<|im_end|>\n<|im_start|>assistant\n```cpp\n#include...\n```<|im_end|>"
}
```

## Step 2: SFT Training

Supervised fine-tuning establishes baseline capability:

```bash
halo-forge sft train \
  --data data/train.jsonl \
  --output models/sft \
  --epochs 3
```

### Why SFT First?

| Stage | Compile Rate |
|-------|--------------|
| Base Qwen 7B | ~5% |
| After SFT | ~15-25% |
| After RAFT | ~45-55% |

RAFT filters model outputs. Without SFT, there's nothing useful to filter.

## Step 3: RAFT Training

Iterative verification loop:

```bash
halo-forge raft train \
  --checkpoint models/sft/final_model \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --cycles 5 \
  --samples-per-prompt 8 \
  --output models/raft
```

### RAFT Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cycles` | 5 | Number of RAFT iterations |
| `samples-per-prompt` | 8 | Samples to generate per prompt |
| `reward-threshold` | 0.5 | Minimum reward to keep |
| `keep-top-percent` | 0.5 | Top % of samples above threshold |

### Cycle Dynamics

```
Cycle 1: Generate → Verify → Filter (keep 40%) → Train
Cycle 2: Generate → Verify → Filter (keep 50%) → Train
Cycle 3: Generate → Verify → Filter (keep 55%) → Train
...
```

Each cycle improves the model's ability to generate code that passes verification.

## Step 4: Benchmark

Evaluate the trained model:

```bash
halo-forge benchmark run \
  --model models/raft/cycle_5_final \
  --prompts data/test.jsonl \
  --verifier gcc \
  --samples 10 \
  --k 1,5,10
```

### pass@k Metrics

- **pass@1**: Probability first sample is correct
- **pass@5**: Probability at least 1 of 5 samples is correct
- **pass@10**: Probability at least 1 of 10 samples is correct

## Complete Example

```bash
# 1. Prepare data
halo-forge data prepare --dataset codeforces_cpp --output data/train.jsonl

# 2. Extract prompts for RAFT
head -200 data/train.jsonl | jq -c '{prompt: .text | split("<|im_start|>user\n")[1] | split("<|im_end|>")[0]}' > data/prompts.jsonl

# 3. Run SFT
halo-forge sft train \
  --data data/train.jsonl \
  --output models/sft \
  --epochs 3

# 4. Run RAFT
halo-forge raft train \
  --checkpoint models/sft/final_model \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --cycles 5 \
  --output models/raft

# 5. Benchmark
halo-forge benchmark run \
  --model models/raft/cycle_5_final \
  --prompts data/test.jsonl \
  --verifier gcc
```

## Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Data prep | 5-10 min | Depends on dataset size |
| SFT | 1-2 hours | 3 epochs, 7B model |
| RAFT (5 cycles) | 8-12 hours | ~2 hours per cycle |
| Benchmark | 30-60 min | Depends on samples |

Total: ~12-16 hours for complete pipeline.
