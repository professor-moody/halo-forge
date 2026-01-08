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
# Using HuggingFace dataset (recommended)
halo-forge sft train \
  --dataset codealpaca \
  --model Qwen/Qwen2.5-Coder-7B \
  --output models/sft \
  --epochs 3

# Or using local data
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

## Complete Example: Code Domain

```bash
# 1. SFT Training (using HuggingFace dataset)
halo-forge sft train \
  --dataset codealpaca \
  --model Qwen/Qwen2.5-Coder-3B \
  --output models/code_sft \
  --epochs 2

# 2. RAFT Training
halo-forge raft train \
  --model models/code_sft \
  --prompts data/rlvr/humaneval_prompts.jsonl \
  --verifier gcc \
  --cycles 5 \
  --output models/code_raft

# 3. Benchmark
halo-forge benchmark run \
  --model models/code_raft \
  --prompts data/rlvr/humaneval_prompts.jsonl \
  --verifier gcc \
  --samples 10
```

---

## Full Pipeline: All Domains

### Reasoning (Math)

```bash
# SFT → RAFT → Benchmark
halo-forge reasoning sft --dataset metamath --model Qwen/Qwen2.5-3B-Instruct --output models/reasoning_sft
halo-forge reasoning train --model models/reasoning_sft --dataset gsm8k --cycles 5 --output models/reasoning_raft
halo-forge reasoning benchmark --model models/reasoning_raft --dataset gsm8k
```

### Audio (ASR)

```bash
# SFT → RAFT → Benchmark
halo-forge audio sft --dataset librispeech_sft --model openai/whisper-small --output models/audio_sft
halo-forge audio train --model models/audio_sft --dataset librispeech --task asr --cycles 3 --output models/audio_raft
halo-forge audio benchmark --model models/audio_raft --dataset librispeech --task asr
```

### VLM (Vision-Language)

```bash
# SFT → RAFT → Benchmark
halo-forge vlm sft --dataset llava --model Qwen/Qwen2-VL-2B-Instruct --output models/vlm_sft
halo-forge vlm train --model models/vlm_sft --dataset textvqa --cycles 3 --output models/vlm_raft
halo-forge vlm benchmark --model models/vlm_raft --dataset textvqa
```

### Agentic (Tool Calling)

```bash
# SFT → RAFT → Benchmark
halo-forge agentic sft --dataset xlam_sft --model Qwen/Qwen2.5-7B-Instruct --output models/agentic_sft
halo-forge agentic train --model models/agentic_sft --dataset xlam --cycles 3 --output models/agentic_raft
halo-forge agentic benchmark --model models/agentic_raft --dataset xlam
```

## Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Data prep | 5-10 min | Depends on dataset size |
| SFT | 1-2 hours | 3 epochs, 7B model |
| RAFT (5 cycles) | 8-12 hours | ~2 hours per cycle |
| Benchmark | 30-60 min | Depends on samples |

Total: ~12-16 hours for complete pipeline.
