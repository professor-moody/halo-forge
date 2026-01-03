---
title: "RAFT Training"
description: "Reward-Ranked Fine-Tuning with compiler verification"
weight: 3
---

RAFT (Reward-Ranked Fine-Tuning) is the core of halo-forge's RLVR approach.

## The Algorithm

```
for cycle in range(num_cycles):
    1. Generate N samples per prompt
    2. Verify all samples with compiler
    3. Filter to samples with reward >= threshold
    4. Fine-tune on filtered samples
    5. Repeat with updated model
```

## Basic Usage

```bash
halo-forge raft train \
  --checkpoint models/sft/final_model \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --cycles 5 \
  --output models/raft
```

## Configuration

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
  top_p: 0.95
  batch_size: 4

training:
  epochs: 1
  batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 5e-5

verifier:
  type: gcc
  run_after_compile: false
```

## Key Parameters

### samples_per_prompt

How many completions to generate for each prompt.

| Value | Exploration | Compute |
|-------|-------------|---------|
| 4 | Low | Fast |
| 8 | Medium | Balanced |
| 16 | High | Slow |

### reward_threshold

Minimum reward to keep a sample.

| Value | Effect |
|-------|--------|
| 0.3 | Keep samples with warnings |
| 0.5 | Keep clean compiles |
| 0.7 | Keep samples that run |
| 1.0 | Only correct outputs |

### temperature

Controls generation diversity.

| Value | Effect |
|-------|--------|
| 0.3 | Conservative, similar outputs |
| 0.7 | Balanced |
| 1.0 | Diverse, more exploration |

## Sample Filtering Strategy

RAFT generates many samples and filters them before training. The filtering strategy significantly impacts what the model learns.

### How Filtering Works

```
1. Generate N samples per prompt (e.g., 374 prompts × 8 samples = 2,992 samples)
2. Verify all samples → each gets a reward score (0.0 to 1.0)
3. Filter by reward_threshold → keep samples with reward >= threshold
4. Sort by reward (highest first)
5. Keep top keep_top_percent of filtered samples
6. Train on final filtered set
```

### keep_top_percent

After filtering by threshold, what percentage of passing samples to keep.

```bash
# CLI usage
halo-forge raft train --keep-percent 0.5 ...  # Keep top 50%
```

| Value | Effect | Use Case |
|-------|--------|----------|
| 0.2 | Very selective (top 20%) | Large datasets, want only best |
| 0.5 | Balanced (top 50%) | Default, works well generally |
| 0.8 | Inclusive (top 80%) | Small datasets, need more data |
| 1.0 | Keep all passing | Very small datasets |

### Filtering Examples

**Example 1: Default settings (50% threshold, 50% keep)**
```
Generated: 2,992 samples
Pass threshold (≥0.5): 1,200 samples (40% compile rate)
Keep top 50%: 600 samples → Training set
```

**Example 2: Selective filtering (50% threshold, 20% keep)**
```
Generated: 2,992 samples
Pass threshold (≥0.5): 1,200 samples
Keep top 20%: 240 samples → Training on only the best
```

**Example 3: Inclusive filtering (30% threshold, 100% keep)**
```
Generated: 2,992 samples
Pass threshold (≥0.3): 1,800 samples (including warnings)
Keep top 100%: 1,800 samples → Maximum training data
```

### Choosing Your Strategy

**Small dataset (< 500 prompts):**
```bash
--keep-percent 0.8 --reward-threshold 0.5
```
Keep more data since you have fewer prompts to begin with.

**Large dataset (> 1000 prompts):**
```bash
--keep-percent 0.3 --reward-threshold 0.5
```
Be selective - train only on high-quality samples.

**Low initial compile rate (< 20%):**
```bash
--keep-percent 1.0 --reward-threshold 0.3
```
Keep everything that compiles (even with warnings) to maximize training signal.

**High initial compile rate (> 50%):**
```bash
--keep-percent 0.5 --reward-threshold 0.7
```
Be selective and only train on samples that actually execute.

### CLI Reference

```bash
halo-forge raft train \
  --model Qwen/Qwen2.5-Coder-7B \
  --prompts data/prompts.jsonl \
  --verifier mbpp \
  --cycles 5 \
  --keep-percent 0.5 \        # Keep top 50% of passing samples
  --reward-threshold 0.5 \    # Min reward to pass
  --output models/production
```

### Configuration File

```yaml
# configs/raft.yaml
raft:
  samples_per_prompt: 8
  reward_threshold: 0.5      # Minimum reward to keep sample
  keep_top_percent: 0.5      # Keep top 50% above threshold
```

## Cycle-by-Cycle Output

```
models/raft/
├── cycle_1/
│   ├── samples.jsonl      # All generated samples
│   ├── kept.jsonl         # Filtered samples used for training
│   ├── checkpoint/        # Model after training
│   └── stats.json         # Verification statistics
├── cycle_2/
│   └── ...
├── cycle_3_final/         # Best performing cycle
└── training_log.json
```

## Monitoring Progress

### Progress Display

halo-forge provides real-time progress during generation:

```
> Generating batch 13/47 ━━━━━━━━━━ 13/47 • 28% • 1:13:10 • 0.15 it/s
```

- **13/47**: Current batch / total batches
- **28%**: Percentage complete
- **1:13:10**: Time elapsed
- **0.15 it/s**: Iterations per second (batches processed)

### Compile Rate Patterns

Watch for these patterns:

**Healthy training:**
```
Cycle 1: 28.4% compile rate (kept 182/640 samples)
Cycle 2: 35.1% compile rate (kept 224/640 samples)
Cycle 3: 39.7% compile rate (kept 254/640 samples)
```

**Plateauing:**
```
Cycle 4: 40.2% compile rate
Cycle 5: 40.5% compile rate
Cycle 6: 39.8% compile rate  # Consider stopping
```

**Degradation:**
```
Cycle 7: 35.2% compile rate  # Stop and use cycle 6
```

## When to Stop

- **Diminishing returns**: < 2% improvement per cycle
- **Degradation**: Performance drops
- **Typically**: 5-6 cycles is optimal

If you see degradation, consider [learning rate decay](/docs/background/learning-rates/).

## Graduated Rewards

halo-forge uses graduated rewards for better gradient flow:

| Outcome | Reward |
|---------|--------|
| Syntax error | 0.0 |
| Compiles with warnings | 0.3 |
| Compiles clean | 0.5 |
| Runs without crash | 0.7 |
| Correct output | 1.0 |

## Caching & Automatic Resume

RAFT automatically caches progress at each stage, allowing seamless resume after crashes or interruptions.

### Cache Files

For each cycle, these files are saved to your output directory:

```
models/raft/
├── cycle_1_samples.jsonl    # All generated completions
├── cycle_1_verified.jsonl   # Verification results with rewards
├── cycle_1_final/           # Trained model checkpoint
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── cycle_2_samples.jsonl
├── cycle_2_verified.jsonl
├── cycle_2_final/
└── ...
```

### Automatic Resume Behavior

When you re-run the same command, RAFT automatically detects existing cache files:

| If this exists... | RAFT will... |
|-------------------|--------------|
| `cycle_N_final/` | Skip entire cycle, load checkpoint |
| `cycle_N_samples.jsonl` | Skip generation, load cached samples |
| `cycle_N_verified.jsonl` | Skip verification, load cached results |
| Nothing | Start from scratch |

### Resume Example

```bash
# Original run crashes during cycle 2 verification
halo-forge raft train --model Qwen/Qwen2.5-Coder-7B --cycles 5 --output models/raft

# State after crash:
# - cycle_1_final/ exists (complete)
# - cycle_2_samples.jsonl exists (generation done)
# - cycle_2_verified.jsonl missing (crashed during verification)

# Resume with SAME command:
halo-forge raft train --model Qwen/Qwen2.5-Coder-7B --cycles 5 --output models/raft

# RAFT will:
# 1. Skip cycle 1 entirely (loads cycle_1_final checkpoint)
# 2. Load cycle_2_samples.jsonl (skip 8+ hours of generation!)
# 3. Resume verification from start
# 4. Continue normally
```

### What Gets Logged

```
Cycle 1 already complete, skipping...
Loading cached samples...
Loaded 2992 samples from cache
Verifying 2992 samples...
```

### Recovery Time Savings

| Crash Point | Time Saved on Resume |
|-------------|---------------------|
| During generation batch 50/100 | 0 (generation restarts) |
| After generation, during verification | ~4-8 hours (full generation) |
| After verification, during training | ~8-10 hours (generation + verification) |
| After training completes | Entire cycle skipped |

### Manual Cache Management

**Clear cache to restart a cycle:**
```bash
rm models/raft/cycle_3_*  # Remove cycle 3 cache files
```

**Force fresh start:**
```bash
rm -rf models/raft/cycle_*  # Remove all cache files
```

### Known Limitation

Currently, **mid-generation resume is not supported**. If a crash occurs during generation (before all batches complete), that cycle's generation restarts from scratch. The samples are only cached after all generation completes.

{{% hint info %}}
**Tip**: For very long runs, consider using `screen` or `tmux` to prevent terminal disconnection issues.
{{% /hint %}}

## Memory Management

RAFT uses chunked verification to prevent OOM:

```python
# Verify in chunks of 200
chunk_size = 200
for i in range(0, len(samples), chunk_size):
    chunk = samples[i:i+chunk_size]
    results.extend(verifier.verify_batch(chunk))
    gc.collect()  # Force garbage collection
```

## Advanced: Different Verifiers per Cycle

```yaml
cycles:
  - verifier: gcc
    reward_threshold: 0.3
  - verifier: gcc
    reward_threshold: 0.5
  - verifier: gcc
    run_after_compile: true
    reward_threshold: 0.7
```

This creates curriculum learning: easier criteria early, harder later.

## Verifier Choice

| Verifier | Use Case |
|----------|----------|
| `gcc` | Linux C/C++ |
| `mingw` | Windows C/C++ (cross-compile) |
| `clang` | Alternative to GCC |
| `humaneval` | Python with HumanEval tests |
| `mbpp` | Python with MBPP tests |
| `custom` | Your own verifier |

See [Verifiers](/docs/verifiers/) for details.

## Tips

1. **Start with 5 cycles** — More isn't always better
2. **Watch for degradation** — Stop if cycle N+1 is worse than N
3. **Use 8 samples/prompt** — Good balance of diversity and compute
4. **Temperature 0.7** — Enough diversity without garbage
5. **Monitor GPU memory** — RAFT is memory-intensive
