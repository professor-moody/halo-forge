---
title: "RAFT Training"
description: "Reward-Ranked Fine-Tuning with compiler verification"
weight: 3
---

RAFT (Reward-Ranked Fine-Tuning) iteratively improves the model using compiler feedback.

## Basic Usage

```bash
halo-forge raft train \
  --checkpoint models/sft/final_model \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --cycles 5
```

## How RAFT Works

```python
for cycle in range(num_cycles):
    # 1. Generate multiple samples per prompt
    samples = model.generate(prompts, n=8)
    
    # 2. Verify with compiler
    results = verifier.verify_batch(samples)
    
    # 3. Filter by reward threshold
    filtered = [s for s, r in zip(samples, results) 
                if r.reward >= 0.5]
    
    # 4. Fine-tune on verified samples
    model.train(filtered)
```

Each cycle:
1. Generates N samples per prompt
2. Verifies all samples with the compiler
3. Filters to keep only samples above threshold
4. Fine-tunes on the filtered set
5. Repeats with improved model

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
  batch_size: 4

training:
  epochs: 1
  batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 5e-5
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_cycles` | 5 | Number of RAFT iterations |
| `samples_per_prompt` | 8 | Samples generated per prompt |
| `reward_threshold` | 0.5 | Minimum reward to include |
| `keep_top_percent` | 0.5 | Top % of samples above threshold |
| `temperature` | 0.7 | Sampling temperature |

## Cycle Dynamics

Typical progression:

| Cycle | Kept Samples | Compile Rate |
|-------|--------------|--------------|
| 1 | 35-45% | 28% |
| 2 | 45-55% | 35% |
| 3 | 50-60% | 40% |
| 4 | 55-65% | 44% |
| 5 | 55-65% | 46% |
| 6+ | Diminishing | May degrade |

> **Note:** Performance often peaks around cycle 5-6, then degrades. Monitor closely.

## Graduated Rewards

halo-forge uses graduated rewards for better gradient flow:

| Outcome | Reward |
|---------|--------|
| Syntax error | 0.0 |
| Compiles with warnings | 0.3 |
| Compiles clean | 0.5 |
| Runs without crash | 0.7 |
| Correct output | 1.0 |

## Caching & Recovery

RAFT automatically caches:
- Generated samples (per cycle)
- Verification results
- Training state checkpoints

Resume after crash:

```bash
halo-forge raft train \
  --config configs/raft.yaml \
  --resume  # Automatically finds last checkpoint
```

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

## Monitoring

Watch for:

- **Compile rate per cycle** — Should increase
- **Average reward** — Should increase
- **Samples kept** — Should stabilize or increase
- **Training loss** — Should decrease within cycle

```bash
# Check statistics
cat models/raft/raft_statistics.json | jq
```

## Tips

1. **Start with 5 cycles** — More isn't always better
2. **Watch for degradation** — Stop if cycle N+1 is worse than N
3. **Use 8 samples/prompt** — Good balance of diversity and compute
4. **Temperature 0.7** — Enough diversity without garbage
5. **Monitor GPU memory** — RAFT is memory-intensive

## Verifier Choice

| Verifier | Use Case |
|----------|----------|
| `gcc` | Linux C/C++ |
| `mingw` | Windows C/C++ (cross-compile) |
| `clang` | Alternative to GCC |
| `pytest` | Python with tests |
| `custom` | Your own verifier |

See [Verifiers](/docs/verifiers/) for details.
