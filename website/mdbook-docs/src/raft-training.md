# RAFT Training

Reward-Ranked Fine-Tuning is the core of halo-forge's RLVR approach.

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
