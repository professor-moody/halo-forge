# RAFT Training Guide

Comprehensive guide to RAFT training with halo-forge, including advanced options for curriculum learning and reward shaping.

## Basic Training

```bash
halo-forge raft train \
  --model Qwen/Qwen2.5-Coder-3B \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --output models/raft_run1
```

## CLI Options

### Core Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model`, `-m` | Qwen/Qwen2.5-Coder-3B | Base model to fine-tune |
| `--checkpoint` | None | SFT checkpoint to start from |
| `--prompts`, `-p` | Required | JSONL file with prompts |
| `--output`, `-o` | models/raft | Output directory |
| `--cycles` | 3 | Number of RAFT cycles |
| `--verifier` | gcc | Verifier type |

### Verifiers

| Verifier | Language | Notes |
|----------|----------|-------|
| `gcc` | C++ | Native Linux compilation |
| `mingw` | C++ | Windows cross-compilation |
| `humaneval` | Python | HumanEval benchmark |
| `mbpp` | Python | MBPP benchmark |
| `rust` | Rust | Requires cargo in toolbox |
| `go` | Go | Requires go in toolbox |

### Reward Filtering

| Flag | Default | Description |
|------|---------|-------------|
| `--reward-threshold` | 0.5 | Minimum reward to keep sample |
| `--keep-percent` | 0.5 | Keep top X% of passing samples |

---

## Curriculum Learning

Train on easy prompts first, then progressively harder ones.

```bash
halo-forge raft train \
  --model Qwen/Qwen2.5-Coder-3B \
  --prompts data/prompts.jsonl \
  --curriculum progressive \
  --output models/raft_curriculum
```

### Strategies

| Strategy | Behavior |
|----------|----------|
| `none` | No curriculum (default) |
| `complexity` | Sort prompts by estimated complexity once |
| `progressive` | Start with easiest 20%, add 20% each cycle |
| `adaptive` | Adjust difficulty based on pass rate |

### How Complexity is Estimated

Prompts are scored 0.0 (easy) to 1.0 (hard) based on:
- **Length**: Longer prompts = harder
- **Keywords**: Technical terms like "syscall", "mutex" = harder
- **Specificity**: Specific values, error handling = harder

### Example

With `--curriculum progressive` and 5 cycles:
- Cycle 1: Easiest 20% of prompts
- Cycle 2: Easiest 40% of prompts
- Cycle 3: Easiest 60% of prompts
- Cycle 4: Easiest 80% of prompts
- Cycle 5: All prompts

---

## Reward Shaping

Dynamically adjust filtering thresholds over training cycles.

```bash
halo-forge raft train \
  --model Qwen/Qwen2.5-Coder-3B \
  --prompts data/prompts.jsonl \
  --reward-shaping annealing \
  --output models/raft_shaped
```

### Strategies

| Strategy | Behavior |
|----------|----------|
| `fixed` | Use constant thresholds (default) |
| `annealing` | Start lenient (threshold=0), increase to target |
| `warmup` | Lenient for first N cycles, then normal |
| `adaptive` | Adjust based on pass rates each cycle |

### Example: Annealing

With `--reward-shaping annealing` and 5 cycles:
- Cycle 1: threshold=0.0, keep=80%
- Cycle 2: threshold=0.12, keep=73%
- Cycle 3: threshold=0.25, keep=65%
- Cycle 4: threshold=0.38, keep=57%
- Cycle 5: threshold=0.50, keep=50%

This allows more samples through early (when model is weak), gradually becoming stricter.

---

## Combining Features

Curriculum and reward shaping work together:

```bash
halo-forge raft train \
  --model Qwen/Qwen2.5-Coder-3B \
  --prompts data/prompts.jsonl \
  --curriculum progressive \
  --reward-shaping warmup \
  --cycles 5 \
  --output models/raft_combined
```

This:
1. Starts with easy prompts (curriculum)
2. Uses lenient filtering early (warmup)
3. Gradually increases both difficulty and filtering strictness

---

## Memory Optimization

For memory-constrained setups:

```python
# In config file or code
RAFTConfig(
    generation_chunk_size=25,      # Fewer prompts per generation batch
    verification_chunk_size=100,   # Smaller verification batches
    clear_cache_every_n_batches=5, # More frequent cache clearing
)
```

Or use smaller models:
```bash
halo-forge raft train --model Qwen/Qwen2.5-Coder-0.5B ...
```

---

## Monitoring

During training, you'll see:
```
======================================================================
RAFT Training: 3 cycles
======================================================================
  Reward threshold: 0.5
  Keep top: 50% of passing samples
  Samples per prompt: 8
  Temperature: 0.70 (fixed)
  Curriculum: progressive
  Reward shaping: annealing

=== Cycle 1/3 ===
Curriculum: 20/100 prompts (avg complexity: 0.15)
Reward shaping: threshold=0.00, keep=80%
...
```

---

## Tips

1. **Start small**: Test with `Qwen/Qwen2.5-Coder-0.5B` before scaling up
2. **Use curriculum for mixed-difficulty prompts**: If your dataset has varying complexity
3. **Use warmup for weak base models**: Gives the model time to learn basics
4. **Check pass rates**: If < 10%, lower threshold or use easier prompts
5. **Monitor memory**: Watch `rocm-smi` during training; reduce chunk sizes if needed

