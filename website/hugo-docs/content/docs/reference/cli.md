---
title: "CLI Reference"
description: "Complete command-line interface reference"
weight: 2
---

## Global Options

```bash
halo-forge [--help] [--version] <command>
```

## Commands

### halo-forge test

Validate installation.

```bash
halo-forge test --level <smoke|standard|full> [--verbose] [--model MODEL]
```

| Level | GPU | Time | Tests |
|-------|-----|------|-------|
| `smoke` | No | 5s | Imports, compiler |
| `standard` | Yes | 2-3 min | Model loading, generation |
| `full` | Yes | 5 min | Complete RAFT cycle |

### halo-forge data

Data generation and preparation.

```bash
# Download public dataset
halo-forge data prepare --dataset <name> --output FILE [--limit N]

# Generate with LLM
halo-forge data generate --prompts FILE --backend <ollama|anthropic|openai> --model MODEL --output FILE

# Split dataset
halo-forge data split --input FILE --train FILE --test FILE [--ratio 0.9]
```

**Datasets**: `codeforces_cpp`, `mbpp`, `humaneval`

### halo-forge sft

Supervised fine-tuning.

```bash
halo-forge sft train [--config FILE] [--data FILE] [--output DIR] [--model MODEL] [--epochs N] [--resume CHECKPOINT]
```

### halo-forge raft

RAFT training.

```bash
halo-forge raft train [--config FILE] [--checkpoint PATH] [--prompts FILE] [--verifier TYPE] [--cycles N] [--output DIR]
```

**Verifiers**: `gcc`, `clang`, `mingw`, `msvc`, `pytest`, `unittest`

### halo-forge benchmark

Model evaluation.

```bash
# Run benchmark
halo-forge benchmark run --model PATH --prompts FILE --verifier TYPE [--samples N] [--k 1,5,10] [--output FILE]

# Full benchmark suite
halo-forge benchmark full [--model MODEL] [--suite <all|small|medium>] [--cycles N] [--output DIR]

# Compare results
halo-forge benchmark compare FILE [FILE ...]
```

### halo-forge info

System information.

```bash
halo-forge info [--gpu] [--memory] [--all]
```

## Examples

```bash
# Quick pipeline test
halo-forge test --level smoke

# Full training run
halo-forge data prepare --dataset codeforces_cpp --output data/train.jsonl
halo-forge sft train --data data/train.jsonl --output models/sft
halo-forge raft train --checkpoint models/sft/final_model --prompts data/prompts.jsonl --verifier gcc --cycles 5
halo-forge benchmark run --model models/raft/cycle_5_final --prompts data/test.jsonl
```
