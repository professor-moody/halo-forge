---
title: "Reasoning Training"
weight: 4
---

# Math & Reasoning Training

**Status:** Experimental (v1.0.0)

Train language models on mathematical reasoning tasks using verified answer checking with SymPy.

## Overview

The reasoning module enables RLVR training for mathematical problem-solving. Unlike simple string matching, answers are verified using:

1. **Numeric comparison** - Direct float comparison with tolerance
2. **Symbolic equivalence** - SymPy-based algebraic comparison
3. **Partial credit** - Reward for showing reasoning steps

## Supported Tasks

| Task | Verifier | Datasets |
|------|----------|----------|
| Grade School Math | MathVerifier | GSM8K |
| Competition Math | MathVerifier | MATH, AIME |

## Quick Start

### List Available Datasets

```bash
halo-forge reasoning datasets
```

### Run Benchmark

```bash
halo-forge reasoning benchmark \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset gsm8k \
  --limit 100
```

### Train with RAFT

```bash
halo-forge reasoning train \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset gsm8k \
  --cycles 4 \
  --output models/reasoning_raft
```

## How It Works

```
Model Completion
       │
       ▼
┌─────────────────────┐
│  AnswerExtractor    │  Extract final answer
│  - \boxed{}         │  from completion
│  - "The answer is"  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  MathVerifier       │  Compare to expected
│  1. Numeric match   │  answer using multiple
│  2. Symbolic match  │  strategies
└─────────┬───────────┘
          │
          ▼
   VerifyResult
   - success: bool
   - reward: 0.0-1.0
```

## Reward Structure

| Outcome | Reward | Description |
|---------|--------|-------------|
| Correct answer | 1.0 | Numeric or symbolic match |
| Wrong + showed work | 0.2 | Reasoning steps present |
| No answer + work | 0.2 | Partial credit |
| No answer, no work | 0.1 | Minimal credit |

## Dependencies

The reasoning module requires SymPy for symbolic verification:

```bash
pip install sympy>=1.12
```

This is included in the halo-forge toolbox containers.

## Next Steps

- [Available Datasets](datasets/) - GSM8K, MATH, and more
- [Testing Guide](testing/) - Validate your setup
