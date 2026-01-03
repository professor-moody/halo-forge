---
title: "Documentation"
description: "Complete documentation for halo-forge RLVR training framework"
---

## What is halo-forge?

halo-forge is an **RLVR (Reinforcement Learning from Verifiable Rewards)** framework that uses compiler feedback as reward signals for iterative model refinement.

### The Problem

| Approach | Limitation |
|----------|------------|
| SFT only | Distribution mismatch — model outputs differ from training data |
| RLHF | Expensive human labeling, inconsistent judgments |
| Self-evaluation | Models hallucinate correctness, signals can be gamed |

### The Solution

A compiler provides a **perfect reward signal** — unambiguous, deterministic feedback about code correctness that cannot be gamed.

## Architecture

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐
│   Data   │ → │   SFT    │ → │   RAFT   │ → │ Benchmark │
└──────────┘    └──────────┘    └──────────┘    └───────────┘
```

1. **Data** — Gather training examples from public datasets or LLM generation
2. **SFT** — Supervised fine-tuning to establish baseline capability  
3. **RAFT** — Iterative verification loop: generate → verify → filter → train
4. **Benchmark** — Evaluate with pass@k metrics

## Results

Production training on Qwen2.5-Coder-7B with 569 C/C++ prompts:

| Stage | Compile Rate | pass@1 |
|-------|-------------|--------|
| SFT Baseline | 15.2% | 18.7% |
| Cycle 1 | 28.4% | 35.2% |
| Cycle 3 | 39.7% | 48.2% |
| Cycle 6 (Peak) | 46.7% | 55.3% |

**3x improvement** over 6 RAFT cycles.

## Quick Navigation

### Getting Started
- [Quick Start](/docs/getting-started/quickstart/) — Get running in 30 minutes
- [Toolbox Setup](/docs/getting-started/toolbox/) — Build the container environment
- [Hardware Notes](/docs/getting-started/hardware/) — Strix Halo configuration

### Training Pipeline
- [Full Pipeline](/docs/training-pipeline/full-pipeline/) — Complete training workflow
- [Data Generation](/docs/training-pipeline/data-generation/) — Prepare training data
- [SFT Training](/docs/training-pipeline/sft/) — Supervised fine-tuning
- [RAFT Training](/docs/training-pipeline/raft/) — Reward-ranked fine-tuning
- [Benchmarking](/docs/training-pipeline/benchmarking/) — Evaluate with pass@k

### Verifiers
- [Verifier Overview](/docs/verifiers/) — Choose your verification strategy
- [Compile Verifiers](/docs/verifiers/compile/) — GCC, Clang, MinGW, MSVC
- [Test Verifiers](/docs/verifiers/test/) — pytest, unittest
- [Custom Verifiers](/docs/verifiers/custom/) — Build your own

### Reference
- [Configuration](/docs/reference/configuration/) — Complete config reference
- [CLI Reference](/docs/reference/cli/) — Command-line interface
- [Troubleshooting](/docs/reference/troubleshooting/) — Common issues

### Background
- [Theory & Research](/docs/background/theory/) — Research foundations
- [RAFT vs PPO vs GRPO](/docs/background/raft-vs-ppo/) — Algorithm comparison
- [Graduated Rewards](/docs/background/graduated-rewards/) — Partial credit system
- [Learning Rate Strategies](/docs/background/learning-rates/) — LR recommendations

### Meta
- [Changelog](/docs/changelog/) — Version history
- [Contributing](/docs/contributing/) — How to contribute
