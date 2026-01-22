---
title: "Documentation"
description: "Complete documentation for halo forge RLVR training framework"
---

## What is halo forge?

halo forge is an **RLVR (Reinforcement Learning from Verifiable Rewards)** framework that uses compiler feedback as reward signals for iterative model refinement.

### The Problem

| Approach | Limitation |
|----------|------------|
| SFT only | Distribution mismatch — model outputs differ from training data |
| RLHF | Expensive human labeling, inconsistent judgments |
| Self-evaluation | Models hallucinate correctness, signals can be gamed |

### The Approach

A compiler provides **deterministic feedback** — objective, reproducible results about code correctness.

## Architecture

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐
│   Data   │ ─► │   SFT    │ ─► │   RAFT   │ ─► │ Benchmark │
└──────────┘    └──────────┘    └──────────┘    └───────────┘
```

1. **Data** — Gather training examples from public datasets or LLM generation
2. **SFT** — Supervised fine-tuning to establish baseline capability  
3. **RAFT** — Iterative verification loop: generate → verify → filter → train
4. **Benchmark** — Evaluate with pass@k metrics

## What to Expect

RAFT training typically shows:

| Cycle | What Happens |
|-------|-------------|
| 1-2 | Largest gains as model learns basic patterns |
| 3-4 | Continued improvement at slower rate |
| 5-6 | Diminishing returns; monitor for plateau |
| 7+ | May see degradation; consider stopping earlier |

Results vary significantly based on model, dataset, hardware, and domain. Run benchmarks to measure improvement on your specific use case.

## Quick Navigation

### Getting Started
- [Quick Start](/docs/getting-started/quickstart/) — Get running in 30 minutes
- [Toolbox Setup](/docs/getting-started/toolbox/) — Build the container environment
- [Hardware Notes](/docs/getting-started/hardware/) — Strix Halo configuration

### Training Pipeline
- **[How to Train](/docs/training-pipeline/how-to-train/)** — Complete step-by-step guide (start here!)
- [Full Pipeline](/docs/training-pipeline/full-pipeline/) — Complete training workflow
- [Data Generation](/docs/training-pipeline/data-generation/) — Prepare training data
- [SFT Training](/docs/training-pipeline/sft/) — Supervised fine-tuning
- [RAFT Training](/docs/training-pipeline/raft/) — Reward-ranked fine-tuning
- [Benchmarking](/docs/training-pipeline/benchmarking/) — Evaluate with pass@k
- [Production Runs](/docs/training-pipeline/production-runs/) — Production training commands

### Verifiers
- [Verifier Overview](/docs/verifiers/) — Choose your verification strategy
- [Compile Verifiers](/docs/verifiers/compile/) — GCC, Clang, MinGW, MSVC
- [Test Verifiers](/docs/verifiers/test/) — pytest, unittest
- [Execution Verifiers](/docs/verifiers/execution/) — Test case verification
- [Multi-Language](/docs/verifiers/multi-language/) — Auto-detect language
- [Custom Verifiers](/docs/verifiers/custom/) — Build your own

### Reference
- **[Command Index](/docs/reference/command-index/)** — Every command and flag
- [Configuration](/docs/reference/configuration/) — Config file reference
- [Web UI](/docs/reference/web-ui/) — Dashboard for training and monitoring
- [Windows Setup](/docs/reference/windows-setup/) — MSVC build server
- [Troubleshooting](/docs/reference/troubleshooting/) — Common issues

### Background
- [Theory & Research](/docs/background/theory/) — Research foundations
- [Graduated Rewards](/docs/background/graduated-rewards/) — Partial credit system
- [Learning Rate Strategies](/docs/background/learning-rates/) — LR recommendations

### Experimental
Features under active development and testing:
- [Experimental Features](/docs/experimental/) — VLM, Audio, Reasoning, Agentic, Inference

### Meta
- [Changelog](/docs/changelog/) — Version history
- [Contributing](/docs/contributing/) — How to contribute
