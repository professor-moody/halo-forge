<p align="center">
  <img src="halo-forge.png" alt="halo forge logo" width="350">
</p>

<h1 align="center">halo forge</h1>

<p align="center">
  A complete RLVR (Reinforcement Learning from Verifier Rewards) training framework for AMD Strix Halo.<br>
  Train language models to generate verified code through iterative refinement using automated verification.
</p>

<p align="center">
  <a href="https://halo-forge.io/docs">Documentation</a> •
  <a href="https://halo-forge.io/docs/getting-started/quickstart/">Quick Start</a> •
  <a href="https://halo-forge.io/docs/contributing/">Contributing</a>
</p>

---

## What is halo forge?

halo forge implements **RAFT (Reward-Ranked Fine-Tuning)**, training code generation models using compiler and test feedback as reward signals. Instead of relying on expensive human labeling, halo forge uses automated verifiers—compilers, test suites, and execution checks—to provide deterministic, scalable feedback for model improvement.

## Quick Install

```bash
# Build the toolbox (ROCm + PyTorch + dependencies)
cd halo-forge/toolbox
./build.sh

# Create and enter toolbox
toolbox create halo-forge --image localhost/halo-forge:latest
toolbox enter halo-forge

# Verify setup
halo-forge test --level smoke
halo-forge info
```

## Quick Start

```bash
# 1. Prepare data
halo-forge data prepare --dataset mbpp --output data/train.jsonl

# 2. SFT training
halo-forge sft train --data data/train.jsonl --output models/sft --epochs 3

# 3. RAFT training
halo-forge raft train \
  --checkpoint models/sft/final_model \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --cycles 5 \
  --output models/raft

# 4. Benchmark
halo-forge benchmark eval --model models/raft/cycle_5_final --benchmark humaneval
```

## Documentation

Full documentation is available at **[halo-forge.io/docs](https://halo-forge.io/docs)**

| Section | Description |
|---------|-------------|
| [Quick Start](https://halo-forge.io/docs/getting-started/quickstart/) | Get running in 30 minutes |
| [How to Train](https://halo-forge.io/docs/training-pipeline/how-to-train/) | Complete step-by-step guide |
| [Command Index](https://halo-forge.io/docs/reference/command-index/) | Every command and flag |
| [Verifiers](https://halo-forge.io/docs/verifiers/) | Verification options (GCC, MinGW, pytest, etc.) |
| [Configuration](https://halo-forge.io/docs/reference/configuration/) | Config file reference |
| [Experimental](https://halo-forge.io/docs/experimental/) | VLM, Audio, Reasoning, Agentic training |

## Hardware

Optimized for **AMD Strix Halo** (gfx1151) with 128GB unified memory. Key settings:

```yaml
bf16: true                     # BF16 is optimal (not 4-bit)
dataloader_num_workers: 0      # Required for unified memory
dataloader_pin_memory: false   # Required for unified memory
```

See [Hardware Notes](https://halo-forge.io/docs/getting-started/hardware/) for details.

## Web UI

```bash
halo-forge ui
# Open http://127.0.0.1:8080
```

Launch training, monitor progress, and view results through a modern dashboard.

## License

Copyright 2025 Halo Forge Labs LLC

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

- AMD for Strix Halo hardware
- [kyuz0](https://github.com/kyuz0/amd-strix-halo-llm-finetuning) for the original fine-tuning toolbox
- TheRock project for ROCm nightlies
- The Strix Halo community for testing and feedback
- RAFT paper authors for the foundational research
