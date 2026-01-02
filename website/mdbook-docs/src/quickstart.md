# Quick Start

Get halo-forge running in under 30 minutes.

## Prerequisites

- AMD Strix Halo hardware (or compatible ROCm GPU)
- Fedora 41+ with Podman/Toolbox
- 128GB unified memory recommended

## 1. Clone and Build

```bash
git clone https://github.com/professor-moody/halo-forge.git
cd halo-forge/toolbox
./build.sh
```

This builds a container with:
- ROCm 7 nightly (TheRock)
- PyTorch nightly for gfx1151
- All Python dependencies

## 2. Enter the Toolbox

```bash
toolbox enter halo-forge
```

## 3. Validate Installation

```bash
# Quick smoke test (no GPU required)
halo-forge test --level smoke

# Full validation with GPU
halo-forge test --level standard
```

Expected output:

```
============================================================
halo-forge Standard Test
Model: Qwen/Qwen2.5-Coder-0.5B
============================================================

  [OK] Import modules (0.0s)
  [OK] Compiler available (0.0s)
  [OK] GPU available (0.0s)
  [OK] Model loading (1.2s)
  [OK] Code generation (21.6s)
  [OK] Code verification (0.3s)

============================================================
Test Results: 6/6 passed
============================================================
```

## 4. Run Demo Benchmark

```bash
halo-forge benchmark full \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --cycles 2
```

This takes ~40 minutes and validates the full RAFT pipeline.

## 5. Train Your Own Model

```bash
# 1. Prepare data
halo-forge data prepare --dataset codeforces_cpp --output data/train.jsonl

# 2. Run SFT
halo-forge sft train --data data/train.jsonl --output models/sft --epochs 1

# 3. Run RAFT
halo-forge raft train \
  --checkpoint models/sft/final_model \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --cycles 3 \
  --output models/raft

# 4. Benchmark
halo-forge benchmark run \
  --model models/raft/cycle_3_final \
  --prompts data/test.jsonl \
  --verifier gcc
```

## Next Steps

- [Full Pipeline](./full-pipeline.md) — Complete training workflow
- [Hardware Notes](./hardware-notes.md) — Strix Halo configuration
- [Verifiers](./verifiers.md) — Choose the right verifier
