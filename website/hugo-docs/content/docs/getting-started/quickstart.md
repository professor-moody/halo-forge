---
title: "Quick Start"
description: "Get halo forge running in under 30 minutes"
weight: 1
---

## Prerequisites

- AMD Strix Halo hardware (gfx1151) with 128GB unified memory
- Fedora 42+ with podman/toolbox
- Kernel 6.16+ recommended for gfx1151 support

## 1. Build the Toolbox

```bash
git clone https://github.com/professor-moody/halo-forge.git
cd halo-forge/toolbox
./build.sh

# Create the toolbox container
toolbox create halo-forge --image localhost/halo-forge:latest

# Enter the toolbox
toolbox enter halo-forge
```

## 2. Verify Setup

```bash
# Quick validation (5 seconds, no GPU needed)
halo-forge test --level smoke

# Full validation with model loading (2-3 minutes)
halo-forge test --level standard

# Check hardware info
halo-forge info
```

Expected test output:

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

## 3. Prepare Training Data

```bash
# Download CodeForces C++ examples
halo-forge data prepare --dataset codeforces_cpp --output data/train.jsonl

# Check what was downloaded
head -1 data/train.jsonl | python -m json.tool
```

## 4. Run SFT Training

```bash
# Quick test (5 minutes)
halo-forge sft train \
  --data data/train.jsonl \
  --output models/sft_test \
  --epochs 1

# Full training (1-2 hours)
halo-forge sft train \
  --data data/train.jsonl \
  --output models/sft \
  --epochs 3
```

## 5. Run RAFT Training

After SFT, improve the model with verification:

```bash
# Create prompts file
echo '{"prompt": "Write a C++ function to sort a vector"}' > data/prompts.jsonl

# Run RAFT with GCC verification
halo-forge raft train \
  --checkpoint models/sft/final_model \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --cycles 3
```

## 6. Benchmark

```bash
halo-forge benchmark run \
  --model models/raft/cycle_3_final \
  --prompts data/test.jsonl \
  --verifier gcc
```

## Test Levels

| Level | Time | GPU | What It Tests |
|-------|------|-----|---------------|
| `smoke` | 5s | No | Imports, compiler, verifier logic |
| `standard` | 2-3 min | Yes | Model loading, generation, verification |
| `full` | 5 min | Yes | Complete mini-RAFT cycle |

## Next Steps

- [Full Pipeline](/docs/training-pipeline/full-pipeline/) — Complete workflow
- [Hardware Notes](/docs/getting-started/hardware/) — Strix Halo configuration
- [Verifiers](/docs/verifiers/) — Choose the right verifier
