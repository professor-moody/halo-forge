---
title: "How to Train"
description: "Complete guide to training code generation models with halo-forge"
weight: 0
---

This guide walks you through training a code generation model from scratch using RAFT. Start with the quick start for immediate results, then explore advanced sections for optimization.

## TL;DR - Quick Start (10 minutes)

Already have the toolbox built? Run training immediately:

```bash
# 1. Enter the toolbox
toolbox enter halo-forge

# 2. Install halo-forge
cd ~/projects/halo-forge && pip install -e .

# 3. Run smoke test
halo-forge test --level smoke

# 4. Start RAFT training (quick validation)
halo-forge raft train \
    --model Qwen/Qwen2.5-Coder-0.5B \
    --prompts data/rlvr/mbpp_train_prompts.jsonl \
    --verifier mbpp \
    --cycles 2 \
    --output models/quick_test
```

That's it. Training will begin and produce checkpoints as it progresses.

---

## Prerequisites Checklist

Before training, ensure you have:

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 24GB VRAM | 48GB+ (Strix Halo) |
| RAM | 32GB | 64GB+ |
| Storage | 50GB SSD | 200GB NVMe |
| Network | Stable connection | Fast for model downloads |

### Software

| Platform | Requirements |
|----------|--------------|
| **Fedora** | Fedora 42+, podman, toolbox |
| **Ubuntu** | Ubuntu 22.04+, Docker |
| **Kernel** | 6.16+ (for gfx1151 without parameters) |

---

## Part 1: Setup

### Option A: Fedora with podman toolbox

```bash
# Clone repository
git clone https://github.com/professor-moody/halo-forge.git
cd halo-forge/toolbox

# Build toolbox
./build.sh --no-cache

# Create and enter
toolbox create halo-forge --image localhost/halo-forge:latest
toolbox enter halo-forge

# Install package
cd ~/projects/halo-forge
pip install -e .
```

### Option B: Ubuntu with Docker (Experimental)

> **Note**: Ubuntu/Docker support is experimental. Fedora toolbox is recommended for production.

```bash
# Clone repository
git clone https://github.com/professor-moody/halo-forge.git
cd halo-forge/toolbox

# Build Docker image
./build-ubuntu.sh --no-cache

# (If GPU not visible) Add udev rules
sudo tee /etc/udev/rules.d/99-amd-kfd.rules >/dev/null <<'EOF'
SUBSYSTEM=="kfd", GROUP="render", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="card[0-9]*", GROUP="render", MODE="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger

# Run container
docker run -it --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  -v ~/projects:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  halo-forge:ubuntu

# Inside container
cd /workspace/halo-forge
pip install -e .
```

### Verify Setup

```bash
# Quick validation (5 seconds, no GPU)
halo-forge test --level smoke

# Standard validation (2-3 minutes, loads model)
halo-forge test --level standard

# Full validation (5 minutes, includes training step)
halo-forge test --level full
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

---

## Part 2: Data Preparation

### Option A: Use Built-in Datasets

halo-forge includes pre-formatted datasets for immediate use:

```bash
# List available datasets
ls data/rlvr/

# Available:
# - humaneval_full.jsonl     (164 problems with tests)
# - mbpp_train_full.jsonl    (374 problems with tests)
# - mbpp_train_prompts.jsonl (prompts only for RAFT)
```

### Option B: Download Public Datasets

```bash
# List available datasets
halo-forge data prepare --list

# Download CodeForces C++ examples
halo-forge data prepare \
  --dataset codeforces_cpp \
  --output data/codeforces.jsonl

# Download MBPP Python examples
halo-forge data prepare \
  --dataset mbpp \
  --output data/mbpp.jsonl
```

### Option C: Generate with LLM

```bash
# List available topics
halo-forge data generate --list

# Generate with DeepSeek (requires API key)
export DEEPSEEK_API_KEY=your_key
halo-forge data generate \
  --topic python_algorithms \
  --backend deepseek \
  --output data/generated.jsonl

# Generate with local Ollama (free)
halo-forge data generate \
  --topic rust_basics \
  --backend ollama \
  --model codellama:13b \
  --output data/rust.jsonl
```

### Create Prompts File

For RAFT training, you need a JSONL file with prompts:

```bash
# Extract prompts from training data
cat data/train.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    prompt = d.get('prompt', d.get('text', ''))[:500]
    if prompt:
        print(json.dumps({'prompt': prompt}))
" > data/prompts.jsonl
```

Or create manually:

```jsonl
{"prompt": "Write a Python function to calculate factorial"}
{"prompt": "Implement binary search in Python"}
{"prompt": "Write a function to check if a string is palindrome"}
```

---

## Part 3: SFT Training (Optional but Recommended)

SFT (Supervised Fine-Tuning) creates a baseline before RAFT. While optional if using a pre-trained coder model, SFT is highly recommended for domain-specific training. It helps the model learn your specific code style, patterns, and requirements before RAFT refinement.

### Basic SFT

```bash
halo-forge sft train \
  --data data/train.jsonl \
  --output models/sft \
  --epochs 3
```

### SFT with Configuration

Create `configs/sft.yaml`:

```yaml
model:
  name: Qwen/Qwen2.5-Coder-7B
  trust_remote_code: true

data:
  train_file: data/train.jsonl
  max_seq_length: 2048

lora:
  r: 16
  alpha: 32
  dropout: 0.05

training:
  output_dir: models/sft
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 2e-4
  bf16: true
  gradient_checkpointing: true
  
  # Critical for Strix Halo
  dataloader_num_workers: 0
  dataloader_pin_memory: false
```

```bash
halo-forge sft train --config configs/sft.yaml
```

---

## Part 4: RAFT Training

RAFT (Reward-rAnked Fine-Tuning) improves the model through iterative verification.

### Understanding the RAFT Cycle

```
┌─────────────────────────────────────────────────────┐
│               RAFT TRAINING CYCLE                    │
├─────────────────────────────────────────────────────┤
│                                                      │
│  GENERATE ──► VERIFY ──► FILTER ──► TRAIN           │
│      │           │          │          │            │
│      ▼           ▼          ▼          ▼            │
│  8 samples   Compile    Keep top    Fine-tune       │
│  per prompt  + Test     by reward   on winners      │
│                                                      │
│  ◄─────────── REPEAT 5-6 TIMES ────────────────►    │
└─────────────────────────────────────────────────────┘
```

### Basic RAFT Training

```bash
halo-forge raft train \
  --model Qwen/Qwen2.5-Coder-7B \
  --prompts data/rlvr/mbpp_train_prompts.jsonl \
  --verifier mbpp \
  --cycles 5 \
  --output models/raft
```

### RAFT with Custom Checkpoint

```bash
# Start from your SFT model
halo-forge raft train \
  --checkpoint models/sft/final_model \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --cycles 5 \
  --output models/raft
```

### Choosing a Verifier

| Verifier | Language | Use Case |
|----------|----------|----------|
| `gcc` | C/C++ | Linux compilation |
| `mingw` | C/C++ | Windows cross-compile |
| `rust` | Rust | Rust compilation |
| `go` | Go | Go compilation |
| `humaneval` | Python | HumanEval benchmark |
| `mbpp` | Python | MBPP benchmark |

### Monitoring Progress

Watch the training output:

```
RAFT CYCLE 1/5
==============
Generating samples... 374 prompts × 8 samples
Verifying 2992 samples...
  Passed: 1023 (34.2%)
  Failed: 1969

Filtering samples...
  Kept: 512 samples (top 50% above threshold)

Training on filtered samples...
  Loss: 0.856 → 0.342

Saving checkpoint to models/raft/cycle_1_final/
```

**Key Metrics to Watch:**

- **Pass rate**: Higher is better - indicates model improvement
- **Loss decrease**: Should trend downward across cycles
- **Kept samples**: More samples = more training signal

### When to Stop

Monitor pass rate across cycles:

```
Cycle 1: 34.2% pass rate
Cycle 2: 42.1% pass rate  (+7.9%)
Cycle 3: 48.5% pass rate  (+6.4%)
Cycle 4: 51.2% pass rate  (+2.7%)
Cycle 5: 52.1% pass rate  (+0.9%)  ← Diminishing returns
Cycle 6: 51.8% pass rate  (-0.3%)  ← Stop here
```

**General guidance:**
- Stop when improvement < 2% per cycle
- Stop if pass rate decreases
- In our testing, 5-6 cycles often worked well

---

## Part 5: Benchmarking

### Run Benchmark

```bash
halo-forge benchmark run \
  --model models/raft/cycle_5_final \
  --prompts data/rlvr/mbpp_validation.jsonl \
  --verifier mbpp \
  --samples 10 \
  --k 1,5,10 \
  --output results/benchmark.json
```

### Compare Models

```bash
# Benchmark baseline
halo-forge benchmark run \
  --model Qwen/Qwen2.5-Coder-7B \
  --prompts data/rlvr/mbpp_validation.jsonl \
  --output results/baseline.json

# Benchmark RAFT model
halo-forge benchmark run \
  --model models/raft/cycle_5_final \
  --prompts data/rlvr/mbpp_validation.jsonl \
  --output results/raft.json

# Compare
python3 -c "
import json
for name in ['baseline', 'raft']:
    with open(f'results/{name}.json') as f:
        data = json.load(f)
        print(f'{name}: pass@1={data[\"pass_at_k\"][\"1\"]:.1%}')
"
```

### Full Benchmark Suite

```bash
# Run full benchmark with before/after comparison
halo-forge benchmark full \
  --model Qwen/Qwen2.5-Coder-7B \
  --prompts data/rlvr/mbpp_train_prompts.jsonl \
  --verifier mbpp \
  --cycles 5 \
  --output results/full_benchmark
```

---

## Part 6: Advanced Topics

### Filtering Strategies

Control which samples are used for training:

```bash
# Keep top 50% of samples above 0.5 reward (default)
halo-forge raft train --keep-percent 0.5 --reward-threshold 0.5 ...

# Selective: top 20% only (large datasets)
halo-forge raft train --keep-percent 0.2 --reward-threshold 0.5 ...

# Inclusive: keep all passing (small datasets)
halo-forge raft train --keep-percent 1.0 --reward-threshold 0.3 ...
```

### Curriculum Learning

Increase difficulty over cycles:

```yaml
# configs/curriculum.yaml
curriculum_strategy: progressive

cycles:
  - verifier: gcc
    reward_threshold: 0.3
  - verifier: gcc
    reward_threshold: 0.5
  - verifier: gcc
    run_after_compile: true
    reward_threshold: 0.7
```

### Custom Verifier

```python
from halo_forge.rlvr.verifiers import Verifier, VerifyResult

class MyVerifier(Verifier):
    def verify(self, code: str) -> VerifyResult:
        # Your verification logic
        success = your_check(code)
        return VerifyResult(
            success=success,
            reward=1.0 if success else 0.0,
            details="Custom verification"
        )
```

### Hyperparameter Tuning

| Parameter | Default | Tuning Notes |
|-----------|---------|--------------|
| `samples_per_prompt` | 8 | More = better diversity, slower |
| `temperature` | 0.7 | Higher = more diverse, lower quality |
| `reward_threshold` | 0.5 | Higher = stricter filtering |
| `keep_top_percent` | 0.5 | Lower = more selective |
| `learning_rate` | 5e-5 | Lower if unstable |

### Memory Optimization (Strix Halo)

For unified memory systems:

```yaml
training:
  batch_size: 2
  gradient_accumulation: 16
  bf16: true  # NOT 4-bit (slower on Strix Halo)
  gradient_checkpointing: true
  
  # Critical for unified memory
  dataloader_num_workers: 0
  dataloader_pin_memory: false
```

---

## Troubleshooting

### Low Pass Rate

**Symptoms:** <20% pass rate, many syntax errors

**Solutions:**
1. Check prompt quality - are they asking for complete code?
2. Lower temperature for more consistent output
3. Add few-shot examples to prompts
4. Run SFT first to establish baseline

### Training Loss Increasing

**Symptoms:** Loss goes up after cycle 4-5

**Solutions:**
1. Stop training - you've peaked
2. Lower learning rate
3. Increase `reward_threshold` to filter stricter
4. Try learning rate decay

### GPU Hang

**Symptoms:** Training freezes, GPU unresponsive

**Solutions:**
1. Ensure `dataloader_num_workers: 0`
2. Ensure `dataloader_pin_memory: false`
3. Add `export HSA_ENABLE_SDMA=0`

### Out of Memory

**Symptoms:** CUDA/ROCm OOM errors

**Solutions:**
1. Reduce `batch_size`
2. Enable `gradient_checkpointing`
3. Use smaller model
4. Reduce `max_seq_length`

### Automatic Resume

RAFT automatically caches progress. If a run crashes:

```bash
# Just re-run the same command
halo-forge raft train --cycles 5 --output models/raft

# Output:
# Cycle 1 already complete, skipping...
# Cycle 2 already complete, skipping...
# Loading cached samples... (resumes cycle 3)
```

See [Troubleshooting](/docs/reference/troubleshooting/) for more solutions.

---

## Next Steps

- [RAFT Details](/docs/training-pipeline/raft/) — Deep dive into RAFT algorithm
- [Verifiers](/docs/verifiers/) — All verifier options
- [Configuration](/docs/reference/configuration/) — Complete config reference
- [Theory & Research](/docs/background/theory/) — Research foundations

