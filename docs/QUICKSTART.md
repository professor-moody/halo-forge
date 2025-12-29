# Quick Start Guide

Get started with halo-forge in under 30 minutes.

## Prerequisites

- AMD Strix Halo hardware (or other ROCm-compatible GPU)
- Fedora 42+ with podman/toolbox
- 64GB+ RAM recommended

## 1. Build the Toolbox

```bash
cd halo-forge/toolbox
chmod +x build.sh
./build.sh

# Create the toolbox container
toolbox create halo-forge --image localhost/halo-forge:latest

# Enter the toolbox
toolbox enter halo-forge
```

## 2. Verify Setup

```bash
# Check hardware detection
halo-forge info

# Expected output for Strix Halo:
# GPU: AMD Radeon Graphics
# Memory: ~96 GB
# Strix Halo: Yes
```

## 3. Prepare Training Data

Choose one of the following:

### Option A: Public Dataset (Recommended for first run)

```bash
# Download CodeForces C++ examples
halo-forge data prepare --dataset codeforces_cpp --output data/train.jsonl

# Check what was downloaded
head -1 data/train.jsonl | python -m json.tool
```

### Option B: Generate with LLM

```bash
# Set your API key
export DEEPSEEK_API_KEY=your_key_here

# Generate Rust async examples
halo-forge data generate --topic rust_async --backend deepseek --output data/rust.jsonl
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

## 5. Run RAFT Training (Optional)

After SFT, improve the model with verification:

```bash
# Create prompts file
echo '{"prompt": "Write a C++ function to sort a vector"}' > data/prompts.jsonl
echo '{"prompt": "Write a C++ function to read a file"}' >> data/prompts.jsonl

# Run RAFT with GCC verification
halo-forge raft train \
  --checkpoint models/sft/final_model \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --cycles 3
```

## 6. Benchmark Your Model

```bash
halo-forge benchmark run \
  --model models/raft/cycle_3_final \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --samples 10 \
  --output results/benchmark.json
```

## Next Steps

- Read [FULL_PIPELINE.md](FULL_PIPELINE.md) for complete training guide
- See [VERIFIERS.md](VERIFIERS.md) for custom verification
- Check [examples/](../examples/) for working examples

## Troubleshooting

### "ROCm not available"

Make sure you're inside the toolbox:
```bash
toolbox enter halo-forge
```

### Out of Memory

Reduce batch size in config or use more gradient accumulation:
```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 64
```

### Slow Performance

Check kernel version and BIOS settings:
- Kernel 6.16+ recommended for gfx1151
- UMA Frame Buffer Size: 16GB+ in BIOS

