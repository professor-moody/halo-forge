# Quick Start Guide

Get started with halo forge in under 30 minutes.

## Prerequisites

- AMD Strix Halo hardware (gfx1151) with 128GB unified memory
- Fedora 42+ with podman/toolbox
- Kernel 6.16+ recommended for gfx1151 support

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
halo forge Standard Test
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

## 7. Launch the Web UI (Optional)

halo-forge includes a web-based dashboard for training, benchmarking, and monitoring:

```bash
# Launch at http://127.0.0.1:8080
halo-forge ui

# Custom port
halo-forge ui --port 8888
```

The UI provides:
- **Dashboard**: GPU status, active jobs, charts
- **Training**: Configure and launch SFT/RAFT runs
- **Benchmark**: Run Code/VLM/Audio/Agentic benchmarks
- **Monitor**: Real-time progress, logs, and metrics
- **Results**: View and compare benchmark scores

See [WEB_UI.md](WEB_UI.md) for full documentation.

## Next Steps

- Read [FULL_PIPELINE.md](FULL_PIPELINE.md) for complete training guide
- See [VERIFIERS.md](VERIFIERS.md) for custom verification
- See [WEB_UI.md](WEB_UI.md) for web interface details
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

