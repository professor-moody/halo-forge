# Benchmarking Guide

This document explains the halo-forge benchmark system, what results to expect, and how to interpret them.

---

## Quick Start

```bash
# Smoke test (no GPU required)
halo-forge test --level smoke

# Demo benchmark (single model)
halo-forge benchmark full --model Qwen/Qwen2.5-Coder-0.5B --cycles 2

# Full benchmark suite
halo-forge benchmark full --suite all
```

---

## Understanding Benchmarks

### Demo Benchmarks vs Production Training

| Aspect | Demo Benchmark | Production Training |
|--------|----------------|---------------------|
| **Purpose** | Validate pipeline works | Train production model |
| **Prompts** | 16 built-in | 500+ domain-specific |
| **Cycles** | 2 | 5-8 |
| **Time** | 40-150 min | 6-24 hours |
| **Expected improvement** | 0-10% | 100-300% |
| **Starting point** | Raw base model | SFT checkpoint |

**Demo benchmarks are for validation**, not performance demonstration. They confirm:
- The pipeline runs correctly on your hardware
- Model loading works
- Verification functions properly
- Training loop executes without errors

**Production training shows actual RAFT effectiveness** with:
- Larger, curated datasets
- More training cycles
- SFT foundation before RAFT
- Full hardware utilization

### Why Demo Results Seem Modest

With only 16 prompts and 2 cycles:
- Very limited training signal
- No SFT foundation to build on
- Insufficient data for meaningful learning
- Small models (0.5B-3B) have less capacity

This is expected. The demo exists to verify your setup, not to demonstrate RAFT's full potential.

---

## Built-in Benchmark Prompts

The demo benchmark uses 16 C++ prompts covering:

| Category | Examples |
|----------|----------|
| Basic I/O | Hello World, print numbers |
| Arithmetic | Sum, factorial, power |
| Arrays | Sum, max element, bubble sort |
| Algorithms | Binary search, GCD, Fibonacci |
| Strings | Reverse, palindrome, vowel count |

Each prompt has an expected output for runtime verification.

---

## Metrics Explained

### Compile Rate

Percentage of generated samples that compile without errors.

```
Compile Rate = (Samples that compile) / (Total samples)
```

A 32% compile rate means 32 out of 100 generated code samples compile successfully.

### pass@k

Probability that at least one of k samples solves the problem correctly.

```
pass@1  = At least 1 of 1 samples works
pass@8  = At least 1 of 8 samples works
```

Higher k values give more chances, so pass@8 > pass@1 typically.

### Tokens per Second

Generation speed. Higher is better.

```
Typical values (Strix Halo):
  0.5B model: 200-250 tok/s
  1.5B model: 150-200 tok/s
  3B model:   100-150 tok/s
  7B model:   60-100 tok/s
```

---

## Running Benchmarks

### Single Model Benchmark

```bash
halo-forge benchmark full \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --cycles 2 \
  --output results/my-benchmark
```

### Benchmark Suite

```bash
# All default models (0.5B, 1.5B, 3B)
halo-forge benchmark full --suite all

# Just small models
halo-forge benchmark full --suite small

# Medium models (0.5B, 1.5B)
halo-forge benchmark full --suite medium
```

### Output Files

```
results/benchmarks/qwen-0.5b/
├── summary.json           # Full results
├── baseline_samples.jsonl # Baseline generated code
├── baseline_hardware.csv  # Hardware metrics
├── cycle_1/
│   ├── samples.jsonl      # Generated samples
│   ├── kept.jsonl         # Filtered samples used for training
│   ├── hardware.csv       # Hardware metrics
│   └── checkpoint/        # Model checkpoint
├── cycle_2/
│   └── ...
├── final_samples.jsonl    # Final evaluation samples
└── final_hardware.csv     # Final hardware metrics
```

---

## Interpreting Results

### Good Demo Results

```
Baseline compile rate: 25-35%
Final compile rate:    25-40%
Improvement:           0-15%
```

For a 2-cycle demo with 16 prompts, this indicates the pipeline is working correctly.

### Warning Signs

| Symptom | Likely Cause |
|---------|--------------|
| 0% compile rate | Verifier not working, or model not generating code |
| No improvement | Normal for small datasets, not a problem |
| Decreasing rate | Model may be overfitting to small dataset |
| Very slow generation | Check GPU utilization, may need restart |

### Hardware Metrics

Note: GPU utilization requires root access to report correctly via rocm-smi.

```json
"hardware": {
  "gpu_peak_memory_gb": 11.9,    // GPU memory used
  "gpu_power_avg_w": 70.8,       // Power consumption
  "gpu_energy_wh": 11.42         // Total energy used
}
```

---

## Production Benchmark Example

For meaningful performance metrics, run production-scale training:

```bash
# 1. Prepare dataset (500+ prompts)
halo-forge data prepare --dataset codeforces --output data/prompts.jsonl

# 2. Run SFT first
halo-forge sft train --config configs/sft_example.yaml

# 3. Run RAFT with more cycles
halo-forge raft train \
  --config configs/raft_example.yaml \
  --checkpoint models/sft/final_model \
  --cycles 6
```

Expected production results:

| Metric | Baseline | After 6 Cycles | Improvement |
|--------|----------|----------------|-------------|
| Compile Rate | 15-25% | 45-55% | 2-3x |
| pass@1 | 10-20% | 40-55% | 3-4x |

---

## Comparing Results

### Cross-Model Comparison

Larger models generally:
- Start with higher baseline
- Show similar relative improvement
- Take longer to train/generate

### Cross-Run Comparison

To fairly compare runs:
- Use same prompts
- Use same number of cycles
- Use same hardware
- Report both baseline and final metrics

---

## Troubleshooting Benchmarks

### Benchmark Crashes

```bash
# Check if GPU is available
python3 -c "import torch; print(torch.cuda.is_available())"

# Check available memory
free -h

# Run with verbose output
halo-forge benchmark full --model MODEL --verbose
```

### Very Slow Generation

```bash
# Check GPU utilization (requires sudo)
sudo rocm-smi --showuse

# Restart Python to clear GPU state
# Then re-run benchmark
```

### Results Not Saved

Check output directory permissions:

```bash
ls -la results/benchmarks/
# Should be writable by your user
```

---

## Contributing Benchmark Results

If you run benchmarks on new hardware, consider contributing:

1. Document your hardware specs
2. Run the full benchmark suite
3. Save all output files
4. Open a GitHub issue with results

This helps the community understand performance across different systems.

