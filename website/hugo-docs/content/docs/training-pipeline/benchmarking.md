---
title: "Benchmarking"
description: "Evaluate model performance with pass@k metrics"
weight: 5
---

This page covers **benchmark reporting** — evaluating trained models with standard metrics for comparison to published results.

> **Note**: Benchmarks produce metrics for papers and comparison. For training-time verification (RAFT loop), use [Verifiers](/docs/verifiers/) which provide graduated reward signals.

---

## Quick Start

```bash
halo-forge benchmark run \
  --model models/raft/cycle_5_final \
  --prompts data/test.jsonl \
  --verifier gcc \
  --samples 20
```

## Metrics

### Compile Rate

Percentage of samples that compile:

```
Compile Rate = Samples that compile / Total samples
```

### pass@k

Probability that at least one of k samples is correct:

| Metric | Meaning |
|--------|---------|
| pass@1 | Success with single attempt |
| pass@5 | Success within 5 attempts |
| pass@10 | Success within 10 attempts |

pass@k is computed using an unbiased estimator when `samples > k`.

## Running Benchmarks

### Single Model

```bash
halo-forge benchmark run \
  --model models/raft/cycle_3_final \
  --prompts data/test.jsonl \
  --verifier gcc \
  --samples 20 \
  --k 1,5,10 \
  --output results/benchmark.json
```

### Compare Models

```bash
# Benchmark each stage
for stage in sft raft/cycle_1 raft/cycle_3 raft/cycle_5; do
  halo-forge benchmark run \
    --model models/$stage/final_model \
    --prompts data/test.jsonl \
    --output results/${stage}.json
done

# Compare results
halo-forge benchmark compare results/*.json
```

### Demo Benchmark

Validate the pipeline with built-in prompts:

```bash
halo-forge benchmark full \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --cycles 2
```

## Output Format

```json
{
  "total": 100,
  "passed": 55,
  "pass_rate": 0.55,
  "pass_at_k": {
    "1": 0.553,
    "5": 0.784,
    "10": 0.891
  },
  "by_category": {
    "algorithms": {"total": 30, "passed": 18},
    "data_structures": {"total": 25, "passed": 15}
  },
  "timing": {
    "generation_time": 1234.5,
    "verification_time": 45.2
  }
}
```

## Demo Results

Quick validation benchmarks (16 prompts, 2 cycles):

| Model | Baseline | After 2 Cycles | Time |
|-------|----------|----------------|------|
| 0.5B | 32.0% | 32.0% | 41 min |
| 1.5B | 67.2% | 67.2% | 52 min |
| 3B | 97.7% | 99.2% | 79 min |

> **Note**: Demo benchmarks validate the pipeline works. Results from extended training runs will vary based on model, dataset, and configuration.

## HumanEval Validation

3B model RAFT training on HumanEval subset (20 prompts, 3 cycles):

| Cycle | Pass Rate | Kept | Loss | Grad Norm |
|-------|-----------|------|------|-----------|
| 1 | 21.2% | 17 | 0.563 | 995.3 |
| 2 | 17.5% | 14 | 0.570 | 194.0 |
| 3 | 22.5% | 18 | 0.526 | 0.47 |

Key observations:
- **Gradient norm stabilized**: 995 → 0.47 across cycles (good convergence)
- **Loss decreased**: 0.563 → 0.526 (model learning)
- **Pass rate variance**: Expected with small dataset (20 prompts)

Total training time: ~54 minutes (18 min/cycle)

## Hardware Metrics

With `--hardware-metrics`:

```json
{
  "hardware": {
    "gpu_peak_memory_gb": 11.9,
    "gpu_power_avg_w": 70.8,
    "gpu_energy_wh": 11.42
  }
}
```
