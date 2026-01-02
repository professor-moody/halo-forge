# Benchmarking

Evaluate model performance with pass@k metrics.

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

> **Note**: Demo benchmarks validate the pipeline works, not RAFT's full potential. Production training shows 2-3x improvements.

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
