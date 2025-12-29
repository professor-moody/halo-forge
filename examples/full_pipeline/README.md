# Full Pipeline Example

Complete example showing the entire halo-forge pipeline.

## What This Example Does

1. **Data Preparation**: Downloads CodeForces C++ and MBPP Python datasets
2. **SFT Training**: Trains base model on combined dataset
3. **RAFT Training**: Improves model with compile verification
4. **Benchmarking**: Evaluates final model

## Files

- `train.py` - Main training script
- `config.yaml` - Configuration file
- `run.sh` - Shell script to run everything

## Quick Start

```bash
# From halo-forge root
cd examples/full_pipeline

# Run the full pipeline
./run.sh

# Or step by step:
python train.py --step data      # Prepare data
python train.py --step sft       # SFT training
python train.py --step raft      # RAFT training
python train.py --step benchmark # Benchmark
```

## Expected Results

On Strix Halo with 96GB unified memory:

| Step | Time | Notes |
|------|------|-------|
| Data | 5 min | Downloads ~5000 examples |
| SFT | 2 hours | 3 epochs, 2048 seq len |
| RAFT | 4 hours | 5 cycles |
| Benchmark | 30 min | pass@1, pass@5, pass@10 |

Expected pass@1 after training: 60-80%

## Configuration

Edit `config.yaml` to adjust:

- Model (default: Qwen2.5-Coder-7B)
- Training hyperparameters
- RAFT cycles
- Verification settings

## Output

```
models/full_pipeline/
├── sft/
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── final_model/
├── raft/
│   ├── cycle_1_final/
│   ├── cycle_2_final/
│   └── cycle_5_final/
└── raft_statistics.json

results/
└── full_pipeline_benchmark.json
```

