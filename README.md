# halo-forge

Complete RLVR (Reinforcement Learning from Verifier Rewards) training framework for AMD Strix Halo.

Train models to generate verified code using the full pipeline: data generation, SFT, RAFT/RLVR training, and benchmarking.

## Features

- **Custom Toolbox**: Self-contained ROCm 7 + PyTorch 2.7 environment optimized for gfx1151
- **Data Generation**: Extract from public datasets (CodeForces, MBPP) or generate with LLMs
- **SFT Training**: QLoRA fine-tuning with Strix Halo optimizations
- **RAFT Training**: Reward-Ranked Fine-Tuning with pluggable verifiers
- **Benchmarking**: Pass@k evaluation with detailed metrics

## Quick Start

### 1. Build the Toolbox

```bash
cd toolbox
./build.sh
toolbox create halo-forge --image localhost/halo-forge:latest
toolbox enter halo-forge
```

### 2. Prepare Data

```bash
# Download public datasets
halo-forge data prepare --dataset codeforces_cpp --output data/

# Or generate with LLM
halo-forge data generate --spec rust_async --backend deepseek --output data/
```

### 3. Train

```bash
# SFT Training
halo-forge sft train --config configs/sft_example.yaml

# RAFT Training (with compilation verification)
halo-forge raft train --config configs/raft_example.yaml --cycles 3
```

### 4. Benchmark

```bash
halo-forge benchmark --model ./models/raft/cycle_3 --prompts data/prompts.jsonl
```

## Hardware Requirements

- **AMD Strix Halo** (gfx1151) with 96GB unified memory
- **Storage**: 100GB+ for models and datasets
- **OS**: Fedora 42+ recommended (for toolbox support)

### Performance Notes

gfx1151 training is currently 30-40x slower than NVIDIA H100 due to driver maturity. This framework is optimized for this hardware with:

- Eager attention (Flash Attention not stable on gfx1151)
- Gradient checkpointing enabled by default
- bf16 mixed precision
- Memory-efficient batch sizes

## Documentation

- [QUICKSTART.md](docs/QUICKSTART.md) - Get running in 5 minutes
- [FULL_PIPELINE.md](docs/FULL_PIPELINE.md) - Complete walkthrough
- [RAFT_VS_GRPO.md](docs/RAFT_VS_GRPO.md) - When to use which training method
- [CUSTOM_VERIFIERS.md](docs/CUSTOM_VERIFIERS.md) - Create your own verifiers
- [HARDWARE_NOTES.md](docs/HARDWARE_NOTES.md) - gfx1151 quirks and optimizations

## Project Structure

```
halo-forge/
├── toolbox/              # Custom ROCm toolbox
├── halo_forge/           # Main Python package
│   ├── data/             # Data generation modules
│   ├── sft/              # SFT training
│   ├── rlvr/             # RAFT/GRPO training + verifiers
│   └── benchmark/        # Evaluation
├── configs/              # Example configurations
├── docs/                 # Documentation
└── examples/             # Working examples
```

## Built-in Verifiers

| Verifier | Description |
|----------|-------------|
| `gcc` | Local GCC compilation |
| `mingw` | Cross-compile to Windows PE |
| `msvc_ssh` | Remote MSVC via SSH |
| `pytest` | Python test runner |

## Built-in Dataset Specs

| Dataset | Description | Size |
|---------|-------------|------|
| `codeforces_cpp` | C++ competitive programming | 4,000 |
| `codeforces_python` | Python competitive programming | 1,000 |
| `mbpp` | Python functions (Google) | 500 |
| `humaneval` | Python (OpenAI) | 164 |

## License

Apache 2.0

## Acknowledgments

- AMD for Strix Halo hardware
- scottt for the ROCm PyTorch wheel
- The Strix Halo community for testing and feedback

