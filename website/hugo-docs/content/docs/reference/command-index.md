---
title: "Command Index"
description: "Complete index of all halo-forge commands and flags"
weight: 1
---

# Command Index

Complete reference for all halo forge commands, subcommands, and flags.

---

## Command Hierarchy

```
halo-forge
├── config
│   └── validate
├── data
│   ├── prepare
│   ├── generate
│   └── validate
├── sft
│   └── train
├── raft
│   └── train
├── benchmark
│   ├── run
│   └── full
├── inference          [EXPERIMENTAL]
│   ├── optimize
│   ├── export
│   └── benchmark
├── vlm                [EXPERIMENTAL]
│   ├── train
│   ├── benchmark
│   └── datasets
├── info
└── test
```

---

## Core Commands (Production Ready)

### halo-forge config validate

Validate a configuration file.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `config` | - | path | Yes | - | Path to config file |
| `--type` | `-t` | string | No | `auto` | Config type: `raft`, `sft`, `auto` |
| `--verbose` | `-v` | flag | No | false | Show config contents |

```bash
halo-forge config validate configs/raft_windows.yaml
halo-forge config validate configs/sft.yaml --type sft --verbose
```

---

### halo-forge data prepare

Download and prepare public datasets.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--dataset` | `-d` | string | No | - | Dataset name |
| `--output` | `-o` | path | No | - | Output file path |
| `--template` | - | string | No | `qwen` | Chat template |
| `--system-prompt` | - | string | No | - | Override system prompt |
| `--list` | - | flag | No | false | List available datasets |

```bash
halo-forge data prepare --list
halo-forge data prepare --dataset humaneval --output data/humaneval.jsonl
halo-forge data prepare --dataset mbpp --template qwen --system-prompt "You are a Python expert."
```

---

### halo-forge data generate

Generate training data using LLM.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--topic` | `-t` | string | No | - | Topic name |
| `--backend` | `-b` | string | No | `deepseek` | LLM backend |
| `--model` | - | string | No | - | Model name for backend |
| `--output` | `-o` | path | No | - | Output file path |
| `--template` | - | string | No | `qwen` | Chat template |
| `--list` | - | flag | No | false | List available topics |

```bash
halo-forge data generate --list
halo-forge data generate --topic windows_api --backend deepseek --output data/windows.jsonl
```

---

### halo-forge data validate

Validate dataset format.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `file` | - | path | Yes | - | Path to JSONL file |
| `--preview` | `-p` | flag | No | false | Show preview of examples |

```bash
halo-forge data validate data/training.jsonl
halo-forge data validate data/training.jsonl --preview
```

---

### halo-forge sft train

Run supervised fine-tuning.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--config` | `-c` | path | No | - | Config file path |
| `--model` | `-m` | string | No | `Qwen/Qwen2.5-Coder-7B` | Base model |
| `--data` | - | path | No | - | Training data file |
| `--output` | `-o` | path | No | `models/sft` | Output directory |
| `--epochs` | - | int | No | 3 | Number of epochs |
| `--resume` | - | path | No | - | Resume from checkpoint |

```bash
halo-forge sft train --model Qwen/Qwen2.5-Coder-3B --data data/sft.jsonl --output models/sft_3b
halo-forge sft train --config configs/sft.yaml --resume models/sft/checkpoint-500
```

---

### halo-forge raft train

Run RAFT (Reward-Ranked Fine-Tuning).

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--config` | `-c` | path | No | - | Config file path |
| `--model` | `-m` | string | No | `Qwen/Qwen2.5-Coder-3B` | Base model |
| `--checkpoint` | - | path | No | - | SFT checkpoint path |
| `--prompts` | `-p` | path | No | - | Prompts file |
| `--output` | `-o` | path | No | `models/raft` | Output directory |
| `--cycles` | - | int | No | 6 | Number of RAFT cycles |
| `--verifier` | - | string | No | `gcc` | Verifier type (see below) |
| `--keep-percent` | - | float | No | 0.5 | Keep top X% of passing samples |
| `--reward-threshold` | - | float | No | 0.5 | Minimum reward to pass |
| `--curriculum` | - | string | No | `none` | Curriculum strategy |
| `--reward-shaping` | - | string | No | `fixed` | Reward shaping strategy |
| `--lr-decay` | - | float | No | 0.85 | LR decay per cycle |
| `--min-lr` | - | float | No | 1e-6 | Minimum learning rate |
| `--system-prompt` | - | string | No | (Windows prompt) | System prompt |
| `--host` | - | string | No | - | MSVC verifier host |
| `--user` | - | string | No | - | MSVC verifier user |
| `--ssh-key` | - | path | No | - | MSVC verifier SSH key |

**Verifier choices:** `gcc`, `mingw`, `msvc`, `rust`, `go`, `dotnet`, `powershell`, `auto`

**Curriculum choices:** `none`, `complexity`, `progressive`, `adaptive`

**Reward shaping choices:** `fixed`, `annealing`, `adaptive`, `warmup`

```bash
# Basic RAFT training
halo-forge raft train \
  --model Qwen/Qwen2.5-Coder-3B \
  --prompts data/prompts.jsonl \
  --verifier mingw \
  --cycles 6 \
  --output models/raft_3b

# With SFT checkpoint and LR decay
halo-forge raft train \
  --checkpoint models/sft_3b/final \
  --prompts data/prompts.jsonl \
  --verifier auto \
  --lr-decay 0.85 \
  --cycles 6

# With MSVC verifier
halo-forge raft train \
  --prompts data/windows.jsonl \
  --verifier msvc \
  --host 10.0.0.152 \
  --user keys \
  --ssh-key ~/.ssh/win
```

---

### halo-forge benchmark run

Run pass@k benchmark.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | path | Yes | - | Model path |
| `--prompts` | `-p` | path | Yes | - | Prompts file |
| `--output` | `-o` | path | No | - | Output file path |
| `--samples` | - | int | No | 10 | Samples per prompt |
| `--k` | - | string | No | `1,5,10` | k values (comma-separated) |
| `--max-prompts` | - | int | No | - | Max prompts to evaluate |
| `--verifier` | - | string | No | `gcc` | Verifier type |
| `--base-model` | - | string | No | `Qwen/Qwen2.5-Coder-7B` | Base model |
| `--system-prompt` | - | string | No | (Windows prompt) | System prompt |
| `--host` | - | string | No | - | MSVC host |
| `--user` | - | string | No | - | MSVC user |
| `--ssh-key` | - | path | No | - | MSVC SSH key |
| `--cross-compile` | - | flag | No | false | Windows cross-compile (rust/go) |
| `--run-after-compile` | - | flag | No | false | Run after compile |

```bash
halo-forge benchmark run \
  --model models/raft_3b/cycle_6 \
  --prompts data/test.jsonl \
  --verifier mingw \
  --samples 10 \
  --output results/benchmark.json
```

---

### halo-forge benchmark full

Run comprehensive RAFT benchmark.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No* | - | Model to benchmark |
| `--suite` | `-s` | string | No* | - | Predefined suite |
| `--cycles` | `-c` | int | No | 2 | RAFT cycles |
| `--output` | `-o` | path | No | `results/benchmarks` | Output directory |
| `--quiet` | `-q` | flag | No | false | Minimal output |

*Either `--model` or `--suite` is required.

**Suite choices:** `all` (0.5B, 1.5B, 3B), `small` (0.5B), `medium` (0.5B, 1.5B)

```bash
halo-forge benchmark full --model Qwen/Qwen2.5-Coder-0.5B --cycles 2
halo-forge benchmark full --suite all --output results/full_benchmark
```

---

### halo-forge info

Show hardware and system information.

```bash
halo-forge info
```

---

### halo-forge test

Run pipeline validation tests.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--level` | `-l` | string | No | `standard` | Test level |
| `--model` | `-m` | string | No | `Qwen/Qwen2.5-Coder-0.5B` | Model for testing |
| `--verbose` | `-v` | flag | No | false | Verbose output |

**Level choices:** `smoke` (no GPU), `standard` (with GPU), `full` (with training)

```bash
halo-forge test --level smoke
halo-forge test --level standard --verbose
halo-forge test --level full --model Qwen/Qwen2.5-Coder-1.5B
```

---

## Experimental Commands

These commands are in active development. APIs may change.

### halo-forge inference optimize

Optimize model for deployment.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | path | Yes | - | Model path |
| `--target-precision` | - | string | No | `int4` | Target precision |
| `--target-latency` | - | float | No | 50.0 | Target latency (ms) |
| `--calibration-data` | - | path | No | - | Calibration data JSONL |
| `--output` | `-o` | path | No | `models/optimized` | Output directory |

**Precision choices:** `int4`, `int8`, `fp16`

```bash
halo-forge inference optimize \
  --model models/raft_7b/cycle_6 \
  --target-precision int4 \
  --output models/optimized
```

---

### halo-forge inference export

Export model to deployment format.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | path | Yes | - | Model path |
| `--format` | `-f` | string | Yes | - | Export format |
| `--quantization` | `-q` | string | No | `Q4_K_M` | GGUF quantization |
| `--output` | `-o` | path | Yes | - | Output path |

**Format choices:** `gguf`, `onnx`

**GGUF quantization types:** `Q4_K_M`, `Q4_K_S`, `Q8_0`, `F16`

```bash
halo-forge inference export \
  --model models/trained \
  --format gguf \
  --quantization Q4_K_M \
  --output models/model.gguf
```

---

### halo-forge inference benchmark

Benchmark inference latency.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | path | Yes | - | Model path |
| `--prompts` | `-p` | path | No | - | Test prompts JSONL |
| `--num-prompts` | - | int | No | 10 | Number of prompts |
| `--max-tokens` | - | int | No | 100 | Max tokens to generate |
| `--warmup` | - | int | No | 3 | Warmup iterations |
| `--measure-memory` | - | flag | No | false | Measure memory usage |

```bash
halo-forge inference benchmark \
  --model models/optimized \
  --num-prompts 50 \
  --measure-memory
```

---

### halo-forge vlm train

Train VLM with RAFT.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No | `Qwen/Qwen2-VL-7B-Instruct` | VLM model |
| `--dataset` | `-d` | string | Yes | - | Dataset name or JSONL |
| `--output` | `-o` | path | No | `models/vlm_raft` | Output directory |
| `--cycles` | - | int | No | 6 | RAFT cycles |
| `--samples-per-prompt` | - | int | No | 4 | Samples per prompt |
| `--perception-weight` | - | float | No | 0.3 | Perception weight |
| `--reasoning-weight` | - | float | No | 0.4 | Reasoning weight |
| `--output-weight` | - | float | No | 0.3 | Output weight |
| `--lr-decay` | - | float | No | 0.85 | LR decay per cycle |
| `--temperature` | - | float | No | 0.7 | Generation temperature |
| `--limit` | - | int | No | - | Limit dataset samples |

**Dataset choices:** `textvqa`, `docvqa`, `chartqa`, `realworldqa`, `mathvista`

```bash
halo-forge vlm train \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --dataset textvqa \
  --cycles 6 \
  --output models/vlm_textvqa
```

---

### halo-forge vlm benchmark

Benchmark VLM on dataset.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | path | Yes | - | VLM model path |
| `--dataset` | `-d` | string | No | `textvqa` | Dataset name |
| `--split` | - | string | No | `validation` | Dataset split |
| `--limit` | - | int | No | 100 | Limit samples |
| `--output` | `-o` | path | No | - | Output file |

```bash
halo-forge vlm benchmark \
  --model models/vlm_raft/cycle_6 \
  --dataset docvqa \
  --limit 200 \
  --output results/vlm_benchmark.json
```

---

### halo-forge vlm datasets

List available VLM datasets.

```bash
halo-forge vlm datasets
```

---

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Configuration error |
| 4 | GPU not available |
| 5 | Verification failed |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PYTORCH_HIP_ALLOC_CONF` | ROCm memory configuration |
| `HF_HOME` | HuggingFace cache directory |
| `CUDA_VISIBLE_DEVICES` | GPU selection |
| `HIP_VISIBLE_DEVICES` | AMD GPU selection |

---

## Quick Reference

### Most Common Commands

```bash
# Test installation
halo-forge test --level smoke

# Train with RAFT
halo-forge raft train --prompts data/prompts.jsonl --verifier mingw --cycles 6

# Benchmark model
halo-forge benchmark run --model models/raft/cycle_6 --prompts data/test.jsonl

# Show info
halo-forge info
```

### Verifier Quick Reference

| Verifier | Language | Cross-compile | Requires |
|----------|----------|---------------|----------|
| `gcc` | C/C++ | No | gcc installed |
| `mingw` | C/C++ | Yes (Windows PE) | mingw-w64 |
| `msvc` | C/C++ | Yes (Windows) | SSH to Windows |
| `rust` | Rust | Yes (Windows) | rustc, cargo |
| `go` | Go | Yes (Windows) | go installed |
| `dotnet` | C# | Yes (Windows PE) | dotnet-sdk |
| `powershell` | PowerShell | No | pwsh |
| `auto` | Multi-lang | Varies | Depends on detected |
