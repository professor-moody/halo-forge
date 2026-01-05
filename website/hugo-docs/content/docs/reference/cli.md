---
title: "CLI Reference"
description: "Complete command-line interface reference"
weight: 2
---

Complete reference for all halo-forge CLI commands and options.

## Command Overview

| Command | Description |
|---------|-------------|
| `halo-forge test` | Validate installation |
| `halo-forge info` | Show system information |
| `halo-forge data prepare` | Download public datasets |
| `halo-forge data generate` | Generate data with LLM |
| `halo-forge data validate` | Validate dataset format |
| `halo-forge sft train` | Run supervised fine-tuning |
| `halo-forge raft train` | Run RAFT training |
| `halo-forge benchmark run` | Evaluate model performance |
| `halo-forge benchmark full` | Complete benchmark suite |

---

## halo-forge test

Validate your installation at various levels.

```bash
halo-forge test [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--level` | string | `smoke` | Test level: `smoke`, `standard`, `full` |
| `--verbose` | flag | false | Show detailed output |
| `--model` | string | `Qwen/Qwen2.5-Coder-0.5B` | Model for generation tests |

### Test Levels

| Level | Time | GPU | What It Tests |
|-------|------|-----|---------------|
| `smoke` | 5s | No | Imports, compiler availability, verifier logic |
| `standard` | 2-3 min | Yes | Model loading, code generation, verification |
| `full` | 5 min | Yes | Complete mini-RAFT cycle with training step |

### Examples

```bash
# Quick validation (no GPU needed)
halo-forge test --level smoke

# Standard test with GPU
halo-forge test --level standard

# Full test with verbose output
halo-forge test --level full --verbose

# Test with specific model
halo-forge test --level standard --model Qwen/Qwen2.5-Coder-3B
```

---

## halo-forge info

Display system and environment information.

```bash
halo-forge info [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--gpu` | flag | false | Show GPU details |
| `--memory` | flag | false | Show memory statistics |
| `--all` | flag | false | Show all available info |

### Examples

```bash
# Basic info
halo-forge info

# GPU details
halo-forge info --gpu

# All information
halo-forge info --all
```

### Sample Output

```
halo-forge v0.2.0
─────────────────────────────────────────────
Python:     3.13.1
PyTorch:    2.6.0.dev20241201+rocm6.3
GPU:        AMD Radeon Graphics (gfx1151)
ROCm:       /opt/rocm-7.0
Memory:     128GB unified
─────────────────────────────────────────────
```

---

## halo-forge data prepare

Download and format public datasets for training.

```bash
halo-forge data prepare [OPTIONS]
```

### Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--dataset` | string | Yes | Dataset name to download |
| `--output` | path | Yes | Output JSONL file path |
| `--template` | string | `qwen` | Chat template format |
| `--system-prompt` | string | - | Custom system prompt |
| `--limit` | int | - | Limit number of examples |
| `--list` | flag | - | List available datasets |

### Available Datasets

| Dataset | Language | Examples | Description |
|---------|----------|----------|-------------|
| `codeforces_cpp` | C++ | ~5000 | Competitive programming |
| `mbpp` | Python | 974 | Mostly Basic Programming Problems |
| `humaneval` | Python | 164 | HumanEval benchmark |
| `apps_intro` | Python | ~5000 | APPS introductory problems |

### Examples

```bash
# List available datasets
halo-forge data prepare --list

# Download CodeForces C++
halo-forge data prepare \
  --dataset codeforces_cpp \
  --output data/codeforces.jsonl

# Download with limit
halo-forge data prepare \
  --dataset mbpp \
  --output data/mbpp.jsonl \
  --limit 500

# Custom template
halo-forge data prepare \
  --dataset humaneval \
  --output data/humaneval.jsonl \
  --template llama
```

---

## halo-forge data generate

Generate training data using LLM backends.

```bash
halo-forge data generate [OPTIONS]
```

### Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--topic` | string | Yes | Topic specification to use |
| `--backend` | string | Yes | LLM backend: `ollama`, `deepseek`, `anthropic`, `openai` |
| `--model` | string | - | Model name (backend-specific) |
| `--output` | path | Yes | Output JSONL file path |
| `--template` | string | `qwen` | Chat template format |
| `--list` | flag | - | List available topics |

### Available Topics

| Topic | Language | Description |
|-------|----------|-------------|
| `python_algorithms` | Python | Algorithm implementations |
| `python_testing` | Python | Test-driven development |
| `rust_basics` | Rust | Rust fundamentals |
| `rust_async` | Rust | Async/await patterns |
| `cpp_systems` | C++ | Systems programming |
| `go_concurrency` | Go | Goroutines and channels |

### Backend Configuration

**Ollama (local, free):**
```bash
halo-forge data generate \
  --topic python_algorithms \
  --backend ollama \
  --model codellama:13b \
  --output data/generated.jsonl
```

**DeepSeek (API, cheap):**
```bash
export DEEPSEEK_API_KEY=your_key
halo-forge data generate \
  --topic rust_async \
  --backend deepseek \
  --output data/rust.jsonl
```

**Anthropic (API):**
```bash
export ANTHROPIC_API_KEY=your_key
halo-forge data generate \
  --topic cpp_systems \
  --backend anthropic \
  --model claude-sonnet-4-20250514 \
  --output data/cpp.jsonl
```

---

## halo-forge data validate

Validate dataset format and get statistics before training.

```bash
halo-forge data validate <file> [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `file` | path | Required | Path to JSONL file to validate |
| `--preview`, `-p` | flag | false | Show preview of examples |

### Supported Formats

| Format | Fields | Use Case |
|--------|--------|----------|
| `sft` | `{"text": "..."}` | SFT training (pre-formatted) |
| `prompt_response` | `{"prompt": "...", "response": "..."}` | Raw data for formatting |
| `prompts_only` | `{"prompt": "..."}` | RAFT training prompts |
| `messages` | `{"messages": [...]}` | Chat format |

### Examples

```bash
# Basic validation
halo-forge data validate data/train.jsonl

# With preview of first 3 examples
halo-forge data validate data/train.jsonl --preview
```

### Sample Output

```
============================================================
DATASET VALIDATION REPORT
============================================================

Status: ✓ VALID
Format: sft

Examples:
  Total:   500
  Valid:   500
  Invalid: 0

Fields Found:
  text:     500
  prompt:   0
  response: 0
  messages: 0

Length Statistics:
  Avg prompt:   1550 chars
  Avg response: 1446 chars
  Max prompt:   3602 chars
  Max response: 6653 chars

============================================================
```

---

## halo-forge sft train

Run supervised fine-tuning on a dataset.

```bash
halo-forge sft train [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | path | - | YAML configuration file |
| `--data` | path | - | Training data JSONL file |
| `--output` | path | `models/sft` | Output directory |
| `--model` | string | `Qwen/Qwen2.5-Coder-7B` | Base model |
| `--epochs` | int | 3 | Number of epochs |
| `--batch-size` | int | 2 | Per-device batch size |
| `--lr` | float | 2e-4 | Learning rate |
| `--resume` | path | - | Resume from checkpoint |

### Examples

```bash
# Basic SFT
halo-forge sft train \
  --data data/train.jsonl \
  --output models/sft \
  --epochs 3

# With configuration file
halo-forge sft train --config configs/sft.yaml

# Resume training
halo-forge sft train \
  --config configs/sft.yaml \
  --resume models/sft/checkpoint-500
```

### Configuration File

```yaml
# configs/sft.yaml
model:
  name: Qwen/Qwen2.5-Coder-7B
  trust_remote_code: true
  attn_implementation: eager

data:
  train_file: data/train.jsonl
  validation_split: 0.05
  max_seq_length: 2048

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

training:
  output_dir: models/sft
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 2e-4
  warmup_ratio: 0.03
  bf16: true
  gradient_checkpointing: true
  dataloader_num_workers: 0
  dataloader_pin_memory: false
```

---

## halo-forge raft train

Run RAFT (Reward-rAnked Fine-Tuning) training.

```bash
halo-forge raft train [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | path | - | YAML configuration file |
| `--checkpoint` | path | - | Starting checkpoint (SFT model) |
| `--model` | string | `Qwen/Qwen2.5-Coder-7B` | Base model (if no checkpoint) |
| `--prompts` | path | Required | Training prompts JSONL |
| `--verifier` | string | `gcc` | Verifier type |
| `--cycles` | int | 5 | Number of RAFT cycles |
| `--samples-per-prompt` | int | 8 | Samples per prompt |
| `--reward-threshold` | float | 0.5 | Minimum reward to keep |
| `--keep-percent` | float | 0.5 | Top percentage to keep |
| `--temperature` | float | 0.7 | Generation temperature |
| `--output` | path | `models/raft` | Output directory |

### Verifier Types

| Verifier | Language | Description |
|----------|----------|-------------|
| `gcc` | C/C++ | GCC compilation |
| `clang` | C/C++ | Clang compilation |
| `mingw` | C/C++ | Windows cross-compile |
| `msvc` | C/C++ | Remote MSVC (requires config) |
| `rust` | Rust | Cargo build |
| `go` | Go | Go build |
| `humaneval` | Python | HumanEval tests |
| `mbpp` | Python | MBPP tests |

### Examples

```bash
# Basic RAFT with GCC
halo-forge raft train \
  --model Qwen/Qwen2.5-Coder-7B \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --cycles 5 \
  --output models/raft

# From SFT checkpoint
halo-forge raft train \
  --checkpoint models/sft/final_model \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --cycles 5

# Python with MBPP tests
halo-forge raft train \
  --model Qwen/Qwen2.5-Coder-7B \
  --prompts data/rlvr/mbpp_train_prompts.jsonl \
  --verifier mbpp \
  --cycles 5

# Selective filtering (large dataset)
halo-forge raft train \
  --model Qwen/Qwen2.5-Coder-7B \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --keep-percent 0.2 \
  --reward-threshold 0.5

# High exploration
halo-forge raft train \
  --model Qwen/Qwen2.5-Coder-7B \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --samples-per-prompt 16 \
  --temperature 0.9
```

### Configuration File

```yaml
# configs/raft.yaml
sft_checkpoint: models/sft/final_model
output_dir: models/raft
prompts: data/prompts.jsonl

raft:
  num_cycles: 5
  samples_per_prompt: 8
  reward_threshold: 0.5
  keep_top_percent: 0.5

generation:
  max_new_tokens: 1024
  temperature: 0.7
  top_p: 0.95
  batch_size: 4

training:
  epochs: 1
  batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 5e-5
  dataloader_num_workers: 0
  dataloader_pin_memory: false

verifier:
  type: gcc
  run_after_compile: false

hardware:
  bf16: true
  gradient_checkpointing: true
```

---

## halo-forge benchmark run

Evaluate model performance on a benchmark.

```bash
halo-forge benchmark run [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | path | Required | Model to evaluate |
| `--prompts` | path | Required | Benchmark prompts |
| `--verifier` | string | `gcc` | Verifier for evaluation |
| `--samples` | int | 10 | Samples per problem |
| `--k` | string | `1,5,10` | k values for pass@k |
| `--temperature` | float | 0.7 | Generation temperature |
| `--output` | path | - | Output JSON file |

### Examples

```bash
# Basic benchmark
halo-forge benchmark run \
  --model models/raft/cycle_5_final \
  --prompts data/test.jsonl \
  --verifier gcc

# With output file
halo-forge benchmark run \
  --model models/raft/cycle_5_final \
  --prompts data/rlvr/mbpp_validation.jsonl \
  --verifier mbpp \
  --samples 20 \
  --k 1,5,10,20 \
  --output results/benchmark.json

# Compare baseline vs trained
halo-forge benchmark run \
  --model Qwen/Qwen2.5-Coder-7B \
  --prompts data/test.jsonl \
  --output results/baseline.json

halo-forge benchmark run \
  --model models/raft/cycle_5_final \
  --prompts data/test.jsonl \
  --output results/trained.json
```

### Output Format

```json
{
  "model": "models/raft/cycle_5_final",
  "prompts": "data/test.jsonl",
  "num_problems": 100,
  "samples_per_problem": 10,
  "pass_rate": 0.523,
  "pass_at_k": {
    "1": 0.312,
    "5": 0.478,
    "10": 0.523
  },
  "generation_time": 1234.5,
  "verification_time": 45.2
}
```

---

## halo-forge benchmark full

Run complete benchmark with before/after comparison.

```bash
halo-forge benchmark full [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | string | `Qwen/Qwen2.5-Coder-0.5B` | Base model |
| `--prompts` | path | - | Training prompts |
| `--verifier` | string | `gcc` | Verifier type |
| `--cycles` | int | 2 | RAFT cycles to run |
| `--suite` | string | `small` | Benchmark suite: `small`, `medium`, `all` |
| `--output` | path | `results/benchmark` | Output directory |

### Examples

```bash
# Quick validation benchmark
halo-forge benchmark full \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --cycles 2

# Full production benchmark
halo-forge benchmark full \
  --model Qwen/Qwen2.5-Coder-7B \
  --prompts data/rlvr/mbpp_train_prompts.jsonl \
  --verifier mbpp \
  --cycles 5 \
  --output results/production

# All model sizes
halo-forge benchmark full --suite all
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DEEPSEEK_API_KEY` | DeepSeek API key for data generation |
| `ANTHROPIC_API_KEY` | Anthropic API key for data generation |
| `OPENAI_API_KEY` | OpenAI API key for data generation |
| `HF_TOKEN` | HuggingFace token for private models |
| `HSA_ENABLE_SDMA` | Set to `0` to prevent GPU hangs |
| `PYTORCH_HIP_ALLOC_CONF` | PyTorch memory allocation config |

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
