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
│   ├── train
│   └── datasets
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
│   ├── sft
│   ├── train
│   ├── benchmark
│   └── datasets
├── audio              [EXPERIMENTAL]
│   ├── sft
│   ├── train
│   ├── benchmark
│   └── datasets
├── reasoning          [EXPERIMENTAL]
│   ├── sft
│   ├── train
│   ├── benchmark
│   └── datasets
├── agentic            [EXPERIMENTAL]
│   ├── sft
│   ├── train
│   ├── benchmark
│   └── datasets
├── info
└── test
```

---

## Global Flags

These flags work with all commands:

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--quiet` | `-q` | flag | Suppress terminal output (logs still written to file) |

### Auto-Logging

All training and benchmark commands automatically log output to `logs/` with timestamped filenames:

```
logs/
├── raft_train_20260110_143052.log
├── sft_train_20260110_121500.log
└── benchmark_run_20260110_160000.log
```

No need for manual `tee` or `PYTHONUNBUFFERED`. Use `--quiet` to suppress terminal output while still capturing logs.

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
| `--dataset` | `-d` | string | No | - | HuggingFace dataset ID or short name |
| `--data` | - | path | No | - | Local training data file (JSONL) |
| `--max-samples` | - | int | No | - | Limit number of training samples |
| `--output` | `-o` | path | No | `models/sft` | Output directory |
| `--epochs` | - | int | No | 3 | Number of epochs |
| `--resume` | - | path | No | - | Resume from checkpoint |
| `--dry-run` | - | flag | No | false | Validate config without training |

**Dataset short names:** `codealpaca`, `metamath`, `gsm8k_sft`, `llava`, `librispeech_sft`, `xlam_sft`, `glaive_sft`

```bash
# Using HuggingFace dataset
halo-forge sft train --dataset codealpaca --model Qwen/Qwen2.5-Coder-3B --output models/sft_3b

# Using local data
halo-forge sft train --data data/sft.jsonl --model Qwen/Qwen2.5-Coder-3B --output models/sft_3b

# With sample limit
halo-forge sft train --dataset metamath --max-samples 50000 --epochs 2

# Resume from checkpoint
halo-forge sft train --config configs/sft.yaml --resume models/sft/checkpoint-500
```

---

### halo-forge sft datasets

List available SFT datasets.

```bash
halo-forge sft datasets
```

Output shows datasets organized by domain (Code, Reasoning, VLM, Audio, Agentic) with HuggingFace IDs and sizes.

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
| `--samples-per-prompt` | - | int | No | 8 | Samples to generate per prompt |
| `--temperature` | - | float | No | 0.7 | Generation temperature |
| `--max-new-tokens` | - | int | No | 1024 | Max tokens to generate |
| `--keep-percent` | - | float | No | 0.5 | Keep top X% of passing samples |
| `--reward-threshold` | - | float | No | 0.5 | Minimum reward to pass |
| `--min-samples` | - | int | No | - | Auto-adjust threshold if fewer pass |
| `--curriculum` | - | string | No | `none` | Curriculum strategy |
| `--curriculum-stats` | - | path | No | - | Historical stats file (for `historical` curriculum) |
| `--curriculum-start` | - | float | No | 0.2 | Start fraction (for `progressive` curriculum) |
| `--curriculum-increment` | - | float | No | 0.2 | Increment per cycle (for `progressive` curriculum) |
| `--reward-shaping` | - | string | No | `fixed` | Reward shaping strategy |
| `--lr-decay` | - | float | No | 0.85 | LR decay per cycle |
| `--min-lr` | - | float | No | 1e-6 | Minimum learning rate |
| `--experimental-attention` | - | flag | No | false | Enable experimental ROCm attention |
| `--system-prompt` | - | string | No | (Windows prompt) | System prompt |
| `--host` | - | string | No | - | MSVC verifier host |
| `--user` | - | string | No | - | MSVC verifier user |
| `--ssh-key` | - | path | No | - | MSVC verifier SSH key |

**Verifier choices:** `gcc`, `mingw`, `msvc`, `rust`, `go`, `dotnet`, `powershell`, `humaneval`, `mbpp`, `python`, `auto`

**Curriculum choices:** `none`, `complexity`, `progressive`, `adaptive`, `historical`

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
| `--verifier` | - | string | No | `gcc` | Verifier type (gcc, humaneval, mbpp, etc.) |
| `--base-model` | - | string | No | `Qwen/Qwen2.5-Coder-7B` | Base model |
| `--system-prompt` | - | string | No | (Windows prompt) | System prompt |
| `--host` | - | string | No | - | MSVC host |
| `--user` | - | string | No | - | MSVC user |
| `--ssh-key` | - | path | No | - | MSVC SSH key |
| `--cross-compile` | - | flag | No | false | Windows cross-compile (rust/go) |
| `--run-after-compile` | - | flag | No | false | Run after compile |
| `--experimental-attention` | - | flag | No | false | Enable experimental ROCm attention |

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

### halo-forge benchmark eval

Evaluate a model on standard benchmarks (HumanEval, MBPP, LiveCodeBench, etc.).

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | Yes | - | Model name or path |
| `--benchmark` | `-b` | string | No | `humaneval` | Benchmark name |
| `--limit` | - | int | No | - | Max samples to evaluate |
| `--output` | `-o` | path | No | - | Output file path |
| `--samples-per-prompt` | - | int | No | 5 | Samples per prompt for pass@k |
| `--run-after-compile` | - | flag | No | false | Run compiled code |
| `--language` | - | string | No | - | Language (cpp, rust, go) |
| `--verifier` | - | string | No | - | Verifier type |

**Benchmark choices:** `humaneval`, `mbpp`, `livecodebench`, `cpp`, `rust`, `go`

```bash
halo-forge benchmark eval --model models/raft/final --benchmark humaneval --limit 164
halo-forge benchmark eval --model models/raft/final --benchmark cpp --language cpp --run-after-compile
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

### halo-forge vlm sft

SFT training for VLM.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No | `Qwen/Qwen2-VL-2B-Instruct` | VLM model |
| `--dataset` | `-d` | string | No | `llava` | SFT dataset name |
| `--max-samples` | - | int | No | - | Limit training samples |
| `--output` | `-o` | path | No | `models/vlm_sft` | Output directory |
| `--epochs` | - | int | No | 2 | Number of epochs |
| `--dry-run` | - | flag | No | false | Validate config only |

```bash
halo-forge vlm sft --dataset llava --model Qwen/Qwen2-VL-2B-Instruct --output models/vlm_sft
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

### halo-forge audio sft

SFT training for audio models.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No | `openai/whisper-small` | Audio model |
| `--dataset` | `-d` | string | No | `librispeech_sft` | SFT dataset name |
| `--max-samples` | - | int | No | - | Limit training samples |
| `--output` | `-o` | path | No | `models/audio_sft` | Output directory |
| `--epochs` | - | int | No | 3 | Number of epochs |
| `--dry-run` | - | flag | No | false | Validate config only |

```bash
halo-forge audio sft --dataset librispeech_sft --model openai/whisper-small --output models/audio_sft
```

---

### halo-forge audio train

Train audio model with RAFT.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No | `openai/whisper-small` | Audio model |
| `--dataset` | `-d` | string | Yes | - | Dataset name |
| `--task` | - | string | No | `asr` | Task: `asr`, `tts`, `classification` |
| `--output` | `-o` | path | No | `models/audio_raft` | Output directory |
| `--cycles` | - | int | No | 4 | RAFT cycles |
| `--lr` | - | float | No | 1e-5 | Learning rate |
| `--lr-decay` | - | float | No | 0.85 | LR decay per cycle |
| `--limit` | - | int | No | - | Limit dataset samples |

**Dataset choices:** `librispeech`, `common_voice`, `audioset`, `speech_commands`

```bash
halo-forge audio train \
  --model openai/whisper-small \
  --dataset librispeech \
  --task asr \
  --cycles 4 \
  --output models/audio_asr
```

---

### halo-forge audio benchmark

Benchmark audio model on dataset.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | path | Yes | - | Audio model path |
| `--dataset` | `-d` | string | No | `librispeech` | Dataset name |
| `--task` | - | string | No | `asr` | Task type |
| `--limit` | - | int | No | 100 | Limit samples |
| `--output` | `-o` | path | No | - | Output file |

```bash
halo-forge audio benchmark \
  --model openai/whisper-small \
  --dataset librispeech \
  --limit 50 \
  --output results/audio_benchmark.json
```

---

### halo-forge audio datasets

List available audio datasets.

```bash
halo-forge audio datasets
```

---

### halo-forge reasoning sft

SFT training for reasoning models.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No | `Qwen/Qwen2.5-3B-Instruct` | Base model |
| `--dataset` | `-d` | string | No | `metamath` | SFT dataset name |
| `--max-samples` | - | int | No | - | Limit training samples |
| `--output` | `-o` | path | No | `models/reasoning_sft` | Output directory |
| `--epochs` | - | int | No | 2 | Number of epochs |
| `--dry-run` | - | flag | No | false | Validate config only |

**SFT Dataset choices:** `metamath`, `gsm8k_sft`

```bash
halo-forge reasoning sft --dataset metamath --model Qwen/Qwen2.5-3B-Instruct --output models/reasoning_sft
```

---

### halo-forge reasoning train

Train on math/reasoning datasets with RAFT.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No | `Qwen/Qwen2.5-7B-Instruct` | Base model |
| `--dataset` | `-d` | string | Yes | - | Dataset name |
| `--output` | `-o` | path | No | `models/reasoning_raft` | Output directory |
| `--cycles` | - | int | No | 4 | RAFT cycles |
| `--lr` | - | float | No | 1e-5 | Learning rate |
| `--lr-decay` | - | float | No | 0.85 | LR decay per cycle |
| `--limit` | - | int | No | - | Limit dataset samples |

**RAFT Dataset choices:** `gsm8k`, `math`, `aime`

```bash
halo-forge reasoning train \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset gsm8k \
  --cycles 4 \
  --output models/reasoning_gsm8k
```

---

### halo-forge reasoning benchmark

Benchmark on math/reasoning dataset.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | path | Yes | - | Model path |
| `--dataset` | `-d` | string | No | `gsm8k` | Dataset name |
| `--limit` | - | int | No | 100 | Limit samples |
| `--output` | `-o` | path | No | - | Output file |

```bash
halo-forge reasoning benchmark \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset gsm8k \
  --limit 100 \
  --output results/reasoning_benchmark.json
```

---

### halo-forge reasoning datasets

List available math/reasoning datasets.

```bash
halo-forge reasoning datasets
```

---

## Agentic Commands (Experimental)

### halo-forge agentic sft

SFT training for tool calling models.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No | `Qwen/Qwen2.5-7B-Instruct` | Base model |
| `--dataset` | `-d` | string | No | `xlam_sft` | SFT dataset name |
| `--max-samples` | - | int | No | - | Limit training samples |
| `--output` | `-o` | path | No | `models/agentic_sft` | Output directory |
| `--epochs` | - | int | No | 2 | Number of epochs |
| `--dry-run` | - | flag | No | false | Validate config only |

**SFT Dataset choices:** `xlam_sft`, `glaive_sft`

```bash
halo-forge agentic sft --dataset xlam_sft --model Qwen/Qwen2.5-7B-Instruct --output models/agentic_sft
```

---

### halo-forge agentic train

Train on tool calling datasets with RAFT.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No | `Qwen/Qwen2.5-7B-Instruct` | Base model |
| `--dataset` | `-d` | string | No | `xlam` | RAFT Dataset: xlam, glaive |
| `--output` | `-o` | path | No | `models/agentic_raft` | Output directory |
| `--cycles` | - | int | No | 5 | RAFT cycles |
| `--lr` | - | float | No | 5e-5 | Learning rate |
| `--lr-decay` | - | float | No | 0.85 | LR decay per cycle |
| `--limit` | - | int | No | - | Limit dataset samples |
| `--dry-run` | - | flag | No | false | Validate config only |

```bash
halo-forge agentic train \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset xlam \
  --cycles 5 \
  --output models/agentic_raft
```

---

### halo-forge agentic benchmark

Benchmark tool calling accuracy.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No | `Qwen/Qwen2.5-7B-Instruct` | Model to benchmark |
| `--dataset` | `-d` | string | No | `xlam` | Dataset: xlam, glaive |
| `--limit` | - | int | No | 100 | Limit samples |
| `--output` | `-o` | path | No | - | Output file |

```bash
halo-forge agentic benchmark \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset xlam \
  --limit 100 \
  --output results/agentic_benchmark.json
```

---

### halo-forge agentic datasets

List available tool calling datasets.

```bash
halo-forge agentic datasets
```

Output:
```
Available Agentic / Tool Calling Datasets
============================================================
  xlam         [Tool Calling] - 60k verified, 3,673 APIs
  glaive       [Tool Calling] - 113k samples, irrelevance
  toolbench    [Tool Calling] - 188k samples, 16k APIs
  hermes       [Tool Calling] - Format reference
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
