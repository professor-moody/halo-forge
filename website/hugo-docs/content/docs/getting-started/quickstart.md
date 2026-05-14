---
title: "Quick Start"
description: "Three practical paths from install to first useful Halo Forge run"
weight: 1
---

Halo Forge is easiest to learn by choosing the path that matches your first hour.

## Path 1: Local Beginner

Use this when you have one workstation and want a first successful training run.

### 1. Install

```bash
git clone https://github.com/professor-moody/halo-forge.git
cd halo-forge
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Apple Silicon users who want MLX inference and MLX-native trainers:

```bash
pip install -e '.[mlx]'
```

### 2. Check the machine

```bash
halo-forge info
halo-forge test --level smoke
```

If `info` reports CPU only, training still works for tiny smoke tests, but use a GPU or Apple Silicon backend for real runs.

### 3. Pick a small model

Start with one of these:

| Goal | Model |
|---|---|
| Code SFT or RAFT | `Qwen/Qwen2.5-Coder-1.5B` |
| Chat or preference tuning | `Qwen/Qwen2.5-3B-Instruct` |
| Reasoning smoke | `Qwen/Qwen2.5-1.5B-Instruct` |
| Apple MLX inference | `mlx-community/Qwen2.5-3B-Instruct-bf16` |
| Liquid tiny structured output experiment | `LiquidAI/LFM2.5-350M` |

See [Choose a Model](/docs/getting-started/choose-a-model/) for the full catalog guidance.

### 4. Run SFT

```bash
halo-forge sft train \
  --dataset codealpaca \
  --model Qwen/Qwen2.5-Coder-1.5B \
  --output models/sft-codealpaca-smoke \
  --epochs 1 \
  --max-samples 200
```

### 5. Evaluate and serve

```bash
halo-forge eval --model models/sft-codealpaca-smoke/final_model --tasks core
halo-forge serve --model models/sft-codealpaca-smoke/final_model
```

## Path 2: Evaluator Demo

Use this when you want to prove the workflow quickly without tuning every knob.

```bash
halo-forge models list --mode sft --backend mps
halo-forge sft train \
  --dataset codealpaca \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --output models/demo-sft \
  --epochs 1 \
  --max-samples 50

halo-forge benchmark full \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --cycles 1
```

Open the dashboard:

```bash
# Terminal 1
halo-forge serve-public

# Terminal 2
cd public_app
npm install
npm run dev
```

Then open `http://127.0.0.1:3000`, use **Start** for the guided first run, open **Models** for the curated catalog, and use **Bundles** to save groups of trained runs for comparison.

## Dashboard First Run

The dashboard is the easiest way to avoid a bad first launch:

1. Open **Start**.
2. Choose a goal: **Code**, **Reasoning**, **Tool use**, or **Apple Silicon**.
3. Review the generated SFT launch. Start keeps the model, dataset, sample count, and output path conservative.
4. Watch **Preflight** before launching. It distinguishes missing inputs, backend detection, and server-side launch checks.
5. Press **Launch** when the checks pass, then open the run monitor from the success state.

Use **Advanced training** after the first clean SFT run when you want RAFT, VLM, audio, preference tuning, or direct knob control.

For model exploration, open **Models** first and use the workstation filter plus intent filters: **First run**, **Code RAFT**, **Apple Silicon**, **VLM**, **Audio**, and **Liquid AI**.

## Path 3: Power User

Use this when you already know your target model, dataset, and backend.

```bash
halo-forge models list --mode raft --provider Qwen
halo-forge raft train \
  --model Qwen/Qwen2.5-Coder-3B \
  --prompts data/rlvr/humaneval_prompts.jsonl \
  --verifier execution \
  --cycles 3 \
  --samples-per-prompt 8 \
  --reward-threshold 0.5 \
  --output models/raft-code

halo-forge eval --model models/raft-code/cycle_3_final --tasks core
halo-forge convert --source models/raft-code/cycle_3_final --format gguf --quant q4 --output models/raft-code.gguf
```

For Apple Silicon, use `--accelerator mlx` for MLX-native inference paths and MLX-format models. For PyTorch training on Apple Silicon, use MPS defaults unless a trainer explicitly supports MLX.

## What To Read Next

- [Choose a Model](/docs/getting-started/choose-a-model/) - model family and backend guidance.
- [Usage Scenarios](/docs/getting-started/scenarios/) - complete recipes by goal.
- [Hardware Notes](/docs/getting-started/hardware/) - backend-specific setup.
- [Command Index](/docs/reference/command-index/) - full CLI reference.
