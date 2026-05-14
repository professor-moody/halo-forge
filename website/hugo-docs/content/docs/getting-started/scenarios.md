---
title: "Usage Scenarios"
description: "Runnable Halo Forge workflows by goal"
weight: 3
---

These scenarios are intentionally small. Scale samples, cycles, and model size after the workflow is producing trustworthy artifacts.

## Code: SFT To RAFT

```bash
halo-forge sft train \
  --dataset codealpaca \
  --model Qwen/Qwen2.5-Coder-1.5B \
  --output models/code-sft \
  --epochs 1 \
  --max-samples 500

halo-forge raft train \
  --checkpoint models/code-sft/final_model \
  --prompts data/rlvr/humaneval_prompts.jsonl \
  --verifier execution \
  --cycles 3 \
  --samples-per-prompt 8 \
  --output models/code-raft
```

Use this when you want the verifier to filter generated code before the next training pass.

## Preference Tuning: DPO

```bash
halo-forge dpo train \
  --dataset ultrafeedback \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output models/chat-dpo \
  --epochs 1 \
  --loss-type sigmoid
```

Use DPO when you want the standard preference baseline. On MLX, Halo Forge supports sigmoid, IPO, hinge, and KTO-pair paths.

## Reasoning: GRPO

```bash
halo-forge grpo train \
  --dataset gsm8k \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --verifier execution \
  --num-generations 4 \
  --reward-threshold 0.5 \
  --output models/reasoning-grpo
```

Use this when reward can be checked mechanically or with a strict verifier.

## VLM: Document Extraction

```bash
halo-forge vlm train \
  --dataset textvqa \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --cycles 2 \
  --limit 24 \
  --output models/vlm-docs \
  --allow-prototype-train
```

For custom forms or invoices, prepare JSONL rows with image paths, prompts, and expected fields before scaling.

## Audio: ASR Adaptation

```bash
halo-forge audio train \
  --dataset librispeech \
  --model openai/whisper-small \
  --task asr \
  --cycles 2 \
  --output models/audio-asr \
  --allow-prototype-train
```

Use this for speech-to-text adaptation. Liquid audio models are interesting, but the safest current Halo Forge path is Whisper-compatible.

## Agentic: Tool Calling

```bash
halo-forge agentic train \
  --dataset xlam \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --cycles 2 \
  --limit 64 \
  --output models/agentic-tools \
  --allow-prototype-train
```

Use this when outputs must follow function-call or tool-call structure.

## Evaluate, Serve, Export

```bash
halo-forge eval --model models/code-sft/final_model --tasks core
halo-forge serve --model models/code-sft/final_model
halo-forge convert --source models/code-sft/final_model --format gguf --quant q4 --output models/code-sft.gguf
```

Evaluation tells you whether training helped. Serving lets you test the artifact behind an OpenAI-compatible API. Export prepares deployment artifacts.

## Apple Silicon MLX

```bash
halo-forge --accelerator mlx models list --backend mlx
halo-forge --accelerator mlx serve \
  --model mlx-community/Qwen2.5-3B-Instruct-bf16 \
  --backend mlx
```

Use MLX-format models on Apple Silicon. For PyTorch training on Apple Silicon, use the MPS backend unless a trainer explicitly supports MLX.
