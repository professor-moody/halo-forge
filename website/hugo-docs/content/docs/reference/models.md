---
title: "Model Catalog Reference"
description: "Catalog schema, model family status, and compatibility guidance"
weight: 6
---

Halo Forge has two different model concepts:

- **Model catalog**: curated upstream/base models you can train, evaluate, serve, or use as starting points.
- **Run collections**: saved groups of trained runs under **Runs**, used for
  comparison and cohort evaluation.

Use the catalog from CLI, API, or dashboard:

```bash
halo-forge models list
halo-forge models list --mode raft --backend rocm
halo-forge models show LiquidAI/LFM2.5-350M
```

```bash
curl http://127.0.0.1:8000/api/public/models | jq
curl "http://127.0.0.1:8000/api/public/models?mode=vlm&status=experimental" | jq
```

## Catalog Fields

Each entry includes:

| Field | Meaning |
|---|---|
| `id` | Hugging Face, MLX, or local-compatible model identifier |
| `provider`, `family` | Human grouping for browsing and filtering |
| `modalities` | `text`, `code`, `vision`, `audio` |
| `tasks` | Practical uses such as `raft`, `preference`, `asr`, `agentic` |
| `trainer_support` | Halo Forge trainers expected to work |
| `backend_support` | `cuda`, `rocm`, `mps`, `mlx`, `cpu` compatibility hints |
| `memory_tier` | Tiny, small, medium, or large |
| `status` | `recommended`, `compatible`, `experimental`, or `deprecated` |
| `known_caveats` | Warnings that should affect operator choice |
| `mlx_variant` | MLX-format sibling when known |

## Status Definitions

| Status | Use it when |
|---|---|
| `recommended` | It is a good default for at least one Halo Forge workflow |
| `compatible` | It should work, but is not the first pick |
| `experimental` | Interesting but not fully proven in Halo Forge |
| `deprecated` | Kept for migration context only |

## Family Guidance

### Qwen and Qwen Coder

Qwen is the safest first choice. Use Qwen Coder for code SFT/RAFT, Qwen Instruct for preference tuning, and Qwen-VL for current VLM workflows.

### Llama, Mistral, Gemma

Good general-purpose baselines. Watch license gates and chat-template differences.

### DeepSeek, StarCoder2, CodeLlama

Useful code alternatives. Test a short verifier-backed run before long jobs because tokenizer and dependency behavior can differ by release.

### Whisper

Current audio training defaults are Whisper-compatible. Start with `openai/whisper-tiny` for smoke tests and `openai/whisper-small` for useful ASR adaptation.

### MLX Community

Use MLX-format repos on Apple Silicon when selecting the MLX backend. Quantization is baked into the artifact; it is not bitsandbytes runtime quantization.

### Liquid AI LFM

Liquid LFM2.5 models are small, edge-oriented, and interesting for structured output, tool use, extraction, reasoning, and on-device deployment. Halo Forge lists them with `experimental` status until each path is tested end-to-end.

Recommended first Liquid experiments:

| Model | Use | Caveat |
|---|---|---|
| `LiquidAI/LFM2.5-350M` | Tiny structured output and tool-use experiments | Liquid’s model card says it is not recommended for knowledge-heavy tasks or programming |
| `LiquidAI/LFM2.5-1.2B-Instruct` | Small chat/tool-use experiments | Verify trainer behavior before long runs |
| `LiquidAI/LFM2.5-1.2B-Thinking` | Reasoning experiments | Treat as experimental until benchmarked locally |
| `LiquidAI/LFM2.5-VL-450M` / `LFM2.5-VL-1.6B` | Visual extraction experiments | Halo Forge VLM adapters need validation |
| `LiquidAI/LFM2.5-Audio-1.5B` | ASR/TTS experiments | Halo Forge audio path is currently Whisper-oriented |

Primary references:

- [Liquid model overview](https://www.liquid.ai/models)
- [LFM2.5 announcement](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)
- [Liquid text models](https://docs.liquid.ai/lfm/models/text-models)
- [Liquid audio models](https://docs.liquid.ai/lfm/models/audio-models)
- [LiquidAI/LFM2.5-350M on Hugging Face](https://huggingface.co/LiquidAI/LFM2.5-350M)

## Compatibility Is A Starting Point

The catalog answers “what should I try first?” It does not replace a smoke run. Before committing hours to a model:

```bash
halo-forge test --level smoke
halo-forge models show MODEL_ID
halo-forge sft train --model MODEL_ID --dataset codealpaca --epochs 1 --max-samples 50 --output models/smoke
```

Then evaluate the artifact before scaling samples, model size, or cycles.
