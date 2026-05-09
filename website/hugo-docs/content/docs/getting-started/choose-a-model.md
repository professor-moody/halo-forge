---
title: "Choose a Model"
description: "How to pick a base model for SFT, RAFT, DPO, GRPO, VLM, audio, and serving"
weight: 2
---

Halo Forge now has a curated model catalog. In the CLI:

```bash
halo-forge models list
halo-forge models list --mode raft --backend mlx
halo-forge models show Qwen/Qwen2.5-Coder-3B
```

In the dashboard, open **Models**. The older **Run Bundles** page is different: it saves groups of your trained runs for comparison.

## First Picks

| Goal | Start with | Why |
|---|---|---|
| Code training | `Qwen/Qwen2.5-Coder-3B` | Best-tested code default for SFT and RAFT |
| Fast code smoke | `Qwen/Qwen2.5-Coder-0.5B` | Small enough for quick validation |
| Preference tuning | `Qwen/Qwen2.5-3B-Instruct` | Good DPO/ORPO/RM default |
| Reasoning | `Qwen/Qwen2.5-1.5B-Instruct` or `Qwen/Qwen2.5-Math-1.5B` | Small reasoning-friendly baselines |
| VLM | `Qwen/Qwen2-VL-2B-Instruct` or `Qwen/Qwen2.5-VL-3B-Instruct` | Safest current VLM adapter path |
| Audio | `openai/whisper-small` | Current Halo Forge audio path is Whisper-oriented |
| Apple MLX | `mlx-community/Qwen2.5-3B-Instruct-bf16` | MLX-format model that avoids HF conversion friction |
| Liquid AI experiment | `LiquidAI/LFM2.5-350M` | Tiny structured-output/tool-use candidate |

## Memory Tiers

| Tier | Typical models | Use |
|---|---|---|
| Tiny | 39M to 700M | Smoke tests, edge demos, CI |
| Small | 1B to 4B | First real local training runs |
| Medium | 7B to 15B | Quality-oriented workstation runs |
| Large | 24B+ | Advanced runs with large unified memory or multi-GPU |

## Family Notes

- **Qwen / Qwen Coder**: default recommendation for most users. Qwen Coder is the safest code SFT/RAFT path.
- **Llama, Mistral, Gemma**: good general baselines when licensing, tokenizer behavior, or ecosystem fit matters.
- **DeepSeek, StarCoder2, CodeLlama**: useful code alternatives. Verify dependency and tokenizer behavior before long runs.
- **Whisper**: current default for audio ASR training.
- **Qwen-VL**: safest current VLM training path in Halo Forge.
- **MLX community models**: use these on Apple Silicon when you want MLX-native inference or trainer paths.
- **Liquid AI LFM**: promising small and efficient models, but treat them as experimental in Halo Forge until each adapter path has been tested.

## Liquid AI Caveats

Liquid AI’s current public docs describe LFM2.5 text models from 350M to 1.2B, including instruction, thinking, and base variants. Their docs also list LFM2.5 vision and audio models. These are interesting for Halo Forge because they are small, edge-oriented, and include MLX/GGUF/vLLM-friendly distribution paths.

Use Liquid text models first for structured output, tool use, extraction, and reasoning experiments. The `LiquidAI/LFM2.5-350M` model card says it is not recommended for knowledge-intensive tasks or programming, so do not use it as your first code model.

Use Liquid VL/audio models as experimental entries only. Halo Forge’s VLM path is adapter-specific and the audio path is currently Whisper-oriented, so Liquid multimodal models need adapter validation before they should be trusted for training results.

Primary references:

- [Liquid model overview](https://www.liquid.ai/models)
- [LFM2.5 announcement](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)
- [Liquid text model docs](https://docs.liquid.ai/lfm/models/text-models)
- [Liquid audio model docs](https://docs.liquid.ai/lfm/models/audio-models)
- [LFM2.5-350M Hugging Face model card](https://huggingface.co/LiquidAI/LFM2.5-350M)

## Rules Of Thumb

1. Start smaller than you think. Prove the data and verifier first.
2. Use Qwen Coder for code unless you have a reason not to.
3. Use instruct models for DPO, ORPO, agentic, and chat refinement.
4. Use MLX-format repos for Apple MLX. Do not expect bitsandbytes-style runtime quantization there.
5. Treat `experimental` catalog entries as “interesting, not guaranteed.”
