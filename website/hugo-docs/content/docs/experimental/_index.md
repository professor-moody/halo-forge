---
title: "Experimental Features"
weight: 80
---

# Experimental Features

This section contains features that are currently in development and testing. These modules extend halo-forge beyond code generation training into new domains.

**Status**: These features are functional but may change significantly as we iterate on the designs.

---

## Available Experimental Modules

### Vision-Language Training (Phase 3)

Train vision-language models using RLVR with perception-aware verification.

| Component | Status | Description |
|-----------|--------|-------------|
| VLMRAFTTrainer | Beta | RAFT training for VLMs |
| VisionVerifier | Beta | Multi-stage verification |
| PerceptionChecker | Beta | YOLOv8 + EasyOCR |
| Dataset Loaders | Beta | TextVQA, DocVQA, ChartQA |

```bash
# Quick start
halo-forge vlm train --model Qwen/Qwen2-VL-7B-Instruct --dataset textvqa
```

[Full VLM Documentation](vlm/)

---

### Inference Optimization (Phase 2)

Optimize trained models for deployment without full retraining.

| Component | Status | Description |
|-----------|--------|-------------|
| InferenceOptimizer | Alpha | End-to-end optimization |
| QATTrainer | Alpha | Quantization-aware training |
| GGUFExporter | Alpha | Export for llama.cpp |
| ONNXExporter | Alpha | Cross-platform export |

```bash
# Quick start
halo-forge inference export --model models/trained --format gguf --output model.gguf
```

[Full Inference Documentation](inference/)

---

### Liquid AI LFM2.5 Models

Liquid AI's [LFM2.5 family](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai) is now experimentally supported for RAFT training. These models feature a hybrid convolutional architecture optimized for edge deployment.

| Model | Parameters | Status | Notes |
|-------|------------|--------|-------|
| LiquidAI/LFM2.5-1.2B-Base | 1.2B | Beta | For custom fine-tuning |
| LiquidAI/LFM2.5-1.2B-Instruct | 1.2B | Beta | General instruction following |
| LiquidAI/LFM2.5-VL-1.6B | 1.6B | Alpha | Vision-language (untested) |

**Quick test:**
```bash
halo-forge benchmark run \
  --model LiquidAI/LFM2.5-1.2B-Instruct \
  --prompts data/prompts.jsonl \
  --verifier mingw \
  --samples 5
```

**Initial results** (5 Windows API prompts, 3 samples each):
- pass@1: 33%
- Compile rate: 40-60%
- PEFT/LoRA: ✅ Compatible
- Generation speed: ~47s per batch of 4

**Why LFM2.5?**
- Efficient 1.2B parameter size = fast iteration cycles
- AMD partnership suggests good ROCm compatibility
- Strong instruction following benchmarks

---

## Stability Levels

| Level | Meaning |
|-------|---------|
| **Alpha** | Early development, API may change significantly |
| **Beta** | Feature complete, API mostly stable |
| **Stable** | Production ready, in main documentation |

---

## Feedback

These features are actively developed based on testing results. If you encounter issues or have suggestions:

1. Check the [Troubleshooting](/docs/reference/troubleshooting/) guide
2. Review the [Changelog](/docs/changelog/) for recent changes
3. Open an issue on GitHub with reproduction steps

---

## Planned Features

Future experimental features:

| Feature | Status |
|---------|--------|
| Audio-Language Training | Planned |
| Reasoning & Math | Planned |
| Cross-Platform GUI | Planned |
