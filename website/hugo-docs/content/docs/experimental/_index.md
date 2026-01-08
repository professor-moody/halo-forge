---
title: "Experimental Features"
weight: 80
---

# Experimental Features

This section contains features that are currently in development and testing. These modules extend halo forge beyond code generation training into new domains.

**Status**: These features are functional but may change significantly as we iterate on the designs.

**Current Version**: v1.1.0

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
# Validate configuration first
halo-forge vlm train --model Qwen/Qwen2-VL-7B-Instruct --dataset textvqa --dry-run

# Then run training
halo-forge vlm train --model Qwen/Qwen2-VL-7B-Instruct --dataset textvqa
```

[Full VLM Documentation](vlm/)

---

### Audio-Language Training (Phase 4)

Train speech and audio models using RLVR with ASR/TTS verification.

| Component | Status | Description |
|-----------|--------|-------------|
| AudioRAFTTrainer | Beta | RAFT training for audio models |
| AudioVerifier | Beta | Multi-task verification |
| ASRChecker | Beta | Word Error Rate (WER) evaluation |
| TTSChecker | Beta | Quality and intelligibility scoring |
| Dataset Loaders | Beta | LibriSpeech, Common Voice, VoxPopuli |

```bash
# List available datasets
halo-forge audio datasets

# Validate configuration
halo-forge audio train --model openai/whisper-small --dataset librispeech --dry-run

# Train with ASR verification
halo-forge audio train --model openai/whisper-small --dataset librispeech --cycles 3
```

[Full Audio Documentation](audio/)

---

### Reasoning & Math Training (Phase 5)

Train models on mathematical and logical reasoning with symbolic verification.

| Component | Status | Description |
|-----------|--------|-------------|
| ReasoningRAFTTrainer | Beta | RAFT training for reasoning |
| ReasoningVerifier | Beta | Multi-method verification |
| MathVerifier | Beta | SymPy symbolic evaluation |
| Dataset Loaders | Beta | GSM8K, MATH, ARC |

```bash
# List available datasets
halo-forge reasoning datasets

# Train on GSM8K
halo-forge reasoning train --model Qwen/Qwen2.5-3B-Instruct --dataset gsm8k --cycles 3
```

[Full Reasoning Documentation](reasoning/)

---

### Agentic / Tool Calling (Phase 6)

Train models to generate structured function calls with schema-aware verification.

| Component | Status | Description |
|-----------|--------|-------------|
| AgenticRAFTTrainer | Beta | RAFT training for tool calling |
| ToolCallingVerifier | Beta | JSON schema validation |
| HermesFormatter | Beta | Hermes chat format conversion |
| Dataset Loaders | Beta | xLAM, Glaive, ToolBench |

**Key Features:**
- Graduated reward structure (partial credit for valid JSON, schema compliance)
- Hermes format output (XML-style tags in ChatML, compatible with Qwen/Llama)
- Supports multi-turn conversations with tool results

```bash
# List available datasets
halo-forge agentic datasets

# Validate configuration
halo-forge agentic train --model Qwen/Qwen2.5-7B-Instruct --dataset xlam --dry-run

# Train on xLAM dataset
halo-forge agentic train --model Qwen/Qwen2.5-7B-Instruct --dataset xlam --cycles 3
```

[Full Agentic Documentation](agentic/)

---

### Inference Optimization (Phase 2)

Optimize trained models for deployment without full retraining.

| Component | Status | Description |
|-----------|--------|-------------|
| InferenceOptimizer | Beta | End-to-end optimization |
| QATTrainer | Alpha | Quantization-aware training |
| GGUFExporter | Beta | Export for llama.cpp |
| ONNXExporter | Alpha | Cross-platform export |

```bash
# Check dependencies and validate config
halo-forge inference optimize --model models/trained --dry-run

# Export to GGUF
halo-forge inference export --model models/trained --format gguf --output model.gguf
```

[Full Inference Documentation](inference/)

---

### Liquid AI LFM2.5 Models

Liquid AI's [LFM2.5 family](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai) is experimentally supported for RAFT training. These models feature a hybrid convolutional architecture optimized for edge deployment.

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
- PEFT/LoRA: Compatible
- Generation speed: ~47s per batch of 4

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

| Feature | Status |
|---------|--------|
| Full Testing / Artifacts | In Progress |
| Cross-Platform GUI | Planned |
