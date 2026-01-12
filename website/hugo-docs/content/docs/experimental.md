---
title: "Experimental Features"
description: "Features under active development: VLM, Audio, Reasoning, Agentic, Inference"
weight: 80
---

This section contains features that are currently in development and testing. These modules extend halo-forge beyond code generation training into new domains.

**Status**: These features are functional but may change significantly as we iterate.

---

## Vision-Language (VLM)

Train vision-language models using RLVR with perception-aware verification.

| Component | Status | Description |
|-----------|--------|-------------|
| VLMRAFTTrainer | Beta | RAFT training for VLMs |
| VisionVerifier | Beta | Multi-stage verification |
| PerceptionChecker | Beta | YOLOv8 + EasyOCR |

### Quick Start

```bash
# Full pipeline
halo-forge vlm sft --dataset llava --model Qwen/Qwen2-VL-2B-Instruct --output models/vlm_sft
halo-forge vlm train --model models/vlm_sft --dataset textvqa --cycles 6 --output models/vlm_raft
halo-forge vlm benchmark --model models/vlm_raft --dataset docvqa --limit 100

# Quick RAFT (skip SFT)
halo-forge vlm train --model Qwen/Qwen2-VL-7B-Instruct --dataset textvqa --cycles 6
```

### Supported Models

| Model | Adapter | Notes |
|-------|---------|-------|
| Qwen/Qwen2-VL-2B-Instruct | qwen_vl | Lightweight |
| Qwen/Qwen2-VL-7B-Instruct | qwen_vl | Recommended |
| llava-hf/llava-1.5-7b-hf | llava | Good baseline |

### Datasets

| Dataset | Task | Size |
|---------|------|------|
| TextVQA | Text reading in images | 45K train |
| DocVQA | Document understanding | 50K train |
| ChartQA | Chart interpretation | 28K train |
| RealWorldQA | Real-world reasoning | 700 test |

### Verification Architecture

The VisionVerifier uses multi-stage verification:
- **Perception** (0.3): Object detection, OCR, spatial reasoning
- **Reasoning** (0.4): Structure, consistency, grounding
- **Output** (0.3): Exact match, fuzzy match, semantic similarity

---

## Audio-Language

Train audio models (ASR, TTS, Classification) using RLVR with task-specific verification.

| Component | Status | Description |
|-----------|--------|-------------|
| AudioRAFTTrainer | Beta | RAFT for audio models |
| AudioVerifier | Beta | Multi-task verification |
| ASRChecker | Beta | Word Error Rate (WER) |

### Quick Start

```bash
# Full pipeline
halo-forge audio sft --dataset librispeech_sft --model openai/whisper-small --output models/audio_sft
halo-forge audio train --model models/audio_sft --dataset librispeech --task asr --cycles 4
halo-forge audio benchmark --model models/audio_raft --dataset librispeech --limit 100

# Quick RAFT
halo-forge audio train --model openai/whisper-small --dataset librispeech --task asr --cycles 4
```

### Supported Models

| Model | Task | Notes |
|-------|------|-------|
| openai/whisper-tiny | ASR | Fast, lightweight |
| openai/whisper-small | ASR | Recommended |
| openai/whisper-medium | ASR | Better accuracy |
| openai/whisper-large-v3 | ASR | Best quality |

### Reward Structure (ASR)

| WER | Reward |
|-----|--------|
| 0% | 1.0 (perfect) |
| 10% | 0.9 |
| 30% | 0.7 |
| 50% | 0.5 |

---

## Reasoning and Math

Train models on mathematical reasoning with SymPy-based verification.

| Component | Status | Description |
|-----------|--------|-------------|
| ReasoningRAFTTrainer | Beta | RAFT for reasoning |
| MathVerifier | Beta | SymPy symbolic evaluation |

### Quick Start

```bash
# Full pipeline
halo-forge reasoning sft --dataset metamath --model Qwen/Qwen2.5-3B-Instruct --output models/reasoning_sft
halo-forge reasoning train --model models/reasoning_sft --dataset gsm8k --cycles 4
halo-forge reasoning benchmark --model models/reasoning_raft --dataset gsm8k --limit 100

# Quick RAFT
halo-forge reasoning train --model Qwen/Qwen2.5-7B-Instruct --dataset gsm8k --cycles 4
```

### Datasets

| Dataset | Task | Description |
|---------|------|-------------|
| GSM8K | Grade school math | Word problems |
| MATH | Competition math | Harder problems |
| MetaMathQA | SFT data | Large scale training |

### Reward Structure

| Outcome | Reward |
|---------|--------|
| Correct answer | 1.0 |
| Wrong + showed work | 0.2 |
| No answer + work | 0.2 |
| No answer, no work | 0.1 |

---

## Agentic / Tool Calling

Train models for reliable function/tool calling with schema-aware verification.

| Component | Status | Description |
|-----------|--------|-------------|
| AgenticRAFTTrainer | Beta | RAFT for tool calling |
| ToolCallingVerifier | Beta | JSON schema validation |
| HermesFormatter | Beta | Hermes chat format |

### Quick Start

```bash
# Full pipeline
halo-forge agentic sft --dataset xlam_sft --model Qwen/Qwen2.5-7B-Instruct --output models/agentic_sft
halo-forge agentic train --model models/agentic_sft --dataset xlam --cycles 5
halo-forge agentic benchmark --model models/agentic_raft --dataset xlam --limit 100

# Quick RAFT
halo-forge agentic train --model Qwen/Qwen2.5-7B-Instruct --dataset xlam --cycles 5
```

### Reward Structure

| Outcome | Reward |
|---------|--------|
| Correct function + args | 1.0 |
| Correct function, wrong args | 0.5 |
| Valid JSON, wrong function | 0.25 |
| No tool call when expected | 0.0 |
| Called when shouldn't | -0.25 |

### Data Format

halo-forge uses the **Hermes format**, standard for Qwen2.5 and NousHermes:

```
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris"}}
</tool_call><|im_end|>
```

---

## Inference Optimization

Optimize trained models for deployment with quantization and export.

| Feature | Description | Command |
|---------|-------------|---------|
| Quantization | INT4/INT8 reduction | `halo-forge inference optimize` |
| GGUF Export | For llama.cpp/Ollama | `halo-forge inference export --format gguf` |
| ONNX Export | Cross-platform | `halo-forge inference export --format onnx` |
| Benchmarking | Measure latency | `halo-forge inference benchmark` |

### Quick Start

```bash
# Optimize
halo-forge inference optimize \
  --model models/raft/final \
  --target-precision int4 \
  --output models/optimized

# Export to GGUF
halo-forge inference export \
  --model models/optimized \
  --format gguf \
  --quantization Q4_K_M \
  --output model.gguf

# Benchmark
halo-forge inference benchmark --model models/optimized --num-prompts 20
```

### Quantization Options

| Precision | Size Reduction | Quality |
|-----------|----------------|---------|
| int4 | ~75% | Good |
| int8 | ~50% | Better |
| fp16 | ~50% | Best |

### GGUF Quantization Types

| Type | Description |
|------|-------------|
| Q4_K_M | Recommended balance |
| Q4_K_S | Smaller, lower quality |
| Q8_0 | Highest quality 8-bit |
| F16 | No quantization |

---

## Stability Levels

| Level | Meaning |
|-------|---------|
| **Alpha** | Early development, API may change significantly |
| **Beta** | Feature complete, API mostly stable |
| **Stable** | Production ready, in main documentation |

---

## Feedback

If you encounter issues or have suggestions:

1. Check the [Troubleshooting](/docs/reference/troubleshooting/) guide
2. Review the [Changelog](/docs/changelog/) for recent changes
3. Open an issue on GitHub with reproduction steps
