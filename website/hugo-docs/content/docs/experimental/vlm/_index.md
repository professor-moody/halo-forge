---
title: "VLM Training"
weight: 10
---

> **⚠️ Experimental Feature**: VLM training is under active development. APIs may change.

# Vision-Language Model Training

Train vision-language models (VLMs) using RLVR with perception-aware verification.

## Overview

Phase 3 of halo-forge extends the RAFT training framework to support vision-language models. This enables training models like Qwen-VL and LLaVA on visual question answering tasks with graduated reward signals.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VLM RAFT Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │
│  │  Image   │───▶│   VLM    │───▶│     Completion       │  │
│  │  Prompt  │    │  Model   │    │                      │  │
│  └──────────┘    └──────────┘    └──────────┬───────────┘  │
│                                              │               │
│                                              ▼               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Multi-Stage Verification                 │  │
│  ├──────────────┬──────────────┬────────────────────────┤  │
│  │  Perception  │  Reasoning   │       Output           │  │
│  │   (0.3)      │   (0.4)      │       (0.3)            │  │
│  │              │              │                        │  │
│  │ • Objects    │ • Structure  │ • Exact match          │  │
│  │ • OCR        │ • Consistency│ • Fuzzy match          │  │
│  │ • Spatial    │ • Grounding  │ • Semantic sim         │  │
│  │ • Counting   │              │                        │  │
│  └──────────────┴──────────────┴────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│                   Combined Reward                           │
│                      (0.0 - 1.0)                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Train on TextVQA

```bash
halo-forge vlm train \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --dataset textvqa \
    --cycles 6 \
    --output models/vlm_raft
```

### Benchmark a Model

```bash
halo-forge vlm benchmark \
    --model models/vlm_raft/cycle_6 \
    --dataset docvqa \
    --limit 100
```

### List Available Datasets

```bash
halo-forge vlm datasets
```

## Supported Models

| Model | Adapter | Notes |
|-------|---------|-------|
| Qwen/Qwen2-VL-2B-Instruct | qwen_vl | Lightweight |
| Qwen/Qwen2-VL-7B-Instruct | qwen_vl | Recommended |
| Qwen/Qwen2-VL-72B-Instruct | qwen_vl | Requires 128GB+ |
| llava-hf/llava-1.5-7b-hf | llava | Good baseline |
| llava-hf/llava-v1.6-34b-hf | llava | High quality |

## Supported Datasets

| Dataset | Task | Size |
|---------|------|------|
| TextVQA | Text reading in images | 45K train |
| DocVQA | Document understanding | 50K train |
| ChartQA | Chart interpretation | 28K train |
| RealWorldQA | Real-world reasoning | 700 test |
| MathVista | Math with visuals | 6K+ test |

## Key Features

### Perception Verification

The `PerceptionChecker` validates visual claims:

- **Object Detection**: Uses YOLOv8 to verify claimed objects exist
- **OCR Verification**: Uses EasyOCR to verify text extraction
- **Spatial Reasoning**: Validates "left of", "above", etc.
- **Counting**: Verifies object counts

### Reasoning Verification

The `ReasoningChecker` validates chain-of-thought quality:

- **Structure**: Proper reasoning steps
- **Consistency**: No contradictions
- **Grounding**: References to visual evidence

### Output Verification

The `OutputChecker` validates final answers:

- **Exact Match**: Direct comparison
- **Fuzzy Match**: Similarity-based
- **Semantic**: Embedding similarity (optional)

## Configuration

```python
from halo_forge.vlm.trainer import VLMRAFTConfig, VLMRAFTTrainer

config = VLMRAFTConfig(
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    num_cycles=6,
    samples_per_prompt=4,
    
    # Verification weights
    perception_weight=0.3,
    reasoning_weight=0.4,
    output_weight=0.3,
    
    # Training
    learning_rate=5e-5,
    lr_decay_per_cycle=0.85,
    temperature=0.7,
)

trainer = VLMRAFTTrainer(config)
trainer.train("textvqa")
```

## Memory Requirements

| Model Size | Training Memory | Inference Memory |
|------------|-----------------|------------------|
| 2B VLM | ~40 GB | ~8 GB |
| 7B VLM | ~75 GB | ~20 GB |
| 72B VLM | ~128 GB+ | ~50 GB |

## Next Steps

- [Perception Verification](verifiers/) - How perception checking works
- [Dataset Loaders](datasets/) - Preparing VLM datasets
- [CLI Reference](/docs/reference/cli/) - All VLM commands
