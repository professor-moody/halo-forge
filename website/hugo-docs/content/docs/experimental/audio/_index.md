---
title: "Audio Training"
weight: 30
---

> **⚠️ Experimental Feature**: Audio training is under active development. APIs may change.

# Audio-Language Model Training

Train audio-language models (ASR, TTS, Classification) using RLVR with task-specific verification.

## Overview

Phase 4 of halo-forge extends the RAFT training framework to support audio-language models. This enables training models like Whisper and Wav2Vec2 on speech tasks with graduated reward signals.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Audio RAFT Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │
│  │  Audio   │───▶│  Model   │───▶│     Prediction       │  │
│  │  Sample  │    │ (Whisper)│    │  (Transcription)     │  │
│  └──────────┘    └──────────┘    └──────────┬───────────┘  │
│                                              │               │
│                                              ▼               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Task-Specific Verification               │  │
│  ├──────────────┬──────────────┬────────────────────────┤  │
│  │     ASR      │     TTS      │    Classification      │  │
│  │              │              │                        │  │
│  │ • WER        │ • Intelli-   │ • Exact match          │  │
│  │ • CER        │   gibility   │ • Fuzzy match          │  │
│  │              │ • Quality    │ • Label aliases        │  │
│  │              │ • Consistency│                        │  │
│  └──────────────┴──────────────┴────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│                      Reward                                 │
│                    (0.0 - 1.0)                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### List Available Datasets

```bash
halo-forge audio datasets
```

### Benchmark a Model

```bash
halo-forge audio benchmark \
    --model openai/whisper-small \
    --dataset librispeech \
    --task asr \
    --limit 100
```

### Train with RAFT

```bash
halo-forge audio train \
    --model openai/whisper-small \
    --dataset librispeech \
    --task asr \
    --cycles 4 \
    --output models/audio_raft
```

## Supported Models

| Model | Adapter | Task | Notes |
|-------|---------|------|-------|
| openai/whisper-tiny | WhisperAdapter | ASR | Fast, lightweight |
| openai/whisper-small | WhisperAdapter | ASR | Recommended |
| openai/whisper-medium | WhisperAdapter | ASR | Better accuracy |
| openai/whisper-large-v3 | WhisperAdapter | ASR | Best quality |
| facebook/wav2vec2-base-960h | Wav2VecAdapter | ASR | English only |

## Supported Tasks

### ASR (Automatic Speech Recognition)

Convert speech to text. Verified using Word Error Rate (WER).

```bash
halo-forge audio train \
    --model openai/whisper-small \
    --dataset librispeech \
    --task asr
```

**Reward Structure:**
- WER 0%: reward = 1.0 (perfect)
- WER 10%: reward = 0.9
- WER 30%: reward = 0.7
- WER 50%: reward = 0.5

### Classification

Classify audio into categories. Verified using exact match.

```bash
halo-forge audio train \
    --model custom/classifier \
    --dataset speech_commands \
    --task classification
```

**Reward Structure:**
- Correct: reward = 1.0
- Incorrect: reward = 0.0

### TTS (Text-to-Speech)

Evaluate synthesized speech quality. Verified using intelligibility, quality, and consistency.

**Reward Structure:**
- Intelligibility (0.4): ASR on generated audio, compare to target
- Quality (0.4): Audio quality metrics (SNR, clipping, dynamics)
- Consistency (0.2): Speaker similarity (if reference provided)

## Configuration

```python
from halo_forge.audio import AudioRAFTTrainer, AudioRAFTConfig

config = AudioRAFTConfig(
    model_name="openai/whisper-small",
    task="asr",
    num_cycles=6,
    samples_per_prompt=4,
    
    # Learning rate
    learning_rate=5e-5,
    lr_decay_per_cycle=0.85,
    
    # Verification
    wer_threshold=0.3,
    
    # Output
    output_dir="models/audio_raft",
)

trainer = AudioRAFTTrainer(config)
trainer.train("librispeech")
```

## Memory Requirements

| Model | Training Memory | Inference Memory |
|-------|-----------------|------------------|
| whisper-tiny | ~4 GB | ~1 GB |
| whisper-small | ~8 GB | ~2 GB |
| whisper-medium | ~16 GB | ~5 GB |
| whisper-large-v3 | ~32 GB+ | ~10 GB |

## Dependencies

```bash
pip install torchaudio librosa jiwer
```

Optional for TTS quality metrics:
```bash
pip install speechbrain  # Speaker embeddings
```

## Next Steps

- [Audio Datasets](./datasets.md) - Available audio datasets
- [Audio Testing](./testing.md) - Test audio features
- [VLM Training](../vlm/) - Vision-language training
