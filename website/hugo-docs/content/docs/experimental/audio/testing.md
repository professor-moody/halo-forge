---
title: "Testing Guide"
weight: 50
---

# Audio Module Testing Guide

Step-by-step guide to testing the halo-forge Audio training module.

## Prerequisites

### Required Dependencies

```bash
pip install torch transformers torchaudio
```

### Optional Dependencies

| Dependency | Required For | Install |
|------------|--------------|---------|
| `torchaudio` | Audio processing | `pip install torchaudio` |
| `librosa` | Feature extraction | `pip install librosa` |
| `jiwer` | WER calculation | `pip install jiwer` |
| `speechbrain` | Speaker embeddings | `pip install speechbrain` |

### Hardware Requirements

- **Minimum**: 8 GB RAM, 4 GB VRAM (for whisper-tiny)
- **Recommended**: 16 GB RAM, 8 GB VRAM (for whisper-small)
- **ROCm Support**: AMD Radeon RX 7900 series or higher

---

## Unit Tests

Run the full audio test suite:

```bash
# From project root
cd /path/to/halo-forge

# Run all audio tests
pytest tests/test_audio.py -v

# Run specific test class
pytest tests/test_audio.py::TestASRChecker -v
pytest tests/test_audio.py::TestAudioRAFTTrainer -v
```

### Expected Test Coverage

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestAudioSample` | 3 | AudioSample dataclass |
| `TestAudioProcessor` | 3 | Audio loading and processing |
| `TestASRChecker` | 5 | WER/CER verification |
| `TestAudioClassificationChecker` | 4 | Classification verification |
| `TestTTSChecker` | 3 | TTS quality verification |
| `TestAudioVerifier` | 3 | Multi-task verifier |
| `TestWhisperAdapter` | 2 | Whisper model adapter |
| `TestAudioRAFTTrainer` | 2 | Trainer configuration |
| `TestIntegration` | 3 | End-to-end tests |

---

## Dry Run Testing

Test commands without actual training:

```bash
# Test CLI parsing and config validation
halo-forge audio train \
  --model openai/whisper-small \
  --dataset librispeech \
  --cycles 3 \
  --output /tmp/audio-test \
  --dry-run

# Expected output:
# Dry run mode - validating configuration only
# 
# Dependencies:
#   ✓ torchaudio
#   ✓ jiwer
# 
# ✓ Dataset: librispeech
# 
# Configuration validated successfully.
```

---

## Manual Testing Steps

### 1. Test Dataset Loading

```bash
# List available datasets
halo-forge audio datasets

# Expected output:
# Available Audio Datasets
# ============================================================
#   librispeech        [ASR           ] - Clean audiobook speech (960h)
#   common_voice       [ASR           ] - Crowdsourced multilingual (2000h+)
#   audioset           [Classification] - Sound event detection (5M clips)
#   speech_commands    [Classification] - Keyword spotting (105k)
```

Test loading in Python:

```python
from halo_forge.audio.data import load_audio_dataset

# Load small subset
dataset = load_audio_dataset("librispeech", limit=10)
print(f"Loaded {len(dataset)} samples")

for sample in dataset:
    print(f"  Text: {sample.text[:50]}...")
    print(f"  Duration: {sample.duration:.2f}s")
```

### 2. Test ASR Verification

```python
from halo_forge.audio.verifiers.asr import ASRChecker

checker = ASRChecker(wer_threshold=0.3)

# Test perfect match
result = checker.verify("hello world", "hello world")
print(f"Perfect match - WER: {result.wer:.1%}, Reward: {result.reward:.2f}")

# Test partial match
result = checker.verify("hello word", "hello world")
print(f"Partial match - WER: {result.wer:.1%}, Reward: {result.reward:.2f}")

# Test mismatch
result = checker.verify("foo bar", "hello world")
print(f"Mismatch - WER: {result.wer:.1%}, Reward: {result.reward:.2f}")
```

### 3. Test Classification Verification

```python
from halo_forge.audio.verifiers.classification import AudioClassificationChecker

checker = AudioClassificationChecker()

# Correct classification
result = checker.verify("dog", "dog")
print(f"Correct - Reward: {result.reward}")

# Wrong classification
result = checker.verify("cat", "dog")
print(f"Wrong - Reward: {result.reward}")

# With label aliases
checker_alias = AudioClassificationChecker(
    label_aliases={"dog": ["canine", "puppy"]}
)
result = checker_alias.verify("canine", "dog")
print(f"Alias match - Reward: {result.reward}")
```

### 4. Test Audio Benchmark

```bash
# Benchmark Whisper on LibriSpeech
halo-forge audio benchmark \
  --model openai/whisper-tiny \
  --dataset librispeech \
  --limit 20

# Expected output:
# Audio Benchmark
# ============================================================
# Model: openai/whisper-tiny
# Dataset: librispeech
# Task: asr
# Limit: 20
# 
# Results:
#   Samples: 20
#   Success rate: XX.X%
#   Average reward: X.XXX
#   Average WER: XX.X%
```

### 5. Test Model Adapters

```python
from halo_forge.audio.models import get_audio_adapter
import numpy as np

# Create adapter
adapter = get_audio_adapter("openai/whisper-tiny")

# Load model (this downloads the model)
adapter.load()

# Test transcription
audio = np.random.randn(16000) * 0.1  # 1 second noise
result = adapter.transcribe(audio)
print(f"Transcription: {result.text}")
```

---

## Validation Checklist

After running tests, verify:

- [ ] All unit tests pass (`pytest tests/test_audio.py`)
- [ ] Dry run mode validates configuration
- [ ] Dataset listing works (`halo-forge audio datasets`)
- [ ] ASRChecker calculates WER correctly
- [ ] ClassificationChecker matches labels
- [ ] Model adapters load and transcribe
- [ ] Benchmark runs on small dataset

---

## Troubleshooting

### "torchaudio not installed"

This is required for audio processing:

```bash
pip install torchaudio
```

For ROCm, ensure compatible version:

```bash
pip install torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

### "jiwer not installed"

Required for accurate WER calculation:

```bash
pip install jiwer
```

Without it, fallback Levenshtein distance is used.

### Model Download Errors

Whisper models are downloaded from HuggingFace:

```bash
# Check HuggingFace login
huggingface-cli login

# Or set token
export HF_TOKEN=your_token_here
```

### Out of Memory (OOM)

- Use smaller models (`whisper-tiny` instead of `whisper-small`)
- Reduce `--limit` for testing
- Use CPU with `--device cpu`

### Audio Loading Errors

Check audio file format:

```python
import torchaudio

# Check supported formats
print(torchaudio.list_audio_backends())

# Try loading
waveform, sr = torchaudio.load("test.wav")
print(f"Sample rate: {sr}, Shape: {waveform.shape}")
```

### ROCm-specific Issues

Set environment variables:

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export HSA_OVERRIDE_GFX_VERSION=11.5.1  # For Strix Halo
```

---

## Next Steps

- [Audio Training](./) - Full training documentation
- [Audio Datasets](./datasets.md) - Available datasets
- [Inference Testing](../inference/testing.md) - Test inference module
- [VLM Testing](../vlm/testing.md) - Test VLM module
