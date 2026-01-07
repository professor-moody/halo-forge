---
title: "Audio Datasets"
weight: 20
---

# Audio Datasets

Guide to obtaining and using datasets for audio-language model training.

## Available Datasets

List available audio datasets:

```bash
halo-forge audio datasets
```

| Dataset | HuggingFace Path | Task | Size |
|---------|------------------|------|------|
| `librispeech` | `librispeech_asr` | ASR | 960h clean speech |
| `common_voice` | `mozilla-foundation/common_voice_11_0` | ASR | 2000h+ multilingual |
| `audioset` | `agkphysics/AudioSet` | Classification | 5M+ clips |
| `speech_commands` | `speech_commands` | Classification | 105k keyword clips |

---

## ASR Datasets

### LibriSpeech

Clean audiobook speech for ASR training. The gold standard for speech recognition.

```bash
halo-forge audio benchmark \
  --model openai/whisper-small \
  --dataset librispeech \
  --limit 100
```

**Splits:**
- `train.clean.100` - 100 hours clean speech
- `train.clean.360` - 360 hours clean speech
- `train.other.500` - 500 hours "other" quality
- `validation.clean` - Validation set
- `test.clean` - Test set

**Use Case:** Primary ASR training and evaluation

### Common Voice

Mozilla's crowdsourced multilingual speech dataset.

```python
from halo_forge.audio.data.loaders import CommonVoiceLoader

# Load English
loader = CommonVoiceLoader(language="en", split="train", limit=1000)
samples = loader.load()

# Load French
loader_fr = CommonVoiceLoader(language="fr", split="train", limit=500)
```

**Languages:** 100+ languages available
**Use Case:** Multilingual ASR, accent training

---

## Classification Datasets

### Speech Commands

Google's keyword spotting dataset. 35 short spoken words.

```bash
halo-forge audio benchmark \
  --model custom/classifier \
  --dataset speech_commands \
  --task classification
```

**Labels:**
```
yes, no, up, down, left, right, on, off, stop, go,
zero, one, two, three, four, five, six, seven, eight, nine
```

**Use Case:** Wake word detection, command recognition

### AudioSet

YouTube audio clips with 632 audio event categories.

```python
from halo_forge.audio.data.loaders import AudioSetLoader

loader = AudioSetLoader(split="train", limit=500)
samples = loader.load()

# Each sample has multiple labels
for sample in samples:
    print(f"Labels: {sample.metadata['all_labels']}")
```

**Categories:** Speech, Music, Animals, Vehicles, Nature, etc.
**Use Case:** Sound event detection, audio tagging

---

## AudioSample Format

Each sample contains:

```python
@dataclass
class AudioSample:
    audio_path: str           # Path to audio file
    text: str                 # Transcript or label
    duration: float           # Duration in seconds
    task: str = "asr"         # asr, tts, classification
    metadata: Dict            # Additional info
    
    # For in-memory audio
    audio_array: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None
```

### Example

```python
sample = AudioSample(
    audio_path="speech.wav",
    text="hello world",
    duration=1.5,
    task="asr",
    metadata={"speaker_id": 123}
)
```

---

## Using Dataset Loaders

### Load Programmatically

```python
from halo_forge.audio.data import load_audio_dataset, list_audio_datasets

# List available datasets
print(list_audio_datasets())
# ['librispeech', 'common_voice', 'audioset', 'speech_commands']

# Load LibriSpeech
dataset = load_audio_dataset("librispeech", split="train", limit=1000)

# Iterate samples
for sample in dataset:
    print(f"Text: {sample.text}")
    print(f"Duration: {sample.duration:.2f}s")
    
    # Access audio array
    if sample.audio_array is not None:
        audio = sample.audio_array
        sr = sample.sample_rate
```

### Export to RLVR Format

```python
dataset = load_audio_dataset("librispeech", limit=1000)

# Export to JSONL
dataset.to_rlvr_format("audio_rlvr.jsonl")
```

Output format:

```json
{
  "audio_path": "/path/to/audio.wav",
  "text": "hello world",
  "duration": 2.5,
  "task": "asr",
  "metadata": {"speaker_id": 123}
}
```

---

## Creating Custom Audio Datasets

### From Local Audio Files

```python
from halo_forge.audio.data import AudioSample, AudioDataset
from pathlib import Path
import json

class CustomASRDataset(AudioDataset):
    @property
    def name(self) -> str:
        return "custom"
    
    @property
    def task(self) -> str:
        return "asr"
    
    def load(self):
        # Load from annotation file
        with open("transcripts.json") as f:
            data = json.load(f)
        
        self.samples = [
            AudioSample(
                audio_path=f"audio/{item['id']}.wav",
                text=item["transcript"],
                duration=item["duration"],
                metadata={"id": item["id"]}
            )
            for item in data
        ]
        return self.samples
```

### From HuggingFace

```python
from datasets import load_dataset
from halo_forge.audio.data import AudioSample
import json

# Load any speech dataset
hf_dataset = load_dataset("your_org/speech_data", split="train")

# Convert to AudioSample format
samples = []
for item in hf_dataset:
    samples.append(AudioSample(
        audio_path=item["audio"]["path"],
        text=item["text"],
        duration=len(item["audio"]["array"]) / item["audio"]["sampling_rate"],
        audio_array=item["audio"]["array"],
        sample_rate=item["audio"]["sampling_rate"],
    ))
```

---

## HuggingFace Sources

### Recommended Datasets

| Dataset | HuggingFace Path | Description |
|---------|------------------|-------------|
| LibriSpeech | `librispeech_asr` | Clean audiobook speech |
| Common Voice | `mozilla-foundation/common_voice_11_0` | Multilingual speech |
| FLEURS | `google/fleurs` | 102 language evaluation |
| VoxCeleb | `voxceleb/voxceleb1` | Speaker verification |
| AudioSet | `agkphysics/AudioSet` | Sound events |
| ESC-50 | `ashraq/esc50` | Environmental sounds |
| UrbanSound8K | `UrbanSound8K` | Urban sounds |

### Loading Directly

```python
from datasets import load_dataset

# Load FLEURS (multilingual)
fleurs = load_dataset("google/fleurs", "en_us", split="train")

# Load VoxCeleb (speaker)
vox = load_dataset("voxceleb/voxceleb1", split="train")

# Access audio
for item in fleurs:
    audio = item["audio"]["array"]
    sr = item["audio"]["sampling_rate"]
    text = item["transcription"]
```

---

## Audio Processing

### AudioProcessor

```python
from halo_forge.audio.data import AudioProcessor

processor = AudioProcessor(
    sample_rate=16000,
    normalize=True,
    mono=True,
    max_duration=30.0,  # Truncate to 30s
)

# Load from file
result = processor.load("speech.wav")
print(f"Duration: {result.duration:.2f}s")
print(f"Waveform shape: {result.waveform.shape}")

# Load from array
import numpy as np
audio = np.random.randn(16000)  # 1 second
result = processor.load_array(audio, original_sr=16000)
```

### Feature Extraction

```python
# Extract mel spectrogram
processor = AudioProcessor(feature_type="mel")
result = processor.load("speech.wav")
mel = result.features  # Shape: [1, n_mels, time]

# Extract MFCC
processor = AudioProcessor(feature_type="mfcc")
result = processor.load("speech.wav")
mfcc = result.features  # Shape: [1, n_mfcc, time]
```

---

## Memory Considerations

Audio datasets can be large:

| Dataset | Disk Size | Memory (1000 samples) |
|---------|-----------|----------------------|
| LibriSpeech-100h | ~6 GB | ~2 GB |
| LibriSpeech-960h | ~60 GB | ~2 GB |
| Common Voice EN | ~80 GB | ~2 GB |
| AudioSet | ~500+ GB | ~3 GB |

### Tips

1. Use `limit` parameter for testing
2. Stream audio instead of loading all at once
3. Use smaller sample rates for development (8kHz vs 16kHz)
4. Process in batches to manage memory

---

## Next Steps

- [Audio Training](./) - Train audio models
- [Audio Testing](./testing.md) - Test audio features
- [Code Datasets](../../training-pipeline/datasets-code/) - Code generation data
- [VLM Datasets](../vlm/datasets.md) - Vision-language data
