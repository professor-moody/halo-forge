"""
Audio Dataset Loaders

Load audio datasets from HuggingFace for RLVR training.

Note: We use torchaudio directly for audio decoding instead of relying on
the datasets library's built-in decoder. This avoids the torchcodec dependency
which is incompatible with PyTorch 2.11.0a0+rocm.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
import json
import logging
import tempfile
import io
import numpy as np

logger = logging.getLogger(__name__)


def _decode_audio_with_librosa(audio_bytes: bytes, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Decode audio bytes using librosa.
    
    We use librosa directly for BytesIO loading because torchaudio.load() 
    with BytesIO requires torchcodec, which is incompatible with 
    PyTorch 2.11.0a0+rocm nightlies.
    
    Args:
        audio_bytes: Raw audio file bytes
        sample_rate: Target sample rate
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa required: pip install librosa")
    
    audio_buffer = io.BytesIO(audio_bytes)
    audio_array, sr = librosa.load(audio_buffer, sr=sample_rate)
    
    return audio_array, sr


def _decode_audio_from_path(audio_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Decode audio from file path using torchaudio.
    
    torchaudio works fine for file paths (no torchcodec needed).
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        import torchaudio
    except ImportError:
        # Fallback to librosa for file paths too
        import librosa
        audio_array, sr = librosa.load(audio_path, sr=sample_rate)
        return audio_array, sr
    
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        sr = sample_rate
    
    # Convert to numpy array (flatten to 1D)
    audio_array = waveform.squeeze().numpy()
    
    return audio_array, sr


def decode_audio(audio_data: dict, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Decode audio from HuggingFace dataset format.
    
    Handles both decoded (array+sampling_rate) and raw (bytes) formats.
    Uses librosa for bytes and torchaudio for file paths.
    
    Args:
        audio_data: Audio dict from HuggingFace dataset
        target_sr: Target sample rate
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    # If already decoded (has 'array' key), return directly
    if isinstance(audio_data, dict) and 'array' in audio_data:
        return np.array(audio_data['array']), audio_data.get('sampling_rate', target_sr)
    
    # If has a file path, use torchaudio (works without torchcodec)
    if isinstance(audio_data, dict) and 'path' in audio_data and audio_data['path']:
        path = audio_data['path']
        if Path(path).exists():
            return _decode_audio_from_path(path, target_sr)
    
    # If raw bytes, use librosa (torchaudio BytesIO requires torchcodec)
    if isinstance(audio_data, dict) and 'bytes' in audio_data:
        audio_bytes = audio_data['bytes']
    elif isinstance(audio_data, bytes):
        audio_bytes = audio_data
    else:
        raise ValueError(f"Unsupported audio format: {type(audio_data)}")
    
    return _decode_audio_with_librosa(audio_bytes, target_sr)


@dataclass
class AudioSample:
    """A single audio training sample."""
    
    audio_path: str  # Path to audio file
    text: str  # Transcript or label
    duration: float  # Duration in seconds
    task: str = "asr"  # asr, tts, classification
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # For in-memory audio (from HuggingFace)
    audio_array: Optional[Any] = None
    sample_rate: Optional[int] = None


class AudioDataset(ABC):
    """Abstract base class for audio datasets."""
    
    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            split: Dataset split (train, validation, test)
            cache_dir: Directory for caching
            limit: Limit number of samples
        """
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir else Path("~/.cache/halo_forge/audio").expanduser()
        self.limit = limit
        self.samples: List[AudioSample] = []
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        pass
    
    @property
    @abstractmethod
    def task(self) -> str:
        """Primary task (asr, tts, classification)."""
        pass
    
    @abstractmethod
    def load(self) -> List[AudioSample]:
        """Load dataset samples."""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self) -> Iterator[AudioSample]:
        return iter(self.samples)
    
    def __getitem__(self, idx: int) -> AudioSample:
        return self.samples[idx]
    
    def to_rlvr_format(self, output_path: str) -> None:
        """
        Export to RLVR JSONL format.
        
        Args:
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            for sample in self.samples:
                record = {
                    'audio_path': sample.audio_path,
                    'text': sample.text,
                    'duration': sample.duration,
                    'task': sample.task,
                    'metadata': sample.metadata,
                }
                f.write(json.dumps(record) + '\n')
        
        logger.info(f"Exported {len(self.samples)} samples to {output_path}")


class LocalAudioManifestLoader(AudioDataset):
    """Load Dataset Lab's canonical audio JSONL without a registry alias.

    Asset paths are resolved relative to the manifest.  Dataset Lab training
    artifacts normally make them absolute, but accepting relative paths keeps
    direct CLI usage portable as well.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        super().__init__(split=split, cache_dir=cache_dir, limit=limit)
        self.manifest_path = Path(manifest_path).expanduser().resolve()

    @property
    def name(self) -> str:
        return self.manifest_path.stem

    @property
    def task(self) -> str:
        return "mixed"

    def load(self) -> List[AudioSample]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Audio manifest not found: {self.manifest_path}")
        samples: List[AudioSample] = []
        with self.manifest_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if self.limit is not None and len(samples) >= self.limit:
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid audio manifest JSON on line {index + 1}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Audio manifest line {index + 1} must be an object")
                raw_audio = row.get("audio", row.get("audio_path"))
                if isinstance(raw_audio, dict):
                    raw_audio = raw_audio.get("path")
                if not isinstance(raw_audio, str) or not raw_audio.strip():
                    raise ValueError(f"Audio manifest line {index + 1} is missing audio")
                audio_path = Path(raw_audio).expanduser()
                if not audio_path.is_absolute():
                    audio_path = (self.manifest_path.parent / audio_path).resolve()
                text = row.get("transcript", row.get("label", row.get("text")))
                if text is None:
                    raise ValueError(
                        f"Audio manifest line {index + 1} requires transcript, label, or text"
                    )
                metadata = row.get("metadata")
                metadata = dict(metadata) if isinstance(metadata, dict) else {}
                metadata.setdefault("manifest_index", index)
                metadata.setdefault("manifest", str(self.manifest_path))
                samples.append(
                    AudioSample(
                        audio_path=str(audio_path),
                        text=str(text),
                        duration=float(row.get("duration", row.get("duration_seconds", 0.0)) or 0.0),
                        task=str(row.get("task") or "asr"),
                        metadata=metadata,
                        sample_rate=(
                            int(row["sample_rate"])
                            if row.get("sample_rate") is not None
                            else None
                        ),
                    )
                )
        if not samples:
            raise ValueError(f"Audio manifest contains no usable records: {self.manifest_path}")
        self.samples = samples
        logger.info("Loaded %d local audio samples from %s", len(samples), self.manifest_path)
        return samples


class LibriSpeechLoader(AudioDataset):
    """Load LibriSpeech dataset for ASR training."""
    
    @property
    def name(self) -> str:
        return "librispeech"
    
    @property
    def task(self) -> str:
        return "asr"
    
    def load(self) -> List[AudioSample]:
        """Load LibriSpeech samples using torchaudio for decoding."""
        try:
            from datasets import load_dataset, Audio
        except ImportError:
            raise ImportError("datasets library required: pip install datasets")
        
        # Map split names
        split_map = {
            "train": "train.clean.100",
            "validation": "validation.clean",
            "test": "test.clean",
        }
        hf_split = split_map.get(self.split, self.split)
        
        logger.info(f"Loading LibriSpeech {hf_split}...")
        
        # Load with audio decoding DISABLED to avoid torchcodec dependency
        # We'll decode audio ourselves using torchaudio
        dataset = load_dataset("librispeech_asr", split=hf_split, trust_remote_code=True)
        
        # Cast audio column to disable decoding - returns raw bytes
        dataset = dataset.cast_column("audio", Audio(decode=False))
        
        samples = []
        for i, item in enumerate(dataset):
            if self.limit and i >= self.limit:
                break
            
            try:
                # Decode audio using torchaudio (our custom decoder)
                audio_data = item["audio"]
                audio_array, sample_rate = decode_audio(audio_data, target_sr=16000)
                duration = len(audio_array) / sample_rate
                
                samples.append(AudioSample(
                    audio_path=audio_data.get("path", f"librispeech_{i}"),
                    text=item["text"],
                    duration=duration,
                    task="asr",
                    metadata={
                        "speaker_id": item["speaker_id"],
                        "chapter_id": item["chapter_id"],
                        "dataset": "librispeech",
                    },
                    audio_array=audio_array,
                    sample_rate=sample_rate,
                ))
            except Exception as e:
                logger.warning(f"Failed to decode sample {i}: {e}")
                continue
        
        self.samples = samples
        logger.info(f"Loaded {len(samples)} LibriSpeech samples")
        return samples


class CommonVoiceLoader(AudioDataset):
    """Load Mozilla Common Voice dataset."""
    
    def __init__(
        self,
        language: str = "en",
        split: str = "train",
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        super().__init__(split, cache_dir, limit)
        self.language = language
    
    @property
    def name(self) -> str:
        return "common_voice"
    
    @property
    def task(self) -> str:
        return "asr"
    
    def load(self) -> List[AudioSample]:
        """Load Common Voice samples using torchaudio for decoding."""
        try:
            from datasets import load_dataset, Audio
        except ImportError:
            raise ImportError("datasets library required: pip install datasets")
        
        logger.info(f"Loading Common Voice ({self.language}) {self.split}...")
        dataset = load_dataset(
            "mozilla-foundation/common_voice_11_0",
            self.language,
            split=self.split,
            trust_remote_code=True,
        )
        
        # Disable built-in audio decoding
        dataset = dataset.cast_column("audio", Audio(decode=False))
        
        samples = []
        for i, item in enumerate(dataset):
            if self.limit and i >= self.limit:
                break
            
            try:
                audio_data = item["audio"]
                audio_array, sample_rate = decode_audio(audio_data, target_sr=16000)
                duration = len(audio_array) / sample_rate
                
                samples.append(AudioSample(
                    audio_path=audio_data.get("path", f"cv_{i}"),
                    text=item["sentence"],
                    duration=duration,
                    task="asr",
                    metadata={
                        "language": self.language,
                        "gender": item.get("gender"),
                        "age": item.get("age"),
                        "dataset": "common_voice",
                    },
                    audio_array=audio_array,
                    sample_rate=sample_rate,
                ))
            except Exception as e:
                logger.warning(f"Failed to decode sample {i}: {e}")
                continue
        
        self.samples = samples
        logger.info(f"Loaded {len(samples)} Common Voice samples")
        return samples


class AudioSetLoader(AudioDataset):
    """Load AudioSet for classification training."""
    
    @property
    def name(self) -> str:
        return "audioset"
    
    @property
    def task(self) -> str:
        return "classification"
    
    def load(self) -> List[AudioSample]:
        """Load AudioSet samples using torchaudio for decoding."""
        try:
            from datasets import load_dataset, Audio
        except ImportError:
            raise ImportError("datasets library required: pip install datasets")
        
        logger.info(f"Loading AudioSet {self.split}...")
        
        # Use a smaller AudioSet variant
        dataset = load_dataset(
            "agkphysics/AudioSet",
            "balanced",
            split=self.split,
            trust_remote_code=True,
        )
        
        # Disable built-in audio decoding
        dataset = dataset.cast_column("audio", Audio(decode=False))
        
        samples = []
        for i, item in enumerate(dataset):
            if self.limit and i >= self.limit:
                break
            
            try:
                audio_data = item["audio"]
                audio_array, sample_rate = decode_audio(audio_data, target_sr=16000)
                duration = len(audio_array) / sample_rate
                
                # AudioSet has multiple labels
                labels = item.get("human_labels", item.get("labels", []))
                label_str = labels[0] if isinstance(labels, list) and labels else str(labels)
                
                samples.append(AudioSample(
                    audio_path=audio_data.get("path", f"audioset_{i}"),
                    text=label_str,
                    duration=duration,
                    task="classification",
                    metadata={
                        "all_labels": labels,
                        "dataset": "audioset",
                    },
                    audio_array=audio_array,
                    sample_rate=sample_rate,
                ))
            except Exception as e:
                logger.warning(f"Failed to decode sample {i}: {e}")
                continue
        
        self.samples = samples
        logger.info(f"Loaded {len(samples)} AudioSet samples")
        return samples


class SpeechCommandsLoader(AudioDataset):
    """Load Speech Commands dataset for keyword classification."""
    
    @property
    def name(self) -> str:
        return "speech_commands"
    
    @property
    def task(self) -> str:
        return "classification"
    
    def load(self) -> List[AudioSample]:
        """Load Speech Commands samples using torchaudio for decoding."""
        try:
            from datasets import load_dataset, Audio
        except ImportError:
            raise ImportError("datasets library required: pip install datasets")
        
        logger.info(f"Loading Speech Commands {self.split}...")
        dataset = load_dataset(
            "speech_commands",
            "v0.02",
            split=self.split,
            trust_remote_code=True,
        )
        
        # Disable built-in audio decoding
        dataset = dataset.cast_column("audio", Audio(decode=False))
        
        samples = []
        for i, item in enumerate(dataset):
            if self.limit and i >= self.limit:
                break
            
            try:
                audio_data = item["audio"]
                audio_array, sample_rate = decode_audio(audio_data, target_sr=16000)
                duration = len(audio_array) / sample_rate
                
                samples.append(AudioSample(
                    audio_path=audio_data.get("path", f"sc_{i}"),
                    text=item["label"],  # Command label (e.g., "yes", "no", "up")
                    duration=duration,
                    task="classification",
                    metadata={
                        "speaker_id": item.get("speaker_id"),
                        "is_unknown": item.get("is_unknown", False),
                        "dataset": "speech_commands",
                    },
                    audio_array=audio_array,
                    sample_rate=sample_rate,
                ))
            except Exception as e:
                logger.warning(f"Failed to decode sample {i}: {e}")
                continue
        
        self.samples = samples
        logger.info(f"Loaded {len(samples)} Speech Commands samples")
        return samples


# Dataset registry
AUDIO_DATASETS = {
    "librispeech": LibriSpeechLoader,
    "common_voice": CommonVoiceLoader,
    "audioset": AudioSetLoader,
    "speech_commands": SpeechCommandsLoader,
}

AUDIO_DATASET_ALIASES = {
    "commonvoice": "common_voice",
}


def load_audio_dataset(
    name: str,
    split: str = "train",
    limit: Optional[int] = None,
    **kwargs
) -> AudioDataset:
    """
    Load an audio dataset by name.
    
    Args:
        name: Dataset name
        split: Dataset split
        limit: Limit number of samples
        **kwargs: Additional arguments for dataset
        
    Returns:
        AudioDataset instance
    """
    local_path = Path(name).expanduser()
    if local_path.is_file():
        loader = LocalAudioManifestLoader(local_path, split=split, limit=limit, **kwargs)
        loader.load()
        return loader

    canonical_name = AUDIO_DATASET_ALIASES.get(name.lower(), name.lower())
    if canonical_name != name:
        logger.warning("Audio dataset alias '%s' normalized to '%s'", name, canonical_name)

    if canonical_name not in AUDIO_DATASETS:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: {list(AUDIO_DATASETS.keys())}. "
            f"Aliases: {list(AUDIO_DATASET_ALIASES.keys())}"
        )
    
    loader_cls = AUDIO_DATASETS[canonical_name]
    loader = loader_cls(split=split, limit=limit, **kwargs)
    loader.load()
    
    return loader


def list_audio_datasets() -> List[str]:
    """List available audio datasets."""
    return list(AUDIO_DATASETS.keys())
