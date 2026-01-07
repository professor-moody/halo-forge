"""
Audio Dataset Loaders

Load audio datasets from HuggingFace for RLVR training.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import json
import logging
import tempfile

logger = logging.getLogger(__name__)


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


class LibriSpeechLoader(AudioDataset):
    """Load LibriSpeech dataset for ASR training."""
    
    @property
    def name(self) -> str:
        return "librispeech"
    
    @property
    def task(self) -> str:
        return "asr"
    
    def load(self) -> List[AudioSample]:
        """Load LibriSpeech samples."""
        try:
            from datasets import load_dataset
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
        dataset = load_dataset("librispeech_asr", split=hf_split)
        
        samples = []
        for i, item in enumerate(dataset):
            if self.limit and i >= self.limit:
                break
            
            # Get audio info
            audio = item["audio"]
            duration = len(audio["array"]) / audio["sampling_rate"]
            
            samples.append(AudioSample(
                audio_path=audio.get("path", f"librispeech_{i}"),
                text=item["text"],
                duration=duration,
                task="asr",
                metadata={
                    "speaker_id": item["speaker_id"],
                    "chapter_id": item["chapter_id"],
                    "dataset": "librispeech",
                },
                audio_array=audio["array"],
                sample_rate=audio["sampling_rate"],
            ))
        
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
        """Load Common Voice samples."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required: pip install datasets")
        
        logger.info(f"Loading Common Voice ({self.language}) {self.split}...")
        dataset = load_dataset(
            "mozilla-foundation/common_voice_11_0",
            self.language,
            split=self.split,
            trust_remote_code=True,
        )
        
        samples = []
        for i, item in enumerate(dataset):
            if self.limit and i >= self.limit:
                break
            
            audio = item["audio"]
            duration = len(audio["array"]) / audio["sampling_rate"]
            
            samples.append(AudioSample(
                audio_path=audio.get("path", f"cv_{i}"),
                text=item["sentence"],
                duration=duration,
                task="asr",
                metadata={
                    "language": self.language,
                    "gender": item.get("gender"),
                    "age": item.get("age"),
                    "dataset": "common_voice",
                },
                audio_array=audio["array"],
                sample_rate=audio["sampling_rate"],
            ))
        
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
        """Load AudioSet samples."""
        try:
            from datasets import load_dataset
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
        
        samples = []
        for i, item in enumerate(dataset):
            if self.limit and i >= self.limit:
                break
            
            audio = item["audio"]
            duration = len(audio["array"]) / audio["sampling_rate"]
            
            # AudioSet has multiple labels
            labels = item.get("human_labels", item.get("labels", []))
            label_str = labels[0] if isinstance(labels, list) and labels else str(labels)
            
            samples.append(AudioSample(
                audio_path=audio.get("path", f"audioset_{i}"),
                text=label_str,
                duration=duration,
                task="classification",
                metadata={
                    "all_labels": labels,
                    "dataset": "audioset",
                },
                audio_array=audio["array"],
                sample_rate=audio["sampling_rate"],
            ))
        
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
        """Load Speech Commands samples."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required: pip install datasets")
        
        logger.info(f"Loading Speech Commands {self.split}...")
        dataset = load_dataset(
            "speech_commands",
            "v0.02",
            split=self.split,
            trust_remote_code=True,
        )
        
        samples = []
        for i, item in enumerate(dataset):
            if self.limit and i >= self.limit:
                break
            
            audio = item["audio"]
            duration = len(audio["array"]) / audio["sampling_rate"]
            
            samples.append(AudioSample(
                audio_path=audio.get("path", f"sc_{i}"),
                text=item["label"],  # Command label (e.g., "yes", "no", "up")
                duration=duration,
                task="classification",
                metadata={
                    "speaker_id": item.get("speaker_id"),
                    "is_unknown": item.get("is_unknown", False),
                    "dataset": "speech_commands",
                },
                audio_array=audio["array"],
                sample_rate=audio["sampling_rate"],
            ))
        
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
    if name not in AUDIO_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(AUDIO_DATASETS.keys())}")
    
    loader_cls = AUDIO_DATASETS[name]
    loader = loader_cls(split=split, limit=limit, **kwargs)
    loader.load()
    
    return loader


def list_audio_datasets() -> List[str]:
    """List available audio datasets."""
    return list(AUDIO_DATASETS.keys())
