"""
Audio Processing

Waveform preprocessing for RLVR training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Tuple
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProcessedAudio:
    """Processed audio result."""
    
    waveform: torch.Tensor  # Shape: [channels, samples] or [samples]
    sample_rate: int
    duration: float  # Duration in seconds
    features: Optional[torch.Tensor] = None  # Mel/MFCC features if extracted
    metadata: dict = field(default_factory=dict)


class AudioProcessor:
    """Audio preprocessing for RLVR training."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        feature_type: str = "raw",  # raw, mel, mfcc
        normalize: bool = True,
        mono: bool = True,
        max_duration: Optional[float] = None,  # Max duration in seconds
    ):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate
            feature_type: Feature extraction type (raw, mel, mfcc)
            normalize: Normalize waveform to [-1, 1]
            mono: Convert to mono
            max_duration: Maximum duration (truncate if longer)
        """
        self.sample_rate = sample_rate
        self.feature_type = feature_type
        self.normalize = normalize
        self.mono = mono
        self.max_duration = max_duration
        
        # Check for torchaudio
        self._torchaudio = None
        try:
            import torchaudio
            self._torchaudio = torchaudio
        except (ImportError, OSError, RuntimeError):
            logger.info("torchaudio is unavailable; checking the librosa/soundfile fallback")

        self._librosa = None
        self._soundfile = None
        if self._torchaudio is None:
            try:
                import librosa
                import soundfile

                self._librosa = librosa
                self._soundfile = soundfile
            except (ImportError, OSError, RuntimeError):
                logger.warning(
                    "Audio decoding requires torchaudio or both librosa and soundfile"
                )
    
    def load(self, audio_path: Union[str, Path]) -> ProcessedAudio:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            ProcessedAudio with waveform and metadata
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self._torchaudio is not None:
            waveform, sr = self._torchaudio.load(str(audio_path))
            if sr != self.sample_rate:
                resampler = self._torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
        elif self._librosa is not None and self._soundfile is not None:
            samples, sr = self._soundfile.read(
                str(audio_path), dtype="float32", always_2d=True
            )
            samples = np.asarray(samples).T
            if sr != self.sample_rate:
                samples = self._librosa.resample(
                    samples,
                    orig_sr=sr,
                    target_sr=self.sample_rate,
                    axis=-1,
                )
            waveform = torch.from_numpy(np.asarray(samples)).float()
        else:
            raise ImportError(
                "audio processing requires torchaudio or both librosa and soundfile"
            )
        
        # Convert to mono
        if self.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Normalize
        if self.normalize:
            max_val = waveform.abs().max()
            if max_val > 0:
                waveform = waveform / max_val
        
        # Truncate if too long
        if self.max_duration is not None:
            max_samples = int(self.max_duration * self.sample_rate)
            if waveform.shape[-1] > max_samples:
                waveform = waveform[..., :max_samples]
        
        # Calculate duration
        duration = waveform.shape[-1] / self.sample_rate
        
        # Extract features if requested
        features = None
        if self.feature_type == "mel":
            features = self._extract_mel(waveform)
        elif self.feature_type == "mfcc":
            features = self._extract_mfcc(waveform)
        
        return ProcessedAudio(
            waveform=waveform.squeeze(0) if self.mono else waveform,
            sample_rate=self.sample_rate,
            duration=duration,
            features=features,
            metadata={"source": str(audio_path)}
        )
    
    def load_array(
        self,
        audio_array: np.ndarray,
        original_sr: int
    ) -> ProcessedAudio:
        """
        Process audio from numpy array.
        
        Args:
            audio_array: Audio samples as numpy array
            original_sr: Original sample rate
            
        Returns:
            ProcessedAudio
        """
        # Convert to tensor
        waveform = torch.from_numpy(audio_array).float()
        
        # Ensure 2D [channels, samples]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        # Resample if necessary
        if original_sr != self.sample_rate:
            if self._torchaudio is not None:
                resampler = self._torchaudio.transforms.Resample(
                    original_sr, self.sample_rate
                )
                waveform = resampler(waveform)
            elif self._librosa is not None and self._soundfile is not None:
                samples = self._librosa.resample(
                    waveform.numpy(),
                    orig_sr=original_sr,
                    target_sr=self.sample_rate,
                    axis=-1,
                )
                waveform = torch.from_numpy(np.asarray(samples)).float()
            else:
                raise ImportError(
                    "audio resampling requires torchaudio or both librosa and soundfile"
                )
        
        # Convert to mono
        if self.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Normalize
        if self.normalize:
            max_val = waveform.abs().max()
            if max_val > 0:
                waveform = waveform / max_val
        
        # Calculate duration
        duration = waveform.shape[-1] / self.sample_rate
        
        return ProcessedAudio(
            waveform=waveform.squeeze(0) if self.mono else waveform,
            sample_rate=self.sample_rate,
            duration=duration,
            features=None,
            metadata={"source": "array"}
        )
    
    def _extract_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram."""
        if self._torchaudio is not None:
            mel_transform = self._torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=80,
                n_fft=400,
                hop_length=160,
                win_length=400,
            )
            return mel_transform(waveform)
        if self._librosa is not None and self._soundfile is not None:
            mel = self._librosa.feature.melspectrogram(
                y=waveform.numpy(),
                sr=self.sample_rate,
                n_mels=80,
                n_fft=400,
                hop_length=160,
                win_length=400,
            )
            return torch.from_numpy(np.asarray(mel)).float()
        raise ImportError(
            "mel extraction requires torchaudio or both librosa and soundfile"
        )
    
    def _extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features."""
        if self._torchaudio is not None:
            mfcc_transform = self._torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=40,
                melkwargs={
                    "n_mels": 80,
                    "n_fft": 400,
                    "hop_length": 160,
                },
            )
            return mfcc_transform(waveform)
        if self._librosa is not None and self._soundfile is not None:
            mfcc = self._librosa.feature.mfcc(
                y=waveform.numpy(), sr=self.sample_rate, n_mfcc=40
            )
            return torch.from_numpy(np.asarray(mfcc)).float()
        raise ImportError(
            "MFCC extraction requires torchaudio or both librosa and soundfile"
        )
    
    def pad_or_truncate(
        self,
        waveform: torch.Tensor,
        target_length: int
    ) -> torch.Tensor:
        """
        Pad or truncate waveform to target length.
        
        Args:
            waveform: Audio waveform
            target_length: Target number of samples
            
        Returns:
            Padded/truncated waveform
        """
        current_length = waveform.shape[-1]
        
        if current_length > target_length:
            # Truncate
            return waveform[..., :target_length]
        elif current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            return torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform


def check_audio_dependencies() -> dict:
    """Check audio processing dependencies."""
    deps = {
        "torchaudio": False,
        "librosa": False,
        "jiwer": False,
    }
    
    try:
        import torchaudio
        deps["torchaudio"] = True
    except ImportError:
        pass
    
    try:
        import librosa
        deps["librosa"] = True
    except ImportError:
        pass
    
    try:
        import jiwer
        deps["jiwer"] = True
    except ImportError:
        pass
    
    return deps
