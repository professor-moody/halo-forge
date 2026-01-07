"""
Audio-Language Training Module

Phase 4 of halo-forge: RLVR training for audio-language models.

Supports:
- ASR (Automatic Speech Recognition)
- TTS (Text-to-Speech) quality verification
- Audio Classification
"""

from halo_forge.audio.data import (
    AudioSample,
    AudioProcessor,
    load_audio_dataset,
    list_audio_datasets,
)
from halo_forge.audio.verifiers import (
    AudioVerifier,
    ASRChecker,
    TTSChecker,
    AudioClassificationChecker,
    AudioVerificationError,
)
from halo_forge.audio.trainer import AudioRAFTTrainer, AudioRAFTConfig
from halo_forge.audio.models import (
    AudioAdapter,
    WhisperAdapter,
    Wav2VecAdapter,
    get_audio_adapter,
)

__all__ = [
    # Data
    "AudioSample",
    "AudioProcessor",
    "load_audio_dataset",
    "list_audio_datasets",
    # Verifiers
    "AudioVerifier",
    "ASRChecker",
    "TTSChecker",
    "AudioClassificationChecker",
    "AudioVerificationError",
    # Trainer
    "AudioRAFTTrainer",
    "AudioRAFTConfig",
    # Models
    "AudioAdapter",
    "WhisperAdapter",
    "Wav2VecAdapter",
    "get_audio_adapter",
]
