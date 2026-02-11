"""
Audio Model Adapters

Unified interface for various audio models.
"""

from halo_forge.audio.models.adapters import (
    AUDIO_TRAINING_SUPPORTED_FAMILIES,
    AudioAdapter,
    WhisperAdapter,
    Wav2VecAdapter,
    get_audio_adapter,
    supports_audio_training,
)

__all__ = [
    "AudioAdapter",
    "WhisperAdapter",
    "Wav2VecAdapter",
    "get_audio_adapter",
    "supports_audio_training",
    "AUDIO_TRAINING_SUPPORTED_FAMILIES",
]
