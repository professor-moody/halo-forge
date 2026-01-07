"""
Audio Model Adapters

Unified interface for various audio models.
"""

from halo_forge.audio.models.adapters import (
    AudioAdapter,
    WhisperAdapter,
    Wav2VecAdapter,
    get_audio_adapter,
)

__all__ = [
    "AudioAdapter",
    "WhisperAdapter",
    "Wav2VecAdapter",
    "get_audio_adapter",
]
