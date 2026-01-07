"""
Audio Data Module

Dataset loaders and audio processors for audio-language training.
"""

from halo_forge.audio.data.loaders import (
    AudioSample,
    AudioDataset,
    LibriSpeechLoader,
    CommonVoiceLoader,
    AudioSetLoader,
    SpeechCommandsLoader,
    load_audio_dataset,
    list_audio_datasets,
    AUDIO_DATASETS,
)
from halo_forge.audio.data.processors import AudioProcessor, ProcessedAudio

__all__ = [
    "AudioSample",
    "AudioDataset",
    "LibriSpeechLoader",
    "CommonVoiceLoader",
    "AudioSetLoader",
    "SpeechCommandsLoader",
    "load_audio_dataset",
    "list_audio_datasets",
    "AUDIO_DATASETS",
    "AudioProcessor",
    "ProcessedAudio",
]
