"""
Audio Model Adapters

Adapters for different audio model architectures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Any, Union
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""
    
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    segments: Optional[List[dict]] = None


class AudioAdapter(ABC):
    """Abstract base class for audio model adapters."""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize adapter.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use (auto-detect if None)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
    
    @abstractmethod
    def load(self) -> None:
        """Load model and processor."""
        pass
    
    @abstractmethod
    def transcribe(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio waveform
            language: Target language (if applicable)
            
        Returns:
            TranscriptionResult
        """
        pass
    
    def generate(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        **kwargs
    ) -> str:
        """Generate text from audio (alias for transcribe)."""
        result = self.transcribe(audio, **kwargs)
        return result.text


class WhisperAdapter(AudioAdapter):
    """Adapter for OpenAI Whisper models."""
    
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        device: Optional[str] = None,
    ):
        super().__init__(model_name, device)
        self.sample_rate = 16000
    
    def load(self) -> None:
        """Load Whisper model and processor."""
        try:
            from transformers import (
                WhisperForConditionalGeneration,
                WhisperProcessor,
            )
        except ImportError:
            raise ImportError(
                "transformers required for Whisper. "
                "Install with: pip install transformers"
            )
        
        logger.info(f"Loading Whisper model: {self.model_name}")
        
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        
        logger.info(f"Whisper loaded on {self.device}")
    
    def transcribe(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        language: Optional[str] = "en",
    ) -> TranscriptionResult:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio: Audio waveform (16kHz)
            language: Target language code
            
        Returns:
            TranscriptionResult
        """
        if self.model is None:
            self.load()
        
        # Convert to numpy if tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Process audio
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)
        
        # Generate with forced language
        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language
            generate_kwargs["task"] = "transcribe"
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features,
                **generate_kwargs,
            )
        
        # Decode
        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        
        return TranscriptionResult(
            text=text.strip(),
            language=language,
        )


class Wav2VecAdapter(AudioAdapter):
    """Adapter for Wav2Vec2 models."""
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        device: Optional[str] = None,
    ):
        super().__init__(model_name, device)
        self.sample_rate = 16000
    
    def load(self) -> None:
        """Load Wav2Vec2 model and processor."""
        try:
            from transformers import (
                Wav2Vec2ForCTC,
                Wav2Vec2Processor,
            )
        except ImportError:
            raise ImportError(
                "transformers required for Wav2Vec2. "
                "Install with: pip install transformers"
            )
        
        logger.info(f"Loading Wav2Vec2 model: {self.model_name}")
        
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        
        logger.info(f"Wav2Vec2 loaded on {self.device}")
    
    def transcribe(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using Wav2Vec2.
        
        Args:
            audio: Audio waveform (16kHz)
            language: Not used for Wav2Vec2
            
        Returns:
            TranscriptionResult
        """
        if self.model is None:
            self.load()
        
        # Convert to numpy if tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Process audio
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        
        # Move to device
        input_values = inputs.input_values.to(self.device)
        
        # Get logits
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(predicted_ids)[0]
        
        return TranscriptionResult(
            text=text.strip(),
            language=language,
        )


def get_audio_adapter(model_name: str, device: Optional[str] = None) -> AudioAdapter:
    """
    Get appropriate adapter for model.
    
    Args:
        model_name: Model name or path
        device: Device to use
        
    Returns:
        AudioAdapter instance
    """
    model_lower = model_name.lower()
    
    if "whisper" in model_lower:
        return WhisperAdapter(model_name, device)
    elif "wav2vec" in model_lower:
        return Wav2VecAdapter(model_name, device)
    else:
        # Default to Whisper
        logger.warning(f"Unknown model type: {model_name}, defaulting to Whisper")
        return WhisperAdapter(model_name, device)
