"""
Audio Verifier Base

Base class for audio verification in RLVR training.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import logging
import warnings

logger = logging.getLogger(__name__)


class AudioVerificationError(Exception):
    """Error during audio verification."""
    pass


class DependencyWarning(UserWarning):
    """Warning for missing optional dependencies."""
    pass


@dataclass
class AudioVerifyResult:
    """Result of audio verification."""
    
    success: bool
    reward: float  # 0.0 to 1.0
    task: str  # asr, tts, classification
    details: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def failure(cls, task: str, reason: str) -> "AudioVerifyResult":
        """Create a failure result."""
        return cls(
            success=False,
            reward=0.0,
            task=task,
            details={"error": reason}
        )


@dataclass
class AudioVerifyConfig:
    """Configuration for audio verification."""
    
    # Task
    task: str = "asr"  # asr, tts, classification
    
    # ASR settings
    wer_threshold: float = 0.3
    use_cer: bool = False
    
    # TTS settings
    intelligibility_weight: float = 0.4
    quality_weight: float = 0.4
    consistency_weight: float = 0.2
    
    # Classification settings
    exact_match: bool = True


class AudioVerifier:
    """
    Multi-task audio verifier.
    
    Delegates to task-specific checkers based on configuration.
    """
    
    def __init__(self, config: Optional[AudioVerifyConfig] = None):
        """
        Initialize audio verifier.
        
        Args:
            config: Verification configuration
        """
        self.config = config or AudioVerifyConfig()
        
        # Check dependencies
        check_audio_verifier_dependencies()
        
        # Create checker based on task
        self._checker = self._create_checker()
    
    def _create_checker(self):
        """Create task-specific checker."""
        from halo_forge.audio.verifiers.asr import ASRChecker
        from halo_forge.audio.verifiers.tts import TTSChecker
        from halo_forge.audio.verifiers.classification import AudioClassificationChecker
        
        if self.config.task == "asr":
            return ASRChecker(
                wer_threshold=self.config.wer_threshold,
                use_cer=self.config.use_cer,
            )
        elif self.config.task == "tts":
            return TTSChecker(
                intelligibility_weight=self.config.intelligibility_weight,
                quality_weight=self.config.quality_weight,
                consistency_weight=self.config.consistency_weight,
            )
        elif self.config.task == "classification":
            return AudioClassificationChecker(
                exact_match=self.config.exact_match,
            )
        else:
            raise ValueError(f"Unknown task: {self.config.task}")
    
    def verify(
        self,
        prediction: str,
        ground_truth: str,
        **kwargs
    ) -> AudioVerifyResult:
        """
        Verify audio model output.
        
        Args:
            prediction: Model prediction (transcript or label)
            ground_truth: Expected output
            **kwargs: Task-specific arguments
            
        Returns:
            AudioVerifyResult with reward
        """
        result = self._checker.verify(prediction, ground_truth, **kwargs)
        return AudioVerifyResult(
            success=result.success,
            reward=result.reward,
            task=self.config.task,
            details=result.details,
        )
    
    def verify_batch(
        self,
        predictions: List[str],
        ground_truths: List[str],
        **kwargs
    ) -> List[AudioVerifyResult]:
        """
        Verify a batch of predictions.
        
        Args:
            predictions: List of predictions
            ground_truths: List of ground truths
            **kwargs: Task-specific arguments
            
        Returns:
            List of AudioVerifyResult
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have same length")
        
        results = []
        for pred, gt in zip(predictions, ground_truths):
            results.append(self.verify(pred, gt, **kwargs))
        
        return results


def check_audio_verifier_dependencies() -> Dict[str, bool]:
    """Check and warn about missing dependencies."""
    deps = {}
    
    # jiwer for WER calculation
    try:
        import jiwer
        deps["jiwer"] = True
    except ImportError:
        deps["jiwer"] = False
        warnings.warn(
            "jiwer not installed. WER calculation will use fallback. "
            "Install with: pip install jiwer",
            DependencyWarning
        )
    
    # torchaudio for audio processing
    try:
        import torchaudio
        deps["torchaudio"] = True
    except ImportError:
        deps["torchaudio"] = False
        warnings.warn(
            "torchaudio not installed. Audio processing unavailable. "
            "Install with: pip install torchaudio",
            DependencyWarning
        )
    
    # librosa for audio feature extraction
    try:
        import librosa
        deps["librosa"] = True
    except ImportError:
        deps["librosa"] = False
        warnings.warn(
            "librosa not installed. Advanced audio features unavailable. "
            "Install with: pip install librosa",
            DependencyWarning
        )
    
    return deps
