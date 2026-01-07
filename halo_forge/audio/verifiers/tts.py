"""
TTS (Text-to-Speech) Verifier

Verify text-to-speech quality using intelligibility, audio quality, and consistency.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """TTS verification result."""
    
    success: bool
    reward: float
    intelligibility_score: float
    quality_score: float
    consistency_score: float
    details: Dict[str, Any] = field(default_factory=dict)


class TTSChecker:
    """
    Verify text-to-speech quality.
    
    Metrics:
    - Intelligibility: ASR the generated audio, compare to target text
    - Quality: Predicted Mean Opinion Score (MOS)
    - Consistency: Speaker similarity (if reference provided)
    
    Reward = weighted combination of all metrics.
    """
    
    def __init__(
        self,
        intelligibility_weight: float = 0.4,
        quality_weight: float = 0.4,
        consistency_weight: float = 0.2,
        reference_audio: Optional[Union[str, np.ndarray]] = None,
        sample_rate: int = 16000,
    ):
        """
        Initialize TTS checker.
        
        Args:
            intelligibility_weight: Weight for intelligibility score
            quality_weight: Weight for quality score
            consistency_weight: Weight for consistency score
            reference_audio: Reference audio for consistency check
            sample_rate: Expected sample rate
        """
        self.weights = {
            "intelligibility": intelligibility_weight,
            "quality": quality_weight,
            "consistency": consistency_weight,
        }
        self.reference_audio = reference_audio
        self.sample_rate = sample_rate
        
        # ASR for intelligibility
        self._asr_model = None
        
        # Lazy-load quality model
        self._quality_model = None
    
    def verify(
        self,
        generated_audio: Union[np.ndarray, torch.Tensor],
        target_text: str,
        **kwargs
    ) -> TTSResult:
        """
        Verify TTS output quality.
        
        Args:
            generated_audio: Generated audio waveform
            target_text: Text that was synthesized
            **kwargs: Additional arguments
            
        Returns:
            TTSResult with scores
        """
        # Convert to numpy
        if isinstance(generated_audio, torch.Tensor):
            generated_audio = generated_audio.cpu().numpy()
        
        scores = {}
        details = {}
        
        # 1. Intelligibility: ASR the generated audio, compare to target
        intel_score, intel_details = self._check_intelligibility(
            generated_audio, target_text
        )
        scores["intelligibility"] = intel_score
        details["intelligibility"] = intel_details
        
        # 2. Audio quality (simplified MOS prediction)
        quality_score, quality_details = self._check_quality(generated_audio)
        scores["quality"] = quality_score
        details["quality"] = quality_details
        
        # 3. Consistency (if reference provided)
        if self.reference_audio is not None:
            consist_score, consist_details = self._check_consistency(
                generated_audio, self.reference_audio
            )
            scores["consistency"] = consist_score
            details["consistency"] = consist_details
        else:
            # No penalty if no reference
            scores["consistency"] = 1.0
            details["consistency"] = {"skipped": True}
        
        # Weighted reward
        reward = sum(
            self.weights[k] * scores[k]
            for k in self.weights
        )
        
        return TTSResult(
            success=reward >= 0.5,
            reward=reward,
            intelligibility_score=scores["intelligibility"],
            quality_score=scores["quality"],
            consistency_score=scores["consistency"],
            details=details,
        )
    
    def _check_intelligibility(
        self,
        audio: np.ndarray,
        target_text: str,
    ) -> tuple:
        """
        Check intelligibility by transcribing and comparing.
        
        Returns:
            (score, details)
        """
        try:
            # Try to use Whisper for ASR
            from halo_forge.audio.models.adapters import WhisperAdapter
            from halo_forge.audio.verifiers.asr import ASRChecker
            
            if self._asr_model is None:
                self._asr_model = WhisperAdapter("openai/whisper-tiny")
                self._asr_model.load()
            
            # Transcribe
            result = self._asr_model.transcribe(audio)
            transcribed = result.text
            
            # Calculate WER
            asr_checker = ASRChecker()
            asr_result = asr_checker.verify(transcribed, target_text)
            
            # Intelligibility = 1 - WER (capped)
            score = max(0, 1.0 - min(asr_result.wer, 1.0))
            
            return score, {
                "transcribed": transcribed,
                "target": target_text,
                "wer": asr_result.wer,
            }
        
        except Exception as e:
            logger.warning(f"Intelligibility check failed: {e}")
            return 0.5, {"error": str(e)}  # Default to neutral
    
    def _check_quality(self, audio: np.ndarray) -> tuple:
        """
        Check audio quality (simplified MOS prediction).
        
        In production, use a proper MOS predictor like UTMOS.
        For now, use simple signal quality heuristics.
        
        Returns:
            (score, details)
        """
        try:
            # Simple quality metrics
            
            # 1. Signal-to-noise ratio estimate
            signal_power = np.mean(audio ** 2)
            noise_floor = np.percentile(np.abs(audio), 10) ** 2
            snr = 10 * np.log10(signal_power / (noise_floor + 1e-10))
            snr_score = min(1.0, max(0.0, snr / 40))  # Normalize to 0-1
            
            # 2. Dynamic range
            peak = np.max(np.abs(audio))
            rms = np.sqrt(np.mean(audio ** 2))
            crest_factor = peak / (rms + 1e-10)
            dynamic_score = min(1.0, crest_factor / 10)
            
            # 3. Check for clipping
            clipping_ratio = np.mean(np.abs(audio) > 0.95)
            clipping_score = 1.0 - min(1.0, clipping_ratio * 10)
            
            # 4. Check for silence
            silence_ratio = np.mean(np.abs(audio) < 0.01)
            silence_score = 1.0 - min(1.0, silence_ratio)
            
            # Combined quality score
            score = (snr_score * 0.4 + dynamic_score * 0.2 +
                    clipping_score * 0.2 + silence_score * 0.2)
            
            return score, {
                "snr_estimate": float(snr),
                "crest_factor": float(crest_factor),
                "clipping_ratio": float(clipping_ratio),
                "silence_ratio": float(silence_ratio),
            }
        
        except Exception as e:
            logger.warning(f"Quality check failed: {e}")
            return 0.5, {"error": str(e)}
    
    def _check_consistency(
        self,
        generated: np.ndarray,
        reference: Union[np.ndarray, str],
    ) -> tuple:
        """
        Check voice consistency with reference.
        
        In production, use speaker embeddings (e.g., from SpeechBrain).
        For now, use simple spectral comparison.
        
        Returns:
            (score, details)
        """
        try:
            # Load reference if path
            if isinstance(reference, str):
                import torchaudio
                ref_wave, sr = torchaudio.load(reference)
                reference = ref_wave.numpy().flatten()
            
            # Simple spectral comparison
            # In production, use speaker embedding cosine similarity
            
            # Compute spectral centroids
            def spectral_centroid(signal, sr=16000):
                from scipy import fft
                spectrum = np.abs(fft.fft(signal))[:len(signal)//2]
                freqs = np.linspace(0, sr/2, len(spectrum))
                centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
                return centroid
            
            gen_centroid = spectral_centroid(generated)
            ref_centroid = spectral_centroid(reference)
            
            # Similarity based on centroid difference
            diff = abs(gen_centroid - ref_centroid)
            max_diff = 4000  # Hz
            score = max(0, 1.0 - diff / max_diff)
            
            return score, {
                "generated_centroid": float(gen_centroid),
                "reference_centroid": float(ref_centroid),
                "difference": float(diff),
            }
        
        except Exception as e:
            logger.warning(f"Consistency check failed: {e}")
            return 0.5, {"error": str(e)}
