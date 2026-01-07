"""
ASR (Automatic Speech Recognition) Verifier

Verify speech-to-text accuracy using Word Error Rate (WER) and Character Error Rate (CER).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class ASRResult:
    """ASR verification result."""
    
    success: bool
    reward: float
    wer: float
    cer: Optional[float] = None
    predicted: str = ""
    ground_truth: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class ASRChecker:
    """
    Verify speech recognition accuracy.
    
    Reward structure:
    - WER 0%: reward = 1.0 (perfect)
    - WER 10%: reward = 0.9
    - WER 30%: reward = 0.7
    - WER 50%: reward = 0.5
    - WER 100%+: reward = 0.0
    """
    
    def __init__(
        self,
        wer_threshold: float = 0.3,
        use_cer: bool = False,
        normalize_text: bool = True,
    ):
        """
        Initialize ASR checker.
        
        Args:
            wer_threshold: WER threshold for success
            use_cer: Also calculate Character Error Rate
            normalize_text: Normalize text before comparison
        """
        self.wer_threshold = wer_threshold
        self.use_cer = use_cer
        self.normalize_text = normalize_text
        
        # Try to use jiwer for accurate WER
        self._jiwer = None
        try:
            import jiwer
            self._jiwer = jiwer
        except ImportError:
            logger.warning("jiwer not installed, using fallback WER calculation")
    
    def verify(
        self,
        prediction: str,
        ground_truth: str,
        **kwargs
    ) -> ASRResult:
        """
        Verify ASR prediction.
        
        Args:
            prediction: Model's transcription
            ground_truth: Expected transcription
            
        Returns:
            ASRResult with WER and reward
        """
        # Normalize text
        if self.normalize_text:
            prediction = self._normalize(prediction)
            ground_truth = self._normalize(ground_truth)
        
        # Calculate WER
        wer = self._calculate_wer(ground_truth, prediction)
        
        # Calculate CER if requested
        cer = None
        if self.use_cer:
            cer = self._calculate_cer(ground_truth, prediction)
        
        # Convert WER to reward (lower WER = higher reward)
        # WER can exceed 1.0 for very bad predictions
        reward = max(0.0, 1.0 - min(wer, 1.0))
        
        # Success if WER below threshold
        success = wer < self.wer_threshold
        
        return ASRResult(
            success=success,
            reward=reward,
            wer=wer,
            cer=cer,
            predicted=prediction,
            ground_truth=ground_truth,
            details={
                "wer": wer,
                "cer": cer,
                "wer_threshold": self.wer_threshold,
            }
        )
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _calculate_wer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate.
        
        WER = (S + D + I) / N
        where:
        - S = substitutions
        - D = deletions
        - I = insertions
        - N = words in reference
        """
        if self._jiwer:
            return self._jiwer.wer(reference, hypothesis)
        
        # Fallback: simple Levenshtein distance
        return self._levenshtein_wer(reference, hypothesis)
    
    def _calculate_cer(self, reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate."""
        if self._jiwer:
            return self._jiwer.cer(reference, hypothesis)
        
        # Fallback
        return self._levenshtein_cer(reference, hypothesis)
    
    def _levenshtein_wer(self, reference: str, hypothesis: str) -> float:
        """Fallback WER using Levenshtein distance on words."""
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        if not ref_words:
            return 1.0 if hyp_words else 0.0
        
        # Dynamic programming
        d = self._levenshtein_distance(ref_words, hyp_words)
        return d / len(ref_words)
    
    def _levenshtein_cer(self, reference: str, hypothesis: str) -> float:
        """Fallback CER using Levenshtein distance on characters."""
        ref_chars = list(reference)
        hyp_chars = list(hypothesis)
        
        if not ref_chars:
            return 1.0 if hyp_chars else 0.0
        
        d = self._levenshtein_distance(ref_chars, hyp_chars)
        return d / len(ref_chars)
    
    def _levenshtein_distance(self, s1: List, s2: List) -> int:
        """Calculate Levenshtein distance between two sequences."""
        m, n = len(s1), len(s2)
        
        # Create distance matrix
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # deletion
                        dp[i][j - 1],      # insertion
                        dp[i - 1][j - 1]   # substitution
                    )
        
        return dp[m][n]
