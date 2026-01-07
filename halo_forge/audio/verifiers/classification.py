"""
Audio Classification Verifier

Verify audio classification accuracy.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Audio classification verification result."""
    
    success: bool
    reward: float
    predicted: str
    ground_truth: str
    is_exact_match: bool
    details: Dict[str, Any] = field(default_factory=dict)


class AudioClassificationChecker:
    """
    Verify audio classification accuracy.
    
    Simple binary reward:
    - 1.0 if correct classification
    - 0.0 if incorrect
    
    Optionally supports fuzzy matching for similar labels.
    """
    
    def __init__(
        self,
        labels: Optional[List[str]] = None,
        exact_match: bool = True,
        case_sensitive: bool = False,
        label_aliases: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize classification checker.
        
        Args:
            labels: Valid class labels (optional)
            exact_match: Require exact string match
            case_sensitive: Case-sensitive comparison
            label_aliases: Dict mapping canonical labels to aliases
        """
        self.labels = labels
        self.exact_match = exact_match
        self.case_sensitive = case_sensitive
        self.label_aliases = label_aliases or {}
        
        # Build reverse alias map
        self._alias_to_canonical = {}
        for canonical, aliases in self.label_aliases.items():
            for alias in aliases:
                key = alias if case_sensitive else alias.lower()
                self._alias_to_canonical[key] = canonical
    
    def verify(
        self,
        prediction: str,
        ground_truth: str,
        **kwargs
    ) -> ClassificationResult:
        """
        Verify classification prediction.
        
        Args:
            prediction: Model's predicted class
            ground_truth: Expected class
            
        Returns:
            ClassificationResult
        """
        # Normalize strings
        pred_norm = prediction.strip()
        gt_norm = ground_truth.strip()
        
        if not self.case_sensitive:
            pred_norm = pred_norm.lower()
            gt_norm = gt_norm.lower()
        
        # Resolve aliases
        pred_canonical = self._alias_to_canonical.get(pred_norm, pred_norm)
        gt_canonical = self._alias_to_canonical.get(gt_norm, gt_norm)
        
        # Check match
        is_match = pred_canonical == gt_canonical
        
        # Fuzzy matching (optional)
        fuzzy_score = 0.0
        if not is_match and not self.exact_match:
            fuzzy_score = self._fuzzy_match(pred_canonical, gt_canonical)
            is_match = fuzzy_score >= 0.8
        
        # Binary reward for classification
        reward = 1.0 if is_match else 0.0
        
        # Check if prediction is valid label
        is_valid_label = True
        if self.labels:
            check_pred = prediction.strip()
            if not self.case_sensitive:
                check_pred = check_pred.lower()
                valid_labels = [l.lower() for l in self.labels]
            else:
                valid_labels = self.labels
            is_valid_label = check_pred in valid_labels
        
        return ClassificationResult(
            success=is_match,
            reward=reward,
            predicted=prediction,
            ground_truth=ground_truth,
            is_exact_match=pred_norm == gt_norm,
            details={
                "normalized_prediction": pred_norm,
                "normalized_ground_truth": gt_norm,
                "canonical_prediction": pred_canonical,
                "canonical_ground_truth": gt_canonical,
                "is_valid_label": is_valid_label,
                "fuzzy_score": fuzzy_score if not self.exact_match else None,
            }
        )
    
    def _fuzzy_match(self, s1: str, s2: str) -> float:
        """
        Calculate fuzzy similarity between strings.
        
        Uses simple character-based Jaccard similarity.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity score 0.0 to 1.0
        """
        if not s1 or not s2:
            return 0.0 if s1 != s2 else 1.0
        
        # Character-level comparison
        set1 = set(s1)
        set2 = set(s2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_valid_labels(self) -> Optional[List[str]]:
        """Get list of valid labels."""
        return self.labels
    
    def add_label_alias(self, canonical: str, alias: str) -> None:
        """
        Add an alias for a label.
        
        Args:
            canonical: Canonical label name
            alias: Alias to add
        """
        if canonical not in self.label_aliases:
            self.label_aliases[canonical] = []
        self.label_aliases[canonical].append(alias)
        
        # Update reverse map
        key = alias if self.case_sensitive else alias.lower()
        self._alias_to_canonical[key] = canonical


# Common audio classification label mappings
SPEECH_COMMANDS_LABELS = [
    "yes", "no", "up", "down", "left", "right", "on", "off",
    "stop", "go", "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
]

AUDIOSET_TOP_LABELS = [
    "Speech", "Music", "Singing", "Dog", "Cat", "Bird",
    "Vehicle", "Water", "Wind", "Thunder", "Silence",
]
