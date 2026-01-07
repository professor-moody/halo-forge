"""
Output Checker

Verifies the final output/answer in VLM completions.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from difflib import SequenceMatcher


@dataclass
class OutputResult:
    """Result of output verification."""
    exact_match: bool
    fuzzy_score: float
    semantic_score: float
    format_score: float
    overall_score: float
    details: Dict[str, Any]


class OutputChecker:
    """
    Verifies the final output/answer accuracy in VLM completions.
    
    Verification Methods:
    1. Exact Match - Direct string comparison
    2. Fuzzy Match - Similarity-based comparison
    3. Semantic Match - Meaning-based comparison (when available)
    4. Format Check - Proper answer formatting
    
    Usage:
        checker = OutputChecker()
        result = checker.verify(completion, ground_truth)
    """
    
    # Answer extraction patterns
    ANSWER_PATTERNS = [
        r"(?:the answer is|answer:)\s*[:\"]?\s*(.+?)(?:\.|$)",
        r"(?:therefore|thus|so),?\s*(?:the answer is|it is)?\s*[:\"]?\s*(.+?)(?:\.|$)",
        r"(?:in conclusion|finally),?\s*(.+?)(?:\.|$)",
        r"^([A-D])(?:\.|:|\))",  # Multiple choice
        r"^\**\s*([A-D])\s*\**",  # Bold multiple choice
    ]
    
    # Common VQA answer formats
    ANSWER_FORMATS = {
        'yes_no': {'yes', 'no'},
        'number': r'^\d+(?:\.\d+)?$',
        'multiple_choice': r'^[A-D]$',
        'color': {'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 
                  'brown', 'black', 'white', 'gray', 'grey'},
    }
    
    def __init__(
        self,
        fuzzy_threshold: float = 0.8,
        use_semantic: bool = False,
        normalize_answers: bool = True
    ):
        """
        Initialize output checker.
        
        Args:
            fuzzy_threshold: Minimum fuzzy match score for success
            use_semantic: Whether to use semantic similarity (requires model)
            normalize_answers: Whether to normalize answers before comparison
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.use_semantic = use_semantic
        self.normalize_answers = normalize_answers
        
        self._semantic_model = None
    
    def normalize(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Lowercase
        text = text.lower().strip()
        
        # Remove punctuation at end
        text = re.sub(r'[.,!?;:]+$', '', text)
        
        # Remove articles
        text = re.sub(r'^(a|an|the)\s+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_answer(self, completion: str) -> Optional[str]:
        """
        Extract the final answer from a completion.
        
        Args:
            completion: Model completion text
            
        Returns:
            Extracted answer or None
        """
        # Try each pattern
        for pattern in self.ANSWER_PATTERNS:
            matches = re.findall(pattern, completion, re.IGNORECASE | re.MULTILINE)
            if matches:
                return matches[-1].strip()  # Return last match (likely final answer)
        
        # Fallback: use last sentence
        sentences = re.split(r'[.!?]\s+', completion)
        if sentences:
            last = sentences[-1].strip()
            # Clean up
            last = re.sub(r'^[.,!?\s]+', '', last)
            return last if last else None
        
        return None
    
    def exact_match(self, answer: str, ground_truth: str) -> bool:
        """
        Check for exact match.
        
        Args:
            answer: Model's answer
            ground_truth: Expected answer
            
        Returns:
            True if exact match
        """
        if self.normalize_answers:
            answer = self.normalize(answer)
            ground_truth = self.normalize(ground_truth)
        
        return answer == ground_truth
    
    def fuzzy_match(self, answer: str, ground_truth: str) -> float:
        """
        Calculate fuzzy similarity score.
        
        Args:
            answer: Model's answer
            ground_truth: Expected answer
            
        Returns:
            Similarity score (0-1)
        """
        if self.normalize_answers:
            answer = self.normalize(answer)
            ground_truth = self.normalize(ground_truth)
        
        return SequenceMatcher(None, answer, ground_truth).ratio()
    
    def semantic_match(self, answer: str, ground_truth: str) -> float:
        """
        Calculate semantic similarity score.
        
        Uses sentence embeddings for comparison.
        
        Args:
            answer: Model's answer
            ground_truth: Expected answer
            
        Returns:
            Semantic similarity score (0-1)
        """
        if not self.use_semantic:
            return 0.0
        
        try:
            # Lazy load sentence transformer
            if self._semantic_model is None:
                from sentence_transformers import SentenceTransformer
                self._semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Get embeddings
            embeddings = self._semantic_model.encode([answer, ground_truth])
            
            # Cosine similarity
            from numpy import dot
            from numpy.linalg import norm
            
            similarity = dot(embeddings[0], embeddings[1]) / (
                norm(embeddings[0]) * norm(embeddings[1])
            )
            
            return float(similarity)
            
        except ImportError:
            return 0.0
    
    def check_format(self, answer: str, expected_format: Optional[str] = None) -> float:
        """
        Check if answer follows expected format.
        
        Args:
            answer: Model's answer
            expected_format: Expected format type
            
        Returns:
            Format score (0-1)
        """
        if not answer:
            return 0.0
        
        answer = answer.strip()
        
        if expected_format:
            if expected_format in self.ANSWER_FORMATS:
                fmt = self.ANSWER_FORMATS[expected_format]
                if isinstance(fmt, set):
                    return 1.0 if self.normalize(answer) in fmt else 0.5
                else:  # regex pattern
                    return 1.0 if re.match(fmt, answer) else 0.5
        
        # Check common formats
        # Yes/No
        if self.normalize(answer) in self.ANSWER_FORMATS['yes_no']:
            return 1.0
        
        # Number
        if re.match(self.ANSWER_FORMATS['number'], answer):
            return 1.0
        
        # Multiple choice
        if re.match(self.ANSWER_FORMATS['multiple_choice'], answer.upper()):
            return 1.0
        
        # Short answer (reasonable length)
        if len(answer.split()) <= 10:
            return 0.8
        
        # Long answer (might be over-explained)
        return 0.6
    
    def verify_multiple_choice(
        self,
        completion: str,
        ground_truth: str,
        choices: Optional[List[str]] = None
    ) -> OutputResult:
        """
        Verify multiple choice answer.
        
        Args:
            completion: Model completion
            ground_truth: Correct answer (A/B/C/D)
            choices: Optional list of choice texts
            
        Returns:
            OutputResult
        """
        # Extract answer
        answer = self.extract_answer(completion)
        
        if not answer:
            return OutputResult(
                exact_match=False,
                fuzzy_score=0.0,
                semantic_score=0.0,
                format_score=0.0,
                overall_score=0.0,
                details={'error': 'Could not extract answer'}
            )
        
        # Normalize to letter
        answer_letter = answer.strip().upper()
        if len(answer_letter) > 1:
            # Try to extract letter
            match = re.search(r'^([A-D])', answer_letter)
            if match:
                answer_letter = match.group(1)
        
        gt_letter = ground_truth.strip().upper()
        
        exact = answer_letter == gt_letter
        
        return OutputResult(
            exact_match=exact,
            fuzzy_score=1.0 if exact else 0.0,
            semantic_score=0.0,
            format_score=1.0 if len(answer_letter) == 1 else 0.5,
            overall_score=1.0 if exact else 0.0,
            details={
                'extracted_answer': answer_letter,
                'ground_truth': gt_letter,
            }
        )
    
    def verify(
        self,
        completion: str,
        ground_truth: str,
        expected_format: Optional[str] = None
    ) -> OutputResult:
        """
        Verify output against ground truth.
        
        Args:
            completion: Model completion text
            ground_truth: Expected answer
            expected_format: Expected answer format
            
        Returns:
            OutputResult with scores
        """
        # Extract answer from completion
        answer = self.extract_answer(completion)
        
        if not answer:
            # If no clear answer, use full completion for comparison
            answer = completion
        
        # Calculate scores
        exact = self.exact_match(answer, ground_truth)
        fuzzy = self.fuzzy_match(answer, ground_truth)
        semantic = self.semantic_match(answer, ground_truth) if self.use_semantic else 0.0
        format_score = self.check_format(answer, expected_format)
        
        # Calculate overall score
        if exact:
            overall = 1.0
        elif fuzzy >= self.fuzzy_threshold:
            overall = 0.8 + (0.2 * (fuzzy - self.fuzzy_threshold) / (1 - self.fuzzy_threshold))
        elif self.use_semantic and semantic >= 0.8:
            overall = 0.7 + (0.3 * semantic)
        else:
            # Weighted combination
            overall = max(
                0.5 * fuzzy + 0.3 * semantic + 0.2 * format_score,
                fuzzy * 0.8  # At least fuzzy score adjusted
            )
        
        return OutputResult(
            exact_match=exact,
            fuzzy_score=fuzzy,
            semantic_score=semantic,
            format_score=format_score,
            overall_score=overall,
            details={
                'extracted_answer': answer[:200] if answer else None,
                'ground_truth': ground_truth[:200],
                'normalized_answer': self.normalize(answer) if answer else None,
                'normalized_ground_truth': self.normalize(ground_truth),
            }
        )
    
    def verify_with_alternatives(
        self,
        completion: str,
        ground_truths: List[str]
    ) -> OutputResult:
        """
        Verify against multiple acceptable answers.
        
        Args:
            completion: Model completion
            ground_truths: List of acceptable answers
            
        Returns:
            OutputResult (best match)
        """
        best_result = None
        best_score = -1
        
        for gt in ground_truths:
            result = self.verify(completion, gt)
            if result.overall_score > best_score:
                best_score = result.overall_score
                best_result = result
        
        if best_result:
            best_result.details['all_ground_truths'] = ground_truths
        
        return best_result or OutputResult(
            exact_match=False,
            fuzzy_score=0.0,
            semantic_score=0.0,
            format_score=0.0,
            overall_score=0.0,
            details={'error': 'No ground truths provided'}
        )
