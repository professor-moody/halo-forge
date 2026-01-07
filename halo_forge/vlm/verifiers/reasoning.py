"""
Reasoning Checker

Verifies the quality and consistency of reasoning chains in VLM outputs.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class ReasoningResult:
    """Result of reasoning verification."""
    structure_score: float      # Does it have proper reasoning structure?
    consistency_score: float    # Are steps logically consistent?
    grounding_score: float      # Are claims grounded in visual evidence?
    overall_score: float
    details: Dict[str, Any]


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    text: str
    step_number: int
    has_evidence: bool
    is_conclusion: bool


class ReasoningChecker:
    """
    Verifies Chain-of-Thought reasoning quality in VLM outputs.
    
    Verification Aspects:
    1. Structure - Does the response have clear reasoning steps?
    2. Consistency - Are steps logically connected?
    3. Grounding - Are claims backed by visual evidence references?
    
    Usage:
        checker = ReasoningChecker()
        result = checker.verify(completion)
    """
    
    # Patterns indicating reasoning structure
    STEP_PATTERNS = [
        r"(?:first|step\s*1|1\.)",
        r"(?:second|step\s*2|2\.)",
        r"(?:third|step\s*3|3\.)",
        r"(?:next|then|after that)",
        r"(?:finally|therefore|thus|so|hence)",
    ]
    
    # Evidence grounding patterns
    EVIDENCE_PATTERNS = [
        r"(?:I (?:can )?see|looking at|observing|noticing)",
        r"(?:the image shows?|the picture displays?|visible in)",
        r"(?:based on|according to|from the image)",
        r"(?:in the (?:image|picture|photo))",
    ]
    
    # Conclusion patterns
    CONCLUSION_PATTERNS = [
        r"(?:therefore|thus|so|hence|in conclusion)",
        r"(?:the answer is|this means|we can conclude)",
        r"(?:finally|as a result)",
    ]
    
    # Contradiction patterns
    CONTRADICTION_MARKERS = [
        r"but\s+also",
        r"however.*opposite",
        r"on one hand.*on the other hand.*contradicts",
    ]
    
    def __init__(
        self,
        min_steps: int = 2,
        require_evidence: bool = True,
        require_conclusion: bool = True
    ):
        """
        Initialize reasoning checker.
        
        Args:
            min_steps: Minimum reasoning steps expected
            require_evidence: Whether to require visual evidence references
            require_conclusion: Whether to require explicit conclusion
        """
        self.min_steps = min_steps
        self.require_evidence = require_evidence
        self.require_conclusion = require_conclusion
    
    def extract_reasoning_steps(self, completion: str) -> List[ReasoningStep]:
        """
        Extract reasoning steps from completion.
        
        Args:
            completion: Model completion text
            
        Returns:
            List of ReasoningStep objects
        """
        steps = []
        
        # Split by common step indicators
        # First try numbered steps
        numbered_pattern = r"(?:^|\n)\s*(?:\d+[\.\):]|step\s*\d+[:\.]?)\s*(.+?)(?=(?:\n\s*(?:\d+[\.\)]|step\s*\d+)|$))"
        numbered_matches = re.findall(numbered_pattern, completion, re.IGNORECASE | re.DOTALL)
        
        if numbered_matches:
            for i, text in enumerate(numbered_matches):
                steps.append(self._create_step(text.strip(), i + 1))
        else:
            # Try sentence-level parsing
            sentences = re.split(r'(?<=[.!?])\s+', completion)
            step_num = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if this is a reasoning step
                is_step = False
                for pattern in self.STEP_PATTERNS:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        is_step = True
                        break
                
                # Also include sentences that reference evidence
                for pattern in self.EVIDENCE_PATTERNS:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        is_step = True
                        break
                
                if is_step or len(sentence) > 30:  # Include substantial sentences
                    step_num += 1
                    steps.append(self._create_step(sentence, step_num))
        
        return steps
    
    def _create_step(self, text: str, step_number: int) -> ReasoningStep:
        """Create a ReasoningStep from text."""
        has_evidence = any(
            re.search(p, text, re.IGNORECASE) 
            for p in self.EVIDENCE_PATTERNS
        )
        
        is_conclusion = any(
            re.search(p, text, re.IGNORECASE)
            for p in self.CONCLUSION_PATTERNS
        )
        
        return ReasoningStep(
            text=text,
            step_number=step_number,
            has_evidence=has_evidence,
            is_conclusion=is_conclusion
        )
    
    def verify_structure(self, steps: List[ReasoningStep]) -> Tuple[float, Dict[str, Any]]:
        """
        Verify reasoning structure.
        
        Checks:
        - Has minimum number of steps
        - Has logical progression
        - Has conclusion
        
        Args:
            steps: List of reasoning steps
            
        Returns:
            (score, details)
        """
        details = {
            'num_steps': len(steps),
            'has_minimum_steps': len(steps) >= self.min_steps,
            'has_conclusion': any(s.is_conclusion for s in steps),
            'step_count_expected': self.min_steps,
        }
        
        score = 0.0
        
        # Score for having steps
        if len(steps) >= self.min_steps:
            score += 0.5
        elif len(steps) > 0:
            score += 0.25 * (len(steps) / self.min_steps)
        
        # Score for having conclusion
        if details['has_conclusion']:
            score += 0.3
        
        # Score for step progression (numbered/ordered)
        if len(steps) >= 2:
            score += 0.2
        
        return min(score, 1.0), details
    
    def verify_consistency(self, steps: List[ReasoningStep]) -> Tuple[float, Dict[str, Any]]:
        """
        Verify logical consistency between steps.
        
        Checks:
        - No contradictions
        - Logical flow
        
        Args:
            steps: List of reasoning steps
            
        Returns:
            (score, details)
        """
        if not steps:
            return 0.5, {'no_steps': True}
        
        full_text = ' '.join(s.text for s in steps)
        
        # Check for contradictions
        contradictions = []
        for pattern in self.CONTRADICTION_MARKERS:
            if re.search(pattern, full_text, re.IGNORECASE):
                contradictions.append(pattern)
        
        details = {
            'num_contradictions': len(contradictions),
            'contradiction_patterns': contradictions,
        }
        
        # Start with full score, deduct for issues
        score = 1.0
        
        # Deduct for contradictions
        if contradictions:
            score -= 0.3 * len(contradictions)
        
        # Check for abrupt topic changes (simple heuristic)
        # This is a simplified check - could be enhanced with embeddings
        
        return max(0.0, min(score, 1.0)), details
    
    def verify_grounding(self, steps: List[ReasoningStep]) -> Tuple[float, Dict[str, Any]]:
        """
        Verify that reasoning is grounded in visual evidence.
        
        Checks:
        - References to image/visual content
        - Evidence-based claims
        
        Args:
            steps: List of reasoning steps
            
        Returns:
            (score, details)
        """
        if not steps:
            return 0.5 if not self.require_evidence else 0.0, {'no_steps': True}
        
        grounded_steps = [s for s in steps if s.has_evidence]
        
        details = {
            'total_steps': len(steps),
            'grounded_steps': len(grounded_steps),
            'grounding_ratio': len(grounded_steps) / len(steps) if steps else 0,
        }
        
        # Calculate score based on grounding ratio
        if self.require_evidence:
            # Expect at least 50% of steps to reference visual evidence
            score = min(1.0, details['grounding_ratio'] * 2)
        else:
            # More lenient if evidence not required
            score = 0.5 + (0.5 * details['grounding_ratio'])
        
        return score, details
    
    def verify(self, completion: str) -> ReasoningResult:
        """
        Verify reasoning quality in a VLM completion.
        
        Args:
            completion: Model completion text
            
        Returns:
            ReasoningResult with scores
        """
        # Extract reasoning steps
        steps = self.extract_reasoning_steps(completion)
        
        # Verify each aspect
        structure_score, structure_details = self.verify_structure(steps)
        consistency_score, consistency_details = self.verify_consistency(steps)
        grounding_score, grounding_details = self.verify_grounding(steps)
        
        # Calculate overall score (weighted)
        overall = (
            0.3 * structure_score +
            0.3 * consistency_score +
            0.4 * grounding_score
        )
        
        return ReasoningResult(
            structure_score=structure_score,
            consistency_score=consistency_score,
            grounding_score=grounding_score,
            overall_score=overall,
            details={
                'steps': [{'text': s.text[:100], 'has_evidence': s.has_evidence, 'is_conclusion': s.is_conclusion} for s in steps],
                'structure': structure_details,
                'consistency': consistency_details,
                'grounding': grounding_details,
            }
        )
    
    def verify_with_context(
        self,
        completion: str,
        prompt: str,
        ground_truth: Optional[str] = None
    ) -> ReasoningResult:
        """
        Verify reasoning with additional context.
        
        Args:
            completion: Model completion text
            prompt: Original prompt/question
            ground_truth: Optional ground truth answer
            
        Returns:
            ReasoningResult with context-aware scores
        """
        result = self.verify(completion)
        
        # Additional context-based verification
        if ground_truth:
            # Check if reasoning leads to correct answer
            gt_lower = ground_truth.lower()
            completion_lower = completion.lower()
            
            if gt_lower in completion_lower:
                # Boost score if correct answer present
                result.overall_score = min(1.0, result.overall_score + 0.1)
                result.details['contains_ground_truth'] = True
            else:
                result.details['contains_ground_truth'] = False
        
        # Check if completion addresses the prompt
        if prompt:
            # Simple relevance check - could be enhanced with embeddings
            prompt_words = set(prompt.lower().split())
            completion_words = set(completion.lower().split())
            overlap = len(prompt_words & completion_words) / len(prompt_words) if prompt_words else 0
            
            result.details['prompt_relevance'] = overlap
            
            # Adjust score based on relevance
            if overlap < 0.1:
                result.overall_score *= 0.8  # Penalize irrelevant responses
        
        return result
