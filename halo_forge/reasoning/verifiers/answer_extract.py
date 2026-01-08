"""
Answer Extractor

Extract final answers from model completions in various formats.
"""

import re
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class AnswerExtractor:
    """
    Extract final answers from mathematical/reasoning completions.
    
    Handles common formats:
    - LaTeX boxed: \\boxed{answer}
    - Final answer patterns: "The answer is X", "Therefore, X"
    - Numeric patterns at end of text
    - Fraction formats: a/b
    """
    
    # Common answer patterns in priority order
    ANSWER_PATTERNS = [
        # LaTeX boxed (highest priority)
        r"\\boxed\{([^}]+)\}",
        r"\\boxed\s*\{([^}]+)\}",
        
        # Explicit answer markers
        r"(?:The )?[Aa]nswer(?:\s+is)?[:\s]+([^\n.]+)",
        r"[Ff]inal [Aa]nswer[:\s]+([^\n.]+)",
        r"[Tt]herefore,?\s+(?:the answer is\s+)?([^\n.]+)",
        r"[Hh]ence,?\s+(?:the answer is\s+)?([^\n.]+)",
        r"[Ss]o,?\s+(?:the answer is\s+)?([^\n.]+)",
        
        # Equals patterns
        r"=\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?)\s*$",
        
        # Dollar amount
        r"\$\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
    ]
    
    # Patterns to clean from extracted answers
    CLEANUP_PATTERNS = [
        r"^\s*",           # Leading whitespace
        r"\s*$",           # Trailing whitespace
        r"^[:\s=]+",       # Leading colons/equals
        r"[.\s]+$",        # Trailing periods
        r"^\$\s*",         # Leading dollar sign (for value extraction)
        r",(?=\d{3})",     # Thousands separators
    ]
    
    def __init__(self, prefer_boxed: bool = True):
        """
        Initialize extractor.
        
        Args:
            prefer_boxed: Prefer \\boxed{} format over other patterns
        """
        self.prefer_boxed = prefer_boxed
    
    def extract(self, text: str) -> Optional[str]:
        """
        Extract final answer from text.
        
        Args:
            text: Model completion text
            
        Returns:
            Extracted answer string, or None if not found
        """
        if not text:
            return None
        
        # Try patterns in priority order
        for pattern in self.ANSWER_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                # Take last match (final answer is usually at the end)
                answer = matches[-1]
                cleaned = self._clean_answer(answer)
                if cleaned:
                    return cleaned
        
        # Fallback: try to find last numeric value
        return self._extract_last_number(text)
    
    def extract_all(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract all potential answers with their patterns.
        
        Args:
            text: Model completion text
            
        Returns:
            List of (answer, pattern_name) tuples
        """
        results = []
        
        for i, pattern in enumerate(self.ANSWER_PATTERNS):
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                cleaned = self._clean_answer(match)
                if cleaned:
                    results.append((cleaned, f"pattern_{i}"))
        
        return results
    
    def extract_boxed(self, text: str) -> Optional[str]:
        """
        Extract answer from \\boxed{} format only.
        
        Args:
            text: Model completion text
            
        Returns:
            Boxed answer or None
        """
        # Handle nested braces in boxed content
        pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        match = re.search(pattern, text)
        if match:
            return self._clean_answer(match.group(1))
        
        # Simple version
        match = re.search(r"\\boxed\{([^}]+)\}", text)
        if match:
            return self._clean_answer(match.group(1))
        
        return None
    
    def _clean_answer(self, answer: str) -> str:
        """Clean extracted answer string."""
        if not answer:
            return ""
        
        cleaned = answer.strip()
        
        # Remove common prefixes/suffixes
        cleaned = re.sub(r"^[:\s=]+", "", cleaned)
        cleaned = re.sub(r"[.\s]+$", "", cleaned)
        cleaned = re.sub(r"^\s*\$\s*", "", cleaned)  # Dollar sign
        cleaned = re.sub(r",(?=\d{3})", "", cleaned)  # Thousands separator
        
        # Remove LaTeX formatting
        cleaned = re.sub(r"\\text\{([^}]+)\}", r"\1", cleaned)
        cleaned = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", cleaned)
        cleaned = re.sub(r"\\textbf\{([^}]+)\}", r"\1", cleaned)
        
        return cleaned.strip()
    
    def _extract_last_number(self, text: str) -> Optional[str]:
        """Extract the last numeric value from text."""
        # Look for numbers (including fractions and decimals)
        pattern = r"([+-]?\d+(?:\.\d+)?(?:/\d+)?)"
        matches = re.findall(pattern, text)
        
        if matches:
            return matches[-1]
        
        return None
    
    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for comparison.
        
        Handles:
        - Whitespace normalization
        - Fraction simplification
        - Decimal/fraction equivalence
        """
        if not answer:
            return ""
        
        normalized = answer.strip().lower()
        
        # Remove spaces
        normalized = re.sub(r"\s+", "", normalized)
        
        # Try to convert to numeric for comparison
        try:
            # Handle fractions
            if "/" in normalized:
                parts = normalized.split("/")
                if len(parts) == 2:
                    num = float(parts[0])
                    den = float(parts[1])
                    if den != 0:
                        return str(num / den)
            
            # Handle percentages
            if normalized.endswith("%"):
                return str(float(normalized[:-1]) / 100)
            
            # Try direct float conversion
            return str(float(normalized))
        except (ValueError, ZeroDivisionError):
            pass
        
        return normalized


def extract_answer(text: str) -> Optional[str]:
    """
    Convenience function to extract answer from text.
    
    Args:
        text: Model completion text
        
    Returns:
        Extracted answer or None
    """
    extractor = AnswerExtractor()
    return extractor.extract(text)
