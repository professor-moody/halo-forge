"""
Math Verifier

SymPy-based mathematical answer verification.
"""

import re
from typing import Optional, Tuple
import logging

from halo_forge.rlvr.verifiers.base import VerifyResult
from halo_forge.reasoning.verifiers.base import (
    ReasoningVerifier,
    ReasoningVerifyResult,
    MathVerificationError,
)
from halo_forge.reasoning.verifiers.answer_extract import AnswerExtractor

logger = logging.getLogger(__name__)


class MathVerifier(ReasoningVerifier):
    """
    Verify mathematical reasoning and answers using SymPy.
    
    Verification strategies:
    1. Numeric comparison (with tolerance)
    2. Symbolic equivalence (via SymPy)
    3. Partial credit for correct work
    """
    
    def __init__(
        self,
        tolerance: float = 1e-6,
        partial_credit_for_work: bool = True,
        require_steps: bool = False,
        max_workers: int = 8,
    ):
        """
        Initialize math verifier.
        
        Args:
            tolerance: Numeric comparison tolerance
            partial_credit_for_work: Give partial credit for showing work
            require_steps: Require step-by-step reasoning
            max_workers: Parallel workers for batch verification
        """
        super().__init__(max_workers)
        self.tolerance = tolerance
        self.partial_credit_for_work = partial_credit_for_work
        self.require_steps = require_steps
        self.extractor = AnswerExtractor()
        
        # Check SymPy availability
        self._sympy_available = False
        try:
            import sympy
            self._sympy_available = True
        except ImportError:
            logger.warning("SymPy not available. Falling back to numeric comparison only.")
    
    def verify(
        self,
        prompt: str,
        completion: str,
        expected_answer: str,
        **kwargs
    ) -> VerifyResult:
        """
        Verify a mathematical completion.
        
        Args:
            prompt: Math problem
            completion: Model's solution
            expected_answer: Expected answer
            
        Returns:
            VerifyResult with reasoning details
        """
        # Extract answer from completion
        extracted = self.extractor.extract(completion)
        
        if not extracted:
            # No answer found
            if self.partial_credit_for_work and self._has_reasoning_steps(completion):
                return VerifyResult(
                    success=False,
                    reward=0.2,  # Partial credit for showing work
                    details={
                        "extracted_answer": None,
                        "expected_answer": expected_answer,
                        "reason": "no_answer_extracted",
                        "has_work": True,
                    }
                )
            return VerifyResult(
                success=False,
                reward=0.1,
                error="Could not extract answer from completion",
                details={"reason": "no_answer_extracted"}
            )
        
        # Try numeric comparison first
        numeric_match, numeric_details = self._numeric_match(extracted, expected_answer)
        if numeric_match:
            return VerifyResult(
                success=True,
                reward=1.0,
                details={
                    "extracted_answer": extracted,
                    "expected_answer": expected_answer,
                    "match_type": "numeric",
                    **numeric_details,
                }
            )
        
        # Try symbolic comparison if available
        if self._sympy_available:
            symbolic_match, symbolic_details = self._symbolic_match(extracted, expected_answer)
            if symbolic_match:
                return VerifyResult(
                    success=True,
                    reward=1.0,
                    details={
                        "extracted_answer": extracted,
                        "expected_answer": expected_answer,
                        "match_type": "symbolic",
                        **symbolic_details,
                    }
                )
        
        # Wrong answer - check for partial credit
        reward = 0.0
        if self.partial_credit_for_work and self._has_reasoning_steps(completion):
            reward = 0.2  # Partial credit for work
        
        return VerifyResult(
            success=False,
            reward=reward,
            details={
                "extracted_answer": extracted,
                "expected_answer": expected_answer,
                "match_type": "none",
                "has_work": self._has_reasoning_steps(completion),
            }
        )
    
    def _numeric_match(
        self,
        extracted: str,
        expected: str
    ) -> Tuple[bool, dict]:
        """
        Compare answers numerically.
        
        Returns:
            (match, details) tuple
        """
        try:
            # Parse extracted value
            extracted_val = self._parse_numeric(extracted)
            expected_val = self._parse_numeric(expected)
            
            if extracted_val is None or expected_val is None:
                return False, {"reason": "non_numeric"}
            
            # Check with tolerance
            if abs(extracted_val - expected_val) <= self.tolerance:
                return True, {
                    "extracted_value": extracted_val,
                    "expected_value": expected_val,
                    "difference": abs(extracted_val - expected_val)
                }
            
            # Check relative tolerance for large numbers
            if expected_val != 0:
                rel_diff = abs(extracted_val - expected_val) / abs(expected_val)
                if rel_diff <= self.tolerance:
                    return True, {
                        "extracted_value": extracted_val,
                        "expected_value": expected_val,
                        "relative_difference": rel_diff
                    }
            
            return False, {
                "extracted_value": extracted_val,
                "expected_value": expected_val,
                "difference": abs(extracted_val - expected_val)
            }
            
        except Exception as e:
            logger.debug(f"Numeric comparison failed: {e}")
            return False, {"error": str(e)}
    
    def _parse_numeric(self, value: str) -> Optional[float]:
        """Parse a string to numeric value."""
        if not value:
            return None
        
        # Clean the value
        cleaned = value.strip()
        cleaned = re.sub(r"\s+", "", cleaned)
        cleaned = re.sub(r",", "", cleaned)  # Remove thousand separators
        
        try:
            # Handle fractions
            if "/" in cleaned:
                parts = cleaned.split("/")
                if len(parts) == 2:
                    return float(parts[0]) / float(parts[1])
            
            # Handle percentages
            if cleaned.endswith("%"):
                return float(cleaned[:-1]) / 100
            
            # Handle pi
            if "pi" in cleaned.lower() or "π" in cleaned:
                import math
                cleaned = re.sub(r"pi|π", str(math.pi), cleaned, flags=re.IGNORECASE)
            
            return float(cleaned)
            
        except (ValueError, ZeroDivisionError):
            return None
    
    def _symbolic_match(
        self,
        extracted: str,
        expected: str
    ) -> Tuple[bool, dict]:
        """
        Compare answers symbolically using SymPy.
        
        Returns:
            (match, details) tuple
        """
        if not self._sympy_available:
            return False, {"reason": "sympy_not_available"}
        
        try:
            import sympy
            from sympy.parsing.sympy_parser import (
                parse_expr,
                standard_transformations,
                implicit_multiplication_application,
            )
            
            transformations = standard_transformations + (implicit_multiplication_application,)
            
            # Parse expressions
            try:
                extracted_expr = parse_expr(
                    self._prepare_for_sympy(extracted),
                    transformations=transformations,
                    evaluate=True
                )
            except Exception as e:
                logger.debug(f"Failed to parse extracted answer: {e}")
                return False, {"reason": f"parse_extracted_failed: {e}"}
            
            try:
                expected_expr = parse_expr(
                    self._prepare_for_sympy(expected),
                    transformations=transformations,
                    evaluate=True
                )
            except Exception as e:
                logger.debug(f"Failed to parse expected answer: {e}")
                return False, {"reason": f"parse_expected_failed: {e}"}
            
            # Check symbolic equality
            diff = sympy.simplify(extracted_expr - expected_expr)
            
            if diff == 0:
                return True, {
                    "extracted_expr": str(extracted_expr),
                    "expected_expr": str(expected_expr)
                }
            
            # Try numeric evaluation as fallback
            try:
                extracted_num = float(extracted_expr.evalf())
                expected_num = float(expected_expr.evalf())
                
                if abs(extracted_num - expected_num) <= self.tolerance:
                    return True, {
                        "extracted_expr": str(extracted_expr),
                        "expected_expr": str(expected_expr),
                        "numeric_match": True
                    }
            except Exception:
                pass
            
            return False, {
                "extracted_expr": str(extracted_expr),
                "expected_expr": str(expected_expr),
                "difference": str(diff)
            }
            
        except Exception as e:
            logger.debug(f"Symbolic comparison failed: {e}")
            return False, {"error": str(e)}
    
    def _prepare_for_sympy(self, expr: str) -> str:
        """Prepare expression string for SymPy parsing."""
        result = expr.strip()
        
        # Replace common LaTeX
        result = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))", result)
        result = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", result)
        result = re.sub(r"\\pi", "pi", result)
        result = re.sub(r"\\times", "*", result)
        result = re.sub(r"\\cdot", "*", result)
        result = re.sub(r"\\div", "/", result)
        result = re.sub(r"\^", "**", result)
        
        # Remove LaTeX formatting
        result = re.sub(r"\\[a-zA-Z]+\{([^}]+)\}", r"\1", result)
        result = re.sub(r"\\[a-zA-Z]+", "", result)
        result = re.sub(r"[{}]", "", result)
        
        return result
    
    def _has_reasoning_steps(self, completion: str) -> bool:
        """Check if completion contains reasoning steps."""
        # Look for step indicators
        step_patterns = [
            r"step\s*\d",
            r"first,?",
            r"second,?",
            r"then,?",
            r"next,?",
            r"therefore",
            r"hence",
            r"because",
            r"since",
            r"let\s+",
            r"we\s+have",
            r"we\s+get",
            r"we\s+know",
            r"this\s+gives",
            r"=",  # At least one equation
        ]
        
        for pattern in step_patterns:
            if re.search(pattern, completion, re.IGNORECASE):
                return True
        
        # Check for multiple lines/equations
        lines = [l.strip() for l in completion.split("\n") if l.strip()]
        if len(lines) >= 3:
            return True
        
        return False


def verify_math_answer(
    completion: str,
    expected: str,
    tolerance: float = 1e-6
) -> ReasoningVerifyResult:
    """
    Convenience function to verify a math answer.
    
    Args:
        completion: Model's solution
        expected: Expected answer
        tolerance: Numeric tolerance
        
    Returns:
        ReasoningVerifyResult
    """
    verifier = MathVerifier(tolerance=tolerance)
    return verifier.verify("", completion, expected)
