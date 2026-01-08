"""
Base Reasoning Verifier

Foundation classes for math and reasoning verification.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import logging

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult

logger = logging.getLogger(__name__)


class MathVerificationError(Exception):
    """Error during mathematical verification."""
    pass


@dataclass
class ReasoningVerifyResult:
    """Result of reasoning verification."""
    
    success: bool
    reward: float
    extracted_answer: Optional[str] = None
    expected_answer: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @classmethod
    def failure(cls, reason: str, partial_reward: float = 0.0) -> "ReasoningVerifyResult":
        """Create a failure result."""
        return cls(
            success=False,
            reward=partial_reward,
            error=reason,
            details={"reason": reason}
        )
    
    @classmethod
    def correct(cls, extracted: str, expected: str) -> "ReasoningVerifyResult":
        """Create a correct answer result."""
        return cls(
            success=True,
            reward=1.0,
            extracted_answer=extracted,
            expected_answer=expected,
            details={"match_type": "exact"}
        )


class ReasoningVerifier(Verifier):
    """
    Abstract base class for reasoning verifiers.
    
    Inherits from base Verifier to follow codebase architecture patterns.
    Overrides verify() with reasoning-specific signature.
    """
    
    def __init__(self, max_workers: int = 8):
        """
        Initialize verifier.
        
        Args:
            max_workers: Maximum parallel workers for batch verification
        """
        super().__init__(max_workers=max_workers)
        self.stats = {"total": 0, "success": 0, "failure": 0}
    
    @abstractmethod
    def verify(
        self,
        prompt: str,
        completion: str,
        expected_answer: str,
        **kwargs
    ) -> VerifyResult:
        """
        Verify a reasoning completion.
        
        Args:
            prompt: Original problem/prompt
            completion: Model's completion with reasoning
            expected_answer: Expected final answer
            
        Returns:
            VerifyResult with reasoning details
        """
        pass
    
    def verify_batch_reasoning(self, samples: list) -> List[VerifyResult]:
        """
        Verify a batch of reasoning samples in parallel.
        
        Args:
            samples: List of (prompt, completion, expected_answer) tuples or dicts
            
        Returns:
            List of VerifyResult
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results: List[Optional[VerifyResult]] = [None] * len(samples)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for i, sample in enumerate(samples):
                if isinstance(sample, (list, tuple)) and len(sample) == 3:
                    prompt, completion, expected = sample
                    future = executor.submit(
                        self.verify, prompt, completion, expected
                    )
                else:
                    # Assume dict with keys
                    future = executor.submit(
                        self.verify,
                        sample["prompt"],
                        sample["completion"],
                        sample["expected_answer"]
                    )
                futures[future] = i
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = VerifyResult(
                        success=False,
                        reward=0.0,
                        details=str(e),
                        error=str(e)
                    )
        
        # Update stats
        self.stats["total"] += len(results)
        self.stats["success"] += sum(1 for r in results if r and r.success)
        self.stats["failure"] += sum(1 for r in results if r and not r.success)
        
        return results


def check_reasoning_dependencies() -> Dict[str, bool]:
    """
    Check availability of reasoning dependencies.
    
    Returns:
        Dict mapping dependency name to availability
    """
    deps = {}
    
    try:
        import sympy
        deps["sympy"] = True
    except ImportError:
        deps["sympy"] = False
        logger.warning("sympy not installed. Install with: pip install sympy")
    
    return deps
