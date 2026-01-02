"""
Abstract Verifier Interface

Base class for all verifiers used in RLVR training.
Verifiers check generated code and return rewards.

Reward Levels (graduated rewards):
- 0.0: Complete failure (syntax errors, doesn't compile)
- 0.3: Compiles with warnings
- 0.5: Compiles clean
- 0.7: Runs without crash
- 1.0: Produces correct output
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import re


class RewardLevel(Enum):
    """
    Standard reward levels for graduated verification.
    
    Use these to provide partial credit for near-successes,
    which helps gradient flow during training.
    """
    FAILURE = 0.0           # Complete failure (syntax errors)
    COMPILE_WARNINGS = 0.3  # Compiles with warnings
    COMPILE_CLEAN = 0.5     # Compiles without warnings
    RUNS_NO_CRASH = 0.7     # Executes without crashing
    CORRECT_OUTPUT = 1.0    # Produces correct output
    
    @classmethod
    def from_compile_result(cls, success: bool, has_warnings: bool = False) -> float:
        """Get reward from compilation result."""
        if not success:
            return cls.FAILURE.value
        return cls.COMPILE_WARNINGS.value if has_warnings else cls.COMPILE_CLEAN.value
    
    @classmethod
    def from_execution_result(cls, compiles: bool, runs: bool, correct: bool) -> float:
        """Get reward from full execution result."""
        if not compiles:
            return cls.FAILURE.value
        if not runs:
            return cls.COMPILE_CLEAN.value
        if not correct:
            return cls.RUNS_NO_CRASH.value
        return cls.CORRECT_OUTPUT.value


@dataclass
class VerifyResult:
    """Result from verifying a code sample."""
    
    success: bool
    reward: float  # 0.0 to 1.0 (see RewardLevel for standard values)
    details: str   # Human-readable explanation
    error: Optional[str] = None  # Error message if failed
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional info
    
    def __repr__(self):
        status = "PASS" if self.success else "FAIL"
        return f"VerifyResult({status}, reward={self.reward:.2f}, details='{self.details[:50]}...')"


class Verifier(ABC):
    """
    Abstract base class for code verification.
    
    Subclasses must implement the `verify` method.
    The `verify_batch` method provides parallel verification by default.
    
    Example:
        class MyVerifier(Verifier):
            def verify(self, code: str) -> VerifyResult:
                # Check the code somehow
                if passes_check(code):
                    return VerifyResult(success=True, reward=1.0, details="Passed")
                else:
                    return VerifyResult(success=False, reward=0.0, details="Failed", error="...")
    """
    
    def __init__(self, max_workers: int = 8):
        """
        Initialize verifier.
        
        Args:
            max_workers: Maximum parallel workers for batch verification
        """
        self.max_workers = max_workers
    
    @abstractmethod
    def verify(self, code: str) -> VerifyResult:
        """
        Verify a single code sample.
        
        Args:
            code: The code to verify
            
        Returns:
            VerifyResult with success, reward, and details
        """
        pass
    
    def verify_batch(
        self, 
        codes: List[str], 
        prompts: Optional[List[str]] = None
    ) -> List[VerifyResult]:
        """
        Verify multiple code samples in parallel.
        
        Args:
            codes: List of code strings to verify
            prompts: Optional list of prompts (for verifiers that need context)
            
        Returns:
            List of VerifyResult objects in same order as input
        """
        results = [None] * len(codes)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            if prompts:
                futures = {
                    executor.submit(self._verify_with_prompt, code, prompt): i 
                    for i, (code, prompt) in enumerate(zip(codes, prompts))
                }
            else:
                futures = {executor.submit(self.verify, code): i for i, code in enumerate(codes)}
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = VerifyResult(
                        success=False,
                        reward=0.0,
                        details="Verification exception",
                        error=str(e)
                    )
        
        return results
    
    def _verify_with_prompt(self, code: str, prompt: str) -> VerifyResult:
        """
        Verify code with prompt context. Override in subclasses that need prompt.
        Default implementation ignores prompt.
        """
        return self.verify(code)
    
    def extract_code(self, text: str) -> str:
        """
        Extract code from model output.
        
        Handles common patterns:
        - Code blocks with ```cpp or ```python
        - <code></code> tags
        - Raw code starting with #include or similar
        
        Override this method for custom extraction logic.
        
        Args:
            text: Raw model output
            
        Returns:
            Extracted code string
        """
        # Try markdown code blocks first
        code_pattern = r'```(?:cpp|c\+\+|python|rust|go)?\s*(.*?)```'
        matches = re.findall(code_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Try <code></code> tags
        code_pattern = r'<code>(.*?)</code>'
        matches = re.findall(code_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Look for code starting with common patterns
        if '#include' in text:
            start = text.find('#include')
            # Find the end by matching braces
            code = text[start:]
            # Try to find main() and its closing brace
            main_match = re.search(r'int\s+main\s*\([^)]*\)\s*\{', code)
            if main_match:
                brace_count = 1
                for i in range(main_match.end(), len(code)):
                    if code[i] == '{':
                        brace_count += 1
                    elif code[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            return code[:i + 1].strip()
            # Fallback: return up to last closing brace
            last_brace = code.rfind('}')
            if last_brace > 0:
                return code[:last_brace + 1].strip()
            return code.strip()
        
        # Return as-is if no patterns match
        return text.strip()
    
    def cleanup(self):
        """
        Cleanup resources.
        
        Override this method if your verifier needs cleanup
        (e.g., closing SSH connections, temporary files).
        """
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


class ChainedVerifier(Verifier):
    """
    Chain multiple verifiers together.
    
    Runs verifiers in sequence, stopping at the first failure.
    Useful for multi-stage verification (e.g., compile then test).
    
    Example:
        verifier = ChainedVerifier([
            GCCVerifier(),
            PytestVerifier(),
        ])
    """
    
    def __init__(self, verifiers: List[Verifier], weights: Optional[List[float]] = None):
        """
        Initialize chained verifier.
        
        Args:
            verifiers: List of verifiers to run in order
            weights: Optional weights for each verifier's reward contribution
                    If None, rewards are averaged
        """
        super().__init__()
        self.verifiers = verifiers
        self.weights = weights or [1.0 / len(verifiers)] * len(verifiers)
    
    def verify(self, code: str) -> VerifyResult:
        """Run all verifiers in sequence."""
        total_reward = 0.0
        all_details = []
        
        for i, verifier in enumerate(self.verifiers):
            result = verifier.verify(code)
            
            if not result.success:
                # Early stop on failure
                return VerifyResult(
                    success=False,
                    reward=total_reward,
                    details=f"Stage {i+1} failed: {result.details}",
                    error=result.error,
                    metadata={"failed_stage": i, "stage_result": result}
                )
            
            total_reward += result.reward * self.weights[i]
            all_details.append(f"Stage {i+1}: {result.details}")
        
        return VerifyResult(
            success=True,
            reward=total_reward,
            details="; ".join(all_details),
            metadata={"stages_passed": len(self.verifiers)}
        )
    
    def cleanup(self):
        """Cleanup all verifiers."""
        for verifier in self.verifiers:
            verifier.cleanup()

