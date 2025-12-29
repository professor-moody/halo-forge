"""
Test Runner Verifiers

Verify code by running tests against it.
Supports pytest, unittest, and custom test runners.
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional, List

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult


class PytestVerifier(Verifier):
    """
    Verify Python code by running pytest.
    
    Can run tests defined in the code itself or external test files.
    
    Example:
        verifier = PytestVerifier()
        result = verifier.verify(python_code_with_tests)
    """
    
    def __init__(
        self,
        test_file: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        timeout: int = 60,
        max_workers: int = 4
    ):
        """
        Initialize pytest verifier.
        
        Args:
            test_file: Optional external test file to run against the code
            extra_args: Extra arguments to pass to pytest
            timeout: Test timeout in seconds
            max_workers: Max parallel test runs
        """
        super().__init__(max_workers=max_workers)
        self.test_file = test_file
        self.extra_args = extra_args or ['-v', '--tb=short']
        self.timeout = timeout
    
    def verify(self, code: str) -> VerifyResult:
        """
        Verify Python code by running pytest.
        
        Args:
            code: Python code (may include test functions)
            
        Returns:
            VerifyResult with test status
        """
        extracted = self.extract_code(code)
        
        # Create temp directory for test
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write the code
            code_file = Path(tmpdir) / "solution.py"
            code_file.write_text(extracted)
            
            # Determine what to test
            if self.test_file:
                # Run external tests against the code
                test_target = self.test_file
            else:
                # Assume code contains its own tests
                test_target = str(code_file)
            
            # Run pytest
            cmd = ['pytest', test_target] + self.extra_args
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                    env={**os.environ, 'PYTHONPATH': tmpdir}
                )
                
                # Parse pytest output
                passed, failed, total = self._parse_pytest_output(result.stdout)
                
                if result.returncode == 0:
                    return VerifyResult(
                        success=True,
                        reward=1.0,
                        details=f"All {total} tests passed",
                        metadata={"passed": passed, "failed": failed, "total": total}
                    )
                else:
                    # Partial credit based on pass rate
                    pass_rate = passed / total if total > 0 else 0.0
                    
                    return VerifyResult(
                        success=False,
                        reward=pass_rate * 0.5,  # Max 0.5 for partial pass
                        details=f"{passed}/{total} tests passed",
                        error=result.stdout[-500:] if result.stdout else result.stderr[-500:],
                        metadata={"passed": passed, "failed": failed, "total": total}
                    )
                    
            except subprocess.TimeoutExpired:
                return VerifyResult(
                    success=False,
                    reward=0.0,
                    details="Test timeout",
                    error=f"Tests exceeded {self.timeout}s timeout"
                )
            except FileNotFoundError:
                return VerifyResult(
                    success=False,
                    reward=0.0,
                    details="pytest not found",
                    error="pytest is not installed"
                )
            except Exception as e:
                return VerifyResult(
                    success=False,
                    reward=0.0,
                    details="Test error",
                    error=str(e)
                )
    
    def _parse_pytest_output(self, output: str) -> tuple:
        """Parse pytest output to extract pass/fail counts."""
        import re
        
        # Look for summary line like "5 passed, 2 failed"
        summary_pattern = r'(\d+) passed'
        failed_pattern = r'(\d+) failed'
        
        passed_match = re.search(summary_pattern, output)
        failed_match = re.search(failed_pattern, output)
        
        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0
        total = passed + failed
        
        return passed, failed, total


class UnittestVerifier(Verifier):
    """
    Verify Python code by running unittest.
    
    Similar to PytestVerifier but uses Python's built-in unittest.
    """
    
    def __init__(
        self,
        timeout: int = 60,
        max_workers: int = 4
    ):
        super().__init__(max_workers=max_workers)
        self.timeout = timeout
    
    def verify(self, code: str) -> VerifyResult:
        """Verify Python code by running unittest."""
        extracted = self.extract_code(code)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            code_file = Path(tmpdir) / "test_solution.py"
            code_file.write_text(extracted)
            
            cmd = ['python', '-m', 'unittest', 'discover', '-s', tmpdir, '-v']
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                if result.returncode == 0:
                    return VerifyResult(
                        success=True,
                        reward=1.0,
                        details="All tests passed"
                    )
                else:
                    return VerifyResult(
                        success=False,
                        reward=0.0,
                        details="Tests failed",
                        error=result.stderr[-500:]
                    )
                    
            except subprocess.TimeoutExpired:
                return VerifyResult(
                    success=False,
                    reward=0.0,
                    details="Test timeout",
                    error=f"Tests exceeded {self.timeout}s"
                )
            except Exception as e:
                return VerifyResult(
                    success=False,
                    reward=0.0,
                    details="Test error",
                    error=str(e)
                )

