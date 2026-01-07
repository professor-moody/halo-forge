"""
Execution-Based Verification with Test Cases

Extends compile verification to support multiple test cases with I/O pairs.
Provides graduated rewards based on test case pass rate.

Reward Structure:
- 0.0: Does not compile
- 0.3: Compiles with warnings
- 0.5: Compiles clean (no tests run)
- 0.5-1.0: Graduated based on test pass rate

Usage:
    verifier = ExecutionVerifier(
        test_cases=[
            {"input": "5\\n", "expected": "25"},
            {"input": "10\\n", "expected": "100"},
        ]
    )
    result = verifier.verify(code)
"""

import subprocess
import tempfile
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from halo_forge.rlvr.verifiers.compile import CompileVerifier
from halo_forge.rlvr.verifiers.base import VerifyResult, RewardLevel


@dataclass
class TestCase:
    """A single test case with input and expected output."""
    input: str
    expected: str
    name: str = ""
    timeout: int = 5
    

class ExecutionVerifier(CompileVerifier):
    """
    Verifies code correctness by compiling and running with multiple test cases.
    
    Extends CompileVerifier to support:
    - Multiple test cases per verification
    - Graduated rewards based on pass rate
    - Flexible output matching (exact, contains, regex)
    
    Reward calculation:
    - 0.0: Does not compile
    - 0.5 + 0.5 * (passed/total): Graduated based on test pass rate
    
    Example:
        verifier = ExecutionVerifier(
            compiler='g++',
            test_cases=[
                {"input": "5", "expected": "25"},
                {"input": "10", "expected": "100"},
            ],
            match_mode='exact'
        )
        result = verifier.verify(code)
    """
    
    MATCH_MODES = ['exact', 'contains', 'regex', 'numeric']
    
    def __init__(
        self,
        compiler: str = 'g++',
        flags: Optional[List[str]] = None,
        test_cases: Optional[List[Dict[str, str]]] = None,
        timeout: int = 30,
        run_timeout: int = 5,
        max_workers: int = 8,
        match_mode: str = 'exact',
        partial_credit: bool = True,
        binary_cache_dir: Optional[str] = None
    ):
        """
        Initialize execution verifier.
        
        Args:
            compiler: Compiler command (e.g., 'g++', 'x86_64-w64-mingw32-g++')
            flags: Compiler flags
            test_cases: List of test cases, each with 'input' and 'expected' keys
            timeout: Compilation timeout in seconds
            run_timeout: Per-test execution timeout in seconds
            max_workers: Max parallel workers
            match_mode: Output matching mode ('exact', 'contains', 'regex', 'numeric')
            partial_credit: If True, give credit for partial passes
            binary_cache_dir: Directory to cache compiled binaries
        """
        super().__init__(
            compiler=compiler,
            flags=flags,
            timeout=timeout,
            max_workers=max_workers,
            run_after_compile=True,  # We handle running ourselves
            run_timeout=run_timeout,
            binary_cache_dir=binary_cache_dir
        )
        
        # Override run_after_compile - we handle test case running
        self.run_after_compile = False
        
        self.test_cases = self._parse_test_cases(test_cases or [])
        self.match_mode = match_mode
        self.partial_credit = partial_credit
        self.default_run_timeout = run_timeout
    
    def _parse_test_cases(self, test_cases: List[Dict]) -> List[TestCase]:
        """Parse test case dicts into TestCase objects."""
        parsed = []
        for i, tc in enumerate(test_cases):
            parsed.append(TestCase(
                input=str(tc.get('input', '')),
                expected=str(tc.get('expected', '')),
                name=tc.get('name', f'test_{i+1}'),
                timeout=tc.get('timeout', 5)
            ))
        return parsed
    
    def set_test_cases(self, test_cases: List[Dict]):
        """Update test cases dynamically."""
        self.test_cases = self._parse_test_cases(test_cases)
    
    def verify(self, code: str) -> VerifyResult:
        """
        Verify code by compiling and running all test cases.
        
        Args:
            code: Source code to verify
            
        Returns:
            VerifyResult with graduated reward based on test pass rate
        """
        # Extract code if wrapped
        extracted = self.extract_code(code)
        
        # Determine file extension
        is_cpp = self._is_cpp(extracted)
        suffix = '.cpp' if is_cpp else '.c'
        
        # Output extension
        is_windows_target = 'mingw' in self.compiler.lower()
        output_ext = '.exe' if is_windows_target else '.out'
        
        # Write source to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=suffix,
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(extracted)
            source_file = f.name
        
        output_file = source_file.replace(suffix, output_ext)
        
        try:
            # Step 1: Compile
            compile_result = self._compile(source_file, output_file)
            
            if not compile_result['success']:
                return VerifyResult(
                    success=False,
                    reward=RewardLevel.FAILURE.value,
                    details="Compilation failed",
                    error=compile_result['error'],
                    metadata={
                        "compiler": self.compiler,
                        "stage": "compile"
                    }
                )
            
            has_warnings = bool(compile_result.get('warnings'))
            
            # Cache binary if requested
            cached_path = None
            if self.binary_cache_dir and os.path.exists(output_file):
                cached_path = self._cache_binary(output_file)
            
            # If no test cases, return compile result
            if not self.test_cases:
                reward = RewardLevel.COMPILE_WARNINGS.value if has_warnings else RewardLevel.COMPILE_CLEAN.value
                return VerifyResult(
                    success=True,
                    reward=reward,
                    details="Compiled successfully (no test cases)",
                    metadata={
                        "compiler": self.compiler,
                        "stage": "compile",
                        "warnings": has_warnings,
                        "binary_path": str(cached_path) if cached_path else None
                    }
                )
            
            # Can't run Windows binaries on Linux
            if is_windows_target:
                return VerifyResult(
                    success=True,
                    reward=RewardLevel.COMPILE_CLEAN.value,
                    details="Compiled for Windows (cannot execute on Linux)",
                    metadata={
                        "compiler": self.compiler,
                        "stage": "compile",
                        "binary_path": str(cached_path) if cached_path else None
                    }
                )
            
            # Step 2: Run test cases
            passed = 0
            failed = 0
            results = []
            
            for tc in self.test_cases:
                tc_result = self._run_test_case(output_file, tc)
                results.append(tc_result)
                
                if tc_result['passed']:
                    passed += 1
                else:
                    failed += 1
            
            total = len(self.test_cases)
            pass_rate = passed / total if total > 0 else 0.0
            
            # Calculate graduated reward
            # 0.5 (compiled) + 0.5 * pass_rate
            if self.partial_credit:
                reward = 0.5 + 0.5 * pass_rate
            else:
                reward = 1.0 if passed == total else 0.5
            
            # Success if more than half pass
            success = pass_rate >= 0.5
            
            return VerifyResult(
                success=success,
                reward=reward,
                details=f"Passed {passed}/{total} test cases ({pass_rate*100:.1f}%)",
                metadata={
                    "compiler": self.compiler,
                    "stage": "execution",
                    "passed": passed,
                    "failed": failed,
                    "total": total,
                    "pass_rate": pass_rate,
                    "test_results": results,
                    "binary_path": str(cached_path) if cached_path else None
                }
            )
            
        finally:
            # Cleanup temp files
            try:
                os.unlink(source_file)
                if os.path.exists(output_file):
                    os.unlink(output_file)
            except OSError:
                pass
    
    def _run_test_case(self, binary_path: str, test_case: TestCase) -> Dict[str, Any]:
        """
        Run a single test case.
        
        Args:
            binary_path: Path to compiled binary
            test_case: TestCase to run
            
        Returns:
            Dict with passed, actual, expected, error keys
        """
        try:
            result = subprocess.run(
                [binary_path],
                input=test_case.input,
                capture_output=True,
                text=True,
                timeout=test_case.timeout
            )
            
            actual = result.stdout.strip()
            expected = test_case.expected.strip()
            
            passed = self._compare_output(actual, expected)
            
            return {
                'name': test_case.name,
                'passed': passed,
                'actual': actual[:500],
                'expected': expected[:500],
                'exit_code': result.returncode,
                'stderr': result.stderr[:200] if result.stderr else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                'name': test_case.name,
                'passed': False,
                'actual': None,
                'expected': test_case.expected[:500],
                'error': f'Timeout after {test_case.timeout}s'
            }
        except Exception as e:
            return {
                'name': test_case.name,
                'passed': False,
                'actual': None,
                'expected': test_case.expected[:500],
                'error': str(e)
            }
    
    def _compare_output(self, actual: str, expected: str) -> bool:
        """
        Compare actual output to expected based on match_mode.
        
        Args:
            actual: Actual program output
            expected: Expected output
            
        Returns:
            True if outputs match according to match_mode
        """
        if self.match_mode == 'exact':
            return actual == expected
        
        elif self.match_mode == 'contains':
            return expected in actual
        
        elif self.match_mode == 'regex':
            try:
                return bool(re.match(expected, actual))
            except re.error:
                return actual == expected
        
        elif self.match_mode == 'numeric':
            # Extract numbers and compare
            actual_nums = re.findall(r'-?\d+\.?\d*', actual)
            expected_nums = re.findall(r'-?\d+\.?\d*', expected)
            
            if len(actual_nums) != len(expected_nums):
                return False
            
            for a, e in zip(actual_nums, expected_nums):
                try:
                    if abs(float(a) - float(e)) > 1e-6:
                        return False
                except ValueError:
                    return False
            return True
        
        return actual == expected
    
    def verify_with_prompt(self, code: str, prompt: str) -> VerifyResult:
        """
        Verify code, extracting test cases from prompt metadata.
        
        Args:
            code: Source code to verify
            prompt: Prompt string (may contain embedded test cases)
            
        Returns:
            VerifyResult with test case results
        """
        # Try to extract test cases from prompt
        test_cases = self._extract_test_cases_from_prompt(prompt)
        if test_cases:
            self.set_test_cases(test_cases)
        
        return self.verify(code)
    
    def _extract_test_cases_from_prompt(self, prompt: str) -> List[Dict]:
        """
        Try to extract test cases from prompt text.
        
        Looks for patterns like:
        - Input: 5  Output: 25
        - Example: input=5, output=25
        
        Args:
            prompt: Prompt text
            
        Returns:
            List of test case dicts, or empty list
        """
        test_cases = []
        
        # Pattern 1: Input: ... Output: ...
        pattern1 = r'Input:\s*([^\n]+)\s*Output:\s*([^\n]+)'
        for match in re.finditer(pattern1, prompt, re.IGNORECASE):
            test_cases.append({
                'input': match.group(1).strip(),
                'expected': match.group(2).strip()
            })
        
        # Pattern 2: Example input/output blocks
        pattern2 = r'Example[^:]*:\s*(?:input[=:]?\s*)?([^\n]+)[\n,]\s*(?:output[=:]?\s*)?([^\n]+)'
        for match in re.finditer(pattern2, prompt, re.IGNORECASE):
            test_cases.append({
                'input': match.group(1).strip(),
                'expected': match.group(2).strip()
            })
        
        return test_cases


class GCCExecutionVerifier(ExecutionVerifier):
    """ExecutionVerifier pre-configured for GCC."""
    
    def __init__(
        self,
        test_cases: Optional[List[Dict[str, str]]] = None,
        flags: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            compiler='g++',
            flags=flags or ['-O2', '-Wall'],
            test_cases=test_cases,
            **kwargs
        )


class ClangExecutionVerifier(ExecutionVerifier):
    """ExecutionVerifier pre-configured for Clang."""
    
    def __init__(
        self,
        test_cases: Optional[List[Dict[str, str]]] = None,
        flags: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            compiler='clang++',
            flags=flags or ['-O2', '-Wall'],
            test_cases=test_cases,
            **kwargs
        )


class MinGWExecutionVerifier(ExecutionVerifier):
    """ExecutionVerifier pre-configured for MinGW (Windows cross-compile).
    
    Note: Can compile to Windows PE but cannot execute on Linux.
    Test cases will be skipped; only compile verification is performed.
    """
    
    def __init__(
        self,
        test_cases: Optional[List[Dict[str, str]]] = None,
        flags: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            compiler='x86_64-w64-mingw32-g++',
            flags=flags or ['-O2', '-Wall', '-static'],
            test_cases=test_cases,
            **kwargs
        )
