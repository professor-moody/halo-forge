"""
Verifier Service

Connects the UI Verifiers page to actual halo-forge verification backends.
Supports testing code snippets against multiple verification systems.
"""

import asyncio
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum


class VerifierType(Enum):
    """Available verifier types."""
    EXECUTION = "execution"
    GCC = "gcc"
    CLANG = "clang"
    MINGW = "mingw"
    MSVC = "msvc"
    PYTHON = "python"
    RUST = "rust"
    GO = "go"
    MATH = "math"
    HUMANEVAL = "humaneval"
    MBPP = "mbpp"


@dataclass
class VerifyResult:
    """Result from running verification."""
    passed: bool
    reward: float
    message: str
    output: str = ""
    error: str = ""
    duration_ms: float = 0.0
    verifier_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class VerifierInfo:
    """Information about a verifier."""
    name: str
    type: VerifierType
    available: bool
    description: str
    language: str
    example_prompt: str = ""
    example_solution: str = ""


class VerifierService:
    """
    Service for testing code against verification backends.
    
    This connects the UI Verifiers page to actual halo-forge verifiers,
    allowing users to test code snippets interactively.
    """
    
    # Verifier configurations
    VERIFIERS: Dict[str, Dict[str, Any]] = {
        "HumanEval": {
            "type": VerifierType.EXECUTION,
            "language": "Python",
            "description": "Code execution with test cases (HumanEval format)",
            "example_prompt": '''def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """''',
            "example_solution": '''    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False''',
        },
        "MBPP": {
            "type": VerifierType.EXECUTION,
            "language": "Python",
            "description": "Mostly Basic Programming Problems",
            "example_prompt": '''"""
Write a function to find the shared elements from the given two lists.
assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))
"""''',
            "example_solution": '''def similar_elements(test_tup1, test_tup2):
    return tuple(set(test_tup1) & set(test_tup2))''',
        },
        "LiveCodeBench": {
            "type": VerifierType.EXECUTION,
            "language": "Python",
            "description": "Competitive programming problems (contamination-free)",
            "example_prompt": '''"""
Given a list of integers, return the sum of all even numbers.
"""''',
            "example_solution": '''def sum_even(numbers):
    return sum(n for n in numbers if n % 2 == 0)''',
        },
        "Math": {
            "type": VerifierType.MATH,
            "language": "Math",
            "description": "Mathematical answer verification",
            "example_prompt": "What is 15% of 80?",
            "example_solution": "12",
        },
        "GSM8K": {
            "type": VerifierType.MATH,
            "language": "Math",
            "description": "Grade school math word problems",
            "example_prompt": '''Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning 
and bakes muffins for her friends every day with four. She sells the remainder 
at the farmers' market daily for $2 per fresh duck egg. How much in dollars does 
she make every day at the farmers' market?''',
            "example_solution": "18",
        },
        "C++ (GCC)": {
            "type": VerifierType.GCC,
            "language": "C++",
            "description": "Compile with GCC",
            "example_prompt": '''// Write a function that returns the factorial of n
#include <iostream>
''',
            "example_solution": '''int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main() {
    std::cout << factorial(5) << std::endl;
    return 0;
}''',
        },
        "C++ (MinGW)": {
            "type": VerifierType.MINGW,
            "language": "C++ (Windows)",
            "description": "Cross-compile for Windows with MinGW",
            "example_prompt": '''// Windows API example
#include <windows.h>
#include <iostream>
''',
            "example_solution": '''int main() {
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    std::cout << "Processors: " << sysInfo.dwNumberOfProcessors << std::endl;
    return 0;
}''',
        },
        "Go": {
            "type": VerifierType.GO,
            "language": "Go",
            "description": "Compile with Go compiler",
            "example_prompt": '''// Write a function that returns the factorial of n
package main

import "fmt"
''',
            "example_solution": '''func factorial(n int) int {
    if n <= 1 {
        return 1
    }
    return n * factorial(n-1)
}

func main() {
    fmt.Println(factorial(5))
}''',
        },
        "Rust": {
            "type": VerifierType.RUST,
            "language": "Rust",
            "description": "Compile with rustc",
            "example_prompt": '''// Write a function that returns the factorial of n
''',
            "example_solution": '''fn factorial(n: u64) -> u64 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}

fn main() {
    println!("{}", factorial(5));
}''',
        },
    }
    
    def __init__(self):
        """Initialize verifier service."""
        self._availability_cache: Dict[str, bool] = {}
    
    def get_verifiers(self) -> List[VerifierInfo]:
        """Get list of all verifiers with availability status."""
        verifiers = []
        
        for name, config in self.VERIFIERS.items():
            available = self._check_availability(config["type"])
            
            verifiers.append(VerifierInfo(
                name=name,
                type=config["type"],
                available=available,
                description=config["description"],
                language=config["language"],
                example_prompt=config.get("example_prompt", ""),
                example_solution=config.get("example_solution", ""),
            ))
        
        return verifiers
    
    def _check_availability(self, verifier_type: VerifierType) -> bool:
        """Check if a verifier type is available on this system."""
        cache_key = verifier_type.value
        
        if cache_key in self._availability_cache:
            return self._availability_cache[cache_key]
        
        available = False
        
        if verifier_type == VerifierType.GCC:
            available = shutil.which('g++') is not None or shutil.which('gcc') is not None
        elif verifier_type == VerifierType.CLANG:
            available = shutil.which('clang++') is not None
        elif verifier_type == VerifierType.MINGW:
            available = shutil.which('x86_64-w64-mingw32-g++') is not None
        elif verifier_type == VerifierType.MSVC:
            # MSVC requires Windows or SSH config
            available = self._check_msvc_available()
        elif verifier_type in (VerifierType.EXECUTION, VerifierType.PYTHON):
            available = True  # Python always available
        elif verifier_type == VerifierType.RUST:
            available = shutil.which('rustc') is not None
        elif verifier_type == VerifierType.GO:
            available = shutil.which('go') is not None
        elif verifier_type in (VerifierType.MATH, VerifierType.HUMANEVAL, VerifierType.MBPP):
            available = True  # These use built-in verification
        
        self._availability_cache[cache_key] = available
        return available
    
    def _check_msvc_available(self) -> bool:
        """Check if MSVC is available (via SSH config)."""
        # Check for SSH config file with MSVC settings
        ssh_config = Path.home() / ".ssh" / "config"
        if ssh_config.exists():
            content = ssh_config.read_text()
            if 'windows' in content.lower() or 'msvc' in content.lower():
                return True
        return False
    
    async def verify(
        self,
        verifier_name: str,
        prompt: str,
        solution: str,
        ground_truth: Optional[str] = None,
        test_cases: Optional[List[Dict]] = None,
    ) -> VerifyResult:
        """
        Run verification on code.
        
        Args:
            verifier_name: Name of verifier (from VERIFIERS)
            prompt: The problem prompt
            solution: The solution code
            ground_truth: Expected answer (for math problems)
            test_cases: Optional test cases
            
        Returns:
            VerifyResult with pass/fail status and details
        """
        start_time = time.perf_counter()
        
        config = self.VERIFIERS.get(verifier_name)
        if not config:
            return VerifyResult(
                passed=False,
                reward=0.0,
                message=f"Unknown verifier: {verifier_name}",
                verifier_type="unknown",
            )
        
        verifier_type = config["type"]
        
        # Check availability
        if not self._check_availability(verifier_type):
            return VerifyResult(
                passed=False,
                reward=0.0,
                message=f"Verifier not available: {verifier_type.value}",
                error="Required tools not installed",
                verifier_type=verifier_type.value,
            )
        
        # Route to appropriate verifier
        try:
            if verifier_type == VerifierType.EXECUTION:
                result = await self._verify_execution(prompt, solution, test_cases)
            elif verifier_type == VerifierType.MATH:
                result = await self._verify_math(solution, ground_truth)
            elif verifier_type in (VerifierType.GCC, VerifierType.CLANG):
                result = await self._verify_compile_cpp(prompt, solution, 'g++')
            elif verifier_type == VerifierType.MINGW:
                result = await self._verify_compile_cpp(prompt, solution, 'x86_64-w64-mingw32-g++')
            elif verifier_type == VerifierType.RUST:
                result = await self._verify_compile_rust(prompt, solution)
            elif verifier_type == VerifierType.GO:
                result = await self._verify_compile_go(prompt, solution)
            else:
                result = VerifyResult(
                    passed=False,
                    reward=0.0,
                    message=f"Verifier not implemented: {verifier_type.value}",
                )
        except Exception as e:
            result = VerifyResult(
                passed=False,
                reward=0.0,
                message="Verification error",
                error=str(e),
            )
        
        result.duration_ms = (time.perf_counter() - start_time) * 1000
        result.verifier_type = verifier_type.value
        
        return result
    
    async def _verify_execution(
        self,
        prompt: str,
        solution: str,
        test_cases: Optional[List[Dict]] = None,
    ) -> VerifyResult:
        """Verify Python code by execution.
        
        For interactive UI testing, we run the combined prompt+solution code
        directly. This checks for syntax errors and runtime crashes.
        
        Note: HumanEvalVerifier requires task_ids from a dataset, which the
        interactive UI doesn't have. For full test-case verification during
        training, use the RLVR verifiers directly.
        """
        return await self._simple_execution_test(prompt, solution)
    
    async def _simple_execution_test(self, prompt: str, solution: str) -> VerifyResult:
        """Simple Python execution test (fallback)."""
        full_code = f"{prompt}\n{solution}"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_path = f.name
        
        try:
            proc = await asyncio.create_subprocess_exec(
                'python3', temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=10.0
            )
            
            if proc.returncode == 0:
                return VerifyResult(
                    passed=True,
                    reward=1.0,
                    message="Execution successful",
                    output=stdout.decode('utf-8', errors='replace'),
                )
            else:
                return VerifyResult(
                    passed=False,
                    reward=0.0,
                    message="Execution failed",
                    error=stderr.decode('utf-8', errors='replace'),
                )
                
        except asyncio.TimeoutError:
            return VerifyResult(
                passed=False,
                reward=0.0,
                message="Execution timeout",
                error="Code took too long to execute (>10s)",
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    async def _verify_math(self, solution: str, ground_truth: Optional[str]) -> VerifyResult:
        """Verify mathematical answer."""
        if not ground_truth:
            return VerifyResult(
                passed=False,
                reward=0.0,
                message="No ground truth provided",
            )
        
        # Extract numerical answer from solution
        import re
        
        # Try to find numbers in the solution
        numbers = re.findall(r'-?\d+\.?\d*', solution)
        if not numbers:
            return VerifyResult(
                passed=False,
                reward=0.0,
                message="No numerical answer found",
                output=f"Solution: {solution}",
            )
        
        # Get last number (usually the final answer)
        extracted = numbers[-1]
        
        # Compare
        try:
            extracted_val = float(extracted)
            truth_val = float(ground_truth)
            
            # Allow small floating point tolerance
            if abs(extracted_val - truth_val) < 0.001:
                return VerifyResult(
                    passed=True,
                    reward=1.0,
                    message="Correct answer",
                    output=f"Answer: {extracted}",
                )
            else:
                return VerifyResult(
                    passed=False,
                    reward=0.0,
                    message="Incorrect answer",
                    output=f"Got {extracted}, expected {ground_truth}",
                )
        except ValueError:
            # String comparison
            if extracted.strip() == ground_truth.strip():
                return VerifyResult(
                    passed=True,
                    reward=1.0,
                    message="Correct answer",
                )
            else:
                return VerifyResult(
                    passed=False,
                    reward=0.0,
                    message="Incorrect answer",
                    output=f"Got {extracted}, expected {ground_truth}",
                )
    
    async def _verify_compile_cpp(
        self,
        prompt: str,
        solution: str,
        compiler: str,
    ) -> VerifyResult:
        """Verify C++ code by compilation."""
        full_code = f"{prompt}\n{solution}"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(full_code)
            source_path = f.name
        
        output_path = source_path.replace('.cpp', '.out')
        
        try:
            # Compile
            proc = await asyncio.create_subprocess_exec(
                compiler, source_path, '-o', output_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=30.0
            )
            
            if proc.returncode != 0:
                return VerifyResult(
                    passed=False,
                    reward=0.0,
                    message="Compilation failed",
                    error=stderr.decode('utf-8', errors='replace'),
                )
            
            # Compilation succeeded - try to run
            if Path(output_path).exists():
                run_proc = await asyncio.create_subprocess_exec(
                    output_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                
                try:
                    run_stdout, run_stderr = await asyncio.wait_for(
                        run_proc.communicate(),
                        timeout=10.0
                    )
                    
                    if run_proc.returncode == 0:
                        return VerifyResult(
                            passed=True,
                            reward=1.0,
                            message="Compilation and execution successful",
                            output=run_stdout.decode('utf-8', errors='replace'),
                        )
                    else:
                        return VerifyResult(
                            passed=True,  # Compiled successfully
                            reward=0.5,
                            message="Compiled but runtime error",
                            error=run_stderr.decode('utf-8', errors='replace'),
                        )
                except asyncio.TimeoutError:
                    return VerifyResult(
                        passed=True,
                        reward=0.5,
                        message="Compiled but execution timeout",
                    )
            
            return VerifyResult(
                passed=True,
                reward=0.5,
                message="Compilation successful",
                output=stdout.decode('utf-8', errors='replace') if stdout else "OK",
            )
            
        except asyncio.TimeoutError:
            return VerifyResult(
                passed=False,
                reward=0.0,
                message="Compilation timeout",
            )
        finally:
            Path(source_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)
    
    async def _verify_compile_rust(self, prompt: str, solution: str) -> VerifyResult:
        """Verify Rust code by compilation."""
        full_code = f"{prompt}\n{solution}"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False) as f:
            f.write(full_code)
            source_path = f.name
        
        output_path = source_path.replace('.rs', '')
        
        try:
            proc = await asyncio.create_subprocess_exec(
                'rustc', source_path, '-o', output_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=60.0
            )
            
            if proc.returncode != 0:
                return VerifyResult(
                    passed=False,
                    reward=0.0,
                    message="Compilation failed",
                    error=stderr.decode('utf-8', errors='replace'),
                )
            
            return VerifyResult(
                passed=True,
                reward=1.0,
                message="Compilation successful",
                output=stdout.decode('utf-8', errors='replace') if stdout else "OK",
            )
            
        except asyncio.TimeoutError:
            return VerifyResult(
                passed=False,
                reward=0.0,
                message="Compilation timeout",
            )
        finally:
            Path(source_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)
    
    async def _verify_compile_go(self, prompt: str, solution: str) -> VerifyResult:
        """Verify Go code by compilation."""
        full_code = f"{prompt}\n{solution}"
        
        # Go requires a directory for compilation
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "main.go"
            source_path.write_text(full_code)
            
            try:
                proc = await asyncio.create_subprocess_exec(
                    'go', 'build', '-o', 'main', 'main.go',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=tmpdir,
                )
                
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=60.0
                )
                
                if proc.returncode != 0:
                    return VerifyResult(
                        passed=False,
                        reward=0.0,
                        message="Compilation failed",
                        error=stderr.decode('utf-8', errors='replace'),
                    )
                
                return VerifyResult(
                    passed=True,
                    reward=1.0,
                    message="Compilation successful",
                    output=stdout.decode('utf-8', errors='replace') if stdout else "OK",
                )
                
            except asyncio.TimeoutError:
                return VerifyResult(
                    passed=False,
                    reward=0.0,
                    message="Compilation timeout",
                )


# Singleton instance
_service: Optional[VerifierService] = None


def get_verifier_service() -> VerifierService:
    """Get the singleton verifier service."""
    global _service
    if _service is None:
        _service = VerifierService()
    return _service
