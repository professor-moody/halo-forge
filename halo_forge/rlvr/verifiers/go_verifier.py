"""
Go Verifier

Verify Go code using `go build` with graduated rewards.

Graduated Rewards:
- 0.0: Does not compile
- 0.3: Compiles with warnings (Go rarely has warnings, uses errors)
- 0.5: Compiles clean
- 0.7: Runs without crash
- 1.0: Produces correct output
"""

import subprocess
import tempfile
import shutil
import os
import resource
from pathlib import Path
from typing import Optional, List
import re

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult, RewardLevel


class GoVerifier(Verifier):
    """
    Go verifier using `go build`.
    
    Creates a temporary Go module, writes code to main.go,
    and compiles using `go build`.
    
    Example:
        # Compile only
        verifier = GoVerifier()
        result = verifier.verify(go_code)
        
        # Compile and run
        verifier = GoVerifier(run_after_compile=True)
        
        # Compile, run, check output
        verifier = GoVerifier(
            run_after_compile=True,
            expected_output="Hello, World!"
        )
    """
    
    def __init__(
        self,
        timeout: int = 30,
        max_workers: int = 8,
        run_after_compile: bool = False,
        run_timeout: int = 5,
        expected_output: Optional[str] = None,
        stdin_input: Optional[str] = None,
        memory_limit_mb: int = 256
    ):
        """
        Initialize Go verifier.
        
        Args:
            timeout: go build timeout in seconds
            max_workers: Max parallel verifications
            run_after_compile: If True, run the compiled binary
            run_timeout: Execution timeout in seconds
            expected_output: If provided, compare output to this string
            stdin_input: Input to provide to stdin when running
            memory_limit_mb: Memory limit for execution
        """
        super().__init__(max_workers=max_workers)
        self.timeout = timeout
        self.run_after_compile = run_after_compile
        self.run_timeout = run_timeout
        self.expected_output = expected_output
        self.stdin_input = stdin_input
        self.memory_limit_mb = memory_limit_mb
    
    def verify(self, code: str) -> VerifyResult:
        """
        Verify Go code by compiling with go build.
        
        Args:
            code: Go source code
            
        Returns:
            VerifyResult with compilation/execution status
        """
        # Extract code from model output
        extracted = self.extract_code(code)
        
        # Create temporary directory for Go module
        project_dir = tempfile.mkdtemp(prefix="go_verify_")
        
        try:
            # Initialize Go module
            self._init_module(project_dir)
            
            # Write source code
            (Path(project_dir) / "main.go").write_text(extracted)
            
            # Step 1: Compile
            compile_result = self._compile(project_dir)
            
            if not compile_result['success']:
                return VerifyResult(
                    success=False,
                    reward=RewardLevel.FAILURE.value,
                    details="Compilation failed",
                    error=compile_result['error'],
                    metadata={
                        "compiler": "go build",
                        "stage": "compile"
                    }
                )
            
            # Go doesn't have warnings (it treats them as errors)
            # So compilation success means clean compile
            if not self.run_after_compile:
                return VerifyResult(
                    success=True,
                    reward=RewardLevel.COMPILE_CLEAN.value,
                    details="Compilation successful",
                    metadata={
                        "compiler": "go build",
                        "stage": "compile"
                    }
                )
            
            # Step 2: Run the binary
            binary_path = Path(project_dir) / "verify_code"
            run_result = self._run(str(binary_path))
            
            if not run_result['success']:
                return VerifyResult(
                    success=False,
                    reward=RewardLevel.COMPILE_CLEAN.value,
                    details="Compiled but crashed during execution",
                    error=run_result['error'],
                    metadata={
                        "compiler": "go build",
                        "stage": "run",
                        "exit_code": run_result.get('exit_code')
                    }
                )
            
            # Step 3: Check output if expected
            if self.expected_output is not None:
                actual_output = run_result['stdout'].strip()
                expected = self.expected_output.strip()
                
                if actual_output == expected:
                    return VerifyResult(
                        success=True,
                        reward=RewardLevel.CORRECT_OUTPUT.value,
                        details="Compiled, ran, and produced correct output",
                        metadata={
                            "compiler": "go build",
                            "stage": "output_check",
                            "output": actual_output
                        }
                    )
                else:
                    return VerifyResult(
                        success=False,
                        reward=RewardLevel.RUNS_NO_CRASH.value,
                        details="Ran but output incorrect",
                        error=f"Expected: {expected[:100]}\nGot: {actual_output[:100]}",
                        metadata={
                            "compiler": "go build",
                            "stage": "output_check",
                            "expected": expected,
                            "actual": actual_output
                        }
                    )
            
            # Ran successfully, no output check
            return VerifyResult(
                success=True,
                reward=RewardLevel.RUNS_NO_CRASH.value,
                details="Compiled and ran successfully",
                metadata={
                    "compiler": "go build",
                    "stage": "run",
                    "stdout": run_result['stdout'][:500]
                }
            )
            
        except subprocess.TimeoutExpired:
            return VerifyResult(
                success=False,
                reward=RewardLevel.FAILURE.value,
                details="Timeout",
                error=f"Exceeded {self.timeout}s timeout"
            )
        except FileNotFoundError:
            return VerifyResult(
                success=False,
                reward=RewardLevel.FAILURE.value,
                details="Go not found",
                error="'go' not found in PATH - is Go installed?"
            )
        except Exception as e:
            return VerifyResult(
                success=False,
                reward=RewardLevel.FAILURE.value,
                details="Verification error",
                error=str(e)
            )
        finally:
            # Cleanup temp project
            shutil.rmtree(project_dir, ignore_errors=True)
    
    def _init_module(self, project_dir: str):
        """Initialize a Go module in the project directory."""
        result = subprocess.run(
            ["go", "mod", "init", "verify_code"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to init go module: {result.stderr}")
    
    def _compile(self, project_dir: str) -> dict:
        """
        Compile Go code.
        
        Returns:
            dict with 'success', 'error' keys
        """
        cmd = ["go", "build", "-o", "verify_code", "."]
        
        result = subprocess.run(
            cmd,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=self.timeout
        )
        
        if result.returncode == 0:
            return {'success': True}
        else:
            # Extract error message
            error_lines = result.stderr.strip().split('\n')[:10]
            return {
                'success': False,
                'error': '\n'.join(error_lines)
            }
    
    def _run(self, executable: str) -> dict:
        """
        Run compiled Go binary with resource limits.
        """
        def set_limits():
            mem_bytes = self.memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            resource.setrlimit(resource.RLIMIT_CPU, (self.run_timeout, self.run_timeout))
        
        try:
            result = subprocess.run(
                [executable],
                input=self.stdin_input,
                capture_output=True,
                text=True,
                timeout=self.run_timeout,
                preexec_fn=set_limits
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'exit_code': 0
                }
            else:
                return {
                    'success': False,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'exit_code': result.returncode,
                    'error': f"Exit code {result.returncode}: {result.stderr[:200]}"
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f"Execution exceeded {self.run_timeout}s timeout",
                'exit_code': -1
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'exit_code': -1
            }
    
    def extract_code(self, text: str) -> str:
        """
        Extract Go code from model output.
        
        Handles:
        - Code blocks with ```go
        - Raw code starting with package main
        """
        # Try markdown code blocks first
        code_pattern = r'```(?:go|golang)?\s*(.*?)```'
        matches = re.findall(code_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Look for Go code patterns
        if 'package main' in text:
            # Extract from package main
            pkg_start = text.find('package main')
            code = text[pkg_start:]
            
            # Try to find the end by looking for main function's closing brace
            main_match = re.search(r'func\s+main\s*\(\s*\)\s*\{', code)
            if main_match:
                brace_count = 1
                for i in range(main_match.end(), len(code)):
                    if code[i] == '{':
                        brace_count += 1
                    elif code[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            return code[:i + 1].strip()
            
            # Fallback: return to last brace
            last_brace = code.rfind('}')
            if last_brace > 0:
                return code[:last_brace + 1].strip()
        
        # Return as-is
        return text.strip()

