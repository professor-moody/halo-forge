"""
Compilation Verifiers

Verify code by attempting to compile it, and optionally run the compiled binary.
Supports GCC (Linux), MinGW (cross-compile to Windows), and Clang.

Graduated Rewards:
- 0.0: Does not compile
- 0.3: Compiles with warnings
- 0.5: Compiles clean
- 0.7: Runs without crash (if run_after_compile=True)
- 1.0: Produces correct output (if expected_output provided)
"""

import subprocess
import tempfile
import os
import resource
from pathlib import Path
from typing import Optional, List, Tuple

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult, RewardLevel


class CompileVerifier(Verifier):
    """
    Base class for compilation-based verification.
    
    Supports:
    - Compile-only verification (default)
    - Compile + run verification
    - Compile + run + output comparison
    
    Graduated rewards based on verification stage reached.
    """
    
    def __init__(
        self,
        compiler: str,
        flags: Optional[List[str]] = None,
        timeout: int = 30,
        max_workers: int = 8,
        run_after_compile: bool = False,
        run_timeout: int = 5,
        expected_output: Optional[str] = None,
        stdin_input: Optional[str] = None,
        memory_limit_mb: int = 256,
        warn_as_error: bool = False
    ):
        """
        Initialize compile verifier.
        
        Args:
            compiler: Compiler command (e.g., 'g++', 'x86_64-w64-mingw32-g++')
            flags: Compiler flags
            timeout: Compilation timeout in seconds
            max_workers: Max parallel compilations
            run_after_compile: If True, run the compiled binary
            run_timeout: Execution timeout in seconds
            expected_output: If provided, compare output to this string
            stdin_input: Input to provide to stdin when running
            memory_limit_mb: Memory limit for execution in MB
            warn_as_error: If True, warnings reduce reward
        """
        super().__init__(max_workers=max_workers)
        self.compiler = compiler
        self.flags = flags or []
        self.timeout = timeout
        self.run_after_compile = run_after_compile
        self.run_timeout = run_timeout
        self.expected_output = expected_output
        self.stdin_input = stdin_input
        self.memory_limit_mb = memory_limit_mb
        self.warn_as_error = warn_as_error
    
    def verify(self, code: str) -> VerifyResult:
        """
        Verify code by compiling (and optionally running) it.
        
        Returns graduated rewards based on verification stage:
        - 0.0: Does not compile
        - 0.3: Compiles with warnings (if warn_as_error=True)
        - 0.5: Compiles clean
        - 0.7: Runs without crash
        - 1.0: Produces correct output
        
        Args:
            code: C/C++ source code
            
        Returns:
            VerifyResult with compilation/execution status
        """
        # Extract code if wrapped in model output
        extracted = self.extract_code(code)
        
        # Determine file extension
        is_cpp = self._is_cpp(extracted)
        suffix = '.cpp' if is_cpp else '.c'
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=suffix,
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(extracted)
            source_file = f.name
        
        output_file = source_file.replace(suffix, '.out')
        
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
            
            # Check for warnings
            has_warnings = bool(compile_result.get('warnings'))
            
            # If not running after compile, return compile result
            if not self.run_after_compile:
                if has_warnings and self.warn_as_error:
                    return VerifyResult(
                        success=True,
                        reward=RewardLevel.COMPILE_WARNINGS.value,
                        details=f"Compiled with warnings: {self.compiler}",
                        metadata={
                            "compiler": self.compiler,
                            "warnings": compile_result.get('warnings'),
                            "stage": "compile"
                        }
                    )
                return VerifyResult(
                    success=True,
                    reward=RewardLevel.COMPILE_CLEAN.value,
                    details=f"Compilation successful: {self.compiler}",
                    metadata={
                        "compiler": self.compiler,
                        "stage": "compile"
                    }
                )
            
            # Step 2: Run (if enabled)
            run_result = self._run(output_file)
            
            if not run_result['success']:
                return VerifyResult(
                    success=False,
                    reward=RewardLevel.COMPILE_CLEAN.value,
                    details="Compiled but crashed during execution",
                    error=run_result['error'],
                    metadata={
                        "compiler": self.compiler,
                        "stage": "run",
                        "exit_code": run_result.get('exit_code')
                    }
                )
            
            # Step 3: Check output (if expected_output provided)
            if self.expected_output is not None:
                actual_output = run_result['stdout'].strip()
                expected = self.expected_output.strip()
                
                if actual_output == expected:
                    return VerifyResult(
                        success=True,
                        reward=RewardLevel.CORRECT_OUTPUT.value,
                        details="Compiled, ran, and produced correct output",
                        metadata={
                            "compiler": self.compiler,
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
                            "compiler": self.compiler,
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
                    "compiler": self.compiler,
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
                details="Compiler not found",
                error=f"Compiler '{self.compiler}' not found in PATH"
            )
        except Exception as e:
            return VerifyResult(
                success=False,
                reward=RewardLevel.FAILURE.value,
                details="Verification error",
                error=str(e)
            )
        finally:
            # Cleanup temp files
            if os.path.exists(source_file):
                os.unlink(source_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def _compile(self, source_file: str, output_file: str) -> dict:
        """
        Compile source file.
        
        Returns:
            dict with 'success', 'error', 'warnings' keys
        """
        # Use -Wall to capture warnings if warn_as_error is enabled
        flags = self.flags.copy()
        if self.warn_as_error and '-Wall' not in flags and '-w' not in flags:
            flags.append('-Wall')
        
        cmd = [self.compiler] + flags + [source_file, '-o', output_file]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout
        )
        
        if result.returncode == 0:
            # Check for warnings in stderr
            warnings = result.stderr.strip() if result.stderr else None
            return {
                'success': True,
                'warnings': warnings
            }
        else:
            error_lines = result.stderr.strip().split('\n')[:5]
            return {
                'success': False,
                'error': '\n'.join(error_lines)
            }
    
    def _run(self, executable: str) -> dict:
        """
        Run compiled executable with resource limits.
        
        Returns:
            dict with 'success', 'stdout', 'stderr', 'exit_code', 'error' keys
        """
        def set_limits():
            """Set resource limits for child process."""
            # Memory limit
            mem_bytes = self.memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            # CPU time limit
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
    
    def _is_cpp(self, code: str) -> bool:
        """Detect if code is C++ (vs C)."""
        cpp_indicators = [
            'iostream', 'std::', 'namespace', 'class ', 'template',
            'vector<', 'string>', 'cout', 'cin', 'nullptr'
        ]
        return any(indicator in code for indicator in cpp_indicators)


class GCCVerifier(CompileVerifier):
    """
    GCC/G++ compiler for Linux code.
    
    Supports compile-only or compile+run verification.
    
    Example:
        # Compile only
        verifier = GCCVerifier()
        result = verifier.verify(code)
        
        # Compile and run
        verifier = GCCVerifier(run_after_compile=True)
        
        # Compile, run, and check output
        verifier = GCCVerifier(
            run_after_compile=True,
            expected_output="Hello, World!",
            stdin_input="test input"
        )
    """
    
    def __init__(
        self,
        flags: Optional[List[str]] = None,
        timeout: int = 30,
        max_workers: int = 8,
        run_after_compile: bool = False,
        run_timeout: int = 5,
        expected_output: Optional[str] = None,
        stdin_input: Optional[str] = None,
        memory_limit_mb: int = 256,
        warn_as_error: bool = False
    ):
        """
        Initialize GCC verifier.
        
        Args:
            flags: Compiler flags (default: -w -O2)
            timeout: Compilation timeout
            max_workers: Max parallel compilations
            run_after_compile: If True, run the compiled binary
            run_timeout: Execution timeout in seconds
            expected_output: If provided, compare output to this string
            stdin_input: Input to provide to stdin when running
            memory_limit_mb: Memory limit for execution
            warn_as_error: If True, warnings reduce reward to 0.3
        """
        default_flags = ['-w', '-O2']
        super().__init__(
            compiler='g++',
            flags=flags or default_flags,
            timeout=timeout,
            max_workers=max_workers,
            run_after_compile=run_after_compile,
            run_timeout=run_timeout,
            expected_output=expected_output,
            stdin_input=stdin_input,
            memory_limit_mb=memory_limit_mb,
            warn_as_error=warn_as_error
        )


class MinGWVerifier(CompileVerifier):
    """
    MinGW cross-compiler for Windows PE executables.
    
    Compiles Linux-hosted code to Windows executables.
    Useful for Windows API code verification on Linux.
    
    Note: run_after_compile is not supported for MinGW
    (cannot run Windows executables on Linux without Wine).
    
    Example:
        verifier = MinGWVerifier()
        result = verifier.verify(windows_api_code)
    """
    
    def __init__(
        self,
        flags: Optional[List[str]] = None,
        timeout: int = 30,
        max_workers: int = 8,
        warn_as_error: bool = False
    ):
        """
        Initialize MinGW verifier.
        
        Args:
            flags: Compiler flags (default: Windows-optimized flags)
            timeout: Compilation timeout
            max_workers: Max parallel compilations
            warn_as_error: If True, warnings reduce reward to 0.3
        """
        default_flags = [
            '-static',
            '-Wl,--subsystem,console',
            '-lntdll',
            '-w',
            '-O2'
        ]
        super().__init__(
            compiler='x86_64-w64-mingw32-g++',
            flags=flags or default_flags,
            timeout=timeout,
            max_workers=max_workers,
            run_after_compile=False,  # Cannot run Windows binaries on Linux
            warn_as_error=warn_as_error
        )


class ClangVerifier(CompileVerifier):
    """
    Clang/Clang++ compiler.
    
    Alternative to GCC with different error messages.
    Supports the same options as GCCVerifier.
    """
    
    def __init__(
        self,
        flags: Optional[List[str]] = None,
        timeout: int = 30,
        max_workers: int = 8,
        run_after_compile: bool = False,
        run_timeout: int = 5,
        expected_output: Optional[str] = None,
        stdin_input: Optional[str] = None,
        memory_limit_mb: int = 256,
        warn_as_error: bool = False
    ):
        default_flags = ['-w', '-O2']
        super().__init__(
            compiler='clang++',
            flags=flags or default_flags,
            timeout=timeout,
            max_workers=max_workers,
            run_after_compile=run_after_compile,
            run_timeout=run_timeout,
            expected_output=expected_output,
            stdin_input=stdin_input,
            memory_limit_mb=memory_limit_mb,
            warn_as_error=warn_as_error
        )
