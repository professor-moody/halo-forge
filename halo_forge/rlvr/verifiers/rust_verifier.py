"""
Rust Verifier

Verify Rust code using cargo build with graduated rewards.

Graduated Rewards:
- 0.0: Does not compile
- 0.3: Compiles with warnings
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


class RustVerifier(Verifier):
    """
    Rust verifier using cargo.
    
    Creates a temporary Cargo project, writes code to src/main.rs,
    and compiles using `cargo build --release`.
    
    Example:
        # Compile only
        verifier = RustVerifier()
        result = verifier.verify(rust_code)
        
        # Compile and run
        verifier = RustVerifier(run_after_compile=True)
        
        # Compile, run, check output
        verifier = RustVerifier(
            run_after_compile=True,
            expected_output="Hello, World!"
        )
    """
    
    def __init__(
        self,
        timeout: int = 60,
        max_workers: int = 4,
        run_after_compile: bool = False,
        run_timeout: int = 5,
        expected_output: Optional[str] = None,
        stdin_input: Optional[str] = None,
        memory_limit_mb: int = 256,
        warn_as_error: bool = False,
        edition: str = "2021"
    ):
        """
        Initialize Rust verifier.
        
        Args:
            timeout: Cargo build timeout in seconds
            max_workers: Max parallel verifications
            run_after_compile: If True, run the compiled binary
            run_timeout: Execution timeout in seconds
            expected_output: If provided, compare output to this string
            stdin_input: Input to provide to stdin when running
            memory_limit_mb: Memory limit for execution
            warn_as_error: If True, warnings reduce reward
            edition: Rust edition (2018, 2021)
        """
        super().__init__(max_workers=max_workers)
        self.timeout = timeout
        self.run_after_compile = run_after_compile
        self.run_timeout = run_timeout
        self.expected_output = expected_output
        self.stdin_input = stdin_input
        self.memory_limit_mb = memory_limit_mb
        self.warn_as_error = warn_as_error
        self.edition = edition
    
    def verify(self, code: str) -> VerifyResult:
        """
        Verify Rust code by compiling with cargo.
        
        Args:
            code: Rust source code
            
        Returns:
            VerifyResult with compilation/execution status
        """
        # Extract code from model output
        extracted = self.extract_code(code)
        
        # Create temporary cargo project
        project_dir = tempfile.mkdtemp(prefix="rust_verify_")
        src_dir = Path(project_dir) / "src"
        src_dir.mkdir()
        
        try:
            # Write Cargo.toml
            cargo_toml = f'''[package]
name = "verify_code"
version = "0.1.0"
edition = "{self.edition}"

[profile.release]
opt-level = 2
'''
            (Path(project_dir) / "Cargo.toml").write_text(cargo_toml)
            
            # Write source code
            (src_dir / "main.rs").write_text(extracted)
            
            # Step 1: Compile with cargo
            compile_result = self._compile(project_dir)
            
            if not compile_result['success']:
                return VerifyResult(
                    success=False,
                    reward=RewardLevel.FAILURE.value,
                    details="Compilation failed",
                    error=compile_result['error'],
                    metadata={
                        "compiler": "rustc (cargo)",
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
                        details="Compiled with warnings",
                        metadata={
                            "compiler": "rustc (cargo)",
                            "warnings": compile_result.get('warnings'),
                            "stage": "compile"
                        }
                    )
                return VerifyResult(
                    success=True,
                    reward=RewardLevel.COMPILE_CLEAN.value,
                    details="Compilation successful",
                    metadata={
                        "compiler": "rustc (cargo)",
                        "stage": "compile"
                    }
                )
            
            # Step 2: Run the binary
            binary_path = Path(project_dir) / "target" / "release" / "verify_code"
            run_result = self._run(str(binary_path))
            
            if not run_result['success']:
                return VerifyResult(
                    success=False,
                    reward=RewardLevel.COMPILE_CLEAN.value,
                    details="Compiled but crashed during execution",
                    error=run_result['error'],
                    metadata={
                        "compiler": "rustc (cargo)",
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
                            "compiler": "rustc (cargo)",
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
                            "compiler": "rustc (cargo)",
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
                    "compiler": "rustc (cargo)",
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
                details="Cargo not found",
                error="'cargo' not found in PATH - is Rust installed?"
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
    
    def _compile(self, project_dir: str) -> dict:
        """
        Compile Rust project with cargo.
        
        Returns:
            dict with 'success', 'error', 'warnings' keys
        """
        cmd = ["cargo", "build", "--release"]
        
        # Suppress color output for cleaner parsing
        env = os.environ.copy()
        env["CARGO_TERM_COLOR"] = "never"
        
        result = subprocess.run(
            cmd,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            env=env
        )
        
        if result.returncode == 0:
            # Check for warnings in stderr
            warnings = None
            if result.stderr:
                warning_match = re.search(r'warning:', result.stderr)
                if warning_match:
                    warnings = result.stderr.strip()
            return {
                'success': True,
                'warnings': warnings
            }
        else:
            # Extract error message
            error_lines = result.stderr.strip().split('\n')[:10]
            return {
                'success': False,
                'error': '\n'.join(error_lines)
            }
    
    def _run(self, executable: str) -> dict:
        """
        Run compiled Rust binary with resource limits.
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
        Extract Rust code from model output.
        
        Handles:
        - Code blocks with ```rust
        - Raw code starting with fn main or use
        """
        # Try markdown code blocks first
        code_pattern = r'```(?:rust|rs)?\s*(.*?)```'
        matches = re.findall(code_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Look for Rust code patterns
        if 'fn main' in text:
            # Find fn main and extract the function
            main_start = text.find('fn main')
            code = text[main_start:]
            
            # Find matching braces
            brace_start = code.find('{')
            if brace_start > 0:
                brace_count = 1
                for i in range(brace_start + 1, len(code)):
                    if code[i] == '{':
                        brace_count += 1
                    elif code[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Include any use statements before fn main
                            prefix_start = 0
                            for line in text[:main_start].split('\n')[::-1]:
                                if line.strip().startswith('use ') or line.strip().startswith('#['):
                                    prefix_start = text.find(line)
                                    break
                            
                            prefix = text[prefix_start:main_start] if prefix_start > 0 else ""
                            return (prefix + code[:i + 1]).strip()
            
            # Fallback: return from fn main to last brace
            last_brace = code.rfind('}')
            if last_brace > 0:
                return code[:last_brace + 1].strip()
        
        # Return as-is
        return text.strip()


# Alias for clarity
CargoVerifier = RustVerifier

