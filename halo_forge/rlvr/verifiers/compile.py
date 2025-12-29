"""
Compilation Verifiers

Verify code by attempting to compile it.
Supports GCC (Linux) and MinGW (cross-compile to Windows).
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional, List

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult


class CompileVerifier(Verifier):
    """
    Base class for compilation-based verification.
    
    Subclasses should set the compiler command and flags.
    """
    
    def __init__(
        self,
        compiler: str,
        flags: Optional[List[str]] = None,
        timeout: int = 30,
        max_workers: int = 8
    ):
        """
        Initialize compile verifier.
        
        Args:
            compiler: Compiler command (e.g., 'g++', 'x86_64-w64-mingw32-g++')
            flags: Compiler flags
            timeout: Compilation timeout in seconds
            max_workers: Max parallel compilations
        """
        super().__init__(max_workers=max_workers)
        self.compiler = compiler
        self.flags = flags or []
        self.timeout = timeout
    
    def verify(self, code: str) -> VerifyResult:
        """
        Verify code by compiling it.
        
        Args:
            code: C/C++ source code
            
        Returns:
            VerifyResult with compilation status
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
            # Build compile command
            cmd = [self.compiler] + self.flags + [source_file, '-o', output_file]
            
            # Run compiler
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
                    details=f"Compilation successful with {self.compiler}",
                    metadata={"compiler": self.compiler}
                )
            else:
                # Extract first few error lines
                error_lines = result.stderr.strip().split('\n')[:5]
                error_summary = '\n'.join(error_lines)
                
                return VerifyResult(
                    success=False,
                    reward=0.0,
                    details="Compilation failed",
                    error=error_summary,
                    metadata={"compiler": self.compiler, "returncode": result.returncode}
                )
                
        except subprocess.TimeoutExpired:
            return VerifyResult(
                success=False,
                reward=0.0,
                details="Compilation timeout",
                error=f"Compilation exceeded {self.timeout}s timeout"
            )
        except FileNotFoundError:
            return VerifyResult(
                success=False,
                reward=0.0,
                details="Compiler not found",
                error=f"Compiler '{self.compiler}' not found in PATH"
            )
        except Exception as e:
            return VerifyResult(
                success=False,
                reward=0.0,
                details="Compilation error",
                error=str(e)
            )
        finally:
            # Cleanup temp files
            if os.path.exists(source_file):
                os.unlink(source_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
    
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
    
    Example:
        verifier = GCCVerifier()
        result = verifier.verify(code)
    """
    
    def __init__(
        self,
        flags: Optional[List[str]] = None,
        timeout: int = 30,
        max_workers: int = 8
    ):
        """
        Initialize GCC verifier.
        
        Args:
            flags: Compiler flags (default: -w -O2)
            timeout: Compilation timeout
            max_workers: Max parallel compilations
        """
        default_flags = ['-w', '-O2']
        super().__init__(
            compiler='g++',
            flags=flags or default_flags,
            timeout=timeout,
            max_workers=max_workers
        )


class MinGWVerifier(CompileVerifier):
    """
    MinGW cross-compiler for Windows PE executables.
    
    Compiles Linux-hosted code to Windows executables.
    Useful for Windows API code verification on Linux.
    
    Example:
        verifier = MinGWVerifier()
        result = verifier.verify(windows_api_code)
    """
    
    def __init__(
        self,
        flags: Optional[List[str]] = None,
        timeout: int = 30,
        max_workers: int = 8
    ):
        """
        Initialize MinGW verifier.
        
        Args:
            flags: Compiler flags (default: Windows-optimized flags)
            timeout: Compilation timeout
            max_workers: Max parallel compilations
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
            max_workers=max_workers
        )


class ClangVerifier(CompileVerifier):
    """
    Clang/Clang++ compiler.
    
    Alternative to GCC with different error messages.
    """
    
    def __init__(
        self,
        flags: Optional[List[str]] = None,
        timeout: int = 30,
        max_workers: int = 8
    ):
        default_flags = ['-w', '-O2']
        super().__init__(
            compiler='clang++',
            flags=flags or default_flags,
            timeout=timeout,
            max_workers=max_workers
        )

