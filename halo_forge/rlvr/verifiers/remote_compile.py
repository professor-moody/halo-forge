"""
Remote Compilation Verifiers

Verify code by compiling on a remote machine via SSH.
Useful for MSVC compilation on Windows from a Linux host.
"""

import subprocess
import tempfile
import os
import uuid
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult, RewardLevel


@dataclass
class RemoteConfig:
    """Configuration for remote SSH connection."""
    host: str
    user: str
    ssh_key: str
    input_dir: str = r"C:\Binaries\input"
    output_dir: str = r"C:\Binaries\output"
    timeout: int = 60


class RemoteMSVCVerifier(Verifier):
    """
    Remote MSVC compiler via SSH.
    
    Uploads code to a Windows machine, compiles with MSVC (cl.exe),
    optionally runs the executable, and returns graduated rewards.
    
    Requirements on Windows host:
    - OpenSSH Server running
    - Visual Studio with MSVC installed
    - MSVC environment loaded in PowerShell profile
    - C:\\Binaries\\input and C:\\Binaries\\output directories
    
    See docs/WINDOWS_SETUP.md for full setup instructions.
    
    Example:
        # Compile-only (default)
        verifier = RemoteMSVCVerifier(
            host="192.168.1.100",
            user="developer",
            ssh_key="~/.ssh/windows_key"
        )
        result = verifier.verify(code)
        
        # Compile and run
        verifier = RemoteMSVCVerifier(
            host="192.168.1.100",
            user="developer",
            ssh_key="~/.ssh/windows_key",
            run_after_compile=True
        )
        
        # Compile, run, and check output
        verifier = RemoteMSVCVerifier(
            host="192.168.1.100",
            user="developer",
            ssh_key="~/.ssh/windows_key",
            run_after_compile=True,
            expected_output="Hello, World!"
        )
        
        # Save compiled binaries for later analysis
        verifier = RemoteMSVCVerifier(
            host="192.168.1.100",
            user="developer",
            ssh_key="~/.ssh/windows_key",
            binary_cache_dir="binaries/msvc"
        )
    
    Reward levels:
    - 0.0: Compile failure
    - 0.5: Compile success (execution disabled or crashed)
    - 0.7: Runs without crash (output not checked or wrong)
    - 1.0: Correct output (matches expected_output)
    """
    
    def __init__(
        self,
        host: str,
        user: str,
        ssh_key: str,
        input_dir: str = r"C:\Binaries\input",
        output_dir: str = r"C:\Binaries\output",
        timeout: int = 60,
        max_workers: int = 8,
        # Execution options
        run_after_compile: bool = False,
        run_timeout: int = 10,
        expected_output: Optional[str] = None,
        stdin_input: Optional[str] = None,
        # Binary caching
        binary_cache_dir: Optional[str] = None,
        keep_remote_binary: bool = False
    ):
        """
        Initialize remote MSVC verifier.
        
        Args:
            host: Windows host IP or hostname
            user: SSH username
            ssh_key: Path to SSH private key
            input_dir: Windows directory for source files
            output_dir: Windows directory for compiled binaries
            timeout: Compilation timeout in seconds
            max_workers: Max parallel compilations
            run_after_compile: If True, execute the compiled .exe
            run_timeout: Execution timeout in seconds
            expected_output: If provided, compare stdout to this string
            stdin_input: Input to provide to stdin when running
            binary_cache_dir: Local directory to cache compiled binaries
            keep_remote_binary: If True, don't delete .exe on Windows
        """
        super().__init__(max_workers=max_workers)
        self.host = host
        self.user = user
        self.ssh_key = os.path.expanduser(ssh_key)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.timeout = timeout
        
        # Execution options
        self.run_after_compile = run_after_compile
        self.run_timeout = run_timeout
        self.expected_output = expected_output
        self.stdin_input = stdin_input
        
        # Binary caching
        self.binary_cache_dir = Path(binary_cache_dir) if binary_cache_dir else None
        self.keep_remote_binary = keep_remote_binary
        
        # Create cache directory if needed
        if self.binary_cache_dir:
            self.binary_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # SSH common options
        self.ssh_opts = [
            '-i', self.ssh_key,
            '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'ConnectTimeout=10'
        ]
    
    def verify(self, code: str) -> VerifyResult:
        """
        Verify code by compiling (and optionally running) on remote Windows.
        
        Args:
            code: C/C++ source code
            
        Returns:
            VerifyResult with graduated reward
        """
        # Extract code if wrapped
        extracted = self.extract_code(code)
        
        # Generate unique filenames
        unique_id = uuid.uuid4().hex[:8]
        source_name = f"sample_{unique_id}.cpp"
        exe_name = f"sample_{unique_id}.exe"
        
        # Write to local temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.cpp',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(extracted)
            local_source = f.name
        
        try:
            # Step 1: Upload source to Windows
            upload_result = self._upload_file(local_source, source_name)
            if not upload_result.success:
                return upload_result
            
            # Step 2: Compile on Windows
            compile_result = self._compile_remote(source_name, exe_name)
            if not compile_result.success:
                self._cleanup_remote(source_name, exe_name)
                return compile_result
            
            # Step 3: Cache binary if requested
            cached_path = None
            if self.binary_cache_dir:
                cached_path = self._download_binary(exe_name)
            
            # Step 4: Run if requested
            if self.run_after_compile:
                run_result = self._run_remote(exe_name)
                
                # Cleanup before returning
                if not self.keep_remote_binary:
                    self._cleanup_remote(source_name, exe_name)
                
                if not run_result['success']:
                    # Compiled but crashed
                    return VerifyResult(
                        success=False,
                        reward=RewardLevel.COMPILE_CLEAN.value,
                        details="Compiled but crashed during execution",
                        error=run_result.get('error', 'Unknown error'),
                        metadata={
                            "compiler": "MSVC",
                            "host": self.host,
                            "stage": "run",
                            "exit_code": run_result.get('exit_code'),
                            "binary_path": str(cached_path) if cached_path else None
                        }
                    )
                
                # Step 5: Check output if expected
                if self.expected_output is not None:
                    actual = run_result.get('stdout', '').strip()
                    expected = self.expected_output.strip()
                    
                    if actual == expected:
                        return VerifyResult(
                            success=True,
                            reward=RewardLevel.CORRECT_OUTPUT.value,
                            details="Compiled, ran, and produced correct output",
                            metadata={
                                "compiler": "MSVC",
                                "host": self.host,
                                "stage": "output_check",
                                "output": actual[:500],
                                "binary_path": str(cached_path) if cached_path else None
                            }
                        )
                    else:
                        return VerifyResult(
                            success=False,
                            reward=RewardLevel.RUNS_NO_CRASH.value,
                            details="Ran but output incorrect",
                            error=f"Expected: {expected[:100]}\nGot: {actual[:100]}",
                            metadata={
                                "compiler": "MSVC",
                                "host": self.host,
                                "stage": "output_check",
                                "expected": expected[:200],
                                "actual": actual[:200],
                                "binary_path": str(cached_path) if cached_path else None
                            }
                        )
                
                # Ran successfully, no output check
                return VerifyResult(
                    success=True,
                    reward=RewardLevel.RUNS_NO_CRASH.value,
                    details="Compiled and ran successfully",
                    metadata={
                        "compiler": "MSVC",
                        "host": self.host,
                        "stage": "run",
                        "stdout": run_result.get('stdout', '')[:500],
                        "binary_path": str(cached_path) if cached_path else None
                    }
                )
            
            # Compile-only mode
            if not self.keep_remote_binary:
                self._cleanup_remote(source_name, exe_name)
            
            return VerifyResult(
                success=True,
                reward=RewardLevel.COMPILE_CLEAN.value,
                details="MSVC compilation successful",
                metadata={
                    "compiler": "MSVC",
                    "host": self.host,
                    "stage": "compile",
                    "binary_path": str(cached_path) if cached_path else None
                }
            )
            
        finally:
            # Cleanup local temp file
            if os.path.exists(local_source):
                os.unlink(local_source)
    
    def _upload_file(self, local_path: str, remote_name: str) -> VerifyResult:
        """Upload file to Windows via SCP."""
        remote_path = f"{self.user}@{self.host}:{self.input_dir}/{remote_name}"
        
        cmd = ['scp'] + self.ssh_opts + [local_path, remote_path]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return VerifyResult(
                    success=False,
                    reward=RewardLevel.FAILURE.value,
                    details="Upload failed",
                    error=result.stderr
                )
            
            return VerifyResult(success=True, reward=0.0, details="Upload successful")
            
        except subprocess.TimeoutExpired:
            return VerifyResult(
                success=False,
                reward=RewardLevel.FAILURE.value,
                details="Upload timeout",
                error="SCP upload timed out"
            )
        except Exception as e:
            return VerifyResult(
                success=False,
                reward=RewardLevel.FAILURE.value,
                details="Upload error",
                error=str(e)
            )
    
    def _compile_remote(self, source_name: str, exe_name: str) -> VerifyResult:
        """Compile on Windows via SSH."""
        win_source = f'{self.input_dir}\\{source_name}'
        win_exe = f'{self.output_dir}\\{exe_name}'
        
        # PowerShell command to compile with MSVC
        ps_cmd = f'''
$clArgs = @('/nologo', '/EHsc', '/O2', '/MT', '/W3', '/DNDEBUG', '/Fe:{win_exe}', '{win_source}')
$output = & cl.exe @clArgs 2>&1
if ($LASTEXITCODE -eq 0) {{
    Write-Output "COMPILE_SUCCESS"
}} else {{
    $err = ($output | Where-Object {{ $_ -match 'error' }} | Select-Object -First 3) -join '; '
    Write-Output "COMPILE_FAILED|$err"
}}
'''
        
        ssh_cmd = ['ssh'] + self.ssh_opts + [f'{self.user}@{self.host}', ps_cmd]
        
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            output = result.stdout.strip()
            
            if "COMPILE_SUCCESS" in output:
                return VerifyResult(
                    success=True,
                    reward=RewardLevel.COMPILE_CLEAN.value,
                    details="MSVC compilation successful",
                    metadata={"compiler": "MSVC", "host": self.host}
                )
            elif "COMPILE_FAILED" in output:
                error = output.split("|", 1)[1] if "|" in output else "Unknown error"
                return VerifyResult(
                    success=False,
                    reward=RewardLevel.FAILURE.value,
                    details="MSVC compilation failed",
                    error=error,
                    metadata={"compiler": "MSVC", "host": self.host}
                )
            else:
                return VerifyResult(
                    success=False,
                    reward=RewardLevel.FAILURE.value,
                    details="Unexpected compilation output",
                    error=output[:200]
                )
                
        except subprocess.TimeoutExpired:
            return VerifyResult(
                success=False,
                reward=RewardLevel.FAILURE.value,
                details="Compilation timeout",
                error=f"Remote compilation exceeded {self.timeout}s"
            )
        except Exception as e:
            return VerifyResult(
                success=False,
                reward=RewardLevel.FAILURE.value,
                details="SSH error",
                error=str(e)
            )
    
    def _run_remote(self, exe_name: str) -> dict:
        """Execute compiled binary on Windows via SSH."""
        win_exe = f'{self.output_dir}\\{exe_name}'
        
        # Build execution command
        if self.stdin_input:
            # Echo input to the process
            escaped_input = self.stdin_input.replace('"', '`"')
            ps_cmd = f'''
$output = echo "{escaped_input}" | & '{win_exe}' 2>&1
$exit = $LASTEXITCODE
Write-Output "EXIT_CODE|$exit"
Write-Output "STDOUT|$output"
'''
        else:
            ps_cmd = f'''
$output = & '{win_exe}' 2>&1
$exit = $LASTEXITCODE
Write-Output "EXIT_CODE|$exit"
Write-Output "STDOUT|$output"
'''
        
        ssh_cmd = ['ssh'] + self.ssh_opts + [f'{self.user}@{self.host}', ps_cmd]
        
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=self.run_timeout
            )
            
            output = result.stdout.strip()
            
            # Parse exit code
            exit_code = None
            stdout = ""
            
            for line in output.split('\n'):
                if line.startswith('EXIT_CODE|'):
                    try:
                        exit_code = int(line.split('|', 1)[1])
                    except ValueError:
                        pass
                elif line.startswith('STDOUT|'):
                    stdout = line.split('|', 1)[1] if '|' in line else ""
            
            if exit_code == 0:
                return {'success': True, 'stdout': stdout, 'exit_code': 0}
            else:
                return {
                    'success': False,
                    'error': f"Process exited with code {exit_code}",
                    'stdout': stdout,
                    'exit_code': exit_code
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f"Execution timeout (>{self.run_timeout}s)",
                'exit_code': -1
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'exit_code': -1}
    
    def _download_binary(self, exe_name: str) -> Optional[Path]:
        """Download compiled binary from Windows via SCP."""
        if not self.binary_cache_dir:
            return None
        
        win_exe = f'{self.output_dir}\\{exe_name}'
        remote_path = f"{self.user}@{self.host}:{win_exe}"
        local_path = self.binary_cache_dir / exe_name
        
        cmd = ['scp'] + self.ssh_opts + [remote_path, str(local_path)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and local_path.exists():
                return local_path
        except Exception:
            pass
        
        return None
    
    def _cleanup_remote(self, source_name: str, exe_name: str):
        """Clean up source and binary on Windows."""
        win_source = f'{self.input_dir}\\{source_name}'
        win_exe = f'{self.output_dir}\\{exe_name}'
        
        ps_cmd = f'''
Remove-Item -Path '{win_source}' -ErrorAction SilentlyContinue
Remove-Item -Path '{win_exe}' -ErrorAction SilentlyContinue
'''
        
        ssh_cmd = ['ssh'] + self.ssh_opts + [f'{self.user}@{self.host}', ps_cmd]
        
        try:
            subprocess.run(ssh_cmd, capture_output=True, timeout=10)
        except Exception:
            pass
    
    def test_connection(self) -> VerifyResult:
        """Test SSH connection to Windows host."""
        ssh_cmd = ['ssh'] + self.ssh_opts + [
            f'{self.user}@{self.host}',
            'echo CONNECTION_OK'
        ]
        
        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
            
            if "CONNECTION_OK" in result.stdout:
                return VerifyResult(
                    success=True,
                    reward=1.0,
                    details=f"Connected to {self.host}"
                )
            else:
                return VerifyResult(
                    success=False,
                    reward=0.0,
                    details="Connection failed",
                    error=result.stderr
                )
        except Exception as e:
            return VerifyResult(
                success=False,
                reward=0.0,
                details="Connection error",
                error=str(e)
            )
    
    def test_msvc(self) -> VerifyResult:
        """Test that MSVC is available on remote host."""
        test_code = '''#include <windows.h>
#include <stdio.h>
int main() {
    printf("Hello from MSVC!\\n");
    return 0;
}
'''
        return self.verify(test_code)
