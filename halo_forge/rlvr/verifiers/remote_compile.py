"""
Remote Compilation Verifiers

Verify code by compiling on a remote machine via SSH.
Useful for MSVC compilation on Windows from a Linux host.
"""

import subprocess
import tempfile
import os
import uuid
from dataclasses import dataclass
from typing import Optional

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult


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
    and returns the result.
    
    Requirements on Windows host:
    - OpenSSH Server running
    - Visual Studio with MSVC installed
    - init-msvc.ps1 script to load MSVC environment
    - C:\\Binaries\\input and C:\\Binaries\\output directories
    
    Example:
        verifier = RemoteMSVCVerifier(
            host="192.168.1.100",
            user="developer",
            ssh_key="~/.ssh/windows_key"
        )
        result = verifier.verify(code)
    """
    
    def __init__(
        self,
        host: str,
        user: str,
        ssh_key: str,
        input_dir: str = r"C:\Binaries\input",
        output_dir: str = r"C:\Binaries\output",
        timeout: int = 60,
        max_workers: int = 8
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
        """
        super().__init__(max_workers=max_workers)
        self.host = host
        self.user = user
        self.ssh_key = ssh_key
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.timeout = timeout
        
        # SSH common options
        self.ssh_opts = [
            '-i', ssh_key,
            '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'ConnectTimeout=10'
        ]
    
    def verify(self, code: str) -> VerifyResult:
        """
        Verify code by compiling with MSVC on remote Windows.
        
        Args:
            code: C/C++ source code
            
        Returns:
            VerifyResult with compilation status
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
            # Upload source to Windows
            scp_result = self._upload_file(local_source, source_name)
            if not scp_result.success:
                return scp_result
            
            # Compile on Windows
            compile_result = self._compile_remote(source_name, exe_name)
            
            return compile_result
            
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
                    reward=0.0,
                    details="Upload failed",
                    error=result.stderr
                )
            
            return VerifyResult(success=True, reward=0.0, details="Upload successful")
            
        except subprocess.TimeoutExpired:
            return VerifyResult(
                success=False,
                reward=0.0,
                details="Upload timeout",
                error="SCP upload timed out"
            )
        except Exception as e:
            return VerifyResult(
                success=False,
                reward=0.0,
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
# Cleanup
Remove-Item -Path '{win_source}' -ErrorAction SilentlyContinue
Remove-Item -Path '{win_exe}' -ErrorAction SilentlyContinue
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
                    reward=1.0,
                    details="MSVC compilation successful",
                    metadata={"compiler": "MSVC", "host": self.host}
                )
            elif "COMPILE_FAILED" in output:
                error = output.split("|", 1)[1] if "|" in output else "Unknown error"
                return VerifyResult(
                    success=False,
                    reward=0.0,
                    details="MSVC compilation failed",
                    error=error,
                    metadata={"compiler": "MSVC", "host": self.host}
                )
            else:
                return VerifyResult(
                    success=False,
                    reward=0.0,
                    details="Unexpected compilation output",
                    error=output[:200]
                )
                
        except subprocess.TimeoutExpired:
            return VerifyResult(
                success=False,
                reward=0.0,
                details="Compilation timeout",
                error=f"Remote compilation exceeded {self.timeout}s"
            )
        except Exception as e:
            return VerifyResult(
                success=False,
                reward=0.0,
                details="SSH error",
                error=str(e)
            )
    
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

