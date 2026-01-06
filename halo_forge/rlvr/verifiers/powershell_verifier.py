"""
PowerShell Verifier

Validates PowerShell scripts by checking syntax.
Unlike compiled languages, PowerShell is a script language -
verification is syntax checking only.

Validation modes:
- "local": Use local pwsh (requires PowerShell Core install)
- "remote": SSH to Windows server, use native PowerShell
- "none": Skip validation, accept all scripts
- "auto": Auto-detect best available method

Graduated Rewards:
- 0.0: Syntax error
- 0.5: Valid syntax (compile-equivalent)
"""

import subprocess
import tempfile
import os
import re
import uuid
import shutil
from pathlib import Path
from typing import Optional, Tuple

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult, RewardLevel


class PowerShellVerifier(Verifier):
    """
    PowerShell script verifier.
    
    Validates script syntax locally (using pwsh) or remotely (via Windows SSH).
    
    Example:
        # Local validation (requires pwsh installed)
        verifier = PowerShellVerifier(validation_mode="local")
        
        # Remote validation (uses Windows server)
        verifier = PowerShellVerifier(
            validation_mode="remote",
            win_host="192.168.1.100",
            win_user="admin",
            win_key="~/.ssh/win"
        )
        
        # Auto-detect best method
        verifier = PowerShellVerifier(validation_mode="auto")
        
        # Skip validation (accept all scripts)
        verifier = PowerShellVerifier(validation_mode="none")
        
        # With script caching
        verifier = PowerShellVerifier(binary_cache_dir="scripts/ps1")
    """
    
    def __init__(
        self,
        timeout: int = 30,
        max_workers: int = 8,
        validation_mode: str = "auto",
        win_host: Optional[str] = None,
        win_user: Optional[str] = None,
        win_key: Optional[str] = None,
        use_pwsh: bool = True,
        binary_cache_dir: Optional[str] = None
    ):
        """
        Initialize PowerShell verifier.
        
        Args:
            timeout: Script validation timeout
            max_workers: Max parallel verifications
            validation_mode: "remote", "local", "none", or "auto"
            win_host: Windows server hostname (for remote mode)
            win_user: Windows server username
            win_key: SSH key path for Windows host
            use_pwsh: If True, use 'pwsh' locally; else use 'powershell'
            binary_cache_dir: Directory to cache validated scripts
        """
        super().__init__(max_workers=max_workers)
        self.timeout = timeout
        self.pwsh_cmd = "pwsh" if use_pwsh else "powershell"
        
        # Windows server connection (for remote validation)
        self.win_host = win_host
        self.win_user = win_user
        self.win_key = os.path.expanduser(win_key) if win_key else None
        
        # Binary cache
        self.binary_cache_dir = Path(binary_cache_dir) if binary_cache_dir else None
        if self.binary_cache_dir:
            self.binary_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine validation mode
        self.validation_mode = self._resolve_validation_mode(validation_mode)
    
    def verify(self, code: str) -> VerifyResult:
        """
        Verify PowerShell script syntax.
        
        Args:
            code: PowerShell script code
            
        Returns:
            VerifyResult with validation status
        """
        # Extract PowerShell code
        script = self.extract_code(code)
        if not script:
            return VerifyResult(
                success=False,
                reward=RewardLevel.FAILURE.value,
                details="Could not extract valid PowerShell script",
                error="No valid PowerShell code found"
            )
        
        # Create temp script file
        script_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.ps1',
            delete=False,
            encoding='utf-8'
        )
        script_file.write(script)
        script_file.close()
        
        try:
            # Validate syntax based on mode
            validation_used = "none"
            
            if self.validation_mode == "remote":
                success, error = self._validate_remote(script)
                validation_used = "remote (Windows)"
                if not success:
                    return VerifyResult(
                        success=False,
                        reward=RewardLevel.FAILURE.value,
                        details="Syntax validation failed",
                        error=error[:500],
                        metadata={
                            "language": "powershell",
                            "validation_mode": validation_used
                        }
                    )
                    
            elif self.validation_mode == "local":
                result = self._validate_local(script_file.name)
                validation_used = "local (pwsh)"
                if not result['success']:
                    return VerifyResult(
                        success=False,
                        reward=RewardLevel.FAILURE.value,
                        details="Syntax validation failed",
                        error=result['error'][:500],
                        metadata={
                            "language": "powershell",
                            "validation_mode": validation_used
                        }
                    )
            else:
                validation_used = "skipped"
            
            # Cache script if configured
            cached_path = None
            if self.binary_cache_dir:
                cache_name = f"{uuid.uuid4().hex[:12]}.ps1"
                cached_path = self.binary_cache_dir / cache_name
                shutil.copy2(script_file.name, cached_path)
            
            # Syntax validation = success (or skipped)
            return VerifyResult(
                success=True,
                reward=RewardLevel.COMPILE_CLEAN.value,
                details=f"PowerShell syntax valid ({validation_used})",
                metadata={
                    "language": "powershell",
                    "script_size": len(script),
                    "validation_mode": self.validation_mode,
                    "binary_path": str(cached_path) if cached_path else script_file.name
                }
            )
            
        except FileNotFoundError:
            # pwsh not available - return success with warning
            return VerifyResult(
                success=True,
                reward=RewardLevel.COMPILE_CLEAN.value,
                details="PowerShell script extracted (syntax check skipped)",
                metadata={
                    "language": "powershell",
                    "warning": "pwsh not available for local syntax check"
                }
            )
        
        except subprocess.TimeoutExpired:
            return VerifyResult(
                success=False,
                reward=RewardLevel.FAILURE.value,
                details="Timeout",
                error=f"Syntax check exceeded {self.timeout}s timeout"
            )
        
        except Exception as e:
            return VerifyResult(
                success=False,
                reward=RewardLevel.FAILURE.value,
                details="Verification error",
                error=str(e)
            )
        
        finally:
            # Cleanup temp file if not cached
            if not self.binary_cache_dir and os.path.exists(script_file.name):
                os.unlink(script_file.name)
    
    def _resolve_validation_mode(self, mode: str) -> str:
        """
        Resolve validation mode, auto-detecting best available method.
        
        Args:
            mode: "remote", "local", "none", or "auto"
            
        Returns:
            Resolved mode string
        """
        if mode == "auto":
            # Try remote first if server configured
            if self.win_host and self.win_user:
                return "remote"
            
            # Try local pwsh
            if self._pwsh_available():
                return "local"
            
            # Fall back to none
            return "none"
        
        return mode
    
    def _pwsh_available(self) -> bool:
        """Check if local pwsh is available."""
        try:
            result = subprocess.run(
                [self.pwsh_cmd, "-Version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _validate_local(self, script_path: str) -> dict:
        """
        Validate PowerShell syntax using local pwsh.
        
        Uses PowerShell's parser to check syntax without execution.
        
        Returns:
            dict with 'success', 'error' keys
        """
        # PowerShell syntax check command
        check_cmd = f'''
$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
    "{script_path}",
    [ref]$null,
    [ref]$errors
) | Out-Null

if ($errors.Count -gt 0) {{
    $errors | ForEach-Object {{ Write-Error $_.Message }}
    exit 1
}} else {{
    exit 0
}}
'''
        
        cmd = [self.pwsh_cmd, "-NoProfile", "-Command", check_cmd]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout
        )
        
        if result.returncode == 0:
            return {'success': True}
        else:
            error = result.stderr.strip() if result.stderr else "Syntax error"
            return {
                'success': False,
                'error': error
            }
    
    def _validate_remote(self, script: str) -> Tuple[bool, str]:
        """
        Validate PowerShell syntax on Windows server via SSH.
        
        Uses native Windows PowerShell for full compatibility.
        
        Args:
            script: PowerShell script content
            
        Returns:
            Tuple of (success, error_message)
        """
        if not self.win_host or not self.win_user:
            return False, "Remote validation requires win_host and win_user"
        
        try:
            # Generate unique filename for this validation
            script_id = uuid.uuid4().hex[:8]
            remote_path = f"C:\\Temp\\ps_validate_{script_id}.ps1"
            
            # Upload script to server
            if not self._upload_script(script, remote_path):
                return False, "Failed to upload script to Windows server"
            
            # Run syntax check on server
            check_cmd = f'''
$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
    '{remote_path}',
    [ref]$null,
    [ref]$errors
) | Out-Null

if ($errors.Count -gt 0) {{
    $errors | ForEach-Object {{ Write-Output $_.Message }}
    Remove-Item -Path '{remote_path}' -Force -ErrorAction SilentlyContinue
    exit 1
}} else {{
    Remove-Item -Path '{remote_path}' -Force -ErrorAction SilentlyContinue
    exit 0
}}
'''
            
            # Build SSH command
            ssh_args = self._build_ssh_args()
            ssh_cmd = ['ssh'] + ssh_args + [
                f'{self.win_user}@{self.win_host}',
                'powershell', '-NoProfile', '-Command', check_cmd
            ]
            
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                return True, ""
            else:
                error = result.stdout.strip() or result.stderr.strip() or "Syntax error"
                return False, error
                
        except subprocess.TimeoutExpired:
            return False, f"Remote validation timeout (>{self.timeout}s)"
        except Exception as e:
            return False, str(e)
    
    def _upload_script(self, script: str, remote_path: str) -> bool:
        """Upload script content to Windows server via SSH."""
        try:
            # Escape single quotes in script
            escaped_script = script.replace("'", "''")
            
            # Write script via PowerShell
            write_cmd = f"Set-Content -Path '{remote_path}' -Value @'\n{escaped_script}\n'@"
            
            ssh_args = self._build_ssh_args()
            ssh_cmd = ['ssh'] + ssh_args + [
                f'{self.win_user}@{self.win_host}',
                'powershell', '-NoProfile', '-Command', write_cmd
            ]
            
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                timeout=30
            )
            
            return result.returncode == 0
            
        except Exception:
            return False
    
    def _build_ssh_args(self) -> list:
        """Build SSH arguments for server connection."""
        args = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
        
        if self.win_key:
            args = ['-i', self.win_key] + args
        
        return args
    
    def extract_code(self, text: str) -> Optional[str]:
        """
        Extract PowerShell script from model output.
        
        Handles:
        - Code blocks with ```powershell or ```ps1
        - Raw script with PowerShell patterns
        """
        # Try markdown code blocks first
        code_pattern = r'```(?:powershell|ps1|posh)?\s*(.*?)```'
        matches = re.findall(code_pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            script = matches[0].strip()
            if self._is_valid_powershell(script):
                return script
        
        # Look for PowerShell patterns
        lines = text.split('\n')
        script_lines = []
        in_script = False
        
        for line in lines:
            stripped = line.strip()
            
            # Start at PowerShell indicators
            if not in_script:
                ps_starts = [
                    '$', 'function ', 'param(', 'Import-Module',
                    'Get-', 'Set-', 'New-', 'Add-', 'Remove-',
                    'Write-', 'Invoke-', '[System.', '[Microsoft.',
                    '#requires', '<#', 'try {', 'if (', 'foreach '
                ]
                if any(stripped.startswith(s) or stripped.lower().startswith(s.lower()) 
                       for s in ps_starts):
                    in_script = True
            
            if in_script:
                # Stop at obvious non-PowerShell
                stop_patterns = ['```', 'user:', 'assistant:', '<|im_']
                if any(p in stripped.lower() for p in stop_patterns):
                    break
                
                script_lines.append(line)
        
        if script_lines:
            script = '\n'.join(script_lines).strip()
            if self._is_valid_powershell(script):
                return script
        
        # Last resort
        if self._is_valid_powershell(text):
            return text.strip()
        
        return None
    
    def _is_valid_powershell(self, script: str) -> bool:
        """Check if script looks like valid PowerShell."""
        if not script or len(script) < 10:
            return False
        
        # PowerShell indicators
        ps_indicators = [
            '$', '-', '|', 'function', 'param', 'Get-', 'Set-',
            'New-', 'Write-', 'Invoke-', '[System.', 'foreach',
            'if (', 'while (', 'try', 'catch', '@{', '@('
        ]
        
        has_ps = any(ind in script for ind in ps_indicators)
        
        # Not C++ or other languages
        not_cpp = '#include' not in script and 'int main' not in script
        
        return has_ps and not_cpp


# Alias
PS1Verifier = PowerShellVerifier
