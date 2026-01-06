"""
.NET/C# Verifier

Compiles C# code to Windows executables using the .NET SDK.
Supports cross-compilation to Windows via `dotnet publish`.

Graduated Rewards:
- 0.0: Does not compile
- 0.5: Compiles clean
"""

import subprocess
import tempfile
import shutil
import uuid
import re
from pathlib import Path
from typing import Optional

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult, RewardLevel


class DotNetVerifier(Verifier):
    """
    .NET/C# verifier using dotnet CLI.
    
    Creates a temporary console project, writes code, and compiles
    using `dotnet publish -c Release -r win-x64`.
    
    Requires:
        dnf install dotnet-sdk-8.0  (Fedora)
        apt install dotnet-sdk-8.0  (Ubuntu)
    
    Example:
        # Compile only
        verifier = DotNetVerifier()
        result = verifier.verify(csharp_code)
        
        # With binary caching
        verifier = DotNetVerifier(binary_cache_dir="binaries/dotnet")
    
    Note:
        Cannot execute Windows binaries on Linux, so this is compile-only.
        The reward is 0.5 (COMPILE_CLEAN) on success.
    """
    
    WINDOWS_RID = "win-x64"
    
    def __init__(
        self,
        timeout: int = 120,
        max_workers: int = 4,
        target_framework: str = "net8.0",
        self_contained: bool = False,
        single_file: bool = False,
        binary_cache_dir: Optional[str] = None
    ):
        """
        Initialize .NET verifier.
        
        Args:
            timeout: dotnet build timeout in seconds
            max_workers: Max parallel verifications
            target_framework: .NET target framework (net8.0, net7.0, etc.)
            self_contained: If True, bundle .NET runtime (larger, portable)
            single_file: If True, produce single executable
            binary_cache_dir: Directory to cache compiled binaries
        """
        super().__init__(max_workers=max_workers)
        self.timeout = timeout
        self.target_framework = target_framework
        self.self_contained = self_contained
        self.single_file = single_file
        self.binary_cache_dir = Path(binary_cache_dir) if binary_cache_dir else None
        
        # Create cache directory if needed
        if self.binary_cache_dir:
            self.binary_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def verify(self, code: str) -> VerifyResult:
        """
        Verify C# code by compiling with dotnet.
        
        Args:
            code: C# source code
            
        Returns:
            VerifyResult with compilation status
        """
        # Extract code from model output
        extracted = self.extract_code(code)
        
        # Create temp project
        project_dir = tempfile.mkdtemp(prefix="dotnet_verify_")
        
        try:
            # Create .csproj
            csproj_content = f'''<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>{self.target_framework}</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>disable</Nullable>
  </PropertyGroup>
</Project>
'''
            (Path(project_dir) / "verify_code.csproj").write_text(csproj_content)
            
            # Write source code
            (Path(project_dir) / "Program.cs").write_text(extracted)
            
            # Compile
            result = self._compile(project_dir)
            
            if not result['success']:
                return VerifyResult(
                    success=False,
                    reward=RewardLevel.FAILURE.value,
                    details="Compilation failed",
                    error=result['error'],
                    metadata={
                        "compiler": "dotnet",
                        "stage": "compile"
                    }
                )
            
            # Cache binary if configured
            cached_path = None
            binary_path = result.get('binary_path')
            if self.binary_cache_dir and binary_path and Path(binary_path).exists():
                cache_name = f"{uuid.uuid4().hex[:12]}.exe"
                cached_path = self.binary_cache_dir / cache_name
                shutil.copy2(binary_path, cached_path)
            
            # Success - compile only (can't run Windows .exe on Linux)
            return VerifyResult(
                success=True,
                reward=RewardLevel.COMPILE_CLEAN.value,
                details="Compilation successful",
                metadata={
                    "compiler": "dotnet",
                    "stage": "compile",
                    "target": self.WINDOWS_RID,
                    "framework": self.target_framework,
                    "binary_path": str(cached_path) if cached_path else binary_path
                }
            )
            
        except subprocess.TimeoutExpired:
            return VerifyResult(
                success=False,
                reward=RewardLevel.FAILURE.value,
                details="Timeout",
                error=f"Compilation exceeded {self.timeout}s timeout"
            )
        
        except FileNotFoundError:
            return VerifyResult(
                success=False,
                reward=RewardLevel.FAILURE.value,
                details="dotnet not found",
                error="'dotnet' not found in PATH - is .NET SDK installed?"
            )
        
        except Exception as e:
            return VerifyResult(
                success=False,
                reward=RewardLevel.FAILURE.value,
                details="Verification error",
                error=str(e)
            )
        
        finally:
            shutil.rmtree(project_dir, ignore_errors=True)
    
    def _compile(self, project_dir: str) -> dict:
        """
        Compile .NET project.
        
        Returns:
            dict with 'success', 'error', 'binary_path' keys
        """
        cmd = [
            "dotnet", "publish",
            "-c", "Release",
            "-r", self.WINDOWS_RID,
            "--nologo",
            "-v", "quiet"
        ]
        
        if self.self_contained:
            cmd.append("--self-contained")
        else:
            cmd.append("--no-self-contained")
        
        if self.single_file:
            cmd.extend(["-p:PublishSingleFile=true"])
        
        result = subprocess.run(
            cmd,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=self.timeout
        )
        
        if result.returncode == 0:
            # Find the output binary
            publish_dir = Path(project_dir) / "bin" / "Release" / self.target_framework / self.WINDOWS_RID / "publish"
            binary_path = publish_dir / "verify_code.exe"
            
            if binary_path.exists():
                return {
                    'success': True,
                    'binary_path': str(binary_path)
                }
            else:
                # Try without publish subfolder
                binary_path = Path(project_dir) / "bin" / "Release" / self.target_framework / self.WINDOWS_RID / "verify_code.exe"
                return {
                    'success': True,
                    'binary_path': str(binary_path) if binary_path.exists() else None
                }
        else:
            error = result.stderr.strip() if result.stderr else result.stdout.strip()
            return {
                'success': False,
                'error': error[:500]
            }
    
    def extract_code(self, text: str) -> str:
        """
        Extract C# code from model output.
        
        Handles:
        - Code blocks with ```csharp or ```cs
        - Raw code starting with using or namespace
        """
        # Try markdown code blocks first
        code_pattern = r'```(?:csharp|cs|c#)?\s*(.*?)```'
        matches = re.findall(code_pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            code = matches[0].strip()
            if self._is_valid_csharp(code):
                return code
        
        # Look for using/namespace pattern
        lines = text.split('\n')
        code_lines = []
        in_code = False
        brace_depth = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Start at using or namespace
            if not in_code:
                if stripped.startswith('using ') or stripped.startswith('namespace '):
                    in_code = True
            
            if in_code:
                code_lines.append(line)
                brace_depth += line.count('{') - line.count('}')
                
                # Stop when all braces are closed
                if brace_depth == 0 and len(code_lines) > 10:
                    break
        
        if code_lines:
            code = '\n'.join(code_lines).strip()
            if self._is_valid_csharp(code):
                return code
        
        # Last resort
        if self._is_valid_csharp(text):
            return text.strip()
        
        return text.strip()
    
    def _is_valid_csharp(self, code: str) -> bool:
        """Check if code looks like valid C#."""
        if not code or len(code) < 30:
            return False
        
        # Must have some C# structure
        csharp_indicators = [
            'class ', 'namespace ', 'using ', 'static void Main',
            'public ', 'private ', 'protected ', 'Console.', 'System.'
        ]
        has_csharp = any(ind in code for ind in csharp_indicators)
        has_structure = '{' in code and '}' in code
        
        return has_csharp and has_structure


# Alias
CSharpVerifier = DotNetVerifier
