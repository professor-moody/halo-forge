"""
Custom Verifier Template

Template and examples for creating custom verifiers.
Copy this file and modify for your specific verification needs.
"""

from typing import Optional
from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult


class CustomVerifier(Verifier):
    """
    Template for creating custom verifiers.
    
    To create your own verifier:
    1. Copy this class
    2. Implement the `verify` method
    3. Optionally override `verify_batch` for custom parallelization
    4. Optionally override `cleanup` for resource management
    
    Example:
        class MyAPIVerifier(Verifier):
            def __init__(self, api_url: str, api_key: str):
                super().__init__()
                self.api_url = api_url
                self.api_key = api_key
            
            def verify(self, code: str) -> VerifyResult:
                response = requests.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"code": code}
                )
                
                if response.json()["success"]:
                    return VerifyResult(
                        success=True,
                        reward=1.0,
                        details="API verification passed"
                    )
                else:
                    return VerifyResult(
                        success=False,
                        reward=0.0,
                        details="API verification failed",
                        error=response.json().get("error")
                    )
    """
    
    def __init__(self, max_workers: int = 8):
        """
        Initialize your verifier.
        
        Add any configuration parameters your verifier needs.
        """
        super().__init__(max_workers=max_workers)
        # Add your initialization here
    
    def verify(self, code: str) -> VerifyResult:
        """
        Verify a code sample.
        
        This is the main method you need to implement.
        
        Args:
            code: The code to verify (may include model output formatting)
            
        Returns:
            VerifyResult with:
                - success: True if verification passed
                - reward: Float 0.0-1.0 indicating quality
                - details: Human-readable explanation
                - error: Error message if failed (optional)
                - metadata: Additional data dict (optional)
        """
        # Extract code from model output if needed
        extracted = self.extract_code(code)
        
        # Implement your verification logic here
        # Example: Check if code contains certain patterns
        
        if "def main" in extracted or "int main" in extracted:
            return VerifyResult(
                success=True,
                reward=1.0,
                details="Code contains main function",
                metadata={"has_main": True}
            )
        else:
            return VerifyResult(
                success=False,
                reward=0.0,
                details="Code missing main function",
                error="No main function found"
            )
    
    def cleanup(self):
        """
        Cleanup resources when done.
        
        Override this if your verifier needs to:
        - Close network connections
        - Delete temporary files
        - Release other resources
        """
        pass


class SubprocessVerifier(Verifier):
    """
    Generic subprocess-based verifier.
    
    Runs an arbitrary command and checks the exit code.
    Useful for quick custom verification.
    
    Example:
        # Verify Rust code with cargo check
        verifier = SubprocessVerifier(
            command="cargo check",
            success_pattern="Finished",
            file_extension=".rs"
        )
    """
    
    def __init__(
        self,
        command: str,
        success_pattern: Optional[str] = None,
        file_extension: str = ".txt",
        timeout: int = 60,
        max_workers: int = 4
    ):
        """
        Initialize subprocess verifier.
        
        Args:
            command: Command to run (use {file} as placeholder for input file)
            success_pattern: Pattern to search for in output (None = use exit code)
            file_extension: Extension for temp file
            timeout: Command timeout
            max_workers: Max parallel runs
        """
        super().__init__(max_workers=max_workers)
        self.command = command
        self.success_pattern = success_pattern
        self.file_extension = file_extension
        self.timeout = timeout
    
    def verify(self, code: str) -> VerifyResult:
        """Run command on code and check result."""
        import subprocess
        import tempfile
        import os
        
        extracted = self.extract_code(code)
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=self.file_extension,
            delete=False
        ) as f:
            f.write(extracted)
            temp_file = f.name
        
        try:
            # Substitute {file} placeholder
            cmd = self.command.replace("{file}", temp_file)
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Check success
            if self.success_pattern:
                success = self.success_pattern in result.stdout
            else:
                success = result.returncode == 0
            
            if success:
                return VerifyResult(
                    success=True,
                    reward=1.0,
                    details="Command succeeded"
                )
            else:
                return VerifyResult(
                    success=False,
                    reward=0.0,
                    details="Command failed",
                    error=result.stderr[:500] or result.stdout[:500]
                )
                
        except subprocess.TimeoutExpired:
            return VerifyResult(
                success=False,
                reward=0.0,
                details="Command timeout",
                error=f"Exceeded {self.timeout}s"
            )
        except Exception as e:
            return VerifyResult(
                success=False,
                reward=0.0,
                details="Command error",
                error=str(e)
            )
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

