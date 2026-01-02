# Custom Verifiers

Build verifiers for any domain with deterministic verification.

## SubprocessVerifier

Run any command as a verifier:

```python
from halo_forge.rlvr.verifiers import SubprocessVerifier

verifier = SubprocessVerifier(
    command=["rustc", "{file}", "-o", "{output}"],
    success_codes=[0],
    timeout=30
)
```

### Placeholders

| Placeholder | Replaced With |
|-------------|---------------|
| `{file}` | Path to generated code |
| `{output}` | Path for output binary |
| `{code}` | The code itself |

## Custom Verifier Class

For complex logic, subclass `Verifier`:

```python
from halo_forge.rlvr.verifiers import Verifier, VerifyResult

class MyVerifier(Verifier):
    def __init__(self, config, max_workers=8):
        super().__init__(max_workers=max_workers)
        self.config = config
    
    def verify(self, code: str) -> VerifyResult:
        # Your verification logic
        is_valid = self.check_code(code)
        
        return VerifyResult(
            success=is_valid,
            reward=1.0 if is_valid else 0.0,
            details="Verification details",
            error="" if is_valid else "Error message"
        )
    
    def cleanup(self):
        # Optional: cleanup resources
        pass
```

## Example: Rust Verifier

```python
class RustVerifier(Verifier):
    def verify(self, code: str) -> VerifyResult:
        with tempfile.NamedTemporaryFile(suffix=".rs", delete=False) as f:
            f.write(code.encode())
            src_path = f.name
        
        try:
            result = subprocess.run(
                ["rustc", src_path, "-o", "/tmp/out"],
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return VerifyResult(
                    success=True,
                    reward=1.0,
                    details="Compilation successful"
                )
            else:
                return VerifyResult(
                    success=False,
                    reward=0.0,
                    details="Compilation failed",
                    error=result.stderr.decode()
                )
        finally:
            os.unlink(src_path)
```

## Example: Theorem Prover

```python
class LeanVerifier(Verifier):
    def verify(self, code: str) -> VerifyResult:
        # Write Lean code
        with open("/tmp/proof.lean", "w") as f:
            f.write(code)
        
        # Run Lean
        result = subprocess.run(
            ["lean", "/tmp/proof.lean"],
            capture_output=True,
            timeout=60
        )
        
        if "error" not in result.stderr.decode().lower():
            return VerifyResult(success=True, reward=1.0, details="Proof valid")
        else:
            return VerifyResult(success=False, reward=0.0, details="Proof invalid")
```

## Domain Applications

| Domain | Verifier Approach |
|--------|-------------------|
| Rust | `rustc` compilation |
| Go | `go build` |
| Formal proofs | Lean, Coq, Z3 |
| SQL | Query execution |
| Shell scripts | ShellCheck + execution |
| APIs | OpenAPI validator |
| Security | EDR detection (see [malagent](https://github.com/professor-moody/malagent)) |

## Integration with RAFT

```python
# Register custom verifier
from halo_forge.rlvr import register_verifier

register_verifier("rust", RustVerifier)

# Use in CLI
# halo-forge raft train --verifier rust ...
```
