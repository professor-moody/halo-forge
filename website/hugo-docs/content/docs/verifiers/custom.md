---
title: "Custom Verifiers"
description: "Create your own verification logic"
weight: 2
---

Create custom verifiers for any verification logic.

## Basic Custom Verifier

```python
from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult

class MyVerifier(Verifier):
    def __init__(self, api_url: str, max_workers: int = 8):
        super().__init__(max_workers=max_workers)
        self.api_url = api_url
    
    def verify(self, code: str) -> VerifyResult:
        # Extract code from model output
        extracted = self.extract_code(code)
        
        # Your verification logic
        if self._check_code(extracted):
            return VerifyResult(
                success=True,
                reward=1.0,
                details="Verification passed"
            )
        else:
            return VerifyResult(
                success=False,
                reward=0.0,
                details="Verification failed",
                error="Code did not pass checks"
            )
    
    def cleanup(self):
        # Close connections, delete temp files
        pass
```

## SubprocessVerifier

Quick custom verification with any command:

```python
from halo_forge.rlvr.verifiers import SubprocessVerifier

# Verify Rust code with cargo check
verifier = SubprocessVerifier(
    command="cargo check --manifest-path {file}/../Cargo.toml",
    success_pattern="Finished",
    file_extension=".rs",
    timeout=60
)

result = verifier.verify(rust_code)
```

### Options

| Option | Description |
|--------|-------------|
| `command` | Command to run (`{file}` = temp file path) |
| `success_pattern` | String that indicates success in output |
| `file_extension` | Extension for temp file |
| `timeout` | Command timeout in seconds |

## API Verifier Example

```python
import requests
from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult

class APIVerifier(Verifier):
    def __init__(self, api_url: str, api_key: str):
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key
    
    def verify(self, code: str) -> VerifyResult:
        extracted = self.extract_code(code)
        
        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"code": extracted},
            timeout=30
        )
        
        data = response.json()
        
        if data["success"]:
            return VerifyResult(
                success=True,
                reward=data.get("score", 1.0),
                details="API verification passed",
                metadata=data
            )
        else:
            return VerifyResult(
                success=False,
                reward=0.0,
                details="API verification failed",
                error=data.get("error", "Unknown error")
            )
```

## Test-Based Verifier

```python
from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult
import subprocess
import tempfile

class CustomTestVerifier(Verifier):
    def __init__(self, test_file: str):
        super().__init__()
        self.test_file = test_file
    
    def verify(self, code: str) -> VerifyResult:
        extracted = self.extract_code(code)
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(extracted)
            code_file = f.name
        
        try:
            result = subprocess.run(
                ['pytest', self.test_file, '--import-mode=importlib'],
                env={'CODE_FILE': code_file},
                capture_output=True,
                text=True,
                timeout=30
            )
            
            passed = result.returncode == 0
            
            return VerifyResult(
                success=passed,
                reward=1.0 if passed else 0.0,
                details=result.stdout[:500],
                error=result.stderr[:500] if not passed else None
            )
        finally:
            os.unlink(code_file)
```

## Overriding Batch Verification

For optimized batch processing:

```python
class BatchOptimizedVerifier(Verifier):
    def verify(self, code: str) -> VerifyResult:
        # Single verification
        ...
    
    def verify_batch(self, codes: list) -> list:
        # Send all to API at once
        response = self.api.batch_verify(codes)
        return [self._parse_result(r) for r in response["results"]]
```

## VerifyResult Fields

```python
@dataclass
class VerifyResult:
    success: bool           # Pass/fail
    reward: float           # 0.0 - 1.0
    details: str            # Human-readable
    error: Optional[str]    # Error message
    metadata: Dict          # Additional data
```

## Best Practices

1. **Extract code first** — Use `self.extract_code(code)` to handle model output
2. **Set timeouts** — Prevent hanging on bad code
3. **Implement cleanup** — Close connections, delete temp files
4. **Use graduated rewards** — Partial credit helps training
5. **Handle exceptions** — Catch and return VerifyResult with error
6. **Limit parallelism** — Set appropriate `max_workers`

## Integration with RAFT

```python
from halo_forge.rlvr import RAFTTrainer

verifier = MyVerifier(api_url="https://api.example.com")

trainer = RAFTTrainer(
    verifier=verifier,
    sft_checkpoint="models/sft/final_model"
)

try:
    trainer.run(prompts, num_cycles=5)
finally:
    verifier.cleanup()
```
