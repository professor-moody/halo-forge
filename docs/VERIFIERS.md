# Verifiers Guide

Verifiers are the core of RLVR training. They check if generated code meets requirements.

## Built-in Verifiers

### GCCVerifier

Compiles C/C++ code with GCC on Linux.

```python
from halo_forge.rlvr.verifiers import GCCVerifier

verifier = GCCVerifier(
    flags=['-w', '-O2'],      # Compiler flags
    timeout=30,               # Compilation timeout
    max_workers=8             # Parallel compilations
)

result = verifier.verify(code)
# result.success: True if compiled
# result.reward: 1.0 if success, 0.0 otherwise
# result.error: Compilation error message
```

### MinGWVerifier

Cross-compiles to Windows PE from Linux.

```python
from halo_forge.rlvr.verifiers import MinGWVerifier

verifier = MinGWVerifier(
    flags=['-static', '-lntdll'],
    timeout=30
)
```

### RemoteMSVCVerifier

Compiles with MSVC on a remote Windows machine.

```python
from halo_forge.rlvr.verifiers import RemoteMSVCVerifier

verifier = RemoteMSVCVerifier(
    host="192.168.1.100",
    user="developer",
    ssh_key="/home/user/.ssh/win",
    timeout=60
)

# Test connection
print(verifier.test_connection())
print(verifier.test_msvc())
```

**Windows Setup Requirements:**
- OpenSSH Server running
- Visual Studio with MSVC
- `C:\Binaries\input` and `C:\Binaries\output` directories

### PytestVerifier

Runs Python tests with pytest.

```python
from halo_forge.rlvr.verifiers import PytestVerifier

# Test code that includes its own tests
verifier = PytestVerifier(timeout=60)

# Or test against external test file
verifier = PytestVerifier(
    test_file="tests/test_solution.py"
)
```

### ChainedVerifier

Chain multiple verifiers together.

```python
from halo_forge.rlvr.verifiers import ChainedVerifier, GCCVerifier, PytestVerifier

# First compile, then test
verifier = ChainedVerifier([
    GCCVerifier(),
    PytestVerifier()
])
```

## Creating Custom Verifiers

### Basic Template

```python
from halo_forge.rlvr.verifiers import Verifier, VerifyResult

class MyVerifier(Verifier):
    def __init__(self, api_url: str, api_key: str, max_workers: int = 4):
        super().__init__(max_workers=max_workers)
        self.api_url = api_url
        self.api_key = api_key
    
    def verify(self, code: str) -> VerifyResult:
        # Extract code from model output
        extracted = self.extract_code(code)
        
        # Your verification logic
        try:
            # Example: Call an API
            response = requests.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"code": extracted}
            )
            
            result = response.json()
            
            if result["success"]:
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
                    error=result.get("error")
                )
        except Exception as e:
            return VerifyResult(
                success=False,
                reward=0.0,
                details="Exception",
                error=str(e)
            )
    
    def cleanup(self):
        # Called when done with verifier
        pass
```

### SubprocessVerifier

Quick way to use any command-line tool:

```python
from halo_forge.rlvr.verifiers import SubprocessVerifier

# Rust cargo check
rust_verifier = SubprocessVerifier(
    command="cargo check --manifest-path /tmp/project/Cargo.toml",
    success_pattern="Finished",
    file_extension=".rs"
)

# Go build
go_verifier = SubprocessVerifier(
    command="go build -o /dev/null {file}",
    file_extension=".go"
)

# ESLint
eslint_verifier = SubprocessVerifier(
    command="eslint {file}",
    file_extension=".js"
)
```

## Partial Rewards

Return partial rewards for near-successes:

```python
def verify(self, code: str) -> VerifyResult:
    extracted = self.extract_code(code)
    
    # Check different quality levels
    compiles = self.try_compile(extracted)
    runs = self.try_run(extracted) if compiles else False
    tests_pass = self.run_tests(extracted) if runs else False
    
    if tests_pass:
        return VerifyResult(success=True, reward=1.0, details="All tests pass")
    elif runs:
        return VerifyResult(success=False, reward=0.7, details="Runs but tests fail")
    elif compiles:
        return VerifyResult(success=False, reward=0.3, details="Compiles but crashes")
    else:
        return VerifyResult(success=False, reward=0.0, details="Does not compile")
```

## Batch Verification

The base class provides parallel batch verification:

```python
# Process many samples in parallel
codes = [code1, code2, code3, ...]
results = verifier.verify_batch(codes)  # Uses max_workers threads
```

Override for custom batch behavior:

```python
def verify_batch(self, codes: List[str]) -> List[VerifyResult]:
    # Custom batch logic, e.g., send all to API at once
    response = self.api.batch_verify(codes)
    return [self._parse_result(r) for r in response["results"]]
```

## Best Practices

1. **Fast Compilation**: Use minimal flags, avoid optimization levels
2. **Reasonable Timeouts**: 30s for compile, 60s for tests
3. **Cleanup Resources**: Implement `cleanup()` for connections, temp files
4. **Meaningful Errors**: Include useful error messages in `result.error`
5. **Parallel Workers**: Match to your CPU cores (8-16 for Strix Halo)

## Integration with RAFT

```python
from halo_forge.rlvr import RAFTTrainer
from halo_forge.rlvr.verifiers import GCCVerifier

verifier = GCCVerifier(max_workers=8)

trainer = RAFTTrainer(
    verifier=verifier,
    sft_checkpoint="models/sft/final_model"
)

trainer.run(prompts, num_cycles=5)
```

