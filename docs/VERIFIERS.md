# Verifiers Guide

Verifiers are the core of RLVR training. They provide the reward signal that guides model improvement by checking if generated code meets requirements.

## Table of Contents

1. [Overview](#overview)
2. [Reward Levels](#reward-levels)
3. [Built-in Verifiers](#built-in-verifiers)
4. [Verification Modes](#verification-modes)
5. [Creating Custom Verifiers](#creating-custom-verifiers)
6. [Integration with RAFT](#integration-with-raft)
7. [Best Practices](#best-practices)
8. [Extensibility](#extensibility)

---

## Overview

A verifier takes generated code and returns a `VerifyResult` containing:

- `success`: Boolean indicating overall pass/fail
- `reward`: Float from 0.0 to 1.0 (used for filtering in RAFT)
- `details`: Human-readable explanation
- `error`: Error message if failed
- `metadata`: Additional information (optional)

```python
from halo_forge.rlvr.verifiers import GCCVerifier, VerifyResult

verifier = GCCVerifier()
result = verifier.verify(code)

print(f"Success: {result.success}")
print(f"Reward: {result.reward}")
print(f"Details: {result.details}")
```

---

## Reward Levels

halo-forge uses graduated rewards to provide partial credit for near-successes. This helps gradient flow during training by distinguishing between different failure modes.

### Standard Reward Levels

| Level | Reward | Meaning |
|-------|--------|---------|
| `FAILURE` | 0.0 | Complete failure (syntax errors, doesn't compile) |
| `COMPILE_WARNINGS` | 0.3 | Compiles with warnings |
| `COMPILE_CLEAN` | 0.5 | Compiles without warnings |
| `RUNS_NO_CRASH` | 0.7 | Executes without crashing |
| `CORRECT_OUTPUT` | 1.0 | Produces correct output |

### Using RewardLevel

```python
from halo_forge.rlvr.verifiers import RewardLevel

# Get reward from compilation result
reward = RewardLevel.from_compile_result(success=True, has_warnings=True)
# Returns 0.3

# Get reward from full execution result
reward = RewardLevel.from_execution_result(
    compiles=True,
    runs=True,
    correct=False
)
# Returns 0.7
```

### Why Graduated Rewards Matter

Binary rewards (0 or 1) create a sparse gradient signal. With graduated rewards:

| Scenario | Binary | Graduated |
|----------|--------|-----------|
| Syntax error | 0.0 | 0.0 |
| Missing semicolon only | 0.0 | 0.0 |
| Compiles with warnings | 1.0 | 0.3 |
| Compiles clean | 1.0 | 0.5 |
| Runs but wrong output | 0.0 | 0.7 |
| Correct output | 1.0 | 1.0 |

The model learns that "almost compiling" is better than "syntax error", creating a smoother learning curve.

---

## Built-in Verifiers

### GCCVerifier

Compiles C/C++ code with GCC on Linux. Supports compile-only or compile+run verification.

```python
from halo_forge.rlvr.verifiers import GCCVerifier

# Compile only (default)
verifier = GCCVerifier(
    flags=['-w', '-O2'],      # Compiler flags
    timeout=30,               # Compilation timeout
    max_workers=8             # Parallel compilations
)

# Compile and run
verifier = GCCVerifier(
    run_after_compile=True,
    run_timeout=5,            # Execution timeout
    memory_limit_mb=256       # Memory limit for execution
)

# Compile, run, and check output
verifier = GCCVerifier(
    run_after_compile=True,
    expected_output="Hello, World!",
    stdin_input="test input"
)

# Treat warnings as partial failure
verifier = GCCVerifier(warn_as_error=True)
```

### ClangVerifier

Alternative to GCC with same interface:

```python
from halo_forge.rlvr.verifiers import ClangVerifier

verifier = ClangVerifier(
    run_after_compile=True,
    expected_output="42"
)
```

### MinGWVerifier

Cross-compiles to Windows PE from Linux. Does not support runtime verification (cannot run Windows binaries on Linux).

```python
from halo_forge.rlvr.verifiers import MinGWVerifier

verifier = MinGWVerifier(
    flags=['-static', '-lntdll'],
    timeout=30,
    warn_as_error=True
)
```

### RemoteMSVCVerifier

Compiles with MSVC on a remote Windows machine via SSH.

```python
from halo_forge.rlvr.verifiers import RemoteMSVCVerifier

verifier = RemoteMSVCVerifier(
    host="192.168.1.100",
    user="developer",
    ssh_key="/home/user/.ssh/win",
    timeout=60
)

# Test connection before training
print(verifier.test_connection())
print(verifier.test_msvc())
```

**Windows Setup Requirements:**
- OpenSSH Server running
- Visual Studio with MSVC
- `C:\Binaries\input` and `C:\Binaries\output` directories

### PytestVerifier

Runs Python tests with pytest. Supports partial rewards based on test pass rate.

```python
from halo_forge.rlvr.verifiers import PytestVerifier

# Test code that includes its own tests
verifier = PytestVerifier(timeout=60)

# Test against external test file
verifier = PytestVerifier(
    test_file="tests/test_solution.py"
)
```

Reward calculation:
- All tests pass: 1.0
- Some tests pass: `pass_rate * 0.5` (max 0.5 for partial)
- No tests pass: 0.0

### UnittestVerifier

Uses Python's built-in unittest instead of pytest.

```python
from halo_forge.rlvr.verifiers import UnittestVerifier

verifier = UnittestVerifier(timeout=60)
```

### ChainedVerifier

Chain multiple verifiers together for multi-stage verification.

```python
from halo_forge.rlvr.verifiers import ChainedVerifier, GCCVerifier, PytestVerifier

# First compile, then test
verifier = ChainedVerifier([
    GCCVerifier(),
    PytestVerifier()
])

# With custom weights
verifier = ChainedVerifier(
    verifiers=[GCCVerifier(), PytestVerifier()],
    weights=[0.3, 0.7]  # Tests weighted higher
)
```

Stops at first failure, returns accumulated reward.

### SubprocessVerifier

Quick way to use any command-line tool as a verifier.

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

---

## Verification Modes

### Mode 1: Compile Only (Default)

Fastest verification. Only checks if code compiles.

```python
verifier = GCCVerifier()  # Compile only
```

Rewards:
- 0.0: Does not compile
- 0.3: Compiles with warnings (if warn_as_error=True)
- 0.5: Compiles clean

### Mode 2: Compile + Run

Checks if code compiles AND runs without crashing.

```python
verifier = GCCVerifier(run_after_compile=True)
```

Rewards:
- 0.0: Does not compile
- 0.5: Compiles but crashes
- 0.7: Runs without crash

### Mode 3: Compile + Run + Output Check

Full verification: compile, run, and verify output.

```python
verifier = GCCVerifier(
    run_after_compile=True,
    expected_output="Expected result",
    stdin_input="Input for stdin"
)
```

Rewards:
- 0.0: Does not compile
- 0.5: Compiles but crashes
- 0.7: Runs but wrong output
- 1.0: Correct output

---

## Creating Custom Verifiers

### Basic Template

```python
from halo_forge.rlvr.verifiers import Verifier, VerifyResult, RewardLevel

class MyVerifier(Verifier):
    def __init__(self, config, max_workers: int = 8):
        super().__init__(max_workers=max_workers)
        self.config = config
    
    def verify(self, code: str) -> VerifyResult:
        # Extract code from model output
        extracted = self.extract_code(code)
        
        # Your verification logic
        try:
            result = your_verification_logic(extracted)
            
            if result.fully_correct:
                return VerifyResult(
                    success=True,
                    reward=RewardLevel.CORRECT_OUTPUT.value,
                    details="Verification passed"
                )
            elif result.partially_correct:
                return VerifyResult(
                    success=False,
                    reward=RewardLevel.RUNS_NO_CRASH.value,
                    details="Partially correct",
                    error=result.error_message
                )
            else:
                return VerifyResult(
                    success=False,
                    reward=RewardLevel.FAILURE.value,
                    details="Verification failed",
                    error=result.error_message
                )
        except Exception as e:
            return VerifyResult(
                success=False,
                reward=RewardLevel.FAILURE.value,
                details="Exception during verification",
                error=str(e)
            )
    
    def cleanup(self):
        # Called when done with verifier
        # Close connections, delete temp files, etc.
        pass
```

### Batch Verification

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

---

## Integration with RAFT

### Basic Usage

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

### With Runtime Verification

```python
verifier = GCCVerifier(
    run_after_compile=True,
    expected_output=None,  # Just check it runs
    run_timeout=5
)

trainer = RAFTTrainer(verifier=verifier, ...)
```

### Multi-Stage Verification

```python
from halo_forge.rlvr.verifiers import ChainedVerifier, GCCVerifier

verifier = ChainedVerifier([
    GCCVerifier(),                    # Stage 1: Compile
    GCCVerifier(run_after_compile=True)  # Stage 2: Run
])

trainer = RAFTTrainer(verifier=verifier, ...)
```

---

## Best Practices

### Performance

1. **Fast Compilation**: Use minimal flags (`-w -O2`), avoid heavy optimization
2. **Reasonable Timeouts**: 30s for compile, 5s for run, 60s for tests
3. **Parallel Workers**: Match to CPU cores (8-16 for Strix Halo)
4. **Memory Limits**: Set `memory_limit_mb` to prevent runaway processes

### Reliability

1. **Cleanup Resources**: Implement `cleanup()` for connections, temp files
2. **Handle Exceptions**: Wrap verification logic in try/except
3. **Meaningful Errors**: Include useful error messages in `result.error`
4. **Timeout Everything**: Set timeouts for compile, run, and any network calls

### Reward Design

1. **Use Graduated Rewards**: Helps gradient flow
2. **Higher Reward = Better**: Ensure reward ordering matches quality
3. **Consistent Thresholds**: Use RewardLevel values for consistency
4. **Document Reward Scheme**: Make it clear what each reward level means

---

## Extensibility

halo-forge verifiers can be extended to domains beyond code compilation.

### Example Domain Verifiers

| Domain | Verification Approach |
|--------|----------------------|
| **Security research** | Detection testing, static analysis |
| **Formal verification** | Theorem provers (Coq, Lean, Z3) |
| **Multi-language** | Additional compilers (Rust, Go, Zig) |
| **Execution testing** | I/O comparison for algorithm correctness |
| **API compliance** | Check generated code against specifications |
| **Documentation** | Verify generated docs compile/render |

### Custom Verifier Checklist

1. Inherit from `Verifier` base class
2. Implement `verify(code: str) -> VerifyResult`
3. Use `RewardLevel` for consistent reward values
4. Implement `cleanup()` if needed
5. Optionally override `verify_batch()` for optimization
6. Set appropriate `max_workers` for parallelism
7. Document your reward scheme

### Verifier Architecture

```
                    Verifier (base class)
                           |
        +------------------+------------------+
        |                  |                  |
  CompileVerifier    TestVerifier      CustomVerifier
        |                  |
   +---------+      +----------+
   |         |      |          |
  GCC     MinGW   Pytest   Unittest
```
