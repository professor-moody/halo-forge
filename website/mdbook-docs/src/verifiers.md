# Verifier Overview

Verifiers provide the reward signal that guides RLVR training.

## How Verifiers Work

A verifier takes generated code and returns:

```python
@dataclass
class VerifyResult:
    success: bool      # Pass/fail
    reward: float      # 0.0 to 1.0
    details: str       # Human-readable explanation
    error: str         # Error message if failed
    metadata: dict     # Additional info
```

## Built-in Verifiers

| Verifier | Language | Use Case |
|----------|----------|----------|
| `GCCVerifier` | C/C++ | Local Linux compilation |
| `ClangVerifier` | C/C++ | Alternative compiler |
| `MinGWVerifier` | C/C++ | Windows cross-compilation |
| `RemoteMSVCVerifier` | C/C++ | Remote Windows build server |
| `PytestVerifier` | Python | Test-based verification |
| `UnittestVerifier` | Python | Standard library tests |
| `SubprocessVerifier` | Any | Custom command |

## Usage Example

```python
from halo_forge.rlvr.verifiers import GCCVerifier

verifier = GCCVerifier()
result = verifier.verify(code)

print(f"Success: {result.success}")
print(f"Reward: {result.reward}")
```

## CLI Usage

```bash
# GCC (default)
halo-forge raft train --verifier gcc ...

# MinGW cross-compilation
halo-forge raft train --verifier mingw ...

# Remote MSVC
halo-forge raft train \
  --verifier msvc \
  --host 192.168.1.100 \
  --user developer \
  --ssh-key ~/.ssh/win \
  ...
```

## Graduated Rewards

Verifiers return graduated rewards for partial success:

| Level | Reward | Meaning |
|-------|--------|---------|
| `FAILURE` | 0.0 | Complete failure |
| `COMPILE_WARNINGS` | 0.3 | Compiles with warnings |
| `COMPILE_CLEAN` | 0.5 | Compiles cleanly |
| `RUNS_NO_CRASH` | 0.7 | Runs without crashing |
| `CORRECT_OUTPUT` | 1.0 | Correct output |

## Batch Verification

For efficiency during training:

```python
results = verifier.verify_batch(code_samples)
```

This parallelizes verification across available CPU cores.

## Next Steps

- [Compile Verifiers](./verifiers-compile.md) — GCC, Clang, MinGW, MSVC
- [Test Verifiers](./verifiers-test.md) — pytest, unittest
- [Custom Verifiers](./verifiers-custom.md) — Build your own
