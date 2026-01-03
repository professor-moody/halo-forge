---
title: "Compile Verifiers"
description: "GCC, MinGW, and Clang verification"
weight: 1
---

Compile verifiers check code by attempting to compile it.

## GCCVerifier

For Linux C/C++ code:

```python
from halo_forge.rlvr.verifiers import GCCVerifier

# Compile only
verifier = GCCVerifier()
result = verifier.verify(code)

# Compile and run
verifier = GCCVerifier(run_after_compile=True)

# Compile, run, and check output
verifier = GCCVerifier(
    run_after_compile=True,
    expected_output="Hello, World!",
    stdin_input="test input"
)
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `flags` | `['-w', '-O2']` | Compiler flags |
| `timeout` | 30 | Compilation timeout (seconds) |
| `max_workers` | 8 | Parallel compilations |
| `run_after_compile` | False | Run the binary |
| `run_timeout` | 5 | Execution timeout |
| `expected_output` | None | Compare output |
| `stdin_input` | None | Input to program |
| `memory_limit_mb` | 256 | Memory limit for execution |
| `warn_as_error` | False | Warnings reduce reward |

## MinGWVerifier

Cross-compile to Windows PE executables:

```python
from halo_forge.rlvr.verifiers import MinGWVerifier

verifier = MinGWVerifier()
result = verifier.verify(windows_api_code)
```

> **Note:** Cannot run Windows binaries on Linux. Use for compile-only verification.

### Default Flags

```python
['-static', '-Wl,--subsystem,console', '-lntdll', '-w', '-O2']
```

## ClangVerifier

Alternative to GCC with different error messages:

```python
from halo_forge.rlvr.verifiers import ClangVerifier

verifier = ClangVerifier()
result = verifier.verify(code)
```

Same options as GCCVerifier.

## Reward Levels

| Stage | Reward | Details |
|-------|--------|---------|
| Syntax error | 0.0 | Doesn't compile |
| Compiles with warnings | 0.3 | `warn_as_error=True` |
| Compiles clean | 0.5 | No errors or warnings |
| Runs without crash | 0.7 | `run_after_compile=True` |
| Correct output | 1.0 | Matches `expected_output` |

## Code Extraction

Verifiers automatically extract code from model output:

```python
# Handles markdown blocks
text = '''Here is the solution:
```cpp
#include <iostream>
int main() { return 0; }
```
'''

verifier = GCCVerifier()
result = verifier.verify(text)  # Extracts code automatically
```

Supported patterns:
- ` ```cpp ... ``` `
- ` ```c++ ... ``` `
- `<code>...</code>`
- Raw code starting with `#include`

## Resource Limits

For execution safety:

```python
verifier = GCCVerifier(
    run_after_compile=True,
    run_timeout=5,        # Kill after 5 seconds
    memory_limit_mb=256   # Limit memory usage
)
```

## Batch Verification

```python
verifier = GCCVerifier(max_workers=8)
codes = [code1, code2, code3, ...]

results = verifier.verify_batch(codes)

for code, result in zip(codes, results):
    if result.success:
        print(f"✓ {result.reward}")
    else:
        print(f"✗ {result.error}")
```

## Cleanup

Verifiers create temp files during verification:

```python
verifier = GCCVerifier()
try:
    results = verifier.verify_batch(codes)
finally:
    verifier.cleanup()  # Remove temp files
```
