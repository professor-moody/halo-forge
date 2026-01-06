---
title: "Compile Verifiers"
description: "Multi-language compilation verification"
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

---

## RustVerifier

For Rust code, with optional Windows cross-compilation:

```python
from halo_forge.rlvr.verifiers import RustVerifier

# Native Rust (Linux)
verifier = RustVerifier()
result = verifier.verify(rust_code)

# Cross-compile to Windows
verifier = RustVerifier(cross_compile=True)
result = verifier.verify(windows_rust_code)
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `timeout` | 60 | Compilation timeout (seconds) |
| `max_workers` | 4 | Parallel compilations |
| `cross_compile` | False | Compile for Windows |
| `binary_cache_dir` | None | Save compiled binaries |

### Cross-Compile Requirements

```bash
# Install Windows target
rustup target add x86_64-pc-windows-gnu
```

---

## GoVerifier

For Go code, with optional Windows cross-compilation:

```python
from halo_forge.rlvr.verifiers import GoVerifier

# Native Go (Linux)
verifier = GoVerifier()
result = verifier.verify(go_code)

# Cross-compile to Windows
verifier = GoVerifier(cross_compile=True)
result = verifier.verify(windows_go_code)
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `timeout` | 60 | Compilation timeout (seconds) |
| `max_workers` | 4 | Parallel compilations |
| `cross_compile` | False | Compile for Windows (GOOS=windows) |
| `binary_cache_dir` | None | Save compiled binaries |

---

## DotNetVerifier

For C# code, cross-compiling to Windows PE:

```python
from halo_forge.rlvr.verifiers import DotNetVerifier

verifier = DotNetVerifier()
result = verifier.verify(csharp_code)
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `timeout` | 120 | Build timeout (seconds) |
| `max_workers` | 4 | Parallel builds |
| `target_framework` | `net8.0` | .NET version |
| `self_contained` | True | Include runtime |
| `single_file` | True | Single executable |
| `binary_cache_dir` | None | Save compiled binaries |

### Requirements

```bash
# .NET SDK 8.0
dotnet --version  # Should show 8.x
```

> **Note:** Creates Windows executables that cannot be run on Linux. Use for compile-only verification.

---

## PowerShellVerifier

For PowerShell scripts with syntax validation:

```python
from halo_forge.rlvr.verifiers import PowerShellVerifier

# Local validation (requires pwsh)
verifier = PowerShellVerifier(validation_mode="local")
result = verifier.verify(ps1_script)

# Remote validation (Windows server via SSH)
verifier = PowerShellVerifier(
    validation_mode="remote",
    win_host="10.0.0.152",
    win_user="keys",
    win_key="~/.ssh/win"
)

# Auto-detect best method
verifier = PowerShellVerifier(validation_mode="auto")
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `timeout` | 30 | Validation timeout |
| `validation_mode` | `auto` | `local`, `remote`, or `auto` |
| `use_pwsh` | True | Use `pwsh` (PowerShell Core) |
| `win_host` | None | Windows SSH host (for remote) |
| `win_user` | None | SSH username |
| `win_key` | None | SSH key path |
| `win_password` | None | SSH password (use key preferred) |
| `binary_cache_dir` | None | Save scripts |

---

## RemoteMSVCVerifier

Compile C/C++ on a remote Windows server via SSH:

```python
from halo_forge.rlvr.verifiers import RemoteMSVCVerifier

verifier = RemoteMSVCVerifier(
    host="10.0.0.152",
    user="keys",
    ssh_key="~/.ssh/win"
)
result = verifier.verify(windows_code)

# With execution
verifier = RemoteMSVCVerifier(
    host="10.0.0.152",
    user="keys",
    ssh_key="~/.ssh/win",
    run_after_compile=True,
    expected_output="Expected output"
)
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `host` | Required | Windows server IP/hostname |
| `user` | Required | SSH username |
| `ssh_key` | None | Path to SSH private key |
| `password` | None | SSH password (key preferred) |
| `timeout` | 60 | Compilation timeout |
| `run_after_compile` | False | Execute after compile |
| `run_timeout` | 10 | Execution timeout |
| `expected_output` | None | Expected stdout |
| `stdin_input` | None | Input to program |
| `binary_cache_dir` | None | Save binaries locally |

### Setup

See [Windows Setup Guide](/docs/reference/windows-setup/) for configuring a Windows build server.

---

## Binary Caching

All compile verifiers support caching compiled binaries for later analysis:

```python
verifier = GCCVerifier(binary_cache_dir="binaries/gcc")
result = verifier.verify(code)
# Binary saved to binaries/gcc/<uuid>.out

verifier = MinGWVerifier(binary_cache_dir="binaries/win")
result = verifier.verify(code)
# Binary saved to binaries/win/<uuid>.exe
```

The cache path is included in the result metadata:

```python
result = verifier.verify(code)
print(result.metadata.get('binary_path'))  # Path to cached binary
```
