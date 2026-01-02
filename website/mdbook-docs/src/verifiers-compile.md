# Compile Verifiers

Verifiers for C/C++ code compilation.

## GCCVerifier

Local compilation with GCC on Linux.

```python
from halo_forge.rlvr.verifiers import GCCVerifier

verifier = GCCVerifier(
    compiler="g++",
    flags=["-std=c++17", "-Wall", "-Wextra"],
    run_after_compile=False
)
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `compiler` | `g++` | Compiler binary |
| `flags` | `["-std=c++17"]` | Compilation flags |
| `run_after_compile` | `False` | Execute after compile |
| `timeout` | `30` | Seconds before timeout |

### Reward Mapping

| Outcome | Reward |
|---------|--------|
| Compile error | 0.0 |
| Compile with warnings | 0.3 |
| Compile clean | 0.5 |
| Runs without crash | 0.7 |
| Correct output | 1.0 |

## ClangVerifier

Same interface as GCC, uses Clang:

```python
from halo_forge.rlvr.verifiers import ClangVerifier

verifier = ClangVerifier(
    compiler="clang++",
    flags=["-std=c++17", "-Wall"]
)
```

## MinGWVerifier

Cross-compile Windows binaries on Linux:

```python
from halo_forge.rlvr.verifiers import MinGWVerifier

verifier = MinGWVerifier(
    compiler="x86_64-w64-mingw32-g++",
    flags=["-std=c++17", "-static"]
)
```

Requires MinGW installed:

```bash
# Fedora
sudo dnf install mingw64-gcc-c++

# Ubuntu
sudo apt install g++-mingw-w64-x86-64
```

## RemoteMSVCVerifier

Compile on a remote Windows machine via SSH:

```python
from halo_forge.rlvr.verifiers import RemoteMSVCVerifier

verifier = RemoteMSVCVerifier(
    host="192.168.1.100",
    user="developer",
    ssh_key="~/.ssh/win_key",
    msvc_path="C:\\Program Files\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat"
)
```

### Windows Setup

1. Install Visual Studio Build Tools
2. Install OpenSSH Server
3. Configure SSH key authentication

See [Windows DEVBOX Setup](https://github.com/professor-moody/halo-forge/blob/main/docs/WINDOWS_SETUP.md) for details.

## Runtime Verification

Enable execution after compilation:

```python
verifier = GCCVerifier(
    run_after_compile=True,
    expected_output="Hello, World!\n"
)
```

This gives higher rewards (0.7-1.0) for code that runs correctly.
