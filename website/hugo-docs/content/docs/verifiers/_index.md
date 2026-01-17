---
title: "Verifiers"
description: "Pluggable verification system for RLVR training"
---

Verifiers are the heart of RLVR — they provide the reward signal that guides training.

> **Important**: Verifiers are **training infrastructure**, not benchmarks. For benchmark reporting (comparing to papers), see [Benchmarking](/docs/training-pipeline/benchmarking/).

## Verifiers vs Benchmarks

Verifiers and benchmarks serve different purposes:

| Aspect | Verifiers (Training) | Benchmarks (Reporting) |
|--------|---------------------|------------------------|
| **Purpose** | Provide reward signal for RAFT | Compare model to published results |
| **Output** | Graduated rewards (0.0 to 1.0) | Metrics (pass@k, accuracy, WER) |
| **When Used** | During training loop | After training is complete |
| **Tooling** | Native halo-forge verifiers | Community tools (VLMEvalKit, etc.) |

**Use verifiers when:** Running RAFT training, need graduated feedback, debugging training

**Use benchmarks when:** Evaluating trained model, comparing to papers, publishing results

---

## Built-in Verifiers

### Compilation Verifiers

| Verifier | Language | Target | Compile | Run | Cross-Compile |
|----------|----------|--------|---------|-----|---------------|
| `GCCVerifier` | C/C++ | Linux ELF | Yes | Yes | - |
| `ClangVerifier` | C/C++ | Linux ELF | Yes | Yes | - |
| `MinGWVerifier` | C/C++ | Windows PE | Yes | No | - |
| `RemoteMSVCVerifier` | C/C++ | Windows PE | Yes | Yes | Requires Windows server |
| `RustVerifier` | Rust | Native/Windows | Yes | Yes | x86_64-pc-windows-gnu |
| `GoVerifier` | Go | Native/Windows | Yes | Yes | GOOS=windows |
| `DotNetVerifier` | C# | Windows PE | Yes | No | win-x64 |
| `PowerShellVerifier` | PowerShell | Script | Syntax | No | - |

### Test Verifiers

| Verifier | Language | Use Case |
|----------|----------|----------|
| `PytestVerifier` | Python | Code with tests |
| `UnittestVerifier` | Python | unittest format |
| `HumanEvalVerifier` | Python | HumanEval benchmark |
| `MBPPVerifier` | Python | MBPP benchmark |
| `SubprocessVerifier` | Any | Custom commands |

## Basic Usage

```python
from halo_forge.rlvr.verifiers import GCCVerifier

verifier = GCCVerifier()
result = verifier.verify(code)

print(result.success)   # True/False
print(result.reward)    # 0.0 - 1.0
print(result.details)   # Human-readable message
```

## Graduated Rewards

Binary rewards create sparse gradients. halo forge uses graduated rewards:

| Outcome | Reward | Signal |
|---------|--------|--------|
| Syntax error | 0.0 | Completely wrong |
| Compiles with warnings | 0.3 | Close but imperfect |
| Compiles clean | 0.5 | Correct syntax |
| Runs without crash | 0.7 | Executable |
| Correct output | 1.0 | Fully correct |

```python
from halo_forge.rlvr.verifiers import RewardLevel

# Get reward from compile result
reward = RewardLevel.from_compile_result(success=True, has_warnings=False)
# Returns 0.5

# Get reward from execution result
reward = RewardLevel.from_execution_result(
    compiles=True, 
    runs=True, 
    correct=False
)
# Returns 0.7
```

## Batch Verification

Verify multiple samples in parallel:

```python
verifier = GCCVerifier(max_workers=8)
codes = [code1, code2, code3, ...]

results = verifier.verify_batch(codes)  # Parallel execution

for result in results:
    print(f"{result.reward}: {result.details}")
```

## With RAFT Training

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

## Verifier Architecture

```
                         Verifier (base class)
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
  CompileVerifier          TestVerifier           CustomVerifier
        │                       │
   ┌────┼────┬────┬────┐   ┌────┴────┐
   │    │    │    │    │   │         │
  GCC MinGW Clang Rust Go Pytest  Unittest
              │
         RemoteMSVC
         DotNet
         PowerShell
```

## Chaining Verifiers

Run multiple verification stages:

```python
from halo_forge.rlvr.verifiers import ChainedVerifier, GCCVerifier

verifier = ChainedVerifier([
    GCCVerifier(),                        # Stage 1: Compile
    GCCVerifier(run_after_compile=True),  # Stage 2: Run
])

result = verifier.verify(code)
# Stops at first failure, accumulates rewards
```

## Cleanup

Always cleanup resources:

```python
verifier = GCCVerifier()

try:
    results = verifier.verify_batch(codes)
finally:
    verifier.cleanup()

# Or use context manager
with GCCVerifier() as verifier:
    results = verifier.verify_batch(codes)
```

---

## Verifier Safety

Verifiers execute model-generated code. Treat verifier runs as untrusted execution.

### Current Safety Characteristics

| Verifier Type | Safety Notes |
|---------------|--------------|
| `SubprocessVerifier` | Uses `shell=True`, inherits host environment |
| Python verifiers | Execute without OS-level resource limits |
| C/C++ verifiers | Apply `setrlimit` for CPU and memory during execution |

### Operational Guardrails (Recommended)

1. **Isolate execution**: Run verifiers in VMs or containers, avoid host mounts
2. **Non-privileged user**: Do not run verifiers as root
3. **Constrain resources**: Use OS limits or container quotas where possible
4. **Restrict network**: Prefer offline execution or blocked egress

These are operational recommendations, not enforced by halo-forge itself.
