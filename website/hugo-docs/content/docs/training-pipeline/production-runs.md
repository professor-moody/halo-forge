---
title: "Production Training Runs"
description: "Step-by-step commands for training all model sizes on the Windows Systems Programming dataset"
weight: 6
---

This guide provides copy-paste commands for running production training on all model sizes using the Windows Systems Programming dataset. Start with the 0.5B model to validate your setup, then scale up.

## Quick Reference

| Model | SFT Time | RAFT Time | Total |
|-------|----------|-----------|-------|
| Qwen2.5-Coder-0.5B | ~30 min | ~1 hour | ~1.5 hours |
| Qwen2.5-Coder-1.5B | ~1 hour | ~2 hours | ~3 hours |
| Qwen2.5-Coder-3B | ~1.5 hours | ~3 hours | ~4.5 hours |
| Qwen2.5-Coder-7B | ~2.5 hours | ~5 hours | ~7.5 hours |

**Dataset**: Windows Curriculum (361 problems, MinGW/MSVC compatible)

**Default Verifier**: MinGW (no Windows machine required)

---

## Pre-Flight Checklist

Before starting, verify your environment is ready.

### Fedora (Toolbox)

```bash
# 1. Check toolbox exists
toolbox list | grep halo-forge

# 2. Check dataset exists
ls -la ~/projects/halo-forge/datasets/windows_curriculum/windows_systems_full_*.jsonl

# 3. Check disk space (need ~50GB free)
df -h ~

# 4. Check GPU
rocm-smi

# 5. Check MinGW is installed
x86_64-w64-mingw32-g++ --version
```

### Ubuntu/Docker

```bash
# 1. Check image exists
docker images | grep halo-forge
# or: podman images | grep halo-forge

# 2-5. Same checks as Fedora (run from host before entering container)
```

---

## Verifier Options

The Windows dataset can be verified with two different compilers:

| Verifier | Platform | Compiles | Executes | Requires |
|----------|----------|----------|----------|----------|
| `mingw`  | Linux    | Yes      | No       | `mingw-w64` package |
| `msvc`   | Remote   | Yes      | Optional | Windows build server |

### MinGW (Recommended for Getting Started)

Use MinGW for quick compile-only verification on Linux. No Windows machine required.

```bash
# Install MinGW (Fedora)
sudo dnf install mingw64-gcc-c++

# Install MinGW (Ubuntu)
sudo apt install mingw-w64

# Quick benchmark with MinGW
halo-forge benchmark run \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --verifier mingw \
  --samples 10 \
  --output results/windows/baseline_mingw.json
```

**Limitation**: MinGW can only verify that code compiles. It cannot run the executables.

### MSVC (Full Verification)

Use MSVC for full compile + run + output verification. Requires a Windows build server.

See the **[Windows Build Server](/docs/reference/windows-setup/)** guide for server configuration.

| Scenario | Recommended Verifier |
|----------|---------------------|
| Getting started / no Windows available | `mingw` (default) |
| Debugging compile issues | `mingw` (faster iteration) |
| Full output verification | `msvc` (with `run_after_compile=True`) |
| Production training (advanced) | `msvc` |

**Recommendation**: Start with MinGW. It's simpler and doesn't require Windows setup.

---

## Initial Setup

Run these commands once to prepare your environment.

### Fedora (Toolbox)

```bash
# Enter toolbox
toolbox enter halo-forge

# Navigate to project
cd ~/projects/halo-forge

# Install halo-forge
pip install -e .

# Create results directory
mkdir -p results/windows
```

### Ubuntu/Docker

```bash
# From the halo-forge directory on your host:
cd /path/to/halo-forge

# Run container with GPU access (Docker)
docker run -it --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  -v $(pwd):/workspace/halo-forge \
  halo-forge:ubuntu

# Or with podman (rootless, recommended on Fedora host):
podman run -it --userns=keep-id \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  -v $(pwd):/workspace/halo-forge:Z \
  halo-forge:ubuntu

# Inside container:
cd /workspace/halo-forge
pip install -e .
mkdir -p results/windows
```

Once inside the container, all `halo-forge` commands work identically on both platforms.

---

## Training Pattern

Each model follows the same four-step pattern:

1. **Baseline** - Measure the untrained model's performance
2. **SFT** - Supervised fine-tuning on solutions (optional but recommended)
3. **RAFT** - Reinforcement learning with compile verification
4. **Benchmark** - Measure the trained model's performance

All commands below use `--verifier mingw` by default. Replace with `--verifier msvc` if you have a Windows build server configured.

---

## 0.5B Model (Start Here)

The smallest model - use this to validate your setup before scaling up.

### Baseline Benchmark

```bash
halo-forge benchmark run \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --verifier mingw \
  --samples 10 \
  --output results/windows/baseline_0.5b.json
```

### SFT Training (~30 min)

```bash
screen -S sft_0.5b

halo-forge sft train \
  --data datasets/windows_curriculum/windows_systems_full_sft.jsonl \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --output models/windows_sft_0.5b \
  --epochs 2

# Detach: Ctrl+A, D
```

### RAFT Training (~1 hour)

```bash
screen -S raft_0.5b

halo-forge raft train \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --verifier mingw \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --checkpoint models/windows_sft_0.5b/final_model \
  --cycles 6 \
  --output models/windows_raft_0.5b

# Detach: Ctrl+A, D
```

### Final Benchmark

```bash
halo-forge benchmark run \
  --model models/windows_raft_0.5b/cycle_6_final \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --verifier mingw \
  --samples 10 \
  --k 1,5,10 \
  --output results/windows/trained_0.5b.json
```

### Compare Results

```bash
echo "=== 0.5B Results ==="
echo "Baseline:" && cat results/windows/baseline_0.5b.json | jq '.pass_at_k'
echo "Trained:" && cat results/windows/trained_0.5b.json | jq '.pass_at_k'
```

---

## 1.5B Model

### Baseline Benchmark

```bash
halo-forge benchmark run \
  --model Qwen/Qwen2.5-Coder-1.5B \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --verifier mingw \
  --samples 10 \
  --output results/windows/baseline_1.5b.json
```

### SFT Training (~1 hour)

```bash
screen -S sft_1.5b

halo-forge sft train \
  --data datasets/windows_curriculum/windows_systems_full_sft.jsonl \
  --model Qwen/Qwen2.5-Coder-1.5B \
  --output models/windows_sft_1.5b \
  --epochs 2

# Detach: Ctrl+A, D
```

### RAFT Training (~2 hours)

```bash
screen -S raft_1.5b

halo-forge raft train \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --verifier mingw \
  --model Qwen/Qwen2.5-Coder-1.5B \
  --checkpoint models/windows_sft_1.5b/final_model \
  --cycles 6 \
  --output models/windows_raft_1.5b

# Detach: Ctrl+A, D
```

### Final Benchmark

```bash
halo-forge benchmark run \
  --model models/windows_raft_1.5b/cycle_6_final \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --verifier mingw \
  --samples 10 \
  --k 1,5,10 \
  --output results/windows/trained_1.5b.json
```

### Compare Results

```bash
echo "=== 1.5B Results ==="
echo "Baseline:" && cat results/windows/baseline_1.5b.json | jq '.pass_at_k'
echo "Trained:" && cat results/windows/trained_1.5b.json | jq '.pass_at_k'
```

---

## 3B Model

### Baseline Benchmark

```bash
halo-forge benchmark run \
  --model Qwen/Qwen2.5-Coder-3B \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --verifier mingw \
  --samples 10 \
  --output results/windows/baseline_3b.json
```

### SFT Training (~1.5 hours)

```bash
screen -S sft_3b

halo-forge sft train \
  --data datasets/windows_curriculum/windows_systems_full_sft.jsonl \
  --model Qwen/Qwen2.5-Coder-3B \
  --output models/windows_sft_3b \
  --epochs 2

# Detach: Ctrl+A, D
```

### RAFT Training (~3 hours)

```bash
screen -S raft_3b

halo-forge raft train \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --verifier mingw \
  --model Qwen/Qwen2.5-Coder-3B \
  --checkpoint models/windows_sft_3b/final_model \
  --cycles 6 \
  --output models/windows_raft_3b

# Detach: Ctrl+A, D
```

### Final Benchmark

```bash
halo-forge benchmark run \
  --model models/windows_raft_3b/cycle_6_final \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --verifier mingw \
  --samples 10 \
  --k 1,5,10 \
  --output results/windows/trained_3b.json
```

### Compare Results

```bash
echo "=== 3B Results ==="
echo "Baseline:" && cat results/windows/baseline_3b.json | jq '.pass_at_k'
echo "Trained:" && cat results/windows/trained_3b.json | jq '.pass_at_k'
```

---

## 7B Model

The largest supported model. Requires gradient checkpointing for memory efficiency.

### Baseline Benchmark

```bash
halo-forge benchmark run \
  --model Qwen/Qwen2.5-Coder-7B \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --verifier mingw \
  --samples 10 \
  --output results/windows/baseline_7b.json
```

### SFT Training (~2.5 hours)

```bash
screen -S sft_7b

# For 7B, use config file to enable gradient checkpointing
halo-forge sft train \
  --config configs/production_7b.yaml \
  --data datasets/windows_curriculum/windows_systems_full_sft.jsonl \
  --output models/windows_sft_7b \
  --epochs 2

# Detach: Ctrl+A, D
```

### RAFT Training (~5 hours)

```bash
screen -S raft_7b

halo-forge raft train \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --verifier mingw \
  --model Qwen/Qwen2.5-Coder-7B \
  --checkpoint models/windows_sft_7b/final_model \
  --cycles 6 \
  --output models/windows_raft_7b

# Detach: Ctrl+A, D
```

### Final Benchmark

```bash
halo-forge benchmark run \
  --model models/windows_raft_7b/cycle_6_final \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --verifier mingw \
  --samples 10 \
  --k 1,5,10 \
  --output results/windows/trained_7b.json
```

### Compare Results

```bash
echo "=== 7B Results ==="
echo "Baseline:" && cat results/windows/baseline_7b.json | jq '.pass_at_k'
echo "Trained:" && cat results/windows/trained_7b.json | jq '.pass_at_k'
```

---

## Summary Report

After training all models, generate a summary report:

```bash
echo "============================================" > results/windows/summary.txt
echo "WINDOWS SYSTEMS PROGRAMMING TRAINING RESULTS" >> results/windows/summary.txt
echo "Date: $(date)" >> results/windows/summary.txt
echo "Dataset: 361 problems (100% compile rate)" >> results/windows/summary.txt
echo "============================================" >> results/windows/summary.txt

for size in 0.5b 1.5b 3b 7b; do
  echo "" >> results/windows/summary.txt
  echo "--- Qwen2.5-Coder-${size} ---" >> results/windows/summary.txt
  
  if [ -f results/windows/baseline_${size}.json ]; then
    echo "Baseline:" >> results/windows/summary.txt
    cat results/windows/baseline_${size}.json | jq -r '"  pass@1: \(.pass_at_k["1"])"' >> results/windows/summary.txt
  fi
  
  if [ -f results/windows/trained_${size}.json ]; then
    echo "Trained:" >> results/windows/summary.txt
    cat results/windows/trained_${size}.json | jq -r '"  pass@1: \(.pass_at_k["1"])"' >> results/windows/summary.txt
  fi
done

cat results/windows/summary.txt
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| GPU hang | `export HSA_ENABLE_SDMA=0` before training |
| OOM error | Use config file with `gradient_checkpointing: true` or reduce batch size |
| MinGW not found | `sudo dnf install mingw64-gcc-c++` or `sudo apt install mingw-w64` |
| MSVC verifier timeout | Check Windows server connectivity |
| Training crash | Re-run same command (auto-resume from checkpoint) |
| Slow training | Check GPU usage with `radeontop` |

---

## Useful Commands

```bash
# Check screen sessions
screen -ls

# Reattach to session
screen -r sft_0.5b

# Monitor GPU
radeontop

# Check disk usage
du -sh models/windows_*

# Kill stuck training
pkill -f "halo-forge"

# View training logs
tail -f models/windows_raft_0.5b/training.log
```

---

## Results Tracking Template

Use this template to track your results:

| Model | Baseline pass@1 | After SFT | After RAFT | Improvement |
|-------|-----------------|-----------|------------|-------------|
| 0.5B | | | | |
| 1.5B | | | | |
| 3B | | | | |
| 7B | | | | |

---

## Verifier Configuration

### MinGW (Default)

MinGW works out of the box with just the `--verifier mingw` flag. No config file required.

### MSVC (Advanced)

For full compile + run + output verification, create `configs/raft_windows_msvc.yaml`:

```yaml
# RAFT with MSVC Verifier for Windows training
base_model: Qwen/Qwen2.5-Coder-0.5B  # Override with --model
num_cycles: 6
output_dir: models/raft

verifier:
  type: msvc
  host: YOUR_WINDOWS_IP
  user: YOUR_USERNAME
  ssh_key: ~/.ssh/win
```

Then use `--config configs/raft_windows_msvc.yaml` with your RAFT commands.

---

## Complete Verifier Reference

### Compilation Verifiers

| Verifier | Language | Target | Compile | Run | Binary Cache | Cross-Compile | Requires |
|----------|----------|--------|---------|-----|--------------|---------------|----------|
| `gcc` | C/C++ | Linux ELF | Yes | Yes | Yes | - | gcc/g++ |
| `clang` | C/C++ | Linux ELF | Yes | Yes | Yes | - | clang/clang++ |
| `mingw` | C/C++ | Windows PE | Yes | No | Yes | - | mingw-w64 |
| `msvc` | C/C++ | Windows PE | Yes | Yes | Yes | - | Windows build server |
| `rust` | Rust | Native/Windows | Yes | Yes | Yes | x86_64-pc-windows-gnu | cargo, rustup |
| `go` | Go | Native/Windows | Yes | Yes | Yes | GOOS=windows | go |
| `dotnet` | C# | Windows PE | Yes | No | Yes | win-x64 | dotnet SDK |
| `powershell` | PS1 | Script | Syntax | No | Yes | - | pwsh or Windows |

### Test Verifiers

| Verifier | Language | Description | Requires |
|----------|----------|-------------|----------|
| `pytest` | Python | Run pytest tests | pytest |
| `unittest` | Python | Run unittest tests | (built-in) |
| `humaneval` | Python | HumanEval benchmark | (built-in) |
| `mbpp` | Python | MBPP benchmark | (built-in) |

### Verifier Quick Reference

```bash
# C/C++ on Linux
halo-forge benchmark run --verifier gcc ...

# Windows PE (cross-compile on Linux, no execution)
halo-forge benchmark run --verifier mingw ...

# Windows PE (compile + run on Windows server)
halo-forge benchmark run --verifier msvc ...

# Rust (native)
halo-forge benchmark run --verifier rust ...

# Go (native)
halo-forge benchmark run --verifier go ...

# C#/.NET (cross-compile to Windows)
halo-forge benchmark run --verifier dotnet ...

# PowerShell (syntax check)
halo-forge benchmark run --verifier powershell ...

# Python (MBPP benchmark)
halo-forge benchmark run --verifier mbpp ...
```

### Graduated Rewards

All verifiers return graduated rewards for partial credit:

| Stage | Reward | Description |
|-------|--------|-------------|
| Compile fail | 0.0 | Does not compile |
| Compile with warnings | 0.3 | Compiles with warnings |
| Compile clean | 0.5 | Compiles without warnings |
| Runs without crash | 0.7 | Executes successfully |
| Correct output | 1.0 | Output matches expected |
