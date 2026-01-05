# Toolbox Setup Guide

Complete guide for building and using the halo-forge toolbox on AMD Strix Halo.

## Prerequisites

- **Hardware**: AMD Strix Halo with 128GB unified memory (gfx1151)
- **OS**: Fedora 42+ with toolbox/podman, **OR** Ubuntu 22.04+ with Docker
- **Kernel**: 6.16+ recommended (eliminates need for kernel parameters)
- **Disk Space**: ~25GB for build, ~50GB for models
- **Network**: Stable internet for downloading ROCm nightlies (~5GB)

## Choose Your Platform

| Platform | Container | Build Script | Status |
|----------|-----------|--------------|--------|
| **Fedora** | podman toolbox | `build.sh` | **Primary supported platform** |
| Ubuntu | Docker | `build-ubuntu.sh` | Experimental |

> **Note**: Ubuntu/Docker support is experimental and less tested than Fedora. Use the Fedora toolbox for production training runs. Reference: [kyuz0/amd-strix-halo-llm-finetuning](https://github.com/kyuz0/amd-strix-halo-llm-finetuning)

### Kernel Parameters (if kernel < 6.16)

Add these to `/etc/default/grub`:

```
GRUB_CMDLINE_LINUX="... amd_iommu=off amdgpu.gttsize=131072 ttm.pages_limit=33554432"
```

Then:
```bash
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
sudo reboot
```

## Quick Start (Fedora/podman)

```bash
# 1. Build the toolbox image
cd toolbox
chmod +x build.sh verify.sh
./build.sh --no-cache

# 2. Create and enter toolbox
toolbox rm -f halo-forge || true
toolbox create halo-forge --image localhost/halo-forge:latest
toolbox enter halo-forge

# 3. Install halo-forge package
cd ~/projects/halo-forge
pip install -e .

# 4. Verify environment
./toolbox/verify.sh

# 5. Run smoke test
halo-forge test --level smoke
```

## Quick Start (Ubuntu/Docker) — EXPERIMENTAL

> **Warning**: Ubuntu/Docker support is experimental. Use Fedora toolbox for production.

```bash
# 1. Build the Docker image
cd toolbox
chmod +x build-ubuntu.sh
./build-ubuntu.sh --no-cache

# 2. (If GPU not visible) Add udev rules first
sudo tee /etc/udev/rules.d/99-amd-kfd.rules >/dev/null <<'EOF'
SUBSYSTEM=="kfd", GROUP="render", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="card[0-9]*", GROUP="render", MODE="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger

# 3. Run the container with GPU access
docker run -it --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  -v ~/projects:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  halo-forge:ubuntu

# 4. Inside container: Install halo-forge package
cd /workspace/halo-forge
pip install -e .

# 5. Run smoke test
halo-forge test --level smoke
```

### Docker Run Options

```bash
# Basic run with GPU
docker run -it --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  -v ~/projects:/workspace \
  halo-forge:ubuntu

# With user mapping (avoids root file ownership)
docker run -it --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  -v ~/projects:/workspace \
  -u $(id -u):$(id -g) \
  halo-forge:ubuntu

# With HuggingFace cache persistence
docker run -it --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  -v ~/projects:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  halo-forge:ubuntu

# Named container (can stop/start later)
docker run -it --name halo-forge-dev \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  -v ~/projects:/workspace \
  halo-forge:ubuntu

# Resume named container
docker start -ai halo-forge-dev
```

## Build Details

### What Gets Installed

The toolbox image includes:

| Component | Version | Notes |
|-----------|---------|-------|
| Fedora | 42 | Base image |
| ROCm | 7.x nightly | Via TheRock tarballs |
| PyTorch | nightly | AMD gfx1151 build |
| bitsandbytes | upstream | Built with gfx1151 target |
| Flash Attention | ROCm fork | main_perf branch |
| Transformers | ≥4.46.0 | HuggingFace |
| PEFT | ≥0.13.0 | LoRA support |
| TRL | ≥0.12.0 | Training library |
| GCC/Clang | system | For code verification |
| MinGW | cross-compiler | Windows verification |

### Build Options

```bash
# Standard build (uses cache)
./build.sh

# Clean build (no cache - recommended for release)
./build.sh --no-cache

# Custom tag
./build.sh --tag v0.2.0
```

### Build Time Estimates

| Mode | Time | Notes |
|------|------|-------|
| With cache | 5-10 min | Incremental updates |
| Without cache | 20-40 min | Full rebuild, depends on network |

## Verification

After entering the toolbox, run the verification script:

```bash
# Full verification
./toolbox/verify.sh

# Quick check (skip GPU test)
./toolbox/verify.sh --quick

# Include GPU memory test
./toolbox/verify.sh --gpu
```

Expected output:
```
╭──────────────────────────────────────────────────────────────╮
│   HALO-FORGE Environment Verification                       │
╰──────────────────────────────────────────────────────────────╯

Python Environment
─────────────────────────────────────────────────────────────
  ✓ Python installed: Python 3.13.x
  ✓ Virtual environment active: /opt/venv

PyTorch & ROCm
─────────────────────────────────────────────────────────────
  ✓ PyTorch installed: 2.x.x
  ✓ GPU detected: AMD Radeon Graphics
  ✓ ROCM_PATH set: /opt/rocm-7.0

...

═══════════════════════════════════════════════════════════════
Summary
═══════════════════════════════════════════════════════════════

  Passed:   20
  Warnings: 0
  Failed:   0

All checks passed! Environment is ready for training.
```

## Training Configuration

### Use BF16 (NOT 4-bit)

4-bit quantization is **2x slower** on Strix Halo due to compute-bound workload. Use BF16:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Recommended for Strix Halo
    device_map="auto",
    attn_implementation="eager"  # Or flash_attention_2
)
```

### Critical Dataloader Settings

Required for unified memory architecture:

```yaml
training:
  dataloader_num_workers: 0   # Required - prevents GPU hangs
  dataloader_pin_memory: false # Required - prevents GPU hangs
```

### Memory Configuration

The Dockerfile sets recommended memory allocation:

```bash
PYTORCH_HIP_ALLOC_CONF="backend:native,expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512"
```

### Actual Memory Usage (7B model, 2048 seq len, BF16 LoRA)

| Metric | Observed |
|--------|----------|
| GTT (GPU system memory) | ~62GB |
| VRAM (dedicated cache) | ~1GB |
| Total system RAM | ~71GB |
| GPU utilization | 95-99% |

## Memory Monitoring

```bash
# Real-time GPU stats
radeontop

# Or use rocm-smi
watch -n 1 rocm-smi

# Check GTT usage
rocm-smi --showmeminfo all
```

> **Note:** PyTorch's `torch.cuda.max_memory_allocated()` reports VRAM only (~24GB), not GTT. Use `radeontop` for accurate unified memory usage.

## Running Training

### RAFT Training

```bash
# Quick validation run (0.5B model, 1 cycle)
halo-forge raft train \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --prompts data/rlvr/mbpp_train_prompts.jsonl \
  --verifier mbpp \
  --cycles 1 \
  --samples-per-prompt 4 \
  --output models/validation

# Production run (7B model, 5 cycles)
halo-forge raft train \
  --model Qwen/Qwen2.5-Coder-7B \
  --prompts data/rlvr/mbpp_train_prompts.jsonl \
  --verifier mbpp \
  --cycles 5 \
  --output models/production_7b
```

## Troubleshooting

### GPU not visible

Check devices are accessible:
```bash
ls -l /dev/dri /dev/kfd
```

If missing, add udev rules:
```bash
sudo tee /etc/udev/rules.d/99-amd-kfd.rules >/dev/null <<'EOF'
SUBSYSTEM=="kfd", GROUP="render", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="card[0-9]*", GROUP="render", MODE="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Low memory visible

Ensure kernel parameters are set correctly. With kernel 6.16+, this should not be needed.

### GPU hang during training

1. Ensure `dataloader_num_workers: 0`
2. Ensure `dataloader_pin_memory: false`
3. Add `export HSA_ENABLE_SDMA=0` to your environment

### Slow generation

1. Verify using BF16 (not 4-bit quantization)
2. Check GPU is at 95%+ compute utilization
3. Ensure only one training process running

### Build fails with network timeout

1. Check internet connectivity
2. Try with VPN if ROCm servers are slow
3. Retry - sometimes TheRock servers are busy

### "Command not found: halo-forge"

```bash
cd ~/projects/halo-forge
pip install -e .
```

## File Structure

```
toolbox/
├── Dockerfile           # Fedora container definition (for podman toolbox)
├── Dockerfile.ubuntu    # Ubuntu container definition (for Docker)
├── build.sh            # Fedora/podman build script
├── build-ubuntu.sh     # Ubuntu/Docker build script
├── verify.sh           # Post-build verification
├── TOOLBOX_SETUP.md    # This file
└── scripts/
    ├── 01-triton-env.sh       # Triton/ROCm paths
    ├── 99-halo-forge-banner.sh # Welcome banner
    └── zz-venv-path-fix.sh    # Ensure venv is first in PATH
```

## Persistence

Your home directory is automatically mounted in the toolbox:
- `~/` on host = `~/` in toolbox
- Models, datasets, checkpoints all persist
- HuggingFace cache at `~/.cache/huggingface/`

## Acknowledgments

Based on [kyuz0/amd-strix-halo-llm-finetuning](https://github.com/kyuz0/amd-strix-halo-llm-finetuning) with community improvements.
