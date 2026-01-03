---
title: "Toolbox Setup"
description: "Build and configure the halo-forge container environment"
weight: 2
---

The halo-forge toolbox is a containerized environment with ROCm and PyTorch configured for Strix Halo.

## Why a Toolbox?

AMD Strix Halo (gfx1151) requires:

- ROCm 7 nightly builds (not yet in stable releases)
- PyTorch compiled specifically for gfx1151
- Specific library versions for unified memory

The toolbox packages all of this into a reproducible environment.

## Building

```bash
cd halo-forge/toolbox
./build.sh
```

Build time: ~15-30 minutes depending on network speed.

### What Gets Installed

| Component | Version | Notes |
|-----------|---------|-------|
| ROCm | 7.x nightly | From TheRock project |
| PyTorch | 2.6 nightly | Built for gfx1151 |
| Transformers | 4.46+ | HuggingFace |
| PEFT | 0.13+ | LoRA adapters |
| TRL | 0.12+ | Training utilities |

## Usage

```bash
# Create the toolbox
toolbox create halo-forge --image localhost/halo-forge:latest

# Enter toolbox
toolbox enter halo-forge

# Check GPU access
python3 -c "import torch; print(torch.cuda.is_available())"

# Exit
exit
```

## Persistence

Your home directory is automatically mounted in the toolbox:

- `~/` on host = `~/` in toolbox
- Models, datasets, checkpoints all persist
- HuggingFace cache at `~/.cache/huggingface/`

## Build Options

```bash
# Build with cache (default, faster)
./build.sh

# Build from scratch (slower, ensures fresh)
./build.sh --no-cache
```

## Updating

```bash
cd halo-forge/toolbox
git pull
./build.sh --no-cache
```

## Troubleshooting

### Build Failures

Ensure you have:

- Sufficient disk space (~20GB for images)
- Network access to github.com, huggingface.co, PyPI

### GPU Not Detected in Toolbox

Check device permissions on the host:

```bash
ls -l /dev/dri /dev/kfd
```

The toolbox automatically mounts these devices, but permissions must allow access.
