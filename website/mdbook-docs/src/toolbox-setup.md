# Toolbox Setup

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
# Enter toolbox
toolbox enter halo-forge

# Check GPU access
python3 -c "import torch; print(torch.cuda.is_available())"

# Exit
exit
```

## Troubleshooting

### GPU Not Detected

Check device permissions:

```bash
ls -l /dev/dri /dev/kfd
```

Add udev rules if needed:

```bash
sudo tee /etc/udev/rules.d/99-amd-kfd.rules >/dev/null <<'EOF'
SUBSYSTEM=="kfd", GROUP="render", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="card[0-9]*", GROUP="render", MODE="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Build Failures

Ensure you have sufficient disk space (~20GB for images) and network access to:
- `github.com`
- `huggingface.co`
- PyPI mirrors

## Updating

```bash
cd halo-forge/toolbox
git pull
./build.sh --no-cache
```
