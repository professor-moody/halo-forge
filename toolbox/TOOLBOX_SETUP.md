# Toolbox Setup Guide

## Prerequisites

- Fedora 42+ with toolbox/podman installed
- AMD Strix Halo hardware (gfx1151)
- Kernel 6.16+ recommended (eliminates need for kernel parameters)

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

## Build the Toolbox

```bash
cd toolbox
chmod +x build.sh
./build.sh
```

This builds a container image with:
- Fedora 42 base
- Python 3.11
- PyTorch 2.7 with ROCm support (scottt's wheel)
- All ML libraries (transformers, peft, trl, etc.)

## Create and Enter

```bash
# Create the toolbox
toolbox create halo-forge --image localhost/halo-forge:latest

# Enter the toolbox
toolbox enter halo-forge
```

## Verify Setup

Inside the toolbox:

```bash
# Check PyTorch sees the GPU
python3 -c "import torch; print(f'ROCm: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Check memory
python3 -c "import torch; print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

Expected output:
```
ROCm: True
GPU: AMD Radeon(TM) Graphics
Memory: 96.0 GB
```

## Persistence

Your home directory is automatically mounted in the toolbox:
- `~/` on host = `~/` in toolbox
- Models, datasets, checkpoints all persist
- HuggingFace cache at `~/.cache/huggingface/`

## Known Issues

### bitsandbytes (4-bit Quantization)

The ROCm 7 PyTorch wheels may have a version mismatch with bitsandbytes:

```
RuntimeError: Configured ROCm binary not found at .../libbitsandbytes_rocm65.so
```

**Workaround**: Use bf16 training without quantization. With 96GB+ unified memory, you don't need 4-bit quantization:

```python
# Use bf16 instead of QLoRA 4-bit:
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Full bf16
    device_map="auto"
)
```

**Actual memory usage (7B model, 2048 seq len, bf16 LoRA)**:

| Metric | Observed |
|--------|----------|
| GTT (GPU system memory) | ~62GB |
| VRAM (dedicated) | ~1GB |
| Total system RAM | ~71GB |
| GPU utilization | 100% |

With 96-128GB unified memory, bf16 LoRA is recommended over 4-bit QLoRA.

**Memory monitoring**: Use `radeontop` (install via `cargo install radeontop`) to see GTT usage. PyTorch's `torch.cuda.max_memory_allocated()` reports incorrectly (~24GB when actual is ~62GB).

**To fix bitsandbytes** (compile from source):

```bash
pip uninstall bitsandbytes
git clone https://github.com/bitsandbytes-foundation/bitsandbytes
cd bitsandbytes
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=hip -S .
make
pip install .
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

### Performance issues

gfx1151 training is 30-40x slower than H100. This is expected due to driver maturity. See `docs/HARDWARE_NOTES.md` for optimization tips.

