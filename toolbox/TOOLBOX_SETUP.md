# Toolbox Setup Guide

## Prerequisites

- Fedora 42+ with toolbox/podman installed
- AMD Strix Halo hardware (gfx1151) with 128GB unified memory
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
- ROCm 7 nightly from TheRock
- PyTorch nightly from AMD gfx1151 nightlies
- bitsandbytes built from upstream with gfx1151 target
- Flash Attention from ROCm fork (main_perf branch)
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
Memory: ~25-30 GB (VRAM cache, actual unified memory is 128GB)
```

> **Note:** PyTorch reports VRAM allocation, not full unified memory. Use `radeontop` to see actual GTT usage.

## Persistence

Your home directory is automatically mounted in the toolbox:
- `~/` on host = `~/` in toolbox
- Models, datasets, checkpoints all persist
- HuggingFace cache at `~/.cache/huggingface/`

## Recommended Settings

### Use BF16 (NOT 4-bit)

4-bit quantization is **2x slower** on Strix Halo due to compute-bound workload and dequantization overhead. Use BF16:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Optimal for Strix Halo
    device_map="auto",
    attn_implementation="eager"  # Or flash_attention_2 with ROCm fork
)
```

### Dataloader Settings

Critical for unified memory architecture:

```yaml
training:
  dataloader_num_workers: 0   # Required - prevents GPU hangs
  dataloader_pin_memory: false # Required - prevents GPU hangs
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
```

> **Note:** PyTorch's `torch.cuda.max_memory_allocated()` reports VRAM only (~24GB), not GTT. Use `radeontop` for accurate unified memory usage.

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

See `docs/HARDWARE_NOTES.md` for detailed optimization guidance.

## Acknowledgments

Based on [kyuz0/amd-strix-halo-llm-finetuning](https://github.com/kyuz0/amd-strix-halo-llm-finetuning) with community improvements.
