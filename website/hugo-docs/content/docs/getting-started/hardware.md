---
title: "Hardware Notes"
description: "Configuration for AMD Strix Halo"
weight: 3
---

AMD Strix Halo with 128GB unified memory requires specific configuration.

## Critical Settings

These settings are **required** for stable training:

```yaml
training:
  dataloader_num_workers: 0    # MUST be 0
  dataloader_pin_memory: false # MUST be false
  bf16: true                   # Recommended precision
```

### Why These Matter

Strix Halo uses unified memory where CPU and GPU share the same RAM. Standard PyTorch data loading optimizations cause issues:

- **`dataloader_num_workers: 0`** — Multiple workers compete for unified memory
- **`dataloader_pin_memory: false`** — Pinned memory is redundant when CPU/GPU share RAM

## Use BF16, Not 4-bit

**BF16 is recommended.** In our testing, 4-bit quantization was slower on Strix Halo due to compute-bound workload and dequantization overhead.

```yaml
bf16: true
fp16: false
load_in_4bit: false  # Don't use QLoRA
```

## GPU Utilization

Expected utilization during training:

| Phase | GPU Util | Memory |
|-------|----------|--------|
| Generation | 95-99% | 40-60GB |
| Verification | 10-20% | 20-30GB |
| Training | 90-98% | 80-100GB |

Monitor with:

```bash
watch -n 1 rocm-smi
rocm-smi --showmeminfo vram
```

> **Note:** PyTorch reports VRAM allocation (~25GB), not full unified memory (~128GB). Use `radeontop` to see actual GTT usage.

## GPU Not Detected

```bash
# Check device access
ls -l /dev/dri /dev/kfd

# Add udev rules
sudo tee /etc/udev/rules.d/99-amd-kfd.rules >/dev/null <<'EOF'
SUBSYSTEM=="kfd", GROUP="render", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="card[0-9]*", GROUP="render", MODE="0666"
EOF

sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Kernel Parameters (if kernel < 6.16)

Add to `/etc/default/grub`:

```
GRUB_CMDLINE_LINUX="... amd_iommu=off amdgpu.gttsize=131072 ttm.pages_limit=33554432"
```

Then:

```bash
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
sudo reboot
```

## Environment Variables

```bash
export HSA_ENABLE_SDMA=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES=0
```

## Recommended Model Sizes

| Model Size | Training Time (5 cycles) | Memory Usage |
|------------|-------------------------|--------------|
| 0.5B | ~2 hours | 30-40GB |
| 1.5B | ~4 hours | 50-60GB |
| 3B | ~6 hours | 70-80GB |
| 7B | ~12 hours | 90-110GB |
