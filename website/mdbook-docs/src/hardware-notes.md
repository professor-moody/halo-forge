# Hardware Notes

Configuration and performance findings for AMD Strix Halo.

## Specifications

| Component | Spec |
|-----------|------|
| GPU | AMD gfx1151 |
| Memory | 128GB unified (shared with CPU) |
| Architecture | RDNA 3.5 |
| ROCm Support | 7.x nightly (TheRock) |

## Key Findings

### BF16 is Optimal

Contrary to typical GPU setups, **4-bit quantization is slower** on Strix Halo due to dequantization overhead. The unified memory architecture makes BF16 the optimal choice:

| Precision | Memory | Speed | Quality |
|-----------|--------|-------|---------|
| BF16 | ~14GB/7B | Fast | Full |
| 4-bit | ~4GB/7B | Slower | Reduced |

### Generation Speed

| Model Size | Tokens/sec |
|------------|------------|
| 0.5B | 220-230 |
| 1.5B | 185-195 |
| 3B | 130-135 |
| 7B | 60-100 |

### Training Throughput

During RAFT training:
- **GPU utilization**: 95-99% (compute-bound)
- **GTT memory**: 40-60GB typical for 7B
- **Power consumption**: ~100W sustained

## Required Settings

These settings are **mandatory** for stable training:

```yaml
training:
  dataloader_num_workers: 0    # Must be 0
  dataloader_pin_memory: false # Must be false
```

Also set in your environment:

```bash
export HSA_ENABLE_SDMA=0
```

## Memory Management

The 128GB unified memory means you rarely hit OOM, but if you do:

1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Enable `gradient_checkpointing: true`
4. Reduce `max_seq_length`

## Monitoring

```bash
# GPU utilization (requires sudo for accurate readings)
sudo rocm-smi --showuse

# Memory usage
rocm-smi --showmeminfo vram

# Temperature
rocm-smi --showtemp

# Continuous monitoring
watch -n 1 rocm-smi
```

## Known Issues

- **GPU hang during training**: Usually caused by `dataloader_num_workers > 0`
- **Slow first generation**: Normal, subsequent generations are faster
- **rocm-smi permissions**: Requires root for accurate utilization stats
