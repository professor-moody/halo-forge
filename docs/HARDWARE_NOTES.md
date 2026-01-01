# Hardware Notes: AMD Strix Halo (gfx1151)

Performance findings and recommendations for training on AMD Strix Halo.

## Hardware Specifications

| Component | Specification |
|-----------|---------------|
| GPU | AMD Strix Halo (RDNA 3.5) |
| GPU ID | gfx1151 |
| Compute Units | 40 CUs (2560 shaders) |
| Memory | 128GB unified LPDDR5X |
| Memory Bandwidth | 273 GB/s |
| TDP | 120W |

## Key Insight: Unified Memory Architecture

Strix Halo uses **unified memory** - the GPU and CPU share the same memory pool. This is fundamentally different from discrete GPUs with dedicated VRAM.

- **VRAM pool**: ~2GB (small dedicated cache)
- **GTT (Graphics Translation Table)**: ~128GB (system memory accessible by GPU)
- **Total available**: 128GB+

This architecture has important implications for training optimization.

---

## Performance Findings

### BF16 is Optimal ✅

We tested various precision modes and found **BF16 is optimal** for Strix Halo:

| Precision | Speed | Notes |
|-----------|-------|-------|
| BF16 | Baseline | Optimal for Strix Halo |
| FP16 | Similar | Less numerically stable |
| 4-bit (NF4) | **2x slower** | Dequantization overhead |

### Why 4-bit Quantization is Slower

1. **Dequantization overhead**: 4-bit requires converting weights back to BF16/FP16 during each forward pass, adding compute cycles.

2. **Compute-bound workload**: Strix Halo runs at 96-99% compute utilization with BF16. Since compute is the bottleneck (not memory), 4-bit's memory savings don't help.

3. **Unified memory negates VRAM savings**: The main benefit of 4-bit is fitting models in limited VRAM. With 128GB unified memory, you're never VRAM-limited.

---

## Optimal Configuration

### Training Settings

```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  bf16: true
  optim: "adamw_torch"
  
  # CRITICAL for unified memory - prevents GPU hangs
  dataloader_num_workers: 0
  dataloader_pin_memory: false
```

### Generation Settings

```yaml
generation:
  batch_size: 8
  samples_per_prompt: 8
  max_new_tokens: 1024
```

### Environment Variables

```bash
# Memory management for unified memory
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"

# Use hipBLASLt for optimized BLAS (already set in toolbox)
export ROCBLAS_USE_HIPBLASLT=1

# Disable SDMA (prevents hangs on some systems)
export HSA_ENABLE_SDMA=0
```

---

## What Works Well

| Feature | Status | Notes |
|---------|--------|-------|
| BF16 training | ✅ | Optimal precision |
| Flash Attention | ✅ | Works via ROCm fork (main_perf branch) |
| LoRA | ✅ | Full support |
| Gradient checkpointing | ✅ | Essential for 7B+ models |
| Large batch sizes | ✅ | 128GB allows generous batching |

## What Doesn't Work Well

| Feature | Status | Notes |
|---------|--------|-------|
| 4-bit quantization | ❌ | 2x slower than BF16 |
| FP16 | ⚠️ | Less stable than BF16 on AMD |
| SDPA attention | ⚠️ | 30-40% slower than eager |
| dataloader_num_workers > 0 | ❌ | Causes GPU hangs |
| dataloader_pin_memory | ❌ | Causes GPU hangs |

---

## Performance Expectations

### RAFT Training (7B model, ~500 prompts, 8 samples/prompt)

| Phase | Time | Notes |
|-------|------|-------|
| Generation | ~7-8 hours | ~6s per sample |
| Verification | ~30-60 min | Parallelized on CPU |
| Training | ~30 min | Per cycle |
| **Total per cycle** | ~8-9 hours | |

### SFT Training

| Dataset Size | Epochs | Estimated Time |
|--------------|--------|----------------|
| 10K samples | 3 | ~8 hours |
| 50K samples | 3 | ~2 days |
| 100K samples | 3 | ~4 days |

---

## Monitoring

### Check GPU utilization

```bash
# Real-time GPU stats
watch -n 1 rocm-smi

# Or use radeontop
radeontop
```

### Expected metrics during training

- **Command Processor - Compute**: 95-99% (compute-bound)
- **VRAM**: 500-800 MB (small cache)
- **GTT**: 20-30 GB (actual GPU memory usage)
- **GFX Activity**: 95-99%

---

## Troubleshooting

### GPU hang during training

1. Ensure `dataloader_num_workers: 0`
2. Ensure `dataloader_pin_memory: false`
3. Add `export HSA_ENABLE_SDMA=0`

### Out of memory

Unlikely with 128GB, but if it happens:
1. Reduce batch size
2. Enable gradient checkpointing
3. Reduce max_seq_length

### Slow generation

1. Verify using BF16 (not 4-bit)
2. Check GPU is at 95%+ utilization
3. Ensure only one training process running

---

## ROCm Stack

The halo-forge toolbox uses:

| Component | Version | Source |
|-----------|---------|--------|
| ROCm | 7.x nightly | TheRock S3 bucket |
| PyTorch | Nightly | AMD nightlies for gfx1151 |
| bitsandbytes | Upstream | Built with gfx1151 target |
| Flash Attention | ROCm fork | main_perf branch |

## Acknowledgments

- AMD for Strix Halo hardware
- kyuz0 for the original fine-tuning toolbox
- TheRock project for ROCm nightlies
- The Strix Halo community for testing and feedback

