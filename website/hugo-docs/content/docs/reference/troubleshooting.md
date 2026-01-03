---
title: "Troubleshooting"
description: "Common issues and solutions"
weight: 2
---

## GPU Issues

### GPU Not Detected

```bash
# Check device access
ls -l /dev/dri /dev/kfd

# Check ROCm
rocm-smi

# Check PyTorch
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Fix:** Add udev rules:

```bash
sudo tee /etc/udev/rules.d/99-amd-kfd.rules >/dev/null <<'EOF'
SUBSYSTEM=="kfd", GROUP="render", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="card[0-9]*", GROUP="render", MODE="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Low GPU Utilization

**Symptom:** GPU at 20-30% during training

**Cause:** Usually data loading bottleneck

**Fix:** Ensure correct settings:

```yaml
dataloader_num_workers: 0      # Must be 0 for Strix Halo
dataloader_pin_memory: false   # Must be false
```

### Wrong Memory Reported

**Symptom:** PyTorch shows ~25GB instead of 128GB

**Explanation:** This is normal. PyTorch reports VRAM allocation, not total unified memory.

**Check actual usage:**

```bash
radeontop  # Shows GTT usage
cat /sys/class/drm/card0/device/mem_info_vram_used
```

## Memory Issues

### Out of Memory During Training

**Fix 1:** Reduce batch size:

```yaml
per_device_train_batch_size: 1  # Reduce from 2
gradient_accumulation_steps: 32  # Increase to compensate
```

**Fix 2:** Enable gradient checkpointing:

```yaml
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
```

**Fix 3:** Use chunked verification (already default in RAFT):

```python
chunk_size = 200  # Verify in chunks
```

### Memory Creep During RAFT

**Symptom:** Memory usage grows each cycle

**Fix:** Ensure cleanup between cycles:

```python
gc.collect()
torch.cuda.empty_cache()
```

Check that verifier cleanup is called:

```python
verifier.cleanup()
```

## Training Issues

### Loss Not Decreasing

**Causes:**
- Learning rate too high or too low
- Data quality issues
- Model capacity too small

**Fixes:**
- Try learning rate: `1e-5`, `2e-5`, `5e-5`
- Check data format
- Use larger model

### Training Hangs

**Cause:** Usually data loading or CUDA sync issues

**Fix:**

```yaml
dataloader_num_workers: 0
dataloader_pin_memory: false
```

### NaN Loss

**Cause:** Gradient explosion or bad data

**Fixes:**
- Reduce learning rate
- Enable gradient clipping: `max_grad_norm: 0.3`
- Check for corrupt data samples

## Verification Issues

### Compiler Not Found

```
Error: Compiler 'g++' not found in PATH
```

**Fix:** Install compiler in toolbox:

```bash
# In toolbox
sudo dnf install gcc-c++

# For MinGW
sudo dnf install mingw64-gcc-c++
```

### Verification Timeout

**Symptom:** Many samples timing out

**Fixes:**
- Increase timeout: `timeout: 60`
- Check for infinite loops in generated code
- Add memory limits: `memory_limit_mb: 256`

### Low Compile Rate

**Symptom:** < 20% samples compile

**Causes:**
- Poor SFT model
- Wrong verifier for language
- Bad prompt format

**Fixes:**
- Train longer SFT
- Check verifier matches language
- Verify prompt format in data

## RAFT Issues

### Cycle Degradation

**Symptom:** Performance drops after cycle 5-6

**Cause:** Overfitting to verification signal

**Fixes:**
- Stop at peak cycle
- Add prompt diversity
- Reduce learning rate per cycle

### Samples Not Cached

**Symptom:** Re-generating samples on resume

**Fix:** Check cache directory exists:

```bash
ls models/raft/cache/
```

Ensure sufficient disk space.

### Verifier Resource Leak

**Symptom:** Temp files accumulating in `/tmp`

**Fix:** Ensure cleanup is called:

```python
try:
    trainer.run(prompts, num_cycles=5)
finally:
    trainer.verifier.cleanup()
```

## Getting Help

1. Check logs in output directory
2. Run with verbose logging
3. Create minimal reproduction
4. Check [GitHub Issues](https://github.com/professor-moody/halo-forge/issues)
