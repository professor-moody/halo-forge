# ROCm Troubleshooting Guide

Common issues and solutions when running halo-forge with AMD GPUs on ROCm.

## Table of Contents

- [GPU Detection Issues](#gpu-detection-issues)
- [Memory Errors](#memory-errors)
- [Performance Issues](#performance-issues)
- [PyTorch/ROCm Compatibility](#pytorchrocm-compatibility)
- [Container Issues](#container-issues)
- [Strix Halo Specific](#strix-halo-specific)

---

## GPU Detection Issues

### GPU not detected by PyTorch

**Symptoms:**
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Solutions:**

1. **Check ROCm installation:**
   ```bash
   rocm-smi
   # Should show your GPU
   ```

2. **Verify device permissions:**
   ```bash
   ls -la /dev/kfd /dev/dri/render*
   # Your user should have rw access
   ```

3. **Add user to render/video groups:**
   ```bash
   sudo usermod -aG render,video $USER
   # Log out and back in
   ```

4. **Check HIP visibility:**
   ```bash
   export HIP_VISIBLE_DEVICES=0
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Wrong GPU architecture detected

**Symptoms:**
```
RuntimeError: HIP error: the file path is invalid
```

**Solution:** Set the correct GPU architecture:
```bash
# For Strix Halo (gfx1151)
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export PYTORCH_ROCM_ARCH=gfx1151

# For RDNA3 desktop (gfx1100)
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
```

---

## Memory Errors

### Out of Memory (OOM) during generation

**Symptoms:**
```
torch.cuda.OutOfMemoryError: HIP out of memory
```

**Solutions:**

1. **Reduce batch size:**
   ```bash
   halo-forge raft train --model ... --generation-batch-size 4
   ```

2. **Use a smaller model:**
   ```bash
   # Try 3B instead of 7B
   halo-forge raft train --model Qwen/Qwen2.5-Coder-3B ...
   ```

3. **Clear cache in code:**
   ```python
   import torch
   import gc
   
   # After generation
   del outputs
   gc.collect()
   torch.cuda.empty_cache()
   ```

4. **Monitor memory usage:**
   ```bash
   watch -n 1 rocm-smi
   ```

### OOM kills during training (system-level)

**Symptoms:**
- Process killed without Python error
- `dmesg | grep -i oom` shows OOM killer

**Solutions:**

1. **Close memory-heavy applications** (browsers, IDEs) during long runs

2. **Reduce gradient accumulation memory:**
   ```bash
   # Smaller effective batch size
   --train-batch-size 1 --gradient-accumulation-steps 32
   ```

3. **Enable gradient checkpointing** (enabled by default in halo-forge)

4. **For Strix Halo unified memory:** The GPU shares system RAM. Ensure you have enough free system memory:
   ```bash
   free -h
   # Should have at least 16GB free for 7B models
   ```

---

## Performance Issues

### Training is very slow

**Possible causes and solutions:**

1. **Check GPU utilization:**
   ```bash
   rocm-smi
   # GPU% should be high during training
   ```

2. **Ensure bfloat16 is used:**
   ```python
   # In your code or config
   torch_dtype=torch.bfloat16
   ```

3. **Check for CPU bottlenecks:**
   - Data loading
   - Tokenization
   - Verification (especially remote MSVC)

4. **Disable torch.compile on older Python:**
   ```bash
   export TORCH_COMPILE_DISABLE=1
   ```

### GPU utilization is low

**Solutions:**

1. **Increase batch size** (if memory allows)

2. **Check for sync points:**
   ```python
   # Avoid frequent .item() or .cpu() calls
   # Use torch.cuda.synchronize() sparingly
   ```

3. **Profile your code:**
   ```python
   with torch.profiler.profile() as prof:
       # Your training code
   print(prof.key_averages().table())
   ```

---

## PyTorch/ROCm Compatibility

### Flash Attention not working

**Symptoms:**
```
ImportError: cannot import name 'flash_attn_func' from 'flash_attn'
```

**Solution:** Flash Attention requires architecture-specific compilation:

```bash
# For gfx1151 (Strix Halo), use pre-compiled wheels or:
pip install flash-attn --no-build-isolation

# Or disable flash attention:
export TRANSFORMERS_NO_FLASH_ATTENTION=1
```

### bitsandbytes errors

**Symptoms:**
```
RuntimeError: CUDA error: no kernel image is available
```

**Solution:** Use ROCm-compatible bitsandbytes:

```bash
# Uninstall CUDA version
pip uninstall bitsandbytes

# Install ROCm fork
pip install bitsandbytes-rocm
# Or build from source for your architecture
```

### Triton compilation errors

**Symptoms:**
```
triton.compiler.CompilationError: ...
```

**Solutions:**

1. **Set Triton cache:**
   ```bash
   export TRITON_CACHE_DIR=/tmp/triton_cache
   ```

2. **Disable Triton:**
   ```bash
   export TRITON_DISABLE=1
   ```

3. **Use specific Triton version:**
   ```bash
   pip install triton==3.0.0
   ```

---

## Container Issues

### toolbox/podman GPU access

**Symptoms:**
GPU not visible inside container.

**Solution:** Ensure proper device mapping:

```bash
# Check Dockerfile has:
# --device=/dev/kfd --device=/dev/dri

# Or run manually:
podman run --device=/dev/kfd --device=/dev/dri \
  --security-opt=label=disable \
  -it halo-forge bash
```

### Permission denied on /dev/kfd

**Solutions:**

1. **Run verify.sh to check permissions:**
   ```bash
   ./toolbox/verify.sh
   ```

2. **Check SELinux (Fedora):**
   ```bash
   # Temporarily disable for testing
   sudo setenforce 0
   
   # Or add proper labels
   sudo chcon -t container_file_t /dev/kfd
   ```

3. **Use security-opt:**
   ```bash
   podman run --security-opt=label=disable ...
   ```

---

## Strix Halo Specific

### Unified Memory Considerations

Strix Halo APUs use unified memory (shared between CPU and GPU). This has implications:

1. **No discrete VRAM:** GPU memory comes from system RAM
2. **Allocation behavior:** Large allocations may succeed but cause swapping
3. **Monitor total memory:** Use `free -h` not just `rocm-smi`

**Recommendations for Strix Halo:**

| Model Size | System RAM Needed | Notes |
|------------|------------------|-------|
| 0.5B-1.5B  | 16GB             | Comfortable |
| 3B         | 24GB             | Works well |
| 7B         | 32GB+            | May need to close other apps |
| 14B+       | 64GB+            | Tight, reduce batch sizes |

### gfx1151 Architecture

Strix Halo uses gfx1151 (RDNA 3.5). Ensure environment is set:

```bash
# In your .bashrc or container profile
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export PYTORCH_ROCM_ARCH=gfx1151
export HIP_VISIBLE_DEVICES=0
```

### Known Issues

1. **torch.compile:** May have issues on Python 3.14+. Disable if problems:
   ```bash
   export TORCH_COMPILE_DISABLE=1
   ```

2. **Power management:** Ensure power profile allows full GPU performance:
   ```bash
   # Check current profile
   cat /sys/class/drm/card*/device/power_dpm_force_performance_level
   
   # Set to high (may require root)
   echo high | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level
   ```

---

## Getting Help

If you encounter issues not covered here:

1. **Check logs:** `dmesg | tail -50` for kernel messages
2. **Verify ROCm:** `rocm-smi --showall`
3. **Test PyTorch:** `python -c "import torch; print(torch.cuda.get_device_properties(0))"`
4. **Open an issue:** Include ROCm version, GPU model, and full error traceback

