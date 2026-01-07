---
title: "Testing Guide"
weight: 50
---

# Inference Module Testing Guide

Step-by-step guide to testing the halo-forge Inference optimization module.

## Prerequisites

### Required Dependencies

```bash
pip install torch transformers accelerate
```

### Optional Dependencies

| Dependency | Required For | Install |
|------------|--------------|---------|
| `bitsandbytes` | INT4/INT8 quantization | `pip install bitsandbytes` |
| `llama-cpp-python` | GGUF export | `pip install llama-cpp-python` |
| `optimum` | ONNX export | `pip install optimum` |
| `onnxruntime` | ONNX inference | `pip install onnxruntime` |

### Hardware Requirements

- **Minimum**: 16 GB RAM, 8 GB VRAM
- **Recommended**: 32 GB RAM, 24 GB VRAM (for 7B models)
- **ROCm Support**: AMD Radeon RX 7900 series or higher

---

## Unit Tests

Run the full inference test suite:

```bash
# From project root
cd /path/to/halo-forge

# Run all inference tests
pytest tests/test_inference.py -v

# Run specific test class
pytest tests/test_inference.py::TestOptimizationConfig -v
pytest tests/test_inference.py::TestInferenceOptimizer -v
pytest tests/test_inference.py::TestGGUFExporter -v
```

### Expected Test Coverage

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestOptimizationConfig` | 3 | Config defaults and validation |
| `TestInferenceOptimizer` | 4 | Optimizer initialization and checks |
| `TestVerifier` | 3 | InferenceOptimizationVerifier |
| `TestQATConfig` | 2 | Quantization-aware training config |
| `TestQATTrainer` | 3 | QAT training workflow |
| `TestCalibration` | 3 | Calibration dataset creation |
| `TestGGUFExporter` | 3 | GGUF export workflow |
| `TestONNXExporter` | 3 | ONNX export workflow |
| `TestIntegration` | 2 | End-to-end workflows |

---

## Dry Run Testing

Test commands without running actual optimization:

```bash
# Test CLI parsing and config validation
halo-forge inference optimize \
  --model Qwen/Qwen2.5-Coder-1.5B \
  --target-precision int4 \
  --output /tmp/test-output \
  --dry-run

# Expected output:
# Dry run mode enabled. Configuration validated successfully.
# Model: Qwen/Qwen2.5-Coder-1.5B
# Target precision: int4
# Output: /tmp/test-output
```

---

## Manual Testing Steps

### 1. Test Quantization

```bash
# INT4 quantization (requires bitsandbytes)
halo-forge inference optimize \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --target-precision int4 \
  --output models/quantized_int4

# Verify output
ls -la models/quantized_int4/
# Should contain: config.json, model.safetensors, tokenizer.json
```

### 2. Test GGUF Export

Using the CLI:

```bash
halo-forge inference export \
  --model models/quantized_int4 \
  --format gguf \
  --quantization Q4_K_M \
  --output models/test.gguf
```

Using the standalone script:

```bash
# List available quantization types
python scripts/export_gguf.py --list-quantizations

# Export with Q4_K_M (recommended)
python scripts/export_gguf.py \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --output test_model.Q4_K_M.gguf \
  --quantization Q4_K_M

# Verify the output
file test_model.Q4_K_M.gguf
# Expected: test_model.Q4_K_M.gguf: data
```

### 3. Test ONNX Export

```bash
halo-forge inference export \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --format onnx \
  --output models/test_onnx/

# Verify output
ls models/test_onnx/
# Should contain: model.onnx, config.json
```

### 4. Test Latency Benchmarking

```bash
halo-forge inference benchmark \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --num-prompts 10
  
# Expected output:
# Benchmarking Qwen/Qwen2.5-Coder-0.5B...
# Average latency: XX.X ms
# P95 latency: XX.X ms
# Throughput: XX.X tokens/sec
```

---

## Validation Checklist

After running tests, verify:

- [ ] All unit tests pass (`pytest tests/test_inference.py`)
- [ ] Dry run mode works without errors
- [ ] Quantized models load and generate text
- [ ] GGUF export produces valid files
- [ ] ONNX export produces valid files
- [ ] Benchmark completes without errors

---

## Troubleshooting

### "bitsandbytes not installed"

This is a warning, not an error. Install if you need quantization:

```bash
pip install bitsandbytes
```

### "llama-cpp-python not installed"

Required only for GGUF export. Install with:

```bash
# CPU-only
pip install llama-cpp-python

# AMD GPU (ROCm)
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --force-reinstall
```

### GGUF export fails

If `llama-cpp-python` doesn't work, clone llama.cpp directly:

```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp && make

# Then export
python scripts/export_gguf.py \
  --model your_model \
  --output model.gguf \
  --llama-cpp-path ~/llama.cpp
```

### Out of Memory (OOM)

- Use smaller models (0.5B or 1.5B) for testing
- Set `--target-precision int4` to reduce memory
- Add `--device cpu` if GPU memory is insufficient

### ROCm-specific Issues

Set environment variables if you encounter HIP errors:

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export HSA_OVERRIDE_GFX_VERSION=11.5.1  # For Strix Halo
```

Or use the experimental attention flag:

```bash
halo-forge inference optimize \
  --model your_model \
  --experimental-attention \
  ...
```

---

## Next Steps

- [GGUF Export Guide](./gguf-export.md) - Detailed GGUF documentation
- [Quantization Options](./) - Full quantization reference
- [VLM Testing Guide](../vlm/testing.md) - Test VLM features
