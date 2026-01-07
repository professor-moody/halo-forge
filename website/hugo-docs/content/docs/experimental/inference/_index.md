---
title: "Inference Optimization"
weight: 20
---

> **⚠️ Experimental Feature**: Inference optimization is under active development. APIs may change.

# Inference Optimization

Optimize trained models for efficient deployment with halo-forge's inference tools.

## Overview

After training with SFT and RAFT, models can be optimized for production:

```
Training Complete → Quantization → Quality Verification → Export
```

## Key Features

| Feature | Description | CLI Command |
|---------|-------------|-------------|
| **Quantization** | Reduce model size with INT4/INT8 | `halo-forge inference optimize` |
| **Quality Verification** | Ensure quality meets thresholds | Automatic during optimize |
| **GGUF Export** | Export for llama.cpp/Ollama | `halo-forge inference export --format gguf` |
| **ONNX Export** | Export for cross-platform inference | `halo-forge inference export --format onnx` |
| **Latency Benchmarking** | Measure inference speed | `halo-forge inference benchmark` |

## Quick Start

```bash
# Optimize a trained model
halo-forge inference optimize \
  --model models/windows_raft_7b/cycle_6_final \
  --target-precision int4 \
  --output models/optimized

# Export to GGUF for Ollama
halo-forge inference export \
  --model models/optimized \
  --format gguf \
  --quantization Q4_K_M \
  --output models/windows-7b.gguf

# Benchmark latency
halo-forge inference benchmark \
  --model models/optimized \
  --num-prompts 20
```

## Quantization Options

| Precision | Size Reduction | Quality | Use Case |
|-----------|----------------|---------|----------|
| `int4` | ~75% | Good | Edge deployment |
| `int8` | ~50% | Better | Server deployment |
| `fp16` | ~50% | Best | High-quality inference |

## Export Formats

### GGUF (llama.cpp)

Best for:
- Local inference with Ollama
- CPU-only systems
- Memory-constrained environments

Quantization types:
- `Q4_K_M` - Recommended balance (quality/size)
- `Q4_K_S` - Smaller, slightly lower quality
- `Q8_0` - Highest quality 8-bit
- `F16` - No quantization (largest)

### ONNX

Best for:
- Cross-platform deployment
- Integration with ONNX Runtime
- TensorRT/OpenVINO optimization
- Web deployment

## Quality Verification

During optimization, halo-forge automatically verifies:

1. **Latency** - Meets target (default: 50ms)
2. **Quality** - Output similarity to original model (default: 95%)

```bash
# Custom thresholds
halo-forge inference optimize \
  --model models/trained \
  --target-latency 100 \
  --target-precision int4
```

## Standalone GGUF Export Script

For a simpler workflow, use the standalone export script:

```bash
# Install llama-cpp-python (one-time)
pip install llama-cpp-python

# Convert directly from trained model to GGUF
python scripts/export_gguf.py \
  --model models/windows_raft_1.5b/final_model \
  --output windows_coder.Q4_K_M.gguf \
  --quantization Q4_K_M
```

See the [GGUF Export Guide](./gguf-export.md) for full documentation.

## Next Steps

- [GGUF Export](./gguf-export.md) - Standalone export script
- [Quantization Guide](./quantization.md) - Detailed quantization options
- [Export Formats](./export.md) - Format-specific documentation
- [Benchmarking](./benchmarking.md) - Measure and compare performance
