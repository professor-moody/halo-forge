---
title: "GGUF Export"
weight: 30
---

# GGUF Export

Export fine-tuned models to GGUF format for llama.cpp, Ollama, and local inference.

## Why GGUF?

- **Fast inference** - Optimized for CPU and GPU
- **Small files** - Quantization reduces size 2-4x
- **No Python needed** - Deploy without Python runtime
- **Ollama compatible** - Easy model serving

## Quick Start

```bash
# Install dependency
pip install llama-cpp-python

# Convert your model
python scripts/export_gguf.py \
  --model models/my_finetuned \
  --output my_model.gguf
```

## Installation

### CPU-only (simplest)

```bash
pip install llama-cpp-python
```

### AMD GPU (ROCm)

```bash
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --force-reinstall
```

### Clone llama.cpp (most control)

```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp && make
```

## Quantization Options

| Type | Size | Quality | Description |
|------|------|---------|-------------|
| Q4_K_M | ~2GB/7B | Good | **Recommended** |
| Q8_0 | ~4GB/7B | Best | Higher quality |
| F16 | ~7GB/7B | Original | No quantization |

```bash
# List all options
python scripts/export_gguf.py --list-quantizations
```

## Using with Ollama

1. Create a `Modelfile`:

```
FROM ./my_model.gguf
SYSTEM "You are an expert Windows systems programmer."
PARAMETER temperature 0.7
```

2. Create and run:

```bash
ollama create mymodel -f Modelfile
ollama run mymodel
```

## Complete Workflow

```bash
# Train
halo-forge raft train \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --model Qwen/Qwen2.5-Coder-1.5B \
  --verifier mingw \
  --cycles 6 \
  --output models/windows_raft

# Export
python scripts/export_gguf.py \
  --model models/windows_raft/final_model \
  --output windows_coder.Q4_K_M.gguf

# Deploy
ollama create windows-coder -f Modelfile
ollama run windows-coder
```

## See Also

- [Full GGUF Export Guide](/home/keys/projects/halo-forge/docs/GGUF_EXPORT.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Ollama](https://ollama.ai)
