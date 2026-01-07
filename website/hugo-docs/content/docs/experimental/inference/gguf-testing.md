---
title: "GGUF Export Testing"
weight: 35
---

# GGUF Export Testing Guide

Step-by-step guide to testing GGUF model export for llama.cpp and Ollama deployment.

## Prerequisites

### Option 1: llama-cpp-python (Recommended)

```bash
# CPU-only (simplest)
pip install llama-cpp-python

# AMD GPU (ROCm)
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --force-reinstall

# NVIDIA GPU (CUDA)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall
```

### Option 2: llama.cpp Clone

```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp

# Build (CPU)
make

# Build (ROCm)
make GGML_HIPBLAS=1

# Build (CUDA)
make GGML_CUDA=1
```

### Optional: Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version
```

---

## Quick Test

Test the export script with a small model:

```bash
# List available quantization types
python scripts/export_gguf.py --list-quantizations

# Expected output:
# Available quantization types:
#   Q4_K_M - 4-bit quantization, medium (recommended)
#   Q4_K_S - 4-bit quantization, small
#   Q8_0   - 8-bit quantization
#   F16    - 16-bit float (no quantization)
#   F32    - 32-bit float (full precision)
```

---

## Step-by-Step Testing

### 1. Export a Small Model

```bash
# Export Qwen 0.5B with Q4_K_M quantization
python scripts/export_gguf.py \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --output qwen_0.5b.Q4_K_M.gguf \
  --quantization Q4_K_M

# Expected output:
# Loading model from Qwen/Qwen2.5-Coder-0.5B...
# Converting to GGUF format...
# Applying Q4_K_M quantization...
# Saved to qwen_0.5b.Q4_K_M.gguf
# File size: ~XXX MB
```

### 2. Verify Export

```bash
# Check file was created
ls -lh qwen_0.5b.Q4_K_M.gguf

# Expected: ~300-400 MB for 0.5B model with Q4_K_M

# Verify file format
file qwen_0.5b.Q4_K_M.gguf
# Expected: data (GGUF uses custom binary format)
```

### 3. Test with llama.cpp

```bash
# If using llama.cpp clone
~/llama.cpp/llama-cli \
  -m qwen_0.5b.Q4_K_M.gguf \
  -p "Write a Python function to sort a list:" \
  -n 100

# Expected: Model loads and generates code
```

### 4. Test with Ollama

Create a `Modelfile`:

```bash
cat > Modelfile << 'EOF'
FROM ./qwen_0.5b.Q4_K_M.gguf

SYSTEM "You are an expert programmer."
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF
```

Import and run:

```bash
# Create model in Ollama
ollama create qwen-test -f Modelfile

# Run interactive chat
ollama run qwen-test

# Test with a prompt
ollama run qwen-test "Write hello world in Python"

# Cleanup when done
ollama rm qwen-test
```

---

## Quantization Comparison Test

Compare different quantization levels:

```bash
# Export with different quantizations
python scripts/export_gguf.py \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --output qwen_0.5b.Q8_0.gguf \
  --quantization Q8_0

python scripts/export_gguf.py \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --output qwen_0.5b.F16.gguf \
  --quantization F16

# Compare file sizes
ls -lh qwen_0.5b.*.gguf

# Expected sizes (approximate for 0.5B model):
# Q4_K_M: ~300 MB
# Q8_0:   ~500 MB
# F16:    ~1 GB
```

---

## End-to-End Workflow Test

Test the complete training-to-deployment pipeline:

### Step 1: Train a Model

```bash
# Quick RAFT training (minimal cycles)
halo-forge raft train \
  --prompts datasets/windows_curriculum/windows_systems_rlvr.jsonl \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --verifier mingw \
  --cycles 2 \
  --samples-per-prompt 2 \
  --output models/test_raft
```

### Step 2: Export to GGUF

```bash
python scripts/export_gguf.py \
  --model models/test_raft/cycle_2_final \
  --output test_raft.Q4_K_M.gguf \
  --quantization Q4_K_M
```

### Step 3: Validate with llama.cpp

```bash
# Test inference
~/llama.cpp/llama-cli \
  -m test_raft.Q4_K_M.gguf \
  -p "Write a Windows API call to create a process:" \
  -n 200
```

### Step 4: Deploy with Ollama

```bash
# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./test_raft.Q4_K_M.gguf
SYSTEM "You are an expert Windows systems programmer specializing in low-level C code."
PARAMETER temperature 0.3
EOF

# Deploy
ollama create windows-coder -f Modelfile
ollama run windows-coder "Create a DLL with DllMain"
```

---

## Validation Checklist

After testing, verify:

- [ ] `--list-quantizations` shows all options
- [ ] Export completes without errors
- [ ] Output file has expected size
- [ ] llama.cpp can load and run the model
- [ ] Ollama can create and run the model
- [ ] Generated output is coherent
- [ ] Fine-tuned knowledge is preserved

---

## Troubleshooting

### "llama-cpp-python not installed"

Install with:

```bash
pip install llama-cpp-python
```

For GPU support, add the appropriate `CMAKE_ARGS`.

### "conversion failed" or "unsupported architecture"

Some model architectures require the llama.cpp clone:

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp && make

# Use --llama-cpp-path flag
python scripts/export_gguf.py \
  --model your_model \
  --output model.gguf \
  --llama-cpp-path ~/llama.cpp
```

### Ollama "unsupported model format"

Ensure the GGUF file was created correctly:

```bash
# Check file header
xxd qwen_0.5b.Q4_K_M.gguf | head -1
# Should start with: 47475546 (GGUF magic bytes)
```

If invalid, re-export with a different method or quantization.

### Model produces garbage output

- Try a higher quality quantization (Q8_0 instead of Q4_K_M)
- Verify the source model works before export
- Check if the model architecture is fully supported

### Export takes too long

- Use a smaller model for testing (0.5B or 1.5B)
- Ensure you have enough disk space (2x model size)
- Close other memory-intensive applications

---

## Size and Quality Reference

| Model Size | Q4_K_M | Q8_0 | F16 |
|------------|--------|------|-----|
| 0.5B | ~300 MB | ~500 MB | ~1 GB |
| 1.5B | ~1 GB | ~1.5 GB | ~3 GB |
| 3B | ~2 GB | ~3 GB | ~6 GB |
| 7B | ~4 GB | ~7 GB | ~14 GB |

Quality typically:
- **F16**: 100% (baseline)
- **Q8_0**: ~99%
- **Q4_K_M**: ~95-97%
- **Q4_K_S**: ~93-95%

---

## Next Steps

- [GGUF Export Guide](./gguf-export.md) - Full documentation
- [Inference Testing](./testing.md) - Test other inference features
- [Ollama Documentation](https://ollama.ai/docs) - Ollama deployment
