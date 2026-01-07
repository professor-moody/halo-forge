# GGUF Export Guide

Export your fine-tuned models to GGUF format for use with llama.cpp, Ollama, and other local inference tools.

## Why GGUF?

- **Fast inference** - Optimized for CPU and GPU
- **Small files** - Quantization reduces model size 2-4x
- **No Python needed** - Deploy without Python runtime
- **Ollama compatible** - Easy model serving
- **Wide support** - Works with llama.cpp, LM Studio, GPT4All, etc.

## Quick Start

```bash
# Install llama-cpp-python (one-time)
pip install llama-cpp-python

# Convert your model
python scripts/export_gguf.py --model models/my_finetuned --output my_model.gguf
```

## Installation Options

### Option 1: CPU-only (simplest)

```bash
pip install llama-cpp-python
```

### Option 2: AMD GPU (ROCm)

For faster conversion with GPU acceleration:

```bash
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Option 3: NVIDIA GPU (CUDA)

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Option 4: Clone llama.cpp (most control)

```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp
make -j$(nproc)

# For ROCm support:
make GGML_HIPBLAS=1 -j$(nproc)
```

## Usage

### Basic Conversion

```bash
python scripts/export_gguf.py \
  --model models/windows_raft_1.5b/final_model \
  --output windows_coder.gguf
```

### With Specific Quantization

```bash
python scripts/export_gguf.py \
  --model models/windows_raft_1.5b/final_model \
  --output windows_coder.Q8_0.gguf \
  --quantization Q8_0
```

### List Quantization Options

```bash
python scripts/export_gguf.py --list-quantizations
```

## Quantization Types

| Type | Size | Quality | Use Case |
|------|------|---------|----------|
| Q4_K_M | ~2GB/7B | Good | **Recommended** - Best balance |
| Q4_K_S | ~2GB/7B | Fair | Memory constrained |
| Q5_K_M | ~2.5GB/7B | Better | Higher quality needed |
| Q8_0 | ~4GB/7B | Best | Quality critical |
| F16 | ~7GB/7B | Original | No quality loss |

**Rule of thumb:**
- For 1-3B models: Use Q8_0 (can afford the space)
- For 7B models: Use Q4_K_M (recommended)
- For 13B+ models: Use Q4_K_S (save memory)

## Using with Ollama

After converting to GGUF:

1. Create a `Modelfile`:

```
FROM ./windows_coder.gguf

SYSTEM "You are an expert Windows systems programmer."

PARAMETER temperature 0.7
PARAMETER num_ctx 4096
```

2. Create and run the model:

```bash
ollama create windows-coder -f Modelfile
ollama run windows-coder
```

3. Use via API:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "windows-coder",
  "prompt": "Write a function to enumerate running processes on Windows"
}'
```

## Using with llama.cpp

```bash
# Interactive mode
./llama.cpp/llama-cli -m windows_coder.gguf -i

# Single prompt
./llama.cpp/llama-cli -m windows_coder.gguf \
  -p "Write a function to read a Windows registry key:" \
  -n 256

# With system prompt
./llama.cpp/llama-cli -m windows_coder.gguf \
  --system-prompt "You are an expert Windows programmer." \
  -p "Write code to list all services" \
  -n 256
```

## Using with LM Studio

1. Copy the `.gguf` file to LM Studio's models directory
2. Open LM Studio and load the model
3. Use the chat interface

## Troubleshooting

### "llama-cpp-python not found"

Install it:
```bash
pip install llama-cpp-python
```

### "llama-quantize not found"

Build llama.cpp:
```bash
cd ~/llama.cpp && make
```

### Conversion fails with memory error

Try a smaller quantization or use a machine with more RAM:
```bash
python scripts/export_gguf.py --model ... --quantization Q4_K_S
```

### Model doesn't work in Ollama

Check the Modelfile syntax and ensure the GGUF path is correct:
```bash
ollama show mymodel --modelfile
```

## Example Workflow

Complete workflow from training to deployment:

```bash
# 1. Train your model
halo-forge raft train \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --model Qwen/Qwen2.5-Coder-1.5B \
  --verifier mingw \
  --cycles 6 \
  --output models/windows_raft_1.5b

# 2. Export to GGUF
python scripts/export_gguf.py \
  --model models/windows_raft_1.5b/final_model \
  --output windows_coder_1.5b.Q4_K_M.gguf \
  --quantization Q4_K_M

# 3. Create Ollama model
cat > Modelfile << 'EOF'
FROM ./windows_coder_1.5b.Q4_K_M.gguf
SYSTEM "You are an expert Windows systems programmer specializing in the Windows API."
PARAMETER temperature 0.7
EOF

ollama create windows-coder -f Modelfile

# 4. Test it
ollama run windows-coder "Write a function to get the current process ID"
```

## See Also

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF inference engine
- [Ollama](https://ollama.ai) - Easy model serving
- [LM Studio](https://lmstudio.ai) - Desktop LLM app
