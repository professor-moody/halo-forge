---
title: "Model Support"
description: "Supported models for halo-forge training"
weight: 6
---

halo forge supports various causal language models for code generation training.

## Recommended Models

### Qwen2.5-Coder Series (Default)

| Model | VRAM Usage | Notes |
|-------|------------|-------|
| `Qwen/Qwen2.5-Coder-0.5B` | ~2 GB | Fast iteration, testing |
| `Qwen/Qwen2.5-Coder-1.5B` | ~4 GB | Good balance |
| `Qwen/Qwen2.5-Coder-3B` | ~8 GB | Recommended for quality |
| `Qwen/Qwen2.5-Coder-7B` | ~14 GB | Best quality, slower |

**Qwen2.5-Coder is the default and best-tested model family.**

```bash
# 3B model (recommended starting point)
halo-forge raft train --model Qwen/Qwen2.5-Coder-3B --prompts data/prompts.jsonl

# 7B model (higher quality)
halo-forge raft train --model Qwen/Qwen2.5-Coder-7B --prompts data/prompts.jsonl
```

### Other Tested Models

| Family | Models | Notes |
|--------|--------|-------|
| DeepSeek-Coder | 1.3B, 6.7B, 33B | Strong code generation |
| CodeLlama | 7B, 13B, 34B | Meta's code models |
| StarCoder2 | 3B, 7B, 15B | BigCode, multi-language |

## Model Selection

### By Use Case

| Use Case | Recommended |
|----------|-------------|
| Quick testing | Qwen2.5-Coder-0.5B |
| Development | Qwen2.5-Coder-3B |
| Production | Qwen2.5-Coder-7B |
| Low VRAM (<8GB) | Qwen2.5-Coder-1.5B |

### By Available VRAM

| VRAM | Recommended Models |
|------|-------------------|
| 8 GB | 0.5B - 3B |
| 16 GB | Up to 7B |
| 24 GB+ | 7B with larger batches |
| 48 GB+ | 13B - 15B |
| 80 GB+ | 33B+ |

## Unified Memory (AMD APUs)

On AMD APUs like Strix Halo with unified memory:

| System RAM | Usable for GPU | Recommended |
|------------|----------------|-------------|
| 32 GB | ~16 GB | 7B (tight) |
| 64 GB | ~32 GB | 7B with headroom |
| 128 GB | ~80 GB | 13B-33B |

## LoRA Configuration

Default LoRA target modules:

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"       # MLP
]
```

These work for Qwen and Llama-style models. Adjust for other architectures.

## Adding New Models

1. Ensure it's a HuggingFace causal LM (`AutoModelForCausalLM`)
2. Check `trust_remote_code=True` support if needed
3. Verify LoRA target modules match the architecture
4. Test with a small prompt set:

```bash
halo-forge raft train \
    --model YOUR_MODEL \
    --prompts data/test_prompts.jsonl \
    --cycles 1 \
    --output models/test
```

## Known Issues

- **DeepSeek-Coder V2**: May require specific transformers version
- **CodeLlama-34B**: Needs multi-GPU or very large VRAM
- **Phi models**: Different architecture, may need config adjustments

## LiquidAI LFM Models

| Model | Status | Notes |
|-------|--------|-------|
| LFM2-1.2B, LFM2.5-1.2B-Base | Supported | Standard CausalLM |
| LFM2.5-VL-1.6B | Unsupported | Custom architecture |
| LFM2.5-Audio-1.5B | Unsupported | Non-standard processor |

For LFM text models, use standard loading. For VLM/Audio benchmarks, use Qwen2-VL or Whisper instead.
