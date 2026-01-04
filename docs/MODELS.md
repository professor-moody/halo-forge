# Model Support Guide

halo-forge supports various causal language models for code generation training.

## Officially Tested Models

### Qwen2.5-Coder Series (Recommended)

| Model | VRAM Usage | Notes |
|-------|------------|-------|
| `Qwen/Qwen2.5-Coder-0.5B` | ~2 GB | Fast iteration, testing |
| `Qwen/Qwen2.5-Coder-1.5B` | ~4 GB | Good balance |
| `Qwen/Qwen2.5-Coder-3B` | ~8 GB | Recommended for quality |
| `Qwen/Qwen2.5-Coder-7B` | ~14 GB | Best quality, slower |

**Qwen2.5-Coder is the default and best-tested model family.** It has excellent code generation capabilities out of the box and responds well to RAFT training.

```bash
# 3B model (recommended starting point)
halo-forge raft train --model Qwen/Qwen2.5-Coder-3B --prompts data/prompts.jsonl

# 7B model (higher quality, needs more VRAM)
halo-forge raft train --model Qwen/Qwen2.5-Coder-7B --prompts data/prompts.jsonl
```

### DeepSeek-Coder

| Model | VRAM Usage | Notes |
|-------|------------|-------|
| `deepseek-ai/deepseek-coder-1.3b-base` | ~4 GB | Fast, good for testing |
| `deepseek-ai/deepseek-coder-6.7b-base` | ~14 GB | Strong code generation |
| `deepseek-ai/deepseek-coder-33b-base` | ~70 GB | Requires multi-GPU |

DeepSeek-Coder models are trained on a large code corpus and perform well on code generation tasks.

```bash
halo-forge raft train --model deepseek-ai/deepseek-coder-6.7b-base --prompts data/prompts.jsonl
```

**Note:** DeepSeek models may require `trust_remote_code=True` which is enabled by default.

### CodeLlama

| Model | VRAM Usage | Notes |
|-------|------------|-------|
| `codellama/CodeLlama-7b-hf` | ~14 GB | Meta's code model |
| `codellama/CodeLlama-13b-hf` | ~28 GB | Stronger reasoning |
| `codellama/CodeLlama-34b-hf` | ~70 GB | Best quality |

CodeLlama models are Meta's specialized code models based on Llama 2.

```bash
halo-forge raft train --model codellama/CodeLlama-7b-hf --prompts data/prompts.jsonl
```

### StarCoder

| Model | VRAM Usage | Notes |
|-------|------------|-------|
| `bigcode/starcoder2-3b` | ~8 GB | BigCode model |
| `bigcode/starcoder2-7b` | ~14 GB | Good general code |
| `bigcode/starcoder2-15b` | ~32 GB | Strong quality |

StarCoder models are trained on The Stack and support many programming languages.

```bash
halo-forge raft train --model bigcode/starcoder2-7b --prompts data/prompts.jsonl
```

## Model Selection Guidelines

### By Use Case

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| Quick testing | Qwen2.5-Coder-0.5B | Fast iteration |
| Development | Qwen2.5-Coder-3B | Good balance |
| Production | Qwen2.5-Coder-7B | Best quality |
| Low VRAM (<8GB) | Qwen2.5-Coder-1.5B | Fits in memory |
| Multi-GPU | DeepSeek-33B | Scales well |

### By Available VRAM

| VRAM | Recommended |
|------|-------------|
| 8 GB | 0.5B - 3B models |
| 16 GB | 7B models |
| 24 GB+ | 7B with larger batches, or 13B |
| 48 GB+ | 13B - 15B models |
| 80 GB+ | 33B+ models |

## Unified Memory (AMD APUs)

On AMD APUs like Strix Halo with unified memory, the GPU shares system RAM. This allows larger models but at a performance cost:

| System RAM | Usable for GPU | Recommended Model |
|------------|----------------|-------------------|
| 32 GB | ~16 GB | 7B models (tight) |
| 64 GB | ~32 GB | 7B with headroom |
| 128 GB | ~80 GB | 13B-33B models |

## LoRA/QLoRA Support

All models work with LoRA for efficient fine-tuning. Default LoRA config targets:

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"       # MLP (Qwen/Llama)
]
```

For models with different architectures (e.g., StarCoder), the target modules may need adjustment.

## Tokenizer Notes

### Chat Templates

halo-forge uses the model's chat template for formatting. Most modern models include this:

```python
messages = [
    {"role": "system", "content": "You are an expert programmer."},
    {"role": "user", "content": prompt}
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

### Padding

- Left padding is used for generation (allows batch generation)
- `pad_token` is set to `eos_token` if not defined

## Adding New Models

To use a model not listed here:

1. Ensure it's a HuggingFace causal LM (`AutoModelForCausalLM`)
2. Check it supports `trust_remote_code=True` if needed
3. Verify LoRA target modules match the architecture
4. Test with a small prompt set first:

```bash
halo-forge raft train \
    --model YOUR_MODEL \
    --prompts data/test_prompts.jsonl \
    --cycles 1 \
    --output models/test
```

## Performance Tips

1. **Start small**: Use 0.5B or 1.5B for rapid iteration
2. **Match batch size to VRAM**: Reduce `--batch-size` if OOM
3. **Use BF16**: All models load in BF16 by default for Strix Halo compatibility
4. **Monitor temperature**: Higher diversity early, lower later (see training docs)

## Known Issues

- **DeepSeek-Coder V2**: May require specific transformers version
- **CodeLlama-34B**: Needs multi-GPU or very large VRAM
- **Phi models**: Different architecture, may need config adjustments

