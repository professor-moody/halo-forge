# Configuration Reference

Complete reference for all configuration options.

## SFT Configuration

```yaml
# configs/sft.yaml
model:
  name: Qwen/Qwen2.5-Coder-7B       # HuggingFace model name
  trust_remote_code: true           # For custom architectures
  
data:
  train_file: data/train.jsonl      # Training data path
  max_seq_length: 2048              # Maximum sequence length
  
training:
  num_train_epochs: 3               # Training epochs
  per_device_train_batch_size: 2    # Batch size per GPU
  gradient_accumulation_steps: 16   # Gradient accumulation
  learning_rate: 2e-5               # Learning rate
  warmup_ratio: 0.03                # Warmup proportion
  weight_decay: 0.01                # L2 regularization
  bf16: true                        # Use BF16 precision
  gradient_checkpointing: true      # Save memory
  dataloader_num_workers: 0         # REQUIRED for Strix Halo
  dataloader_pin_memory: false      # REQUIRED for Strix Halo

lora:
  r: 64                             # LoRA rank
  alpha: 128                        # LoRA alpha (scaling)
  dropout: 0.05                     # LoRA dropout
  target_modules:                   # Modules to adapt
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

output:
  dir: models/sft                   # Output directory
  save_steps: 500                   # Checkpoint frequency
  logging_steps: 10                 # Log frequency
```

## RAFT Configuration

```yaml
# configs/raft.yaml
sft_checkpoint: models/sft/final_model  # Starting checkpoint
output_dir: models/raft                  # Output directory
prompts: data/prompts.jsonl              # Training prompts

raft:
  num_cycles: 5                     # Number of RAFT cycles
  samples_per_prompt: 8             # Samples to generate
  reward_threshold: 0.5             # Minimum reward to keep
  keep_top_percent: 0.5             # Top % of samples to keep

generation:
  max_new_tokens: 1024              # Max tokens to generate
  temperature: 0.7                  # Sampling temperature
  top_p: 0.95                       # Nucleus sampling
  batch_size: 4                     # Generation batch size

training:
  epochs: 1                         # Epochs per cycle
  batch_size: 2                     # Training batch size
  gradient_accumulation_steps: 16   # Gradient accumulation
  learning_rate: 5e-5               # Learning rate

verifier:
  type: gcc                         # Verifier type
  run_after_compile: false          # Execute after compile
  timeout: 30                       # Verification timeout
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HSA_ENABLE_SDMA` | `0` | Required for Strix Halo |
| `PYTORCH_HIP_ALLOC_CONF` | - | Memory allocator config |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache |
| `ANTHROPIC_API_KEY` | - | For Claude data generation |
| `OPENAI_API_KEY` | - | For OpenAI data generation |

## CLI Options

Most config options can be overridden via CLI:

```bash
halo-forge sft train \
  --config configs/sft.yaml \
  --epochs 5 \                    # Override num_train_epochs
  --learning-rate 1e-5 \          # Override learning_rate
  --output models/sft_v2          # Override output.dir
```
