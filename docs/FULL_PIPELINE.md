# Full Pipeline Guide

Complete guide to training a code generation model with halo-forge.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    halo-forge Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. DATA GENERATION                                            │
│   ┌─────────────────┐    ┌─────────────────┐                   │
│   │ Public Datasets │ or │ LLM Generation  │                   │
│   │ (CodeForces,    │    │ (DeepSeek,      │                   │
│   │  MBPP, etc.)    │    │  Claude, etc.)  │                   │
│   └────────┬────────┘    └────────┬────────┘                   │
│            │                      │                             │
│            └──────────┬───────────┘                             │
│                       ▼                                         │
│   2. SFT TRAINING                                               │
    │   ┌─────────────────────────────────────────┐                  │
    │   │ LoRA Fine-tuning (BF16)                  │                  │
    │   │ - BF16 precision (optimal)               │                  │
    │   │ - Gradient checkpointing                 │                  │
    │   │ - Early stopping                         │                  │
    │   └────────────────────┬────────────────────┘                  │
│                        ▼                                        │
│   3. RAFT TRAINING (RLVR)                                       │
│   ┌─────────────────────────────────────────┐                  │
│   │ Iterative Verification Loop              │                  │
│   │   ┌─────────┐  ┌─────────┐  ┌─────────┐│                  │
│   │   │Generate │→ │ Verify  │→ │ Filter  ││                  │
│   │   └─────────┘  └─────────┘  └────┬────┘│                  │
│   │        ↑                         │     │                  │
│   │        └─────────────────────────┘     │                  │
│   │                 Train on filtered      │                  │
│   └────────────────────┬────────────────────┘                  │
│                        ▼                                        │
│   4. BENCHMARK                                                  │
│   ┌─────────────────────────────────────────┐                  │
│   │ pass@k Evaluation                        │                  │
│   │ - Generate multiple samples              │                  │
│   │ - Verify with same verifier              │                  │
│   │ - Compute pass@1, pass@5, pass@10        │                  │
│   └─────────────────────────────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Data Generation

### Public Datasets

Best for getting started. No API keys needed.

```bash
# List available datasets
halo-forge data prepare --list

# Download CodeForces C++ (recommended)
halo-forge data prepare \
  --dataset codeforces_cpp \
  --output data/codeforces.jsonl \
  --template qwen

# Download multiple and combine
halo-forge data prepare --dataset mbpp --output data/mbpp.jsonl
cat data/codeforces.jsonl data/mbpp.jsonl > data/train.jsonl
```

### LLM Generation

Better for domain-specific training. Requires API key.

```bash
# List available topics
halo-forge data generate --list

# Generate with DeepSeek (cheap, good quality)
export DEEPSEEK_API_KEY=your_key

halo-forge data generate \
  --topic rust_async \
  --backend deepseek \
  --output data/rust_generated.jsonl

# Generate with local Ollama (free)
halo-forge data generate \
  --topic python_testing \
  --backend ollama \
  --model codellama:13b \
  --output data/python_generated.jsonl
```

### Custom Generation

For specific domains, create your own spec:

```python
from halo_forge.data.llm_generate import TopicSpec, TrainingDataGenerator, get_backend

my_spec = TopicSpec(
    name="my_domain",
    description="Your domain description",
    categories=["cat1", "cat2", "cat3"],
    complexity_levels=["easy", "medium", "hard"],
    examples_per_category=25,
    system_prompt="You are an expert in ..."
)

backend = get_backend("deepseek")
generator = TrainingDataGenerator(backend, my_spec)
generator.generate_all("data/my_domain.jsonl")
```

## Step 2: SFT Training

### Basic Training

```bash
halo-forge sft train \
  --data data/train.jsonl \
  --output models/sft \
  --epochs 3
```

### With Configuration File

Create `configs/sft.yaml`:

```yaml
model:
  name: Qwen/Qwen2.5-Coder-7B
  trust_remote_code: true
  attn_implementation: eager

data:
  train_file: data/train.jsonl
  validation_split: 0.05
  max_seq_length: 2048

lora:
  r: 16
  alpha: 32
  dropout: 0.05

# Note: BF16 is optimal for Strix Halo (4-bit is 2x slower)

training:
  output_dir: models/sft
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 2e-4
  gradient_checkpointing: true
  bf16: true
```

Then run:

```bash
halo-forge sft train --config configs/sft.yaml
```

### Resume Training

```bash
halo-forge sft train \
  --config configs/sft.yaml \
  --resume models/sft/checkpoint-500
```

## Step 3: RAFT Training

RAFT improves the model by training only on verified outputs.

### Setup Verifier

Choose based on your target language:

| Language | Verifier | Notes |
|----------|----------|-------|
| C/C++ (Linux) | `gcc` | Fast, local |
| C/C++ (Windows) | `mingw` | Cross-compile |
| C/C++ (MSVC) | `msvc` | Remote Windows |
| Python | `pytest` | With tests |
| Any | `custom` | Your own |

### Prepare Prompts

Create a JSONL file with training prompts:

```bash
cat > data/prompts.jsonl << 'EOF'
{"prompt": "Write a function to calculate factorial"}
{"prompt": "Implement binary search in C++"}
{"prompt": "Write a thread-safe queue class"}
EOF
```

### Run RAFT

```bash
# With GCC verifier
halo-forge raft train \
  --checkpoint models/sft/final_model \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --cycles 5 \
  --output models/raft

# With remote MSVC
halo-forge raft train \
  --checkpoint models/sft/final_model \
  --prompts data/prompts.jsonl \
  --verifier msvc \
  --host 192.168.1.100 \
  --user developer \
  --ssh-key ~/.ssh/win \
  --cycles 5
```

### RAFT Configuration

Create `configs/raft.yaml`:

```yaml
sft_checkpoint: models/sft/final_model
output_dir: models/raft
prompts: data/prompts.jsonl

raft:
  num_cycles: 5
  samples_per_prompt: 8
  reward_threshold: 0.5
  keep_top_percent: 0.5

generation:
  max_new_tokens: 1024
  temperature: 0.7
  batch_size: 4

training:
  epochs: 1
  batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 5e-5

verifier:
  type: gcc
```

## Step 4: Benchmarking

### Run Benchmark

```bash
halo-forge benchmark run \
  --model models/raft/cycle_5_final \
  --prompts data/test_prompts.jsonl \
  --verifier gcc \
  --samples 20 \
  --k 1,5,10,20 \
  --output results/benchmark.json
```

### Compare Models

```bash
# Benchmark SFT
halo-forge benchmark run \
  --model models/sft/final_model \
  --prompts data/test.jsonl \
  --output results/sft_benchmark.json

# Benchmark RAFT cycle 3
halo-forge benchmark run \
  --model models/raft/cycle_3_final \
  --prompts data/test.jsonl \
  --output results/raft3_benchmark.json

# Benchmark RAFT cycle 5
halo-forge benchmark run \
  --model models/raft/cycle_5_final \
  --prompts data/test.jsonl \
  --output results/raft5_benchmark.json

# Compare
python -c "
import json
for f in ['sft', 'raft3', 'raft5']:
    with open(f'results/{f}_benchmark.json') as fp:
        data = json.load(fp)
        print(f'{f}: pass@1={data[\"pass_at_k\"][\"1\"]:.1%}')
"
```

## Complete Example

Full pipeline from scratch:

```bash
# Setup
toolbox enter halo-forge
cd ~/training

# 1. Get data
halo-forge data prepare --dataset codeforces_cpp --output data/train.jsonl

# 2. Split for test
head -4000 data/train.jsonl > data/train_split.jsonl
tail -500 data/train.jsonl > data/test_split.jsonl

# Extract prompts for RAFT
cat data/test_split.jsonl | python -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    # Extract problem from text
    text = d.get('text', '')
    if 'user' in text.lower():
        start = text.find('user') + 5
        end = text.find('<|im_end|>', start)
        prompt = text[start:end].strip()
        print(json.dumps({'prompt': prompt}))
" > data/prompts.jsonl

# 3. SFT
halo-forge sft train \
  --data data/train_split.jsonl \
  --output models/sft \
  --epochs 3

# 4. RAFT
halo-forge raft train \
  --checkpoint models/sft/final_model \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --cycles 5 \
  --output models/raft

# 5. Benchmark
halo-forge benchmark run \
  --model models/raft/cycle_5_final \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --samples 10 \
  --output results/final_benchmark.json

echo "Training complete!"
cat results/final_benchmark.json | python -c "
import json, sys
data = json.load(sys.stdin)
print(f'Pass rate: {data[\"pass_rate\"]:.1%}')
for k, v in data['pass_at_k'].items():
    print(f'pass@{k}: {v:.1%}')
"
```

## Monitoring

### TensorBoard

```bash
tensorboard --logdir models/sft/logs --port 6006

# Or for RAFT
tensorboard --logdir models/raft/cycle_1/logs --port 6006
```

### GPU Usage

```bash
watch -n 1 rocm-smi
```

### Training Progress

RAFT saves statistics after each cycle:

```bash
cat models/raft/raft_statistics.json | python -m json.tool
```

