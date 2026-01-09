---
title: "Code Datasets"
weight: 15
---

# Code Datasets

Guide to obtaining and using datasets for code generation training.

## Available Datasets

### Built-in Datasets

List available datasets with the CLI:

```bash
halo-forge data prepare --list
```

| Dataset | Source | Language | Size | Use Case |
|---------|--------|----------|------|----------|
| `codeforces_cpp` | open-r1/codeforces-cots | C++ | ~4000 | Competitive programming |
| `codeforces_python` | open-r1/codeforces-cots | Python | ~1000 | Competitive programming |
| `codeforces_rust` | open-r1/codeforces-cots | Rust | ~500 | Systems programming |
| `mbpp` | google-research-datasets/mbpp | Python | ~1000 | Basic Python |
| `humaneval` | openai/openai_humaneval | Python | 164 | Function synthesis |
| `humaneval_plus` | evalplus/humanevalplus | Python | 164 | Extended HumanEval |
| `livecodebench` | livecodebench/code_generation_lite | Multiple | Variable | Real-world coding |

---

## Downloading Datasets

### Single Dataset

```bash
# Download and format CodeForces C++
halo-forge data prepare \
  --dataset codeforces_cpp \
  --output data/codeforces_cpp.jsonl

# Download MBPP
halo-forge data prepare \
  --dataset mbpp \
  --output data/mbpp.jsonl
```

### Multiple Datasets

```bash
# Download several datasets
for ds in codeforces_cpp mbpp humaneval_plus; do
  halo-forge data prepare --dataset $ds --output data/${ds}.jsonl
done

# Combine for training
cat data/*.jsonl > data/combined_train.jsonl
```

---

## Sample Data Locations

The repository includes sample data for quick testing:

| Path | Description |
|------|-------------|
| `data/rlvr/humaneval_prompts.jsonl` | HumanEval prompts for RLVR |
| `data/rlvr/mbpp_train_prompts.jsonl` | MBPP prompts for RLVR |
| `data/samples/codeforces_cpp_500.jsonl` | 500 CodeForces C++ samples |
| `datasets/windows_curriculum/` | Windows systems programming |

### Using Sample Data

```bash
# Quick RAFT training with HumanEval
halo-forge raft train \
  --prompts data/rlvr/humaneval_prompts.jsonl \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --verifier humaneval \
  --cycles 3

# SFT with CodeForces samples
halo-forge sft train \
  --data data/samples/codeforces_cpp_500_sft.jsonl \
  --model Qwen/Qwen2.5-Coder-0.5B
```

---

## Windows Curriculum

A specialized dataset for Windows systems programming:

```bash
# Location
ls datasets/windows_curriculum/

# Files:
# - windows_systems_full_rlvr.jsonl (361 problems, RLVR format)
# - windows_systems_full_sft.jsonl (SFT format)
# - curriculum_order_full.json (tier metadata)
```

### Tier Structure

| Tier | Level | Problems | Topics |
|------|-------|----------|--------|
| 1 | Foundations | 84 | Process info, file I/O, registry |
| 2 | Core APIs | 128 | Process enum, memory mapping, pipes |
| 3 | Intermediate | 72 | PE parsing, tokens, services |
| 4 | Advanced | 77 | ETW, native API, syscalls |

### Training with Windows Curriculum

```bash
# RAFT training (requires Windows build server)
halo-forge raft train \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --model Qwen/Qwen2.5-Coder-1.5B \
  --verifier msvc \
  --cycles 6 \
  --output models/windows_coder
```

---

## Data Formats

### RLVR Format (for RAFT training)

```json
{
  "prompt": "Write a function to sort a list...",
  "test_cases": [
    {"input": "[3,1,2]", "expected": "[1,2,3]"}
  ]
}
```

### SFT Format (for supervised training)

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert programmer."},
    {"role": "user", "content": "Write a function to sort a list..."},
    {"role": "assistant", "content": "```python\ndef sort_list(lst):\n    return sorted(lst)\n```"}
  ]
}
```

---

## Creating Custom Datasets

### From HuggingFace

```python
from datasets import load_dataset
import json

# Load any HuggingFace dataset
ds = load_dataset("your_org/your_dataset", split="train")

# Convert to RLVR format
with open("custom_rlvr.jsonl", "w") as f:
    for item in ds:
        record = {
            "prompt": item["problem"],
            "test_cases": item.get("test_cases", [])
        }
        f.write(json.dumps(record) + "\n")
```

### Using LLM Generation

```bash
# Generate data with DeepSeek
halo-forge data generate \
  --topic rust_async \
  --backend deepseek \
  --num-examples 100 \
  --output data/rust_async.jsonl
```

---

## HuggingFace Sources

### Recommended Datasets

| Dataset | HuggingFace Path | Description |
|---------|------------------|-------------|
| CodeForces | `open-r1/codeforces-cots` | Competitive programming with CoT |
| MBPP | `google-research-datasets/mbpp` | Basic Python problems |
| HumanEval | `openai/openai_humaneval` | Classic benchmark |
| HumanEval+ | `evalplus/humanevalplus` | Extended tests |
| LiveCodeBench | `livecodebench/code_generation_lite` | Modern benchmarks |
| Apps | `codeparrot/apps` | Large coding dataset |
| CodeContests | `deepmind/code_contests` | Competition problems |

### Downloading Directly

```python
from datasets import load_dataset

# Load CodeForces
cf = load_dataset("open-r1/codeforces-cots", split="train")

# Filter by language
cpp_problems = cf.filter(lambda x: x["language"] == "cpp")

# Save
cpp_problems.to_json("codeforces_cpp.jsonl")
```

---

## Dataset Validation

Validate dataset format before training:

```bash
# Validate format
halo-forge data validate --file data/custom.jsonl

# Expected output:
# Validated 1000 samples
# Format: RLVR
# Fields: prompt, test_cases
# Errors: 0
```

---

## Next Steps

- [Training Pipeline](../how-to-train/) - Use datasets for training
- [RAFT Training](../raft/) - RLVR training workflow
- [Windows Setup](../../reference/windows-setup/) - For Windows curriculum
