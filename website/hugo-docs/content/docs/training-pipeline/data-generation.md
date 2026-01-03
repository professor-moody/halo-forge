---
title: "Data Generation"
description: "Preparing training data for SFT and RAFT"
weight: 2
---

## Data Format

halo-forge expects JSONL files with this structure:

```json
{"prompt": "Write a function to calculate factorial", "completion": "int factorial(int n) {...}"}
{"prompt": "Implement binary search", "completion": "int binary_search(int arr[], int n, int target) {...}"}
```

For RAFT, you only need prompts:

```json
{"prompt": "Write a function to calculate factorial"}
{"prompt": "Implement binary search"}
```

## Public Datasets

### CodeForces

```bash
halo-forge data prepare \
  --dataset codeforces_cpp \
  --output data/codeforces.jsonl \
  --limit 1000
```

Best for: C++ code generation, algorithmic problems

### MBPP

```bash
halo-forge data prepare \
  --dataset mbpp \
  --output data/mbpp.jsonl
```

Best for: Python functions, simpler problems

### HumanEval

```bash
halo-forge data prepare \
  --dataset humaneval \
  --output data/humaneval.jsonl
```

Best for: Evaluation benchmark, Python

## LLM Generation

Generate domain-specific training data using LLMs.

### With Ollama (Local)

```bash
halo-forge data generate \
  --prompts prompts.txt \
  --backend ollama \
  --model deepseek-coder:6.7b \
  --output generated.jsonl \
  --samples 3
```

### With Claude API

```bash
export ANTHROPIC_API_KEY=your_key

halo-forge data generate \
  --prompts prompts.txt \
  --backend anthropic \
  --model claude-3-sonnet \
  --output generated.jsonl
```

### Prompt File Format

```text
Write a function to reverse a linked list
Implement a thread-safe queue in C++
Create a binary search tree with insert and delete
```

## Data Quality Tips

1. **Diversity matters**: Include various problem types
2. **Verify examples**: Run through your verifier before training
3. **Balance difficulty**: Mix easy and hard problems
4. **Clean formatting**: Consistent code style helps

## Splitting Data

```bash
# Create train/test split
halo-forge data split \
  --input data/all.jsonl \
  --train data/train.jsonl \
  --test data/test.jsonl \
  --ratio 0.9
```
