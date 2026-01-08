---
title: "Agentic / Tool Calling"
description: "Train models for reliable function and tool calling"
weight: 5
---

# Agentic / Tool Calling Training

Train small language models (1B-13B parameters) for reliable function/tool calling using RLVR.

---

## Overview

Agentic AI capabilities require models that can:

1. **Understand tool schemas** - Parse JSON function definitions
2. **Generate valid calls** - Produce syntactically correct JSON
3. **Select correct functions** - Choose the right tool for the task
4. **Extract correct arguments** - Map user intent to parameters
5. **Know when NOT to call** - Detect irrelevant queries

halo forge trains these capabilities through verifier-guided RAFT.

---

## Target Benchmarks (BFCL)

The Berkeley Function Calling Leaderboard (BFCL) is the standard benchmark.

| Metric | Minimum | Target |
|--------|---------|--------|
| Single-Turn | 75% | 85% |
| Parallel Calls | 55% | 70% |
| Multi-Turn | 60% | 75% |
| Irrelevance Detection | 70% | 85% |

**Reference:** xLAM-7B-fc-r achieves 88.24% on BFCL V4.

---

## Quick Start

### List Available Datasets

```bash
halo-forge agentic datasets
```

### Run Benchmark

```bash
halo-forge agentic benchmark \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset xlam \
  --limit 100
```

### Train with RAFT

```bash
halo-forge agentic train \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset xlam \
  --cycles 5 \
  --output models/agentic_raft
```

---

## Data Format: Hermes

halo forge uses the **Hermes format**, the standard for Qwen2.5, NousHermes, and most open models:

```
<|im_start|>system
You are a function calling AI model.
<tools>
[{"type":"function","function":{"name":"get_weather","parameters":{...}}}]
</tools>
<|im_end|>
<|im_start|>user
What's the weather in Paris?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris"}}
</tool_call><|im_end|>
```

---

## ToolCallingVerifier

The verifier provides graduated rewards based on correctness:

| Outcome | Reward | Description |
|---------|--------|-------------|
| Correct function + args | 1.0 | Perfect result |
| Correct function, wrong args | 0.5 | Partial credit |
| Valid JSON, wrong function | 0.25 | Format correct |
| No tool call when expected | 0.0 | Failure |
| Called when shouldn't | -0.25 | False positive penalty |

---

## Training Pipeline

### 1. SFT Phase

Fine-tune on tool calling examples with proper loss masking:
- Train only on assistant + `<tool_call>` tokens
- Mask system, user, and `<tool_response>` content

### 2. RAFT Phase

Iterative improvement with verification:
- Generate multiple completions per prompt
- Verify with ToolCallingVerifier
- Filter to top 25% (more selective than other domains)
- Train on high-reward samples

### 3. Constrained Decoding

For production, use grammar constraints to ensure valid JSON:

```bash
# vLLM
vllm serve model --enable-auto-tool-choice --tool-call-parser hermes

# llama.cpp
./llama-server -m model.gguf --grammar-file tool_call.gbnf
```

---

## Inference Tips

**Use temperature 0.0 for production** - Up to 10% accuracy improvement over T=0.7.

```python
INFERENCE_CONFIG = {
    "temperature": 0.0,
    "max_tokens": 512,
    "stop": ["<|im_end|>", "</tool_call>"],
}
```

---

## Module Structure

```
halo_forge/agentic/
├── __init__.py
├── trainer.py              # AgenticRAFTTrainer
├── verifiers/
│   ├── __init__.py
│   └── base.py             # ToolCallingVerifier
└── data/
    ├── __init__.py
    ├── loaders.py          # xLAM, Glaive loaders
    └── formatters.py       # Hermes format conversion
```

---

## References

- [xLAM Paper](https://arxiv.org/pdf/2409.03215) - Large Action Models
- [BFCL Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [vLLM Tool Calling](https://docs.vllm.ai/en/latest/features/tool_calling.html)
