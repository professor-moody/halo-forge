---
title: "Agentic Testing"
description: "Testing guide for tool calling module"
weight: 2
---

# Testing the Agentic Module

Step-by-step guide to validate the tool calling training pipeline.

---

## Prerequisites

```bash
# Enter the toolbox
toolbox enter halo-forge

# Verify module is available
python -c "from halo_forge.agentic import ToolCallingVerifier; print('OK')"
```

---

## Level 1: Unit Tests

Run the agentic unit tests:

```bash
python -m pytest tests/test_agentic.py -v
```

Expected: 32 tests passing

### Key Tests

| Test | Description |
|------|-------------|
| `test_verify_correct_call` | Correct function + args → reward 0.75 |
| `test_verify_wrong_function` | Valid JSON, wrong function → reward 0.25 |
| `test_verify_false_positive` | Called when shouldn't → reward -0.25 |
| `test_extract_tool_calls` | Parse `<tool_call>` from output |
| `test_filter_completions` | Top K% filtering logic |

---

## Level 2: CLI Validation

### List Datasets

```bash
halo-forge agentic datasets
```

Expected output:
```
Available Agentic / Tool Calling Datasets
============================================================
  xlam         [Tool Calling] - 60k verified, 3,673 APIs
  glaive       [Tool Calling] - 113k samples, irrelevance
  toolbench    [Tool Calling] - 188k samples, 16k APIs
```

### Dry Run Training

```bash
halo-forge agentic train --dry-run
```

Expected:
```
Agentic / Tool Calling RAFT Training
============================================================
Model: Qwen/Qwen2.5-7B-Instruct
Dataset: xlam
...
Configuration valid!
```

---

## Level 3: Verifier Testing

Test the ToolCallingVerifier directly:

```python
from halo_forge.agentic import ToolCallingVerifier

verifier = ToolCallingVerifier()

# Test correct call
output = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
expected = [{"name": "get_weather", "arguments": {"city": "Paris"}}]

result = verifier.verify(output, expected_calls=expected)
print(f"Success: {result.success}")  # True
print(f"Reward: {result.reward}")    # 0.75

# Test false positive
result = verifier.verify(output, expected_calls=[], is_irrelevant=True)
print(f"Reward: {result.reward}")    # -0.25
```

---

## Level 4: Data Loading

Test dataset loading (requires network):

```python
from halo_forge.agentic.data import XLAMLoader

loader = XLAMLoader()
samples = loader.load(limit=10)

print(f"Loaded: {len(samples)} samples")
for s in samples[:2]:
    print(f"  Query: {s.messages[0]['content'][:50]}...")
    print(f"  Tools: {len(s.tools)}")
```

---

## Level 5: Benchmark Run

Run a small benchmark (requires GPU):

```bash
halo-forge agentic benchmark \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset xlam \
  --limit 20 \
  --output results/agentic_benchmark.json
```

Expected metrics:
- `accuracy`: Overall correct rate
- `json_valid_rate`: Valid JSON output rate
- `function_accuracy`: Correct function selection rate

---

## Level 6: Training Run

Run a short training (requires GPU, ~2-4 hours):

```bash
halo-forge agentic train \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset xlam \
  --limit 500 \
  --cycles 2 \
  --output models/agentic_test
```

### Monitor with TensorBoard

```bash
tensorboard --logdir models/agentic_test/tensorboard
```

Metrics to watch:
- `success_rate` - Should increase over cycles
- `avg_reward` - Should trend upward
- `kept_samples` - High-quality samples per cycle

---

## Troubleshooting

### "No tool call found in output"

The model isn't generating `<tool_call>` tags. Check:
- Is the prompt in correct Hermes format?
- Is the model instruction-tuned?
- Try a different model (Qwen2.5 recommended)

### Low accuracy on irrelevance detection

The model calls tools when it shouldn't. Solutions:
- Include more irrelevance samples in training data (15%+)
- Use Glaive dataset (7,500 irrelevance examples)

### JSON syntax errors

Small models make JSON mistakes. Solutions:
- Use constrained decoding with GBNF grammar
- Use temperature 0.0 for production
- Consider larger model (7B+)

---

## Validation Checklist

| Test | Command | Expected |
|------|---------|----------|
| Unit tests | `pytest tests/test_agentic.py` | 32 passed |
| CLI datasets | `halo-forge agentic datasets` | Lists 4 datasets |
| Dry run | `halo-forge agentic train --dry-run` | Config valid |
| Verifier | Python test above | Rewards correct |
| Data load | Python test above | Samples load |
| Benchmark | `halo-forge agentic benchmark` | JSON output |
