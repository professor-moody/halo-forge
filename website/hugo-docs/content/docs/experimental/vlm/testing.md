---
title: "Testing Guide"
weight: 50
---

# VLM Module Testing Guide

Step-by-step guide to testing the halo forge Vision-Language Model training module.

## Prerequisites

### Required Dependencies

```bash
pip install torch transformers accelerate pillow
```

### Verification Dependencies

| Dependency | Required For | Install |
|------------|--------------|---------|
| `ultralytics` | Object detection (PerceptionChecker) | `pip install ultralytics` |
| `easyocr` | OCR verification | `pip install easyocr` |
| `sentence-transformers` | Semantic similarity | `pip install sentence-transformers` |

### Hardware Requirements

- **Minimum**: 32 GB RAM, 16 GB VRAM (for 2B VLM)
- **Recommended**: 64 GB RAM, 48 GB VRAM (for 7B VLM)
- **ROCm Support**: AMD Radeon RX 7900 series or higher

---

## Unit Tests

Run the full VLM test suite:

```bash
# From project root
cd /path/to/halo-forge

# Run all VLM tests
pytest tests/test_vlm.py tests/test_vlm_data.py -v

# Run specific test files
pytest tests/test_vlm.py -v        # Verifier tests
pytest tests/test_vlm_data.py -v   # Data loader tests

# Run specific test classes
pytest tests/test_vlm.py::TestPerceptionChecker -v
pytest tests/test_vlm.py::TestVisionVerifier -v
pytest tests/test_vlm_data.py::TestTextVQALoader -v
```

### Expected Test Coverage

**test_vlm.py:**

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestDetection` | 3 | Detection dataclass |
| `TestPerceptionChecker` | 4 | Object detection, OCR |
| `TestPerceptionResult` | 2 | Perception results |
| `TestReasoningChecker` | 3 | Chain-of-thought validation |
| `TestReasoningStep` | 2 | Reasoning step dataclass |
| `TestOutputChecker` | 3 | Answer verification |
| `TestVisionVerifier` | 4 | Multi-stage verification |
| `TestIntegration` | 2 | End-to-end workflows |

**test_vlm_data.py:**

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestVLMSample` | 2 | Sample dataclass |
| `TestVLMDataset` | 3 | Abstract dataset |
| `TestTextVQALoader` | 3 | TextVQA dataset |
| `TestDocVQALoader` | 3 | DocVQA dataset |
| `TestChartQALoader` | 3 | ChartQA dataset |
| `TestRealWorldQALoader` | 3 | RealWorldQA dataset |
| `TestVLMPreprocessor` | 3 | Image preprocessing |
| `TestIntegration` | 2 | End-to-end data loading |

---

## Dry Run Testing

Test commands without running actual training:

```bash
# Test CLI parsing and config validation
halo-forge vlm train \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --dataset textvqa \
  --cycles 3 \
  --output /tmp/vlm-test \
  --dry-run

# Expected output:
# Dry run mode enabled. Configuration validated successfully.
# Model: Qwen/Qwen2-VL-2B-Instruct
# Dataset: textvqa
# Cycles: 3
# Output: /tmp/vlm-test
```

---

## Manual Testing Steps

### 1. Test Dataset Loaders

```bash
# List available VLM datasets
halo-forge vlm datasets

# Expected output:
# Available VLM Datasets:
#   - textvqa (45K samples)
#   - docvqa (50K samples)
#   - chartqa (28K samples)
#   - realworldqa (700 samples)
#   - mathvista (6K+ samples)
```

Test loading individual datasets in Python:

```python
from halo_forge.vlm.data.loaders import (
    TextVQALoader, DocVQALoader, ChartQALoader, RealWorldQALoader
)

# Test TextVQA
loader = TextVQALoader(split="train", limit=10)
samples = loader.load()
print(f"Loaded {len(samples)} TextVQA samples")
print(f"First sample: {samples[0]}")

# Test DocVQA
loader = DocVQALoader(split="train", limit=10)
samples = loader.load()
print(f"Loaded {len(samples)} DocVQA samples")
```

### 2. Test Perception Checker

```python
from halo_forge.vlm.verifiers.perception import PerceptionChecker
from PIL import Image

# Initialize checker (requires ultralytics, easyocr)
checker = PerceptionChecker()

# Test with a sample image
image = Image.open("test_image.jpg")
response = "I can see a red car and a stop sign."

result = checker.check(image, response)
print(f"Perception score: {result.score:.2f}")
print(f"Detected objects: {result.detected_objects}")
print(f"OCR text: {result.ocr_text}")
```

### 3. Test Reasoning Checker

```python
from halo_forge.vlm.verifiers.reasoning import ReasoningChecker

checker = ReasoningChecker()

response = """
Let me analyze this step by step:
1. First, I see a document with a table
2. The table has columns for date and amount
3. Looking at the total row, the amount is $1,234
Therefore, the total is $1,234.
"""

result = checker.check(response)
print(f"Reasoning score: {result.score:.2f}")
print(f"Steps found: {len(result.steps)}")
print(f"Is coherent: {result.is_coherent}")
```

### 4. Test Output Checker

```python
from halo_forge.vlm.verifiers.output import OutputChecker

checker = OutputChecker()

# Exact match
result = checker.check(
    prediction="$1,234",
    ground_truth="$1,234"
)
print(f"Exact match score: {result.score:.2f}")

# Fuzzy match
result = checker.check(
    prediction="The answer is 1234 dollars",
    ground_truth="$1,234"
)
print(f"Fuzzy match score: {result.score:.2f}")
```

### 5. Test Full VisionVerifier

```python
from halo_forge.vlm.verifiers.base import VisionVerifier, VisionVerifyConfig
from PIL import Image

config = VisionVerifyConfig(
    perception_weight=0.3,
    reasoning_weight=0.4,
    output_weight=0.3
)
verifier = VisionVerifier(config)

image = Image.open("test_chart.png")
response = """
Looking at the chart, I can see three bars representing sales data.
The blue bar (Q1) is at 100, red bar (Q2) at 150, green bar (Q3) at 200.
The total sales across all quarters is 450.
"""
ground_truth = "450"

result = verifier.verify(image, response, ground_truth)
print(f"Overall score: {result.overall_score:.2f}")
print(f"Perception: {result.perception_score:.2f}")
print(f"Reasoning: {result.reasoning_score:.2f}")
print(f"Output: {result.output_score:.2f}")
```

### 6. Test VLM Benchmark

```bash
# Benchmark on a small subset
halo-forge vlm benchmark \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --dataset textvqa \
  --limit 20

# Expected output:
# Benchmarking Qwen/Qwen2-VL-2B-Instruct on textvqa...
# Samples: 20
# Average perception score: X.XX
# Average reasoning score: X.XX
# Average output score: X.XX
# Overall accuracy: XX.X%
```

---

## Validation Checklist

After running tests, verify:

- [ ] All unit tests pass (`pytest tests/test_vlm.py tests/test_vlm_data.py`)
- [ ] Dry run mode works without errors
- [ ] All dataset loaders work (`halo-forge vlm datasets`)
- [ ] PerceptionChecker detects objects correctly
- [ ] ReasoningChecker validates chain-of-thought
- [ ] OutputChecker compares answers correctly
- [ ] VisionVerifier combines all stages
- [ ] VLM benchmark runs on small subset

---

## Troubleshooting

### "ultralytics not installed"

This is a warning. Install if you need object detection:

```bash
pip install ultralytics
```

### "easyocr not installed"

Required for OCR verification:

```bash
pip install easyocr
```

Note: First run downloads language models (~100MB).

### "sentence-transformers not installed"

Required for semantic similarity:

```bash
pip install sentence-transformers
```

### Dataset Loading Errors

Most VLM datasets require HuggingFace authentication:

```bash
huggingface-cli login
```

Or set the token:

```bash
export HF_TOKEN=your_token_here
```

### Image Loading Errors

Ensure images are valid and accessible:

```python
from PIL import Image

# Test loading
try:
    img = Image.open("test.jpg")
    img.verify()
    print("Image is valid")
except Exception as e:
    print(f"Image error: {e}")
```

### Out of Memory (OOM)

VLM models are memory-intensive:

- Use 2B models for testing (Qwen2-VL-2B)
- Reduce batch size with `--batch-size 1`
- Use `--limit` to test on fewer samples
- Enable gradient checkpointing in config

### ROCm-specific Issues

For AMD GPUs with attention issues:

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
halo-forge vlm train --experimental-attention ...
```

---

## Next Steps

- [VLM Training Guide](./) - Full training documentation
- [Perception Verification](./verifiers/) - How perception works
- [Dataset Loaders](./datasets/) - Preparing VLM datasets
- [Inference Testing](../inference/testing.md) - Test inference module
