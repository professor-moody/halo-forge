---
title: "Testing Guide"
weight: 2
---

# Reasoning Module Testing

Validate your reasoning training setup.

## Prerequisites

Ensure you're in the halo-forge toolbox:

```bash
toolbox enter halo-forge
```

## Quick Validation

### 1. Check Dependencies

```bash
python -c "import sympy; print(f'SymPy: {sympy.__version__}')"
```

Expected output: `SymPy: 1.12` or higher

### 2. Test Imports

```python
from halo_forge.reasoning import MathVerifier, ReasoningRAFTTrainer
from halo_forge.reasoning.data import GSM8KLoader, MATHLoader
from halo_forge.reasoning.verifiers import AnswerExtractor

print("All imports successful!")
```

### 3. Test MathVerifier

```python
from halo_forge.reasoning import MathVerifier

verifier = MathVerifier()

# Test numeric verification
result = verifier.verify(
    prompt="What is 2 + 2?",
    completion="Let me calculate: 2 + 2 = 4. \\boxed{4}",
    expected_answer="4"
)

print(f"Success: {result.success}")
print(f"Reward: {result.reward}")
# Expected: Success: True, Reward: 1.0
```

### 4. Test AnswerExtractor

```python
from halo_forge.reasoning.verifiers import AnswerExtractor

extractor = AnswerExtractor()

# Test boxed format
answer = extractor.extract("After calculation, \\boxed{42}")
print(f"Boxed: {answer}")  # Expected: 42

# Test "answer is" format
answer = extractor.extract("The answer is 3.14159")
print(f"Text: {answer}")  # Expected: 3.14159
```

## Run Unit Tests

```bash
pytest tests/test_reasoning.py -v
```

Expected: All tests passing

## CLI Testing

### List Datasets

```bash
halo-forge reasoning datasets
```

Expected output:
```
Available Math/Reasoning Datasets
============================================================
  gsm8k        [Grade School] - 8.5K problems, 2-8 step solutions
  math         [Competition ] - 12.5K problems, 7 subjects, 5 levels
  aime         [Competition ] - AIME problems (hard)
```

### Benchmark (Dry Run)

Test without downloading the full dataset:

```bash
halo-forge reasoning benchmark \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --limit 5 \
  --dry-run
```

### Full Benchmark

Run an actual benchmark (downloads model and dataset):

```bash
halo-forge reasoning benchmark \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset gsm8k \
  --limit 20 \
  --output results/reasoning_test.json
```

## Verification Strategies

The MathVerifier uses multiple strategies:

### Numeric Match

```python
# These all match "4"
"4"
"4.0"
"4.00"
```

### Symbolic Match (via SymPy)

```python
# These are symbolically equivalent
"x^2 + 2x + 1"
"(x + 1)^2"
```

### Partial Credit

If the answer is wrong but reasoning steps are shown, partial credit (0.2) is awarded.

## Common Issues

### SymPy Not Installed

```
ImportError: No module named 'sympy'
```

Fix: `pip install sympy>=1.12`

### Answer Not Extracted

If verification fails with "Could not extract answer":
- Ensure the model outputs `\boxed{answer}` format
- Or uses "The answer is X" pattern

### Low Accuracy on MATH

MATH dataset is very challenging. Try:
1. Start with GSM8K first
2. Use a stronger base model (7B+)
3. Increase training cycles
