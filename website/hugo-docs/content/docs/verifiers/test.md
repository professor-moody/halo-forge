---
title: "Test Verifiers"
description: "pytest and unittest verification"
weight: 2
---

Test verifiers check code by running tests against it.

## PytestVerifier

For Python code with pytest:

```python
from halo_forge.rlvr.verifiers import PytestVerifier

verifier = PytestVerifier()
result = verifier.verify(python_code_with_tests)
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `test_file` | None | External test file to run |
| `extra_args` | `['-v', '--tb=short']` | pytest arguments |
| `timeout` | 60 | Test timeout (seconds) |
| `max_workers` | 4 | Parallel test runs |

### Inline Tests

If code contains its own tests:

```python
code = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def test_factorial():
    assert factorial(5) == 120
    assert factorial(0) == 1
'''

result = verifier.verify(code)
```

### External Tests

Run external tests against generated code:

```python
verifier = PytestVerifier(test_file="tests/test_solution.py")
result = verifier.verify(solution_code)
```

The external test file can import from the generated code.

## UnittestVerifier

For Python's built-in unittest:

```python
from halo_forge.rlvr.verifiers import UnittestVerifier

verifier = UnittestVerifier()
result = verifier.verify(code_with_unittest)
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `timeout` | 60 | Test timeout (seconds) |
| `max_workers` | 4 | Parallel test runs |

## Rewards

Test verifiers use graduated rewards based on test pass rate:

| Outcome | Reward |
|---------|--------|
| All tests pass | 1.0 |
| 75%+ pass | 0.75 |
| 50%+ pass | 0.5 |
| Some pass | 0.25 |
| None pass | 0.0 |

## HumanEval/MBPP Verifiers

For standard benchmarks:

```python
from halo_forge.rlvr.verifiers import HumanEvalVerifier, MBPPVerifier

# HumanEval
verifier = HumanEvalVerifier("data/rlvr/humaneval_full.jsonl")
result = verifier.verify(code, task_id="HumanEval/0")

# MBPP
verifier = MBPPVerifier("data/rlvr/mbpp_train_full.jsonl")
result = verifier.verify(code, task_id="mbpp/1")
```

These verifiers:
- Load test cases from the dataset
- Combine generated code with tests
- Run pytest and return results

## Batch Verification

```python
verifier = PytestVerifier(max_workers=8)
codes = [code1, code2, code3, ...]
prompts = [prompt1, prompt2, prompt3, ...]

# Prompts help look up correct test cases
results = verifier.verify_batch(codes, prompts=prompts)
```

## Best Practices

1. **Set reasonable timeouts** — Tests shouldn't run forever
2. **Use isolated environments** — Prevent side effects between tests
3. **Include edge cases** — Tests should cover corner cases
4. **Clean up resources** — Tests should not leave temp files

## CLI Usage

```bash
# With pytest verifier
halo-forge raft train \
  --checkpoint models/sft/final_model \
  --prompts data/prompts.jsonl \
  --verifier pytest \
  --cycles 5

# With HumanEval
halo-forge raft train \
  --checkpoint models/sft/final_model \
  --prompts data/humaneval_prompts.jsonl \
  --verifier humaneval \
  --cycles 5
```
