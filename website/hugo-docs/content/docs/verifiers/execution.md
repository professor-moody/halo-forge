---
title: "Execution Verifier"
weight: 25
---

# Execution Verifier

The `ExecutionVerifier` extends compile verification to support **multiple test cases** with input/output pairs. It provides graduated rewards based on test case pass rate.

## Overview

Unlike simple compile verifiers that only check if code compiles, the ExecutionVerifier:

1. **Compiles** the code
2. **Runs** the compiled binary with test case inputs
3. **Compares** actual output to expected output
4. **Calculates** graduated reward based on pass rate

## Reward Structure

| Stage | Reward | Description |
|-------|--------|-------------|
| Compile fail | 0.0 | Code doesn't compile |
| Compile with warnings | 0.3 | Compiles with warnings |
| Compile clean | 0.5 | Compiles without warnings |
| Test cases | 0.5 + 0.5 × pass_rate | Graduated by test pass rate |

For example:
- Pass 0/10 tests: 0.5 reward (compiled but all tests fail)
- Pass 5/10 tests: 0.75 reward
- Pass 10/10 tests: 1.0 reward

## Basic Usage

```python
from halo_forge.rlvr.verifiers import ExecutionVerifier

verifier = ExecutionVerifier(
    test_cases=[
        {"input": "5\n", "expected": "25"},
        {"input": "10\n", "expected": "100"},
    ]
)

code = '''
#include <iostream>
int main() {
    int n;
    std::cin >> n;
    std::cout << n * n << std::endl;
    return 0;
}
'''

result = verifier.verify(code)
print(f"Success: {result.success}")
print(f"Reward: {result.reward}")
print(f"Details: {result.details}")
```

## Test Case Format

Test cases are dictionaries with `input` and `expected` keys:

```python
test_cases = [
    {
        "input": "5\n",           # stdin input
        "expected": "25",          # expected stdout
        "name": "test_square_5",   # optional: test name
        "timeout": 5               # optional: per-test timeout
    },
    {
        "input": "hello world\n",
        "expected": "HELLO WORLD"
    }
]
```

## Output Matching Modes

The `match_mode` parameter controls how outputs are compared:

| Mode | Description | Use Case |
|------|-------------|----------|
| `exact` | Exact string match (default) | Precise output required |
| `contains` | Expected is substring of actual | Partial output matching |
| `regex` | Expected is regex pattern | Flexible matching |
| `numeric` | Extract and compare numbers | Floating-point tolerance |

```python
# Exact match (default)
verifier = ExecutionVerifier(test_cases=tests, match_mode='exact')

# Check if output contains expected
verifier = ExecutionVerifier(test_cases=tests, match_mode='contains')

# Regex pattern matching
verifier = ExecutionVerifier(test_cases=tests, match_mode='regex')

# Numeric comparison with tolerance
verifier = ExecutionVerifier(test_cases=tests, match_mode='numeric')
```

## Pre-configured Variants

### GCCExecutionVerifier

Pre-configured for GCC:

```python
from halo_forge.rlvr.verifiers import GCCExecutionVerifier

verifier = GCCExecutionVerifier(
    test_cases=my_tests,
    flags=['-O2', '-Wall']
)
```

### ClangExecutionVerifier

Pre-configured for Clang:

```python
from halo_forge.rlvr.verifiers import ClangExecutionVerifier

verifier = ClangExecutionVerifier(test_cases=my_tests)
```

### MinGWExecutionVerifier

For Windows cross-compilation (compile-only, cannot execute on Linux):

```python
from halo_forge.rlvr.verifiers import MinGWExecutionVerifier

verifier = MinGWExecutionVerifier(test_cases=my_tests)
# Note: Test cases will be skipped; only compile verification
```

## Dynamic Test Cases

Update test cases based on the prompt:

```python
verifier = ExecutionVerifier()

# Set test cases dynamically
verifier.set_test_cases([
    {"input": "1 2 3\n", "expected": "6"},
    {"input": "10 20 30\n", "expected": "60"},
])

result = verifier.verify(code)
```

## Extracting Test Cases from Prompts

The verifier can attempt to extract test cases from prompt text:

```python
result = verifier.verify_with_prompt(code, prompt)
```

This looks for patterns like:
- `Input: 5  Output: 25`
- `Example: input=5, output=25`

## Configuration Options

```python
verifier = ExecutionVerifier(
    compiler='g++',                    # Compiler command
    flags=['-O2', '-Wall'],            # Compiler flags
    test_cases=my_tests,               # Test case list
    timeout=30,                        # Compile timeout (seconds)
    run_timeout=5,                     # Per-test timeout (seconds)
    max_workers=8,                     # Parallel workers
    match_mode='exact',                # Output matching mode
    partial_credit=True,               # Give partial credit for some passes
    binary_cache_dir='binaries/'       # Cache compiled binaries
)
```

## Using with RAFT Training

```python
from halo_forge.rlvr import RAFTTrainer
from halo_forge.rlvr.verifiers import ExecutionVerifier

# Load test cases from your dataset
with open('data/prompts_with_tests.jsonl') as f:
    data = [json.loads(line) for line in f]

# Create verifier with test cases from first prompt
verifier = ExecutionVerifier(
    test_cases=data[0].get('test_cases', [])
)

trainer = RAFTTrainer(verifier=verifier, config=config)
trainer.run(prompts, num_cycles=6)
```

## CLI Usage

```bash
# Not directly available via CLI yet, but you can use the
# execution verifier through the Python API or config files

# Config file example:
# verifier:
#   type: execution
#   test_cases:
#     - input: "5\n"
#       expected: "25"
#   match_mode: exact
```

## Best Practices

1. **Provide multiple test cases** - More tests = better reward signal
2. **Include edge cases** - Empty input, large numbers, special characters
3. **Set reasonable timeouts** - Prevent infinite loops from hanging
4. **Use `partial_credit=True`** - Better gradient signal during training
5. **Consider `numeric` mode** - For floating-point outputs

## Comparison with Compile Verifiers

| Feature | CompileVerifier | ExecutionVerifier |
|---------|-----------------|-------------------|
| Checks compilation | ✓ | ✓ |
| Runs code | Optional | ✓ |
| Multiple test cases | ✗ | ✓ |
| Graduated rewards | Basic | Rich |
| Training signal | Weak | Strong |

Use `ExecutionVerifier` when you have test cases available for richer reward signals during RAFT training.
