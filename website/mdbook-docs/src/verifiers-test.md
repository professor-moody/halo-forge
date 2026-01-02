# Test Verifiers

Verifiers that run test suites against generated code.

## PytestVerifier

Run pytest tests against generated Python code.

```python
from halo_forge.rlvr.verifiers import PytestVerifier

verifier = PytestVerifier(
    test_template="tests/template.py",
    timeout=60
)
```

### Test Template

The template receives the generated code as `{code}`:

```python
# tests/template.py
{code}

def test_function():
    assert function_name(5) == expected_output
    assert function_name(0) == edge_case
```

### Reward Mapping

| Outcome | Reward |
|---------|--------|
| Syntax error | 0.0 |
| Import error | 0.1 |
| Some tests pass | 0.3-0.9 |
| All tests pass | 1.0 |

## UnittestVerifier

Standard library unittest framework:

```python
from halo_forge.rlvr.verifiers import UnittestVerifier

verifier = UnittestVerifier(
    test_file="tests/test_cases.py"
)
```

## RLVRPytestVerifier

Specialized verifier for datasets with embedded tests (MBPP, HumanEval):

```python
from halo_forge.rlvr.verifiers import RLVRPytestVerifier

verifier = RLVRPytestVerifier()

# prompt_data includes test cases
result = verifier.verify(code, prompt_data)
```

### Data Format

```json
{
  "prompt": "Write a function to find the maximum",
  "test_cases": [
    "assert find_max([1, 2, 3]) == 3",
    "assert find_max([-1, -2]) == -1"
  ]
}
```

## Combining with RAFT

```yaml
# configs/raft_pytest.yaml
verifier:
  type: pytest
  test_template: tests/template.py
  timeout: 60

raft:
  reward_threshold: 0.5  # At least some tests must pass
```

## Tips

- **Timeout**: Set generous timeouts for complex tests
- **Isolation**: Each verification runs in a clean environment
- **Edge cases**: Include edge cases in your test templates
