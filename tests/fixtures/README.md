# Test Fixtures

This directory contains test data for halo-forge unit tests.

## Contents

- `sample_prompts.jsonl` - Sample prompts for testing benchmark/training
- `vlm_samples.jsonl` - Sample VLM prompts with image references
- `mock_responses.json` - Mock model responses for testing verification
- `conftest.py` - Shared pytest fixtures

## Usage

Fixtures are automatically available via pytest's `conftest.py` mechanism.

```python
def test_example(sample_prompts, vlm_samples):
    # fixtures are injected automatically
    assert len(sample_prompts) > 0
```

## Generating Test Images

Test images are created programmatically in tests to avoid storing binary files.
See `test_vlm.py` and `test_vlm_data.py` for examples.
