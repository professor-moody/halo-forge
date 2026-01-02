# RLVR Production Run: Data Preparation Guide

## Overview

For RLVR training with pytest verification, you need datasets that include **both prompts AND test cases**. The standard halo-forge data specs only include prompts and solutions.

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  HuggingFace                    Output Files                    │
│  ───────────                    ────────────                    │
│                                                                  │
│  openai/openai_humaneval  ──►  humaneval_prompts.jsonl          │
│  (164 problems)                humaneval_full.jsonl (with tests)│
│                                humaneval_validation.jsonl (20)   │
│                                                                  │
│  google-research-datasets/     mbpp_train_prompts.jsonl         │
│  mbpp (974 problems)      ──►  mbpp_train_full.jsonl (with tests)│
│                                mbpp_validation.jsonl (20)        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING FLOW                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  RAFT Training                 Verification                      │
│  ─────────────                 ────────────                      │
│                                                                  │
│  mbpp_train_prompts.jsonl  ──► Model generates code              │
│  (prompts only)                      │                           │
│                                      ▼                           │
│  mbpp_train_full.jsonl     ──► RLVRPytestVerifier combines:      │
│  (has test cases)              generated_code + test_cases       │
│                                      │                           │
│                                      ▼                           │
│                                pytest runs tests                 │
│                                      │                           │
│                                      ▼                           │
│                                reward signal ──► RAFT training   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     BENCHMARK FLOW                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  humaneval_full.jsonl  ──►  Generate N samples per problem       │
│  (164 problems)                    │                             │
│                                    ▼                             │
│                             Verify each sample                   │
│                                    │                             │
│                                    ▼                             │
│                             Calculate pass@1, pass@5, pass@10    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Commands

### 1. Prepare Datasets

```bash
# In toolbox
toolbox enter halo-forge

# Run preparation script
python scripts/prepare_rlvr_datasets.py \
    --output-dir data/rlvr \
    --dataset both \
    --validation-size 20
```

This creates:
```
data/rlvr/
├── humaneval_prompts.jsonl      # 164 prompts for generation
├── humaneval_full.jsonl         # 164 prompts + tests for verification
├── humaneval_solutions.jsonl    # Reference solutions (for analysis)
├── humaneval_validation.jsonl   # 20 problems for quick testing
├── mbpp_train_prompts.jsonl     # 974 prompts for generation
├── mbpp_train_full.jsonl        # 974 prompts + tests for verification
├── mbpp_train_solutions.jsonl   # Reference solutions
├── mbpp_validation.jsonl        # 20 problems for quick testing
├── mbpp_sanitized_prompts.jsonl # ~427 cleaner problems (optional)
├── mbpp_sanitized_full.jsonl    # Sanitized with tests
└── dataset_summary.json         # Stats and file paths
```

### 2. File Formats

**Prompts file** (for RAFT generation):
```json
{"task_id": "HumanEval/0", "prompt": "def has_close_elements(numbers: List[float]...", "entry_point": "has_close_elements"}
```

**Full file** (for verification):
```json
{"task_id": "HumanEval/0", "prompt": "...", "tests": "def check(candidate):\n    assert...", "entry_point": "has_close_elements"}
```

### 3. Verification Integration

The `RLVRPytestVerifier` class handles combining generated code with tests:

```python
from scripts.rlvr_pytest_verifier import HumanEvalVerifier, MBPPVerifier

# For HumanEval benchmarking
verifier = HumanEvalVerifier("data/rlvr/humaneval_full.jsonl")
result = verifier.verify(generated_code, task_id="HumanEval/0")

# For MBPP training
verifier = MBPPVerifier("data/rlvr/mbpp_train_full.jsonl")
result = verifier.verify(generated_code, task_id="123")
```

## Production Run Configuration

### Validation Run (3B, 2-3 hours)

```yaml
# configs/validation_run.yaml
model:
  name: Qwen/Qwen2.5-Coder-3B
  
data:
  prompts: data/rlvr/humaneval_validation.jsonl
  tests: data/rlvr/humaneval_full.jsonl
  
raft:
  cycles: 3
  samples_per_prompt: 4
  
verifier:
  type: rlvr_pytest
  dataset_type: humaneval
  timeout: 30
```

### Production Run (7B, 12-24 hours)

```yaml
# configs/production_run.yaml
model:
  name: Qwen/Qwen2.5-Coder-7B
  
data:
  prompts: data/rlvr/mbpp_train_prompts.jsonl
  tests: data/rlvr/mbpp_train_full.jsonl
  
raft:
  cycles: 5
  samples_per_prompt: 8
  
verifier:
  type: rlvr_pytest
  dataset_type: mbpp
  timeout: 30
  
checkpointing:
  save_every_cycle: true
  output_dir: models/production_raft
```

### Benchmark Configuration

```yaml
# configs/benchmark.yaml
model:
  checkpoints:
    - models/base  # Qwen2.5-Coder-7B raw
    - models/sft   # After SFT (if applicable)
    - models/production_raft/cycle_1
    - models/production_raft/cycle_3
    - models/production_raft/cycle_5
    
data:
  prompts: data/rlvr/humaneval_full.jsonl
  
benchmark:
  samples_per_problem: 10
  k_values: [1, 5, 10]
  
verifier:
  type: rlvr_pytest
  dataset_type: humaneval
```

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Training data | MBPP (974) | Larger, more diverse |
| Benchmark data | HumanEval (164) | Industry standard, comparable |
| Validation subset | 20 problems | Quick iteration, ~10 min |
| Samples per prompt | 8 | Balance diversity vs. compute |
| Temperature | 0.7 | Good diversity for RAFT |

## Troubleshooting

**"No test cases found for task"**
- Verify the task_id format matches (HumanEval uses "HumanEval/N", MBPP uses integers)
- Check the full dataset file was loaded correctly

**"Test execution timed out"**
- Increase timeout (some problems need more time)
- Check for infinite loops in generated code

**"Import errors in test"**
- HumanEval may need imports (typing, etc.)
- Add common imports to test file header

## Next Steps

1. Run `prepare_rlvr_datasets.py` to generate all data files
2. Verify with a single problem manually
3. Run validation with 3B model
4. Fix any issues
5. Start production 7B run