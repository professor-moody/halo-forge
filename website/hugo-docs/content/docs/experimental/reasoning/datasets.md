---
title: "Reasoning Datasets"
weight: 1
---

# Math & Reasoning Datasets

Available datasets for mathematical reasoning training.

## Supported Datasets

### GSM8K (Grade School Math)

**Size:** 8,500 problems  
**Difficulty:** Elementary to middle school  
**Source:** [HuggingFace](https://huggingface.co/datasets/gsm8k)

Problems require 2-8 reasoning steps. Ideal for teaching basic mathematical reasoning.

```bash
# Benchmark
halo-forge reasoning benchmark --dataset gsm8k --limit 100

# Training
halo-forge reasoning train --dataset gsm8k --cycles 4
```

**Example problem:**
```
Janet's ducks lay 16 eggs per day. She eats three for breakfast 
every morning and bakes muffins for her friends every day with 
four. She sells the remainder at the farmers' market daily for 
$2 per fresh duck egg. How much in dollars does she make every 
day at the farmers' market?
```

### MATH (Competition Mathematics)

**Size:** 12,500 problems  
**Difficulty:** High school to competition level  
**Subjects:** Algebra, Geometry, Number Theory, Counting, Probability, Precalculus, Intermediate Algebra  
**Source:** [HuggingFace](https://huggingface.co/datasets/lighteval/MATH)

Competition-level problems across 5 difficulty levels.

```bash
# Benchmark
halo-forge reasoning benchmark --dataset math --limit 50

# Training (requires strong base model)
halo-forge reasoning train --dataset math --cycles 4
```

### AIME (American Invitational Mathematics Examination)

**Size:** ~300 problems  
**Difficulty:** Advanced competition  
**Source:** Historical AIME exams

Very challenging problems. Recommended only for models that already perform well on GSM8K and MATH.

## Data Format

All datasets are loaded via HuggingFace and converted to a standard format:

```python
@dataclass
class ReasoningSample:
    question: str      # The math problem
    answer: str        # Expected final answer
    solution: str      # Step-by-step solution (if available)
    difficulty: str    # Difficulty level (if available)
    subject: str       # Math subject (if available)
```

## Loading Datasets in Python

```python
from halo_forge.reasoning.data import load_math_dataset

# Load GSM8K
gsm8k = load_math_dataset("gsm8k", split="train", limit=1000)

# Load MATH
math_data = load_math_dataset("math", split="train", limit=500)

for sample in gsm8k:
    print(f"Q: {sample.question[:50]}...")
    print(f"A: {sample.answer}")
```

## Dataset Statistics

| Dataset | Train | Test | Avg Steps | Difficulty |
|---------|-------|------|-----------|------------|
| GSM8K | 7,473 | 1,319 | 4.2 | Easy-Medium |
| MATH | 7,500 | 5,000 | 6.8 | Medium-Hard |
| AIME | ~300 | - | 8+ | Very Hard |

## Curriculum Learning

For best results, train in order of difficulty:

1. **GSM8K** (4 cycles) - Build basic reasoning
2. **MATH (Level 1-3)** (4 cycles) - Intermediate skills
3. **MATH (Level 4-5)** (2 cycles) - Advanced techniques

```bash
# Stage 1: GSM8K
halo-forge reasoning train --dataset gsm8k --cycles 4 --output models/stage1

# Stage 2: MATH
halo-forge reasoning train --dataset math --cycles 4 --output models/stage2
```
