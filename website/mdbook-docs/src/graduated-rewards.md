# Graduated Rewards

Why partial credit matters for RLVR training.

## The Problem with Binary Rewards

Binary rewards (0 or 1) create sparse gradients:

```
Code with syntax error    → 0.0
Code that almost compiles → 0.0  # Same as complete failure!
Code that compiles        → 1.0
```

The model can't distinguish between "terrible" and "almost there".

## Graduated Reward Scale

halo-forge uses a graduated scale:

| Outcome | Reward | Signal |
|---------|--------|--------|
| Syntax error | 0.0 | Complete failure |
| Missing includes | 0.1 | Structural issues |
| Type errors | 0.2 | Logic issues |
| Compiles with warnings | 0.3 | Almost there |
| Compiles clean | 0.5 | Compilation success |
| Runs, crashes | 0.6 | Runtime issues |
| Runs, wrong output | 0.7 | Logic correct-ish |
| Runs, partial output | 0.8 | Nearly correct |
| Correct output | 1.0 | Full success |

## How It Works

```python
from halo_forge.rlvr.verifiers import RewardLevel

# From compilation result
reward = RewardLevel.from_compile_result(
    success=True,
    has_warnings=True
)
# Returns 0.3

# From execution result
reward = RewardLevel.from_execution_result(
    compiles=True,
    runs=True,
    correct=False
)
# Returns 0.7
```

## Impact on Training

### Binary Rewards

```
Cycle 1: 15% success → train on 15%
Cycle 2: 18% success → train on 18%
Cycle 3: 20% success → train on 20%
```

Limited learning signal.

### Graduated Rewards

```
Cycle 1: 15% full success, but 40% above threshold
Cycle 2: 22% full success, 55% above threshold
Cycle 3: 30% full success, 65% above threshold
```

More samples contribute to learning.

## Threshold Selection

The `reward_threshold` parameter controls what's kept:

| Threshold | Effect |
|-----------|--------|
| 0.3 | Keep code with warnings |
| 0.5 | Keep clean compiles (recommended) |
| 0.7 | Keep code that runs |
| 1.0 | Only correct output |

Start with 0.5 (compilation) and increase as model improves.

## Curriculum Learning

Use graduated thresholds across cycles:

```yaml
cycles:
  - reward_threshold: 0.3  # Early: accept warnings
  - reward_threshold: 0.5  # Mid: require clean compile
  - reward_threshold: 0.7  # Late: require execution
```

This naturally progresses from "make it compile" to "make it work".

## Research Background

The graduated reward approach is inspired by:

- **Reward shaping** in RL literature
- **Curriculum learning** (Bengio et al., 2009)
- **Sparse reward problem** in robotics RL

Key insight: intermediate rewards accelerate learning when the final goal is hard to reach directly.
