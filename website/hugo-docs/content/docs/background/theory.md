---
title: "Theory & Research"
description: "RLVR paradigm and research foundations"
weight: 1
---

## The RLVR Paradigm

**Reinforcement Learning from Verifiable Rewards (RLVR)** uses deterministic, programmatic signals instead of human feedback.

### Traditional RLHF Problems

| Problem | RLHF | RLVR |
|---------|------|------|
| Signal source | Human annotators | Compilers, tests, APIs |
| Consistency | Varies by annotator | Deterministic |
| Scale | Expensive to scale | Unlimited |
| Gaming | Can be manipulated | Cannot be fooled |
| Latency | Days to weeks | Milliseconds |

### When to Use RLVR

RLVR works when you have:

1. **Verifiable correctness** — A program can check if output is correct
2. **Binary or graduated signal** — Clear pass/fail or quality score
3. **Fast feedback** — Verification completes quickly

Examples:
- Code generation (compiler)
- Math problems (symbolic checker)
- SQL queries (database execution)
- API compliance (schema validator)

## RAFT: Reward-Ranked Fine-Tuning

RAFT is simpler than PPO/GRPO while achieving similar results.

### The Algorithm

```
Initialize model M from SFT checkpoint
For each cycle:
    1. Generate N samples per prompt using M
    2. Verify all samples, get rewards
    3. Filter to samples with reward >= threshold
    4. Take top K% of filtered samples
    5. Fine-tune M on selected samples
    6. Repeat
```

### Why RAFT Works

1. **Iterated rejection sampling** — Each cycle moves the distribution toward higher rewards
2. **Simple optimization** — Just SFT on good examples, no value networks
3. **Stable training** — No reward hacking, no mode collapse
4. **Memory efficient** — 1x model memory vs 2-4x for PPO

### Algorithm Comparison

| Method | Memory | Stability | Complexity |
|--------|--------|-----------|------------|
| RAFT | 1x | Tends to be stable | Lower |
| PPO | 4x | Requires tuning | Higher |
| GRPO | 2x | Moderate | Moderate |
| DPO | 1x | Tends to be stable | Lower |

RAFT uses offline training (generate-verify-filter-train) rather than online RL updates.

## Graduated Rewards

Binary rewards create sparse gradients:

```
reward = 1.0 if compiles else 0.0  # Bad: no gradient for "almost compiles"
```

Graduated rewards provide denser signal:

```python
if not compiles:
    reward = 0.0      # Syntax error
elif has_warnings:
    reward = 0.3      # Close
elif not runs:
    reward = 0.5      # Compiles but crashes
elif wrong_output:
    reward = 0.7      # Runs but incorrect
else:
    reward = 1.0      # Full reward
```

This helps training because:
- Near-misses get partial credit
- Model learns "direction" to improve
- Faster convergence to working code

## Curriculum Effects

RAFT exhibits natural curriculum learning:

| Early Cycles | Late Cycles |
|--------------|-------------|
| Fix syntax errors | Optimize logic |
| Add missing includes | Handle edge cases |
| Fix type mismatches | Improve structure |

The filter threshold creates implicit curriculum:
- Cycle 1: Keep anything that compiles
- Cycle 5: Keep only high-quality samples

## Scaling Laws

Empirical observations:

1. **More samples per prompt** → Better filtering, higher peak
2. **More prompts** → More diversity, less overfitting
3. **More cycles** → Diminishing returns after 5-6
4. **Higher threshold** → Fewer samples, potentially better quality

## Research Foundations

### Key Papers

1. **RAFT: Reward rAnked FineTuning** (Dong et al., 2023)
   - Original RAFT algorithm
   - [OpenReview](https://openreview.net/forum?id=m7p5O7zblY)

2. **STaR: Self-Taught Reasoner** (Zelikman et al., 2022)
   - Related iterative self-improvement
   - [arXiv](https://arxiv.org/abs/2203.14465)

3. **DeepSeek-Coder-V2** (2024)
   - Uses RLVR for code models
   - Scaled to 236B parameters

4. **Qwen2.5-Coder** (2024)
   - Code-focused model family
   - Base for halo-forge experiments

### Related Work

- **Rejection sampling fine-tuning** (RSO)
- **Constitutional AI** (Anthropic)
- **Self-play** (AlphaGo, AlphaCode)
- **Iterative distillation**

## Open Questions

1. **Cycle count** — When to stop? Current heuristic: monitor for degradation
2. **Prompt diversity** — How much variation needed to prevent overfitting?
3. **Reward shaping** — Suitable reward function for different domains?
4. **Transfer** — Do RAFT improvements transfer across tasks?

## Contributions

halo-forge contributes:

1. **Framework** — Reusable RLVR training infrastructure
2. **Verifiers** — Pluggable verification system
3. **Hardware optimization** — Strix Halo unified memory configuration
4. **Documentation** — Practical guidance for RLVR training
