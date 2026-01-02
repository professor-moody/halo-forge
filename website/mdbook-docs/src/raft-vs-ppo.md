# RAFT vs PPO vs GRPO

Comparing reinforcement learning approaches for code generation.

## Overview

| Method | Type | Memory | Complexity | Stability |
|--------|------|--------|------------|-----------|
| PPO | Online RL | 4x model | High | Tricky |
| GRPO | Online RL | 2x model | Medium | Better |
| RAFT | Offline | 1x model | Low | Stable |

## PPO (Proximal Policy Optimization)

The standard RL approach:

```
1. Generate samples
2. Compute rewards
3. Estimate advantage
4. Update policy with clipping
5. Update value function
```

**Requires**:
- Policy model
- Reference model (frozen)
- Value model
- Reward model

**Memory**: ~4x single model

**Challenges**:
- Hyperparameter sensitive (clip ratio, KL penalty)
- Value function estimation is hard
- Reward hacking

## GRPO (Group Relative Policy Optimization)

DeepSeek's improvement on PPO:

```
1. Generate multiple samples per prompt
2. Rank within group
3. Use relative rewards (no value function)
4. Update policy
```

**Memory**: ~2x single model

**Benefits over PPO**:
- No value function needed
- More stable gradients
- Less hyperparameter tuning

## RAFT (Reward-Ranked Fine-Tuning)

Our approach — offline, simple:

```
1. Generate samples
2. Verify with compiler
3. Filter by threshold
4. SFT on filtered samples
5. Repeat
```

**Memory**: 1x model

**Benefits**:
- No reward model needed
- No value function
- Just standard SFT training
- Very stable

## When to Use What

| Scenario | Recommendation |
|----------|----------------|
| Limited compute | RAFT |
| Binary verification (compile) | RAFT |
| Complex reward shaping | PPO/GRPO |
| Research/experimentation | GRPO |
| Production simplicity | RAFT |

## Our Experience

We tried GRPO on Strix Halo:

- **Speed**: ~10x slower than RAFT
- **Memory**: Hit limits with 7B model
- **Results**: Similar quality to RAFT

RAFT's simplicity won for our use case.

## Code Comparison

### RAFT
```python
for cycle in range(5):
    samples = generate(prompts, n=8)
    filtered = [s for s in samples if verify(s).reward >= 0.5]
    model.train(filtered)
```

### GRPO
```python
for step in range(1000):
    samples = generate(prompts, n=8)
    rewards = [verify(s).reward for s in samples]
    # Complex: compute group-relative advantages
    # Complex: policy gradient with KL penalty
    model.update(samples, advantages)
```

## References

- PPO: Schulman et al., 2017
- GRPO: DeepSeek-R1, 2024
- RAFT: Dong et al., 2023
