# Theory & Research

Research foundations behind halo-forge.

## The Problem

Standard fine-tuning approaches have limitations:

### SFT (Supervised Fine-Tuning)
```
Input: "Write a sorting function"
Output: <human-written solution>
```

Problems:
- **Distribution mismatch**: Model outputs differ from training data
- **Error amplification**: Model may memorize patterns, not understand
- **No negative signal**: Model never sees what NOT to do

### RLHF (Reinforcement Learning from Human Feedback)
```
Model generates → Human rates → Train on preferences
```

Problems:
- **Expensive**: Humans must evaluate each output
- **Inconsistent**: Human judgment varies
- **Slow**: Can't scale to millions of samples

## The Solution: RLVR

Replace human feedback with automated verification:

```
Model generates → Verifier checks → Train on verified outputs
```

Benefits:
- **Free**: No human labeling cost
- **Consistent**: Same input always gives same result
- **Scalable**: Verify thousands per minute
- **Perfect precision**: Compiler never misses syntax errors

## RAFT Algorithm

We implement **RAFT (Reward-Ranked Fine-Tuning)** from:

> Dong et al., "RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment", TMLR 2023

```python
for cycle in range(num_cycles):
    samples = generate(prompts, n=8)
    results = verify_batch(samples)
    filtered = filter(samples, results, threshold=0.5)
    model = train(model, filtered)
```

### Why RAFT over PPO?

| Approach | Complexity | Memory | Stability |
|----------|------------|--------|-----------|
| PPO | High | 4x model | Requires tuning |
| GRPO | Medium | 2x model | Better |
| RAFT | Low | 1x model | Simple, stable |

RAFT is essentially "iterated rejection sampling" — simple but effective.

## Graduated Rewards

Binary rewards create sparse gradients. We use graduated rewards:

| Outcome | Binary | Graduated |
|---------|--------|-----------|
| Syntax error | 0.0 | 0.0 |
| Compile warnings | 1.0 | 0.3 |
| Compiles clean | 1.0 | 0.5 |
| Runs, wrong output | 0.0 | 0.7 |
| Correct output | 1.0 | 1.0 |

This creates smoother learning: "almost compiling" > "syntax error".

## Related Work

- **STaR** (Zelikman et al., 2022): Self-Taught Reasoner, bootstrap from correct outputs
- **RFT** (Yuan et al., 2023): Rejection sampling fine-tuning
- **Code Llama**: Used RLHF but reported high annotation costs

## Limitations

RAFT with compile verification only ensures code compiles, not correctness:

- Code may compile but produce wrong output
- Edge cases may not be handled
- Efficiency is not optimized

For correctness, use test-based verifiers (pytest).

## References

1. Dong et al., "RAFT: Reward rAnked FineTuning", TMLR 2023
2. Zelikman et al., "STaR: Self-Taught Reasoner", NeurIPS 2022
3. Chen et al., "HumanEval: Evaluating Large Language Models", 2021
