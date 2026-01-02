# halo-forge

An RLVR framework using compiler feedback as reward signals for iterative model refinement.

---

## Motivation

Traditional approaches to improving code generation models face limitations:

| Approach | Limitation |
|----------|------------|
| SFT only | Distribution mismatch — model outputs differ from training data |
| RLHF | Expensive human labeling, inconsistent judgments |
| Self-evaluation | Models hallucinate correctness, signals can be gamed |

> **Core insight:** A compiler provides a perfect reward signal — unambiguous, deterministic feedback about code correctness that cannot be gamed.

## Approach

halo-forge implements RAFT (Reward-Ranked Fine-Tuning), which is essentially iterated rejection sampling:

```python
for cycle in range(num_cycles):
    # 1. Generate samples
    samples = model.generate(prompts, n=8)
    
    # 2. Verify with compiler
    results = verifier.verify_batch(samples)
    
    # 3. Filter by reward threshold
    filtered = [s for s, r in zip(samples, results) 
                if r.reward >= 0.5]
    
    # 4. Fine-tune on verified samples
    model.train(filtered)
```

This is simpler than PPO/GRPO (1x model memory vs 2-4x), stable to train, and produces comparable results.

## Results

Production training on Qwen2.5-Coder-7B with 569 C/C++ prompts:

| Stage | Compile Rate | pass@1 |
|-------|-------------|--------|
| SFT Baseline | 15.2% | 18.7% |
| Cycle 1 | 28.4% | 35.2% |
| Cycle 3 | 39.7% | 48.2% |
| Cycle 6 (Peak) | 46.7% | 55.3% |

**3x improvement** over 6 RAFT cycles. Diminishing returns observed after cycle 6.

## Graduated Rewards

Binary rewards create sparse gradients. halo-forge uses graduated rewards:

| Outcome | Reward |
|---------|--------|
| Syntax error | 0.0 |
| Compiles with warnings | 0.3 |
| Compiles clean | 0.5 |
| Runs without crash | 0.7 |
| Correct output | 1.0 |

## Hardware

Optimized for AMD Strix Halo (gfx1151) with 128GB unified memory:

- **BF16 is optimal** — 4-bit quantization actually slower due to dequantization overhead
- **GPU utilization** reaches 95-99% during training
- **Generation speed:** ~80 tok/s for 7B models

## Getting Started

```bash
# Build the toolbox
cd halo-forge/toolbox && ./build.sh

# Enter container
toolbox enter halo-forge

# Validate
halo-forge test --level smoke
```

See [Quick Start](./quickstart.md) for full installation, or [Theory](./theory.md) for research background.

## Related Projects

- [malagent](https://github.com/professor-moody/malagent) — Applies RLVR to security research (EDR evasion with Elastic Security as verifier)

## License

Apache 2.0
