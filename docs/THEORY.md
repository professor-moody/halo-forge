# Theory & Research Behind halo-forge

This document explains the theoretical foundations, research that informed the design, and lessons learned during development.

## Overview

halo-forge implements **RLVR (Reinforcement Learning from Verifier Rewards)** - a technique for improving code generation models by training on verified outputs rather than relying solely on human preferences or model self-evaluation.

The key insight: **A compiler (or test suite) is a perfect reward signal** - it gives unambiguous, deterministic feedback about code correctness.

## Theoretical Foundations

### The Problem with Traditional Fine-Tuning

Standard SFT (Supervised Fine-Tuning) trains on human-written examples:

```
Input: "Write a sorting function"
Output: <human-written solution>
```

Problems:
1. **Distribution mismatch**: Model's own outputs differ from training data
2. **Error amplification**: Model may memorize patterns, not understand
3. **No negative signal**: Model never sees what NOT to do

### RLHF and Its Limitations

RLHF (Reinforcement Learning from Human Feedback) addresses some issues:

```
Model generates → Human rates → Train on preferences
```

Problems for code:
1. **Expensive**: Humans must evaluate each output
2. **Inconsistent**: Human judgment varies
3. **Slow**: Can't scale to millions of samples
4. **Imperfect**: Humans miss subtle bugs

### RLVR: The Verifier Approach

RLVR replaces human feedback with automated verification:

```
Model generates → Verifier checks → Train on verified outputs
```

Benefits:
1. **Free**: No human labeling cost
2. **Consistent**: Same input always gives same result
3. **Scalable**: Verify thousands per minute
4. **Perfect precision**: Compiler never misses syntax errors

The verifier provides a **ground truth reward signal** that's impossible to game.

## RAFT: Our Implementation

We implement RLVR using **RAFT (Reward-Ranked Fine-Tuning)**, a simpler alternative to full RL algorithms like PPO.

### RAFT Algorithm

```
for cycle in range(num_cycles):
    1. Generate N samples per prompt using current model
    2. Verify all samples with verifier (compile, test, etc.)
    3. Assign rewards based on verification result
    4. Filter to keep top K% samples (reward >= threshold)
    5. SFT on filtered samples
    6. Repeat with updated model
```

### Why RAFT Over PPO/GRPO?

| Approach | Complexity | Memory | Stability |
|----------|------------|--------|-----------|
| PPO | High | 4x model | Requires tuning |
| GRPO | Medium | 2x model | Better but still tricky |
| RAFT | Low | 1x model | Just works |

RAFT is essentially "iterated rejection sampling" - simple to implement, stable to train, and produces similar results with less engineering effort.

### Our GRPO Experience

We initially tried GRPO (Group Relative Policy Optimization):
- Required loading reference model (2x memory)
- Gradient computation was unstable on ROCm
- Hyperparameter sensitivity was high

**Lesson**: Simpler algorithms often outperform complex ones when you have a strong reward signal.

## Key Research Influences

### DeepSeek-R1 (2024)

DeepSeek's work on reasoning models showed that:
- Models can learn to reason through chain-of-thought
- Verification-based training produces more reliable outputs
- Even simple verifiers (format checking) improve reliability

### STaR (Self-Taught Reasoner)

The STaR paper demonstrated:
- Models can bootstrap from their own correct outputs
- Iterative refinement converges to better performance
- Key is having a reliable "correctness" signal

### Code Generation Benchmarks

HumanEval and MBPP showed:
- pass@k is the right metric for code
- Models need multiple attempts (temperature sampling)
- Compilation is necessary but not sufficient

## Strix Halo Optimizations

### Unified Memory Architecture

Strix Halo's 96GB unified memory changes training dynamics:

1. **No GPU memory limit worry**: Can load 14B models in bf16
2. **But slower than discrete**: Bandwidth is lower than VRAM
3. **CPU-GPU coordination**: Need to avoid ping-ponging

**Our optimizations**:
- `dataloader_num_workers=0` (avoid multiprocess issues)
- `dataloader_pin_memory=False` (unified memory doesn't benefit)
- `gradient_checkpointing` with `use_reentrant=False` (ROCm stability)

### ROCm Considerations

ROCm 7 on gfx1151 has specific requirements:

1. **Eager attention**: Flash attention not yet optimized
2. **Native wheels**: Use scottt's PyTorch builds, not emulation
3. **Kernel serialization**: `AMD_SERIALIZE_KERNEL=1` for debugging

### Memory Usage (Strix Halo)

Actual observed memory during 7B bf16 LoRA training (2048 seq len):

| Metric | Value |
|--------|-------|
| **GTT (GPU system memory)** | ~62GB |
| **VRAM (dedicated)** | ~1GB |
| **Total system RAM** | ~71GB |
| **GPU utilization** | 100% |

The GTT (Graphics Translation Table) is where unified memory training happens. Monitor with `radeontop` or similar tools.

**Note**: `torch.cuda.max_memory_allocated()` reports incorrectly on unified memory architecture - it shows ~24GB when actual usage is ~62GB. Always verify with system tools.

## Observations from Training

### Compilation Rate Progression

Typical RAFT progression for C++ code generation:

| Cycle | Compile Rate | Notes |
|-------|--------------|-------|
| SFT only | 20-30% | Many syntax errors |
| Cycle 1 | 35-45% | Basic structure improved |
| Cycle 2 | 45-55% | Fewer missing semicolons |
| Cycle 3 | 55-65% | Better type handling |
| Cycle 4+ | 60-70% | Diminishing returns |

### What Models Learn

During RAFT, we observed models learning:

1. **Syntax correctness**: Matching braces, semicolons
2. **Include statements**: Right headers for functions used
3. **Type consistency**: Matching function signatures
4. **Error handling**: Adding null checks, return values

### What Models Don't Learn (easily)

Some things require more than compile verification:

1. **Algorithm correctness**: Code compiles but wrong answer
2. **Edge cases**: Works for examples, fails on boundaries
3. **Efficiency**: O(n²) when O(n) is possible
4. **Security**: Buffer overflows that compile fine

**Lesson**: Verifier quality determines learning quality.

### Timing Benchmarks (Strix Halo)

| Operation | Time | Notes |
|-----------|------|-------|
| Load 7B QLoRA | 2-3 min | First load |
| Generate 100 samples | 10-15 min | batch_size=4 |
| Verify 100 (GCC) | 30 sec | 8 workers |
| Verify 100 (MSVC/SSH) | 2-3 min | Network overhead |
| Train 1 epoch on 100 | 5-10 min | Depends on seq_len |
| Full RAFT cycle | 30-45 min | Generate + verify + train |

### Resource Usage

During RAFT training:
- **GPU utilization**: 70-90% during generation/training
- **Memory**: 40-60GB typical (out of 96GB)
- **Power**: ~100W sustained

## Multi-Stage Verification

For complex verification, we recommend staged rewards:

```
Stage 1: Format Check (regex) → 0.0 or continue
Stage 2: Compile (GCC/MSVC) → 0.5 or continue  
Stage 3: Tests Pass (pytest) → 1.0 if all pass
```

This creates a **curriculum**:
- First, model learns correct syntax
- Then, model learns to compile
- Finally, model learns to pass tests

### Reward Shaping Matters

We found that binary rewards (0 or 1) work worse than staged:

| Reward Scheme | Final Success Rate |
|---------------|-------------------|
| Binary (pass=1, else=0) | 55% |
| Staged (format=0.1, compile=0.5, pass=1.0) | 68% |

**Lesson**: Partial credit helps gradient flow.

## Future Directions

### Execution-Based Verification

Current limitation: We only check if code compiles, not if it runs correctly.

Next step: Run generated code in sandbox, verify output.

### Formal Verification

For security-critical code: Use theorem provers or static analyzers as verifiers.

### Multi-Language

Current framework supports any language with a compiler/interpreter verifier.

### Scaling

Larger models (14B, 70B) with more RAFT cycles may push boundaries further.

## References

### Core RAFT Paper

**RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment**  
Hanze Dong, Wei Xiong, Deepanshu Goyal, et al.  
Published: TMLR, 23 Nov 2023  
[Paper (OpenReview)](https://openreview.net/forum?id=m7p5O7zblY) | [Code (LMFlow)](https://github.com/OptimalScale/LMFlow)

> "Utilizing a reward model and a sufficient number of samples, our approach selects the high-quality samples, discarding those that exhibit undesired behavior, and subsequently enhancing the model by fine-tuning on these filtered samples."

This is the foundational paper for our RAFT implementation. The key insight is that rejection sampling + SFT is simpler and often as effective as full RL algorithms. The LMFlow repository provides the reference implementation.

### Kyle Avery's Dante Methodology

**Kyle Avery's research** on domain-specific model training informed our approach to:
- High-quality, curated dataset construction
- Domain-expert prompt engineering
- Iterative refinement based on real-world testing
- Evaluation methodology for specialized tasks

The "Dante" approach emphasizes quality over quantity in training data.

### Other Key References

1. **DeepSeek-R1** (2024)
   - Technical report on reasoning models with chain-of-thought
   - Demonstrated verification-based training for reliability
   - [DeepSeek Technical Reports](https://github.com/deepseek-ai)

2. **STaR: Self-Taught Reasoner** (Zelikman et al., 2022)
   - Bootstrap reasoning from correct outputs
   - Iterative self-improvement methodology
   - [Paper](https://arxiv.org/abs/2203.14465)

3. **RLHF: Training language models to follow instructions with human feedback** (Ouyang et al., 2022)
   - Original InstructGPT paper
   - Foundation for alignment research
   - [Paper](https://arxiv.org/abs/2203.02155)

4. **HumanEval: Evaluating Large Language Models Trained on Code** (Chen et al., 2021)
   - OpenAI's code evaluation benchmark
   - Introduced pass@k metric
   - [Paper](https://arxiv.org/abs/2107.03374)

5. **kyuz0's AMD Strix Halo LLM Fine-tuning Guide**
   - Community resource for Strix Halo training
   - QLoRA configuration for ROCm
   - [GitHub](https://github.com/kyuz0/amd-strix-halo-llm-finetuning)

6. **GRPO: Group Relative Policy Optimization** (DeepSeek, 2024)
   - Simplified RLHF alternative
   - Our initial approach before settling on RAFT
   - More complex than RAFT but potentially more powerful

### AMD/ROCm Resources

- **scottt's PyTorch wheels for gfx1151**: Native support for Strix Halo
- **ROCm 7.0 documentation**: HIP and ROCm optimization guides
- **Kernel parameters for Strix Halo**: `amdgpu.gttsize`, `ttm.pages_limit`

