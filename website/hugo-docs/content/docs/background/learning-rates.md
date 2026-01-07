---
title: "Learning Rate Strategies"
description: "Experimental learning rate recommendations for RAFT training"
weight: 4
---

> **Experimental**: These recommendations are based on observations, not exhaustive validation.

## Why This Matters

During production training, we observed **degradation after cycle 6** using constant LR:

| Cycle | Observation | LR Used | Result |
|-------|-------------|---------|--------|
| 1-6 | Improving | 5e-5 | Peak performance |
| 7 | Declining | 5e-5 | Degradation begins |
| 8 | Declining | 5e-5 | Further degradation |

**Hypothesis**: Decreasing LR across cycles might prevent late-cycle degradation.

## Quick Reference

| Phase | Learning Rate | Notes |
|-------|---------------|-------|
| SFT (LoRA) | 2e-4 | Well-established |
| RAFT Cycle 1 | 5e-5 | ~1/4 of SFT LR |
| RAFT Cycle 5 | 2e-5 to 3e-5 | Theoretical decay |

**Rule of thumb**: RAFT LR ≈ SFT LR / 4

## Why Different Learning Rates?

### SFT vs RAFT

| Aspect | SFT | RAFT |
|--------|-----|------|
| Data source | Fixed, curated | Model-generated (filtered) |
| Goal | Learn new capabilities | Refine behavior |
| Risk | Overfit to training data | Mode collapse |
| Recommended LR | 2e-4 | 5e-5 |

### The Distribution Shift Problem

Each RAFT cycle trains on data from the previous model:

```
Cycle 1: Train on outputs from base model
Cycle 2: Train on outputs from Cycle 1 model
...
```

This creates compounding effects:
- Good: Model improves at generating verified code
- Bad: Distribution narrows, diversity decreases
- Risk: Collapse to single "safe" pattern

Lower LR in later cycles makes smaller updates as distribution contracts.

## Decay Strategies

### Strategy A: Constant (Baseline)

```yaml
training:
  learning_rate: 5e-5  # same all cycles
```

Simple, but may cause degradation after 5-6 cycles.

### Strategy B: Exponential Decay (Recommended)

```python
def get_cycle_lr(base_lr: float, cycle: int, decay: float = 0.85) -> float:
    return base_lr * (decay ** (cycle - 1))
```

| Cycle | LR (0.85 decay) |
|-------|-----------------|
| 1 | 5.0e-5 |
| 2 | 4.25e-5 |
| 3 | 3.61e-5 |
| 4 | 3.07e-5 |
| 5 | 2.61e-5 |

### Strategy C: Manual Schedule

```yaml
training:
  learning_rate_schedule:
    cycle_1: 5.0e-5
    cycle_2: 4.0e-5
    cycle_3: 3.0e-5
    cycle_4: 2.5e-5
    cycle_5: 2.0e-5
```

Full control, but requires tuning.

## Decay Factor Comparison

```
Base LR: 5e-5 across 5 cycles

Factor      | Cycle 1 | Cycle 3 | Cycle 5 | Notes
────────────┼─────────┼─────────┼─────────┼───────────────
Constant    | 5.0e-5  | 5.0e-5  | 5.0e-5  | Baseline
0.95 (gentle)| 5.0e-5 | 4.5e-5  | 4.1e-5  | Minimal decay
0.85 (std)  | 5.0e-5  | 3.6e-5  | 2.6e-5  | Recommended
0.70 (aggro)| 5.0e-5  | 2.5e-5  | 1.2e-5  | If std fails
```

## Diagnostic Signals

### LR Too High

- Training loss oscillates or spikes
- Cycle N+1 worse than Cycle N
- Gradient norm frequently hits max
- Outputs become repetitive

### LR Too Low

- Training loss barely moves
- Multiple cycles, same verification rate
- Very small gradient norms (< 0.05)

## Interaction Effects

| Change | LR Adjustment |
|--------|---------------|
| LoRA rank ↑ | Slightly lower |
| Batch size ↑ | Slightly higher |
| Temperature ↑ | Can go higher |
| Smaller dataset | Lower |

## Within-Cycle Settings

For short training (1 epoch on filtered samples):

```yaml
training:
  warmup_steps: 0       # Few total steps, skip warmup
  lr_scheduler_type: linear
```

With gradient accumulation:
```
500 samples / 2 batch / 16 accumulation = ~16 optimizer steps
```

With so few steps, within-cycle scheduling has minimal effect. Focus on getting base LR right.

## Recommendations

1. **Start with 5e-5 constant** — Establish baseline
2. **Monitor gradient norms** — Should be 0.1-0.2, not clipping
3. **Watch for degradation** — If cycle N+1 drops, reduce LR
4. **Try 0.85 decay** — If constant causes late degradation
5. **Log everything** — Data drives decisions
