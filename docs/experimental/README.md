# Experimental Research

This section contains experimental research, theoretical frameworks, and untested hypotheses for RLVR training optimization.

**Status**: EXPERIMENTAL - Not validated in production runs

---

## Contents

| Document | Description |
|----------|-------------|
| [LEARNING_RATE_THEORY.md](LEARNING_RATE_THEORY.md) | Deep dive into learning rate strategies for RAFT |
| [LR_QUICK_REFERENCE.md](LR_QUICK_REFERENCE.md) | Quick reference card for LR selection |
| [configs/](configs/) | Ready-to-use experimental config files |

## Motivation

During our RAFT training experiments on AMD Strix Halo, we observed:

- **Cycles 1-6**: Consistent improvement in verification rate
- **Cycle 6**: Peak performance
- **Cycles 7-8**: Performance degradation when using constant LR

We used **constant learning rate (5e-5)** throughout all 8 cycles.

This led us to hypothesize that:

1. **Distribution narrowing** occurs as the model trains on its own filtered outputs
2. **Constant LR becomes too aggressive** for the narrowing distribution
3. **Decaying LR across cycles** might prevent late-cycle degradation

## Experimental Configs

```
configs/
├── raft_constant_lr.yaml      # Baseline (what we ran before)
├── raft_decay_lr.yaml         # Standard 0.85 decay per cycle
└── raft_aggressive_decay.yaml # Aggressive 0.7 decay per cycle
```

## Calculator Tool

```bash
# Compare decay strategies
python scripts/lr_schedule_calculator.py --compare

# Visualize
python scripts/lr_schedule_calculator.py --compare --visualize

# Get recommendations
python scripts/lr_schedule_calculator.py --recommendations
```

## Hardware Context

All experiments target **AMD Strix Halo**:
- 128GB unified LPDDR5X memory
- gfx1151 architecture
- **Compute-bound** (96-99% GPU utilization with BF16)
- BF16 optimal, 4-bit quantization NOT recommended

See [HARDWARE_NOTES.md](../HARDWARE_NOTES.md) for details.

## What We Know vs. What We're Testing

| Aspect | Known (Validated) | Testing (Hypothesis) |
|--------|-------------------|---------------------|
| BF16 is optimal | Yes | - |
| 5e-5 LR works for RAFT | Yes (cycles 1-6) | - |
| Degradation after cycle 6 | Yes (observed) | - |
| LR decay prevents degradation | Unknown | Testing |
| Optimal decay factor | Unknown | 0.7 vs 0.85 vs 0.95 |

## Contributing

When running experiments:

1. **Log everything** - Use the metrics in the configs
2. **One variable at a time** - Don't change LR and temperature together
3. **Document results** - Update this section with findings
4. **Compare to baseline** - Always run constant LR baseline for comparison

---

*Last updated: January 2, 2026*

