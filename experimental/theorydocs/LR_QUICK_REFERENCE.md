# Learning Rate Quick Reference Card

> **⚠️ EXPERIMENTAL & THEORETICAL** - All values are hypotheses, not validated best practices.

---

## At-a-Glance Recommendations

```
┌─────────────────────────────────────────────────────────────────┐
│                    LEARNING RATE STARTING POINTS                 │
│                         (EXPERIMENTAL)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase              │  Learning Rate  │  Confidence             │
│  ──────────────────────────────────────────────────────────────│
│  SFT (LoRA)         │  2e-4           │  High (well-studied)    │
│  RAFT Cycle 1       │  5e-5           │  Medium                 │
│  RAFT Cycle 5       │  2e-5 to 3e-5   │  Low (theoretical)      │
│                                                                  │
│  Rule of thumb: RAFT LR ≈ SFT LR / 4                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Decay Factor Comparison

```
Base LR: 5e-5 across 5 cycles

Decay Factor │ Cycle 1 │ Cycle 3 │ Cycle 5 │ Notes
─────────────┼─────────┼─────────┼─────────┼──────────────────────
Constant     │ 5.0e-5  │ 5.0e-5  │ 5.0e-5  │ Baseline comparison
0.95 (gentle)│ 5.0e-5  │ 4.5e-5  │ 4.1e-5  │ Minimal decay
0.85 (std)   │ 5.0e-5  │ 3.6e-5  │ 2.6e-5  │ Recommended start
0.70 (aggro) │ 5.0e-5  │ 2.5e-5  │ 1.2e-5  │ If std insufficient
```

---

## Decision Tree

```
                    ┌─────────────────────┐
                    │ Starting RAFT?      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Use 5e-5, decay 0.85│
                    │ (experimental)      │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                      ▼
         ┌─────────────────┐    ┌─────────────────┐
         │ Loss oscillates │    │ Loss flat       │
         │ or degrades?    │    │ no improvement? │
         └────────┬────────┘    └────────┬────────┘
                  │                      │
                  ▼                      ▼
         ┌─────────────────┐    ┌─────────────────┐
         │ REDUCE LR       │    │ INCREASE LR     │
         │ - Try 0.70 decay│    │ - Try 7e-5 base │
         │ - Or lower base │    │ - Or 0.95 decay │
         └─────────────────┘    └─────────────────┘
```

---

## Warning Signs

### LR Too High
- [ ] Training loss spikes or oscillates
- [ ] Validation loss increasing while train loss decreases
- [ ] `gradient_norm` frequently = `max_grad_norm` (clipping)
- [ ] Cycle N+1 worse than Cycle N
- [ ] Generated outputs become repetitive

### LR Too Low  
- [ ] Training loss barely moves
- [ ] Multiple cycles, same verification rate
- [ ] Very small gradient norms (< 0.05)
- [ ] Feels like wasted compute

---

## Config Snippets

### Constant LR (Baseline)
```yaml
training:
  learning_rate: 5e-5  # same all cycles
```

### Exponential Decay
```yaml
training:
  base_learning_rate: 5e-5
  lr_decay_factor: 0.85
```

### Manual Schedule
```yaml
training:
  learning_rate_schedule:
    cycle_1: 5.0e-5
    cycle_2: 4.0e-5
    cycle_3: 3.0e-5
    cycle_4: 2.5e-5
    cycle_5: 2.0e-5
```

---

## Key Interactions

| If you change... | Consider adjusting LR... |
|------------------|-------------------------|
| LoRA rank (r) ↑  | Slightly lower |
| Batch size ↑     | Slightly higher |
| Temperature ↑    | Can go slightly higher |
| Dataset smaller  | Lower (less signal) |

---

## Commands

```bash
# Compare decay strategies
python scripts/lr_schedule_calculator.py --compare

# Visualize schedules
python scripts/lr_schedule_calculator.py --compare --visualize

# Generate YAML for specific config
python scripts/lr_schedule_calculator.py --base-lr 5e-5 --decay 0.85 --yaml

# Print recommendations
python scripts/lr_schedule_calculator.py --recommendations
```

---

## Experimental Configs Location

```
configs/experimental/
├── raft_constant_lr.yaml      # Baseline (no decay)
├── raft_decay_lr.yaml         # Standard decay (0.85)
└── raft_aggressive_decay.yaml # Aggressive decay (0.70)
```

---

## Remember

1. **These are hypotheses**, not proven best practices
2. **Start with baseline**, then adjust based on observations
3. **Log everything** - you'll need data to make decisions
4. **One variable at a time** - don't change LR and temperature together
5. **Trust the loss curves** - they tell you if LR is right

---

*Last updated: January 2, 2026 (pre-production run)*
