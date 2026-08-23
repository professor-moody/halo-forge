---
title: "Recommended training plan"
weight: 25
---

After you publish a clean Dataset Version, Halo Forge presents one recommended
plan based on the data, scenario, verified runtime, and current workstation.
Normal mode explains the model, approach, data use, expected download, memory,
and proof time without asking for low-level optimization settings.

```text
Recommended plan → Prepare and check → Ready for proof run → Start proof run
```

**Prepare and check** confirms any displayed public-model download, resolves an
immutable model commit, verifies required model files, and performs disposable
capacity work. Scratch weights and optimizer state are removed afterward. A
capacity check never counts as training and never reads protected evaluation
splits.

The check runs through the selected trainer adapter using deterministic
median- and maximum-cost training records. It verifies model loading,
collation, forward and backward passes, optimizer allocation, and a disposable
optimizer step. Halo Forge retains only identities, shapes, hashes, timings,
and resource measurements—not the source examples.

If the confirmed shape does not fit, Halo Forge may try verified gradient
checkpointing and then a smaller group of examples with compensating
accumulation. It does not silently change the model, maximum text length,
objective, learning rate, verifier, reward behavior, or dataset.

After a successful check, **Start proof run** is the single primary action. The
proof continues into **Check training result** before a full run is offered.

Advanced configuration and existing direct trainer commands remain available.
When a plan revision is supplied, it takes precedence and contradictory raw
settings are rejected.
