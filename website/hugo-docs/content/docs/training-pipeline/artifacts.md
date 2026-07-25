---
title: "Training Artifacts"
description: "Files written by Halo Forge training runs"
weight: 22
---

Every dashboard launch writes into a dedicated output directory under `~/.halo-forge/runs` unless you use the CLI and choose another path.

Common files:

| File | Purpose |
|---|---|
| `launch_context.json` | reproducible launch command and UI args |
| `training_summary.json` | normalized run summary for Runs and Models |
| `final_model/` | final artifact when training succeeds |
| `latest_checkpoint.json` | resume metadata for cycle-based methods |
| `*_training.log` | captured stdout/stderr log |

The browser cannot open arbitrary local files directly in every environment,
so **Runs** and **Models** show the local workstation path.
