---
title: "Dashboard Training"
description: "Use the Halo Forge dashboard as the primary operator surface"
weight: 22
---

The dashboard is the normal path for local workstation training. The CLI remains the automation path, but the dashboard is designed for operators who want to choose a goal, check preflight, launch, monitor, serve, and inspect results without memorizing every flag.

## Start vs Train

- **Start** is the beginner-safe path. It stays SFT-only and chooses conservative defaults.
- **Train** is the full method surface. It supports SFT, RAFT, DPO, ORPO, RM, GRPO, VLM, audio, reasoning, and agentic training.
- **Runs** monitors active and completed work.
- **Results** answers what to do next: open the run, serve the final model, compare, or inspect local paths.

## Default Output Path

Dashboard launches save under:

```bash
~/.halo-forge/runs/<method>-<goal-or-template>-<model-slug>
```

This avoids installed-app permission failures from repo-relative `models/...` paths.

## Method Preconditions

| Method | Needs |
|---|---|
| SFT | model, dataset, writable output path |
| RAFT | model, prompt file, verifier |
| DPO/ORPO | model, preference dataset |
| RM | model, preference dataset |
| GRPO | model, prompt dataset, verifier |
| VLM | compatible VLM family and image-text data |
| Audio | audio dependencies, task, audio data |
| Reasoning | compatible text model and reasoning data |
| Agentic | tool-call traces or structured-output data |

When a method is capability-gated, the dashboard shows the reason and keeps the CLI path documented.

## Serving After Training

When a run produces a final model or adapter path, open **Results** and choose **Serve model**. Halo Forge manages one local serve process at a time and sends Playground to the managed endpoint by default.
