---
title: "Dashboard Training"
description: "Use the Halo Forge dashboard as the primary operator surface"
weight: 22
---

The dashboard is the normal interactive path for local workstation training.
The CLI remains an equal automation path over the same catalog and services.
Both support choosing a goal, checking preflight, launching, monitoring,
evaluating, serving, and inspecting artifacts without changing data formats.

On ROCm and CUDA workstations, **Setup** first prepares and qualifies a pinned
managed runtime. Train then uses that runtime automatically. A detected GPU is
never shown as ready until the runtime has completed a real optimizer update
and saved-artifact reload. See [Managed Accelerator Runtimes](/docs/reference/managed-runtimes/).

## Guided vs Advanced Train

- **Train → Guided** is the beginner-safe path with conservative defaults and
  structured own-data selection.
- **Train → Advanced** exposes direct method configuration for SFT, RAFT, DPO,
  ORPO, RM, GRPO, VLM, audio, reasoning, and agentic training.
- **Runs** monitors active and completed work and owns completed-run actions.
- **Models** owns trained artifacts, serving, conversion, qualification, and export.

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
The guided picker only exposes methods certified for the active runtime,
trainer adapter, and capacity adapter.

## Serving After Training

When a run produces a final model or adapter, open **Runs → Artifacts** or the
shared **Models** library and choose **Serve**. Halo Forge manages one local
serve process at a time and sends **Models → Serve & Test** to the managed
endpoint by default.
