---
title: "Verified training paths"
weight: 44
---

Halo Forge does not unlock guided training merely because a GPU or a generic optimizer works.

The readiness sequence is **runtime ready → path verified → plan ready → beta qualified**. Path verification sends a pinned fixture through Dataset Lab and the shipped trainer, proves that trainable-parameter hashes changed, reloads the saved artifact, and records replay and lineage evidence.

Use Setup to prepare the accelerator runtime and verify instruction training. A selected path can be verified on demand only when its real certification executor exists. Paths without one remain visibly unavailable with an exact reason; generic diagnostics cannot unlock them. External accelerator work is never terminated; Halo Forge waits and resumes from a checksummed boundary.
