---
title: "Labs V11–V15"
weight: 68
description: "Outcome validation, controlled adaptation studies, grounded data, specialized task models, and deterministic agent environments."
---

Halo Forge’s next five Lab phases extend the managed own-data workflow without
adding a project system or an unrestricted experiment matrix:

```text
Own data → proof run → validated outcome → full run
         → controlled adaptation study
         → grounded reviewed data
         → specialized task models
         → replayable local environments and trajectories
```

## Outcome validation

After a proof run, **Assess proof outcome** checks the optimizer update,
artifact and replay integrity, dataset identity, and compatible development
evidence. Technical completion and quality change are reported separately. A
full run requires the completed assessment or a reasoned operator override.

## Adaptation studies

Experiments → **Studies** supports paired A/B, dose response, and bounded 2×2
designs. The default paired seeds are 17, 42, and 101. Domain uptake and
general-capability retention remain separate evidence.

## Grounded data

From an immutable corpus Dataset Version, **Create grounded data** proposes
cited training or development-evaluation records. Every citation retains its
document, source span, and source hash. Generated records remain suggestions
until a person creates and completes a Review Studio queue.

## Specialized task models

Guided Own Data supports text and media classification, multi-label
classification, embedding pairs, and reranking. Verified task artifacts include
their model head, processor, label or retrieval contract, fixed-input
verification, and replay identity. Local serving exposes classifications,
embeddings, and reranking without claiming generative-model compatibility.

## Agent environments

Evaluate → **Environments** provides deterministic local fixtures, episode
suites, step evidence, snapshots, exact trace replay, comparisons, and reviewed
trajectory publication. The first release cannot silently write to external
systems and does not add online environment reinforcement learning.

All five phases use additive schema v18, replay v9, the durable Activity
scheduler, bounded APIs, and existing Dataset, Review, Evaluation, Artifact,
and training services.
