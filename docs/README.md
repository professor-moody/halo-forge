# halo forge Documentation Bridge

The canonical user documentation is **[halo-forge.io/docs](https://halo-forge.io/docs/)**.

This repo-local `docs/` directory is kept for release checklists, engineering references, and offline artifacts. Normal product links should point to the website docs.

## Quick Links

| Section | Description |
|---------|-------------|
| [Quick Start](https://halo-forge.io/docs/getting-started/quickstart/) | Get running in 30 minutes |
| [How to Train](https://halo-forge.io/docs/training-pipeline/how-to-train/) | Complete step-by-step guide |
| [Choose a Training Method](https://halo-forge.io/docs/training-pipeline/methods/) | SFT, CPT, RAFT, DPO, ORPO, RM, GRPO, VLM, audio, reasoning, and agentic guide |
| [Dashboard Training](https://halo-forge.io/docs/reference/dashboard-training/) | Goal-first dashboard launch workflow |
| [Use Your Own Data](https://halo-forge.io/docs/data/own-data/) | Guided file/folder import, scenario mapping, validation, immutable versions, and training handoff |
| [Fix Training Data](https://halo-forge.io/docs/data/repair/) | Reviewed, non-destructive repair overlays with exact previews and lineage |
| [Adapt a Model to Documents](https://halo-forge.io/docs/data/corpus-adaptation/) | Text/Markdown/HTML/PDF/DOCX extraction, corpus packing, and CPT |
| [Workstation Surfaces](https://halo-forge.io/docs/reference/workstation-surfaces/) | Desktop, local-browser, remote-browser, and CLI parity plus platform truth |
| [Review Studio](https://halo-forge.io/docs/review-studio/) | Deterministic acquisition, multimodal review, immutable label sets, and Dataset Lab handoff |
| [Reward Integrity](https://halo-forge.io/docs/reward-integrity/) | Same-output training-signal capture, independent sentinel audits, and reviewed gates |
| [Command Index](https://halo-forge.io/docs/reference/command-index/) | Every command and flag |
| [Public Frontend](https://halo-forge.io/docs/reference/public-frontend/) | Local/remote training, run inspection, docs, and connection surface |
| [Verifiers](https://halo-forge.io/docs/verifiers/) | Verification options |
| [Experimental](https://halo-forge.io/docs/experimental/) | VLM, Audio, Reasoning, Agentic readiness tiers and modality caveats |
| [Contributing](https://halo-forge.io/docs/contributing/) | How to contribute |

## Repo-Local Artifacts

These files contain release or engineering detail that may not be the best first stop for users:

| Document | Description |
|----------|-------------|
| [LABS_V11_V15.md](LABS_V11_V15.md) | Outcome validation, adaptation studies, grounding, specialized task models, and agent environments |
| [DATASET_LAB.md](DATASET_LAB.md) | Local text, VLM, and audio dataset factory |
| [EXPERIMENT_OPERATIONS.md](EXPERIMENT_OPERATIONS.md) | Repeats, sweeps, durable work queue, evidence, and recovery |
| [ARTIFACT_STUDIO.md](ARTIFACT_STUDIO.md) | Content-addressed models, qualification, serving, export, and safe cleanup |
| [ADAPTIVE_EVIDENCE.md](ADAPTIVE_EVIDENCE.md) | Checkpoint policies, seed-aware analysis, reviewed decisions, and evidence bundles |
| [REVIEW_STUDIO.md](REVIEW_STUDIO.md) | Active-data proposals, multimodal review, adjudication, and immutable label sets |
| [VERIFIER_RELIABILITY.md](VERIFIER_RELIABILITY.md) | Immutable verifier identity, replicated calibration, qualification, and exact downstream bindings |
| [REWARD_INTEGRITY.md](REWARD_INTEGRITY.md) | Immutable reward systems, boundary signal capture, same-output audits, and replay identity |
| [OWN_DATA_WORKFLOW.md](OWN_DATA_WORKFLOW.md) | Managed own-data source, mapping, validation, version, and trainer handoff |
| [DATASET_REPAIR.md](DATASET_REPAIR.md) | Deterministic repair overlays, source-drift handling, publication, and replay |
| [CORPUS_ADAPTATION.md](CORPUS_ADAPTATION.md) | Immutable document extraction, corpus preparation, token packing, and CPT |
| [WORKSTATION_SURFACES.md](WORKSTATION_SURFACES.md) | Equal operator surfaces, platform status, and native dataset chooser contract |
| [VERIFIED_TRAINING_PATHS.md](VERIFIED_TRAINING_PATHS.md) | Runtime, real trainer-path, plan, and workstation beta certification states |
| [VERIFIERS.md](VERIFIERS.md) | Verifier guide with safety considerations |
| [MODELS.md](MODELS.md) | Supported models reference |
| [HARDWARE_NOTES.md](HARDWARE_NOTES.md) | AMD Strix Halo configuration details |
| [GGUF_EXPORT.md](GGUF_EXPORT.md) | GGUF export guide for llama.cpp/Ollama |
| [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) | Release verification checklist |

## Experimental Configs

Experimental learning rate configurations are in [`configs/experimental/`](../configs/experimental/):

| Config | Description |
|--------|-------------|
| `raft_aggressive_decay.yaml` | Aggressive LR decay (0.7 factor) |
| `raft_constant_lr.yaml` | Constant LR baseline |
| `raft_decay_lr.yaml` | Standard LR decay (0.85 factor) |
