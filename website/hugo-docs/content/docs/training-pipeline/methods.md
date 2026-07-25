---
title: "Choose a Training Method"
description: "Pick the right Halo Forge trainer from the dashboard or CLI"
weight: 1
---

Halo Forge exposes the same training service in the dashboard and CLI. Use
**Train → Guided** for a bounded first run, then switch to **Advanced** when you
need an exact method or less common controls.

| Goal | Start with | Move to | Data shape |
|---|---|---|---|
| Adapt to a document corpus without labels | CPT | SFT after labeled examples exist | extracted document text |
| Domain/style adaptation | SFT | DPO or ORPO | prompt/completion |
| Code with executable checks | SFT | RAFT or GRPO | prompts plus verifier |
| Preference alignment | DPO or ORPO | RM, then GRPO | prompt/chosen/rejected |
| Verifier-grounded reasoning | SFT or reasoning | GRPO | prompts plus verifier |
| Vision-language | VLM | VLM with verifier gating | image + prompt + answer |
| Audio | Audio | Audio with task verifier | audio + transcript/label |
| Tool use | Agentic | GRPO with schema/tool verifier | messages/tool calls |

## Dashboard Flow

1. Open **Train**.
2. Choose a goal: Documents, Code, Reasoning, Tool use, Vision, Audio, or
   Preferences.
3. Choose the method Halo Forge should run.
4. Review the generated launch, preflight, output path, and backend notes.
5. Launch, then monitor the run from **Runs** and inspect artifacts in **Models**.

The dashboard intentionally hides unusual flags until the advanced drawer is opened. The CLI remains available for exact reproducibility and scripting.

## Method Guide

- **SFT** learns from labeled examples. Use it first unless you already have preference data.
- **CPT** continues causal next-token training over a reviewed, tokenizer-packed
  corpus. Use it when the source is documents rather than labeled examples.
- **RAFT** generates multiple answers, verifies them, keeps the useful ones, then trains.
- **DPO** uses chosen/rejected pairs to improve behavior without an explicit reward model.
- **ORPO** uses the same pair data as DPO but skips the reference model.
- **RM** trains a reward scorer from chosen/rejected pairs.
- **GRPO** uses a verifier as reward and performs group-relative policy updates.
- **VLM**, **audio**, **reasoning**, and **agentic** are domain-specific training surfaces with capability gates where needed.

## Related Pages

- [Dashboard training](/docs/reference/dashboard-training/)
- [Corpus adaptation](/docs/data/corpus-adaptation/)
- [SFT](/docs/training-pipeline/sft/)
- [RAFT](/docs/training-pipeline/raft/)
- [Preference tuning](/docs/training-pipeline/preference-tuning/)
- [Reward models](/docs/training-pipeline/reward-models/)
- [GRPO](/docs/training-pipeline/grpo/)
- [Dataset formats](/docs/training-pipeline/dataset-formats/)
- [Artifacts](/docs/training-pipeline/artifacts/)
