---
title: "Dataset Formats"
description: "Data shapes expected by each training method"
weight: 21
---

| Method | Required shape |
|---|---|
| SFT | prompt/completion, text, or chat messages |
| RAFT | prompt rows for generation and verification |
| DPO/ORPO/RM | prompt, chosen, rejected |
| GRPO | prompt rows plus a verifier |
| VLM | image path or image payload, prompt, answer |
| Audio | audio path or decoded audio field, transcript or label |
| Reasoning | question and answer, optionally reasoning trace |
| Agentic | messages, tool schema, tool call, final answer |

Local JSONL paths can be selected in **Train** with **Custom local file**. Built-in dataset short names use the same registry as the CLI.
