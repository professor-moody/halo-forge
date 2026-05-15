---
title: "Preference Tuning"
description: "DPO and ORPO training from chosen/rejected examples"
weight: 14
---

Preference tuning uses pair data: each row has a prompt, a chosen answer, and a rejected answer.

Use **DPO** when you want the standard reference-model objective. Use **ORPO** when you want a simpler reference-free pass with lower memory pressure.

## Dashboard

Open **Train**, choose **Preferences**, then choose **DPO** or **ORPO**. The generated launch defaults to UltraFeedback-style pair data and a small batch size.

## CLI

```bash
halo-forge dpo train --dataset ultrafeedback --model Qwen/Qwen2.5-3B-Instruct --output ~/.halo-forge/runs/dpo-chat
halo-forge orpo train --dataset ultrafeedback --model Qwen/Qwen2.5-3B-Instruct --output ~/.halo-forge/runs/orpo-chat
```

## Data Format

JSONL rows should contain:

```json
{"prompt": "Explain X", "chosen": "Better answer", "rejected": "Worse answer"}
```

DPO also supports `loss_type` values such as `sigmoid`, `ipo`, `hinge`, and `kto_pair`.
