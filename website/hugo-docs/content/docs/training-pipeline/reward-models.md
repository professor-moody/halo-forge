---
title: "Reward Models"
description: "Train a scorer from chosen/rejected examples"
weight: 15
---

Reward-model training builds a scorer that estimates which answer is better for a prompt. It uses the same preference-pair data as DPO and ORPO.

## Dashboard

Open **Train**, choose **Preferences**, then choose **Reward model**. Use this when you need a reusable scorer for later ranking or RL workflows.

## CLI

```bash
halo-forge rm train --dataset ultrafeedback --model Qwen/Qwen2.5-3B-Instruct --output ~/.halo-forge/runs/rm-chat
```

## Outputs

Reward-model runs write `training_summary.json`, logs, and a final model artifact when training completes. Results shows the local workstation path even when the browser cannot open the file directly.
