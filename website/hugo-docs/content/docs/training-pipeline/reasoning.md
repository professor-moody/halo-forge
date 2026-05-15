---
title: "Reasoning Training"
description: "Math and multi-step reasoning training"
weight: 19
---

Reasoning training focuses on math, answer extraction, and multi-step problem solving.

## Dashboard

Open **Train**, choose **Reasoning**, then choose SFT, Reasoning, or GRPO. Use SFT for format and traces; use GRPO when a verifier can score the final answer.

## CLI

```bash
halo-forge reasoning train --dataset gsm8k --model Qwen/Qwen2.5-1.5B-Instruct --output ~/.halo-forge/runs/reasoning-gsm8k
```

Run a probe or eval after training to catch regressions on general tasks.
