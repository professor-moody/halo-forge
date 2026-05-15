---
title: "Tool-Use And Agentic Training"
description: "Function-calling and structured tool-use training"
weight: 20
---

Agentic training teaches models to produce structured tool calls and multi-turn traces.

## Dashboard

Open **Train**, choose **Tool use**, then choose SFT, Agentic, or GRPO. Use schema or tool-call verifiers when the output shape matters.

## CLI

```bash
halo-forge agentic train --dataset xlam_sft --model Qwen/Qwen2.5-1.5B-Instruct --output ~/.halo-forge/runs/agentic-tools
```

Good custom data records the user request, available tools, expected tool call, and final answer.
