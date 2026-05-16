---
title: "GRPO"
description: "Verifier-grounded RL with group-relative advantages"
weight: 16
---

GRPO samples multiple completions per prompt, scores them with a verifier, and updates the policy using group-relative advantages.

Use GRPO when you have a programmatic reward: execution, tests, schema validation, exact answer checks, or an LLM judge.

## Dashboard

Open **Train**, choose **Reasoning**, **Code**, **Tool use**, or **Preferences**, then choose **GRPO**. Pick the verifier before launch. Preflight checks the model, dataset, output path, and backend state.

## CLI

```bash
halo-forge grpo train \
  --dataset gsm8k \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --verifier json_schema \
  --num-generations 4 \
  --output ~/.halo-forge/runs/grpo-reasoning
```

Start with small group sizes, then increase `--num-generations` once verifier yield is healthy.
