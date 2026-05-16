---
title: "Vision-Language Training"
description: "VLM training for image and text tasks"
weight: 17
---

Vision-language training adapts models for visual question answering, document extraction, screenshots, charts, and image-grounded reasoning.

## Dashboard

Open **Train**, choose **Vision**, then choose **Vision-language**. Some VLM families are capability-gated until local dependencies and deterministic qualification are confirmed.

## CLI

```bash
halo-forge vlm train --dataset textvqa --model Qwen/Qwen2-VL-2B-Instruct --output ~/.halo-forge/runs/vlm-textvqa
```

Use custom JSONL when your task needs local image paths or structured expected answers.
