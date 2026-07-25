---
title: "halo forge"
---

# halo forge

**Cross-vendor local finetuning workstation.**
SFT · DPO · GRPO · RAFT, with verifier-grounded rewards, on ROCm · CUDA · Apple MLX · Apple MPS.

Halo Forge keeps one guided data-to-training workflow across supported accelerator backends. ROCm and CUDA guided paths are exposed only after their pinned managed runtime passes real hardware qualification.

Pick a goal. Choose a catalog model. Pick an algorithm and verifier. Train. Evaluate. Serve.

Start fast:

- [I want my first local training run](/docs/getting-started/quickstart/)
- [I want to choose a model](/docs/getting-started/choose-a-model/)
- [I want runnable scenarios](/docs/getting-started/scenarios/)
- [I want to serve or export](/docs/serving/)

```bash
# Strix Halo / RTX 4090 / Apple M-series — same commands.
halo-forge sft train   --dataset codealpaca --model Qwen/Qwen2.5-Coder-3B
halo-forge dpo train   --dataset ultrafeedback --model Qwen/Qwen2.5-3B-Instruct
halo-forge grpo train  --data prompts.jsonl --verifier execution --num-generations 8

halo-forge eval        --model ./models/sft/final_model --tasks core
halo-forge merge       --base Qwen/Qwen2.5-3B-Instruct --adapter ./my-lora --output ./shipped
halo-forge convert     --source ./shipped --format gguf --quant q4 --output ./out.gguf --verify
halo-forge serve       --model ./shipped
```

[**Documentation →**](/docs/)
