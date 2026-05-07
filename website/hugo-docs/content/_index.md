---
title: "halo forge"
---

# halo forge

**Cross-vendor local finetuning workstation.**
SFT · DPO · GRPO · RAFT, with verifier-grounded rewards, on ROCm · CUDA · Apple MLX · Apple MPS.

The single thing that makes halo-forge different from every adjacent project (axolotl, llama-factory, unsloth, mlx-lm-lora, torchtune): **it runs natively on every modern accelerator**, not just CUDA.

Pick a model. Pick an algorithm. Pick a verifier. Pick a backend. Train. Evaluate. Serve.

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
