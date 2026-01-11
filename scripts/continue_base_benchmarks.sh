#!/bin/bash
# continue_base_benchmarks.sh - Continue from after MBPP
# Picks up from where run_all_base_benchmarks.sh left off

set -e
mkdir -p results/{reasoning,vlm,audio,agentic}/{qwen_base,lfm_base}

# ============================================================================
# REASONING DOMAIN
# ============================================================================

echo "=== Reasoning (Qwen2.5-3B): GSM8K ==="
python -m halo_forge.cli reasoning benchmark \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset gsm8k --split test --limit 500 \
  --output results/reasoning/qwen_base/gsm8k.json

echo "=== Reasoning (LFM2.5-1.2B): GSM8K ==="
python -m halo_forge.cli reasoning benchmark \
  --model LiquidAI/LFM2.5-1.2B-Base \
  --dataset gsm8k --split test --limit 500 \
  --output results/reasoning/lfm_base/gsm8k.json

# ============================================================================
# VLM DOMAIN
# ============================================================================

echo "=== VLM (Qwen2-VL-2B): TextVQA ==="
python -m halo_forge.cli vlm benchmark \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --dataset textvqa --split validation --limit 500 \
  --output results/vlm/qwen_base/textvqa.json

echo "=== VLM (LFM2.5-VL-1.6B): TextVQA ==="
python -m halo_forge.cli vlm benchmark \
  --model LiquidAI/LFM2.5-VL-1.6B \
  --dataset textvqa --split validation --limit 500 \
  --output results/vlm/lfm_base/textvqa.json

# ============================================================================
# AUDIO DOMAIN
# ============================================================================

echo "=== Audio (Whisper-Small): LibriSpeech ==="
python -m halo_forge.cli audio benchmark \
  --model openai/whisper-small \
  --dataset librispeech --task asr --limit 500 \
  --output results/audio/qwen_base/librispeech.json

echo "=== Audio (LFM2.5-Audio-1.5B): LibriSpeech ==="
python -m halo_forge.cli audio benchmark \
  --model LiquidAI/LFM2.5-Audio-1.5B \
  --dataset librispeech --task asr --limit 500 \
  --output results/audio/lfm_base/librispeech.json

# ============================================================================
# AGENTIC DOMAIN
# ============================================================================

echo "=== Agentic (Qwen2.5-7B): xLAM ==="
python -m halo_forge.cli agentic benchmark \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset xlam --limit 500 \
  --output results/agentic/qwen_base/xlam.json

echo "=== Agentic (LFM2.5-1.2B): xLAM ==="
python -m halo_forge.cli agentic benchmark \
  --model LiquidAI/LFM2.5-1.2B-Base \
  --dataset xlam --limit 500 \
  --output results/agentic/lfm_base/xlam.json

# ============================================================================
echo ""
echo "=== All Remaining Base Model Benchmarks Complete ==="
echo ""
echo "Results saved to:"
echo "  - results/reasoning/{qwen_base,lfm_base}/"
echo "  - results/vlm/{qwen_base,lfm_base}/"
echo "  - results/audio/{qwen_base,lfm_base}/"
echo "  - results/agentic/{qwen_base,lfm_base}/"
