#!/bin/bash
# continue_from_vlm.sh - Continue benchmarks from VLM onward
# 
# Already completed (DO NOT RERUN):
#   ✓ results/code/lfm_base/humaneval.json
#   ✓ results/code/lfm_base/mbpp.json
#   ✓ results/reasoning/lfm_base/gsm8k.json
#
# This script picks up from VLM benchmarks onward.

set -e

echo "=============================================="
echo "  HALO-FORGE Base Model Benchmark Continuation"
echo "=============================================="
echo ""
echo "Already completed:"
echo "  ✓ Code (LFM): HumanEval"
echo "  ✓ Code (LFM): MBPP"  
echo "  ✓ Reasoning (LFM): GSM8K"
echo ""
echo "Starting from: VLM benchmarks"
echo "=============================================="
echo ""

mkdir -p results/{vlm,audio,agentic}/{qwen_base,lfm_base}

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
echo "=============================================="
echo "  All Remaining Benchmarks Complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - results/vlm/{qwen_base,lfm_base}/"
echo "  - results/audio/{qwen_base,lfm_base}/"
echo "  - results/agentic/{qwen_base,lfm_base}/"
echo ""
echo "Combined with previously completed:"
echo "  - results/code/lfm_base/"
echo "  - results/reasoning/lfm_base/"
