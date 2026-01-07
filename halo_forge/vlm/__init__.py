"""
Vision-Language Model (VLM) Training Module

RLVR training for vision-language models with perception-aware verification.

Key Components:
- VLMRAFTTrainer: RAFT trainer adapted for VLMs
- VisionVerifier: Multi-stage verification (perception, reasoning, output)
- Dataset loaders: TextVQA, DocVQA, ChartQA
- Model adapters: Qwen-VL, LLaVA
"""

from halo_forge.vlm.trainer import VLMRAFTTrainer
from halo_forge.vlm.verifiers import VisionVerifier, PerceptionChecker, ReasoningChecker, OutputChecker

__all__ = [
    "VLMRAFTTrainer",
    "VisionVerifier",
    "PerceptionChecker",
    "ReasoningChecker",
    "OutputChecker",
]
