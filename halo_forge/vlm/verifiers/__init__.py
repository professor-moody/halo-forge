"""
VLM Verifiers Module

Multi-stage verification for vision-language model outputs.
"""

from halo_forge.vlm.verifiers.base import (
    VisionVerifier,
    VisionVerifyResult,
    VLMVerificationError,
    ImageLoadError,
    DependencyWarning,
    check_vlm_dependencies,
)
from halo_forge.vlm.verifiers.perception import PerceptionChecker, PerceptionResult
from halo_forge.vlm.verifiers.reasoning import ReasoningChecker, ReasoningResult
from halo_forge.vlm.verifiers.output import OutputChecker, OutputResult

__all__ = [
    # Core verifier
    "VisionVerifier",
    "VisionVerifyResult",
    # Individual checkers
    "PerceptionChecker",
    "PerceptionResult",
    "ReasoningChecker",
    "ReasoningResult",
    "OutputChecker",
    "OutputResult",
    # Error classes
    "VLMVerificationError",
    "ImageLoadError",
    "DependencyWarning",
    # Utilities
    "check_vlm_dependencies",
]
