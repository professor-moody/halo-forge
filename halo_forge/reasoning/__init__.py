"""
Reasoning Module

Phase 5: Math and reasoning verification for RLVR training.

Provides verifiers for mathematical correctness using SymPy,
answer extraction, and dataset loaders for math problems.
"""

from halo_forge.reasoning.verifiers import (
    MathVerifier,
    AnswerExtractor,
    ReasoningVerifyResult,
    MathVerificationError,
)
from halo_forge.reasoning.data import (
    MathSample,
    GSM8KLoader,
    MATHLoader,
    load_math_dataset,
    list_math_datasets,
)
from halo_forge.reasoning.trainer import (
    ReasoningRAFTTrainer,
    ReasoningRAFTConfig,
)

__all__ = [
    # Verifiers
    "MathVerifier",
    "AnswerExtractor",
    "ReasoningVerifyResult",
    "MathVerificationError",
    # Data
    "MathSample",
    "GSM8KLoader",
    "MATHLoader",
    "load_math_dataset",
    "list_math_datasets",
    # Training
    "ReasoningRAFTTrainer",
    "ReasoningRAFTConfig",
]
