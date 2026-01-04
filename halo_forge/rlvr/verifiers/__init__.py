"""Verifier implementations for RLVR training."""

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult, ChainedVerifier, RewardLevel
from halo_forge.rlvr.verifiers.compile import GCCVerifier, MinGWVerifier, ClangVerifier
from halo_forge.rlvr.verifiers.remote_compile import RemoteMSVCVerifier, RemoteConfig
from halo_forge.rlvr.verifiers.test_runner import PytestVerifier, UnittestVerifier
from halo_forge.rlvr.verifiers.custom import CustomVerifier, SubprocessVerifier
from halo_forge.rlvr.verifiers.pytest_verifier import (
    RLVRPytestVerifier,
    HumanEvalVerifier,
    MBPPVerifier,
)
from halo_forge.rlvr.verifiers.rust_verifier import RustVerifier, CargoVerifier
from halo_forge.rlvr.verifiers.go_verifier import GoVerifier

__all__ = [
    # Base
    "Verifier",
    "VerifyResult",
    "ChainedVerifier",
    "RewardLevel",
    # Compile - C/C++
    "GCCVerifier",
    "MinGWVerifier",
    "ClangVerifier",
    # Compile - Rust
    "RustVerifier",
    "CargoVerifier",  # Alias for RustVerifier
    # Compile - Go
    "GoVerifier",
    # Remote
    "RemoteMSVCVerifier",
    "RemoteConfig",
    # Test
    "PytestVerifier",
    "UnittestVerifier",
    # RLVR Dataset Verifiers (HumanEval/MBPP)
    "RLVRPytestVerifier",
    "HumanEvalVerifier",
    "MBPPVerifier",
    # Custom
    "CustomVerifier",
    "SubprocessVerifier",
]

