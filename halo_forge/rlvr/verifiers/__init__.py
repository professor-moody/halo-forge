"""Verifier implementations for RLVR training."""

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult, ChainedVerifier, RewardLevel
from halo_forge.rlvr.verifiers.compile import GCCVerifier, MinGWVerifier, ClangVerifier
from halo_forge.rlvr.verifiers.execution import (
    ExecutionVerifier,
    GCCExecutionVerifier,
    MinGWExecutionVerifier,
    ClangExecutionVerifier,
    TestCase,
)
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
from halo_forge.rlvr.verifiers.dotnet_verifier import DotNetVerifier, CSharpVerifier
from halo_forge.rlvr.verifiers.powershell_verifier import PowerShellVerifier, PS1Verifier
from halo_forge.rlvr.verifiers.multi_language import MultiLanguageVerifier, AutoVerifier, LanguageConfig

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
    # Execution - C/C++ with test cases
    "ExecutionVerifier",
    "GCCExecutionVerifier",
    "MinGWExecutionVerifier",
    "ClangExecutionVerifier",
    "TestCase",
    # Compile - Rust
    "RustVerifier",
    "CargoVerifier",  # Alias for RustVerifier
    # Compile - Go
    "GoVerifier",
    # Compile - .NET/C#
    "DotNetVerifier",
    "CSharpVerifier",  # Alias for DotNetVerifier
    # Script - PowerShell
    "PowerShellVerifier",
    "PS1Verifier",  # Alias for PowerShellVerifier
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
    # Multi-Language (auto-detection)
    "MultiLanguageVerifier",
    "AutoVerifier",  # Alias for MultiLanguageVerifier
    "LanguageConfig",
]

