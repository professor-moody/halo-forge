"""
Training Verifiers for RAFT

These verifiers provide **graduated reward signals** for the RAFT training loop.
They are NOT benchmarks — they are training infrastructure.

Graduated Reward Ladder:
- 0.0: Failed to generate valid code
- 0.3: Valid syntax, failed to compile
- 0.5: Compiled, failed tests
- 0.7: Passed some tests
- 1.0: Passed all tests

This gradient enables RAFT to learn from partial successes, not just perfect solutions.

For benchmark reporting (paper comparison), use halo_forge.benchmark.
See docs/VERIFIERS.md for the full verifier documentation.
"""

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
# Track V2 — LLM-as-judge. Imported here so the @register_verifier
# decorator on LLMJudgeVerifier fires at package init and the registry
# carries it without a separate plugin install.
from halo_forge.rlvr.verifiers.llm_judge import LLMJudgeVerifier
# Track V3 — schema verifiers (JSON-structure / JSON-schema / regex).
# Track V4 — reference-metric verifiers (BLEU / ROUGE / chrF).
# Both rely on the V1 plugin registry; importing the modules fires
# the @register_verifier decorators so the short names are available
# from list_registered_verifiers() at startup.
from halo_forge.rlvr.verifiers.schema import (
    JSONSchemaVerifier,
    JSONStructureVerifier,
    RegexFormatVerifier,
)
from halo_forge.rlvr.verifiers.metrics import (
    BLEUVerifier,
    ChrFVerifier,
    ROUGEVerifier,
)

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
    # Track V1 — plugin registry. Decorator-based registration plus lazy
    # discovery of `~/.halo-forge/verifiers/*.py` and entry-point packages.
    "register_verifier",
    "get_verifier",
    "list_registered_verifiers",
    # Track V2 — LLM-as-judge with rubric + pluggable judge callable.
    "LLMJudgeVerifier",
    # Track V3 — schema verifiers.
    "JSONStructureVerifier",
    "JSONSchemaVerifier",
    "RegexFormatVerifier",
    # Track V4 — reference-metric verifiers.
    "BLEUVerifier",
    "ROUGEVerifier",
    "ChrFVerifier",
]


# V1 — register the verifiers halo-forge ships under canonical short
# names so the registry is non-empty out of the box. The seeding helper
# is idempotent and the discovery of user plugins still happens lazily
# on the first `get_verifier` / `list_registered_verifiers` call.
from halo_forge.rlvr.verifiers.registry import (  # noqa: E402
    get_verifier,
    list_registered_verifiers,
    register_verifier,
    _seed_builtin_registrations,
)

_seed_builtin_registrations()

