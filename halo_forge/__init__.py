"""
halo-forge: Complete RLVR Training Framework for AMD Strix Halo

A standalone framework for verification-guided code generation training,
including data generation, SFT, RAFT/RLVR training, and benchmarking.
"""

from halo_forge.version import DISPLAY_VERSION, PACKAGE_VERSION

__version__ = PACKAGE_VERSION
__display_version__ = DISPLAY_VERSION
__author__ = "keys"

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult

__all__ = [
    "Verifier",
    "VerifyResult",
    "__version__",
    "__display_version__",
]
