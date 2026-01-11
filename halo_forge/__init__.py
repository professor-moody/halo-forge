"""
halo-forge: Complete RLVR Training Framework for AMD Strix Halo

A standalone framework for verification-guided code generation training,
including data generation, SFT, RAFT/RLVR training, and benchmarking.
"""

__version__ = "1.0.0"
__author__ = "keys"

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult

__all__ = [
    "Verifier",
    "VerifyResult",
    "__version__",
]

