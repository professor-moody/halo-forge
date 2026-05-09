"""Agentic / tool-calling training module.

Verifier and data helpers are cheap to import; trainer classes are resolved
lazy so importing ``halo_forge.agentic`` does not require torch.
"""

from halo_forge.agentic.verifiers import (
    ToolCallingVerifier,
    ToolCallVerifyResult,
    ToolCallingVerifyConfig,
)
from halo_forge.agentic.data import (
    ToolCallSample,
    XLAMLoader,
    GlaiveLoader,
    list_agentic_datasets,
)

__all__ = [
    "ToolCallingVerifier",
    "ToolCallVerifyResult",
    "ToolCallingVerifyConfig",
    "AgenticRAFTTrainer",
    "AgenticRAFTConfig",
    "ToolCallSample",
    "XLAMLoader",
    "GlaiveLoader",
    "list_agentic_datasets",
]


def __getattr__(name: str):
    if name in {"AgenticRAFTTrainer", "AgenticRAFTConfig"}:
        from halo_forge.agentic import trainer

        return getattr(trainer, name)
    raise AttributeError(name)
