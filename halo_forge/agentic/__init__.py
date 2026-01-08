"""
Agentic / Tool Calling Module

Train language models for reliable function/tool calling using RLVR.

This module provides:
- ToolCallingVerifier: Verify tool call correctness with graduated rewards
- AgenticRAFTTrainer: RAFT training loop for tool calling
- Dataset loaders: xLAM, Glaive, ToolBench
- Hermes format conversion

Example:
    from halo_forge.agentic import ToolCallingVerifier, AgenticRAFTTrainer
    from halo_forge.agentic.data import XLAMLoader
    
    # Load dataset
    loader = XLAMLoader()
    samples = loader.load(limit=1000)
    
    # Create verifier
    verifier = ToolCallingVerifier()
    
    # Verify a tool call
    result = verifier.verify(
        output='<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>',
        expected_calls=[{"name": "get_weather", "arguments": {"location": "Paris"}}]
    )
"""

from halo_forge.agentic.verifiers import (
    ToolCallingVerifier,
    ToolCallVerifyResult,
    ToolCallingVerifyConfig,
)
from halo_forge.agentic.trainer import (
    AgenticRAFTTrainer,
    AgenticRAFTConfig,
)
from halo_forge.agentic.data import (
    ToolCallSample,
    XLAMLoader,
    GlaiveLoader,
    list_agentic_datasets,
)

__all__ = [
    # Verifiers
    "ToolCallingVerifier",
    "ToolCallVerifyResult",
    "ToolCallingVerifyConfig",
    # Trainer
    "AgenticRAFTTrainer",
    "AgenticRAFTConfig",
    # Data
    "ToolCallSample",
    "XLAMLoader",
    "GlaiveLoader",
    "list_agentic_datasets",
]
