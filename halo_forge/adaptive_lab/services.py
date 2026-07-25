"""Compatibility import for callers that use the plural service module."""

from .service import AdaptiveLabError, AdaptiveLabService, EvidenceBundleExecutionError

__all__ = ["AdaptiveLabError", "AdaptiveLabService", "EvidenceBundleExecutionError"]
