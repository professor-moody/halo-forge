"""Transport-neutral response types for the Artifact Studio facade."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class StudioQueueReceipt:
    """Durable domain and scheduler identities returned at enqueue time."""

    domain_kind: str
    domain_id: str
    work_item_id: Optional[str]
    status: str
    reused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ServingReservation:
    """A serving profile plus its workstation resource reservation.

    ``reserved`` is deliberately distinct from ``serving``: this facade does
    not claim a model server process has started merely because a lease exists.
    """

    serving_id: str
    profile_revision: Mapping[str, Any]
    state: str
    lease: Optional[Mapping[str, Any]] = None
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "serving_id": self.serving_id,
            "profile_revision": dict(self.profile_revision),
            "state": self.state,
            "lease": None if self.lease is None else dict(self.lease),
            "reason": self.reason,
        }


class ArtifactStudioError(RuntimeError):
    """Base error for facade-level lifecycle failures."""


class UnsupportedArtifactCapability(ArtifactStudioError):
    """The requested conversion or execution backend is not implemented."""


class PromotionBlocked(ArtifactStudioError):
    """Qualification evidence does not permit an unreviewed promotion."""


__all__ = [
    "ArtifactStudioError",
    "PromotionBlocked",
    "ServingReservation",
    "StudioQueueReceipt",
    "UnsupportedArtifactCapability",
]
