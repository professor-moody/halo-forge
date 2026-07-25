"""Public V17 product-completion types.

The types intentionally contain user-facing labels alongside exact technical
identity. Normal clients can render the labels and keep hashes/IDs behind an
advanced disclosure without inventing their own status translations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


class Serializable:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SetupRemediation(Serializable):
    id: str
    label: str
    description: str
    automatic: bool
    action: str
    blocker: Optional[str] = None


@dataclass(frozen=True)
class DistributionCapability(Serializable):
    platform: str
    architecture: str
    execution_surfaces: tuple[str, ...]
    desktop_package: Optional[str]
    desktop_status: str
    signature_state: str
    runtime_version: str
    supported_backends: tuple[str, ...]
    unavailable_reason: Optional[str] = None


@dataclass(frozen=True)
class WorkstationReadiness(Serializable):
    id: str
    status: str
    display_status: str
    summary: str
    checks: tuple[Dict[str, Any], ...]
    remediations: tuple[SetupRemediation, ...]
    capability: DistributionCapability
    content_hash: str
    created_at: str
    primary_action: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class DatasetIssue(Serializable):
    id: str
    session_id: str
    ordinal: int
    record_id: Optional[str]
    source_index: Optional[int]
    code: str
    category: str
    severity: str
    field_path: Optional[str]
    message: str
    suggested_actions: tuple[str, ...]
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetRepairAction(Serializable):
    ordinal: int
    issue_code: str
    action_kind: str
    reason: str
    record_id: Optional[str] = None
    source_index: Optional[int] = None
    field_path: Optional[str] = None
    value: Any = None
    before_hash: Optional[str] = None
    after_hash: Optional[str] = None


@dataclass(frozen=True)
class DatasetRepairPlanRevision(Serializable):
    id: str
    session_id: str
    revision_number: int
    source_fingerprint: str
    content_hash: str
    actions: tuple[DatasetRepairAction, ...]
    created_at: str


@dataclass(frozen=True)
class DatasetRepairSession(Serializable):
    id: str
    source_id: Optional[str]
    inspection_id: Optional[str]
    dataset_version_id: Optional[str]
    source_uri: str
    source_fingerprint: str
    scenario_revision_id: Optional[str]
    status: str
    stage: str
    progress: Dict[str, Any]
    issue_summary: Dict[str, Any]
    latest_plan_revision_id: Optional[str]
    latest_preview_id: Optional[str]
    published_repair_revision_id: Optional[str]
    work_item_id: Optional[str]
    error: Optional[str]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class DatasetRepairPreview(Serializable):
    id: str
    session_id: str
    plan_revision_id: str
    source_fingerprint: str
    status: str
    exact: bool
    counts: Dict[str, int]
    issue_counts: Dict[str, int]
    split_impact: Dict[str, Any]
    sample: tuple[Dict[str, Any], ...]
    content_hash: Optional[str]
    storage_path: Optional[str]
    work_item_id: Optional[str]
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]


@dataclass(frozen=True)
class SupportBundlePreview(Serializable):
    categories: tuple[str, ...]
    included: tuple[Dict[str, Any], ...]
    excluded_by_default: tuple[str, ...]
    redaction_policy: str


@dataclass(frozen=True)
class SupportBundle(Serializable):
    id: str
    status: str
    categories: tuple[str, ...]
    preview: Dict[str, Any]
    manifest: Dict[str, Any]
    storage_path: Optional[str]
    content_hash: Optional[str]
    work_item_id: Optional[str]
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]


@dataclass(frozen=True)
class ReleaseQualification(Serializable):
    id: str
    platform: str
    architecture: str
    package_type: str
    signature_state: str
    smoke_status: str
    supported_backends: tuple[str, ...]
    evidence: Dict[str, Any]
    content_hash: str
    work_item_id: Optional[str]
    created_at: str
    status: str = "completed"
    progress: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    completed_at: Optional[str] = None
