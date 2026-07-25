"""Public dataclasses for Halo Forge Labs V11-V15."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class Serializable:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GuidedAction(Serializable):
    id: str
    label: str
    href: Optional[str] = None
    method: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    tone: str = "primary"


@dataclass(frozen=True)
class ActionableGuidance(Serializable):
    context_kind: str
    context_id: str
    display_status: str
    summary: str
    primary_action: Optional[GuidedAction]
    secondary_actions: tuple[GuidedAction, ...] = ()
    blockers: tuple[str, ...] = ()
    technical_details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutcomePreparation(Serializable):
    assessment_id: str
    proof_run_id: str
    suite_revision_id: Optional[str]
    base_evaluation_id: Optional[str]
    candidate_evaluation_id: Optional[str]
    status: str
    stage: str
    work_item_id: Optional[str]
    guidance: ActionableGuidance


@dataclass(frozen=True)
class StudyLaunchPlan(Serializable):
    protocol_revision_id: str
    arm_count: int
    seed_count: int
    run_count: int
    estimated_seconds_low: Optional[float]
    estimated_seconds_high: Optional[float]
    estimated_storage_bytes: Optional[int]
    blockers: tuple[str, ...] = ()
    work_item_id: Optional[str] = None


@dataclass(frozen=True)
class GroundingGenerationPreview(Serializable):
    profile_revision_id: str
    preset: str
    candidate_limit: int
    preview_items: tuple[Dict[str, Any], ...]
    teacher: Dict[str, Any]
    verifier: Dict[str, Any]
    request_estimate: Dict[str, Any]
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnvironmentPermissionSummary(Serializable):
    local_files: bool
    local_sqlite: bool
    loopback_services: bool
    external_writes: bool
    max_steps: int
    timeout_seconds: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnvironmentSubjectExecution(Serializable):
    episode_id: str
    execution_kind: str
    subject_ref: str
    parent_episode_id: Optional[str]
    status: str
    work_item_id: Optional[str]


@dataclass(frozen=True)
class ScenarioEvaluationStarter(Serializable):
    id: str
    scenario_revision_id: str
    adapter_id: str
    primary_metric: str
    direction: str
    required_fields: tuple[str, ...] = ()
    minimum_records: int = 20
    notes: str = ""


@dataclass(frozen=True)
class ScenarioOutcomeProfile(Serializable):
    id: str
    scenario_revision_id: str
    version: str
    content_hash: str
    practical_margin: float
    evaluation_starters: tuple[ScenarioEvaluationStarter, ...]
    diagnostic_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingResourceProjection(Serializable):
    elapsed_seconds_low: Optional[float] = None
    elapsed_seconds_high: Optional[float] = None
    peak_memory_bytes: Optional[int] = None
    output_bytes_low: Optional[int] = None
    output_bytes_high: Optional[int] = None
    training_tokens: Optional[int] = None
    confidence: str = "unavailable"
    basis: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingOutcomeFinding(Serializable):
    id: str
    assessment_id: str
    ordinal: int
    category: str
    severity: str
    summary: str
    evidence: Dict[str, Any]
    why_it_matters: str
    safe_remedies: List[str]
    available_actions: List[str]
    content_hash: str
    created_at: str


@dataclass(frozen=True)
class TrainingOutcomeAssessment(Serializable):
    id: str
    proof_run_id: str
    scenario_revision_id: str
    profile_id: str
    status: str
    technical_status: str
    quality_status: str
    base_evaluation_id: Optional[str]
    candidate_evaluation_id: Optional[str]
    comparison_hash: Optional[str]
    resource_projection: Dict[str, Any]
    diagnostics: Dict[str, Any]
    summary: Dict[str, Any]
    content_hash: Optional[str]
    work_item_id: Optional[str]
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]
    stage: str = "completed"
    progress: Dict[str, Any] = field(default_factory=dict)
    request: Dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False


@dataclass(frozen=True)
class TrainingOutcomeDecision(Serializable):
    id: str
    assessment_id: Optional[str]
    proof_run_id: str
    decision: str
    reason: str
    full_run_id: Optional[str]
    context: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class StudyFactor(Serializable):
    name: str
    levels: tuple[Any, ...]


@dataclass(frozen=True)
class StudyArm(Serializable):
    id: str
    protocol_revision_id: str
    ordinal: int
    name: str
    is_control: bool
    factor_values: Dict[str, Any]
    launch_config: Dict[str, Any]
    content_hash: str


@dataclass(frozen=True)
class StudyAssignment(Serializable):
    id: str
    protocol_revision_id: str
    arm_id: str
    seed: int
    ordinal: int
    run_group_id: Optional[str]
    run_id: Optional[str]
    status: str
    created_at: str


@dataclass(frozen=True)
class PlannedContrast(Serializable):
    id: str
    name: str
    left_arm_id: str
    right_arm_id: str
    metric: str
    direction: str
    conclusion_kind: str
    practical_margin: float
    exploratory: bool = False


@dataclass(frozen=True)
class AdaptationStudy(Serializable):
    id: str
    name: str
    description: Optional[str]
    status: str
    latest_protocol_revision_id: Optional[str]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class AdaptationStudyProtocolRevision(Serializable):
    id: str
    study_id: str
    revision_number: int
    design_kind: str
    question: str
    definition: Dict[str, Any]
    content_hash: str
    created_at: str
    arms: tuple[StudyArm, ...] = ()
    assignments: tuple[StudyAssignment, ...] = ()
    contrasts: tuple[PlannedContrast, ...] = ()
    launch_status: str = "not_started"
    launch_progress: Dict[str, Any] = field(default_factory=dict)
    launch_work_item_id: Optional[str] = None
    launch_error: Optional[str] = None


@dataclass(frozen=True)
class StudyAnalysis(Serializable):
    id: str
    protocol_revision_id: str
    status: str
    analysis: Dict[str, Any]
    content_hash: str
    evidence_classification: str
    work_item_id: Optional[str]
    bundle_path: Optional[str]
    created_at: str
    completed_at: Optional[str]
    stage: str = "completed"
    progress: Dict[str, Any] = field(default_factory=dict)
    request: Dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False
    error: Optional[str] = None


@dataclass(frozen=True)
class StudyDeviation(Serializable):
    id: str
    protocol_revision_id: str
    reason: str
    change: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class StudyDecision(Serializable):
    id: str
    protocol_revision_id: str
    analysis_id: Optional[str]
    decision: str
    reason: str
    created_at: str


@dataclass(frozen=True)
class GroundingProfile(Serializable):
    id: str
    name: str
    description: Optional[str]
    latest_revision_id: Optional[str]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class GroundingProfileRevision(Serializable):
    id: str
    profile_id: str
    revision_number: int
    definition: Dict[str, Any]
    content_hash: str
    created_at: str


@dataclass(frozen=True)
class GroundingCoverage(Serializable):
    documents_total: int = 0
    documents_covered: int = 0
    spans_total: int = 0
    citations_valid: int = 0
    citations_invalid: int = 0
    source_concentration: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundingCitation(Serializable):
    id: str
    candidate_id: str
    ordinal: int
    document_id: str
    source_ref: str
    span_start: Optional[int]
    span_end: Optional[int]
    locator: Dict[str, Any]
    quoted_hash: str
    structural_valid: bool
    semantic_status: str
    evidence: Dict[str, Any]


@dataclass(frozen=True)
class GroundingVerification(Serializable):
    structural_valid: bool
    semantic_status: str
    reasons: List[str] = field(default_factory=list)
    verifier_trace: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundedCandidate(Serializable):
    id: str
    batch_id: str
    ordinal: int
    task_type: str
    status: str
    document_id: str
    source_ref: str
    source_hash: str
    prompt: Dict[str, Any]
    output: Dict[str, Any]
    verifier: Dict[str, Any]
    content_hash: str
    rejection_reason: Optional[str]
    created_at: str
    citations: tuple[GroundingCitation, ...] = ()


@dataclass(frozen=True)
class GroundedGenerationBatch(Serializable):
    id: str
    profile_revision_id: str
    source_version_id: Optional[str]
    extraction_id: Optional[str]
    status: str
    stage: str
    intended_destination: str
    request: Dict[str, Any]
    source_hash: str
    content_hash: Optional[str]
    candidate_count: int
    accepted_count: int
    rejected_count: int
    coverage: Dict[str, Any]
    work_item_id: Optional[str]
    bundle_path: Optional[str]
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]
    progress: Dict[str, Any] = field(default_factory=dict)
    resume_cursor: Dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False


@dataclass(frozen=True)
class SpecializedTaskDescriptor(Serializable):
    id: str
    label: str
    task_kind: str
    modality: str
    canonical_schema: str
    trainer_mode: str
    metrics: tuple[str, ...]
    available: bool
    unavailable_reason: Optional[str] = None


@dataclass(frozen=True)
class TaskLabelSchemaRevision(Serializable):
    id: str
    schema_id: str
    revision_number: int
    definition: Dict[str, Any]
    content_hash: str
    created_at: str


@dataclass(frozen=True)
class SpecializedTaskArtifactMetadata(Serializable):
    artifact_occurrence_id: str
    task_kind: str
    modality: str
    label_schema_revision_id: Optional[str]
    model_head_hash: str
    processor_hash: str
    loss_adapter: str
    loss_adapter_version: str
    retrieval_corpus_hash: Optional[str]
    metadata: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class ClassificationPrediction(Serializable):
    labels: List[str]
    scores: Dict[str, float]
    model_artifact_id: Optional[str] = None


@dataclass(frozen=True)
class EmbeddingResult(Serializable):
    embedding: List[float]
    dimensions: int
    model_artifact_id: Optional[str] = None


@dataclass(frozen=True)
class RerankResult(Serializable):
    items: List[Dict[str, Any]]
    model_artifact_id: Optional[str] = None


@dataclass(frozen=True)
class EnvironmentToolDescriptor(Serializable):
    id: str
    name: str
    ordinal: int
    definition: Dict[str, Any]
    implementation_hash: str


@dataclass(frozen=True)
class AgentEnvironment(Serializable):
    id: str
    name: str
    description: Optional[str]
    latest_revision_id: Optional[str]
    archived: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class AgentEnvironmentRevision(Serializable):
    id: str
    environment_id: str
    revision_number: int
    adapter_id: str
    adapter_version: str
    implementation_hash: str
    definition: Dict[str, Any]
    fixture_hash: str
    content_hash: str
    storage_path: str
    created_at: str
    tools: tuple[EnvironmentToolDescriptor, ...] = ()


@dataclass(frozen=True)
class EpisodeSuiteRevision(Serializable):
    id: str
    suite_id: str
    revision_number: int
    environment_revision_id: str
    definition: Dict[str, Any]
    content_hash: str
    created_at: str


@dataclass(frozen=True)
class EnvironmentStateSnapshot(Serializable):
    state: Dict[str, Any]
    content_hash: str
    path: Optional[str] = None


@dataclass(frozen=True)
class AgentEpisodeStep(Serializable):
    episode_id: str
    ordinal: int
    observation: Dict[str, Any]
    raw_output: Optional[str]
    action: Dict[str, Any]
    tool_call: Optional[Dict[str, Any]]
    tool_result: Optional[Dict[str, Any]]
    state_delta: Dict[str, Any]
    state_hash: str
    verifier: Dict[str, Any]
    latency_ms: Optional[float]
    error: Optional[str]
    created_at: str


@dataclass(frozen=True)
class AgentEpisode(Serializable):
    id: str
    suite_revision_id: str
    suite_item_id: str
    subject_type: str
    subject_ref: str
    subject_hash: str
    seed: int
    status: str
    terminal_reason: Optional[str]
    metrics: Dict[str, Any]
    initial_state_hash: str
    final_state_hash: Optional[str]
    snapshot_path: Optional[str]
    trace_hash: Optional[str]
    work_item_id: Optional[str]
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]
    stage: str = "completed"
    progress: Dict[str, Any] = field(default_factory=dict)
    request: Dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False
    parent_episode_id: Optional[str] = None


@dataclass(frozen=True)
class TrajectorySet(Serializable):
    id: str
    name: str
    latest_revision_id: Optional[str]
    created_at: str
    updated_at: str
    status: str = "ready"
    stage: str = "ready"
    progress: Dict[str, Any] = field(default_factory=dict)
    work_item_id: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class TrajectorySetRevision(Serializable):
    id: str
    trajectory_set_id: str
    revision_number: int
    content_hash: str
    storage_path: str
    row_count: int
    provenance: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class EnvironmentEvaluationComparison(Serializable):
    suite_revision_id: str
    base_subject_hash: str
    candidate_subject_hash: str
    counts: Dict[str, int]
    metric_deltas: Dict[str, Optional[float]]
    compatible: bool
    reasons: List[str] = field(default_factory=list)
