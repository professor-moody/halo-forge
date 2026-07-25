"""SQLite schema for the run database (Track F-G commit 1; F-J commit 1)."""

from __future__ import annotations

SCHEMA_VERSION = 23


# A single ``runs`` table with all the headline fields the UI filters /
# sorts on; the rest of the training_summary lives behind ``raw_json``.
#
# Indexes are added on the columns we actually filter / sort on:
# modality, status, timestamp, model_name. They cost ~10% on inserts
# and pay back orders of magnitude on the list view at any nontrivial
# row count.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    -- Filesystem id used by the existing ResultsService. Kept so we
    -- can round-trip through that surface during the migration.
    fs_id TEXT,
    modality TEXT NOT NULL,
    model_name TEXT NOT NULL,
    base_model_name TEXT,
    active_model_name TEXT,
    status TEXT NOT NULL,
    -- ISO 8601 timestamp; sortable as text since Python's datetime.isoformat
    -- emits zero-padded fields. Indexed for the "newest first" default sort.
    timestamp TEXT,
    output_dir TEXT NOT NULL,
    -- Headline metrics
    cycles_executed INTEGER DEFAULT 0,
    total_train_steps INTEGER DEFAULT 0,
    final_train_loss REAL,
    weights_updated INTEGER DEFAULT 0,
    final_update_reason TEXT,
    failure_reason TEXT,
    effectiveness_verdict TEXT,
    quality_status TEXT,
    keep_rate REAL,
    dominant_rejection_reason TEXT,
    final_model_path TEXT,
    seed INTEGER,
    -- Free-form JSON for everything else the UI might want to read on
    -- the detail page (yield_diagnostics, recovery, cycle_losses, etc).
    raw_json TEXT,
    -- Sync provenance — the mtime of the source training_summary.json
    -- when this row was last upserted. Lets sync skip unchanged rows.
    source_mtime REAL,
    indexed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_modality ON runs (modality);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs (timestamp);
CREATE INDEX IF NOT EXISTS idx_runs_model ON runs (model_name);
-- `sync_from_filesystem` probes `WHERE output_dir = ?` once per discovered
-- training_summary.json, on the runs-search request path. Without this index
-- that walk is a full table scan per file — quadratic in the run count.
CREATE INDEX IF NOT EXISTS idx_runs_output_dir ON runs (output_dir);

-- Track F-Q (run forking) reservation. Empty in commit 1.
-- Each row is "child run forked from parent at cycle N".
CREATE TABLE IF NOT EXISTS run_lineage (
    -- Managed launches allocate a canonical id before a filesystem summary is
    -- indexed, so lineage must be attachable while the child is still active.
    child_run_id TEXT NOT NULL,
    parent_run_id TEXT NOT NULL,
    forked_at_cycle INTEGER,
    notes TEXT,
    PRIMARY KEY (child_run_id, parent_run_id)
);

-- Track F-J — model registry. A registry entry is a named bundle of
-- runs the user wants to compare / promote / share as a unit. Most
-- payload fields are JSON blobs (run_ids, tags) to keep the schema
-- simple — the cohort dashboard reads them flat.
CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    base_model TEXT,
    run_ids TEXT NOT NULL DEFAULT '[]',  -- JSON array of run_id strings
    tags TEXT NOT NULL DEFAULT '[]',     -- JSON array of strings
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_registry_name ON model_registry (name);
CREATE INDEX IF NOT EXISTS idx_registry_base_model ON model_registry (base_model);

-- Dataset Lab v1. Raw sources remain external and are fingerprinted; immutable
-- derived versions point at their atomically-published managed directory.
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    modality TEXT NOT NULL,
    canonical_schema TEXT NOT NULL,
    latest_version_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets (name);
CREATE INDEX IF NOT EXISTS idx_datasets_modality ON datasets (modality);

CREATE TABLE IF NOT EXISTS dataset_sources (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    uri TEXT NOT NULL,
    config TEXT,
    split TEXT,
    revision TEXT,
    fingerprint TEXT NOT NULL,
    size_bytes INTEGER,
    row_count INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    refreshed_from_source_id TEXT REFERENCES dataset_sources(id),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dataset_sources_dataset ON dataset_sources (dataset_id);
CREATE INDEX IF NOT EXISTS idx_dataset_sources_fingerprint ON dataset_sources (fingerprint);

CREATE TABLE IF NOT EXISTS dataset_versions (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    source_id TEXT REFERENCES dataset_sources(id),
    parent_version_id TEXT REFERENCES dataset_versions(id),
    status TEXT NOT NULL,
    content_hash TEXT,
    recipe_hash TEXT NOT NULL,
    recipe_json TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    split_counts_json TEXT NOT NULL DEFAULT '{}',
    statistics_json TEXT NOT NULL DEFAULT '{}',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    source_fingerprints_json TEXT NOT NULL DEFAULT '{}',
    assets_materialized INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_dataset_versions_dataset ON dataset_versions (dataset_id);
CREATE INDEX IF NOT EXISTS idx_dataset_versions_status ON dataset_versions (status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_versions_identity
    ON dataset_versions (dataset_id, content_hash, recipe_hash)
    WHERE content_hash IS NOT NULL AND status = 'completed';

CREATE TABLE IF NOT EXISTS dataset_version_parents (
    version_id TEXT NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
    parent_version_id TEXT NOT NULL REFERENCES dataset_versions(id),
    role TEXT NOT NULL DEFAULT 'parent',
    weight REAL,
    PRIMARY KEY (version_id, parent_version_id, role)
);

CREATE INDEX IF NOT EXISTS idx_dataset_version_parents_parent
    ON dataset_version_parents (parent_version_id);

CREATE TABLE IF NOT EXISTS dataset_jobs (
    id TEXT PRIMARY KEY,
    dataset_id TEXT REFERENCES datasets(id) ON DELETE CASCADE,
    version_id TEXT REFERENCES dataset_versions(id) ON DELETE SET NULL,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'queued',
    processed_records INTEGER NOT NULL DEFAULT 0,
    total_records INTEGER,
    accepted_records INTEGER NOT NULL DEFAULT 0,
    rejected_records INTEGER NOT NULL DEFAULT 0,
    output_size_bytes INTEGER NOT NULL DEFAULT 0,
    logs_json TEXT NOT NULL DEFAULT '[]',
    request_json TEXT NOT NULL DEFAULT '{}',
    checkpoint_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    work_item_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_dataset_jobs_dataset ON dataset_jobs (dataset_id);
CREATE INDEX IF NOT EXISTS idx_dataset_jobs_status ON dataset_jobs (status);
CREATE INDEX IF NOT EXISTS idx_dataset_jobs_created ON dataset_jobs (created_at);

CREATE TABLE IF NOT EXISTS run_datasets (
    -- Training launch IDs are attached before the asynchronous run has
    -- necessarily been indexed in `runs`, so this is intentionally not an FK.
    run_id TEXT NOT NULL,
    dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(id),
    role TEXT NOT NULL DEFAULT 'train',
    split TEXT NOT NULL DEFAULT 'train',
    training_artifact_id TEXT REFERENCES training_artifacts(id) ON DELETE SET NULL,
    attached_at TEXT NOT NULL,
    PRIMARY KEY (run_id, dataset_version_id, split, role)
);

CREATE INDEX IF NOT EXISTS idx_run_datasets_version
    ON run_datasets (dataset_version_id);

-- Dataset Lab v2 trainer-ready, content-addressed artifact catalog. The
-- manifest itself remains on disk; SQLite stores the searchable identity and
-- the immutable role-to-version bindings used to render it.
CREATE TABLE IF NOT EXISTS training_artifacts (
    id TEXT PRIMARY KEY,
    artifact_hash TEXT NOT NULL UNIQUE,
    adapter_id TEXT NOT NULL,
    adapter_version TEXT NOT NULL,
    trainer_mode TEXT NOT NULL,
    model_id TEXT,
    tokenizer_revision TEXT,
    chat_template_hash TEXT,
    manifest_path TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_training_artifacts_adapter
    ON training_artifacts (adapter_id, adapter_version, trainer_mode);

CREATE TABLE IF NOT EXISTS training_artifact_bindings (
    artifact_id TEXT NOT NULL REFERENCES training_artifacts(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(id),
    split TEXT NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (artifact_id, role, dataset_version_id, split)
);

CREATE INDEX IF NOT EXISTS idx_training_artifact_bindings_version
    ON training_artifact_bindings (dataset_version_id);

-- Dataset Lab v2 benchmark definitions. Revisions are append-only; editing a
-- suite creates another revision and advances latest_revision_id.
CREATE TABLE IF NOT EXISTS benchmark_suites (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    latest_revision_id TEXT,
    purpose TEXT NOT NULL DEFAULT 'unspecified'
        CHECK (purpose IN ('development', 'holdout', 'unspecified')),
    purpose_v4 TEXT,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmark_suite_revisions (
    id TEXT PRIMARY KEY,
    suite_id TEXT NOT NULL REFERENCES benchmark_suites(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    items_json TEXT NOT NULL,
    generation_settings_json TEXT NOT NULL DEFAULT '{}',
    evaluator_versions_json TEXT NOT NULL DEFAULT '{}',
    primary_metric TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('maximize', 'minimize')),
    created_at TEXT NOT NULL,
    UNIQUE (suite_id, revision_number),
    UNIQUE (suite_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_benchmark_revisions_suite
    ON benchmark_suite_revisions (suite_id, revision_number DESC);

-- Evaluations double as persistent background-job records. The request and
-- result JSON preserve adapter-specific details while headline columns remain
-- queryable. A completed reuse_key is unique to prevent duplicate work.
CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    suite_revision_id TEXT NOT NULL REFERENCES benchmark_suite_revisions(id),
    adapter_id TEXT NOT NULL,
    adapter_version TEXT NOT NULL,
    subject_type TEXT NOT NULL,
    subject_ref TEXT NOT NULL,
    subject_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'queued',
    processed_samples INTEGER NOT NULL DEFAULT 0,
    total_samples INTEGER,
    request_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL DEFAULT '{}',
    logs_json TEXT NOT NULL DEFAULT '[]',
    artifact_path TEXT,
    reuse_key TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    work_item_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_evaluations_suite
    ON evaluations (suite_revision_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evaluations_status
    ON evaluations (status, created_at);
CREATE INDEX IF NOT EXISTS idx_evaluations_subject
    ON evaluations (subject_type, subject_ref, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evaluations_subject_hash
    ON evaluations (subject_hash);
CREATE UNIQUE INDEX IF NOT EXISTS idx_evaluations_completed_reuse
    ON evaluations (reuse_key) WHERE status = 'completed';

CREATE TABLE IF NOT EXISTS evaluation_metrics (
    evaluation_id TEXT NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('maximize', 'minimize')),
    suite_item_id TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (evaluation_id, name, suite_item_id)
);

CREATE INDEX IF NOT EXISTS idx_evaluation_metrics_name
    ON evaluation_metrics (name, value);

CREATE TABLE IF NOT EXISTS evaluation_samples (
    evaluation_id TEXT NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    suite_item_id TEXT NOT NULL,
    record_id TEXT,
    input_json TEXT,
    expected_json TEXT,
    output_json TEXT,
    score REAL,
    passed INTEGER,
    latency_ms REAL,
    error TEXT,
    verifier_trace_json TEXT,
    generation_seed INTEGER,
    evidence_kind TEXT NOT NULL DEFAULT 'legacy',
    valid INTEGER NOT NULL DEFAULT 0,
    mineable INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER,
    output_tokens INTEGER,
    finish_reason TEXT,
    template_hash TEXT,
    runtime_versions_json TEXT NOT NULL DEFAULT '{}',
    score_direction TEXT,
    score_threshold REAL,
    coverage REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (evaluation_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_evaluation_samples_record
    ON evaluation_samples (record_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_samples_item
    ON evaluation_samples (suite_item_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_samples_passed
    ON evaluation_samples (evaluation_id, passed);

-- Lab v3 durable workstation queue. Work is claimed transactionally in
-- priority/FIFO order; accelerator-heavy entries additionally hold the single
-- ``accelerator`` resource lease. Retained serving leases have no expiry and
-- can only be removed explicitly by their owner/operator.
CREATE TABLE IF NOT EXISTS work_items (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    stage TEXT NOT NULL DEFAULT 'queued',
    resource_class TEXT NOT NULL DEFAULT 'accelerator',
    priority INTEGER NOT NULL DEFAULT 0,
    launch_spec_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL DEFAULT '{}',
    progress_json TEXT NOT NULL DEFAULT '{}',
    resource_requirements_json TEXT NOT NULL DEFAULT '{}',
    domain_kind TEXT,
    domain_id TEXT,
    run_group_id TEXT,
    canonical_run_id TEXT,
    log_path TEXT,
    worker_id TEXT,
    worker_pid INTEGER,
    worker_pid_started_at REAL,
    claim_token TEXT,
    heartbeat_at TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    not_before TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_work_items_dispatch
    ON work_items (status, resource_class, priority DESC, created_at, id);
CREATE INDEX IF NOT EXISTS idx_work_items_run ON work_items (canonical_run_id);
CREATE INDEX IF NOT EXISTS idx_work_items_heartbeat
    ON work_items (status, heartbeat_at);
CREATE TABLE IF NOT EXISTS work_item_dependencies (
    work_item_id TEXT NOT NULL REFERENCES work_items(id) ON DELETE CASCADE,
    depends_on_work_item_id TEXT NOT NULL REFERENCES work_items(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    PRIMARY KEY (work_item_id, depends_on_work_item_id),
    CHECK (work_item_id <> depends_on_work_item_id)
);

CREATE INDEX IF NOT EXISTS idx_work_item_dependencies_parent
    ON work_item_dependencies (depends_on_work_item_id);

CREATE TABLE IF NOT EXISTS resource_leases (
    resource_key TEXT PRIMARY KEY,
    holder_type TEXT NOT NULL CHECK (holder_type IN ('work_item', 'serving')),
    holder_id TEXT NOT NULL,
    work_item_id TEXT REFERENCES work_items(id) ON DELETE CASCADE,
    lease_token TEXT NOT NULL UNIQUE,
    retained INTEGER NOT NULL DEFAULT 0,
    acquired_at TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL,
    expires_at TEXT,
    holder_pid INTEGER,
    holder_pid_started_at REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    CHECK (
        (holder_type = 'work_item' AND work_item_id IS NOT NULL AND retained = 0
            AND expires_at IS NOT NULL)
        OR
        (holder_type = 'serving' AND work_item_id IS NULL AND retained = 1
            AND expires_at IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_resource_leases_holder
    ON resource_leases (holder_type, holder_id);

-- A deliberately narrow repeat/sweep hierarchy. This is not a generic
-- experiment matrix: groups own ordered trials, trials own seeded runs, and
-- runs own checkpoint-gated segments.
CREATE TABLE IF NOT EXISTS run_groups (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('repeat', 'sweep')),
    status TEXT NOT NULL DEFAULT 'draft',
    trainer_mode TEXT NOT NULL,
    resolved_launch_config_json TEXT NOT NULL DEFAULT '{}',
    dataset_bindings_json TEXT NOT NULL DEFAULT '[]',
    base_subject_json TEXT NOT NULL DEFAULT '{}',
    development_suite_revision_id TEXT REFERENCES benchmark_suite_revisions(id),
    holdout_suite_revision_id TEXT REFERENCES benchmark_suite_revisions(id),
    search_space_json TEXT NOT NULL DEFAULT '{}',
    seeds_json TEXT NOT NULL DEFAULT '[]',
    budgets_json TEXT NOT NULL DEFAULT '{}',
    sampler_state_json TEXT NOT NULL DEFAULT '{}',
    pruning_policy_json TEXT NOT NULL DEFAULT '{}',
    checkpoint_policy_revision_id TEXT REFERENCES checkpoint_policy_revisions(id),
    resolved_checkpoint_plan_json TEXT NOT NULL DEFAULT '{}',
    parent_group_id TEXT REFERENCES run_groups(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_run_groups_status ON run_groups (status, created_at);

CREATE TABLE IF NOT EXISTS run_group_trials (
    id TEXT PRIMARY KEY,
    run_group_id TEXT NOT NULL REFERENCES run_groups(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    config_hash TEXT NOT NULL,
    sampled_config_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'queued',
    objective_metric TEXT,
    objective_direction TEXT CHECK (
        objective_direction IS NULL OR objective_direction IN ('maximize', 'minimize')
    ),
    objective_value REAL,
    seed_coverage INTEGER NOT NULL DEFAULT 0,
    required_seed_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (run_group_id, ordinal),
    UNIQUE (run_group_id, config_hash)
);

CREATE INDEX IF NOT EXISTS idx_run_group_trials_status
    ON run_group_trials (run_group_id, status, ordinal);

CREATE TABLE IF NOT EXISTS trial_runs (
    id TEXT PRIMARY KEY,
    trial_id TEXT NOT NULL REFERENCES run_group_trials(id) ON DELETE CASCADE,
    run_id TEXT NOT NULL UNIQUE,
    ordinal INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (trial_id, ordinal),
    UNIQUE (trial_id, seed)
);

CREATE INDEX IF NOT EXISTS idx_trial_runs_trial ON trial_runs (trial_id, ordinal);

CREATE TABLE IF NOT EXISTS trial_segments (
    id TEXT PRIMARY KEY,
    trial_run_id TEXT NOT NULL REFERENCES trial_runs(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    unit TEXT NOT NULL,
    start_value INTEGER NOT NULL,
    end_value INTEGER NOT NULL,
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    checkpoint_artifact_id TEXT,
    decision TEXT CHECK (
        decision IS NULL OR decision IN ('continue', 'pause', 'stop', 'prune', 'complete')
    ),
    decision_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    UNIQUE (trial_run_id, ordinal),
    CHECK (end_value > start_value)
);

CREATE INDEX IF NOT EXISTS idx_trial_segments_run
    ON trial_segments (trial_run_id, ordinal);

CREATE TABLE IF NOT EXISTS model_artifacts (
    id TEXT PRIMARY KEY,
    artifact_hash TEXT NOT NULL UNIQUE,
    artifact_kind TEXT NOT NULL CHECK (
        artifact_kind IN ('checkpoint', 'final_model', 'adapter')
    ),
    run_id TEXT NOT NULL,
    run_group_id TEXT REFERENCES run_groups(id) ON DELETE SET NULL,
    trial_id TEXT REFERENCES run_group_trials(id) ON DELETE SET NULL,
    trial_segment_id TEXT REFERENCES trial_segments(id) ON DELETE SET NULL,
    parent_artifact_id TEXT REFERENCES model_artifacts(id) ON DELETE SET NULL,
    model_id TEXT NOT NULL,
    tokenizer_revision TEXT,
    chat_template_hash TEXT,
    backend TEXT NOT NULL,
    format TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    step INTEGER,
    cycle INTEGER,
    verification_status TEXT NOT NULL DEFAULT 'unverified',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_model_artifacts_run
    ON model_artifacts (run_id, artifact_kind, created_at);
CREATE INDEX IF NOT EXISTS idx_model_artifacts_group
    ON model_artifacts (run_group_id, trial_id);

-- Append-only record-level exposure provenance. Nullable target columns let an
-- exposure be recorded early and inherited as dataset/run/artifact identities
-- become available without mutating the source event.
CREATE TABLE IF NOT EXISTS exposure_ledger (
    id TEXT PRIMARY KEY,
    suite_revision_id TEXT NOT NULL REFERENCES benchmark_suite_revisions(id),
    suite_item_id TEXT NOT NULL,
    exposure_type TEXT NOT NULL,
    dataset_version_id TEXT REFERENCES dataset_versions(id) ON DELETE SET NULL,
    run_group_id TEXT REFERENCES run_groups(id) ON DELETE SET NULL,
    run_id TEXT,
    model_artifact_id TEXT REFERENCES model_artifacts(id) ON DELETE SET NULL,
    inherited_from_id TEXT REFERENCES exposure_ledger(id) ON DELETE SET NULL,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_exposure_suite_item
    ON exposure_ledger (suite_revision_id, suite_item_id);
CREATE INDEX IF NOT EXISTS idx_exposure_dataset
    ON exposure_ledger (dataset_version_id);
CREATE INDEX IF NOT EXISTS idx_exposure_run
    ON exposure_ledger (run_id);
CREATE INDEX IF NOT EXISTS idx_exposure_artifact
    ON exposure_ledger (model_artifact_id);

-- Lab v4 workstation control plane. Attempts and events are append-only so
-- retries never erase the history needed to diagnose or reproduce a launch.
CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    pid INTEGER,
    pid_started_at REAL,
    version TEXT,
    capabilities_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    started_at TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL,
    stopped_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_workers_status
    ON workers (status, heartbeat_at);

CREATE TABLE IF NOT EXISTS work_item_attempts (
    id TEXT PRIMARY KEY,
    work_item_id TEXT NOT NULL REFERENCES work_items(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    status TEXT NOT NULL,
    worker_id TEXT REFERENCES workers(id) ON DELETE SET NULL,
    worker_pid INTEGER,
    worker_pid_started_at REAL,
    claim_token TEXT,
    output_dir TEXT,
    result_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    UNIQUE (work_item_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_work_item_attempts_work
    ON work_item_attempts (work_item_id, ordinal DESC);

CREATE TABLE IF NOT EXISTS work_item_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    work_item_id TEXT NOT NULL REFERENCES work_items(id) ON DELETE CASCADE,
    attempt_id TEXT REFERENCES work_item_attempts(id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_work_item_events_work
    ON work_item_events (work_item_id, sequence);
CREATE INDEX IF NOT EXISTS idx_work_item_events_sequence
    ON work_item_events (sequence);

CREATE TABLE IF NOT EXISTS telemetry_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    work_item_id TEXT NOT NULL REFERENCES work_items(id) ON DELETE CASCADE,
    attempt_id TEXT REFERENCES work_item_attempts(id) ON DELETE SET NULL,
    sampled_at TEXT NOT NULL,
    cpu_percent REAL,
    process_rss_bytes INTEGER,
    system_memory_used_bytes INTEGER,
    system_memory_total_bytes INTEGER,
    gpu_percent REAL,
    device_memory_used_bytes INTEGER,
    device_memory_total_bytes INTEGER,
    power_watts REAL,
    temperature_c REAL,
    throughput_tokens_per_second REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_telemetry_samples_work
    ON telemetry_samples (work_item_id, sampled_at);

CREATE TABLE IF NOT EXISTS telemetry_rollups (
    work_item_id TEXT NOT NULL REFERENCES work_items(id) ON DELETE CASCADE,
    attempt_id TEXT REFERENCES work_item_attempts(id) ON DELETE SET NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    ended_at TEXT,
    aggregates_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (work_item_id, attempt_id)
);

-- Content identity is distinct from occurrences. Multiple runs may point at
-- the same immutable bytes without stealing one another's provenance.
CREATE TABLE IF NOT EXISTS artifact_blobs (
    id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    format TEXT NOT NULL,
    dtype TEXT,
    quantization TEXT,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    integrity_state TEXT NOT NULL DEFAULT 'unverified',
    manifest_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    last_verified_at TEXT
);

CREATE TABLE IF NOT EXISTS artifact_locations (
    id TEXT PRIMARY KEY,
    blob_id TEXT NOT NULL REFERENCES artifact_blobs(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    storage_mode TEXT NOT NULL CHECK (storage_mode IN ('referenced', 'managed', 'trash')),
    state TEXT NOT NULL DEFAULT 'available',
    size_bytes INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    last_verified_at TEXT,
    trash_expires_at TEXT,
    UNIQUE (blob_id, path)
);

CREATE INDEX IF NOT EXISTS idx_artifact_locations_blob
    ON artifact_locations (blob_id, state, storage_mode);

CREATE TABLE IF NOT EXISTS artifact_occurrences (
    id TEXT PRIMARY KEY,
    blob_id TEXT NOT NULL REFERENCES artifact_blobs(id),
    artifact_kind TEXT NOT NULL CHECK (artifact_kind IN (
        'checkpoint', 'adapter', 'final_model', 'merged_model',
        'converted_model', 'quantized_model', 'export_bundle'
    )),
    legacy_model_artifact_id TEXT,
    run_id TEXT,
    run_group_id TEXT REFERENCES run_groups(id) ON DELETE SET NULL,
    trial_id TEXT REFERENCES run_group_trials(id) ON DELETE SET NULL,
    trial_segment_id TEXT REFERENCES trial_segments(id) ON DELETE SET NULL,
    model_id TEXT NOT NULL,
    tokenizer_revision TEXT,
    chat_template_hash TEXT,
    backend TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0,
    tags_json TEXT NOT NULL DEFAULT '[]',
    notes TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_artifact_occurrences_blob
    ON artifact_occurrences (blob_id, created_at);
CREATE INDEX IF NOT EXISTS idx_artifact_occurrences_run
    ON artifact_occurrences (run_id, artifact_kind, created_at);

CREATE TABLE IF NOT EXISTS artifact_operations (
    id TEXT PRIMARY KEY,
    operation_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    operation_hash TEXT NOT NULL,
    resolved_spec_json TEXT NOT NULL DEFAULT '{}',
    input_occurrences_json TEXT NOT NULL DEFAULT '[]',
    output_occurrence_id TEXT REFERENCES artifact_occurrences(id) ON DELETE SET NULL,
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    logs_json TEXT NOT NULL DEFAULT '[]',
    result_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_artifact_operations_status
    ON artifact_operations (status, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_artifact_operations_completed_reuse
    ON artifact_operations (operation_hash) WHERE status = 'completed';

CREATE TABLE IF NOT EXISTS artifact_edges (
    child_occurrence_id TEXT NOT NULL REFERENCES artifact_occurrences(id) ON DELETE CASCADE,
    parent_occurrence_id TEXT NOT NULL REFERENCES artifact_occurrences(id),
    relation TEXT NOT NULL,
    ordinal INTEGER NOT NULL DEFAULT 0,
    operation_id TEXT REFERENCES artifact_operations(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (child_occurrence_id, parent_occurrence_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_artifact_edges_parent
    ON artifact_edges (parent_occurrence_id, relation);

CREATE TABLE IF NOT EXISTS artifact_aliases (
    alias TEXT PRIMARY KEY,
    occurrence_id TEXT NOT NULL REFERENCES artifact_occurrences(id),
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_alias_events (
    id TEXT PRIMARY KEY,
    alias TEXT NOT NULL,
    previous_occurrence_id TEXT REFERENCES artifact_occurrences(id),
    occurrence_id TEXT NOT NULL REFERENCES artifact_occurrences(id),
    override_reason TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_artifact_alias_events_alias
    ON artifact_alias_events (alias, created_at);

CREATE TABLE IF NOT EXISTS qualification_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    latest_revision_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS qualification_profile_revisions (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL REFERENCES qualification_profiles(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    quality_suite_revision_id TEXT NOT NULL REFERENCES benchmark_suite_revisions(id),
    operational_suite_revision_id TEXT NOT NULL REFERENCES benchmark_suite_revisions(id),
    holdout_suite_revision_id TEXT REFERENCES benchmark_suite_revisions(id),
    thresholds_json TEXT NOT NULL DEFAULT '[]',
    target_backend TEXT,
    generation_settings_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (profile_id, revision_number),
    UNIQUE (profile_id, content_hash)
);

CREATE TABLE IF NOT EXISTS artifact_qualifications (
    id TEXT PRIMARY KEY,
    profile_revision_id TEXT NOT NULL REFERENCES qualification_profile_revisions(id),
    occurrence_id TEXT NOT NULL REFERENCES artifact_occurrences(id),
    parent_occurrence_id TEXT REFERENCES artifact_occurrences(id),
    status TEXT NOT NULL DEFAULT 'queued',
    decision TEXT CHECK (decision IS NULL OR decision IN ('pass', 'warn', 'fail')),
    reasons_json TEXT NOT NULL DEFAULT '[]',
    quality_evaluation_id TEXT REFERENCES evaluations(id) ON DELETE SET NULL,
    performance_evaluation_id TEXT REFERENCES evaluations(id) ON DELETE SET NULL,
    holdout_evaluation_id TEXT REFERENCES evaluations(id) ON DELETE SET NULL,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_artifact_qualifications_occurrence
    ON artifact_qualifications (occurrence_id, created_at);

CREATE TABLE IF NOT EXISTS serving_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    latest_revision_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS serving_profile_revisions (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL REFERENCES serving_profiles(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    occurrence_id TEXT NOT NULL REFERENCES artifact_occurrences(id),
    backend TEXT NOT NULL,
    endpoint_settings_json TEXT NOT NULL DEFAULT '{}',
    generation_settings_json TEXT NOT NULL DEFAULT '{}',
    resource_requirements_json TEXT NOT NULL DEFAULT '{}',
    chat_template_hash TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (profile_id, revision_number),
    UNIQUE (profile_id, content_hash)
);

CREATE TABLE IF NOT EXISTS cleanup_plans (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'preview',
    request_json TEXT NOT NULL DEFAULT '{}',
    entries_json TEXT NOT NULL DEFAULT '[]',
    reclaimed_bytes INTEGER NOT NULL DEFAULT 0,
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    reviewed_at TEXT,
    applied_at TEXT
);

CREATE TABLE IF NOT EXISTS playground_sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    primary_occurrence_id TEXT REFERENCES artifact_occurrences(id),
    comparison_occurrence_id TEXT REFERENCES artifact_occurrences(id),
    settings_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS playground_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES playground_sessions(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    occurrence_id TEXT REFERENCES artifact_occurrences(id),
    generation_json TEXT NOT NULL DEFAULT '{}',
    evidence_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (session_id, ordinal)
);

-- Lab v5 adaptive checkpoint and research-evidence records. Policy revisions,
-- gate decisions, cohort snapshots, and research decisions are append-only.
-- Evidence bundles have a mutable publication lifecycle, while their resolved
-- request and published manifest remain content-addressed.
CREATE TABLE IF NOT EXISTS checkpoint_policies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    latest_revision_id TEXT,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoint_policy_revisions (
    id TEXT PRIMARY KEY,
    policy_id TEXT NOT NULL REFERENCES checkpoint_policies(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    development_suite_revision_id TEXT NOT NULL
        REFERENCES benchmark_suite_revisions(id),
    primary_metric TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('maximize', 'minimize')),
    definition_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (policy_id, revision_number),
    UNIQUE (policy_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_checkpoint_policy_revisions_policy
    ON checkpoint_policy_revisions (policy_id, revision_number DESC);

CREATE TABLE IF NOT EXISTS checkpoint_gate_decisions (
    id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    policy_revision_id TEXT NOT NULL REFERENCES checkpoint_policy_revisions(id),
    plan_hash TEXT NOT NULL,
    run_group_id TEXT REFERENCES run_groups(id) ON DELETE SET NULL,
    trial_run_id TEXT REFERENCES trial_runs(id) ON DELETE SET NULL,
    trial_segment_id TEXT REFERENCES trial_segments(id) ON DELETE SET NULL,
    checkpoint_occurrence_id TEXT REFERENCES artifact_occurrences(id) ON DELETE SET NULL,
    boundary_index INTEGER NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('continue', 'pause', 'stop')),
    automatic INTEGER NOT NULL DEFAULT 0,
    reasons_json TEXT NOT NULL DEFAULT '[]',
    evidence_json TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT NOT NULL,
    override_of_id TEXT REFERENCES checkpoint_gate_decisions(id),
    override_reason TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_checkpoint_gate_decisions_run
    ON checkpoint_gate_decisions (run_group_id, trial_run_id, boundary_index);
CREATE INDEX IF NOT EXISTS idx_checkpoint_gate_decisions_segment
    ON checkpoint_gate_decisions (trial_segment_id, created_at);

CREATE TABLE IF NOT EXISTS cohort_analysis_snapshots (
    id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    run_group_id TEXT REFERENCES run_groups(id) ON DELETE SET NULL,
    baseline_subject_id TEXT,
    primary_metric TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('maximize', 'minimize')),
    status TEXT NOT NULL DEFAULT 'completed',
    request_json TEXT NOT NULL,
    analysis_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cohort_analysis_group
    ON cohort_analysis_snapshots (run_group_id, created_at DESC);

CREATE TABLE IF NOT EXISTS research_decisions (
    id TEXT PRIMARY KEY,
    analysis_snapshot_id TEXT NOT NULL
        REFERENCES cohort_analysis_snapshots(id),
    selected_subject_json TEXT NOT NULL,
    rejected_subjects_json TEXT NOT NULL DEFAULT '[]',
    exclusions_json TEXT NOT NULL DEFAULT '[]',
    rationale TEXT NOT NULL,
    override_reason TEXT,
    fork_spec_json TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_research_decisions_analysis
    ON research_decisions (analysis_snapshot_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_decisions_hash
    ON research_decisions (content_hash);

CREATE TABLE IF NOT EXISTS evidence_bundles (
    id TEXT PRIMARY KEY,
    analysis_snapshot_id TEXT NOT NULL
        REFERENCES cohort_analysis_snapshots(id),
    research_decision_id TEXT REFERENCES research_decisions(id),
    status TEXT NOT NULL DEFAULT 'queued',
    content_hash TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    request_json TEXT NOT NULL DEFAULT '{}',
    manifest_json TEXT NOT NULL DEFAULT '{}',
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_evidence_bundles_status
    ON evidence_bundles (status, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_bundles_completed_hash
    ON evidence_bundles (content_hash) WHERE status = 'completed';

CREATE TABLE IF NOT EXISTS workspace_drafts (
    id TEXT PRIMARY KEY,
    owner_key TEXT NOT NULL DEFAULT 'local',
    draft_kind TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT 'default',
    content_json TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (owner_key, draft_kind, name)
);

CREATE INDEX IF NOT EXISTS idx_workspace_drafts_expiry
    ON workspace_drafts (expires_at);

-- Halo Forge Lab v6: reviewed human-feedback and active-data workflow.
-- Schema revisions, acquisition batches, review events, and label-set revisions
-- are immutable.  Mutable tables only hold discoverability/lifecycle pointers;
-- review state can always be rebuilt from the append-only event stream.
CREATE TABLE IF NOT EXISTS annotation_schemas (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_annotation_schemas_name
    ON annotation_schemas (name);

CREATE TABLE IF NOT EXISTS annotation_schema_revisions (
    id TEXT PRIMARY KEY,
    schema_id TEXT NOT NULL REFERENCES annotation_schemas(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    modality TEXT NOT NULL,
    task_type TEXT NOT NULL,
    definition_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (schema_id, revision_number),
    UNIQUE (schema_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_annotation_schema_revisions_schema
    ON annotation_schema_revisions (schema_id, revision_number);

CREATE TABLE IF NOT EXISTS acquisition_batches (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ready',
    stage TEXT NOT NULL DEFAULT 'complete',
    request_json TEXT NOT NULL DEFAULT '{}',
    source_hash TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    seed INTEGER NOT NULL DEFAULT 0,
    row_count INTEGER NOT NULL DEFAULT 0,
    processed_records INTEGER NOT NULL DEFAULT 0,
    total_records INTEGER,
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    error TEXT,
    eligibility_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_acquisition_batches_status
    ON acquisition_batches (status, created_at);

CREATE TABLE IF NOT EXISTS acquisition_candidates (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES acquisition_batches(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    record_id TEXT NOT NULL,
    record_hash TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_ref TEXT,
    source_record_id TEXT,
    record_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    source_json TEXT NOT NULL DEFAULT '{}',
    stratum TEXT NOT NULL DEFAULT 'explicit',
    score REAL,
    created_at TEXT NOT NULL,
    UNIQUE (batch_id, ordinal),
    UNIQUE (batch_id, record_id)
);

CREATE INDEX IF NOT EXISTS idx_acquisition_candidates_record
    ON acquisition_candidates (record_id, record_hash);
CREATE INDEX IF NOT EXISTS idx_acquisition_candidates_batch
    ON acquisition_candidates (batch_id, ordinal);

CREATE TABLE IF NOT EXISTS review_queues (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    acquisition_batch_id TEXT NOT NULL
        REFERENCES acquisition_batches(id),
    schema_revision_id TEXT NOT NULL
        REFERENCES annotation_schema_revisions(id),
    policy_json TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT NOT NULL UNIQUE,
    current_pass INTEGER NOT NULL DEFAULT 1,
    latest_label_set_revision_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_review_queues_status
    ON review_queues (status, updated_at);

CREATE TABLE IF NOT EXISTS review_items (
    id TEXT PRIMARY KEY,
    queue_id TEXT NOT NULL REFERENCES review_queues(id) ON DELETE CASCADE,
    candidate_id TEXT NOT NULL REFERENCES acquisition_candidates(id),
    ordinal INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    active_event_id TEXT,
    projection_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (queue_id, candidate_id),
    UNIQUE (queue_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_review_items_queue_status
    ON review_items (queue_id, status, ordinal);

CREATE TABLE IF NOT EXISTS review_events (
    id TEXT PRIMARY KEY,
    queue_id TEXT NOT NULL REFERENCES review_queues(id) ON DELETE CASCADE,
    item_id TEXT NOT NULL REFERENCES review_items(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    pass_number INTEGER NOT NULL DEFAULT 1,
    reviewer_key TEXT NOT NULL DEFAULT 'local',
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    expected_active_event_id TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    supersedes_event_id TEXT REFERENCES review_events(id),
    created_at TEXT NOT NULL,
    UNIQUE (queue_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_review_events_item
    ON review_events (item_id, created_at, id);
CREATE INDEX IF NOT EXISTS idx_review_events_item_pass_page
    ON review_events (item_id, pass_number, created_at, id);

CREATE TABLE IF NOT EXISTS review_suggestions (
    id TEXT PRIMARY KEY,
    item_id TEXT NOT NULL REFERENCES review_items(id) ON DELETE CASCADE,
    pass_number INTEGER NOT NULL DEFAULT 1,
    provider TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    output_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (item_id, pass_number, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_review_suggestions_item
    ON review_suggestions (item_id, pass_number, created_at);
CREATE INDEX IF NOT EXISTS idx_review_suggestions_item_page
    ON review_suggestions (item_id, pass_number, created_at, id);

CREATE TABLE IF NOT EXISTS label_sets (
    id TEXT PRIMARY KEY,
    queue_id TEXT NOT NULL REFERENCES review_queues(id),
    name TEXT NOT NULL,
    latest_revision_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (queue_id, name)
);

CREATE TABLE IF NOT EXISTS label_set_revisions (
    id TEXT PRIMARY KEY,
    label_set_id TEXT NOT NULL REFERENCES label_sets(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    excluded_count INTEGER NOT NULL DEFAULT 0,
    manifest_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (label_set_id, revision_number),
    UNIQUE (label_set_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_label_set_revisions_set
    ON label_set_revisions (label_set_id, revision_number);

CREATE TABLE IF NOT EXISTS label_set_items (
    revision_id TEXT NOT NULL
        REFERENCES label_set_revisions(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    review_item_id TEXT NOT NULL REFERENCES review_items(id),
    record_id TEXT NOT NULL,
    record_hash TEXT NOT NULL,
    annotation_json TEXT NOT NULL DEFAULT '{}',
    output_records_json TEXT NOT NULL DEFAULT '[]',
    lineage_json TEXT NOT NULL DEFAULT '{}',
    excluded INTEGER NOT NULL DEFAULT 0,
    exclusion_reason TEXT,
    PRIMARY KEY (revision_id, ordinal),
    UNIQUE (revision_id, review_item_id)
);

CREATE INDEX IF NOT EXISTS idx_label_set_items_record
    ON label_set_items (record_id, record_hash);

-- Halo Forge Lab v7: verifier reliability and reward calibration.  Profile,
-- protocol, and policy identities are split from their immutable revisions so
-- names can remain stable while every research input stays reproducible.
CREATE TABLE IF NOT EXISTS verifier_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    latest_revision_id TEXT,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_verifier_profiles_name
    ON verifier_profiles (name);

CREATE TABLE IF NOT EXISTS verifier_profile_revisions (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL REFERENCES verifier_profiles(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    family TEXT NOT NULL CHECK (
        family IN ('deterministic', 'llm_judge', 'reward_model', 'chain')
    ),
    reliability_adapter_id TEXT NOT NULL,
    reliability_adapter_version TEXT NOT NULL,
    implementation_kind TEXT NOT NULL,
    implementation_ref TEXT NOT NULL,
    implementation_fingerprint TEXT,
    qualifiable INTEGER NOT NULL DEFAULT 1,
    qualification_blockers_json TEXT NOT NULL DEFAULT '[]',
    modality TEXT NOT NULL,
    task_type TEXT NOT NULL,
    input_mapping_json TEXT NOT NULL DEFAULT '{}',
    output_contract_json TEXT NOT NULL DEFAULT '{}',
    reward_min REAL NOT NULL,
    reward_max REAL NOT NULL,
    reward_direction TEXT NOT NULL CHECK (
        reward_direction IN ('maximize', 'minimize')
    ),
    threshold REAL,
    tie_policy TEXT NOT NULL DEFAULT 'error',
    error_behavior TEXT NOT NULL DEFAULT 'fail_closed',
    definition_json TEXT NOT NULL DEFAULT '{}',
    sanitized_configuration_hash TEXT NOT NULL,
    runtime_contract_json TEXT NOT NULL DEFAULT '{}',
    runtime_contract_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (profile_id, revision_number),
    UNIQUE (profile_id, content_hash),
    CHECK (reward_max > reward_min),
    CHECK (threshold IS NULL OR (threshold >= reward_min AND threshold <= reward_max))
);

CREATE INDEX IF NOT EXISTS idx_verifier_profile_revisions_profile
    ON verifier_profile_revisions (profile_id, revision_number DESC);
CREATE INDEX IF NOT EXISTS idx_verifier_profile_revisions_capability
    ON verifier_profile_revisions (family, modality, task_type, qualifiable);
CREATE INDEX IF NOT EXISTS idx_verifier_profile_revisions_fingerprint
    ON verifier_profile_revisions (implementation_fingerprint);

CREATE TABLE IF NOT EXISTS verifier_revision_components (
    revision_id TEXT NOT NULL
        REFERENCES verifier_profile_revisions(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    child_revision_id TEXT NOT NULL
        REFERENCES verifier_profile_revisions(id),
    weight REAL NOT NULL DEFAULT 1.0,
    veto INTEGER NOT NULL DEFAULT 0,
    required INTEGER NOT NULL DEFAULT 1,
    configuration_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (revision_id, ordinal),
    UNIQUE (revision_id, child_revision_id),
    CHECK (ordinal >= 0),
    CHECK (weight >= 0),
    CHECK (revision_id <> child_revision_id)
);

CREATE INDEX IF NOT EXISTS idx_verifier_revision_components_child
    ON verifier_revision_components (child_revision_id, revision_id);

CREATE TABLE IF NOT EXISTS verifier_calibration_protocols (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    latest_revision_id TEXT,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verifier_calibration_protocol_revisions (
    id TEXT PRIMARY KEY,
    protocol_id TEXT NOT NULL
        REFERENCES verifier_calibration_protocols(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    definition_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (protocol_id, revision_number),
    UNIQUE (protocol_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_verifier_protocol_revisions_protocol
    ON verifier_calibration_protocol_revisions (protocol_id, revision_number DESC);

CREATE TABLE IF NOT EXISTS verifier_qualification_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    latest_revision_id TEXT,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verifier_qualification_profile_revisions (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL
        REFERENCES verifier_qualification_profiles(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    template_kind TEXT NOT NULL CHECK (
        template_kind IN ('strict_oracle', 'human_aligned', 'exploratory', 'custom')
    ),
    promotable INTEGER NOT NULL DEFAULT 1,
    requirements_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (profile_id, revision_number),
    UNIQUE (profile_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_verifier_qualification_revisions_profile
    ON verifier_qualification_profile_revisions (profile_id, revision_number DESC);

-- Calibrations are durable job/domain records.  Their request identity remains
-- immutable while lifecycle/progress columns are updated by the scheduler.
CREATE TABLE IF NOT EXISTS verifier_calibrations (
    id TEXT PRIMARY KEY,
    verifier_revision_id TEXT NOT NULL
        REFERENCES verifier_profile_revisions(id),
    protocol_revision_id TEXT NOT NULL
        REFERENCES verifier_calibration_protocol_revisions(id),
    qualification_profile_revision_id TEXT NOT NULL
        REFERENCES verifier_qualification_profile_revisions(id),
    source_kind TEXT NOT NULL CHECK (
        source_kind IN ('label_set', 'benchmark_suite')
    ),
    source_revision_id TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    source_purpose TEXT NOT NULL DEFAULT 'unspecified',
    status TEXT NOT NULL DEFAULT 'queued',
    stage TEXT NOT NULL DEFAULT 'queued',
    processed_records INTEGER NOT NULL DEFAULT 0,
    total_records INTEGER,
    sample_count INTEGER NOT NULL DEFAULT 0,
    request_json TEXT NOT NULL DEFAULT '{}',
    partition_json TEXT NOT NULL DEFAULT '{}',
    runtime_identity_json TEXT NOT NULL DEFAULT '{}',
    runtime_identity_hash TEXT NOT NULL,
    protocol_hash TEXT NOT NULL,
    qualification_hash TEXT NOT NULL,
    reuse_key TEXT NOT NULL,
    artifact_path TEXT,
    manifest_hash TEXT,
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_verifier_calibrations_revision
    ON verifier_calibrations (verifier_revision_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_verifier_calibrations_status
    ON verifier_calibrations (status, created_at);
CREATE INDEX IF NOT EXISTS idx_verifier_calibrations_source
    ON verifier_calibrations (source_kind, source_revision_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_verifier_calibrations_completed_reuse
    ON verifier_calibrations (reuse_key) WHERE status = 'completed';

CREATE TABLE IF NOT EXISTS verifier_calibration_samples (
    calibration_id TEXT NOT NULL
        REFERENCES verifier_calibrations(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    record_id TEXT NOT NULL,
    record_hash TEXT NOT NULL,
    group_id TEXT NOT NULL,
    partition TEXT NOT NULL CHECK (
        partition IN ('calibration', 'confirmation')
    ),
    repeat_index INTEGER NOT NULL DEFAULT 0,
    orientation TEXT NOT NULL DEFAULT 'canonical',
    probe_kind TEXT NOT NULL DEFAULT 'base',
    seed INTEGER,
    reference_json TEXT NOT NULL DEFAULT '{}',
    observation_json TEXT NOT NULL DEFAULT '{}',
    reward REAL,
    passed INTEGER,
    latency_ms REAL,
    error TEXT,
    runtime_identity_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (calibration_id, ordinal),
    UNIQUE (
        calibration_id, record_id, repeat_index, orientation, probe_kind
    ),
    CHECK (ordinal >= 0),
    CHECK (repeat_index >= 0)
);

CREATE INDEX IF NOT EXISTS idx_verifier_calibration_samples_record
    ON verifier_calibration_samples (calibration_id, record_id, partition);
CREATE INDEX IF NOT EXISTS idx_verifier_calibration_samples_group
    ON verifier_calibration_samples (calibration_id, group_id, partition);
CREATE INDEX IF NOT EXISTS idx_verifier_calibration_samples_error
    ON verifier_calibration_samples (calibration_id, error);
CREATE INDEX IF NOT EXISTS idx_verifier_calibration_samples_outcome
    ON verifier_calibration_samples (calibration_id, partition, passed, ordinal);

CREATE TABLE IF NOT EXISTS verifier_calibration_metrics (
    calibration_id TEXT NOT NULL
        REFERENCES verifier_calibrations(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    partition TEXT NOT NULL DEFAULT 'calibration',
    subgroup TEXT NOT NULL DEFAULT '',
    value REAL,
    ci_low REAL,
    ci_high REAL,
    direction TEXT CHECK (
        direction IS NULL OR direction IN ('maximize', 'minimize')
    ),
    available INTEGER NOT NULL DEFAULT 1,
    missing_reason TEXT,
    record_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (calibration_id, name, partition, subgroup),
    CHECK (
        (available = 1 AND value IS NOT NULL AND missing_reason IS NULL)
        OR (available = 0 AND value IS NULL AND missing_reason IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_verifier_calibration_metrics_name
    ON verifier_calibration_metrics (name, partition, value);

-- Decisions, alias events, and usage bindings are append-only evidence.  A
-- later operator action writes a new row referencing the superseded decision.
CREATE TABLE IF NOT EXISTS verifier_qualification_decisions (
    id TEXT PRIMARY KEY,
    calibration_id TEXT NOT NULL
        REFERENCES verifier_calibrations(id),
    qualification_profile_revision_id TEXT NOT NULL
        REFERENCES verifier_qualification_profile_revisions(id),
    scope TEXT NOT NULL CHECK (
        scope IN ('development', 'operational', 'confirmation')
    ),
    decision TEXT NOT NULL CHECK (decision IN ('pass', 'warn', 'fail')),
    runtime_state TEXT NOT NULL DEFAULT 'compatible' CHECK (
        runtime_state IN ('compatible', 'stale_runtime', 'unavailable')
    ),
    reasons_json TEXT NOT NULL DEFAULT '[]',
    evidence_json TEXT NOT NULL DEFAULT '{}',
    override INTEGER NOT NULL DEFAULT 0,
    override_note TEXT,
    supersedes_decision_id TEXT REFERENCES verifier_qualification_decisions(id),
    created_at TEXT NOT NULL,
    CHECK (override = 0 OR (override_note IS NOT NULL AND length(trim(override_note)) > 0))
);

CREATE INDEX IF NOT EXISTS idx_verifier_qualification_decisions_calibration
    ON verifier_qualification_decisions (calibration_id, scope, created_at DESC);

CREATE TABLE IF NOT EXISTS verifier_aliases (
    profile_id TEXT NOT NULL REFERENCES verifier_profiles(id) ON DELETE CASCADE,
    alias TEXT NOT NULL CHECK (alias IN ('candidate', 'approved')),
    revision_id TEXT NOT NULL REFERENCES verifier_profile_revisions(id),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, alias)
);

CREATE TABLE IF NOT EXISTS verifier_alias_events (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL REFERENCES verifier_profiles(id),
    alias TEXT NOT NULL CHECK (alias IN ('candidate', 'approved')),
    previous_revision_id TEXT REFERENCES verifier_profile_revisions(id),
    revision_id TEXT NOT NULL REFERENCES verifier_profile_revisions(id),
    qualification_decision_id TEXT
        REFERENCES verifier_qualification_decisions(id),
    override INTEGER NOT NULL DEFAULT 0,
    note TEXT,
    created_at TEXT NOT NULL,
    CHECK (override = 0 OR (note IS NOT NULL AND length(trim(note)) > 0))
);

CREATE INDEX IF NOT EXISTS idx_verifier_alias_events_profile
    ON verifier_alias_events (profile_id, alias, created_at DESC);

CREATE TABLE IF NOT EXISTS verifier_bindings (
    id TEXT PRIMARY KEY,
    verifier_revision_id TEXT NOT NULL
        REFERENCES verifier_profile_revisions(id),
    domain_kind TEXT NOT NULL CHECK (
        domain_kind IN (
            'dataset', 'dataset_version', 'run', 'evaluation', 'suggestion',
            'dataset_output', 'review_suggestion', 'evidence_bundle',
            'training_artifact', 'replay'
        )
    ),
    domain_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'verifier',
    qualification_decision_id TEXT
        REFERENCES verifier_qualification_decisions(id),
    legacy_unqualified INTEGER NOT NULL DEFAULT 0,
    development_exposed INTEGER NOT NULL DEFAULT 0,
    binding_hash TEXT NOT NULL UNIQUE,
    context_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_verifier_bindings_revision
    ON verifier_bindings (verifier_revision_id, domain_kind, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_verifier_bindings_domain
    ON verifier_bindings (domain_kind, domain_id, role, created_at DESC);

-- Halo Forge Lab v8: immutable reward-system identity, captured training
-- signals, and same-output reward-integrity evidence.  This is deliberately
-- additive to v10: no verifier reliability table is rebuilt or repurposed.
CREATE TABLE IF NOT EXISTS reward_systems (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    latest_revision_id TEXT,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reward_systems_name
    ON reward_systems (name);

CREATE TABLE IF NOT EXISTS reward_system_revisions (
    id TEXT PRIMARY KEY,
    system_id TEXT NOT NULL REFERENCES reward_systems(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    optimizer_verifier_revision_id TEXT NOT NULL
        REFERENCES verifier_profile_revisions(id),
    modality TEXT NOT NULL,
    task_type TEXT NOT NULL,
    input_mapping_json TEXT NOT NULL DEFAULT '{}',
    reward_mapping_json TEXT NOT NULL DEFAULT '{}',
    definition_json TEXT NOT NULL DEFAULT '{}',
    runtime_contract_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (system_id, revision_number),
    UNIQUE (system_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_reward_system_revisions_system
    ON reward_system_revisions (system_id, revision_number DESC);
CREATE INDEX IF NOT EXISTS idx_reward_system_revisions_optimizer
    ON reward_system_revisions (optimizer_verifier_revision_id, created_at DESC);

CREATE TABLE IF NOT EXISTS reward_system_auditors (
    reward_system_revision_id TEXT NOT NULL
        REFERENCES reward_system_revisions(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('primary_sentinel', 'diagnostic')),
    verifier_revision_id TEXT NOT NULL REFERENCES verifier_profile_revisions(id),
    correlated INTEGER NOT NULL DEFAULT 0,
    correlation_reasons_json TEXT NOT NULL DEFAULT '[]',
    configuration_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (reward_system_revision_id, ordinal),
    UNIQUE (reward_system_revision_id, verifier_revision_id),
    CHECK (ordinal >= 0)
);

CREATE INDEX IF NOT EXISTS idx_reward_system_auditors_verifier
    ON reward_system_auditors (verifier_revision_id, role);

CREATE TABLE IF NOT EXISTS reward_audit_protocols (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    latest_revision_id TEXT,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reward_audit_protocol_revisions (
    id TEXT PRIMARY KEY,
    protocol_id TEXT NOT NULL REFERENCES reward_audit_protocols(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    capture_mode TEXT NOT NULL CHECK (
        capture_mode IN ('balanced_256', 'broad_512', 'exhaustive', 'custom')
    ),
    definition_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (protocol_id, revision_number),
    UNIQUE (protocol_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_reward_audit_protocol_revisions_protocol
    ON reward_audit_protocol_revisions (protocol_id, revision_number DESC);

CREATE TABLE IF NOT EXISTS reward_integrity_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    latest_revision_id TEXT,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reward_integrity_profile_revisions (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL
        REFERENCES reward_integrity_profiles(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    template_kind TEXT NOT NULL CHECK (
        template_kind IN (
            'strict_integrity', 'human_aligned_integrity', 'exploratory', 'custom'
        )
    ),
    promotable INTEGER NOT NULL DEFAULT 1,
    requirements_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (profile_id, revision_number),
    UNIQUE (profile_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_reward_integrity_profile_revisions_profile
    ON reward_integrity_profile_revisions (profile_id, revision_number DESC);

-- Direct Train launches reuse the checkpoint segment state machine while
-- retaining one canonical run identity.  Run rows are indexed asynchronously,
-- so run_id intentionally is not a foreign key.
CREATE TABLE IF NOT EXISTS direct_run_segments (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    unit TEXT NOT NULL CHECK (unit IN ('step', 'cycle', 'epoch', 'final')),
    start_value INTEGER NOT NULL,
    end_value INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    checkpoint_occurrence_id TEXT REFERENCES artifact_occurrences(id) ON DELETE SET NULL,
    decision TEXT CHECK (
        decision IS NULL OR decision IN (
            'continue', 'pause', 'stop', 'fork', 'complete', 'incomplete_evidence'
        )
    ),
    decision_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    UNIQUE (run_id, ordinal),
    CHECK (start_value >= 0),
    CHECK (end_value >= start_value)
);

CREATE INDEX IF NOT EXISTS idx_direct_run_segments_run
    ON direct_run_segments (run_id, ordinal);

CREATE TABLE IF NOT EXISTS training_signal_shards (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    direct_run_segment_id TEXT
        REFERENCES direct_run_segments(id) ON DELETE SET NULL,
    trial_segment_id TEXT REFERENCES trial_segments(id) ON DELETE SET NULL,
    reward_system_revision_id TEXT NOT NULL REFERENCES reward_system_revisions(id),
    protocol_revision_id TEXT NOT NULL REFERENCES reward_audit_protocol_revisions(id),
    capability_id TEXT NOT NULL,
    capture_fidelity TEXT NOT NULL CHECK (
        capture_fidelity IN (
            'exact', 'sampled', 'aggregate_only', 'unavailable', 'not_recorded'
        )
    ),
    boundary_unit TEXT NOT NULL,
    boundary_value INTEGER NOT NULL,
    trace_hash TEXT NOT NULL,
    retained_set_hash TEXT NOT NULL,
    event_count INTEGER NOT NULL DEFAULT 0,
    distinct_record_count INTEGER NOT NULL DEFAULT 0,
    aggregate_json TEXT NOT NULL DEFAULT '{}',
    dataset_identity_json TEXT NOT NULL DEFAULT '{}',
    producer_model_hash TEXT NOT NULL,
    checkpoint_hash TEXT NOT NULL,
    runtime_identity_json TEXT NOT NULL DEFAULT '{}',
    storage_path TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    sealed INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, trace_hash),
    CHECK (boundary_value >= 0),
    CHECK (event_count >= 0),
    CHECK (distinct_record_count >= 0)
);

CREATE INDEX IF NOT EXISTS idx_training_signal_shards_run
    ON training_signal_shards (run_id, boundary_value, created_at);
CREATE INDEX IF NOT EXISTS idx_training_signal_shards_system
    ON training_signal_shards (reward_system_revision_id, created_at DESC);

CREATE TABLE IF NOT EXISTS reward_integrity_audits (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    direct_run_segment_id TEXT
        REFERENCES direct_run_segments(id) ON DELETE SET NULL,
    trial_segment_id TEXT REFERENCES trial_segments(id) ON DELETE SET NULL,
    signal_shard_id TEXT NOT NULL REFERENCES training_signal_shards(id),
    reward_system_revision_id TEXT NOT NULL REFERENCES reward_system_revisions(id),
    protocol_revision_id TEXT NOT NULL REFERENCES reward_audit_protocol_revisions(id),
    integrity_profile_revision_id TEXT NOT NULL
        REFERENCES reward_integrity_profile_revisions(id),
    development_suite_revision_id TEXT REFERENCES benchmark_suite_revisions(id),
    status TEXT NOT NULL DEFAULT 'queued',
    stage TEXT NOT NULL DEFAULT 'queued',
    processed_samples INTEGER NOT NULL DEFAULT 0,
    total_samples INTEGER,
    distinct_record_count INTEGER NOT NULL DEFAULT 0,
    request_json TEXT NOT NULL DEFAULT '{}',
    runtime_identity_json TEXT NOT NULL DEFAULT '{}',
    runtime_identity_hash TEXT NOT NULL,
    reuse_key TEXT NOT NULL,
    artifact_path TEXT,
    manifest_hash TEXT,
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_reward_integrity_audits_run
    ON reward_integrity_audits (run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reward_integrity_audits_status
    ON reward_integrity_audits (status, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_reward_integrity_audits_completed_reuse
    ON reward_integrity_audits (reuse_key) WHERE status = 'completed';

CREATE TABLE IF NOT EXISTS reward_integrity_samples (
    audit_id TEXT NOT NULL REFERENCES reward_integrity_audits(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    snapshot_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    record_hash TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    group_id TEXT NOT NULL,
    candidate_ordinal INTEGER NOT NULL DEFAULT 0,
    selection_class TEXT NOT NULL DEFAULT 'uniform_core' CHECK (
        selection_class IN (
            'uniform_core', 'verifier_error', 'threshold_adjacent',
            'highest_reward', 'component_disagreement', 'exhaustive'
        )
    ),
    diagnostic INTEGER NOT NULL DEFAULT 0,
    input_json TEXT NOT NULL DEFAULT '{}',
    output_json TEXT NOT NULL DEFAULT '{}',
    expected_json TEXT,
    media_json TEXT NOT NULL DEFAULT '[]',
    generation_json TEXT NOT NULL DEFAULT '{}',
    lineage_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (audit_id, ordinal),
    UNIQUE (audit_id, snapshot_id),
    CHECK (ordinal >= 0),
    CHECK (candidate_ordinal >= 0)
);

CREATE INDEX IF NOT EXISTS idx_reward_integrity_samples_record
    ON reward_integrity_samples (audit_id, record_id, candidate_ordinal);
CREATE INDEX IF NOT EXISTS idx_reward_integrity_samples_selection
    ON reward_integrity_samples (audit_id, diagnostic, selection_class, ordinal);

CREATE TABLE IF NOT EXISTS reward_integrity_observations (
    audit_id TEXT NOT NULL,
    sample_ordinal INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (
        role IN ('optimizer', 'primary_sentinel', 'diagnostic')
    ),
    auditor_ordinal INTEGER NOT NULL DEFAULT 0,
    verifier_revision_id TEXT NOT NULL REFERENCES verifier_profile_revisions(id),
    reward REAL,
    normalized_reward REAL,
    passed INTEGER,
    parsed_value_json TEXT,
    raw_output_json TEXT,
    details_json TEXT NOT NULL DEFAULT '{}',
    component_trace_json TEXT NOT NULL DEFAULT '[]',
    latency_ms REAL,
    error TEXT,
    runtime_identity_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (audit_id, sample_ordinal, role, auditor_ordinal),
    FOREIGN KEY (audit_id, sample_ordinal)
        REFERENCES reward_integrity_samples(audit_id, ordinal) ON DELETE CASCADE,
    CHECK (auditor_ordinal >= 0),
    CHECK (
        normalized_reward IS NULL
        OR (normalized_reward >= 0.0 AND normalized_reward <= 1.0)
    ),
    CHECK (latency_ms IS NULL OR latency_ms >= 0)
);

CREATE INDEX IF NOT EXISTS idx_reward_integrity_observations_verifier
    ON reward_integrity_observations (verifier_revision_id, role, audit_id);

CREATE TABLE IF NOT EXISTS reward_integrity_metrics (
    audit_id TEXT NOT NULL REFERENCES reward_integrity_audits(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    subgroup TEXT NOT NULL DEFAULT '',
    population TEXT NOT NULL DEFAULT 'uniform_core',
    value REAL,
    ci_low REAL,
    ci_high REAL,
    direction TEXT CHECK (
        direction IS NULL OR direction IN ('maximize', 'minimize')
    ),
    available INTEGER NOT NULL DEFAULT 1,
    missing_reason TEXT,
    record_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (audit_id, name, subgroup, population),
    CHECK (
        (available = 1 AND value IS NOT NULL AND missing_reason IS NULL)
        OR (available = 0 AND value IS NULL AND missing_reason IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_reward_integrity_metrics_name
    ON reward_integrity_metrics (name, value);

CREATE TABLE IF NOT EXISTS reward_integrity_decisions (
    id TEXT PRIMARY KEY,
    audit_id TEXT NOT NULL REFERENCES reward_integrity_audits(id),
    integrity_profile_revision_id TEXT NOT NULL
        REFERENCES reward_integrity_profile_revisions(id),
    decision TEXT NOT NULL CHECK (
        decision IN ('pass', 'warn', 'fail', 'incomplete_evidence')
    ),
    action TEXT NOT NULL CHECK (
        action IN ('continue', 'pause', 'stop', 'fork', 'report_only')
    ),
    reasons_json TEXT NOT NULL DEFAULT '[]',
    evidence_json TEXT NOT NULL DEFAULT '{}',
    override INTEGER NOT NULL DEFAULT 0,
    override_note TEXT,
    supersedes_decision_id TEXT REFERENCES reward_integrity_decisions(id),
    created_at TEXT NOT NULL,
    CHECK (
        override = 0
        OR (override_note IS NOT NULL AND length(trim(override_note)) > 0)
    )
);

CREATE INDEX IF NOT EXISTS idx_reward_integrity_decisions_audit
    ON reward_integrity_decisions (audit_id, created_at DESC);

CREATE TABLE IF NOT EXISTS reward_integrity_bindings (
    id TEXT PRIMARY KEY,
    reward_system_revision_id TEXT NOT NULL REFERENCES reward_system_revisions(id),
    protocol_revision_id TEXT REFERENCES reward_audit_protocol_revisions(id),
    integrity_profile_revision_id TEXT REFERENCES reward_integrity_profile_revisions(id),
    audit_id TEXT REFERENCES reward_integrity_audits(id),
    domain_kind TEXT NOT NULL CHECK (
        domain_kind IN (
            'run', 'run_group', 'trial', 'segment', 'checkpoint', 'artifact',
            'qualification', 'evidence_bundle', 'replay'
        )
    ),
    domain_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'reward_system',
    binding_hash TEXT NOT NULL UNIQUE,
    context_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reward_integrity_bindings_domain
    ON reward_integrity_bindings (domain_kind, domain_id, role, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reward_integrity_bindings_system
    ON reward_integrity_bindings (reward_system_revision_id, created_at DESC);

-- Revisions and published evidence are immutable/append-only. Lifecycle rows
-- (segments, audits, identity heads) deliberately remain mutable.
CREATE TRIGGER IF NOT EXISTS immutable_reward_system_revisions_update
BEFORE UPDATE ON reward_system_revisions
BEGIN
    SELECT RAISE(ABORT, 'reward system revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_reward_system_revisions_delete
BEFORE DELETE ON reward_system_revisions
BEGIN
    SELECT RAISE(ABORT, 'reward system revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_reward_system_auditors_update
BEFORE UPDATE ON reward_system_auditors
BEGIN
    SELECT RAISE(ABORT, 'reward system auditors are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_reward_system_auditors_delete
BEFORE DELETE ON reward_system_auditors
BEGIN
    SELECT RAISE(ABORT, 'reward system auditors are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_reward_audit_protocol_revisions_update
BEFORE UPDATE ON reward_audit_protocol_revisions
BEGIN
    SELECT RAISE(ABORT, 'reward audit protocol revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_reward_audit_protocol_revisions_delete
BEFORE DELETE ON reward_audit_protocol_revisions
BEGIN
    SELECT RAISE(ABORT, 'reward audit protocol revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_reward_integrity_profile_revisions_update
BEFORE UPDATE ON reward_integrity_profile_revisions
BEGIN
    SELECT RAISE(ABORT, 'reward integrity profile revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_reward_integrity_profile_revisions_delete
BEFORE DELETE ON reward_integrity_profile_revisions
BEGIN
    SELECT RAISE(ABORT, 'reward integrity profile revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_training_signal_shards_update
BEFORE UPDATE ON training_signal_shards
BEGIN
    SELECT RAISE(ABORT, 'sealed training signal shards are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_training_signal_shards_delete
BEFORE DELETE ON training_signal_shards
BEGIN
    SELECT RAISE(ABORT, 'sealed training signal shards are immutable');
END;
CREATE TRIGGER IF NOT EXISTS append_only_reward_integrity_samples_update
BEFORE UPDATE ON reward_integrity_samples
BEGIN
    SELECT RAISE(ABORT, 'reward integrity samples are append-only');
END;
CREATE TRIGGER IF NOT EXISTS append_only_reward_integrity_samples_delete
BEFORE DELETE ON reward_integrity_samples
BEGIN
    SELECT RAISE(ABORT, 'reward integrity samples are append-only');
END;
CREATE TRIGGER IF NOT EXISTS append_only_reward_integrity_observations_update
BEFORE UPDATE ON reward_integrity_observations
BEGIN
    SELECT RAISE(ABORT, 'reward integrity observations are append-only');
END;
CREATE TRIGGER IF NOT EXISTS append_only_reward_integrity_observations_delete
BEFORE DELETE ON reward_integrity_observations
BEGIN
    SELECT RAISE(ABORT, 'reward integrity observations are append-only');
END;
CREATE TRIGGER IF NOT EXISTS append_only_reward_integrity_metrics_update
BEFORE UPDATE ON reward_integrity_metrics
BEGIN
    SELECT RAISE(ABORT, 'reward integrity metrics are append-only');
END;
CREATE TRIGGER IF NOT EXISTS append_only_reward_integrity_metrics_delete
BEFORE DELETE ON reward_integrity_metrics
BEGIN
    SELECT RAISE(ABORT, 'reward integrity metrics are append-only');
END;
CREATE TRIGGER IF NOT EXISTS append_only_reward_integrity_decisions_update
BEFORE UPDATE ON reward_integrity_decisions
BEGIN
    SELECT RAISE(ABORT, 'reward integrity decisions are append-only');
END;
CREATE TRIGGER IF NOT EXISTS append_only_reward_integrity_decisions_delete
BEFORE DELETE ON reward_integrity_decisions
BEGIN
    SELECT RAISE(ABORT, 'reward integrity decisions are append-only');
END;
CREATE TRIGGER IF NOT EXISTS append_only_reward_integrity_bindings_update
BEFORE UPDATE ON reward_integrity_bindings
BEGIN
    SELECT RAISE(ABORT, 'reward integrity bindings are append-only');
END;
CREATE TRIGGER IF NOT EXISTS append_only_reward_integrity_bindings_delete
BEFORE DELETE ON reward_integrity_bindings
BEGIN
    SELECT RAISE(ABORT, 'reward integrity bindings are append-only');
END;

-- SQLite triggers make the database-level contract match the public API:
-- revisions and evidence rows cannot be edited or deleted in-place.
CREATE TRIGGER IF NOT EXISTS immutable_verifier_profile_revisions_update
BEFORE UPDATE ON verifier_profile_revisions
BEGIN
    SELECT RAISE(ABORT, 'verifier profile revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_verifier_profile_revisions_delete
BEFORE DELETE ON verifier_profile_revisions
BEGIN
    SELECT RAISE(ABORT, 'verifier profile revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_verifier_components_update
BEFORE UPDATE ON verifier_revision_components
BEGIN
    SELECT RAISE(ABORT, 'verifier revision components are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_verifier_components_delete
BEFORE DELETE ON verifier_revision_components
BEGIN
    SELECT RAISE(ABORT, 'verifier revision components are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_verifier_protocol_revisions_update
BEFORE UPDATE ON verifier_calibration_protocol_revisions
BEGIN
    SELECT RAISE(ABORT, 'verifier calibration protocol revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_verifier_protocol_revisions_delete
BEFORE DELETE ON verifier_calibration_protocol_revisions
BEGIN
    SELECT RAISE(ABORT, 'verifier calibration protocol revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_verifier_qualification_revisions_update
BEFORE UPDATE ON verifier_qualification_profile_revisions
BEGIN
    SELECT RAISE(ABORT, 'verifier qualification profile revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_verifier_qualification_revisions_delete
BEFORE DELETE ON verifier_qualification_profile_revisions
BEGIN
    SELECT RAISE(ABORT, 'verifier qualification profile revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS append_only_verifier_samples_update
BEFORE UPDATE ON verifier_calibration_samples
BEGIN
    SELECT RAISE(ABORT, 'verifier calibration samples are append-only');
END;
CREATE TRIGGER IF NOT EXISTS append_only_verifier_samples_delete
BEFORE DELETE ON verifier_calibration_samples
BEGIN
    SELECT RAISE(ABORT, 'verifier calibration samples are append-only');
END;
CREATE TRIGGER IF NOT EXISTS append_only_verifier_metrics_update
BEFORE UPDATE ON verifier_calibration_metrics
BEGIN
    SELECT RAISE(ABORT, 'verifier calibration metrics are append-only');
END;
CREATE TRIGGER IF NOT EXISTS append_only_verifier_metrics_delete
BEFORE DELETE ON verifier_calibration_metrics
BEGIN
    SELECT RAISE(ABORT, 'verifier calibration metrics are append-only');
END;
CREATE TRIGGER IF NOT EXISTS append_only_verifier_decisions_update
BEFORE UPDATE ON verifier_qualification_decisions
BEGIN
    SELECT RAISE(ABORT, 'verifier qualification decisions are append-only');
END;
CREATE TRIGGER IF NOT EXISTS append_only_verifier_decisions_delete
BEFORE DELETE ON verifier_qualification_decisions
BEGIN
    SELECT RAISE(ABORT, 'verifier qualification decisions are append-only');
END;
CREATE TRIGGER IF NOT EXISTS append_only_verifier_alias_events_update
BEFORE UPDATE ON verifier_alias_events
BEGIN
    SELECT RAISE(ABORT, 'verifier alias history is append-only');
END;
CREATE TRIGGER IF NOT EXISTS append_only_verifier_alias_events_delete
BEFORE DELETE ON verifier_alias_events
BEGIN
    SELECT RAISE(ABORT, 'verifier alias history is append-only');
END;
CREATE TRIGGER IF NOT EXISTS append_only_verifier_bindings_update
BEFORE UPDATE ON verifier_bindings
BEGIN
    SELECT RAISE(ABORT, 'verifier bindings are append-only');
END;
CREATE TRIGGER IF NOT EXISTS append_only_verifier_bindings_delete
BEFORE DELETE ON verifier_bindings
BEGIN
    SELECT RAISE(ABORT, 'verifier bindings are append-only');
END;

-- Dataset Lab v9 guided own-data imports.  Import sessions and files are
-- lifecycle records; completed inspections are immutable evidence derived
-- from one exact source fingerprint, adapter, and scenario-registry revision.
CREATE TABLE IF NOT EXISTS dataset_imports (
    id TEXT PRIMARY KEY,
    source_kind TEXT NOT NULL CHECK (
        source_kind IN ('upload', 'workstation_path', 'huggingface', 'desktop_reference')
    ),
    status TEXT NOT NULL DEFAULT 'draft' CHECK (
        status IN (
            'draft', 'uploading', 'ready', 'inspecting', 'completed',
            'failed', 'cancelled', 'published', 'expired'
        )
    ),
    display_name TEXT,
    source_uri TEXT,
    source_config TEXT,
    source_split TEXT,
    source_revision TEXT,
    resolved_revision TEXT,
    scenario_revision_id TEXT,
    fingerprint TEXT,
    expected_size_bytes INTEGER,
    received_size_bytes INTEGER NOT NULL DEFAULT 0,
    file_count INTEGER NOT NULL DEFAULT 0,
    staging_path TEXT,
    managed_source_path TEXT,
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    published_dataset_id TEXT REFERENCES datasets(id) ON DELETE SET NULL,
    published_source_id TEXT REFERENCES dataset_sources(id) ON DELETE SET NULL,
    latest_inspection_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    expires_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_dataset_imports_status
    ON dataset_imports (status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_dataset_imports_fingerprint
    ON dataset_imports (fingerprint);
CREATE INDEX IF NOT EXISTS idx_dataset_imports_expiry
    ON dataset_imports (expires_at, status);

CREATE TABLE IF NOT EXISTS dataset_import_files (
    id TEXT PRIMARY KEY,
    import_id TEXT NOT NULL REFERENCES dataset_imports(id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending', 'uploading', 'complete', 'failed', 'cancelled')
    ),
    media_type TEXT,
    size_bytes INTEGER NOT NULL,
    received_bytes INTEGER NOT NULL DEFAULT 0,
    expected_sha256 TEXT,
    content_sha256 TEXT,
    staging_path TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE (import_id, relative_path)
);

CREATE INDEX IF NOT EXISTS idx_dataset_import_files_import
    ON dataset_import_files (import_id, relative_path);
CREATE INDEX IF NOT EXISTS idx_dataset_import_files_status
    ON dataset_import_files (import_id, status);

CREATE TABLE IF NOT EXISTS dataset_source_inspections (
    id TEXT PRIMARY KEY,
    import_id TEXT REFERENCES dataset_imports(id) ON DELETE SET NULL,
    source_id TEXT REFERENCES dataset_sources(id) ON DELETE SET NULL,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (
        status IN ('queued', 'running', 'completed', 'failed', 'cancelled', 'interrupted')
    ),
    source_fingerprint TEXT NOT NULL,
    import_adapter_version TEXT NOT NULL,
    scenario_registry_revision TEXT NOT NULL,
    scenario_revision_id TEXT,
    sample_seed INTEGER NOT NULL DEFAULT 42,
    total_records INTEGER NOT NULL DEFAULT 0,
    valid_records INTEGER NOT NULL DEFAULT 0,
    invalid_records INTEGER NOT NULL DEFAULT 0,
    sample_count INTEGER NOT NULL DEFAULT 0,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    fields_json TEXT NOT NULL DEFAULT '[]',
    candidates_json TEXT NOT NULL DEFAULT '[]',
    preview_json TEXT NOT NULL DEFAULT '[]',
    issues_json TEXT NOT NULL DEFAULT '[]',
    statistics_json TEXT NOT NULL DEFAULT '{}',
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE (
        source_fingerprint, import_adapter_version, scenario_registry_revision
    )
);

CREATE INDEX IF NOT EXISTS idx_dataset_source_inspections_import
    ON dataset_source_inspections (import_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dataset_source_inspections_source
    ON dataset_source_inspections (source_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dataset_source_inspections_status
    ON dataset_source_inspections (status, created_at DESC);

-- One immutable inspection can be reused by multiple imports with identical
-- content.  The link preserves the active import identity without mutating
-- the completed inspection or silently publishing an earlier import path.
CREATE TABLE IF NOT EXISTS dataset_import_inspections (
    import_id TEXT NOT NULL REFERENCES dataset_imports(id) ON DELETE CASCADE,
    inspection_id TEXT NOT NULL
        REFERENCES dataset_source_inspections(id) ON DELETE CASCADE,
    linked_at TEXT NOT NULL,
    PRIMARY KEY (import_id, inspection_id)
);

CREATE INDEX IF NOT EXISTS idx_dataset_import_inspections_inspection
    ON dataset_import_inspections (inspection_id, linked_at DESC);

CREATE TRIGGER IF NOT EXISTS immutable_dataset_import_inspection_links_update
BEFORE UPDATE ON dataset_import_inspections
BEGIN
    SELECT RAISE(ABORT, 'dataset import inspection links are immutable');
END;

CREATE TRIGGER IF NOT EXISTS immutable_completed_dataset_inspections_update
BEFORE UPDATE ON dataset_source_inspections
WHEN OLD.status = 'completed'
BEGIN
    SELECT RAISE(ABORT, 'completed dataset inspections are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_completed_dataset_inspections_delete
BEFORE DELETE ON dataset_source_inspections
WHEN OLD.status = 'completed'
BEGIN
    SELECT RAISE(ABORT, 'completed dataset inspections are immutable');
END;

-- Halo Forge Lab v10 corpus extraction.  Lifecycle rows are mutable only
-- until completion; a completed extraction and all of its indexed items are
-- immutable evidence for one exact source/configuration identity.  Payload
-- bytes live in a separately checksummed, content-addressed bundle.
CREATE TABLE IF NOT EXISTS document_extractions (
    id TEXT PRIMARY KEY,
    import_id TEXT REFERENCES dataset_imports(id) ON DELETE SET NULL,
    source_id TEXT REFERENCES dataset_sources(id) ON DELETE SET NULL,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (
        status IN (
            'queued', 'running', 'completed', 'failed', 'cancelled', 'interrupted'
        )
    ),
    source_kind TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    reuse_key TEXT NOT NULL UNIQUE,
    config_json TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT,
    bundle_path TEXT,
    manifest_hash TEXT,
    document_count INTEGER NOT NULL DEFAULT 0,
    item_count INTEGER NOT NULL DEFAULT 0,
    quarantined_count INTEGER NOT NULL DEFAULT 0,
    extracted_text_bytes INTEGER NOT NULL DEFAULT 0,
    statistics_json TEXT NOT NULL DEFAULT '{}',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    CHECK (document_count >= 0),
    CHECK (item_count >= 0),
    CHECK (quarantined_count >= 0),
    CHECK (extracted_text_bytes >= 0),
    CHECK (
        status != 'completed'
        OR (
            content_hash IS NOT NULL
            AND bundle_path IS NOT NULL
            AND manifest_hash IS NOT NULL
            AND completed_at IS NOT NULL
            AND item_count = document_count + quarantined_count
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_document_extractions_source
    ON document_extractions (
        source_fingerprint, extractor_version, config_hash, status
    );
CREATE INDEX IF NOT EXISTS idx_document_extractions_import
    ON document_extractions (import_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_document_extractions_dataset_source
    ON document_extractions (source_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_document_extractions_status
    ON document_extractions (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_document_extractions_content
    ON document_extractions (content_hash) WHERE content_hash IS NOT NULL;

CREATE TABLE IF NOT EXISTS document_extraction_items (
    extraction_id TEXT NOT NULL
        REFERENCES document_extractions(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    document_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('extracted', 'quarantined')),
    source_uri TEXT NOT NULL,
    relative_path TEXT NOT NULL DEFAULT '',
    source_kind TEXT NOT NULL,
    media_type TEXT NOT NULL,
    title TEXT,
    content_hash TEXT,
    text_char_count INTEGER NOT NULL DEFAULT 0,
    text_byte_count INTEGER NOT NULL DEFAULT 0,
    bundle_member TEXT NOT NULL,
    bundle_ordinal INTEGER NOT NULL,
    locator_json TEXT NOT NULL DEFAULT '{}',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    error_code TEXT,
    error TEXT,
    PRIMARY KEY (extraction_id, ordinal),
    UNIQUE (extraction_id, document_id),
    CHECK (ordinal >= 0),
    CHECK (bundle_ordinal >= 0),
    CHECK (text_char_count >= 0),
    CHECK (text_byte_count >= 0),
    CHECK (
        (status = 'extracted' AND content_hash IS NOT NULL
            AND error_code IS NULL AND error IS NULL)
        OR
        (status = 'quarantined' AND content_hash IS NULL
            AND error_code IS NOT NULL AND error IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_document_extraction_items_document
    ON document_extraction_items (document_id);
CREATE INDEX IF NOT EXISTS idx_document_extraction_items_content
    ON document_extraction_items (content_hash) WHERE content_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_document_extraction_items_status
    ON document_extraction_items (extraction_id, status, ordinal);
CREATE INDEX IF NOT EXISTS idx_document_extraction_items_source_uri
    ON document_extraction_items (source_uri);
CREATE INDEX IF NOT EXISTS idx_document_extraction_items_error
    ON document_extraction_items (error_code, extraction_id)
    WHERE status = 'quarantined';

CREATE TRIGGER IF NOT EXISTS immutable_completed_document_extractions_update
BEFORE UPDATE ON document_extractions
WHEN OLD.status = 'completed'
BEGIN
    SELECT RAISE(ABORT, 'completed document extractions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_completed_document_extractions_delete
BEFORE DELETE ON document_extractions
WHEN OLD.status = 'completed'
BEGIN
    SELECT RAISE(ABORT, 'completed document extractions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_document_extraction_items_update
BEFORE UPDATE ON document_extraction_items
BEGIN
    SELECT RAISE(ABORT, 'document extraction items are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_document_extraction_items_delete
BEFORE DELETE ON document_extraction_items
BEGIN
    SELECT RAISE(ABORT, 'document extraction items are immutable');
END;
CREATE TRIGGER IF NOT EXISTS sealed_document_extraction_items_insert
BEFORE INSERT ON document_extraction_items
WHEN (
    SELECT status FROM document_extractions WHERE id = NEW.extraction_id
) = 'completed'
BEGIN
    SELECT RAISE(ABORT, 'completed document extractions cannot accept items');
END;

-- Halo Forge Lab v11: guided proof-run outcomes and reviewed full-run gates.
CREATE TABLE IF NOT EXISTS scenario_outcome_profiles (
    id TEXT PRIMARY KEY,
    scenario_revision_id TEXT NOT NULL,
    version TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    definition_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (scenario_revision_id, version)
);

CREATE TABLE IF NOT EXISTS training_outcome_assessments (
    id TEXT PRIMARY KEY,
    proof_run_id TEXT NOT NULL,
    scenario_revision_id TEXT NOT NULL,
    profile_id TEXT NOT NULL REFERENCES scenario_outcome_profiles(id),
    stage TEXT NOT NULL DEFAULT 'queued',
    progress_json TEXT NOT NULL DEFAULT '{}',
    request_json TEXT NOT NULL DEFAULT '{}',
    resume_cursor_json TEXT NOT NULL DEFAULT '{}',
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (
        status IN (
            'queued', 'running', 'improved', 'regressed', 'mixed',
            'no_clear_change', 'incomplete_evidence', 'technical_failure',
            'failed', 'cancelled'
        )
    ),
    technical_status TEXT NOT NULL,
    quality_status TEXT NOT NULL,
    base_evaluation_id TEXT REFERENCES evaluations(id) ON DELETE SET NULL,
    candidate_evaluation_id TEXT REFERENCES evaluations(id) ON DELETE SET NULL,
    comparison_hash TEXT,
    resource_projection_json TEXT NOT NULL DEFAULT '{}',
    diagnostics_json TEXT NOT NULL DEFAULT '{}',
    summary_json TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT,
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_training_outcomes_proof
    ON training_outcome_assessments (proof_run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_outcomes_status
    ON training_outcome_assessments (status, created_at DESC);

CREATE TABLE IF NOT EXISTS training_outcome_findings (
    id TEXT PRIMARY KEY,
    assessment_id TEXT NOT NULL
        REFERENCES training_outcome_assessments(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    summary TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    why_it_matters TEXT NOT NULL,
    safe_remedies_json TEXT NOT NULL DEFAULT '[]',
    available_actions_json TEXT NOT NULL DEFAULT '[]',
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (assessment_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_training_outcome_findings
    ON training_outcome_findings (assessment_id, severity, ordinal);

CREATE TABLE IF NOT EXISTS training_outcome_decisions (
    id TEXT PRIMARY KEY,
    assessment_id TEXT REFERENCES training_outcome_assessments(id),
    proof_run_id TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (
        decision IN ('evaluate', 'repair', 'retry', 'fork', 'start_full_run', 'override')
    ),
    reason TEXT NOT NULL,
    full_run_id TEXT,
    context_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_training_outcome_decisions_proof
    ON training_outcome_decisions (proof_run_id, created_at DESC);

-- Halo Forge Lab v12: bounded adaptation studies over existing run groups.
CREATE TABLE IF NOT EXISTS adaptation_studies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    latest_protocol_revision_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS adaptation_study_protocol_revisions (
    id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL REFERENCES adaptation_studies(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    design_kind TEXT NOT NULL CHECK (
        design_kind IN ('paired_ab', 'dose_response', 'factorial_2x2')
    ),
    question TEXT NOT NULL,
    definition_json TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    launch_status TEXT NOT NULL DEFAULT 'not_started',
    launch_progress_json TEXT NOT NULL DEFAULT '{}',
    launch_work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    launch_error TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (study_id, revision_number)
);

CREATE TABLE IF NOT EXISTS adaptation_study_arms (
    id TEXT PRIMARY KEY,
    protocol_revision_id TEXT NOT NULL
        REFERENCES adaptation_study_protocol_revisions(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    name TEXT NOT NULL,
    is_control INTEGER NOT NULL DEFAULT 0,
    factor_values_json TEXT NOT NULL DEFAULT '{}',
    launch_config_json TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT NOT NULL,
    UNIQUE (protocol_revision_id, ordinal),
    UNIQUE (protocol_revision_id, name)
);

CREATE TABLE IF NOT EXISTS adaptation_study_assignments (
    id TEXT PRIMARY KEY,
    protocol_revision_id TEXT NOT NULL
        REFERENCES adaptation_study_protocol_revisions(id) ON DELETE CASCADE,
    arm_id TEXT NOT NULL REFERENCES adaptation_study_arms(id) ON DELETE CASCADE,
    seed INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    run_group_id TEXT REFERENCES run_groups(id) ON DELETE SET NULL,
    run_id TEXT,
    status TEXT NOT NULL DEFAULT 'planned',
    created_at TEXT NOT NULL,
    UNIQUE (protocol_revision_id, arm_id, seed)
);

CREATE TABLE IF NOT EXISTS adaptation_study_contrasts (
    id TEXT PRIMARY KEY,
    protocol_revision_id TEXT NOT NULL
        REFERENCES adaptation_study_protocol_revisions(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    name TEXT NOT NULL,
    left_arm_id TEXT NOT NULL REFERENCES adaptation_study_arms(id),
    right_arm_id TEXT NOT NULL REFERENCES adaptation_study_arms(id),
    metric TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('maximize', 'minimize')),
    conclusion_kind TEXT NOT NULL CHECK (
        conclusion_kind IN ('superiority', 'equivalence', 'non_inferiority')
    ),
    practical_margin REAL NOT NULL,
    exploratory INTEGER NOT NULL DEFAULT 0,
    UNIQUE (protocol_revision_id, ordinal)
);

CREATE TABLE IF NOT EXISTS adaptation_study_analyses (
    id TEXT PRIMARY KEY,
    protocol_revision_id TEXT NOT NULL
        REFERENCES adaptation_study_protocol_revisions(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'queued',
    progress_json TEXT NOT NULL DEFAULT '{}',
    request_json TEXT NOT NULL DEFAULT '{}',
    resume_cursor_json TEXT NOT NULL DEFAULT '{}',
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    analysis_json TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    evidence_classification TEXT NOT NULL CHECK (
        evidence_classification IN ('causal', 'comparative', 'incomplete')
    ),
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    bundle_path TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS adaptation_study_deviations (
    id TEXT PRIMARY KEY,
    protocol_revision_id TEXT NOT NULL
        REFERENCES adaptation_study_protocol_revisions(id) ON DELETE CASCADE,
    reason TEXT NOT NULL,
    change_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS adaptation_study_decisions (
    id TEXT PRIMARY KEY,
    protocol_revision_id TEXT NOT NULL
        REFERENCES adaptation_study_protocol_revisions(id) ON DELETE CASCADE,
    analysis_id TEXT REFERENCES adaptation_study_analyses(id),
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_adaptation_protocols_study
    ON adaptation_study_protocol_revisions (study_id, revision_number DESC);
CREATE INDEX IF NOT EXISTS idx_adaptation_assignments_protocol
    ON adaptation_study_assignments (protocol_revision_id, ordinal);

-- Halo Forge Lab v13: citation-grounded generation over immutable corpora.
CREATE TABLE IF NOT EXISTS grounding_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    latest_revision_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS grounding_profile_revisions (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL REFERENCES grounding_profiles(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    definition_json TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    UNIQUE (profile_id, revision_number)
);

CREATE TABLE IF NOT EXISTS grounded_generation_batches (
    id TEXT PRIMARY KEY,
    profile_revision_id TEXT NOT NULL
        REFERENCES grounding_profile_revisions(id),
    source_version_id TEXT REFERENCES dataset_versions(id),
    extraction_id TEXT REFERENCES document_extractions(id),
    status TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'queued',
    intended_destination TEXT NOT NULL CHECK (
        intended_destination IN ('training', 'development_evaluation')
    ),
    request_json TEXT NOT NULL DEFAULT '{}',
    progress_json TEXT NOT NULL DEFAULT '{}',
    resume_cursor_json TEXT NOT NULL DEFAULT '{}',
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    source_hash TEXT NOT NULL,
    content_hash TEXT,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    accepted_count INTEGER NOT NULL DEFAULT 0,
    rejected_count INTEGER NOT NULL DEFAULT 0,
    coverage_json TEXT NOT NULL DEFAULT '{}',
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    bundle_path TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS grounded_candidates (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL
        REFERENCES grounded_generation_batches(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL,
    document_id TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    prompt_json TEXT NOT NULL DEFAULT '{}',
    output_json TEXT NOT NULL DEFAULT '{}',
    verifier_json TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT NOT NULL,
    rejection_reason TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (batch_id, ordinal)
);

CREATE TABLE IF NOT EXISTS grounding_citations (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES grounded_candidates(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    document_id TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    span_start INTEGER,
    span_end INTEGER,
    locator_json TEXT NOT NULL DEFAULT '{}',
    quoted_hash TEXT NOT NULL,
    structural_valid INTEGER NOT NULL,
    semantic_status TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (candidate_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_grounded_batches_status
    ON grounded_generation_batches (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_grounded_candidates_batch
    ON grounded_candidates (batch_id, status, ordinal);

-- Halo Forge Lab v14: specialized non-generative task identities.
CREATE TABLE IF NOT EXISTS task_label_schemas (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    task_kind TEXT NOT NULL,
    modality TEXT NOT NULL,
    latest_revision_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_label_schema_revisions (
    id TEXT PRIMARY KEY,
    schema_id TEXT NOT NULL REFERENCES task_label_schemas(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    definition_json TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    UNIQUE (schema_id, revision_number)
);

CREATE TABLE IF NOT EXISTS specialized_artifact_metadata (
    artifact_occurrence_id TEXT PRIMARY KEY
        REFERENCES artifact_occurrences(id) ON DELETE CASCADE,
    task_kind TEXT NOT NULL,
    modality TEXT NOT NULL,
    label_schema_revision_id TEXT REFERENCES task_label_schema_revisions(id),
    model_head_hash TEXT NOT NULL,
    processor_hash TEXT NOT NULL,
    loss_adapter TEXT NOT NULL,
    loss_adapter_version TEXT NOT NULL,
    retrieval_corpus_hash TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

-- Halo Forge Lab v15: deterministic local environments and trajectories.
CREATE TABLE IF NOT EXISTS agent_environments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    latest_revision_id TEXT,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_environment_revisions (
    id TEXT PRIMARY KEY,
    environment_id TEXT NOT NULL
        REFERENCES agent_environments(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    adapter_id TEXT NOT NULL,
    adapter_version TEXT NOT NULL,
    implementation_hash TEXT NOT NULL,
    definition_json TEXT NOT NULL,
    fixture_hash TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    storage_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (environment_id, revision_number)
);

CREATE TABLE IF NOT EXISTS environment_tools (
    id TEXT PRIMARY KEY,
    environment_revision_id TEXT NOT NULL
        REFERENCES agent_environment_revisions(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    name TEXT NOT NULL,
    definition_json TEXT NOT NULL,
    implementation_hash TEXT NOT NULL,
    UNIQUE (environment_revision_id, ordinal),
    UNIQUE (environment_revision_id, name)
);

CREATE TABLE IF NOT EXISTS episode_suites (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    purpose TEXT NOT NULL CHECK (
        purpose IN ('development', 'operational', 'holdout')
    ),
    latest_revision_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episode_suite_revisions (
    id TEXT PRIMARY KEY,
    suite_id TEXT NOT NULL REFERENCES episode_suites(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    environment_revision_id TEXT NOT NULL
        REFERENCES agent_environment_revisions(id),
    definition_json TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    UNIQUE (suite_id, revision_number)
);

CREATE TABLE IF NOT EXISTS agent_episodes (
    id TEXT PRIMARY KEY,
    suite_revision_id TEXT NOT NULL REFERENCES episode_suite_revisions(id),
    suite_item_id TEXT NOT NULL,
    subject_type TEXT NOT NULL,
    subject_ref TEXT NOT NULL,
    subject_hash TEXT NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'queued',
    progress_json TEXT NOT NULL DEFAULT '{}',
    request_json TEXT NOT NULL DEFAULT '{}',
    resume_cursor_json TEXT NOT NULL DEFAULT '{}',
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    parent_episode_id TEXT REFERENCES agent_episodes(id) ON DELETE SET NULL,
    terminal_reason TEXT,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    initial_state_hash TEXT NOT NULL,
    final_state_hash TEXT,
    snapshot_path TEXT,
    trace_hash TEXT,
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE (suite_revision_id, suite_item_id, subject_hash, seed)
);

CREATE TABLE IF NOT EXISTS agent_episode_steps (
    episode_id TEXT NOT NULL REFERENCES agent_episodes(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    observation_json TEXT NOT NULL,
    raw_output TEXT,
    action_json TEXT NOT NULL,
    tool_call_json TEXT,
    tool_result_json TEXT,
    state_delta_json TEXT NOT NULL DEFAULT '{}',
    state_hash TEXT NOT NULL,
    verifier_json TEXT NOT NULL DEFAULT '{}',
    latency_ms REAL,
    error TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (episode_id, ordinal)
);

CREATE TABLE IF NOT EXISTS trajectory_sets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    latest_revision_id TEXT,
    status TEXT NOT NULL DEFAULT 'ready',
    stage TEXT NOT NULL DEFAULT 'ready',
    progress_json TEXT NOT NULL DEFAULT '{}',
    request_json TEXT NOT NULL DEFAULT '{}',
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trajectory_set_revisions (
    id TEXT PRIMARY KEY,
    trajectory_set_id TEXT NOT NULL REFERENCES trajectory_sets(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    storage_path TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (trajectory_set_id, revision_number)
);

CREATE TABLE IF NOT EXISTS trajectory_set_items (
    revision_id TEXT NOT NULL
        REFERENCES trajectory_set_revisions(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    episode_id TEXT NOT NULL REFERENCES agent_episodes(id),
    output_adapter TEXT NOT NULL,
    record_json TEXT NOT NULL,
    record_hash TEXT NOT NULL,
    PRIMARY KEY (revision_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_agent_episodes_suite
    ON agent_episodes (suite_revision_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_episode_steps
    ON agent_episode_steps (episode_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_future_outcome_work
    ON training_outcome_assessments (work_item_id, status);
CREATE INDEX IF NOT EXISTS idx_future_grounding_work
    ON grounded_generation_batches (work_item_id, status);
CREATE INDEX IF NOT EXISTS idx_future_episode_work
    ON agent_episodes (work_item_id, status);

-- Halo Forge Lab v17: cross-platform readiness, immutable data repair, and
-- privacy-safe support bundles. Repair plans are overlays: source rows and
-- binary media are never updated in place.
CREATE TABLE IF NOT EXISTS workstation_readiness_assessments (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('ready', 'attention', 'blocked')),
    platform TEXT NOT NULL,
    architecture TEXT NOT NULL,
    app_version TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    checks_json TEXT NOT NULL DEFAULT '[]',
    remediations_json TEXT NOT NULL DEFAULT '[]',
    capability_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_repair_sessions (
    id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES dataset_sources(id) ON DELETE SET NULL,
    inspection_id TEXT REFERENCES dataset_source_inspections(id) ON DELETE SET NULL,
    dataset_version_id TEXT REFERENCES dataset_versions(id) ON DELETE SET NULL,
    source_uri TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL,
    scenario_revision_id TEXT,
    status TEXT NOT NULL CHECK (
        status IN (
            'draft', 'scanning', 'ready', 'previewing', 'published',
            'stale', 'failed', 'cancelled'
        )
    ),
    stage TEXT NOT NULL DEFAULT 'draft',
    progress_json TEXT NOT NULL DEFAULT '{}',
    issue_summary_json TEXT NOT NULL DEFAULT '{}',
    latest_plan_revision_id TEXT,
    latest_preview_id TEXT,
    published_repair_revision_id TEXT,
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_repair_plan_revisions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES dataset_repair_sessions(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    source_fingerprint TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    definition_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (session_id, revision_number)
);

CREATE TABLE IF NOT EXISTS dataset_repair_issues (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES dataset_repair_sessions(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    record_id TEXT,
    source_index INTEGER,
    code TEXT NOT NULL,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    field_path TEXT,
    message TEXT NOT NULL,
    suggested_actions_json TEXT NOT NULL DEFAULT '[]',
    evidence_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (session_id, ordinal)
);

CREATE TABLE IF NOT EXISTS dataset_repair_actions (
    revision_id TEXT NOT NULL
        REFERENCES dataset_repair_plan_revisions(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    record_id TEXT,
    source_index INTEGER,
    issue_code TEXT NOT NULL,
    action_kind TEXT NOT NULL,
    field_path TEXT,
    value_json TEXT,
    reason TEXT NOT NULL,
    before_hash TEXT,
    after_hash TEXT,
    PRIMARY KEY (revision_id, ordinal)
);

CREATE TABLE IF NOT EXISTS dataset_repair_previews (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES dataset_repair_sessions(id) ON DELETE CASCADE,
    plan_revision_id TEXT NOT NULL
        REFERENCES dataset_repair_plan_revisions(id) ON DELETE CASCADE,
    source_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'completed', 'failed', 'cancelled')
    ),
    exact INTEGER NOT NULL DEFAULT 0,
    counts_json TEXT NOT NULL DEFAULT '{}',
    issue_counts_json TEXT NOT NULL DEFAULT '{}',
    split_impact_json TEXT NOT NULL DEFAULT '{}',
    sample_json TEXT NOT NULL DEFAULT '[]',
    content_hash TEXT,
    storage_path TEXT,
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS dataset_repair_revisions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES dataset_repair_sessions(id),
    plan_revision_id TEXT NOT NULL REFERENCES dataset_repair_plan_revisions(id),
    preview_id TEXT NOT NULL REFERENCES dataset_repair_previews(id),
    source_fingerprint TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    repaired_record_set_hash TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS support_bundles (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (
        status IN ('draft', 'queued', 'running', 'completed', 'failed', 'cancelled')
    ),
    categories_json TEXT NOT NULL DEFAULT '[]',
    preview_json TEXT NOT NULL DEFAULT '{}',
    manifest_json TEXT NOT NULL DEFAULT '{}',
    storage_path TEXT,
    content_hash TEXT,
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS release_qualifications (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'completed' CHECK (
        status IN ('queued', 'running', 'completed', 'failed', 'cancelled')
    ),
    platform TEXT NOT NULL,
    architecture TEXT NOT NULL,
    package_type TEXT NOT NULL,
    signature_state TEXT NOT NULL,
    smoke_status TEXT NOT NULL,
    supported_backends_json TEXT NOT NULL DEFAULT '[]',
    evidence_json TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT NOT NULL,
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    progress_json TEXT NOT NULL DEFAULT '{}',
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_repair_sessions_source
    ON dataset_repair_sessions (source_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_repair_sessions_inspection
    ON dataset_repair_sessions (inspection_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_repair_actions_record
    ON dataset_repair_actions (revision_id, record_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_repair_issues_session
    ON dataset_repair_issues (session_id, severity, category, ordinal);
CREATE INDEX IF NOT EXISTS idx_repair_issues_record
    ON dataset_repair_issues (record_id, session_id);
CREATE INDEX IF NOT EXISTS idx_repair_previews_work
    ON dataset_repair_previews (work_item_id, status);
CREATE INDEX IF NOT EXISTS idx_support_bundles_work
    ON support_bundles (work_item_id, status);

CREATE TRIGGER IF NOT EXISTS immutable_repair_plan_revision_update
BEFORE UPDATE ON dataset_repair_plan_revisions
BEGIN
    SELECT RAISE(ABORT, 'dataset repair plan revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_repair_plan_revision_delete
BEFORE DELETE ON dataset_repair_plan_revisions
BEGIN
    SELECT RAISE(ABORT, 'dataset repair plan revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_repair_action_update
BEFORE UPDATE ON dataset_repair_actions
BEGIN
    SELECT RAISE(ABORT, 'dataset repair actions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_repair_action_delete
BEFORE DELETE ON dataset_repair_actions
BEGIN
    SELECT RAISE(ABORT, 'dataset repair actions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_repair_revision_update
BEFORE UPDATE ON dataset_repair_revisions
BEGIN
    SELECT RAISE(ABORT, 'dataset repair revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_repair_revision_delete
BEFORE DELETE ON dataset_repair_revisions
BEGIN
    SELECT RAISE(ABORT, 'dataset repair revisions are immutable');
END;

-- Halo Forge Lab v18: immutable guided training plans, exact model
-- preparation identity, disposable capacity checks, and explicit decisions.
CREATE TABLE IF NOT EXISTS training_plans (
    id TEXT PRIMARY KEY,
    dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(id),
    scenario_revision_id TEXT,
    status TEXT NOT NULL CHECK (
        status IN ('draft', 'recommended', 'preparing', 'checking', 'ready', 'blocked', 'stale')
    ),
    latest_revision_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_plan_revisions (
    id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL REFERENCES training_plans(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('draft', 'resolved')),
    content_hash TEXT NOT NULL UNIQUE,
    profile_id TEXT NOT NULL,
    profile_version TEXT NOT NULL,
    dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(id),
    scenario_revision_id TEXT,
    trainer_mode TEXT NOT NULL,
    backend TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_revision TEXT,
    resolved_model_commit TEXT,
    definition_json TEXT NOT NULL,
    reasons_json TEXT NOT NULL DEFAULT '[]',
    forecast_json TEXT NOT NULL DEFAULT '{}',
    compute_shape_hash TEXT NOT NULL,
    runtime_hash TEXT NOT NULL,
    runtime_profile_revision_id TEXT REFERENCES managed_runtime_revisions(id),
    training_path_revision_id TEXT REFERENCES training_path_profile_revisions(id),
    training_path_certification_id TEXT REFERENCES training_path_certifications(id),
    created_at TEXT NOT NULL,
    UNIQUE (plan_id, revision_number)
);

CREATE TABLE IF NOT EXISTS model_preparations (
    id TEXT PRIMARY KEY,
    plan_revision_id TEXT NOT NULL REFERENCES training_plan_revisions(id),
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'completed', 'failed', 'cancelled', 'blocked')
    ),
    requested_model_id TEXT NOT NULL,
    requested_revision TEXT,
    resolved_commit TEXT,
    cache_path TEXT,
    manifest_path TEXT,
    manifest_hash TEXT,
    size_bytes INTEGER,
    access_json TEXT NOT NULL DEFAULT '{}',
    progress_json TEXT NOT NULL DEFAULT '{}',
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS training_capacity_checks (
    id TEXT PRIMARY KEY,
    plan_revision_id TEXT NOT NULL REFERENCES training_plan_revisions(id),
    model_preparation_id TEXT REFERENCES model_preparations(id),
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'ready', 'ready_with_adjustment', 'blocked', 'failed', 'cancelled', 'stale')
    ),
    stage TEXT NOT NULL DEFAULT 'queued',
    capability_id TEXT NOT NULL,
    capability_version TEXT NOT NULL,
    compute_shape_hash TEXT NOT NULL,
    runtime_hash TEXT NOT NULL,
    selected_adjustment_json TEXT NOT NULL DEFAULT '{}',
    forecast_json TEXT NOT NULL DEFAULT '{}',
    progress_json TEXT NOT NULL DEFAULT '{}',
    primary_remedy_json TEXT NOT NULL DEFAULT '{}',
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS training_capacity_attempts (
    id TEXT PRIMARY KEY,
    capacity_check_id TEXT NOT NULL REFERENCES training_capacity_checks(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    configuration_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'passed', 'failed', 'cancelled')),
    sample_identity_json TEXT NOT NULL DEFAULT '{}',
    measurements_json TEXT NOT NULL DEFAULT '{}',
    error_class TEXT,
    error TEXT,
    scratch_cleaned INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE (capacity_check_id, ordinal)
);

CREATE TABLE IF NOT EXISTS training_plan_decisions (
    id TEXT PRIMARY KEY,
    plan_revision_id TEXT NOT NULL REFERENCES training_plan_revisions(id),
    decision TEXT NOT NULL CHECK (
        decision IN ('confirmed', 'alternative_selected', 'override', 'proof_launched', 'full_run_derived')
    ),
    reason TEXT,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_training_plans (
    run_id TEXT PRIMARY KEY,
    plan_revision_id TEXT NOT NULL REFERENCES training_plan_revisions(id),
    capacity_check_id TEXT REFERENCES training_capacity_checks(id),
    role TEXT NOT NULL CHECK (role IN ('proof', 'full', 'manual')),
    attached_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_training_plans_dataset
    ON training_plans (dataset_version_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_plan_revisions_plan
    ON training_plan_revisions (plan_id, revision_number DESC);
CREATE INDEX IF NOT EXISTS idx_model_preparations_work
    ON model_preparations (work_item_id, status);
CREATE INDEX IF NOT EXISTS idx_capacity_checks_work
    ON training_capacity_checks (work_item_id, status);
CREATE INDEX IF NOT EXISTS idx_capacity_attempts_check
    ON training_capacity_attempts (capacity_check_id, ordinal);

CREATE TRIGGER IF NOT EXISTS immutable_training_plan_revision_update
BEFORE UPDATE ON training_plan_revisions
BEGIN
    SELECT RAISE(ABORT, 'training plan revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_training_plan_revision_delete
BEFORE DELETE ON training_plan_revisions
BEGIN
    SELECT RAISE(ABORT, 'training plan revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_training_plan_decision_update
BEFORE UPDATE ON training_plan_decisions
BEGIN
    SELECT RAISE(ABORT, 'training plan decisions are append-only');
END;
CREATE TRIGGER IF NOT EXISTS immutable_training_plan_decision_delete
BEFORE DELETE ON training_plan_decisions
BEGIN
    SELECT RAISE(ABORT, 'training plan decisions are append-only');
END;

-- Halo Forge Labs v19/v20: content-addressed accelerator runtimes and
-- conservative external-occupancy decisions. Profiles are stable names;
-- revisions, qualification evidence, bindings, and preflight decisions are
-- immutable research/operational inputs.
CREATE TABLE IF NOT EXISTS managed_runtime_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    accelerator_family TEXT NOT NULL CHECK (
        accelerator_family IN ('native', 'rocm', 'cuda')
    ),
    description TEXT,
    latest_revision_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS managed_runtime_revisions (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL REFERENCES managed_runtime_profiles(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    adapter_id TEXT NOT NULL,
    adapter_version TEXT NOT NULL,
    engine TEXT NOT NULL CHECK (engine IN ('native', 'podman', 'docker')),
    base_image TEXT,
    base_image_digest TEXT,
    derived_image_ref TEXT,
    dependency_lock_json TEXT NOT NULL DEFAULT '{}',
    configuration_json TEXT NOT NULL DEFAULT '{}',
    trainer_contracts_json TEXT NOT NULL DEFAULT '[]',
    download_bytes INTEGER,
    installed_bytes INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE (profile_id, revision_number)
);

CREATE TABLE IF NOT EXISTS runtime_preparations (
    id TEXT PRIMARY KEY,
    runtime_revision_id TEXT NOT NULL REFERENCES managed_runtime_revisions(id),
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'completed', 'failed', 'cancelled', 'blocked')
    ),
    stage TEXT NOT NULL DEFAULT 'queued',
    engine TEXT NOT NULL,
    image_id TEXT,
    image_digest TEXT,
    storage_path TEXT,
    manifest_path TEXT,
    manifest_hash TEXT,
    progress_json TEXT NOT NULL DEFAULT '{}',
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS runtime_qualifications (
    id TEXT PRIMARY KEY,
    runtime_revision_id TEXT NOT NULL REFERENCES managed_runtime_revisions(id),
    preparation_id TEXT REFERENCES runtime_preparations(id),
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'vendor_supported', 'local_verified',
                   'failed', 'stale', 'cancelled', 'blocked')
    ),
    stage TEXT NOT NULL DEFAULT 'queued',
    host_identity_hash TEXT NOT NULL,
    device_identity_hash TEXT NOT NULL,
    runtime_identity_hash TEXT NOT NULL,
    qualification_hash TEXT,
    evidence_path TEXT,
    progress_json TEXT NOT NULL DEFAULT '{}',
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS runtime_qualification_steps (
    qualification_id TEXT NOT NULL REFERENCES runtime_qualifications(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    step_id TEXT NOT NULL,
    label TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'running', 'passed', 'failed', 'skipped', 'cancelled')
    ),
    command_hash TEXT,
    result_json TEXT NOT NULL DEFAULT '{}',
    log_path TEXT,
    started_at TEXT,
    completed_at TEXT,
    PRIMARY KEY (qualification_id, ordinal),
    UNIQUE (qualification_id, step_id)
);

CREATE TABLE IF NOT EXISTS runtime_bindings (
    id TEXT PRIMARY KEY,
    runtime_revision_id TEXT NOT NULL REFERENCES managed_runtime_revisions(id),
    qualification_id TEXT REFERENCES runtime_qualifications(id),
    domain_kind TEXT NOT NULL,
    domain_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'execution',
    runtime_identity_hash TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (domain_kind, domain_id, role)
);

CREATE TABLE IF NOT EXISTS accelerator_preflight_decisions (
    id TEXT PRIMARY KEY,
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    runtime_revision_id TEXT REFERENCES managed_runtime_revisions(id),
    accelerator_family TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (
        decision IN ('idle', 'waiting', 'unknown', 'contention', 'override')
    ),
    sample_count INTEGER NOT NULL,
    evidence_hash TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    override_reason TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runtime_revisions_profile
    ON managed_runtime_revisions (profile_id, revision_number DESC);
CREATE INDEX IF NOT EXISTS idx_runtime_preparations_revision
    ON runtime_preparations (runtime_revision_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runtime_qualifications_revision
    ON runtime_qualifications (runtime_revision_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runtime_qualification_steps
    ON runtime_qualification_steps (qualification_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_runtime_bindings_domain
    ON runtime_bindings (domain_kind, domain_id);
CREATE INDEX IF NOT EXISTS idx_accelerator_preflight_work
    ON accelerator_preflight_decisions (work_item_id, created_at DESC);

CREATE TRIGGER IF NOT EXISTS immutable_managed_runtime_revision_update
BEFORE UPDATE ON managed_runtime_revisions
BEGIN
    SELECT RAISE(ABORT, 'managed runtime revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_managed_runtime_revision_delete
BEFORE DELETE ON managed_runtime_revisions
BEGIN
    SELECT RAISE(ABORT, 'managed runtime revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_runtime_binding_update
BEFORE UPDATE ON runtime_bindings
BEGIN
    SELECT RAISE(ABORT, 'runtime bindings are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_runtime_binding_delete
BEFORE DELETE ON runtime_bindings
BEGIN
    SELECT RAISE(ABORT, 'runtime bindings are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_accelerator_preflight_update
BEFORE UPDATE ON accelerator_preflight_decisions
BEGIN
    SELECT RAISE(ABORT, 'accelerator preflight decisions are append-only');
END;
CREATE TRIGGER IF NOT EXISTS immutable_accelerator_preflight_delete
BEFORE DELETE ON accelerator_preflight_decisions
BEGIN
    SELECT RAISE(ABORT, 'accelerator preflight decisions are append-only');
END;

-- Halo Forge Lab v21: real, progressive training-path certification. Runtime
-- qualification proves the accelerator stack; these records separately prove
-- that a shipped Dataset Lab renderer and trainer changed parameters and
-- published a reloadable artifact. A diagnostic tensor update is intentionally
-- insufficient evidence for any row in this catalog.
CREATE TABLE IF NOT EXISTS training_path_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    scenario_revision_id TEXT,
    trainer_mode TEXT NOT NULL,
    model_id TEXT NOT NULL,
    description TEXT,
    latest_revision_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_path_profile_revisions (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL REFERENCES training_path_profiles(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    runtime_family TEXT NOT NULL CHECK (runtime_family IN ('native', 'rocm', 'cuda')),
    backend TEXT NOT NULL,
    scenario_revision_id TEXT,
    trainer_mode TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    tokenizer_processor_hash TEXT NOT NULL,
    fixture_id TEXT NOT NULL,
    fixture_hash TEXT NOT NULL,
    trainer_adapter_version TEXT NOT NULL,
    capacity_adapter_version TEXT NOT NULL,
    configuration_json TEXT NOT NULL DEFAULT '{}',
    expected_artifacts_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    UNIQUE (profile_id, revision_number)
);

CREATE TABLE IF NOT EXISTS training_path_certifications (
    id TEXT PRIMARY KEY,
    path_revision_id TEXT NOT NULL REFERENCES training_path_profile_revisions(id),
    runtime_revision_id TEXT NOT NULL REFERENCES managed_runtime_revisions(id),
    runtime_qualification_id TEXT NOT NULL REFERENCES runtime_qualifications(id),
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'waiting_for_accelerator', 'verified',
                   'failed', 'stale', 'cancelled', 'blocked')
    ),
    stage TEXT NOT NULL DEFAULT 'queued',
    host_identity_hash TEXT NOT NULL,
    device_identity_hash TEXT NOT NULL,
    runtime_identity_hash TEXT NOT NULL,
    source_identity_hash TEXT NOT NULL,
    certification_hash TEXT,
    evidence_path TEXT,
    progress_json TEXT NOT NULL DEFAULT '{}',
    resume_cursor_json TEXT NOT NULL DEFAULT '{}',
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS training_path_certification_steps (
    certification_id TEXT NOT NULL REFERENCES training_path_certifications(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    step_id TEXT NOT NULL,
    label TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'running', 'passed', 'failed', 'skipped', 'cancelled')
    ),
    input_hash TEXT,
    result_json TEXT NOT NULL DEFAULT '{}',
    evidence_hash TEXT,
    log_path TEXT,
    started_at TEXT,
    completed_at TEXT,
    PRIMARY KEY (certification_id, ordinal),
    UNIQUE (certification_id, step_id)
);

CREATE TABLE IF NOT EXISTS training_path_certification_attempts (
    id TEXT PRIMARY KEY,
    certification_id TEXT NOT NULL REFERENCES training_path_certifications(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'passed', 'failed', 'cancelled')),
    resume_from_step INTEGER NOT NULL DEFAULT 0,
    output_dir TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE (certification_id, ordinal)
);

CREATE TABLE IF NOT EXISTS training_path_evidence_bindings (
    id TEXT PRIMARY KEY,
    certification_id TEXT NOT NULL REFERENCES training_path_certifications(id),
    domain_kind TEXT NOT NULL,
    domain_id TEXT NOT NULL,
    role TEXT NOT NULL,
    evidence_hash TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (certification_id, domain_kind, domain_id, role)
);

CREATE TABLE IF NOT EXISTS workstation_certifications (
    id TEXT PRIMARY KEY,
    runtime_revision_id TEXT NOT NULL REFERENCES managed_runtime_revisions(id),
    runtime_qualification_id TEXT NOT NULL REFERENCES runtime_qualifications(id),
    instruction_path_revision_id TEXT NOT NULL REFERENCES training_path_profile_revisions(id),
    instruction_path_certification_id TEXT REFERENCES training_path_certifications(id),
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'waiting_for_accelerator', 'beta_qualified',
                   'incomplete', 'failed', 'cancelled', 'stale')
    ),
    stage TEXT NOT NULL DEFAULT 'queued',
    host_identity_hash TEXT NOT NULL,
    device_identity_hash TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    qualification_hash TEXT,
    report_path TEXT,
    support_bundle_id TEXT,
    progress_json TEXT NOT NULL DEFAULT '{}',
    resume_cursor_json TEXT NOT NULL DEFAULT '{}',
    work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_training_path_revisions_profile
    ON training_path_profile_revisions (profile_id, revision_number DESC);
CREATE INDEX IF NOT EXISTS idx_training_path_certifications_revision
    ON training_path_certifications (path_revision_id, runtime_revision_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_path_certifications_work
    ON training_path_certifications (work_item_id, status);
CREATE INDEX IF NOT EXISTS idx_training_path_steps
    ON training_path_certification_steps (certification_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_training_path_evidence_domain
    ON training_path_evidence_bindings (domain_kind, domain_id);
CREATE INDEX IF NOT EXISTS idx_workstation_certifications_work
    ON workstation_certifications (work_item_id, status);

CREATE TRIGGER IF NOT EXISTS immutable_training_path_revision_update
BEFORE UPDATE ON training_path_profile_revisions
BEGIN
    SELECT RAISE(ABORT, 'training path profile revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_training_path_revision_delete
BEFORE DELETE ON training_path_profile_revisions
BEGIN
    SELECT RAISE(ABORT, 'training path profile revisions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_training_path_evidence_update
BEFORE UPDATE ON training_path_evidence_bindings
BEGIN
    SELECT RAISE(ABORT, 'training path evidence bindings are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_training_path_evidence_delete
BEFORE DELETE ON training_path_evidence_bindings
BEGIN
    SELECT RAISE(ABORT, 'training path evidence bindings are immutable');
END;
"""


def initial_meta_rows() -> list[tuple[str, str]]:
    """Rows the schema needs to be self-describing on disk."""
    return [("schema_version", str(SCHEMA_VERSION))]
