"""Lab v4 persistence facade for control-plane and artifact records.

The existing :mod:`halo_forge.run_db.db` module remains the compatibility
surface for runs, Dataset Lab, evaluations, and v3 experiments.  This facade
owns the v4 tables so callers do not need to reach into SQLite directly and so
the legacy row shapes can remain stable.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .db import RunDatabase


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _load(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class WorkerRecord:
    id: str
    status: str
    pid: Optional[int]
    pid_started_at: Optional[float]
    version: Optional[str]
    capabilities_json: str
    metadata_json: str
    started_at: str
    heartbeat_at: str
    stopped_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["capabilities"] = _load(value.pop("capabilities_json"), {})
        value["metadata"] = _load(value.pop("metadata_json"), {})
        return value


@dataclass(frozen=True)
class WorkAttemptRecord:
    id: str
    work_item_id: str
    ordinal: int
    status: str
    worker_id: Optional[str]
    worker_pid: Optional[int]
    worker_pid_started_at: Optional[float]
    claim_token: Optional[str]
    output_dir: Optional[str]
    result_json: str
    error: Optional[str]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["result"] = _load(value.pop("result_json"), {})
        return value


@dataclass(frozen=True)
class WorkEventRecord:
    sequence: int
    id: str
    work_item_id: str
    attempt_id: Optional[str]
    event_type: str
    payload_json: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["payload"] = _load(value.pop("payload_json"), {})
        return value


@dataclass(frozen=True)
class ArtifactBlobRecord:
    id: str
    content_hash: str
    artifact_type: str
    format: str
    dtype: Optional[str]
    quantization: Optional[str]
    size_bytes: int
    integrity_state: str
    manifest_json: str
    created_at: str
    last_verified_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["manifest"] = _load(value.pop("manifest_json"), {})
        return value


@dataclass(frozen=True)
class ArtifactLocationRecord:
    id: str
    blob_id: str
    path: str
    storage_mode: str
    state: str
    size_bytes: int
    metadata_json: str
    created_at: str
    last_verified_at: Optional[str]
    trash_expires_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["metadata"] = _load(value.pop("metadata_json"), {})
        return value


@dataclass(frozen=True)
class ArtifactOccurrenceRecord:
    id: str
    blob_id: str
    artifact_kind: str
    legacy_model_artifact_id: Optional[str]
    run_id: Optional[str]
    run_group_id: Optional[str]
    trial_id: Optional[str]
    trial_segment_id: Optional[str]
    model_id: str
    tokenizer_revision: Optional[str]
    chat_template_hash: Optional[str]
    backend: str
    pinned: bool
    tags_json: str
    notes: Optional[str]
    metadata_json: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["pinned"] = bool(value["pinned"])
        value["tags"] = _load(value.pop("tags_json"), [])
        value["metadata"] = _load(value.pop("metadata_json"), {})
        return value


@dataclass(frozen=True)
class ArtifactOperationRecord:
    id: str
    operation_type: str
    status: str
    operation_hash: str
    resolved_spec_json: str
    input_occurrences_json: str
    output_occurrence_id: Optional[str]
    work_item_id: Optional[str]
    logs_json: str
    result_json: str
    error: Optional[str]
    created_at: str
    updated_at: str
    started_at: Optional[str]
    completed_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        for raw, cooked, default in (
            ("resolved_spec_json", "resolved_spec", {}),
            ("input_occurrences_json", "input_occurrence_ids", []),
            ("logs_json", "logs", []),
            ("result_json", "result", {}),
        ):
            value[cooked] = _load(value.pop(raw), default)
        return value


@dataclass(frozen=True)
class QualificationProfileRevisionRecord:
    id: str
    profile_id: str
    revision_number: int
    content_hash: str
    quality_suite_revision_id: str
    operational_suite_revision_id: str
    holdout_suite_revision_id: Optional[str]
    thresholds_json: str
    target_backend: Optional[str]
    generation_settings_json: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["thresholds"] = _load(value.pop("thresholds_json"), [])
        value["generation_settings"] = _load(value.pop("generation_settings_json"), {})
        return value


@dataclass(frozen=True)
class ArtifactQualificationRecord:
    id: str
    profile_revision_id: str
    occurrence_id: str
    parent_occurrence_id: Optional[str]
    status: str
    decision: Optional[str]
    reasons_json: str
    quality_evaluation_id: Optional[str]
    performance_evaluation_id: Optional[str]
    holdout_evaluation_id: Optional[str]
    metrics_json: str
    work_item_id: Optional[str]
    created_at: str
    updated_at: str
    completed_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["reasons"] = _load(value.pop("reasons_json"), [])
        value["metrics"] = _load(value.pop("metrics_json"), {})
        return value


@dataclass(frozen=True)
class ServingProfileRevisionRecord:
    id: str
    profile_id: str
    revision_number: int
    content_hash: str
    occurrence_id: str
    backend: str
    endpoint_settings_json: str
    generation_settings_json: str
    resource_requirements_json: str
    chat_template_hash: Optional[str]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        for raw, cooked in (
            ("endpoint_settings_json", "endpoint_settings"),
            ("generation_settings_json", "generation_settings"),
            ("resource_requirements_json", "resource_requirements"),
        ):
            value[cooked] = _load(value.pop(raw), {})
        return value


class LabV4Catalog:
    """Typed access to Lab v4 operational and artifact persistence."""

    def __init__(self, database: RunDatabase):
        self.database = database

    @property
    def _conn(self):
        return self.database._conn

    @property
    def _lock(self):
        return self.database._lock

    # -- workers, attempts, and events ---------------------------------

    def register_worker(
        self,
        *,
        worker_id: str,
        pid: Optional[int],
        pid_started_at: Optional[float],
        version: Optional[str] = None,
        capabilities: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> WorkerRecord:
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO workers
                    (id, status, pid, pid_started_at, version, capabilities_json,
                     metadata_json, started_at, heartbeat_at, stopped_at)
                VALUES (?, 'online', ?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(id) DO UPDATE SET status = 'online', pid = excluded.pid,
                    pid_started_at = excluded.pid_started_at, version = excluded.version,
                    capabilities_json = excluded.capabilities_json,
                    metadata_json = excluded.metadata_json,
                    heartbeat_at = excluded.heartbeat_at, stopped_at = NULL
                """,
                (
                    worker_id,
                    pid,
                    pid_started_at,
                    version,
                    _json(dict(capabilities or {})),
                    _json(dict(metadata or {})),
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return self.get_worker(worker_id)  # type: ignore[return-value]

    def heartbeat_worker(self, worker_id: str) -> Optional[WorkerRecord]:
        with self._lock:
            changed = self._conn.execute(
                "UPDATE workers SET heartbeat_at = ?, status = 'online' WHERE id = ?",
                (_now(), worker_id),
            ).rowcount
            self._conn.commit()
        return self.get_worker(worker_id) if changed else None

    def stop_worker(self, worker_id: str) -> Optional[WorkerRecord]:
        now = _now()
        with self._lock:
            changed = self._conn.execute(
                "UPDATE workers SET status = 'offline', stopped_at = ?, "
                "heartbeat_at = ? WHERE id = ?",
                (now, now, worker_id),
            ).rowcount
            self._conn.commit()
        return self.get_worker(worker_id) if changed else None

    def get_worker(self, worker_id: str) -> Optional[WorkerRecord]:
        row = self._conn.execute("SELECT * FROM workers WHERE id = ?", (worker_id,)).fetchone()
        return WorkerRecord(**dict(row)) if row else None

    def list_workers(self, *, limit: int = 100) -> List[WorkerRecord]:
        rows = self._conn.execute(
            "SELECT * FROM workers ORDER BY heartbeat_at DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
        return [WorkerRecord(**dict(row)) for row in rows]

    def start_attempt(
        self,
        work_item_id: str,
        *,
        worker_id: Optional[str],
        worker_pid: Optional[int],
        worker_pid_started_at: Optional[float],
        claim_token: Optional[str],
        output_dir: Optional[str] = None,
    ) -> WorkAttemptRecord:
        now = _now()
        with self._lock:
            previous = self._conn.execute(
                "SELECT COALESCE(MAX(ordinal), 0) AS value FROM work_item_attempts "
                "WHERE work_item_id = ?",
                (work_item_id,),
            ).fetchone()
            ordinal = int(previous["value"]) + 1
            identifier = f"{work_item_id}-attempt-{ordinal}"
            self._conn.execute(
                """
                INSERT INTO work_item_attempts
                    (id, work_item_id, ordinal, status, worker_id, worker_pid,
                     worker_pid_started_at, claim_token, output_dir, result_json,
                     created_at, started_at)
                VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, '{}', ?, ?)
                """,
                (
                    identifier,
                    work_item_id,
                    ordinal,
                    worker_id,
                    worker_pid,
                    worker_pid_started_at,
                    claim_token,
                    output_dir,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        self.add_event(
            work_item_id,
            "attempt_started",
            attempt_id=identifier,
            payload={"ordinal": ordinal, "output_dir": output_dir},
        )
        return self.get_attempt(identifier)  # type: ignore[return-value]

    def finish_attempt(
        self,
        attempt_id: str,
        *,
        status: str,
        result: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[WorkAttemptRecord]:
        if status not in {
            "completed",
            "failed",
            "cancelled",
            "interrupted",
            "needs_reconciliation",
        }:
            raise ValueError("invalid terminal attempt status")
        with self._lock:
            row = self._conn.execute(
                "SELECT work_item_id FROM work_item_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE work_item_attempts SET status = ?, result_json = ?, error = ?, "
                "completed_at = ? WHERE id = ?",
                (status, _json(dict(result or {})), error, _now(), attempt_id),
            )
            self._conn.commit()
        self.add_event(
            str(row["work_item_id"]),
            "attempt_finished",
            attempt_id=attempt_id,
            payload={"status": status, "error": error},
        )
        return self.get_attempt(attempt_id)

    def get_attempt(self, attempt_id: str) -> Optional[WorkAttemptRecord]:
        row = self._conn.execute(
            "SELECT * FROM work_item_attempts WHERE id = ?", (attempt_id,)
        ).fetchone()
        return WorkAttemptRecord(**dict(row)) if row else None

    def list_attempts(self, work_item_id: str) -> List[WorkAttemptRecord]:
        rows = self._conn.execute(
            "SELECT * FROM work_item_attempts WHERE work_item_id = ? ORDER BY ordinal",
            (work_item_id,),
        ).fetchall()
        return [WorkAttemptRecord(**dict(row)) for row in rows]

    def add_event(
        self,
        work_item_id: str,
        event_type: str,
        *,
        attempt_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> WorkEventRecord:
        identifier = uuid.uuid4().hex
        with self._lock:
            self._conn.execute(
                "INSERT INTO work_item_events "
                "(id, work_item_id, attempt_id, event_type, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    identifier,
                    work_item_id,
                    attempt_id,
                    event_type,
                    _json(dict(payload or {})),
                    _now(),
                ),
            )
            self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM work_item_events WHERE id = ?", (identifier,)
        ).fetchone()
        assert row is not None
        return WorkEventRecord(**dict(row))

    def list_events(
        self,
        *,
        after_sequence: int = 0,
        work_item_id: Optional[str] = None,
        limit: int = 500,
    ) -> List[WorkEventRecord]:
        params: list[Any] = [max(0, int(after_sequence))]
        where = "sequence > ?"
        if work_item_id:
            where += " AND work_item_id = ?"
            params.append(work_item_id)
        params.append(max(1, min(5000, int(limit))))
        rows = self._conn.execute(
            f"SELECT * FROM work_item_events WHERE {where} ORDER BY sequence LIMIT ?",
            params,
        ).fetchall()
        return [WorkEventRecord(**dict(row)) for row in rows]

    def record_telemetry(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        """Persist one honest two-second sample; missing metrics remain NULL."""

        work_item_id = str(sample.get("work_item_id") or "")
        sampled_at = str(sample.get("sampled_at") or "")
        if not work_item_id or not sampled_at:
            raise ValueError("telemetry requires work_item_id and sampled_at")
        columns = (
            "cpu_percent",
            "process_rss_bytes",
            "system_memory_used_bytes",
            "system_memory_total_bytes",
            "gpu_percent",
            "device_memory_used_bytes",
            "device_memory_total_bytes",
            "power_watts",
            "temperature_c",
            "throughput_tokens_per_second",
        )
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO telemetry_samples
                    (work_item_id, attempt_id, sampled_at, cpu_percent,
                     process_rss_bytes, system_memory_used_bytes,
                     system_memory_total_bytes, gpu_percent,
                     device_memory_used_bytes, device_memory_total_bytes,
                     power_watts, temperature_c, throughput_tokens_per_second,
                     metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    work_item_id,
                    sample.get("attempt_id"),
                    sampled_at,
                    *(sample.get(column) for column in columns),
                    _json(dict(sample.get("metadata") or {})),
                ),
            )
            self._conn.commit()
            identifier = int(cursor.lastrowid)
        return {"id": identifier, **dict(sample)}

    def list_telemetry(
        self,
        work_item_id: str,
        *,
        after: Optional[str] = None,
        limit: int = 2000,
    ) -> List[Dict[str, Any]]:
        where = "work_item_id = ?"
        params: list[Any] = [work_item_id]
        if after:
            where += " AND sampled_at > ?"
            params.append(after)
        params.append(max(1, min(10000, int(limit))))
        rows = self._conn.execute(
            f"SELECT * FROM telemetry_samples WHERE {where} " "ORDER BY sampled_at, id LIMIT ?",
            params,
        ).fetchall()
        result = []
        for row in rows:
            value = dict(row)
            value["metadata"] = _load(value.pop("metadata_json"), {})
            result.append(value)
        return result

    def finalize_telemetry_rollup(
        self, work_item_id: str, *, attempt_id: Optional[str] = None
    ) -> Dict[str, Any]:
        clauses = ["work_item_id = ?"]
        params: list[Any] = [work_item_id]
        if attempt_id is not None:
            clauses.append("attempt_id = ?")
            params.append(attempt_id)
        rows = self._conn.execute(
            f"SELECT * FROM telemetry_samples WHERE {' AND '.join(clauses)} " "ORDER BY sampled_at",
            params,
        ).fetchall()
        metric_names = (
            "cpu_percent",
            "process_rss_bytes",
            "system_memory_used_bytes",
            "system_memory_total_bytes",
            "gpu_percent",
            "device_memory_used_bytes",
            "device_memory_total_bytes",
            "power_watts",
            "temperature_c",
            "throughput_tokens_per_second",
        )
        aggregates: Dict[str, Any] = {}
        for name in metric_names:
            values = [float(row[name]) for row in rows if row[name] is not None]
            aggregates[name] = {
                "count": len(values),
                "minimum": min(values) if values else None,
                "maximum": max(values) if values else None,
                "mean": (sum(values) / len(values)) if values else None,
            }
        started_at = str(rows[0]["sampled_at"]) if rows else None
        ended_at = str(rows[-1]["sampled_at"]) if rows else None
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO telemetry_rollups
                    (work_item_id, attempt_id, sample_count, started_at, ended_at,
                     aggregates_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(work_item_id, attempt_id) DO UPDATE SET
                    sample_count = excluded.sample_count,
                    started_at = excluded.started_at,
                    ended_at = excluded.ended_at,
                    aggregates_json = excluded.aggregates_json,
                    updated_at = excluded.updated_at
                """,
                (
                    work_item_id,
                    attempt_id,
                    len(rows),
                    started_at,
                    ended_at,
                    _json(aggregates),
                    now,
                ),
            )
            self._conn.commit()
        return {
            "work_item_id": work_item_id,
            "attempt_id": attempt_id,
            "sample_count": len(rows),
            "started_at": started_at,
            "ended_at": ended_at,
            "aggregates": aggregates,
            "updated_at": now,
        }

    def get_telemetry_rollup(
        self, work_item_id: str, *, attempt_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        if attempt_id is None:
            row = self._conn.execute(
                "SELECT * FROM telemetry_rollups WHERE work_item_id = ? "
                "ORDER BY updated_at DESC LIMIT 1",
                (work_item_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM telemetry_rollups WHERE work_item_id = ? " "AND attempt_id = ?",
                (work_item_id, attempt_id),
            ).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["aggregates"] = _load(value.pop("aggregates_json"), {})
        return value

    def prune_telemetry(self, *, before: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM telemetry_samples WHERE sampled_at < ?", (before,)
            )
            self._conn.commit()
        return int(cursor.rowcount)

    # -- artifacts ------------------------------------------------------

    def upsert_blob(
        self,
        *,
        content_hash: str,
        artifact_type: str,
        format: str,
        size_bytes: int = 0,
        dtype: Optional[str] = None,
        quantization: Optional[str] = None,
        integrity_state: str = "unverified",
        manifest: Optional[Mapping[str, Any]] = None,
        blob_id: Optional[str] = None,
    ) -> ArtifactBlobRecord:
        existing = self.find_blob(content_hash)
        if existing:
            return existing
        identifier = blob_id or f"blob-{content_hash}"
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO artifact_blobs
                    (id, content_hash, artifact_type, format, dtype, quantization,
                     size_bytes, integrity_state, manifest_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    content_hash,
                    artifact_type,
                    format,
                    dtype,
                    quantization,
                    max(0, int(size_bytes)),
                    integrity_state,
                    _json(dict(manifest or {})),
                    _now(),
                ),
            )
            self._conn.commit()
        return self.get_blob(identifier)  # type: ignore[return-value]

    def find_blob(self, content_hash: str) -> Optional[ArtifactBlobRecord]:
        row = self._conn.execute(
            "SELECT * FROM artifact_blobs WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return ArtifactBlobRecord(**dict(row)) if row else None

    def get_blob(self, blob_id: str) -> Optional[ArtifactBlobRecord]:
        row = self._conn.execute("SELECT * FROM artifact_blobs WHERE id = ?", (blob_id,)).fetchone()
        return ArtifactBlobRecord(**dict(row)) if row else None

    def verify_blob(self, blob_id: str, *, state: str) -> Optional[ArtifactBlobRecord]:
        with self._lock:
            changed = self._conn.execute(
                "UPDATE artifact_blobs SET integrity_state = ?, last_verified_at = ? "
                "WHERE id = ?",
                (state, _now(), blob_id),
            ).rowcount
            self._conn.commit()
        return self.get_blob(blob_id) if changed else None

    def add_location(
        self,
        *,
        blob_id: str,
        path: str,
        storage_mode: str,
        size_bytes: int = 0,
        state: str = "available",
        metadata: Optional[Mapping[str, Any]] = None,
        location_id: Optional[str] = None,
        trash_expires_at: Optional[str] = None,
    ) -> ArtifactLocationRecord:
        if storage_mode not in {"referenced", "managed", "trash"}:
            raise ValueError("storage_mode must be referenced, managed, or trash")
        identifier = location_id or uuid.uuid4().hex
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO artifact_locations
                    (id, blob_id, path, storage_mode, state, size_bytes,
                     metadata_json, created_at, trash_expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(blob_id, path) DO UPDATE SET
                    storage_mode = excluded.storage_mode, state = excluded.state,
                    size_bytes = excluded.size_bytes, metadata_json = excluded.metadata_json,
                    trash_expires_at = excluded.trash_expires_at
                """,
                (
                    identifier,
                    blob_id,
                    path,
                    storage_mode,
                    state,
                    max(0, int(size_bytes)),
                    _json(dict(metadata or {})),
                    _now(),
                    trash_expires_at,
                ),
            )
            self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM artifact_locations WHERE blob_id = ? AND path = ?",
            (blob_id, path),
        ).fetchone()
        assert row is not None
        return ArtifactLocationRecord(**dict(row))

    def list_locations(self, blob_id: str) -> List[ArtifactLocationRecord]:
        rows = self._conn.execute(
            "SELECT * FROM artifact_locations WHERE blob_id = ? ORDER BY created_at",
            (blob_id,),
        ).fetchall()
        return [ArtifactLocationRecord(**dict(row)) for row in rows]

    def create_occurrence(
        self,
        *,
        blob_id: str,
        artifact_kind: str,
        model_id: str,
        backend: str,
        occurrence_id: Optional[str] = None,
        run_id: Optional[str] = None,
        run_group_id: Optional[str] = None,
        trial_id: Optional[str] = None,
        trial_segment_id: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        chat_template_hash: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        legacy_model_artifact_id: Optional[str] = None,
    ) -> ArtifactOccurrenceRecord:
        allowed = {
            "checkpoint",
            "adapter",
            "final_model",
            "merged_model",
            "converted_model",
            "quantized_model",
            "export_bundle",
        }
        if artifact_kind not in allowed:
            raise ValueError(f"unsupported artifact occurrence kind: {artifact_kind}")
        identifier = occurrence_id or uuid.uuid4().hex
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO artifact_occurrences
                    (id, blob_id, artifact_kind, legacy_model_artifact_id, run_id,
                     run_group_id, trial_id, trial_segment_id, model_id,
                     tokenizer_revision, chat_template_hash, backend, metadata_json,
                     created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    blob_id,
                    artifact_kind,
                    legacy_model_artifact_id,
                    run_id,
                    run_group_id,
                    trial_id,
                    trial_segment_id,
                    model_id,
                    tokenizer_revision,
                    chat_template_hash,
                    backend,
                    _json(dict(metadata or {})),
                    _now(),
                ),
            )
            self._conn.commit()
        return self.get_occurrence(identifier)  # type: ignore[return-value]

    def get_occurrence(self, occurrence_id: str) -> Optional[ArtifactOccurrenceRecord]:
        row = self._conn.execute(
            "SELECT * FROM artifact_occurrences WHERE id = ?", (occurrence_id,)
        ).fetchone()
        if not row:
            return None
        value = dict(row)
        value["pinned"] = bool(value["pinned"])
        return ArtifactOccurrenceRecord(**value)

    def list_occurrences(
        self,
        *,
        run_id: Optional[str] = None,
        artifact_kind: Optional[str] = None,
        pinned: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ArtifactOccurrenceRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (("run_id", run_id), ("artifact_kind", artifact_kind)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if pinned is not None:
            clauses.append("pinned = ?")
            params.append(int(pinned))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend((max(1, min(1000, int(limit))), max(0, int(offset))))
        rows = self._conn.execute(
            f"SELECT * FROM artifact_occurrences {where} "
            "ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        result: list[ArtifactOccurrenceRecord] = []
        for row in rows:
            value = dict(row)
            value["pinned"] = bool(value["pinned"])
            result.append(ArtifactOccurrenceRecord(**value))
        return result

    def update_occurrence_annotations(
        self,
        occurrence_id: str,
        *,
        pinned: Optional[bool] = None,
        tags: Optional[Sequence[str]] = None,
        notes: Optional[str] = None,
    ) -> Optional[ArtifactOccurrenceRecord]:
        current = self.get_occurrence(occurrence_id)
        if current is None:
            return None
        with self._lock:
            self._conn.execute(
                "UPDATE artifact_occurrences SET pinned = ?, tags_json = ?, notes = ? "
                "WHERE id = ?",
                (
                    int(current.pinned if pinned is None else pinned),
                    current.tags_json if tags is None else _json(sorted(set(tags))),
                    current.notes if notes is None else notes,
                    occurrence_id,
                ),
            )
            self._conn.commit()
        return self.get_occurrence(occurrence_id)

    def add_edge(
        self,
        *,
        child_occurrence_id: str,
        parent_occurrence_id: str,
        relation: str,
        ordinal: int = 0,
        operation_id: Optional[str] = None,
    ) -> None:
        if child_occurrence_id == parent_occurrence_id:
            raise ValueError("artifact cannot be its own parent")
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO artifact_edges "
                "(child_occurrence_id, parent_occurrence_id, relation, ordinal, "
                "operation_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    child_occurrence_id,
                    parent_occurrence_id,
                    relation,
                    max(0, int(ordinal)),
                    operation_id,
                    _now(),
                ),
            )
            self._conn.commit()

    def lineage(self, occurrence_id: str) -> Dict[str, Any]:
        parents = self._conn.execute(
            "SELECT * FROM artifact_edges WHERE child_occurrence_id = ? "
            "ORDER BY ordinal, parent_occurrence_id",
            (occurrence_id,),
        ).fetchall()
        children = self._conn.execute(
            "SELECT * FROM artifact_edges WHERE parent_occurrence_id = ? "
            "ORDER BY created_at, child_occurrence_id",
            (occurrence_id,),
        ).fetchall()
        return {
            "occurrence_id": occurrence_id,
            "parents": [dict(row) for row in parents],
            "children": [dict(row) for row in children],
        }

    def create_operation(
        self,
        *,
        operation_type: str,
        operation_hash: str,
        resolved_spec: Mapping[str, Any],
        input_occurrence_ids: Sequence[str],
        operation_id: Optional[str] = None,
        work_item_id: Optional[str] = None,
    ) -> ArtifactOperationRecord:
        completed = self.find_completed_operation(operation_hash)
        if completed:
            return completed
        identifier = operation_id or uuid.uuid4().hex
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO artifact_operations
                    (id, operation_type, status, operation_hash, resolved_spec_json,
                     input_occurrences_json, work_item_id, logs_json, result_json,
                     created_at, updated_at)
                VALUES (?, ?, 'queued', ?, ?, ?, ?, '[]', '{}', ?, ?)
                """,
                (
                    identifier,
                    operation_type,
                    operation_hash,
                    _json(dict(resolved_spec)),
                    _json(list(input_occurrence_ids)),
                    work_item_id,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return self.get_operation(identifier)  # type: ignore[return-value]

    def get_operation(self, operation_id: str) -> Optional[ArtifactOperationRecord]:
        row = self._conn.execute(
            "SELECT * FROM artifact_operations WHERE id = ?", (operation_id,)
        ).fetchone()
        return ArtifactOperationRecord(**dict(row)) if row else None

    def find_completed_operation(self, operation_hash: str) -> Optional[ArtifactOperationRecord]:
        row = self._conn.execute(
            "SELECT * FROM artifact_operations WHERE operation_hash = ? "
            "AND status = 'completed' ORDER BY completed_at DESC LIMIT 1",
            (operation_hash,),
        ).fetchone()
        return ArtifactOperationRecord(**dict(row)) if row else None

    def list_operations(
        self, *, status: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[ArtifactOperationRecord]:
        where = "WHERE status = ?" if status else ""
        params: list[Any] = [status] if status else []
        params.extend((max(1, min(1000, int(limit))), max(0, int(offset))))
        rows = self._conn.execute(
            f"SELECT * FROM artifact_operations {where} "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [ArtifactOperationRecord(**dict(row)) for row in rows]

    def update_operation(
        self,
        operation_id: str,
        *,
        status: Optional[str] = None,
        output_occurrence_id: Optional[str] = None,
        work_item_id: Optional[str] = None,
        logs: Optional[Sequence[Any]] = None,
        result: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[ArtifactOperationRecord]:
        current = self.get_operation(operation_id)
        if current is None:
            return None
        resolved_status = status or current.status
        now = _now()
        started = current.started_at or (now if resolved_status == "running" else None)
        completed = (
            now if resolved_status in {"completed", "failed", "cancelled"} else current.completed_at
        )
        with self._lock:
            self._conn.execute(
                """
                UPDATE artifact_operations SET status = ?, output_occurrence_id = ?,
                    work_item_id = ?, logs_json = ?, result_json = ?, error = ?,
                    updated_at = ?, started_at = ?, completed_at = ? WHERE id = ?
                """,
                (
                    resolved_status,
                    output_occurrence_id or current.output_occurrence_id,
                    work_item_id or current.work_item_id,
                    current.logs_json if logs is None else _json(list(logs)),
                    current.result_json if result is None else _json(dict(result)),
                    error,
                    now,
                    started,
                    completed,
                    operation_id,
                ),
            )
            self._conn.commit()
        return self.get_operation(operation_id)

    def set_alias(
        self,
        alias: str,
        occurrence_id: str,
        *,
        override_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        alias = alias.strip().lower()
        if not alias:
            raise ValueError("artifact alias is required")
        now = _now()
        with self._lock:
            existing = self._conn.execute(
                "SELECT occurrence_id FROM artifact_aliases WHERE alias = ?", (alias,)
            ).fetchone()
            previous = str(existing["occurrence_id"]) if existing else None
            self._conn.execute(
                "INSERT INTO artifact_aliases (alias, occurrence_id, updated_at) "
                "VALUES (?, ?, ?) ON CONFLICT(alias) DO UPDATE SET "
                "occurrence_id = excluded.occurrence_id, updated_at = excluded.updated_at",
                (alias, occurrence_id, now),
            )
            self._conn.execute(
                "INSERT INTO artifact_alias_events "
                "(id, alias, previous_occurrence_id, occurrence_id, override_reason, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, alias, previous, occurrence_id, override_reason, now),
            )
            self._conn.commit()
        return {
            "alias": alias,
            "occurrence_id": occurrence_id,
            "previous_occurrence_id": previous,
            "updated_at": now,
        }

    def aliases_for(self, occurrence_id: str) -> List[str]:
        rows = self._conn.execute(
            "SELECT alias FROM artifact_aliases WHERE occurrence_id = ? ORDER BY alias",
            (occurrence_id,),
        ).fetchall()
        return [str(row["alias"]) for row in rows]

    # -- qualification and serving profiles ----------------------------

    def create_qualification_profile_revision(
        self,
        *,
        name: str,
        content_hash: str,
        quality_suite_revision_id: str,
        operational_suite_revision_id: str,
        thresholds: Sequence[Mapping[str, Any]],
        generation_settings: Mapping[str, Any],
        profile_id: Optional[str] = None,
        revision_id: Optional[str] = None,
        description: Optional[str] = None,
        holdout_suite_revision_id: Optional[str] = None,
        target_backend: Optional[str] = None,
    ) -> QualificationProfileRevisionRecord:
        now = _now()
        with self._lock:
            if profile_id is None:
                found = self._conn.execute(
                    "SELECT id FROM qualification_profiles WHERE name = ?", (name,)
                ).fetchone()
                profile_id = str(found["id"]) if found else uuid.uuid4().hex
            self._conn.execute(
                "INSERT OR IGNORE INTO qualification_profiles "
                "(id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (profile_id, name, description, now, now),
            )
            previous = self._conn.execute(
                "SELECT COALESCE(MAX(revision_number), 0) AS value "
                "FROM qualification_profile_revisions WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            ordinal = int(previous["value"]) + 1
            identifier = revision_id or uuid.uuid4().hex
            self._conn.execute(
                """
                INSERT INTO qualification_profile_revisions
                    (id, profile_id, revision_number, content_hash,
                     quality_suite_revision_id, operational_suite_revision_id,
                     holdout_suite_revision_id, thresholds_json, target_backend,
                     generation_settings_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    profile_id,
                    ordinal,
                    content_hash,
                    quality_suite_revision_id,
                    operational_suite_revision_id,
                    holdout_suite_revision_id,
                    _json(list(thresholds)),
                    target_backend,
                    _json(dict(generation_settings)),
                    now,
                ),
            )
            self._conn.execute(
                "UPDATE qualification_profiles SET latest_revision_id = ?, "
                "updated_at = ? WHERE id = ?",
                (identifier, now, profile_id),
            )
            self._conn.commit()
        return self.get_qualification_profile_revision(identifier)  # type: ignore[return-value]

    def get_qualification_profile_revision(
        self, revision_id: str
    ) -> Optional[QualificationProfileRevisionRecord]:
        row = self._conn.execute(
            "SELECT * FROM qualification_profile_revisions WHERE id = ?", (revision_id,)
        ).fetchone()
        return QualificationProfileRevisionRecord(**dict(row)) if row else None

    def list_qualification_profile_revisions(
        self, *, profile_id: Optional[str] = None
    ) -> List[QualificationProfileRevisionRecord]:
        where = "WHERE profile_id = ?" if profile_id else ""
        params = (profile_id,) if profile_id else ()
        rows = self._conn.execute(
            f"SELECT * FROM qualification_profile_revisions {where} " "ORDER BY created_at DESC",
            params,
        ).fetchall()
        return [QualificationProfileRevisionRecord(**dict(row)) for row in rows]

    def create_qualification(
        self,
        *,
        profile_revision_id: str,
        occurrence_id: str,
        parent_occurrence_id: Optional[str] = None,
        qualification_id: Optional[str] = None,
        work_item_id: Optional[str] = None,
    ) -> ArtifactQualificationRecord:
        identifier = qualification_id or uuid.uuid4().hex
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO artifact_qualifications
                    (id, profile_revision_id, occurrence_id, parent_occurrence_id,
                     status, reasons_json, metrics_json, work_item_id,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, 'queued', '[]', '{}', ?, ?, ?)
                """,
                (
                    identifier,
                    profile_revision_id,
                    occurrence_id,
                    parent_occurrence_id,
                    work_item_id,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return self.get_qualification(identifier)  # type: ignore[return-value]

    def get_qualification(self, qualification_id: str) -> Optional[ArtifactQualificationRecord]:
        row = self._conn.execute(
            "SELECT * FROM artifact_qualifications WHERE id = ?", (qualification_id,)
        ).fetchone()
        return ArtifactQualificationRecord(**dict(row)) if row else None

    def list_qualifications(
        self, *, occurrence_id: Optional[str] = None, limit: int = 100
    ) -> List[ArtifactQualificationRecord]:
        where = "WHERE occurrence_id = ?" if occurrence_id else ""
        params: list[Any] = [occurrence_id] if occurrence_id else []
        params.append(max(1, min(1000, int(limit))))
        rows = self._conn.execute(
            f"SELECT * FROM artifact_qualifications {where} " "ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [ArtifactQualificationRecord(**dict(row)) for row in rows]

    def update_qualification(
        self,
        qualification_id: str,
        *,
        status: str,
        decision: Optional[str] = None,
        reasons: Sequence[str] = (),
        metrics: Optional[Mapping[str, Any]] = None,
        quality_evaluation_id: Optional[str] = None,
        performance_evaluation_id: Optional[str] = None,
        holdout_evaluation_id: Optional[str] = None,
    ) -> Optional[ArtifactQualificationRecord]:
        completed = _now() if status in {"completed", "failed", "cancelled"} else None
        with self._lock:
            changed = self._conn.execute(
                """
                UPDATE artifact_qualifications SET status = ?, decision = ?,
                    reasons_json = ?, metrics_json = ?, quality_evaluation_id = ?,
                    performance_evaluation_id = ?, holdout_evaluation_id = ?,
                    updated_at = ?, completed_at = ? WHERE id = ?
                """,
                (
                    status,
                    decision,
                    _json(list(reasons)),
                    _json(dict(metrics or {})),
                    quality_evaluation_id,
                    performance_evaluation_id,
                    holdout_evaluation_id,
                    _now(),
                    completed,
                    qualification_id,
                ),
            ).rowcount
            self._conn.commit()
        return self.get_qualification(qualification_id) if changed else None

    def create_serving_profile_revision(
        self,
        *,
        name: str,
        content_hash: str,
        occurrence_id: str,
        backend: str,
        endpoint_settings: Mapping[str, Any],
        generation_settings: Mapping[str, Any],
        resource_requirements: Mapping[str, Any],
        profile_id: Optional[str] = None,
        revision_id: Optional[str] = None,
        chat_template_hash: Optional[str] = None,
    ) -> ServingProfileRevisionRecord:
        now = _now()
        with self._lock:
            if profile_id is None:
                found = self._conn.execute(
                    "SELECT id FROM serving_profiles WHERE name = ?", (name,)
                ).fetchone()
                profile_id = str(found["id"]) if found else uuid.uuid4().hex
            self._conn.execute(
                "INSERT OR IGNORE INTO serving_profiles "
                "(id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (profile_id, name, now, now),
            )
            previous = self._conn.execute(
                "SELECT COALESCE(MAX(revision_number), 0) AS value "
                "FROM serving_profile_revisions WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            ordinal = int(previous["value"]) + 1
            identifier = revision_id or uuid.uuid4().hex
            self._conn.execute(
                """
                INSERT INTO serving_profile_revisions
                    (id, profile_id, revision_number, content_hash, occurrence_id,
                     backend, endpoint_settings_json, generation_settings_json,
                     resource_requirements_json, chat_template_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    profile_id,
                    ordinal,
                    content_hash,
                    occurrence_id,
                    backend,
                    _json(dict(endpoint_settings)),
                    _json(dict(generation_settings)),
                    _json(dict(resource_requirements)),
                    chat_template_hash,
                    now,
                ),
            )
            self._conn.execute(
                "UPDATE serving_profiles SET latest_revision_id = ?, updated_at = ? "
                "WHERE id = ?",
                (identifier, now, profile_id),
            )
            self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM serving_profile_revisions WHERE id = ?", (identifier,)
        ).fetchone()
        assert row is not None
        return ServingProfileRevisionRecord(**dict(row))


__all__ = [
    "ArtifactBlobRecord",
    "ArtifactLocationRecord",
    "ArtifactOccurrenceRecord",
    "ArtifactOperationRecord",
    "ArtifactQualificationRecord",
    "LabV4Catalog",
    "QualificationProfileRevisionRecord",
    "ServingProfileRevisionRecord",
    "WorkerRecord",
    "WorkAttemptRecord",
    "WorkEventRecord",
]
