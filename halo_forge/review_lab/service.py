"""Application service for reviewed acquisition, annotation, and label publication."""

from __future__ import annotations

import copy
import json
import os
import shutil
import sqlite3
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from halo_forge.run_db.db import RunDatabase
from halo_forge.verifier_lab.failure_mining import VERIFIER_FAILURE_SELECTORS

from ._canonical import bytes_hash, canonical_json, content_hash, stable_id, utc_now
from .acquisition import STRATEGY_KINDS, comparison_records, plan_acquisition
from .acquisition_storage import AcquisitionManifestStore
from .errors import (
    ReviewConflictError,
    ReviewIntegrityError,
    ReviewStateError,
    ReviewValidationError,
)
from .models import (
    AcquisitionBatch,
    AcquisitionCandidate,
    AcquisitionStrategy,
    AnnotationSchema,
    AnnotationSchemaRevision,
    LabelSet,
    LabelSetItem,
    LabelSetRevision,
    LabelSetVerification,
    ReviewEvent,
    ReviewItem,
    ReviewPolicy,
    ReviewQueue,
    ReviewSuggestion,
)
from .registry import (
    MODALITY_TASKS,
    OutputAdapterRegistry,
    normalize_annotation,
    validate_schema_definition,
)

EVENT_TYPES = frozenset(
    {
        "label",
        "correct",
        "retract",
        "defer",
        "exclude",
        "include",
        "flag",
        "unflag",
        "note",
        "reveal_suggestion",
        "adjudicate",
    }
)
QUEUE_STATUSES = frozenset({"active", "paused", "archived"})
_RESOLVED_STATUSES = frozenset({"resolved", "excluded"})


def _loads(value: Any, default: Any) -> Any:
    try:
        return json.loads(value) if value else copy.deepcopy(default)
    except (TypeError, json.JSONDecodeError):
        return copy.deepcopy(default)


def _without_credentials(value: Any) -> Any:
    blocked = {"api_key", "token", "authorization", "password", "secret"}
    if isinstance(value, Mapping):
        return {
            str(key): _without_credentials(item)
            for key, item in value.items()
            if str(key).strip().lower() not in blocked
        }
    if isinstance(value, (list, tuple)):
        return [_without_credentials(item) for item in value]
    return copy.deepcopy(value)


def _schema(row: sqlite3.Row) -> AnnotationSchema:
    return AnnotationSchema(
        id=str(row["id"]),
        name=str(row["name"]),
        description=row["description"],
        archived=bool(row["archived"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _schema_revision(row: sqlite3.Row) -> AnnotationSchemaRevision:
    return AnnotationSchemaRevision(
        id=str(row["id"]),
        schema_id=str(row["schema_id"]),
        revision_number=int(row["revision_number"]),
        content_hash=str(row["content_hash"]),
        modality=str(row["modality"]),
        task_type=str(row["task_type"]),
        definition=_loads(row["definition_json"], {}),
        created_at=str(row["created_at"]),
    )


def _batch(row: sqlite3.Row) -> AcquisitionBatch:
    return AcquisitionBatch(
        id=str(row["id"]),
        name=str(row["name"]),
        status=str(row["status"]),
        stage=str(row["stage"]),
        request=_loads(row["request_json"], {}),
        source_hash=str(row["source_hash"]),
        content_hash=str(row["content_hash"]),
        seed=int(row["seed"]),
        row_count=int(row["row_count"]),
        processed_records=int(row["processed_records"]),
        total_records=(None if row["total_records"] is None else int(row["total_records"])),
        work_item_id=row["work_item_id"],
        error=row["error"],
        eligibility=_loads(row["eligibility_json"], {}),
        metadata=_loads(row["metadata_json"], {}),
        created_at=str(row["created_at"]),
        completed_at=row["completed_at"],
    )


def _candidate(row: sqlite3.Row) -> AcquisitionCandidate:
    return AcquisitionCandidate(
        id=str(row["id"]),
        batch_id=str(row["batch_id"]),
        ordinal=int(row["ordinal"]),
        record_id=str(row["record_id"]),
        record_hash=str(row["record_hash"]),
        source_kind=str(row["source_kind"]),
        source_ref=row["source_ref"],
        source_record_id=row["source_record_id"],
        record=_loads(row["record_json"], {}),
        evidence=_loads(row["evidence_json"], {}),
        source=_loads(row["source_json"], {}),
        stratum=str(row["stratum"]),
        score=None if row["score"] is None else float(row["score"]),
        created_at=str(row["created_at"]),
    )


def _queue(row: sqlite3.Row) -> ReviewQueue:
    return ReviewQueue(
        id=str(row["id"]),
        name=str(row["name"]),
        status=str(row["status"]),
        acquisition_batch_id=str(row["acquisition_batch_id"]),
        schema_revision_id=str(row["schema_revision_id"]),
        policy=_loads(row["policy_json"], {}),
        content_hash=str(row["content_hash"]),
        current_pass=int(row["current_pass"]),
        latest_label_set_revision_id=row["latest_label_set_revision_id"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        completed_at=row["completed_at"],
    )


def _item(row: sqlite3.Row) -> ReviewItem:
    keys = set(row.keys())
    return ReviewItem(
        id=str(row["id"]),
        queue_id=str(row["queue_id"]),
        candidate_id=str(row["candidate_id"]),
        ordinal=int(row["ordinal"]),
        status=str(row["status"]),
        active_event_id=row["active_event_id"],
        projection=_loads(row["projection_json"], {}),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        record=_loads(row["record_json"], {}) if "record_json" in keys else None,
        evidence=_loads(row["evidence_json"], {}) if "evidence_json" in keys else None,
        source=_loads(row["source_json"], {}) if "source_json" in keys else None,
        record_id=(str(row["record_id"]) if "record_id" in keys else None),
        record_hash=(str(row["record_hash"]) if "record_hash" in keys else None),
    )


def _event(row: sqlite3.Row) -> ReviewEvent:
    return ReviewEvent(
        id=str(row["id"]),
        queue_id=str(row["queue_id"]),
        item_id=str(row["item_id"]),
        event_type=str(row["event_type"]),
        pass_number=int(row["pass_number"]),
        reviewer_key=str(row["reviewer_key"]),
        idempotency_key=str(row["idempotency_key"]),
        request_hash=str(row["request_hash"]),
        expected_active_event_id=row["expected_active_event_id"],
        payload=_loads(row["payload_json"], {}),
        supersedes_event_id=row["supersedes_event_id"],
        created_at=str(row["created_at"]),
    )


def _suggestion(row: sqlite3.Row) -> ReviewSuggestion:
    return ReviewSuggestion(
        id=str(row["id"]),
        item_id=str(row["item_id"]),
        pass_number=int(row["pass_number"]),
        provider=str(row["provider"]),
        model_revision=str(row["model_revision"]),
        content_hash=str(row["content_hash"]),
        output=_loads(row["output_json"], None),
        provenance=_loads(row["provenance_json"], {}),
        created_at=str(row["created_at"]),
    )


def _label_set(row: sqlite3.Row) -> LabelSet:
    return LabelSet(
        id=str(row["id"]),
        queue_id=str(row["queue_id"]),
        name=str(row["name"]),
        latest_revision_id=row["latest_revision_id"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _label_revision(row: sqlite3.Row) -> LabelSetRevision:
    return LabelSetRevision(
        id=str(row["id"]),
        label_set_id=str(row["label_set_id"]),
        revision_number=int(row["revision_number"]),
        content_hash=str(row["content_hash"]),
        storage_path=str(row["storage_path"]),
        row_count=int(row["row_count"]),
        excluded_count=int(row["excluded_count"]),
        manifest=_loads(row["manifest_json"], {}),
        created_at=str(row["created_at"]),
    )


def _label_item(row: sqlite3.Row) -> LabelSetItem:
    return LabelSetItem(
        revision_id=str(row["revision_id"]),
        ordinal=int(row["ordinal"]),
        review_item_id=str(row["review_item_id"]),
        record_id=str(row["record_id"]),
        record_hash=str(row["record_hash"]),
        annotation=_loads(row["annotation_json"], {}),
        output_records=_loads(row["output_records_json"], []),
        lineage=_loads(row["lineage_json"], {}),
        excluded=bool(row["excluded"]),
        exclusion_reason=row["exclusion_reason"],
    )


@dataclass(frozen=True)
class _LabelSetPublicationSnapshot:
    """All mutable review state captured by one SQLite read transaction."""

    queue: ReviewQueue
    schema_revision: AnnotationSchemaRevision
    items: List[LabelSetItem]
    rendering: Dict[str, Any]
    statistics: Dict[str, Any]
    acquisition: Optional[AcquisitionBatch]
    suggestions: Dict[str, ReviewSuggestion]


class ReviewLabService:
    """One transport-neutral facade over the complete reviewed-data lifecycle."""

    def __init__(
        self,
        db: RunDatabase,
        root: str | Path | None = None,
        *,
        output_adapters: Optional[OutputAdapterRegistry] = None,
    ) -> None:
        self.db = db
        self.root = Path(root or Path.home() / ".halo-forge" / "reviews").expanduser()
        self.output_adapters = output_adapters or OutputAdapterRegistry()
        self.acquisition_manifests = AcquisitionManifestStore(self.root)

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": list(MODALITY_TASKS),
            "annotation_capabilities": [
                {"id": modality, "task_types": list(tasks)}
                for modality, tasks in MODALITY_TASKS.items()
            ],
            "acquisition_strategies": sorted(STRATEGY_KINDS),
            "strategies": sorted(STRATEGY_KINDS),
            "acquisition_source_kinds": [
                "evaluation",
                "evaluation_comparison",
                "verifier_calibration",
                "reward_integrity_audit",
                "run_samples",
                "dataset_version",
                "playground_session",
                "jsonl",
            ],
            "verifier_failure_selectors": sorted(VERIFIER_FAILURE_SELECTORS),
            "review_policies": ["one_pass", "two_pass"],
            "event_types": sorted(EVENT_TYPES),
            "output_adapters": self.output_adapters.descriptors(),
            "max_event_batch_size": 1000,
            "protected_suite_purposes": [
                "operational",
                "holdout",
                "final_holdout",
                "test",
                "canary",
            ],
            "protected_splits": ["test", "canary"],
        }

    # -- annotation schemas -------------------------------------------------

    def create_schema(
        self,
        *,
        name: str,
        modality: str,
        task_type: str,
        definition: Optional[Mapping[str, Any]] = None,
        description: Optional[str] = None,
        schema_id: Optional[str] = None,
    ) -> Tuple[AnnotationSchema, AnnotationSchemaRevision]:
        if not str(name).strip():
            raise ReviewValidationError("annotation schema name is required")
        normalized = validate_schema_definition(modality, task_type, definition)
        adapter = self.output_adapters.get(str(normalized["output_adapter_id"]))
        if not adapter.compatible(str(normalized["modality"]), str(normalized["task_type"])):
            raise ReviewValidationError("configured output adapter is incompatible with schema")
        now = utc_now()
        identifier = schema_id or uuid.uuid4().hex
        revision_hash = content_hash(normalized)
        revision_id = stable_id("asr", {"schema_id": identifier, "content_hash": revision_hash})
        with self.db._lock:
            try:
                self.db._conn.execute(
                    """INSERT INTO annotation_schemas
                       (id,name,description,archived,created_at,updated_at)
                       VALUES (?,?,?,0,?,?)""",
                    (identifier, str(name).strip(), description, now, now),
                )
                self.db._conn.execute(
                    """INSERT INTO annotation_schema_revisions
                       (id,schema_id,revision_number,content_hash,modality,task_type,
                        definition_json,created_at) VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        revision_id,
                        identifier,
                        1,
                        revision_hash,
                        normalized["modality"],
                        normalized["task_type"],
                        canonical_json(normalized),
                        now,
                    ),
                )
                self.db._conn.commit()
            except Exception:
                self.db._conn.rollback()
                raise
        return self.get_schema(identifier), self.get_schema_revision(revision_id)  # type: ignore[return-value]

    def revise_schema(
        self,
        schema_id: str,
        *,
        definition: Mapping[str, Any],
        modality: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> AnnotationSchemaRevision:
        schema = self.get_schema(schema_id)
        if schema is None:
            raise ReviewValidationError(f"unknown annotation schema: {schema_id}")
        prior = self.list_schema_revisions(schema_id)
        if not prior:
            raise ReviewStateError("annotation schema has no initial revision")
        latest = prior[-1]
        normalized = validate_schema_definition(
            modality or latest.modality, task_type or latest.task_type, definition
        )
        adapter = self.output_adapters.get(str(normalized["output_adapter_id"]))
        if not adapter.compatible(str(normalized["modality"]), str(normalized["task_type"])):
            raise ReviewValidationError("configured output adapter is incompatible with schema")
        revision_hash = content_hash(normalized)
        for value in prior:
            if value.content_hash == revision_hash:
                return value
        now = utc_now()
        revision_number = latest.revision_number + 1
        identifier = stable_id("asr", {"schema_id": schema_id, "content_hash": revision_hash})
        with self.db._lock:
            self.db._conn.execute(
                """INSERT INTO annotation_schema_revisions
                   (id,schema_id,revision_number,content_hash,modality,task_type,
                    definition_json,created_at) VALUES (?,?,?,?,?,?,?,?)""",
                (
                    identifier,
                    schema_id,
                    revision_number,
                    revision_hash,
                    normalized["modality"],
                    normalized["task_type"],
                    canonical_json(normalized),
                    now,
                ),
            )
            self.db._conn.execute(
                "UPDATE annotation_schemas SET updated_at=? WHERE id=?", (now, schema_id)
            )
            self.db._conn.commit()
        revision = self.get_schema_revision(identifier)
        assert revision is not None
        return revision

    def get_schema(self, schema_id: str) -> Optional[AnnotationSchema]:
        row = self.db._conn.execute(
            "SELECT * FROM annotation_schemas WHERE id=?", (schema_id,)
        ).fetchone()
        return _schema(row) if row else None

    def list_schemas(
        self, *, archived: Optional[bool] = False, limit: int = 100, offset: int = 0
    ) -> List[AnnotationSchema]:
        clauses: List[str] = []
        params: List[Any] = []
        if archived is not None:
            clauses.append("archived=?")
            params.append(1 if archived else 0)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.extend((min(1000, max(0, int(limit))), max(0, int(offset))))
        rows = self.db._conn.execute(
            "SELECT * FROM annotation_schemas"
            + where
            + " ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [_schema(row) for row in rows]

    def get_schema_revision(self, revision_id: str) -> Optional[AnnotationSchemaRevision]:
        row = self.db._conn.execute(
            "SELECT * FROM annotation_schema_revisions WHERE id=?", (revision_id,)
        ).fetchone()
        return _schema_revision(row) if row else None

    def list_schema_revisions(self, schema_id: str) -> List[AnnotationSchemaRevision]:
        rows = self.db._conn.execute(
            "SELECT * FROM annotation_schema_revisions WHERE schema_id=? ORDER BY revision_number",
            (schema_id,),
        ).fetchall()
        return [_schema_revision(row) for row in rows]

    # -- deterministic acquisition -----------------------------------------

    def create_acquisition(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        strategies: Optional[Sequence[AcquisitionStrategy | Mapping[str, Any] | str]] = None,
        seed: int = 0,
        filters: Any = None,
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        batch_id: Optional[str] = None,
        work_item_id: Optional[str] = None,
        check_cancelled: Optional[Callable[[], None]] = None,
    ) -> AcquisitionBatch:
        plan = plan_acquisition(
            records,
            strategies=strategies,
            seed=seed,
            filters=filters,
            metadata=metadata,
            check_cancelled=check_cancelled,
        )
        selected = plan.selected
        request = plan.request
        source_hash = plan.source_hash
        batch_hash = plan.content_hash
        existing_row = self.db._conn.execute(
            "SELECT * FROM acquisition_batches WHERE content_hash=?", (batch_hash,)
        ).fetchone()
        if existing_row:
            existing = _batch(existing_row)
            self.acquisition_manifests.publish(existing.id, plan, check_cancelled=check_cancelled)
            if batch_id and batch_id != existing.id:
                pending = self.get_acquisition(batch_id)
                if pending is not None and pending.status in {"queued", "running"}:
                    alias_metadata = {
                        **pending.metadata,
                        "reused_batch_id": existing.id,
                        "canonical_content_hash": existing.content_hash,
                    }
                    with self.db._lock:
                        self.db._conn.execute(
                            """UPDATE acquisition_batches
                               SET status='reused',stage='reused',row_count=?,
                                   processed_records=?,total_records=?,error=NULL,
                                   eligibility_json=?,metadata_json=?,completed_at=?
                               WHERE id=?""",
                            (
                                existing.row_count,
                                existing.processed_records,
                                existing.total_records,
                                canonical_json(existing.eligibility),
                                canonical_json(alias_metadata),
                                utc_now(),
                                batch_id,
                            ),
                        )
                        self.db._conn.commit()
            return existing
        identifier = batch_id or plan.default_batch_id
        pending = self.get_acquisition(identifier)
        if pending is not None and pending.status == "ready":
            if pending.content_hash != batch_hash:
                raise ReviewIntegrityError(
                    "acquisition batch identifier already contains different content"
                )
            self.acquisition_manifests.publish(identifier, plan, check_cancelled=check_cancelled)
            return pending
        if pending is not None and pending.status not in {
            "queued",
            "running",
            "failed",
            "cancelled",
            "interrupted",
            "needs_reconciliation",
        }:
            raise ReviewStateError(
                f"acquisition batch cannot be finalized from status {pending.status}"
            )
        self.acquisition_manifests.publish(identifier, plan, check_cancelled=check_cancelled)
        now = utc_now()
        with self.db._lock:
            try:
                self.db._conn.execute("BEGIN IMMEDIATE")
                current = self.db._conn.execute(
                    "SELECT status FROM acquisition_batches WHERE id=?", (identifier,)
                ).fetchone()
                if current is None:
                    self.db._conn.execute(
                        """INSERT INTO acquisition_batches
                           (id,name,status,stage,request_json,source_hash,content_hash,seed,
                            row_count,processed_records,total_records,work_item_id,error,
                            eligibility_json,metadata_json,created_at,completed_at)
                           VALUES (?,?, 'ready','complete',?,?,?,?,?,?,?,?,NULL,?,?,?,?)""",
                        (
                            identifier,
                            str(name or f"Review proposal {now[:10]}").strip(),
                            canonical_json(request),
                            source_hash,
                            batch_hash,
                            int(seed),
                            len(selected),
                            int(plan.eligibility["supplied"]),
                            int(plan.eligibility["supplied"]),
                            work_item_id,
                            canonical_json(plan.eligibility),
                            canonical_json(plan.metadata),
                            now,
                            now,
                        ),
                    )
                else:
                    self.db._conn.execute(
                        """UPDATE acquisition_batches
                           SET name=?,status='ready',stage='complete',request_json=?,
                               source_hash=?,content_hash=?,seed=?,row_count=?,
                               processed_records=?,total_records=?,work_item_id=COALESCE(?,work_item_id),
                               error=NULL,eligibility_json=?,metadata_json=?,completed_at=?
                           WHERE id=?""",
                        (
                            str(name or pending.name or f"Review proposal {now[:10]}").strip(),
                            canonical_json(request),
                            source_hash,
                            batch_hash,
                            int(seed),
                            len(selected),
                            int(plan.eligibility["supplied"]),
                            int(plan.eligibility["supplied"]),
                            work_item_id,
                            canonical_json(plan.eligibility),
                            canonical_json(plan.metadata),
                            now,
                            identifier,
                        ),
                    )
                    self.db._conn.execute(
                        "DELETE FROM acquisition_candidates WHERE batch_id=?", (identifier,)
                    )
                for ordinal, value in enumerate(selected):
                    candidate_id = stable_id(
                        "cand",
                        {"batch_id": identifier, "ordinal": ordinal, "record_id": value.record_id},
                    )
                    self.db._conn.execute(
                        """INSERT INTO acquisition_candidates
                           (id,batch_id,ordinal,record_id,record_hash,source_kind,
                            source_ref,source_record_id,record_json,evidence_json,
                            source_json,stratum,score,created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            candidate_id,
                            identifier,
                            ordinal,
                            value.record_id,
                            value.record_hash,
                            value.source_kind,
                            value.source_ref,
                            value.source_record_id,
                            canonical_json(value.record),
                            canonical_json(value.evidence),
                            canonical_json(value.source),
                            value.stratum,
                            value.score,
                            now,
                        ),
                    )
                self.db._conn.commit()
            except Exception:
                self.db._conn.rollback()
                raise
        batch = self.get_acquisition(identifier)
        assert batch is not None
        return batch

    def queue_acquisition(
        self,
        *,
        batch_id: str,
        work_item_id: str,
        request: Mapping[str, Any],
        name: Optional[str] = None,
        seed: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
        total_records: Optional[int] = None,
    ) -> AcquisitionBatch:
        """Create the durable domain placeholder before source resolution starts."""

        existing = self.get_acquisition(batch_id)
        if existing is not None:
            return existing
        now = utc_now()
        pending_identity = {
            "batch_id": batch_id,
            "request": dict(request),
            "metadata": dict(metadata or {}),
        }
        with self.db._lock:
            self.db._conn.execute(
                """INSERT INTO acquisition_batches
                   (id,name,status,stage,request_json,source_hash,content_hash,seed,
                    row_count,processed_records,total_records,work_item_id,error,
                    eligibility_json,metadata_json,created_at,completed_at)
                   VALUES (?,?,'queued','resolving_sources',?,?,?,?,0,0,?,?,NULL,'{}',?,?,NULL)""",
                (
                    batch_id,
                    str(name or f"Review proposal {now[:10]}").strip(),
                    canonical_json(dict(request)),
                    content_hash({"pending_source": pending_identity}),
                    content_hash({"pending_acquisition": pending_identity}),
                    int(seed),
                    None if total_records is None else max(0, int(total_records)),
                    work_item_id,
                    canonical_json(dict(metadata or {})),
                    now,
                ),
            )
            self.db._conn.commit()
        result = self.get_acquisition(batch_id)
        assert result is not None
        return result

    def update_acquisition_progress(
        self,
        batch_id: str,
        *,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        processed_records: Optional[int] = None,
        total_records: Optional[int] = None,
    ) -> AcquisitionBatch:
        batch = self.get_acquisition(batch_id)
        if batch is None:
            raise ReviewValidationError(f"unknown acquisition batch: {batch_id}")
        changes: List[str] = []
        values: List[Any] = []
        for column, value in (
            ("status", status),
            ("stage", stage),
            ("processed_records", processed_records),
            ("total_records", total_records),
        ):
            if value is None:
                continue
            changes.append(f"{column}=?")
            values.append(value)
        if changes:
            with self.db._lock:
                self.db._conn.execute(
                    f"UPDATE acquisition_batches SET {','.join(changes)} WHERE id=?",
                    (*values, batch_id),
                )
                self.db._conn.commit()
        result = self.get_acquisition(batch_id)
        assert result is not None
        return result

    def mark_acquisition_failed(self, batch_id: str, error: str) -> AcquisitionBatch:
        now = utc_now()
        with self.db._lock:
            self.db._conn.execute(
                """UPDATE acquisition_batches
                   SET status='failed',stage='failed',error=?,completed_at=? WHERE id=?""",
                (str(error), now, batch_id),
            )
            self.db._conn.commit()
        result = self.get_acquisition(batch_id)
        if result is None:
            raise ReviewValidationError(f"unknown acquisition batch: {batch_id}")
        return result

    def cancel_acquisition(self, batch_id: str) -> AcquisitionBatch:
        batch = self.get_acquisition(batch_id)
        if batch is None:
            raise ReviewValidationError(f"unknown acquisition batch: {batch_id}")
        if batch.status == "ready":
            raise ReviewStateError("completed immutable acquisition batches cannot be cancelled")
        if batch.status != "cancelled":
            now = utc_now()
            with self.db._lock:
                self.db._conn.execute(
                    """UPDATE acquisition_batches SET status='cancelled',stage='cancelled',
                       completed_at=? WHERE id=?""",
                    (now, batch_id),
                )
                self.db._conn.commit()
        result = self.get_acquisition(batch_id)
        assert result is not None
        return result

    def retry_acquisition(self, batch_id: str) -> AcquisitionBatch:
        batch = self.get_acquisition(batch_id)
        if batch is None:
            raise ReviewValidationError(f"unknown acquisition batch: {batch_id}")
        if batch.status not in {
            "failed",
            "cancelled",
            "interrupted",
            "needs_reconciliation",
        }:
            raise ReviewStateError(f"acquisition batch is not retryable: {batch.status}")
        with self.db._lock:
            self.db._conn.execute(
                """UPDATE acquisition_batches SET status='queued',stage='resolving_sources',
                   error=NULL,completed_at=NULL WHERE id=?""",
                (batch_id,),
            )
            self.db._conn.commit()
        result = self.get_acquisition(batch_id)
        assert result is not None
        return result

    def create_evaluation_comparison_acquisition(
        self,
        base_records: Iterable[Mapping[str, Any]],
        candidate_records: Iterable[Mapping[str, Any]],
        *,
        strategies: Optional[Sequence[AcquisitionStrategy | Mapping[str, Any] | str]] = None,
        seed: int = 0,
        filters: Any = None,
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        batch_id: Optional[str] = None,
        work_item_id: Optional[str] = None,
    ) -> AcquisitionBatch:
        return self.create_acquisition(
            comparison_records(base_records, candidate_records),
            strategies=strategies,
            seed=seed,
            filters=filters,
            name=name,
            metadata={"source_type": "evaluation_comparison", **dict(metadata or {})},
            batch_id=batch_id,
            work_item_id=work_item_id,
        )

    def get_acquisition(self, batch_id: str) -> Optional[AcquisitionBatch]:
        row = self.db._conn.execute(
            "SELECT * FROM acquisition_batches WHERE id=?", (batch_id,)
        ).fetchone()
        return _batch(row) if row else None

    def list_acquisitions(
        self, *, status: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[AcquisitionBatch]:
        where = " WHERE status=?" if status else ""
        params: List[Any] = [status] if status else []
        params.extend((min(1000, max(0, int(limit))), max(0, int(offset))))
        rows = self.db._conn.execute(
            "SELECT * FROM acquisition_batches"
            + where
            + " ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [_batch(row) for row in rows]

    def list_acquisition_candidates(
        self, batch_id: str, *, limit: int = 100, offset: int = 0
    ) -> List[AcquisitionCandidate]:
        rows = self.db._conn.execute(
            """SELECT * FROM acquisition_candidates WHERE batch_id=?
               ORDER BY ordinal LIMIT ? OFFSET ?""",
            (batch_id, min(1000, max(0, int(limit))), max(0, int(offset))),
        ).fetchall()
        return [_candidate(row) for row in rows]

    def iter_acquisition_candidates(
        self, batch_id: str, *, page_size: int = 1000
    ) -> Iterator[AcquisitionCandidate]:
        """Yield every acquisition candidate while keeping API pages bounded."""

        size = min(1000, max(1, int(page_size)))
        offset = 0
        while True:
            page = self.list_acquisition_candidates(
                batch_id,
                limit=size,
                offset=offset,
            )
            if not page:
                return
            yield from page
            offset += len(page)
            if len(page) < size:
                return

    # -- review queues ------------------------------------------------------

    @staticmethod
    def _default_policy(task_type: str) -> ReviewPolicy:
        mode = (
            "two_pass"
            if task_type in {"pairwise", "ranking", "text_correction", "structured_correction"}
            else "one_pass"
        )
        return ReviewPolicy(mode=mode)

    def create_queue(
        self,
        batch_id: str,
        schema_revision_id: str,
        *,
        name: Optional[str] = None,
        policy: Optional[ReviewPolicy | Mapping[str, Any]] = None,
        queue_id: Optional[str] = None,
    ) -> ReviewQueue:
        batch = self.get_acquisition(batch_id)
        revision = self.get_schema_revision(schema_revision_id)
        if batch is None:
            raise ReviewValidationError(f"unknown acquisition batch: {batch_id}")
        if batch.status != "ready":
            raise ReviewStateError(f"acquisition batch is not ready: {batch.status}")
        manifest_path = self.acquisition_manifests.path_for(batch.id) / "manifest.json"
        if manifest_path.exists():
            verification = self.acquisition_manifests.verify(
                batch.id, expected_content_hash=batch.content_hash
            )
            if not verification.valid:
                raise ReviewIntegrityError(
                    "acquisition batch failed checksum verification: "
                    + "; ".join(verification.errors)
                )
        if revision is None:
            raise ReviewValidationError(f"unknown annotation schema revision: {schema_revision_id}")
        default_policy = self._default_policy(revision.task_type)
        resolved = ReviewPolicy.from_value(policy, default_mode=default_policy.mode)
        if resolved.mode not in {"one_pass", "two_pass"}:
            raise ReviewValidationError("review policy mode must be one_pass or two_pass")
        queue_name = str(name or f"{batch.name} review").strip()
        identity = {
            "batch_id": batch_id,
            "schema_revision_id": schema_revision_id,
            "policy": resolved.to_dict(),
            "name": queue_name,
        }
        queue_hash = content_hash(identity)
        existing_row = self.db._conn.execute(
            "SELECT * FROM review_queues WHERE content_hash=?", (queue_hash,)
        ).fetchone()
        if existing_row:
            return _queue(existing_row)
        identifier = queue_id or uuid.uuid4().hex
        now = utc_now()
        candidate_count_row = self.db._conn.execute(
            "SELECT COUNT(*) AS value FROM acquisition_candidates WHERE batch_id=?",
            (batch_id,),
        ).fetchone()
        candidate_count = int(candidate_count_row["value"] if candidate_count_row else 0)
        if candidate_count != batch.row_count:
            raise ReviewIntegrityError("acquisition candidate count does not match batch manifest")
        with self.db._lock:
            try:
                self.db._conn.execute("BEGIN IMMEDIATE")
                self.db._conn.execute(
                    """INSERT INTO review_queues
                       (id,name,status,acquisition_batch_id,schema_revision_id,
                        policy_json,content_hash,current_pass,latest_label_set_revision_id,
                        created_at,updated_at,completed_at)
                       VALUES (?,?, 'active',?,?,?,?,1,NULL,?,?,NULL)""",
                    (
                        identifier,
                        queue_name,
                        batch_id,
                        schema_revision_id,
                        canonical_json(resolved.to_dict()),
                        queue_hash,
                        now,
                        now,
                    ),
                )
                for candidate in self.iter_acquisition_candidates(batch_id):
                    item_id = stable_id(
                        "review_item", {"queue_id": identifier, "candidate_id": candidate.id}
                    )
                    projection = self._project_events(
                        [], resolved, current_pass=1, queue_id=identifier, item_id=item_id
                    )
                    self.db._conn.execute(
                        """INSERT INTO review_items
                           (id,queue_id,candidate_id,ordinal,status,active_event_id,
                            projection_json,created_at,updated_at)
                           VALUES (?,?,?,?,?,NULL,?,?,?)""",
                        (
                            item_id,
                            identifier,
                            candidate.id,
                            candidate.ordinal,
                            projection["status"],
                            canonical_json(projection),
                            now,
                            now,
                        ),
                    )
                self.db._conn.commit()
            except Exception:
                self.db._conn.rollback()
                raise
        queue = self.get_queue(identifier)
        assert queue is not None
        return queue

    def get_queue(self, queue_id: str) -> Optional[ReviewQueue]:
        row = self.db._conn.execute(
            "SELECT * FROM review_queues WHERE id=?", (queue_id,)
        ).fetchone()
        return _queue(row) if row else None

    def list_queues(
        self,
        *,
        status: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ReviewQueue]:
        clauses: List[str] = []
        params: List[Any] = []
        if status:
            clauses.append("status=?")
            params.append(status)
        search = str(query or "").strip()
        if search:
            clauses.append("(name LIKE ? OR id LIKE ?)")
            pattern = f"%{search}%"
            params.extend((pattern, pattern))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.extend((min(1000, max(0, int(limit))), max(0, int(offset))))
        rows = self.db._conn.execute(
            "SELECT * FROM review_queues" + where + " ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [_queue(row) for row in rows]

    def count_queues(
        self, *, status: Optional[str] = None, query: Optional[str] = None
    ) -> int:
        clauses: List[str] = []
        params: List[Any] = []
        if status:
            clauses.append("status=?")
            params.append(status)
        search = str(query or "").strip()
        if search:
            clauses.append("(name LIKE ? OR id LIKE ?)")
            pattern = f"%{search}%"
            params.extend((pattern, pattern))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        row = self.db._conn.execute(
            "SELECT COUNT(*) AS value FROM review_queues" + where, params
        ).fetchone()
        return int(row["value"] if row is not None else 0)

    def clone_queue(
        self,
        queue_id: str,
        *,
        name: Optional[str] = None,
        policy: Optional[ReviewPolicy | Mapping[str, Any]] = None,
    ) -> ReviewQueue:
        source = self.get_queue(queue_id)
        if source is None:
            raise ReviewValidationError(f"unknown review queue: {queue_id}")
        return self.create_queue(
            source.acquisition_batch_id,
            source.schema_revision_id,
            name=name or f"{source.name} copy",
            policy=policy or source.policy,
        )

    def _set_queue_status(self, queue_id: str, status: str) -> ReviewQueue:
        if status not in QUEUE_STATUSES:
            raise ReviewValidationError(f"unsupported review queue status: {status}")
        queue = self.get_queue(queue_id)
        if queue is None:
            raise ReviewValidationError(f"unknown review queue: {queue_id}")
        if queue.status == "archived" and status != "archived":
            raise ReviewStateError("archived review queues cannot be reopened")
        now = utc_now()
        with self.db._lock:
            self.db._conn.execute(
                "UPDATE review_queues SET status=?,updated_at=? WHERE id=?",
                (status, now, queue_id),
            )
            self.db._conn.commit()
        return self.get_queue(queue_id)  # type: ignore[return-value]

    def pause_queue(self, queue_id: str) -> ReviewQueue:
        return self._set_queue_status(queue_id, "paused")

    def resume_queue(self, queue_id: str) -> ReviewQueue:
        return self._set_queue_status(queue_id, "active")

    def archive_queue(self, queue_id: str) -> ReviewQueue:
        return self._set_queue_status(queue_id, "archived")

    @staticmethod
    def _joined_item_sql() -> str:
        return """SELECT i.*,c.record_id,c.record_hash,c.record_json,c.evidence_json,c.source_json
                  FROM review_items i JOIN acquisition_candidates c ON c.id=i.candidate_id"""

    def _sanitize_item(self, item: ReviewItem, queue: ReviewQueue) -> ReviewItem:
        projection = copy.deepcopy(item.projection)
        policy = ReviewPolicy.from_value(queue.policy)
        if (
            policy.mode == "two_pass"
            and policy.blind_second_pass
            and queue.current_pass == 2
            and not projection.get("pass_2")
            and not projection.get("adjudication")
            and projection.get("pass_1")
        ):
            projection["pass_1"] = {"hidden": True}
        return ReviewItem(**{**item.to_dict(), "projection": projection})

    def get_item(self, item_id: str, *, include_hidden: bool = False) -> Optional[ReviewItem]:
        row = self.db._conn.execute(
            self._joined_item_sql() + " WHERE i.id=?", (item_id,)
        ).fetchone()
        if not row:
            return None
        value = _item(row)
        queue = self.get_queue(value.queue_id)
        return value if include_hidden or queue is None else self._sanitize_item(value, queue)

    def list_items(
        self,
        queue_id: str,
        *,
        status: Optional[str] = None,
        pass_number: Optional[int] = None,
        query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        include_hidden: bool = False,
    ) -> List[ReviewItem]:
        queue = self.get_queue(queue_id)
        if pass_number is not None and (
            queue is None or int(pass_number) != int(queue.current_pass)
        ):
            return []
        where = " WHERE i.queue_id=?"
        params: List[Any] = [queue_id]
        if status:
            where += " AND i.status=?"
            params.append(status)
        search = str(query or "").strip()
        if search:
            pattern = f"%{search}%"
            where += (
                " AND (c.record_id LIKE ? OR c.record_json LIKE ? "
                "OR c.evidence_json LIKE ?)"
            )
            params.extend((pattern, pattern, pattern))
        params.extend((min(1000, max(0, int(limit))), max(0, int(offset))))
        rows = self.db._conn.execute(
            self._joined_item_sql() + where + " ORDER BY i.ordinal LIMIT ? OFFSET ?", params
        ).fetchall()
        values = [_item(row) for row in rows]
        if include_hidden or queue is None:
            return values
        return [self._sanitize_item(value, queue) for value in values]

    def count_items(
        self,
        queue_id: str,
        *,
        status: Optional[str] = None,
        pass_number: Optional[int] = None,
        query: Optional[str] = None,
    ) -> int:
        """Count the same indexed item scope returned by :meth:`list_items`.

        A queue has exactly one active review pass.  Treating that queue field
        as the pass predicate avoids loading or parsing every cached projection
        merely to answer a pass-specific page request.
        """

        queue = self.get_queue(queue_id)
        if queue is None:
            return 0
        if pass_number is not None and int(pass_number) != int(queue.current_pass):
            return 0
        sql = (
            "SELECT COUNT(*) AS value FROM review_items i "
            "JOIN acquisition_candidates c ON c.id=i.candidate_id WHERE i.queue_id=?"
        )
        params: List[Any] = [queue_id]
        if status:
            sql += " AND i.status=?"
            params.append(status)
        search = str(query or "").strip()
        if search:
            pattern = f"%{search}%"
            sql += (
                " AND (c.record_id LIKE ? OR c.record_json LIKE ? "
                "OR c.evidence_json LIKE ?)"
            )
            params.extend((pattern, pattern, pattern))
        row = self.db._conn.execute(sql, params).fetchone()
        return int(row["value"] if row is not None else 0)

    def next_item(
        self, queue_id: str, *, after_ordinal: Optional[int] = None
    ) -> Optional[ReviewItem]:
        params: List[Any] = [queue_id]
        where = "i.queue_id=? AND i.status NOT IN ('resolved','excluded')"
        if after_ordinal is not None:
            where += " AND i.ordinal>?"
            params.append(int(after_ordinal))
        row = self.db._conn.execute(
            self._joined_item_sql() + " WHERE " + where + " ORDER BY i.ordinal LIMIT 1",
            params,
        ).fetchone()
        if row is None and after_ordinal is not None:
            row = self.db._conn.execute(
                self._joined_item_sql()
                + " WHERE i.queue_id=? AND i.status NOT IN ('resolved','excluded') ORDER BY i.ordinal LIMIT 1",
                (queue_id,),
            ).fetchone()
        if not row:
            return None
        value = _item(row)
        queue = self.get_queue(queue_id)
        return value if queue is None else self._sanitize_item(value, queue)

    # -- append-only review state ------------------------------------------

    @staticmethod
    def _project_events(
        events: Sequence[ReviewEvent],
        policy: ReviewPolicy,
        *,
        current_pass: int,
        queue_id: str,
        item_id: str,
    ) -> Dict[str, Any]:
        decisions: Dict[int, Dict[str, Any]] = {}
        adjudication: Optional[Dict[str, Any]] = None
        excluded = False
        exclusion_reason: Optional[str] = None
        flagged = False
        deferred = False
        reveals: List[str] = []
        reveals_by_pass: Dict[int, List[str]] = {1: [], 2: []}
        correction_count = 0
        active_event_id: Optional[str] = None
        for event in events:
            active_event_id = event.id
            payload = event.payload
            if event.event_type in {"label", "correct"}:
                updated_decision = {
                    "event_id": event.id,
                    "reviewer_key": event.reviewer_key,
                    "annotation": copy.deepcopy(payload.get("annotation", {})),
                }
                if (
                    event.event_type == "correct"
                    and adjudication
                    and adjudication.get("event_id") == event.supersedes_event_id
                ):
                    adjudication = {
                        **updated_decision,
                        "reason": payload.get("reason"),
                    }
                else:
                    decisions[event.pass_number] = updated_decision
                deferred = False
                correction_count += int(event.event_type == "correct")
            elif event.event_type == "retract":
                current = decisions.get(event.pass_number)
                if current and current.get("event_id") == event.supersedes_event_id:
                    decisions.pop(event.pass_number, None)
                if adjudication and adjudication.get("event_id") == event.supersedes_event_id:
                    adjudication = None
            elif event.event_type == "adjudicate":
                adjudication = {
                    "event_id": event.id,
                    "reviewer_key": event.reviewer_key,
                    "annotation": copy.deepcopy(payload.get("annotation", {})),
                    "reason": payload.get("reason"),
                }
            elif event.event_type == "exclude":
                excluded = True
                exclusion_reason = str(payload.get("reason") or "")
            elif event.event_type == "include":
                excluded = False
                exclusion_reason = None
            elif event.event_type == "flag":
                flagged = True
            elif event.event_type == "unflag":
                flagged = False
            elif event.event_type == "defer":
                deferred = True
            elif event.event_type == "reveal_suggestion":
                suggestion_id = str(payload.get("suggestion_id") or "")
                if suggestion_id:
                    reveals.append(suggestion_id)
                    reveals_by_pass.setdefault(event.pass_number, []).append(suggestion_id)

        first = decisions.get(1)
        second = decisions.get(2)
        agreement: Optional[bool] = None
        if first and second:
            agreement = content_hash(first["annotation"]) == content_hash(second["annotation"])
        if flagged:
            status = "flagged"
        elif excluded:
            status = "excluded"
        elif policy.mode == "one_pass":
            status = "resolved" if first else "pending"
        elif current_pass == 1:
            status = "pass1_complete" if first else "pending"
        elif not second:
            status = "pending_second_pass"
        elif agreement or adjudication:
            status = "resolved"
        else:
            status = "conflict"
        if deferred and status in {"pending", "pending_second_pass"}:
            status = "deferred"
        active_annotation = None
        if status == "resolved":
            active_annotation = copy.deepcopy(
                (adjudication or second or first or {}).get("annotation")
            )
        flip = int(content_hash({"queue_id": queue_id, "item_id": item_id})[:8], 16) % 2 == 1
        return {
            "status": status,
            "current_pass": int(current_pass),
            "active_event_id": active_event_id,
            "pass_1": first,
            "pass_2": second,
            "adjudication": adjudication,
            "active_annotation": active_annotation,
            "agreement": agreement,
            "excluded": excluded,
            "exclusion_reason": exclusion_reason,
            "flagged": flagged,
            "deferred": deferred,
            "revealed_suggestion_ids": list(dict.fromkeys(reveals)),
            "revealed_suggestion_ids_by_pass": {
                str(pass_number): list(dict.fromkeys(values))
                for pass_number, values in sorted(reveals_by_pass.items())
            },
            "correction_count": correction_count,
            "presentation": {"pass_2_flip_candidates": flip},
        }

    def _events_for_item_locked(self, item_id: str) -> List[ReviewEvent]:
        rows = self.db._conn.execute(
            "SELECT * FROM review_events WHERE item_id=? ORDER BY rowid", (item_id,)
        ).fetchall()
        return [_event(row) for row in rows]

    def rebuild_queue_projections(self, queue_id: str) -> Dict[str, Any]:
        """Recreate every cached item state solely from append-only events."""

        queue = self.get_queue(queue_id)
        if queue is None:
            raise ReviewValidationError(f"unknown review queue: {queue_id}")
        policy = ReviewPolicy.from_value(queue.policy)
        now = utc_now()
        rebuilt = 0
        with self.db._lock:
            try:
                self.db._conn.execute("BEGIN IMMEDIATE")
                item_rows = self.db._conn.execute(
                    "SELECT id FROM review_items WHERE queue_id=? ORDER BY ordinal",
                    (queue_id,),
                ).fetchall()
                for item_row in item_rows:
                    item_id = str(item_row["id"])
                    projection = self._project_events(
                        self._events_for_item_locked(item_id),
                        policy,
                        current_pass=queue.current_pass,
                        queue_id=queue_id,
                        item_id=item_id,
                    )
                    self.db._conn.execute(
                        """UPDATE review_items
                           SET status=?,active_event_id=?,projection_json=?,updated_at=?
                           WHERE id=?""",
                        (
                            projection["status"],
                            projection["active_event_id"],
                            canonical_json(projection),
                            now,
                            item_id,
                        ),
                    )
                    rebuilt += 1
                self.db._conn.execute(
                    "UPDATE review_queues SET updated_at=? WHERE id=?", (now, queue_id)
                )
                self.db._conn.commit()
            except Exception:
                self.db._conn.rollback()
                raise
        return {"queue_id": queue_id, "rebuilt": rebuilt}

    def _normalize_event_payload(
        self,
        event_type: str,
        payload: Optional[Mapping[str, Any]],
        revision: AnnotationSchemaRevision,
    ) -> Dict[str, Any]:
        raw = copy.deepcopy(dict(payload or {}))
        if event_type in {"label", "correct", "adjudicate"}:
            annotation = normalize_annotation(revision.definition, raw)
            result: Dict[str, Any] = {"annotation": annotation}
            for key in ("reason", "note"):
                if raw.get(key) is not None:
                    result[key] = raw[key]
            if (
                event_type in {"correct", "adjudicate"}
                and not str(result.get("reason") or "").strip()
            ):
                raise ReviewValidationError(f"{event_type} requires a reason")
            return result
        if event_type == "exclude" and not str(raw.get("reason") or "").strip():
            raise ReviewValidationError("excluding a review item requires a reason")
        if event_type == "retract" and not str(raw.get("reason") or "").strip():
            raise ReviewValidationError("retracting a review event requires a reason")
        return raw

    def _submit_event_locked(
        self,
        *,
        item_id: str,
        event_type: str,
        payload: Optional[Mapping[str, Any]],
        idempotency_key: str,
        expected_active_event_id: Optional[str],
        pass_number: Optional[int],
        supersedes_event_id: Optional[str],
        reviewer_key: str,
    ) -> ReviewEvent:
        normalized_type = str(event_type).strip().lower().replace("-", "_")
        if normalized_type not in EVENT_TYPES:
            raise ReviewValidationError(f"unsupported review event type: {event_type}")
        if not str(idempotency_key).strip():
            raise ReviewValidationError("idempotency_key is required")
        if not str(reviewer_key).strip():
            raise ReviewValidationError("reviewer_key is required")
        row = self.db._conn.execute(
            """SELECT i.*,q.status AS queue_status,q.current_pass,q.policy_json,
                      q.schema_revision_id
               FROM review_items i JOIN review_queues q ON q.id=i.queue_id
               WHERE i.id=?""",
            (item_id,),
        ).fetchone()
        if row is None:
            raise ReviewValidationError(f"unknown review item: {item_id}")
        queue_id = str(row["queue_id"])
        if str(row["queue_status"]) != "active":
            raise ReviewStateError(f"review queue is {row['queue_status']}")
        current_pass = int(row["current_pass"])
        policy = ReviewPolicy.from_value(_loads(row["policy_json"], {}))
        resolved_pass = int(pass_number or current_pass)
        if resolved_pass not in {1, 2} or resolved_pass > policy.passes:
            raise ReviewValidationError("pass_number is incompatible with review policy")
        if (
            normalized_type not in {"adjudicate", "reveal_suggestion"}
            and resolved_pass != current_pass
        ):
            raise ReviewStateError("events may only modify the queue's current review pass")
        revision_row = self.db._conn.execute(
            "SELECT * FROM annotation_schema_revisions WHERE id=?",
            (row["schema_revision_id"],),
        ).fetchone()
        assert revision_row is not None
        revision = _schema_revision(revision_row)
        normalized_payload = self._normalize_event_payload(normalized_type, payload, revision)
        request_identity = {
            "item_id": item_id,
            "event_type": normalized_type,
            "pass_number": resolved_pass,
            "reviewer_key": str(reviewer_key),
            "payload": normalized_payload,
            "expected_active_event_id": expected_active_event_id,
            "supersedes_event_id": supersedes_event_id,
        }
        request_hash = content_hash(request_identity)
        prior = self.db._conn.execute(
            "SELECT * FROM review_events WHERE queue_id=? AND idempotency_key=?",
            (queue_id, str(idempotency_key)),
        ).fetchone()
        if prior:
            existing = _event(prior)
            if existing.request_hash != request_hash:
                raise ReviewConflictError("idempotency key was already used for another event")
            return existing
        actual_active = row["active_event_id"]
        if actual_active != expected_active_event_id:
            raise ReviewConflictError(
                f"stale review item: expected active event {expected_active_event_id!r}, "
                f"found {actual_active!r}"
            )
        events = self._events_for_item_locked(item_id)
        current_projection = self._project_events(
            events, policy, current_pass=current_pass, queue_id=queue_id, item_id=item_id
        )
        if normalized_type in {"correct", "retract"}:
            if not supersedes_event_id:
                raise ReviewValidationError(f"{normalized_type} requires supersedes_event_id")
            active_decision = current_projection.get(f"pass_{resolved_pass}")
            if current_projection.get("adjudication") and (
                normalized_type == "retract"
                or current_projection["adjudication"].get("event_id") == supersedes_event_id
            ):
                active_decision = current_projection["adjudication"]
            if not active_decision or active_decision.get("event_id") != supersedes_event_id:
                raise ReviewConflictError("superseded event is not the active decision")
        if normalized_type == "adjudicate":
            if policy.mode != "two_pass" or current_pass != 2:
                raise ReviewStateError("adjudication requires an active two-pass second review")
            if current_projection["status"] != "conflict":
                raise ReviewStateError("only conflicting two-pass items can be adjudicated")
        if normalized_type == "reveal_suggestion":
            suggestion_id = str(normalized_payload.get("suggestion_id") or "")
            suggestion = self.db._conn.execute(
                """SELECT id FROM review_suggestions
                   WHERE id=? AND item_id=? AND pass_number=?""",
                (suggestion_id, item_id, resolved_pass),
            ).fetchone()
            if suggestion is None:
                raise ReviewValidationError(
                    "unknown suggestion for this review item and review pass"
                )
        event_id = stable_id(
            "revent", {"queue_id": queue_id, "idempotency_key": str(idempotency_key)}
        )
        now = utc_now()
        self.db._conn.execute(
            """INSERT INTO review_events
               (id,queue_id,item_id,event_type,pass_number,reviewer_key,idempotency_key,
                request_hash,expected_active_event_id,payload_json,supersedes_event_id,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                event_id,
                queue_id,
                item_id,
                normalized_type,
                resolved_pass,
                str(reviewer_key).strip(),
                str(idempotency_key),
                request_hash,
                expected_active_event_id,
                canonical_json(normalized_payload),
                supersedes_event_id,
                now,
            ),
        )
        inserted = self.db._conn.execute(
            "SELECT * FROM review_events WHERE id=?", (event_id,)
        ).fetchone()
        assert inserted is not None
        event = _event(inserted)
        projection = self._project_events(
            [*events, event],
            policy,
            current_pass=current_pass,
            queue_id=queue_id,
            item_id=item_id,
        )
        self.db._conn.execute(
            """UPDATE review_items SET status=?,active_event_id=?,projection_json=?,updated_at=?
               WHERE id=?""",
            (projection["status"], event_id, canonical_json(projection), now, item_id),
        )
        self.db._conn.execute("UPDATE review_queues SET updated_at=? WHERE id=?", (now, queue_id))
        return event

    def submit_event(
        self,
        item_id: str,
        event_type: str,
        payload: Optional[Mapping[str, Any]],
        *,
        idempotency_key: str,
        expected_active_event_id: Optional[str],
        pass_number: Optional[int] = None,
        supersedes_event_id: Optional[str] = None,
        reviewer_key: str = "local",
    ) -> ReviewEvent:
        with self.db._lock:
            try:
                self.db._conn.execute("BEGIN IMMEDIATE")
                result = self._submit_event_locked(
                    item_id=item_id,
                    event_type=event_type,
                    payload=payload,
                    idempotency_key=idempotency_key,
                    expected_active_event_id=expected_active_event_id,
                    pass_number=pass_number,
                    supersedes_event_id=supersedes_event_id,
                    reviewer_key=reviewer_key,
                )
                self.db._conn.commit()
                return result
            except Exception:
                self.db._conn.rollback()
                raise

    def submit_event_batch(
        self,
        queue_id: str,
        events: Sequence[Mapping[str, Any]],
        *,
        reviewer_key: str = "local",
    ) -> List[ReviewEvent]:
        if not events or len(events) > 1000:
            raise ReviewValidationError("event batch must contain between 1 and 1000 events")
        with self.db._lock:
            try:
                self.db._conn.execute("BEGIN IMMEDIATE")
                results: List[ReviewEvent] = []
                for value in events:
                    raw = dict(value)
                    item_row = self.db._conn.execute(
                        "SELECT queue_id FROM review_items WHERE id=?", (raw.get("item_id"),)
                    ).fetchone()
                    if item_row is None or str(item_row["queue_id"]) != queue_id:
                        raise ReviewValidationError("every batch item must belong to queue_id")
                    results.append(
                        self._submit_event_locked(
                            item_id=str(raw.get("item_id")),
                            event_type=str(raw.get("event_type") or "label"),
                            payload=raw.get("payload"),
                            idempotency_key=str(raw.get("idempotency_key") or ""),
                            expected_active_event_id=raw.get("expected_active_event_id"),
                            pass_number=raw.get("pass_number"),
                            supersedes_event_id=raw.get("supersedes_event_id"),
                            reviewer_key=str(raw.get("reviewer_key") or reviewer_key),
                        )
                    )
                self.db._conn.commit()
                return results
            except Exception:
                self.db._conn.rollback()
                raise

    def adjudicate(
        self,
        item_id: str,
        annotation: Mapping[str, Any],
        *,
        reason: str,
        idempotency_key: str,
        expected_active_event_id: Optional[str],
        reviewer_key: str = "local",
    ) -> ReviewEvent:
        return self.submit_event(
            item_id,
            "adjudicate",
            {"annotation": dict(annotation), "reason": reason},
            idempotency_key=idempotency_key,
            expected_active_event_id=expected_active_event_id,
            pass_number=2,
            reviewer_key=reviewer_key,
        )

    def _hide_first_pass_events(self, item_id: str, *, include_hidden: bool) -> bool:
        if include_hidden:
            return False
        item = self.get_item(item_id, include_hidden=True)
        queue = self.get_queue(item.queue_id) if item else None
        if item is None or queue is None:
            return False
        policy = ReviewPolicy.from_value(queue.policy)
        return bool(
            policy.mode == "two_pass"
            and policy.blind_second_pass
            and queue.current_pass == 2
            and not item.projection.get("pass_2")
        )

    def list_events(
        self,
        item_id: str,
        *,
        include_hidden: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[ReviewEvent]:
        sql = "SELECT * FROM review_events WHERE item_id=?"
        params: List[Any] = [item_id]
        if self._hide_first_pass_events(item_id, include_hidden=include_hidden):
            sql += " AND pass_number<>1"
        sql += " ORDER BY created_at,id"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend((min(1000, max(0, int(limit))), max(0, int(offset))))
        elif offset:
            sql += " LIMIT -1 OFFSET ?"
            params.append(max(0, int(offset)))
        rows = self.db._conn.execute(sql, params).fetchall()
        return [_event(row) for row in rows]

    def count_events(self, item_id: str, *, include_hidden: bool = False) -> int:
        sql = "SELECT COUNT(*) AS value FROM review_events WHERE item_id=?"
        params: List[Any] = [item_id]
        if self._hide_first_pass_events(item_id, include_hidden=include_hidden):
            sql += " AND pass_number<>1"
        row = self.db._conn.execute(sql, params).fetchone()
        return int(row["value"] if row is not None else 0)

    def start_second_pass(self, queue_id: str) -> ReviewQueue:
        queue = self.get_queue(queue_id)
        if queue is None:
            raise ReviewValidationError(f"unknown review queue: {queue_id}")
        policy = ReviewPolicy.from_value(queue.policy)
        if policy.mode != "two_pass":
            raise ReviewStateError("queue does not use two-pass review")
        if queue.current_pass == 2:
            return queue
        if queue.status != "active":
            raise ReviewStateError(f"review queue is {queue.status}")
        now = utc_now()
        with self.db._lock:
            try:
                self.db._conn.execute("BEGIN IMMEDIATE")
                item_rows = self.db._conn.execute(
                    "SELECT * FROM review_items WHERE queue_id=? ORDER BY ordinal", (queue_id,)
                ).fetchall()
                for row in item_rows:
                    projection = self._project_events(
                        self._events_for_item_locked(str(row["id"])),
                        policy,
                        current_pass=1,
                        queue_id=queue_id,
                        item_id=str(row["id"]),
                    )
                    if projection["status"] not in {"pass1_complete", "excluded"}:
                        raise ReviewStateError(
                            "every unexcluded item must complete pass one before pass two"
                        )
                self.db._conn.execute(
                    "UPDATE review_queues SET current_pass=2,updated_at=? WHERE id=?",
                    (now, queue_id),
                )
                for row in item_rows:
                    projection = self._project_events(
                        self._events_for_item_locked(str(row["id"])),
                        policy,
                        current_pass=2,
                        queue_id=queue_id,
                        item_id=str(row["id"]),
                    )
                    self.db._conn.execute(
                        "UPDATE review_items SET status=?,projection_json=?,updated_at=? WHERE id=?",
                        (
                            projection["status"],
                            canonical_json(projection),
                            now,
                            row["id"],
                        ),
                    )
                self.db._conn.commit()
            except Exception:
                self.db._conn.rollback()
                raise
        return self.get_queue(queue_id)  # type: ignore[return-value]

    # Suggestions are immutable completed evidence. Queued/running generation
    # lifecycle lives in the durable WorkItem referenced by the launch spec.
    def create_suggestion(
        self,
        item_id: str,
        *,
        provider: str,
        model_revision: str,
        output: Any,
        provenance: Optional[Mapping[str, Any]] = None,
        pass_number: Optional[int] = None,
        suggestion_id: Optional[str] = None,
    ) -> ReviewSuggestion:
        item = self.get_item(item_id, include_hidden=True)
        if item is None:
            raise ReviewValidationError(f"unknown review item: {item_id}")
        queue = self.get_queue(item.queue_id)
        assert queue is not None
        policy = ReviewPolicy.from_value(queue.policy)
        if not policy.allow_suggestions:
            raise ReviewStateError("review queue policy disables model suggestions")
        resolved_pass = int(pass_number or queue.current_pass)
        if queue.status != "active":
            raise ReviewStateError(f"review queue is {queue.status}")
        if resolved_pass != queue.current_pass or resolved_pass > policy.passes:
            raise ReviewStateError(
                "suggestions may only be generated for the queue's active review pass"
            )
        safe_provenance = _without_credentials(dict(provenance or {}))
        suggestion_hash = content_hash(
            {
                "item_id": item_id,
                "pass_number": resolved_pass,
                "provider": provider,
                "model_revision": model_revision,
                "output": output,
                "provenance": safe_provenance,
            }
        )
        existing = self.db._conn.execute(
            """SELECT * FROM review_suggestions
               WHERE item_id=? AND pass_number=? AND content_hash=?""",
            (item_id, resolved_pass, suggestion_hash),
        ).fetchone()
        if existing:
            return _suggestion(existing)
        identifier = suggestion_id or stable_id("suggestion", suggestion_hash)
        now = utc_now()
        with self.db._lock:
            self.db._conn.execute(
                """INSERT INTO review_suggestions
                   (id,item_id,pass_number,provider,model_revision,content_hash,
                    output_json,provenance_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    identifier,
                    item_id,
                    resolved_pass,
                    str(provider),
                    str(model_revision),
                    suggestion_hash,
                    canonical_json(output),
                    canonical_json(safe_provenance),
                    now,
                ),
            )
            self.db._conn.commit()
        return self.get_suggestion(identifier, include_hidden=True)  # type: ignore[return-value]

    def get_suggestion(
        self, suggestion_id: str, *, include_hidden: bool = False
    ) -> Optional[ReviewSuggestion]:
        row = self.db._conn.execute(
            "SELECT * FROM review_suggestions WHERE id=?", (suggestion_id,)
        ).fetchone()
        if not row:
            return None
        value = _suggestion(row)
        if include_hidden:
            return value
        item = self.get_item(value.item_id, include_hidden=True)
        queue = self.get_queue(item.queue_id) if item is not None else None
        current_pass = queue.current_pass if queue is not None else value.pass_number
        projection = item.projection if item else {}
        by_pass = projection.get("revealed_suggestion_ids_by_pass")
        if isinstance(by_pass, Mapping):
            revealed = set(by_pass.get(str(current_pass)) or [])
        else:  # read-compatible projection created before per-pass reveal tracking
            revealed = set(projection.get("revealed_suggestion_ids") or [])
        if value.id in revealed:
            return value
        return ReviewSuggestion(**{**value.to_dict(), "output": None})

    def list_suggestions(
        self,
        item_id: str,
        *,
        pass_number: Optional[int] = None,
        include_hidden: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[ReviewSuggestion]:
        sql = "SELECT * FROM review_suggestions WHERE item_id=?"
        params: List[Any] = [item_id]
        if pass_number is not None:
            sql += " AND pass_number=?"
            params.append(int(pass_number))
        sql += " ORDER BY created_at,id"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend((min(1000, max(0, int(limit))), max(0, int(offset))))
        elif offset:
            sql += " LIMIT -1 OFFSET ?"
            params.append(max(0, int(offset)))
        rows = self.db._conn.execute(sql, params).fetchall()
        values = [_suggestion(row) for row in rows]
        if include_hidden:
            return values
        item = self.get_item(item_id, include_hidden=True)
        queue = self.get_queue(item.queue_id) if item is not None else None
        current_pass = queue.current_pass if queue is not None else int(pass_number or 1)
        projection = item.projection if item else {}
        by_pass = projection.get("revealed_suggestion_ids_by_pass")
        if isinstance(by_pass, Mapping):
            revealed = set(by_pass.get(str(current_pass)) or [])
        else:
            revealed = set(projection.get("revealed_suggestion_ids") or [])
        return [
            (
                value
                if value.id in revealed
                else ReviewSuggestion(**{**value.to_dict(), "output": None})
            )
            for value in values
        ]

    def count_suggestions(self, item_id: str, *, pass_number: Optional[int] = None) -> int:
        sql = "SELECT COUNT(*) AS value FROM review_suggestions WHERE item_id=?"
        params: List[Any] = [item_id]
        if pass_number is not None:
            sql += " AND pass_number=?"
            params.append(int(pass_number))
        row = self.db._conn.execute(sql, params).fetchone()
        return int(row["value"] if row is not None else 0)

    def statistics(self, queue_id: str) -> Dict[str, Any]:
        queue = self.get_queue(queue_id)
        if queue is None:
            raise ReviewValidationError(f"unknown review queue: {queue_id}")
        rows = self.db._conn.execute(
            "SELECT status,projection_json FROM review_items WHERE queue_id=? ORDER BY ordinal",
            (queue_id,),
        ).fetchall()
        status_counts = Counter(str(row["status"]) for row in rows)
        projections = [_loads(row["projection_json"], {}) for row in rows]
        resolved = sum(int(value.get("status") in _RESOLVED_STATUSES) for value in projections)
        paired = [value for value in projections if value.get("agreement") is not None]
        agreements = sum(int(value.get("agreement") is True) for value in paired)
        event_rows = self.db._conn.execute(
            """SELECT event_type,COUNT(*) AS count FROM review_events
               WHERE queue_id=? GROUP BY event_type""",
            (queue_id,),
        ).fetchall()
        event_counts = {str(row["event_type"]): int(row["count"]) for row in event_rows}
        class_balance: Counter[str] = Counter()
        suggestion_compared = 0
        suggestion_agreements = 0
        for projection in projections:
            annotation = projection.get("active_annotation")
            if isinstance(annotation, Mapping):
                if isinstance(annotation.get("accepted"), bool):
                    class_balance["accepted" if annotation["accepted"] else "rejected"] += 1
                elif annotation.get("label") is not None:
                    class_balance[str(annotation["label"])] += 1
                elif isinstance(annotation.get("labels"), list):
                    for label in annotation["labels"]:
                        class_balance[str(label)] += 1
            revealed = {str(value) for value in projection.get("revealed_suggestion_ids") or []}
            if not revealed or not isinstance(annotation, Mapping):
                continue
            marks = ",".join("?" for _ in revealed)
            suggestion_rows = self.db._conn.execute(
                f"SELECT output_json FROM review_suggestions WHERE id IN ({marks})",
                tuple(sorted(revealed)),
            ).fetchall()
            for suggestion_row in suggestion_rows:
                output = _loads(suggestion_row["output_json"], None)
                if isinstance(output, Mapping) and isinstance(output.get("annotation"), Mapping):
                    output = output["annotation"]
                suggestion_compared += 1
                suggestion_agreements += int(output == annotation)
        current_stream_hash = content_hash(
            [
                {
                    "item_id": row["item_id"],
                    "event_id": row["id"],
                    "request_hash": row["request_hash"],
                }
                for row in self.db._conn.execute(
                    "SELECT id,item_id,request_hash FROM review_events WHERE queue_id=? ORDER BY rowid",
                    (queue_id,),
                ).fetchall()
            ]
        )
        latest_hash = None
        if queue.latest_label_set_revision_id:
            latest = self.get_label_set_revision(queue.latest_label_set_revision_id)
            latest_hash = latest.manifest.get("event_stream_hash") if latest else None
        return {
            "queue_id": queue_id,
            "total": len(rows),
            "resolved": resolved,
            "coverage": (resolved / len(rows)) if rows else 0.0,
            "status_counts": dict(sorted(status_counts.items())),
            "excluded": status_counts.get("excluded", 0),
            "flagged": status_counts.get("flagged", 0),
            "conflicts": status_counts.get("conflict", 0),
            "two_pass_compared": len(paired),
            "two_pass_agreements": agreements,
            "two_pass_agreement_rate": (agreements / len(paired)) if paired else None,
            "class_balance": dict(sorted(class_balance.items())),
            "suggestion_compared": suggestion_compared,
            "suggestion_agreements": suggestion_agreements,
            "suggestion_agreement_rate": (
                suggestion_agreements / suggestion_compared if suggestion_compared else None
            ),
            "event_counts": event_counts,
            "correction_rate": (event_counts.get("correct", 0) / len(rows) if rows else 0.0),
            "unpublished_changes": bool(latest_hash and latest_hash != current_stream_hash),
            "event_stream_hash": current_stream_hash,
        }

    # -- immutable label-set publication ----------------------------------

    def get_label_set(self, label_set_id: str) -> Optional[LabelSet]:
        row = self.db._conn.execute(
            "SELECT * FROM label_sets WHERE id=?", (label_set_id,)
        ).fetchone()
        return _label_set(row) if row else None

    def list_label_sets(
        self,
        *,
        queue_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[LabelSet]:
        sql = "SELECT * FROM label_sets"
        params: List[Any] = []
        if queue_id:
            sql += " WHERE queue_id=?"
            params.append(queue_id)
        params.extend((min(1000, max(0, int(limit))), max(0, int(offset))))
        rows = self.db._conn.execute(
            sql + " ORDER BY updated_at DESC LIMIT ? OFFSET ?", params
        ).fetchall()
        return [_label_set(row) for row in rows]

    def get_label_set_revision(self, revision_id: str) -> Optional[LabelSetRevision]:
        row = self.db._conn.execute(
            "SELECT * FROM label_set_revisions WHERE id=?", (revision_id,)
        ).fetchone()
        return _label_revision(row) if row else None

    def list_label_set_revisions(self, label_set_id: str) -> List[LabelSetRevision]:
        rows = self.db._conn.execute(
            """SELECT * FROM label_set_revisions WHERE label_set_id=?
               ORDER BY revision_number""",
            (label_set_id,),
        ).fetchall()
        return [_label_revision(row) for row in rows]

    def list_label_set_items(
        self, revision_id: str, *, limit: int = 1000, offset: int = 0
    ) -> List[LabelSetItem]:
        rows = self.db._conn.execute(
            """SELECT * FROM label_set_items WHERE revision_id=?
               ORDER BY ordinal LIMIT ? OFFSET ?""",
            (revision_id, min(10000, max(0, int(limit))), max(0, int(offset))),
        ).fetchall()
        return [_label_item(row) for row in rows]

    def iter_label_set_items(
        self, revision_id: str, *, page_size: int = 1000
    ) -> Iterator[LabelSetItem]:
        """Yield every immutable label-set item without a hidden row cap."""

        size = min(10000, max(1, int(page_size)))
        offset = 0
        while True:
            page = self.list_label_set_items(
                revision_id,
                limit=size,
                offset=offset,
            )
            if not page:
                return
            yield from page
            offset += len(page)
            if len(page) < size:
                return

    def _label_set_for_queue(self, queue_id: str, name: str) -> LabelSet:
        row = self.db._conn.execute(
            "SELECT * FROM label_sets WHERE queue_id=? AND name=?", (queue_id, name)
        ).fetchone()
        if row:
            return _label_set(row)
        now = utc_now()
        identifier = stable_id("label_set", {"queue_id": queue_id, "name": name})
        with self.db._lock:
            self.db._conn.execute(
                """INSERT OR IGNORE INTO label_sets
                   (id,queue_id,name,latest_revision_id,created_at,updated_at)
                   VALUES (?,?,?,NULL,?,?)""",
                (identifier, queue_id, name, now, now),
            )
            self.db._conn.commit()
        value = self.get_label_set(identifier)
        assert value is not None
        return value

    def _publication_items(
        self,
        queue: ReviewQueue,
        revision: AnnotationSchemaRevision,
        *,
        output_adapter_id: Optional[str],
        build_mode: Optional[str],
    ) -> Tuple[List[LabelSetItem], Dict[str, Any]]:
        adapter_id = str(output_adapter_id or revision.definition.get("output_adapter_id") or "")
        adapter = self.output_adapters.get(adapter_id)
        if not adapter.compatible(revision.modality, revision.task_type):
            raise ReviewValidationError("output adapter is incompatible with annotation schema")
        resolved_mode = str(build_mode or adapter.default_build_mode)
        if resolved_mode not in adapter.build_modes:
            raise ReviewValidationError(
                f"adapter {adapter.id} does not support build mode {resolved_mode}"
            )
        rows = self.db._conn.execute(
            self._joined_item_sql() + " WHERE i.queue_id=? ORDER BY i.ordinal", (queue.id,)
        ).fetchall()
        result: List[LabelSetItem] = []
        for row in rows:
            item = _item(row)
            projection = item.projection
            if projection.get("flagged") or item.status not in _RESOLVED_STATUSES:
                raise ReviewStateError(
                    "label publication requires every item to be resolved or explicitly excluded"
                )
            excluded = bool(projection.get("excluded"))
            annotation = copy.deepcopy(projection.get("active_annotation") or {})
            if excluded:
                outputs: List[Dict[str, Any]] = []
            else:
                outputs = adapter.render(item.record or {}, annotation, build_mode=resolved_mode)
            event_rows = self.db._conn.execute(
                """SELECT id,event_type,pass_number,reviewer_key,request_hash
                   FROM review_events WHERE item_id=? ORDER BY rowid""",
                (item.id,),
            ).fetchall()
            suggestions = self.list_suggestions(item.id)
            lineage = {
                "review_item_id": item.id,
                "acquisition_batch_id": queue.acquisition_batch_id,
                "candidate_id": item.candidate_id,
                "source": copy.deepcopy(item.source or {}),
                "schema_revision_id": revision.id,
                "review_events": [dict(row) for row in event_rows],
                "suggestions": [
                    {
                        "id": value.id,
                        "content_hash": value.content_hash,
                        "provider": value.provider,
                        "model_revision": value.model_revision,
                        "revealed": value.id
                        in set(projection.get("revealed_suggestion_ids") or []),
                    }
                    for value in suggestions
                ],
            }
            result.append(
                LabelSetItem(
                    revision_id="",
                    ordinal=item.ordinal,
                    review_item_id=item.id,
                    record_id=str(item.record_id),
                    record_hash=str(item.record_hash),
                    annotation=annotation,
                    output_records=outputs,
                    lineage=lineage,
                    excluded=excluded,
                    exclusion_reason=projection.get("exclusion_reason"),
                )
            )
        return result, {
            "adapter": adapter.descriptor(),
            "build_mode": resolved_mode,
        }

    def _publication_snapshot(
        self,
        queue_id: str,
        *,
        output_adapter_id: Optional[str],
        build_mode: Optional[str],
    ) -> _LabelSetPublicationSnapshot:
        """Capture a coherent point-in-time label-set input.

        Publication performs filesystem work after this method returns, so it
        cannot hold a database transaction for the entire operation.  The
        mutable inputs that determine rendered labels and their identity are
        therefore copied while one SQLite read transaction is active.  A
        correction committed concurrently is either wholly before or wholly
        after this snapshot; it cannot contribute only its event hash,
        statistics, or lineage to an older item projection.
        """

        with self.db._lock:
            try:
                # A deferred transaction establishes the snapshot on the
                # first SELECT and, in WAL mode, still permits another
                # connection to append review events while publication reads.
                self.db._conn.execute("BEGIN")
                queue = self.get_queue(queue_id)
                if queue is None:
                    raise ReviewValidationError(f"unknown review queue: {queue_id}")
                schema_revision = self.get_schema_revision(queue.schema_revision_id)
                assert schema_revision is not None
                items, rendering = self._publication_items(
                    queue,
                    schema_revision,
                    output_adapter_id=output_adapter_id,
                    build_mode=build_mode,
                )
                statistics = self.statistics(queue_id)
                acquisition = self.get_acquisition(queue.acquisition_batch_id)
                suggestion_rows = self.db._conn.execute(
                    """SELECT s.* FROM review_suggestions s
                       JOIN review_items i ON i.id=s.item_id
                       WHERE i.queue_id=? ORDER BY s.created_at,s.id""",
                    (queue_id,),
                ).fetchall()
                suggestions = {str(row["id"]): _suggestion(row) for row in suggestion_rows}
                self.db._conn.commit()
            except Exception:
                self.db._conn.rollback()
                raise
        return _LabelSetPublicationSnapshot(
            queue=queue,
            schema_revision=schema_revision,
            items=items,
            rendering=rendering,
            statistics=statistics,
            acquisition=acquisition,
            suggestions=suggestions,
        )

    @staticmethod
    def _write_bytes(path: Path, payload: bytes) -> str:
        path.write_bytes(payload)
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        return bytes_hash(payload)

    def publish_label_set(
        self,
        queue_id: str,
        *,
        name: Optional[str] = None,
        output_adapter_id: Optional[str] = None,
        build_mode: Optional[str] = None,
        check_cancelled: Optional[Callable[[], None]] = None,
    ) -> LabelSetRevision:
        cancel = check_cancelled or (lambda: None)
        cancel()
        snapshot = self._publication_snapshot(
            queue_id,
            output_adapter_id=output_adapter_id,
            build_mode=build_mode,
        )
        queue = snapshot.queue
        schema_revision = snapshot.schema_revision
        publication_items = snapshot.items
        rendering = snapshot.rendering
        statistics = snapshot.statistics
        cancel()
        label_set = self._label_set_for_queue(queue_id, str(name or f"{queue.name} labels").strip())
        event_stream_hash = statistics["event_stream_hash"]
        identity = {
            "queue_content_hash": queue.content_hash,
            "schema_revision_hash": schema_revision.content_hash,
            "event_stream_hash": event_stream_hash,
            "rendering": rendering,
            "items": [
                {
                    "ordinal": value.ordinal,
                    "record_id": value.record_id,
                    "record_hash": value.record_hash,
                    "annotation": value.annotation,
                    "output_records": value.output_records,
                    "lineage": value.lineage,
                    "excluded": value.excluded,
                    "exclusion_reason": value.exclusion_reason,
                }
                for value in publication_items
            ],
        }
        revision_hash = content_hash(identity)
        existing_row = self.db._conn.execute(
            """SELECT * FROM label_set_revisions
               WHERE label_set_id=? AND content_hash=?""",
            (label_set.id, revision_hash),
        ).fetchone()
        if existing_row:
            existing = _label_revision(existing_row)
            verification = self.verify_label_set(existing.id)
            if not verification.valid:
                raise ReviewIntegrityError("reused label-set revision failed verification")
            return existing
        prior = self.list_label_set_revisions(label_set.id)
        revision_number = len(prior) + 1
        revision_id = stable_id(
            "label_rev", {"label_set_id": label_set.id, "content_hash": revision_hash}
        )
        final_dir = self.root / "label-sets" / label_set.id / revision_id
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        stage = final_dir.parent / f".stage-{revision_id}-{uuid.uuid4().hex}"
        stage.mkdir(parents=True, exist_ok=False)
        now = utc_now()
        try:
            canonical_lines = [
                canonical_json(record)
                for value in publication_items
                for record in value.output_records
            ]
            item_lines = [
                canonical_json(
                    {
                        "ordinal": value.ordinal,
                        "review_item_id": value.review_item_id,
                        "record_id": value.record_id,
                        "record_hash": value.record_hash,
                        "annotation": value.annotation,
                        "excluded": value.excluded,
                        "exclusion_reason": value.exclusion_reason,
                    }
                )
                for value in publication_items
            ]
            lineage_lines = [canonical_json(value.lineage) for value in publication_items]
            acquisition = snapshot.acquisition
            suggestion_provenance: List[Dict[str, Any]] = []
            exposure_identity: List[Dict[str, Any]] = []
            for value in publication_items:
                source = value.lineage.get("source")
                if isinstance(source, Mapping) and source.get("suite_revision_id"):
                    exposure_identity.append(
                        {
                            "review_item_id": value.review_item_id,
                            "record_id": value.record_id,
                            "suite_revision_id": source.get("suite_revision_id"),
                            "suite_item_id": source.get("suite_item_id")
                            or source.get("record_id")
                            or value.record_id,
                            "purpose": source.get("purpose"),
                            "source_kind": source.get("kind"),
                            "source_ref": source.get("ref"),
                        }
                    )
                for suggestion_ref in value.lineage.get("suggestions") or []:
                    suggestion = snapshot.suggestions.get(str(suggestion_ref.get("id") or ""))
                    if suggestion is None:
                        continue
                    suggestion_provenance.append(
                        {
                            "id": suggestion.id,
                            "review_item_id": value.review_item_id,
                            "content_hash": suggestion.content_hash,
                            "provider": suggestion.provider,
                            "model_revision": suggestion.model_revision,
                            "pass_number": suggestion.pass_number,
                            "revealed": bool(suggestion_ref.get("revealed")),
                            "provenance": suggestion.provenance,
                        }
                    )
            publication_provenance = {
                "queue": {
                    "id": queue.id,
                    "content_hash": queue.content_hash,
                    "policy": queue.policy,
                },
                "acquisition": acquisition.to_dict() if acquisition is not None else None,
                "annotation_schema": schema_revision.to_dict(),
                "rendering": rendering,
                "suggestions": suggestion_provenance,
            }
            payloads = {
                "canonical.jsonl": (
                    "\n".join(canonical_lines) + ("\n" if canonical_lines else "")
                ).encode("utf-8"),
                "items.jsonl": ("\n".join(item_lines) + ("\n" if item_lines else "")).encode(
                    "utf-8"
                ),
                "lineage.jsonl": (
                    "\n".join(lineage_lines) + ("\n" if lineage_lines else "")
                ).encode("utf-8"),
                "statistics.json": (canonical_json(statistics) + "\n").encode("utf-8"),
                "provenance.json": (canonical_json(publication_provenance) + "\n").encode("utf-8"),
                "exposure.json": (canonical_json(exposure_identity) + "\n").encode("utf-8"),
            }
            checksums = {
                filename: self._write_bytes(stage / filename, payload)
                for filename, payload in payloads.items()
            }
            manifest = {
                "format": "halo-forge-label-set",
                "format_version": 1,
                "label_set_id": label_set.id,
                "revision_id": revision_id,
                "revision_number": revision_number,
                "content_hash": revision_hash,
                "queue_id": queue.id,
                "acquisition_batch_id": queue.acquisition_batch_id,
                "schema_revision_id": schema_revision.id,
                "event_stream_hash": event_stream_hash,
                "row_count": len(canonical_lines),
                "review_item_count": len(publication_items),
                "excluded_count": sum(int(value.excluded) for value in publication_items),
                "rendering": rendering,
                "statistics_file": "statistics.json",
                "provenance_file": "provenance.json",
                "exposure_file": "exposure.json",
                "checksums": checksums,
                "created_at": now,
            }
            self._write_bytes(
                stage / "manifest.json", (canonical_json(manifest) + "\n").encode("utf-8")
            )
            self._write_bytes(
                stage / "checksums.json", (canonical_json(checksums) + "\n").encode("utf-8")
            )
            cancel()
            if final_dir.exists():
                existing_manifest = _loads((final_dir / "manifest.json").read_text(), {})
                if existing_manifest.get("content_hash") != revision_hash:
                    raise ReviewIntegrityError("label-set publication path contains other content")
                shutil.rmtree(stage)
            else:
                os.replace(stage, final_dir)
                try:
                    parent_fd = os.open(final_dir.parent, os.O_RDONLY)
                    try:
                        os.fsync(parent_fd)
                    finally:
                        os.close(parent_fd)
                except OSError:
                    pass
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        with self.db._lock:
            try:
                self.db._conn.execute("BEGIN IMMEDIATE")
                self.db._conn.execute(
                    """INSERT INTO label_set_revisions
                       (id,label_set_id,revision_number,content_hash,storage_path,row_count,
                        excluded_count,manifest_json,created_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        revision_id,
                        label_set.id,
                        revision_number,
                        revision_hash,
                        str(final_dir),
                        len(canonical_lines),
                        manifest["excluded_count"],
                        canonical_json(manifest),
                        now,
                    ),
                )
                for value in publication_items:
                    self.db._conn.execute(
                        """INSERT INTO label_set_items
                           (revision_id,ordinal,review_item_id,record_id,record_hash,
                            annotation_json,output_records_json,lineage_json,excluded,
                            exclusion_reason) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (
                            revision_id,
                            value.ordinal,
                            value.review_item_id,
                            value.record_id,
                            value.record_hash,
                            canonical_json(value.annotation),
                            canonical_json(value.output_records),
                            canonical_json(value.lineage),
                            1 if value.excluded else 0,
                            value.exclusion_reason,
                        ),
                    )
                self.db._conn.execute(
                    "UPDATE label_sets SET latest_revision_id=?,updated_at=? WHERE id=?",
                    (revision_id, now, label_set.id),
                )
                self.db._conn.execute(
                    """UPDATE review_queues
                       SET latest_label_set_revision_id=?,updated_at=? WHERE id=?""",
                    (revision_id, now, queue_id),
                )
                self.db._conn.commit()
            except Exception:
                self.db._conn.rollback()
                raise
        result = self.get_label_set_revision(revision_id)
        assert result is not None
        verification = self.verify_label_set(revision_id)
        if not verification.valid:
            raise ReviewIntegrityError("new label-set revision failed verification")
        return result

    def verify_label_set(self, revision_id: str) -> LabelSetVerification:
        revision = self.get_label_set_revision(revision_id)
        if revision is None:
            raise ReviewValidationError(f"unknown label-set revision: {revision_id}")
        root = Path(revision.storage_path)
        errors: List[str] = []
        observed: Dict[str, str] = {}
        if not root.is_dir():
            return LabelSetVerification(
                revision_id=revision_id,
                valid=False,
                checksums={},
                errors=["storage directory is missing"],
            )
        manifest_path = root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return LabelSetVerification(
                revision_id=revision_id,
                valid=False,
                checksums={},
                errors=[f"manifest unreadable: {exc}"],
            )
        if canonical_json(manifest) != canonical_json(revision.manifest):
            errors.append("manifest does not match immutable catalog metadata")
        if manifest.get("content_hash") != revision.content_hash:
            errors.append("manifest content hash does not match catalog")

        def manifest_count(name: str) -> Optional[int]:
            try:
                return int(manifest.get(name, -1))
            except (TypeError, ValueError):
                errors.append(f"manifest {name} is invalid")
                return None

        if manifest_count("row_count") != revision.row_count:
            errors.append("manifest row count does not match catalog")
        if manifest_count("excluded_count") != revision.excluded_count:
            errors.append("manifest excluded count does not match catalog")
        item_count_row = self.db._conn.execute(
            "SELECT COUNT(*) AS value FROM label_set_items WHERE revision_id=?",
            (revision_id,),
        ).fetchone()
        item_count = int(item_count_row["value"] if item_count_row is not None else 0)
        if manifest_count("review_item_count") != item_count:
            errors.append("manifest review item count does not match catalog")
        checksums_path = root / "checksums.json"
        try:
            checksum_document = json.loads(checksums_path.read_text(encoding="utf-8"))
            if canonical_json(checksum_document) != canonical_json(
                dict(manifest.get("checksums") or {})
            ):
                errors.append("checksums document does not match manifest")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"checksums document unreadable: {exc}")
        for filename, expected in dict(manifest.get("checksums") or {}).items():
            path = root / str(filename)
            if not path.is_file():
                errors.append(f"missing file: {filename}")
                continue
            observed[str(filename)] = bytes_hash(path.read_bytes())
            if observed[str(filename)] != str(expected):
                errors.append(f"checksum mismatch: {filename}")
        return LabelSetVerification(
            revision_id=revision_id,
            valid=not errors,
            checksums=observed,
            errors=errors,
        )

    def render_label_set(
        self,
        revision_id: str,
        *,
        output_adapter_id: Optional[str] = None,
        build_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        revision = self.get_label_set_revision(revision_id)
        if revision is None:
            raise ReviewValidationError(f"unknown label-set revision: {revision_id}")
        label_set = self.get_label_set(revision.label_set_id)
        assert label_set is not None
        queue = self.get_queue(label_set.queue_id)
        assert queue is not None
        schema_revision = self.get_schema_revision(queue.schema_revision_id)
        assert schema_revision is not None
        stored_items = list(self.iter_label_set_items(revision_id))
        configured = revision.manifest.get("rendering") or {}
        adapter_id = str(
            output_adapter_id
            or (configured.get("adapter") or {}).get("id")
            or schema_revision.definition.get("output_adapter_id")
        )
        adapter = self.output_adapters.get(adapter_id)
        resolved_mode = str(
            build_mode or configured.get("build_mode") or adapter.default_build_mode
        )
        items: List[Dict[str, Any]] = []
        records: List[Dict[str, Any]] = []
        for value in stored_items:
            if value.excluded:
                outputs: List[Dict[str, Any]] = []
            elif output_adapter_id is None and build_mode is None:
                outputs = copy.deepcopy(value.output_records)
            else:
                source = self.get_item(value.review_item_id, include_hidden=True)
                if source is None:
                    raise ReviewIntegrityError("label-set source review item is missing")
                outputs = adapter.render(
                    source.record or {}, value.annotation, build_mode=resolved_mode
                )
            records.extend(outputs)
            items.append(
                {
                    "ordinal": value.ordinal,
                    "review_item_id": value.review_item_id,
                    "record_id": value.record_id,
                    "record_hash": value.record_hash,
                    "annotation": copy.deepcopy(value.annotation),
                    "output_records": outputs,
                    "excluded": value.excluded,
                    "exclusion_reason": value.exclusion_reason,
                    "lineage": copy.deepcopy(value.lineage),
                }
            )
        return {
            "revision": revision.to_dict(),
            "output_adapter": adapter.descriptor(),
            "build_mode": resolved_mode,
            "records": records,
            "items": items,
        }

    def preview_dataset_build(
        self,
        revision_id: str,
        *,
        target_records: Optional[Sequence[Mapping[str, Any]]] = None,
        output_adapter_id: Optional[str] = None,
        build_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        from halo_forge.data_lab.identity import deterministic_record_id, record_hash

        rendered = self.render_label_set(
            revision_id,
            output_adapter_id=output_adapter_id,
            build_mode=build_mode,
        )
        target = [copy.deepcopy(dict(value)) for value in (target_records or [])]
        target_by_id = {deterministic_record_id(value): value for value in target}
        output_by_id = {deterministic_record_id(value): value for value in rendered["records"]}
        mode = str(rendered["build_mode"])
        matched = sorted(set(target_by_id).intersection(output_by_id))
        changed = [
            record_id
            for record_id in matched
            if record_hash(target_by_id[record_id]) != record_hash(output_by_id[record_id])
        ]
        if mode == "append":
            final_records = [*target, *rendered["records"]]
        elif mode in {"replace_by_record_id", "annotate"}:
            final_records = [
                copy.deepcopy(output_by_id.get(deterministic_record_id(value), value))
                for value in target
            ]
            final_records.extend(
                value for record_id, value in output_by_id.items() if record_id not in target_by_id
            )
        elif mode == "filter":
            accepted = set(output_by_id)
            final_records = (
                [value for value in target if deterministic_record_id(value) in accepted]
                if target
                else copy.deepcopy(rendered["records"])
            )
        else:
            raise ReviewValidationError(f"unsupported dataset build mode: {mode}")
        removed = sorted(set(target_by_id).difference(output_by_id)) if mode == "filter" else []
        return {
            "label_set_revision_id": revision_id,
            "build_mode": mode,
            "output_adapter": rendered["output_adapter"],
            "input_count": len(target),
            "review_output_count": len(rendered["records"]),
            "final_count": len(final_records),
            "added_record_ids": sorted(set(output_by_id).difference(target_by_id)),
            "matched_record_ids": matched,
            "changed_record_ids": changed,
            "removed_record_ids": removed,
            "records": final_records,
            "items": rendered["items"],
        }

    def build_dataset(
        self,
        revision_id: str,
        *,
        target_records: Optional[Sequence[Mapping[str, Any]]] = None,
        output_adapter_id: Optional[str] = None,
        build_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a deterministic Dataset Lab handoff; publication stays reviewed.

        Dataset Lab owns logical dataset creation, parent selection, validation,
        contamination checks, and immutable version publication.  Keeping that
        final mutation in its service avoids a second partial catalog writer.
        """

        preview = self.preview_dataset_build(
            revision_id,
            target_records=target_records,
            output_adapter_id=output_adapter_id,
            build_mode=build_mode,
        )
        return {
            "status": "ready_for_dataset_lab",
            "label_set_revision_id": revision_id,
            "records": preview["records"],
            "provenance": {
                "source": "review_label_set",
                "label_set_revision_id": revision_id,
                "build_mode": preview["build_mode"],
                "output_adapter": preview["output_adapter"],
            },
            "preview": {
                key: value for key, value in preview.items() if key not in {"records", "items"}
            },
        }

    def execute_work_item(self, work_item: str | Mapping[str, Any] | Any) -> Dict[str, Any]:
        """Execute one resolved durable-work launch specification idempotently."""

        if isinstance(work_item, str):
            record = self.db.get_work_item(work_item)
            if record is None:
                raise ReviewValidationError(f"unknown work item: {work_item}")
            payload: Mapping[str, Any] = record.to_dict()
        elif isinstance(work_item, Mapping):
            payload = work_item
        elif hasattr(work_item, "to_dict"):
            payload = work_item.to_dict()
        else:
            raise TypeError("work_item must be an id, mapping, or work-item record")
        spec: Any = payload.get("launch_spec") or payload.get("launch_spec_json") or payload
        if isinstance(spec, str):
            spec = json.loads(spec)
        if not isinstance(spec, Mapping):
            raise ReviewValidationError("review work launch_spec must be an object")
        action = (
            str(spec.get("action") or spec.get("operation") or payload.get("kind") or "")
            .strip()
            .lower()
        )
        if action == "build_review_batch":
            if spec.get("base_records") is not None or spec.get("candidate_records") is not None:
                result = self.create_evaluation_comparison_acquisition(
                    spec.get("base_records") or [],
                    spec.get("candidate_records") or [],
                    strategies=spec.get("strategies"),
                    seed=int(spec.get("seed", 0)),
                    filters=spec.get("filters"),
                    name=spec.get("name"),
                    metadata=spec.get("metadata"),
                    batch_id=spec.get("batch_id"),
                    work_item_id=str(payload.get("id") or "") or None,
                )
            else:
                result = self.create_acquisition(
                    spec.get("records") or [],
                    strategies=spec.get("strategies"),
                    seed=int(spec.get("seed", 0)),
                    filters=spec.get("filters"),
                    name=spec.get("name"),
                    metadata=spec.get("metadata"),
                    batch_id=spec.get("batch_id"),
                    work_item_id=str(payload.get("id") or "") or None,
                )
            return result.to_dict()
        if action == "generate_review_suggestion":
            if "output" not in spec:
                raise ReviewStateError(
                    "suggestion work requires a provider executor to resolve output first"
                )
            return self.create_suggestion(
                str(spec.get("item_id") or ""),
                provider=str(spec.get("provider") or ""),
                model_revision=str(spec.get("model_revision") or ""),
                output=spec["output"],
                provenance=spec.get("provenance"),
                pass_number=spec.get("pass_number"),
                suggestion_id=spec.get("suggestion_id"),
            ).to_dict()
        if action == "publish_label_set":
            return self.publish_label_set(
                str(spec.get("queue_id") or payload.get("domain_id") or ""),
                name=spec.get("name"),
                output_adapter_id=spec.get("output_adapter_id"),
                build_mode=spec.get("build_mode"),
            ).to_dict()
        raise ReviewValidationError(f"unsupported review work-item action: {action!r}")
