"""SQLite persistence for immutable verifier identities and reliability evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from halo_forge.run_db.db import RunDatabase

from .models import (
    ResolvedVerifierBinding,
    VerifierAlias,
    VerifierAliasEvent,
    VerifierCalibration,
    VerifierCalibrationMetric,
    VerifierCalibrationProtocol,
    VerifierCalibrationProtocolRevision,
    VerifierCalibrationSample,
    VerifierObservation,
    VerifierProfile,
    VerifierProfileRevision,
    VerifierQualificationDecision,
    VerifierQualificationProfile,
    VerifierQualificationProfileRevision,
    VerifierRevisionComponent,
    VerifierRewardContract,
)

VERIFIER_FAMILIES = frozenset({"deterministic", "llm_judge", "reward_model", "chain"})
SOURCE_KINDS = frozenset({"label_set", "benchmark_suite"})
PROTECTED_SOURCE_PURPOSES = frozenset(
    {
        "operational",
        "holdout",
        "final_holdout",
        "test",
        "canary",
        "protected_lineage",
        "reward_model_training",
    }
)
QUALIFICATION_TEMPLATES = frozenset({"strict_oracle", "human_aligned", "exploratory", "custom"})

_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "token",
        "authorization",
        "password",
        "passwd",
        "secret",
        "client_secret",
        "credential",
        "credentials",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _loads(value: Any, default: Any) -> Any:
    try:
        return json.loads(value) if value else copy.deepcopy(default)
    except (TypeError, json.JSONDecodeError):
        return copy.deepcopy(default)


def _is_secret_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return normalized in _SECRET_KEYS or normalized.endswith(("_secret", "_password", "_token"))


def _scrub_url(value: str) -> str:
    """Remove credentials and known credential query parameters from URLs."""

    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if not parsed.scheme or not parsed.netloc:
        if re.match(r"(?i)^\s*(bearer|basic)\s+\S+", value):
            return "[REDACTED]"
        return value
    hostname = parsed.hostname or ""
    if parsed.port:
        hostname = f"{hostname}:{parsed.port}"
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not _is_secret_key(key)
    ]
    return urlunsplit((parsed.scheme, hostname, parsed.path, urlencode(query), parsed.fragment))


def scrub_secrets(value: Any) -> Any:
    """Recursively remove credentials before hashing or persistence."""

    if isinstance(value, Mapping):
        return {
            str(key): scrub_secrets(item) for key, item in value.items() if not _is_secret_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [scrub_secrets(item) for item in value]
    if isinstance(value, str):
        return _scrub_url(value)
    return copy.deepcopy(value)


def _observation_details(value: Any) -> Dict[str, Any]:
    """Keep legacy string details readable without assuming a mapping."""

    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    return {"message": value}


def _profile(row: sqlite3.Row) -> VerifierProfile:
    return VerifierProfile(
        id=str(row["id"]),
        name=str(row["name"]),
        description=row["description"],
        latest_revision_id=row["latest_revision_id"],
        archived=bool(row["archived"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _component(row: sqlite3.Row) -> VerifierRevisionComponent:
    return VerifierRevisionComponent(
        revision_id=str(row["revision_id"]),
        ordinal=int(row["ordinal"]),
        child_revision_id=str(row["child_revision_id"]),
        weight=float(row["weight"]),
        veto=bool(row["veto"]),
        required=bool(row["required"]),
        configuration=_loads(row["configuration_json"], {}),
    )


def _profile_revision(
    row: sqlite3.Row, components: Optional[List[VerifierRevisionComponent]] = None
) -> VerifierProfileRevision:
    reward_raw = _loads(row["output_contract_json"], {})
    reward_raw.update(
        {
            "minimum": float(row["reward_min"]),
            "maximum": float(row["reward_max"]),
            "direction": str(row["reward_direction"]),
            "threshold": None if row["threshold"] is None else float(row["threshold"]),
            "tie_policy": str(row["tie_policy"]),
            "error_behavior": str(row["error_behavior"]),
        }
    )
    return VerifierProfileRevision(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        revision_number=int(row["revision_number"]),
        content_hash=str(row["content_hash"]),
        family=str(row["family"]),
        reliability_adapter_id=str(row["reliability_adapter_id"]),
        reliability_adapter_version=str(row["reliability_adapter_version"]),
        implementation_kind=str(row["implementation_kind"]),
        implementation_ref=str(row["implementation_ref"]),
        implementation_fingerprint=row["implementation_fingerprint"],
        qualifiable=bool(row["qualifiable"]),
        qualification_blockers=_loads(row["qualification_blockers_json"], []),
        modality=str(row["modality"]),
        task_type=str(row["task_type"]),
        input_mapping=_loads(row["input_mapping_json"], {}),
        reward_contract=VerifierRewardContract.from_value(reward_raw),
        definition=_loads(row["definition_json"], {}),
        sanitized_configuration_hash=str(row["sanitized_configuration_hash"]),
        runtime_contract=_loads(row["runtime_contract_json"], {}),
        runtime_contract_hash=str(row["runtime_contract_hash"]),
        created_at=str(row["created_at"]),
        components=list(components or []),
    )


def _protocol(row: sqlite3.Row) -> VerifierCalibrationProtocol:
    return VerifierCalibrationProtocol(
        id=str(row["id"]),
        name=str(row["name"]),
        description=row["description"],
        latest_revision_id=row["latest_revision_id"],
        archived=bool(row["archived"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _protocol_revision(row: sqlite3.Row) -> VerifierCalibrationProtocolRevision:
    return VerifierCalibrationProtocolRevision(
        id=str(row["id"]),
        protocol_id=str(row["protocol_id"]),
        revision_number=int(row["revision_number"]),
        content_hash=str(row["content_hash"]),
        definition=_loads(row["definition_json"], {}),
        created_at=str(row["created_at"]),
    )


def _qualification_profile(row: sqlite3.Row) -> VerifierQualificationProfile:
    return VerifierQualificationProfile(
        id=str(row["id"]),
        name=str(row["name"]),
        description=row["description"],
        latest_revision_id=row["latest_revision_id"],
        archived=bool(row["archived"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _qualification_revision(row: sqlite3.Row) -> VerifierQualificationProfileRevision:
    return VerifierQualificationProfileRevision(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        revision_number=int(row["revision_number"]),
        content_hash=str(row["content_hash"]),
        template_kind=str(row["template_kind"]),
        promotable=bool(row["promotable"]),
        requirements=_loads(row["requirements_json"], {}),
        created_at=str(row["created_at"]),
    )


def _calibration(row: sqlite3.Row) -> VerifierCalibration:
    return VerifierCalibration(
        id=str(row["id"]),
        verifier_revision_id=str(row["verifier_revision_id"]),
        protocol_revision_id=str(row["protocol_revision_id"]),
        qualification_profile_revision_id=str(row["qualification_profile_revision_id"]),
        source_kind=str(row["source_kind"]),
        source_revision_id=str(row["source_revision_id"]),
        source_hash=str(row["source_hash"]),
        source_purpose=str(row["source_purpose"]),
        status=str(row["status"]),
        stage=str(row["stage"]),
        processed_records=int(row["processed_records"]),
        total_records=None if row["total_records"] is None else int(row["total_records"]),
        sample_count=int(row["sample_count"]),
        request=_loads(row["request_json"], {}),
        partition=_loads(row["partition_json"], {}),
        runtime_identity=_loads(row["runtime_identity_json"], {}),
        runtime_identity_hash=str(row["runtime_identity_hash"]),
        protocol_hash=str(row["protocol_hash"]),
        qualification_hash=str(row["qualification_hash"]),
        reuse_key=str(row["reuse_key"]),
        artifact_path=row["artifact_path"],
        manifest_hash=row["manifest_hash"],
        work_item_id=row["work_item_id"],
        cancel_requested=bool(row["cancel_requested"]),
        retry_count=int(row["retry_count"]),
        error=row["error"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


def _sample(row: sqlite3.Row) -> VerifierCalibrationSample:
    observation_raw = _loads(row["observation_json"], {})
    observation = VerifierObservation(
        reward=None if row["reward"] is None else float(row["reward"]),
        passed=None if row["passed"] is None else bool(row["passed"]),
        parsed_value=observation_raw.get("parsed_value"),
        raw_output=observation_raw.get("raw_output"),
        details=_observation_details(observation_raw.get("details")),
        component_trace=list(observation_raw.get("component_trace") or []),
        latency_ms=None if row["latency_ms"] is None else float(row["latency_ms"]),
        error=row["error"],
        runtime_identity=_loads(row["runtime_identity_json"], {}),
    )
    return VerifierCalibrationSample(
        calibration_id=str(row["calibration_id"]),
        ordinal=int(row["ordinal"]),
        record_id=str(row["record_id"]),
        record_hash=str(row["record_hash"]),
        group_id=str(row["group_id"]),
        partition=str(row["partition"]),
        repeat_index=int(row["repeat_index"]),
        orientation=str(row["orientation"]),
        probe_kind=str(row["probe_kind"]),
        seed=None if row["seed"] is None else int(row["seed"]),
        reference=_loads(row["reference_json"], {}),
        observation=observation,
        metadata=_loads(row["metadata_json"], {}),
        created_at=str(row["created_at"]),
    )


def _metric(row: sqlite3.Row) -> VerifierCalibrationMetric:
    return VerifierCalibrationMetric(
        calibration_id=str(row["calibration_id"]),
        name=str(row["name"]),
        partition=str(row["partition"]),
        subgroup=str(row["subgroup"]),
        value=None if row["value"] is None else float(row["value"]),
        ci_low=None if row["ci_low"] is None else float(row["ci_low"]),
        ci_high=None if row["ci_high"] is None else float(row["ci_high"]),
        direction=row["direction"],
        available=bool(row["available"]),
        missing_reason=row["missing_reason"],
        record_count=int(row["record_count"]),
        metadata=_loads(row["metadata_json"], {}),
        created_at=str(row["created_at"]),
    )


def _decision(row: sqlite3.Row) -> VerifierQualificationDecision:
    return VerifierQualificationDecision(
        id=str(row["id"]),
        calibration_id=str(row["calibration_id"]),
        qualification_profile_revision_id=str(row["qualification_profile_revision_id"]),
        scope=str(row["scope"]),
        decision=str(row["decision"]),
        runtime_state=str(row["runtime_state"]),
        reasons=_loads(row["reasons_json"], []),
        evidence=_loads(row["evidence_json"], {}),
        override=bool(row["override"]),
        override_note=row["override_note"],
        supersedes_decision_id=row["supersedes_decision_id"],
        created_at=str(row["created_at"]),
    )


def _alias(row: sqlite3.Row) -> VerifierAlias:
    return VerifierAlias(
        profile_id=str(row["profile_id"]),
        alias=str(row["alias"]),
        revision_id=str(row["revision_id"]),
        updated_at=str(row["updated_at"]),
    )


def _alias_event(row: sqlite3.Row) -> VerifierAliasEvent:
    return VerifierAliasEvent(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        alias=str(row["alias"]),
        previous_revision_id=row["previous_revision_id"],
        revision_id=str(row["revision_id"]),
        qualification_decision_id=row["qualification_decision_id"],
        override=bool(row["override"]),
        note=row["note"],
        created_at=str(row["created_at"]),
    )


def _binding(row: sqlite3.Row) -> ResolvedVerifierBinding:
    return ResolvedVerifierBinding(
        id=str(row["id"]),
        verifier_revision_id=str(row["verifier_revision_id"]),
        domain_kind=str(row["domain_kind"]),
        domain_id=str(row["domain_id"]),
        role=str(row["role"]),
        qualification_decision_id=row["qualification_decision_id"],
        legacy_unqualified=bool(row["legacy_unqualified"]),
        development_exposed=bool(row["development_exposed"]),
        binding_hash=str(row["binding_hash"]),
        context=_loads(row["context_json"], {}),
        created_at=str(row["created_at"]),
    )


class VerifierLabStore:
    """One catalog shared by the API, CLI, scheduler, and guided dashboard."""

    def __init__(self, db: RunDatabase):
        self.db = db

    # -- verifier profiles -------------------------------------------------

    def create_profile(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> VerifierProfile:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Verifier profile name is required")
        now = _now()
        identifier = profile_id or _new_id("verifier")
        with self.db._lock:
            self.db._conn.execute(
                """
                INSERT INTO verifier_profiles
                    (id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (identifier, clean_name, description, now, now),
            )
            self.db._conn.commit()
        return self.get_profile(identifier)

    def get_profile(self, profile_id: str) -> VerifierProfile:
        row = self.db._conn.execute(
            "SELECT * FROM verifier_profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown verifier profile: {profile_id}")
        return _profile(row)

    def list_profiles(
        self, *, include_archived: bool = False, limit: int = 100, offset: int = 0
    ) -> List[VerifierProfile]:
        limit = max(1, min(int(limit), 1000))
        clause = "" if include_archived else "WHERE archived = 0"
        rows = self.db._conn.execute(
            f"SELECT * FROM verifier_profiles {clause} ORDER BY name, id LIMIT ? OFFSET ?",
            (limit, max(0, int(offset))),
        ).fetchall()
        return [_profile(row) for row in rows]

    def archive_profile(self, profile_id: str, *, archived: bool = True) -> VerifierProfile:
        now = _now()
        with self.db._lock:
            cursor = self.db._conn.execute(
                "UPDATE verifier_profiles SET archived = ?, updated_at = ? WHERE id = ?",
                (int(archived), now, profile_id),
            )
            if cursor.rowcount != 1:
                self.db._conn.rollback()
                raise KeyError(f"Unknown verifier profile: {profile_id}")
            self.db._conn.commit()
        return self.get_profile(profile_id)

    def _component_rows(self, revision_id: str) -> List[VerifierRevisionComponent]:
        rows = self.db._conn.execute(
            """
            SELECT * FROM verifier_revision_components
            WHERE revision_id = ? ORDER BY ordinal
            """,
            (revision_id,),
        ).fetchall()
        return [_component(row) for row in rows]

    def get_profile_revision(self, revision_id: str) -> VerifierProfileRevision:
        row = self.db._conn.execute(
            "SELECT * FROM verifier_profile_revisions WHERE id = ?", (revision_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown verifier profile revision: {revision_id}")
        return _profile_revision(row, self._component_rows(revision_id))

    def list_profile_revisions(
        self,
        *,
        profile_id: Optional[str] = None,
        family: Optional[str] = None,
        modality: Optional[str] = None,
        task_type: Optional[str] = None,
        qualifiable: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[VerifierProfileRevision]:
        clauses: List[str] = []
        values: List[Any] = []
        for column, value in (
            ("profile_id", profile_id),
            ("family", family),
            ("modality", modality),
            ("task_type", task_type),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                values.append(value)
        if qualifiable is not None:
            clauses.append("qualifiable = ?")
            values.append(int(qualifiable))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        values.extend([max(1, min(int(limit), 1000)), max(0, int(offset))])
        rows = self.db._conn.execute(
            "SELECT * FROM verifier_profile_revisions"
            + where
            + " ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
            values,
        ).fetchall()
        return [_profile_revision(row, self._component_rows(str(row["id"]))) for row in rows]

    def _assert_component_graph_acyclic(self, child_revision_ids: Sequence[str]) -> None:
        seen: set[str] = set()
        active: set[str] = set()

        def visit(revision_id: str) -> None:
            if revision_id in active:
                raise ValueError(f"Verifier chain cycle detected at {revision_id}")
            if revision_id in seen:
                return
            if (
                self.db._conn.execute(
                    "SELECT 1 FROM verifier_profile_revisions WHERE id = ?", (revision_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(f"Unknown child verifier revision: {revision_id}")
            active.add(revision_id)
            rows = self.db._conn.execute(
                "SELECT child_revision_id FROM verifier_revision_components WHERE revision_id = ?",
                (revision_id,),
            ).fetchall()
            for row in rows:
                visit(str(row["child_revision_id"]))
            active.remove(revision_id)
            seen.add(revision_id)

        for child_revision_id in child_revision_ids:
            visit(child_revision_id)

    def create_profile_revision(
        self,
        profile_id: str,
        definition: Mapping[str, Any],
        *,
        components: Optional[Sequence[Mapping[str, Any] | VerifierRevisionComponent]] = None,
        revision_id: Optional[str] = None,
    ) -> VerifierProfileRevision:
        self.get_profile(profile_id)
        resolved = scrub_secrets(dict(definition))
        family = str(resolved.get("family", "deterministic")).strip().lower()
        if family not in VERIFIER_FAMILIES:
            raise ValueError(f"Unknown verifier family: {family}")
        implementation = dict(resolved.get("implementation") or {})
        implementation_kind = str(implementation.get("kind", "builtin")).strip().lower()
        implementation_ref = str(
            implementation.get("ref") or implementation.get("class") or ""
        ).strip()
        if not implementation_ref:
            raise ValueError("A verifier implementation reference is required")
        implementation_fingerprint = implementation.get("fingerprint")
        if implementation_fingerprint is not None:
            implementation_fingerprint = str(implementation_fingerprint).strip() or None
        reward_contract = VerifierRewardContract.from_value(resolved.get("reward_contract"))
        modality = str(resolved.get("modality", "text")).strip().lower()
        task_type = str(resolved.get("task_type", "binary")).strip().lower()
        adapter = dict(resolved.get("reliability_adapter") or {})
        adapter_id = str(adapter.get("id", f"{family}-reliability")).strip()
        adapter_version = str(adapter.get("version", "1")).strip()
        if not adapter_id or not adapter_version:
            raise ValueError("Reliability adapter ID and version are required")
        runtime_contract = scrub_secrets(
            resolved.get("runtime_requirements") or resolved.get("runtime_contract") or {}
        )
        blockers = [str(value) for value in resolved.get("qualification_blockers", [])]
        if not implementation_fingerprint:
            blockers.append("implementation_unfingerprinted")
        if implementation.get("pinned") is False:
            blockers.append("implementation_unpinned")
        blockers = sorted(set(blockers))
        qualifiable = bool(resolved.get("qualifiable", not blockers)) and not blockers

        raw_components: List[Dict[str, Any]] = []
        for ordinal, value in enumerate(components or []):
            if isinstance(value, VerifierRevisionComponent):
                raw = value.to_dict()
            else:
                raw = dict(value)
            child_revision_id = str(
                raw.get("child_revision_id") or raw.get("revision_id") or ""
            ).strip()
            if not child_revision_id:
                raise ValueError("Every chain component needs child_revision_id")
            raw_components.append(
                {
                    "ordinal": ordinal,
                    "child_revision_id": child_revision_id,
                    "weight": float(raw.get("weight", 1.0)),
                    "veto": bool(raw.get("veto", False)),
                    "required": bool(raw.get("required", True)),
                    "configuration": scrub_secrets(raw.get("configuration") or {}),
                }
            )
        if family == "chain" and not raw_components:
            raise ValueError("A verifier chain requires at least one ordered component")
        if family != "chain" and raw_components:
            raise ValueError("Only chain verifier revisions may declare components")
        child_ids = [str(value["child_revision_id"]) for value in raw_components]
        if len(child_ids) != len(set(child_ids)):
            raise ValueError("A verifier revision cannot contain the same child twice")
        if any(float(value["weight"]) < 0 for value in raw_components):
            raise ValueError("Verifier component weights cannot be negative")
        self._assert_component_graph_acyclic(child_ids)

        identity = {"definition": resolved, "components": raw_components}
        revision_hash = content_hash(identity)
        runtime_hash = content_hash(runtime_contract)
        configuration_hash = content_hash(resolved)
        now = _now()
        identifier = revision_id or _new_id("verifier-revision")
        with self.db._lock:
            existing = self.db._conn.execute(
                """
                SELECT id FROM verifier_profile_revisions
                WHERE profile_id = ? AND content_hash = ?
                """,
                (profile_id, revision_hash),
            ).fetchone()
            if existing is not None:
                return self.get_profile_revision(str(existing["id"]))
            revision_number = int(
                self.db._conn.execute(
                    """
                    SELECT COALESCE(MAX(revision_number), 0) + 1
                    FROM verifier_profile_revisions WHERE profile_id = ?
                    """,
                    (profile_id,),
                ).fetchone()[0]
            )
            try:
                self.db._conn.execute(
                    """
                    INSERT INTO verifier_profile_revisions (
                        id, profile_id, revision_number, content_hash, family,
                        reliability_adapter_id, reliability_adapter_version,
                        implementation_kind, implementation_ref,
                        implementation_fingerprint, qualifiable,
                        qualification_blockers_json, modality, task_type,
                        input_mapping_json, output_contract_json, reward_min,
                        reward_max, reward_direction, threshold, tie_policy,
                        error_behavior, definition_json,
                        sanitized_configuration_hash, runtime_contract_json,
                        runtime_contract_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        profile_id,
                        revision_number,
                        revision_hash,
                        family,
                        adapter_id,
                        adapter_version,
                        implementation_kind,
                        implementation_ref,
                        implementation_fingerprint,
                        int(qualifiable),
                        _canonical_json(blockers),
                        modality,
                        task_type,
                        _canonical_json(resolved.get("input_mapping") or {}),
                        _canonical_json(reward_contract.to_dict()),
                        reward_contract.minimum,
                        reward_contract.maximum,
                        reward_contract.direction,
                        reward_contract.threshold,
                        reward_contract.tie_policy,
                        reward_contract.error_behavior,
                        _canonical_json(resolved),
                        configuration_hash,
                        _canonical_json(runtime_contract),
                        runtime_hash,
                        now,
                    ),
                )
                for value in raw_components:
                    self.db._conn.execute(
                        """
                        INSERT INTO verifier_revision_components (
                            revision_id, ordinal, child_revision_id, weight,
                            veto, required, configuration_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            identifier,
                            value["ordinal"],
                            value["child_revision_id"],
                            value["weight"],
                            int(value["veto"]),
                            int(value["required"]),
                            _canonical_json(value["configuration"]),
                        ),
                    )
                self.db._conn.execute(
                    """
                    UPDATE verifier_profiles
                    SET latest_revision_id = ?, updated_at = ? WHERE id = ?
                    """,
                    (identifier, now, profile_id),
                )
                self.db._conn.commit()
            except Exception:
                self.db._conn.rollback()
                raise
        return self.get_profile_revision(identifier)

    # -- immutable protocol and qualification-policy revisions ------------

    def create_protocol(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        protocol_id: Optional[str] = None,
    ) -> VerifierCalibrationProtocol:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Calibration protocol name is required")
        identifier = protocol_id or _new_id("verifier-protocol")
        now = _now()
        with self.db._lock:
            self.db._conn.execute(
                """
                INSERT INTO verifier_calibration_protocols
                    (id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (identifier, clean_name, description, now, now),
            )
            self.db._conn.commit()
        return self.get_protocol(identifier)

    def get_protocol(self, protocol_id: str) -> VerifierCalibrationProtocol:
        row = self.db._conn.execute(
            "SELECT * FROM verifier_calibration_protocols WHERE id = ?", (protocol_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown verifier calibration protocol: {protocol_id}")
        return _protocol(row)

    def list_protocols(
        self, *, limit: int = 100, offset: int = 0
    ) -> List[VerifierCalibrationProtocol]:
        rows = self.db._conn.execute(
            """
            SELECT * FROM verifier_calibration_protocols
            WHERE archived = 0 ORDER BY name, id LIMIT ? OFFSET ?
            """,
            (max(1, min(int(limit), 1000)), max(0, int(offset))),
        ).fetchall()
        return [_protocol(row) for row in rows]

    def create_protocol_revision(
        self,
        protocol_id: str,
        definition: Mapping[str, Any],
        *,
        revision_id: Optional[str] = None,
    ) -> VerifierCalibrationProtocolRevision:
        self.get_protocol(protocol_id)
        resolved = scrub_secrets(dict(definition))
        revision_hash = content_hash(resolved)
        identifier = revision_id or _new_id("verifier-protocol-revision")
        now = _now()
        with self.db._lock:
            existing = self.db._conn.execute(
                """
                SELECT id FROM verifier_calibration_protocol_revisions
                WHERE protocol_id = ? AND content_hash = ?
                """,
                (protocol_id, revision_hash),
            ).fetchone()
            if existing is not None:
                return self.get_protocol_revision(str(existing["id"]))
            number = int(
                self.db._conn.execute(
                    """
                    SELECT COALESCE(MAX(revision_number), 0) + 1
                    FROM verifier_calibration_protocol_revisions
                    WHERE protocol_id = ?
                    """,
                    (protocol_id,),
                ).fetchone()[0]
            )
            self.db._conn.execute(
                """
                INSERT INTO verifier_calibration_protocol_revisions
                    (id, protocol_id, revision_number, content_hash,
                     definition_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (identifier, protocol_id, number, revision_hash, _canonical_json(resolved), now),
            )
            self.db._conn.execute(
                """
                UPDATE verifier_calibration_protocols
                SET latest_revision_id = ?, updated_at = ? WHERE id = ?
                """,
                (identifier, now, protocol_id),
            )
            self.db._conn.commit()
        return self.get_protocol_revision(identifier)

    def get_protocol_revision(self, revision_id: str) -> VerifierCalibrationProtocolRevision:
        row = self.db._conn.execute(
            "SELECT * FROM verifier_calibration_protocol_revisions WHERE id = ?",
            (revision_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown verifier calibration protocol revision: {revision_id}")
        return _protocol_revision(row)

    def list_protocol_revisions(
        self, protocol_id: str, *, limit: int = 100, offset: int = 0
    ) -> List[VerifierCalibrationProtocolRevision]:
        rows = self.db._conn.execute(
            """
            SELECT * FROM verifier_calibration_protocol_revisions
            WHERE protocol_id = ? ORDER BY revision_number DESC LIMIT ? OFFSET ?
            """,
            (protocol_id, max(1, min(int(limit), 1000)), max(0, int(offset))),
        ).fetchall()
        return [_protocol_revision(row) for row in rows]

    def create_qualification_profile(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> VerifierQualificationProfile:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Verifier qualification profile name is required")
        identifier = profile_id or _new_id("verifier-qualification")
        now = _now()
        with self.db._lock:
            self.db._conn.execute(
                """
                INSERT INTO verifier_qualification_profiles
                    (id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (identifier, clean_name, description, now, now),
            )
            self.db._conn.commit()
        return self.get_qualification_profile(identifier)

    def get_qualification_profile(self, profile_id: str) -> VerifierQualificationProfile:
        row = self.db._conn.execute(
            "SELECT * FROM verifier_qualification_profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown verifier qualification profile: {profile_id}")
        return _qualification_profile(row)

    def list_qualification_profiles(
        self, *, limit: int = 100, offset: int = 0
    ) -> List[VerifierQualificationProfile]:
        rows = self.db._conn.execute(
            """
            SELECT * FROM verifier_qualification_profiles
            WHERE archived = 0 ORDER BY name, id LIMIT ? OFFSET ?
            """,
            (max(1, min(int(limit), 1000)), max(0, int(offset))),
        ).fetchall()
        return [_qualification_profile(row) for row in rows]

    def create_qualification_profile_revision(
        self,
        profile_id: str,
        *,
        template_kind: str,
        requirements: Mapping[str, Any],
        promotable: Optional[bool] = None,
        revision_id: Optional[str] = None,
    ) -> VerifierQualificationProfileRevision:
        self.get_qualification_profile(profile_id)
        kind = template_kind.strip().lower()
        if kind not in QUALIFICATION_TEMPLATES:
            raise ValueError(f"Unknown verifier qualification template: {kind}")
        if kind == "exploratory" and promotable is True:
            raise ValueError("The exploratory template can never grant a promotable pass")
        resolved_promotable = kind != "exploratory" if promotable is None else bool(promotable)
        resolved = scrub_secrets(dict(requirements))
        identity = {
            "template_kind": kind,
            "promotable": resolved_promotable,
            "requirements": resolved,
        }
        revision_hash = content_hash(identity)
        identifier = revision_id or _new_id("verifier-qualification-revision")
        now = _now()
        with self.db._lock:
            existing = self.db._conn.execute(
                """
                SELECT id FROM verifier_qualification_profile_revisions
                WHERE profile_id = ? AND content_hash = ?
                """,
                (profile_id, revision_hash),
            ).fetchone()
            if existing is not None:
                return self.get_qualification_profile_revision(str(existing["id"]))
            number = int(
                self.db._conn.execute(
                    """
                    SELECT COALESCE(MAX(revision_number), 0) + 1
                    FROM verifier_qualification_profile_revisions
                    WHERE profile_id = ?
                    """,
                    (profile_id,),
                ).fetchone()[0]
            )
            self.db._conn.execute(
                """
                INSERT INTO verifier_qualification_profile_revisions
                    (id, profile_id, revision_number, content_hash, template_kind,
                     promotable, requirements_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    profile_id,
                    number,
                    revision_hash,
                    kind,
                    int(resolved_promotable),
                    _canonical_json(resolved),
                    now,
                ),
            )
            self.db._conn.execute(
                """
                UPDATE verifier_qualification_profiles
                SET latest_revision_id = ?, updated_at = ? WHERE id = ?
                """,
                (identifier, now, profile_id),
            )
            self.db._conn.commit()
        return self.get_qualification_profile_revision(identifier)

    def get_qualification_profile_revision(
        self, revision_id: str
    ) -> VerifierQualificationProfileRevision:
        row = self.db._conn.execute(
            "SELECT * FROM verifier_qualification_profile_revisions WHERE id = ?",
            (revision_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown verifier qualification profile revision: {revision_id}")
        return _qualification_revision(row)

    def list_qualification_profile_revisions(
        self, profile_id: str, *, limit: int = 100, offset: int = 0
    ) -> List[VerifierQualificationProfileRevision]:
        rows = self.db._conn.execute(
            """
            SELECT * FROM verifier_qualification_profile_revisions
            WHERE profile_id = ? ORDER BY revision_number DESC LIMIT ? OFFSET ?
            """,
            (profile_id, max(1, min(int(limit), 1000)), max(0, int(offset))),
        ).fetchall()
        return [_qualification_revision(row) for row in rows]

    # -- calibration jobs and immutable evidence --------------------------

    def create_calibration(
        self,
        *,
        verifier_revision_id: str,
        protocol_revision_id: str,
        qualification_profile_revision_id: str,
        source_kind: str,
        source_revision_id: str,
        source_hash: str,
        source_purpose: str = "unspecified",
        request: Optional[Mapping[str, Any]] = None,
        partition: Optional[Mapping[str, Any]] = None,
        runtime_identity: Optional[Mapping[str, Any]] = None,
        total_records: Optional[int] = None,
        work_item_id: Optional[str] = None,
        calibration_id: Optional[str] = None,
    ) -> VerifierCalibration:
        verifier_revision = self.get_profile_revision(verifier_revision_id)
        protocol = self.get_protocol_revision(protocol_revision_id)
        qualification = self.get_qualification_profile_revision(qualification_profile_revision_id)
        kind = source_kind.strip().lower()
        purpose = source_purpose.strip().lower().replace("-", "_")
        if kind not in SOURCE_KINDS:
            raise ValueError(f"Unknown calibration source kind: {kind}")
        if purpose in PROTECTED_SOURCE_PURPOSES:
            raise ValueError(
                f"Calibration source purpose {source_purpose!r} is protected and ineligible"
            )
        if not source_revision_id.strip() or not source_hash.strip():
            raise ValueError("Calibration source revision and hash are required")
        clean_request = scrub_secrets(dict(request or {}))
        clean_partition = scrub_secrets(dict(partition or {}))
        clean_runtime = scrub_secrets(dict(runtime_identity or {}))
        runtime_hash = content_hash(clean_runtime)
        reuse_key = content_hash(
            {
                "verifier_revision_hash": verifier_revision.content_hash,
                "source_kind": kind,
                "source_revision_id": source_revision_id,
                "source_hash": source_hash,
                "protocol_hash": protocol.content_hash,
                "qualification_hash": qualification.content_hash,
                "request": clean_request,
                "partition": clean_partition,
                "runtime_identity_hash": runtime_hash,
            }
        )
        identifier = calibration_id or _new_id("verifier-calibration")
        now = _now()
        with self.db._lock:
            existing = self.db._conn.execute(
                """SELECT * FROM verifier_calibrations
                   WHERE reuse_key=? AND status IN ('queued','running','interrupted','completed')
                   ORDER BY CASE status WHEN 'completed' THEN 0 ELSE 1 END,
                            created_at DESC LIMIT 1""",
                (reuse_key,),
            ).fetchone()
            if existing is not None:
                return _calibration(existing)
            self.db._conn.execute(
                """
                INSERT INTO verifier_calibrations (
                    id, verifier_revision_id, protocol_revision_id,
                    qualification_profile_revision_id, source_kind,
                    source_revision_id, source_hash, source_purpose, status,
                    stage, total_records, request_json, partition_json,
                    runtime_identity_json, runtime_identity_hash, protocol_hash,
                    qualification_hash, reuse_key, work_item_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', 'queued', ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    verifier_revision_id,
                    protocol_revision_id,
                    qualification_profile_revision_id,
                    kind,
                    source_revision_id,
                    source_hash,
                    purpose,
                    total_records,
                    _canonical_json(clean_request),
                    _canonical_json(clean_partition),
                    _canonical_json(clean_runtime),
                    runtime_hash,
                    protocol.content_hash,
                    qualification.content_hash,
                    reuse_key,
                    work_item_id,
                    now,
                    now,
                ),
            )
            self.db._conn.commit()
        return self.get_calibration(identifier)

    def find_reusable_calibration(self, reuse_key: str) -> Optional[VerifierCalibration]:
        row = self.db._conn.execute(
            """
            SELECT * FROM verifier_calibrations
            WHERE reuse_key = ? AND status = 'completed'
            ORDER BY completed_at DESC LIMIT 1
            """,
            (reuse_key,),
        ).fetchone()
        return _calibration(row) if row else None

    def get_calibration(self, calibration_id: str) -> VerifierCalibration:
        row = self.db._conn.execute(
            "SELECT * FROM verifier_calibrations WHERE id = ?", (calibration_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown verifier calibration: {calibration_id}")
        return _calibration(row)

    def list_calibrations(
        self,
        *,
        verifier_revision_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[VerifierCalibration]:
        clauses: List[str] = []
        values: List[Any] = []
        if verifier_revision_id:
            clauses.append("verifier_revision_id = ?")
            values.append(verifier_revision_id)
        if status:
            clauses.append("status = ?")
            values.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        values.extend([max(1, min(int(limit), 1000)), max(0, int(offset))])
        rows = self.db._conn.execute(
            "SELECT * FROM verifier_calibrations"
            + where
            + " ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
            values,
        ).fetchall()
        return [_calibration(row) for row in rows]

    def update_calibration(self, calibration_id: str, **changes: Any) -> VerifierCalibration:
        allowed = {
            "status",
            "stage",
            "processed_records",
            "total_records",
            "sample_count",
            "artifact_path",
            "manifest_hash",
            "work_item_id",
            "cancel_requested",
            "retry_count",
            "error",
            "started_at",
            "completed_at",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(
                "Immutable calibration identity fields cannot be updated: "
                + ", ".join(sorted(unknown))
            )
        if not changes:
            return self.get_calibration(calibration_id)
        normalized = dict(changes)
        for key in ("cancel_requested",):
            if key in normalized:
                normalized[key] = int(bool(normalized[key]))
        normalized["updated_at"] = _now()
        assignments = ", ".join(f"{key} = ?" for key in normalized)
        with self.db._lock:
            cursor = self.db._conn.execute(
                f"UPDATE verifier_calibrations SET {assignments} WHERE id = ?",
                [*normalized.values(), calibration_id],
            )
            if cursor.rowcount != 1:
                self.db._conn.rollback()
                raise KeyError(f"Unknown verifier calibration: {calibration_id}")
            self.db._conn.commit()
        return self.get_calibration(calibration_id)

    def append_sample(
        self,
        calibration_id: str,
        *,
        ordinal: int,
        record_id: str,
        record_hash: str,
        group_id: Optional[str],
        partition: str,
        repeat_index: int,
        orientation: str,
        probe_kind: str,
        seed: Optional[int],
        reference: Mapping[str, Any],
        observation: VerifierObservation | Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> VerifierCalibrationSample:
        calibration = self.get_calibration(calibration_id)
        revision = self.get_profile_revision(calibration.verifier_revision_id)
        if not isinstance(observation, VerifierObservation):
            if callable(getattr(observation, "to_dict", None)):
                raw = dict(observation.to_dict())
            else:
                raw = dict(observation)
            observation = VerifierObservation(
                reward=raw.get("reward"),
                passed=raw.get("passed"),
                parsed_value=raw.get("parsed_value"),
                raw_output=raw.get("raw_output"),
                details=_observation_details(raw.get("details")),
                component_trace=list(raw.get("component_trace") or []),
                latency_ms=raw.get("latency_ms"),
                error=raw.get("error"),
                runtime_identity=dict(raw.get("runtime_identity") or {}),
            )
        if observation.reward is not None and not (
            revision.reward_contract.minimum
            <= float(observation.reward)
            <= revision.reward_contract.maximum
        ):
            raise ValueError(
                f"Reward {observation.reward} is outside the declared "
                f"[{revision.reward_contract.minimum}, {revision.reward_contract.maximum}] contract"
            )
        partition = partition.strip().lower()
        if partition not in {"calibration", "confirmation"}:
            raise ValueError("Calibration sample partition must be calibration or confirmation")
        now = _now()
        raw_observation = observation.to_dict()
        with self.db._lock:
            self.db._conn.execute(
                """
                INSERT INTO verifier_calibration_samples (
                    calibration_id, ordinal, record_id, record_hash, group_id,
                    partition, repeat_index, orientation, probe_kind, seed,
                    reference_json, observation_json, reward, passed, latency_ms,
                    error, runtime_identity_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    calibration_id,
                    int(ordinal),
                    record_id,
                    record_hash,
                    group_id or record_id,
                    partition,
                    int(repeat_index),
                    orientation,
                    probe_kind,
                    seed,
                    _canonical_json(scrub_secrets(reference)),
                    _canonical_json(scrub_secrets(raw_observation)),
                    observation.reward,
                    None if observation.passed is None else int(observation.passed),
                    observation.latency_ms,
                    observation.error,
                    _canonical_json(scrub_secrets(observation.runtime_identity)),
                    _canonical_json(scrub_secrets(metadata or {})),
                    now,
                ),
            )
            self.db._conn.execute(
                """
                UPDATE verifier_calibrations
                SET sample_count = sample_count + 1, updated_at = ? WHERE id = ?
                """,
                (now, calibration_id),
            )
            self.db._conn.commit()
        row = self.db._conn.execute(
            """
            SELECT * FROM verifier_calibration_samples
            WHERE calibration_id = ? AND ordinal = ?
            """,
            (calibration_id, int(ordinal)),
        ).fetchone()
        assert row is not None
        return _sample(row)

    def list_samples(
        self,
        calibration_id: str,
        *,
        partition: Optional[str] = None,
        record_id: Optional[str] = None,
        error_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[VerifierCalibrationSample]:
        clauses = ["calibration_id = ?"]
        values: List[Any] = [calibration_id]
        if partition:
            clauses.append("partition = ?")
            values.append(partition)
        if record_id:
            clauses.append("record_id = ?")
            values.append(record_id)
        if error_only:
            clauses.append("error IS NOT NULL")
        values.extend([max(1, min(int(limit), 1000)), max(0, int(offset))])
        rows = self.db._conn.execute(
            "SELECT * FROM verifier_calibration_samples WHERE "
            + " AND ".join(clauses)
            + " ORDER BY ordinal LIMIT ? OFFSET ?",
            values,
        ).fetchall()
        return [_sample(row) for row in rows]

    def list_samples_for_record_ids(
        self,
        calibration_id: str,
        record_ids: Sequence[str],
        *,
        partition: str = "calibration",
        maximum_record_ids: int = 256,
        maximum_samples: int = 20_000,
    ) -> Dict[str, List[VerifierCalibrationSample]]:
        """Fetch one bounded record chunk for failure-mining joins.

        Calibration acquisition joins immutable source rows to replicated
        observations.  Batching the indexed record-ID lookup avoids both a
        full sample materialization and one database query per source record.
        """

        self.get_calibration(calibration_id)
        identifiers = list(
            dict.fromkeys(str(value) for value in record_ids if str(value))
        )
        if not identifiers:
            return {}
        if len(identifiers) > max(1, int(maximum_record_ids)):
            raise ValueError(
                f"calibration sample grouping accepts at most {maximum_record_ids} record IDs"
            )
        normalized_partition = str(partition).strip().lower()
        if normalized_partition not in {"calibration", "confirmation"}:
            raise ValueError("sample partition must be calibration or confirmation")
        placeholders = ",".join("?" for _ in identifiers)
        bound = max(1, int(maximum_samples))
        rows = self.db._conn.execute(
            f"""SELECT * FROM verifier_calibration_samples
                WHERE calibration_id = ? AND partition = ?
                  AND record_id IN ({placeholders})
                ORDER BY record_id, ordinal LIMIT ?""",
            (calibration_id, normalized_partition, *identifiers, bound + 1),
        ).fetchall()
        if len(rows) > bound:
            raise ValueError(
                "calibration protocol produced too many observations for one bounded join chunk"
            )
        result: Dict[str, List[VerifierCalibrationSample]] = {
            record_id: [] for record_id in identifiers
        }
        for row in rows:
            result.setdefault(str(row["record_id"]), []).append(_sample(row))
        return result

    def query_samples(
        self,
        calibration_id: str,
        *,
        partition: Optional[str] = None,
        outcome: Optional[str] = None,
        perturbation: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[VerifierCalibrationSample], int, int, int]:
        """Return one bounded, server-filtered page with a matching count."""

        self.get_calibration(calibration_id)
        clauses = ["calibration_id = ?"]
        values: List[Any] = [calibration_id]
        if partition:
            clauses.append("partition = ?")
            values.append(str(partition))
        if perturbation:
            clauses.append("probe_kind = ?")
            values.append(str(perturbation))
        normalized_outcome = str(outcome or "").strip().lower()
        if normalized_outcome == "error":
            clauses.append("error IS NOT NULL")
        elif normalized_outcome == "passed":
            clauses.append("passed = 1 AND error IS NULL")
        elif normalized_outcome == "failed":
            clauses.append("passed = 0 AND error IS NULL")
        elif normalized_outcome in {"false_accept", "false_reject"}:
            truthy = (
                "LOWER(CAST(json_extract(reference_json, '$.expected') AS TEXT)) "
                "IN ('1','true','yes','pass','passed','accept','accepted')"
            )
            clauses.append("passed = ? AND error IS NULL AND " + (truthy if normalized_outcome == "false_reject" else f"NOT ({truthy})"))
            values.append(0 if normalized_outcome == "false_reject" else 1)
        elif normalized_outcome in {"repeat_flip", "order_flip"}:
            grouping_filter = "calibration_id = ?"
            grouping_values: List[Any] = [calibration_id]
            if partition:
                grouping_filter += " AND partition = ?"
                grouping_values.append(str(partition))
            if normalized_outcome == "repeat_flip":
                grouping_filter += " AND repeat_index >= 0"
            else:
                grouping_filter += " AND orientation <> 'canonical'"
            clauses.append(
                "record_id IN (SELECT record_id FROM verifier_calibration_samples "
                f"WHERE {grouping_filter} AND error IS NULL GROUP BY record_id "
                "HAVING MIN(passed) <> MAX(passed))"
            )
            values.extend(grouping_values)
        elif normalized_outcome:
            raise ValueError(
                "outcome must be passed, failed, error, false_accept, "
                "false_reject, repeat_flip, or order_flip"
            )
        search = str(query or "").strip()
        if search:
            pattern = f"%{search}%"
            clauses.append(
                "(record_id LIKE ? OR reference_json LIKE ? OR observation_json LIKE ?)"
            )
            values.extend((pattern, pattern, pattern))
        where = " AND ".join(clauses)
        page_limit = max(1, min(1000, int(limit)))
        page_offset = max(0, int(offset))
        rows = self.db._conn.execute(
            f"""SELECT * FROM verifier_calibration_samples WHERE {where}
                ORDER BY ordinal LIMIT ? OFFSET ?""",
            (*values, page_limit, page_offset),
        ).fetchall()
        total = int(
            self.db._conn.execute(
                f"SELECT COUNT(*) AS value FROM verifier_calibration_samples WHERE {where}",
                values,
            ).fetchone()["value"]
        )
        return [_sample(row) for row in rows], total, page_limit, page_offset

    def compare_sample_page(
        self,
        base_calibration_id: str,
        candidate_calibration_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[
        List[
            tuple[
                Optional[VerifierCalibrationSample],
                Optional[VerifierCalibrationSample],
            ]
        ],
        int,
        int,
        int,
    ]:
        """Page an indexed stable-identity join across two calibrations.

        Only the requested page is decoded into Python objects.  SQLite joins
        against the calibration/record expansion identity index and computes
        the total, avoiding the former full materialization of both sample
        collections in application memory.
        """

        self.get_calibration(base_calibration_id)
        self.get_calibration(candidate_calibration_id)
        page_limit = max(1, min(1000, int(limit)))
        page_offset = max(0, int(offset))
        key_columns = (
            "partition",
            "record_id",
            "repeat_index",
            "orientation",
            "probe_kind",
        )
        sample_columns = (
            "calibration_id",
            "ordinal",
            "record_id",
            "record_hash",
            "group_id",
            "partition",
            "repeat_index",
            "orientation",
            "probe_kind",
            "seed",
            "reference_json",
            "observation_json",
            "reward",
            "passed",
            "latency_ms",
            "error",
            "runtime_identity_json",
            "metadata_json",
            "created_at",
        )
        keys = ", ".join(key_columns)
        join = " AND ".join(f"{{side}}.{name} = sample_keys.{name}" for name in key_columns)
        selected = ", ".join(
            [
                *(f"base.{name} AS base_{name}" for name in sample_columns),
                *(f"candidate.{name} AS candidate_{name}" for name in sample_columns),
            ]
        )
        query = f"""
            WITH sample_keys AS (
                SELECT {keys} FROM verifier_calibration_samples
                WHERE calibration_id = ?
                UNION
                SELECT {keys} FROM verifier_calibration_samples
                WHERE calibration_id = ?
            )
            SELECT {selected}
            FROM sample_keys
            LEFT JOIN verifier_calibration_samples AS base
              ON base.calibration_id = ? AND {join.format(side='base')}
            LEFT JOIN verifier_calibration_samples AS candidate
              ON candidate.calibration_id = ? AND {join.format(side='candidate')}
            ORDER BY sample_keys.partition, sample_keys.record_id,
                     sample_keys.repeat_index, sample_keys.orientation,
                     sample_keys.probe_kind
            LIMIT ? OFFSET ?
        """
        rows = self.db._conn.execute(
            query,
            (
                base_calibration_id,
                candidate_calibration_id,
                base_calibration_id,
                candidate_calibration_id,
                page_limit,
                page_offset,
            ),
        ).fetchall()
        total = int(
            self.db._conn.execute(
                f"""
                SELECT COUNT(*) AS value FROM (
                    SELECT {keys} FROM verifier_calibration_samples
                    WHERE calibration_id = ?
                    UNION
                    SELECT {keys} FROM verifier_calibration_samples
                    WHERE calibration_id = ?
                )
                """,
                (base_calibration_id, candidate_calibration_id),
            ).fetchone()["value"]
        )

        pairs: List[
            tuple[
                Optional[VerifierCalibrationSample],
                Optional[VerifierCalibrationSample],
            ]
        ] = []
        for row in rows:
            decoded: List[Optional[VerifierCalibrationSample]] = []
            for prefix in ("base", "candidate"):
                if row[f"{prefix}_calibration_id"] is None:
                    decoded.append(None)
                    continue
                payload = {
                    name: row[f"{prefix}_{name}"] for name in sample_columns
                }
                decoded.append(_sample(payload))
            pairs.append((decoded[0], decoded[1]))
        return pairs, total, page_limit, page_offset

    def append_metric(
        self,
        calibration_id: str,
        *,
        name: str,
        value: Optional[float],
        partition: str = "calibration",
        subgroup: str = "",
        ci_low: Optional[float] = None,
        ci_high: Optional[float] = None,
        direction: Optional[str] = None,
        available: bool = True,
        missing_reason: Optional[str] = None,
        record_count: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> VerifierCalibrationMetric:
        self.get_calibration(calibration_id)
        numeric_values = [item for item in (value, ci_low, ci_high) if item is not None]
        if any(not math.isfinite(float(item)) for item in numeric_values):
            raise ValueError("Calibration metrics and intervals must be finite")
        if available and value is None:
            raise ValueError("An available metric requires a value")
        if available and missing_reason is not None:
            raise ValueError("An available metric cannot have a missing reason")
        if not available and (value is not None or not missing_reason):
            raise ValueError("An unavailable metric requires only a missing reason")
        now = _now()
        with self.db._lock:
            self.db._conn.execute(
                """
                INSERT INTO verifier_calibration_metrics (
                    calibration_id, name, partition, subgroup, value, ci_low,
                    ci_high, direction, available, missing_reason, record_count,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    calibration_id,
                    name,
                    partition,
                    subgroup,
                    value,
                    ci_low,
                    ci_high,
                    direction,
                    int(available),
                    missing_reason,
                    int(record_count),
                    _canonical_json(scrub_secrets(metadata or {})),
                    now,
                ),
            )
            self.db._conn.commit()
        row = self.db._conn.execute(
            """
            SELECT * FROM verifier_calibration_metrics
            WHERE calibration_id = ? AND name = ? AND partition = ? AND subgroup = ?
            """,
            (calibration_id, name, partition, subgroup),
        ).fetchone()
        assert row is not None
        return _metric(row)

    def list_metrics(self, calibration_id: str) -> List[VerifierCalibrationMetric]:
        rows = self.db._conn.execute(
            """
            SELECT * FROM verifier_calibration_metrics WHERE calibration_id = ?
            ORDER BY partition, name, subgroup
            """,
            (calibration_id,),
        ).fetchall()
        return [_metric(row) for row in rows]

    def query_metrics(
        self,
        calibration_id: str,
        *,
        partition: Optional[str] = None,
        subgroup: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[VerifierCalibrationMetric], int, int, int]:
        clauses = ["calibration_id=?"]
        params: List[Any] = [calibration_id]
        if partition:
            clauses.append("partition=?")
            params.append(str(partition))
        if subgroup is not None:
            clauses.append("subgroup=?")
            params.append(str(subgroup))
        where = " AND ".join(clauses)
        page_limit = max(1, min(1000, int(limit)))
        page_offset = max(0, int(offset))
        rows = self.db._conn.execute(
            f"""SELECT * FROM verifier_calibration_metrics WHERE {where}
                ORDER BY partition, subgroup, name LIMIT ? OFFSET ?""",
            (*params, page_limit, page_offset),
        ).fetchall()
        total = int(
            self.db._conn.execute(
                f"SELECT COUNT(*) AS value FROM verifier_calibration_metrics WHERE {where}",
                params,
            ).fetchone()["value"]
        )
        return [_metric(row) for row in rows], total, page_limit, page_offset

    def append_decision(
        self,
        calibration_id: str,
        *,
        scope: str,
        decision: str,
        reasons: Sequence[str],
        evidence: Optional[Mapping[str, Any]] = None,
        runtime_state: str = "compatible",
        override: bool = False,
        override_note: Optional[str] = None,
        supersedes_decision_id: Optional[str] = None,
        decision_id: Optional[str] = None,
    ) -> VerifierQualificationDecision:
        calibration = self.get_calibration(calibration_id)
        if scope not in {"development", "operational", "confirmation"}:
            raise ValueError(f"Unknown verifier qualification scope: {scope}")
        if decision not in {"pass", "warn", "fail"}:
            raise ValueError(f"Unknown verifier qualification decision: {decision}")
        if runtime_state not in {"compatible", "stale_runtime", "unavailable"}:
            raise ValueError(f"Unknown verifier runtime state: {runtime_state}")
        if override and not str(override_note or "").strip():
            raise ValueError("A verifier qualification override requires a note")
        if supersedes_decision_id:
            previous = self.get_decision(supersedes_decision_id)
            if previous.calibration_id != calibration_id:
                raise ValueError("A decision can only supersede one from the same calibration")
        identifier = decision_id or _new_id("verifier-decision")
        now = _now()
        with self.db._lock:
            self.db._conn.execute(
                """
                INSERT INTO verifier_qualification_decisions (
                    id, calibration_id, qualification_profile_revision_id,
                    scope, decision, runtime_state, reasons_json, evidence_json,
                    override, override_note, supersedes_decision_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    calibration_id,
                    calibration.qualification_profile_revision_id,
                    scope,
                    decision,
                    runtime_state,
                    _canonical_json(list(reasons)),
                    _canonical_json(scrub_secrets(evidence or {})),
                    int(override),
                    override_note,
                    supersedes_decision_id,
                    now,
                ),
            )
            self.db._conn.commit()
        return self.get_decision(identifier)

    def get_decision(self, decision_id: str) -> VerifierQualificationDecision:
        row = self.db._conn.execute(
            "SELECT * FROM verifier_qualification_decisions WHERE id = ?", (decision_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown verifier qualification decision: {decision_id}")
        return _decision(row)

    def list_decisions(self, calibration_id: str) -> List[VerifierQualificationDecision]:
        rows = self.db._conn.execute(
            """
            SELECT * FROM verifier_qualification_decisions
            WHERE calibration_id = ? ORDER BY created_at, id
            """,
            (calibration_id,),
        ).fetchall()
        return [_decision(row) for row in rows]

    def query_decisions(
        self,
        *,
        verifier_revision_id: Optional[str] = None,
        calibration_id: Optional[str] = None,
        decision: Optional[str] = None,
        scope: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[VerifierQualificationDecision], int, int, int]:
        clauses: List[str] = []
        values: List[Any] = []
        if verifier_revision_id:
            clauses.append("c.verifier_revision_id = ?")
            values.append(verifier_revision_id)
        if calibration_id:
            clauses.append("d.calibration_id = ?")
            values.append(calibration_id)
        if decision:
            clauses.append("d.decision = ?")
            values.append(str(decision).strip().lower())
        if scope:
            clauses.append("d.scope = ?")
            values.append(str(scope).strip().lower())
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        page_limit = max(1, min(1000, int(limit)))
        page_offset = max(0, int(offset))
        rows = self.db._conn.execute(
            """SELECT d.* FROM verifier_qualification_decisions d
               JOIN verifier_calibrations c ON c.id = d.calibration_id"""
            + where
            + " ORDER BY d.created_at DESC, d.id DESC LIMIT ? OFFSET ?",
            (*values, page_limit, page_offset),
        ).fetchall()
        total = int(
            self.db._conn.execute(
                """SELECT COUNT(*) AS value FROM verifier_qualification_decisions d
                   JOIN verifier_calibrations c ON c.id = d.calibration_id"""
                + where,
                values,
            ).fetchone()["value"]
        )
        return [_decision(row) for row in rows], total, page_limit, page_offset

    # -- promotion aliases and exact downstream usage ---------------------

    def promote_alias(
        self,
        revision_id: str,
        *,
        alias: str,
        qualification_decision_id: Optional[str] = None,
        override: bool = False,
        note: Optional[str] = None,
    ) -> VerifierAlias:
        revision = self.get_profile_revision(revision_id)
        clean_alias = alias.strip().lower()
        if clean_alias not in {"candidate", "approved"}:
            raise ValueError("Verifier aliases are limited to candidate and approved")
        if override and not str(note or "").strip():
            raise ValueError("An alias promotion override requires a note")
        if not override:
            if qualification_decision_id is None:
                raise ValueError("A normal verifier promotion requires a qualification decision")
            decision = self.get_decision(qualification_decision_id)
            calibration = self.get_calibration(decision.calibration_id)
            if calibration.verifier_revision_id != revision_id:
                raise ValueError("Qualification decision belongs to a different verifier revision")
            if decision.decision != "pass" or decision.runtime_state != "compatible":
                raise ValueError("Only a compatible passing qualification can promote a verifier")
            qualification = self.get_qualification_profile_revision(
                decision.qualification_profile_revision_id
            )
            if not qualification.promotable:
                raise ValueError("This qualification profile can never grant promotion")
        elif qualification_decision_id is not None:
            self.get_decision(qualification_decision_id)
        now = _now()
        event_id = _new_id("verifier-alias-event")
        with self.db._lock:
            previous = self.db._conn.execute(
                """
                SELECT revision_id FROM verifier_aliases
                WHERE profile_id = ? AND alias = ?
                """,
                (revision.profile_id, clean_alias),
            ).fetchone()
            self.db._conn.execute(
                """
                INSERT INTO verifier_alias_events (
                    id, profile_id, alias, previous_revision_id, revision_id,
                    qualification_decision_id, override, note, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    revision.profile_id,
                    clean_alias,
                    None if previous is None else previous["revision_id"],
                    revision_id,
                    qualification_decision_id,
                    int(override),
                    note,
                    now,
                ),
            )
            self.db._conn.execute(
                """
                INSERT INTO verifier_aliases (profile_id, alias, revision_id, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(profile_id, alias) DO UPDATE SET
                    revision_id = excluded.revision_id,
                    updated_at = excluded.updated_at
                """,
                (revision.profile_id, clean_alias, revision_id, now),
            )
            self.db._conn.commit()
        return self.get_alias(revision.profile_id, clean_alias)

    def get_alias(self, profile_id: str, alias: str) -> VerifierAlias:
        row = self.db._conn.execute(
            "SELECT * FROM verifier_aliases WHERE profile_id = ? AND alias = ?",
            (profile_id, alias),
        ).fetchone()
        if row is None:
            raise KeyError(f"Verifier profile {profile_id} has no {alias!r} alias")
        return _alias(row)

    def list_alias_history(
        self, profile_id: str, *, alias: Optional[str] = None
    ) -> List[VerifierAliasEvent]:
        if alias:
            rows = self.db._conn.execute(
                """
                SELECT * FROM verifier_alias_events
                WHERE profile_id = ? AND alias = ? ORDER BY created_at, id
                """,
                (profile_id, alias),
            ).fetchall()
        else:
            rows = self.db._conn.execute(
                """
                SELECT * FROM verifier_alias_events
                WHERE profile_id = ? ORDER BY created_at, id
                """,
                (profile_id,),
            ).fetchall()
        return [_alias_event(row) for row in rows]

    def bind_revision(
        self,
        revision_id: str,
        *,
        domain_kind: str,
        domain_id: str,
        role: str = "verifier",
        qualification_decision_id: Optional[str] = None,
        legacy_unqualified: bool = False,
        development_exposed: bool = False,
        context: Optional[Mapping[str, Any]] = None,
    ) -> ResolvedVerifierBinding:
        self.get_profile_revision(revision_id)
        if qualification_decision_id:
            decision = self.get_decision(qualification_decision_id)
            calibration = self.get_calibration(decision.calibration_id)
            if calibration.verifier_revision_id != revision_id:
                raise ValueError("Qualification decision belongs to a different verifier revision")
        clean_context = scrub_secrets(dict(context or {}))
        identity = {
            "verifier_revision_id": revision_id,
            "domain_kind": domain_kind,
            "domain_id": domain_id,
            "role": role,
            "qualification_decision_id": qualification_decision_id,
            "legacy_unqualified": bool(legacy_unqualified),
            "development_exposed": bool(development_exposed),
            "context": clean_context,
        }
        binding_hash = content_hash(identity)
        existing = self.db._conn.execute(
            "SELECT * FROM verifier_bindings WHERE binding_hash = ?", (binding_hash,)
        ).fetchone()
        if existing is not None:
            return _binding(existing)
        identifier = _new_id("verifier-binding")
        now = _now()
        with self.db._lock:
            self.db._conn.execute(
                """
                INSERT INTO verifier_bindings (
                    id, verifier_revision_id, domain_kind, domain_id, role,
                    qualification_decision_id, legacy_unqualified,
                    development_exposed, binding_hash, context_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    revision_id,
                    domain_kind,
                    domain_id,
                    role,
                    qualification_decision_id,
                    int(legacy_unqualified),
                    int(development_exposed),
                    binding_hash,
                    _canonical_json(clean_context),
                    now,
                ),
            )
            self.db._conn.commit()
        return self.get_binding(identifier)

    def get_binding(self, binding_id: str) -> ResolvedVerifierBinding:
        row = self.db._conn.execute(
            "SELECT * FROM verifier_bindings WHERE id = ?", (binding_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown verifier binding: {binding_id}")
        return _binding(row)

    def list_bindings(
        self,
        *,
        revision_id: Optional[str] = None,
        domain_kind: Optional[str] = None,
        domain_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ResolvedVerifierBinding]:
        clauses: List[str] = []
        values: List[Any] = []
        for column, value in (
            ("verifier_revision_id", revision_id),
            ("domain_kind", domain_kind),
            ("domain_id", domain_id),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                values.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        values.extend([max(1, min(int(limit), 1000)), max(0, int(offset))])
        rows = self.db._conn.execute(
            "SELECT * FROM verifier_bindings"
            + where
            + " ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
            values,
        ).fetchall()
        return [_binding(row) for row in rows]

    def runtime_compatibility(
        self, revision_id: str, runtime_identity: Mapping[str, Any]
    ) -> Dict[str, Any]:
        revision = self.get_profile_revision(revision_id)
        current_hash = content_hash(scrub_secrets(runtime_identity))
        compatible = current_hash == revision.runtime_contract_hash
        return {
            "verifier_revision_id": revision_id,
            "expected_runtime_hash": revision.runtime_contract_hash,
            "current_runtime_hash": current_hash,
            "state": "compatible" if compatible else "stale_runtime",
            "compatible": compatible,
        }


__all__ = [
    "PROTECTED_SOURCE_PURPOSES",
    "QUALIFICATION_TEMPLATES",
    "SOURCE_KINDS",
    "VERIFIER_FAMILIES",
    "VerifierLabStore",
    "content_hash",
    "scrub_secrets",
]
