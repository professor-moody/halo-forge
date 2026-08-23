"""SQLite persistence for immutable reward systems and integrity evidence."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from halo_forge.run_db.db import RunDatabase

from .models import (
    Page,
    RewardAuditProtocol,
    RewardAuditProtocolRevision,
    RewardIntegrityAudit,
    RewardIntegrityBinding,
    RewardIntegrityDecision,
    RewardIntegrityMetric,
    RewardIntegrityObservation,
    RewardIntegrityProfile,
    RewardIntegrityProfileRevision,
    RewardIntegritySample,
    RewardSystem,
    RewardSystemAuditor,
    RewardSystemRevision,
    TrainingSignalShard,
)
from .storage import canonical_json, content_hash


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _loads(value: Any, default: Any) -> Any:
    try:
        return json.loads(value) if value is not None else default
    except (TypeError, json.JSONDecodeError):
        return default


def _dictish(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    return {"message": value}


def _page(limit: int, offset: int) -> tuple[int, int]:
    return max(1, min(int(limit), 1000)), max(0, int(offset))


def _identity(row: sqlite3.Row, cls: Any) -> Any:
    return cls(
        id=str(row["id"]),
        name=str(row["name"]),
        description=row["description"],
        latest_revision_id=row["latest_revision_id"],
        archived=bool(row["archived"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


class RewardIntegrityStore:
    def __init__(self, db: RunDatabase):
        self.db = db

    # -- reward systems -------------------------------------------------

    def create_system(
        self, *, name: str, description: Optional[str] = None, system_id: Optional[str] = None
    ) -> RewardSystem:
        name = str(name).strip()
        if not name:
            raise ValueError("reward system name is required")
        identifier, now = system_id or _id("reward-system"), _now()
        with self.db._lock:
            try:
                self.db._conn.execute(
                    "INSERT INTO reward_systems "
                    "(id,name,description,created_at,updated_at) VALUES (?,?,?,?,?)",
                    (identifier, name, description, now, now),
                )
                self.db._conn.commit()
            except sqlite3.IntegrityError as exc:
                self.db._conn.rollback()
                raise ValueError(f"reward system {name!r} already exists") from exc
        return self.get_system(identifier)

    def get_system(self, system_id: str) -> RewardSystem:
        row = self.db._conn.execute(
            "SELECT * FROM reward_systems WHERE id=?", (system_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown reward system: {system_id}")
        return _identity(row, RewardSystem)

    def list_systems(
        self, *, include_archived: bool = False, limit: int = 100, offset: int = 0
    ) -> Page[RewardSystem]:
        limit, offset = _page(limit, offset)
        where = "" if include_archived else "WHERE archived=0"
        total = int(
            self.db._conn.execute(f"SELECT COUNT(*) FROM reward_systems {where}").fetchone()[0]
        )
        rows = self.db._conn.execute(
            f"SELECT * FROM reward_systems {where} ORDER BY name,id LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return Page([_identity(row, RewardSystem) for row in rows], total, limit, offset)

    def _auditors(self, revision_id: str) -> List[RewardSystemAuditor]:
        rows = self.db._conn.execute(
            "SELECT * FROM reward_system_auditors "
            "WHERE reward_system_revision_id=? ORDER BY ordinal",
            (revision_id,),
        ).fetchall()
        return [
            RewardSystemAuditor(
                reward_system_revision_id=str(row["reward_system_revision_id"]),
                ordinal=int(row["ordinal"]),
                role=str(row["role"]),
                verifier_revision_id=str(row["verifier_revision_id"]),
                correlated=bool(row["correlated"]),
                correlation_reasons=list(_loads(row["correlation_reasons_json"], [])),
                configuration=dict(_loads(row["configuration_json"], {})),
            )
            for row in rows
        ]

    def _system_revision(self, row: sqlite3.Row) -> RewardSystemRevision:
        return RewardSystemRevision(
            id=str(row["id"]),
            system_id=str(row["system_id"]),
            revision_number=int(row["revision_number"]),
            content_hash=str(row["content_hash"]),
            optimizer_verifier_revision_id=str(row["optimizer_verifier_revision_id"]),
            modality=str(row["modality"]),
            task_type=str(row["task_type"]),
            input_mapping=dict(_loads(row["input_mapping_json"], {})),
            reward_mapping=dict(_loads(row["reward_mapping_json"], {})),
            definition=dict(_loads(row["definition_json"], {})),
            runtime_contract_hash=str(row["runtime_contract_hash"]),
            created_at=str(row["created_at"]),
            auditors=self._auditors(str(row["id"])),
        )

    def create_system_revision(
        self,
        system_id: str,
        *,
        optimizer_verifier_revision_id: str,
        modality: str,
        task_type: str,
        auditors: Sequence[Mapping[str, Any]],
        input_mapping: Optional[Mapping[str, Any]] = None,
        reward_mapping: Optional[Mapping[str, Any]] = None,
        definition: Optional[Mapping[str, Any]] = None,
        runtime_contract_hash: Optional[str] = None,
        revision_id: Optional[str] = None,
    ) -> RewardSystemRevision:
        # Reward systems may contain endpoint and runtime descriptors copied
        # from guided forms.  Persist their reproducible, credential-free
        # shape only; credentials remain in the provider configuration that
        # resolves the verifier revision at execution time.
        from halo_forge.verifier_lab.store import scrub_secrets

        self.get_system(system_id)
        if (
            self.db._conn.execute(
                "SELECT 1 FROM verifier_profile_revisions WHERE id=?",
                (optimizer_verifier_revision_id,),
            ).fetchone()
            is None
        ):
            raise KeyError(f"unknown optimizer verifier revision: {optimizer_verifier_revision_id}")
        resolved_auditors: List[Dict[str, Any]] = []
        for ordinal, raw_value in enumerate(auditors):
            raw = dict(raw_value)
            verifier_id = str(raw.get("verifier_revision_id") or "").strip()
            if not verifier_id:
                raise ValueError("every reward-system auditor requires verifier_revision_id")
            if (
                self.db._conn.execute(
                    "SELECT 1 FROM verifier_profile_revisions WHERE id=?", (verifier_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(f"unknown auditor verifier revision: {verifier_id}")
            resolved_auditors.append(
                {
                    "ordinal": ordinal,
                    "role": str(raw.get("role", "diagnostic")).strip().lower(),
                    "verifier_revision_id": verifier_id,
                    "correlated": bool(raw.get("correlated", False)),
                    "correlation_reasons": sorted(
                        {str(item) for item in raw.get("correlation_reasons", [])}
                    ),
                    "configuration": scrub_secrets(raw.get("configuration") or {}),
                }
            )
        primary = [item for item in resolved_auditors if item["role"] == "primary_sentinel"]
        diagnostic = [item for item in resolved_auditors if item["role"] == "diagnostic"]
        if len(primary) != 1:
            raise ValueError("a reward system requires exactly one primary sentinel")
        if len(diagnostic) > 3:
            raise ValueError("a reward system supports at most three diagnostic auditors")
        if any(
            item["role"] not in {"primary_sentinel", "diagnostic"} for item in resolved_auditors
        ):
            raise ValueError("unknown reward-system auditor role")
        auditor_ids = [item["verifier_revision_id"] for item in resolved_auditors]
        if len(auditor_ids) != len(set(auditor_ids)):
            raise ValueError("a verifier revision may occur only once in a reward system")
        identity = {
            "optimizer_verifier_revision_id": optimizer_verifier_revision_id,
            "modality": str(modality).strip().lower(),
            "task_type": str(task_type).strip().lower(),
            "input_mapping": scrub_secrets(dict(input_mapping or {})),
            "reward_mapping": scrub_secrets(dict(reward_mapping or {})),
            "definition": scrub_secrets(dict(definition or {})),
            "auditors": resolved_auditors,
        }
        if not identity["modality"] or not identity["task_type"]:
            raise ValueError("reward-system modality and task_type are required")
        revision_hash = content_hash(identity)
        runtime_hash = runtime_contract_hash or content_hash(
            {
                "optimizer": optimizer_verifier_revision_id,
                "auditors": auditor_ids,
                "runtime": identity["definition"].get("runtime_requirements", {}),
            }
        )
        identifier, now = revision_id or _id("reward-system-revision"), _now()
        with self.db._lock:
            existing = self.db._conn.execute(
                "SELECT * FROM reward_system_revisions WHERE system_id=? AND content_hash=?",
                (system_id, revision_hash),
            ).fetchone()
            if existing is not None:
                return self._system_revision(existing)
            number = int(
                self.db._conn.execute(
                    "SELECT COALESCE(MAX(revision_number),0)+1 FROM reward_system_revisions "
                    "WHERE system_id=?",
                    (system_id,),
                ).fetchone()[0]
            )
            try:
                self.db._conn.execute(
                    """
                    INSERT INTO reward_system_revisions
                        (id,system_id,revision_number,content_hash,
                         optimizer_verifier_revision_id,modality,task_type,
                         input_mapping_json,reward_mapping_json,definition_json,
                         runtime_contract_hash,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        identifier,
                        system_id,
                        number,
                        revision_hash,
                        optimizer_verifier_revision_id,
                        identity["modality"],
                        identity["task_type"],
                        canonical_json(identity["input_mapping"]),
                        canonical_json(identity["reward_mapping"]),
                        canonical_json(identity["definition"]),
                        runtime_hash,
                        now,
                    ),
                )
                for item in resolved_auditors:
                    self.db._conn.execute(
                        """
                        INSERT INTO reward_system_auditors
                            (reward_system_revision_id,ordinal,role,verifier_revision_id,
                             correlated,correlation_reasons_json,configuration_json)
                        VALUES (?,?,?,?,?,?,?)
                        """,
                        (
                            identifier,
                            item["ordinal"],
                            item["role"],
                            item["verifier_revision_id"],
                            int(item["correlated"]),
                            canonical_json(item["correlation_reasons"]),
                            canonical_json(item["configuration"]),
                        ),
                    )
                self.db._conn.execute(
                    "UPDATE reward_systems SET latest_revision_id=?,updated_at=? WHERE id=?",
                    (identifier, now, system_id),
                )
                self.db._conn.commit()
            except Exception:
                self.db._conn.rollback()
                raise
        return self.get_system_revision(identifier)

    def get_system_revision(self, revision_id: str) -> RewardSystemRevision:
        row = self.db._conn.execute(
            "SELECT * FROM reward_system_revisions WHERE id=?", (revision_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown reward system revision: {revision_id}")
        return self._system_revision(row)

    def list_system_revisions(
        self, *, system_id: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> Page[RewardSystemRevision]:
        limit, offset = _page(limit, offset)
        where, params = ("WHERE system_id=?", [system_id]) if system_id else ("", [])
        total = int(
            self.db._conn.execute(
                f"SELECT COUNT(*) FROM reward_system_revisions {where}", params
            ).fetchone()[0]
        )
        rows = self.db._conn.execute(
            f"SELECT * FROM reward_system_revisions {where} "
            "ORDER BY created_at DESC,id LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        return Page([self._system_revision(row) for row in rows], total, limit, offset)

    # -- generic immutable named definitions ----------------------------

    def _create_named(
        self, table: str, *, name: str, description: Optional[str], identifier: str
    ) -> None:
        name, now = str(name).strip(), _now()
        if not name:
            raise ValueError("name is required")
        with self.db._lock:
            self.db._conn.execute(
                f"INSERT INTO {table} (id,name,description,created_at,updated_at) VALUES (?,?,?,?,?)",
                (identifier, name, description, now, now),
            )
            self.db._conn.commit()

    def _get_named(self, table: str, identifier: str, cls: Any) -> Any:
        row = self.db._conn.execute(f"SELECT * FROM {table} WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError(f"unknown {table.rstrip('s')}: {identifier}")
        return _identity(row, cls)

    def _list_named(self, table: str, cls: Any, *, limit: int, offset: int) -> Page[Any]:
        limit, offset = _page(limit, offset)
        total = int(
            self.db._conn.execute(f"SELECT COUNT(*) FROM {table} WHERE archived=0").fetchone()[0]
        )
        rows = self.db._conn.execute(
            f"SELECT * FROM {table} WHERE archived=0 ORDER BY name,id LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return Page([_identity(row, cls) for row in rows], total, limit, offset)

    def create_protocol(
        self, *, name: str, description: Optional[str] = None, protocol_id: Optional[str] = None
    ) -> RewardAuditProtocol:
        identifier = protocol_id or _id("reward-audit-protocol")
        self._create_named(
            "reward_audit_protocols", name=name, description=description, identifier=identifier
        )
        return self.get_protocol(identifier)

    def get_protocol(self, protocol_id: str) -> RewardAuditProtocol:
        return self._get_named("reward_audit_protocols", protocol_id, RewardAuditProtocol)

    def list_protocols(self, *, limit: int = 100, offset: int = 0) -> Page[RewardAuditProtocol]:
        return self._list_named(
            "reward_audit_protocols", RewardAuditProtocol, limit=limit, offset=offset
        )

    def create_protocol_revision(
        self,
        protocol_id: str,
        definition: Mapping[str, Any],
        *,
        capture_mode: Optional[str] = None,
        revision_id: Optional[str] = None,
    ) -> RewardAuditProtocolRevision:
        self.get_protocol(protocol_id)
        resolved = dict(definition)
        mode = str(capture_mode or resolved.get("capture_mode", "custom")).strip().lower()
        if mode not in {"balanced_256", "broad_512", "exhaustive", "custom"}:
            raise ValueError("unknown reward audit capture mode")
        resolved["capture_mode"] = mode
        digest, identifier, now = (
            content_hash(resolved),
            revision_id or _id("reward-audit-protocol-revision"),
            _now(),
        )
        with self.db._lock:
            existing = self.db._conn.execute(
                "SELECT * FROM reward_audit_protocol_revisions WHERE protocol_id=? AND content_hash=?",
                (protocol_id, digest),
            ).fetchone()
            if existing is not None:
                return self._protocol_revision(existing)
            number = int(
                self.db._conn.execute(
                    "SELECT COALESCE(MAX(revision_number),0)+1 FROM reward_audit_protocol_revisions WHERE protocol_id=?",
                    (protocol_id,),
                ).fetchone()[0]
            )
            self.db._conn.execute(
                "INSERT INTO reward_audit_protocol_revisions "
                "(id,protocol_id,revision_number,content_hash,capture_mode,definition_json,created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (identifier, protocol_id, number, digest, mode, canonical_json(resolved), now),
            )
            self.db._conn.execute(
                "UPDATE reward_audit_protocols SET latest_revision_id=?,updated_at=? WHERE id=?",
                (identifier, now, protocol_id),
            )
            self.db._conn.commit()
        return self.get_protocol_revision(identifier)

    @staticmethod
    def _protocol_revision(row: sqlite3.Row) -> RewardAuditProtocolRevision:
        return RewardAuditProtocolRevision(
            id=str(row["id"]),
            protocol_id=str(row["protocol_id"]),
            revision_number=int(row["revision_number"]),
            content_hash=str(row["content_hash"]),
            capture_mode=str(row["capture_mode"]),
            definition=dict(_loads(row["definition_json"], {})),
            created_at=str(row["created_at"]),
        )

    def get_protocol_revision(self, revision_id: str) -> RewardAuditProtocolRevision:
        row = self.db._conn.execute(
            "SELECT * FROM reward_audit_protocol_revisions WHERE id=?", (revision_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown reward audit protocol revision: {revision_id}")
        return self._protocol_revision(row)

    def list_protocol_revisions(
        self, *, protocol_id: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> Page[RewardAuditProtocolRevision]:
        limit, offset = _page(limit, offset)
        where, params = ("WHERE protocol_id=?", [protocol_id]) if protocol_id else ("", [])
        total = int(
            self.db._conn.execute(
                f"SELECT COUNT(*) FROM reward_audit_protocol_revisions {where}", params
            ).fetchone()[0]
        )
        rows = self.db._conn.execute(
            f"SELECT * FROM reward_audit_protocol_revisions {where} ORDER BY created_at DESC,id LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        return Page([self._protocol_revision(row) for row in rows], total, limit, offset)

    def create_integrity_profile(
        self, *, name: str, description: Optional[str] = None, profile_id: Optional[str] = None
    ) -> RewardIntegrityProfile:
        identifier = profile_id or _id("reward-integrity-profile")
        self._create_named(
            "reward_integrity_profiles", name=name, description=description, identifier=identifier
        )
        return self.get_integrity_profile(identifier)

    def get_integrity_profile(self, profile_id: str) -> RewardIntegrityProfile:
        return self._get_named("reward_integrity_profiles", profile_id, RewardIntegrityProfile)

    def list_integrity_profiles(
        self, *, limit: int = 100, offset: int = 0
    ) -> Page[RewardIntegrityProfile]:
        return self._list_named(
            "reward_integrity_profiles", RewardIntegrityProfile, limit=limit, offset=offset
        )

    def create_integrity_profile_revision(
        self,
        profile_id: str,
        *,
        template_kind: str,
        requirements: Mapping[str, Any],
        promotable: Optional[bool] = None,
        revision_id: Optional[str] = None,
    ) -> RewardIntegrityProfileRevision:
        self.get_integrity_profile(profile_id)
        kind = str(template_kind).strip().lower()
        if kind not in {"strict_integrity", "human_aligned_integrity", "exploratory", "custom"}:
            raise ValueError("unknown reward integrity profile template")
        can_promote = kind != "exploratory" if promotable is None else bool(promotable)
        if kind == "exploratory" and can_promote:
            raise ValueError("exploratory integrity profiles cannot be promotable")
        identity = {
            "template_kind": kind,
            "promotable": can_promote,
            "requirements": dict(requirements),
        }
        digest, identifier, now = (
            content_hash(identity),
            revision_id or _id("reward-integrity-profile-revision"),
            _now(),
        )
        with self.db._lock:
            existing = self.db._conn.execute(
                "SELECT * FROM reward_integrity_profile_revisions WHERE profile_id=? AND content_hash=?",
                (profile_id, digest),
            ).fetchone()
            if existing is not None:
                return self._integrity_profile_revision(existing)
            number = int(
                self.db._conn.execute(
                    "SELECT COALESCE(MAX(revision_number),0)+1 FROM reward_integrity_profile_revisions WHERE profile_id=?",
                    (profile_id,),
                ).fetchone()[0]
            )
            self.db._conn.execute(
                "INSERT INTO reward_integrity_profile_revisions "
                "(id,profile_id,revision_number,content_hash,template_kind,promotable,requirements_json,created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    profile_id,
                    number,
                    digest,
                    kind,
                    int(can_promote),
                    canonical_json(identity["requirements"]),
                    now,
                ),
            )
            self.db._conn.execute(
                "UPDATE reward_integrity_profiles SET latest_revision_id=?,updated_at=? WHERE id=?",
                (identifier, now, profile_id),
            )
            self.db._conn.commit()
        return self.get_integrity_profile_revision(identifier)

    @staticmethod
    def _integrity_profile_revision(row: sqlite3.Row) -> RewardIntegrityProfileRevision:
        return RewardIntegrityProfileRevision(
            id=str(row["id"]),
            profile_id=str(row["profile_id"]),
            revision_number=int(row["revision_number"]),
            content_hash=str(row["content_hash"]),
            template_kind=str(row["template_kind"]),
            promotable=bool(row["promotable"]),
            requirements=dict(_loads(row["requirements_json"], {})),
            created_at=str(row["created_at"]),
        )

    def get_integrity_profile_revision(self, revision_id: str) -> RewardIntegrityProfileRevision:
        row = self.db._conn.execute(
            "SELECT * FROM reward_integrity_profile_revisions WHERE id=?", (revision_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown reward integrity profile revision: {revision_id}")
        return self._integrity_profile_revision(row)

    def list_integrity_profile_revisions(
        self, *, profile_id: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> Page[RewardIntegrityProfileRevision]:
        limit, offset = _page(limit, offset)
        where, params = ("WHERE profile_id=?", [profile_id]) if profile_id else ("", [])
        total = int(
            self.db._conn.execute(
                f"SELECT COUNT(*) FROM reward_integrity_profile_revisions {where}", params
            ).fetchone()[0]
        )
        rows = self.db._conn.execute(
            f"SELECT * FROM reward_integrity_profile_revisions {where} ORDER BY created_at DESC,id LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        return Page([self._integrity_profile_revision(row) for row in rows], total, limit, offset)

    # -- sealed traces --------------------------------------------------

    def create_signal_shard(self, **values: Any) -> TrainingSignalShard:
        from halo_forge.verifier_lab.store import scrub_secrets

        required = (
            "run_id",
            "reward_system_revision_id",
            "protocol_revision_id",
            "capability_id",
            "capture_fidelity",
            "boundary_unit",
            "trace_hash",
            "retained_set_hash",
            "producer_model_hash",
            "checkpoint_hash",
            "storage_path",
            "manifest_hash",
        )
        missing = [name for name in required if not str(values.get(name, "")).strip()]
        if missing:
            raise ValueError(f"missing signal shard fields: {', '.join(missing)}")
        identifier, now = values.get("shard_id") or _id("training-signal-shard"), _now()
        with self.db._lock:
            existing = self.db._conn.execute(
                "SELECT * FROM training_signal_shards WHERE run_id=? AND trace_hash=?",
                (values["run_id"], values["trace_hash"]),
            ).fetchone()
            if existing is not None:
                return self._signal_shard(existing)
            self.db._conn.execute(
                """INSERT INTO training_signal_shards
                (id,run_id,direct_run_segment_id,trial_segment_id,reward_system_revision_id,
                 protocol_revision_id,capability_id,capture_fidelity,boundary_unit,boundary_value,
                 trace_hash,retained_set_hash,event_count,distinct_record_count,aggregate_json,
                 dataset_identity_json,producer_model_hash,checkpoint_hash,runtime_identity_json,
                 storage_path,manifest_hash,sealed,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    identifier,
                    values["run_id"],
                    values.get("direct_run_segment_id"),
                    values.get("trial_segment_id"),
                    values["reward_system_revision_id"],
                    values["protocol_revision_id"],
                    values["capability_id"],
                    values["capture_fidelity"],
                    values["boundary_unit"],
                    int(values.get("boundary_value", 0)),
                    values["trace_hash"],
                    values["retained_set_hash"],
                    int(values.get("event_count", 0)),
                    int(values.get("distinct_record_count", 0)),
                    canonical_json(values.get("aggregate") or {}),
                    canonical_json(scrub_secrets(values.get("dataset_identity") or {})),
                    values["producer_model_hash"],
                    values["checkpoint_hash"],
                    canonical_json(scrub_secrets(values.get("runtime_identity") or {})),
                    str(values["storage_path"]),
                    values["manifest_hash"],
                    1,
                    now,
                ),
            )
            self.db._conn.commit()
        return self.get_signal_shard(identifier)

    @staticmethod
    def _signal_shard(row: sqlite3.Row) -> TrainingSignalShard:
        return TrainingSignalShard(
            id=str(row["id"]),
            run_id=str(row["run_id"]),
            direct_run_segment_id=row["direct_run_segment_id"],
            trial_segment_id=row["trial_segment_id"],
            reward_system_revision_id=str(row["reward_system_revision_id"]),
            protocol_revision_id=str(row["protocol_revision_id"]),
            capability_id=str(row["capability_id"]),
            capture_fidelity=str(row["capture_fidelity"]),
            boundary_unit=str(row["boundary_unit"]),
            boundary_value=int(row["boundary_value"]),
            trace_hash=str(row["trace_hash"]),
            retained_set_hash=str(row["retained_set_hash"]),
            event_count=int(row["event_count"]),
            distinct_record_count=int(row["distinct_record_count"]),
            aggregate=dict(_loads(row["aggregate_json"], {})),
            dataset_identity=dict(_loads(row["dataset_identity_json"], {})),
            producer_model_hash=str(row["producer_model_hash"]),
            checkpoint_hash=str(row["checkpoint_hash"]),
            runtime_identity=dict(_loads(row["runtime_identity_json"], {})),
            storage_path=str(row["storage_path"]),
            manifest_hash=str(row["manifest_hash"]),
            sealed=bool(row["sealed"]),
            created_at=str(row["created_at"]),
        )

    def get_signal_shard(self, shard_id: str) -> TrainingSignalShard:
        row = self.db._conn.execute(
            "SELECT * FROM training_signal_shards WHERE id=?", (shard_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown training signal shard: {shard_id}")
        return self._signal_shard(row)

    def list_signal_shards(
        self, *, run_id: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> Page[TrainingSignalShard]:
        limit, offset = _page(limit, offset)
        where, params = ("WHERE run_id=?", [run_id]) if run_id else ("", [])
        total = int(
            self.db._conn.execute(
                f"SELECT COUNT(*) FROM training_signal_shards {where}", params
            ).fetchone()[0]
        )
        rows = self.db._conn.execute(
            f"SELECT * FROM training_signal_shards {where} ORDER BY created_at DESC,id LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        return Page([self._signal_shard(row) for row in rows], total, limit, offset)

    # -- audit lifecycle + append-only evidence -------------------------

    def create_audit(self, **values: Any) -> RewardIntegrityAudit:
        from halo_forge.verifier_lab.store import scrub_secrets

        required = (
            "run_id",
            "signal_shard_id",
            "reward_system_revision_id",
            "protocol_revision_id",
            "integrity_profile_revision_id",
            "runtime_identity_hash",
            "reuse_key",
        )
        missing = [name for name in required if not str(values.get(name, "")).strip()]
        if missing:
            raise ValueError(f"missing reward audit fields: {', '.join(missing)}")
        identifier, now = values.get("audit_id") or _id("reward-integrity-audit"), _now()
        with self.db._lock:
            if values.get("status") == "completed":
                existing = self.db._conn.execute(
                    "SELECT * FROM reward_integrity_audits WHERE reuse_key=? AND status='completed'",
                    (values["reuse_key"],),
                ).fetchone()
                if existing is not None:
                    return self._audit(existing)
            self.db._conn.execute(
                """INSERT INTO reward_integrity_audits
                (id,run_id,direct_run_segment_id,trial_segment_id,signal_shard_id,
                 reward_system_revision_id,protocol_revision_id,integrity_profile_revision_id,
                 development_suite_revision_id,status,stage,processed_samples,total_samples,
                 distinct_record_count,request_json,runtime_identity_json,runtime_identity_hash,
                 reuse_key,artifact_path,manifest_hash,work_item_id,cancel_requested,retry_count,
                 error,created_at,updated_at,started_at,completed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    identifier,
                    values["run_id"],
                    values.get("direct_run_segment_id"),
                    values.get("trial_segment_id"),
                    values["signal_shard_id"],
                    values["reward_system_revision_id"],
                    values["protocol_revision_id"],
                    values["integrity_profile_revision_id"],
                    values.get("development_suite_revision_id"),
                    values.get("status", "queued"),
                    values.get("stage", "queued"),
                    int(values.get("processed_samples", 0)),
                    values.get("total_samples"),
                    int(values.get("distinct_record_count", 0)),
                    canonical_json(scrub_secrets(values.get("request") or {})),
                    canonical_json(scrub_secrets(values.get("runtime_identity") or {})),
                    values["runtime_identity_hash"],
                    values["reuse_key"],
                    values.get("artifact_path"),
                    values.get("manifest_hash"),
                    values.get("work_item_id"),
                    int(bool(values.get("cancel_requested", False))),
                    int(values.get("retry_count", 0)),
                    values.get("error"),
                    now,
                    now,
                    now if values.get("status") == "running" else None,
                    now if values.get("status") == "completed" else None,
                ),
            )
            self.db._conn.commit()
        return self.get_audit(identifier)

    @staticmethod
    def _audit(row: sqlite3.Row) -> RewardIntegrityAudit:
        return RewardIntegrityAudit(
            id=str(row["id"]),
            run_id=str(row["run_id"]),
            direct_run_segment_id=row["direct_run_segment_id"],
            trial_segment_id=row["trial_segment_id"],
            signal_shard_id=str(row["signal_shard_id"]),
            reward_system_revision_id=str(row["reward_system_revision_id"]),
            protocol_revision_id=str(row["protocol_revision_id"]),
            integrity_profile_revision_id=str(row["integrity_profile_revision_id"]),
            development_suite_revision_id=row["development_suite_revision_id"],
            status=str(row["status"]),
            stage=str(row["stage"]),
            processed_samples=int(row["processed_samples"]),
            total_samples=None if row["total_samples"] is None else int(row["total_samples"]),
            distinct_record_count=int(row["distinct_record_count"]),
            request=dict(_loads(row["request_json"], {})),
            runtime_identity=dict(_loads(row["runtime_identity_json"], {})),
            runtime_identity_hash=str(row["runtime_identity_hash"]),
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

    def get_audit(self, audit_id: str) -> RewardIntegrityAudit:
        row = self.db._conn.execute(
            "SELECT * FROM reward_integrity_audits WHERE id=?", (audit_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown reward integrity audit: {audit_id}")
        return self._audit(row)

    def list_audits(
        self,
        *,
        run_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Page[RewardIntegrityAudit]:
        limit, offset = _page(limit, offset)
        clauses, params = [], []
        for column, value in (("run_id", run_id), ("status", status)):
            if value is not None:
                clauses.append(f"{column}=?")
                params.append(value)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        total = int(
            self.db._conn.execute(
                f"SELECT COUNT(*) FROM reward_integrity_audits {where}", params
            ).fetchone()[0]
        )
        rows = self.db._conn.execute(
            f"SELECT * FROM reward_integrity_audits {where} ORDER BY created_at DESC,id LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        return Page([self._audit(row) for row in rows], total, limit, offset)

    def update_audit(self, audit_id: str, **changes: Any) -> RewardIntegrityAudit:
        current = self.get_audit(audit_id)
        allowed = {
            "status",
            "stage",
            "processed_samples",
            "total_samples",
            "distinct_record_count",
            "artifact_path",
            "manifest_hash",
            "work_item_id",
            "cancel_requested",
            "retry_count",
            "error",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unknown audit lifecycle fields: {sorted(unknown)}")
        if current.status == "completed" and changes:
            identical = all(getattr(current, key) == value for key, value in changes.items())
            if not identical:
                raise ValueError("completed reward audit identity is immutable")
            return current
        assignments, params = ["updated_at=?"], [_now()]
        for key, value in changes.items():
            assignments.append(f"{key}=?")
            params.append(int(value) if key == "cancel_requested" else value)
        if changes.get("status") == "running" and current.started_at is None:
            assignments.append("started_at=?")
            params.append(_now())
        if changes.get("status") == "completed":
            assignments.append("completed_at=?")
            params.append(_now())
        params.append(audit_id)
        with self.db._lock:
            self.db._conn.execute(
                f"UPDATE reward_integrity_audits SET {','.join(assignments)} WHERE id=?", params
            )
            self.db._conn.commit()
        return self.get_audit(audit_id)

    def add_sample(self, sample: Mapping[str, Any]) -> RewardIntegritySample:
        now = _now()
        with self.db._lock:
            self.db._conn.execute(
                """INSERT OR IGNORE INTO reward_integrity_samples
                (audit_id,ordinal,snapshot_id,record_id,record_hash,instance_id,group_id,
                 candidate_ordinal,selection_class,diagnostic,input_json,output_json,
                 expected_json,media_json,generation_json,lineage_json,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    sample["audit_id"],
                    int(sample["ordinal"]),
                    sample["snapshot_id"],
                    sample["record_id"],
                    sample["record_hash"],
                    sample["instance_id"],
                    sample["group_id"],
                    int(sample.get("candidate_ordinal", 0)),
                    sample.get("selection_class", "uniform_core"),
                    int(bool(sample.get("diagnostic", False))),
                    canonical_json(sample.get("input") or {}),
                    canonical_json(sample.get("output")),
                    (
                        None
                        if sample.get("expected") is None
                        else canonical_json(sample.get("expected"))
                    ),
                    canonical_json(sample.get("media") or []),
                    canonical_json(sample.get("generation") or {}),
                    canonical_json(sample.get("lineage") or {}),
                    now,
                ),
            )
            self.db._conn.commit()
        row = self.db._conn.execute(
            "SELECT * FROM reward_integrity_samples WHERE audit_id=? AND ordinal=?",
            (sample["audit_id"], int(sample["ordinal"])),
        ).fetchone()
        return self._sample(row)

    def add_samples(self, samples: Sequence[Mapping[str, Any]]) -> None:
        """Insert one bounded hydration page in a single transaction."""

        if not samples:
            return
        now = _now()
        rows = []
        for sample in samples:
            rows.append(
                (
                    sample["audit_id"],
                    int(sample["ordinal"]),
                    sample["snapshot_id"],
                    sample["record_id"],
                    sample["record_hash"],
                    sample["instance_id"],
                    sample["group_id"],
                    int(sample.get("candidate_ordinal", 0)),
                    sample.get("selection_class", "uniform_core"),
                    int(bool(sample.get("diagnostic", False))),
                    canonical_json(sample.get("input") or {}),
                    canonical_json(sample.get("output")),
                    (
                        None
                        if sample.get("expected") is None
                        else canonical_json(sample.get("expected"))
                    ),
                    canonical_json(sample.get("media") or []),
                    canonical_json(sample.get("generation") or {}),
                    canonical_json(sample.get("lineage") or {}),
                    now,
                )
            )
        with self.db._lock:
            self.db._conn.executemany(
                """INSERT OR IGNORE INTO reward_integrity_samples
                (audit_id,ordinal,snapshot_id,record_id,record_hash,instance_id,group_id,
                 candidate_ordinal,selection_class,diagnostic,input_json,output_json,
                 expected_json,media_json,generation_json,lineage_json,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            self.db._conn.commit()

    @staticmethod
    def _sample(row: sqlite3.Row) -> RewardIntegritySample:
        return RewardIntegritySample(
            audit_id=str(row["audit_id"]),
            ordinal=int(row["ordinal"]),
            snapshot_id=str(row["snapshot_id"]),
            record_id=str(row["record_id"]),
            record_hash=str(row["record_hash"]),
            instance_id=str(row["instance_id"]),
            group_id=str(row["group_id"]),
            candidate_ordinal=int(row["candidate_ordinal"]),
            selection_class=str(row["selection_class"]),
            diagnostic=bool(row["diagnostic"]),
            input=dict(_loads(row["input_json"], {})),
            output=_loads(row["output_json"], None),
            expected=_loads(row["expected_json"], None),
            media=list(_loads(row["media_json"], [])),
            generation=dict(_loads(row["generation_json"], {})),
            lineage=dict(_loads(row["lineage_json"], {})),
            created_at=str(row["created_at"]),
        )

    def list_samples(
        self,
        audit_id: str,
        *,
        diagnostic: Optional[bool] = None,
        population: Optional[str] = None,
        outcome: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Page[RewardIntegritySample]:
        limit, offset = _page(limit, offset)
        clauses, params = ["s.audit_id=?"], [audit_id]
        if diagnostic is not None:
            clauses.append("s.diagnostic=?")
            params.append(int(diagnostic))
        if population:
            normalized_population = str(population).strip().lower()
            if normalized_population in {"uniform_core", "core"}:
                clauses.append("s.diagnostic=0")
            elif normalized_population == "diagnostic":
                clauses.append("s.diagnostic=1")
            elif normalized_population not in {"all", "any"}:
                raise ValueError("population must be uniform_core, diagnostic, or all")
        normalized_outcome = str(outcome or "").strip().lower()
        if normalized_outcome:
            if normalized_outcome == "agreement":
                clauses.append(
                    "o.passed IS NOT NULL AND p.passed IS NOT NULL AND o.passed=p.passed"
                )
            elif normalized_outcome == "optimizer_only":
                clauses.append("o.passed=1 AND p.passed=0")
            elif normalized_outcome == "sentinel_only":
                clauses.append("o.passed=0 AND p.passed=1")
            elif normalized_outcome == "disagreement":
                clauses.append(
                    "o.passed IS NOT NULL AND p.passed IS NOT NULL AND o.passed<>p.passed"
                )
            elif normalized_outcome == "error":
                clauses.append("(o.error IS NOT NULL OR p.error IS NOT NULL)")
            elif normalized_outcome not in {"all", "any"}:
                raise ValueError(
                    "outcome must be agreement, disagreement, optimizer_only, "
                    "sentinel_only, error, or all"
                )
        search = str(query or "").strip()
        if search:
            pattern = f"%{search}%"
            clauses.append(
                "(s.record_id LIKE ? OR s.snapshot_id LIKE ? OR "
                "s.input_json LIKE ? OR s.output_json LIKE ?)"
            )
            params.extend((pattern, pattern, pattern, pattern))
        joins = (
            "LEFT JOIN reward_integrity_observations o "
            "ON o.audit_id=s.audit_id AND o.sample_ordinal=s.ordinal AND o.role='optimizer' "
            "LEFT JOIN reward_integrity_observations p "
            "ON p.audit_id=s.audit_id AND p.sample_ordinal=s.ordinal "
            "AND p.role='primary_sentinel'"
        )
        where = " AND ".join(clauses)
        total = int(
            self.db._conn.execute(
                f"SELECT COUNT(DISTINCT s.ordinal) FROM reward_integrity_samples s "
                f"{joins} WHERE {where}",
                params,
            ).fetchone()[0]
        )
        rows = self.db._conn.execute(
            f"SELECT DISTINCT s.* FROM reward_integrity_samples s {joins} "
            f"WHERE {where} ORDER BY s.ordinal LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        return Page([self._sample(row) for row in rows], total, limit, offset)

    def add_observation(self, value: Mapping[str, Any]) -> RewardIntegrityObservation:
        now = _now()
        with self.db._lock:
            self.db._conn.execute(
                """INSERT OR IGNORE INTO reward_integrity_observations
                (audit_id,sample_ordinal,role,auditor_ordinal,verifier_revision_id,reward,
                 normalized_reward,passed,parsed_value_json,raw_output_json,details_json,
                 component_trace_json,latency_ms,error,runtime_identity_json,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    value["audit_id"],
                    int(value["sample_ordinal"]),
                    value["role"],
                    int(value.get("auditor_ordinal", 0)),
                    value["verifier_revision_id"],
                    value.get("reward"),
                    value.get("normalized_reward"),
                    None if value.get("passed") is None else int(bool(value["passed"])),
                    (
                        None
                        if value.get("parsed_value") is None
                        else canonical_json(value.get("parsed_value"))
                    ),
                    (
                        None
                        if value.get("raw_output") is None
                        else canonical_json(value.get("raw_output"))
                    ),
                    canonical_json(value.get("details") or {}),
                    canonical_json(value.get("component_trace") or []),
                    value.get("latency_ms"),
                    value.get("error"),
                    canonical_json(value.get("runtime_identity") or {}),
                    now,
                ),
            )
            self.db._conn.commit()
        row = self.db._conn.execute(
            "SELECT * FROM reward_integrity_observations WHERE audit_id=? AND sample_ordinal=? AND role=? AND auditor_ordinal=?",
            (
                value["audit_id"],
                int(value["sample_ordinal"]),
                value["role"],
                int(value.get("auditor_ordinal", 0)),
            ),
        ).fetchone()
        return self._observation(row)

    def add_observations(self, values: Sequence[Mapping[str, Any]]) -> None:
        """Insert one bounded observation page in a single transaction."""

        if not values:
            return
        now = _now()
        rows = []
        for value in values:
            rows.append(
                (
                    value["audit_id"],
                    int(value["sample_ordinal"]),
                    value["role"],
                    int(value.get("auditor_ordinal", 0)),
                    value["verifier_revision_id"],
                    value.get("reward"),
                    value.get("normalized_reward"),
                    None if value.get("passed") is None else int(bool(value["passed"])),
                    (
                        None
                        if value.get("parsed_value") is None
                        else canonical_json(value.get("parsed_value"))
                    ),
                    (
                        None
                        if value.get("raw_output") is None
                        else canonical_json(value.get("raw_output"))
                    ),
                    canonical_json(value.get("details") or {}),
                    canonical_json(value.get("component_trace") or []),
                    value.get("latency_ms"),
                    value.get("error"),
                    canonical_json(value.get("runtime_identity") or {}),
                    now,
                )
            )
        with self.db._lock:
            self.db._conn.executemany(
                """INSERT OR IGNORE INTO reward_integrity_observations
                (audit_id,sample_ordinal,role,auditor_ordinal,verifier_revision_id,reward,
                 normalized_reward,passed,parsed_value_json,raw_output_json,details_json,
                 component_trace_json,latency_ms,error,runtime_identity_json,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            self.db._conn.commit()

    @staticmethod
    def _observation(row: sqlite3.Row) -> RewardIntegrityObservation:
        return RewardIntegrityObservation(
            audit_id=str(row["audit_id"]),
            sample_ordinal=int(row["sample_ordinal"]),
            role=str(row["role"]),
            auditor_ordinal=int(row["auditor_ordinal"]),
            verifier_revision_id=str(row["verifier_revision_id"]),
            reward=None if row["reward"] is None else float(row["reward"]),
            normalized_reward=(
                None if row["normalized_reward"] is None else float(row["normalized_reward"])
            ),
            passed=None if row["passed"] is None else bool(row["passed"]),
            parsed_value=_loads(row["parsed_value_json"], None),
            raw_output=_loads(row["raw_output_json"], None),
            details=_dictish(_loads(row["details_json"], {})),
            component_trace=list(_loads(row["component_trace_json"], [])),
            latency_ms=None if row["latency_ms"] is None else float(row["latency_ms"]),
            error=row["error"],
            runtime_identity=dict(_loads(row["runtime_identity_json"], {})),
            created_at=str(row["created_at"]),
        )

    def list_observations(
        self, audit_id: str, *, limit: int = 1000, offset: int = 0
    ) -> Page[RewardIntegrityObservation]:
        limit, offset = _page(limit, offset)
        total = int(
            self.db._conn.execute(
                "SELECT COUNT(*) FROM reward_integrity_observations WHERE audit_id=?", (audit_id,)
            ).fetchone()[0]
        )
        rows = self.db._conn.execute(
            "SELECT * FROM reward_integrity_observations WHERE audit_id=? ORDER BY sample_ordinal,role,auditor_ordinal LIMIT ? OFFSET ?",
            (audit_id, limit, offset),
        ).fetchall()
        return Page([self._observation(row) for row in rows], total, limit, offset)

    def add_metrics(self, metrics: Sequence[RewardIntegrityMetric]) -> List[RewardIntegrityMetric]:
        now = _now()
        with self.db._lock:
            for metric in metrics:
                self.db._conn.execute(
                    """INSERT INTO reward_integrity_metrics
                    (audit_id,name,subgroup,population,value,ci_low,ci_high,direction,
                     available,missing_reason,record_count,metadata_json,created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        metric.audit_id,
                        metric.name,
                        metric.subgroup,
                        metric.population,
                        metric.value,
                        metric.ci_low,
                        metric.ci_high,
                        metric.direction,
                        int(metric.available),
                        metric.missing_reason,
                        metric.record_count,
                        canonical_json(metric.metadata),
                        now,
                    ),
                )
            self.db._conn.commit()
        return self.list_metrics(metrics[0].audit_id, limit=1000).items if metrics else []

    @staticmethod
    def _metric(row: sqlite3.Row) -> RewardIntegrityMetric:
        return RewardIntegrityMetric(
            audit_id=str(row["audit_id"]),
            name=str(row["name"]),
            value=None if row["value"] is None else float(row["value"]),
            available=bool(row["available"]),
            record_count=int(row["record_count"]),
            subgroup=str(row["subgroup"]),
            population=str(row["population"]),
            ci_low=None if row["ci_low"] is None else float(row["ci_low"]),
            ci_high=None if row["ci_high"] is None else float(row["ci_high"]),
            direction=row["direction"],
            missing_reason=row["missing_reason"],
            metadata=dict(_loads(row["metadata_json"], {})),
            created_at=str(row["created_at"]),
        )

    def list_metrics(
        self,
        audit_id: str,
        *,
        subgroup: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Page[RewardIntegrityMetric]:
        limit, offset = _page(limit, offset)
        where = "audit_id=?"
        params: list[Any] = [audit_id]
        if subgroup is not None:
            where += " AND subgroup=?"
            params.append(str(subgroup))
        total = int(
            self.db._conn.execute(
                f"SELECT COUNT(*) FROM reward_integrity_metrics WHERE {where}", params
            ).fetchone()[0]
        )
        rows = self.db._conn.execute(
            f"SELECT * FROM reward_integrity_metrics WHERE {where} "
            "ORDER BY name,subgroup,population LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        return Page([self._metric(row) for row in rows], total, limit, offset)

    def add_decision(
        self,
        *,
        audit_id: str,
        integrity_profile_revision_id: str,
        decision: str,
        action: str,
        reasons: Sequence[str],
        evidence: Mapping[str, Any],
        override: bool = False,
        override_note: Optional[str] = None,
        supersedes_decision_id: Optional[str] = None,
        decision_id: Optional[str] = None,
    ) -> RewardIntegrityDecision:
        if decision not in {"pass", "warn", "fail", "incomplete_evidence"}:
            raise ValueError("unknown reward-integrity decision")
        if action not in {"continue", "pause", "stop", "fork", "report_only"}:
            raise ValueError("unknown reward-integrity action")
        if override and not str(override_note or "").strip():
            raise ValueError("an override requires a note")
        identifier, now = decision_id or _id("reward-integrity-decision"), _now()
        with self.db._lock:
            self.db._conn.execute(
                "INSERT INTO reward_integrity_decisions "
                "(id,audit_id,integrity_profile_revision_id,decision,action,reasons_json,evidence_json,override,override_note,supersedes_decision_id,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    audit_id,
                    integrity_profile_revision_id,
                    decision,
                    action,
                    canonical_json(list(reasons)),
                    canonical_json(dict(evidence)),
                    int(override),
                    override_note,
                    supersedes_decision_id,
                    now,
                ),
            )
            self.db._conn.commit()
        return self.get_decision(identifier)

    @staticmethod
    def _decision(row: sqlite3.Row) -> RewardIntegrityDecision:
        return RewardIntegrityDecision(
            id=str(row["id"]),
            audit_id=str(row["audit_id"]),
            integrity_profile_revision_id=str(row["integrity_profile_revision_id"]),
            decision=str(row["decision"]),
            action=str(row["action"]),
            reasons=list(_loads(row["reasons_json"], [])),
            evidence=dict(_loads(row["evidence_json"], {})),
            override=bool(row["override"]),
            override_note=row["override_note"],
            supersedes_decision_id=row["supersedes_decision_id"],
            created_at=str(row["created_at"]),
        )

    def get_decision(self, decision_id: str) -> RewardIntegrityDecision:
        row = self.db._conn.execute(
            "SELECT * FROM reward_integrity_decisions WHERE id=?", (decision_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown reward integrity decision: {decision_id}")
        return self._decision(row)

    def list_decisions(
        self, audit_id: str, *, limit: int = 100, offset: int = 0
    ) -> Page[RewardIntegrityDecision]:
        limit, offset = _page(limit, offset)
        total = int(
            self.db._conn.execute(
                "SELECT COUNT(*) FROM reward_integrity_decisions WHERE audit_id=?", (audit_id,)
            ).fetchone()[0]
        )
        rows = self.db._conn.execute(
            "SELECT * FROM reward_integrity_decisions WHERE audit_id=? ORDER BY created_at,id LIMIT ? OFFSET ?",
            (audit_id, limit, offset),
        ).fetchall()
        return Page([self._decision(row) for row in rows], total, limit, offset)

    def bind(
        self,
        *,
        reward_system_revision_id: str,
        domain_kind: str,
        domain_id: str,
        role: str = "reward_system",
        protocol_revision_id: Optional[str] = None,
        integrity_profile_revision_id: Optional[str] = None,
        audit_id: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        binding_id: Optional[str] = None,
    ) -> RewardIntegrityBinding:
        from halo_forge.verifier_lab.store import scrub_secrets

        payload = {
            "reward_system_revision_id": reward_system_revision_id,
            "protocol_revision_id": protocol_revision_id,
            "integrity_profile_revision_id": integrity_profile_revision_id,
            "audit_id": audit_id,
            "domain_kind": domain_kind,
            "domain_id": domain_id,
            "role": role,
            "context": scrub_secrets(dict(context or {})),
        }
        digest, identifier, now = (
            content_hash(payload),
            binding_id or _id("reward-integrity-binding"),
            _now(),
        )
        with self.db._lock:
            existing = self.db._conn.execute(
                "SELECT * FROM reward_integrity_bindings WHERE binding_hash=?", (digest,)
            ).fetchone()
            if existing is not None:
                return self._binding(existing)
            self.db._conn.execute(
                "INSERT INTO reward_integrity_bindings (id,reward_system_revision_id,protocol_revision_id,integrity_profile_revision_id,audit_id,domain_kind,domain_id,role,binding_hash,context_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    reward_system_revision_id,
                    protocol_revision_id,
                    integrity_profile_revision_id,
                    audit_id,
                    domain_kind,
                    domain_id,
                    role,
                    digest,
                    canonical_json(payload["context"]),
                    now,
                ),
            )
            self.db._conn.commit()
        return self.get_binding(identifier)

    @staticmethod
    def _binding(row: sqlite3.Row) -> RewardIntegrityBinding:
        return RewardIntegrityBinding(
            id=str(row["id"]),
            reward_system_revision_id=str(row["reward_system_revision_id"]),
            protocol_revision_id=row["protocol_revision_id"],
            integrity_profile_revision_id=row["integrity_profile_revision_id"],
            audit_id=row["audit_id"],
            domain_kind=str(row["domain_kind"]),
            domain_id=str(row["domain_id"]),
            role=str(row["role"]),
            binding_hash=str(row["binding_hash"]),
            context=dict(_loads(row["context_json"], {})),
            created_at=str(row["created_at"]),
        )

    def get_binding(self, binding_id: str) -> RewardIntegrityBinding:
        row = self.db._conn.execute(
            "SELECT * FROM reward_integrity_bindings WHERE id=?", (binding_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown reward integrity binding: {binding_id}")
        return self._binding(row)

    def list_bindings(
        self,
        *,
        domain_kind: Optional[str] = None,
        domain_id: Optional[str] = None,
        reward_system_revision_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Page[RewardIntegrityBinding]:
        limit, offset = _page(limit, offset)
        clauses, params = [], []
        for column, value in (
            ("domain_kind", domain_kind),
            ("domain_id", domain_id),
            ("reward_system_revision_id", reward_system_revision_id),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                params.append(value)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        total = int(
            self.db._conn.execute(
                f"SELECT COUNT(*) FROM reward_integrity_bindings {where}", params
            ).fetchone()[0]
        )
        rows = self.db._conn.execute(
            f"SELECT * FROM reward_integrity_bindings {where} ORDER BY created_at DESC,id LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        return Page([self._binding(row) for row in rows], total, limit, offset)


__all__ = ["RewardIntegrityStore"]
