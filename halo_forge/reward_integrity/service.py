"""Capability-independent service layer for reward-integrity workflows."""

from __future__ import annotations

import json
from pathlib import Path
import hashlib
import math
import sqlite3
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from halo_forge.run_db.db import RunDatabase

from .metrics import (
    IntegrityEvidence,
    compute_integrity_metrics,
    grouped_percentile_bootstrap,
    normalize_reward,
)
from .models import (
    Page,
    ResolvedRewardBinding,
    RewardIntegrityAudit,
    RewardIntegrityComparison,
    RewardIntegrityComparisonPair,
    RewardIntegrityDecision,
    RewardIntegrityMetric,
    RewardIntegritySample,
    RewardSystem,
    RewardSystemRevision,
    TrainingSignalShard,
    TrainingSignalSnapshot,
)
from .storage import RewardIntegrityStorage, canonical_json, content_hash
from .store import RewardIntegrityStore

PROTOCOL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "balanced_256": {
        "capture_mode": "balanced_256",
        "seed": 42,
        "uniform_core_limit": 192,
        "diagnostic_limit": 64,
        "diagnostic_strata": [
            "verifier_error",
            "threshold_adjacent",
            "highest_reward",
            "component_disagreement",
        ],
        "full_snapshot_required_for_gating": True,
    },
    "broad_512": {
        "capture_mode": "broad_512",
        "seed": 42,
        "uniform_core_limit": 384,
        "diagnostic_limit": 128,
        "diagnostic_strata": [
            "verifier_error",
            "threshold_adjacent",
            "highest_reward",
            "component_disagreement",
        ],
        "full_snapshot_required_for_gating": True,
    },
    "exhaustive": {
        "capture_mode": "exhaustive",
        "seed": 42,
        "uniform_core_limit": None,
        "diagnostic_limit": 0,
        "full_snapshot_required_for_gating": True,
    },
}


_STRICT_METRICS = {
    "paired_coverage": {"direction": "maximize", "pass": 0.99, "warn": 0.97},
    "sentinel_error_rate": {"direction": "minimize", "pass": 0.01, "warn": 0.03},
    "pass_agreement": {"direction": "maximize", "pass": 0.95, "warn": 0.90},
    "optimizer_only_acceptance": {
        "direction": "minimize",
        "pass": 0.02,
        "warn": 0.05,
    },
    "spearman": {
        "direction": "maximize",
        "pass": 0.90,
        "warn": 0.80,
        "required_when_available": True,
    },
    "absolute_mean_reward_gap": {
        "direction": "minimize",
        "pass": 0.05,
        "warn": 0.10,
    },
    "top_tail_disagreement": {
        "direction": "minimize",
        "pass": 0.05,
        "warn": 0.10,
    },
}

_HUMAN_METRICS = {
    "paired_coverage": {"direction": "maximize", "pass": 0.97, "warn": 0.95},
    "sentinel_error_rate": {"direction": "minimize", "pass": 0.03, "warn": 0.05},
    "pass_agreement": {"direction": "maximize", "pass": 0.85, "warn": 0.75},
    "optimizer_only_acceptance": {
        "direction": "minimize",
        "pass": 0.10,
        "warn": 0.20,
    },
    "spearman": {
        "direction": "maximize",
        "pass": 0.70,
        "warn": 0.50,
        "required_when_available": True,
    },
    "absolute_mean_reward_gap": {
        "direction": "minimize",
        "pass": 0.15,
        "warn": 0.25,
    },
    "top_tail_disagreement": {
        "direction": "minimize",
        "pass": 0.15,
        "warn": 0.25,
    },
}

PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "strict_integrity": {
        "minimum_records": {"pass": 100, "warn": 20},
        "metrics": _STRICT_METRICS,
        "report_only": False,
    },
    "human_aligned_integrity": {
        "minimum_records": {"pass": 100, "warn": 20},
        "metrics": _HUMAN_METRICS,
        "report_only": False,
    },
    "exploratory": {
        "minimum_records": {"pass": None, "warn": 20},
        "metrics": {},
        "report_only": True,
    },
}


def _value(value: Any) -> Dict[str, Any]:
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    return dict(value)


def _content_hash_sequence(values: Any) -> str:
    """Hash an ordered JSON sequence without creating the sequence in memory."""

    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for value in values:
        if not first:
            digest.update(b",")
        digest.update(canonical_json(value).encode("utf-8"))
        first = False
    digest.update(b"]")
    return digest.hexdigest()


class RewardIntegrityService:
    def __init__(
        self,
        database: RunDatabase,
        *,
        root: str | Path | None = None,
        bootstrap_defaults: bool = True,
        scheduler: Any = None,
        gate_hook: Optional[Callable[[RewardIntegrityAudit, RewardIntegrityDecision], None]] = None,
    ) -> None:
        self.db = database
        self.store = RewardIntegrityStore(database)
        self.storage = RewardIntegrityStorage(root)
        if scheduler is None:
            from halo_forge.workstation_jobs import WorkstationScheduler

            scheduler = WorkstationScheduler(database)
        self.scheduler = scheduler
        self.gate_hook = gate_hook
        self.default_ids: Dict[str, str] = {}
        if bootstrap_defaults:
            self.default_ids = self.ensure_builtin_defaults()

    @staticmethod
    def _all_pages(list_method: Callable[..., Page[Any]], *args: Any, **kwargs: Any) -> list[Any]:
        values: list[Any] = []
        offset = 0
        while True:
            page = list_method(*args, limit=1000, offset=offset, **kwargs)
            values.extend(page.items)
            offset += len(page.items)
            if offset >= page.total or not page.items:
                return values

    @staticmethod
    def _iter_pages(
        list_method: Callable[..., Page[Any]],
        *args: Any,
        page_size: int = 1000,
        **kwargs: Any,
    ) -> Any:
        offset = 0
        while True:
            page = list_method(*args, limit=page_size, offset=offset, **kwargs)
            if not page.items:
                return
            yield from page.items
            offset += len(page.items)
            if offset >= page.total:
                return

    # -- guided defaults ------------------------------------------------

    def ensure_builtin_defaults(self) -> Dict[str, str]:
        resolved: Dict[str, str] = {}
        for name, definition in PROTOCOL_DEFAULTS.items():
            protocol_id = f"builtin-reward-audit-protocol-{name.replace('_', '-')}"
            try:
                protocol = self.store.get_protocol(protocol_id)
            except KeyError:
                protocol = self.store.create_protocol(
                    name=name,
                    description=f"Halo Forge built-in {name} reward capture protocol.",
                    protocol_id=protocol_id,
                )
            revision = self.store.create_protocol_revision(
                protocol.id,
                definition,
                capture_mode=name,
                revision_id=f"{protocol_id}-revision-1",
            )
            resolved[f"protocol:{name}"] = revision.id
        for name, requirements in PROFILE_DEFAULTS.items():
            profile_id = f"builtin-reward-integrity-profile-{name.replace('_', '-')}"
            try:
                profile = self.store.get_integrity_profile(profile_id)
            except KeyError:
                profile = self.store.create_integrity_profile(
                    name=name,
                    description=f"Halo Forge built-in {name} decision policy.",
                    profile_id=profile_id,
                )
            revision = self.store.create_integrity_profile_revision(
                profile.id,
                template_kind=name,
                requirements=requirements,
                promotable=name != "exploratory",
                revision_id=f"{profile_id}-revision-1",
            )
            resolved[f"profile:{name}"] = revision.id
        self.default_ids = resolved
        return dict(resolved)

    def capabilities(self) -> Dict[str, Any]:
        from halo_forge.training_signal import TRAINING_SIGNAL_CAPABILITIES

        return {
            "schema_version": 11,
            "items": [value.to_dict() for value in TRAINING_SIGNAL_CAPABILITIES.list()],
            "capture_fidelities": [
                "exact",
                "sampled",
                "aggregate_only",
                "unavailable",
                "not_recorded",
            ],
            "protocols": [dict(value) for value in PROTOCOL_DEFAULTS.values()],
            "integrity_profiles": [
                {"template_kind": name, **dict(value)} for name, value in PROFILE_DEFAULTS.items()
            ],
            "same_output_required": True,
            "diagnostics_contribute_to_population_metrics": False,
            "maximum_diagnostic_auditors": 3,
        }

    # -- thin catalog conveniences -------------------------------------

    def create_system(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        definition: Optional[Mapping[str, Any]] = None,
        system_id: Optional[str] = None,
    ) -> Any:
        if definition is not None:
            validation = self.validate_system_definition(definition)
            blockers = [
                str(value)
                for value in validation.get("blockers", [])
                if not str(value).startswith("primary_sentinel_correlated:")
            ]
            if blockers:
                raise ValueError("invalid reward system: " + "; ".join(blockers))
        system = self.store.create_system(name=name, description=description, system_id=system_id)
        if definition is None:
            return system
        raw = dict(definition)
        revision = self.create_system_revision(
            system.id,
            optimizer_verifier_revision_id=str(raw.pop("optimizer_verifier_revision_id")),
            modality=str(raw.pop("modality", "text")),
            task_type=str(raw.pop("task_type", "binary")),
            auditors=list(raw.pop("auditors", [])),
            input_mapping=dict(raw.pop("input_mapping", {})),
            reward_mapping=dict(raw.pop("reward_mapping", {})),
            definition=raw,
        )
        return {"system": system.to_dict(), "revision": revision.to_dict()}

    def get_system(self, identifier: str) -> Any:
        return self.store.get_system(identifier)

    def list_systems(
        self,
        *,
        query: Optional[str] = None,
        modality: Optional[str] = None,
        task_type: Optional[str] = None,
        trainer: Optional[str] = None,
        backend: str = "hf",
        qualified_only: bool = False,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        if not query and not modality and not task_type and not trainer and not qualified_only:
            return self.store.list_systems(
                include_archived=include_archived, limit=limit, offset=offset
            )
        clauses = [] if include_archived else ["s.archived=0"]
        params: list[Any] = []
        if query:
            clauses.append("(s.name LIKE ? OR s.description LIKE ? OR s.id LIKE ?)")
            pattern = f"%{query}%"
            params.extend((pattern, pattern, pattern))
        if modality:
            clauses.append("r.modality=?")
            params.append(modality)
        if task_type:
            clauses.append("r.task_type=?")
            params.append(task_type)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        page_limit, page_offset = max(1, min(int(limit), 1000)), max(0, int(offset))
        join = "LEFT JOIN reward_system_revisions r ON r.id=s.latest_revision_id"
        candidate_total = int(
            self.db._conn.execute(
                f"SELECT COUNT(*) FROM reward_systems s {join} {where}", params
            ).fetchone()[0]
        )
        if trainer or qualified_only:
            capability_id: Optional[str] = None
            if trainer:
                from halo_forge.training_signal import TRAINING_SIGNAL_CAPABILITIES

                capability_id = TRAINING_SIGNAL_CAPABILITIES.resolve(
                    str(trainer), str(backend or "hf")
                ).id
            selected: list[RewardSystem] = []
            matched = 0
            scan_offset = 0
            while scan_offset < candidate_total:
                rows = self.db._conn.execute(
                    f"SELECT s.* FROM reward_systems s {join} {where} "
                    "ORDER BY s.name,s.id LIMIT 500 OFFSET ?",
                    [*params, scan_offset],
                ).fetchall()
                if not rows:
                    break
                scan_offset += len(rows)
                for row in rows:
                    revision_id = row["latest_revision_id"]
                    if not revision_id:
                        continue
                    revision = self.store.get_system_revision(str(revision_id))
                    if capability_id:
                        declared = revision.definition.get(
                            "compatible_training_signal_capabilities"
                        )
                        compatible_ids = (
                            {str(value) for value in declared}
                            if isinstance(declared, (list, tuple))
                            else set(
                                self._default_signal_capability_ids(
                                    revision.modality, revision.task_type
                                )
                            )
                        )
                        if capability_id not in compatible_ids:
                            continue
                    if qualified_only:
                        verifier_blockers = self._verifier_blockers(
                            revision.optimizer_verifier_revision_id,
                            modality=revision.modality,
                            task_type=revision.task_type,
                            runtime_identity=None,
                        )
                        primary = revision.primary_sentinel
                        if primary is None or primary.correlated:
                            continue
                        verifier_blockers.extend(
                            self._verifier_blockers(
                                primary.verifier_revision_id,
                                modality=revision.modality,
                                task_type=revision.task_type,
                                runtime_identity=None,
                            )
                        )
                        if verifier_blockers:
                            continue
                    if page_offset <= matched < page_offset + page_limit:
                        selected.append(
                            RewardSystem(
                                id=str(row["id"]),
                                name=str(row["name"]),
                                description=row["description"],
                                latest_revision_id=row["latest_revision_id"],
                                archived=bool(row["archived"]),
                                created_at=str(row["created_at"]),
                                updated_at=str(row["updated_at"]),
                            )
                        )
                    matched += 1
            return Page(selected, matched, page_limit, page_offset)
        total = candidate_total
        rows = self.db._conn.execute(
            f"SELECT s.* FROM reward_systems s {join} {where} "
            "ORDER BY s.name,s.id LIMIT ? OFFSET ?",
            [*params, page_limit, page_offset],
        ).fetchall()
        items = [
            RewardSystem(
                id=str(row["id"]),
                name=str(row["name"]),
                description=row["description"],
                latest_revision_id=row["latest_revision_id"],
                archived=bool(row["archived"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
            )
            for row in rows
        ]
        return Page(items, total, page_limit, page_offset)

    def get_system_detail(self, identifier: str) -> Optional[Dict[str, Any]]:
        try:
            revision = self.store.get_system_revision(identifier)
            system = self.store.get_system(revision.system_id)
        except KeyError:
            try:
                system = self.store.get_system(identifier)
            except KeyError:
                return None
            revision = (
                self.store.get_system_revision(system.latest_revision_id)
                if system.latest_revision_id
                else None
            )
        return {
            "system": system.to_dict(),
            "revision": revision.to_dict() if revision else None,
            "revisions": self.store.list_system_revisions(
                system_id=system.id, limit=1000
            ).to_dict(),
            "usage_count": (
                self.store.list_bindings(reward_system_revision_id=revision.id, limit=1).total
                if revision
                else 0
            ),
        }

    def validate_system_definition(self, definition: Mapping[str, Any]) -> Dict[str, Any]:
        raw = dict(definition)
        blockers: list[str] = []
        optimizer_id = str(raw.get("optimizer_verifier_revision_id") or "").strip()
        if not optimizer_id:
            blockers.append("optimizer_verifier_revision_required")
        elif self._verifier_row(optimizer_id) is None:
            blockers.append("optimizer_verifier_revision_unknown")
        auditors = [dict(value) for value in raw.get("auditors") or []]
        primary = [value for value in auditors if value.get("role") == "primary_sentinel"]
        if len(primary) != 1:
            blockers.append("exactly_one_primary_sentinel_required")
        if len([value for value in auditors if value.get("role") == "diagnostic"]) > 3:
            blockers.append("too_many_diagnostic_auditors")
        for item in auditors:
            verifier_id = str(item.get("verifier_revision_id") or "")
            if self._verifier_row(verifier_id) is None:
                blockers.append(f"auditor_verifier_revision_unknown:{verifier_id}")
        if optimizer_id and primary:
            correlation = self._correlation(
                optimizer_id, str(primary[0].get("verifier_revision_id") or "")
            )
            if correlation:
                blockers.append("primary_sentinel_correlated:" + ",".join(correlation))
        declared_capabilities = raw.get("compatible_training_signal_capabilities")
        if declared_capabilities is not None:
            if not isinstance(declared_capabilities, (list, tuple)) or not declared_capabilities:
                blockers.append("compatible_training_signal_capabilities_required")
            else:
                from halo_forge.training_signal import TRAINING_SIGNAL_CAPABILITIES

                for capability_id in declared_capabilities:
                    try:
                        TRAINING_SIGNAL_CAPABILITIES.get(str(capability_id))
                    except KeyError:
                        blockers.append(
                            f"training_signal_capability_unknown:{capability_id}"
                        )
        return {"valid": not blockers, "blockers": sorted(set(blockers))}

    @staticmethod
    def _default_signal_capability_ids(modality: str, task_type: str) -> list[str]:
        """Derive a conservative immutable trainer set for guided creation.

        Operators may narrow this set explicitly.  The fallback keeps legacy
        V8/API clients useful while preventing an audio or vision reward
        contract from silently appearing in unrelated text trainers.
        """

        normalized_modality = str(modality).strip().lower().replace("-", "_")
        normalized_task = str(task_type).strip().lower().replace("-", "_")
        if normalized_modality in {"audio", "speech"}:
            return ["audio:hf"]
        if normalized_modality in {"vlm", "vision", "image", "image_text"}:
            return ["vlm:hf"]
        if normalized_modality in {"multimodal", "any"}:
            return [
                "raft:hf",
                "raft:mlx",
                "grpo:hf",
                "grpo:mlx",
                "reasoning:hf",
                "agentic:hf",
                "vlm:hf",
                "audio:hf",
            ]
        text_capabilities = [
            "raft:hf",
            "raft:mlx",
            "grpo:hf",
            "grpo:mlx",
            "reasoning:hf",
        ]
        if normalized_task in {
            "agentic",
            "tool",
            "tool_use",
            "tool_trace",
            "structured_tool_trace",
        }:
            return ["agentic:hf"]
        return text_capabilities

    def create_system_revision(self, system_id: str, **values: Any) -> RewardSystemRevision:
        validation = self.validate_system_definition(
            {
                "optimizer_verifier_revision_id": values.get(
                    "optimizer_verifier_revision_id"
                ),
                "auditors": list(values.get("auditors") or []),
            }
        )
        blockers = [
            str(value)
            for value in validation.get("blockers", [])
            if not str(value).startswith("primary_sentinel_correlated:")
        ]
        if blockers:
            raise ValueError("invalid reward system: " + "; ".join(blockers))
        auditors = []
        optimizer = str(values["optimizer_verifier_revision_id"])
        optimizer_row = self._verifier_row(optimizer)
        if optimizer_row is None:
            raise ValueError(f"unknown optimizer verifier revision: {optimizer}")
        definition = dict(values.get("definition") or {})
        reward_mapping = dict(values.get("reward_mapping") or {})
        # Older guided clients placed these executable settings beside the
        # mapping. Canonicalize them into the one contract the trainer uses so
        # the stored form, replay identity, and runtime cannot disagree.
        for key in (
            "failure_behavior",
            "failure_reward",
            "filtering",
            "scaling",
            "centering",
            "keep_policy",
            "reward_shaping",
        ):
            if key in definition and key not in reward_mapping:
                reward_mapping[key] = definition.pop(key)
        raw_minimum = float(optimizer_row["reward_min"])
        raw_maximum = float(optimizer_row["reward_max"])
        raw_direction = str(optimizer_row["reward_direction"])
        normalization = reward_mapping.get("normalization")
        if normalization is None or (
            isinstance(normalization, str)
            and normalization in {"linear", "linear_0_1"}
        ):
            reward_mapping["normalization"] = {
                "minimum": raw_minimum,
                "maximum": raw_maximum,
                "direction": raw_direction,
            }
        elif not isinstance(normalization, Mapping):
            raise ValueError("reward normalization must be a pinned mapping")
        reward_mapping.setdefault("minimum", 0.0)
        reward_mapping.setdefault("maximum", 1.0)
        failure_behavior = str(
            reward_mapping.get("failure_behavior") or "reject"
        ).strip().lower()
        failure_behavior = {
            "fail_closed": "reject",
            "error": "raise",
        }.get(failure_behavior, failure_behavior)
        if failure_behavior not in {"reject", "raise", "abstain"}:
            raise ValueError("failure_behavior must be reject, raise, or abstain")
        reward_mapping["failure_behavior"] = failure_behavior
        filtering = reward_mapping.get("filtering")
        if isinstance(filtering, str):
            reward_mapping["filtering"] = {"mode": filtering}
        elif filtering is not None and not isinstance(filtering, Mapping):
            raise ValueError("reward filtering must be a mapping or named mode")
        scaling = reward_mapping.get("scaling", 1.0)
        if isinstance(scaling, str):
            if scaling not in {"linear", "identity", "none"}:
                raise ValueError("unknown reward scaling mode")
            reward_mapping["scaling"] = 1.0
        centering = reward_mapping.get("centering", 0.0)
        if isinstance(centering, str):
            if centering not in {"none", "zero"}:
                raise ValueError("unknown reward centering mode")
            reward_mapping["centering"] = 0.0
        output_minimum = float(reward_mapping["minimum"])
        output_maximum = float(reward_mapping["maximum"])
        normalized_contract = dict(reward_mapping["normalization"])
        normalized_minimum = float(
            normalized_contract.get("minimum", normalized_contract.get("min", raw_minimum))
        )
        normalized_maximum = float(
            normalized_contract.get("maximum", normalized_contract.get("max", raw_maximum))
        )
        normalized_direction = str(
            normalized_contract.get("direction") or raw_direction
        )
        scale_value = reward_mapping.get("scale", reward_mapping.get("scaling", 1.0))
        if isinstance(scale_value, Mapping):
            scale_value = scale_value.get("factor", scale_value.get("scale", 1.0))
        center_value = reward_mapping.get(
            "center", reward_mapping.get("centering", 0.0)
        )
        if isinstance(center_value, Mapping):
            center_value = center_value.get("value", center_value.get("center", 0.0))
        numeric_values = [
            output_minimum,
            output_maximum,
            normalized_minimum,
            normalized_maximum,
            float(scale_value),
            float(center_value),
            float(reward_mapping.get("failure_reward", output_minimum)),
        ]
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("reward mapping requires finite numeric values")
        if output_maximum <= output_minimum or normalized_maximum <= normalized_minimum:
            raise ValueError("reward mapping ranges must have maximum greater than minimum")
        if normalized_direction not in {"maximize", "minimize"}:
            raise ValueError("reward normalization direction must be maximize or minimize")
        if reward_mapping.get("threshold") is None and optimizer_row["threshold"] is not None:
            raw_threshold = float(optimizer_row["threshold"])
            normalized_threshold = (raw_threshold - raw_minimum) / (
                raw_maximum - raw_minimum
            )
            if raw_direction == "minimize":
                normalized_threshold = 1.0 - normalized_threshold
            reward_mapping["threshold"] = normalized_threshold
        if reward_mapping.get("threshold") is not None:
            threshold = float(reward_mapping["threshold"])
            if not math.isfinite(threshold) or not output_minimum <= threshold <= output_maximum:
                raise ValueError("reward threshold is outside the mapped reward range")
        values["reward_mapping"] = reward_mapping
        declared_capabilities = definition.get(
            "compatible_training_signal_capabilities"
        )
        if declared_capabilities is None:
            declared_capabilities = self._default_signal_capability_ids(
                str(values.get("modality") or "text"),
                str(values.get("task_type") or "binary"),
            )
        definition["compatible_training_signal_capabilities"] = sorted(
            {str(value) for value in declared_capabilities}
        )
        capability_validation = self.validate_system_definition(
            {
                "optimizer_verifier_revision_id": values.get(
                    "optimizer_verifier_revision_id"
                ),
                "auditors": list(values.get("auditors") or []),
                "compatible_training_signal_capabilities": definition[
                    "compatible_training_signal_capabilities"
                ],
            }
        )
        capability_blockers = [
            str(value)
            for value in capability_validation.get("blockers", [])
            if str(value).startswith("training_signal_capability_unknown:")
            or str(value) == "compatible_training_signal_capabilities_required"
        ]
        if capability_blockers:
            raise ValueError(
                "invalid reward system: " + "; ".join(capability_blockers)
            )
        values["definition"] = definition
        for raw_value in values.get("auditors") or []:
            raw = dict(raw_value)
            reasons = self._correlation(optimizer, str(raw.get("verifier_revision_id") or ""))
            raw["correlated"] = bool(reasons)
            raw["correlation_reasons"] = reasons
            auditors.append(raw)
        values["auditors"] = auditors
        return self.store.create_system_revision(system_id, **values)

    def get_system_revision(self, identifier: str) -> Any:
        return self.store.get_system_revision(identifier)

    def list_system_revisions(self, **values: Any) -> Any:
        return self.store.list_system_revisions(**values)

    def create_protocol(
        self, *, name: str, definition: Mapping[str, Any], description: Optional[str] = None
    ) -> Dict[str, Any]:
        protocol = self.store.create_protocol(name=name, description=description)
        revision = self.store.create_protocol_revision(protocol.id, definition)
        return {"protocol": protocol.to_dict(), "revision": revision.to_dict()}

    def get_protocol(self, identifier: str) -> Any:
        return self.store.get_protocol(identifier)

    def list_protocols(self, **values: Any) -> Any:
        return self.store.list_protocols(**values)

    def create_protocol_revision(
        self, protocol_id: str, definition: Mapping[str, Any], **values: Any
    ) -> Any:
        return self.store.create_protocol_revision(protocol_id, definition, **values)

    def get_protocol_revision(self, identifier: str) -> Any:
        return self.store.get_protocol_revision(identifier)

    def list_protocol_revisions(self, **values: Any) -> Any:
        return self.store.list_protocol_revisions(**values)

    def get_protocol_detail(self, identifier: str) -> Optional[Dict[str, Any]]:
        try:
            revision = self.store.get_protocol_revision(identifier)
            protocol = self.store.get_protocol(revision.protocol_id)
        except KeyError:
            try:
                protocol = self.store.get_protocol(identifier)
            except KeyError:
                return None
            revision = (
                self.store.get_protocol_revision(protocol.latest_revision_id)
                if protocol.latest_revision_id
                else None
            )
        return {
            "protocol": protocol.to_dict(),
            "revision": revision.to_dict() if revision else None,
            "revisions": self.store.list_protocol_revisions(
                protocol_id=protocol.id, limit=1000
            ).to_dict(),
        }

    def create_integrity_profile(
        self,
        *,
        name: str,
        template_kind: str,
        requirements: Mapping[str, Any],
        description: Optional[str] = None,
        promotable: Optional[bool] = None,
    ) -> Dict[str, Any]:
        profile = self.store.create_integrity_profile(name=name, description=description)
        revision = self.store.create_integrity_profile_revision(
            profile.id,
            template_kind=template_kind,
            requirements=requirements,
            promotable=promotable,
        )
        return {"profile": profile.to_dict(), "revision": revision.to_dict()}

    def get_integrity_profile(self, identifier: str) -> Any:
        return self.store.get_integrity_profile(identifier)

    def list_integrity_profiles(self, **values: Any) -> Any:
        return self.store.list_integrity_profiles(**values)

    def create_integrity_profile_revision(self, profile_id: str, **values: Any) -> Any:
        return self.store.create_integrity_profile_revision(profile_id, **values)

    def get_integrity_profile_revision(self, identifier: str) -> Any:
        return self.store.get_integrity_profile_revision(identifier)

    def list_integrity_profile_revisions(self, **values: Any) -> Any:
        return self.store.list_integrity_profile_revisions(**values)

    def get_integrity_profile_detail(self, identifier: str) -> Optional[Dict[str, Any]]:
        try:
            revision = self.store.get_integrity_profile_revision(identifier)
            profile = self.store.get_integrity_profile(revision.profile_id)
        except KeyError:
            try:
                profile = self.store.get_integrity_profile(identifier)
            except KeyError:
                return None
            revision = (
                self.store.get_integrity_profile_revision(profile.latest_revision_id)
                if profile.latest_revision_id
                else None
            )
        return {
            "profile": profile.to_dict(),
            "revision": revision.to_dict() if revision else None,
            "revisions": self.store.list_integrity_profile_revisions(
                profile_id=profile.id, limit=1000
            ).to_dict(),
        }

    # -- verifier identity and binding ---------------------------------

    def _verifier_row(self, revision_id: str) -> Any:
        return self.db._conn.execute(
            "SELECT * FROM verifier_profile_revisions WHERE id=?", (revision_id,)
        ).fetchone()

    def _leaf_fingerprints(self, revision_id: str) -> set[str]:
        row = self._verifier_row(revision_id)
        if row is None:
            return set()
        children = self.db._conn.execute(
            "SELECT child_revision_id FROM verifier_revision_components WHERE revision_id=?",
            (revision_id,),
        ).fetchall()
        if children:
            result: set[str] = set()
            for child in children:
                result.update(self._leaf_fingerprints(str(child["child_revision_id"])))
            return result
        fingerprint = row["implementation_fingerprint"]
        return {str(fingerprint)} if fingerprint else set()

    def _leaf_artifact_fingerprints(self, revision_id: str) -> set[str]:
        row = self._verifier_row(revision_id)
        if row is None:
            return set()
        children = self.db._conn.execute(
            "SELECT child_revision_id FROM verifier_revision_components WHERE revision_id=?",
            (revision_id,),
        ).fetchall()
        if children:
            result: set[str] = set()
            for child in children:
                result.update(
                    self._leaf_artifact_fingerprints(str(child["child_revision_id"]))
                )
            return result
        try:
            definition = json.loads(row["definition_json"] or "{}")
        except json.JSONDecodeError:
            return set()
        values: set[str] = set()

        def collect(value: Any, key: str = "") -> None:
            if isinstance(value, Mapping):
                for child_key, child_value in value.items():
                    collect(child_value, str(child_key).lower())
            elif isinstance(value, list):
                for child_value in value:
                    collect(child_value, key)
            elif value not in {None, ""} and (
                "artifact_hash" in key
                or key in {"verified_artifact", "model_content_hash", "weights_hash"}
            ):
                values.add(str(value))

        collect(definition)
        return values

    def _correlation(self, optimizer_id: str, auditor_id: str) -> list[str]:
        reasons: list[str] = []
        if optimizer_id == auditor_id:
            reasons.append("same_verifier_revision")
        optimizer = self._verifier_row(optimizer_id)
        auditor = self._verifier_row(auditor_id)
        if optimizer is None or auditor is None:
            return reasons
        if str(optimizer["profile_id"]) == str(auditor["profile_id"]):
            reasons.append("same_verifier_profile")
        left, right = self._leaf_fingerprints(optimizer_id), self._leaf_fingerprints(auditor_id)
        if not left or not right:
            reasons.append("unfingerprintable_leaf")
        elif left & right:
            reasons.append("overlapping_implementation_fingerprint")
        left_artifacts = self._leaf_artifact_fingerprints(optimizer_id)
        right_artifacts = self._leaf_artifact_fingerprints(auditor_id)
        if left_artifacts & right_artifacts:
            reasons.append("overlapping_artifact_fingerprint")
        return sorted(set(reasons))

    def _verifier_blockers(
        self,
        revision_id: str,
        *,
        modality: str,
        task_type: str,
        runtime_identity: Optional[Mapping[str, Any]],
    ) -> list[str]:
        row = self._verifier_row(revision_id)
        if row is None:
            return ["verifier_revision_missing"]
        blockers: list[str] = []
        if not bool(row["qualifiable"]):
            blockers.append("verifier_unqualifiable")
        if str(row["modality"]) not in {modality, "any", "multimodal"}:
            blockers.append("modality_incompatible")
        if str(row["task_type"]) != task_type:
            blockers.append("task_type_incompatible")
        aliases = self.db._conn.execute(
            "SELECT alias FROM verifier_aliases WHERE profile_id=? AND revision_id=?",
            (row["profile_id"], revision_id),
        ).fetchall()
        usable_alias = False
        overridden_alias = False
        for alias_row in aliases:
            event = self.db._conn.execute(
                "SELECT override FROM verifier_alias_events "
                "WHERE profile_id=? AND alias=? AND revision_id=? "
                "ORDER BY created_at DESC,id DESC LIMIT 1",
                (row["profile_id"], alias_row["alias"], revision_id),
            ).fetchone()
            # An alias without an event is a readable v7/early-migration
            # identity. New promotions always have an event, whose override
            # bit keeps them out of normal guided gating.
            if event is not None and bool(event["override"]):
                overridden_alias = True
                continue
            usable_alias = True
        if not usable_alias:
            blockers.append("verifier_not_candidate_or_approved")
            if overridden_alias:
                blockers.append("verifier_qualification_override_excluded")

        # Compare against the current implementation/runtime contract instead
        # of comparing two unrelated hashes (the contract hash is not an
        # observed runtime identity). This also catches implementation or
        # artifact drift before a managed launch acquires the workstation.
        try:
            from halo_forge.verifier_lab import VerifierLabService

            compatibility = VerifierLabService(
                self.db,
                root=self.storage.root / "evaluations" / "verifier-calibrations",
                scheduler=self.scheduler,
            ).runtime_compatibility(revision_id, actual=runtime_identity)
            if str(compatibility.get("state") or "stale_runtime") != "compatible":
                blockers.append("stale_runtime")
        except Exception:
            blockers.append("stale_runtime")
        return blockers

    def resolve_binding(
        self,
        reward_system_revision_id: str,
        protocol_revision_id: Optional[str] = None,
        integrity_profile_revision_id: Optional[str] = None,
        *,
        trainer: Optional[str] = None,
        backend: Optional[str] = None,
        boundaries: Sequence[int] = (),
        runtime_identity: Optional[Mapping[str, Any]] = None,
    ) -> ResolvedRewardBinding:
        system = self.store.get_system_revision(reward_system_revision_id)
        protocol = self.store.get_protocol_revision(
            protocol_revision_id
            or self.default_ids.get("protocol:balanced_256")
            or self.ensure_builtin_defaults()["protocol:balanced_256"]
        )
        profile = self.store.get_integrity_profile_revision(
            integrity_profile_revision_id
            or self.default_ids.get("profile:human_aligned_integrity")
            or self.ensure_builtin_defaults()["profile:human_aligned_integrity"]
        )
        blockers = [
            f"optimizer:{item}"
            for item in self._verifier_blockers(
                system.optimizer_verifier_revision_id,
                modality=system.modality,
                task_type=system.task_type,
                runtime_identity=runtime_identity,
            )
        ]
        primary = system.primary_sentinel
        if primary is None:
            blockers.append("primary_sentinel_missing")
        else:
            blockers.extend(
                f"primary_sentinel:{item}"
                for item in self._verifier_blockers(
                    primary.verifier_revision_id,
                    modality=system.modality,
                    task_type=system.task_type,
                    runtime_identity=runtime_identity,
                )
            )
            if primary.correlated:
                blockers.append("primary_sentinel_correlated")
        if protocol.definition.get("full_snapshot_required_for_gating") is not True:
            blockers.append("protocol_not_gating_eligible")
        if profile.template_kind == "exploratory" or not profile.promotable:
            blockers.append("integrity_profile_report_only")
        capability = None
        if trainer or backend:
            if not trainer or not backend:
                blockers.append("trainer_and_backend_required_together")
            else:
                from halo_forge.training_signal import (
                    TRAINING_SIGNAL_CAPABILITIES,
                    CaptureFidelity,
                )

                try:
                    capability = TRAINING_SIGNAL_CAPABILITIES.resolve(
                        str(trainer), str(backend)
                    )
                except KeyError:
                    blockers.append("training_signal_capability_unavailable")
                else:
                    declared = system.definition.get(
                        "compatible_training_signal_capabilities"
                    )
                    compatible_ids = (
                        {str(value) for value in declared}
                        if isinstance(declared, (list, tuple))
                        else set(
                            self._default_signal_capability_ids(
                                system.modality, system.task_type
                            )
                        )
                    )
                    if capability.id not in compatible_ids:
                        blockers.append("training_signal_capability_incompatible")
                    if capability.fidelity not in {
                        CaptureFidelity.EXACT,
                        CaptureFidelity.SAMPLED,
                    }:
                        blockers.append("training_signal_capture_not_gating_eligible")
        for value in boundaries:
            if isinstance(value, str):
                _unit, boundary_value = self._parse_boundary(value)
            else:
                try:
                    boundary_value = int(value)
                except (TypeError, ValueError):
                    blockers.append("invalid_boundary")
                    continue
            if boundary_value < 0:
                blockers.append("invalid_boundary")
            if (
                capability is not None
                and not capability.resumable
                and str(value).strip().lower() not in {"final", "last"}
            ):
                blockers.append("trainer_supports_final_boundary_only")
        return ResolvedRewardBinding(
            reward_system_revision=system,
            protocol_revision=protocol,
            integrity_profile_revision=profile,
            gating_eligible=not blockers,
            blockers=sorted(set(blockers)),
        )

    # -- trace publication ---------------------------------------------

    def create_signal_shard(
        self,
        *,
        run_id: str,
        segment_id: str,
        reward_system_revision_id: str,
        protocol_revision_id: str,
        capability_id: str,
        capture_fidelity: str,
        boundary_unit: str,
        boundary_value: int,
        snapshots: Sequence[TrainingSignalSnapshot | Mapping[str, Any]],
        aggregate: Mapping[str, Any],
        dataset_identity: Mapping[str, Any],
        producer_model_hash: str,
        checkpoint_hash: str,
        runtime_identity: Mapping[str, Any],
        direct_run_segment_id: Optional[str] = None,
        trial_segment_id: Optional[str] = None,
    ) -> TrainingSignalShard:
        if capture_fidelity not in {
            "exact",
            "sampled",
            "aggregate_only",
            "unavailable",
            "not_recorded",
        }:
            raise ValueError("unknown training-signal capture fidelity")
        values = [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in snapshots]
        snapshot_ids = [str(item.get("snapshot_id") or "") for item in values]
        if any(not item for item in snapshot_ids) or len(snapshot_ids) != len(set(snapshot_ids)):
            raise ValueError("training-signal snapshot IDs must be present and unique")
        trace_identity = {
            "run_id": run_id,
            "segment_id": segment_id,
            "reward_system_revision_id": reward_system_revision_id,
            "protocol_revision_id": protocol_revision_id,
            "capability_id": capability_id,
            "capture_fidelity": capture_fidelity,
            "boundary": {"unit": boundary_unit, "value": int(boundary_value)},
            "snapshot_ids": snapshot_ids,
            "aggregate": dict(aggregate),
            "dataset_identity": dict(dataset_identity),
            "producer_model_hash": producer_model_hash,
            "checkpoint_hash": checkpoint_hash,
            "runtime_identity": dict(runtime_identity),
        }
        trace_hash = content_hash(trace_identity)
        retained_set_hash = content_hash(snapshot_ids)
        destination = self.storage.signal_path(run_id, segment_id, trace_hash)
        bundle = self.storage.publish(
            destination,
            {
                "trace.json": trace_identity,
                "snapshots.jsonl": values,
                "aggregate.json": dict(aggregate),
            },
            identity={"kind": "training_signal", "trace_hash": trace_hash},
        )
        records = {
            str(
                (item.get("record") or {}).get("group_id")
                or (item.get("record") or {}).get("record_id")
            )
            for item in values
        }
        return self.store.create_signal_shard(
            run_id=run_id,
            direct_run_segment_id=direct_run_segment_id,
            trial_segment_id=trial_segment_id,
            reward_system_revision_id=reward_system_revision_id,
            protocol_revision_id=protocol_revision_id,
            capability_id=capability_id,
            capture_fidelity=capture_fidelity,
            boundary_unit=boundary_unit,
            boundary_value=boundary_value,
            trace_hash=trace_hash,
            retained_set_hash=retained_set_hash,
            event_count=int(aggregate.get("event_count", len(values))),
            distinct_record_count=len(records - {"None", ""}),
            aggregate=aggregate,
            dataset_identity=dataset_identity,
            producer_model_hash=producer_model_hash,
            checkpoint_hash=checkpoint_hash,
            runtime_identity=runtime_identity,
            storage_path=bundle.path,
            manifest_hash=bundle.manifest_hash,
        )

    def get_signal_shard(self, identifier: str) -> Any:
        return self.store.get_signal_shard(identifier)

    def list_signal_shards(self, **values: Any) -> Any:
        return self.store.list_signal_shards(**values)

    def register_training_signal_shard(
        self,
        shard: Any,
        *,
        reward_system_revision_id: str,
        protocol_revision_id: str,
        dataset_identity: Optional[Mapping[str, Any]] = None,
        runtime_identity: Optional[Mapping[str, Any]] = None,
        producer_model_hash: Optional[str] = None,
        boundary_unit: Optional[str] = None,
        boundary_value: Optional[int] = None,
        direct_run_segment_id: Optional[str] = None,
        trial_segment_id: Optional[str] = None,
    ) -> TrainingSignalShard:
        """Register a shard sealed by :mod:`halo_forge.training_signal`.

        Registration retains the trainer-published path and checksums instead
        of copying or translating the exact captured output rows.
        """

        from halo_forge.training_signal import verify_training_signal_shard

        raw = shard.to_dict() if hasattr(shard, "to_dict") else dict(shard)
        path = Path(str(raw["path"])).expanduser().resolve()
        verified = verify_training_signal_shard(path)
        if not verified.get("valid"):
            raise ValueError(
                "invalid training signal shard: "
                + "; ".join(verified.get("problems") or ["verification failed"])
            )
        manifest_payload = (path / "manifest.json").read_bytes()
        manifest = json.loads(manifest_payload)
        parsed_unit, parsed_value = self._parse_boundary(str(raw.get("boundary") or "final"))
        samples_path = path / "samples.jsonl"
        inventory = sqlite3.connect("")
        inventory.executescript(
            """
            CREATE TABLE records (value TEXT PRIMARY KEY);
            CREATE TABLE models (value TEXT PRIMARY KEY);
            CREATE TABLE runtimes (value TEXT PRIMARY KEY);
            """
        )
        if samples_path.is_file():
            with samples_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    record_id = (item.get("record") or {}).get("record_id")
                    if record_id:
                        inventory.execute(
                            "INSERT OR IGNORE INTO records VALUES (?)", (str(record_id),)
                        )
                    model_hash = item.get("producer_model_hash")
                    if model_hash:
                        inventory.execute(
                            "INSERT OR IGNORE INTO models VALUES (?)", (str(model_hash),)
                        )
                    if item.get("runtime_identity"):
                        runtime_hash = content_hash(dict(item.get("runtime_identity") or {}))
                        inventory.execute(
                            "INSERT OR IGNORE INTO runtimes VALUES (?)", (runtime_hash,)
                        )
            inventory.commit()
        model_count = int(inventory.execute("SELECT COUNT(*) FROM models").fetchone()[0])
        discovered_model = (
            str(inventory.execute("SELECT value FROM models").fetchone()[0])
            if model_count == 1
            else None
        )
        resolved_producer = producer_model_hash or (
            discovered_model
            if discovered_model is not None
            else (
                _content_hash_sequence(
                    row[0]
                    for row in inventory.execute("SELECT value FROM models ORDER BY value")
                )
                if model_count
                else "unavailable"
            )
        )
        resolved_runtime = dict(runtime_identity or {})
        if not resolved_runtime:
            runtime_count = int(
                inventory.execute("SELECT COUNT(*) FROM runtimes").fetchone()[0]
            )
            if runtime_count:
                if runtime_count <= 128:
                    resolved_runtime = {
                        "sample_runtime_hashes": [
                            str(row[0])
                            for row in inventory.execute(
                                "SELECT value FROM runtimes ORDER BY value"
                            )
                        ]
                    }
                else:
                    resolved_runtime = {
                        "sample_runtime_count": runtime_count,
                        "sample_runtime_hash": _content_hash_sequence(
                            row[0]
                            for row in inventory.execute(
                                "SELECT value FROM runtimes ORDER BY value"
                            )
                        ),
                    }
        distinct_record_count = int(
            inventory.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        )
        inventory.close()
        return self.store.create_signal_shard(
            shard_id=str(raw.get("shard_id") or ""),
            run_id=str(raw["run_id"]),
            direct_run_segment_id=direct_run_segment_id,
            trial_segment_id=trial_segment_id,
            reward_system_revision_id=reward_system_revision_id,
            protocol_revision_id=protocol_revision_id,
            capability_id=str(raw["capability_id"]),
            capture_fidelity=str(raw["capture_fidelity"]),
            boundary_unit=boundary_unit or parsed_unit,
            boundary_value=(parsed_value if boundary_value is None else int(boundary_value)),
            trace_hash=str(raw["trace_hash"]),
            retained_set_hash=str(
                manifest.get("retained_ids_hash")
                or content_hash(manifest.get("retained_ids") or [])
            ),
            event_count=int(raw.get("observed_count", 0)),
            distinct_record_count=distinct_record_count,
            aggregate=dict(raw.get("aggregate") or {}),
            dataset_identity=dict(dataset_identity or {}),
            producer_model_hash=resolved_producer,
            checkpoint_hash=str(raw.get("checkpoint_hash") or "unavailable"),
            runtime_identity=resolved_runtime,
            storage_path=str(path),
            manifest_hash=hashlib.sha256(manifest_payload).hexdigest(),
        )

    @staticmethod
    def _parse_boundary(boundary: str) -> tuple[str, int]:
        lowered = boundary.strip().lower()
        if lowered == "final":
            return "final", 0
        for unit in ("step", "cycle", "epoch"):
            if lowered.startswith(unit):
                suffix = lowered[len(unit) :].lstrip(":=_- ")
                try:
                    return unit, int(suffix)
                except ValueError:
                    return unit, 0
        try:
            return "step", int(lowered)
        except ValueError:
            return "final", 0

    def verify_signal_shard(self, identifier: str) -> Dict[str, Any]:
        shard = self.store.get_signal_shard(identifier)
        manifest_path = Path(shard.storage_path) / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
            if manifest.get("format_version") in {1, 2} and "trace_hash" in manifest:
                from halo_forge.training_signal import verify_training_signal_shard

                result = verify_training_signal_shard(shard.storage_path)
                if not result.get("valid"):
                    raise ValueError(
                        "invalid training signal shard: " + "; ".join(result.get("problems") or [])
                    )
                actual_manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                if actual_manifest_hash != shard.manifest_hash:
                    raise ValueError("training-signal manifest identity drifted")
                return {**result, "manifest_hash": actual_manifest_hash}
        verified = self.storage.verify(shard.storage_path)
        if verified.manifest_hash != shard.manifest_hash:
            raise ValueError("training-signal manifest identity drifted")
        if verified.content_hash != verified.manifest.get("content_hash"):
            raise ValueError("training-signal content identity drifted")
        return {
            "valid": True,
            "manifest_hash": verified.manifest_hash,
            "content_hash": verified.content_hash,
            "trace_hash": (verified.manifest.get("identity") or {}).get(
                "trace_hash"
            ),
            "size_bytes": verified.size_bytes,
        }

    def audit_signal_evidence_problems(self, audit_id: str) -> list[str]:
        """Return deterministic sealed-trace invalidities for one audit.

        This check deliberately does not catch unexpected filesystem or
        database failures. Those may be transient workstation failures and
        must continue through the scheduler's retry path. A successfully read
        but missing, corrupt, unsealed, or identity-drifted trace is different:
        it is scientific evidence that cannot support a score and therefore
        becomes an ``incomplete_evidence`` gate decision.
        """

        audit = self.store.get_audit(audit_id)
        shard = self.store.get_signal_shard(audit.signal_shard_id)
        problems: list[str] = []
        if not shard.sealed:
            problems.append("training_signal_not_sealed")
        for field in (
            "run_id",
            "reward_system_revision_id",
            "protocol_revision_id",
        ):
            if getattr(audit, field) != getattr(shard, field):
                problems.append(f"training_signal_identity_mismatch:{field}")
        try:
            verification = self.verify_signal_shard(shard.id)
        except ValueError as exc:
            problems.append(f"training_signal_integrity_invalid:{exc}")
        else:
            verified_trace_hash = verification.get("trace_hash")
            if (
                verified_trace_hash not in {None, ""}
                and str(verified_trace_hash) != shard.trace_hash
            ):
                problems.append("training_signal_identity_mismatch:trace_hash")
        return list(dict.fromkeys(problems))

    def auditor_runtime_identity_problems(
        self,
        reward_system_revision_id: str,
        runtime_identity: Mapping[str, Any],
    ) -> list[str]:
        """Return pinned auditor identities that are deterministically stale.

        ``runtime_compatibility`` reports contract or implementation drift as
        data. If resolving that report itself raises, the caller must retain
        the exception because provider, storage, or workstation failures are
        retryable execution errors rather than scientific evidence.
        """

        from halo_forge.verifier_lab import VerifierLabService

        system = self.store.get_system_revision(reward_system_revision_id)
        verifiers = VerifierLabService(self.db)
        problems: list[str] = []
        identities = [
            ("optimizer", system.optimizer_verifier_revision_id),
            *(
                (auditor.role, auditor.verifier_revision_id)
                for auditor in sorted(system.auditors, key=lambda item: item.ordinal)
            ),
        ]
        seen: set[str] = set()
        for role, revision_id in identities:
            if revision_id in seen:
                continue
            seen.add(revision_id)
            compatibility = verifiers.runtime_compatibility(
                revision_id,
                actual=runtime_identity or None,
            )
            if bool(compatibility.get("compatible")):
                continue
            fields = sorted(
                {
                    str(value.get("field") or "unspecified")
                    for value in compatibility.get("mismatches") or []
                    if isinstance(value, Mapping)
                }
            )
            suffix = ",".join(fields) if fields else str(
                compatibility.get("state") or "stale_runtime"
            )
            problems.append(
                "stale_verifier_runtime:"
                f"{role}:{revision_id}:{suffix}"
            )
        return problems

    # -- audit execution ------------------------------------------------

    def audit_resource_requirements(
        self, reward_system_revision_id: str
    ) -> Dict[str, Any]:
        """Classify every sentinel/diagnostic execution locality once."""

        system = self.store.get_system_revision(reward_system_revision_id)
        primary = system.primary_sentinel
        auditor_resources: list[dict[str, Any]] = []
        for auditor in sorted(system.auditors, key=lambda value: value.ordinal):
            row = self._verifier_row(auditor.verifier_revision_id)
            family = str(row["family"]) if row is not None else "unknown"
            definition = (
                json.loads(row["definition_json"] or "{}") if row is not None else {}
            )
            endpoint_type = str(
                definition.get("endpoint_type")
                or (definition.get("provider") or {}).get("endpoint_type")
                or ""
            ).lower()
            local = family == "reward_model" or endpoint_type in {
                "ollama",
                "local",
                "openai_compatible_local",
            }
            auditor_resources.append(
                {
                    "role": auditor.role,
                    "ordinal": auditor.ordinal,
                    "verifier_revision_id": auditor.verifier_revision_id,
                    "family": family,
                    "endpoint_type": endpoint_type or None,
                    "local_model": local,
                }
            )
        local_model = any(value["local_model"] for value in auditor_resources)
        resource_class = "accelerator" if local_model else "cpu"
        primary_resources = next(
            (
                value
                for value in auditor_resources
                if value["role"] == "primary_sentinel"
            ),
            {},
        )
        return {
            "resource_class": resource_class,
            "exclusive_heavy_operation": local_model,
            "reward_system_revision_id": system.id,
            "primary_sentinel_revision_id": (
                primary.verifier_revision_id if primary else None
            ),
            "family": primary_resources.get("family", "unknown"),
            "endpoint_type": primary_resources.get("endpoint_type"),
            "auditors": auditor_resources,
            "local_auditor_revision_ids": [
                value["verifier_revision_id"]
                for value in auditor_resources
                if value["local_model"]
            ],
        }

    def _assert_current_auditor_runtime(
        self,
        system: RewardSystemRevision,
        runtime_identity: Mapping[str, Any],
        *,
        context: str,
    ) -> None:
        """Refuse scientific reuse after any pinned auditor has drifted.

        The reuse key pins the runtime identity recorded by the request, but
        cannot detect a later implementation, artifact, or implicit local
        runtime change. Re-run Verifier Lab's compatibility check against the
        current process before returning immutable evidence.
        """

        from halo_forge.verifier_lab import VerifierLabService

        verifiers = VerifierLabService(self.db)
        stale: list[str] = []
        for auditor in sorted(system.auditors, key=lambda item: item.ordinal):
            compatibility = verifiers.runtime_compatibility(
                auditor.verifier_revision_id,
                actual=runtime_identity or None,
            )
            if not bool(compatibility.get("compatible")):
                stale.append(auditor.verifier_revision_id)
        if stale:
            raise ValueError(
                f"{context} refuses stale sentinel runtime or implementation: "
                + ", ".join(stale)
            )

    def validate_development_suite_revision(
        self, revision_id: str
    ) -> Dict[str, str]:
        """Resolve the optional checkpoint-quality suite before training starts.

        Reward-integrity audits may track an independent development evaluation,
        but protected benchmark purposes must never influence a training boundary.
        This check is intentionally reusable by CLI and HTTP preflight so a typo
        or protected suite cannot consume a training segment before failing.
        """

        identifier = str(revision_id or "").strip()
        if not identifier:
            raise ValueError("development suite revision is required")
        suite = self.db._conn.execute(
            """SELECT r.id,COALESCE(s.purpose_v4,s.purpose,'unspecified') purpose
                 FROM benchmark_suite_revisions r
                 JOIN benchmark_suites s ON s.id=r.suite_id
                WHERE r.id=?""",
            (identifier,),
        ).fetchone()
        if suite is None:
            raise ValueError("development suite revision is missing")
        purpose = str(suite["purpose"] or "unspecified").strip().lower()
        if purpose not in {"development", "unspecified"}:
            raise ValueError(
                f"{purpose} suite evidence cannot guide a reward-integrity training gate"
            )
        return {"id": str(suite["id"]), "purpose": purpose}

    def create_audit(
        self,
        *,
        signal_shard_id: str,
        integrity_profile_revision_id: Optional[str] = None,
        runtime_identity: Optional[Mapping[str, Any]] = None,
        development_suite_revision_id: Optional[str] = None,
        request: Optional[Mapping[str, Any]] = None,
        work_item_id: Optional[str] = None,
        audit_id: Optional[str] = None,
        submit: bool = True,
    ) -> RewardIntegrityAudit:
        shard = self.store.get_signal_shard(signal_shard_id)
        if development_suite_revision_id:
            self.validate_development_suite_revision(development_suite_revision_id)
        profile_id = (
            integrity_profile_revision_id
            or self.default_ids.get("profile:human_aligned_integrity")
            or self.ensure_builtin_defaults()["profile:human_aligned_integrity"]
        )
        self.store.get_integrity_profile_revision(profile_id)
        from halo_forge.verifier_lab.store import scrub_secrets

        runtime = scrub_secrets(dict(runtime_identity or {}))
        audit_request = scrub_secrets(dict(request or {}))
        reuse = content_hash(
            {
                "trace_hash": shard.trace_hash,
                "reward_system_revision_id": shard.reward_system_revision_id,
                "protocol_revision_id": shard.protocol_revision_id,
                "integrity_profile_revision_id": profile_id,
                "runtime_identity": runtime,
                "development_suite_revision_id": development_suite_revision_id,
                "request": audit_request,
            }
        )
        completed_row = self.db._conn.execute(
            "SELECT id FROM reward_integrity_audits "
            "WHERE reuse_key=? AND status='completed' LIMIT 1",
            (reuse,),
        ).fetchone()
        if completed_row is not None:
            # Reuse is scientific identity reuse, not merely a matching row.
            # A damaged bundle remains visible for reconciliation and must not
            # silently cause a fresh audit with the same immutable reuse key.
            completed = self.store.get_audit(str(completed_row["id"]))
            self.verify_audit_bundle(completed.id)
            system = self.store.get_system_revision(
                completed.reward_system_revision_id
            )
            self._assert_current_auditor_runtime(
                system,
                runtime,
                context="completed reward-integrity audit reuse",
            )
            return completed
        existing_row = self.db._conn.execute(
            "SELECT id FROM reward_integrity_audits "
            "WHERE reuse_key=? AND status<>'cancelled' "
            "ORDER BY created_at,id LIMIT 1",
            (reuse,),
        ).fetchone()
        if existing_row is not None:
            # Training-process retries can revisit a sealed boundary before its
            # dependent evaluation/audit chain finishes. Reuse the durable
            # domain row instead of creating a second evaluation chain for the
            # same immutable trace and scientific request.
            return self.store.get_audit(str(existing_row["id"]))
        audit = self.store.create_audit(
            audit_id=audit_id,
            run_id=shard.run_id,
            direct_run_segment_id=shard.direct_run_segment_id,
            trial_segment_id=shard.trial_segment_id,
            signal_shard_id=shard.id,
            reward_system_revision_id=shard.reward_system_revision_id,
            protocol_revision_id=shard.protocol_revision_id,
            integrity_profile_revision_id=profile_id,
            development_suite_revision_id=development_suite_revision_id,
            status="queued",
            stage="queued",
            request=audit_request,
            runtime_identity=runtime,
            runtime_identity_hash=content_hash(runtime),
            reuse_key=reuse,
            work_item_id=work_item_id,
        )
        if self.store.list_samples(audit.id, limit=1).total == 0:
            self._hydrate_trace(audit, shard)
        audit = self.store.get_audit(audit.id)
        if submit and not audit.work_item_id:
            resources = self.audit_resource_requirements(
                audit.reward_system_revision_id
            )
            work = self.scheduler.enqueue(
                kind="reward_integrity_audit",
                launch_spec={
                    "handler": "reward_integrity.execute_audit",
                    "operation": "reward_integrity_audit",
                    "audit_id": audit.id,
                    "reward_integrity_root": str(self.storage.root),
                },
                resource_class=str(resources["resource_class"]),
                resource_requirements=resources,
                domain_kind="reward_integrity_audit",
                domain_id=audit.id,
                canonical_run_id=audit.run_id,
                max_retries=2,
            )
            audit = self.store.update_audit(audit.id, work_item_id=work.id)
        return audit

    def _hydrate_trace(self, audit: RewardIntegrityAudit, shard: TrainingSignalShard) -> None:
        canonical_path = Path(shard.storage_path) / "snapshots.jsonl"
        trainer_path = Path(shard.storage_path) / "samples.jsonl"
        path = canonical_path if canonical_path.is_file() else trainer_path
        system = self.store.get_system_revision(audit.reward_system_revision_id)
        optimizer_contract = self._optimizer_contract(system)
        count = 0
        sample_page: list[Dict[str, Any]] = []
        observation_page: list[Dict[str, Any]] = []
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                record = dict(raw.get("record") or {})
                selection = dict(raw.get("selection") or {})
                if raw.get("selection_stratum") is not None:
                    stratum = str(raw.get("selection_stratum"))
                    selection.setdefault(
                        "class",
                        "uniform_core" if stratum in {"exact", "uniform_fill"} else stratum,
                    )
                    selection.setdefault(
                        "diagnostic",
                        stratum not in {"exact", "uniform_core", "uniform_fill"},
                    )
                group_id = str(
                    record.get("group_id")
                    or record.get("record_id")
                    or raw["snapshot_id"]
                )
                sample_page.append(
                    {
                        "audit_id": audit.id,
                        "ordinal": count,
                        "snapshot_id": raw["snapshot_id"],
                        "record_id": record.get("record_id") or raw["snapshot_id"],
                        "record_hash": record.get("record_hash")
                        or content_hash(raw.get("input")),
                        "instance_id": record.get("instance_id") or raw["snapshot_id"],
                        "group_id": group_id,
                        "candidate_ordinal": raw.get("candidate_ordinal", 0),
                        "selection_class": selection.get("class", "uniform_core"),
                        "diagnostic": bool(selection.get("diagnostic", False)),
                        "input": raw.get("input")
                        or {"prompt": raw.get("prompt"), "context": raw.get("context")},
                        "output": raw.get("output"),
                        "expected": raw.get("expected"),
                        "media": raw.get("media") or [],
                        "generation": raw.get("generation")
                        or raw.get("generation_settings")
                        or {},
                        "lineage": record.get("lineage") or record.get("source") or {},
                    }
                )
                observation = dict(
                    raw.get("optimizer_observation")
                    or raw.get("training_observation")
                    or {}
                )
                reward = observation.get("reward")
                normalized = None
                error = observation.get("error")
                if reward is not None and not error:
                    try:
                        normalized = normalize_reward(reward, optimizer_contract)
                    except ValueError as exc:
                        error = str(exc)
                observation_page.append(
                    {
                        "audit_id": audit.id,
                        "sample_ordinal": count,
                        "role": "optimizer",
                        "auditor_ordinal": 0,
                        "verifier_revision_id": system.optimizer_verifier_revision_id,
                        **observation,
                        "normalized_reward": normalized,
                        "error": error,
                    }
                )
                count += 1
                if len(sample_page) >= 500:
                    self.store.add_samples(sample_page)
                    self.store.add_observations(observation_page)
                    sample_page.clear()
                    observation_page.clear()
        self.store.add_samples(sample_page)
        self.store.add_observations(observation_page)
        distinct = int(
            self.db._conn.execute(
                "SELECT COUNT(DISTINCT group_id) FROM reward_integrity_samples WHERE audit_id=?",
                (audit.id,),
            ).fetchone()[0]
        )
        self.store.update_audit(
            audit.id, total_samples=count, distinct_record_count=distinct
        )

    def _reward_contract(self, verifier_revision_id: str) -> Dict[str, Any]:
        row = self._verifier_row(verifier_revision_id)
        if row is None:
            raise KeyError(f"unknown verifier revision: {verifier_revision_id}")
        return {
            "minimum": float(row["reward_min"]),
            "maximum": float(row["reward_max"]),
            "direction": str(row["reward_direction"]),
            "threshold": row["threshold"],
        }

    def _optimizer_contract(self, system: RewardSystemRevision) -> Dict[str, Any]:
        """Return the post-mapping optimizer contract seen by training."""

        mapping = dict(system.reward_mapping or {})
        filtering = dict(mapping.get("filtering") or {})
        return {
            "minimum": float(mapping.get("minimum", 0.0)),
            "maximum": float(mapping.get("maximum", 1.0)),
            "direction": "maximize",
            "threshold": mapping.get("threshold", filtering.get("threshold")),
        }

    def add_audit_evidence(
        self, audit_id: str, evidence: Sequence[Mapping[str, Any]]
    ) -> RewardIntegrityAudit:
        audit = self.store.get_audit(audit_id)
        system = self.store.get_system_revision(audit.reward_system_revision_id)
        primary = system.primary_sentinel
        if primary is None:
            raise ValueError("reward system has no primary sentinel")
        contracts = {
            item.verifier_revision_id: self._reward_contract(item.verifier_revision_id)
            for item in system.auditors
        }
        for item in evidence:
            snapshot_id = str(item.get("snapshot_id") or "")
            sample_row = self.db._conn.execute(
                "SELECT * FROM reward_integrity_samples WHERE audit_id=? AND snapshot_id=?",
                (audit_id, snapshot_id),
            ).fetchone()
            if sample_row is None:
                raise KeyError(f"unknown audit snapshot: {snapshot_id}")
            sample = self.store._sample(sample_row)
            values = [
                (
                    "primary_sentinel",
                    primary.ordinal,
                    primary.verifier_revision_id,
                    item.get("primary_sentinel") or {},
                )
            ]
            for ordinal, diagnostic in enumerate(item.get("diagnostics") or [], start=1):
                auditor = next(
                    (
                        value
                        for value in system.auditors
                        if value.role == "diagnostic"
                        and value.ordinal == int(diagnostic.get("auditor_ordinal", ordinal))
                    ),
                    None,
                )
                if auditor is None:
                    raise ValueError("diagnostic observation does not match a configured auditor")
                values.append(
                    ("diagnostic", auditor.ordinal, auditor.verifier_revision_id, diagnostic)
                )
            for role, auditor_ordinal, verifier_id, raw_value in values:
                raw = _value(raw_value)
                reward, error = raw.get("reward"), raw.get("error")
                normalized = None
                if reward is not None and not error:
                    try:
                        normalized = normalize_reward(reward, contracts[verifier_id])
                    except ValueError as exc:
                        error = str(exc)
                self.store.add_observation(
                    {
                        "audit_id": audit_id,
                        "sample_ordinal": sample.ordinal,
                        "role": role,
                        "auditor_ordinal": auditor_ordinal,
                        "verifier_revision_id": verifier_id,
                        **raw,
                        "normalized_reward": normalized,
                        "error": error,
                    }
                )
        processed = self.db._conn.execute(
            "SELECT COUNT(DISTINCT sample_ordinal) FROM reward_integrity_observations WHERE audit_id=? AND role='primary_sentinel'",
            (audit_id,),
        ).fetchone()[0]
        return self.store.update_audit(
            audit_id, processed_samples=int(processed), stage="scoring", status="running"
        )

    def _integrity_evidence(self, audit_id: str) -> list[IntegrityEvidence]:
        rows = self.db._conn.execute(
            """SELECT s.snapshot_id,s.group_id,s.diagnostic,s.lineage_json,
                      o.reward AS optimizer_reward,o.passed AS optimizer_passed,o.error AS optimizer_error,
                      o.component_trace_json AS optimizer_component_trace,
                      p.reward AS sentinel_reward,p.passed AS sentinel_passed,p.error AS sentinel_error
               FROM reward_integrity_samples s
               LEFT JOIN reward_integrity_observations o ON o.audit_id=s.audit_id AND o.sample_ordinal=s.ordinal AND o.role='optimizer'
               LEFT JOIN reward_integrity_observations p ON p.audit_id=s.audit_id AND p.sample_ordinal=s.ordinal AND p.role='primary_sentinel'
               WHERE s.audit_id=? ORDER BY s.ordinal""",
            (audit_id,),
        ).fetchall()

        def subgroup(value: Any) -> str:
            try:
                lineage = json.loads(value or "{}")
            except (TypeError, json.JSONDecodeError):
                return ""
            metadata = lineage.get("metadata") if isinstance(lineage, Mapping) else None
            source = metadata if isinstance(metadata, Mapping) else lineage
            if not isinstance(source, Mapping):
                return ""
            for key in ("subgroup", "category", "task", "topic", "domain"):
                candidate = source.get(key)
                if candidate not in {None, ""}:
                    return f"{key}:{candidate}"
            return ""

        def component_disagreement(value: Any) -> Optional[bool]:
            try:
                trace = json.loads(value or "[]")
            except (TypeError, json.JSONDecodeError):
                return None
            if not isinstance(trace, list) or len(trace) < 2:
                return None
            passes = [
                item.get("passed")
                for item in trace
                if isinstance(item, Mapping) and item.get("passed") is not None
            ]
            if len(passes) >= 2:
                return len({bool(item) for item in passes}) > 1
            rewards = [
                item.get("reward")
                for item in trace
                if isinstance(item, Mapping) and item.get("reward") is not None
            ]
            if len(rewards) >= 2:
                return len({float(item) for item in rewards}) > 1
            return None

        return [
            IntegrityEvidence(
                snapshot_id=str(row["snapshot_id"]),
                group_id=str(row["group_id"]),
                optimizer_reward=row["optimizer_reward"],
                sentinel_reward=row["sentinel_reward"],
                optimizer_passed=(
                    None if row["optimizer_passed"] is None else bool(row["optimizer_passed"])
                ),
                sentinel_passed=(
                    None if row["sentinel_passed"] is None else bool(row["sentinel_passed"])
                ),
                optimizer_error=row["optimizer_error"],
                sentinel_error=row["sentinel_error"],
                diagnostic=bool(row["diagnostic"]),
                subgroup=subgroup(row["lineage_json"]),
                component_disagreement=component_disagreement(
                    row["optimizer_component_trace"]
                ),
            )
            for row in rows
        ]

    def analyze_audit(
        self,
        audit_id: str,
        optimizer_contract: Optional[Mapping[str, Any]] = None,
        sentinel_contract: Optional[Mapping[str, Any]] = None,
        *,
        bootstrap_resamples: int = 10_000,
    ) -> list[RewardIntegrityMetric]:
        existing = self.store.list_metrics(audit_id, limit=1000)
        if existing.total:
            return existing.items
        audit = self.store.get_audit(audit_id)
        system = self.store.get_system_revision(audit.reward_system_revision_id)
        primary = system.primary_sentinel
        if primary is None:
            raise ValueError("reward system has no primary sentinel")
        evidence = self._integrity_evidence(audit_id)
        optimizer_scale = (
            optimizer_contract or self._optimizer_contract(system)
        )
        sentinel_scale = sentinel_contract or self._reward_contract(
            primary.verifier_revision_id
        )
        metrics = compute_integrity_metrics(
            audit_id,
            evidence,
            optimizer_scale,
            sentinel_scale,
            bootstrap_resamples=bootstrap_resamples,
        )
        # Subgroup estimates are only emitted when the stable-record replicate
        # count meets the V8 minimum; small slices remain visibly unavailable
        # rather than presenting noisy pseudo-precision.
        subgroup_names = sorted({item.subgroup for item in evidence if item.subgroup})
        for name in subgroup_names:
            values = [item for item in evidence if item.subgroup == name]
            if len({item.group_id for item in values}) < 20:
                continue
            metrics.extend(
                RewardIntegrityMetric(
                    **{
                        **metric.to_dict(),
                        "subgroup": name,
                    }
                )
                for metric in compute_integrity_metrics(
                    audit_id,
                    values,
                    optimizer_scale,
                    sentinel_scale,
                    bootstrap_resamples=bootstrap_resamples,
                )
            )
        metrics.extend(
            self._boundary_trend_metrics(
                audit,
                bootstrap_resamples=bootstrap_resamples,
            )
        )
        self.store.add_metrics(metrics)
        self.store.update_audit(audit_id, stage="analyzed")
        return self.store.list_metrics(audit_id, limit=1000).items

    def _boundary_trend_metrics(
        self,
        audit: RewardIntegrityAudit,
        *,
        bootstrap_resamples: int,
    ) -> list[RewardIntegrityMetric]:
        """Compare shared stable records with the immediately prior boundary.

        On-policy additions/removals are reported as distributional coverage;
        they are never paired into a causal-looking trend estimate.
        """

        current_shard = self.store.get_signal_shard(audit.signal_shard_id)
        previous = self.db._conn.execute(
            """SELECT a.id
                 FROM reward_integrity_audits a
                 JOIN training_signal_shards s ON s.id=a.signal_shard_id
                WHERE a.run_id=? AND a.reward_system_revision_id=?
                  AND a.status='completed' AND s.boundary_value < ?
                ORDER BY s.boundary_value DESC,a.completed_at DESC LIMIT 1""",
            (
                audit.run_id,
                audit.reward_system_revision_id,
                current_shard.boundary_value,
            ),
        ).fetchone()
        if previous is None:
            return []
        previous_id = str(previous["id"])
        rows = self.db._conn.execute(
            """WITH current_values AS (
                     SELECT s.record_id,
                            AVG(CASE WHEN o.role='optimizer' THEN o.normalized_reward END) optimizer,
                            AVG(CASE WHEN o.role='primary_sentinel' THEN o.normalized_reward END) sentinel
                       FROM reward_integrity_samples s
                       JOIN reward_integrity_observations o
                         ON o.audit_id=s.audit_id AND o.sample_ordinal=s.ordinal
                      WHERE s.audit_id=? AND s.diagnostic=0 AND o.error IS NULL
                      GROUP BY s.record_id
                 ), previous_values AS (
                     SELECT s.record_id,
                            AVG(CASE WHEN o.role='optimizer' THEN o.normalized_reward END) optimizer,
                            AVG(CASE WHEN o.role='primary_sentinel' THEN o.normalized_reward END) sentinel
                       FROM reward_integrity_samples s
                       JOIN reward_integrity_observations o
                         ON o.audit_id=s.audit_id AND o.sample_ordinal=s.ordinal
                      WHERE s.audit_id=? AND s.diagnostic=0 AND o.error IS NULL
                      GROUP BY s.record_id
                 )
                 SELECT c.record_id,c.optimizer-p.optimizer optimizer_delta,
                        c.sentinel-p.sentinel sentinel_delta
                   FROM current_values c JOIN previous_values p USING(record_id)
                  WHERE c.optimizer IS NOT NULL AND p.optimizer IS NOT NULL
                    AND c.sentinel IS NOT NULL AND p.sentinel IS NOT NULL
                  ORDER BY c.record_id""",
            (audit.id, previous_id),
        ).fetchall()
        current_count = int(
            self.db._conn.execute(
                "SELECT COUNT(DISTINCT record_id) FROM reward_integrity_samples "
                "WHERE audit_id=? AND diagnostic=0",
                (audit.id,),
            ).fetchone()[0]
        )
        previous_count = int(
            self.db._conn.execute(
                "SELECT COUNT(DISTINCT record_id) FROM reward_integrity_samples "
                "WHERE audit_id=? AND diagnostic=0",
                (previous_id,),
            ).fetchone()[0]
        )
        shared = len(rows)
        if not rows:
            return [
                RewardIntegrityMetric(
                    audit_id=audit.id,
                    name="boundary_shared_record_coverage",
                    value=0.0,
                    available=True,
                    record_count=0,
                    direction="maximize",
                    metadata={
                        "previous_audit_id": previous_id,
                        "pairing": "distributional",
                        "current_record_count": current_count,
                        "previous_record_count": previous_count,
                    },
                )
            ]
        evidence = [
            IntegrityEvidence(
                snapshot_id=str(row["record_id"]),
                group_id=str(row["record_id"]),
                optimizer_reward=float(row["optimizer_delta"]),
                sentinel_reward=float(row["sentinel_delta"]),
                optimizer_passed=None,
                sentinel_passed=None,
            )
            for row in rows
        ]

        def optimizer_mean(values: Sequence[IntegrityEvidence]) -> Optional[float]:
            return sum(float(item.optimizer_reward) for item in values) / len(values) if values else None

        def sentinel_mean(values: Sequence[IntegrityEvidence]) -> Optional[float]:
            return sum(float(item.sentinel_reward) for item in values) / len(values) if values else None

        optimizer_value = optimizer_mean(evidence)
        sentinel_value = sentinel_mean(evidence)
        optimizer_low, optimizer_high = grouped_percentile_bootstrap(
            evidence,
            optimizer_mean,
            resamples=bootstrap_resamples,
            seed=42,
        )
        sentinel_low, sentinel_high = grouped_percentile_bootstrap(
            evidence,
            sentinel_mean,
            resamples=bootstrap_resamples,
            seed=42,
        )
        inversion = bool(
            optimizer_value is not None
            and sentinel_value is not None
            and optimizer_low is not None
            and sentinel_high is not None
            and optimizer_value > 0
            and sentinel_value < 0
            and optimizer_low > 0
            and sentinel_high < 0
        )
        metadata = {
            "previous_audit_id": previous_id,
            "pairing": "shared_stable_record",
            "shared_record_count": shared,
            "current_only_record_count": max(0, current_count - shared),
            "previous_only_record_count": max(0, previous_count - shared),
        }
        return [
            RewardIntegrityMetric(
                audit_id=audit.id,
                name="boundary_shared_record_coverage",
                value=shared / max(current_count, previous_count, 1),
                available=True,
                record_count=shared,
                direction="maximize",
                metadata=metadata,
            ),
            RewardIntegrityMetric(
                audit_id=audit.id,
                name="boundary_optimizer_reward_delta",
                value=optimizer_value,
                available=True,
                record_count=shared,
                ci_low=optimizer_low,
                ci_high=optimizer_high,
                direction=None,
                metadata=metadata,
            ),
            RewardIntegrityMetric(
                audit_id=audit.id,
                name="boundary_sentinel_reward_delta",
                value=sentinel_value,
                available=True,
                record_count=shared,
                ci_low=sentinel_low,
                ci_high=sentinel_high,
                direction=None,
                metadata=metadata,
            ),
            RewardIntegrityMetric(
                audit_id=audit.id,
                name="optimizer_up_sentinel_down",
                value=1.0 if inversion else 0.0,
                available=True,
                record_count=shared,
                direction="minimize",
                metadata=metadata,
            ),
        ]

    def decide_audit(self, audit_id: str) -> RewardIntegrityDecision:
        audit = self.store.get_audit(audit_id)
        prior = self.store.list_decisions(audit_id, limit=1000)
        if prior.total:
            return prior.items[-1]
        profile = self.store.get_integrity_profile_revision(audit.integrity_profile_revision_id)
        metrics = {item.name: item for item in self.store.list_metrics(audit_id, limit=1000).items}
        requirements = profile.requirements
        record_count = audit.distinct_record_count
        minimum = dict(requirements.get("minimum_records") or {})
        reasons: list[str] = []
        decision = "pass"
        if record_count < int(minimum.get("warn") or 20):
            decision = "incomplete_evidence"
            reasons.append(f"fewer_than_{int(minimum.get('warn') or 20)}_distinct_records")
        elif profile.template_kind == "exploratory" or requirements.get("report_only"):
            decision = "warn"
            reasons.append("exploratory_profile_is_report_only")
        else:
            for name, rule_value in dict(requirements.get("metrics") or {}).items():
                rule = dict(rule_value)
                metric = metrics.get(name)
                if metric is None or not metric.available:
                    if rule.get("required_when_available"):
                        continue
                    decision = "incomplete_evidence"
                    reasons.append(f"required_metric_unavailable:{name}")
                    continue
                direction = str(rule.get("direction", metric.direction or "maximize"))
                pass_ok = (
                    metric.value >= float(rule["pass"])
                    if direction == "maximize"
                    else metric.value <= float(rule["pass"])
                )
                warn_ok = (
                    metric.value >= float(rule["warn"])
                    if direction == "maximize"
                    else metric.value <= float(rule["warn"])
                )
                if not warn_ok:
                    if decision != "incomplete_evidence":
                        decision = "fail"
                    reasons.append(f"{name}_outside_warn_threshold")
                elif not pass_ok and decision == "pass":
                    decision = "warn"
                    reasons.append(f"{name}_outside_pass_threshold")
            inversion = metrics.get("optimizer_up_sentinel_down")
            if (
                decision != "incomplete_evidence"
                and inversion is not None
                and inversion.available
                and inversion.value == 1.0
            ):
                decision = "fail"
                reasons.append("matched_optimizer_up_sentinel_down_trend")
            pass_minimum = minimum.get("pass")
            if pass_minimum is not None and record_count < int(pass_minimum) and decision == "pass":
                decision = "warn"
                reasons.append(f"fewer_than_{int(pass_minimum)}_distinct_records_caps_at_warn")
        action = (
            "report_only"
            if requirements.get("report_only")
            else ("pause" if decision in {"fail", "incomplete_evidence"} else "continue")
        )
        result = self.store.add_decision(
            audit_id=audit_id,
            integrity_profile_revision_id=profile.id,
            decision=decision,
            action=action,
            reasons=reasons or ["all_integrity_requirements_satisfied"],
            evidence={"record_count": record_count, "metric_names": sorted(metrics)},
        )
        self._apply_gate_state(audit, result)
        return result

    def complete_incomplete_evidence(
        self,
        audit_id: str,
        *,
        reasons: Sequence[str],
        classification: str = "deterministic_evidence_invalid",
    ) -> RewardIntegrityAudit:
        """Publish an evidence-invalid audit without inventing measurements.

        A corrupt trace or stale pinned auditor is a terminal scientific input
        problem, not a failed metric threshold and not a transient work error.
        It therefore produces an append-only pause decision and a valid audit
        bundle, leaving the canonical run at a resumable review boundary.
        """

        normalized_reasons = list(
            dict.fromkeys(str(value).strip() for value in reasons if str(value).strip())
        )
        if not normalized_reasons:
            raise ValueError("incomplete evidence requires at least one exact reason")
        audit = self.store.get_audit(audit_id)
        if audit.status == "completed":
            return audit
        audit = self.store.update_audit(
            audit_id,
            status="running",
            stage="incomplete_evidence",
            error="; ".join(normalized_reasons),
        )
        prior = self.store.list_decisions(audit_id, limit=1000).items
        previous = prior[-1] if prior else None
        if (
            previous is not None
            and previous.decision == "incomplete_evidence"
            and previous.action == "pause"
            and previous.reasons == normalized_reasons
        ):
            decision = previous
        else:
            decision = self.store.add_decision(
                audit_id=audit_id,
                integrity_profile_revision_id=audit.integrity_profile_revision_id,
                decision="incomplete_evidence",
                action="pause",
                reasons=normalized_reasons,
                evidence={
                    "classification": classification,
                    "signal_shard_id": audit.signal_shard_id,
                    "scientific_metrics_computed": False,
                    "invalidity_reasons": normalized_reasons,
                },
                supersedes_decision_id=previous.id if previous is not None else None,
            )
        self._apply_gate_state(audit, decision)
        return self.publish_audit_bundle(audit_id)

    def _apply_gate_state(
        self, audit: RewardIntegrityAudit, decision: RewardIntegrityDecision
    ) -> None:
        """Project an append-only decision onto resumable run/segment state."""

        if decision.action == "pause":
            run_status, segment_status, segment_decision = (
                "awaiting_review",
                "awaiting_review",
                ("incomplete_evidence" if decision.decision == "incomplete_evidence" else "pause"),
            )
        elif decision.action == "continue":
            final_direct_boundary = False
            if audit.direct_run_segment_id:
                final_direct_boundary = self.db._conn.execute(
                    """SELECT NOT EXISTS (
                           SELECT 1 FROM direct_run_segments later
                            WHERE later.run_id=current.run_id
                              AND later.ordinal>current.ordinal
                       )
                         FROM direct_run_segments current WHERE current.id=?""",
                    (audit.direct_run_segment_id,),
                ).fetchone()
                final_direct_boundary = bool(
                    final_direct_boundary and final_direct_boundary[0]
                )
            run_status = "completed" if final_direct_boundary else "running"
            segment_status = "completed" if final_direct_boundary else "reviewed"
            segment_decision = "complete" if final_direct_boundary else "continue"
        elif decision.action == "stop":
            run_status, segment_status, segment_decision = "stopped", "stopped", "stop"
        elif decision.action == "fork":
            run_status, segment_status, segment_decision = "stopped", "stopped", "fork"
        else:
            run_status = segment_status = segment_decision = None
        with self.db._lock:
            if run_status is not None:
                if decision.action == "continue" and run_status == "running":
                    self.db._conn.execute(
                        "UPDATE runs SET status=? WHERE run_id=? "
                        "AND status NOT IN ('completed','succeeded')",
                        (run_status, audit.run_id),
                    )
                else:
                    self.db._conn.execute(
                        "UPDATE runs SET status=? WHERE run_id=?",
                        (run_status, audit.run_id),
                    )
            if audit.direct_run_segment_id and segment_status is not None:
                self.db._conn.execute(
                    "UPDATE direct_run_segments SET status=?,decision=?,decision_reason=?,"
                    "updated_at=? WHERE id=?",
                    (
                        segment_status,
                        segment_decision,
                        "; ".join(decision.reasons),
                        decision.created_at,
                        audit.direct_run_segment_id,
                    ),
                )
            self.db._conn.commit()
        if self.gate_hook is not None:
            self.gate_hook(audit, decision)

    def publish_audit_bundle(self, audit_id: str) -> RewardIntegrityAudit:
        audit = self.store.get_audit(audit_id)
        system = self.store.get_system_revision(audit.reward_system_revision_id)
        metrics = self.store.list_metrics(audit_id, limit=1000).items
        decisions = self.store.list_decisions(audit_id, limit=1000).items
        processed_samples = int(
            self.db._conn.execute(
                "SELECT COUNT(DISTINCT sample_ordinal) "
                "FROM reward_integrity_observations "
                "WHERE audit_id=? AND role='primary_sentinel'",
                (audit_id,),
            ).fetchone()[0]
        )
        # Atomic publication must finish before the database lifecycle row can
        # become immutable. Serialize the state that this successful publish
        # commits rather than the preceding running/analyzed state. The
        # manifest hash remains external because it covers audit.json itself.
        audit_document = audit.to_dict()
        audit_document.update(
            {
                "status": "completed",
                "stage": "published",
                "processed_samples": processed_samples,
                "artifact_path": str(self.storage.audit_path(audit_id)),
            }
        )
        bundle = self.storage.publish_streaming(
            self.storage.audit_path(audit_id),
            {
                "audit.json": audit_document,
                "reward-system.json": system.to_dict(),
                "metrics.json": [item.to_dict() for item in metrics],
                "decisions.json": [item.to_dict() for item in decisions],
            },
            jsonl_documents={
                "samples.jsonl": (
                    item.to_dict()
                    for item in self._iter_pages(self.store.list_samples, audit_id)
                ),
                "observations.jsonl": (
                    item.to_dict()
                    for item in self._iter_pages(self.store.list_observations, audit_id)
                ),
            },
            identity={
                "kind": "reward_integrity_audit",
                "audit_id": audit_id,
                "reuse_key": audit.reuse_key,
            },
        )
        return self.store.update_audit(
            audit_id,
            status="completed",
            stage="published",
            artifact_path=bundle.path,
            manifest_hash=bundle.manifest_hash,
            processed_samples=processed_samples,
        )

    def execute_audit(
        self,
        audit_id: str,
        *,
        sentinel: Callable[[RewardIntegritySample], Mapping[str, Any]],
        diagnostic_auditors: Sequence[Callable[[RewardIntegritySample], Mapping[str, Any]]] = (),
        bootstrap_resamples: int = 10_000,
    ) -> RewardIntegrityAudit:
        evidence_problems = self.audit_signal_evidence_problems(audit_id)
        if evidence_problems:
            return self.complete_incomplete_evidence(
                audit_id,
                reasons=evidence_problems,
                classification="sealed_signal_invalid",
            )
        audit = self.store.update_audit(audit_id, status="running", stage="scoring")
        system = self.store.get_system_revision(audit.reward_system_revision_id)
        primary_auditor = system.primary_sentinel
        if primary_auditor is None:
            raise ValueError("reward system has no primary sentinel")
        configured_diagnostics = sorted(
            (item for item in system.auditors if item.role == "diagnostic"),
            key=lambda item: item.ordinal,
        )
        if len(diagnostic_auditors) > len(configured_diagnostics):
            raise ValueError("more diagnostic callables were supplied than configured auditors")
        offset = 0
        while True:
            page = self.store.list_samples(audit_id, limit=256, offset=offset)
            if not page.items:
                break
            ordinals = [item.ordinal for item in page.items]
            placeholders = ",".join("?" for _ in ordinals)
            existing = {
                (int(row[0]), str(row[1]), int(row[2]))
                for row in self.db._conn.execute(
                    "SELECT sample_ordinal,role,auditor_ordinal "
                    "FROM reward_integrity_observations "
                    f"WHERE audit_id=? AND sample_ordinal IN ({placeholders})",
                    [audit_id, *ordinals],
                ).fetchall()
            }
            evidence = []
            for sample in page.items:
                if self.store.get_audit(audit_id).cancel_requested:
                    return self.store.update_audit(
                        audit_id, status="cancelled", stage="cancelled"
                    )
                primary = (
                    sentinel(sample)
                    if (
                        sample.ordinal,
                        "primary_sentinel",
                        primary_auditor.ordinal,
                    )
                    not in existing
                    else {}
                )
                diagnostics = [
                    dict(
                        callable_value(sample),
                        auditor_ordinal=configured_diagnostics[index].ordinal,
                    )
                    for index, callable_value in enumerate(diagnostic_auditors)
                    if (
                        sample.ordinal,
                        "diagnostic",
                        configured_diagnostics[index].ordinal,
                    )
                    not in existing
                ]
                if primary or diagnostics:
                    evidence.append(
                        {
                            "snapshot_id": sample.snapshot_id,
                            "primary_sentinel": primary,
                            "diagnostics": diagnostics,
                        }
                    )
            if evidence:
                self.add_audit_evidence(audit_id, evidence)
            offset += len(page.items)
            if offset >= page.total:
                break
        self.analyze_audit(audit_id, bootstrap_resamples=bootstrap_resamples)
        self.decide_audit(audit_id)
        return self.publish_audit_bundle(audit_id)

    # -- operator/query conveniences -----------------------------------

    def get_audit(self, identifier: str) -> RewardIntegrityAudit:
        return self.store.get_audit(identifier)

    def get_audit_detail(self, identifier: str) -> Optional[Dict[str, Any]]:
        try:
            audit = self.store.get_audit(identifier)
        except KeyError:
            return None
        decisions = self.store.list_decisions(identifier, limit=1000)
        shard = self.store.get_signal_shard(audit.signal_shard_id)
        linked_bindings = [
            value
            for value in self.store.list_bindings(
                domain_kind="run", domain_id=audit.run_id, limit=1000
            ).items
            if value.audit_id == audit.id
        ]
        return {
            "audit": audit.to_dict(),
            "signal_shard": {
                "id": shard.id,
                "boundary_unit": shard.boundary_unit,
                "boundary_value": shard.boundary_value,
                "capture_fidelity": shard.capture_fidelity,
                "capability_id": shard.capability_id,
                "event_count": shard.event_count,
                "distinct_record_count": shard.distinct_record_count,
                "trace_hash": shard.trace_hash,
            },
            "metrics": self.store.list_metrics(identifier, limit=1000).to_dict(),
            "decisions": decisions.to_dict(),
            "latest_decision": (decisions.items[-1].to_dict() if decisions.items else None),
            "sample_count": self.store.list_samples(identifier, limit=1).total,
            "bindings": [value.to_dict() for value in linked_bindings],
            "development_evaluation": self.development_evaluation_evidence(
                audit.id, require_complete=False
            ),
        }

    def development_evaluation_evidence(
        self,
        audit_id: str,
        *,
        require_complete: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Resolve the independently scheduled checkpoint-quality evidence.

        Development metrics are deliberately not interpreted here. V8 does not
        pin a quality threshold into reward-integrity profiles, so this link is
        completion/evidence tracking and never alters the integrity decision.
        """

        audit = self.store.get_audit(audit_id)
        binding = next(
            (
                value
                for value in self.store.list_bindings(
                    domain_kind="run", domain_id=audit.run_id, limit=1000
                ).items
                if value.audit_id == audit.id
                and value.context.get("development_evaluation_id")
            ),
            None,
        )
        if binding is None:
            if (
                require_complete
                and audit.development_suite_revision_id
                and audit.request.get("source") == "training_boundary"
            ):
                raise ValueError(
                    "managed reward audit has no linked development evaluation"
                )
            return None
        evaluation_id = str(binding.context["development_evaluation_id"])
        evaluation = self.db.get_evaluation(evaluation_id)
        if evaluation is None:
            message = f"linked development evaluation is missing: {evaluation_id}"
            if require_complete:
                raise ValueError(message)
            return {
                "evaluation_id": evaluation_id,
                "work_item_id": binding.context.get(
                    "development_evaluation_work_item_id"
                ),
                "suite_revision_id": audit.development_suite_revision_id,
                "status": "missing",
                "stage": "needs_reconciliation",
                "gate_semantics": "completion_evidence_only",
                "changes_reward_integrity_decision": False,
                "error": message,
            }
        if evaluation.suite_revision_id != audit.development_suite_revision_id:
            message = "linked development evaluation uses a different suite revision"
            if require_complete:
                raise ValueError(message)
            return {
                "evaluation_id": evaluation.id,
                "work_item_id": evaluation.work_item_id,
                "suite_revision_id": evaluation.suite_revision_id,
                "status": "incompatible",
                "stage": "needs_reconciliation",
                "gate_semantics": "completion_evidence_only",
                "changes_reward_integrity_decision": False,
                "error": message,
            }
        work_item_id = str(
            binding.context.get("development_evaluation_work_item_id") or ""
        )
        work_item = self.db.get_work_item(work_item_id) if work_item_id else None
        if require_complete:
            if evaluation.status != "completed":
                raise ValueError(
                    "linked development evaluation has not completed successfully"
                )
            if work_item_id and (work_item is None or work_item.status != "completed"):
                raise ValueError(
                    "linked development evaluation work has not completed verification"
                )
            subject = dict(evaluation.request.get("subject") or {})
            payload = dict(subject.get("payload") or {})
            evaluated_hash = str(payload.get("content_hash") or "")
            checkpoint_hash = self.store.get_signal_shard(
                audit.signal_shard_id
            ).checkpoint_hash
            if evaluated_hash and evaluated_hash != checkpoint_hash:
                raise ValueError(
                    "linked development evaluation resolved a different checkpoint"
                )
        return {
            "evaluation_id": evaluation.id,
            "work_item_id": work_item_id or evaluation.work_item_id,
            "suite_revision_id": evaluation.suite_revision_id,
            "status": evaluation.status,
            "stage": evaluation.stage,
            "subject_hash": evaluation.subject_hash,
            "artifact_path": evaluation.artifact_path,
            "gate_semantics": "completion_evidence_only",
            "changes_reward_integrity_decision": False,
            "work_status": (work_item.status if work_item is not None else None),
        }

    def list_audits(self, *, include_shard: bool = True, **values: Any) -> Page[Any]:
        page = self.store.list_audits(**values)
        if not include_shard:
            return page
        items = []
        for audit in page.items:
            shard = self.store.get_signal_shard(audit.signal_shard_id)
            items.append(
                {
                    **audit.to_dict(),
                    "boundary_unit": shard.boundary_unit,
                    "boundary_value": shard.boundary_value,
                    "capture_fidelity": shard.capture_fidelity,
                    "capability_id": shard.capability_id,
                    "trace_hash": shard.trace_hash,
                }
            )
        return Page(items, page.total, page.limit, page.offset)

    def list_audit_samples(
        self,
        audit_id: str,
        *,
        include_observations: bool = True,
        **values: Any,
    ) -> Any:
        page = self.store.list_samples(audit_id, **values)
        if not include_observations or not page.items:
            return page
        ordinals = [item.ordinal for item in page.items]
        placeholders = ",".join("?" for _ in ordinals)
        rows = self.db._conn.execute(
            "SELECT * FROM reward_integrity_observations "
            f"WHERE audit_id=? AND sample_ordinal IN ({placeholders}) "
            "ORDER BY sample_ordinal,role,auditor_ordinal",
            [audit_id, *ordinals],
        ).fetchall()
        grouped: Dict[int, list[Any]] = {}
        for row in rows:
            observation = self.store._observation(row)
            grouped.setdefault(observation.sample_ordinal, []).append(observation)
        items = []
        for sample in page.items:
            observations = grouped.get(sample.ordinal, [])
            optimizer = next((value for value in observations if value.role == "optimizer"), None)
            primary = next(
                (value for value in observations if value.role == "primary_sentinel"),
                None,
            )
            diagnostics = [value.to_dict() for value in observations if value.role == "diagnostic"]
            items.append(
                {
                    **sample.to_dict(),
                    "optimizer_observation": (optimizer.to_dict() if optimizer else None),
                    "primary_sentinel_observation": (primary.to_dict() if primary else None),
                    "diagnostic_observations": diagnostics,
                }
            )
        return Page(items, page.total, page.limit, page.offset)

    def list_audit_metrics(self, audit_id: str, **values: Any) -> Any:
        return self.store.list_metrics(audit_id, **values)

    def cancel_audit(self, audit_id: str) -> RewardIntegrityAudit:
        audit = self.store.get_audit(audit_id)
        if audit.status == "completed":
            raise ValueError("a completed reward audit cannot be cancelled")
        work = None
        if audit.work_item_id:
            work = self.scheduler.cancel(audit.work_item_id)
        terminal = work is not None and work.status == "cancelled"
        return self.store.update_audit(
            audit_id,
            cancel_requested=True,
            **(
                {"status": "cancelled", "stage": "cancelled"}
                if terminal
                else {"stage": "cancelling"}
            ),
        )

    def retry_audit(
        self, audit_id: str, *, reason: str = "operator requested reward-audit retry"
    ) -> RewardIntegrityAudit:
        audit = self.store.get_audit(audit_id)
        if audit.status == "completed":
            raise ValueError("a completed reward audit is immutable")
        if not str(reason).strip():
            raise ValueError("an operator-forced retry requires a reason")
        if audit.work_item_id:
            self.db.retry_work_item(audit.work_item_id, force=True, reason=reason)
        return self.store.update_audit(
            audit_id,
            status="queued",
            stage="queued",
            cancel_requested=False,
            retry_count=audit.retry_count + 1,
            error=None,
        )

    def review_audit(
        self, audit_id: str, *, action: str, reason: str, checkpoint: Optional[str] = None
    ) -> Any:
        reason = str(reason).strip()
        if not reason:
            raise ValueError("reward audit review requires a reason")
        if action == "create_review_proposal":
            audit = self.store.get_audit(audit_id)
            return {
                "action": "create_review_proposal",
                "audit_id": audit_id,
                "source": {
                    "kind": "reward_integrity_audit",
                    "id": audit_id,
                    "run_id": audit.run_id,
                },
                "reason": reason,
                "resolves_pause": False,
            }
        if action not in {"continue", "stop", "fork"}:
            raise ValueError(
                "review action must be continue, stop, fork, or create_review_proposal"
            )
        audit = self.store.get_audit(audit_id)
        decisions = self.store.list_decisions(audit_id, limit=1000).items
        if not decisions:
            raise ValueError("audit has no published decision to review")
        previous = decisions[-1]
        decision = self.store.add_decision(
            audit_id=audit_id,
            integrity_profile_revision_id=audit.integrity_profile_revision_id,
            decision=previous.decision,
            action=action,
            reasons=[f"operator_{action}"],
            evidence={"checkpoint": checkpoint},
            override=True,
            override_note=reason,
            supersedes_decision_id=previous.id,
        )
        self._apply_gate_state(audit, decision)
        return decision

    def sync_audit_replay(
        self,
        audit_id: str,
        *,
        decision: Optional[RewardIntegrityDecision] = None,
    ) -> Dict[str, Any]:
        """Project the latest immutable audit decision into replay V4.

        Review events are committed before replay publication, so transports
        call this method after a successful Continue/Stop/Fork action. Missing
        manifests are truthful ``not_recorded`` results and V1-V3 manifests
        remain byte-for-byte read-only.
        """

        from halo_forge.replay import sync_reward_integrity_decision

        audit = self.store.get_audit(audit_id)
        run = self.db.get_run(audit.run_id)
        if run is None or not str(run.output_dir or "").strip():
            return {"status": "not_recorded", "reason": "run_output_unavailable"}
        if decision is None:
            decisions = self.store.list_decisions(audit_id, limit=1000).items
            if not decisions:
                return {"status": "not_recorded", "reason": "decision_unavailable"}
            decision = decisions[-1]
        return sync_reward_integrity_decision(
            Path(str(run.output_dir)).expanduser() / "replay.json",
            run_id=audit.run_id,
            audit=audit.to_dict(),
            decision=decision.to_dict(),
        )

    def compare_audits(
        self,
        left_audit_id: str,
        right_audit_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> RewardIntegrityComparison:
        limit = max(1, min(1_000, int(limit)))
        offset = max(0, int(offset))
        left_audit = self.store.get_audit(left_audit_id)
        right_audit = self.store.get_audit(right_audit_id)
        left_metrics = {
            item.name: item for item in self.store.list_metrics(left_audit_id, limit=1000).items
        }
        right_metrics = {
            item.name: item for item in self.store.list_metrics(right_audit_id, limit=1000).items
        }
        counts = self.db._conn.execute(
            """SELECT
                   (SELECT COUNT(*) FROM reward_integrity_samples WHERE audit_id=?) AS left_count,
                   (SELECT COUNT(*) FROM reward_integrity_samples WHERE audit_id=?) AS right_count,
                   (SELECT COUNT(*) FROM reward_integrity_samples l
                      JOIN reward_integrity_samples r ON r.snapshot_id=l.snapshot_id
                     WHERE l.audit_id=? AND r.audit_id=?) AS shared_snapshots,
                   (SELECT COUNT(DISTINCT l.record_id) FROM reward_integrity_samples l
                      JOIN reward_integrity_samples r ON r.record_id=l.record_id
                     WHERE l.audit_id=? AND r.audit_id=?) AS shared_records
               """,
            (
                left_audit_id,
                right_audit_id,
                left_audit_id,
                right_audit_id,
                left_audit_id,
                right_audit_id,
            ),
        ).fetchone()
        left_count = int(counts["left_count"] or 0)
        right_count = int(counts["right_count"] or 0)
        shared_snapshots = int(counts["shared_snapshots"] or 0)
        shared_records = int(counts["shared_records"] or 0)

        pair_rows: list[sqlite3.Row] = []
        pair_total = 0
        if shared_snapshots:
            pairing = "paired_snapshot"
            pairing_reason = (
                "Evidence is joined by the exact immutable snapshot_id. "
                "Both sides expose the captured output and verifier observations."
            )
            pair_total = shared_snapshots
            pair_rows = self.db._conn.execute(
                """SELECT l.ordinal AS left_ordinal,r.ordinal AS right_ordinal,
                          l.snapshot_id AS snapshot_id,l.record_id AS record_id,
                          1 AS pair_ordinal
                     FROM reward_integrity_samples l
                     JOIN reward_integrity_samples r ON r.snapshot_id=l.snapshot_id
                    WHERE l.audit_id=? AND r.audit_id=?
                    ORDER BY l.snapshot_id
                    LIMIT ? OFFSET ?""",
                (left_audit_id, right_audit_id, limit, offset),
            ).fetchall()
        elif shared_records:
            pairing = "matched_input"
            pairing_reason = (
                "Evidence shares only the stable record_id. Outputs were produced at "
                "different on-policy boundaries, so this join is distributional, "
                "non-causal, and must not be interpreted as a boundary effect."
            )
            matched_cte = """
                WITH left_ranked AS (
                    SELECT ordinal,snapshot_id,record_id,
                           ROW_NUMBER() OVER (
                               PARTITION BY record_id
                               ORDER BY candidate_ordinal,snapshot_id,ordinal
                           ) AS pair_ordinal
                      FROM reward_integrity_samples WHERE audit_id=?
                ), right_ranked AS (
                    SELECT ordinal,snapshot_id,record_id,
                           ROW_NUMBER() OVER (
                               PARTITION BY record_id
                               ORDER BY candidate_ordinal,snapshot_id,ordinal
                           ) AS pair_ordinal
                      FROM reward_integrity_samples WHERE audit_id=?
                )
            """
            pair_total = int(
                self.db._conn.execute(
                    matched_cte
                    + "SELECT COUNT(*) FROM left_ranked l JOIN right_ranked r "
                    "ON r.record_id=l.record_id AND r.pair_ordinal=l.pair_ordinal",
                    (left_audit_id, right_audit_id),
                ).fetchone()[0]
            )
            pair_rows = self.db._conn.execute(
                matched_cte
                + """SELECT l.ordinal AS left_ordinal,r.ordinal AS right_ordinal,
                            NULL AS snapshot_id,l.record_id AS record_id,
                            l.pair_ordinal AS pair_ordinal
                       FROM left_ranked l JOIN right_ranked r
                         ON r.record_id=l.record_id
                        AND r.pair_ordinal=l.pair_ordinal
                      ORDER BY l.record_id,l.pair_ordinal
                      LIMIT ? OFFSET ?""",
                (left_audit_id, right_audit_id, limit, offset),
            ).fetchall()
        else:
            pairing = "aggregate_only"
            left_fidelity = self.store.get_signal_shard(
                left_audit.signal_shard_id
            ).capture_fidelity
            right_fidelity = self.store.get_signal_shard(
                right_audit.signal_shard_id
            ).capture_fidelity
            if {
                str(left_fidelity),
                str(right_fidelity),
            } & {"aggregate_only", "unavailable", "not_recorded"}:
                pairing_reason = (
                    "At least one audit has no retained per-output evidence. "
                    "Only aggregate metric deltas are available; no evidence pairs are returned."
                )
            else:
                pairing_reason = (
                    "The audits share neither snapshot_id nor stable record_id. "
                    "Only aggregate metric deltas are available; no evidence pairs are returned."
                )

        left_payloads = self._comparison_sample_payloads(
            left_audit_id, [int(row["left_ordinal"]) for row in pair_rows]
        )
        right_payloads = self._comparison_sample_payloads(
            right_audit_id, [int(row["right_ordinal"]) for row in pair_rows]
        )
        pairs: list[RewardIntegrityComparisonPair] = []
        for row in pair_rows:
            left_payload = left_payloads[int(row["left_ordinal"])]
            right_payload = right_payloads[int(row["right_ordinal"])]
            snapshot_id = None if row["snapshot_id"] is None else str(row["snapshot_id"])
            pair_id = snapshot_id or "comparison_pair_" + content_hash(
                {
                    "left_audit_id": left_audit_id,
                    "right_audit_id": right_audit_id,
                    "record_id": str(row["record_id"]),
                    "pair_ordinal": int(row["pair_ordinal"]),
                }
            )
            pairs.append(
                RewardIntegrityComparisonPair(
                    id=pair_id,
                    pairing=pairing,
                    record_id=str(row["record_id"]),
                    snapshot_id=snapshot_id,
                    left_snapshot_id=str(left_payload["snapshot_id"]),
                    right_snapshot_id=str(right_payload["snapshot_id"]),
                    same_output=canonical_json(left_payload.get("output"))
                    == canonical_json(right_payload.get("output")),
                    left=left_payload,
                    right=right_payload,
                )
            )
        deltas = {
            name: (
                right_metrics[name].value - left_metrics[name].value
                if left_metrics[name].value is not None and right_metrics[name].value is not None
                else None
            )
            for name in sorted(left_metrics.keys() & right_metrics.keys())
        }
        return RewardIntegrityComparison(
            left_audit_id=left_audit_id,
            right_audit_id=right_audit_id,
            pairing=pairing,
            pairing_reason=pairing_reason,
            shared_snapshot_count=shared_snapshots,
            shared_record_count=shared_records,
            metric_deltas=deltas,
            unmatched_left=max(0, left_count - pair_total),
            unmatched_right=max(0, right_count - pair_total),
            pairs=pairs,
            pair_total=pair_total,
            limit=limit,
            offset=offset,
        )

    def _comparison_sample_payloads(
        self, audit_id: str, ordinals: Sequence[int]
    ) -> Dict[int, Dict[str, Any]]:
        """Hydrate one bounded comparison page without materializing an audit."""

        if not ordinals:
            return {}
        unique_ordinals = sorted({int(value) for value in ordinals})
        placeholders = ",".join("?" for _ in unique_ordinals)
        sample_rows = self.db._conn.execute(
            "SELECT * FROM reward_integrity_samples "
            f"WHERE audit_id=? AND ordinal IN ({placeholders}) ORDER BY ordinal",
            [audit_id, *unique_ordinals],
        ).fetchall()
        observation_rows = self.db._conn.execute(
            "SELECT * FROM reward_integrity_observations "
            f"WHERE audit_id=? AND sample_ordinal IN ({placeholders}) "
            "ORDER BY sample_ordinal,role,auditor_ordinal",
            [audit_id, *unique_ordinals],
        ).fetchall()
        observations: Dict[int, list[Any]] = {}
        for row in observation_rows:
            value = self.store._observation(row)
            observations.setdefault(value.sample_ordinal, []).append(value)
        payloads: Dict[int, Dict[str, Any]] = {}
        for row in sample_rows:
            sample = self.store._sample(row)
            values = observations.get(sample.ordinal, [])
            optimizer = next((value for value in values if value.role == "optimizer"), None)
            sentinel = next(
                (value for value in values if value.role == "primary_sentinel"), None
            )
            payloads[sample.ordinal] = {
                **sample.to_dict(),
                "optimizer_observation": optimizer.to_dict() if optimizer else None,
                "primary_sentinel_observation": sentinel.to_dict() if sentinel else None,
                "diagnostic_observations": [
                    value.to_dict() for value in values if value.role == "diagnostic"
                ],
            }
        return payloads

    def verify_audit_bundle(self, audit_id: str) -> Dict[str, Any]:
        audit = self.store.get_audit(audit_id)
        if not audit.artifact_path or not audit.manifest_hash:
            raise ValueError("reward audit is not published")
        bundle = self.storage.verify(audit.artifact_path)
        if bundle.manifest_hash != audit.manifest_hash:
            raise ValueError("reward audit manifest identity drifted")
        return {
            "valid": True,
            "manifest_hash": bundle.manifest_hash,
            "content_hash": bundle.content_hash,
            "size_bytes": bundle.size_bytes,
        }

    def list_usage(
        self, reward_system_revision_id: str, *, limit: int = 100, offset: int = 0
    ) -> Any:
        return self.store.list_bindings(
            reward_system_revision_id=reward_system_revision_id, limit=limit, offset=offset
        )

    def bind(self, **values: Any) -> Any:
        return self.store.bind(**values)


__all__ = [
    "PROFILE_DEFAULTS",
    "PROTOCOL_DEFAULTS",
    "RewardIntegrityService",
]
