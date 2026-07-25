"""Operational service for verifier identity, calibration, and qualification.

This module is the single transport-neutral implementation used by the public
API, CLI, durable worker, Dataset Lab, Evaluation Lab, and training handoff.
It keeps the historical verifier plugin API intact while making every new
reliability decision immutable and replayable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from halo_forge.run_db import LabV4Catalog, RunDatabase

from .models import (
    ResolvedVerifierBinding,
    VerifierCalibration,
    VerifierCalibrationComparison,
    VerifierCalibrationMetric,
    VerifierCalibrationSample,
    VerifierObservation,
    VerifierProfileRevision,
    VerifierQualificationDecision,
    VerifierRewardContract,
)
from .store import (
    PROTECTED_SOURCE_PURPOSES,
    VerifierLabStore,
    content_hash,
    scrub_secrets,
)


class VerifierLabError(RuntimeError):
    """Base error for verifier reliability operations."""


class CalibrationCancelled(VerifierLabError):
    """Raised at a complete record/protocol boundary after cancellation."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        return dict(result) if isinstance(result, Mapping) else {"value": result}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected a mapping-compatible value, got {type(value).__name__}")


def _observation_details(value: Any) -> Dict[str, Any]:
    """Normalize historical free-text details at the reliability boundary."""

    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    return {"message": value}


def _page(values: Sequence[Any], *, total: int, limit: int, offset: int) -> Dict[str, Any]:
    return {
        "items": [_as_dict(value) for value in values],
        "total": int(total),
        "limit": max(1, int(limit)),
        "offset": max(0, int(offset)),
    }


def _file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


_VERIFIER_TOOLCHAIN_EXECUTABLES: dict[str, tuple[str, ...]] = {
    "cargo": ("cargo", "rustc"),
    "clang": ("clang",),
    "clang_execution": ("clang",),
    "gcc": ("gcc", "g++"),
    "gcc_execution": ("gcc", "g++"),
    "go": ("go",),
    "mingw": ("x86_64-w64-mingw32-g++",),
    "mingw_execution": ("x86_64-w64-mingw32-g++",),
    "rust": ("cargo", "rustc"),
}


def _default_runtime_contract(
    *, family: str, implementation_ref: str
) -> Dict[str, Any]:
    from .fingerprints import runtime_contract_snapshot

    packages = ["halo-forge"]
    if implementation_ref in {"pytest", "rlvr_pytest"}:
        packages.append("pytest")
    if implementation_ref == "json_schema":
        packages.append("jsonschema")
    if family == "reward_model":
        packages.extend(("torch", "transformers", "tokenizers"))
    return runtime_contract_snapshot(
        package_names=tuple(packages),
        executable_names=_VERIFIER_TOOLCHAIN_EXECUTABLES.get(
            implementation_ref, ()
        ),
        include_accelerator=family == "reward_model",
    )


def _qualification_defaults(template: str) -> Dict[str, Any]:
    """Persist the reviewed v7 template thresholds, not mutable UI defaults."""

    universal = {
        "coverage": {"pass": 0.99, "warn": 0.97, "direction": "maximize"},
        "error_rate": {"pass": 0.01, "warn": 0.03, "direction": "minimize"},
    }
    if template == "strict_oracle":
        return {
            "template": template,
            "universal": universal,
            "primary_agreement": {"pass": 0.98, "warn": 0.95},
            "false_accept_rate": {"pass": 0.01, "warn": 0.03},
            "false_reject_rate": {"pass": 0.02, "warn": 0.05},
            "exact_repeat_agreement": {"pass": 1.0, "warn": 1.0},
        }
    if template == "human_aligned":
        return {
            "template": template,
            "universal": universal,
            "task": {
                "binary": {"primary": [0.80, 0.70]},
                "categorical": {"primary": [0.80, 0.70]},
                "multi_label": {"primary": [0.75, 0.65]},
                "pairwise": {"primary": [0.75, 0.65], "order_consistency": [0.95, 0.90]},
                "ranking": {"primary": [0.60, 0.45]},
                "scalar": {"primary": [0.70, 0.50], "normalized_mae": [0.15, 0.25]},
            },
            "repeat_agreement": {"pass": 0.95, "warn": 0.90},
        }
    if template == "exploratory":
        return {"template": template, "promotable": False, "report_only": True}
    return {"template": "custom"}


class VerifierLabService:
    """One-researcher verifier reliability control plane."""

    def __init__(
        self,
        database: RunDatabase,
        *,
        root: Optional[str | Path] = None,
        scheduler: Any = None,
        adapter_registry: Any = None,
    ) -> None:
        self.db = database
        self.store = VerifierLabStore(database)
        self.root = Path(
            root
            or os.environ.get("HALOFORGE_VERIFIER_CALIBRATION_ROOT")
            or Path.home() / ".halo-forge" / "evaluations" / "verifier-calibrations"
        ).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if scheduler is None:
            from halo_forge.workstation_jobs import WorkstationScheduler

            scheduler = WorkstationScheduler(database)
        self.scheduler = scheduler
        if adapter_registry is None:
            from .adapters import VerifierReliabilityAdapterRegistry

            adapter_registry = VerifierReliabilityAdapterRegistry.from_existing_registry()
        self.adapters = adapter_registry
        self._artifact_reward_adapters: Dict[str, Any] = {}

    def _active_alias_event(
        self, profile_id: str, alias: str, revision_id: str
    ) -> Optional[Any]:
        """Return the append-only event backing the current alias value."""

        try:
            current = self.store.get_alias(profile_id, alias)
        except KeyError:
            return None
        if current.revision_id != revision_id:
            return None
        history = self.store.list_alias_history(profile_id, alias=alias)
        return history[-1] if history else None

    # -- catalog and immutable profiles ---------------------------------

    def _artifact_reward_adapter(self, revision: VerifierProfileRevision) -> Any:
        cached = self._artifact_reward_adapters.get(revision.id)
        if cached is not None:
            return cached
        implementation = dict(revision.definition.get("implementation") or {})
        occurrence_id = str(implementation.get("ref") or revision.implementation_ref)
        catalog = LabV4Catalog(self.db)
        occurrence = catalog.get_occurrence(occurrence_id)
        if occurrence is None:
            raise ValueError(f"Unknown reward-model artifact occurrence: {occurrence_id}")
        blob = catalog.get_blob(occurrence.blob_id)
        if blob is None or blob.integrity_state not in {"verified", "valid"}:
            raise ValueError("Reward-model artifact integrity is not verified")
        if str(blob.format).strip().lower() not in {
            "hf",
            "huggingface",
            "safetensors",
            "transformers",
        }:
            raise ValueError("Reward-model artifact is not in a loadable Transformers format")
        locations = [
            value
            for value in catalog.list_locations(blob.id)
            if value.state in {"available", "verified", "valid"}
            and Path(value.path).expanduser().exists()
        ]
        if not locations:
            raise FileNotFoundError("Reward-model artifact has no available local location")
        managed = [value for value in locations if value.storage_mode == "managed"]
        selected = (managed or locations)[-1]
        from halo_forge.artifact_lab.hashing import hash_path

        actual_hash = hash_path(Path(selected.path).expanduser()).content_hash
        if actual_hash != blob.content_hash:
            raise ValueError(
                "Reward-model artifact content drifted from its verified blob identity"
            )
        from .adapters import ArtifactRewardModelReliabilityAdapter

        adapter = ArtifactRewardModelReliabilityAdapter(
            key=occurrence_id,
            model_path=selected.path,
            content_hash=blob.content_hash,
            modality=revision.modality,
            task_type=revision.task_type,
            tokenizer_revision=(
                revision.definition.get("tokenizer_revision")
                or (revision.definition.get("configuration") or {}).get(
                    "tokenizer_revision"
                )
            ),
        )
        self._artifact_reward_adapters[revision.id] = adapter
        return adapter

    def _assert_implementation_identity(
        self,
        revision: VerifierProfileRevision,
        *,
        rehash_artifact: bool = True,
    ) -> None:
        implementation = dict(revision.definition.get("implementation") or {})
        if revision.family == "chain":
            for component in revision.components:
                self._assert_implementation_identity(
                    self.store.get_profile_revision(component.child_revision_id),
                    rehash_artifact=rehash_artifact,
                )
            return
        if str(implementation.get("kind")) == "artifact":
            if rehash_artifact:
                # Execution boundaries re-hash the selected local occurrence.
                self._artifact_reward_adapters.pop(revision.id, None)
                self._artifact_reward_adapter(revision)
            else:
                occurrence = LabV4Catalog(self.db).get_occurrence(
                    str(implementation.get("ref") or revision.implementation_ref)
                )
                blob = (
                    LabV4Catalog(self.db).get_blob(occurrence.blob_id)
                    if occurrence is not None
                    else None
                )
                if (
                    blob is None
                    or blob.integrity_state not in {"verified", "valid"}
                    or blob.content_hash != revision.implementation_fingerprint
                ):
                    raise VerifierLabError(
                        "Reward-model artifact identity or verified state drifted"
                    )
            return
        from .fingerprints import fingerprint_registered_verifier

        try:
            current = fingerprint_registered_verifier(revision.implementation_ref)
        except Exception:
            if not revision.qualifiable:
                return
            raise
        if (
            revision.implementation_fingerprint
            and current.fingerprint != revision.implementation_fingerprint
        ):
            raise VerifierLabError(
                "Verifier implementation fingerprint drifted; create a new profile revision"
            )

    def capabilities(self) -> Dict[str, Any]:
        from .qualification import qualification_templates

        capabilities = list(self.adapters.capabilities())
        capabilities.append(
            {
                "key": "verified_reward_model_artifact",
                "family": "reward_model",
                "adapter_id": "artifact_reward_model",
                "adapter_version": "1",
                "implementation": "Artifact Studio occurrence",
                "origin": "artifact_library",
                "fingerprint": None,
                "qualifiable": True,
                "modalities": ["text"],
                "tasks": ["scalar", "pairwise", "ranking"],
                "requires_verified_artifact": True,
            }
        )
        return {
            "items": capabilities,
            "total": len(capabilities),
            "families": ["deterministic", "llm_judge", "reward_model", "chain"],
            "task_types": [
                "binary",
                "categorical",
                "multi_label",
                "scalar",
                "pairwise",
                "ranking",
            ],
            "modalities": ["text", "tool", "vlm", "audio"],
            "qualification_templates": qualification_templates(),
            "defaults": {
                "confirmation_partition": {"calibration": 0.70, "confirmation": 0.30},
                "partition_seed": 42,
                "stochastic_seeds": [17, 42, 101],
                "bootstrap_resamples": 10_000,
                "bootstrap_seed": 42,
                "concurrency": 1,
            },
        }

    def _resolve_profile_definition(
        self, definition: Mapping[str, Any]
    ) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
        from .fingerprints import fingerprint_registered_verifier

        value = scrub_secrets(dict(definition))
        family = str(value.get("family") or "deterministic").strip().lower()
        requested_modality = str(value.get("modality") or "text").strip().lower()
        requested_task = str(value.get("task_type") or "binary").strip().lower()
        implementation = value.get("implementation")
        if isinstance(implementation, str):
            implementation = {"kind": "builtin", "ref": implementation}
        implementation = dict(implementation or {})
        components = [dict(item) for item in value.pop("components", [])]
        if family == "chain":
            implementation.setdefault("kind", "chain")
            implementation.setdefault("ref", "ordered_chain")
            if not components:
                raise ValueError("A verifier chain requires ordered components")
            for component in components:
                child_id = str(
                    component.get("child_revision_id") or component.get("revision_id") or ""
                ).strip()
                child = self.store.get_profile_revision(child_id)
                qualified = False
                for alias in ("approved", "candidate"):
                    event = self._active_alias_event(child.profile_id, alias, child.id)
                    if event is not None and not event.override:
                        qualified = True
                        break
                if not qualified:
                    raise ValueError(
                        f"Chain child {child_id} must be candidate-qualified before use"
                    )
                if child.task_type != requested_task or child.modality not in {
                    requested_modality,
                    "any",
                    "multimodal",
                }:
                    raise ValueError(
                        f"Chain child {child_id} is incompatible with the chain task or modality"
                    )
            value.setdefault("reliability_adapter", {"id": "chain", "version": "1"})
        elif str(implementation.get("kind") or "builtin") == "artifact":
            if family != "reward_model":
                raise ValueError("Artifact verifier implementations must be reward models")
            if requested_modality != "text" or requested_task not in {
                "scalar",
                "pairwise",
                "ranking",
            }:
                raise ValueError(
                    "The local artifact reward-model adapter supports text scalar, pairwise, or ranking tasks"
                )
            reference = str(implementation.get("ref") or "").strip()
            occurrence = LabV4Catalog(self.db).get_occurrence(reference)
            if occurrence is None:
                raise ValueError(f"Unknown reward-model artifact occurrence: {reference}")
            blob = LabV4Catalog(self.db).get_blob(occurrence.blob_id)
            if blob is None or blob.integrity_state not in {"verified", "valid"}:
                raise ValueError("Reward-model artifacts must have verified content integrity")
            if str(blob.format).strip().lower() not in {
                "hf",
                "huggingface",
                "safetensors",
                "transformers",
            }:
                raise ValueError(
                    "Reward-model calibration requires a loadable Hugging Face artifact"
                )
            supplied = implementation.get("fingerprint")
            if supplied and str(supplied) != blob.content_hash:
                raise ValueError("Reward-model artifact fingerprint does not match its blob")
            implementation.update(
                fingerprint=blob.content_hash,
                pinned=True,
                artifact_blob_id=blob.id,
            )
            value.setdefault(
                "reliability_adapter", {"id": "artifact_reward_model", "version": "1"}
            )
        else:
            reference = str(implementation.get("ref") or "").strip().lower()
            if not reference:
                raise ValueError("A verifier implementation is required")
            try:
                fingerprint = fingerprint_registered_verifier(reference)
            except Exception as exc:
                blockers = list(value.get("qualification_blockers") or [])
                blockers.append(f"implementation_unresolvable:{type(exc).__name__}")
                value["qualification_blockers"] = blockers
                value["qualifiable"] = False
            else:
                try:
                    capability = self.adapters.get(reference).capability()
                except Exception:
                    capability = None
                if capability is not None and (
                    requested_modality not in capability.modalities
                    or requested_task not in capability.tasks
                ):
                    raise ValueError(
                        f"Verifier {reference!r} does not declare {requested_modality}/{requested_task} support"
                    )
                supplied = implementation.get("fingerprint")
                if supplied and supplied != fingerprint.fingerprint:
                    raise ValueError("Implementation fingerprint drifted from the selected verifier")
                implementation.update(
                    kind=fingerprint.origin,
                    ref=reference,
                    fingerprint=fingerprint.fingerprint,
                    pinned=fingerprint.qualifiable,
                    class_path=fingerprint.class_path,
                )
                value.setdefault(
                    "reliability_adapter",
                    {"id": "registered_verifier", "version": "1"},
                )
        value["family"] = family
        value["implementation"] = implementation
        configuration = dict(value.get("configuration") or {})
        if str(implementation.get("ref") or "") == "llm_judge":
            if configuration.get("endpoint_type"):
                value["endpoint_type"] = str(configuration["endpoint_type"])
            model_revision = value.get("model_revision") or configuration.pop(
                "model_revision", None
            )
            model_revision = str(
                model_revision or configuration.get("judge_model") or ""
            ).strip()
            if not model_revision or model_revision.lower() == "default":
                raise ValueError(
                    "LLM judge profiles require a pinned model name or revision"
                )
            value["model_revision"] = model_revision
            configuration["judge_model"] = model_revision
            if value.get("rubric"):
                configuration["rubric"] = str(value["rubric"])
            # Endpoint type is credential-free provenance. The configured
            # provider integration resolves its URL/key from the environment;
            # it is not an LLMJudgeVerifier constructor argument.
            configuration.pop("endpoint_type", None)
        value["configuration"] = configuration
        contract = VerifierRewardContract.from_value(value.get("reward_contract"))
        value["reward_contract"] = contract.to_dict()
        value["modality"] = requested_modality
        value["task_type"] = requested_task
        value.setdefault("input_mapping", {})
        requested_runtime = value.get("runtime_contract") or value.get(
            "runtime_requirements"
        )
        value["runtime_contract"] = (
            dict(requested_runtime)
            if isinstance(requested_runtime, Mapping) and requested_runtime
            else _default_runtime_contract(
                family=family,
                implementation_ref=str(implementation.get("ref") or ""),
            )
        )
        value.pop("runtime_requirements", None)
        return value, components

    def validate_profile_definition(self, definition: Mapping[str, Any]) -> Dict[str, Any]:
        value, components = self._resolve_profile_definition(definition)
        return {
            "valid": True,
            "resolved_definition": value,
            "components": components,
            "content_hash": content_hash({"definition": value, "components": components}),
            "warnings": list(value.get("qualification_blockers") or []),
        }

    def create_profile(
        self, *, name: str, description: Optional[str], definition: Mapping[str, Any]
    ) -> Dict[str, Any]:
        value, components = self._resolve_profile_definition(definition)
        profile = self.store.create_profile(name=name, description=description)
        try:
            revision = self.store.create_profile_revision(
                profile.id, value, components=components
            )
        except Exception:
            with self.db._lock:
                self.db._conn.execute("DELETE FROM verifier_profiles WHERE id=?", (profile.id,))
                self.db._conn.commit()
            raise
        return {"profile": profile.to_dict(), "revision": revision.to_dict()}

    def revise_profile(
        self, profile_id: str, *, definition: Mapping[str, Any]
    ) -> Dict[str, Any]:
        value, components = self._resolve_profile_definition(definition)
        revision = self.store.create_profile_revision(
            profile_id, value, components=components
        )
        return {
            "profile": self.store.get_profile(profile_id).to_dict(),
            "revision": revision.to_dict(),
        }

    def list_profiles(
        self,
        *,
        query: Optional[str] = None,
        family: Optional[str] = None,
        modality: Optional[str] = None,
        task_type: Optional[str] = None,
        qualified_only: bool = False,
        include_overridden: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        clauses = ["p.archived=0", "p.latest_revision_id=r.id"]
        params: list[Any] = []
        for column, value in (
            ("r.family", family),
            ("r.modality", modality),
            ("r.task_type", task_type),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                params.append(str(value))
        search = str(query or "").strip()
        if search:
            clauses.append("(p.name LIKE ? OR p.description LIKE ? OR p.id LIKE ?)")
            pattern = f"%{search}%"
            params.extend((pattern, pattern, pattern))
        if qualified_only:
            override_clause = ""
            if not include_overridden:
                override_clause = """
                    AND COALESCE((
                        SELECT e.override FROM verifier_alias_events e
                        WHERE e.profile_id=a.profile_id AND e.alias=a.alias
                          AND e.revision_id=a.revision_id
                        ORDER BY e.created_at DESC, e.id DESC LIMIT 1
                    ), 0)=0
                """
            clauses.append(
                """EXISTS (
                    SELECT 1 FROM verifier_aliases a
                    WHERE a.profile_id=p.id AND a.revision_id=r.id
                """
                + override_clause
                + ")"
            )
        where = " WHERE " + " AND ".join(clauses)
        total = int(
            self.db._conn.execute(
                "SELECT COUNT(*) AS value FROM verifier_profiles p "
                "JOIN verifier_profile_revisions r ON r.id=p.latest_revision_id"
                + where,
                params,
            ).fetchone()["value"]
        )
        rows = self.db._conn.execute(
            "SELECT p.id AS profile_id, r.id AS revision_id "
            "FROM verifier_profiles p JOIN verifier_profile_revisions r "
            "ON r.id=p.latest_revision_id"
            + where
            + " ORDER BY p.name, p.id LIMIT ? OFFSET ?",
            (*params, max(1, min(int(limit), 1000)), max(0, int(offset))),
        ).fetchall()
        items: list[Dict[str, Any]] = []
        for row in rows:
            profile = self.store.get_profile(str(row["profile_id"]))
            revision = self.store.get_profile_revision(str(row["revision_id"]))
            aliases = []
            overridden_aliases = []
            for alias in ("candidate", "approved"):
                event = self._active_alias_event(profile.id, alias, revision.id)
                if event is None:
                    continue
                if event.override:
                    overridden_aliases.append(alias)
                else:
                    aliases.append(alias)
            runtime = self.runtime_compatibility(revision.id)
            if qualified_only and not aliases and not (
                include_overridden and overridden_aliases
            ):
                continue
            if qualified_only and runtime["state"] != "compatible" and not (
                include_overridden and overridden_aliases
            ):
                continue
            items.append(
                {
                    **profile.to_dict(),
                    "latest_revision": revision.to_dict(),
                    "aliases": aliases,
                    "overridden_aliases": overridden_aliases,
                    "runtime_compatibility": runtime,
                    "guided_eligible": (
                        bool(aliases)
                        and revision.qualifiable
                        and runtime["state"] == "compatible"
                    ),
                }
            )
        return _page(items, total=total, limit=limit, offset=offset)

    def list_profile_revisions(
        self, profile_id: str, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        self.store.get_profile(profile_id)
        values = self.store.list_profile_revisions(
            profile_id=profile_id, limit=limit, offset=offset
        )
        total = int(
            self.db._conn.execute(
                "SELECT COUNT(*) AS value FROM verifier_profile_revisions WHERE profile_id=?",
                (profile_id,),
            ).fetchone()["value"]
        )
        return _page(values, total=total, limit=limit, offset=offset)

    def get_profile_detail(self, identifier: str) -> Optional[Dict[str, Any]]:
        try:
            revision = self.store.get_profile_revision(identifier)
            profile = self.store.get_profile(revision.profile_id)
        except KeyError:
            try:
                profile = self.store.get_profile(identifier)
            except KeyError:
                return None
            if not profile.latest_revision_id:
                return {"profile": profile.to_dict(), "revisions": []}
            revision = self.store.get_profile_revision(profile.latest_revision_id)
        revisions = self.store.list_profile_revisions(profile_id=profile.id, limit=1000)
        calibrations = self.store.list_calibrations(
            verifier_revision_id=revision.id, limit=100
        )
        aliases: list[Dict[str, Any]] = []
        for alias in ("candidate", "approved"):
            try:
                aliases.append(self.store.get_alias(profile.id, alias).to_dict())
            except KeyError:
                pass
        return {
            "profile": profile.to_dict(),
            "revision": revision.to_dict(),
            "revisions": [value.to_dict() for value in revisions],
            "aliases": aliases,
            "alias_history": [
                value.to_dict() for value in self.store.list_alias_history(profile.id)
            ],
            "calibrations": [value.to_dict() for value in calibrations],
            "usage_count": len(self.store.list_bindings(revision_id=revision.id, limit=1000)),
        }

    # -- immutable protocol and policy records ---------------------------

    def create_protocol(
        self, *, name: str, description: Optional[str], definition: Mapping[str, Any]
    ) -> Dict[str, Any]:
        protocol = self.store.create_protocol(name=name, description=description)
        revision = self.store.create_protocol_revision(protocol.id, definition)
        return {"protocol": protocol.to_dict(), "revision": revision.to_dict()}

    def revise_protocol(self, protocol_id: str, *, definition: Mapping[str, Any]) -> Dict[str, Any]:
        revision = self.store.create_protocol_revision(protocol_id, definition)
        return {
            "protocol": self.store.get_protocol(protocol_id).to_dict(),
            "revision": revision.to_dict(),
        }

    def list_protocols(self, *, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        values = self.store.list_protocols(limit=limit, offset=offset)
        total = int(self.db._conn.execute(
            "SELECT COUNT(*) AS value FROM verifier_calibration_protocols WHERE archived=0"
        ).fetchone()["value"])
        return _page(values, total=total, limit=limit, offset=offset)

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
            "revisions": [
                value.to_dict()
                for value in self.store.list_protocol_revisions(protocol.id, limit=1000)
            ],
        }

    def create_qualification_profile(
        self,
        *,
        name: str,
        description: Optional[str],
        template_kind: str,
        requirements: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        from .qualification import QUALIFICATION_TEMPLATES

        if template_kind not in QUALIFICATION_TEMPLATES:
            raise ValueError("unknown verifier qualification template")
        profile = self.store.create_qualification_profile(
            name=name, description=description
        )
        resolved = dict(requirements or _qualification_defaults(template_kind))
        revision = self.store.create_qualification_profile_revision(
            profile.id,
            template_kind=template_kind,
            promotable=template_kind != "exploratory",
            requirements=resolved,
        )
        return {"profile": profile.to_dict(), "revision": revision.to_dict()}

    def revise_qualification_profile(
        self,
        profile_id: str,
        *,
        template_kind: str,
        requirements: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        from .qualification import QUALIFICATION_TEMPLATES

        if template_kind not in QUALIFICATION_TEMPLATES:
            raise ValueError("unknown verifier qualification template")
        revision = self.store.create_qualification_profile_revision(
            profile_id,
            template_kind=template_kind,
            promotable=template_kind != "exploratory",
            requirements=dict(requirements or _qualification_defaults(template_kind)),
        )
        return {
            "profile": self.store.get_qualification_profile(profile_id).to_dict(),
            "revision": revision.to_dict(),
        }

    def list_qualification_profiles(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        values = self.store.list_qualification_profiles(limit=limit, offset=offset)
        total = int(self.db._conn.execute(
            "SELECT COUNT(*) AS value FROM verifier_qualification_profiles WHERE archived=0"
        ).fetchone()["value"])
        return _page(values, total=total, limit=limit, offset=offset)

    def get_qualification_profile_detail(self, identifier: str) -> Optional[Dict[str, Any]]:
        try:
            revision = self.store.get_qualification_profile_revision(identifier)
            profile = self.store.get_qualification_profile(revision.profile_id)
        except KeyError:
            try:
                profile = self.store.get_qualification_profile(identifier)
            except KeyError:
                return None
            revision = (
                self.store.get_qualification_profile_revision(profile.latest_revision_id)
                if profile.latest_revision_id
                else None
            )
        return {
            "profile": profile.to_dict(),
            "revision": revision.to_dict() if revision else None,
            "revisions": [
                value.to_dict()
                for value in self.store.list_qualification_profile_revisions(
                    profile.id, limit=1000
                )
            ],
        }

    # -- source eligibility and deterministic protocol expansion ---------

    @staticmethod
    def _reference_value(payload: Mapping[str, Any], task_type: str) -> Any:
        for key in (
            "expected",
            "reference",
            "label",
            "value",
            "score",
            "ranking",
            "order",
            "winner",
            "chosen",
            "labels",
        ):
            if key in payload and payload.get(key) is not None:
                return payload.get(key)
        if task_type == "binary" and "accepted" in payload:
            return bool(payload.get("accepted"))
        return None

    def _label_set_source(
        self, revision_id: str, task_type: str, modality: Optional[str] = None
    ) -> tuple[Dict[str, Any], Iterator[Dict[str, Any]]]:
        from halo_forge.review_lab import ReviewLabService

        reviews = ReviewLabService(self.db)
        revision = reviews.get_label_set_revision(revision_id)
        if revision is None:
            raise KeyError(f"Unknown Label Set revision: {revision_id}")
        verification = reviews.verify_label_set(revision_id)
        if not verification.valid:
            raise ValueError("Calibration Label Set failed checksum verification")
        manifest = dict(revision.manifest or {})
        purpose = str(
            manifest.get("purpose") or manifest.get("source_purpose") or "development"
        ).strip().lower()
        if purpose in PROTECTED_SOURCE_PURPOSES or bool(
            manifest.get("protected_lineage")
        ):
            raise ValueError(f"Calibration source {purpose!r} is protected")
        if manifest.get("unresolved_count") or manifest.get("conflict_count"):
            raise ValueError("Calibration Label Set contains unresolved review evidence")
        schema_revision_id = str(manifest.get("schema_revision_id") or "")
        schema_revision = (
            reviews.get_schema_revision(schema_revision_id)
            if schema_revision_id
            else None
        )
        if schema_revision is not None:
            if schema_revision.task_type != task_type:
                raise ValueError(
                    "Calibration Label Set annotation task is incompatible with the verifier"
                )
            if modality and schema_revision.modality not in {
                modality,
                "any",
                "multimodal",
            } and modality not in {"any", "multimodal"}:
                raise ValueError(
                    "Calibration Label Set modality is incompatible with the verifier"
                )

        def records() -> Iterator[Dict[str, Any]]:
            for item in reviews.iter_label_set_items(revision_id):
                if item.excluded:
                    continue
                annotation = dict(item.annotation or {})
                output_records = list(item.output_records or [])
                base = dict(output_records[0]) if output_records else {}
                if task_type in {"pairwise", "ranking"} and output_records:
                    winner = output_records[0].get("chosen")
                    losers = [
                        value.get("rejected")
                        for value in output_records
                        if value.get("rejected") is not None
                    ]
                    if winner is not None and losers:
                        candidates = [winner, *losers]
                        base["candidates"] = candidates
                        base["candidate_ids"] = [
                            f"candidate-{content_hash(value)[:16]}"
                            for value in candidates
                        ]
                        base["expected"] = (
                            winner if task_type == "pairwise" else candidates
                        )
                expected = self._reference_value(annotation, task_type)
                if expected is None:
                    expected = self._reference_value(base, task_type)
                if task_type in {"pairwise", "ranking"} and base.get("expected") is not None:
                    # Rendered preference outputs contain resolved human choices;
                    # they are more stable than source-index annotations.
                    expected = base["expected"]
                if expected is None:
                    raise ValueError(
                        f"Label Set record {item.record_id} has no resolved human reference"
                    )
                lineage = dict(item.lineage or {})
                source_lineage = dict(lineage.get("source") or {})
                lineage_purpose = str(
                    source_lineage.get("purpose")
                    or source_lineage.get("suite_purpose")
                    or ""
                ).strip().lower()
                lineage_split = str(source_lineage.get("split") or "").strip().lower()
                if (
                    lineage_purpose in PROTECTED_SOURCE_PURPOSES
                    or lineage_split in PROTECTED_SOURCE_PURPOSES
                    or source_lineage.get("protected_lineage") is True
                ):
                    raise ValueError(
                        f"Label Set record {item.record_id} descends from protected evidence"
                    )
                if bool(
                    lineage.get("reward_model_training")
                    or source_lineage.get("reward_model_training")
                ):
                    base["reward_model_training"] = True
                lineage_source = lineage.get("source")
                if isinstance(lineage_source, Mapping):
                    for key in (
                        "suite_revision_id",
                        "suite_item_id",
                        "purpose",
                        "split",
                        "protected_lineage",
                    ):
                        if lineage_source.get(key) is not None:
                            base[key] = lineage_source[key]
                base.update(
                    record_id=item.record_id,
                    content_hash=item.record_hash,
                    expected=expected,
                    reference=expected,
                    annotation=annotation,
                    group_id=str(
                        lineage.get("group_id")
                        or lineage.get("related_group_id")
                        or item.record_id
                    ),
                    media_hash=lineage.get("media_hash"),
                    media_hashes=list(
                        lineage.get("media_hashes")
                        or lineage.get("asset_hashes")
                        or []
                    ),
                    subgroup=dict(lineage.get("subgroup") or {}),
                )
                yield base

        return (
            {
                "kind": "label_set",
                "revision_id": revision.id,
                "hash": revision.content_hash,
                "purpose": purpose,
                "count": max(0, revision.row_count - revision.excluded_count),
                "manifest": manifest,
            },
            records(),
        )

    def _benchmark_source(
        self, revision_id: str, task_type: str
    ) -> tuple[Dict[str, Any], Iterator[Dict[str, Any]]]:
        revision = self.db.get_benchmark_suite_revision(revision_id)
        if revision is None:
            raise KeyError(f"Unknown benchmark-suite revision: {revision_id}")
        suite = self.db.get_benchmark_suite(revision.suite_id)
        if suite is None:
            raise ValueError("Benchmark suite identity is missing")
        purpose = str(suite.purpose or "unspecified").strip().lower()
        if purpose not in {"development", "unspecified"}:
            raise ValueError(
                f"{purpose} benchmark evidence cannot calibrate or tune a verifier"
            )

        def records() -> Iterator[Dict[str, Any]]:
            seen: set[str] = set()
            for ordinal, raw in enumerate(revision.items):
                item = dict(raw) if isinstance(raw, Mapping) else {"input": raw}
                record_id = str(
                    item.get("record_id") or item.get("id") or f"suite-item-{ordinal}"
                )
                if record_id in seen:
                    raise ValueError(f"Duplicate benchmark record identity: {record_id}")
                seen.add(record_id)
                expected = self._reference_value(item, task_type)
                if expected is None:
                    raise ValueError(f"Benchmark item {record_id} has no reference label")
                item.update(
                    record_id=record_id,
                    content_hash=str(
                        item.get("content_hash")
                        or content_hash(
                            {
                                key: value
                                for key, value in item.items()
                                if key
                                not in {
                                    "id",
                                    "record_id",
                                    "group_id",
                                    "related_group_id",
                                    "ordinal",
                                    "presentation_order",
                                    "metadata",
                                }
                            }
                        )
                    ),
                    expected=expected,
                    reference=expected,
                    group_id=str(
                        item.get("group_id")
                        or item.get("related_group_id")
                        or record_id
                    ),
                    media_hashes=list(item.get("media_hashes") or item.get("asset_hashes") or []),
                    subgroup=dict(item.get("subgroup") or item.get("metadata") or {}),
                )
                yield item

        revision_hash = str(
            getattr(revision, "content_hash", None)
            or content_hash(
                {
                    "suite_id": revision.suite_id,
                    "revision": revision.revision_number,
                    "items": revision.items,
                    "settings": revision.generation_settings,
                    "evaluators": revision.evaluator_versions,
                }
            )
        )
        return (
            {
                "kind": "benchmark_suite",
                "revision_id": revision.id,
                "hash": revision_hash,
                "purpose": purpose,
                "count": len(revision.items),
                "suite_id": revision.suite_id,
            },
            records(),
        )

    def _source(
        self,
        source_kind: str,
        source_revision_id: str,
        task_type: str,
        modality: Optional[str] = None,
    ) -> tuple[Dict[str, Any], Iterator[Dict[str, Any]]]:
        kind = str(source_kind).strip().lower()
        if kind in {"label_set", "label_set_revision"}:
            return self._label_set_source(source_revision_id, task_type, modality)
        if kind in {"benchmark_suite", "benchmark_suite_revision"}:
            return self._benchmark_source(source_revision_id, task_type)
        raise ValueError("Calibration source must be label_set or benchmark_suite")

    def _protocol_for(
        self,
        revision: VerifierProfileRevision,
        definition: Mapping[str, Any],
        *,
        confirmation: bool,
    ) -> Any:
        from .protocol import CalibrationProtocol

        deterministic = bool(
            definition.get(
                "deterministic", self._revision_is_deterministic(revision)
            )
        )
        return CalibrationProtocol(
            family=revision.family,
            task_type=revision.task_type,
            deterministic=deterministic,
            seed=int(definition.get("seed", 42)),
            confirmation_requested=confirmation,
            confirmation_fraction=float(definition.get("confirmation_fraction", 0.30)),
            stochastic_seeds=tuple(definition.get("stochastic_seeds") or (17, 42, 101)),
            temperature=float(definition.get("temperature", 0.0)),
            top_p=float(definition.get("top_p", 1.0)),
            concurrency=int(definition.get("concurrency", 1)),
            ranking_orientation_cap=int(definition.get("ranking_orientation_cap", 4)),
            production_batch_size=definition.get("production_batch_size"),
            reward_model_dtype=definition.get("reward_model_dtype"),
            reviewed_probe_kinds=tuple(definition.get("reviewed_probe_kinds") or ()),
        )

    def _revision_is_deterministic(
        self,
        revision: VerifierProfileRevision,
        *,
        _visiting: Optional[set[str]] = None,
    ) -> bool:
        """Infer replication semantics through an ordered verifier chain."""

        if revision.family in {"deterministic", "reward_model"}:
            return True
        if revision.family != "chain":
            return False
        visiting = set(_visiting or ())
        if revision.id in visiting:
            raise ValueError("verifier chain cycle detected while resolving protocol")
        visiting.add(revision.id)
        return all(
            self._revision_is_deterministic(
                self.store.get_profile_revision(component.child_revision_id),
                _visiting=visiting,
            )
            for component in revision.components
        )

    # -- durable calibration lifecycle ----------------------------------

    def launch_calibration(
        self,
        *,
        verifier_revision_id: str,
        source_kind: str,
        source_revision_id: str,
        protocol_revision_id: str,
        qualification_profile_revision_id: str,
        confirmation: bool = False,
        runtime_identity: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        from .fingerprints import runtime_identity as capture_runtime

        source_kind = {
            "label_set_revision": "label_set",
            "benchmark_suite_revision": "benchmark_suite",
        }.get(str(source_kind).strip().lower(), str(source_kind).strip().lower())
        verifier = self.store.get_profile_revision(verifier_revision_id)
        self._assert_implementation_identity(verifier)
        protocol_revision = self.store.get_protocol_revision(protocol_revision_id)
        self.store.get_qualification_profile_revision(
            qualification_profile_revision_id
        )
        source, _ = self._source(
            source_kind,
            source_revision_id,
            verifier.task_type,
            verifier.modality,
        )
        if verifier.family == "reward_model" and bool(
            source.get("manifest", {}).get("reward_model_training")
        ):
            raise ValueError("Reward-model training records cannot calibrate that reward model")
        protocol = self._protocol_for(
            verifier, protocol_revision.definition, confirmation=confirmation
        )
        runtime = capture_runtime(runtime_identity or {})
        request = {
            "confirmation": bool(confirmation),
            "protocol": protocol.to_dict(),
            "source": {key: value for key, value in source.items() if key != "manifest"},
        }
        calibration = self.store.create_calibration(
            verifier_revision_id=verifier.id,
            protocol_revision_id=protocol_revision.id,
            qualification_profile_revision_id=qualification_profile_revision_id,
            source_kind=source_kind,
            source_revision_id=source_revision_id,
            source_hash=str(source["hash"]),
            source_purpose=str(source["purpose"]),
            request=request,
            partition={"requested": bool(confirmation), "seed": 42},
            runtime_identity=runtime,
            total_records=int(source["count"]),
        )
        if calibration.status == "completed":
            verification = self.verify_calibration(calibration.id)
            if not verification.get("valid"):
                raise VerifierLabError(
                    "An identical completed calibration exists but its bundle is corrupt"
                )
            return {**calibration.to_dict(), "reused": True}
        if calibration.work_item_id:
            # Identical queued/running requests share one durable work item.
            return {
                **calibration.to_dict(),
                "work_item_id": calibration.work_item_id,
                "reused": True,
                "reuse_pending": True,
            }
        reusable = self.store.find_reusable_calibration(calibration.reuse_key)
        if reusable is not None and reusable.id != calibration.id:
            verification = self.verify_calibration(reusable.id)
            if verification.get("valid"):
                with self.db._lock:
                    self.db._conn.execute(
                        "DELETE FROM verifier_calibrations WHERE id=?", (calibration.id,)
                    )
                    self.db._conn.commit()
                return {**reusable.to_dict(), "reused": True}
        local_model = verifier.family == "reward_model" or bool(
            verifier.definition.get("runtime_contract", {}).get("accelerator")
        )
        if verifier.family == "llm_judge":
            endpoint = str(
                verifier.definition.get("endpoint_type")
                or verifier.definition.get("configuration", {}).get("endpoint_type")
                or ""
            ).lower()
            local_model = endpoint in {"ollama", "local", "openai_compatible_local"}
        work_item_id = f"verifier-calibration-work-{uuid.uuid4().hex}"
        work_item = self.scheduler.enqueue(
            kind="verifier_calibration",
            launch_spec={
                "handler": "verifier_lab.run_calibration",
                "calibration_id": calibration.id,
                "calibration_root": str(self.root),
            },
            resource_class="accelerator" if local_model else "cpu",
            resource_requirements={
                "exclusive_heavy_operation": True,
                "output_path": str(self.root),
                "provider_concurrency": 1,
            },
            domain_kind="verifier_calibration",
            domain_id=calibration.id,
            log_path=str(self.root / ".logs" / f"{calibration.id}.log"),
            max_retries=2,
            work_item_id=work_item_id,
        )
        calibration = self.store.update_calibration(
            calibration.id, work_item_id=work_item.id
        )
        return {**calibration.to_dict(), "work_item_id": work_item.id, "reused": False}

    def get_calibration(self, calibration_id: str) -> Optional[VerifierCalibration]:
        try:
            return self.store.get_calibration(calibration_id)
        except KeyError:
            return None

    def get_calibration_detail(self, calibration_id: str) -> Optional[Dict[str, Any]]:
        value = self.get_calibration(calibration_id)
        if value is None:
            return None
        metrics, metric_total, metric_limit, _ = self.store.query_metrics(
            value.id, limit=200, offset=0
        )
        decisions, decision_total, decision_limit, _ = self.store.query_decisions(
            calibration_id=value.id, limit=100, offset=0
        )
        return {
            **value.to_dict(),
            "metrics": [metric.to_dict() for metric in metrics],
            "metrics_page": {
                "total": metric_total,
                "limit": metric_limit,
                "offset": 0,
            },
            "decisions": [decision.to_dict() for decision in decisions],
            "decisions_page": {
                "total": decision_total,
                "limit": decision_limit,
                "offset": 0,
            },
            "integrity": self.verify_calibration(value.id) if value.artifact_path else None,
            "runtime_compatibility": self.runtime_compatibility(value.verifier_revision_id),
        }

    def list_calibrations(
        self,
        *,
        verifier_revision_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        values = self.store.list_calibrations(
            verifier_revision_id=verifier_revision_id,
            status=status,
            limit=limit,
            offset=offset,
        )
        clauses: list[str] = []
        params: list[Any] = []
        if verifier_revision_id:
            clauses.append("verifier_revision_id=?")
            params.append(verifier_revision_id)
        if status:
            clauses.append("status=?")
            params.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        total = int(
            self.db._conn.execute(
                "SELECT COUNT(*) AS value FROM verifier_calibrations" + where, params
            ).fetchone()["value"]
        )
        return _page(values, total=total, limit=limit, offset=offset)

    @staticmethod
    def _mapped_item(
        revision: VerifierProfileRevision, item: Mapping[str, Any]
    ) -> Dict[str, Any]:
        mapped_item = dict(item)
        mapping_source = (
            item.get("record") if isinstance(item.get("record"), Mapping) else item
        )
        for target, source in revision.input_mapping.items():
            current: Any = mapping_source
            for part in str(source).split("."):
                if not isinstance(current, Mapping) or part not in current:
                    current = None
                    break
                current = current[part]
            if current is not None:
                mapped_item[str(target)] = current
        return mapped_item

    def _invoke_revision(
        self,
        revision: VerifierProfileRevision,
        item: Mapping[str, Any],
        *,
        runtime: Optional[Mapping[str, Any]] = None,
        isolated_pass: Optional[Any] = None,
    ) -> VerifierObservation:
        mapped_item = self._mapped_item(revision, item)
        runtime_value = dict(runtime or {})
        if (
            revision.family == "deterministic"
            and bool(runtime_value.get("fresh_process_requested"))
        ):
            try:
                from .process_runner import RESULT_PREFIX

                request = {
                    "implementation_ref": revision.implementation_ref,
                    "configuration": dict(
                        revision.definition.get("configuration") or {}
                    ),
                    "modality": revision.modality,
                    "task_type": revision.task_type,
                    "reward_contract": revision.reward_contract.to_dict(),
                    "item": mapped_item,
                    "runtime": runtime_value,
                }
                timeout = max(
                    1.0,
                    min(
                        3600.0,
                        float(revision.runtime_contract.get("timeout_seconds", 300.0)),
                    ),
                )
                revision_pass = (
                    isolated_pass.get(revision.id)
                    if isinstance(isolated_pass, Mapping)
                    else isolated_pass
                )
                if revision_pass is not None:
                    isolated = revision_pass.invoke(request)
                    return_code = 0
                else:
                    completed = subprocess.run(
                        [sys.executable, "-m", "halo_forge.verifier_lab.process_runner"],
                        input=json.dumps(request, sort_keys=True, allow_nan=False),
                        text=True,
                        capture_output=True,
                        timeout=timeout,
                        check=False,
                    )
                    result_line = next(
                        (
                            line[len(RESULT_PREFIX) :]
                            for line in reversed(completed.stdout.splitlines())
                            if line.startswith(RESULT_PREFIX)
                        ),
                        None,
                    )
                    if result_line is None:
                        detail = (completed.stderr or completed.stdout).strip()[-500:]
                        raise RuntimeError(
                            "isolated verifier returned no observation"
                            + (f": {detail}" if detail else "")
                        )
                    isolated = json.loads(result_line)
                    return_code = completed.returncode
                if return_code != 0 or (
                    isolated.get("error")
                    and "runtime_identity" not in isolated
                    and "passed" not in isolated
                ):
                    raise RuntimeError(str(isolated.get("error") or "isolated verifier failed"))
                return VerifierObservation(
                    reward=isolated.get("reward"),
                    passed=isolated.get("passed"),
                    parsed_value=isolated.get("parsed_value"),
                    raw_output=isolated.get("raw_output"),
                    details=_observation_details(isolated.get("details")),
                    component_trace=[
                        dict(value) for value in isolated.get("component_trace") or []
                    ],
                    latency_ms=isolated.get("latency_ms"),
                    error=isolated.get("error"),
                    runtime_identity=dict(
                        isolated.get("runtime_identity") or runtime_value
                    ),
                )
            except subprocess.TimeoutExpired:
                return VerifierObservation(
                    reward=None,
                    passed=(
                        False
                        if revision.reward_contract.error_behavior == "fail_closed"
                        else None
                    ),
                    details={"process_isolation": "fresh_interpreter"},
                    error="TimeoutError: isolated verifier invocation timed out",
                    runtime_identity=runtime_value,
                )
            except Exception as exc:
                return VerifierObservation(
                    reward=None,
                    passed=(
                        False
                        if revision.reward_contract.error_behavior == "fail_closed"
                        else None
                    ),
                    details={"process_isolation": "fresh_interpreter"},
                    error=f"{type(exc).__name__}: {exc}",
                    runtime_identity=runtime_value,
                )
        if revision.family == "chain":
            from .chains import aggregate_chain_observations

            children: Dict[str, Any] = {}
            for component in revision.components:
                child = self.store.get_profile_revision(component.child_revision_id)
                children[child.id] = self._invoke_revision(
                    child,
                    mapped_item,
                    runtime=runtime,
                    isolated_pass=isolated_pass,
                ).to_dict()
            aggregated = aggregate_chain_observations(
                [
                    {
                        "revision_id": value.child_revision_id,
                        "order_index": value.ordinal,
                        "weight": value.weight,
                        "veto": value.veto,
                    }
                    for value in revision.components
                ],
                children,
                aggregation=str(
                    revision.definition.get("aggregation")
                    or revision.definition.get("aggregation_rule")
                    or "weighted_mean"
                ),
                contract=revision.reward_contract,
                runtime_identity=runtime,
            )
        else:
            reference = revision.implementation_ref
            configuration = dict(revision.definition.get("configuration") or {})
            if reference == "llm_judge":
                generation = dict((runtime or {}).get("generation_settings") or {})
                for key in ("seed", "temperature", "top_p"):
                    if generation.get(key) is not None:
                        configuration[key] = generation[key]
                configuration["max_workers"] = 1
                endpoint_type = str(
                    revision.definition.get("endpoint_type") or "openai_compatible"
                ).strip().lower()
                if endpoint_type in {
                    "ollama",
                    "local",
                    "openai_compatible",
                    "openai_compatible_local",
                    "hosted",
                }:
                    from halo_forge.data_lab.integrations import configured_teacher

                    provider_parameters = {
                        **configuration,
                        "endpoint_type": (
                            "openai_compatible"
                            if endpoint_type
                            in {"local", "openai_compatible_local", "hosted"}
                            else endpoint_type
                        ),
                        "teacher_model": configuration.get("judge_model") or "default",
                        "max_tokens": 8,
                        "sampling": {
                            "max_tokens": 8,
                            "temperature": generation.get("temperature", 0.0),
                            "top_p": generation.get("top_p", 1.0),
                            "seed": generation.get("seed"),
                        },
                    }
                    configuration["judge_callable"] = lambda prompt: configured_teacher(
                        prompt, provider_parameters, row=mapped_item
                    )
                allowed_judge_arguments = {
                    "rubric",
                    "scoring_scale",
                    "judge_model",
                    "prompt",
                    "judge_callable",
                    "base_url",
                    "api_key",
                    "timeout_s",
                    "max_workers",
                    "seed",
                    "temperature",
                    "top_p",
                }
                configuration = {
                    key: value
                    for key, value in configuration.items()
                    if key in allowed_judge_arguments
                    and not (key == "api_key" and value == "<redacted>")
                }
            try:
                if revision.family == "reward_model" and str(
                    (revision.definition.get("implementation") or {}).get("kind")
                ) == "artifact":
                    adapter = self._artifact_reward_adapter(revision)
                else:
                    try:
                        from .adapters import RegistryVerifierReliabilityAdapter

                        adapter = RegistryVerifierReliabilityAdapter(
                            reference,
                            configuration=configuration,
                            family=(
                                revision.family
                                if revision.family in {"deterministic", "llm_judge"}
                                else None
                            ),
                            modalities=(revision.modality,),
                            tasks=(revision.task_type,),
                        )
                    except Exception:
                        adapter = self.adapters.get(
                            reference,
                            version=revision.reliability_adapter_version,
                        )
            except Exception as exc:
                return VerifierObservation(
                    reward=None,
                    passed=False,
                    details={"implementation_ref": reference},
                    error=f"Verifier implementation unavailable: {exc}",
                    runtime_identity=dict(runtime or {}),
                )
            try:
                aggregated = adapter.invoke(
                    mapped_item,
                    contract=revision.reward_contract,
                    runtime=runtime,
                )
            except Exception as exc:
                # Contract violations and parser/runtime failures are evidence
                # about reliability. Persist them as rejected observations;
                # never clamp a reward or abort the whole replicated batch.
                aggregated = VerifierObservation(
                    reward=None,
                    passed=(
                        False
                        if revision.reward_contract.error_behavior == "fail_closed"
                        else None
                    ),
                    details={"implementation_ref": reference},
                    error=f"{type(exc).__name__}: {exc}",
                    runtime_identity=dict(runtime or {}),
                )
        raw = aggregated.to_dict() if hasattr(aggregated, "to_dict") else dict(aggregated)
        return VerifierObservation(
            reward=raw.get("reward"),
            passed=raw.get("passed"),
            parsed_value=raw.get("parsed_value"),
            raw_output=raw.get("raw_output"),
            details=_observation_details(raw.get("details")),
            component_trace=[dict(value) for value in raw.get("component_trace") or []],
            latency_ms=raw.get("latency_ms"),
            error=raw.get("error"),
            runtime_identity=dict(raw.get("runtime_identity") or runtime or {}),
        )

    def _deterministic_isolation_revisions(
        self, revision: VerifierProfileRevision
    ) -> tuple[VerifierProfileRevision, ...]:
        """Return deterministic leaves that need one interpreter per pass.

        A mixed or deterministic chain is invoked by the parent process, but
        each deterministic child still has the same fresh-process replication
        contract as a top-level deterministic verifier.  Returning unique
        leaves lets the calibration worker keep one process open for each
        child and repetition instead of spawning a process for every record.
        """

        result: list[VerifierProfileRevision] = []
        seen: set[str] = set()
        visiting: set[str] = set()

        def visit(current: VerifierProfileRevision) -> None:
            if current.id in visiting:
                raise ValueError("verifier chain cycle detected while preparing isolation")
            if current.id in seen:
                return
            if current.family == "deterministic":
                seen.add(current.id)
                result.append(current)
                return
            if current.family != "chain":
                seen.add(current.id)
                return
            visiting.add(current.id)
            for component in current.components:
                visit(self.store.get_profile_revision(component.child_revision_id))
            visiting.remove(current.id)
            seen.add(current.id)

        visit(revision)
        return tuple(result)

    @staticmethod
    def _validate_scalar_reference_contract(
        revision: VerifierProfileRevision,
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        """Reject invalid human scalar references before any verifier runs."""

        if revision.task_type != "scalar":
            return
        contract = revision.reward_contract
        for record in records:
            record_id = str(record.get("record_id") or "<unknown>")
            expected = record.get("expected")
            if isinstance(expected, bool):
                raise ValueError(
                    f"Scalar human reference for record {record_id!r} must be numeric"
                )
            try:
                value = float(expected)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Scalar human reference for record {record_id!r} must be numeric"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"Scalar human reference for record {record_id!r} must be finite"
                )
            if value < contract.minimum or value > contract.maximum:
                raise ValueError(
                    f"Scalar human reference for record {record_id!r} is outside the "
                    f"immutable reward contract [{contract.minimum}, {contract.maximum}]"
                )

    def invoke_revision(
        self, revision_id: str, item: Mapping[str, Any]
    ) -> VerifierObservation:
        return self._invoke_revision(self.store.get_profile_revision(revision_id), item)

    @staticmethod
    def _prediction(observation: VerifierObservation, task_type: str) -> Any:
        if task_type == "binary":
            return (
                observation.passed
                if observation.passed is not None
                else observation.parsed_value
            )
        if task_type == "scalar":
            return (
                observation.reward
                if observation.reward is not None
                else observation.parsed_value
            )
        if observation.parsed_value is not None:
            return observation.parsed_value
        if task_type in {"categorical", "multi_label", "pairwise"}:
            return observation.passed
        return observation.reward

    @staticmethod
    def _reward_model_batch_metrics(
        samples: Sequence[VerifierCalibrationSample], *, dtype: str
    ) -> Dict[str, float]:
        from .protocol import dtype_score_tolerance, reward_model_batch_consistent

        paired: Dict[str, Dict[str, Optional[float]]] = {}
        for sample in samples:
            if sample.probe_kind not in {"batch_size_one", "production_batch_size"}:
                continue
            paired.setdefault(sample.record_id, {})[sample.probe_kind] = (
                sample.observation.reward
            )
        tolerance = dtype_score_tolerance(dtype)
        deltas: list[float] = []
        consistent = 0
        complete = 0
        for values in paired.values():
            one = values.get("batch_size_one")
            production = values.get("production_batch_size")
            if one is None or production is None:
                continue
            complete += 1
            deltas.append(abs(float(one) - float(production)))
            if reward_model_batch_consistent(float(one), float(production), dtype=dtype):
                consistent += 1
        requested = len(paired)
        return {
            "reward_model_batch_consistency": (
                float(consistent) / float(complete) if complete else 0.0
            ),
            "reward_model_batch_evidence_coverage": (
                float(complete) / float(requested) if requested else 0.0
            ),
            "reward_model_batch_max_delta": max(deltas) if deltas else 0.0,
            "reward_model_batch_tolerance": float(tolerance),
            "reward_model_batch_record_count": float(complete),
        }

    def _reward_model_production_observations(
        self,
        revision: VerifierProfileRevision,
        invocations: Sequence[Any],
        *,
        batch_size: int,
        runtime: Mapping[str, Any],
    ) -> Dict[str, VerifierObservation]:
        """Run the parity probe as real multi-record model batches.

        A larger ``batch_size`` argument on a one-record call is not batch
        evidence.  Only adapters exposing ``invoke_batch`` can produce this
        probe; unsupported integrations return explicit error observations.
        """

        if not invocations:
            return {}
        try:
            if str((revision.definition.get("implementation") or {}).get("kind")) == "artifact":
                adapter = self._artifact_reward_adapter(revision)
            else:
                adapter = self.adapters.get(
                    revision.implementation_ref,
                    version=revision.reliability_adapter_version,
                )
            invoke_batch = getattr(adapter, "invoke_batch", None)
            if not callable(invoke_batch):
                raise TypeError("reward-model adapter does not support real batched scoring")
            mapped = [self._mapped_item(revision, value.payload) for value in invocations]
            observations = list(
                invoke_batch(
                    mapped,
                    contract=revision.reward_contract,
                    batch_size=max(1, int(batch_size)),
                    runtime={
                        **dict(runtime),
                        "batch_parity_probe": True,
                        "requested_production_batch_size": max(1, int(batch_size)),
                    },
                )
            )
            if len(observations) != len(invocations):
                raise ValueError("reward-model batch adapter returned the wrong item count")
            return {
                invocation.invocation_id: observation
                for invocation, observation in zip(invocations, observations)
            }
        except Exception as exc:
            return {
                value.invocation_id: VerifierObservation(
                    reward=None,
                    passed=(
                        False
                        if revision.reward_contract.error_behavior == "fail_closed"
                        else None
                    ),
                    details={
                        "batch_parity_probe": True,
                        "requested_production_batch_size": max(1, int(batch_size)),
                        "actual_batch_record_count": 0,
                    },
                    error=f"{type(exc).__name__}: {exc}",
                    runtime_identity=dict(runtime),
                )
                for value in invocations
            }

    @staticmethod
    def _metric_direction(name: str) -> Optional[str]:
        from .metrics import metric_direction

        return metric_direction(name)

    def _persist_metric_result(
        self,
        calibration_id: str,
        result: Mapping[str, Any],
        *,
        partition: str,
    ) -> None:
        with self.db._lock:
            self.db._conn.execute(
                "DELETE FROM verifier_calibration_metrics WHERE calibration_id=? AND partition=?",
                (calibration_id, partition),
            )
            self.db._conn.commit()
        primary = dict(result.get("primary_metric") or {})
        interval = dict(result.get("primary_metric_interval") or {})
        bootstrap = dict(result.get("bootstrap") or {})
        subgroup_analysis = dict(result.get("subgroup_analysis") or {})
        # The reliability engine names percentile bounds ``lower``/``upper``.
        # Accept the public/storage-shaped aliases as well so an injected
        # adapter can round-trip an already-normalized interval.
        interval_low = interval.get("ci_low", interval.get("lower"))
        interval_high = interval.get("ci_high", interval.get("upper"))
        sections = ("universal", "task", "probability")
        persisted: set[tuple[str, str]] = set()
        for section in sections:
            raw = result.get(section)
            if not isinstance(raw, Mapping):
                continue
            structured = {
                str(name): value
                for name, value in raw.items()
                if isinstance(value, bool)
                or (value is not None and not isinstance(value, (int, float)))
            }
            if section == "task" and result.get("threshold_curve") is not None:
                structured["threshold_curve"] = result["threshold_curve"]
            scalar_names = [
                str(name)
                for name, value in raw.items()
                if not isinstance(value, bool)
                and (value is None or isinstance(value, (int, float)))
            ]
            primary_name = str(primary.get("name") or "")
            diagnostic_anchor = (
                primary_name
                if section == "task" and primary_name in scalar_names
                else (scalar_names[0] if scalar_names else None)
            )
            for name, value in raw.items():
                if isinstance(value, bool) or (
                    value is not None and not isinstance(value, (int, float))
                ):
                    continue
                metric_name = str(name)
                key = (metric_name, "")
                if key in persisted:
                    metric_name = f"{section}.{metric_name}"
                    key = (metric_name, "")
                persisted.add(key)
                is_primary = str(primary.get("name")) == str(name)
                diagnostics: Dict[str, Any] = (
                    {**structured, "structured": structured}
                    if structured and str(name) == diagnostic_anchor
                    else {}
                )
                reliability_diagnostics: Dict[str, Any] = (
                    {
                        "bootstrap": bootstrap,
                        "primary_metric_interval": interval,
                        **(
                            {"subgroup_analysis": subgroup_analysis}
                            if subgroup_analysis
                            else {}
                        ),
                    }
                    if is_primary
                    else {}
                )
                if value is None:
                    self.store.append_metric(
                        calibration_id,
                        name=metric_name,
                        value=None,
                        partition=partition,
                        available=False,
                        missing_reason="metric unavailable for the compatible evidence",
                        record_count=int(
                            (result.get("task") or {}).get(
                                "record_count",
                                (result.get("universal") or {}).get("record_count", 0),
                            )
                            or 0
                        ),
                        metadata={
                            "section": section,
                            "primary": is_primary,
                            **reliability_diagnostics,
                            **diagnostics,
                        },
                    )
                    continue
                self.store.append_metric(
                    calibration_id,
                    name=metric_name,
                    value=float(value),
                    partition=partition,
                    ci_low=(
                        float(interval_low)
                        if is_primary and interval_low is not None
                        else None
                    ),
                    ci_high=(
                        float(interval_high)
                        if is_primary and interval_high is not None
                        else None
                    ),
                    direction=self._metric_direction(str(name)),
                    record_count=int(
                        (result.get("task") or {}).get(
                            "record_count", (result.get("universal") or {}).get("record_count", 0)
                        )
                        or 0
                    ),
                    metadata={
                        "section": section,
                        "primary": is_primary,
                        **reliability_diagnostics,
                        **diagnostics,
                    },
                )
        for subgroup, payload in dict(result.get("subgroups") or {}).items():
            if not isinstance(payload, Mapping) or not payload.get("available"):
                self.store.append_metric(
                    calibration_id,
                    name="subgroup",
                    value=None,
                    partition=partition,
                    subgroup=str(subgroup),
                    available=False,
                    missing_reason=str(
                        (payload or {}).get("reason") or "subgroup evidence unavailable"
                    ),
                    record_count=int((payload or {}).get("record_count") or 0),
                    metadata={"section": "subgroup", "payload": dict(payload or {})},
                )
                continue
            metrics = dict(payload.get("metrics") or {})
            primary_name = str(primary.get("name") or "")
            primary_value = metrics.get(primary_name)
            if isinstance(primary_value, (int, float)) and not isinstance(primary_value, bool):
                self.store.append_metric(
                    calibration_id,
                    name=primary_name,
                    value=float(primary_value),
                    partition=partition,
                    subgroup=str(subgroup),
                    direction=self._metric_direction(primary_name),
                    record_count=int(payload.get("record_count") or 0),
                    metadata={
                        "section": "subgroup",
                        "primary": True,
                        "payload": dict(payload),
                    },
                )

    def _all_samples(
        self, calibration_id: str, *, partition: Optional[str] = None
    ) -> Iterator[VerifierCalibrationSample]:
        offset = 0
        while True:
            page = self.store.list_samples(
                calibration_id,
                partition=partition,
                limit=1000,
                offset=offset,
            )
            if not page:
                return
            yield from page
            offset += len(page)

    def _bundle_files(
        self, calibration: VerifierCalibration, staging: Path
    ) -> Dict[str, str]:
        profile = self.store.get_profile_revision(calibration.verifier_revision_id)
        protocol = self.store.get_protocol_revision(calibration.protocol_revision_id)
        qualification = self.store.get_qualification_profile_revision(
            calibration.qualification_profile_revision_id
        )
        payloads: Dict[str, bytes] = {
            "profile.json": _canonical_bytes(profile.to_dict()),
            "source.json": _canonical_bytes(
                {
                    "kind": calibration.source_kind,
                    "revision_id": calibration.source_revision_id,
                    "hash": calibration.source_hash,
                    "purpose": calibration.source_purpose,
                    "partition": calibration.partition,
                }
            ),
            "protocol.json": _canonical_bytes(protocol.to_dict()),
            "runtime.json": _canonical_bytes(calibration.runtime_identity),
            "qualification.json": _canonical_bytes(
                {
                    "profile": qualification.to_dict(),
                    "decisions": [
                        value.to_dict()
                        for value in self.store.list_decisions(calibration.id)
                    ],
                }
            ),
            "metrics.json": _canonical_bytes(
                [value.to_dict() for value in self.store.list_metrics(calibration.id)]
            ),
        }
        staging.mkdir(parents=True, exist_ok=False)
        checksums: Dict[str, str] = {}
        for name, data in payloads.items():
            path = staging / name
            path.write_bytes(data)
            checksums[name] = hashlib.sha256(data).hexdigest()
        samples_path = staging / "samples.jsonl"
        with samples_path.open("wb") as handle:
            for sample in self._all_samples(calibration.id):
                handle.write(_canonical_bytes(sample.to_dict()))
        checksums["samples.jsonl"] = _file_sha256(samples_path)
        return checksums

    def _publish_bundle(self, calibration_id: str) -> VerifierCalibration:
        calibration = self.store.get_calibration(calibration_id)
        final = self.root / calibration.id
        staging = self.root / ".staging" / f"{calibration.id}-{uuid.uuid4().hex}"
        staging.parent.mkdir(parents=True, exist_ok=True)
        try:
            checksums = self._bundle_files(calibration, staging)
            manifest = {
                "format_version": 1,
                "calibration_id": calibration.id,
                "verifier_revision_id": calibration.verifier_revision_id,
                "source_hash": calibration.source_hash,
                "protocol_hash": calibration.protocol_hash,
                "qualification_hash": calibration.qualification_hash,
                "runtime_identity_hash": calibration.runtime_identity_hash,
                "sample_count": calibration.sample_count,
                "checksums": checksums,
                "created_at": _now(),
            }
            manifest_bytes = _canonical_bytes(manifest)
            (staging / "manifest.json").write_bytes(manifest_bytes)
            manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
            if final.exists():
                # Publication and cataloging are intentionally two separate
                # durable boundaries.  A worker can die after the atomic
                # rename but before ``artifact_path`` is committed.  Verify
                # the immutable directory directly so that retry adopts it
                # instead of rejecting a perfectly valid publication merely
                # because the catalog pointer is still empty.
                existing = self._verify_calibration_bundle(calibration, final)
                if existing.get("valid"):
                    shutil.rmtree(staging, ignore_errors=True)
                    return self.store.update_calibration(
                        calibration.id,
                        artifact_path=str(final),
                        manifest_hash=str(existing.get("manifest_hash") or manifest_hash),
                    )
                raise VerifierLabError("Existing calibration bundle failed integrity verification")
            os.replace(staging, final)
            return self.store.update_calibration(
                calibration.id,
                artifact_path=str(final),
                manifest_hash=manifest_hash,
            )
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def run_calibration(
        self, calibration_id: str, *, work_item_id: Optional[str] = None
    ) -> VerifierCalibration:
        from .metrics import CalibrationEvidence, compute_calibration_metrics
        from .protocol import (
            grouped_calibration_confirmation_partition,
            iter_calibration_protocol,
        )

        calibration = self.store.get_calibration(calibration_id)
        # Recover the narrow crash window after the verified directory became
        # visible but before either cataloging or the terminal lifecycle update
        # committed.  Adopting here also avoids recomputing metrics (and
        # potentially duplicating catalog rows) before `_publish_bundle` gets a
        # chance to notice the existing final directory.
        recovered_final = self.root / calibration.id
        if recovered_final.is_dir() and calibration.status != "completed":
            recovered = self._verify_calibration_bundle(calibration, recovered_final)
            if recovered.get("valid"):
                return self.store.update_calibration(
                    calibration.id,
                    status="completed",
                    stage="completed",
                    artifact_path=str(recovered_final),
                    manifest_hash=str(recovered.get("manifest_hash") or ""),
                    processed_records=(
                        calibration.total_records
                        if calibration.total_records is not None
                        else calibration.processed_records
                    ),
                    completed_at=calibration.completed_at or _now(),
                    cancel_requested=False,
                    error=None,
                )
        if calibration.status == "completed":
            if self.verify_calibration(calibration.id).get("valid"):
                return calibration
            raise VerifierLabError("Completed calibration bundle is corrupt")
        if calibration.status not in {"queued", "interrupted", "failed"}:
            raise VerifierLabError(f"Calibration is {calibration.status}")
        revision = self.store.get_profile_revision(calibration.verifier_revision_id)
        self._assert_implementation_identity(revision)
        protocol_revision = self.store.get_protocol_revision(calibration.protocol_revision_id)
        protocol = self._protocol_for(
            revision,
            protocol_revision.definition,
            confirmation=bool(calibration.request.get("confirmation")),
        )
        started = _now()
        self.store.update_calibration(
            calibration.id,
            status="running",
            stage="loading_reference",
            started_at=calibration.started_at or started,
            error=None,
            cancel_requested=False,
            **({"work_item_id": work_item_id} if work_item_id else {}),
        )
        try:
            source_meta, source_iter = self._source(
                calibration.source_kind,
                calibration.source_revision_id,
                revision.task_type,
                revision.modality,
            )
            records = list(source_iter)
            self._validate_scalar_reference_contract(revision, records)
            if revision.family == "reward_model" and any(
                bool(value.get("reward_model_training")) for value in records
            ):
                raise ValueError(
                    "Reward-model calibration cannot include its training records"
                )
            by_id = {str(value["record_id"]): value for value in records}
            if bool(calibration.request.get("confirmation")):
                partition = grouped_calibration_confirmation_partition(
                    records,
                    seed=42,
                    confirmation_fraction=protocol.confirmation_fraction,
                )
                if not partition.calibration_record_ids:
                    raise ValueError(
                        "Related records form one group and cannot create a leakage-free partition"
                    )
                partition_value = partition.to_dict()
            else:
                partition_value = {
                    "calibration_record_ids": sorted(by_id),
                    "confirmation_record_ids": [],
                    "group_count": len(by_id),
                    "leakage": {"record_id": [], "content_hash": [], "media_hash": []},
                }
            with self.db._lock:
                self.db._conn.execute(
                    "UPDATE verifier_calibrations SET partition_json=?, updated_at=? WHERE id=?",
                    (
                        json.dumps(partition_value, sort_keys=True, separators=(",", ":")),
                        _now(),
                        calibration.id,
                    ),
                )
                self.db._conn.commit()
            partition_by_id = {
                record_id: "calibration"
                for record_id in partition_value["calibration_record_ids"]
            }
            partition_by_id.update(
                {
                    record_id: "confirmation"
                    for record_id in partition_value["confirmation_record_ids"]
                }
            )
            existing_rows = self.db._conn.execute(
                """SELECT record_id,repeat_index,orientation,probe_kind
                   FROM verifier_calibration_samples WHERE calibration_id=?""",
                (calibration.id,),
            ).fetchall()
            existing = {
                (
                    str(row["record_id"]),
                    int(row["repeat_index"]),
                    str(row["orientation"]),
                    str(row["probe_kind"]),
                )
                for row in existing_rows
            }
            self.store.update_calibration(calibration.id, stage="invoking_verifier")
            completed_records: set[str] = set()
            production_invocations = [
                value
                for value in iter_calibration_protocol(records, protocol)
                if value.perturbation == "production_batch_size"
                and (
                    value.record_id,
                    value.repetition_index,
                    value.orientation,
                    value.perturbation,
                )
                not in existing
            ]
            production_observations: Dict[str, VerifierObservation] = {}
            if production_invocations:
                production_observations = self._reward_model_production_observations(
                    revision,
                    production_invocations,
                    batch_size=int(protocol.production_batch_size or 1),
                    runtime=calibration.runtime_identity,
                )

            with ExitStack() as isolated_processes:
                isolated_by_repeat: Dict[int, Dict[str, Any]] = {}
                isolation_revisions = self._deterministic_isolation_revisions(revision)
                if isolation_revisions:
                    from .process_runner import IsolatedVerifierPass

                    # Open one worker for each deterministic leaf and protocol
                    # pass.  Mixed chains use the stochastic pass count, while
                    # all-deterministic protocols retain their two fresh runs.
                    pending_repeats = (
                        (0, 1)
                        if protocol.deterministic
                        else tuple(range(len(protocol.stochastic_seeds)))
                    )
                    for repetition_index in pending_repeats:
                        isolated_by_repeat[repetition_index] = {}
                        for isolated_revision in isolation_revisions:
                            timeout_seconds = float(
                                isolated_revision.runtime_contract.get(
                                    "timeout_seconds", 300.0
                                )
                            )
                            isolated_by_repeat[repetition_index][
                                isolated_revision.id
                            ] = isolated_processes.enter_context(
                                IsolatedVerifierPass(timeout_seconds=timeout_seconds)
                            )

                for ordinal, invocation in enumerate(
                    iter_calibration_protocol(records, protocol)
                ):
                    current = self.store.get_calibration(calibration.id)
                    if current.cancel_requested:
                        raise CalibrationCancelled("Calibration cancellation requested")
                    key = (
                        invocation.record_id,
                        invocation.repetition_index,
                        invocation.orientation,
                        invocation.perturbation,
                    )
                    if key not in existing:
                        source_record = by_id[invocation.record_id]
                        observation = production_observations.get(invocation.invocation_id)
                        if observation is None:
                            invocation_runtime = {
                                **calibration.runtime_identity,
                                "generation_settings": dict(invocation.generation_settings),
                                "fresh_process_requested": invocation.fresh_process,
                            }
                            isolated_pass = isolated_by_repeat.get(
                                invocation.repetition_index
                            )
                            if (
                                isolated_pass is not None
                                and getattr(self._invoke_revision, "__func__", None)
                                is VerifierLabService._invoke_revision
                            ):
                                observation = self._invoke_revision(
                                    revision,
                                    invocation.payload,
                                    runtime=invocation_runtime,
                                    isolated_pass=isolated_pass,
                                )
                            else:
                                observation = self._invoke_revision(
                                    revision,
                                    invocation.payload,
                                    runtime=invocation_runtime,
                                )
                        observation_details = (
                            observation.details
                            if isinstance(observation.details, Mapping)
                            else {}
                        )
                        seed_honored = observation.runtime_identity.get(
                            "seed_honored",
                            observation_details.get("seed_honored"),
                        )
                        declared_seed_support = revision.definition.get("seed_support")
                        if invocation.seed is not None and seed_honored is None:
                            if declared_seed_support is False or str(
                                declared_seed_support or ""
                            ).strip().lower() in {
                                "ignored",
                                "unsupported",
                                "false",
                                "none",
                            }:
                                seed_honored = False
                        scrubbed_source = scrub_secrets(source_record)
                        self.store.append_sample(
                            calibration.id,
                            ordinal=ordinal,
                            record_id=invocation.record_id,
                            record_hash=str(
                                source_record.get("content_hash")
                                or content_hash(source_record)
                            ),
                            group_id=str(
                                source_record.get("group_id") or invocation.record_id
                            ),
                            partition=partition_by_id.get(
                                invocation.record_id, "calibration"
                            ),
                            repeat_index=invocation.repetition_index,
                            orientation=invocation.orientation,
                            probe_kind=invocation.perturbation,
                            seed=invocation.seed,
                            reference={
                                "expected": source_record.get("expected"),
                                "input": scrubbed_source,
                            },
                            observation=observation,
                            metadata={
                                "invocation_id": invocation.invocation_id,
                                "seed_honored": seed_honored,
                                "fresh_process_requested": invocation.fresh_process,
                                "isolated_pass": invocation.repetition_index,
                                "production_batch_size": invocation.production_batch_size,
                                "subgroup": dict(source_record.get("subgroup") or {}),
                            },
                        )
                        existing.add(key)
                    completed_records.add(invocation.record_id)
                    if ordinal % 25 == 0:
                        self.store.update_calibration(
                            calibration.id,
                            processed_records=len(completed_records),
                            total_records=len(records),
                        )

                self.store.update_calibration(
                    calibration.id,
                    processed_records=len(completed_records),
                    total_records=len(records),
                )

            self.store.update_calibration(calibration.id, stage="computing_metrics")
            for partition_name in ("calibration", "confirmation"):
                # Materialize one statistical partition at a time.  Task-level
                # rank/correlation metrics still require one partition's
                # stable-record arrays, but calibration and confirmation no
                # longer coexist as duplicate Python object graphs.
                selected = list(
                    self._all_samples(calibration.id, partition=partition_name)
                )
                if not selected:
                    continue
                evidence = [
                    CalibrationEvidence(
                        record_id=sample.record_id,
                        expected=sample.reference.get("expected"),
                        predicted=self._prediction(sample.observation, revision.task_type),
                        reward=sample.observation.reward,
                        passed=sample.observation.passed,
                        error=sample.observation.error,
                        error_kind=(
                            sample.observation.details.get("error_kind")
                            or sample.metadata.get("error_kind")
                        ),
                        timeout=bool(
                            sample.observation.details.get("timeout")
                            or sample.metadata.get("timeout")
                            or (
                                sample.observation.error
                                and "timeout" in sample.observation.error.lower()
                            )
                        ),
                        latency_ms=sample.observation.latency_ms,
                        repetition_index=sample.repeat_index,
                        seed=sample.seed,
                        seed_honored=sample.metadata.get("seed_honored"),
                        orientation=sample.orientation,
                        perturbation=sample.probe_kind,
                        probability=(
                            sample.observation.reward
                            if revision.reward_contract.probability_semantics
                            else None
                        ),
                        subgroup=dict(sample.metadata.get("subgroup") or {}),
                        component_trace=tuple(sample.observation.component_trace),
                    )
                    for sample in selected
                ]
                result = compute_calibration_metrics(
                    evidence,
                    task_type=revision.task_type,
                    reward_contract=revision.reward_contract,
                    probability_semantics=revision.reward_contract.probability_semantics,
                    bootstrap_resamples=int(
                        protocol_revision.definition.get("bootstrap_resamples", 10_000)
                    ),
                    bootstrap_seed=42,
                )
                if revision.task_type == "binary":
                    from .qualification import binary_threshold_curve

                    span = (
                        revision.reward_contract.maximum
                        - revision.reward_contract.minimum
                    )
                    thresholds = [
                        revision.reward_contract.minimum + span * index / 20.0
                        for index in range(21)
                    ]
                    result = dict(result)
                    result["threshold_curve"] = binary_threshold_curve(
                        evidence,
                        reward_contract=revision.reward_contract,
                        thresholds=thresholds,
                    )
                if revision.family == "reward_model" and protocol.production_batch_size:
                    result = dict(result)
                    universal = dict(result.get("universal") or {})
                    universal.update(
                        self._reward_model_batch_metrics(
                            selected,
                            dtype=str(protocol.reward_model_dtype),
                        )
                    )
                    result["universal"] = universal
                self._persist_metric_result(
                    calibration.id, result, partition=partition_name
                )
                # Release the completed partition before the next paginated
                # partition is loaded (assignment would otherwise briefly keep
                # both lists alive while evaluating its right-hand side).
                del selected, evidence, result
            self.store.update_calibration(calibration.id, stage="publishing")
            published = self._publish_bundle(calibration.id)
            return self.store.update_calibration(
                published.id,
                status="completed",
                stage="completed",
                processed_records=len(records),
                total_records=len(records),
                completed_at=_now(),
                error=None,
            )
        except CalibrationCancelled:
            self.store.update_calibration(
                calibration.id,
                status="cancelled",
                stage="cancelled",
                completed_at=_now(),
                error="Calibration cancelled by operator",
            )
            raise
        except Exception as exc:
            self.store.update_calibration(
                calibration.id,
                status="failed",
                stage="failed",
                completed_at=_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    def _verify_calibration_bundle(
        self,
        calibration: VerifierCalibration,
        root: Path,
    ) -> Dict[str, Any]:
        """Verify one published bundle without trusting its catalog pointer.

        This is also the recovery verifier used immediately after an atomic
        rename.  Identity fields in the manifest are checked against the
        immutable calibration row before a missing ``artifact_path`` may be
        repaired.
        """

        if not root.is_dir():
            return {
                "calibration_id": calibration.id,
                "valid": False,
                "errors": [f"calibration bundle directory is unavailable: {root}"],
            }
        manifest_path = root / "manifest.json"
        errors: list[str] = []
        checksums: Dict[str, str] = {}
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "calibration_id": calibration.id,
                "valid": False,
                "errors": [f"manifest unavailable: {exc}"],
            }
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        if calibration.manifest_hash and manifest_hash != calibration.manifest_hash:
            errors.append("manifest hash differs from the catalog")
        if manifest.get("format_version") != 1:
            errors.append("unsupported calibration bundle format")
        if str(manifest.get("calibration_id")) != calibration.id:
            errors.append("manifest calibration identity differs")
        try:
            manifest_sample_count = int(manifest.get("sample_count"))
        except (TypeError, ValueError):
            manifest_sample_count = -1
        if manifest_sample_count != int(calibration.sample_count):
            errors.append("manifest sample count differs")
        expected_identity = {
            "verifier_revision_id": calibration.verifier_revision_id,
            "source_hash": calibration.source_hash,
            "protocol_hash": calibration.protocol_hash,
            "qualification_hash": calibration.qualification_hash,
            "runtime_identity_hash": calibration.runtime_identity_hash,
        }
        for field, expected in expected_identity.items():
            if str(manifest.get(field) or "") != str(expected):
                errors.append(f"manifest {field} differs")
        manifest_checksums = manifest.get("checksums")
        if not isinstance(manifest_checksums, Mapping) or not manifest_checksums:
            errors.append("manifest has no bundle checksums")
            manifest_checksums = {}
        required_files = {
            "profile.json",
            "source.json",
            "protocol.json",
            "runtime.json",
            "qualification.json",
            "metrics.json",
            "samples.jsonl",
        }
        missing_checksum_entries = required_files - {
            str(name) for name in manifest_checksums
        }
        for name in sorted(missing_checksum_entries):
            errors.append(f"manifest is missing checksum: {name}")
        for name, expected in dict(manifest_checksums).items():
            relative = Path(str(name))
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"invalid bundle file path: {name}")
                continue
            path = root / relative
            if not path.is_file():
                errors.append(f"missing bundle file: {name}")
                continue
            actual = _file_sha256(path)
            checksums[str(name)] = actual
            if actual != expected:
                errors.append(f"checksum mismatch: {name}")
        return {
            "calibration_id": calibration.id,
            "valid": not errors,
            "manifest_hash": manifest_hash,
            "checksums": checksums,
            "errors": errors,
        }

    def verify_calibration(self, calibration_id: str) -> Dict[str, Any]:
        calibration = self.store.get_calibration(calibration_id)
        if not calibration.artifact_path:
            return {
                "calibration_id": calibration_id,
                "valid": False,
                "errors": ["calibration has no published bundle"],
            }
        result = self._verify_calibration_bundle(
            calibration,
            Path(calibration.artifact_path),
        )
        decision_checks: list[Dict[str, Any]] = []
        decision_errors: list[str] = []
        for decision in self.store.list_decisions(calibration.id):
            artifact = decision.evidence.get("decision_artifact")
            if not isinstance(artifact, Mapping):
                decision_errors.append(
                    f"qualification decision {decision.id} has no integrity artifact"
                )
                continue
            path = Path(str(artifact.get("path") or ""))
            expected = str(artifact.get("sha256") or "")
            actual = _file_sha256(path) if path.is_file() else None
            valid = bool(actual and expected and actual == expected)
            decision_checks.append(
                {
                    "decision_id": decision.id,
                    "path": str(path),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "valid": valid,
                }
            )
            if not valid:
                decision_errors.append(
                    f"qualification decision artifact failed integrity: {decision.id}"
                )
        result["qualification_decisions"] = decision_checks
        if decision_errors:
            result["errors"] = [*list(result.get("errors") or []), *decision_errors]
            result["valid"] = False
        return result

    def cancel_calibration(self, calibration_id: str) -> VerifierCalibration:
        calibration = self.store.get_calibration(calibration_id)
        if calibration.status in {"completed", "cancelled", "failed"}:
            return calibration
        updated = self.store.update_calibration(
            calibration.id, cancel_requested=True, stage="cancelling"
        )
        if updated.work_item_id:
            self.scheduler.cancel(updated.work_item_id)
            return self.store.get_calibration(updated.id)
        return updated

    def prepare_retry(self, calibration_id: str) -> VerifierCalibration:
        calibration = self.store.get_calibration(calibration_id)
        if calibration.status == "completed":
            raise VerifierLabError("Completed immutable calibrations cannot be retried")
        return self.store.update_calibration(
            calibration.id,
            status="queued",
            stage="resume_pending",
            cancel_requested=False,
            retry_count=calibration.retry_count + 1,
            error=None,
            completed_at=None,
        )

    def retry_calibration(self, calibration_id: str) -> VerifierCalibration:
        calibration = self.prepare_retry(calibration_id)
        if calibration.work_item_id:
            item = self.scheduler.retry(
                calibration.work_item_id,
                reason="operator requested verifier calibration retry",
                force=True,
                sync_domain=False,
            )
            if item is None:
                raise VerifierLabError("Calibration work item could not be retried")
        return self.store.get_calibration(calibration.id)

    def wait_for_calibration(
        self, calibration_id: str, *, timeout_seconds: float = 3600.0
    ) -> VerifierCalibration:
        deadline = time.monotonic() + max(0.1, float(timeout_seconds))
        while time.monotonic() < deadline:
            value = self.store.get_calibration(calibration_id)
            if value.status in {"completed", "failed", "cancelled"}:
                return value
            time.sleep(0.1)
        raise TimeoutError(f"Timed out waiting for verifier calibration {calibration_id}")

    def list_calibration_samples(
        self,
        calibration_id: str,
        *,
        partition: Optional[str] = None,
        outcome: Optional[str] = None,
        perturbation: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        values, total, page_limit, page_offset = self.store.query_samples(
            calibration_id,
            partition=partition,
            outcome=outcome,
            perturbation=perturbation,
            query=query,
            limit=limit,
            offset=offset,
        )
        return {
            "items": [value.to_dict() for value in values],
            "total": total,
            "limit": page_limit,
            "offset": page_offset,
        }

    @staticmethod
    def _nested_metrics(metrics: Sequence[VerifierCalibrationMetric]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "universal": {},
            "task": {},
            "probability": {},
            "primary_metric": {},
            "subgroups": {},
        }
        for metric in metrics:
            if metric.subgroup:
                payload = metric.metadata.get("payload")
                if isinstance(payload, Mapping):
                    result["subgroups"][metric.subgroup] = dict(payload)
                elif metric.available:
                    result["subgroups"].setdefault(
                        metric.subgroup,
                        {"available": True, "record_count": metric.record_count, "metrics": {}},
                    )["metrics"][metric.name] = metric.value
                continue
            section = str(metric.metadata.get("section") or "task")
            structured = metric.metadata.get("structured")
            if isinstance(structured, Mapping):
                result.setdefault(section, {}).update(dict(structured))
            if not metric.available:
                continue
            name = metric.name.split(".", 1)[-1]
            result.setdefault(section, {})[name] = metric.value
            for diagnostic_name in (
                "confusion_matrix",
                "per_class",
                "per_label",
                "ece_bins",
                "missing_evidence_records",
                "threshold_curve",
            ):
                if diagnostic_name in metric.metadata:
                    result.setdefault(section, {})[diagnostic_name] = metric.metadata[
                        diagnostic_name
                    ]
            if metric.metadata.get("primary"):
                result["primary_metric"] = {
                    "name": name,
                    "value": metric.value,
                    "direction": metric.direction,
                }
        return result

    def qualify_calibration(
        self,
        calibration_id: str,
        *,
        scope: str,
        override_note: Optional[str] = None,
    ) -> VerifierQualificationDecision:
        from .qualification import qualify_calibration

        calibration = self.store.get_calibration(calibration_id)
        if calibration.status != "completed":
            raise VerifierLabError("Only completed calibrations can be qualified")
        if not self.verify_calibration(calibration.id).get("valid"):
            raise VerifierLabError("Calibration integrity verification failed")
        revision = self.store.get_profile_revision(calibration.verifier_revision_id)
        policy = self.store.get_qualification_profile_revision(
            calibration.qualification_profile_revision_id
        )
        requested_scope = str(scope).strip().lower()
        metric_partition = "confirmation" if requested_scope == "confirmation" else "calibration"
        metrics = [
            value
            for value in self.store.list_metrics(calibration.id)
            if value.partition == metric_partition
        ]
        if not metrics:
            raise VerifierLabError(f"No {metric_partition} evidence is available")
        nested = self._nested_metrics(metrics)
        result = qualify_calibration(
            nested,
            task_type=revision.task_type,
            template=policy.template_kind,
            scope=requested_scope,
            required_classes=tuple(policy.requirements.get("required_classes") or ()),
            requirements=policy.requirements,
        )
        runtime = self.runtime_compatibility(revision.id)
        decision = result.decision
        reasons = list(result.reasons)
        if not revision.qualifiable:
            decision = "fail"
            blockers = list(revision.definition.get("qualification_blockers") or [])
            reasons.append(
                "Verifier implementation is not fingerprintable and cannot normally qualify"
                + (f": {', '.join(map(str, blockers))}" if blockers else "")
            )
        for warning in result.warnings:
            if warning not in reasons:
                reasons.append(str(warning))
        if runtime["state"] != "compatible":
            decision = "fail"
            reasons.append("Verifier runtime contract is stale")
        override = bool(override_note)
        if override:
            reasons.append("Operator override recorded; guided promotion remains excluded")
        previous = self.store.list_decisions(calibration.id)
        supersedes = next(
            (value.id for value in reversed(previous) if value.scope == requested_scope), None
        )
        decision_id = f"verifier-decision-{uuid.uuid4().hex}"
        decision_evidence = {**result.evidence, "warnings": list(result.warnings)}
        decision_payload = {
            "format_version": 1,
            "id": decision_id,
            "calibration_id": calibration.id,
            "verifier_revision_id": revision.id,
            "qualification_profile_revision_id": policy.id,
            "scope": requested_scope,
            "decision": decision,
            "reasons": reasons or [f"{decision} under {policy.template_kind}"],
            "runtime_state": str(runtime["state"]),
            "override": override,
            "override_note": override_note,
            "supersedes_decision_id": supersedes,
            "evidence": decision_evidence,
            "created_at": _now(),
        }
        decision_bytes = _canonical_bytes(decision_payload)
        decision_hash = hashlib.sha256(decision_bytes).hexdigest()
        decision_root = self.root / "qualification-decisions" / calibration.id
        decision_root.mkdir(parents=True, exist_ok=True)
        decision_path = decision_root / f"{decision_id}.json"
        decision_staging = decision_root / f".{decision_id}-{uuid.uuid4().hex}.tmp"
        decision_staging.write_bytes(decision_bytes)
        if _file_sha256(decision_staging) != decision_hash:
            decision_staging.unlink(missing_ok=True)
            raise VerifierLabError("Qualification decision checksum verification failed")
        os.replace(decision_staging, decision_path)
        return self.store.append_decision(
            calibration.id,
            scope=requested_scope,
            decision=decision,
            reasons=decision_payload["reasons"],
            evidence={
                **decision_evidence,
                "decision_artifact": {
                    "path": str(decision_path),
                    "sha256": decision_hash,
                    "format_version": 1,
                },
            },
            runtime_state=str(runtime["state"]),
            override=override,
            override_note=override_note,
            supersedes_decision_id=supersedes,
            decision_id=decision_id,
        )

    def _revision_decisions(self, revision_id: str) -> list[VerifierQualificationDecision]:
        calibrations = self.store.list_calibrations(
            verifier_revision_id=revision_id, status="completed", limit=1000
        )
        decisions: list[VerifierQualificationDecision] = []
        for calibration in reversed(calibrations):
            decisions.extend(self.store.list_decisions(calibration.id))
        return decisions

    def promote_revision(
        self,
        revision_id: str,
        *,
        alias: str,
        override_note: Optional[str] = None,
    ) -> Dict[str, Any]:
        from .qualification import promotion_eligibility

        revision = self.store.get_profile_revision(revision_id)
        try:
            self._assert_implementation_identity(revision)
        except Exception:
            if not override_note:
                raise
        if not revision.qualifiable and not override_note:
            raise ValueError(
                "Unpinned or unfingerprintable verifier revisions require a promotion override note"
            )
        decisions = self._revision_decisions(revision.id)
        latest: Dict[str, VerifierQualificationDecision] = {}
        for decision in decisions:
            latest[decision.scope] = decision
        eligibility = promotion_eligibility(
            [value.to_dict() for value in latest.values()],
            # V7's approved alias is stronger than candidate: confirmation
            # evidence is mandatory, not merely required when an operator
            # happened to request a partition on the first calibration.
            confirmation_required=True,
        )
        requested = str(alias).strip().lower()
        required_scopes = (
            ("development", "operational", "confirmation")
            if requested == "approved"
            else ("development", "operational")
        )
        integrity_issues = [
            scope
            for scope in required_scopes
            if scope in latest
            and not self.verify_calibration(latest[scope].calibration_id).get("valid")
        ]
        if integrity_issues:
            eligibility = {
                **eligibility,
                requested: False,
                "integrity_issues": [
                    f"{scope}:corrupt_evidence" for scope in integrity_issues
                ],
            }
        allowed = bool(eligibility.get(requested)) and not integrity_issues
        if not allowed and not override_note:
            missing = eligibility.get(f"{requested}_missing_scopes") or []
            if integrity_issues:
                missing = [*missing, *[f"{scope}:corrupt_evidence" for scope in integrity_issues]]
            raise ValueError(
                f"Verifier revision cannot become {requested}; missing passing scopes: "
                + ", ".join(missing)
            )
        anchor_scope = "confirmation" if requested == "approved" else "operational"
        decision = latest.get(anchor_scope)
        if decision is not None and (
            decision.decision != "pass" or decision.override
        ):
            decision = None
        promoted = self.store.promote_alias(
            revision.id,
            alias=requested,
            qualification_decision_id=decision.id if decision and allowed else None,
            override=not allowed,
            note=override_note,
        )
        return {
            "alias": promoted.to_dict(),
            "eligibility": eligibility,
            "override": not allowed,
            "history": [
                value.to_dict()
                for value in self.store.list_alias_history(revision.profile_id, alias=requested)
            ],
        }

    def runtime_compatibility(
        self, revision_id: str, actual: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        from .fingerprints import runtime_identity_for_contract
        from .qualification import runtime_compatibility

        revision = self.store.get_profile_revision(revision_id)
        actual_identity = runtime_identity_for_contract(
            revision.runtime_contract, actual
        )
        result = runtime_compatibility(revision.runtime_contract, actual_identity)
        try:
            self._assert_implementation_identity(revision, rehash_artifact=False)
        except Exception as exc:
            mismatches = list(result.get("mismatches") or [])
            mismatches.append(
                {
                    "field": "implementation_identity",
                    "expected": revision.implementation_fingerprint,
                    "actual": None,
                    "reason": str(exc),
                }
            )
            result = {
                **result,
                "status": "stale_runtime",
                "compatible": False,
                "mismatches": mismatches,
            }
        return {
            "verifier_revision_id": revision_id,
            **result,
            "state": str(result.get("status") or "stale_runtime"),
        }

    def compare_calibrations(
        self,
        base_id: str,
        candidate_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        from .comparison import compare_calibrations, compare_joined_calibration_samples

        base = self.store.get_calibration(base_id)
        candidate = self.store.get_calibration(candidate_id)
        left_revision = self.store.get_profile_revision(base.verifier_revision_id)
        right_revision = self.store.get_profile_revision(candidate.verifier_revision_id)
        left_decisions = self.store.list_decisions(base.id)
        right_decisions = self.store.list_decisions(candidate.id)
        left = {
            **base.to_dict(),
            "task_type": left_revision.task_type,
            "reward_contract_hash": content_hash(
                left_revision.reward_contract.to_dict()
            ),
            "metrics": [value.to_dict() for value in self.store.list_metrics(base.id)],
            "decisions": [value.to_dict() for value in left_decisions],
        }
        right = {
            **candidate.to_dict(),
            "task_type": right_revision.task_type,
            "reward_contract_hash": content_hash(
                right_revision.reward_contract.to_dict()
            ),
            "metrics": [value.to_dict() for value in self.store.list_metrics(candidate.id)],
            "decisions": [value.to_dict() for value in right_decisions],
        }
        comparison = compare_calibrations(left, right)
        joined, total, page_limit, page_offset = self.store.compare_sample_page(
            base.id,
            candidate.id,
            offset=offset,
            limit=limit,
        )
        samples = compare_joined_calibration_samples(
            joined,
            total=total,
            offset=page_offset,
            limit=page_limit,
            task_type=left_revision.task_type,
            reward_direction=left_revision.reward_contract.direction,
        )
        return {**comparison.to_dict(), "samples": samples}

    def list_usage(
        self, revision_id: str, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        self.store.get_profile_revision(revision_id)
        values = self.store.list_bindings(
            revision_id=revision_id, limit=limit, offset=offset
        )
        total = int(
            self.db._conn.execute(
                "SELECT COUNT(*) AS value FROM verifier_bindings WHERE verifier_revision_id=?",
                (revision_id,),
            ).fetchone()["value"]
        )
        return _page(values, total=total, limit=limit, offset=offset)

    def list_qualification_decisions(
        self,
        *,
        verifier_revision_id: Optional[str] = None,
        calibration_id: Optional[str] = None,
        decision: Optional[str] = None,
        scope: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        values, total, page_limit, page_offset = self.store.query_decisions(
            verifier_revision_id=verifier_revision_id,
            calibration_id=calibration_id,
            decision=decision,
            scope=scope,
            limit=limit,
            offset=offset,
        )
        return _page(values, total=total, limit=page_limit, offset=page_offset)

    def list_alias_history(
        self,
        profile_id: str,
        *,
        alias: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        self.store.get_profile(profile_id)
        values = list(reversed(self.store.list_alias_history(profile_id, alias=alias)))
        page_limit = max(1, min(1000, int(limit)))
        page_offset = max(0, int(offset))
        return _page(
            values[page_offset : page_offset + page_limit],
            total=len(values),
            limit=page_limit,
            offset=page_offset,
        )

    def resolve_binding(
        self,
        revision_id: str,
        *,
        modality: Optional[str] = None,
        task_type: Optional[str] = None,
        require_qualified: bool = False,
        include_overridden: bool = False,
    ) -> Dict[str, Any]:
        revision = self.store.get_profile_revision(revision_id)
        if modality and revision.modality not in {modality, "any", "multimodal"}:
            raise ValueError(
                f"Verifier revision modality {revision.modality!r} is incompatible with {modality!r}"
            )
        if task_type and revision.task_type != task_type:
            raise ValueError(
                f"Verifier task {revision.task_type!r} is incompatible with {task_type!r}"
            )
        aliases: list[str] = []
        overridden_aliases: list[str] = []
        for alias in ("candidate", "approved"):
            event = self._active_alias_event(revision.profile_id, alias, revision.id)
            if event is None:
                continue
            if event.override:
                overridden_aliases.append(alias)
            else:
                aliases.append(alias)
        runtime = self.runtime_compatibility(revision.id)
        if require_qualified and (
            not revision.qualifiable
            or runtime["state"] != "compatible"
            or (not aliases and not (include_overridden and overridden_aliases))
        ):
            raise ValueError("Guided use requires a candidate- or approved-qualified verifier")
        decisions = self._revision_decisions(revision.id)
        selected_alias = "approved" if "approved" in aliases else (
            "candidate" if "candidate" in aliases else None
        )
        alias_event = (
            self._active_alias_event(revision.profile_id, selected_alias, revision.id)
            if selected_alias
            else None
        )
        decision = (
            next(
                (
                    value
                    for value in decisions
                    if value.id == alias_event.qualification_decision_id
                ),
                None,
            )
            if alias_event is not None and alias_event.qualification_decision_id
            else None
        )
        return {
            "verifier_profile_revision_id": revision.id,
            "revision_hash": revision.content_hash,
            "family": revision.family,
            "implementation_ref": revision.implementation_ref,
            "implementation_fingerprint": revision.implementation_fingerprint,
            "reliability_adapter": {
                "id": revision.reliability_adapter_id,
                "version": revision.reliability_adapter_version,
            },
            "sanitized_configuration_hash": revision.sanitized_configuration_hash,
            "reward_contract": revision.reward_contract.to_dict(),
            "qualification_scope": {
                "aliases": aliases,
                "overridden_aliases": overridden_aliases,
                "decision_id": decision.id if decision else None,
                "qualified": bool(aliases),
            },
            "runtime_compatibility": runtime,
            "legacy_unqualified": False,
            "guided_eligible": bool(aliases) and runtime["state"] == "compatible",
            "override_warning": (
                "Verifier promotion was overridden; enable overridden revisions explicitly."
                if overridden_aliases and not aliases
                else None
            ),
        }

    def bind_revision(
        self,
        revision_id: str,
        *,
        domain_kind: str,
        domain_id: str,
        role: str = "verifier",
        context: Optional[Mapping[str, Any]] = None,
    ) -> ResolvedVerifierBinding:
        self._assert_implementation_identity(
            self.store.get_profile_revision(revision_id)
        )
        resolved = self.resolve_binding(revision_id)
        decision_id = resolved["qualification_scope"].get("decision_id")
        calibration_sources = self.store.list_calibrations(
            verifier_revision_id=revision_id, status="completed", limit=1000
        )
        development_exposed = any(
            value.source_purpose in {"development", "unspecified"}
            for value in calibration_sources
        )
        binding = self.store.bind_revision(
            revision_id,
            domain_kind=domain_kind,
            domain_id=domain_id,
            role=role,
            qualification_decision_id=decision_id,
            development_exposed=development_exposed,
            context={**dict(context or {}), "resolved": resolved},
        )
        if development_exposed:
            for calibration in calibration_sources:
                target = {
                    "dataset_version_id": domain_id if domain_kind == "dataset_version" else None,
                    "run_id": domain_id if domain_kind == "run" else None,
                    "model_artifact_id": domain_id if domain_kind == "model_artifact" else None,
                }
                if not any(target.values()):
                    continue
                exposures: list[tuple[str, str]] = []
                if calibration.source_kind == "benchmark_suite":
                    suite_revision = self.db.get_benchmark_suite_revision(
                        calibration.source_revision_id
                    )
                    if suite_revision is not None:
                        exposures.extend(
                            (
                                suite_revision.id,
                                str(
                                    (item or {}).get("id")
                                    if isinstance(item, Mapping)
                                    else ordinal
                                ),
                            )
                            for ordinal, item in enumerate(suite_revision.items)
                        )
                elif calibration.source_kind == "label_set":
                    from halo_forge.review_lab import ReviewLabService

                    label_revision = ReviewLabService(self.db).get_label_set_revision(
                        calibration.source_revision_id
                    )
                    if label_revision is not None:
                        exposure_path = Path(label_revision.storage_path) / str(
                            label_revision.manifest.get("exposure_file")
                            or "exposure.json"
                        )
                        if exposure_path.is_file():
                            payload = json.loads(exposure_path.read_text(encoding="utf-8"))
                            exposures.extend(
                                (
                                    str(value["suite_revision_id"]),
                                    str(value["suite_item_id"]),
                                )
                                for value in payload
                                if isinstance(value, Mapping)
                                and value.get("suite_revision_id")
                                and value.get("suite_item_id")
                            )
                for suite_revision_id, suite_item_id in exposures:
                    try:
                        self.db.record_exposure(
                            suite_revision_id=suite_revision_id,
                            suite_item_id=suite_item_id,
                            exposure_type="verifier_calibration_inherited",
                            provenance={
                                "calibration_id": calibration.id,
                                "verifier_revision_id": revision_id,
                                "binding_id": binding.id,
                            },
                            **target,
                        )
                    except Exception:
                        # The binding remains authoritative when a legacy
                        # target has no exposure-ledger-compatible identity.
                        continue
        return binding


__all__ = [
    "CalibrationCancelled",
    "VerifierLabError",
    "VerifierLabService",
]
