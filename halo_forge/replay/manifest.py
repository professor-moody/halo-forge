"""Replay manifest capture + load + diff (Track T15)."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)


MANIFEST_VERSION = 14
MANIFEST_FILENAME = "replay.json"


# ---------- environment fingerprint ----------------------------------------


_TRACKED_PACKAGES = (
    "halo_forge",
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "mlx",
    "mlx_lm",
    "datasets",
    "bitsandbytes",
    "vllm",
    "numpy",
)


@dataclass
class EnvironmentFingerprint:
    """Snapshot of the host + package state at run time.

    Compared at replay time to surface divergence (different torch,
    different MLX, different OS) without refusing the replay outright.
    """

    python: str
    platform: str
    backend: str
    chip_name: Optional[str] = None
    chip_brand: Optional[str] = None
    macos_version: Optional[str] = None
    metal_version: Optional[str] = None
    packages: Dict[str, Optional[str]] = field(default_factory=dict)

    @classmethod
    def capture(cls) -> "EnvironmentFingerprint":
        apple = _capture_apple_environment()
        return cls(
            python=sys.version.split()[0],
            platform=f"{platform.system()} {platform.machine()}",
            backend=_detect_backend_name_safe(),
            chip_name=apple.get("chip_name"),
            chip_brand=apple.get("chip_brand"),
            macos_version=apple.get("macos_version"),
            metal_version=apple.get("metal_version"),
            packages={name: _get_pkg_version(name) for name in _TRACKED_PACKAGES},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _detect_backend_name_safe() -> str:
    try:
        from halo_forge.backend import get_backend

        return get_backend().name
    except Exception:
        return "unknown"


def _capture_apple_environment() -> Dict[str, Optional[str]]:
    if sys.platform != "darwin":
        return {}
    data: Dict[str, Optional[str]] = {
        "macos_version": platform.mac_ver()[0] or None,
    }
    try:
        from halo_forge.telemetry.apple_silicon import AppleSiliconTelemetry
        from halo_forge.utils.apple_chip import detect_metal_version, parse_chip_brand

        chip_name = AppleSiliconTelemetry._detect_device_name()
        chip = parse_chip_brand(chip_name)
        data.update(
            {
                "chip_name": chip_name,
                "chip_brand": chip.brand if chip else None,
                "metal_version": detect_metal_version(),
            }
        )
    except Exception:
        data.setdefault("chip_name", None)
        data.setdefault("chip_brand", None)
        data.setdefault("metal_version", None)
    return data


def _get_pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(name)
    except Exception:
        return None


# ---------- dataset hashing ------------------------------------------------


def hash_dataset_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 over a local dataset file. Returns the hex digest.

    Streams the file so multi-GB JSONLs don't load into memory. The
    digest is stable across hosts; a hash mismatch at replay time
    means the file changed (rebuilt, mistakenly overwritten, swapped).
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------- manifest -------------------------------------------------------


@dataclass
class ReplayManifest:
    """Everything we need to re-launch a run identically."""

    manifest_version: int
    run_id: str
    modality: str
    timestamp: str
    model_name: str
    seed: int
    pythonhashseed: Optional[str]
    config: Dict[str, Any]
    dataset: Dict[str, Any]
    environment: Dict[str, Any]
    cli_args: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    # V3 pins the complete verifier reliability identity used by data,
    # evaluation, or training.  V1/V2 manifests load with an empty mapping so
    # historical runs remain readable without pretending they were qualified.
    verifier: Dict[str, Any] = field(default_factory=dict)
    # V4 pins the complete training-time reward system and integrity evidence.
    # Older manifests load with an empty mapping and remain readable without
    # pretending that an aggregate reward history was a captured signal trace.
    reward_integrity: Dict[str, Any] = field(default_factory=dict)
    # V5 pins corpus extraction, immutable corpus/version identity, exact
    # tokenizer + packing identities, budget semantics, and adaptation mode.
    # V1-V4 manifests load with an empty mapping.
    corpus_training: Dict[str, Any] = field(default_factory=dict)
    # V6 pins proof-run outcome evidence and the reviewed full-run decision.
    training_outcome: Dict[str, Any] = field(default_factory=dict)
    # V7 pins an adaptation study protocol, arm, assignment, and deviations.
    adaptation_study: Dict[str, Any] = field(default_factory=dict)
    # V8 pins specialized task, label-schema, model-head, and retrieval identity.
    specialized_task: Dict[str, Any] = field(default_factory=dict)
    # V9 pins deterministic environment, suite, snapshot, and trajectory identity.
    agent_environment: Dict[str, Any] = field(default_factory=dict)
    # V10 records the durable guided-operation graph used to prepare outcome
    # evidence, launch studies, verify grounding, and execute environment
    # subjects. Older manifests remain readable with an empty mapping.
    operational_completion: Dict[str, Any] = field(default_factory=dict)
    # V11 pins reviewed, non-destructive Dataset Repair identity together with
    # the workstation and distribution capability seen at launch.
    product_completion: Dict[str, Any] = field(default_factory=dict)
    # V12 pins the immutable guided training plan, exact prepared model,
    # disposable capacity evidence, selected safe adjustment, and decision.
    training_plan: Dict[str, Any] = field(default_factory=dict)
    # V13 pins the managed runtime image/revision, host and device identity,
    # qualification bundle, and conservative accelerator-occupancy decision.
    managed_runtime: Dict[str, Any] = field(default_factory=dict)
    # V14 distinguishes generic runtime readiness from a real, fixture-backed
    # Halo Forge trainer path and the workstation beta evidence. Older replay
    # files remain readable without inferring certification from tensor tests.
    training_path: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def capture_manifest(
    *,
    run_id: str,
    modality: str,
    model_name: str,
    seed: int,
    config: Any,
    dataset_file: Optional[Path] = None,
    dataset_id: Optional[str] = None,
    dataset_revision: Optional[str] = None,
    dataset_version: Optional[Dict[str, Any]] = None,
    dataset_bindings: Optional[List[Dict[str, Any]]] = None,
    training_artifact: Optional[Dict[str, Any]] = None,
    verifier_binding: Optional[Dict[str, Any]] = None,
    reward_integrity_binding: Optional[Dict[str, Any]] = None,
    corpus_training_binding: Optional[Dict[str, Any]] = None,
    training_outcome_binding: Optional[Dict[str, Any]] = None,
    adaptation_study_binding: Optional[Dict[str, Any]] = None,
    specialized_task_binding: Optional[Dict[str, Any]] = None,
    agent_environment_binding: Optional[Dict[str, Any]] = None,
    operational_completion_binding: Optional[Dict[str, Any]] = None,
    product_completion_binding: Optional[Dict[str, Any]] = None,
    training_plan_binding: Optional[Dict[str, Any]] = None,
    managed_runtime_binding: Optional[Dict[str, Any]] = None,
    training_path_binding: Optional[Dict[str, Any]] = None,
    cli_args: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> ReplayManifest:
    """Build a `ReplayManifest` for the active run."""
    import os

    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        config_dict = dataclasses.asdict(config)
    elif isinstance(config, dict):
        config_dict = dict(config)
    else:
        config_dict = {"value": str(config)}

    dataset: Dict[str, Any] = {}
    if dataset_version is not None and dataset_bindings is not None:
        raise ValueError("dataset_version and dataset_bindings are mutually exclusive")
    if dataset_bindings is not None:
        normalized_bindings: List[Dict[str, Any]] = []
        for raw_binding in dataset_bindings:
            binding = dict(raw_binding)
            role = str(binding.get("role") or "").strip().lower()
            version_id = str(
                binding.get("dataset_version_id") or binding.get("version_id") or ""
            ).strip()
            split = str(binding.get("split") or "").strip()
            if not role or not version_id or not split:
                raise ValueError("dataset_bindings require role, dataset_version_id, and split")
            binding["role"] = role
            binding["dataset_version_id"] = version_id
            binding.pop("version_id", None)
            normalized_bindings.append(binding)
        if not normalized_bindings:
            raise ValueError("dataset_bindings cannot be empty")
        dataset = {
            "kind": "managed_versions",
            "bindings": normalized_bindings,
            "training_artifact": dict(training_artifact or {}),
        }
    elif dataset_version is not None:
        version_payload = dict(dataset_version)
        version_id = str(
            version_payload.get("version_id") or version_payload.get("id") or ""
        ).strip()
        if not version_id:
            raise ValueError("dataset_version requires version_id or id")
        dataset = {
            "kind": "managed_version",
            **version_payload,
            "version_id": version_id,
        }
        dataset.pop("id", None)
    elif dataset_file is not None:
        path = Path(dataset_file)
        if path.exists() and path.is_file():
            dataset = {
                "kind": "local_file",
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": hash_dataset_file(path),
            }
        else:
            dataset = {"kind": "local_file", "path": str(path), "missing": True}
    elif dataset_id:
        dataset = {
            "kind": "huggingface",
            "id": dataset_id,
            "revision": dataset_revision,
        }

    verifier: Dict[str, Any] = {}
    if verifier_binding is not None:
        raw_verifier = dict(verifier_binding)
        revision_id = str(
            raw_verifier.get("verifier_profile_revision_id")
            or raw_verifier.get("profile_revision_id")
            or ""
        ).strip()
        legacy = bool(raw_verifier.get("legacy_unqualified", not revision_id))
        verifier = {
            "verifier_profile_revision_id": revision_id or None,
            "revision_hash": raw_verifier.get("revision_hash"),
            "family": raw_verifier.get("family"),
            "implementation_ref": raw_verifier.get("implementation_ref"),
            "reliability_adapter": dict(raw_verifier.get("reliability_adapter") or {}),
            "implementation_fingerprint": raw_verifier.get("implementation_fingerprint"),
            "sanitized_configuration_hash": raw_verifier.get("sanitized_configuration_hash"),
            "reward_contract": dict(raw_verifier.get("reward_contract") or {}),
            "qualification_scope": dict(raw_verifier.get("qualification_scope") or {}),
            "runtime_compatibility": dict(raw_verifier.get("runtime_compatibility") or {}),
            "legacy_unqualified": legacy,
            "legacy_warning": (
                raw_verifier.get("legacy_warning")
                or (
                    "This run used a raw verifier configuration without an immutable "
                    "reliability qualification."
                    if legacy
                    else None
                )
            ),
        }

    reward_integrity: Dict[str, Any] = {}
    if reward_integrity_binding is not None:
        raw_reward = dict(reward_integrity_binding)
        auditors = [dict(value) for value in raw_reward.get("auditors") or []]
        reward_integrity = {
            "reward_system_revision_id": raw_reward.get("reward_system_revision_id"),
            "reward_system_hash": raw_reward.get("reward_system_hash"),
            "optimizer_verifier_revision_id": raw_reward.get(
                "optimizer_verifier_revision_id"
            ),
            "auditors": auditors,
            "reward_mapping_hash": raw_reward.get("reward_mapping_hash"),
            "protocol_revision_id": raw_reward.get("protocol_revision_id"),
            "protocol_hash": raw_reward.get("protocol_hash"),
            "integrity_profile_revision_id": raw_reward.get(
                "integrity_profile_revision_id"
            ),
            "integrity_profile_hash": raw_reward.get("integrity_profile_hash"),
            "boundaries": list(raw_reward.get("boundaries") or []),
            "signal_capability": dict(raw_reward.get("signal_capability") or {}),
            "trace_manifests": [
                dict(value) for value in raw_reward.get("trace_manifests") or []
            ],
            "audit_decisions": [
                dict(value) for value in raw_reward.get("audit_decisions") or []
            ],
            "runtime_compatibility": dict(
                raw_reward.get("runtime_compatibility") or {}
            ),
            "legacy_unmonitored": bool(raw_reward.get("legacy_unmonitored", False)),
        }

    corpus_training: Dict[str, Any] = {}
    if corpus_training_binding is not None:
        raw_corpus = dict(corpus_training_binding)
        adaptation = str(raw_corpus.get("adaptation") or "").strip().lower()
        if adaptation not in {"lora", "full"}:
            raise ValueError(
                "corpus_training adaptation must be explicitly set to 'lora' or 'full'"
            )
        budget_mode = str(raw_corpus.get("budget_mode") or "").strip().lower()
        if budget_mode not in {"tokens", "passes"}:
            raise ValueError("corpus_training budget_mode must be 'tokens' or 'passes'")
        target_tokens = raw_corpus.get("target_tokens")
        corpus_passes = raw_corpus.get("corpus_passes")
        if budget_mode == "tokens":
            if target_tokens is None or int(target_tokens) <= 0:
                raise ValueError(
                    "corpus_training target_tokens is required for token budgets"
                )
            target_tokens = int(target_tokens)
            corpus_passes = None
        else:
            if corpus_passes is None or float(corpus_passes) <= 0:
                raise ValueError(
                    "corpus_training corpus_passes is required for pass budgets"
                )
            corpus_passes = float(corpus_passes)
            target_tokens = None
        tokenizer_hash = str(raw_corpus.get("tokenizer_hash") or "").strip()
        packing_plan_hash = str(raw_corpus.get("packing_plan_hash") or "").strip()
        if not tokenizer_hash or not packing_plan_hash:
            raise ValueError(
                "corpus_training requires tokenizer_hash and packing_plan_hash"
            )
        corpus_training = {
            "extraction_identity": copy.deepcopy(
                raw_corpus.get("extraction_identity") or {}
            ),
            "corpus_identity": copy.deepcopy(
                raw_corpus.get("corpus_identity")
                or raw_corpus.get("dataset_identity")
                or {}
            ),
            "corpus_version": raw_corpus.get("corpus_version")
            or raw_corpus.get("dataset_version_id")
            or raw_corpus.get("version_id"),
            "tokenizer_identity": copy.deepcopy(
                raw_corpus.get("tokenizer_identity") or {}
            ),
            "tokenizer_hash": tokenizer_hash,
            "packing_plan": copy.deepcopy(raw_corpus.get("packing_plan") or {}),
            "packing_plan_hash": packing_plan_hash,
            "budget_mode": budget_mode,
            "target_tokens": target_tokens,
            "corpus_passes": corpus_passes,
            "adaptation": adaptation,
            "objective": "causal_next_token",
            "training_artifact": copy.deepcopy(
                raw_corpus.get("training_artifact")
                or raw_corpus.get("training_artifact_identity")
                or training_artifact
                or {}
            ),
        }

    return ReplayManifest(
        manifest_version=MANIFEST_VERSION,
        run_id=run_id,
        modality=modality,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z") or "",
        model_name=model_name,
        seed=int(seed),
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        config=config_dict,
        dataset=dataset,
        environment=EnvironmentFingerprint.capture().to_dict(),
        cli_args=list(cli_args or []),
        notes=notes,
        verifier=verifier,
        reward_integrity=reward_integrity,
        corpus_training=corpus_training,
        training_outcome=copy.deepcopy(training_outcome_binding or {}),
        adaptation_study=copy.deepcopy(adaptation_study_binding or {}),
        specialized_task=copy.deepcopy(specialized_task_binding or {}),
        agent_environment=copy.deepcopy(agent_environment_binding or {}),
        operational_completion=copy.deepcopy(operational_completion_binding or {}),
        product_completion=copy.deepcopy(product_completion_binding or {}),
        training_plan=copy.deepcopy(training_plan_binding or {}),
        managed_runtime=copy.deepcopy(managed_runtime_binding or {}),
        training_path=copy.deepcopy(training_path_binding or {}),
    )


def save_manifest(manifest: ReplayManifest, output_dir: Path) -> Path:
    """Write the manifest to `<output_dir>/replay.json` and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest.to_dict(), indent=2, default=str))
    return path


def _decision_snapshot(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the immutable decision fields stored in replay history."""

    result = value.get("decision", value.get("result"))
    return {
        "decision_id": value.get("id", value.get("decision_id")),
        "result": result,
        "decision": result,
        "action": value.get("action"),
        "reasons": list(value.get("reasons") or []),
        "evidence": dict(value.get("evidence") or {}),
        "override": bool(value.get("override", False)),
        "override_note": value.get("override_note"),
        "supersedes_decision_id": value.get("supersedes_decision_id"),
        "created_at": value.get("created_at"),
    }


def _write_json_durably(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish JSON with a same-directory atomic replacement and fsync."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            directory = None
        if directory is not None:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def sync_reward_integrity_decision(
    source: Path,
    *,
    run_id: str,
    audit: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> Dict[str, Any]:
    """Atomically project an audit decision into a current replay manifest.

    The latest decision remains available on the audit entry for existing
    readers, while ``decision_history`` retains every immutable decision and
    operator override observed for that audit. Replay V4 remains writable for
    reward-decision compatibility; V1-V3 files are historical read-only
    evidence, while V5 adds corpus-training identity.
    """

    path = Path(source)
    if path.is_dir():
        path = path / MANIFEST_FILENAME
    if not path.is_file():
        return {"status": "not_recorded", "reason": "replay_missing", "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"existing replay manifest is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("existing replay manifest must be a JSON object")
    version = int(payload.get("manifest_version") or 0)
    if version < 4 or version > MANIFEST_VERSION:
        return {
            "status": "legacy_read_only",
            "manifest_version": version,
            "path": str(path),
        }
    captured_run_id = str(payload.get("run_id") or "")
    if captured_run_id and captured_run_id != str(run_id):
        raise RuntimeError("replay run identity does not match the reward audit")

    audit_id = str(audit.get("id") or audit.get("audit_id") or "").strip()
    if not audit_id:
        raise ValueError("reward audit identity is required for replay synchronization")
    decision_value = _decision_snapshot(decision)
    decision_id = str(decision_value.get("decision_id") or "").strip()
    if not decision_id:
        raise ValueError("reward decision identity is required for replay synchronization")

    raw_reward = payload.get("reward_integrity") or {}
    if not isinstance(raw_reward, Mapping):
        raise RuntimeError("reward_integrity replay data must be a JSON object")
    reward = dict(raw_reward)
    entries = [
        dict(value)
        for value in reward.get("audit_decisions") or []
        if isinstance(value, Mapping)
    ]
    existing_index = next(
        (
            index
            for index, value in enumerate(entries)
            if str(value.get("audit_id") or "") == audit_id
        ),
        None,
    )
    existing = dict(entries[existing_index]) if existing_index is not None else {}
    history = [
        dict(value)
        for value in existing.get("decision_history") or []
        if isinstance(value, Mapping)
    ]
    existing_decision_id = str(existing.get("decision_id") or "").strip()
    if existing_decision_id and not any(
        str(value.get("decision_id") or "") == existing_decision_id
        for value in history
    ):
        history.append(_decision_snapshot(existing))
    if not any(
        str(value.get("decision_id") or "") == decision_id for value in history
    ):
        history.append(decision_value)

    update = {
        "audit_id": audit_id,
        "status": str(audit.get("status") or ""),
        "audit_manifest_hash": audit.get("manifest_hash")
        or audit.get("audit_manifest_hash"),
        "integrity_profile_revision_id": audit.get(
            "integrity_profile_revision_id"
        ),
        "work_item_id": audit.get("work_item_id"),
        **decision_value,
        "decision_history": history,
    }
    if existing_index is None:
        entries.append(update)
    else:
        entries[existing_index] = {**existing, **update}
    reward["audit_decisions"] = entries
    payload["reward_integrity"] = reward
    _write_json_durably(path, payload)

    published = json.loads(path.read_text(encoding="utf-8"))
    published_entry = next(
        (
            value
            for value in (published.get("reward_integrity") or {}).get(
                "audit_decisions", []
            )
            if isinstance(value, Mapping)
            and str(value.get("audit_id") or "") == audit_id
        ),
        None,
    )
    if published_entry is None or str(published_entry.get("decision_id") or "") != decision_id:
        raise RuntimeError("atomic replay verification did not retain latest decision")
    if published_entry.get("audit_manifest_hash") != update["audit_manifest_hash"]:
        raise RuntimeError("atomic replay verification did not retain audit identity")
    if not any(
        str(value.get("decision_id") or "") == decision_id
        for value in published_entry.get("decision_history") or []
        if isinstance(value, Mapping)
    ):
        raise RuntimeError("atomic replay verification did not retain decision history")
    return {
        "status": "updated",
        "path": str(path),
        "sha256": hash_dataset_file(path),
        "decision_id": decision_id,
        "decision_count": len(history),
    }


def load_manifest(source: Path) -> ReplayManifest:
    """Load a manifest from a path that's either the manifest file
    itself or the run's output directory.

    Tolerates older / future manifest_version fields by warning instead
    of refusing — the captured config + seed are forward/backward
    compatible enough that an older client can usually still replay.
    """
    p = Path(source)
    if p.is_dir():
        p = p / MANIFEST_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"Replay manifest not found at {p}")

    data = json.loads(p.read_text())
    version = data.get("manifest_version", 0)
    if version != MANIFEST_VERSION:
        logger.warning(
            "Replay manifest version %s does not match current %s — "
            "config + seed should still replay; environment diff may be skewed.",
            version,
            MANIFEST_VERSION,
        )

    # Construct via field-by-field assignment so missing fields don't
    # explode on older manifests. We treat absent fields as None / [].
    return ReplayManifest(
        manifest_version=int(data.get("manifest_version", 0)),
        run_id=str(data.get("run_id") or ""),
        modality=str(data.get("modality") or ""),
        timestamp=str(data.get("timestamp") or ""),
        model_name=str(data.get("model_name") or ""),
        seed=int(data.get("seed") or 0),
        pythonhashseed=data.get("pythonhashseed"),
        config=dict(data.get("config") or {}),
        dataset=dict(data.get("dataset") or {}),
        environment=dict(data.get("environment") or {}),
        cli_args=list(data.get("cli_args") or []),
        notes=data.get("notes"),
        verifier=dict(data.get("verifier") or {}),
        reward_integrity=dict(data.get("reward_integrity") or {}),
        corpus_training=dict(data.get("corpus_training") or {}),
        training_outcome=dict(data.get("training_outcome") or {}),
        adaptation_study=dict(data.get("adaptation_study") or {}),
        specialized_task=dict(data.get("specialized_task") or {}),
        agent_environment=dict(data.get("agent_environment") or {}),
        operational_completion=dict(data.get("operational_completion") or {}),
        product_completion=dict(data.get("product_completion") or {}),
        training_plan=dict(data.get("training_plan") or {}),
        managed_runtime=dict(data.get("managed_runtime") or {}),
        training_path=dict(data.get("training_path") or {}),
    )


# ---------- environment diff -----------------------------------------------


def compare_environments(captured: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a structured diff between two environment fingerprints.

    Returns:
        ``{
            "matched": bool,
            "differences": [
                {"key": "torch.version", "captured": "2.4.0", "current": "2.5.1"},
                {"key": "platform", "captured": "Darwin arm64", "current": "Linux x86_64"},
                ...
            ]
        }``
    """
    differences: List[Dict[str, Any]] = []

    for top_key in ("python", "platform", "backend"):
        if captured.get(top_key) != current.get(top_key):
            differences.append(
                {
                    "key": top_key,
                    "captured": captured.get(top_key),
                    "current": current.get(top_key),
                }
            )

    captured_pkgs = captured.get("packages") or {}
    current_pkgs = current.get("packages") or {}
    all_keys = set(captured_pkgs) | set(current_pkgs)
    for k in sorted(all_keys):
        if captured_pkgs.get(k) != current_pkgs.get(k):
            differences.append(
                {
                    "key": f"packages.{k}",
                    "captured": captured_pkgs.get(k),
                    "current": current_pkgs.get(k),
                }
            )

    return {
        "matched": not differences,
        "differences": differences,
    }


def compare_verifier_identities(
    captured: Dict[str, Any], current: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare the immutable verifier fields required for exact replay."""

    differences: List[Dict[str, Any]] = []
    fields = (
        "verifier_profile_revision_id",
        "revision_hash",
        "family",
        "implementation_ref",
        "implementation_fingerprint",
        "sanitized_configuration_hash",
        "reward_contract",
        "reliability_adapter",
    )
    for key in fields:
        if captured.get(key) != current.get(key):
            differences.append(
                {"key": key, "captured": captured.get(key), "current": current.get(key)}
            )
    runtime = dict(current.get("runtime_compatibility") or {})
    runtime_state = runtime.get("state") or runtime.get("status")
    if runtime_state != "compatible":
        differences.append(
            {
                "key": "runtime_compatibility",
                "captured": dict(captured.get("runtime_compatibility") or {}),
                "current": runtime,
            }
        )
    return {"matched": not differences, "differences": differences}


def compare_reward_identities(
    captured: Dict[str, Any], current: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare the immutable V4 reward-system fields required for exact replay."""

    differences: List[Dict[str, Any]] = []
    fields = (
        "reward_system_revision_id",
        "reward_system_hash",
        "optimizer_verifier_revision_id",
        "auditors",
        "reward_mapping_hash",
        "protocol_revision_id",
        "protocol_hash",
        "integrity_profile_revision_id",
        "integrity_profile_hash",
        "boundaries",
        "signal_capability",
        "trace_manifests",
        "audit_decisions",
    )
    for key in fields:
        if captured.get(key) != current.get(key):
            differences.append(
                {"key": key, "captured": captured.get(key), "current": current.get(key)}
            )
    runtime = dict(current.get("runtime_compatibility") or {})
    runtime_state = runtime.get("state") or runtime.get("status")
    if runtime and runtime_state != "compatible":
        differences.append(
            {
                "key": "runtime_compatibility",
                "captured": dict(captured.get("runtime_compatibility") or {}),
                "current": runtime,
            }
        )
    return {"matched": not differences, "differences": differences}


def compare_corpus_training_identities(
    captured: Dict[str, Any], current: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare the immutable V5 corpus-training inputs required for replay."""

    differences: List[Dict[str, Any]] = []
    for key in (
        "extraction_identity",
        "corpus_identity",
        "corpus_version",
        "tokenizer_identity",
        "tokenizer_hash",
        "packing_plan",
        "packing_plan_hash",
        "budget_mode",
        "target_tokens",
        "corpus_passes",
        "adaptation",
        "objective",
        "training_artifact",
    ):
        if captured.get(key) != current.get(key):
            differences.append(
                {"key": key, "captured": captured.get(key), "current": current.get(key)}
            )
    return {"matched": not differences, "differences": differences}


__all__ = [
    "MANIFEST_VERSION",
    "MANIFEST_FILENAME",
    "EnvironmentFingerprint",
    "ReplayManifest",
    "capture_manifest",
    "save_manifest",
    "load_manifest",
    "compare_environments",
    "compare_verifier_identities",
    "compare_reward_identities",
    "compare_corpus_training_identities",
    "hash_dataset_file",
    "sync_reward_integrity_decision",
]
