#!/usr/bin/env python3
"""
halo-forge CLI

Unified command-line interface for the halo forge framework.

Usage:
    halo-forge data prepare --dataset codeforces_cpp --output data/train.jsonl
    halo-forge data generate --topic rust_async --backend deepseek --output data/rust.jsonl
    halo-forge sft train --model Qwen/Qwen2.5-Coder-0.5B --data data/train.jsonl
    halo-forge raft train --model Qwen/Qwen2.5-Coder-0.5B --prompts data/prompts.jsonl
    halo-forge benchmark run --model models/raft/cycle_3 --prompts data/test.jsonl
    halo-forge test --level standard  # Validate pipeline
    halo-forge info  # Show hardware info
"""

# Pre-parse for --experimental-attention BEFORE any torch imports
# This must happen before any imports that could trigger torch loading
import sys
import os

if "--experimental-attention" in sys.argv:
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import argparse
import copy
import hashlib
import json
import math
import subprocess
import time
import shutil
import tempfile
import uuid
import webbrowser
from pathlib import Path
from typing import List, Dict, Any, Mapping, Optional

from halo_forge.capabilities import check_modality_train_capability
from halo_forge.utils.accelerator import (
    detect_gpu_kind,
    empty_accelerator_cache,
    get_device_map,
    recommended_attn_impl,
    recommended_dtype,
)

# ANSI color codes
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
BOLD = "\033[1m"
NC = "\033[0m"  # No Color

# RAFT verifier choices must stay in sync with cmd_raft_train dispatch.
RAFT_TRAIN_SUPPORTED_VERIFIERS = (
    "gcc",
    "mingw",
    "msvc",
    "humaneval",
    "mbpp",
    "rust",
    "go",
    "auto",
    "execution",
)

MODALITY_TRAIN_COMMANDS = ("vlm", "audio", "reasoning", "agentic")


def _cli_option_present(*names: str) -> bool:
    """Return whether an option was explicitly supplied, ignoring parser defaults."""

    for token in sys.argv[1:]:
        for name in names:
            if token == name or token.startswith(f"{name}="):
                return True
    return False


def _verifier_database_path(args: Any) -> Optional[str]:
    value = getattr(args, "database", None)
    return str(value) if value else None


def _training_signal_backend(args: Any) -> str:
    accelerator = str(getattr(args, "accelerator", None) or "").strip().lower()
    model = str(getattr(args, "model", None) or "")
    return "mlx" if accelerator == "mlx" or model.startswith("mlx-community/") else "hf"


def _prepare_reward_integrity(args: Any, *, trainer: str) -> Optional[dict[str, Any]]:
    """Resolve one monitored reward system before resolving its optimizer verifier."""

    revision_id = str(getattr(args, "reward_system_revision", None) or "").strip()
    audit_options = any(
        (
            str(getattr(args, "reward_audit_protocol_revision", None) or "").strip(),
            str(getattr(args, "reward_integrity_profile_revision", None) or "").strip(),
            str(
                getattr(args, "reward_development_suite_revision", None) or ""
            ).strip(),
            list(getattr(args, "reward_audit_boundary", None) or []),
        )
    )
    if not revision_id:
        if audit_options:
            raise ValueError(
                "Reward audit options require --reward-system-revision"
            )
        replay = getattr(args, "_verifier_replay_binding", None)
        if isinstance(replay, dict) and replay:
            setattr(
                args,
                "_reward_integrity_replay_binding",
                {"legacy_unmonitored": True},
            )
        return None

    protocol_id = str(
        getattr(args, "reward_audit_protocol_revision", None) or ""
    ).strip()
    profile_id = str(
        getattr(args, "reward_integrity_profile_revision", None) or ""
    ).strip()
    if not protocol_id or not profile_id:
        raise ValueError(
            "--reward-system-revision requires --reward-audit-protocol-revision "
            "and --reward-integrity-profile-revision"
        )
    boundaries = [
        str(value).strip()
        for value in list(getattr(args, "reward_audit_boundary", None) or ["final"])
        if str(value).strip()
    ]
    backend = _training_signal_backend(args)
    from halo_forge.reward_integrity import RewardIntegrityService
    from halo_forge.run_db import get_database
    from halo_forge.training_signal import TRAINING_SIGNAL_CAPABILITIES

    service = RewardIntegrityService(get_database(_verifier_database_path(args)))
    development_suite_revision_id = str(
        getattr(args, "reward_development_suite_revision", None) or ""
    ).strip()
    if development_suite_revision_id:
        service.validate_development_suite_revision(development_suite_revision_id)
    resolved = service.resolve_binding(
        revision_id,
        protocol_revision_id=protocol_id,
        integrity_profile_revision_id=profile_id,
        trainer=trainer,
        backend=backend,
        boundaries=boundaries,
        runtime_identity=None,
    )
    resolved_value = resolved.to_dict() if hasattr(resolved, "to_dict") else dict(resolved)
    if not bool(resolved_value.get("gating_eligible", False)):
        blockers = [str(value) for value in resolved_value.get("blockers") or []]
        raise ValueError(
            "; ".join(blockers)
            or "The selected reward system is not eligible for training-time gating"
        )
    reward_revision = dict(resolved_value.get("reward_system_revision") or {})
    optimizer_revision_id = str(
        reward_revision.get("optimizer_verifier_revision_id") or ""
    ).strip()
    if not optimizer_revision_id:
        raise ValueError("Reward system has no optimizer verifier revision")
    reward_mapping = dict(reward_revision.get("reward_mapping") or {})
    setattr(args, "_reward_system_mapping", reward_mapping)
    supplied_revision = str(
        getattr(args, "verifier_profile_revision", None) or ""
    ).strip()
    if supplied_revision and supplied_revision != optimizer_revision_id:
        raise ValueError(
            "--reward-system-revision conflicts with --verifier-profile-revision"
        )
    args.verifier_profile_revision = optimizer_revision_id
    capability = TRAINING_SIGNAL_CAPABILITIES.resolve(trainer, backend)
    if capability.fidelity.value not in {"exact", "sampled"}:
        raise ValueError(
            f"{trainer}/{backend} signal capture is {capability.fidelity.value} and cannot gate"
        )
    if not capability.resumable and any(
        str(value).lower() != "final" for value in boundaries
    ):
        raise ValueError(f"{trainer}/{backend} supports final-boundary audits only")
    if hasattr(resolved, "to_replay_dict"):
        replay = dict(resolved.to_replay_dict())
    else:
        protocol = dict(resolved_value.get("protocol_revision") or {})
        profile = dict(resolved_value.get("integrity_profile_revision") or {})
        replay = {
            "reward_system_revision_id": reward_revision.get("id"),
            "reward_system_hash": reward_revision.get("content_hash"),
            "optimizer_verifier_revision_id": optimizer_revision_id,
            "auditors": list(reward_revision.get("auditors") or []),
            "reward_mapping_hash": reward_revision.get("reward_mapping_hash")
            or reward_revision.get("content_hash"),
            "protocol_revision_id": protocol.get("id"),
            "protocol_hash": protocol.get("content_hash"),
            "integrity_profile_revision_id": profile.get("id"),
            "integrity_profile_hash": profile.get("content_hash"),
            "runtime_compatibility": {"state": "compatible", "backend": backend},
        }
    replay["boundaries"] = boundaries
    replay["signal_capability"] = capability.to_dict()
    if development_suite_revision_id:
        replay["development_suite_revision_id"] = development_suite_revision_id
    setattr(args, "_reward_integrity_replay_binding", replay)
    setattr(args, "_reward_integrity_resolved", resolved)
    setattr(args, "_training_signal_capability", capability)
    from halo_forge.runtime_determinism import RUN_ID_ENV, build_run_id

    os.environ.setdefault(RUN_ID_ENV, build_run_id(trainer))
    return replay


def _enqueue_managed_reward_training(args: Any, *, trainer: str) -> bool:
    """Turn a top-level audited CLI launch into durable segmented work.

    Scheduler child processes carry ``HALOFORGE_WORK_ITEM_ID`` and execute the
    bounded trainer command normally.  The operator-facing process only plans
    the immutable boundaries and optionally runs the local worker until the
    canonical run completes or needs review.
    """

    if not getattr(args, "reward_system_revision", None):
        return False
    if os.environ.get("HALOFORGE_WORK_ITEM_ID"):
        return False
    if bool(getattr(args, "dry_run", False)):
        return False

    import time
    from datetime import datetime, timezone

    from halo_forge.reward_integrity.direct_segments import (
        build_segment_launch_spec,
        resolve_boundary_values,
    )
    from halo_forge.run_db import RunRecord, get_database
    from halo_forge.runtime_determinism import RUN_ID_ENV, build_run_id
    from halo_forge.training_signal import TRAINING_SIGNAL_CAPABILITIES
    from halo_forge.training_signal.session import default_audit_boundaries
    from halo_forge.workstation_jobs import WorkstationScheduler

    backend = _training_signal_backend(args)
    capability = TRAINING_SIGNAL_CAPABILITIES.resolve(trainer, backend)
    max_steps = getattr(args, "max_steps", None)
    if trainer == "grpo" and backend == "hf" and max_steps:
        total = max(1, int(max_steps))
        resumable = capability.resumable
        unit = "step"
    elif trainer in {"raft", "vlm", "audio", "reasoning", "agentic"}:
        total = max(1, int(getattr(args, "cycles", None) or 1))
        resumable = capability.resumable
        unit = capability.boundary_unit
    else:
        total = max(1, int(getattr(args, "epochs", None) or 1))
        resumable = capability.resumable
        unit = capability.boundary_unit
    # HF GRPO without a resolved finite step budget cannot truthfully stop and
    # resume at an intermediate scheduler boundary.
    if trainer == "grpo" and backend == "hf" and not max_steps:
        resumable = False
        unit = "final"
    requested = list(getattr(args, "reward_audit_boundary", None) or [])
    if not requested:
        requested = default_audit_boundaries(total) if resumable else ["final"]
        args.reward_audit_boundary = list(requested)
    boundaries = resolve_boundary_values(requested, total=total, resumable=resumable)

    run_id = str(os.environ.get(RUN_ID_ENV) or build_run_id(trainer))
    os.environ[RUN_ID_ENV] = run_id
    output = Path(str(getattr(args, "output", "") or "")).expanduser().resolve()
    if not output.name:
        raise ValueError("managed audited training requires --output")
    if output.exists():
        raise ValueError(
            f"managed audited training will not reuse an existing output: {output}"
        )
    database = get_database(_verifier_database_path(args))
    scheduler = WorkstationScheduler(database)
    if database.list_work_items(canonical_run_id=run_id, limit=1):
        raise ValueError(f"managed audited run already exists: {run_id}")

    raw_config: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key.startswith("_") or callable(value):
            continue
        if isinstance(value, Path):
            raw_config[key] = str(value)
        elif isinstance(value, (str, int, float, bool, list, tuple, dict)) or value is None:
            raw_config[key] = value
    resolved_config = {
        **raw_config,
        "mode": trainer,
        "run_id": run_id,
        "output_dir": str(output),
        "_resolved_signal_backend": backend,
        "reward_audit_boundaries": list(requested),
    }
    canonical_command = [sys.executable, "-m", "halo_forge.cli", *sys.argv[1:]]
    base_launch = {
        "operation": "managed_training",
        "command": canonical_command,
        "canonical_command": canonical_command,
        "cwd": str(Path.cwd()),
        "output_dir": str(output),
        "resolved_launch_config": resolved_config,
        "source_ui_page": "cli",
    }
    now = datetime.now(timezone.utc).isoformat()
    database.upsert_run(
        RunRecord(
            run_id=run_id,
            fs_id=run_id,
            modality=trainer,
            model_name=str(getattr(args, "model", "") or ""),
            status="queued",
            timestamp=now,
            output_dir=str(output),
            seed=int(getattr(args, "seed", 42) or 42),
            raw_json=json.dumps(
                {
                    "run_id": run_id,
                    "status": "queued",
                    "launch_config": resolved_config,
                },
                sort_keys=True,
                default=str,
            ),
        )
    )
    segments: list[dict[str, Any]] = []
    try:
        previous = 0
        for ordinal, boundary in enumerate(boundaries):
            segment = database.create_direct_run_segment(
                run_id=run_id,
                ordinal=ordinal,
                unit=str(
                    unit if unit in {"step", "cycle", "epoch", "final"} else "final"
                ),
                start_value=previous,
                end_value=boundary,
            )
            if ordinal:
                segment = database.update_direct_run_segment(
                    str(segment["id"]),
                    status="blocked",
                    decision_reason="waiting for the previous boundary audit",
                )
            segments.append(segment)
            previous = boundary
        first_spec = build_segment_launch_spec(
            base_launch,
            segment={**segments[0], "is_final": len(segments) == 1},
        )
        work = scheduler.enqueue(
            kind="training",
            launch_spec=first_spec,
            resource_class="accelerator",
            resource_requirements={
                "exclusive_heavy_operation": True,
                "output_path": str(output.parent),
                "projected_disk_bytes": 0,
                "projected_ram_bytes": 0,
            },
            domain_kind="run",
            domain_id=run_id,
            canonical_run_id=run_id,
            log_path=str(output.parent / ".halo-forge-logs" / f"{run_id}.log"),
            max_retries=1,
            work_item_id=f"training-segment-work-{segments[0]['id']}",
        )
        database.update_direct_run_segment(
            str(segments[0]["id"]), work_item_id=work.id
        )
    except Exception as exc:
        # Leave a truthful, inspectable terminal projection when planning fails
        # after the canonical run has been minted.  No segment should look
        # runnable when its first work item was never durably enqueued.
        reason = f"managed audit setup failed: {type(exc).__name__}: {exc}"
        for segment in segments:
            database.update_direct_run_segment(
                str(segment["id"]), status="failed", decision_reason=reason
            )
        failed = database.get_run(run_id)
        if failed is not None:
            failed.status = "failed"
            failed.failure_reason = reason
            failed_payload = failed.raw
            failed_payload.update({"status": "failed", "failure_reason": reason})
            failed.raw_json = json.dumps(failed_payload, sort_keys=True, default=str)
            database.upsert_run(failed)
        raise
    print(f"Managed audited run queued: {run_id}")
    print(f"  work_item_id={work.id}")
    print(f"  boundaries={','.join(str(value) for value in requested)}")
    if not bool(getattr(args, "wait", False)):
        return True

    from halo_forge.workstation_jobs import WorkstationWorker

    worker = WorkstationWorker(scheduler)
    terminal = {"completed", "succeeded", "failed", "cancelled", "stopped", "awaiting_review"}
    while True:
        run = database.get_run(run_id)
        if run is not None and run.status in terminal:
            print(f"Managed audited run {run_id}: {run.status}")
            if run.status in {"failed", "cancelled", "stopped"}:
                raise RuntimeError(f"managed audited run ended with status {run.status}")
            return True
        processed = worker.run_once()
        if processed is None:
            time.sleep(0.1)


def _bind_reward_integrity(
    args: Any,
    *,
    domain_kind: str,
    domain_id: str,
    role: str = "training_gate",
) -> None:
    revision_id = str(getattr(args, "reward_system_revision", None) or "").strip()
    if not revision_id or not domain_id:
        return
    from halo_forge.reward_integrity import RewardIntegrityService
    from halo_forge.run_db import get_database

    RewardIntegrityService(get_database(_verifier_database_path(args))).bind(
        reward_system_revision_id=revision_id,
        protocol_revision_id=str(args.reward_audit_protocol_revision),
        integrity_profile_revision_id=str(args.reward_integrity_profile_revision),
        domain_kind=domain_kind,
        domain_id=str(domain_id),
        role=role,
        context={
            "boundaries": list(getattr(args, "reward_audit_boundary", None) or ["final"]),
            "development_suite_revision_id": (
                str(
                    getattr(args, "reward_development_suite_revision", None) or ""
                ).strip()
                or None
            ),
            "managed": bool(getattr(args, "_managed_dataset_replay", None)),
        },
    )


def _build_training_signal_session(
    args: Any,
    *,
    trainer: str,
    output_dir: str | Path,
    total_boundaries: int,
    reward_threshold: float,
    record_resolver: Any = None,
) -> Any:
    """Create a boundary-aware capture session for an audited launch."""

    if not getattr(args, "reward_system_revision", None):
        return None
    from halo_forge.runtime_determinism import RUN_ID_ENV, build_run_id
    from halo_forge.training_signal.session import (
        BoundarySignalSession,
        default_audit_boundaries,
    )
    from halo_forge.training_signal.models import normalize_producer_model_identity
    from halo_forge.verifier_lab.store import scrub_secrets

    run_id = os.environ.get(RUN_ID_ENV) or build_run_id(trainer)
    os.environ[RUN_ID_ENV] = run_id
    boundaries = list(getattr(args, "reward_audit_boundary", None) or [])
    if not boundaries:
        boundaries = default_audit_boundaries(total_boundaries)
        args.reward_audit_boundary = list(boundaries)
        replay = getattr(args, "_reward_integrity_replay_binding", None)
        if isinstance(replay, dict):
            replay["boundaries"] = list(boundaries)
    resolved = getattr(args, "_reward_integrity_resolved", None)
    protocol = "balanced_256"
    if resolved is not None:
        protocol_revision = getattr(resolved, "protocol_revision", None)
        protocol = str(getattr(protocol_revision, "capture_mode", protocol) or protocol)
    capability = getattr(args, "_training_signal_capability")
    if record_resolver is None:
        record_resolver = getattr(args, "_training_record_resolver", None)
    if (
        isinstance(getattr(args, "_managed_dataset_replay", None), dict)
        and record_resolver is None
    ):
        raise ValueError(
            "audited managed training requires a format v3 row-lineage index"
        )
    root = Path(
        os.environ.get("HALOFORGE_TRAINING_SIGNAL_ROOT")
        or Path.home() / ".halo-forge" / "training-signals"
    )
    producer_reference = next(
        (
            str(value).strip()
            for value in (
                getattr(args, "resume", None),
                getattr(args, "resume_from_checkpoint", None),
                getattr(args, "rollout_model", None),
                getattr(args, "checkpoint", None),
                getattr(args, "sft_checkpoint", None),
                getattr(args, "model", None),
            )
            if value is not None and str(value).strip()
        ),
        "",
    )
    producer_path: Optional[Path] = None
    producer_source = "model_reference"
    explicit_hash = str(
        os.environ.get("HALOFORGE_PRODUCER_MODEL_CONTENT_HASH") or ""
    ).strip()
    if explicit_hash:
        producer_hash, producer_identity = normalize_producer_model_identity(
            explicit_hash,
            {
                "content_available": True,
                "content_hash": explicit_hash,
                "identity_source": "managed_environment",
                "reference": producer_reference or None,
            },
        )
    else:
        resume_cycle = int(getattr(args, "resume_from_cycle", 0) or 0)
        candidate_values: list[tuple[str, str]] = []
        for name in (
            "resume",
            "resume_from_checkpoint",
            "rollout_model",
        ):
            value = getattr(args, name, None)
            if value is not None and str(value).strip():
                candidate_values.append((name, str(value).strip()))
        if resume_cycle > 0:
            candidate_values.append(
                (
                    "resume_cycle_model",
                    str(Path(output_dir).expanduser() / f"cycle_{resume_cycle - 1}" / "model"),
                )
            )
        for name in ("checkpoint", "sft_checkpoint"):
            value = getattr(args, name, None)
            if value is not None and str(value).strip():
                candidate_values.append((name, str(value).strip()))
        model_value = getattr(args, "model", None)
        if model_value is not None and str(model_value).strip():
            candidate_values.append(("model", str(model_value).strip()))
        for source_name, value in candidate_values:
            path = Path(value).expanduser()
            if path.exists():
                producer_path = path.resolve()
                producer_source = source_name
                break
        if producer_path is not None:
            try:
                from halo_forge.artifact_lab import hash_path

                digest = hash_path(producer_path)
                producer_hash, producer_identity = normalize_producer_model_identity(
                    digest.content_hash,
                    {
                        "content_available": True,
                        "content_hash": digest.content_hash,
                        "identity_source": producer_source,
                        "reference": str(producer_path),
                        "size_bytes": digest.size_bytes,
                        "file_count": digest.file_count,
                    },
                )
            except ValueError:
                # An empty checkpoint/model directory is not evidence of
                # model content; preserve it as a reference identity instead.
                producer_hash, producer_identity = normalize_producer_model_identity(
                    producer_reference or str(producer_path),
                    {
                        "content_available": False,
                        "identity_source": producer_source,
                        "reference": str(producer_path),
                        "unavailable_reason": "empty_model_payload",
                    },
                )
        else:
            producer_hash, producer_identity = normalize_producer_model_identity(
                producer_reference,
                {
                    "content_available": False,
                    "identity_source": producer_source,
                    "reference": producer_reference or None,
                    "unavailable_reason": (
                        "external_or_unresolved_reference"
                        if producer_reference
                        else "producer_reference_missing"
                    ),
                },
            )
    producer_identity = scrub_secrets(producer_identity)
    session = BoundarySignalSession(
        root,
        run_id=run_id,
        trainer=trainer,
        capability=capability,
        total_boundaries=max(1, int(total_boundaries)),
        boundaries=boundaries,
        protocol=protocol,
        reward_threshold=float(reward_threshold),
        attempt_id=str(os.environ.get("HALOFORGE_ATTEMPT_ID") or "attempt-1"),
        record_resolver=record_resolver,
        producer_model_hash=producer_hash,
        producer_model_identity=producer_identity,
    )
    setattr(args, "_training_signal_session", session)
    return session


def _checkpoint_content_candidate(
    summary: Mapping[str, Any], output_dir: str | Path
) -> Optional[Path]:
    """Return the exact trainer output whose bytes define the boundary."""

    candidates = (
        summary.get("final_model_path"),
        summary.get("checkpoint_path"),
        output_dir,
    )
    for value in candidates:
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if path.exists():
            return path.resolve()
    return None


def _checkpoint_content_hash(summary: Mapping[str, Any], output_dir: str | Path) -> str:
    from halo_forge.artifact_lab.hashing import hash_path
    from halo_forge.training_signal.models import content_hash

    candidate = _checkpoint_content_candidate(summary, output_dir)
    if candidate is not None:
        try:
            return hash_path(candidate).content_hash
        except Exception:
            pass
    return content_hash(
        {
            "run_id": summary.get("run_id"),
            "output_dir": str(output_dir),
            "final_model_path": summary.get("final_model_path"),
        }
    )


def _project_managed_checkpoint_path(
    checkpoint_path: Path,
    *,
    execution_output: str | Path,
    work_item: Any,
) -> Path:
    """Map an attempt-local checkpoint to its atomic publication location."""

    execution_root = Path(execution_output).expanduser().resolve()
    checkpoint = checkpoint_path.expanduser().resolve()
    try:
        relative = checkpoint.relative_to(execution_root)
    except ValueError as exc:
        raise ValueError(
            "development evaluation checkpoint is outside the managed training output"
        ) from exc
    launch_spec = dict(getattr(work_item, "launch_spec", {}) or {})
    segmented = launch_spec.get("operation") == "managed_training_segment"
    final_segment = bool(launch_spec.get("final_segment"))
    if segmented and not final_segment:
        publication_root = launch_spec.get("segment_output_dir")
    else:
        publication_root = launch_spec.get("output_dir")
    if not publication_root:
        raise ValueError("managed training work has no checkpoint publication location")
    return Path(str(publication_root)).expanduser().resolve() / relative


def _launch_boundary_development_evaluation(
    *,
    database: Any,
    scheduler: Any,
    suite_revision_id: str,
    run_id: str,
    signal_shard_id: str,
    direct_run_segment_id: str,
    checkpoint_hash: str,
    checkpoint_path: Path,
    execution_output: str | Path,
    current_work_item: Any,
) -> Any:
    """Queue independent checkpoint evidence behind its training segment.

    This evaluation is completion evidence only. Its metrics are never folded
    into the reward-integrity pass/warn/fail decision because V8 does not pin a
    development-quality threshold in the integrity profile.
    """

    from halo_forge.evaluation_lab import EvaluationLabService

    published_checkpoint = _project_managed_checkpoint_path(
        checkpoint_path,
        execution_output=execution_output,
        work_item=current_work_item,
    )
    evaluations = EvaluationLabService(database, scheduler=scheduler)
    try:
        return evaluations.launch_evaluation(
            suite_revision_id=suite_revision_id,
            subject={
                "type": "checkpoint",
                "run_id": run_id,
                "path": str(published_checkpoint),
            },
            request={
                "source": "reward_integrity_training_boundary",
                "gate_semantics": "completion_evidence_only",
                "changes_reward_integrity_decision": False,
                "training_signal_shard_id": signal_shard_id,
                "direct_run_segment_id": direct_run_segment_id,
                "checkpoint_hash": checkpoint_hash,
            },
            dependencies=[str(current_work_item.id)],
            submit=True,
        )
    finally:
        evaluations.shutdown(wait=False, cancel_futures=False)


def _seal_training_signal_session(
    args: Any, summary: Mapping[str, Any], output_dir: str | Path
) -> list[Any]:
    session = getattr(args, "_training_signal_session", None)
    if session is None:
        return []
    existing = getattr(args, "_training_signal_shards", None)
    if isinstance(existing, list):
        return existing
    checkpoint_path = _checkpoint_content_candidate(summary, output_dir)
    checkpoint_hash = _checkpoint_content_hash(summary, output_dir)
    shards = list(session.finalize(checkpoint_hash=checkpoint_hash))
    setattr(args, "_training_signal_shards", shards)
    from halo_forge.reward_integrity import RewardIntegrityService
    from halo_forge.run_db import get_database
    from halo_forge.workstation_jobs import WorkstationScheduler

    database = get_database(_verifier_database_path(args))
    service = RewardIntegrityService(database)
    scheduler = WorkstationScheduler(database)
    reward_revision_id = str(args.reward_system_revision)
    protocol_revision_id = str(args.reward_audit_protocol_revision)
    profile_revision_id = str(args.reward_integrity_profile_revision)
    managed = getattr(args, "_managed_dataset_replay", None)
    dataset_identity = (
        {
            "bindings": list(managed.get("bindings") or []),
            "training_artifact": dict(managed.get("training_artifact") or {}),
        }
        if isinstance(managed, dict)
        else {
            "path": str(
                getattr(args, "data", None)
                or getattr(args, "prompts", None)
                or getattr(args, "dataset", None)
                or ""
            )
        }
    )
    capability = getattr(args, "_training_signal_capability")
    registered = []
    audits = []
    current_work_item_id = str(os.environ.get("HALOFORGE_WORK_ITEM_ID") or "").strip()
    current_work_item = (
        database.get_work_item(current_work_item_id) if current_work_item_id else None
    )
    if current_work_item_id and current_work_item is None:
        current_work_item_id = ""
    development_suite_revision_id = (
        str(getattr(args, "reward_development_suite_revision", None) or "").strip()
        or None
    )
    existing_segments = {
        int(value["ordinal"]): value
        for value in database.list_direct_run_segments(session.run_id)
    }
    managed_segment_id = str(
        os.environ.get("HALOFORGE_DIRECT_RUN_SEGMENT_ID") or ""
    ).strip()
    managed_segment = (
        database.get_direct_run_segment(managed_segment_id)
        if managed_segment_id
        else None
    )
    if managed_segment is not None and str(managed_segment["run_id"]) != session.run_id:
        raise ValueError("managed direct-run segment belongs to a different run")
    previous_boundary = 0
    for local_ordinal, shard in enumerate(shards):
        resolved_boundary = (
            int(shard.boundary) if str(shard.boundary).isdigit() else local_ordinal + 1
        )
        ordinal = (
            int(managed_segment["ordinal"])
            if managed_segment is not None
            else local_ordinal
        )
        segment = managed_segment or existing_segments.get(ordinal)
        expected_start = (
            int(segment["start_value"])
            if managed_segment is not None
            else previous_boundary
        )
        if segment is None:
            segment = database.create_direct_run_segment(
                run_id=session.run_id,
                ordinal=ordinal,
                unit=(
                    str(capability.boundary_unit)
                    if str(capability.boundary_unit) in {"step", "cycle", "epoch", "final"}
                    else "final"
                ),
                start_value=expected_start,
                end_value=resolved_boundary,
                work_item_id=current_work_item_id or None,
            )
        elif (
            int(segment["start_value"]) != expected_start
            or int(segment["end_value"]) != resolved_boundary
        ):
            raise ValueError(
                "direct-run segment identity conflicts with the sealed audit schedule"
            )
        segment = database.update_direct_run_segment(
            str(segment["id"]),
            status="completed",
            decision="complete",
            decision_reason="checkpoint and training-signal trace published",
        )
        registered_shard = service.register_training_signal_shard(
            shard,
            reward_system_revision_id=reward_revision_id,
            protocol_revision_id=protocol_revision_id,
            dataset_identity=dataset_identity,
            runtime_identity=dict(getattr(session, "runtime_identity", {}) or {}),
            boundary_unit=str(capability.boundary_unit),
            boundary_value=resolved_boundary,
            direct_run_segment_id=str(segment["id"]),
        )
        registered.append(registered_shard)
        audit_work_id = f"reward-audit-work-{uuid.uuid4().hex}"
        audit = service.create_audit(
            signal_shard_id=registered_shard.id,
            integrity_profile_revision_id=profile_revision_id,
            development_suite_revision_id=development_suite_revision_id,
            runtime_identity={"backend": _training_signal_backend(args)},
            request={
                "same_output": True,
                "source": "training_boundary",
                "boundary_unit": registered_shard.boundary_unit,
                "boundary_value": registered_shard.boundary_value,
                **(
                    {
                        "development_evaluation": {
                            "suite_revision_id": development_suite_revision_id,
                            "checkpoint_hash": checkpoint_hash,
                            "gate_semantics": "completion_evidence_only",
                            "changes_reward_integrity_decision": False,
                        }
                    }
                    if development_suite_revision_id
                    else {}
                ),
            },
            submit=False,
        )
        reused_audit = audit.status == "completed"
        existing_scheduled_audit = bool(audit.work_item_id) and not reused_audit
        development_evaluation = None
        development_evaluation_work_id = ""
        if reused_audit:
            service.verify_audit_bundle(audit.id)
            audit_work_id = str(audit.work_item_id or "")
        elif existing_scheduled_audit:
            audit_work_id = str(audit.work_item_id)
        else:
            if development_suite_revision_id:
                if current_work_item is None:
                    raise ValueError(
                        "boundary development evaluation requires managed scheduler work"
                    )
                if checkpoint_path is None:
                    raise ValueError(
                        "boundary development evaluation has no published checkpoint candidate"
                    )
                launched_evaluation = _launch_boundary_development_evaluation(
                    database=database,
                    scheduler=scheduler,
                    suite_revision_id=development_suite_revision_id,
                    run_id=session.run_id,
                    signal_shard_id=registered_shard.id,
                    direct_run_segment_id=str(segment["id"]),
                    checkpoint_hash=checkpoint_hash,
                    checkpoint_path=checkpoint_path,
                    execution_output=output_dir,
                    current_work_item=current_work_item,
                )
                development_evaluation = launched_evaluation.evaluation
                development_evaluation_work_id = str(
                    development_evaluation.work_item_id or ""
                )
                if (
                    development_evaluation.status != "completed"
                    and not development_evaluation_work_id
                ):
                    raise ValueError(
                        "boundary development evaluation has no durable work item"
                    )
            service.bind(
                reward_system_revision_id=reward_revision_id,
                protocol_revision_id=protocol_revision_id,
                integrity_profile_revision_id=profile_revision_id,
                audit_id=audit.id,
                domain_kind="run",
                domain_id=session.run_id,
                role="training_audit",
                context={
                    "signal_shard_id": registered_shard.id,
                    "work_item_id": audit_work_id,
                    "boundary_value": registered_shard.boundary_value,
                    **(
                        {
                            "development_evaluation_id": development_evaluation.id,
                            "development_evaluation_work_item_id": (
                                development_evaluation_work_id or None
                            ),
                            "development_evaluation_semantics": (
                                "completion_evidence_only"
                            ),
                        }
                        if development_evaluation is not None
                        else {}
                    ),
                },
            )
            resources = service.audit_resource_requirements(reward_revision_id)
            resource_class = str(resources["resource_class"])
            audit_dependencies = (
                [development_evaluation_work_id]
                if development_evaluation_work_id
                else ([current_work_item_id] if current_work_item_id else [])
            )
            audit_work = scheduler.enqueue(
                kind="reward_integrity_audit",
                launch_spec={
                    "handler": "reward_integrity.execute_audit",
                    "operation": "reward_integrity_audit",
                    "audit_id": audit.id,
                    "reward_integrity_root": str(
                        Path.home() / ".halo-forge" / "evaluations" / "reward-audits"
                    ),
                },
                resource_class=resource_class,
                resource_requirements=resources,
                domain_kind="reward_integrity_audit",
                domain_id=audit.id,
                canonical_run_id=session.run_id,
                dependencies=audit_dependencies,
                max_retries=1,
                work_item_id=audit_work_id,
            )
            audit = service.store.update_audit(audit.id, work_item_id=audit_work.id)
        audits.append(audit)
        if (
            not reused_audit
            and bool(getattr(args, "wait", False))
            and not current_work_item_id
        ):
            claimed = scheduler.claim(work_item_id=audit_work_id)
            if claimed is not None:
                try:
                    from halo_forge.reward_integrity.runtime import execute_pinned_audit

                    completed = execute_pinned_audit(database, audit.id)
                    scheduler.complete(
                        claimed,
                        result={"audit_id": completed.id, "status": completed.status},
                    )
                    audit = completed
                    audits[-1] = audit
                except Exception as exc:
                    database.finish_work_item(
                        claimed.id,
                        claim_token=claimed.claim_token,
                        result={"audit_id": audit.id},
                        error=str(exc),
                    )
                    raise
        previous_boundary = resolved_boundary
    setattr(args, "_registered_training_signal_shards", registered)
    setattr(args, "_reward_integrity_audits", audits)
    replay = getattr(args, "_reward_integrity_replay_binding", None)
    if isinstance(replay, dict):
        replay["trace_manifests"] = [
            {
                "shard_id": shard.shard_id,
                "trace_hash": shard.trace_hash,
                "path": shard.path,
                "boundary": shard.boundary,
                "checkpoint_hash": shard.checkpoint_hash,
                "capture_fidelity": shard.capture_fidelity,
                "observed_count": shard.observed_count,
                "retained_count": shard.retained_count,
            }
            for shard in shards
        ]
        replay_audits = []
        for audit in audits:
            decisions = service.store.list_decisions(audit.id, limit=1000)
            latest = decisions.items[-1].to_dict() if decisions.items else {}
            replay_audits.append(
                {
                    "audit_id": audit.id,
                    "status": audit.status,
                    "audit_manifest_hash": audit.manifest_hash,
                    "integrity_profile_revision_id": (
                        audit.integrity_profile_revision_id
                    ),
                    "work_item_id": audit.work_item_id,
                    "decision_id": latest.get("id"),
                    "result": latest.get("decision"),
                    "decision": latest.get("decision"),
                    "action": latest.get("action"),
                    "reasons": list(latest.get("reasons") or []),
                }
            )
        replay["audit_decisions"] = replay_audits
    return shards


def _prepare_profile_verifier(
    args: Any,
    *,
    consumer: str,
    modality: Optional[str] = None,
    training: bool = False,
) -> Any:
    """Resolve one immutable verifier revision and build its exact runtime bridge.

    Parser defaults are deliberately not treated as raw configuration.  An
    explicitly entered raw value may coexist only when it agrees with the
    immutable revision; otherwise the launch is ambiguous and is refused.
    """

    revision_id = str(getattr(args, "verifier_profile_revision", None) or "").strip()
    if not revision_id:
        raw_name = str(getattr(args, "verifier", None) or "").strip()
        if raw_name:
            setattr(
                args,
                "_verifier_replay_binding",
                {
                    "implementation_ref": raw_name,
                    "legacy_unqualified": True,
                    "legacy_warning": (
                        "This run used a raw verifier configuration without an immutable "
                        "reliability qualification."
                    ),
                },
            )
        return None

    from halo_forge.run_db import get_database
    from halo_forge.verifier_lab.runtime import (
        ProfileRevisionVerifier,
        register_profile_verifier,
    )
    from halo_forge.verifier_lab.service import VerifierLabService

    database_path = _verifier_database_path(args)
    service = VerifierLabService(get_database(database_path))
    resolved = service.resolve_binding(revision_id, modality=modality)
    revision = service.store.get_profile_revision(revision_id)
    contract = revision.reward_contract
    reward_mapping = dict(getattr(args, "_reward_system_mapping", {}) or {})
    filtering = dict(reward_mapping.get("filtering") or {})
    mapped_threshold = reward_mapping.get("threshold", filtering.get("threshold"))
    if training and contract.direction != "maximize":
        raise ValueError(
            "Profile-backed training currently requires reward direction='maximize'; "
            "evaluation and calibration continue to support minimize contracts"
        )

    raw_name = str(getattr(args, "verifier", None) or "").strip()
    if _cli_option_present("--verifier") and raw_name and raw_name != revision.implementation_ref:
        raise ValueError(
            "--verifier-profile-revision conflicts with the explicitly supplied "
            f"--verifier {raw_name!r}"
        )
    raw_config_options = tuple(
        name
        for name in (
            "--verifier-config",
            "--host",
            "--user",
            "--ssh-key",
            "--unsafe-verifier-execution",
            "--run-after-compile",
            "--cross-compile",
        )
        if _cli_option_present(name)
    )
    if raw_config_options:
        raise ValueError(
            "--verifier-profile-revision conflicts with raw verifier configuration: "
            + ", ".join(raw_config_options)
        )
    threshold_option = None
    threshold_value = None
    if _cli_option_present("--reward-threshold"):
        threshold_option = "--reward-threshold"
        threshold_value = getattr(args, "reward_threshold", None)
    elif _cli_option_present("--threshold"):
        threshold_option = "--threshold"
        threshold_value = getattr(args, "threshold", None)
    if threshold_option:
        expected_threshold = (
            mapped_threshold if mapped_threshold is not None else contract.threshold
        )
        if expected_threshold is None or not math.isclose(
            float(threshold_value), float(expected_threshold), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"--verifier-profile-revision conflicts with {threshold_option}; "
                "the immutable reward contract controls the threshold"
            )

    setattr(args, "_verifier_replay_binding", dict(resolved))
    setattr(args, "_verifier_profile_contract", contract.to_dict())
    setattr(args, "_verifier_profile_revision", revision)
    effective_threshold = (
        mapped_threshold
        if mapped_threshold is not None
        else (contract.threshold if contract.threshold is not None else contract.minimum)
    )
    if hasattr(args, "reward_threshold"):
        args.reward_threshold = effective_threshold
    if hasattr(args, "threshold"):
        args.threshold = effective_threshold

    keep_policy = dict(reward_mapping.get("keep_policy") or {})
    mapped_keep = reward_mapping.get(
        "keep_percent", keep_policy.get("keep_percent", filtering.get("keep_percent"))
    )
    if mapped_keep is not None and hasattr(args, "keep_percent"):
        if _cli_option_present("--keep-percent") and not math.isclose(
            float(getattr(args, "keep_percent")),
            float(mapped_keep),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "--reward-system-revision conflicts with --keep-percent; "
                "the immutable reward system controls filtering"
            )
        args.keep_percent = float(mapped_keep)
    shaping = dict(reward_mapping.get("shaping") or {})
    mapped_shaping = reward_mapping.get("reward_shaping", shaping.get("strategy"))
    if mapped_shaping is not None and hasattr(args, "reward_shaping"):
        if (
            _cli_option_present("--reward-shaping")
            and str(getattr(args, "reward_shaping")) != str(mapped_shaping)
        ):
            raise ValueError(
                "--reward-system-revision conflicts with --reward-shaping; "
                "the immutable reward system controls shaping"
            )
        args.reward_shaping = str(mapped_shaping)

    if consumer == "registry":
        if getattr(args, "reward_system_revision", None):
            from halo_forge.reward_integrity import register_reward_mapped_verifier

            args.verifier = register_reward_mapped_verifier(
                revision_id, reward_mapping, database=database_path
            )
        else:
            args.verifier = register_profile_verifier(revision_id, database=database_path)
        return args.verifier
    runtime = ProfileRevisionVerifier(revision_id, database=database_path)
    if getattr(args, "reward_system_revision", None):
        from halo_forge.reward_integrity import RewardMappedVerifier

        runtime = RewardMappedVerifier(runtime, reward_mapping)
    setattr(args, "_profile_verifier_runtime", runtime)
    return runtime


def _bind_profile_verifier(
    args: Any,
    *,
    domain_kind: str,
    domain_id: str,
    role: str,
) -> None:
    revision_id = str(getattr(args, "verifier_profile_revision", None) or "").strip()
    if not revision_id or not domain_id:
        return
    key = (domain_kind, str(domain_id), role)
    already = set(getattr(args, "_verifier_bound_domains", set()))
    if key in already:
        return
    from halo_forge.run_db import get_database
    from halo_forge.verifier_lab.service import VerifierLabService

    service = VerifierLabService(get_database(_verifier_database_path(args)))
    service.bind_revision(
        revision_id,
        domain_kind=domain_kind,
        domain_id=str(domain_id),
        role=role,
        context={"consumer": role},
    )
    already.add(key)
    setattr(args, "_verifier_bound_domains", already)


def _positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_dataset_binding(value: str) -> dict[str, str]:
    """Parse ``role=version:split`` used by every managed training command."""
    raw = str(value or "").strip()
    if "=" not in raw:
        raise ValueError("dataset binding must use role=version:split")
    role, identity = (part.strip() for part in raw.split("=", 1))
    if ":" in identity:
        version_id, split = (part.strip() for part in identity.rsplit(":", 1))
    else:
        version_id, split = identity, role
    if not role or not version_id or not split:
        raise ValueError("dataset binding must use role=version:split")
    return {"role": role.lower(), "dataset_version_id": version_id, "split": split}


def _artifact_db_bindings(artifact: Any) -> list[dict[str, Any]]:
    """Return per-source binding counts from the published artifact manifest."""
    resolved: dict[tuple[str, str, str], int] = {}
    try:
        manifest = json.loads((Path(artifact.path) / "manifest.json").read_text(encoding="utf-8"))
        for value in manifest.get("resolved_bindings") or []:
            key = (
                str(value.get("role") or "train"),
                str(value.get("dataset_version_id") or ""),
                str(value.get("split") or "train"),
            )
            resolved[key] = int(value.get("row_count") or 0)
    except (OSError, json.JSONDecodeError):
        pass
    values = []
    for binding in artifact.bindings:
        value = binding.to_dict()
        value["row_count"] = resolved.get(
            (binding.role, binding.dataset_version_id, binding.split),
            int(artifact.row_counts.get(binding.role, 0)),
        )
        values.append(value)
    return values


def _managed_replay_identity(lab: Any, artifact: Any, db: Any) -> dict[str, Any]:
    """Build the immutable Dataset Lab identity captured by direct CLI runs."""
    resolved = {
        (
            str(value.get("role") or "train"),
            str(value.get("dataset_version_id") or ""),
            str(value.get("split") or "train"),
        ): dict(value)
        for value in artifact.resolved_bindings
    }
    bindings: list[dict[str, Any]] = []
    for binding in artifact.bindings:
        version = lab.store.get_any(
            binding.dataset_version_id,
            dataset_id=binding.dataset_id,
        )
        version_manifest: dict[str, Any] = {}
        try:
            version_manifest = json.loads(
                (Path(version.path) / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            # The renderer already verified the version.  This fallback keeps
            # replay finalization useful for legacy stores with sparse manifests.
            pass
        catalog_version = db.get_dataset_version(binding.dataset_version_id)
        source_record = (
            db.get_dataset_source(catalog_version.source_id)
            if catalog_version is not None and catalog_version.source_id
            else None
        )
        guided_source = (
            dict((source_record.metadata or {}).get("guided_own_data") or {})
            if source_record is not None
            else {}
        )
        source_fingerprints: Any = None
        if catalog_version is not None and catalog_version.source_fingerprints:
            source_fingerprints = catalog_version.source_fingerprints
        if not source_fingerprints:
            source_fingerprints = {
                "source": version_manifest.get("source_fingerprint") or version.source_fingerprint,
                "assets": version_manifest.get("asset_fingerprints") or [],
            }
        resolved_binding = resolved.get(
            (binding.role, binding.dataset_version_id, binding.split), {}
        )
        materialized = bool(
            version_manifest.get("materialized_assets", version.materialized_assets)
        )
        binding_identity = {
                "role": binding.role,
                "dataset_id": version.dataset_id,
                "dataset_version_id": version.version_id,
                "split": binding.split,
                "content_hash": version.content_hash,
                "recipe_hash": version.recipe_hash,
                "source_fingerprints": source_fingerprints,
                "assets_materialized": materialized,
                "asset_materialization_state": ("materialized" if materialized else "referenced"),
                "row_count": int(resolved_binding.get("row_count") or 0),
                "split_content_hash": resolved_binding.get("content_fingerprint"),
                "exposed_to_trainer": bool(resolved_binding.get("exposed_to_trainer", False)),
            }
        extraction_identity = copy.deepcopy(
            guided_source.get("corpus_extraction") or {}
        )
        if extraction_identity:
            binding_identity["extraction_identity"] = extraction_identity
        bindings.append(binding_identity)

    artifact_payload = {
        "artifact_id": artifact.artifact_id,
        "artifact_hash": artifact.artifact_hash,
        "format_version": int(getattr(artifact, "format_version", 2)),
        "adapter_id": artifact.adapter_id,
        "adapter_version": artifact.adapter_version,
        "trainer_mode": artifact.trainer_mode,
        "model": artifact.model,
        "tokenizer_revision": artifact.tokenizer_revision,
        "chat_template_hash": artifact.chat_template_hash,
        "validation_policy": dict(artifact.validation_policy),
        "token_statistics": dict(artifact.token_statistics),
        "row_counts": dict(artifact.row_counts),
        "split_paths": dict(artifact.split_paths),
        "asset_roots": list(artifact.asset_roots),
        "model_revision": getattr(artifact, "model_revision", None),
        "model_hash": getattr(artifact, "model_hash", None),
        "tokenizer_hash": getattr(artifact, "tokenizer_hash", None),
        "packing_plan": copy.deepcopy(
            getattr(artifact, "packing_plan", None)
        ),
        "packing_plan_hash": getattr(artifact, "packing_plan_hash", None),
        "split_fidelity": copy.deepcopy(
            getattr(artifact, "split_fidelity", {}) or {}
        ),
    }
    return {"bindings": bindings, "training_artifact": artifact_payload}


def _finalize_managed_training_replay(
    args: Any,
    modality: str,
    output_dir: str | Path,
    config: Any,
    summary: Optional[dict[str, Any]] = None,
) -> Optional[Path]:
    """Write replay.json after a successful managed Dataset Lab CLI run.

    Manual paths and built-in dataset names deliberately remain on the legacy
    path: without ``_managed_dataset_replay`` this helper is a no-op.
    """
    summary = dict(summary or {})
    database = getattr(args, "database", None)
    _seal_training_signal_session(args, summary, output_dir)
    completed_run_id = str(summary.get("run_id") or os.environ.get("HALO_FORGE_RUN_ID") or "").strip()
    if completed_run_id:
        _bind_profile_verifier(
            args,
            domain_kind="run",
            domain_id=completed_run_id,
            role=f"{modality}_training",
        )
        _bind_reward_integrity(
            args,
            domain_kind="run",
            domain_id=completed_run_id,
        )
    managed = getattr(args, "_managed_dataset_replay", None)
    reward_binding = getattr(args, "_reward_integrity_replay_binding", None)
    if (
        not isinstance(managed, dict)
        and not isinstance(reward_binding, dict)
        and modality != "cpt"
    ):
        return None

    from halo_forge.replay import capture_manifest, save_manifest

    run_id = str(
        (managed.get("run_id") if isinstance(managed, dict) else None)
        or completed_run_id
        or os.environ.get("HALO_FORGE_RUN_ID")
        or ""
    ).strip()
    if not run_id:
        raise RuntimeError("managed Dataset Lab launch is missing its canonical run ID")
    summary_run_id = str((summary or {}).get("run_id") or "").strip()
    if summary_run_id and summary_run_id != run_id:
        raise RuntimeError(
            "trainer run ID diverged from the preallocated Dataset Lab run ID: "
            f"{summary_run_id!r} != {run_id!r}"
        )
    model_name = str(
        getattr(config, "model_name", None)
        or getattr(config, "base_model", None)
        or getattr(args, "model", None)
        or ""
    )
    seed = int(
        (summary or {}).get("seed")
        if (summary or {}).get("seed") is not None
        else getattr(config, "seed", getattr(args, "seed", 42))
    )
    corpus_training_binding: Optional[dict[str, Any]] = None
    if modality == "cpt":
        artifact_identity = (
            dict(managed.get("training_artifact") or {})
            if isinstance(managed, dict)
            else {}
        )
        train_binding = next(
            (
                dict(value)
                for value in (
                    list(managed.get("bindings") or [])
                    if isinstance(managed, dict)
                    else []
                )
                if str(value.get("role") or "") == "train"
            ),
            {},
        )
        packing_plan = copy.deepcopy(
            summary.get("packing_plan")
            or artifact_identity.get("packing_plan")
            or {}
        )
        tokenizer_hash = str(
            summary.get("tokenizer_hash")
            or artifact_identity.get("tokenizer_hash")
            or ""
        ).strip()
        packing_plan_hash = str(
            summary.get("packing_plan_hash")
            or artifact_identity.get("packing_plan_hash")
            or ""
        ).strip()
        if not tokenizer_hash or not packing_plan_hash:
            raise RuntimeError(
                "completed CPT training is missing its tokenizer or packing-plan identity"
            )
        corpus_identity: dict[str, Any]
        if train_binding:
            corpus_identity = copy.deepcopy(train_binding)
        else:
            train_file = Path(
                str(
                    getattr(config, "train_file", None)
                    or getattr(args, "train_file", None)
                    or ""
                )
            ).expanduser()
            if not train_file.is_file():
                raise RuntimeError(
                    "completed CPT training is missing its corpus file identity"
                )
            digest = hashlib.sha256()
            with train_file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            corpus_identity = {
                "path": str(train_file.resolve()),
                "content_hash": digest.hexdigest(),
                "size_bytes": train_file.stat().st_size,
                "identity_kind": "manual_path",
            }
        corpus_training_binding = {
            "extraction_identity": copy.deepcopy(
                train_binding.get("extraction_identity") or {}
            ),
            "corpus_identity": corpus_identity,
            "corpus_version": train_binding.get("dataset_version_id"),
            "tokenizer_identity": {
                "model": model_name,
                "revision": getattr(config, "tokenizer_revision", None),
                "hash": tokenizer_hash,
            },
            "tokenizer_hash": tokenizer_hash,
            "packing_plan": packing_plan,
            "packing_plan_hash": packing_plan_hash,
            "budget_mode": str(getattr(config, "budget_mode", "") or ""),
            "target_tokens": getattr(config, "target_tokens", None),
            "corpus_passes": getattr(config, "corpus_passes", None),
            "adaptation": str(getattr(config, "adaptation", "") or ""),
            "training_artifact": artifact_identity,
        }

    try:
        operational_env = json.loads(
            os.environ.get("HALO_FORGE_OPERATIONAL_COMPLETION") or "{}"
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        operational_env = {}
    if not isinstance(operational_env, dict):
        operational_env = {}

    def evidence_value(name: str) -> Any:
        value = getattr(args, name, None)
        if value not in (None, "", [], {}):
            return value
        if isinstance(config, dict):
            value = config.get(name)
        else:
            value = getattr(config, name, None)
        return (
            value
            if value not in (None, "", [], {})
            else operational_env.get(name)
        )

    outcome_assessment_id = str(
        evidence_value("outcome_assessment_id") or ""
    ).strip()
    outcome_binding: Optional[dict[str, Any]] = None
    operational_completion: dict[str, Any] = {}
    if outcome_assessment_id:
        outcome_binding = {"assessment_id": outcome_assessment_id}
        try:
            from halo_forge.run_db import get_database

            assessment = get_database()._conn.execute(
                """SELECT proof_run_id,base_evaluation_id,candidate_evaluation_id,
                          comparison_hash,status
                   FROM training_outcome_assessments WHERE id=?""",
                (outcome_assessment_id,),
            ).fetchone()
            if assessment is not None:
                outcome_binding.update(
                    proof_run_id=assessment["proof_run_id"],
                    base_evaluation_id=assessment["base_evaluation_id"],
                    candidate_evaluation_id=assessment["candidate_evaluation_id"],
                    comparison_hash=assessment["comparison_hash"],
                    status=assessment["status"],
                )
                operational_completion["prepared_evaluation_ids"] = [
                    value
                    for value in (
                        assessment["base_evaluation_id"],
                        assessment["candidate_evaluation_id"],
                    )
                    if value
                ]
                decision = get_database()._conn.execute(
                    """SELECT id,decision,reason,full_run_id
                       FROM training_outcome_decisions
                       WHERE assessment_id=? ORDER BY created_at DESC LIMIT 1""",
                    (outcome_assessment_id,),
                ).fetchone()
                if decision is not None:
                    operational_completion["reviewed_decision"] = {
                        "id": decision["id"],
                        "decision": decision["decision"],
                        "reason": decision["reason"],
                        "full_run_id": decision["full_run_id"],
                    }
        except Exception:
            # The immutable assessment identity remains sufficient for legacy
            # databases; exact prepared-evaluation identities stay unavailable.
            pass
    assignment_id = evidence_value("study_assignment_id")
    if not assignment_id:
        assignments_by_seed = evidence_value("study_assignment_ids_by_seed")
        if isinstance(assignments_by_seed, dict):
            assignment_id = assignments_by_seed.get(str(seed))
    study_binding = {
        key: value
        for key, value in {
            "study_id": evidence_value("study_id"),
            "protocol_revision_id": evidence_value("study_protocol_revision_id"),
            "arm_id": evidence_value("study_arm_id"),
            "assignment_id": assignment_id,
            "factor_values": evidence_value("study_factor_values"),
            "contrast_ids": evidence_value("study_contrast_ids"),
            "deviation_ids": evidence_value("study_deviation_ids"),
        }.items()
        if value not in (None, "", [], {})
    }
    if study_binding.get("assignment_id"):
        operational_completion["study_launch_assignment_id"] = study_binding[
            "assignment_id"
        ]

    product_completion: dict[str, Any] = {}
    try:
        from halo_forge.product_lab import ProductLabService
        from halo_forge.run_db import get_database

        database = get_database(_verifier_database_path(args))
        if isinstance(managed, dict):
            for binding in managed.get("bindings") or []:
                version_id = str(binding.get("dataset_version_id") or "")
                version = database.get_dataset_version(version_id) if version_id else None
                if version is None:
                    continue
                provenance = dict(version.provenance or {})
                repair_step = next(
                    (
                        dict(step.get("details") or {})
                        for step in provenance.get("steps") or []
                        if isinstance(step, dict)
                        and step.get("kind") == "repair_overlay"
                        and isinstance(step.get("details"), dict)
                    ),
                    {},
                )
                repair = dict(provenance.get("dataset_repair") or repair_step or {})
                if repair:
                    product_completion.update(
                        dataset_repair_revision_id=(
                            repair.get("revision_id")
                            or repair.get("id")
                            or repair.get("repair_revision_id")
                        ),
                        repair_content_hash=(
                            repair.get("content_hash")
                            or repair.get("repair_content_hash")
                        ),
                        repaired_record_set_hash=repair.get("repaired_record_set_hash"),
                        source_fingerprint=repair.get("source_fingerprint"),
                    )
                    break
        product = ProductLabService(database)
        readiness = product.assess_readiness()
        product_completion.update(
            workstation_readiness_id=readiness.id,
            workstation_readiness_hash=readiness.content_hash,
            distribution_capability=readiness.capability.to_dict(),
        )
    except Exception:
        # Existing direct CLI launches remain compatible when the v20 catalog
        # or optional workstation probes are unavailable.
        product_completion = {}

    training_plan_binding = {
        key: value
        for key, value in {
            "training_plan_revision_id": getattr(
                args, "training_plan_revision", None
            ),
            "training_capacity_check_id": getattr(
                args, "training_capacity_check_id", None
            ),
            "model_preparation_id": getattr(args, "model_preparation_id", None),
            "compute_shape_hash": getattr(args, "training_compute_shape_hash", None),
            "selected_adjustment": getattr(
                args, "training_capacity_adjustment", None
            ),
            "decision_id": getattr(args, "training_plan_decision_id", None),
            "plan_content_hash": getattr(args, "training_plan_content_hash", None),
            "runtime_hash": getattr(args, "training_plan_runtime_hash", None),
            "recommendation_reasons": getattr(
                args, "training_plan_recommendation_reasons", None
            ),
            "resource_forecast": getattr(args, "training_plan_forecast", None),
            "resolved_model_commit": getattr(args, "resolved_model_commit", None),
            "model_preparation_manifest_hash": getattr(
                args, "model_preparation_manifest_hash", None
            ),
            "role": "proof" if bool(getattr(args, "proof_run", False)) else "full",
        }.items()
        if value not in (None, "", {}, [])
    }
    managed_runtime_binding: dict[str, Any] = {}
    training_path_binding: dict[str, Any] = {}
    runtime_revision_id = str(
        getattr(args, "runtime_profile_revision", None) or ""
    ).strip()
    if runtime_revision_id:
        try:
            from halo_forge.managed_runtime import ManagedRuntimeService
            from halo_forge.run_db import get_database

            if not hasattr(database, "_conn"):
                database = get_database(_verifier_database_path(args))
            runtime_service = ManagedRuntimeService(database)
            revision = runtime_service.get_revision(runtime_revision_id)
            qualification = runtime_service.latest_qualification(runtime_revision_id)
            if revision is None or qualification is None:
                raise ValueError("managed runtime identity is incomplete")
            preflight = database._conn.execute(
                "SELECT * FROM accelerator_preflight_decisions "
                "WHERE runtime_revision_id=? ORDER BY created_at DESC LIMIT 1",
                (runtime_revision_id,),
            ).fetchone()
            managed_runtime_binding = {
                "runtime_profile_revision_id": revision.id,
                "runtime_content_hash": revision.content_hash,
                "base_image_digest": revision.base_image_digest,
                "derived_image_ref": revision.derived_image_ref,
                "qualification_id": qualification.id,
                "qualification_hash": qualification.qualification_hash,
                "runtime_identity_hash": qualification.runtime_identity_hash,
                "host_identity_hash": qualification.host_identity_hash,
                "device_identity_hash": qualification.device_identity_hash,
                "occupancy_decision": dict(preflight) if preflight else None,
            }
        except Exception as exc:
            managed_runtime_binding = {
                "runtime_profile_revision_id": runtime_revision_id,
                "identity_error": str(exc),
            }
    path_revision_id = str(
        getattr(args, "training_path_revision_id", None) or ""
    ).strip()
    path_certification_id = str(
        getattr(args, "training_path_certification_id", None) or ""
    ).strip()
    if path_revision_id or path_certification_id:
        try:
            from halo_forge.training_path_certification import (
                TrainingPathCertificationService,
            )
            from halo_forge.run_db import get_database

            if not hasattr(database, "_conn"):
                database = get_database(_verifier_database_path(args))
            path_service = TrainingPathCertificationService(database)
            path_revision = path_service.get_revision(path_revision_id)
            certification = path_service.get_certification(path_certification_id)
            if path_revision is None or certification is None:
                raise ValueError("training-path identity is incomplete")
            verification = path_service.verify(certification.id)
            if not verification["valid"]:
                raise ValueError("training-path certification is stale")
            training_path_binding = {
                "runtime_core_qualification_id": certification.runtime_qualification_id,
                "training_path_revision_id": path_revision.id,
                "training_path_revision_hash": path_revision.content_hash,
                "training_path_certification_id": certification.id,
                "training_path_certification_hash": certification.certification_hash,
                "fixture_id": path_revision.fixture_id,
                "fixture_hash": path_revision.fixture_hash,
                "exact_model_commit": getattr(args, "resolved_model_commit", None),
                "trainer_adapter_version": path_revision.trainer_adapter_version,
                "capacity_adapter_version": path_revision.capacity_adapter_version,
                "host_identity_hash": certification.host_identity_hash,
            }
        except Exception as exc:
            training_path_binding = {
                "training_path_revision_id": path_revision_id or None,
                "training_path_certification_id": path_certification_id or None,
                "identity_error": str(exc),
            }

    manifest = capture_manifest(
        run_id=run_id,
        modality=modality,
        model_name=model_name,
        seed=seed,
        config=config,
        dataset_bindings=(
            list(managed.get("bindings") or []) if isinstance(managed, dict) else None
        ),
        training_artifact=(
            dict(managed.get("training_artifact") or {})
            if isinstance(managed, dict)
            else None
        ),
        dataset_file=(
            next(
                (
                    Path(str(value)).expanduser()
                    for value in (
                        getattr(args, "data", None),
                        getattr(args, "train_file", None),
                        getattr(args, "prompts", None),
                        getattr(args, "dataset", None),
                    )
                    if value and Path(str(value)).expanduser().is_file()
                ),
                None,
            )
            if not isinstance(managed, dict)
            else None
        ),
        dataset_id=(
            str(getattr(args, "dataset", None) or "") or None
            if not isinstance(managed, dict)
            else None
        ),
        verifier_binding=(
            dict(getattr(args, "_verifier_replay_binding"))
            if isinstance(getattr(args, "_verifier_replay_binding", None), dict)
            else None
        ),
        reward_integrity_binding=(
            dict(reward_binding) if isinstance(reward_binding, dict) else None
        ),
        training_outcome_binding=outcome_binding,
        adaptation_study_binding=study_binding or None,
        corpus_training_binding=corpus_training_binding,
        operational_completion_binding=operational_completion or None,
        product_completion_binding=product_completion or None,
        training_plan_binding=training_plan_binding or None,
        managed_runtime_binding=managed_runtime_binding or None,
        training_path_binding=training_path_binding or None,
        cli_args=list(sys.argv[1:]),
    )
    path = save_manifest(manifest, Path(output_dir).expanduser())
    print(f"Replay manifest: {path}")
    return path


def _apply_managed_dataset_args(
    args: Any, trainer_mode: str, target: str
) -> Optional[dict[str, Any]]:
    """Render managed Dataset Lab inputs and point a trainer at the artifact.

    This is deliberately shared by all training commands so dashboard, API,
    and CLI resolve the same adapter and content-addressed bundle.
    """
    version_id = str(getattr(args, "dataset_version", None) or "").strip()
    raw_bindings = list(getattr(args, "dataset_binding", None) or [])
    if not version_id and not raw_bindings:
        return None

    from halo_forge.data_lab import DatasetLab
    from halo_forge.runtime_determinism import RUN_ID_ENV, build_run_id
    from halo_forge.run_db import get_database

    run_id = os.environ.get(RUN_ID_ENV) or build_run_id(trainer_mode)
    os.environ[RUN_ID_ENV] = run_id
    bindings = [_parse_dataset_binding(value) for value in raw_bindings]
    if version_id:
        if any(binding["role"] == "train" for binding in bindings):
            raise ValueError(
                "--dataset-version is the train shorthand and cannot be combined "
                "with a train=... --dataset-binding"
            )
        bindings.insert(
            0,
            {"role": "train", "dataset_version_id": version_id, "split": "train"},
        )

    root = (
        Path(os.environ.get("HALOFORGE_DATASET_ROOT") or (Path.home() / ".halo-forge" / "datasets"))
        .expanduser()
        .resolve()
    )
    lab = DatasetLab(root)
    render_options: dict[str, Any] = {
        "trainer_mode": trainer_mode,
        "model": str(getattr(args, "model", None) or "") or None,
        "tokenizer_revision": getattr(args, "tokenizer_revision", None),
        "validation_fraction": float(
            getattr(args, "validation_split", 0.05) or 0.0
        ),
        "seed": int(getattr(args, "seed", 42) or 42),
    }
    if trainer_mode == "cpt":
        render_options.update(
            model_revision=getattr(args, "model_revision", None),
            model_hash=getattr(args, "model_hash", None),
            tokenizer_hash=getattr(args, "tokenizer_hash", None),
            max_sequence_length=int(
                getattr(args, "max_seq_length", 2048) or 2048
            ),
            packing=str(
                getattr(
                    args,
                    "packing",
                    "paragraph_eos_non_overlap_v1",
                )
            ),
            budget_mode=str(getattr(args, "budget_mode", "passes") or "passes"),
            target_tokens=getattr(args, "target_tokens", None),
            corpus_passes=getattr(args, "corpus_passes", None),
            effective_batch_size=max(
                1,
                int(getattr(args, "batch_size", 1) or 1)
                * int(getattr(args, "gradient_accumulation", 1) or 1),
            ),
        )
    artifact = lab.training_artifacts.render(bindings, **render_options)
    from halo_forge.training_signal import record_resolver_from_training_artifact

    record_resolver = record_resolver_from_training_artifact(artifact, role="train")
    if record_resolver is None:
        raise ValueError(
            "managed Dataset Lab training requires a format v3 row-lineage index"
        )
    setattr(args, "_training_record_resolver", record_resolver)
    train_path = artifact.split_paths.get("train")
    if not train_path:
        raise ValueError("rendered training artifact has no trainer-visible train split")
    setattr(args, target, train_path)
    # Managed data always wins over a manually entered built-in dataset name.
    if target == "data" and hasattr(args, "dataset"):
        args.dataset = None
    validation_path = artifact.split_paths.get("validation")
    if validation_path:
        if hasattr(args, "validation_data"):
            args.validation_data = validation_path
        if hasattr(args, "validation_file"):
            args.validation_file = validation_path
    if trainer_mode == "cpt":
        args.model = artifact.model
        args.model_revision = artifact.model_revision
        args.model_hash = artifact.model_hash
        args.tokenizer_revision = artifact.tokenizer_revision
        args.tokenizer_hash = artifact.tokenizer_hash
        args.training_artifact_id = artifact.artifact_id
        args.training_artifact_hash = artifact.artifact_hash
        args.expected_packing_plan_hash = artifact.packing_plan_hash

    db_bindings = _artifact_db_bindings(artifact)
    db = get_database()
    stored = db.create_training_artifact(
        artifact_id=artifact.artifact_id,
        artifact_hash=artifact.artifact_hash,
        adapter_id=artifact.adapter_id,
        adapter_version=artifact.adapter_version,
        trainer_mode=artifact.trainer_mode,
        model_id=artifact.model,
        tokenizer_revision=artifact.tokenizer_revision,
        chat_template_hash=artifact.chat_template_hash,
        manifest_path=str(Path(artifact.path) / "manifest.json"),
        bindings=db_bindings,
        metadata=artifact.to_dict(),
    )
    for binding in artifact.bindings:
        db.attach_run_dataset(
            run_id=run_id,
            dataset_version_id=binding.dataset_version_id,
            role=binding.role,
            split=binding.split,
            training_artifact_id=stored.id,
        )
    _bind_profile_verifier(
        args,
        domain_kind="run",
        domain_id=run_id,
        role=f"{trainer_mode}_training",
    )
    payload = artifact.to_dict()
    setattr(args, "training_artifact", payload)
    replay_identity = _managed_replay_identity(lab, artifact, db)
    replay_identity["run_id"] = run_id
    setattr(args, "_managed_dataset_replay", replay_identity)
    print(
        f"Dataset artifact: {artifact.artifact_id} "
        f"({artifact.adapter_id}@{artifact.adapter_version}, run {run_id})"
    )
    return payload


def _enforce_modality_train_contract(modality: str, args) -> None:
    """Validate modality train gating/model support contract."""
    if modality not in MODALITY_TRAIN_COMMANDS:
        return

    check = check_modality_train_capability(
        modality=modality,
        model_name=getattr(args, "model", ""),
        allow_prototype_train=getattr(args, "allow_prototype_train", False),
        dry_run=getattr(args, "dry_run", False),
    )
    if not check.allowed:
        print(f"{RED}{check.message}{NC}")
        sys.exit(2)


def _enforce_training_outcome_or_exit(modality: str, summary: dict) -> None:
    """Fail non-zero when a train command produced no optimizer updates."""
    effectiveness = summary.get("effectiveness")
    if isinstance(effectiveness, dict) and effectiveness.get("verdict") == "fail":
        reason_text = ",".join(effectiveness.get("reasons") or []) or "effectiveness_failed"
        steps = int(summary.get("total_train_steps_executed", 0) or 0)
        print(
            f"{RED}TRAINING_CONTRACT_ERROR modality={modality} "
            f"reason={reason_text} total_train_steps_executed={steps}{NC}"
        )
        print(
            "Training completed but failed the effectiveness contract. "
            "Check sample filtering, optimizer updates, artifact writes, and evaluation deltas."
        )
        sys.exit(2)

    if summary.get("weights_updated", False):
        return

    reason = summary.get("final_update_reason", "no_updates")
    steps = int(summary.get("total_train_steps_executed", 0) or 0)
    print(
        f"{RED}TRAINING_CONTRACT_ERROR modality={modality} "
        f"reason={reason} total_train_steps_executed={steps}{NC}"
    )
    print(
        "Training completed without any optimizer updates. "
        "Check dataset quality, model support, and adapter configuration."
    )
    sys.exit(2)


def _print_training_run_metadata(summary: dict) -> None:
    """Print deterministic runtime metadata when available."""
    run_id = summary.get("run_id")
    if run_id:
        print(f"Run ID: {run_id}")
    if summary.get("seed") is not None:
        print(f"Seed: {summary['seed']}")
    resume_from_cycle = summary.get("resume_from_cycle")
    if resume_from_cycle is not None:
        print(f"Resume from cycle: {resume_from_cycle}")
    resumed_from = summary.get("resumed_from_checkpoint")
    if isinstance(resumed_from, dict) and resumed_from.get("model_dir"):
        print(f"Resumed checkpoint: {resumed_from['model_dir']}")


def _print_completed_training_summary(
    modality: str,
    output_dir: str,
    summary: dict,
    *,
    args: Any = None,
    config: Any = None,
) -> None:
    """Print canonical post-train summary details."""
    _enforce_training_outcome_or_exit(modality, summary)
    print(f"\n{GREEN}Training complete!{NC}")
    print(f"Output: {output_dir}")
    if summary.get("final_model_path"):
        print(f"Final model: {summary['final_model_path']}")
    _print_training_run_metadata(summary)
    print(f"Train steps executed: {int(summary.get('total_train_steps_executed', 0) or 0)}")
    final_loss = summary.get("final_train_loss")
    if isinstance(final_loss, (int, float)):
        print(f"Final train loss: {final_loss:.4f}")
    effectiveness = summary.get("effectiveness")
    if isinstance(effectiveness, dict):
        print(f"Effectiveness verdict: {effectiveness.get('verdict', 'unknown')}")
    if args is not None:
        run_id = str(summary.get("run_id") or "").strip()
        if run_id:
            _bind_profile_verifier(
                args,
                domain_kind="run",
                domain_id=run_id,
                role=f"{modality}_training",
            )
        _finalize_managed_training_replay(
            args,
            modality,
            output_dir,
            config if config is not None else {},
            summary,
        )


# =============================================================================
# Auto-Logging System
# =============================================================================


class TeeWriter:
    """
    Write to both stdout and a log file simultaneously.

    Implements tee-style output without requiring external commands.
    Used for automatic logging of all training/benchmark commands.
    """

    def __init__(self, log_path: Path, quiet: bool = False):
        """
        Initialize TeeWriter.

        Args:
            log_path: Path to log file
            quiet: If True, suppress terminal output (log file only)
        """
        self.log_path = log_path
        self.quiet = quiet
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", buffering=1, encoding="utf-8")  # Line buffered

    def write(self, message: str):
        """Write to both terminal and log file."""
        # Always write to log file
        self.log_file.write(message)

        # Write to terminal unless quiet mode
        if not self.quiet:
            try:
                self.terminal.write(message)
            except UnicodeEncodeError:
                # Legacy Windows consoles may still use cp1252. Keep the
                # complete UTF-8 log while rendering unsupported terminal
                # glyphs as replacements instead of aborting the command.
                encoding = getattr(self.terminal, "encoding", None) or "ascii"
                safe_message = message.encode(encoding, errors="replace").decode(encoding)
                self.terminal.write(safe_message)

    def flush(self):
        """Flush both outputs."""
        self.log_file.flush()
        if not self.quiet:
            self.terminal.flush()

    def close(self):
        """Close log file and restore stdout."""
        self.log_file.close()
        sys.stdout = self.terminal

    def isatty(self):
        """Check if terminal is a TTY (for color support)."""
        return not self.quiet and self.terminal.isatty()


def _default_app_log_dir() -> Path:
    configured = str(os.environ.get("HALO_FORGE_LOG_DIR") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".halo-forge" / "logs"


def setup_auto_logging(command_name: str, output_dir: str = "logs", quiet: bool = False) -> Path:
    """
    Configure automatic logging with timestamped file.

    Creates logs/ directory if needed and redirects stdout to both
    terminal and log file (unless quiet mode).

    Args:
        command_name: Name of command being run (e.g., 'raft_train')
        output_dir: Directory for log files (default: 'logs')
        quiet: If True, suppress terminal output

    Returns:
        Path to log file
    """
    from datetime import datetime

    log_dir = Path(output_dir).expanduser()
    if not log_dir.is_absolute() and output_dir == "logs" and os.environ.get("HALO_FORGE_LOG_DIR"):
        log_dir = _default_app_log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        if log_dir.is_absolute() or output_dir != "logs":
            raise
        log_dir = _default_app_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{command_name}_{timestamp}.log"

    # Install TeeWriter
    tee = TeeWriter(log_path, quiet=quiet)
    sys.stdout = tee

    # Also capture stderr if not quiet
    if not quiet:
        sys.stderr = tee

    return log_path


def print_banner():
    """Print the halo forge banner."""
    # Disable colors when output is piped to file
    if sys.stdout.isatty():
        c, nc = CYAN, NC
    else:
        c, nc = "", ""

    width = 63
    lines = [
        "HALO-FORGE",
        "Local AI Training Framework",
        "ROCm / CUDA / MPS / MLX / CPU",
    ]
    print(f"\n{c}╔{'═' * width}╗")
    for line in lines:
        print(f"║{line.center(width)}║")
    print(f"╚{'═' * width}╝{nc}\n")


def cmd_data_synthesize(args):
    """Synthesize a training dataset (Track D1: teacher → verifier → filter)."""
    from pathlib import Path

    from halo_forge.data.synthesize import synthesize_dataset

    profile_verifier = _prepare_profile_verifier(args, consumer="direct", modality="text")

    print_banner()
    print(f"{GREEN}halo-forge data synthesize{NC}")
    print("=" * 60)
    print(f"  seeds:    {args.seeds}")
    print(f"  output:   {args.output}")
    print(f"  teacher:  {args.teacher_model} ({args.base_url or 'default endpoint'})")
    print(f"  verifier: {args.verifier}")
    print(f"  shape:    {args.kind} (n={args.n_per_prompt})")
    print(f"  threshold: {args.threshold}")
    print()

    try:
        result = synthesize_dataset(
            seeds=args.seeds,
            output_path=Path(args.output),
            teacher_model=args.teacher_model,
            base_url=args.base_url,
            api_key=args.api_key,
            system_prompt=args.system_prompt,
            verifier_name=args.verifier,
            verifier=profile_verifier,
            verifier_profile_revision_id=getattr(args, "verifier_profile_revision", None),
            n_per_prompt=args.n_per_prompt,
            reward_threshold=args.threshold,
            output_kind=args.kind,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    except Exception as exc:
        print(f"{RED}Synthesis failed:{NC} {exc}")
        sys.exit(1)

    pct_kept = 100.0 * result.n_accepted / max(1, result.n_generated)
    print(f"{GREEN}Done{NC} in {result.duration_seconds:.1f}s")
    print(f"  seeds:     {result.n_seeds:>8,}")
    print(f"  generated: {result.n_generated:>8,}")
    print(f"  accepted:  {result.n_accepted:>8,}  ({pct_kept:.1f}%)")
    print(f"  avg reward: {result.avg_reward:.3f}")
    print(f"  output:    {result.output_path}")
    _bind_profile_verifier(
        args,
        domain_kind="dataset_output",
        domain_id=str(Path(result.output_path).expanduser().resolve()),
        role="synthesis_filter",
    )


def cmd_data_score(args):
    """Score a JSONL dataset by quality and filter by threshold or top-K%
    (Track D3)."""
    from pathlib import Path

    from halo_forge.data.quality import QualityScore, score_file

    profile_verifier = _prepare_profile_verifier(args, consumer="direct", modality="text")
    scorer = None
    score_threshold = float(args.threshold)
    if profile_verifier is not None:
        contract = dict(getattr(args, "_verifier_profile_contract", {}) or {})
        minimum = float(contract.get("minimum", 0.0))
        maximum = float(contract.get("maximum", 1.0))
        direction = str(contract.get("direction") or "maximize")

        def verifier_scorer(record: Any) -> QualityScore:
            candidate = record
            if isinstance(record, dict):
                candidate = next(
                    (
                        record.get(key)
                        for key in ("completion", "chosen", "response", "output", "text")
                        if record.get(key) is not None
                    ),
                    json.dumps(record, sort_keys=True, default=str),
                )
            result = profile_verifier.verify(candidate=candidate, record=record)
            if result.error:
                return QualityScore(
                    score=0.0,
                    components={"verifier": 0.0},
                    rejected=True,
                    reason="verifier_error",
                )
            normalized = (float(result.reward) - minimum) / (maximum - minimum)
            if direction == "minimize":
                normalized = 1.0 - normalized
            return QualityScore(score=normalized, components={"verifier": normalized})

        scorer = verifier_scorer
        raw_threshold = contract.get("threshold")
        if raw_threshold is None:
            score_threshold = 0.0
        else:
            score_threshold = (float(raw_threshold) - minimum) / (maximum - minimum)
            if direction == "minimize":
                score_threshold = 1.0 - score_threshold

    print_banner()
    print(f"{GREEN}halo-forge data score{NC}")
    print("=" * 60)
    print(f"  input:  {args.input}")
    print(f"  output: {args.output}")
    if args.top_k_pct is not None:
        print(f"  filter: keep top {args.top_k_pct:.0%}")
    else:
        print(f"  filter: score >= {args.threshold}")
    print()

    try:
        result = score_file(
            input_path=Path(args.input),
            output_path=Path(args.output),
            threshold=score_threshold,
            keep_top_k_pct=args.top_k_pct,
            **({"scorer": scorer} if scorer is not None else {}),
        )
    except Exception as exc:
        print(f"{RED}Score failed:{NC} {exc}")
        sys.exit(1)

    pct_kept = 100.0 * result.n_kept / max(1, result.n_input)
    print(f"{GREEN}Done{NC} in {result.duration_seconds:.2f}s")
    print(f"  input:    {result.n_input:>8,}")
    print(f"  kept:     {result.n_kept:>8,}  ({pct_kept:.1f}%)")
    print(f"  rejected: {result.n_rejected:>8,}")
    if result.reasons:
        print(f"  rejection reasons (by weakest component):")
        for reason, count in sorted(result.reasons.items(), key=lambda kv: kv[1], reverse=True):
            print(f"    {reason:>14}: {count:>5}")
    _bind_profile_verifier(
        args,
        domain_kind="dataset_output",
        domain_id=str(Path(args.output).expanduser().resolve()),
        role="quality_scoring",
    )


def cmd_data_dedup(args):
    """Deduplicate a JSONL dataset (Track D2)."""
    from pathlib import Path

    from halo_forge.data.dedup import dedup_file

    print_banner()
    print(f"{GREEN}halo-forge data dedup{NC}")
    print("=" * 60)
    print(f"  input:  {args.input}")
    print(f"  output: {args.output}")
    print(f"  method: {args.method}")
    if args.method == "fuzzy":
        print(f"  threshold: {args.threshold}")
    print()

    try:
        result = dedup_file(
            input_path=Path(args.input),
            output_path=Path(args.output),
            method=args.method,
            threshold=args.threshold,
            key=args.key,
            case_sensitive=args.case_sensitive,
        )
    except Exception as exc:
        print(f"{RED}Dedup failed:{NC} {exc}")
        sys.exit(1)

    pct_removed = 100.0 * result.n_removed / max(1, result.n_input)
    print(f"{GREEN}Done{NC} in {result.duration_seconds:.2f}s")
    print(f"  input:    {result.n_input:>8,}")
    print(f"  kept:     {result.n_output:>8,}")
    print(f"  removed:  {result.n_removed:>8,}  ({pct_removed:.1f}%)")


def cmd_data_validate(args):
    """Validate dataset format."""
    from halo_forge.data.validator import validate_dataset

    result = validate_dataset(args.file, preview=args.preview)

    if not result.valid:
        sys.exit(1)


def _dataset_lab(args):
    """Return the shared Dataset Lab facade for CLI commands."""
    from pathlib import Path
    import os

    from halo_forge.data_lab import DatasetLab
    from halo_forge.run_db import get_database

    root = getattr(args, "root", None) or os.environ.get("HALOFORGE_DATASET_ROOT")
    database = get_database(getattr(args, "database", None))
    return DatasetLab(
        Path(root).expanduser() if root else Path.home() / ".halo-forge" / "datasets",
        database=database,
    )


def _dataset_lab_service(args):
    """Create the same SQLite-backed service used by the dashboard/API."""
    from halo_forge.public_api.service import PublicApiService
    from halo_forge.run_db import get_database

    lab = _dataset_lab(args)
    database = get_database(getattr(args, "database", None))
    return (
        PublicApiService(
            database=database,
            dataset_lab=lab,
            dataset_storage_root=lab.root,
        ),
        lab,
    )


def _guided_own_data_service(args):
    """Create the transport-neutral own-data service used by API and CLI."""
    from halo_forge.own_data import GuidedOwnDataService
    from halo_forge.run_db import get_database
    from halo_forge.workstation_jobs import (
        WorkstationScheduler,
        sample_workstation_capacity,
    )

    lab = _dataset_lab(args)
    database = get_database(getattr(args, "database", None))
    scheduler = WorkstationScheduler(database)
    return (
        GuidedOwnDataService(
            database,
            datasets_root=lab.root,
            imports_root=lab.root.parent / "imports",
            scheduler=scheduler,
            capacity_probe=lambda path: sample_workstation_capacity(
                path, include_accelerator=False
            ),
        ),
        lab,
    )


def _guided_active_backend_name() -> str:
    """Resolve the same active runtime used by managed dashboard launches."""

    try:
        from halo_forge.backend import get_backend

        return str(get_backend().name)
    except Exception:
        return "unknown"


def _print_dataset_lab_payload(payload):
    """Render Dataset Lab dataclasses and mappings consistently."""
    import json

    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    elif isinstance(payload, list):
        payload = [item.to_dict() if hasattr(item, "to_dict") else item for item in payload]
    print(json.dumps(payload, indent=2, default=str))


def cmd_data_lab_add(args):
    """Register a local or pinned Hugging Face Dataset Lab source."""
    import json

    if bool(args.path) == bool(args.hf_id):
        print(f"{RED}Choose exactly one of --path or --hf-id.{NC}")
        sys.exit(2)
    mapping = {}
    if args.mapping:
        try:
            mapping = json.loads(args.mapping)
        except json.JSONDecodeError as exc:
            print(f"{RED}Invalid --mapping JSON:{NC} {exc}")
            sys.exit(2)
        if not isinstance(mapping, dict):
            print(f"{RED}Invalid --mapping JSON:{NC} expected an object")
            sys.exit(2)
    for assignment in args.map_fields or []:
        target, separator, source_field = assignment.partition("=")
        target = target.strip()
        source_field = source_field.strip()
        if not separator or not target or not source_field:
            print(f"{RED}Invalid --map value:{NC} expected target=source")
            sys.exit(2)
        if target in mapping and mapping[target] != source_field:
            print(f"{RED}Conflicting mapping for {target}:{NC} {mapping[target]!r} vs {source_field!r}")
            sys.exit(2)
        mapping[target] = source_field

    scenario = None
    if args.scenario:
        from halo_forge.own_data import GuidedOwnDataService
        from halo_forge.own_data.registry import TRAINING_SCENARIOS

        try:
            scenario = TRAINING_SCENARIOS.get(args.scenario)
        except KeyError:
            print(f"{RED}Unknown training scenario:{NC} {args.scenario}")
            sys.exit(2)
        if not scenario.available:
            print(
                f"{RED}Scenario is unavailable:{NC} "
                f"{scenario.unavailable_reason or 'no verified trainer contract'}"
            )
            sys.exit(2)
        runtime_available, runtime_reason = GuidedOwnDataService._runtime_scenario_status(
            scenario,
            _guided_active_backend_name(),
        )
        if not runtime_available:
            print(
                f"{RED}Scenario is unavailable on this workstation:{NC} "
                f"{runtime_reason or 'no verified trainer runtime'}"
            )
            sys.exit(2)
        if args.kind and args.kind != scenario.canonical_schema:
            print(
                f"{RED}--kind conflicts with --scenario:{NC} "
                f"{scenario.id} requires {scenario.canonical_schema}"
            )
            sys.exit(2)
        requested_modality = "image" if args.modality == "vlm" else args.modality
        if requested_modality and requested_modality != scenario.modality:
            print(
                f"{RED}--modality conflicts with --scenario:{NC} "
                f"{scenario.id} requires {scenario.modality}"
            )
            sys.exit(2)
    if args.accept_recommended and scenario is None:
        print(f"{RED}--accept-recommended requires --scenario.{NC}")
        sys.exit(2)
    if args.accept_recommended and scenario is not None:
        import copy

        for target, value in scenario.safe_constants.items():
            mapping.setdefault(target, {"kind": "constant", "value": copy.deepcopy(value)})
        missing = [target for target in scenario.required_fields if target not in mapping]
        if missing:
            print(
                f"{RED}Confirmed mapping is incomplete:{NC} "
                + ", ".join(missing)
                + ". Use repeatable --map or run `halo-forge data inspect` first."
            )
            sys.exit(2)
    source = {
        "kind": "local" if args.path else "huggingface",
        "uri": args.path or args.hf_id,
        "config": args.config,
        "split": args.split,
        "revision": args.revision,
    }
    if scenario is not None and args.accept_recommended:
        import copy

        recipe = copy.deepcopy(scenario.default_recipe)
        recipe["steps"][0]["fields"] = copy.deepcopy(mapping)
        mapping_v2 = {
            target: (
                value
                if isinstance(value, dict)
                else {"kind": "direct", "source": str(value)}
            )
            for target, value in mapping.items()
        }
        source["metadata"] = {
            "guided_own_data": {
                "format_version": 1,
                "scenario_revision_id": scenario.revision_id,
                "field_mapping": copy.deepcopy(mapping),
                "preparation_plan": {
                    "scenario_revision_id": scenario.revision_id,
                    "mapping_plan": {
                        "version": 2,
                        "scenario_revision_id": scenario.revision_id,
                        "confirmed": True,
                        "mappings": mapping_v2,
                    },
                    "recipe": recipe,
                    "sampled": True,
                    "warnings": [
                        "This plan was explicitly accepted from CLI mappings; exact counts are published by the build."
                    ],
                },
            }
        }
    service, lab = _dataset_lab_service(args)
    try:
        result = service.create_dataset(
            {
                "name": args.name,
                "canonical_schema": scenario.canonical_schema if scenario else args.kind,
                "modality": scenario.modality if scenario else args.modality,
                "field_mapping": mapping,
                "scenario_revision_id": scenario.revision_id if scenario else None,
                "accept_recommended": bool(args.accept_recommended),
                "source": source,
            }
        )
    except Exception as exc:
        print(f"{RED}Dataset registration failed:{NC} {exc}")
        sys.exit(1)
    finally:
        lab.close()
    _print_dataset_lab_payload(result)


def cmd_data_lab_scenarios(args):
    """Inspect the same versioned own-data scenarios used by the dashboard."""
    from pathlib import Path

    from halo_forge.own_data.registry import TRAINING_SCENARIOS

    action = str(args.scenarios_action)
    service, lab = _guided_own_data_service(args)
    backend_name = _guided_active_backend_name()
    try:
        if action == "list":
            _print_dataset_lab_payload(
                service.list_scenarios(
                    backend_name=backend_name,
                    include_unavailable=bool(args.include_unavailable),
                    modality=args.modality,
                    limit=1000,
                    offset=0,
                )
            )
            return
        if action == "advise":
            fields = []
            for value in args.field or []:
                fields.extend(
                    part.strip() for part in str(value).split(",") if part.strip()
                )
            payload = {
                "goal": args.goal,
                "modality": args.modality,
                "source_layout": args.source_layout,
                "source_fields": fields,
                "include_unavailable": bool(args.include_unavailable),
            }
            _print_dataset_lab_payload(
                service.scenario_advice(payload, backend_name=backend_name)
            )
            return
        scenario = TRAINING_SCENARIOS.get(args.scenario)
        if action == "show":
            _print_dataset_lab_payload(
                service.get_scenario(
                    scenario.revision_id,
                    backend_name=backend_name,
                    include_examples=True,
                )
            )
            return
        filename, fixture_files = TRAINING_SCENARIOS.template_files(
            scenario.revision_id, example_id=args.example
        )
        content = fixture_files[filename]
        if args.output:
            output = Path(args.output).expanduser()
            written = []
            if len(fixture_files) > 1:
                root = output if output.is_dir() or not output.suffix else output.parent
                manifest_output = root / filename if root == output else output
                root.mkdir(parents=True, exist_ok=True)
                for relative_path, file_content in fixture_files.items():
                    target = (
                        manifest_output if relative_path == filename else root / relative_path
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(file_content)
                    written.append(str(target.resolve()))
                output = manifest_output
            else:
                if output.is_dir():
                    output = output / filename
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(content)
                written.append(str(output.resolve()))
            _print_dataset_lab_payload(
                {
                    "scenario_id": scenario.id,
                    "scenario_revision_id": scenario.revision_id,
                    "path": str(output.resolve()),
                    "size_bytes": len(content),
                    "files": written,
                }
            )
        elif args.json:
            import base64
            import hashlib

            _print_dataset_lab_payload(
                {
                    "scenario_id": scenario.id,
                    "scenario_revision_id": scenario.revision_id,
                    "filename": filename,
                    "content": content.decode("utf-8"),
                    "files": [
                        {
                            "path": path,
                            "size_bytes": len(file_content),
                            "sha256": hashlib.sha256(file_content).hexdigest(),
                            "content_base64": (
                                None
                                if path == filename
                                else base64.b64encode(file_content).decode("ascii")
                            ),
                        }
                        for path, file_content in fixture_files.items()
                    ],
                }
            )
        else:
            if len(fixture_files) > 1:
                raise ValueError(
                    "This multimodal fixture includes media assets; use --output DIRECTORY."
                )
            sys.stdout.write(content.decode("utf-8"))
    except (KeyError, OSError, ValueError) as exc:
        print(f"{RED}Scenario command failed:{NC} {exc}")
        sys.exit(1)
    finally:
        lab.close()


def _guided_import_source_payload(args):
    selected = [
        bool(getattr(args, "path", None)),
        bool(getattr(args, "hf_id", None)),
        bool(getattr(args, "import_id", None)),
    ]
    if sum(selected) != 1:
        raise ValueError("Choose exactly one of --path, --hf-id, or --import-id.")
    if getattr(args, "import_id", None):
        return None
    if getattr(args, "hf_id", None):
        return {
            "source_kind": "huggingface",
            "source_uri": args.hf_id,
            "name": getattr(args, "name", None) or args.hf_id,
            "config": getattr(args, "config", None),
            "split": getattr(args, "split", None) or "train",
            "revision": getattr(args, "revision", None),
            "scenario_revision_id": getattr(args, "scenario", None),
            "capacity_override_reason": getattr(
                args, "capacity_override_reason", None
            ),
        }
    return {
        "source_kind": (
            "upload" if bool(getattr(args, "managed", False)) else "workstation_path"
        ),
        "source_uri": args.path,
        "name": getattr(args, "name", None),
        "scenario_revision_id": getattr(args, "scenario", None),
        "capacity_override_reason": getattr(args, "capacity_override_reason", None),
    }


def _upload_cli_source(engine, import_id: str, source_value: str) -> None:
    """Stream a local CLI source into the same resumable managed upload store."""
    import hashlib
    import mimetypes
    from pathlib import Path

    source = Path(source_value).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_symlink():
        raise ValueError("symbolic-link dataset sources are not accepted")
    if source.is_file():
        files = [(source.name, source)]
    else:
        files = []
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError(
                    f"unsafe symbolic link in dataset source: {path.relative_to(source)}"
                )
            if path.is_file():
                files.append((path.relative_to(source).as_posix(), path))
    if not files:
        raise ValueError("the selected source contains no files")
    for relative_path, path in files:
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"empty upload files are not supported: {relative_path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        file_record = engine.create_import_file(
            import_id,
            {
                "relative_path": relative_path,
                "size_bytes": size,
                "content_type": mimetypes.guess_type(relative_path)[0],
                "content_hash": digest.hexdigest(),
            },
        )
        start = 0
        with path.open("rb") as handle:
            while start < size:
                content = handle.read(min(4 * 1024 * 1024, size - start))
                end = start + len(content) - 1
                engine.upload_chunk(
                    import_id,
                    str(file_record["id"]),
                    content,
                    start=start,
                    end=end,
                    total=size,
                    chunk_sha256=hashlib.sha256(content).hexdigest(),
                )
                start = end + 1


def _execute_cli_inspection(engine, import_id: str, *, scenario: str | None, force: bool):
    from halo_forge.workstation_jobs import WorkstationWorker

    requested = engine.request_inspection(
        import_id, scenario_revision_id=scenario, force=force
    )
    inspection = requested["inspection"]
    if inspection.get("status") != "completed":
        work_item_id = str(requested.get("work_item_id") or "").strip()
        if not work_item_id or engine.scheduler is None:
            raise RuntimeError("inspection was not durably scheduled")
        terminal = WorkstationWorker(engine.scheduler).run_once(
            work_item_id=work_item_id
        )
        if terminal is None:
            current = engine.db.get_work_item(work_item_id)
            state = str(current.status if current is not None else "missing")
            reason = str(
                (current.error if current is not None else None)
                or (
                    (current.progress or {}).get("blocked_reason")
                    if current is not None
                    else None
                )
                or "the work item is not ready"
            )
            raise RuntimeError(f"inspection work is {state}: {reason}")
        if terminal.status not in {"completed", "succeeded"}:
            raise RuntimeError(
                f"inspection work ended as {terminal.status}: "
                f"{terminal.error or (terminal.progress or {}).get('blocked_reason') or 'no result was published'}"
            )
        inspection = engine.get_inspection(str(inspection["id"]))
        if inspection is None or inspection.get("status") != "completed":
            raise RuntimeError("inspection work completed without a published inspection")
    return {
        "import": engine.get_import(import_id),
        "inspection": inspection,
        "work_item_id": requested.get("work_item_id"),
        "reused": bool(requested.get("reused", False)),
    }


def cmd_data_lab_import(args):
    """Create a persistent referenced, managed-upload, or Hugging Face import."""
    engine, lab = _guided_own_data_service(args)
    try:
        if args.managed and not args.path:
            raise ValueError("--managed can only be used with --path")
        source_payload = _guided_import_source_payload(args)
        if source_payload is None:
            import_id = str(args.import_id)
            session = engine.get_import(import_id)
            if session is None:
                raise KeyError(import_id)
        else:
            managed_path = source_payload.pop("source_uri", None) if args.managed else None
            if managed_path is not None:
                # A managed CLI import copies through the exact same checksummed
                # chunk contract as a browser upload.
                source_payload.pop("source_uri", None)
            session = engine.create_import(source_payload)
            import_id = str(session["id"])
            if managed_path is not None:
                _upload_cli_source(engine, import_id, str(managed_path))
                session = engine.get_import(import_id)
        if args.no_inspect:
            _print_dataset_lab_payload(session)
        else:
            _print_dataset_lab_payload(
                _execute_cli_inspection(
                    engine,
                    import_id,
                    scenario=getattr(args, "scenario", None),
                    force=bool(getattr(args, "force", False)),
                )
            )
    except Exception as exc:
        print(f"{RED}Dataset import failed:{NC} {exc}")
        sys.exit(1)
    finally:
        lab.close()


def cmd_data_lab_inspect(args):
    """Stream and persist a full-source inspection without registering data."""
    engine, lab = _guided_own_data_service(args)
    try:
        source_payload = _guided_import_source_payload(args)
        if source_payload is None:
            import_id = str(args.import_id)
            if engine.get_import(import_id) is None:
                raise KeyError(import_id)
        else:
            # `data inspect` is intentionally non-destructive: workstation
            # paths stay referenced. Managed copying belongs to `data import`.
            source_payload["source_kind"] = (
                "huggingface"
                if source_payload["source_kind"] == "huggingface"
                else "workstation_path"
            )
            session = engine.create_import(source_payload)
            import_id = str(session["id"])
        _print_dataset_lab_payload(
            _execute_cli_inspection(
                engine,
                import_id,
                scenario=getattr(args, "scenario", None),
                force=bool(getattr(args, "force", False)),
            )
        )
    except Exception as exc:
        print(f"{RED}Dataset inspection failed:{NC} {exc}")
        sys.exit(1)
    finally:
        lab.close()


def cmd_data_lab_extract(args):
    """Extract canonical corpus documents with durable identity and checksums."""

    service, lab = _dataset_lab_service(args)
    engine = service._corpus_extraction_engine()
    try:
        if args.list:
            _print_dataset_lab_payload(
                service.list_document_extractions(
                    import_id=args.import_id,
                    source_id=args.source_id,
                    status=args.status,
                    limit=args.limit,
                    offset=args.offset,
                )
            )
            return
        if args.extraction:
            if args.preview:
                result = service.preview_document_extraction(
                    args.extraction,
                    limit=args.limit,
                    offset=args.offset,
                    include_text=not args.no_text,
                )
            elif args.verify:
                result = service.verify_document_extraction(args.extraction)
            elif args.cancel:
                result = service.cancel_document_extraction(args.extraction)
            elif args.retry:
                result = engine.retry(
                    args.extraction, synchronous=not args.no_wait
                )
            else:
                result = service.get_document_extraction(args.extraction)
            _print_dataset_lab_payload(result)
            return
        selected = [
            bool(args.path),
            bool(args.import_id),
            bool(args.source_id),
        ]
        if sum(selected) != 1:
            raise ValueError(
                "choose exactly one of --path, --import-id, or --source-id"
            )
        config = {
            "text_columns": args.text_column or [],
            "title_column": args.title_column,
            "id_column": args.id_column,
            "metadata_columns": args.metadata_column or [],
            "min_text_chars": args.min_text_chars,
            "include_hidden": bool(args.include_hidden),
            "pdf_page_documents": not args.single_pdf_document,
        }
        result = engine.launch(
            args.path,
            import_id=args.import_id,
            source_id=args.source_id,
            config=config,
            synchronous=not args.no_wait,
        )
        _print_dataset_lab_payload(result)
    except Exception as exc:
        print(f"{RED}Document extraction failed:{NC} {exc}")
        sys.exit(1)
    finally:
        lab.close()


def cmd_data_lab_corpus_profile(args):
    """Show exact document and split statistics for a corpus version."""

    service, lab = _dataset_lab_service(args)
    try:
        _print_dataset_lab_payload(service.corpus_profile(args.version))
    except Exception as exc:
        print(f"{RED}Corpus profile failed:{NC} {exc}")
        sys.exit(1)
    finally:
        lab.close()


def cmd_data_lab_list(args):
    service, lab = _dataset_lab_service(args)
    try:
        _print_dataset_lab_payload(service.list_datasets())
    finally:
        lab.close()


def cmd_data_lab_show(args):
    service, lab = _dataset_lab_service(args)
    try:
        payload = service.get_dataset(args.dataset)
        if payload is None:
            payload = service.get_dataset_source(args.dataset)
        if payload is None:
            raise KeyError(args.dataset)
        if args.profile:
            dataset_id = payload.get("dataset_id") or payload.get("id")
            payload["profile"] = service.dataset_statistics(str(dataset_id))
        _print_dataset_lab_payload(payload)
    except Exception as exc:
        print(f"{RED}Dataset lookup failed:{NC} {exc}")
        sys.exit(1)
    finally:
        lab.close()


def cmd_data_lab_build(args):
    service, lab = _dataset_lab_service(args)
    try:
        from halo_forge.data_lab import Recipe

        if args.recommended_recipe:
            dataset = service.get_dataset(args.dataset)
            if dataset is None:
                raise KeyError(args.dataset)
            recipe_value = None
            for source in dataset.get("sources") or []:
                guided = (source.get("metadata") or {}).get("guided_own_data") or {}
                preparation = guided.get("preparation_plan") or {}
                if preparation.get("recipe"):
                    recipe_value = preparation["recipe"]
                    break
            if recipe_value is None:
                raise ValueError(
                    "this dataset has no confirmed guided preparation plan; provide --recipe"
                )
            recipe = Recipe.from_value(recipe_value)
        else:
            recipe = Recipe.from_value(args.recipe)
        started = service.build_dataset(
            args.dataset,
            {"recipe": recipe.to_dict(), "materialize_assets": args.materialize_assets},
        )
        lab.job_manager.wait(started["id"])
        completed = service.get_dataset_job(started["id"])
        if not completed or completed.get("status") != "completed":
            raise RuntimeError((completed or {}).get("error") or "dataset build did not complete")
        result = service.get_dataset_version(str(completed["version_id"]))
    except Exception as exc:
        print(f"{RED}Dataset build failed:{NC} {exc}")
        sys.exit(1)
    finally:
        lab.close()
    _print_dataset_lab_payload(result)


def cmd_data_lab_versions(args):
    service, lab = _dataset_lab_service(args)
    try:
        if args.dataset:
            result = service.list_dataset_versions(args.dataset)
        else:
            items = []
            for dataset in service.list_datasets(limit=500)["items"]:
                items.extend(service.list_dataset_versions(dataset["id"])["items"])
            result = {"items": items}
        _print_dataset_lab_payload(result)
    finally:
        lab.close()


def cmd_data_lab_preview(args):
    service, lab = _dataset_lab_service(args)
    try:
        if service.get_dataset(args.dataset) is not None:
            page = service.preview_dataset(args.dataset, offset=args.offset, limit=args.limit)
        elif service.get_dataset_version(args.dataset) is not None:
            page = service.preview_dataset_version(
                args.dataset, split=args.split, offset=args.offset, limit=args.limit
            )
        else:
            raise KeyError(args.dataset)
        _print_dataset_lab_payload(page)
    except Exception as exc:
        print(f"{RED}Dataset preview failed:{NC} {exc}")
        sys.exit(1)
    finally:
        lab.close()


def cmd_data_lab_export(args):
    service, lab = _dataset_lab_service(args)
    try:
        result = service.export_dataset_version(
            args.version, {"output": args.output, "split": args.split, "format": args.format}
        )
    except Exception as exc:
        print(f"{RED}Dataset export failed:{NC} {exc}")
        sys.exit(1)
    finally:
        lab.close()
    _print_dataset_lab_payload({"version_id": args.version, **result})


def cmd_data_lab_materialize(args):
    service, lab = _dataset_lab_service(args)
    try:
        started = service.materialize_dataset_version(args.version, {})
        lab.job_manager.wait(started["id"])
        result = service.get_dataset_job(started["id"])
    except Exception as exc:
        print(f"{RED}Dataset materialization failed:{NC} {exc}")
        sys.exit(1)
    finally:
        lab.close()
    _print_dataset_lab_payload(result)


def cmd_data_lab_jobs(args):
    service, lab = _dataset_lab_service(args)
    try:
        if args.cancel:
            result = service.cancel_dataset_job(args.cancel)
        elif args.retry:
            result = service.retry_dataset_job(args.retry)
        elif args.job:
            result = service.get_dataset_job(args.job)
        else:
            result = service.list_dataset_jobs()
        _print_dataset_lab_payload(result)
    finally:
        lab.close()


def cmd_data_lab_render(args):
    """Render and catalog a content-addressed trainer artifact."""
    service, lab = _dataset_lab_service(args)
    try:
        bindings = [_parse_dataset_binding(value) for value in (args.binding or [])]
        if args.version:
            if any(binding["role"] == "train" for binding in bindings):
                raise ValueError("version shorthand conflicts with an explicit train binding")
            bindings.insert(
                0,
                {"role": "train", "dataset_version_id": args.version, "split": args.split},
            )
        artifact = lab.render_training_artifact(
            bindings,
            trainer_mode=args.trainer,
            adapter_id=args.adapter,
            model=args.model,
            model_revision=args.model_revision,
            model_hash=args.model_hash,
            tokenizer_revision=args.tokenizer_revision,
            tokenizer_hash=args.tokenizer_hash,
            max_sequence_length=args.max_seq_length,
            packing=args.packing,
            budget_mode=args.budget_mode,
            target_tokens=args.target_tokens,
            corpus_passes=args.corpus_passes,
            effective_batch_size=args.effective_batch_size,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        db_bindings = _artifact_db_bindings(artifact)
        service._dataset_database().create_training_artifact(
            artifact_id=artifact.artifact_id,
            artifact_hash=artifact.artifact_hash,
            adapter_id=artifact.adapter_id,
            adapter_version=artifact.adapter_version,
            trainer_mode=artifact.trainer_mode,
            model_id=artifact.model,
            tokenizer_revision=artifact.tokenizer_revision,
            chat_template_hash=artifact.chat_template_hash,
            manifest_path=str(Path(artifact.path) / "manifest.json"),
            bindings=db_bindings,
            metadata=artifact.to_dict(),
        )
        _print_dataset_lab_payload(artifact)
    except Exception as exc:
        print(f"{RED}Dataset rendering failed:{NC} {exc}")
        sys.exit(1)
    finally:
        lab.close()


def cmd_data_lab_compare(args):
    """Compare immutable versions by stable record identity."""
    lab = _dataset_lab(args)
    try:
        _print_dataset_lab_payload(lab.compare_versions(args.left, args.right))
    except Exception as exc:
        print(f"{RED}Dataset comparison failed:{NC} {exc}")
        sys.exit(1)
    finally:
        lab.close()


def cmd_data_lab_mine(args):
    """Preview or build a reviewed dataset from evaluation evidence."""

    service, lab = _dataset_lab_service(args)
    selector = {
        "kind": args.selector,
        "task": args.task,
        "category": args.category,
        "failure_reason": args.failure_reason,
        "min_score": args.min_score,
        "max_score": args.max_score,
        "min_reward": args.min_reward,
        "max_reward": args.max_reward,
    }
    selector = {key: value for key, value in selector.items() if value is not None}
    payload = {
        "base_id": args.base,
        "candidate_id": args.candidate,
        "selector": selector,
        "excluded_record_ids": list(args.exclude or []),
    }
    try:
        if bool(args.dataset) != bool(args.parent_version):
            raise ValueError("--dataset and --parent-version must be supplied together")
        if not args.dataset:
            result = service.preview_failure_mining(payload)
        else:
            started = service.build_failure_mined_dataset(
                {
                    **payload,
                    "dataset_id": args.dataset,
                    "parent_version_id": args.parent_version,
                    "target_split": args.target_split,
                    "mode": args.mode,
                    "materialize_assets": args.materialize_assets,
                }
            )
            lab.job_manager.wait(str(started["id"]))
            result = service.get_dataset_job(str(started["id"]))
            if not result or result.get("status") != "completed":
                raise RuntimeError(
                    (result or {}).get("error") or "failure-mining build did not complete"
                )
        _print_dataset_lab_payload(result)
    except Exception as exc:
        print(f"{RED}Dataset failure mining failed:{NC} {exc}")
        sys.exit(1)
    finally:
        lab.close()


def _review_public_service(args):
    """Create the SQLite-backed Review Studio service used by API and dashboard."""

    from halo_forge.public_api.service import PublicApiService
    from halo_forge.run_db import get_database

    root = getattr(args, "root", None) or os.environ.get("HALOFORGE_REVIEW_ROOT")
    return PublicApiService(
        database=get_database(getattr(args, "database", None)),
        review_storage_root=(Path(root).expanduser() if root else None),
    )


def _review_json(value: Optional[str], *, default: Any, label: str) -> Any:
    if not value:
        return default
    candidate = Path(value).expanduser()
    raw = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml

            return yaml.safe_load(raw)
        except Exception as exc:
            raise ValueError(f"Invalid {label} JSON/YAML: {exc}") from exc


def _review_records(value: Optional[str]) -> list[dict[str, Any]]:
    if not value:
        return []
    path = Path(value).expanduser()
    if path.is_file() and path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"records line {line_number} must be an object")
                rows.append(dict(row))
        return rows
    value_object = _review_json(value, default=[], label="records")
    if isinstance(value_object, dict):
        value_object = value_object.get("records") or value_object.get("items") or []
    if not isinstance(value_object, list) or not all(isinstance(row, dict) for row in value_object):
        raise ValueError("records must be a JSON/JSONL array of objects")
    return [dict(row) for row in value_object]


def cmd_review(args):
    """Operate the Human Feedback and Active Data Studio through one service layer."""

    service = _review_public_service(args)
    try:
        command = args.review_command
        if command == "capabilities":
            result = service.get_review_capabilities()
        elif command == "schema":
            action = args.schema_action
            if action == "list":
                result = service.list_annotation_schemas(limit=args.limit, offset=args.offset)
            elif action == "show":
                result = (
                    service.get_annotation_schema_revision(args.identifier)
                    if args.revision
                    else service.get_annotation_schema(args.identifier)
                )
                if result is None:
                    raise KeyError(args.identifier)
            elif action == "validate":
                result = service.validate_annotation_schema(
                    {
                        "name": args.name or "Validation only",
                        "modality": args.modality,
                        "task_type": args.task_type,
                        "definition": _review_json(
                            args.definition, default={}, label="annotation definition"
                        ),
                    }
                )
            elif action == "create":
                result = service.create_annotation_schema(
                    {
                        "schema_id": args.schema_id,
                        "name": args.name,
                        "description": args.description,
                        "modality": args.modality,
                        "task_type": args.task_type,
                        "definition": _review_json(
                            args.definition, default={}, label="annotation definition"
                        ),
                    }
                )
            else:
                result = service.revise_annotation_schema(
                    args.schema_id,
                    {
                        "modality": args.modality,
                        "task_type": args.task_type,
                        "definition": _review_json(
                            args.definition, default={}, label="annotation definition"
                        ),
                    },
                )
        elif command == "acquire":
            action = args.acquire_action
            if action == "list":
                result = service.list_acquisition_batches(
                    status=args.status, limit=args.limit, offset=args.offset
                )
            elif action == "create":
                if args.spec:
                    payload = _review_json(args.spec, default={}, label="acquisition spec")
                else:
                    strategies = []
                    for raw_strategy in args.strategy:
                        parsed = _review_json(
                            raw_strategy, default={}, label="acquisition strategy"
                        )
                        value = {"kind": parsed} if isinstance(parsed, str) else dict(parsed)
                        if args.quota is not None and "quota" not in value:
                            value["quota"] = args.quota
                        options = dict(value.get("options") or {})
                        if args.embedding_revision:
                            options.setdefault("embedding_revision", args.embedding_revision)
                        if args.score_direction:
                            options.setdefault("direction", args.score_direction)
                        if options:
                            value["options"] = options
                        strategies.append(value)
                    source = None
                    if args.source_kind:
                        reference = args.candidate_evaluation or args.source_ref
                        source = {
                            "kind": args.source_kind,
                            "ref": reference,
                            "split": args.source_split,
                            "base_id": args.base_evaluation,
                            "candidate_id": args.candidate_evaluation,
                            "selector": args.failure_selector,
                            "options": _review_json(
                                args.selector_options,
                                default={},
                                label="verifier failure selector options",
                            ),
                        }
                    payload = {
                        "name": args.name,
                        "seed": args.seed,
                        "strategies": strategies or None,
                    }
                    if args.records:
                        payload["records"] = _review_records(args.records)
                    elif source is not None:
                        payload["sources"] = [source]
                result = service.create_acquisition_batch(payload)
            elif action == "show":
                result = service.get_acquisition_batch(args.batch_id)
                if result is None:
                    raise KeyError(args.batch_id)
                if args.candidates:
                    result["candidates"] = service.list_acquisition_candidates(
                        args.batch_id, limit=args.limit, offset=args.offset
                    )
            elif action == "cancel":
                result = service.cancel_acquisition_batch(args.batch_id)
            else:
                result = service.retry_acquisition_batch(args.batch_id)
        elif command == "queue":
            action = args.queue_action
            if action == "list":
                result = service.list_review_queues(
                    status=args.status, limit=args.limit, offset=args.offset
                )
            elif action == "show":
                result = service.get_review_queue(args.queue_id)
                if result is None:
                    raise KeyError(args.queue_id)
            elif action == "create":
                result = service.create_review_queue(
                    {
                        "batch_id": args.batch,
                        "schema_revision_id": args.schema,
                        "name": args.name,
                        "policy": _review_json(args.policy, default={}, label="review policy"),
                    }
                )
            elif action == "clone":
                result = service.clone_review_queue(
                    args.queue_id,
                    {
                        "name": args.name,
                        "batch_id": args.batch,
                        "schema_revision_id": args.schema,
                        "policy": (
                            _review_json(args.policy, default={}, label="review policy")
                            if args.policy
                            else None
                        ),
                    },
                )
            else:
                result = service.update_review_queue_state(
                    args.queue_id, action=action, reason=getattr(args, "reason", None)
                )
        elif command == "items":
            result = service.list_review_items(
                args.queue_id,
                status=args.status,
                pass_number=args.pass_number,
                limit=args.limit,
                offset=args.offset,
            )
        elif command == "item":
            result = service.get_review_item(args.item_id)
            if result is None:
                raise KeyError(args.item_id)
        elif command in {"submit", "correct", "exclude", "flag"}:
            if command == "submit":
                event_type = args.event_type
                payload = _review_json(args.label, default={}, label="label")
            elif command == "correct":
                event_type = "correct"
                correction = _review_json(args.label, default={}, label="label")
                if not isinstance(correction, dict):
                    raise ValueError("corrected label must be a JSON/YAML object")
                payload = {**correction, "reason": args.reason}
            elif command == "exclude":
                event_type = "exclude"
                payload = {"reason": args.reason}
            else:
                event_type = "flag"
                payload = {"reason": args.reason}
            result = service.submit_review_event(
                args.item_id,
                {
                    "event_type": event_type,
                    "payload": payload,
                    "idempotency_key": args.idempotency_key or f"cli-{uuid.uuid4().hex}",
                    "expected_active_event_id": args.expected_active_event,
                    "pass_number": args.pass_number,
                    "supersedes_event_id": getattr(args, "supersedes_event", None),
                },
            )
        elif command == "adjudicate":
            result = service.adjudicate_review_item(
                args.item_id,
                {
                    "payload": _review_json(args.label, default={}, label="adjudicated label"),
                    "reason": args.reason,
                    "idempotency_key": args.idempotency_key or f"cli-{uuid.uuid4().hex}",
                    "expected_active_event_id": args.expected_active_event,
                },
            )
        elif command == "suggestions":
            if args.suggestions_action == "generate":
                _prepare_profile_verifier(args, consumer="direct")
                result = service.generate_review_suggestions(
                    args.item_id,
                    {
                        "provider": args.provider,
                        "model": args.model,
                        "prompt": args.prompt,
                        "pass_number": args.pass_number,
                        "output": (
                            _review_json(args.output, default=None, label="suggestion output")
                            if args.output
                            else None
                        ),
                        "parameters": _review_json(
                            args.parameters, default={}, label="suggestion parameters"
                        ),
                        "verifier_profile_revision_id": getattr(
                            args, "verifier_profile_revision", None
                        ),
                        "provenance": (
                            {"resolved_verifier": getattr(args, "_verifier_replay_binding", None)}
                            if getattr(args, "verifier_profile_revision", None)
                            else {}
                        ),
                    },
                )
                if isinstance(result, dict) and result.get("id"):
                    _bind_profile_verifier(
                        args,
                        domain_kind="review_suggestion",
                        domain_id=str(result["id"]),
                        role="suggestion_verifier",
                    )
            else:
                result = service.list_review_suggestions(
                    args.item_id,
                    pass_number=args.pass_number,
                    limit=args.limit,
                    offset=args.offset,
                )
        elif command == "stats":
            result = service.get_review_queue_statistics(args.queue_id)
        elif command == "label-set":
            action = args.label_set_action
            if action == "list":
                result = service.list_label_sets(limit=args.limit, offset=args.offset)
            elif action == "show":
                result = service.get_label_set_revision(args.identifier)
                if result is None:
                    result = service.get_label_set(args.identifier)
                if result is None:
                    raise KeyError(args.identifier)
            elif action == "publish":
                result = service.publish_label_set(
                    args.queue,
                    {
                        "name": args.name,
                        "output_adapter_id": args.adapter,
                        "build_mode": args.mode,
                    },
                )
            elif action == "verify":
                result = service.verify_label_set_revision(args.revision_id)
            elif action == "preview":
                result = service.preview_label_set_dataset(
                    args.revision_id,
                    {
                        "output_adapter_id": args.adapter,
                        "build_mode": args.mode,
                        "dataset_id": args.dataset,
                        "parent_version_id": args.parent_version,
                        "target_split": args.target_split,
                    },
                )
            else:
                result = service.build_label_set_dataset(
                    args.revision_id,
                    {
                        "output_adapter_id": args.adapter,
                        "build_mode": args.mode,
                        "dataset_id": args.dataset,
                        "parent_version_id": args.parent_version,
                        "target_split": args.target_split,
                        "name": args.name,
                        "materialize_assets": args.materialize_assets,
                    },
                )
        else:  # pragma: no cover - argparse keeps this unreachable
            raise ValueError(f"unknown review command: {command}")
        _print_dataset_lab_payload(result)
    except Exception as exc:
        if getattr(args, "json", False):
            print(json.dumps({"error": str(exc), "type": type(exc).__name__}))
        else:
            print(f"{RED}Review Studio operation failed:{NC} {exc}", file=sys.stderr)
        raise SystemExit(1)


def _load_cli_mapping(value: str, *, label: str) -> dict[str, Any]:
    """Load a JSON/YAML mapping from an inline value or a file path."""

    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{label} cannot be empty")
    path = Path(raw).expanduser()
    text = path.read_text(encoding="utf-8") if path.is_file() else raw
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        import yaml

        parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON/YAML object")
    return dict(parsed)


def _parse_cli_ints(values: Optional[List[str]], *, label: str) -> Optional[list[int]]:
    if not values:
        return None
    parsed: list[int] = []
    for value in values:
        for token in str(value).replace(",", " ").split():
            try:
                parsed.append(int(token))
            except ValueError as exc:
                raise ValueError(f"{label} must contain integers; got {token!r}") from exc
    if not parsed:
        raise ValueError(f"{label} cannot be empty")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{label} must not contain duplicates")
    return parsed


def _experiment_runtime(args: Any):
    from halo_forge.orchestration import ExperimentOrchestrationService
    from halo_forge.run_db import get_database
    from halo_forge.workstation_jobs import WorkstationScheduler

    database = get_database(getattr(args, "database", None))
    scheduler = WorkstationScheduler(database)
    return database, scheduler, ExperimentOrchestrationService(database, scheduler=scheduler)


def _adaptive_public_service(database: Any, scheduler: Any):
    """Return the shared adaptive/evidence service used by CLI and dashboard."""

    from halo_forge.public_api.service import PublicApiService

    return PublicApiService(database=database, workstation_scheduler=scheduler)


def _experiment_emit(payload: Any, *, as_json: bool, kind: str) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    if kind == "group-list":
        rows = list(payload)
        if not rows:
            print("No experiment groups.")
            return
        for row in rows:
            print(
                f"{row['id']}  {row.get('status', 'unknown'):<18} "
                f"{row.get('kind', 'group'):<7} {row.get('name', '')}"
            )
        return
    if kind == "work-list":
        rows = list(payload)
        if not rows:
            print("No workstation jobs.")
            return
        for row in rows:
            position = row.get("queue_position")
            suffix = f" queue={position}" if position is not None else ""
            print(
                f"{row['id']}  {row.get('status', 'unknown'):<12} "
                f"{row.get('kind', 'work')}{suffix}"
            )
        return
    if kind == "comparison":
        print(
            f"{payload['metric']} ({payload['direction']}): "
            f"right-left={payload.get('right_minus_left')} winner={payload.get('winner')}"
        )
        return
    if kind == "fork":
        print("Resolved fork payload (not launched):")
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    if kind == "worker":
        completed = payload.get("completed_work_item")
        print(
            f"Worker recovery: adopted={len(payload.get('adopted', []))} "
            f"interrupted={len(payload.get('interrupted', []))}"
        )
        print(f"Completed work item: {completed.get('id') if completed else 'none'}")
        return
    if isinstance(payload, dict) and "name" in payload:
        print(
            f"{payload.get('name')} ({payload.get('id')})\n"
            f"Status: {payload.get('status')}  Trials: {len(payload.get('trials') or [])}  "
            f"Work items: {len(payload.get('work_items') or [])}"
        )
        return
    if isinstance(payload, dict) and "id" in payload:
        print(f"{payload['id']}  {payload.get('status', 'unknown')}  " f"{payload.get('kind', '')}")
        return
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _experiment_failure(args: Any, exc: Exception) -> None:
    if getattr(args, "json", False):
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}))
    else:
        print(f"{RED}Experiment operation failed:{NC} {exc}", file=sys.stderr)
    raise SystemExit(1)


def _sweep_create_payload(args: Any) -> dict[str, Any]:
    loaded = (
        _load_cli_mapping(args.spec, label="run-group spec") if getattr(args, "spec", None) else {}
    )
    if isinstance(loaded.get("spec"), dict):
        payload = {key: value for key, value in loaded.items() if key != "spec"}
        spec = dict(loaded["spec"])
        nested = True
    else:
        payload = dict(loaded)
        spec = payload
        nested = False

    if not loaded and getattr(args, "kind", None) is None:
        spec["kind"] = "repeat"
    for cli_name, spec_name in (
        ("name", "name"),
        ("kind", "kind"),
        ("trials", "n_trials"),
        ("sampler", "sampler"),
        ("sampler_seed", "sampler_seed"),
    ):
        value = getattr(args, cli_name, None)
        if value is not None:
            spec[spec_name] = value

    seeds = _parse_cli_ints(getattr(args, "seeds", None), label="seeds")
    if seeds is not None:
        spec["seeds"] = seeds

    base_config = dict(spec.get("base_config") or {})
    if getattr(args, "base_config", None):
        base_config.update(_load_cli_mapping(args.base_config, label="base configuration"))
    for cli_name, config_name in (
        ("model", "model"),
        ("backend", "backend"),
        ("output_root", "output_root"),
        ("max_steps", "max_steps"),
        ("cycles", "cycles"),
    ):
        value = getattr(args, cli_name, None)
        if value is not None:
            base_config[config_name] = value
    if base_config or "base_config" in spec:
        spec["base_config"] = base_config

    if getattr(args, "search_space", None):
        spec["search_space"] = _load_cli_mapping(args.search_space, label="search space")

    pruning = dict(spec.get("pruning") or {})
    if getattr(args, "pruning", None):
        pruning.update(_load_cli_mapping(args.pruning, label="pruning policy"))
    if getattr(args, "prune", None) is not None:
        pruning["enabled"] = bool(args.prune)
    budgets = _parse_cli_ints(getattr(args, "budgets", None), label="budgets")
    if budgets is not None:
        pruning["budgets"] = budgets
        pruning.setdefault("enabled", True)
    if getattr(args, "reduction_factor", None) is not None:
        pruning["reduction_factor"] = args.reduction_factor
    if pruning or "pruning" in spec:
        spec["pruning"] = pruning

    if nested:
        payload["spec"] = spec

    if getattr(args, "trainer", None) is not None:
        payload["trainer_mode"] = args.trainer
    if getattr(args, "suite", None) is not None:
        payload["development_suite_revision_id"] = args.suite
    if getattr(args, "holdout_suite", None) is not None:
        payload["holdout_suite_revision_id"] = args.holdout_suite
    if getattr(args, "priority", None) is not None:
        payload["priority"] = args.priority
    if getattr(args, "max_retries", None) is not None:
        payload["max_retries"] = args.max_retries
    if getattr(args, "checkpoint_policy", None) is not None:
        payload["checkpoint_policy_revision_id"] = args.checkpoint_policy

    raw_bindings = list(getattr(args, "dataset_binding", None) or [])
    if getattr(args, "dataset_version", None):
        raw_bindings.insert(0, f"train={args.dataset_version}:train")
    if raw_bindings:
        payload["dataset_bindings"] = [_parse_dataset_binding(value) for value in raw_bindings]
    return payload


def _work_item_payload(database: Any, item: Any) -> dict[str, Any]:
    payload = item.to_dict()
    payload["queue_position"] = database.work_item_queue_position(item.id)
    payload["blockers"] = database.work_item_blockers(item.id)
    return payload


def cmd_sweep(args: Any) -> None:
    """Manage durable repeat and sweep experiment groups."""

    try:
        _database, _scheduler, service = _experiment_runtime(args)
        action = args.sweep_command
        if action == "create":
            result = service.create_group_from_payload(_sweep_create_payload(args))
            kind = "group"
        elif action == "list":
            result = service.list_run_groups(kind=args.kind, status=args.status, limit=args.limit)
            kind = "group-list"
        elif action == "show":
            result = service.get_run_group_detail(args.group_id)
            kind = "group"
        elif action == "cancel":
            result = service.cancel_run_group(args.group_id)
            kind = "group"
        elif action == "resume":
            result = service.resume_run_group(
                args.group_id,
                **({"reason": args.reason} if getattr(args, "reason", None) else {}),
            )
            kind = "group"
        elif action == "checkpoints":
            result = service.get_checkpoint_trajectory(args.group_id)
            kind = "checkpoint-trajectory"
        elif action == "analyze":
            public = _adaptive_public_service(_database, _scheduler)
            result = public.create_run_group_analysis(
                args.group_id,
                {
                    "baseline_subject_id": args.baseline,
                    "practical_delta": args.practical_delta,
                    "equivalence_delta": args.equivalence_delta,
                    "confidence": args.confidence,
                    "bootstrap_resamples": args.bootstrap_resamples,
                    "bootstrap_seed": args.bootstrap_seed,
                },
            )
            kind = "analysis"
        elif action == "decide":
            public = _adaptive_public_service(_database, _scheduler)
            analysis_id = args.analysis
            if not analysis_id:
                analyses = public.list_run_group_analyses(args.group_id, limit=1)["items"]
                if not analyses:
                    raise ValueError("no cohort analysis exists; run `sweep analyze` first")
                analysis_id = analyses[0]["id"]
            result = public.create_research_decision(
                {
                    "analysis_snapshot_id": analysis_id,
                    "selected_subject": {
                        "run_group_id": args.group_id,
                        "trial_id": args.select,
                    },
                    "rejected_subjects": [{"trial_id": value} for value in (args.reject or [])],
                    "exclusions": [
                        {"id": value, "reason": "operator exclusion"}
                        for value in (args.exclude or [])
                    ],
                    "rationale": args.rationale,
                    "override_reason": args.override_reason,
                    "fork_spec": (
                        {"run_group_id": args.group_id, "trial_id": args.select}
                        if args.fork
                        else {}
                    ),
                }
            )
            kind = "decision"
        elif action == "report":
            public = _adaptive_public_service(_database, _scheduler)
            analysis_id = args.analysis
            if not analysis_id:
                analyses = public.list_run_group_analyses(args.group_id, limit=1)["items"]
                if not analyses:
                    raise ValueError("no cohort analysis exists; run `sweep analyze` first")
                analysis_id = analyses[0]["id"]
            result = public.create_evidence_bundle(
                {
                    "analysis_snapshot_id": analysis_id,
                    "research_decision_id": args.decision,
                    "formats": args.format or ["markdown", "html", "json", "csv", "svg"],
                }
            )
            kind = "evidence-bundle"
        elif action == "compare":
            result = service.compare_run_groups(args.left_group_id, args.right_group_id)
            kind = "comparison"
        elif action == "fork-best":
            seeds = _parse_cli_ints(args.seeds, label="seeds")
            result = service.build_fork_best_payload(args.group_id, name=args.name, seeds=seeds)
            detail = service.get_run_group_detail(args.group_id)
            source_trial = next(
                trial for trial in detail["trials"] if trial["id"] == result["source_trial_id"]
            )
            eligible_runs = [
                run
                for run in source_trial.get("runs", [])
                if run.get("objective_value") is not None
            ]
            parent_run = None
            if eligible_runs:
                reverse = detail["objective"]["direction"] == "maximize"
                parent_run = sorted(
                    eligible_runs,
                    key=lambda run: (
                        (
                            -float(run["objective_value"])
                            if reverse
                            else float(run["objective_value"])
                        ),
                        int(run.get("ordinal", 0)),
                        str(run.get("run_id", "")),
                    ),
                )[0]
            resolved = dict(result["base_config"])
            if parent_run is not None:
                resolved["seed"] = int(parent_run["seed"])
                result["parent_run_id"] = parent_run["run_id"]
            result["resolved_launch_config"] = resolved
            result["parent_context"] = {
                "run_group_id": detail["id"],
                "trial_id": source_trial["id"],
                "run_id": parent_run.get("run_id") if parent_run else None,
                "seed": parent_run.get("seed") if parent_run else None,
                "objective": detail["objective"],
                "objective_value": (parent_run.get("objective_value") if parent_run else None),
            }
            result["launch_started"] = False
            kind = "fork"
        else:  # pragma: no cover - argparse enforces the command set
            raise ValueError(f"unknown sweep command: {action}")
        _experiment_emit(result, as_json=args.json, kind=kind)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        _experiment_failure(args, exc)


def _checkpoint_policy_payload(args: Any) -> dict[str, Any]:
    """Build one immutable checkpoint-policy revision from CLI inputs."""

    payload = (
        _load_cli_mapping(args.spec, label="checkpoint policy")
        if getattr(args, "spec", None)
        else {}
    )
    if isinstance(payload.get("definition"), dict):
        definition = dict(payload.pop("definition"))
        payload.update(definition)
    for cli_name, field_name in (
        ("policy_id", "policy_id"),
        ("name", "name"),
        ("description", "description"),
        ("suite_revision", "development_suite_revision_id"),
        ("metric", "primary_metric"),
        ("direction", "direction"),
    ):
        value = getattr(args, cli_name, None)
        if value is not None:
            payload[field_name] = value
    if getattr(args, "schedule", None):
        payload["schedule"] = _load_cli_mapping(args.schedule, label="checkpoint schedule")
    if getattr(args, "rules", None):
        rules = _load_json_argument(args.rules, default=[])
        if not isinstance(rules, list):
            raise ValueError("checkpoint rules must be a JSON array")
        payload["rules"] = rules
    automatic = getattr(args, "automatic_actions", None)
    if automatic is not None:
        payload["automatic_actions"] = bool(automatic)
    capabilities = list(getattr(args, "capability", None) or [])
    if capabilities:
        payload["compatible_capabilities"] = capabilities
    payload.setdefault("version", 1)
    payload.setdefault("revision_number", 1)
    payload.setdefault("schedule", {"mode": "final", "unit": "step"})
    payload.setdefault("rules", [])
    payload.setdefault("guardrail_suite_revision_ids", [])
    payload.setdefault("automatic_actions", False)
    payload.setdefault("compatible_capabilities", [])
    return payload


def cmd_checkpoint_policy(args: Any) -> None:
    """Create and inspect versioned adaptive checkpoint policies."""

    try:
        database, scheduler, _service = _experiment_runtime(args)
        public = _adaptive_public_service(database, scheduler)
        action = args.checkpoint_policy_command
        if action == "list":
            result = public.list_checkpoint_policies(
                trainer_mode=args.trainer,
                limit=args.limit,
                offset=args.offset,
            )
            kind = "checkpoint-policy-list"
        elif action == "show":
            result = public.get_checkpoint_policy(args.policy)
            if result is None:
                raise ValueError(f"unknown checkpoint policy: {args.policy}")
            kind = "checkpoint-policy"
        elif action == "create":
            result = public.create_checkpoint_policy(_checkpoint_policy_payload(args))
            kind = "checkpoint-policy"
        elif action == "validate":
            from halo_forge.adaptive_lab import CheckpointPolicyRevision

            payload = _checkpoint_policy_payload(args)
            payload.setdefault("policy_id", "validation")
            payload.setdefault("name", "Validation policy")
            normalized = CheckpointPolicyRevision.from_dict(payload)
            result = {"valid": True, "policy": normalized.to_dict()}
            kind = "checkpoint-policy"
        else:  # pragma: no cover - argparse enforces choices
            raise ValueError(f"unknown checkpoint-policy command: {action}")
        _experiment_emit(result, as_json=args.json, kind=kind)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        _experiment_failure(args, exc)


def cmd_jobs(args: Any) -> None:
    """Inspect and operate the durable single-workstation queue."""

    try:
        database, scheduler, _service = _experiment_runtime(args)
        action = args.jobs_command
        if action == "list":
            statuses = [args.status] if args.status else None
            kinds = [args.kind] if args.kind else None
            result = [
                _work_item_payload(database, item)
                for item in database.list_work_items(
                    statuses=statuses,
                    kinds=kinds,
                    canonical_run_id=args.run_id,
                    limit=args.limit,
                    offset=args.offset,
                )
            ]
            kind = "work-list"
        elif action == "show":
            item = database.get_work_item(args.work_item_id)
            if item is None:
                raise ValueError(f"unknown work item: {args.work_item_id}")
            result = _work_item_payload(database, item)
            kind = "work"
        elif action == "cancel":
            item = scheduler.cancel(args.work_item_id)
            if item is None:
                raise ValueError(f"unknown work item: {args.work_item_id}")
            result = _work_item_payload(database, item)
            kind = "work"
        elif action == "retry":
            item = scheduler.retry(args.work_item_id)
            if item is None:
                raise ValueError(
                    "work item is not failed, interrupted, or cancelled: " f"{args.work_item_id}"
                )
            result = _work_item_payload(database, item)
            kind = "work"
        elif action == "worker":
            import signal

            from halo_forge.workstation_jobs.worker import WorkstationWorker

            recovery = scheduler.recover_or_adopt() if args.recover else None
            worker = WorkstationWorker(
                scheduler,
                poll_interval=args.poll_interval,
                heartbeat_interval=args.heartbeat_interval,
                terminate_timeout=args.terminate_timeout,
            )
            received_signal: list[int] = []

            def request_stop(signum: int, _frame: Any) -> None:
                received_signal.append(signum)
                worker.stop()

            previous_handlers = {
                signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
            }
            try:
                for signum in previous_handlers:
                    signal.signal(signum, request_stop)
                completed = worker.run_once() if args.once else None
                if not args.once:
                    worker.watch()
            finally:
                for signum, previous in previous_handlers.items():
                    signal.signal(signum, previous)
            result = {
                "adopted": [item.to_dict() for item in (recovery.adopted if recovery else ())],
                "interrupted": [
                    item.to_dict() for item in (recovery.interrupted if recovery else ())
                ],
                "completed_work_item": completed.to_dict() if completed else None,
            }
            kind = "worker"
            if received_signal:
                result["stopped_by_signal"] = received_signal[0]
        else:  # pragma: no cover - argparse enforces the command set
            raise ValueError(f"unknown jobs command: {action}")
        _experiment_emit(result, as_json=args.json, kind=kind)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as exc:
        _experiment_failure(args, exc)


def _artifact_runtime(args: Any):
    """Open the shared v4 artifact catalog and local content store."""

    from halo_forge.artifact_lab import ArtifactStore
    from halo_forge.artifact_studio import ArtifactStudioService
    from halo_forge.run_db import LabV4Catalog, get_database
    from halo_forge.workstation_jobs import WorkstationScheduler

    database = get_database(getattr(args, "database", None))
    root = getattr(args, "artifact_root", None)
    store = ArtifactStore(root) if root else ArtifactStore()
    catalog = LabV4Catalog(database)
    scheduler = WorkstationScheduler(database)
    return ArtifactStudioService(
        store=store,
        catalog=catalog,
        scheduler=scheduler,
    )


def _artifact_emit(payload: Any, *, as_json: bool) -> None:
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    if isinstance(payload, dict) and payload.get("domain_id"):
        print(
            f"{payload.get('domain_kind', 'operation')}: {payload['domain_id']}\n"
            f"Work item: {payload.get('work_item_id') or 'reused completed result'}\n"
            f"Status: {payload.get('status', 'unknown')}"
        )
        if payload.get("reused"):
            print("Reused: yes")
        if payload.get("execution_note"):
            print(f"Note: {payload['execution_note']}")
        return
    if isinstance(payload, dict) and "items" in payload:
        rows = list(payload.get("items") or [])
        if not rows:
            print("No artifacts found.")
            return
        for row in rows:
            occurrence = row.get("occurrence") or row
            blob = row.get("blob") or {}
            print(
                f"{occurrence.get('id', '(content only)')}  "
                f"{occurrence.get('artifact_kind', blob.get('artifact_type', 'artifact')):<18} "
                f"{blob.get('content_hash', '')[:12]}  "
                f"{occurrence.get('model_id', '')}"
            )
        if payload.get("has_more"):
            print("More results are available; increase --offset to continue.")
        return
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _artifact_failure(args: Any, exc: Exception) -> None:
    if getattr(args, "json", False):
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}))
    else:
        print(f"{RED}Artifact operation failed:{NC} {exc}", file=sys.stderr)
    raise SystemExit(1)


def _artifact_mapping(value: Optional[str], *, label: str) -> dict[str, Any]:
    if not value:
        return {}
    return _load_cli_mapping(value, label=label)


def _resolved_artifact_id(service: Any, identifier: str) -> str:
    view = service.show_artifact(identifier)
    occurrence = view.get("occurrence")
    if not occurrence:
        raise ValueError(f"{identifier!r} identifies content but not a model artifact occurrence")
    return str(occurrence["id"])


def _numeric_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    """Flatten persisted qualification metrics without treating booleans as numbers."""

    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            nested = f"{prefix}.{key}" if prefix else str(key)
            result.update(_numeric_metrics(child, nested))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = float(value)
    return result


def _artifact_compare_payload(
    service: Any,
    parent_identifier: str,
    candidate_identifier: str,
    *,
    profile_revision_id: Optional[str],
) -> dict[str, Any]:
    parent_id = _resolved_artifact_id(service, parent_identifier)
    candidate_id = _resolved_artifact_id(service, candidate_identifier)
    catalog = service.catalog
    parent_records = [
        item
        for item in catalog.list_qualifications(occurrence_id=parent_id, limit=1000)
        if item.status == "completed"
        and (not profile_revision_id or item.profile_revision_id == profile_revision_id)
    ]
    candidate_records = [
        item
        for item in catalog.list_qualifications(occurrence_id=candidate_id, limit=1000)
        if item.status == "completed"
        and (not profile_revision_id or item.profile_revision_id == profile_revision_id)
    ]
    candidate_by_profile = {item.profile_revision_id: item for item in candidate_records}
    selected_parent = next(
        (item for item in parent_records if item.profile_revision_id in candidate_by_profile),
        None,
    )
    if selected_parent is None:
        scope = (
            f" under qualification profile {profile_revision_id}"
            if profile_revision_id
            else " under the same qualification profile revision"
        )
        raise ValueError("Both artifacts need completed qualification evidence" + scope)
    selected_candidate = candidate_by_profile[selected_parent.profile_revision_id]
    profile = catalog.get_qualification_profile_revision(selected_parent.profile_revision_id)
    if profile is None:
        raise ValueError(
            f"Missing qualification profile revision: {selected_parent.profile_revision_id}"
        )
    directions: dict[str, str] = {}
    for rule in profile.to_dict().get("thresholds") or []:
        metric = str(rule.get("metric") or "").strip()
        stage = str(rule.get("stage") or "").strip()
        direction = str(rule.get("direction") or "maximize").strip().lower()
        if metric:
            directions[metric] = direction
            if stage:
                directions[f"{stage}.{metric}"] = direction
    parent_metrics = _numeric_metrics(selected_parent.to_dict().get("metrics") or {})
    candidate_metrics = _numeric_metrics(selected_candidate.to_dict().get("metrics") or {})
    metric_names = sorted(set(parent_metrics) | set(candidate_metrics))
    deltas = []
    for metric in metric_names:
        parent_value = parent_metrics.get(metric)
        candidate_value = candidate_metrics.get(metric)
        raw_delta = (
            None
            if parent_value is None or candidate_value is None
            else candidate_value - parent_value
        )
        direction = directions.get(metric) or directions.get(metric.rsplit(".", 1)[-1])
        favorable_delta = (
            None
            if raw_delta is None or direction is None
            else raw_delta if direction == "maximize" else -raw_delta
        )
        deltas.append(
            {
                "metric": metric,
                "direction": direction or "unscored",
                "parent": parent_value,
                "candidate": candidate_value,
                "raw_delta": raw_delta,
                "favorable_delta": favorable_delta,
            }
        )
    return {
        "profile_revision": profile.to_dict(),
        "parent": {
            "artifact_id": parent_id,
            "qualification": selected_parent.to_dict(),
        },
        "candidate": {
            "artifact_id": candidate_id,
            "qualification": selected_candidate.to_dict(),
        },
        "deltas": deltas,
    }


def _artifact_cleanup(service: Any, args: Any) -> dict[str, Any]:
    if getattr(args, "restore", None):
        return service.restore_artifact(args.restore)
    if getattr(args, "purge", False):
        return {"purged": service.purge_trash(), "retention_days": 7}
    if getattr(args, "apply", None):
        note = str(getattr(args, "review_note", None) or "").strip()
        if not note:
            raise ValueError("--review-note is required when applying a cleanup plan")
        return service.queue_cleanup(
            args.apply,
            review_note=note,
            priority=int(getattr(args, "priority", 0)),
            max_retries=int(getattr(args, "max_retries", 1)),
        ).to_dict()
    return {**service.preview_cleanup().to_dict(), "status": "preview"}


def cmd_artifact(args: Any) -> None:
    """Operate the content-addressed Artifact Studio."""

    try:
        service = _artifact_runtime(args)
        action = args.artifact_command
        if action == "list":
            result = service.list_artifacts(
                artifact_kind=args.kind,
                pinned=args.pinned,
                run_id=args.run_id,
                limit=args.limit,
                offset=args.offset,
            )
        elif action == "show":
            result = service.show_artifact(args.artifact)
        elif action == "import":
            result = service.import_artifact(
                args.source,
                artifact_kind=args.kind,
                artifact_format=args.format,
                model_id=args.model_id,
                backend=args.backend,
                managed=args.managed,
                dtype=args.dtype,
                quantization=args.quantization,
                occurrence_id=args.occurrence_id,
                run_id=args.run_id,
                tokenizer_revision=args.tokenizer_revision,
                chat_template_hash=args.chat_template_hash,
                metadata=_artifact_mapping(args.metadata, label="artifact metadata"),
            )
        elif action == "lineage":
            result = service.lineage(args.artifact)
        elif action == "verify":
            result = service.verify_artifact(args.artifact)
        elif action == "merge":
            input_ids = [_resolved_artifact_id(service, value) for value in args.artifacts]
            result = service.queue_merge(
                input_occurrence_ids=input_ids,
                base_model=args.base_model,
                mode=args.mode,
                method=args.method,
                weights=args.weights,
                bake_after_merge=args.bake_after_merge,
                priority=args.priority,
                max_retries=args.max_retries,
            ).to_dict()
        elif action == "convert":
            occurrence_id = _resolved_artifact_id(service, args.artifact)
            result = service.queue_convert(
                occurrence_id=occurrence_id,
                target_format=args.format,
                quantization=args.quantization,
                priority=args.priority,
                max_retries=args.max_retries,
                allow_unquantized_fallback=args.allow_unquantized_fallback,
            ).to_dict()
            result["quantization_method"] = (
                "post_training" if args.quantization.lower() in {"q4", "q8"} else "dtype_conversion"
            )
            result["qat"] = False
        elif action == "qualify":
            occurrence_id = _resolved_artifact_id(service, args.artifact)
            parent_id = _resolved_artifact_id(service, args.parent) if args.parent else None
            result = service.queue_qualification(
                occurrence_id=occurrence_id,
                profile_revision_id=args.profile,
                parent_occurrence_id=parent_id,
                execution_request=_artifact_mapping(
                    args.request, label="qualification execution request"
                ),
                priority=args.priority,
                max_retries=args.max_retries,
            ).to_dict()
            result["execution_note"] = (
                "Qualification completes only when a configured evaluation worker "
                "publishes verified quality, operational, and evidence results."
            )
        elif action == "compare":
            result = _artifact_compare_payload(
                service,
                args.parent,
                args.candidate,
                profile_revision_id=args.profile,
            )
        elif action in {"pin", "unpin"}:
            occurrence_id = _resolved_artifact_id(service, args.artifact)
            result = service.pin_artifact(occurrence_id, pinned=action == "pin")
        elif action == "tag":
            occurrence_id = _resolved_artifact_id(service, args.artifact)
            result = service.tag_artifact(occurrence_id, args.tags, replace=args.replace)
        elif action == "promote":
            occurrence_id = _resolved_artifact_id(service, args.artifact)
            target_alias = args.alias_option or args.alias
            if not target_alias:
                raise ValueError(
                    "promotion target is required: candidate or approved "
                    "(positional, --to, or --alias)"
                )
            result = service.promote(
                occurrence_id,
                target_alias,
                override_note=args.override_note,
            )
        elif action == "serve":
            occurrence_id = _resolved_artifact_id(service, args.artifact)
            result = service.queue_serving(
                occurrence_id,
                name=args.name,
                backend=args.backend,
                endpoint_settings=_artifact_mapping(args.endpoint, label="endpoint settings"),
                generation_settings=_artifact_mapping(args.generation, label="generation settings"),
                resource_requirements=_artifact_mapping(
                    args.resources, label="serving resource requirements"
                ),
                serving_id=args.serving_id,
                start_process=True,
                priority=args.priority,
                max_retries=args.max_retries,
            ).to_dict()
            result["server_started"] = False
            result["next_action"] = "The supervised worker will start and monitor the server."
        elif action == "export":
            occurrence_id = _resolved_artifact_id(service, args.artifact)
            model_card = args.model_card
            if model_card and Path(model_card).expanduser().is_file():
                model_card = Path(model_card).expanduser().read_text(encoding="utf-8")
            result = service.queue_export(
                occurrence_id=occurrence_id,
                destination=args.destination,
                replay_identity=_artifact_mapping(args.replay_identity, label="replay identity"),
                dataset_identity=_artifact_mapping(args.dataset_identity, label="dataset identity"),
                license_metadata=_artifact_mapping(args.license_metadata, label="license metadata"),
                model_card=model_card,
                priority=args.priority,
                max_retries=args.max_retries,
            ).to_dict()
        elif action == "cleanup":
            result = _artifact_cleanup(service, args)
        else:  # pragma: no cover - argparse enforces the command set
            raise ValueError(f"unknown artifact command: {action}")
        _artifact_emit(result, as_json=args.json)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        _artifact_failure(args, exc)


def cmd_storage(args: Any) -> None:
    """Inspect storage and execute reviewed Artifact Studio cleanup plans."""

    try:
        service = _artifact_runtime(args)
        if args.storage_command == "status":
            result = service.storage_inventory()
        elif args.storage_command == "cleanup":
            result = _artifact_cleanup(service, args)
        else:  # pragma: no cover - argparse enforces the command set
            raise ValueError(f"unknown storage command: {args.storage_command}")
        _artifact_emit(result, as_json=args.json)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        _artifact_failure(args, exc)


def cmd_config_validate(args):
    """Validate training config file."""
    import yaml
    from pathlib import Path

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    errors = []
    warnings = []

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML syntax: {e}")
        sys.exit(1)

    print(f"Validating config: {config_path}")
    print("=" * 50)

    # Required fields based on config type
    config_type = args.type
    if not config_type:
        if "raft" in str(config_path).lower():
            config_type = "raft"
        elif "sft" in str(config_path).lower():
            config_type = "sft"
        else:
            config_type = "auto"

    def _get_nested(cfg: dict, path: str):
        """Fetch nested config value by dot-delimited path."""
        current = cfg
        for key in path.split("."):
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _number(value):
        """Coerce YAML numeric scalars, including scientific notation strings."""
        if isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    if config_type == "raft":
        required = ["output_dir", "prompts"]
    elif config_type == "sft":
        required = ["model.name", "data.train_file", "training.output_dir"]
    else:
        required = []

    # Check required fields
    for field in required:
        if "." in field:
            value = _get_nested(config, field)
            if value is None:
                errors.append(f"Missing required field: {field}")
        else:
            if field not in config:
                errors.append(f"Missing required field: {field}")

    # Validate specific fields
    lr = config.get("learning_rate")
    if lr is None:
        lr = _get_nested(config, "training.learning_rate")
    if lr is not None:
        numeric_lr = _number(lr)
        if numeric_lr is None or numeric_lr <= 0:
            errors.append(f"Invalid learning_rate: {lr} (must be positive number)")
        elif numeric_lr > 1e-3:
            warnings.append(f"learning_rate={lr} seems high (typical: 1e-5 to 5e-5)")

    decay = config.get("lr_decay_per_cycle")
    if decay is None:
        decay = _get_nested(config, "lr_decay_per_cycle")
    if decay is not None:
        numeric_decay = _number(decay)
        if numeric_decay is None or not 0 < numeric_decay <= 1:
            errors.append(f"Invalid lr_decay_per_cycle: {decay} (must be 0 < x <= 1)")

    cycles = config.get("num_cycles")
    if cycles is None:
        cycles = _get_nested(config, "raft.num_cycles")
    if cycles is not None:
        if not isinstance(cycles, int) or cycles < 1:
            errors.append(f"Invalid num_cycles: {cycles} (must be positive integer)")
        elif cycles > 10:
            warnings.append(f"num_cycles={cycles} is high (typical: 3-6)")

    temp = config.get("temperature")
    if temp is None:
        temp = _get_nested(config, "generation.temperature")
    if temp is not None:
        numeric_temp = _number(temp)
        if numeric_temp is None or not 0 < numeric_temp <= 2:
            errors.append(f"Invalid temperature: {temp} (must be 0 < x <= 2)")

    threshold = config.get("reward_threshold")
    if threshold is None:
        threshold = _get_nested(config, "raft.reward_threshold")
    if threshold is not None:
        numeric_threshold = _number(threshold)
        if numeric_threshold is None or not 0 <= numeric_threshold <= 1:
            errors.append(f"Invalid reward_threshold: {threshold} (must be 0 <= x <= 1)")

    # Print results
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  ✗ {e}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  ⚠ {w}")

    if not errors and not warnings:
        print("✓ Config is valid")
    elif not errors:
        print(f"\n✓ Config is valid ({len(warnings)} warnings)")

    # Print config summary
    if args.verbose:
        print("\nConfig contents:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    if errors:
        sys.exit(1)


def cmd_data_prepare(args):
    """Prepare dataset from public sources."""
    from halo_forge.data.public_datasets import DatasetPreparer, get_dataset_spec, list_datasets

    if args.list:
        print("Available datasets:")
        for name in list_datasets():
            print(f"  - {name}")
        return

    if not args.dataset:
        print("Error: --dataset required")
        print("Use --list to see available datasets")
        sys.exit(1)

    spec = get_dataset_spec(args.dataset)
    preparer = DatasetPreparer(spec, system_prompt=args.system_prompt)

    output = args.output or f"data/{args.dataset}.jsonl"
    preparer.prepare(output, template=args.template)


def cmd_data_generate(args):
    """Generate data with LLM."""
    from halo_forge.data.llm_generate import (
        TrainingDataGenerator,
        get_backend,
        get_topic_spec,
        list_topics,
    )

    if args.list:
        print("Available topics:")
        for name in list_topics():
            print(f"  - {name}")
        return

    if not args.topic:
        print("Error: --topic required")
        print("Use --list to see available topics")
        sys.exit(1)

    spec = get_topic_spec(args.topic)
    backend = get_backend(args.backend, model=args.model)
    generator = TrainingDataGenerator(backend, spec)

    output = args.output or f"data/{args.topic}_generated.jsonl"
    generator.generate_all(output, template=args.template)


_CPT_CONFIG_ARG_FIELDS = {
    "model": "model_name",
    "model_revision": "model_revision",
    "model_hash": "model_hash",
    "tokenizer_revision": "tokenizer_revision",
    "tokenizer_hash": "tokenizer_hash",
    "training_artifact_id": "training_artifact_id",
    "training_artifact_hash": "training_artifact_hash",
    "expected_packing_plan_hash": "expected_packing_plan_hash",
    "adaptation": "adaptation",
    "output": "output_dir",
    "seed": "seed",
    "max_seq_length": "max_sequence_length",
    "packing": "packing",
    "budget_mode": "budget_mode",
    "target_tokens": "target_tokens",
    "corpus_passes": "corpus_passes",
    "batch_size": "batch_size",
    "gradient_accumulation": "gradient_accumulation_steps",
    "learning_rate": "learning_rate",
    "warmup_ratio": "warmup_ratio",
    "weight_decay": "weight_decay",
    "max_grad_norm": "max_grad_norm",
    "max_steps": "max_steps",
    "optim": "optim",
    "lora_rank": "lora_r",
    "lora_alpha": "lora_alpha",
    "lora_dropout": "lora_dropout",
    "use_dora": "use_dora",
    "use_rslora": "use_rslora",
    "init_lora_weights": "init_lora_weights",
    "load_in_4bit": "load_in_4bit",
    "validation_split": "validation_fraction",
    "save_steps": "save_steps",
    "eval_steps": "eval_steps",
    "save_total_limit": "save_total_limit",
    "logging_steps": "logging_steps",
}


def _prepare_cpt_cli_config(args):
    """Return a YAML/default base plus the flags the operator supplied."""

    from halo_forge.cpt import CPTConfig

    explicit = {
        name
        for name in (
            *_CPT_CONFIG_ARG_FIELDS,
            "target_modules",
            "no_bf16",
            "no_gradient_checkpointing",
        )
        if getattr(args, name, None) is not None
    }
    configured = CPTConfig.from_yaml(args.config) if args.config else None
    adaptation = (
        getattr(args, "adaptation", None)
        or (configured.adaptation if configured is not None else None)
    )
    if not adaptation:
        raise ValueError(
            "CPT requires an explicit --adaptation lora or --adaptation full"
        )
    base = configured or CPTConfig(adaptation=adaptation)

    # Dataset Lab rendering happens before the final config object is built.
    # Populate only omitted parser attributes so it sees the same YAML/default
    # values without erasing which options were explicit CLI overrides.
    for argument, field in _CPT_CONFIG_ARG_FIELDS.items():
        if getattr(args, argument, None) is None:
            setattr(args, argument, getattr(base, field))
    if getattr(args, "target_modules", None) is None:
        args.target_modules = ",".join(base.target_modules or [])
    if getattr(args, "no_bf16", None) is None:
        args.no_bf16 = not bool(base.bf16)
    if getattr(args, "no_gradient_checkpointing", None) is None:
        args.no_gradient_checkpointing = not bool(base.gradient_checkpointing)
    if not getattr(args, "train_file", None):
        args.train_file = base.train_file
    if not getattr(args, "validation_file", None):
        args.validation_file = base.validation_file
    return base, explicit


def _resolve_cpt_cli_config(args, base, explicit):
    """Overlay explicitly supplied CLI options on one CPT base config."""

    from dataclasses import asdict

    from halo_forge.cpt import CPTConfig

    values = asdict(base)
    for argument, field in _CPT_CONFIG_ARG_FIELDS.items():
        if argument in explicit:
            values[field] = getattr(args, argument)
    if "target_modules" in explicit:
        values["target_modules"] = [
            value.strip()
            for value in str(args.target_modules).split(",")
            if value.strip()
        ]
    if "no_bf16" in explicit:
        values["bf16"] = not bool(args.no_bf16)
    if "no_gradient_checkpointing" in explicit:
        values["gradient_checkpointing"] = not bool(
            args.no_gradient_checkpointing
        )
    # Managed Dataset Lab resolution may replace these after the initial CLI
    # overlay, so the final artifact paths always take precedence.
    values["train_file"] = getattr(args, "train_file", None)
    values["validation_file"] = getattr(args, "validation_file", None)
    if isinstance(getattr(args, "training_artifact", None), dict):
        for argument, field in (
            ("model", "model_name"),
            ("model_revision", "model_revision"),
            ("model_hash", "model_hash"),
            ("tokenizer_revision", "tokenizer_revision"),
            ("tokenizer_hash", "tokenizer_hash"),
            ("training_artifact_id", "training_artifact_id"),
            ("training_artifact_hash", "training_artifact_hash"),
            ("expected_packing_plan_hash", "expected_packing_plan_hash"),
        ):
            values[field] = getattr(args, argument, None)
    # These are input aliases populated by CPTConfig after validation. Keeping
    # stale base values would make a legitimate CLI model/adaptation override
    # appear contradictory to itself.
    values.pop("model", None)
    values.pop("adaptation_mode", None)
    return CPTConfig.from_mapping(values)


def cmd_cpt_train(args):
    """Run continued causal pretraining on a canonical document corpus."""

    from halo_forge.cpt import get_cpt_trainer
    from halo_forge.cpt.trainer import load_corpus_jsonl

    base_config, explicit = _prepare_cpt_cli_config(args)
    _apply_managed_dataset_args(args, "cpt", "train_file")
    if not getattr(args, "train_file", None):
        raise ValueError(
            "CPT requires --train-file or --dataset-version/--dataset-binding"
        )
    config = _resolve_cpt_cli_config(args, base_config, explicit)

    print_banner()
    print(f"{GREEN}Continued Pretraining (CPT){NC}")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Corpus: {config.train_file}")
    if config.validation_file:
        print(f"Validation: {config.validation_file}")
    print(f"Adaptation: {config.adaptation}")
    print(
        f"Packing: {config.packing} at {config.max_sequence_length} tokens"
    )
    if config.budget_mode == "tokens":
        print(f"Budget: {config.target_tokens} training tokens")
    else:
        print(f"Budget: {config.corpus_passes} corpus passes")
    print(f"Output: {config.output_dir}")

    if args.dry_run:
        records = load_corpus_jsonl(config.train_file)
        validation_records = (
            load_corpus_jsonl(config.validation_file)
            if config.validation_file
            else []
        )
        print(
            f"{GREEN}Configuration valid:{NC} "
            f"{len(records)} training documents, "
            f"{len(validation_records)} supplied validation documents"
        )
        return

    trainer = get_cpt_trainer(config)
    summary = trainer.train(
        train_file=config.train_file,
        validation_file=config.validation_file,
        resume_from_checkpoint=args.resume,
    )
    _print_completed_training_summary(
        "cpt", config.output_dir, summary, args=args, config=config
    )


def cmd_sft_train(args):
    """Run SFT training. Dispatches to the right backend (PyTorch / MLX)."""
    _apply_managed_dataset_args(args, "sft", "data")
    from halo_forge.sft.trainer import SFTConfig
    from halo_forge.sft._dispatch import get_sft_trainer

    print_banner()
    print(f"{GREEN}SFT Training{NC}")
    print("=" * 60)

    # Require either --dataset or --data
    dataset = getattr(args, "dataset", None)
    data = getattr(args, "data", None)
    validation_data = getattr(args, "validation_data", None)
    max_samples = getattr(args, "max_samples", None)
    dry_run = getattr(args, "dry_run", False)

    if not dataset and not data:
        print(f"{RED}Error: Either --dataset or --data is required{NC}")
        print()
        print("Examples:")
        print("  halo-forge sft train --dataset codealpaca --model Qwen/Qwen2.5-Coder-3B")
        print("  halo-forge sft train --data my_data.jsonl --model Qwen/Qwen2.5-Coder-3B")
        print()
        print("Available datasets:")
        print("  codealpaca, metamath, gsm8k_sft, llava, xlam_sft, glaive_sft")
        print("  Run 'halo-forge sft datasets' to see all options")
        sys.exit(1)

    # Extract all CLI arguments with defaults
    batch_size = getattr(args, "batch_size", 2)
    learning_rate = getattr(args, "learning_rate", 2e-4)
    warmup_ratio = getattr(args, "warmup_ratio", 0.03)
    weight_decay = getattr(args, "weight_decay", 0.01)
    max_grad_norm = getattr(args, "max_grad_norm", 0.3)
    gradient_accumulation = getattr(args, "gradient_accumulation", 16)
    lora_rank = getattr(args, "lora_rank", 16)
    lora_alpha = getattr(args, "lora_alpha", 32)
    lora_dropout = getattr(args, "lora_dropout", 0.05)
    no_lora = getattr(args, "no_lora", False)
    no_gradient_checkpointing = getattr(args, "no_gradient_checkpointing", False)
    save_steps = getattr(args, "save_steps", 500)
    eval_steps = getattr(args, "eval_steps", 250)
    save_total_limit = getattr(args, "save_total_limit", 3)
    early_stopping_patience = getattr(args, "early_stopping_patience", 5)
    validation_split = getattr(args, "validation_split", 0.05)
    max_seq_length = getattr(args, "max_seq_length", 2048)
    use_dora = getattr(args, "use_dora", False)
    use_rslora = getattr(args, "use_rslora", False)
    init_lora_weights = getattr(args, "init_lora_weights", "true")
    optim = getattr(args, "optim", "adamw_torch")

    if args.config:
        config = SFTConfig.from_yaml(args.config)
        # CLI args override config file
        if args.model:
            config.model_name = args.model
        if dataset:
            config.dataset = dataset
        if data:
            config.train_file = data
        if validation_data:
            config.validation_file = validation_data
        if max_samples:
            config.max_samples = max_samples
        if args.output:
            config.output_dir = args.output
        if args.epochs:
            config.num_epochs = args.epochs
        # Apply other overrides
        config.batch_size = batch_size
        config.learning_rate = learning_rate
        config.warmup_ratio = warmup_ratio
        config.weight_decay = weight_decay
        config.max_grad_norm = max_grad_norm
        config.gradient_accumulation_steps = gradient_accumulation
        config.lora_r = lora_rank
        config.lora_alpha = lora_alpha
        config.lora_dropout = lora_dropout
        config.save_steps = save_steps
        config.eval_steps = eval_steps
        config.save_total_limit = save_total_limit
        config.early_stopping_patience = early_stopping_patience
        config.validation_split = validation_split
        config.max_seq_length = max_seq_length
        config.use_dora = use_dora
        config.use_rslora = use_rslora
        config.init_lora_weights = init_lora_weights
        config.optim = optim
        config.seed = getattr(args, "seed", 42)
        if getattr(args, "max_steps", None) is not None:
            config.max_steps = args.max_steps
        config.enable_neural_accelerators = getattr(args, "enable_neural_accelerators", False)
        config.capture_parameter_hashes = bool(
            getattr(args, "capture_parameter_hashes", False)
        )
        if no_gradient_checkpointing:
            config.gradient_checkpointing = False
    else:
        config = SFTConfig(
            model_name=args.model or "Qwen/Qwen2.5-Coder-7B",
            dataset=dataset,
            train_file=data,
            validation_file=validation_data,
            max_samples=max_samples,
            output_dir=args.output,
            num_epochs=args.epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation,
            lora_r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            save_steps=save_steps,
            eval_steps=eval_steps,
            save_total_limit=save_total_limit,
            early_stopping_patience=early_stopping_patience,
            validation_split=validation_split,
            max_seq_length=max_seq_length,
            gradient_checkpointing=not no_gradient_checkpointing,
            use_dora=use_dora,
            use_rslora=use_rslora,
            init_lora_weights=init_lora_weights,
            optim=optim,
            seed=getattr(args, "seed", 42),
            max_steps=getattr(args, "max_steps", None),
            enable_neural_accelerators=getattr(args, "enable_neural_accelerators", False),
            capture_parameter_hashes=bool(
                getattr(args, "capture_parameter_hashes", False)
            ),
        )

    # Disable LoRA if requested (full fine-tuning)
    if no_lora:
        config.lora_r = 0  # Trainer will skip LoRA setup if rank is 0

    print(f"Model: {config.model_name}")
    if config.dataset:
        print(f"Dataset: {config.dataset}")
    elif config.train_file:
        print(f"Data file: {config.train_file}")
    if config.max_samples:
        print(f"Max samples: {config.max_samples}")
    print(f"Output: {config.output_dir}")
    print(f"Epochs: {config.num_epochs}")
    print(
        f"Batch size: {config.batch_size} (x{config.gradient_accumulation_steps} accum = {config.batch_size * config.gradient_accumulation_steps} effective)"
    )
    print(f"Learning rate: {config.learning_rate}")
    if config.lora_r > 0:
        print(f"LoRA: rank={config.lora_r}, alpha={config.lora_alpha}")
    else:
        print(f"LoRA: disabled (full fine-tuning)")
    print()

    if dry_run:
        print(f"{YELLOW}Dry run mode - validating configuration only{NC}")
        print()
        # Validate dataset exists
        if config.dataset:
            from halo_forge.sft.datasets import get_sft_dataset_spec, is_huggingface_id

            spec = get_sft_dataset_spec(config.dataset)
            if spec:
                print(f"{GREEN}✓{NC} Dataset: {spec.name} ({spec.huggingface_id})")
            elif is_huggingface_id(config.dataset):
                print(f"{GREEN}✓{NC} HuggingFace dataset: {config.dataset}")
            else:
                print(f"{RED}✗{NC} Unknown dataset: {config.dataset}")
                sys.exit(1)
        print(f"{GREEN}Configuration valid!{NC}")
        return

    trainer = get_sft_trainer(config)
    summary = trainer.train(resume_from_checkpoint=args.resume)
    _print_completed_training_summary("sft", config.output_dir, summary, args=args, config=config)


def cmd_dpo_train(args):
    """Run DPO training (Track T1 / phase Q1).

    Wraps `trl.DPOTrainer` so we get the published loss-math (sigmoid, IPO,
    hinge, KTO-pair, RPO, cDPO via label-smoothing) for free; halo-forge owns
    the run-id, output_dir, training_summary contract, and recovery guidance
    so the public API + frontend treat DPO runs identically to SFT/RAFT.
    """
    _apply_managed_dataset_args(args, "dpo", "data")
    from halo_forge.dpo import DPOConfig, get_dpo_trainer

    print_banner()
    print(f"{GREEN}DPO Training{NC}")
    print("=" * 60)

    dataset = getattr(args, "dataset", None)
    data = getattr(args, "data", None)
    if not dataset and not data:
        print(f"{RED}Error: Either --dataset or --data is required{NC}")
        print()
        print("Examples:")
        print("  halo-forge dpo train --dataset ultrafeedback --model Qwen/Qwen2.5-3B-Instruct")
        print("  halo-forge dpo train --data my_pairs.jsonl --model meta-llama/Llama-3.2-3B")
        print()
        print("Available preference datasets:")
        print("  ultrafeedback, orca_dpo, hh_rlhf, py_dpo")
        print("  Run 'halo-forge dpo datasets' to see all options")
        sys.exit(1)

    config = DPOConfig(
        model_name=args.model,
        train_file=data,
        validation_file=getattr(args, "validation_data", None),
        dataset=dataset,
        max_samples=getattr(args, "max_samples", None),
        validation_split=getattr(args, "validation_split", 0.05),
        max_seq_length=getattr(args, "max_seq_length", 1024),
        max_prompt_length=getattr(args, "max_prompt_length", 512),
        beta=getattr(args, "beta", 0.1),
        loss_type=getattr(args, "loss_type", "sigmoid"),
        reference_free=getattr(args, "reference_free", False),
        label_smoothing=getattr(args, "label_smoothing", 0.0),
        output_dir=args.output,
        num_epochs=getattr(args, "epochs", 1),
        batch_size=getattr(args, "batch_size", 1),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation", 16),
        learning_rate=getattr(args, "learning_rate", 5e-6),
        warmup_ratio=getattr(args, "warmup_ratio", 0.1),
        weight_decay=getattr(args, "weight_decay", 0.0),
        max_grad_norm=getattr(args, "max_grad_norm", 1.0),
        lora_r=getattr(args, "lora_rank", 16),
        lora_alpha=getattr(args, "lora_alpha", 32),
        lora_dropout=getattr(args, "lora_dropout", 0.05),
        use_dora=getattr(args, "use_dora", False),
        use_rslora=getattr(args, "use_rslora", False),
        init_lora_weights=getattr(args, "init_lora_weights", "true"),
        optim=getattr(args, "optim", "adamw_torch"),
        save_steps=getattr(args, "save_steps", 200),
        eval_steps=getattr(args, "eval_steps", 100),
        save_total_limit=getattr(args, "save_total_limit", 3),
        load_in_4bit=getattr(args, "load_in_4bit", False),
        enable_neural_accelerators=getattr(args, "enable_neural_accelerators", False),
        gradient_checkpointing=not getattr(args, "no_gradient_checkpointing", False),
        seed=getattr(args, "seed", 42),
        max_steps=getattr(args, "max_steps", None),
    )

    if getattr(args, "dry_run", False):
        print("Dry run: configuration validated. No training started.")
        print(f"  model={config.model_name} dataset={config.dataset or '(local)'}")
        print(f"  beta={config.beta} loss_type={config.loss_type}")
        return

    trainer = get_dpo_trainer(config)
    summary = trainer.train(resume_from_checkpoint=args.resume)
    _print_completed_training_summary("dpo", config.output_dir, summary, args=args, config=config)


def cmd_orpo_train(args):
    """Run ORPO training (Track T17b).

    Reference-free preference optimization in a single pass — combines
    NLL on the chosen response with a log-odds preference term against
    the rejected. No reference model copy in memory, no separate RM/PPO,
    competitive with DPO on chat refinement.
    """
    _apply_managed_dataset_args(args, "orpo", "data")
    from halo_forge.orpo import ORPOConfig, get_orpo_trainer

    print_banner()
    print(f"{GREEN}ORPO Training{NC}")
    print("=" * 60)

    dataset = getattr(args, "dataset", None)
    data = getattr(args, "data", None)
    if not dataset and not data:
        print(f"{RED}Error: Either --dataset or --data is required{NC}")
        print()
        print("Examples:")
        print("  halo-forge orpo train --dataset ultrafeedback --model Qwen/Qwen2.5-3B-Instruct")
        print("  halo-forge orpo train --data my_pairs.jsonl --model meta-llama/Llama-3.2-3B")
        print()
        print("ORPO consumes the same prompt/chosen/rejected layout as DPO —")
        print("  ultrafeedback, orca_dpo, hh_rlhf, py_dpo all work.")
        sys.exit(1)

    config = ORPOConfig(
        model_name=args.model,
        train_file=data,
        validation_file=getattr(args, "validation_data", None),
        dataset=dataset,
        max_samples=getattr(args, "max_samples", None),
        validation_split=getattr(args, "validation_split", 0.05),
        max_seq_length=getattr(args, "max_seq_length", 1024),
        max_prompt_length=getattr(args, "max_prompt_length", 512),
        beta=getattr(args, "beta", 0.1),
        output_dir=args.output,
        num_epochs=getattr(args, "epochs", 1),
        batch_size=getattr(args, "batch_size", 1),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation", 16),
        learning_rate=getattr(args, "learning_rate", 8e-6),
        warmup_ratio=getattr(args, "warmup_ratio", 0.1),
        weight_decay=getattr(args, "weight_decay", 0.0),
        max_grad_norm=getattr(args, "max_grad_norm", 1.0),
        lora_r=getattr(args, "lora_rank", 16),
        lora_alpha=getattr(args, "lora_alpha", 32),
        lora_dropout=getattr(args, "lora_dropout", 0.05),
        use_dora=getattr(args, "use_dora", False),
        use_rslora=getattr(args, "use_rslora", False),
        init_lora_weights=getattr(args, "init_lora_weights", "true"),
        optim=getattr(args, "optim", "adamw_torch"),
        save_steps=getattr(args, "save_steps", 200),
        eval_steps=getattr(args, "eval_steps", 100),
        save_total_limit=getattr(args, "save_total_limit", 3),
        load_in_4bit=getattr(args, "load_in_4bit", False),
        enable_neural_accelerators=getattr(args, "enable_neural_accelerators", False),
        gradient_checkpointing=not getattr(args, "no_gradient_checkpointing", False),
        seed=getattr(args, "seed", 42),
        max_steps=getattr(args, "max_steps", None),
    )

    if getattr(args, "dry_run", False):
        print("Dry run: configuration validated. No training started.")
        print(f"  model={config.model_name} dataset={config.dataset or '(local)'}")
        print(f"  beta={config.beta}")
        return

    trainer = get_orpo_trainer(config)
    summary = trainer.train(resume_from_checkpoint=args.resume)
    _print_completed_training_summary("orpo", config.output_dir, summary, args=args, config=config)


def cmd_merge(args):
    """Merge LoRA adapters (Tracks T12 + T13).

    Two operations:
      bake     — merge a single LoRA adapter into its base. Output is a
                 standard HF checkpoint, no LoRA infrastructure required.
      combine  — combine N LoRA adapters into one (linear / ties / dare).
                 Optionally bake the combined adapter into the base in
                 the same step.
    """
    from halo_forge.inference.merge import (
        list_supported_methods,
        merge as run_merge,
    )

    print_banner()
    print(f"{GREEN}halo-forge merge{NC}")
    print("=" * 60)

    if getattr(args, "list", False):
        print("Operations: bake, combine")
        print(f"Combine methods: {', '.join(list_supported_methods())}")
        return

    print(f"  mode:   {args.mode}")
    print(f"  base:   {args.base}")
    print(f"  output: {args.output}")
    if args.mode == "bake":
        print(f"  adapter: {args.adapter}")
    else:
        print(f"  adapters: {args.adapters}")
        print(f"  weights:  {args.weights or '(uniform)'}")
        print(f"  method:   {args.method}")
        if args.bake_after_merge:
            print(f"  + bake-after-merge")
    print()

    adapter_paths = (
        [s.strip() for s in args.adapters.split(",") if s.strip()]
        if args.mode == "combine"
        else None
    )
    weights = None
    if args.mode == "combine" and args.weights:
        weights = [float(w) for w in args.weights.split(",")]

    try:
        result = run_merge(
            operation=args.mode,
            base_model=args.base,
            output_path=args.output,
            adapter_path=args.adapter if args.mode == "bake" else None,
            adapter_paths=adapter_paths,
            weights=weights,
            method=args.method,
            bake_after_merge=args.bake_after_merge,
            trust_remote_code=args.trust_remote_code,
            svd_rank=args.svd_rank,
        )
    except Exception as exc:
        print(f"{RED}Merge failed:{NC} {exc}")
        sys.exit(1)

    size_mb = (result.bytes_written or 0) / (1024 * 1024)
    print(f"{GREEN}Merged{NC} → {result.output_path}")
    print(f"  size: {size_mb:.1f} MB")
    if result.notes:
        print(f"  note: {result.notes}")


def cmd_probe(args):
    """Run a mid-training general-benchmark probe (Track V9)."""
    from pathlib import Path

    from halo_forge.eval import DEFAULT_PROBE_TASKS, MidTrainingProbe

    print_banner()
    print(f"{GREEN}halo-forge probe{NC} (mid-training general-benchmark probe)")
    print("=" * 60)

    tasks = (
        [t.strip() for t in args.tasks.split(",") if t.strip()]
        if args.tasks
        else list(DEFAULT_PROBE_TASKS)
    )
    print(f"  model:    {args.model}")
    print(f"  tasks:    {tasks}")
    print(f"  limit:    {args.limit} samples per task")
    print(f"  baseline: {args.baseline or '(no persistence)'}")
    print(f"  tolerance: {args.tolerance}")
    print()

    probe = MidTrainingProbe(
        model_name=args.model,
        baseline_path=Path(args.baseline) if args.baseline else None,
        tasks=tasks,
        limit=args.limit,
        every_n_cycles=1,  # CLI invocation is one-shot
        regression_tolerance=args.tolerance,
        backend=args.backend,
    )

    try:
        report = probe.run(cycle=args.cycle, notes=args.notes)
    except Exception as exc:
        print(f"{RED}Probe failed:{NC} {exc}")
        sys.exit(1)

    print(f"{GREEN}Done{NC} in {report.duration_seconds:.1f}s")
    if not report.has_baseline:
        print(f"{YELLOW}No baseline yet — current values written as the baseline.{NC}")
    print()
    for d in report.task_deltas:
        marker = (
            f"{RED}REGRESS{NC}"
            if d.regression
            else (f"{GREEN}    OK{NC}" if d.delta is not None and d.delta >= 0 else "       ")
        )
        delta_str = f"  Δ={d.delta:+.4f}" if d.delta is not None else ""
        print(f"  {marker} {d.task:<22} " f"{d.primary_metric:>22} = {d.value:>7.4f}{delta_str}")

    if report.avg_delta is not None:
        print(f"\n  avg delta vs baseline: {report.avg_delta:+.4f}")
    if report.has_regression:
        regressed = report.regressed_tasks()
        print(f"\n{RED}Regression on {len(regressed)} task(s):{NC} " f"{', '.join(regressed)}")
        sys.exit(2)


def _evaluation_lab_service():
    from halo_forge.evaluation_lab import EvaluationLabService
    from halo_forge.run_db import get_database

    return EvaluationLabService(get_database())


def _load_json_argument(value: Optional[str], *, default: Any) -> Any:
    if not value:
        return default
    candidate = Path(value).expanduser()
    raw = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    return json.loads(raw)


def _spawn_evaluation_worker(evaluation_id: str) -> int:
    """Start a detached process that claims one persistent evaluation job."""

    process = subprocess.Popen(
        [sys.executable, "-m", "halo_forge.evaluation_lab.worker", str(evaluation_id)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
        env=os.environ.copy(),
    )
    return int(process.pid)


def cmd_eval_suite(args):
    service = _evaluation_lab_service()
    try:
        if args.suite_action == "list":
            _print_dataset_lab_payload(
                {"items": [item.to_dict() for item in service.db.list_benchmark_suites()]}
            )
            return
        if args.suite_action == "show":
            suite = service.db.get_benchmark_suite(args.suite_id)
            if suite is None:
                raise KeyError(args.suite_id)
            _print_dataset_lab_payload(
                {
                    **suite.to_dict(),
                    "revisions": [
                        item.to_dict()
                        for item in service.db.list_benchmark_suite_revisions(args.suite_id)
                    ],
                }
            )
            return
        items = _load_json_argument(args.items, default=[])
        if isinstance(items, dict):
            items = items.get("items") or []
        generation = _load_json_argument(args.generation_settings, default={})
        evaluators = _load_json_argument(args.evaluator_versions, default={})
        if args.suite_action == "create":
            suite, revision = service.create_suite(
                name=args.name,
                description=args.description,
                purpose=args.purpose,
                items=items,
                primary_metric=args.primary_metric,
                direction=args.direction,
                generation_settings=generation,
                evaluator_versions=evaluators,
            )
            _print_dataset_lab_payload(
                {"suite": suite.to_dict(), "revision": revision.to_dict() if revision else None}
            )
        else:
            revision = service.create_revision(
                suite_id=args.suite_id,
                items=items,
                primary_metric=args.primary_metric,
                direction=args.direction,
                generation_settings=generation,
                evaluator_versions=evaluators,
            )
            _print_dataset_lab_payload(revision)
    except Exception as exc:
        print(f"{RED}Evaluation suite operation failed:{NC} {exc}")
        sys.exit(1)
    finally:
        service.shutdown()


def cmd_eval_run(args):
    service = _evaluation_lab_service()
    wait_for_result = bool(args.wait)
    try:
        _prepare_profile_verifier(args, consumer="direct")
        subject = {"type": args.subject_type, "ref": args.subject}
        if getattr(args, "subject_revision", None):
            subject["revision"] = args.subject_revision
        if getattr(args, "run_id", None):
            subject["run_id"] = args.run_id
            if args.subject_type == "checkpoint":
                subject["checkpoint"] = args.subject
        elif args.subject_type in {"final_model", "checkpoint"}:
            subject["path"] = args.subject
        request = _load_json_argument(args.request, default={})
        if getattr(args, "verifier_profile_revision", None):
            if any(
                request.get(key) not in (None, "", {})
                for key in ("verifier", "verifier_config", "pass_threshold", "reward_threshold")
            ):
                raise ValueError(
                    "--verifier-profile-revision conflicts with raw verifier request fields"
                )
            request["verifier_profile_revision_id"] = args.verifier_profile_revision
        launched = service.launch_evaluation(
            suite_revision_id=args.suite_revision,
            adapter_id=args.adapter,
            subject=subject,
            request=request,
            submit=wait_for_result,
        )
        evaluation = launched.evaluation
        _bind_profile_verifier(
            args,
            domain_kind="evaluation",
            domain_id=evaluation.id,
            role="evaluation_verifier",
        )
        if wait_for_result and not launched.reused:
            evaluation = service.jobs.wait(evaluation.id)
        elif not launched.reused:
            try:
                worker_pid = _spawn_evaluation_worker(evaluation.id)
            except Exception as exc:
                service.db.update_evaluation(
                    evaluation.id,
                    status="failed",
                    stage="worker_launch_failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            payload = evaluation.to_dict()
            payload["worker_pid"] = worker_pid
            evaluation_payload = payload
        else:
            evaluation_payload = evaluation.to_dict()
        _print_dataset_lab_payload(
            {
                "evaluation": (evaluation.to_dict() if wait_for_result else evaluation_payload),
                "reused": launched.reused,
            }
        )
    except Exception as exc:
        print(f"{RED}Evaluation launch failed:{NC} {exc}")
        sys.exit(1)
    finally:
        service.shutdown(wait=wait_for_result)


def cmd_eval_jobs(args):
    service = _evaluation_lab_service()
    detached = False
    try:
        if args.cancel:
            payload = service.jobs.cancel(args.cancel).to_dict()
        elif args.retry:
            evaluation = service.jobs.retry(args.retry, submit=False)
            try:
                worker_pid = _spawn_evaluation_worker(evaluation.id)
            except Exception as exc:
                service.db.update_evaluation(
                    evaluation.id,
                    status="failed",
                    stage="worker_launch_failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            payload = evaluation.to_dict()
            payload["worker_pid"] = worker_pid
            detached = True
        elif args.evaluation_id:
            payload = service.evaluation_detail(args.evaluation_id)
        else:
            payload = {
                "items": [
                    item.to_dict()
                    for item in service.db.list_evaluations(status=args.status, limit=args.limit)
                ]
            }
        _print_dataset_lab_payload(payload)
    except Exception as exc:
        print(f"{RED}Evaluation job operation failed:{NC} {exc}")
        sys.exit(1)
    finally:
        service.shutdown(wait=not detached)


def cmd_eval_compare(args):
    service = _evaluation_lab_service()
    try:
        _print_dataset_lab_payload(service.compare(args.base, args.candidate))
    except Exception as exc:
        print(f"{RED}Evaluation comparison failed:{NC} {exc}")
        sys.exit(1)
    finally:
        service.shutdown()


def cmd_eval_history(args):
    try:
        database, scheduler, _operations = _experiment_runtime(args)
        public = _adaptive_public_service(database, scheduler)
        _print_dataset_lab_payload(
            public.evaluation_history(
                subject_ref=args.subject,
                suite_revision_id=args.suite_revision,
                limit=args.limit,
            )
        )
    except Exception as exc:
        print(f"{RED}Evaluation history failed:{NC} {exc}")
        sys.exit(1)


def cmd_eval_drift(args):
    try:
        database, scheduler, _operations = _experiment_runtime(args)
        public = _adaptive_public_service(database, scheduler)
        _print_dataset_lab_payload(
            public.evaluation_drift(
                base_id=args.base,
                candidate_id=args.candidate,
                subject_ref=args.subject,
                suite_revision_id=args.suite_revision,
                practical_delta=args.practical_delta,
            )
        )
    except Exception as exc:
        print(f"{RED}Evaluation drift comparison failed:{NC} {exc}")
        sys.exit(1)


def cmd_eval(args):
    """Run lm-evaluation-harness benchmarks (Track V8)."""
    eval_command = getattr(args, "eval_command", None)
    if eval_command == "suite":
        return cmd_eval_suite(args)
    if eval_command == "run":
        return cmd_eval_run(args)
    if eval_command == "jobs":
        return cmd_eval_jobs(args)
    if eval_command == "compare":
        return cmd_eval_compare(args)
    if eval_command == "history":
        return cmd_eval_history(args)
    if eval_command == "drift":
        return cmd_eval_drift(args)

    from pathlib import Path

    from halo_forge.eval import list_curated_task_groups, run_lm_eval

    print_banner()
    print(f"{GREEN}halo-forge eval{NC}")
    print("=" * 60)

    if getattr(args, "list_tasks", False):
        groups = list_curated_task_groups()
        print(f"Curated task groups (use any of these as --tasks <name>):")
        for name, members in groups.items():
            print(f"  {CYAN}{name:<22}{NC} {', '.join(members)}")
        print()
        print("Or pass any lm-eval task name directly (e.g. mmlu_pro_law).")
        return

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"  model:    {args.model}")
    print(f"  backend:  {args.backend}")
    print(f"  tasks:    {tasks}")
    if args.limit:
        print(f"  limit:    {args.limit} per task")
    if args.output:
        print(f"  output:   {args.output}")
    print()

    try:
        result = run_lm_eval(
            model_name=args.model,
            tasks=tasks,
            limit=args.limit,
            batch_size=args.batch_size,
            backend=args.backend,
            output_dir=Path(args.output) if args.output else None,
        )
    except Exception as exc:
        print(f"{RED}Eval failed:{NC} {exc}")
        sys.exit(1)

    print(f"{GREEN}Done{NC} in {result.duration_seconds:.1f}s")
    print(f"  tasks completed: {result.n_tasks_completed}")
    if result.n_tasks_failed:
        print(f"  tasks failed:    {result.n_tasks_failed}")
    print()
    for task in result.task_results:
        marker = f"{RED}✗{NC}" if task.error else f"{GREEN}✓{NC}"
        print(
            f"  {marker} {task.task:<24} "
            f"{task.primary_metric:>22} = "
            f"{task.value:>7.4f}" + (f"  (n={task.n_samples})" if task.n_samples else "")
        )
    avg = result.average_score()
    if avg is not None:
        print(f"\n  {GREEN}average primary metric:{NC} {avg:.4f}")
    if result.n_tasks_completed == 0:
        print(f"\n{RED}No eval tasks completed successfully; results are not trustworthy.{NC}")
        sys.exit(1)


def cmd_token(args):
    """Manage API tokens for the public API (Track P1)."""
    from halo_forge.auth import TokenStore, default_store_path

    store = TokenStore()
    sub = args.token_command

    print_banner()
    print(f"{GREEN}halo-forge token{NC}")
    print("=" * 60)

    if sub == "create":
        try:
            secret = store.add_token(name=args.name, note=args.note)
        except ValueError as exc:
            print(f"{RED}Error:{NC} {exc}")
            sys.exit(1)
        print(f"  name:   {args.name}")
        print(f"  store:  {default_store_path()}")
        print()
        print(f"{YELLOW}Save this token now — it won't be shown again:{NC}")
        print(f"  {secret}")
        print()
        print("Use it with:")
        print(f"  curl -H 'Authorization: Bearer {secret}' http://<host>/api/public/health")
    elif sub == "list":
        tokens = store.list_tokens()
        if not tokens:
            print("(no tokens)")
            print(f"Store: {default_store_path()}")
            print("Create one: halo-forge token create <name>")
            return
        print(f"Store: {default_store_path()}")
        print()
        print(f"  {'NAME':<24} {'CREATED':<26} {'LAST USED':<26} NOTE")
        for t in tokens:
            print(
                f"  {t.name:<24} {t.created_at:<26} "
                f"{(t.last_used_at or '—'):<26} {t.note or ''}"
            )
    elif sub == "revoke":
        ok = store.revoke(args.name)
        if ok:
            print(f"{GREEN}Revoked{NC} {args.name!r}")
        else:
            print(f"{YELLOW}No token named{NC} {args.name!r}")
            sys.exit(1)


def cmd_replay(args):
    """Show or execute the replay command for a captured run (Track T15)."""
    from pathlib import Path

    from halo_forge.replay import (
        EnvironmentFingerprint,
        compare_environments,
        compare_reward_identities,
        compare_verifier_identities,
        load_manifest,
        hash_dataset_file,
    )

    print_banner()
    print(f"{GREEN}halo-forge replay{NC}")
    print("=" * 60)

    try:
        manifest = load_manifest(Path(args.source))
    except FileNotFoundError as exc:
        print(f"{RED}Error:{NC} {exc}")
        sys.exit(1)

    print(f"  run_id:    {manifest.run_id}")
    print(f"  modality:  {manifest.modality}")
    print(f"  model:     {manifest.model_name}")
    print(f"  seed:      {manifest.seed}")
    print(f"  timestamp: {manifest.timestamp}")
    print()

    # Environment diff vs the active host.
    current = EnvironmentFingerprint.capture().to_dict()
    diff = compare_environments(manifest.environment, current)
    if diff["matched"]:
        print(f"{GREEN}Environment matches{NC} the captured run.")
    else:
        print(f"{YELLOW}Environment differs from the captured run:{NC}")
        for d in diff["differences"][:20]:
            print(f"  {d['key']:>26}: {d['captured']!r} -> {d['current']!r}")
        if len(diff["differences"]) > 20:
            print(f"  ... {len(diff['differences']) - 20} more")

    dataset_mismatch = False
    dataset_info = manifest.dataset or {}
    if dataset_info.get("kind") == "local_file" and dataset_info.get("sha256"):
        dataset_path = Path(str(dataset_info.get("path") or ""))
        if not dataset_path.exists():
            dataset_mismatch = True
            print(f"{RED}Dataset missing:{NC} {dataset_path}")
        else:
            current_sha = hash_dataset_file(dataset_path)
            if current_sha != dataset_info.get("sha256"):
                dataset_mismatch = True
                print(f"{RED}Dataset hash mismatch:{NC} {dataset_path}")
                print(f"  captured: {dataset_info.get('sha256')}")
                print(f"  current:  {current_sha}")

    verifier_mismatch = False
    verifier_differences: list[dict[str, Any]] = []
    verifier_info = dict(manifest.verifier or {})
    verifier_revision_id = str(verifier_info.get("verifier_profile_revision_id") or "").strip()
    if verifier_info.get("legacy_unqualified"):
        print()
        print(
            f"{YELLOW}Legacy unqualified verifier:{NC} "
            + str(
                verifier_info.get("legacy_warning")
                or "this replay does not pin an immutable verifier revision"
            )
        )
    elif verifier_revision_id:
        try:
            from halo_forge.run_db import get_database
            from halo_forge.verifier_lab.service import VerifierLabService

            current_verifier = VerifierLabService(get_database()).resolve_binding(
                verifier_revision_id
            )
            verifier_diff = compare_verifier_identities(verifier_info, current_verifier)
            verifier_mismatch = not verifier_diff["matched"]
            verifier_differences = list(verifier_diff["differences"])
        except Exception as exc:
            verifier_mismatch = True
            verifier_differences = [
                {
                    "key": "verifier_profile_revision_id",
                    "captured": verifier_revision_id,
                    "current": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ]
        if verifier_mismatch:
            print()
            print(f"{RED}Verifier identity differs from the captured run:{NC}")
            for difference in verifier_differences[:20]:
                print(
                    f"  {difference.get('key', 'identity'):>26}: "
                    f"{difference.get('captured')!r} -> {difference.get('current')!r}"
                )
                if difference.get("error"):
                    print(f"  {'error':>26}: {difference['error']}")
        else:
            print()
            print(f"{GREEN}Verifier identity matches{NC} the captured run.")
    elif manifest.manifest_version >= 3 and verifier_info:
        verifier_mismatch = True
        verifier_differences = [
            {
                "key": "verifier_profile_revision_id",
                "captured": None,
                "current": None,
                "error": "v3 verifier identity is incomplete",
            }
        ]
        print(f"{RED}Verifier identity is incomplete in this v3 replay manifest.{NC}")

    reward_mismatch = False
    reward_differences: list[dict[str, Any]] = []
    reward_info = dict(manifest.reward_integrity or {})
    reward_revision_id = str(
        reward_info.get("reward_system_revision_id") or ""
    ).strip()
    if reward_info.get("legacy_unmonitored"):
        print()
        print(
            f"{YELLOW}Legacy unmonitored reward launch:{NC} "
            "no sealed same-output training signal was captured."
        )
    elif reward_revision_id:
        try:
            current_reward = _resolve_current_reward_identity(reward_info)
            reward_diff = compare_reward_identities(reward_info, current_reward)
            reward_mismatch = not reward_diff["matched"]
            reward_differences = list(reward_diff["differences"])
        except Exception as exc:
            reward_mismatch = True
            reward_differences = [
                {
                    "key": "reward_system_revision_id",
                    "captured": reward_revision_id,
                    "current": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ]
        print()
        if reward_mismatch:
            print(f"{RED}Reward-system identity differs from the captured run:{NC}")
            for difference in reward_differences[:20]:
                print(
                    f"  {difference.get('key', 'identity'):>26}: "
                    f"{difference.get('captured')!r} -> {difference.get('current')!r}"
                )
                if difference.get("error"):
                    print(f"  {'error':>26}: {difference['error']}")
        else:
            print(f"{GREEN}Reward-system identity matches{NC} the captured run.")
    elif manifest.manifest_version >= 4 and reward_info:
        reward_mismatch = True
        reward_differences = [
            {
                "key": "reward_system_revision_id",
                "captured": None,
                "current": None,
                "error": "v4 reward-system identity is incomplete",
            }
        ]
        print(f"{RED}Reward-system identity is incomplete in this v4 replay manifest.{NC}")

    # Reconstruct the launch command.
    cmd = _reconstruct_launch_command(manifest)
    print()
    print(f"{GREEN}Reproducible launch command:{NC}")
    print(f"  {' '.join(cmd)}")

    if args.launch:
        allow_verifier_drift = bool(getattr(args, "allow_verifier_drift", False))
        allow_reward_drift = bool(getattr(args, "allow_reward_drift", False))
        if (
            allow_verifier_drift
            and not str(getattr(args, "verifier_drift_reason", None) or "").strip()
        ):
            print()
            print(
                f"{RED}Refusing to launch{NC}: --allow-verifier-drift requires "
                "--verifier-drift-reason."
            )
            sys.exit(2)
        if (
            allow_reward_drift
            and not str(getattr(args, "reward_drift_reason", None) or "").strip()
        ):
            print()
            print(
                f"{RED}Refusing to launch{NC}: --allow-reward-drift requires "
                "--reward-drift-reason."
            )
            sys.exit(2)
        if not diff["matched"] and not args.force:
            print()
            print(
                f"{RED}Refusing to launch{NC}: environment differs. "
                "Pass --force to launch anyway."
            )
            sys.exit(2)
        if dataset_mismatch and not args.allow_dataset_drift:
            print()
            print(
                f"{RED}Refusing to launch{NC}: dataset content differs from replay manifest. "
                "Pass --allow-dataset-drift to launch anyway."
            )
            sys.exit(2)
        if verifier_mismatch and not allow_verifier_drift:
            print()
            print(
                f"{RED}Refusing to launch{NC}: verifier identity is missing or drifted. "
                "Pass --allow-verifier-drift with --verifier-drift-reason to override."
            )
            sys.exit(2)
        if verifier_mismatch:
            _record_verifier_drift_override(
                Path(args.source),
                manifest.run_id,
                verifier_differences,
                str(args.verifier_drift_reason).strip(),
            )
        if reward_mismatch and not allow_reward_drift:
            print()
            print(
                f"{RED}Refusing to launch{NC}: reward-system identity is missing or drifted. "
                "Pass --allow-reward-drift with --reward-drift-reason to override."
            )
            sys.exit(2)
        if reward_mismatch:
            _record_replay_drift_override(
                Path(args.source),
                manifest.run_id,
                event_type="allow_reward_drift",
                differences=reward_differences,
                reason=str(args.reward_drift_reason).strip(),
            )
        print()
        print(f"{GREEN}Launching...{NC}")
        # Replay invokes our own CLI re-entrantly so env vars + logging
        # are wired the same way as a normal launch.
        import subprocess

        completed = subprocess.run(cmd, check=False)
        sys.exit(completed.returncode)


def _record_verifier_drift_override(
    source: Path,
    run_id: str,
    differences: list[dict[str, Any]],
    reason: str,
) -> Path:
    """Append an operator-authorized verifier drift event beside replay.json."""

    return _record_replay_drift_override(
        source,
        run_id,
        event_type="allow_verifier_drift",
        differences=differences,
        reason=reason,
    )


def _record_replay_drift_override(
    source: Path,
    run_id: str,
    *,
    event_type: str,
    differences: list[dict[str, Any]],
    reason: str,
) -> Path:
    """Append one explicit, operator-authored replay drift event."""

    from datetime import datetime, timezone

    manifest_path = source / "replay.json" if source.is_dir() else source
    target = manifest_path.parent / "replay_overrides.jsonl"
    event = {
        "event_id": f"replay-override-{uuid.uuid4().hex}",
        "event_type": event_type,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "differences": differences,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _resolve_current_reward_identity(captured: dict[str, Any]) -> dict[str, Any]:
    """Resolve the current immutable reward identity for an exact replay check."""

    from halo_forge.reward_integrity import RewardIntegrityService
    from halo_forge.run_db import get_database

    service = RewardIntegrityService(get_database())
    resolved = service.resolve_binding(
        str(captured.get("reward_system_revision_id") or ""),
        protocol_revision_id=str(captured.get("protocol_revision_id") or "") or None,
        integrity_profile_revision_id=(
            str(captured.get("integrity_profile_revision_id") or "") or None
        ),
        boundaries=list(captured.get("boundaries") or []),
    )
    from halo_forge.training_signal import load_training_signal_shard

    verified_traces: list[dict[str, Any]] = []
    for expected in captured.get("trace_manifests") or []:
        loaded = load_training_signal_shard(str(expected.get("path") or ""))
        verified_traces.append(
            {
                "shard_id": loaded.shard_id,
                "trace_hash": loaded.trace_hash,
                "path": loaded.path,
                "boundary": loaded.boundary,
                "checkpoint_hash": loaded.checkpoint_hash,
                "capture_fidelity": loaded.capture_fidelity,
                "observed_count": loaded.observed_count,
                "retained_count": loaded.retained_count,
            }
        )
    verified_audits: list[dict[str, Any]] = []
    for expected in captured.get("audit_decisions") or []:
        audit_id = str(expected.get("audit_id") or "")
        audit = service.get_audit(audit_id)
        decisions = service.store.list_decisions(audit_id, limit=1000).items
        latest = decisions[-1] if decisions else None
        if audit.status == "completed":
            service.verify_audit_bundle(audit.id)
        value = {
            "audit_id": audit.id,
            "status": audit.status,
            "integrity_profile_revision_id": audit.integrity_profile_revision_id,
            "work_item_id": audit.work_item_id,
        }
        if latest is not None:
            value.update(
                decision_id=latest.id,
                decision=latest.decision,
                action=latest.action,
                reasons=list(latest.reasons),
                audit_manifest_hash=audit.manifest_hash,
            )
        verified_audits.append(value)
    if hasattr(resolved, "to_replay_dict"):
        current = dict(resolved.to_replay_dict())
        current["trace_manifests"] = verified_traces
        current["audit_decisions"] = verified_audits
        return current
    value = resolved.to_dict() if hasattr(resolved, "to_dict") else dict(resolved)
    if "reward_system_revision_id" in value:
        value["trace_manifests"] = verified_traces
        value["audit_decisions"] = verified_audits
        return value
    revision = dict(value.get("reward_system_revision") or {})
    protocol = dict(value.get("protocol_revision") or {})
    profile = dict(value.get("integrity_profile_revision") or {})
    captured_capability = dict(captured.get("signal_capability") or {})
    capability_id = str(captured_capability.get("id") or "").strip()
    if capability_id:
        from halo_forge.training_signal import TRAINING_SIGNAL_CAPABILITIES

        current_capability = TRAINING_SIGNAL_CAPABILITIES.get(capability_id).to_dict()
    else:
        current_capability = {}
    blockers = [str(item) for item in value.get("blockers") or []]
    stale_runtime = any("stale_runtime" in item for item in blockers)
    gating_eligible = bool(value.get("gating_eligible", False))
    runtime_state = (
        "stale_runtime"
        if stale_runtime
        else ("compatible" if gating_eligible else "ineligible")
    )
    return {
        "reward_system_revision_id": revision.get("id"),
        "reward_system_hash": revision.get("content_hash"),
        "optimizer_verifier_revision_id": revision.get(
            "optimizer_verifier_revision_id"
        ),
        "auditors": list(revision.get("auditors") or []),
        "reward_mapping_hash": revision.get("reward_mapping_hash")
        or revision.get("content_hash"),
        "protocol_revision_id": protocol.get("id"),
        "protocol_hash": protocol.get("content_hash"),
        "integrity_profile_revision_id": profile.get("id"),
        "integrity_profile_hash": profile.get("content_hash"),
        "boundaries": list(captured.get("boundaries") or []),
        "signal_capability": current_capability,
        "trace_manifests": verified_traces,
        "audit_decisions": verified_audits,
        "runtime_compatibility": {
            "state": runtime_state,
            "compatible": gating_eligible,
            "blockers": blockers,
        },
    }


def _reconstruct_launch_command(manifest) -> list[str]:
    """Translate a replay manifest back into a ``halo-forge`` CLI invocation.

    Maps the modality + the captured config onto the right subcommand
    and forwards the small set of fields the CLI exposes. Fields the
    CLI doesn't accept stay in the manifest but aren't part of the
    command — they're already represented by the seed + config keys
    that *are* exposed.
    """
    cfg = manifest.config or {}
    modality = (manifest.modality or "").lower()

    # Map modality → subcommand. Each path picks a small set of keys
    # to forward; the full config is already on disk in replay.json
    # for anyone wanting the exhaustive picture.
    if modality == "sft":
        subcmd = ["sft", "train"]
    elif modality == "raft":
        subcmd = ["raft", "train"]
    elif modality == "dpo":
        subcmd = ["dpo", "train"]
    elif modality.startswith("dpo_mlx"):
        subcmd = ["dpo", "train"]
    elif modality == "grpo":
        subcmd = ["grpo", "train"]
    elif modality.startswith("grpo_mlx"):
        subcmd = ["grpo", "train"]
    elif modality == "orpo":
        subcmd = ["orpo", "train"]
    elif modality == "rm":
        subcmd = ["rm", "train"]
    elif modality in {"vlm", "audio", "reasoning", "agentic"}:
        subcmd = [modality, "train"]
    else:
        # Unknown modality — show "halo-forge --help" and let the user
        # reconstruct manually from the config they can see.
        return ["halo-forge", "# unknown modality:", modality, "see replay.json"]

    cmd = ["halo-forge", *subcmd]
    model_name = cfg.get("model_name") or cfg.get("base_model") or manifest.model_name
    if model_name:
        cmd += ["--model", str(model_name)]
    dataset_info = manifest.dataset or {}
    managed_bindings = (
        list(dataset_info.get("bindings") or [])
        if dataset_info.get("kind") == "managed_versions"
        else []
    )
    if managed_bindings:
        for binding in managed_bindings:
            cmd += [
                "--dataset-binding",
                (f"{binding['role']}={binding['dataset_version_id']}:" f"{binding['split']}"),
            ]
    elif cfg.get("dataset"):
        cmd += ["--dataset", str(cfg["dataset"])]
    elif cfg.get("train_file"):
        cmd += ["--data", str(cfg["train_file"])]
    if cfg.get("output_dir"):
        cmd += ["--output", str(cfg["output_dir"])]
    if cfg.get("num_epochs"):
        cmd += ["--epochs", str(cfg["num_epochs"])]
    if cfg.get("max_samples"):
        cmd += ["--max-samples", str(cfg["max_samples"])]
    verifier_revision_id = str(
        (manifest.verifier or {}).get("verifier_profile_revision_id") or ""
    ).strip()
    if verifier_revision_id:
        cmd += ["--verifier-profile-revision", verifier_revision_id]
    if "seed" in cfg:
        cmd += ["--seed", str(cfg["seed"])]
    return cmd


def cmd_convert(args):
    """Convert a model between formats (Track I5).

    Wraps mlx_lm.convert / GGUFExporter / HF dtype-recast behind one
    consistent CLI vocabulary. ``--quant q4`` means "4-bit affine
    quantization with group size 64" regardless of which format you
    target; the dispatch translates to the underlying tool's args.
    """
    from halo_forge.inference.convert import (
        convert as run_convert,
        list_supported_formats,
        list_supported_quants,
    )

    print_banner()
    print(f"{GREEN}halo-forge convert{NC}")
    print("=" * 60)

    if getattr(args, "list", False):
        print(f"Supported formats: {', '.join(list_supported_formats())}")
        print(f"Supported quants:  {', '.join(list_supported_quants())}")
        return

    print(f"  source: {args.source}")
    print(f"  format: {args.format}")
    print(f"  quant:  {args.quant}")
    print(f"  output: {args.output}")
    print()
    try:
        result = run_convert(
            source=args.source,
            output_path=args.output,
            target_format=args.format,
            quantization=args.quant,
            trust_remote_code=args.trust_remote_code,
            allow_unquantized_fallback=getattr(args, "allow_unquantized_fallback", False),
        )
    except Exception as exc:
        print(f"{RED}Conversion failed:{NC} {exc}")
        sys.exit(1)

    size_mb = (result.bytes_written or 0) / (1024 * 1024)
    print(f"{GREEN}Converted{NC} -> {result.output_path}")
    print(f"  size: {size_mb:.1f} MB")
    if result.notes:
        print(f"  note: {result.notes}")

    # Track I4 — opt-in round-trip verification right after conversion.
    if getattr(args, "verify", False):
        from halo_forge.inference.verify_export import verify_export

        print()
        print(f"{GREEN}Verifying export round-trip{NC}")
        try:
            report = verify_export(
                source_model=args.source,
                exported_path=result.output_path,
                target_format=args.format,
            )
        except NotImplementedError as exc:
            print(f"{RED}Verification unsupported:{NC} {exc}")
            sys.exit(1)
        print(f"  prompts:               {report.n_prompts}")
        print(f"  exact match rate:      {report.exact_match_rate:.2%}")
        print(f"  avg char overlap:      {report.avg_char_overlap:.3f}")
        print(f"  first-token match:     {report.avg_first_token_match:.2%}")
        print(f"  duration:              {report.duration_seconds:.1f}s")
        if report.passed:
            print(f"{GREEN}Round-trip verification passed.{NC}")
        else:
            print(
                f"{RED}Round-trip verification failed{NC} — exported model "
                f"diverges from source. Failures: {len(report.failures)}/"
                f"{report.n_prompts}"
            )
            sys.exit(1)


def cmd_serve(args):
    """Run an OpenAI-compatible serving endpoint (Track I1).

    Spins up uvicorn on `--host`/`--port` (default 127.0.0.1:8001) serving
    `--model` via the active backend. The model loads lazily on the first
    request so `halo-forge serve` returns control quickly even for large
    weights — the first chat call eats the load cost.
    """
    if args.backend:
        backend_display = f"{args.backend} (forced)"
    else:
        try:
            from halo_forge.backend import get_backend

            backend_display = f"{get_backend().name} (auto)"
        except Exception:
            backend_display = "auto"

    print_banner()
    print(f"{GREEN}halo-forge serve{NC} — OpenAI-compatible endpoint")
    print("=" * 60)
    print(f"  model:               {args.model}")
    print(f"  bind:                {args.host}:{args.port}")
    print(f"  backend:             {backend_display}")
    print(f"  adapter load:        lazy (first generation request)")
    print(f"  streaming:           OpenAI SSE supported")
    print(f"  trust remote code:   {bool(args.trust_remote_code)}")
    print(f"  health:              http://{args.host}:{args.port}/health")
    print(f"  models:              http://{args.host}:{args.port}/v1/models")
    print()

    if getattr(args, "check", False):
        print(f"{GREEN}Serve preflight OK.{NC} No server started.")
        return

    print("Try:")
    print(f"  curl http://{args.host}:{args.port}/v1/models")
    print(
        f"  curl http://{args.host}:{args.port}/v1/chat/completions "
        '-H "Content-Type: application/json" '
        '-d \'{"model":"' + args.model + '","messages":[{"role":"user","content":"hi"}]}\''
    )
    print()

    import uvicorn

    from halo_forge.serving.app import create_serving_app

    app = create_serving_app(
        model_name=args.model,
        backend_name=args.backend,
        trust_remote_code=args.trust_remote_code,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_serve_public(args):
    """Run the dashboard FastAPI (Track F-* surface).

    This is the API the public_app SPA talks to — runs/eval/registry/
    playground/lineage/verifiers/metrics. Different process from the
    OpenAI-compatible inference server (`halo-forge serve`); a typical
    workstation runs both side by side on different ports.
    """
    print_banner()
    print(f"{GREEN}halo-forge serve-public{NC} — dashboard API")
    print("=" * 60)
    is_loopback = str(args.host) in {"127.0.0.1", "localhost", "::1"}
    remote_app_host = "<workstation-host>" if str(args.host) == "0.0.0.0" else str(args.host)
    print(f"  bind:        {args.host}:{args.port}")
    print(f"  health:      http://{args.host}:{args.port}/api/public/health")
    print(f"  local app:   cd public_app && npm run dev")
    print(f"  open app:    http://127.0.0.1:3000")
    print(f"  app command: halo-forge dashboard")
    if not is_loopback:
        print(f"  remote app:  cd public_app && npm run dev -- --host 0.0.0.0")
        print(f"  remote URL:  http://{remote_app_host}:3000")
    print(f"  remote auth: {'loopback bypass' if is_loopback else 'required'}")
    print()

    if getattr(args, "check", False):
        print(f"{GREEN}Dashboard API preflight OK.{NC} No server started.")
        return

    if is_loopback:
        print("Local development:")
        print("  Terminal 1: halo-forge serve-public")
        print("  Terminal 2: cd public_app && npm install && npm run dev")
    else:
        print("Remote development:")
        print("  Terminal 1a: halo-forge token create dashboard")
        print(f"  Terminal 1b: halo-forge serve-public --host {args.host}")
        print("  Terminal 2:  cd public_app && npm install && npm run dev -- --host 0.0.0.0")
        print(f"  Browser:    http://{remote_app_host}:3000")
    print()

    import uvicorn

    from halo_forge.public_api.app import create_app

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _dashboard_is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def _dashboard_open_url(host: str, port: int) -> str:
    display_host = "<workstation-host>" if host == "0.0.0.0" else host
    return f"http://{display_host}:{port}"


def _public_app_source_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "public_app"


def _build_public_app() -> bool:
    app_dir = _public_app_source_dir()
    package_json = app_dir / "package.json"
    if not package_json.is_file():
        print(f"{RED}Dashboard source not found at {app_dir}.{NC}")
        return False
    if shutil.which("npm") is None:
        print(f"{RED}npm is required to build the dashboard assets, but it was not found.{NC}")
        print("Install Node.js/npm, then run:")
        print("  cd public_app && npm install && npm run build")
        return False

    node_modules = app_dir / "node_modules"
    if not node_modules.is_dir():
        print(f"{CYAN}Installing dashboard dependencies...{NC}")
        install = subprocess.run(["npm", "install"], cwd=app_dir)
        if install.returncode != 0:
            return False

    print(f"{CYAN}Building dashboard assets...{NC}")
    build = subprocess.run(["npm", "run", "build"], cwd=app_dir)
    return build.returncode == 0


def cmd_dashboard(args):
    """Run the user-facing dashboard as a single app command."""
    from halo_forge.public_api.app import create_app, find_frontend_dist

    host = str(args.host)
    port = int(args.port)
    is_loopback = _dashboard_is_loopback(host)
    app_url = _dashboard_open_url(host, port)
    dist = find_frontend_dist()

    print_banner()
    print(f"{GREEN}halo-forge dashboard{NC} — workstation app")
    print("=" * 60)
    print(f"  bind:        {host}:{port}")
    print(f"  open app:    {app_url}")
    print(f"  health:      http://{host}:{port}/api/public/health")
    print(f"  remote auth: {'loopback bypass' if is_loopback else 'required'}")
    if dist is None:
        print("  app assets:  missing; will build from public_app/")
        print("  build:       cd public_app && npm install && npm run build")
    else:
        print(f"  app assets:  {dist}")
    print()

    if getattr(args, "check", False):
        print(f"{GREEN}Dashboard preflight OK.{NC} No server started.")
        return

    if dist is None:
        if getattr(args, "no_build", False):
            print(f"{RED}Dashboard assets are not built.{NC}")
            print(
                "Run `cd public_app && npm install && npm run build`, then rerun `halo-forge dashboard`."
            )
            sys.exit(2)
        if not _build_public_app():
            print(f"{RED}Could not build dashboard assets.{NC}")
            sys.exit(2)
        dist = find_frontend_dist()
        if dist is None:
            print(
                f"{RED}Dashboard build finished, but public_app/dist/index.html was not found.{NC}"
            )
            sys.exit(2)

    if not is_loopback:
        print("Remote workstation:")
        print("  Create a dashboard token first: halo-forge token create dashboard")
        print(f"  Open from another machine:     {app_url}")
        print("  Paste the hfk_... token in Connection.")
    else:
        print(f"Open {app_url}")
    print()

    if getattr(args, "open", False) and is_loopback:
        webbrowser.open(app_url)
    elif getattr(args, "open", False):
        print("Skipping --open for non-loopback host; open the URL from the remote browser.")

    import uvicorn

    app = create_app(frontend_dist=dist, serve_frontend=True)
    uvicorn.run(app, host=host, port=port, log_level="info")


def cmd_rm_train(args):
    """Train a Bradley-Terry reward model (Track T3)."""
    _apply_managed_dataset_args(args, "rm", "data")
    from halo_forge.rm import RMConfig, get_rm_trainer

    print_banner()
    print(f"{GREEN}Reward-Model Training{NC}")
    print("=" * 60)

    dataset = getattr(args, "dataset", None)
    data = getattr(args, "data", None)
    if not dataset and not data:
        print(f"{RED}Error: Either --dataset or --data is required{NC}")
        print()
        print("Examples:")
        print("  halo-forge rm train --dataset ultrafeedback --model Qwen/Qwen2.5-3B-Instruct")
        print("  halo-forge rm train --data my_pairs.jsonl --model Qwen/Qwen2.5-3B-Instruct")
        sys.exit(1)

    config = RMConfig(
        model_name=args.model,
        train_file=data,
        validation_file=getattr(args, "validation_data", None),
        dataset=dataset,
        max_samples=getattr(args, "max_samples", None),
        max_length=getattr(args, "max_length", 1024),
        output_dir=args.output,
        num_epochs=getattr(args, "epochs", 1),
        batch_size=getattr(args, "batch_size", 4),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation", 4),
        learning_rate=getattr(args, "learning_rate", 1e-5),
        warmup_ratio=getattr(args, "warmup_ratio", 0.05),
        weight_decay=getattr(args, "weight_decay", 0.0),
        max_grad_norm=getattr(args, "max_grad_norm", 1.0),
        lora_r=getattr(args, "lora_rank", 8),
        lora_alpha=getattr(args, "lora_alpha", 16),
        lora_dropout=getattr(args, "lora_dropout", 0.05),
        use_dora=getattr(args, "use_dora", False),
        use_rslora=getattr(args, "use_rslora", False),
        init_lora_weights=getattr(args, "init_lora_weights", "true"),
        optim=getattr(args, "optim", "adamw_torch"),
        load_in_4bit=getattr(args, "load_in_4bit", False),
        enable_neural_accelerators=getattr(args, "enable_neural_accelerators", False),
        center_rewards_coefficient=getattr(args, "center_rewards_coefficient", 0.01),
        gradient_checkpointing=not getattr(args, "no_gradient_checkpointing", False),
        seed=getattr(args, "seed", 42),
        max_steps=getattr(args, "max_steps", None),
        save_steps=getattr(args, "save_steps", 200),
        save_total_limit=getattr(args, "save_total_limit", 3),
    )

    if getattr(args, "dry_run", False):
        print("Dry run: configuration validated. No training started.")
        print(f"  model={config.model_name} dataset={config.dataset or '(local)'}")
        print(f"  lora_rank={config.lora_r} lr={config.learning_rate}")
        return

    trainer = get_rm_trainer(config)
    summary = trainer.train(resume_from_checkpoint=args.resume)
    _print_completed_training_summary("rm", config.output_dir, summary, args=args, config=config)


def cmd_grpo_train(args):
    """Run GRPO training (Track T2 / phase Q1).

    Wraps trl.GRPOTrainer for the PyTorch path; MLX path uses an
    in-house reference-free/reference-model implementation. Verifier comes from
    the plugin registry (V1) — `--verifier execution` (default), or any
    registered short name (e.g. `llm_judge` from V2).
    """
    _prepare_reward_integrity(args, trainer="grpo")
    if _enqueue_managed_reward_training(args, trainer="grpo"):
        return
    _prepare_profile_verifier(args, consumer="registry", modality="text", training=True)
    _apply_managed_dataset_args(args, "grpo", "data")
    from halo_forge.grpo import GRPOConfig, get_grpo_trainer

    print_banner()
    print(f"{GREEN}GRPO Training{NC}")
    print("=" * 60)

    dataset = getattr(args, "dataset", None)
    data = getattr(args, "data", None)
    if not dataset and not data:
        print(f"{RED}Error: Either --dataset or --data is required{NC}")
        print()
        print("Examples:")
        print(
            "  halo-forge grpo train --data prompts.jsonl --model Qwen/Qwen2.5-3B-Instruct --verifier execution"
        )
        print("  halo-forge grpo train --dataset gsm8k --verifier execution --num-generations 8")
        sys.exit(1)

    config = GRPOConfig(
        model_name=args.model,
        train_file=data,
        dataset=dataset,
        max_samples=getattr(args, "max_samples", None),
        max_prompt_length=getattr(args, "max_prompt_length", 512),
        max_completion_length=getattr(args, "max_completion_length", 512),
        num_generations=getattr(args, "num_generations", 4),
        beta=getattr(args, "beta", 0.04),
        epsilon=getattr(args, "epsilon", 0.2),
        temperature=getattr(args, "temperature", 0.9),
        scale_rewards=not getattr(args, "no_scale_rewards", False),
        reference_free=getattr(args, "reference_free", False),
        verifier_name=getattr(args, "verifier", "execution"),
        reward_threshold=getattr(args, "reward_threshold", 0.0),
        output_dir=args.output,
        num_epochs=getattr(args, "epochs", 1),
        batch_size=getattr(args, "batch_size", 1),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation", 16),
        learning_rate=getattr(args, "learning_rate", 1e-6),
        warmup_ratio=getattr(args, "warmup_ratio", 0.1),
        weight_decay=getattr(args, "weight_decay", 0.0),
        max_grad_norm=getattr(args, "max_grad_norm", 1.0),
        lora_r=getattr(args, "lora_rank", 16),
        lora_alpha=getattr(args, "lora_alpha", 32),
        lora_dropout=getattr(args, "lora_dropout", 0.05),
        use_dora=getattr(args, "use_dora", False),
        use_rslora=getattr(args, "use_rslora", False),
        init_lora_weights=getattr(args, "init_lora_weights", "true"),
        optim=getattr(args, "optim", "adamw_torch"),
        load_in_4bit=getattr(args, "load_in_4bit", False),
        rollout_engine=getattr(args, "rollout_engine", "auto"),
        enable_neural_accelerators=getattr(args, "enable_neural_accelerators", False),
        gradient_checkpointing=not getattr(args, "no_gradient_checkpointing", False),
        seed=getattr(args, "seed", 42),
        max_steps=getattr(args, "max_steps", None),
        save_steps=getattr(args, "save_steps", 100),
        save_total_limit=getattr(args, "save_total_limit", 3),
    )

    if getattr(args, "dry_run", False):
        print("Dry run: configuration validated. No training started.")
        print(f"  model={config.model_name} dataset={config.dataset or '(local)'}")
        print(f"  num_generations={config.num_generations} beta={config.beta}")
        print(f"  verifier={config.verifier_name} reference_free={config.reference_free}")
        return

    signal_session = _build_training_signal_session(
        args,
        trainer="grpo",
        output_dir=config.output_dir,
        total_boundaries=max(1, int(config.max_steps or 1)),
        reward_threshold=config.reward_threshold,
    )
    trainer = get_grpo_trainer(config, signal_sink=signal_session)
    summary = trainer.train(resume_from_checkpoint=args.resume)
    _print_completed_training_summary("grpo", config.output_dir, summary, args=args, config=config)


def cmd_dpo_datasets(args):
    """List the canonical preference datasets halo-forge ships short names for."""
    from halo_forge.dpo.datasets import list_preference_datasets

    print_banner()
    print(f"{GREEN}Available preference datasets (DPO){NC}")
    print("=" * 60)
    print()
    for ds in list_preference_datasets():
        print(f"  {CYAN}{ds.name:<16}{NC} [{ds.size_hint:>6}] {ds.description}")
        print(f"                  HuggingFace: {ds.huggingface_id}")
    print()
    print("Usage:")
    print("  halo-forge dpo train --dataset ultrafeedback --model Qwen/Qwen2.5-3B-Instruct")
    print()


def cmd_sft_datasets(args):
    """List available SFT datasets."""
    from halo_forge.sft.datasets import list_sft_datasets

    print_banner()
    print(f"{GREEN}Available SFT Datasets{NC}")
    print("=" * 60)
    print()

    # Group by domain
    domains = ["code", "reasoning", "vlm", "audio", "agentic"]

    for domain in domains:
        datasets = list_sft_datasets(domain)
        if datasets:
            print(f"{YELLOW}{domain.upper()}{NC}")
            for ds in datasets:
                print(f"  {CYAN}{ds.name:<20}{NC} [{ds.size_hint:>6}] {ds.description}")
                print(f"                         HuggingFace: {ds.huggingface_id}")
            print()

    print("Usage:")
    print("  halo-forge sft train --dataset codealpaca --model Qwen/Qwen2.5-Coder-3B")
    print("  halo-forge sft train --dataset metamath --model Qwen/Qwen2.5-3B-Instruct")
    print()


def _resolve_model_path(model_path: str) -> tuple:
    """
    Resolve a model path that may be a base model ID or SFT output directory.

    Handles three cases:
    1. HuggingFace model ID (e.g., "Qwen/Qwen2.5-Coder-3B") - returns as-is
    2. SFT output directory with final_model/ subdirectory - auto-detects
    3. Direct LoRA adapter directory - reads base_model from adapter_config

    Returns:
        tuple: (base_model, sft_checkpoint) where base_model is the HuggingFace ID
               and sft_checkpoint is the path to the LoRA adapters (or None if fresh)
    """
    from pathlib import Path

    model_path_obj = Path(model_path)

    # Case 1: Not a local path, assume it's a HuggingFace model ID
    if not model_path_obj.exists():
        return (model_path, None)

    # Check for final_model subdirectory (SFT output pattern)
    final_model_path = model_path_obj / "final_model"
    if final_model_path.exists() and (final_model_path / "adapter_config.json").exists():
        checkpoint_path = final_model_path
    elif (model_path_obj / "adapter_config.json").exists():
        checkpoint_path = model_path_obj
    else:
        # It's a local path but not a LoRA adapter - might be a merged model
        return (model_path, None)

    # Read base model from adapter config
    adapter_config_path = checkpoint_path / "adapter_config.json"
    try:
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model = adapter_config.get("base_model_name_or_path")
        if base_model:
            return (base_model, str(checkpoint_path))
    except (json.JSONDecodeError, IOError):
        pass

    # Fallback: couldn't read config
    return (model_path, None)


def _load_prompts_jsonl(prompts_file: str) -> tuple:
    """
    Load prompts from a JSONL file.

    Returns:
        (prompts, invalid_lines)
    """
    prompts = []
    invalid_lines = []
    with open(prompts_file) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                invalid_lines.append((line_num, str(e)))
                continue
            prompts.append(data.get("prompt", data.get("text", "")))
    return prompts, invalid_lines


def cmd_raft_train(args):
    """Run RAFT training."""
    _prepare_reward_integrity(args, trainer="raft")
    if _enqueue_managed_reward_training(args, trainer="raft"):
        return
    profile_verifier = _prepare_profile_verifier(
        args, consumer="direct", modality="text", training=True
    )
    _apply_managed_dataset_args(args, "raft", "prompts")
    # Note: --experimental-attention is handled at script startup (before imports)

    print_banner()
    print(f"{GREEN}RAFT Training{NC}")
    print("=" * 60)

    import yaml
    from halo_forge.rlvr.raft_trainer import RAFTTrainer, RAFTConfig
    from halo_forge.rlvr.verifiers import (
        GCCVerifier,
        MinGWVerifier,
        RemoteMSVCVerifier,
        HumanEvalVerifier,
        MBPPVerifier,
    )

    # Load config
    if args.config:
        try:
            with open(args.config) as f:
                cfg_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error: Invalid YAML syntax in {args.config}: {e}")
            sys.exit(1)
    else:
        cfg_dict = {}

    if profile_verifier is not None and cfg_dict.get("verifier"):
        raise ValueError(
            "--verifier-profile-revision conflicts with verifier configuration in --config"
        )

    # Setup verifier
    verifier_type = args.verifier or cfg_dict.get("verifier", {}).get("type", "gcc")
    verifier_policy = (
        "unsafe_host" if getattr(args, "unsafe_verifier_execution", False) else "sandbox"
    )
    if verifier_policy == "unsafe_host":
        print("WARNING: --unsafe-verifier-execution runs generated-code verifiers on the host.")

    if profile_verifier is not None:
        verifier = profile_verifier
    elif verifier_type == "gcc":
        verifier = GCCVerifier(execution_policy=verifier_policy)
    elif verifier_type == "mingw":
        verifier = MinGWVerifier(execution_policy=verifier_policy)
    elif verifier_type == "msvc":
        # CLI args take precedence over config file
        msvc_host = getattr(args, "host", None) or cfg_dict.get("verifier", {}).get("host")
        msvc_user = getattr(args, "user", None) or cfg_dict.get("verifier", {}).get("user")
        msvc_key = getattr(args, "ssh_key", None) or cfg_dict.get("verifier", {}).get("ssh_key")

        if not msvc_host or not msvc_user or not msvc_key:
            print("Error: MSVC verifier requires --host, --user, and --ssh-key.")
            print("\nExample:")
            print("  halo-forge raft train --verifier msvc \\")
            print("    --host 10.0.0.152 --user keys --ssh-key ~/.ssh/win \\")
            print("    --prompts data/prompts.jsonl")
            print("\nOr in config file (configs/raft_windows_msvc.yaml):")
            print("  verifier:")
            print("    type: msvc")
            print("    host: 10.0.0.152")
            print("    user: keys")
            print("    ssh_key: ~/.ssh/win")
            print("\nOr use MinGW for local cross-compilation (no Windows needed):")
            print("  halo-forge raft train --verifier mingw ...")
            sys.exit(1)

        verifier = RemoteMSVCVerifier(host=msvc_host, user=msvc_user, ssh_key=msvc_key)
    elif verifier_type == "humaneval":
        dataset_path = cfg_dict.get("verifier", {}).get("dataset", "data/rlvr/humaneval_full.jsonl")
        verifier = HumanEvalVerifier(dataset_path, execution_policy=verifier_policy)
    elif verifier_type == "mbpp":
        dataset_path = cfg_dict.get("verifier", {}).get(
            "dataset", "data/rlvr/mbpp_train_full.jsonl"
        )
        verifier = MBPPVerifier(dataset_path, execution_policy=verifier_policy)
    elif verifier_type == "rust" or verifier_type == "cargo":
        from halo_forge.rlvr.verifiers import RustVerifier

        run_after = cfg_dict.get("verifier", {}).get("run_after_compile", False)
        verifier = RustVerifier(run_after_compile=run_after, execution_policy=verifier_policy)
    elif verifier_type == "go":
        from halo_forge.rlvr.verifiers import GoVerifier

        run_after = cfg_dict.get("verifier", {}).get("run_after_compile", False)
        verifier = GoVerifier(run_after_compile=run_after, execution_policy=verifier_policy)
    elif verifier_type == "auto":
        from halo_forge.rlvr.verifiers import MultiLanguageVerifier

        run_after = cfg_dict.get("verifier", {}).get("run_after_compile", False)
        binary_cache = cfg_dict.get("verifier", {}).get("binary_cache_dir")
        verifier = MultiLanguageVerifier(
            run_after_compile=run_after,
            binary_cache_dir=binary_cache,
            execution_policy=verifier_policy,
        )
    elif verifier_type == "execution":
        from halo_forge.rlvr.verifiers import ExecutionVerifier

        test_cases = cfg_dict.get("verifier", {}).get("test_cases", [])
        match_mode = cfg_dict.get("verifier", {}).get("match_mode", "exact")
        verifier = ExecutionVerifier(
            test_cases=test_cases,
            match_mode=match_mode,
            execution_policy=verifier_policy,
        )
    else:
        print(f"Unknown verifier: {verifier_type}")
        print(f"Available: {', '.join(RAFT_TRAIN_SUPPORTED_VERIFIERS)}")
        sys.exit(1)

    # Create config
    raft_cfg = cfg_dict.get("raft", {}) if isinstance(cfg_dict.get("raft", {}), dict) else {}
    generation_cfg = (
        cfg_dict.get("generation", {}) if isinstance(cfg_dict.get("generation", {}), dict) else {}
    )
    training_cfg = (
        cfg_dict.get("training", {}) if isinstance(cfg_dict.get("training", {}), dict) else {}
    )
    lora_cfg = cfg_dict.get("lora", {}) if isinstance(cfg_dict.get("lora", {}), dict) else {}

    def _prefer_nested(nested, top_level, default, name: str):
        if nested is not None and top_level is not None and nested != top_level:
            print(f"[!] Using raft.{name} ({nested}) over top-level {name} ({top_level})")
        if nested is not None:
            return nested
        if top_level is not None:
            return top_level
        return default

    keep_percent = getattr(args, "keep_percent", None)
    if keep_percent is None:
        keep_percent = _prefer_nested(
            raft_cfg.get("keep_top_percent"),
            cfg_dict.get("keep_top_percent"),
            0.5,
            "keep_top_percent",
        )

    reward_threshold = getattr(args, "reward_threshold", None)
    if reward_threshold is None:
        reward_threshold = _prefer_nested(
            raft_cfg.get("reward_threshold"),
            cfg_dict.get("reward_threshold"),
            0.5,
            "reward_threshold",
        )
    if profile_verifier is not None:
        profile_contract = dict(getattr(args, "_verifier_profile_contract", {}) or {})
        reward_threshold = (
            profile_contract.get("threshold")
            if profile_contract.get("threshold") is not None
            else profile_contract.get("minimum", reward_threshold)
        )

    curriculum = getattr(args, "curriculum", None) or cfg_dict.get("curriculum_strategy", "none")
    curriculum_stats = getattr(args, "curriculum_stats", None) or cfg_dict.get(
        "curriculum_stats_path", None
    )
    curriculum_start = getattr(args, "curriculum_start", None) or cfg_dict.get(
        "curriculum_progressive_start", 0.2
    )
    curriculum_increment = getattr(args, "curriculum_increment", None) or cfg_dict.get(
        "curriculum_progressive_increment", 0.2
    )
    reward_shaping = getattr(args, "reward_shaping", None) or cfg_dict.get(
        "reward_shaping_strategy", "fixed"
    )
    system_prompt = getattr(args, "system_prompt", None) or cfg_dict.get(
        "system_prompt", "You are an expert Windows systems programmer."
    )
    lr_decay = getattr(args, "lr_decay", None) or cfg_dict.get("lr_decay_per_cycle", 0.85)
    min_lr = getattr(args, "min_lr", None) or cfg_dict.get("min_lr", 1e-6)

    # New generation parameters
    samples_per_prompt = getattr(args, "samples_per_prompt", None)
    if samples_per_prompt is None:
        samples_per_prompt = raft_cfg.get("samples_per_prompt", 8)

    temperature = getattr(args, "temperature", None)
    if temperature is None:
        temperature = generation_cfg.get("temperature", 0.7)

    max_new_tokens = getattr(args, "max_new_tokens", None)
    if max_new_tokens is None:
        max_new_tokens = generation_cfg.get("max_new_tokens", 1024)

    min_samples = getattr(args, "min_samples", None)
    if min_samples is None:
        min_samples = raft_cfg.get("min_samples")

    # Training hyperparameters from training.* section
    learning_rate = getattr(args, "learning_rate", None)
    if learning_rate is None:
        learning_rate = (
            training_cfg.get("learning_rate") or training_cfg.get("base_learning_rate") or 5e-5
        )

    batch_size = getattr(args, "batch_size", None)
    if batch_size is None:
        batch_size = training_cfg.get("batch_size") or 2

    gradient_accumulation = getattr(args, "gradient_accumulation", None)
    if gradient_accumulation is None:
        gradient_accumulation = training_cfg.get("gradient_accumulation_steps") or 16

    warmup_steps = getattr(args, "warmup_steps", None)
    if warmup_steps is None:
        warmup_steps = training_cfg.get("warmup_steps") or 10

    # Support training.lr_decay_factor as alias for lr_decay_per_cycle
    if lr_decay == 0.85:  # default value, check if config has different
        config_lr_decay = training_cfg.get("lr_decay_factor")
        if config_lr_decay is not None:
            lr_decay = config_lr_decay

    # LoRA configuration from lora.* section
    lora_rank = getattr(args, "lora_rank", None)
    if lora_rank is None:
        lora_rank = lora_cfg.get("r") or 16

    lora_alpha = getattr(args, "lora_alpha", None)
    if lora_alpha is None:
        lora_alpha = lora_cfg.get("alpha") or 32

    lora_dropout = lora_cfg.get("dropout") or 0.05

    # Resolve model path - handles SFT output directories automatically
    # This allows: --model models/code_sft (where adapters are in models/code_sft/final_model)
    model_arg = args.model or cfg_dict.get("base_model", "Qwen/Qwen2.5-Coder-3B")
    checkpoint_arg = args.checkpoint or cfg_dict.get("sft_checkpoint")

    if checkpoint_arg:
        # Explicit checkpoint provided - use as-is
        base_model = model_arg
        sft_checkpoint = checkpoint_arg
    else:
        # Auto-detect from --model argument
        base_model, sft_checkpoint = _resolve_model_path(model_arg)
        if sft_checkpoint:
            print(f"  > Auto-detected SFT adapter: {sft_checkpoint}")
            print(f"  > Base model: {base_model}")
        else:
            # No adapter found - will train from scratch
            sft_checkpoint = cfg_dict.get("sft_checkpoint", "models/sft/final_model")

    num_cycles = args.cycles
    if num_cycles is None:
        num_cycles = _prefer_nested(
            raft_cfg.get("num_cycles"), cfg_dict.get("num_cycles"), 3, "num_cycles"
        )

    config = RAFTConfig(
        base_model=base_model,
        sft_checkpoint=sft_checkpoint,
        output_dir=args.output or cfg_dict.get("output_dir", "models/raft"),
        seed=getattr(args, "seed", cfg_dict.get("seed", 42)),
        num_cycles=num_cycles,
        keep_top_percent=keep_percent,
        reward_threshold=reward_threshold,
        allow_compile_only_training=getattr(args, "allow_compile_only_training", False),
        curriculum_strategy=curriculum,
        curriculum_stats_path=curriculum_stats,
        curriculum_progressive_start=curriculum_start,
        curriculum_progressive_increment=curriculum_increment,
        reward_shaping_strategy=reward_shaping,
        system_prompt=system_prompt,
        # Training hyperparameters
        learning_rate=learning_rate,
        train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        warmup_steps=warmup_steps,
        lr_decay_per_cycle=lr_decay,
        min_lr=min_lr,
        # LoRA configuration
        lora_r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        # Generation
        samples_per_prompt=samples_per_prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        min_samples_per_cycle=min_samples,
    )

    # Load prompts
    prompts = []
    prompts_file = args.prompts or cfg_dict.get("prompts")
    if prompts_file:
        prompts, invalid_lines = _load_prompts_jsonl(prompts_file)
        if invalid_lines:
            print(f"Error: Invalid JSONL in prompts file: {prompts_file}")
            for line_num, err in invalid_lines[:5]:
                print(f"  Line {line_num}: {err}")
            if len(invalid_lines) > 5:
                print(f"  ... {len(invalid_lines) - 5} more invalid lines")
            sys.exit(1)

    if not prompts:
        print("Error: No prompts provided")
        print("Use --prompts or set in config")
        sys.exit(1)
    max_prompts = getattr(args, "max_prompts", None)
    if max_prompts is not None:
        if int(max_prompts) <= 0:
            print("Error: --max-prompts must be greater than zero")
            sys.exit(1)
        # Dataset-version artifacts have deterministic ordering.  Taking the
        # ordered prefix makes proof-run selection stable without materializing
        # a second prompt file or changing the immutable trainer artifact.
        prompts = prompts[: int(max_prompts)]

    # Phase 5: when --accelerator mlx is requested we have two paths.
    # Default (5b): MLXRAFTTrainer — rollout + verify + SFT all on MLX.
    # Opt-in (5a): --rollout-only — keep the PyTorch RAFTTrainer but swap
    # in MLXRolloutGenerator for the rollout step. The 5a hybrid is useful
    # when the user has an existing PyTorch checkpoint they want to keep
    # training but wants fast Apple Silicon rollouts.
    signal_session = _build_training_signal_session(
        args,
        trainer="raft",
        output_dir=config.output_dir,
        total_boundaries=config.num_cycles,
        reward_threshold=config.reward_threshold,
    )
    completed_trainer = None
    if getattr(args, "accelerator", "auto") == "mlx":
        mlx_model = getattr(args, "rollout_model", None) or args.model
        if getattr(args, "rollout_only", False):
            from halo_forge.rlvr.mlx_rollout import MLXRolloutGenerator

            print(
                f"[mlx-5a] Hybrid mode: MLX rollouts ({mlx_model}) + PyTorch policy update ({args.model})"
            )
            trainer = RAFTTrainer(
                verifier=verifier,
                config=config,
                rollout_generator=MLXRolloutGenerator(mlx_model),
                signal_sink=signal_session,
            )
            trainer.run(prompts, num_cycles=config.num_cycles)
            completed_trainer = trainer
        else:
            from halo_forge.rlvr.mlx_raft_trainer import MLXRAFTTrainer

            print(f"[mlx-5b] Native MLX RAFT: rollout + verify + SFT on MLX ({mlx_model})")
            mlx_trainer = MLXRAFTTrainer(
                verifier=verifier,
                config=config,
                rollout_model=mlx_model,
                signal_sink=signal_session,
            )
            mlx_trainer.run(prompts, num_cycles=config.num_cycles)
            completed_trainer = mlx_trainer
    else:
        # Track I6 — pluggable rollout engine. The torch fallback is the
        # historical default; vllm is the CUDA/ROCm fast path; mlx is the
        # Apple Silicon equivalent (vLLM doesn't run on MLX, but
        # mlx_lm.generate is the same throughput story on Apple's
        # unified-memory hardware).
        rollout_engine = getattr(args, "rollout_engine", "auto")
        if rollout_engine == "vllm":
            from halo_forge.rlvr.vllm_rollout import VLLMRolloutGenerator

            print(f"[i6] Using vLLM rollouts for fast continuous-batched generation")
            trainer = RAFTTrainer(
                verifier=verifier,
                config=config,
                rollout_generator=VLLMRolloutGenerator(args.model),
                signal_sink=signal_session,
            )
        elif rollout_engine == "mlx":
            from halo_forge.rlvr.mlx_rollout import MLXRolloutGenerator

            mlx_model = getattr(args, "rollout_model", None) or args.model
            print(f"[i6] Using MLX rollouts ({mlx_model}) — Apple Silicon fast path")
            trainer = RAFTTrainer(
                verifier=verifier,
                config=config,
                rollout_generator=MLXRolloutGenerator(mlx_model),
                signal_sink=signal_session,
            )
        else:
            trainer = RAFTTrainer(
                verifier=verifier, config=config, signal_sink=signal_session
            )
        trainer.run(prompts, num_cycles=config.num_cycles)
        completed_trainer = trainer

    _finalize_managed_training_replay(
        args,
        "raft",
        config.output_dir,
        config,
        getattr(completed_trainer, "training_summary", {}),
    )


def cmd_benchmark(args):
    """Run benchmark."""
    # Note: --experimental-attention is handled at script startup (before imports)

    print_banner()
    print(f"{GREEN}Benchmark{NC}")
    print("=" * 60)

    from halo_forge.benchmark.pass_at_k import Benchmark
    from halo_forge.rlvr.verifiers import (
        GCCVerifier,
        MinGWVerifier,
        RemoteMSVCVerifier,
        RustVerifier,
        GoVerifier,
        DotNetVerifier,
        PowerShellVerifier,
        MultiLanguageVerifier,
        AutoVerifier,
    )

    # Setup verifier
    verifier_policy = (
        "unsafe_host" if getattr(args, "unsafe_verifier_execution", False) else "sandbox"
    )
    if verifier_policy == "unsafe_host":
        print("WARNING: --unsafe-verifier-execution runs generated-code verifiers on the host.")
    if args.verifier == "gcc":
        verifier = GCCVerifier(execution_policy=verifier_policy)
    elif args.verifier == "mingw":
        verifier = MinGWVerifier(execution_policy=verifier_policy)
    elif args.verifier == "rust":
        verifier = RustVerifier(
            cross_compile=getattr(args, "cross_compile", False), execution_policy=verifier_policy
        )
    elif args.verifier == "go":
        verifier = GoVerifier(
            cross_compile=getattr(args, "cross_compile", False), execution_policy=verifier_policy
        )
    elif args.verifier == "dotnet":
        verifier = DotNetVerifier(execution_policy=verifier_policy)
    elif args.verifier == "powershell":
        verifier = PowerShellVerifier()
    elif args.verifier in ("auto", "multi"):
        # Auto-detect language from code
        verifier = MultiLanguageVerifier(
            run_after_compile=getattr(args, "run_after_compile", False),
            execution_policy=verifier_policy,
        )
    elif args.verifier in ("humaneval", "python"):
        from halo_forge.rlvr.verifiers import HumanEvalVerifier

        dataset_path = getattr(args, "dataset", None) or "data/rlvr/humaneval_full.jsonl"
        verifier = HumanEvalVerifier(dataset_path, execution_policy=verifier_policy)
    elif args.verifier == "mbpp":
        from halo_forge.rlvr.verifiers import MBPPVerifier

        dataset_path = getattr(args, "dataset", None) or "data/rlvr/mbpp_train_full.jsonl"
        verifier = MBPPVerifier(dataset_path, execution_policy=verifier_policy)
    elif args.verifier == "msvc":
        # Validate required MSVC parameters
        missing = []
        if not args.host:
            missing.append("--host")
        if not args.user:
            missing.append("--user")
        if not args.ssh_key:
            missing.append("--ssh-key")

        if missing:
            print(f"Error: MSVC verifier requires: {', '.join(missing)}")
            print("\nExample:")
            print("  halo-forge benchmark run --verifier msvc \\")
            print("    --host 10.0.0.152 --user keys --ssh-key ~/.ssh/win \\")
            print("    --model Qwen/Qwen2.5-Coder-0.5B \\")
            print("    --prompts data/prompts.jsonl")
            print("\nOr use MinGW for local cross-compilation (no Windows needed):")
            print("  halo-forge benchmark run --verifier mingw ...")
            sys.exit(1)

        verifier = RemoteMSVCVerifier(host=args.host, user=args.user, ssh_key=args.ssh_key)
    else:
        print(f"Unknown verifier: {args.verifier}")
        print(
            "Available verifiers: gcc, mingw, msvc, rust, go, dotnet, powershell, auto, humaneval, mbpp, python"
        )
        sys.exit(1)

    # Resolve model path - handles SFT/RAFT output directories automatically
    model_arg = args.model
    base_model_arg = args.base_model

    if not base_model_arg:
        # Auto-detect from model path
        detected_base, detected_checkpoint = _resolve_model_path(model_arg)
        if detected_checkpoint:
            print(f"  > Auto-detected adapter: {detected_checkpoint}")
            print(f"  > Base model: {detected_base}")
            model_path = detected_checkpoint
            base_model_arg = detected_base
        else:
            model_path = model_arg
    else:
        model_path = model_arg

    # Create benchmark
    benchmark = Benchmark(
        model_path=model_path,
        verifier=verifier,
        base_model=base_model_arg,
        system_prompt=args.system_prompt,
    )

    # Parse k values
    k_values = [int(k) for k in args.k.split(",")]

    # Run
    result = benchmark.run(
        prompts=args.prompts,
        samples_per_prompt=args.samples,
        k_values=k_values,
        max_prompts=args.max_prompts,
        output_path=args.output,
    )


def cmd_benchmark_full(args):
    """Run comprehensive RAFT benchmark with hardware monitoring."""
    try:
        from halo_forge import ui

        use_rich = True
    except ImportError:
        use_rich = False

    from halo_forge.benchmark import BenchmarkRunner, run_benchmark_suite, DEFAULT_MODELS

    if use_rich:
        ui.print_banner()
        ui.print_header("RAFT Benchmark", f"Comprehensive training benchmark with metrics")

    # Handle --suite option
    if args.suite:
        if args.suite == "all":
            models = DEFAULT_MODELS
        elif args.suite == "small":
            models = [DEFAULT_MODELS[0]]  # Just 0.5B
        elif args.suite == "medium":
            models = DEFAULT_MODELS[:2]  # 0.5B and 1.5B
        else:
            print(f"Unknown suite: {args.suite}")
            print("Valid suites: all, small, medium")
            sys.exit(1)

        results = run_benchmark_suite(
            models=models,
            output_dir=args.output,
            n_cycles=args.cycles,
            verbose=not args.quiet,
        )

        # Print comparison
        if use_rich:
            ui.print_header("Results Summary")
        print(f"\nBenchmark complete. Results saved to: {args.output}")

        for r in results:
            improvement = (
                (r.final.compile_rate - r.baseline.compile_rate) if r.final and r.baseline else 0
            )
            print(
                f"  {r.model_short}: {r.baseline.compile_rate:.1%} -> {r.final.compile_rate:.1%} (+{improvement:.1%})"
            )

    else:
        # Single model benchmark
        runner = BenchmarkRunner(
            model_name=args.model,
            output_dir=args.output,
            n_cycles=args.cycles,
            verbose=not args.quiet,
        )

        result = runner.run()
        print(f"\nBenchmark complete. Results saved to: {args.output}/summary.json")


def cmd_benchmark_eval(args):
    """Run code evaluation on standard benchmarks (HumanEval, MBPP, LiveCodeBench)."""
    from pathlib import Path
    from halo_forge.benchmark import run_benchmark

    print_banner()
    print(f"{GREEN}Code Benchmark: {args.benchmark}{NC}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Benchmark: {args.benchmark}")
    if args.limit:
        print(f"Limit: {args.limit}")

    run_after_compile = getattr(args, "run_after_compile", False)
    is_compiled_language = args.benchmark in {"cpp", "rust", "go"}
    if is_compiled_language:
        mode = "MVR (full verification)" if run_after_compile else "MVP (compile-only)"
        print(f"Mode: {mode}")
        if getattr(args, "language", None):
            print(f"Language: {args.language}")
        if getattr(args, "verifier", None):
            print(f"Verifier: {args.verifier}")
    else:
        print("Mode: dataset-faithful Python benchmark evaluation")
    print("=" * 60)

    output = Path(args.output) if args.output else None

    result = run_benchmark(
        model=args.model,
        benchmark=args.benchmark,
        limit=args.limit,
        output=output,
        samples_per_prompt=getattr(args, "samples_per_prompt", 5),
        run_after_compile=run_after_compile,
        language=getattr(args, "language", None),
        verifier=getattr(args, "verifier", None),
    )

    if "error" in result:
        print(f"\n{RED}Error: {result['error']}{NC}")
        sys.exit(1)

    print(f"\n{GREEN}Results:{NC}")
    execution_path = result.get("execution_path")
    if execution_path:
        print(f"  execution_path: {execution_path}")
    for key, value in result.get("metrics", {}).items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print(f"\nSamples evaluated: {result.get('samples', 0)}")

    if output:
        print(f"Results saved to: {output}")


def cmd_plot_training(args):
    """Generate charts from TensorBoard training logs."""
    from pathlib import Path

    # Import the plotting module
    try:
        from scripts.plot_training import (
            load_training_metrics,
            load_multiple_runs,
            generate_all_charts,
            plot_loss_curve,
            plot_learning_rate,
            plot_grad_norm,
            plot_training_summary,
            plot_comparison,
        )
    except ImportError:
        # Fallback: run as subprocess
        import subprocess

        cmd = [sys.executable, "scripts/plot_training.py"] + args.log_dirs
        if args.output:
            cmd.extend(["--output", args.output])
        if args.compare:
            cmd.append("--compare")
        if args.only:
            cmd.extend(["--only", args.only])
        if args.name:
            cmd.extend(["--name", args.name])
        subprocess.run(cmd)
        return

    log_dirs = [Path(d) for d in args.log_dirs]

    # Comparison mode
    if args.compare and len(log_dirs) > 1:
        runs = load_multiple_runs(log_dirs)
        if not runs:
            print("Error: No valid training runs found")
            return

        output_dir = Path(args.output) if args.output else Path("figures/comparison")
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating comparison charts in {output_dir}...")
        plot_comparison(runs, output_dir / "loss_comparison.png", "train_loss")
        plot_comparison(runs, output_dir / "lr_comparison.png", "learning_rate")
        print(f"\nDone! Comparison charts saved to {output_dir}")
        return

    # Single run mode
    log_dir = log_dirs[0]

    try:
        metrics = load_training_metrics(log_dir, name=args.name)
    except Exception as e:
        print(f"Error loading training logs: {e}")
        return

    print(
        f"Loaded {metrics.name}: {metrics.total_steps} steps, final loss {metrics.final_loss:.4f}"
    )

    output_dir = Path(args.output) if args.output else log_dir.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating charts in {output_dir}...")

    if args.only == "loss":
        plot_loss_curve(metrics, output_dir / "loss_curve.png")
    elif args.only == "lr":
        plot_learning_rate(metrics, output_dir / "learning_rate.png")
    elif args.only == "grad":
        plot_grad_norm(metrics, output_dir / "grad_norm.png")
    elif args.only == "summary":
        plot_training_summary(metrics, output_dir / "training_summary.png")
    else:
        generate_all_charts(metrics, output_dir)

    print(f"\nDone! Charts saved to {output_dir}")


def cmd_plot_benchmarks(args):
    """Generate charts from benchmark results."""
    import subprocess

    cmd = [sys.executable, "scripts/plot_benchmarks.py", args.results_dir]
    if args.output:
        cmd.extend(["--output", args.output])
    subprocess.run(cmd)


def cmd_info(args):
    """Show hardware info."""

    def _backend_info() -> dict[str, Any]:
        info: dict[str, Any] = {
            "name": "unknown",
            "device": "unknown",
            "supports_neural_accelerators": False,
            "chip": None,
        }
        try:
            from halo_forge.backend import get_backend

            backend = get_backend()
            info["name"] = backend.name
            info["device"] = backend.device()
            info["supports_neural_accelerators"] = bool(
                getattr(backend.capabilities, "supports_neural_accelerators", False)
            )
        except Exception:
            pass

        try:
            from halo_forge.telemetry.apple_silicon import AppleSiliconTelemetry

            chip = AppleSiliconTelemetry._detect_chip_info(
                AppleSiliconTelemetry._detect_device_name()
            )
            if chip is not None:
                info["chip"] = chip.to_dict()
        except Exception:
            pass
        if not info.get("chip") or str(info.get("chip", {}).get("brand", "")).lower() in {
            "arm",
            "apple silicon",
        }:
            try:
                from halo_forge.backend.mlx_readiness import _metal_device, _optional_int
                from halo_forge.utils.apple_chip import parse_chip_brand, with_gpu_cores

                metal = _metal_device()
                if metal:
                    parsed = with_gpu_cores(
                        parse_chip_brand(str(metal.get("model") or "")),
                        _optional_int(metal.get("gpu_cores")),
                    )
                    if parsed is not None:
                        info["chip"] = parsed.to_dict()
            except Exception:
                pass
        return info

    def _hardware_lines(backend_info: dict[str, Any]) -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = [
            ("info", f"Backend: {backend_info['name']} ({backend_info['device']})")
        ]
        chip = backend_info.get("chip")
        if isinstance(chip, dict):
            gpu = f", gpu_cores={chip['gpu_cores']}" if chip.get("gpu_cores") is not None else ""
            lines.append(
                (
                    "info",
                    f"Chip: {chip['brand']} (gen={chip['generation']}, "
                    f"variant={chip.get('variant') or 'base'}{gpu})",
                )
            )
        if backend_info["name"] in {"mps", "mlx"}:
            lines.append(("success", "Apple Silicon accelerator detected"))
        elif backend_info["name"] in {"cuda", "rocm", "rocm_gfx1151"}:
            lines.append(("success", f"{backend_info['name'].upper()} accelerator detected"))
        elif backend_info["name"] == "cpu":
            lines.append(("warning", "No hardware accelerator backend is active"))
        if backend_info["name"] in {"mps", "mlx"}:
            lines.append(("info", "PyTorch CUDA/ROCm not active; using Apple backend"))
        lines.append(
            (
                "info",
                "Neural Accelerators: "
                + (
                    "available"
                    if backend_info.get("supports_neural_accelerators")
                    else "unavailable"
                ),
            )
        )
        return lines

    backend_info = _backend_info()

    try:
        from halo_forge import ui

        ui.print_banner()
        try:
            import torch

            if torch.cuda.is_available() and backend_info["name"] in {
                "cuda",
                "rocm",
                "rocm_gfx1151",
            }:
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                memory_gb = props.total_memory / 1e9

                rocm_version = ""
                if hasattr(torch.version, "hip"):
                    rocm_version = torch.version.hip or ""

                ui.print_hardware_info(
                    gpu_name=gpu_name,
                    memory_gb=memory_gb,
                    rocm_version=rocm_version,
                    pytorch_version=torch.__version__,
                )
        except Exception:
            pass

        for level, line in _hardware_lines(backend_info):
            if level == "success":
                ui.print_success(line)
            elif level == "warning":
                ui.print_warning(line)
            else:
                ui.print_info(line)
    except ImportError:
        print_banner()
        for level, line in _hardware_lines(backend_info):
            prefix = {"success": "[OK]", "warning": "[!]", "info": ">"}[level]
            print(f"{prefix} {line}")


def cmd_doctor(args):
    """Run environment readiness checks."""
    if args.doctor_command != "mlx":
        print(f"Unknown doctor check: {args.doctor_command}", file=sys.stderr)
        sys.exit(1)

    from halo_forge.backend.mlx_readiness import check_mlx_readiness

    readiness = check_mlx_readiness()
    payload = readiness.to_dict()
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        versions = payload.get("package_versions") or {}
        chip = payload.get("chip") or {}
        metal = payload.get("metal_device") or {}
        print(f"MLX readiness: {payload['status']}")
        print(f"  executable: {payload['executable']}")
        print(
            f"  packages: mlx={versions.get('mlx') or 'missing'}, mlx-lm={versions.get('mlx-lm') or 'missing'}"
        )
        if payload.get("macos_version"):
            print(f"  macOS: {payload['macos_version']}")
        if chip:
            print(f"  chip: {chip.get('brand') or chip.get('raw_brand') or chip}")
        if metal:
            print(f"  Metal device: {metal.get('model') or metal}")
        for label in ("warnings", "errors", "suggested_fixes"):
            values = [str(item) for item in payload.get(label, []) if item]
            if values:
                print(f"  {label.replace('_', ' ')}:")
                for value in values:
                    print(f"    - {value}")
        if readiness.executable:
            print()
            print("Next:")
            print(
                "  halo-forge --accelerator mlx sft train "
                "--model mlx-community/Qwen2.5-0.5B-Instruct-bf16 "
                "--dataset codealpaca --output models/sft_mlx_quickstart"
            )

    if readiness.executable:
        return
    sys.exit(2 if readiness.status == "unavailable" else 1)


def cmd_models(args):
    """List and inspect curated upstream/base models."""
    from halo_forge.models.catalog import CATALOG_VERSION, get_model, list_models

    if args.models_command == "show":
        item = get_model(args.model_id)
        if item is None:
            print(f"Unknown model: {args.model_id}", file=sys.stderr)
            sys.exit(1)
        if getattr(args, "json", False):
            print(json.dumps(item, indent=2, sort_keys=True))
            return
        print(f"{item['id']}")
        print(f"  label: {item['label']}")
        print(f"  provider: {item['provider']}")
        print(f"  family: {item['family']} ({item['parameter_count']})")
        print(f"  status: {item['status']}")
        print(f"  memory: {item['memory_tier']}")
        print(f"  modalities: {', '.join(item['modalities'])}")
        print(f"  trainers: {', '.join(item['trainer_support'])}")
        print(f"  backends: {', '.join(item['backend_support'])}")
        if item.get("mlx_variant"):
            print(f"  mlx_variant: {item['mlx_variant']}")
        print(f"  use: {item['recommended_use']}")
        caveats = item.get("known_caveats") or []
        if caveats:
            print("  caveats:")
            for caveat in caveats:
                print(f"    - {caveat}")
        return

    filters = {
        "mode": getattr(args, "mode", None),
        "backend": getattr(args, "backend", None),
        "modality": getattr(args, "modality", None),
        "provider": getattr(args, "provider", None),
        "status": getattr(args, "status", None),
        "memory_tier": getattr(args, "memory_tier", None),
    }
    items = list_models({k: v for k, v in filters.items() if v})
    if getattr(args, "json", False):
        print(
            json.dumps(
                {"catalog_version": CATALOG_VERSION, "items": items}, indent=2, sort_keys=True
            )
        )
        return
    print(f"halo-forge model catalog {CATALOG_VERSION} ({len(items)} models)")
    print()
    print(f"{'MODEL':<42} {'STATUS':<12} {'MEM':<7} {'TRAINERS':<22} USE")
    print("-" * 110)
    for item in items:
        trainers = ",".join(item["trainer_support"][:4])
        if len(item["trainer_support"]) > 4:
            trainers += ",..."
        print(
            f"{item['id']:<42.42} {item['status']:<12} {item['memory_tier']:<7} "
            f"{trainers:<22.22} {item['recommended_use']}"
        )


# =============================================================================
# Test Command
# =============================================================================

# Built-in test prompts for pipeline validation
TEST_PROMPTS = [
    {
        "prompt": "Write a C++ program that prints 'Hello, World!' to stdout.",
        "expected_output": "Hello, World!",
    },
    {
        "prompt": "Write a C++ function that returns the sum of two integers a and b, then call it in main to print the result of 5 + 3.",
        "expected_output": "8",
    },
    {
        "prompt": "Write a C++ program that prints the numbers 1 through 5, each on a new line.",
        "expected_output": "1\n2\n3\n4\n5",
    },
]


class TestRunner:
    """Pipeline test runner with multiple test levels."""

    def __init__(self, verbose: bool = False, model: str = "Qwen/Qwen2.5-Coder-0.5B"):
        self.verbose = verbose
        self.model_name = model
        self.results = {"passed": [], "failed": [], "skipped": []}

        # Try to use rich UI
        try:
            from halo_forge import ui

            self.ui = ui
            self.use_rich = True
        except ImportError:
            self.ui = None
            self.use_rich = False

    def log(self, msg: str, level: str = "info"):
        """Log message if verbose or if it's an error."""
        if self.verbose or level in ("error", "result"):
            if self.use_rich:
                if level == "ok":
                    self.ui.print_step(msg, "success")
                elif level == "fail":
                    self.ui.print_step(msg, "error")
                elif level == "skip":
                    self.ui.print_step(msg, "skip")
                elif level == "error":
                    self.ui.print_error(msg)
                else:
                    self.ui.print_dim(f"  {msg}")
            else:
                prefix = {
                    "info": "  ",
                    "ok": "  [OK] ",
                    "fail": "  [FAIL] ",
                    "skip": "  [SKIP] ",
                    "error": "  [ERROR] ",
                    "result": "",
                }
                print(f"{prefix.get(level, '  ')}{msg}")

    def run_test(self, name: str, test_fn, skip_condition: bool = False, skip_reason: str = ""):
        """Run a single test with timing."""
        if skip_condition:
            self.results["skipped"].append(name)
            if self.use_rich:
                self.ui.print_step(name, "skip", skip_reason)
            else:
                self.log(f"{name}: {skip_reason}", "skip")
            return None

        start = time.time()
        try:
            result = test_fn()
            elapsed = time.time() - start
            self.results["passed"].append(name)
            if self.use_rich:
                self.ui.print_step(name, "success", time_s=elapsed)
            else:
                self.log(f"{name} ({elapsed:.1f}s)", "ok")
            return result
        except Exception as e:
            elapsed = time.time() - start
            self.results["failed"].append(name)
            if self.use_rich:
                self.ui.print_step(name, "error", str(e), time_s=elapsed)
            else:
                self.log(f"{name} ({elapsed:.1f}s): {e}", "fail")
            if self.verbose:
                import traceback

                traceback.print_exc()
            return None

    def print_summary(self):
        """Print test summary."""
        if self.use_rich:
            self.ui.print_test_results(self.results)
            return len(self.results["failed"]) == 0

        # Fallback plain output
        total = (
            len(self.results["passed"]) + len(self.results["failed"]) + len(self.results["skipped"])
        )
        passed = len(self.results["passed"])
        failed = len(self.results["failed"])
        skipped = len(self.results["skipped"])

        print(f"\n{'='*60}")
        print(f"Test Results: {passed}/{total} passed", end="")
        if skipped:
            print(f", {skipped} skipped", end="")
        if failed:
            print(f", {failed} FAILED", end="")
        print()

        if failed:
            print(f"\nFailed tests:")
            for name in self.results["failed"]:
                print(f"  - {name}")

        print(f"{'='*60}")

        return failed == 0

    # =========================================================================
    # Smoke Tests (no GPU required)
    # =========================================================================

    def test_imports(self) -> bool:
        """Test that all modules import correctly."""
        # Core modules
        from halo_forge.rlvr.verifiers import GCCVerifier, VerifyResult, RewardLevel
        from halo_forge.rlvr.raft_trainer import RAFTTrainer
        from halo_forge.sft.trainer import SFTTrainer
        from halo_forge.utils.hardware import print_hardware_info

        return True

    def test_compiler_available(self) -> bool:
        """Test that g++ is available."""
        if not shutil.which("g++"):
            raise RuntimeError("g++ not found in PATH")
        return True

    def test_verifier_basic(self) -> bool:
        """Test verifier with known good/bad code."""
        from halo_forge.rlvr.verifiers import GCCVerifier

        verifier = GCCVerifier()

        # Test valid code
        valid = '#include <iostream>\nint main() { std::cout << "test"; return 0; }'
        result = verifier.verify(valid)
        if result.reward == 0.0:
            raise RuntimeError(f"Valid code got reward 0: {result.details}")

        # Test invalid code
        invalid = "this is not valid C++ code at all"
        result = verifier.verify(invalid)
        if result.reward > 0.0:
            raise RuntimeError("Invalid code got positive reward")

        return True

    # =========================================================================
    # Standard Tests (GPU required)
    # =========================================================================

    def test_gpu_available(self) -> bool:
        """Test accelerator availability (CUDA/ROCm or Apple Silicon MPS).

        On CPU-only hosts, logs a warning and returns True so smoke tests can
        still proceed against tiny models. The trainer paths themselves still
        prefer an accelerator and tune accordingly.
        """
        import torch
        from halo_forge.utils.accelerator import detect_gpu_kind, GPU_KIND_CPU, GPU_KIND_MPS

        kind = detect_gpu_kind()
        if kind == GPU_KIND_CPU:
            self.log("WARNING: no accelerator detected; running on CPU. Training will be slow.")
            return True

        if kind == GPU_KIND_MPS:
            self.log("Accelerator: Apple Silicon (MPS). Memory probe unavailable on this backend.")
            return True

        device_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem_gb = props.total_memory / 1e9

        self.log(f"GPU: {device_name}, Memory: {mem_gb:.1f} GB")
        return True

    def test_model_load(self) -> Any:
        """Test model loading."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.log(f"Loading {self.model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=recommended_dtype(),
            device_map=get_device_map(),
            trust_remote_code=True,
        )

        self.log(f"Loaded: {model.num_parameters() / 1e6:.1f}M parameters")

        return model, tokenizer

    def test_generation(self, model, tokenizer) -> List[Dict]:
        """Test code generation."""
        import torch

        results = []

        for i, item in enumerate(TEST_PROMPTS):
            prompt = item["prompt"]

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful coding assistant. Write clean, working C++ code.",
                },
                {"role": "user", "content": prompt},
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            self.log(f"Prompt {i+1}: {prompt[:40]}...")
            self.log(f"Generated: {generated[:60]}...")

            results.append(
                {
                    "prompt": prompt,
                    "generated": generated,
                    "expected_output": item.get("expected_output"),
                }
            )

        return results

    def test_verification(self, samples: List[Dict]) -> List[Dict]:
        """Test verification of generated samples."""
        from halo_forge.rlvr.verifiers import GCCVerifier

        # Create verifier with run_after_compile to test execution
        verifier = GCCVerifier(run_after_compile=True, timeout=30, run_timeout=5)

        verified = []
        for i, sample in enumerate(samples):
            result = verifier.verify(sample["generated"])

            status = "PASS" if result.success else "FAIL"
            self.log(f"Sample {i+1}: {status} (reward={result.reward:.2f})")

            verified.append(
                {
                    **sample,
                    "success": result.success,
                    "reward": result.reward,
                    "details": result.details,
                }
            )

        passed = sum(1 for v in verified if v["success"])
        avg_reward = sum(v["reward"] for v in verified) / len(verified) if verified else 0

        self.log(f"Verification: {passed}/{len(verified)} passed, avg_reward={avg_reward:.2f}")

        return verified

    # =========================================================================
    # Full Tests (includes training)
    # =========================================================================

    def test_training_step(self, model, tokenizer, verified_samples: List[Dict]) -> bool:
        """Test a minimal SFT training step."""
        from transformers import TrainingArguments
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset

        # Prepare data - keep samples with any reward
        kept = [s for s in verified_samples if s["reward"] > 0]
        if not kept:
            self.log("No samples passed verification, using all for test")
            kept = verified_samples

        # Format for SFT
        training_data = []
        for sample in kept:
            training_data.append(
                {
                    "messages": [
                        {"role": "system", "content": "You are a helpful coding assistant."},
                        {"role": "user", "content": sample["prompt"]},
                        {"role": "assistant", "content": sample["generated"]},
                    ]
                }
            )

        dataset = Dataset.from_list(training_data)

        self.log(f"Training on {len(dataset)} samples...")

        # Minimal training config
        with tempfile.TemporaryDirectory(prefix="halo_forge_test_") as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=2e-5,
                logging_steps=1,
                save_steps=9999,
                max_steps=2,  # Just 2 steps
                bf16=True,
                dataloader_num_workers=0,
                dataloader_pin_memory=False,
                report_to="none",
            )

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                processing_class=tokenizer,
            )

            result = trainer.train()

            self.log(f"Training: {result.global_step} steps, loss={result.training_loss:.4f}")

        return True

    # =========================================================================
    # Test Level Runners
    # =========================================================================

    def run_smoke(self) -> bool:
        """Run smoke tests (no GPU required)."""
        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header("Smoke Test", "Quick validation without GPU")
        else:
            print(f"\n{'='*60}")
            print("halo forge Smoke Test")
            print(f"{'='*60}\n")

        self.run_test("Import modules", self.test_imports)
        self.run_test("Compiler available", self.test_compiler_available)
        self.run_test("Verifier basic", self.test_verifier_basic)

        return self.print_summary()

    def run_standard(self) -> bool:
        """Run standard tests (GPU required)."""
        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header("Standard Test", f"Model: {self.model_name}")
        else:
            print(f"\n{'='*60}")
            print("halo forge Standard Test")
            print(f"Model: {self.model_name}")
            print(f"{'='*60}\n")

        # Smoke tests first
        self.run_test("Import modules", self.test_imports)
        self.run_test("Compiler available", self.test_compiler_available)

        # GPU tests
        gpu_ok = self.run_test("GPU available", self.test_gpu_available)
        if gpu_ok is None:
            if self.use_rich:
                self.ui.print_error("Cannot continue without GPU")
            else:
                print("\nCannot continue without GPU")
            return self.print_summary()

        # Model loading
        result = self.run_test("Model loading", self.test_model_load)
        if result is None:
            return self.print_summary()
        model, tokenizer = result

        # Generation
        samples = self.run_test("Code generation", lambda: self.test_generation(model, tokenizer))
        if samples is None:
            return self.print_summary()

        # Verification
        self.run_test("Code verification", lambda: self.test_verification(samples))

        return self.print_summary()

    def run_full(self) -> bool:
        """Run full tests including training."""
        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header("Full Pipeline Test", f"Model: {self.model_name}")
        else:
            print(f"\n{'='*60}")
            print("halo forge Full Pipeline Test")
            print(f"Model: {self.model_name}")
            print(f"{'='*60}\n")

        # Smoke tests
        self.run_test("Import modules", self.test_imports)
        self.run_test("Compiler available", self.test_compiler_available)

        # GPU tests
        gpu_ok = self.run_test("GPU available", self.test_gpu_available)
        if gpu_ok is None:
            if self.use_rich:
                self.ui.print_error("Cannot continue without GPU")
            else:
                print("\nCannot continue without GPU")
            return self.print_summary()

        # Model loading
        result = self.run_test("Model loading", self.test_model_load)
        if result is None:
            return self.print_summary()
        model, tokenizer = result

        # Generation
        samples = self.run_test("Code generation", lambda: self.test_generation(model, tokenizer))
        if samples is None:
            return self.print_summary()

        # Verification
        verified = self.run_test("Code verification", lambda: self.test_verification(samples))
        if verified is None:
            verified = samples  # Use unverified for training test

        # Training step
        self.run_test("Training step", lambda: self.test_training_step(model, tokenizer, verified))

        return self.print_summary()

    def test_modality_fixtures(self) -> bool:
        """Validate deterministic modality fixture pack shape."""
        fixture_dir = Path("tests/fixtures/modality")
        required_files = (
            "vlm_samples.jsonl",
            "audio_samples.jsonl",
            "reasoning_samples.jsonl",
            "agentic_samples.jsonl",
        )
        if not fixture_dir.exists():
            raise RuntimeError(f"Missing fixture directory: {fixture_dir}")

        for filename in required_files:
            path = fixture_dir / filename
            if not path.exists():
                raise RuntimeError(f"Missing fixture file: {path}")
            with open(path, encoding="utf-8") as f:
                first_line = f.readline().strip()
            if not first_line:
                raise RuntimeError(f"Fixture file is empty: {path}")
            try:
                json.loads(first_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL fixture in {path}: {exc}") from exc
        return True

    def test_modality_train_smoke(self) -> Dict[str, Dict[str, Any]]:
        """Run tiny deterministic modality train-smoke flows."""
        from types import SimpleNamespace

        from halo_forge.modality_baseline import build_modality_entries_from_runs
        from halo_forge.modality_artifacts import persist_cycle_artifacts
        from halo_forge.training_contracts import build_cycle_summary
        from halo_forge.vlm.trainer import VLMRAFTConfig, VLMRAFTTrainer, VLMSampleResult
        from halo_forge.audio.trainer import AudioRAFTConfig, AudioRAFTTrainer, AudioRAFTCycleResult
        from halo_forge.reasoning.trainer import ReasoningRAFTConfig, ReasoningRAFTTrainer
        from halo_forge.reasoning.data import MathSample
        from halo_forge.agentic.trainer import (
            AgenticRAFTConfig,
            AgenticRAFTTrainer,
            AgenticRAFTCycleResult,
        )

        class _FakeSaveComponent:
            def __init__(self, marker: str):
                self.marker = marker

            def save_pretrained(self, target_dir: str):
                target = Path(target_dir) / f"{self.marker}.txt"
                target.write_text(self.marker, encoding="utf-8")

        with tempfile.TemporaryDirectory(prefix="halo_forge_modality_test_") as tmp_dir:
            output_root = Path(tmp_dir)
            run_payloads: Dict[str, Dict[str, Any]] = {}

            # VLM smoke
            vlm = VLMRAFTTrainer(
                VLMRAFTConfig(num_cycles=1, output_dir=str(output_root / "vlm"), seed=7)
            )
            vlm._setup = lambda: (
                setattr(
                    vlm,
                    "adapter",
                    SimpleNamespace(
                        model=_FakeSaveComponent("vlm_model"),
                        tokenizer=_FakeSaveComponent("vlm_tokenizer"),
                        processor=_FakeSaveComponent("vlm_processor"),
                        cleanup=lambda: None,
                    ),
                ),
                setattr(vlm, "verifier", SimpleNamespace(cleanup=lambda: None)),
            )
            vlm.generate_samples = lambda prompts, spp: [
                VLMSampleResult(
                    image="fixture.png",
                    prompt="describe",
                    completion="answer",
                    ground_truth="answer",
                    reward=1.0,
                    success=True,
                    details={},
                )
            ]
            vlm.filter_samples = lambda samples: samples
            vlm.train_on_samples = lambda samples, cycle: {
                "train_steps_executed": 1,
                "train_loss": 0.1,
                "weights_updated": True,
                "update_reason": "updated",
                "optimizer_steps": 1,
                "skipped_batches_non_finite": 0,
            }
            vlm_summary = vlm.train(
                prompts=[
                    SimpleNamespace(image="fixture.png", prompt="describe", ground_truth="answer")
                ]
            )
            if not vlm_summary.get("final_model_path"):
                raise RuntimeError("VLM smoke did not emit final_model_path")
            run_payloads["vlm"] = {"summary": vlm_summary, "output_dir": output_root / "vlm"}

            # Audio smoke
            audio = AudioRAFTTrainer(
                AudioRAFTConfig(num_cycles=1, output_dir=str(output_root / "audio"), seed=7)
            )
            audio.adapter = SimpleNamespace(
                model=_FakeSaveComponent("audio_model"),
                tokenizer=_FakeSaveComponent("audio_tokenizer"),
                processor=_FakeSaveComponent("audio_processor"),
            )
            audio._init_adapter = lambda: None
            audio._init_verifier = lambda: None
            audio._train_cycle = lambda cycle, samples: AudioRAFTCycleResult(
                cycle=cycle,
                samples_generated=1,
                samples_verified=1,
                samples_kept=1,
                average_reward=1.0,
                learning_rate=1e-5,
                metrics=build_cycle_summary(
                    cycle=cycle,
                    learning_rate=1e-5,
                    samples_seen=1,
                    samples_kept=1,
                    cycle_duration_seconds=0.01,
                    update_metrics={
                        "train_steps_executed": 1,
                        "train_loss": 0.1,
                        "weights_updated": True,
                        "update_reason": "updated",
                        "optimizer_steps": 1,
                        "skipped_batches_non_finite": 0,
                    },
                ),
            )
            audio.train(samples=[SimpleNamespace()])
            if not audio.training_summary.get("final_model_path"):
                raise RuntimeError("Audio smoke did not emit final_model_path")
            run_payloads["audio"] = {
                "summary": audio.training_summary,
                "output_dir": output_root / "audio",
            }

            # Reasoning smoke
            reasoning = ReasoningRAFTTrainer(
                ReasoningRAFTConfig(num_cycles=1, output_dir=str(output_root / "reasoning"), seed=7)
            )
            reasoning.model = _FakeSaveComponent("reasoning_model")
            reasoning.tokenizer = _FakeSaveComponent("reasoning_tokenizer")

            def _reasoning_cycle(samples, cycle):
                metrics = build_cycle_summary(
                    cycle=cycle,
                    learning_rate=1e-5,
                    samples_seen=1,
                    samples_kept=1,
                    cycle_duration_seconds=0.01,
                    update_metrics={
                        "train_steps_executed": 1,
                        "train_loss": 0.1,
                        "weights_updated": True,
                        "update_reason": "updated",
                        "optimizer_steps": 1,
                        "skipped_batches_non_finite": 0,
                    },
                    extra={"accuracy": 1.0, "avg_reward": 1.0},
                )
                persist_cycle_artifacts(
                    output_dir=reasoning.output_dir,
                    modality="reasoning",
                    model_name=reasoning.config.model_name,
                    cycle=cycle,
                    update_metrics=metrics,
                    model=reasoning.model,
                    tokenizer=reasoning.tokenizer,
                )
                return metrics

            reasoning.train_cycle = _reasoning_cycle
            reasoning_summary = reasoning.train(samples=[MathSample(question="1+1", answer="2")])
            if not reasoning_summary.get("final_model_path"):
                raise RuntimeError("Reasoning smoke did not emit final_model_path")
            run_payloads["reasoning"] = {
                "summary": reasoning_summary,
                "output_dir": output_root / "reasoning",
            }

            # Agentic smoke
            agentic = AgenticRAFTTrainer(
                AgenticRAFTConfig(num_cycles=1, output_dir=str(output_root / "agentic"), seed=7)
            )
            agentic.model = _FakeSaveComponent("agentic_model")
            agentic.tokenizer = _FakeSaveComponent("agentic_tokenizer")
            agentic._run_cycle = lambda samples, cycle: AgenticRAFTCycleResult(
                cycle=cycle,
                total_samples=1,
                verified_samples=1,
                avg_reward=1.0,
                success_rate=1.0,
                training_samples=1,
                metrics=build_cycle_summary(
                    cycle=cycle,
                    learning_rate=1e-5,
                    samples_seen=1,
                    samples_kept=1,
                    cycle_duration_seconds=0.01,
                    update_metrics={
                        "train_steps_executed": 1,
                        "train_loss": 0.1,
                        "weights_updated": True,
                        "update_reason": "updated",
                        "optimizer_steps": 1,
                        "skipped_batches_non_finite": 0,
                    },
                ),
            )
            agentic_summary = agentic.train(
                samples=[SimpleNamespace(prompt="prompt", expected_calls=[], is_irrelevant=False)]
            )
            if not agentic_summary.get("final_model_path"):
                raise RuntimeError("Agentic smoke did not emit final_model_path")
            run_payloads["agentic"] = {
                "summary": agentic_summary,
                "output_dir": output_root / "agentic",
            }

            return build_modality_entries_from_runs(run_payloads)

    def run_modality(
        self,
        baseline_file: Optional[str] = None,
        write_baseline: bool = False,
        compare_baseline: bool = False,
    ) -> bool:
        """Run deterministic modality training smoke checks."""
        from halo_forge.modality_baseline import (
            DEFAULT_MODALITY_BASELINE_FILE,
            build_baseline_payload,
            compare_baseline_payloads,
            compute_fixture_pack_fingerprint,
            format_drift_lines,
            load_baseline_file,
            validate_baseline_payload,
            write_baseline_file,
        )

        baseline_path = Path(baseline_file) if baseline_file else DEFAULT_MODALITY_BASELINE_FILE

        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header("Modality Smoke", "Deterministic tiny-run validation")
        else:
            print(f"\n{'='*60}")
            print("halo forge Modality Smoke Test")
            print(f"{'='*60}\n")

        self.run_test("Modality fixtures", self.test_modality_fixtures)

        try:
            import torch  # noqa: F401

            has_torch = True
        except Exception:
            has_torch = False

        modality_entries = self.run_test(
            "Modality train smoke",
            self.test_modality_train_smoke,
            skip_condition=not has_torch,
            skip_reason="torch not available in environment",
        )

        def _write_baseline() -> bool:
            if not has_torch or not isinstance(modality_entries, dict):
                raise RuntimeError("Cannot write baseline without modality smoke runtime data")
            payload = build_baseline_payload(modality_entries=modality_entries, seed=42)
            write_baseline_file(baseline_path, payload)
            self.log(f"Wrote modality baseline: {baseline_path}", "info")
            return True

        self.run_test(
            "Modality baseline write",
            _write_baseline,
            skip_condition=not write_baseline,
            skip_reason="--write-baseline not requested",
        )

        def _compare_baseline() -> bool:
            if not baseline_path.exists():
                raise RuntimeError(f"Baseline file not found: {baseline_path}")
            expected = load_baseline_file(baseline_path)
            schema_errors = validate_baseline_payload(expected)
            if schema_errors:
                raise RuntimeError("Invalid baseline schema: " + "; ".join(schema_errors))

            if not has_torch or not isinstance(modality_entries, dict):
                expected_fingerprint = str(expected.get("fixture_pack", ""))
                current_fingerprint = compute_fixture_pack_fingerprint()
                if expected_fingerprint != current_fingerprint:
                    raise RuntimeError(
                        "Fixture pack fingerprint mismatch without runtime smoke coverage. "
                        f"expected={expected_fingerprint} actual={current_fingerprint}"
                    )
                self.log(
                    "Torch runtime unavailable; validated baseline schema + fixture fingerprint only.",
                    "skip",
                )
                return True

            current = build_baseline_payload(modality_entries=modality_entries, seed=42)
            drifts = compare_baseline_payloads(expected, current)
            if drifts:
                raise RuntimeError("\n".join(format_drift_lines(drifts)))
            return True

        self.run_test(
            "Modality baseline compare",
            _compare_baseline,
            skip_condition=not compare_baseline,
            skip_reason="--compare-baseline not requested",
        )
        return self.print_summary()

    def run_ops_e2e(
        self,
        report_file: Optional[str] = None,
        strict: bool = False,
        seed: int = 42,
        fixture_pack: str = "",
    ) -> bool:
        """Run deterministic non-code ops E2E launch reliability checks."""
        from halo_forge.ops_e2e_reliability import (
            DEFAULT_OPS_E2E_REPORT_FILE,
            OPS_E2E_STATUSES,
            OpsE2EModuleResult,
            build_ops_e2e_report,
            compute_ops_e2e_reliability,
            validate_ops_e2e_module,
            write_ops_e2e_report,
        )

        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header("Ops E2E Reliability", "Launch/stop/relaunch contract checks")
        else:
            print(f"\n{'='*60}")
            print("halo forge Ops E2E Reliability")
            print(f"{'='*60}\n")

        def _resolve_fixture_pack(pack: str) -> Optional[Path]:
            text = str(pack or "").strip()
            if not text:
                return None
            if "/" in text or text.startswith("."):
                root = Path(text).expanduser()
                if not root.is_absolute():
                    root = (Path.cwd() / root).resolve()
                return root
            return (Path.cwd() / "tests" / "fixtures" / "ops_e2e" / text).resolve()

        def _run_ops_e2e() -> bool:
            pack_root = _resolve_fixture_pack(fixture_pack)
            if pack_root:
                if not pack_root.exists() or not pack_root.is_dir():
                    raise RuntimeError(f"Fixture pack directory not found: {pack_root}")
                entries: Dict[str, OpsE2EModuleResult] = {}
                for module in ("vlm", "audio", "reasoning", "agentic", "inference", "benchmark"):
                    module_dir = pack_root / module
                    if not module_dir.exists() or not module_dir.is_dir():
                        raise RuntimeError(f"Fixture pack missing module directory: {module_dir}")
                    entries[module] = validate_ops_e2e_module(
                        module=module,
                        output_dir=module_dir,
                        seed=seed,
                    )
                entries["ui_ops"] = validate_ops_e2e_module(
                    module="ui_ops",
                    output_dir=Path.cwd(),
                    seed=seed,
                )
                report = build_ops_e2e_report(module_entries=entries, seed=seed, source="cli_test")
            else:
                report = compute_ops_e2e_reliability(seed=seed, source="cli_test")

            for module in (
                "vlm",
                "audio",
                "reasoning",
                "agentic",
                "inference",
                "benchmark",
                "ui_ops",
            ):
                entry = report.modules[module]
                resume_ok = (
                    bool(entry.resume_latest_ok) if entry.resume_latest_ok is not None else False
                )
                print(
                    "OPS_E2E "
                    f"module={module} status={entry.status} "
                    f"launch={1 if entry.launch_ok else 0} "
                    f"stop={1 if entry.stop_ok else 0} "
                    f"relaunch={1 if entry.relaunch_ok else 0} "
                    f"resume={1 if resume_ok else 0}"
                )
                if entry.status not in OPS_E2E_STATUSES:
                    raise RuntimeError(f"Invalid E2E status for module={module}: {entry.status}")

            report_path = Path(report_file) if report_file else DEFAULT_OPS_E2E_REPORT_FILE
            write_ops_e2e_report(report_path, report)
            self.log(f"Wrote ops E2E report: {report_path}", "info")

            if strict:
                failing = [
                    module for module, entry in report.modules.items() if entry.status == "fail"
                ]
                if failing:
                    raise RuntimeError("Failing modules: " + ", ".join(sorted(failing)))
            return True

        self.run_test("Ops E2E launch reliability", _run_ops_e2e)
        return self.print_summary()

    def run_ops_burnin(
        self,
        *,
        burnin_profile: str = "tiny-v1",
        seed: int = 42,
        report_file: Optional[str] = None,
        baseline_file: Optional[str] = None,
        write_baseline: bool = False,
        compare_baseline: bool = False,
        strict: bool = False,
    ) -> bool:
        """Run bounded dataset-backed non-code burn-in checks."""
        from halo_forge.ops_dataset_burnin import (
            DEFAULT_OPS_BURNIN_BASELINE_FILE,
            DEFAULT_OPS_BURNIN_REPORT_FILE,
            OPS_BURNIN_STATUSES,
            build_burnin_baseline_payload,
            compare_burnin_baselines,
            compute_ops_dataset_burnin,
            format_burnin_drift_lines,
            load_burnin_baseline_file,
            validate_burnin_baseline_payload,
            write_burnin_baseline_file,
            write_ops_burnin_report,
        )

        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header("Ops Dataset Burn-In", "Bounded non-code runtime trust checks")
        else:
            print(f"\n{'='*60}")
            print("halo forge Ops Dataset Burn-In")
            print(f"{'='*60}\n")

        report_path = Path(report_file) if report_file else DEFAULT_OPS_BURNIN_REPORT_FILE
        baseline_path = Path(baseline_file) if baseline_file else DEFAULT_OPS_BURNIN_BASELINE_FILE

        report = None
        current_baseline = None
        hard_drifts: List[Dict[str, Any]] = []
        warn_drifts: List[Dict[str, Any]] = []

        def _run_burnin() -> bool:
            nonlocal report, current_baseline
            report = compute_ops_dataset_burnin(
                profile=burnin_profile,
                seed=seed,
                source="cli_test",
                execute_commands=False,
                fixture_pack="v1",
            )
            for module in (
                "vlm",
                "audio",
                "reasoning",
                "agentic",
                "inference",
                "benchmark",
                "ui_ops",
            ):
                entry = report.modules[module]
                print(
                    "OPS_BURNIN "
                    f"module={module} status={entry.status} "
                    f"errors={len(entry.errors)} warnings={len(entry.warnings)}"
                )
                if entry.status not in OPS_BURNIN_STATUSES:
                    raise RuntimeError(
                        f"Invalid burn-in status for module={module}: {entry.status}"
                    )
            write_ops_burnin_report(report_path, report)
            self.log(f"Wrote ops dataset burn-in report: {report_path}", "info")
            current_baseline = build_burnin_baseline_payload(report)
            return True

        self.run_test("Ops dataset burn-in", _run_burnin)

        def _write_baseline() -> bool:
            if current_baseline is None:
                raise RuntimeError("Burn-in baseline payload unavailable")
            write_burnin_baseline_file(baseline_path, current_baseline)
            self.log(f"Wrote burn-in baseline: {baseline_path}", "info")
            return True

        self.run_test(
            "Ops burn-in baseline write",
            _write_baseline,
            skip_condition=not write_baseline,
            skip_reason="--write-baseline not requested",
        )

        def _compare_baseline() -> bool:
            nonlocal hard_drifts, warn_drifts
            if current_baseline is None:
                raise RuntimeError("Burn-in baseline payload unavailable")
            if not baseline_path.exists():
                raise RuntimeError(f"Baseline file not found: {baseline_path}")
            expected = load_burnin_baseline_file(baseline_path)
            schema_errors = validate_burnin_baseline_payload(expected)
            if schema_errors:
                raise RuntimeError("Invalid burn-in baseline schema: " + "; ".join(schema_errors))
            drifts = compare_burnin_baselines(expected=expected, current=current_baseline)
            if drifts:
                for line in format_burnin_drift_lines(drifts):
                    print(line)
            hard_drifts = [drift for drift in drifts if drift.get("severity") == "hard"]
            warn_drifts = [drift for drift in drifts if drift.get("severity") != "hard"]
            if hard_drifts:
                raise RuntimeError("Hard burn-in contract drift detected")
            return True

        self.run_test(
            "Ops burn-in baseline compare",
            _compare_baseline,
            skip_condition=not compare_baseline,
            skip_reason="--compare-baseline not requested",
        )

        if strict and report is not None:
            failing = [module for module, entry in report.modules.items() if entry.status == "fail"]
            if failing:
                self.failures += 1
                self.log("Failing modules: " + ", ".join(sorted(failing)), "fail")
            elif warn_drifts:
                self.log(
                    f"Burn-in warning drift detected ({len(warn_drifts)} warn drift(s))",
                    "warn",
                )
        return self.print_summary()

    def run_all_modules(
        self,
        *,
        profile: str = "bounded-v1",
        seed: int = 42,
        report_file: Optional[str] = None,
        module_filters: Optional[List[str]] = None,
        strict: bool = False,
        fixture_pack: str = "",
    ) -> bool:
        """Run all-module readiness checks for coding + non-coding surfaces."""
        from halo_forge.all_module_readiness import (
            ALL_MODULES,
            ALL_MODULE_READINESS_STATUSES,
            DEFAULT_ALL_MODULE_READINESS_REPORT_FILE,
            AllModuleReadiness,
            build_all_module_readiness_report,
            compute_all_module_readiness,
            default_output_map,
            validate_all_module,
            write_all_module_readiness_report,
        )

        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header("All Module Parity", "Coding + non-coding readiness checks")
        else:
            print(f"\n{'='*60}")
            print("halo forge All Module Parity")
            print(f"{'='*60}\n")

        selected_modules = []
        for module in module_filters or []:
            key = str(module or "").strip().lower()
            if not key:
                continue
            if key not in ALL_MODULES:
                raise RuntimeError(f"Unsupported module filter: {key}")
            if key not in selected_modules:
                selected_modules.append(key)
        if not selected_modules:
            selected_modules = list(ALL_MODULES)

        def _resolve_fixture_pack(pack: str) -> Optional[Path]:
            text = str(pack or "").strip()
            if not text:
                return None
            if "/" in text or text.startswith("."):
                root = Path(text).expanduser()
                if not root.is_absolute():
                    root = (Path.cwd() / root).resolve()
                return root
            return (Path.cwd() / "tests" / "fixtures" / "all_modules" / text).resolve()

        def _run_all_module_checks() -> bool:
            pack_root = _resolve_fixture_pack(fixture_pack)
            if pack_root:
                if not pack_root.exists() or not pack_root.is_dir():
                    raise RuntimeError(f"Fixture pack directory not found: {pack_root}")
                entries: Dict[str, AllModuleReadiness] = {}
                for module in selected_modules:
                    if module == "ui_ops":
                        module_dir = Path.cwd()
                    else:
                        module_dir = pack_root / module
                        if not module_dir.exists() or not module_dir.is_dir():
                            raise RuntimeError(
                                f"Fixture pack missing module directory: {module_dir}"
                            )
                    entries[module] = validate_all_module(
                        module=module,
                        output_dir=module_dir,
                        seed=seed,
                        require_artifacts=True,
                    )
                report = build_all_module_readiness_report(
                    module_entries=entries,
                    seed=seed,
                    source="cli_test",
                )
            else:
                base_output_map = default_output_map()
                output_map = {module: base_output_map[module] for module in selected_modules}
                report = compute_all_module_readiness(
                    output_map=output_map,
                    seed=seed,
                    source="cli_test",
                    require_artifacts=False,
                )

            for module in selected_modules:
                entry = report.modules[module]
                print(
                    "ALL_READY "
                    f"module={module} status={entry.status} "
                    f"errors={len(entry.errors)} warnings={len(entry.warnings)}"
                )
                if entry.status not in ALL_MODULE_READINESS_STATUSES:
                    raise RuntimeError(
                        f"Invalid all-module status for module={module}: {entry.status}"
                    )

            report_path = (
                Path(report_file) if report_file else DEFAULT_ALL_MODULE_READINESS_REPORT_FILE
            )
            write_all_module_readiness_report(report_path, report)
            self.log(f"Wrote all-module readiness report: {report_path}", "info")

            if strict:
                failing = [
                    module for module in selected_modules if report.modules[module].status == "fail"
                ]
                if failing:
                    raise RuntimeError("Failing modules: " + ", ".join(sorted(failing)))
            return True

        self.run_test(f"All-module readiness ({profile})", _run_all_module_checks)
        return self.print_summary()

    def run_all_module_qualification(
        self,
        *,
        qualification_profile: str = "contract-v1",
        seed: int = 42,
        report_file: Optional[str] = None,
        baseline_file: Optional[str] = None,
        write_baseline: bool = False,
        compare_baseline: bool = False,
        strict: bool = False,
        module_filters: Optional[List[str]] = None,
        fixture_pack: str = "",
        show_fix_commands: bool = False,
    ) -> bool:
        """Run all-module qualification lifecycle checks with optional drift compare."""
        from halo_forge.all_module_readiness import ALL_MODULES
        from halo_forge.all_module_qualification import (
            ALL_MODULE_QUALIFICATION_STATUSES,
            DEFAULT_ALL_MODULE_QUALIFICATION_BASELINE_FILE,
            DEFAULT_ALL_MODULE_QUALIFICATION_REPORT_FILE,
            build_qualification_baseline_payload,
            compare_qualification_baselines,
            compute_all_module_qualification,
            format_qualification_drift_lines,
            format_qualification_issue_lines,
            load_qualification_baseline_file,
            validate_qualification_baseline_payload,
            write_all_module_qualification_report,
            write_qualification_baseline_file,
        )

        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header(
                "All-Module Qualification", "Bounded lifecycle qualification checks"
            )
        else:
            print(f"\n{'='*60}")
            print("halo forge All-Module Qualification")
            print(f"{'='*60}\n")

        selected_modules: List[str] = []
        for module in module_filters or []:
            key = str(module or "").strip().lower()
            if not key:
                continue
            if key not in ALL_MODULES:
                raise RuntimeError(f"Unsupported module filter: {key}")
            if key not in selected_modules:
                selected_modules.append(key)
        if not selected_modules:
            selected_modules = list(ALL_MODULES)

        report_path = (
            Path(report_file) if report_file else DEFAULT_ALL_MODULE_QUALIFICATION_REPORT_FILE
        )
        baseline_path = (
            Path(baseline_file) if baseline_file else DEFAULT_ALL_MODULE_QUALIFICATION_BASELINE_FILE
        )

        report = None
        current_baseline = None
        hard_drifts: List[Dict[str, Any]] = []
        warn_drifts: List[Dict[str, Any]] = []

        def _resolve_fixture_output_map(pack: str) -> Dict[str, str]:
            text = str(pack or "").strip()
            if not text:
                return {}
            if "/" in text or text.startswith("."):
                pack_root = Path(text).expanduser()
                if not pack_root.is_absolute():
                    pack_root = (Path.cwd() / pack_root).resolve()
            else:
                pack_root = (Path.cwd() / "tests" / "fixtures" / "all_modules" / text).resolve()

            if not pack_root.exists() or not pack_root.is_dir():
                raise RuntimeError(f"Fixture pack directory not found: {pack_root}")

            output_map: Dict[str, str] = {}
            for module in ALL_MODULES:
                if module == "ui_ops":
                    output_map[module] = str(Path.cwd())
                    continue
                module_dir = pack_root / module
                if not module_dir.exists() or not module_dir.is_dir():
                    raise RuntimeError(f"Fixture pack missing module directory: {module_dir}")
                output_map[module] = str(module_dir)
            return output_map

        def _run_qualification() -> bool:
            nonlocal report, current_baseline
            output_map: Dict[str, str] = {}
            if qualification_profile == "fixture-v1":
                output_map = _resolve_fixture_output_map(fixture_pack or "v1")
            elif fixture_pack:
                output_map = _resolve_fixture_output_map(fixture_pack)

            report = compute_all_module_qualification(
                output_map=output_map or None,
                seed=seed,
                profile=qualification_profile,
                source="cli_test",
                module_filters=selected_modules,
            )

            for module in selected_modules:
                entry = report.modules[module]
                print(
                    "ALL_QUAL "
                    f"module={module} status={entry.status} "
                    f"errors={len(entry.errors)} warnings={len(entry.warnings)}"
                )
                for line in format_qualification_issue_lines(
                    entry,
                    show_fix_commands=show_fix_commands,
                ):
                    print(line)
                if entry.status not in ALL_MODULE_QUALIFICATION_STATUSES:
                    raise RuntimeError(
                        f"Invalid qualification status for module={module}: {entry.status}"
                    )

            write_all_module_qualification_report(report_path, report)
            self.log(f"Wrote all-module qualification report: {report_path}", "info")
            current_baseline = build_qualification_baseline_payload(report)
            return True

        self.run_test(f"All-module qualification ({qualification_profile})", _run_qualification)

        def _write_baseline() -> bool:
            if current_baseline is None:
                raise RuntimeError("Qualification baseline payload unavailable")
            write_qualification_baseline_file(baseline_path, current_baseline)
            self.log(f"Wrote qualification baseline: {baseline_path}", "info")
            return True

        self.run_test(
            "Qualification baseline write",
            _write_baseline,
            skip_condition=not write_baseline,
            skip_reason="--write-baseline not requested",
        )

        def _compare_baseline() -> bool:
            nonlocal hard_drifts, warn_drifts
            if current_baseline is None:
                raise RuntimeError("Qualification baseline payload unavailable")
            if not baseline_path.exists():
                raise RuntimeError(f"Baseline file not found: {baseline_path}")
            expected = load_qualification_baseline_file(baseline_path)
            schema_errors = validate_qualification_baseline_payload(expected)
            if schema_errors:
                raise RuntimeError(
                    "Invalid qualification baseline schema: " + "; ".join(schema_errors)
                )
            drifts = compare_qualification_baselines(expected=expected, current=current_baseline)
            if drifts:
                for line in format_qualification_drift_lines(drifts):
                    print(line)
            hard_drifts = [drift for drift in drifts if drift.get("severity") == "hard"]
            warn_drifts = [drift for drift in drifts if drift.get("severity") != "hard"]
            if hard_drifts:
                raise RuntimeError("Hard qualification drift detected")
            return True

        self.run_test(
            "Qualification baseline compare",
            _compare_baseline,
            skip_condition=not compare_baseline,
            skip_reason="--compare-baseline not requested",
        )

        if strict and report is not None:
            failing = [
                module for module in selected_modules if report.modules[module].status == "fail"
            ]
            if failing:
                self.failures += 1
                self.log("Failing modules: " + ", ".join(sorted(failing)), "fail")
            elif warn_drifts:
                self.log(
                    f"Qualification warning drift detected ({len(warn_drifts)} warn drift(s))",
                    "warn",
                )
        return self.print_summary()

    def run_all_module_bootstrap(
        self,
        *,
        bootstrap_profile: str = "contract-v1",
        seed: int = 42,
        output_root: Optional[str] = None,
        report_file: Optional[str] = None,
        module_filters: Optional[List[str]] = None,
        strict: bool = False,
    ) -> bool:
        """Run bounded all-module bootstrap evidence generation."""
        from halo_forge.all_module_readiness import ALL_MODULES
        from halo_forge.all_module_bootstrap import (
            DEFAULT_ALL_MODULE_BOOTSTRAP_OUTPUT_ROOT,
            DEFAULT_ALL_MODULE_BOOTSTRAP_REPORT_FILE,
            compute_all_module_bootstrap,
            write_all_module_bootstrap_report,
        )

        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header(
                "All-Module Bootstrap",
                "Bounded evidence generation for readiness remediation",
            )
        else:
            print(f"\n{'='*60}")
            print("halo forge All-Module Bootstrap")
            print(f"{'='*60}\n")

        selected_modules: List[str] = []
        for module in module_filters or []:
            key = str(module or "").strip().lower()
            if not key:
                continue
            if key not in ALL_MODULES:
                raise RuntimeError(f"Unsupported module filter: {key}")
            if key not in selected_modules:
                selected_modules.append(key)
        if not selected_modules:
            selected_modules = list(ALL_MODULES)

        report_path = Path(report_file) if report_file else DEFAULT_ALL_MODULE_BOOTSTRAP_REPORT_FILE
        output_root_path = (
            Path(output_root) if output_root else DEFAULT_ALL_MODULE_BOOTSTRAP_OUTPUT_ROOT
        )

        report = None

        def _run_bootstrap() -> bool:
            nonlocal report
            report = compute_all_module_bootstrap(
                bootstrap_profile=bootstrap_profile,
                seed=seed,
                source="cli_test",
                output_root=output_root_path,
                module_filters=selected_modules,
                strict=strict,
            )

            for module in selected_modules:
                entry = report.modules[module]
                print(
                    "ALL_BOOTSTRAP "
                    f"module={module} status={entry.status} "
                    f"created={len(entry.artifacts_created)} "
                    f"errors={len(entry.errors)} warnings={len(entry.warnings)}"
                )

            write_all_module_bootstrap_report(report_path, report)
            self.log(f"Wrote all-module bootstrap report: {report_path}", "info")
            return True

        self.run_test(
            f"All-module bootstrap ({bootstrap_profile})",
            _run_bootstrap,
        )

        if strict and report is not None:
            failing = [
                module for module in selected_modules if report.modules[module].status == "fail"
            ]
            if failing:
                self.failures += 1
                self.log("Failing modules: " + ", ".join(sorted(failing)), "fail")
        return self.print_summary()

    def run_all_module_live(
        self,
        *,
        live_profile: str = "live-smoke-v1",
        seed: int = 42,
        output_root: Optional[str] = None,
        report_file: Optional[str] = None,
        module_filters: Optional[List[str]] = None,
        strict: bool = False,
    ) -> bool:
        """Run bounded all-module live execution probes."""
        from halo_forge.all_module_readiness import ALL_MODULES
        from halo_forge.all_module_live_execution import (
            DEFAULT_ALL_MODULE_LIVE_OUTPUT_ROOT,
            DEFAULT_ALL_MODULE_LIVE_REPORT_FILE,
            compute_all_module_live_execution,
            write_all_module_live_execution_report,
        )

        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header(
                "All-Module Live Execution",
                "Bounded live-local/smoke probe closure checks",
            )
        else:
            print(f"\n{'='*60}")
            print("halo forge All-Module Live Execution")
            print(f"{'='*60}\n")

        selected_modules: List[str] = []
        for module in module_filters or []:
            key = str(module or "").strip().lower()
            if not key:
                continue
            if key not in ALL_MODULES:
                raise RuntimeError(f"Unsupported module filter: {key}")
            if key not in selected_modules:
                selected_modules.append(key)
        if not selected_modules:
            selected_modules = list(ALL_MODULES)

        report_path = Path(report_file) if report_file else DEFAULT_ALL_MODULE_LIVE_REPORT_FILE
        output_root_path = Path(output_root) if output_root else DEFAULT_ALL_MODULE_LIVE_OUTPUT_ROOT

        report = None

        def _run_live() -> bool:
            nonlocal report
            report = compute_all_module_live_execution(
                live_profile=live_profile,
                seed=seed,
                source="cli_test",
                output_root=output_root_path,
                module_filters=selected_modules,
                strict=strict,
            )

            for module in selected_modules:
                entry = report.modules[module]
                print(
                    "ALL_LIVE "
                    f"module={module} status={entry.status} "
                    f"launch={1 if entry.launch_ok else 0} "
                    f"monitor={1 if entry.monitor_ok else 0} "
                    f"results={1 if entry.results_ok else 0} "
                    f"errors={len(entry.errors)} warnings={len(entry.warnings)}"
                )

            write_all_module_live_execution_report(report_path, report)
            self.log(f"Wrote all-module live execution report: {report_path}", "info")
            return True

        self.run_test(
            f"All-module live execution ({live_profile})",
            _run_live,
        )

        if strict and report is not None:
            failing = [
                module for module in selected_modules if report.modules[module].status == "fail"
            ]
            if failing:
                self.failures += 1
                self.log("Failing modules: " + ", ".join(sorted(failing)), "fail")
        return self.print_summary()

    def run_walkthroughs(
        self,
        *,
        profile: str = "contract-v1",
        seed: int = 42,
        report_file: Optional[str] = None,
        module_filters: Optional[List[str]] = None,
        strict: bool = False,
        execute: bool = False,
    ) -> bool:
        """Run all-module walkthrough contracts for local/operator validation."""
        from halo_forge.all_module_walkthroughs import (
            DEFAULT_WALKTHROUGH_REPORT_FILE,
            WALKTHROUGH_PROFILES,
            compute_walkthroughs,
            write_walkthrough_report,
        )

        if profile not in WALKTHROUGH_PROFILES:
            raise RuntimeError(
                f"Invalid walkthrough profile: {profile}. "
                f"Expected one of {', '.join(WALKTHROUGH_PROFILES)}"
            )

        selected_modules: List[str] = []
        for module in module_filters or []:
            key = str(module or "").strip().lower()
            if not key:
                continue
            if key not in (
                "config",
                "data",
                "info",
                "plot",
                "sft",
                "raft",
                "benchmark_code",
                "benchmark_non_code",
                "inference",
                "vlm",
                "audio",
                "reasoning",
                "agentic",
                "ui_ops",
            ):
                raise RuntimeError(f"Unsupported walkthrough module filter: {key}")
            if key not in selected_modules:
                selected_modules.append(key)

        if not selected_modules:
            selected_modules = []

        def _run_walkthroughs() -> bool:
            report = compute_walkthroughs(
                modules=selected_modules,
                seed=seed,
                profile=profile,
                execute=execute,
            )
            modules_to_print = selected_modules or list(report.modules.keys())
            for module in modules_to_print:
                entry = report.modules[module]
                print(
                    "WALKTHROUGH "
                    f"module={module} status={entry.status} "
                    f"steps={len(entry.steps)} errors={len(entry.errors)} warnings={len(entry.warnings)}"
                )

            path = Path(report_file) if report_file else DEFAULT_WALKTHROUGH_REPORT_FILE
            write_walkthrough_report(path, report)
            self.log(f"Wrote walkthrough report: {path}", "info")

            if strict:
                failing = [
                    module for module in modules_to_print if report.modules[module].status == "fail"
                ]
                if failing:
                    raise RuntimeError("Failing modules: " + ", ".join(sorted(failing)))
            return True

        self.run_test(f"All-module walkthroughs ({profile})", _run_walkthroughs)
        return self.print_summary()


def cmd_test(args):
    """Run pipeline validation tests."""
    baseline_levels = {"modality", "ops-burnin", "all-module-qualification"}
    if args.level not in baseline_levels and (args.write_baseline or args.compare_baseline):
        print(
            f"{RED}Error: --write-baseline/--compare-baseline are supported only with "
            f"--level modality, --level ops-burnin, or --level all-module-qualification{NC}"
        )
        sys.exit(2)

    runner = TestRunner(verbose=args.verbose, model=args.model)

    if args.level == "smoke":
        success = runner.run_smoke()
    elif args.level == "standard":
        success = runner.run_standard()
    elif args.level == "full":
        success = runner.run_full()
    elif args.level == "modality":
        success = runner.run_modality(
            baseline_file=args.baseline_file,
            write_baseline=args.write_baseline,
            compare_baseline=args.compare_baseline,
        )
    elif args.level == "ops-e2e":
        success = runner.run_ops_e2e(
            report_file=args.report_file,
            strict=args.strict,
            seed=args.seed,
            fixture_pack=args.fixture_pack,
        )
    elif args.level == "ops-burnin":
        report_file = args.report_file
        baseline_file = args.baseline_file
        if report_file == "results/readiness/ops_e2e_launch_reliability.v1.json":
            report_file = "results/readiness/ops_dataset_burnin.v1.json"
        if baseline_file == "tests/baselines/modality_runtime_baseline.v1.json":
            baseline_file = "tests/baselines/ops_dataset_burnin_baseline.v1.json"
        success = runner.run_ops_burnin(
            burnin_profile=args.burnin_profile,
            seed=args.seed,
            report_file=report_file,
            baseline_file=baseline_file,
            write_baseline=args.write_baseline,
            compare_baseline=args.compare_baseline,
            strict=args.strict,
        )
    elif args.level == "all-modules":
        report_file = args.report_file
        if report_file == "results/readiness/ops_e2e_launch_reliability.v1.json":
            report_file = "results/readiness/all_modules_readiness.v1.json"
        success = runner.run_all_modules(
            profile=args.profile,
            seed=args.seed,
            report_file=report_file,
            module_filters=args.module,
            strict=args.strict,
            fixture_pack=args.fixture_pack,
        )
    elif args.level == "walkthroughs":
        report_file = args.report_file
        profile = args.profile
        if report_file == "results/readiness/ops_e2e_launch_reliability.v1.json":
            report_file = (
                ".internal_docs/research_testing/walkthroughs/reports/"
                "all_module_e2e_walkthrough_report.v1.json"
            )
        if profile == "bounded-v1":
            profile = "contract-v1"
        success = runner.run_walkthroughs(
            profile=profile,
            seed=args.seed,
            report_file=report_file,
            module_filters=args.module,
            strict=args.strict,
            execute=args.execute,
        )
    elif args.level == "all-module-qualification":
        report_file = args.report_file
        baseline_file = args.baseline_file
        if report_file == "results/readiness/ops_e2e_launch_reliability.v1.json":
            report_file = "results/readiness/all_module_qualification.v1.json"
        if baseline_file == "tests/baselines/modality_runtime_baseline.v1.json":
            baseline_file = "tests/baselines/all_module_qualification_baseline.v1.json"
        success = runner.run_all_module_qualification(
            qualification_profile=args.qualification_profile,
            seed=args.seed,
            report_file=report_file,
            baseline_file=baseline_file,
            write_baseline=args.write_baseline,
            compare_baseline=args.compare_baseline,
            strict=args.strict,
            module_filters=args.module,
            fixture_pack=args.fixture_pack,
            show_fix_commands=args.show_fix_commands,
        )
    elif args.level == "all-module-bootstrap":
        report_file = args.report_file
        if report_file == "results/readiness/ops_e2e_launch_reliability.v1.json":
            report_file = "results/readiness/all_module_bootstrap.v1.json"
        success = runner.run_all_module_bootstrap(
            bootstrap_profile=args.bootstrap_profile,
            seed=args.seed,
            output_root=args.output_root,
            report_file=report_file,
            module_filters=args.module,
            strict=args.strict,
        )
    elif args.level == "all-module-live":
        report_file = args.report_file
        output_root = args.output_root
        if report_file == "results/readiness/ops_e2e_launch_reliability.v1.json":
            report_file = "results/readiness/all_module_live_execution.v1.json"
        if output_root == "results/bootstrap":
            output_root = "results/live_probes"
        success = runner.run_all_module_live(
            live_profile=args.live_profile,
            seed=args.seed,
            output_root=output_root,
            report_file=report_file,
            module_filters=args.module,
            strict=args.strict,
        )
    else:
        print(f"Unknown test level: {args.level}")
        print(
            "Valid levels: smoke, standard, full, modality, ops-e2e, ops-burnin, "
            "all-modules, walkthroughs, all-module-qualification, all-module-bootstrap, all-module-live"
        )
        sys.exit(1)

    sys.exit(0 if success else 1)


def cmd_inference_optimize(args):
    """Optimize model for inference."""
    from halo_forge.inference import (
        InferenceOptimizer,
        OptimizationConfig,
        check_dependencies,
        validate_config,
    )

    print_banner()
    print(f"{GREEN}Inference Optimization{NC}")
    print("=" * 60)
    print(f"Optimizing model: {args.model}")
    print(f"Target precision: {args.target_precision}")
    print(f"Target latency: {args.target_latency}ms")

    # MLX path: bitsandbytes-style quantization is not applicable; weights are
    # quantized at conversion time (use mlx-community/...-4bit models or
    # `python -m mlx_lm.convert`). We surface that and run a smoke generation
    # instead of the full PyTorch optimize/calibrate pipeline.
    if getattr(args, "accelerator", "auto") == "mlx":
        from halo_forge.backend.mlx import MLXInferenceAdapter

        print("\n[MLX] Skipping torch optimize/calibrate; running smoke generation.")
        adapter = MLXInferenceAdapter(args.model)
        adapter.load()
        out = adapter.generate(
            "Write a function to sort a list.",
            max_tokens=64,
            temperature=0.7,
        )
        print("\nSample generation:")
        print(out)
        adapter.cleanup()
        print(f"\n{GREEN}MLX smoke OK.{NC} For pre-quantized weights see docs/MLX.md.")
        return

    config = OptimizationConfig(
        target_precision=args.target_precision,
        target_latency_ms=args.target_latency,
        output_dir=args.output,
    )

    # Handle --dry-run
    if getattr(args, "dry_run", False):
        print("\n[DRY RUN] Validating configuration and dependencies...")

        # Check dependencies
        deps = check_dependencies()
        print("\nDependencies:")
        for dep, available in deps.items():
            status = f"{GREEN}✓{NC}" if available else f"{RED}✗{NC}"
            print(f"  {status} {dep}")

        # Validate config
        try:
            warnings = validate_config(config)
            if warnings:
                print("\nWarnings:")
                for w in warnings:
                    print(f"  {YELLOW}⚠{NC} {w}")
            else:
                print(f"\n{GREEN}Configuration valid!{NC}")
        except Exception as e:
            print(f"\n{RED}Configuration error: {e}{NC}")
            sys.exit(1)

        # Check model path
        from pathlib import Path

        model_path = Path(args.model)
        if model_path.exists():
            print(f"\n{GREEN}✓{NC} Model path exists: {args.model}")
        else:
            print(f"\n{YELLOW}⚠{NC} Model path not found locally (may be HuggingFace ID)")

        print(f"\n{GREEN}[DRY RUN] All checks passed!{NC}")
        return

    optimizer = InferenceOptimizer(config)

    # Simple eval prompts for verification
    eval_prompts = [
        "Write a function to sort a list.",
        "Implement a binary search.",
        "Create a linked list class.",
    ]

    result = optimizer.optimize(
        model_path=args.model, calibration_data=args.calibration_data, eval_prompts=eval_prompts
    )

    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETE")
    print("=" * 50)
    print(f"Success: {result['success']}")
    if result.get("verification"):
        metrics = result["verification"]["metrics"]
        print(f"Latency: {metrics.get('avg_latency_ms', 0):.1f}ms")
        print(f"Quality: {metrics.get('quality_score', 0):.2%}")
    print(f"Output: {args.output}")


def cmd_inference_export(args):
    """Export model to deployment format."""
    print_banner()
    print(f"{GREEN}Model Export{NC}")
    print("=" * 60)
    print(f"Exporting model: {args.model}")
    print(f"Format: {args.format}")
    print(f"Output: {args.output}")

    if args.format == "gguf":
        from halo_forge.inference.export import GGUFExporter

        print(f"Quantization: {args.quantization}")

        # Load model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, device_map="cpu"  # Export on CPU
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        exporter = GGUFExporter()
        output_path = exporter.export(
            model, args.output, tokenizer=tokenizer, quantization=args.quantization
        )

        print(f"\nExported to: {output_path}")

    elif args.format == "onnx":
        from halo_forge.inference.export import ONNXExporter

        # Load model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        exporter = ONNXExporter()
        output_path = exporter.export(model, args.output, tokenizer=tokenizer)

        print(f"\nExported to: {output_path}")


def cmd_inference_benchmark(args):
    """Benchmark inference latency."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import time

    print_banner()
    print(f"{GREEN}Inference Benchmark{NC}")
    print("=" * 60)
    print(f"Benchmarking: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Warmup iterations: {args.warmup}")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=recommended_dtype(), device_map=get_device_map(), trust_remote_code=True
    )

    # Get test prompts
    if args.prompts:
        with open(args.prompts) as f:
            prompts = [json.loads(line).get("prompt", "") for line in f][: args.num_prompts]
    else:
        prompts = [
            "Write a function to calculate fibonacci numbers.",
            "Implement a binary search tree.",
            "Create a simple HTTP server.",
            "Write a sorting algorithm.",
            "Implement a stack data structure.",
        ][: args.num_prompts]

    print(f"Testing with {len(prompts)} prompts...\n")

    # Warmup
    print("Warmup...")
    for i, prompt in enumerate(prompts[: args.warmup]):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)

    # Benchmark
    print("Benchmarking...")
    latencies = []
    tokens_generated = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        num_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

        latencies.append(latency_ms)
        tokens_generated.append(num_tokens)

    # Calculate metrics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    total_tokens = sum(tokens_generated)
    total_time = sum(latencies) / 1000
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Prompts tested: {len(prompts)}")
    print(f"Avg latency:    {avg_latency:.1f}ms")
    print(f"Min latency:    {min_latency:.1f}ms")
    print(f"Max latency:    {max_latency:.1f}ms")
    print(f"Tokens/second:  {tokens_per_second:.1f}")

    if args.measure_memory and torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak memory:    {memory_mb:.0f}MB")


def cmd_vlm_train(args):
    """Train VLM with RAFT."""
    _prepare_reward_integrity(args, trainer="vlm")
    if _enqueue_managed_reward_training(args, trainer="vlm"):
        return
    profile_verifier = _prepare_profile_verifier(
        args, consumer="direct", modality="vlm", training=True
    )
    _apply_managed_dataset_args(args, "vlm", "dataset")
    from halo_forge.vlm import VLMRAFTTrainer
    from halo_forge.vlm.trainer import VLMRAFTConfig
    from halo_forge.vlm.data import list_vlm_datasets
    from halo_forge.vlm.verifiers import check_vlm_dependencies

    print_banner()

    print(f"\n{GREEN}VLM RAFT Training{NC}")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Dataset:     {args.dataset}")
    print(f"Output:      {args.output}")
    print(f"Cycles:      {args.cycles}")
    print(f"Seed:        {args.seed}")
    print("=" * 60)

    _enforce_modality_train_contract("vlm", args)

    # Handle --dry-run
    if getattr(args, "dry_run", False):
        print("\n[DRY RUN] Validating configuration and dependencies...")

        # Check VLM dependencies
        deps = check_vlm_dependencies()
        print("\nVLM Dependencies:")
        for dep, available in deps.items():
            status = f"{GREEN}✓{NC}" if available else f"{YELLOW}⚠{NC}"
            print(f"  {status} {dep}")

        # Check dataset
        if args.dataset.endswith(".jsonl"):
            from pathlib import Path

            dataset_path = Path(args.dataset)
            if dataset_path.exists():
                # Count samples
                with open(dataset_path) as f:
                    count = sum(1 for _ in f)
                print(f"\n{GREEN}✓{NC} Dataset: {args.dataset} ({count} samples)")
            else:
                print(f"\n{RED}✗{NC} Dataset not found: {args.dataset}")
                sys.exit(1)
        else:
            available = list_vlm_datasets()
            if args.dataset in available:
                print(f"\n{GREEN}✓{NC} Dataset: {args.dataset} (HuggingFace)")
            else:
                print(f"\n{RED}✗{NC} Unknown dataset: {args.dataset}")
                print(f"  Available: {', '.join(available)}")
                sys.exit(1)

        # Validate config values
        print("\nConfiguration:")
        print(f"  Cycles: {args.cycles}")
        print(f"  Samples/prompt: {args.samples_per_prompt}")
        print(f"  Perception weight: {args.perception_weight}")
        print(f"  Reasoning weight: {args.reasoning_weight}")
        print(f"  Output weight: {args.output_weight}")
        print(f"  LR decay: {args.lr_decay}")
        print(f"  Temperature: {args.temperature}")

        # Check model (just print - can't validate without loading)
        print(f"\nModel: {args.model}")
        print(f"  (Model will be loaded at training start)")

        print(f"\n{GREEN}[DRY RUN] All checks passed!{NC}")
        return

    # Create config
    config = VLMRAFTConfig(
        model_name=args.model,
        output_dir=args.output,
        num_cycles=args.cycles,
        samples_per_prompt=args.samples_per_prompt,
        reward_threshold=args.reward_threshold,
        perception_weight=args.perception_weight,
        reasoning_weight=args.reasoning_weight,
        output_weight=args.output_weight,
        lr_decay_per_cycle=args.lr_decay,
        temperature=args.temperature,
        enable_neural_accelerators=getattr(args, "enable_neural_accelerators", False),
        seed=args.seed,
    )
    if profile_verifier is not None:
        contract = dict(getattr(args, "_verifier_profile_contract", {}) or {})
        config.reward_threshold = float(
            contract.get("threshold")
            if contract.get("threshold") is not None
            else contract.get("minimum", config.reward_threshold)
        )

    # Load dataset
    if args.dataset.endswith(".jsonl"):
        dataset_path = args.dataset
    else:
        available = list_vlm_datasets()
        if args.dataset not in available:
            print(f"{RED}Error: Unknown dataset '{args.dataset}'{NC}")
            print(f"Available: {', '.join(available)}")
            sys.exit(1)
        dataset_path = args.dataset

    # Create trainer and run
    signal_session = _build_training_signal_session(
        args,
        trainer="vlm",
        output_dir=config.output_dir,
        total_boundaries=config.num_cycles,
        reward_threshold=config.reward_threshold,
    )
    trainer = VLMRAFTTrainer(config, signal_sink=signal_session)
    if profile_verifier is not None:
        trainer.verifier = profile_verifier

    try:
        summary = trainer.train(
            dataset_path,
            resume_from=getattr(args, "resume_from_cycle", 0),
        )
    except ValueError as e:
        print(f"{RED}Training error: {e}{NC}")
        sys.exit(2)
    finally:
        trainer.cleanup()

    total_steps = int(summary.get("total_train_steps_executed", 0))
    final_loss = summary.get("final_train_loss")
    _enforce_training_outcome_or_exit("vlm", summary)

    print(f"\n{GREEN}Training complete!{NC}")
    print(f"Output: {args.output}")
    if summary.get("final_model_path"):
        print(f"Final model: {summary['final_model_path']}")
    _print_training_run_metadata(summary)
    print(f"Train steps executed: {total_steps}")
    if isinstance(final_loss, (int, float)):
        print(f"Final train loss: {final_loss:.4f}")
    _finalize_managed_training_replay(args, "vlm", args.output, config, summary)


def cmd_vlm_sft(args):
    """SFT training for VLM."""
    from halo_forge.sft.trainer import SFTTrainer, SFTConfig

    print_banner()
    print(f"{GREEN}VLM SFT Training{NC}")
    print("=" * 60)

    dataset = getattr(args, "dataset", "llava")
    max_samples = getattr(args, "max_samples", None)
    dry_run = getattr(args, "dry_run", False)

    print(f"Model: {args.model}")
    print(f"Dataset: {dataset}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print()

    if dry_run:
        print(f"{YELLOW}Dry run mode - validating configuration only{NC}")
        from halo_forge.sft.datasets import get_sft_dataset_spec, is_huggingface_id

        spec = get_sft_dataset_spec(dataset)
        if spec:
            print(f"{GREEN}✓{NC} Dataset: {spec.name} ({spec.huggingface_id})")
        elif is_huggingface_id(dataset):
            print(f"{GREEN}✓{NC} HuggingFace dataset: {dataset}")
        else:
            print(f"{RED}✗{NC} Unknown dataset: {dataset}")
            sys.exit(1)
        print(f"{GREEN}Configuration valid!{NC}")
        return

    config = SFTConfig(
        model_name=args.model,
        dataset=dataset,
        max_samples=max_samples,
        output_dir=args.output,
        num_epochs=args.epochs,
    )

    trainer = SFTTrainer(config)
    summary = trainer.train()
    _print_completed_training_summary("vlm_sft", args.output, summary)


def cmd_vlm_benchmark(args):
    """Benchmark VLM on dataset."""
    from halo_forge.vlm.data import load_vlm_dataset
    from halo_forge.vlm.models import get_vlm_adapter
    from halo_forge.vlm.verifiers import VisionVerifier

    print_banner()

    print(f"\n{GREEN}VLM Benchmark{NC}")
    print("=" * 60)
    print(f"Model:   {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Split:   {args.split}")
    print(f"Limit:   {args.limit}")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_vlm_dataset(args.dataset, split=args.split, limit=args.limit)

    # Load model
    print("\nLoading model...")
    adapter = get_vlm_adapter(args.model)
    adapter.load()

    # Initialize verifier
    verifier = VisionVerifier()

    # Run benchmark
    print(f"\nBenchmarking {len(dataset)} samples...")
    results = []
    correct = 0
    total_reward = 0.0

    from tqdm import tqdm

    for sample in tqdm(dataset, desc="Evaluating"):
        # Generate
        output = adapter.generate(
            image=sample.load_image(),
            prompt=sample.prompt,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
        )

        # Verify
        result = verifier.verify(
            image=sample.load_image(),
            prompt=sample.prompt,
            completion=output.text,
            ground_truth=sample.ground_truth,
        )

        results.append(
            {
                "prompt": sample.prompt[:100],
                "ground_truth": sample.ground_truth,
                "completion": output.text[:200],
                "reward": result.reward,
                "success": result.success,
            }
        )

        if result.success:
            correct += 1
        total_reward += result.reward

    # Print results
    print("\n" + "=" * 60)
    print("VLM BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total samples:  {len(results)}")
    print(f"Correct:        {correct} ({correct/len(results)*100:.1f}%)")
    print(f"Avg reward:     {total_reward/len(results):.3f}")

    # Save results if output specified
    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "dataset": args.dataset,
                    "split": args.split,
                    "accuracy": correct / len(results),
                    "avg_reward": total_reward / len(results),
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to: {args.output}")

    # Cleanup
    adapter.cleanup()
    verifier.cleanup()


def cmd_vlm_datasets(args):
    """List available VLM datasets."""
    from halo_forge.vlm.data import list_vlm_datasets

    print_banner()

    print(f"\n{GREEN}Available VLM Datasets{NC}")
    print("=" * 60)

    datasets = list_vlm_datasets()

    dataset_info = {
        "textvqa": "Text reading in natural images",
        "docvqa": "Document understanding",
        "chartqa": "Chart interpretation",
        "realworldqa": "Real-world visual reasoning",
        "mathvista": "Mathematical reasoning with visuals",
    }

    for name in datasets:
        desc = dataset_info.get(name, "Vision-language dataset")
        print(f"  {name:15} - {desc}")


# =============================================================================
# Audio Commands
# =============================================================================


def cmd_audio_datasets(args):
    """List available audio datasets."""
    from halo_forge.audio.data import list_audio_datasets

    print_banner()

    print(f"\n{GREEN}Available Audio Datasets{NC}")
    print("=" * 60)

    dataset_info = {
        "librispeech": ("ASR", "Clean audiobook speech (960h)"),
        "common_voice": ("ASR", "Crowdsourced multilingual (2000h+)"),
        "audioset": ("Classification", "Sound event detection (5M clips)"),
        "speech_commands": ("Classification", "Keyword spotting (105k)"),
    }

    datasets = list_audio_datasets()

    for name in datasets:
        task, desc = dataset_info.get(name, ("Unknown", "Audio dataset"))
        print(f"  {name:18} [{task:14}] - {desc}")

    print()
    print("Usage:")
    print("  halo-forge audio benchmark --model openai/whisper-small --dataset librispeech")
    print("  halo-forge audio train --model openai/whisper-small --dataset librispeech --seed 42")


def cmd_audio_sft(args):
    """SFT training for audio."""
    from halo_forge.sft.trainer import SFTTrainer, SFTConfig

    print_banner()
    print(f"{GREEN}Audio SFT Training{NC}")
    print("=" * 60)

    dataset = getattr(args, "dataset", "librispeech_sft")
    max_samples = getattr(args, "max_samples", None)
    dry_run = getattr(args, "dry_run", False)

    print(f"Model: {args.model}")
    print(f"Dataset: {dataset}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print()

    if dry_run:
        print(f"{YELLOW}Dry run mode - validating configuration only{NC}")
        from halo_forge.sft.datasets import get_sft_dataset_spec, is_huggingface_id

        spec = get_sft_dataset_spec(dataset)
        if spec:
            print(f"{GREEN}✓{NC} Dataset: {spec.name} ({spec.huggingface_id})")
        elif is_huggingface_id(dataset):
            print(f"{GREEN}✓{NC} HuggingFace dataset: {dataset}")
        else:
            print(f"{RED}✗{NC} Unknown dataset: {dataset}")
            sys.exit(1)
        print(f"{GREEN}Configuration valid!{NC}")
        return

    config = SFTConfig(
        model_name=args.model,
        dataset=dataset,
        max_samples=max_samples,
        output_dir=args.output,
        num_epochs=args.epochs,
    )

    trainer = SFTTrainer(config)
    summary = trainer.train()
    _print_completed_training_summary("audio_sft", args.output, summary)


def cmd_audio_benchmark(args):
    """Benchmark audio model."""
    from halo_forge.audio import AudioRAFTTrainer, AudioRAFTConfig

    print_banner()

    print(f"\n{GREEN}Audio Benchmark{NC}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Limit: {args.limit}")

    # Check dependencies
    try:
        from halo_forge.audio.data.processors import check_audio_dependencies

        deps = check_audio_dependencies()

        if not deps.get("torchaudio"):
            print(f"\n{YELLOW}Warning: torchaudio not installed{NC}")
            print("Install with: pip install torchaudio")
    except ImportError as e:
        print(f"\n{RED}Error: {e}{NC}")
        sys.exit(1)

    # Create config
    config = AudioRAFTConfig(
        model_name=args.model,
        task=args.task,
        wer_threshold=0.3,
    )

    # Run benchmark
    trainer = AudioRAFTTrainer(config)
    results = trainer.benchmark(args.dataset, limit=args.limit)
    results["model"] = args.model
    results["benchmark"] = args.dataset
    results["task"] = args.task

    print(f"\n{GREEN}Results:{NC}")
    print(f"  Samples: {results['samples']}")
    print(f"  Success rate: {results['success_rate']:.1%}")
    print(f"  Average reward: {results['average_reward']:.3f}")

    if args.task == "asr":
        print(f"  Average WER: {results.get('average_wer', 'N/A'):.1%}")

    # Save results
    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


def cmd_audio_train(args):
    """Train audio model with RAFT."""
    _prepare_reward_integrity(args, trainer="audio")
    if _enqueue_managed_reward_training(args, trainer="audio"):
        return
    profile_verifier = _prepare_profile_verifier(
        args, consumer="direct", modality="audio", training=True
    )
    _apply_managed_dataset_args(args, "audio", "dataset")
    from halo_forge.audio import AudioRAFTTrainer, AudioRAFTConfig

    print_banner()

    print(f"\n{GREEN}Audio RAFT Training{NC}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Cycles: {args.cycles}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")

    _enforce_modality_train_contract("audio", args)

    if args.dry_run:
        print(f"\n{YELLOW}Dry run mode - validating configuration only{NC}")

        # Check dependencies
        try:
            from halo_forge.audio.data.processors import check_audio_dependencies

            deps = check_audio_dependencies()

            print(f"\nDependencies:")
            for dep, installed in deps.items():
                status = f"{GREEN}✓{NC}" if installed else f"{RED}✗{NC}"
                print(f"  {status} {dep}")

            # Try loading dataset info
            from halo_forge.audio.data import list_audio_datasets

            if args.dataset in list_audio_datasets():
                print(f"\n{GREEN}✓{NC} Dataset: {args.dataset}")
            else:
                print(f"\n{YELLOW}⚠{NC} Dataset: {args.dataset} (custom path)")

            print(f"\n{GREEN}Configuration validated successfully.{NC}")
        except Exception as e:
            print(f"\n{RED}Validation error: {e}{NC}")
            sys.exit(1)
        return

    # Create config
    config = AudioRAFTConfig(
        model_name=args.model,
        task=args.task,
        num_cycles=args.cycles,
        reward_threshold=args.reward_threshold,
        learning_rate=args.lr,
        lr_decay_per_cycle=args.lr_decay,
        output_dir=args.output,
        enable_neural_accelerators=getattr(args, "enable_neural_accelerators", False),
        seed=args.seed,
    )
    if profile_verifier is not None:
        contract = dict(getattr(args, "_verifier_profile_contract", {}) or {})
        config.reward_threshold = float(
            contract.get("threshold")
            if contract.get("threshold") is not None
            else contract.get("minimum", config.reward_threshold)
        )

    # Run training
    signal_session = _build_training_signal_session(
        args,
        trainer="audio",
        output_dir=config.output_dir,
        total_boundaries=config.num_cycles,
        reward_threshold=config.reward_threshold,
    )
    trainer = AudioRAFTTrainer(config, signal_sink=signal_session)
    if profile_verifier is not None:
        trainer.verifier = profile_verifier
    try:
        results = trainer.train(
            args.dataset,
            resume_from_cycle=getattr(args, "resume_from_cycle", 0),
            limit=getattr(args, "limit", None),
        )
    except ValueError as e:
        print(f"{RED}Training error: {e}{NC}")
        sys.exit(2)
    summary = getattr(trainer, "training_summary", {})
    total_steps = int(summary.get("total_train_steps_executed", 0))
    final_loss = summary.get("final_train_loss")
    _enforce_training_outcome_or_exit("audio", summary)

    print(f"\n{GREEN}Training complete!{NC}")
    print(f"Final model saved to: {args.output}")
    if summary.get("final_model_path"):
        print(f"Final model: {summary['final_model_path']}")
    _print_training_run_metadata(summary)
    print(f"Train steps executed: {total_steps}")
    if isinstance(final_loss, (int, float)):
        print(f"Final train loss: {final_loss:.4f}")
    _finalize_managed_training_replay(args, "audio", args.output, config, summary)

    print("\nUsage:")
    print("  halo-forge vlm train --dataset textvqa --model Qwen/Qwen2-VL-7B-Instruct --seed 42")
    print("  halo-forge vlm benchmark --dataset docvqa --model path/to/model")


def main():
    if sys.version_info >= (3, 14):
        print(
            "halo-forge supports Python >=3.10,<3.14. "
            f"Current interpreter is {sys.version.split()[0]}. "
            "Create a Python 3.10-3.13 environment and rerun the command.",
            file=sys.stderr,
        )
        sys.exit(2)
    parser = argparse.ArgumentParser(
        prog="halo-forge",
        description="Multi-backend RLVR training framework for AMD ROCm, Apple Silicon, and CUDA",
    )
    from halo_forge.version import version_line

    parser.add_argument("--version", action="version", version=version_line())

    # Global flags
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress terminal output (logs still written to file)",
    )
    # Compute accelerator override. Distinct from the per-subcommand `--backend`
    # flag in `data generate` (which selects the LLM API: deepseek/anthropic/...).
    # Default: auto-detect via halo_forge.backend.get_backend().
    # `mlx` requires the `[mlx]` extra and currently only powers inference paths.
    parser.add_argument(
        "--accelerator",
        choices=["auto", "rocm", "rocm_gfx1151", "cuda", "mps", "mlx", "cpu"],
        default="auto",
        help='Compute accelerator to target. "auto" (default) detects ROCm/CUDA/MPS/CPU; '
        'pass "mlx" explicitly to use Apple MLX for supported training and inference paths. '
        "Sets HALOFORGE_BACKEND for downstream code.",
    )

    def add_apple_runtime_flags(train_parser, *, neural_accelerators: bool = True):
        train_parser.add_argument(
            "--no-caffeinate",
            action="store_true",
            help="Dashboard round-trip flag: opt out of macOS caffeinate wrapping for launched training jobs",
        )
        if neural_accelerators:
            train_parser.add_argument(
                "--enable-neural-accelerators",
                action="store_true",
                help="Annotate and validate experimental Apple M5+ Neural Accelerator opt-in (no kernel routing yet)",
            )

    def add_dataset_training_flags(train_parser, *, consumes_verifier=False):
        train_parser.add_argument(
            "--training-plan-revision",
            help="Exact confirmed V18 training-plan revision; replaces normal low-level defaults",
        )
        train_parser.add_argument(
            "--runtime-profile-revision",
            help="Exact qualified managed runtime revision for this launch",
        )
        train_parser.add_argument(
            "--dataset-version",
            help="Dataset Lab version id (train-role shorthand; takes precedence over a manual dataset)",
        )
        train_parser.add_argument(
            "--dataset-binding",
            action="append",
            default=[],
            metavar="ROLE=VERSION:SPLIT",
            help="Repeatable explicit Dataset Lab binding (train, validation, test, or canary)",
        )
        if consumes_verifier:
            train_parser.add_argument(
                "--verifier-profile-revision",
                help="Exact qualified Verifier Reliability profile revision",
            )
            train_parser.add_argument(
                "--reward-system-revision",
                help="Immutable monitored reward-system revision (takes precedence over raw verifier fields)",
            )
            train_parser.add_argument(
                "--reward-audit-protocol-revision",
                help="Immutable same-output capture protocol revision",
            )
            train_parser.add_argument(
                "--reward-integrity-profile-revision",
                help="Immutable pass/warn/fail integrity-policy revision",
            )
            train_parser.add_argument(
                "--reward-development-suite-revision",
                help=(
                    "Optional development suite pinned for independent quality tracking; "
                    "it is not scored by the same-output integrity audit"
                ),
            )
            train_parser.add_argument(
                "--reward-audit-boundary",
                action="append",
                default=[],
                metavar="STEP|CYCLE|FINAL",
                help="Repeatable selected audit boundary; audited launches use managed execution",
            )
            train_parser.add_argument(
                "--wait",
                action="store_true",
                help="Wait for a managed audited launch to reach a terminal or review state",
            )

    subparsers = parser.add_subparsers(dest="command", required=True)

    from halo_forge.verifier_cli import add_verifier_parser
    from halo_forge.reward_cli import add_reward_parser
    from halo_forge.lab_v11_v15.cli import (
        add_future_lab_parsers,
        add_ground_parser,
    )
    from halo_forge.product_cli import add_product_v17_parsers
    from halo_forge.training_plan.cli import add_training_plan_parser
    from halo_forge.managed_runtime.cli import add_managed_runtime_parsers

    add_verifier_parser(subparsers)
    add_reward_parser(subparsers)
    add_future_lab_parsers(subparsers)
    add_training_plan_parser(subparsers)
    add_managed_runtime_parsers(subparsers)

    # config command
    config_parser = subparsers.add_parser("config", help="Configuration utilities")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)

    # config validate
    config_validate_parser = config_subparsers.add_parser("validate", help="Validate config file")
    config_validate_parser.add_argument("config", help="Path to config file")
    config_validate_parser.add_argument(
        "--type",
        "-t",
        choices=["raft", "sft", "auto"],
        default="auto",
        help="Config type (auto-detected from filename if not specified)",
    )
    config_validate_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show config contents"
    )

    # data command
    data_parser = subparsers.add_parser("data", help="Data preparation")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    add_ground_parser(data_subparsers)
    add_product_v17_parsers(subparsers, data_subparsers)

    # data prepare
    prepare_parser = data_subparsers.add_parser("prepare", help="Prepare public dataset")
    prepare_parser.add_argument("--dataset", "-d", help="Dataset name")
    prepare_parser.add_argument("--output", "-o", help="Output file path")
    prepare_parser.add_argument("--template", default="qwen", help="Chat template")
    prepare_parser.add_argument("--system-prompt", help="Override system prompt")
    prepare_parser.add_argument("--list", action="store_true", help="List available datasets")

    # data generate
    generate_parser = data_subparsers.add_parser("generate", help="Generate with LLM")
    generate_parser.add_argument("--topic", "-t", help="Topic name")
    generate_parser.add_argument("--backend", "-b", default="deepseek", help="LLM backend")
    generate_parser.add_argument("--model", help="Model name for backend")
    generate_parser.add_argument("--output", "-o", help="Output file path")
    generate_parser.add_argument("--template", default="qwen", help="Chat template")
    generate_parser.add_argument("--list", action="store_true", help="List available topics")

    # data validate
    validate_parser = data_subparsers.add_parser("validate", help="Validate dataset format")
    validate_parser.add_argument("file", help="Path to JSONL file to validate")
    validate_parser.add_argument(
        "--preview", "-p", action="store_true", help="Show preview of examples"
    )

    # data synthesize (Track D1) — teacher → verifier → filter pipeline.
    synth_parser = data_subparsers.add_parser(
        "synthesize",
        help="Generate synthetic training data from prompts via a teacher model + verifier filter",
    )
    synth_parser.add_argument(
        "--seeds", "-i", required=True, help="JSONL or text file of seed prompts (one per line)"
    )
    synth_parser.add_argument("--output", "-o", required=True, help="Output JSONL path")
    synth_parser.add_argument(
        "--teacher-model",
        default="default",
        help="Model name for the OpenAI-compatible teacher endpoint",
    )
    synth_parser.add_argument(
        "--base-url",
        help="Teacher endpoint base URL (default: http://127.0.0.1:8001/v1 — "
        "a local halo-forge serve process)",
    )
    synth_parser.add_argument(
        "--api-key", help="Teacher endpoint API key (env: HALOFORGE_TEACHER_API_KEY)"
    )
    synth_parser.add_argument(
        "--system-prompt", help="System message prepended to every teacher call"
    )
    synth_parser.add_argument(
        "--verifier",
        default="json_structure",
        help="V1 verifier short name to score completions "
        "(execution, llm_judge, bleu, json_schema, regex_format, ...)",
    )
    synth_parser.add_argument(
        "--verifier-profile-revision",
        help="Use an immutable qualified verifier revision instead of raw verifier fields",
    )
    synth_parser.add_argument(
        "--n-per-prompt",
        type=int,
        default=1,
        help="Completions sampled per prompt (>=2 required for --kind preference)",
    )
    synth_parser.add_argument(
        "--threshold", type=float, default=0.5, help="Reward threshold for acceptance (default 0.5)"
    )
    synth_parser.add_argument(
        "--kind",
        default="sft",
        choices=["sft", "preference"],
        help="sft → {prompt, completion}; preference → {prompt, chosen, rejected}",
    )
    synth_parser.add_argument(
        "--max-tokens", type=int, default=512, help="Teacher max_tokens per call (default 512)"
    )
    synth_parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Teacher sampling temperature (default 0.8 for diverse generation)",
    )

    # data score (Track D3) — heuristic quality scoring + filter.
    score_parser = data_subparsers.add_parser(
        "score",
        help="Score JSONL records by heuristic quality and filter by threshold / top-K%%",
    )
    score_parser.add_argument("--input", "-i", required=True)
    score_parser.add_argument("--output", "-o", required=True)
    score_parser.add_argument(
        "--verifier-profile-revision",
        help="Optional immutable verifier revision for verifier-backed scoring",
    )
    score_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Composite score below which rows are dropped (default 0.5)",
    )
    score_parser.add_argument(
        "--top-k-pct",
        type=float,
        help="Keep top K%% by score instead of using --threshold (e.g. 0.5 = top 50%%)",
    )

    # data dedup (Track D2)
    dedup_parser = data_subparsers.add_parser(
        "dedup",
        help="Deduplicate a JSONL dataset (exact / fuzzy MinHash)",
    )
    dedup_parser.add_argument("--input", "-i", required=True, help="Input JSONL path")
    dedup_parser.add_argument("--output", "-o", required=True, help="Output JSONL path (deduped)")
    dedup_parser.add_argument(
        "--method",
        default="exact",
        choices=["exact", "fuzzy"],
        help="exact = SHA256 over normalized text; fuzzy = MinHash + LSH",
    )
    dedup_parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="[fuzzy] Jaccard similarity threshold (default 0.85)",
    )
    dedup_parser.add_argument(
        "--key", default="text", help='Field name when records are dicts (default "text")'
    )
    dedup_parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Skip the lowercase-and-trim normalization step",
    )
    dedup_parser.add_argument(
        "--num-perm", type=int, default=128, help="[fuzzy] MinHash permutations (default 128)"
    )
    dedup_parser.add_argument(
        "--shingle-n", type=int, default=5, help="[fuzzy] Word n-gram shingle size (default 5)"
    )

    # Dataset Lab — catalog, immutable versions, recipes, and jobs.
    def add_dataset_lab_root(parser):
        parser.add_argument(
            "--root",
            help="Dataset Lab storage root (default: ~/.halo-forge/datasets)",
        )
        parser.add_argument("--database", help="Override the Halo Forge SQLite path")
        parser.add_argument("--json", action="store_true", help="Emit JSON")

    data_scenarios_parser = data_subparsers.add_parser(
        "scenarios", help="Browse verified own-data training scenarios and examples"
    )
    data_scenarios_actions = data_scenarios_parser.add_subparsers(
        dest="scenarios_action", required=True
    )
    data_scenarios_list = data_scenarios_actions.add_parser(
        "list", help="List guided training scenarios"
    )
    add_dataset_lab_root(data_scenarios_list)
    data_scenarios_list.add_argument(
        "--include-unavailable",
        action="store_true",
        help="Include scenarios without a verified trainer contract",
    )
    data_scenarios_list.add_argument(
        "--modality", choices=["text", "image", "audio"], help="Filter by modality"
    )
    data_scenarios_show = data_scenarios_actions.add_parser(
        "show", help="Show one scenario, its mappings, and its examples"
    )
    add_dataset_lab_root(data_scenarios_show)
    data_scenarios_show.add_argument("scenario", help="Scenario id or immutable revision id")
    data_scenarios_template = data_scenarios_actions.add_parser(
        "template", help="Write or print a working scenario fixture"
    )
    add_dataset_lab_root(data_scenarios_template)
    data_scenarios_template.add_argument(
        "scenario", help="Scenario id or immutable revision id"
    )
    data_scenarios_template.add_argument("--example", help="Specific example id")
    data_scenarios_template.add_argument(
        "--output", "-o", help="Output file or directory; omit to print the JSONL fixture"
    )
    data_scenarios_advise = data_scenarios_actions.add_parser(
        "advise",
        help="Explain which verified scenarios fit a goal and source shape",
    )
    add_dataset_lab_root(data_scenarios_advise)
    data_scenarios_advise.add_argument(
        "--goal",
        default="",
        help="Plain-language description of what the model should learn",
    )
    data_scenarios_advise.add_argument(
        "--modality",
        choices=["text", "image", "audio"],
        help="Known source modality",
    )
    data_scenarios_advise.add_argument(
        "--source-layout",
        help="Known layout such as jsonl, csv, markdown, pdf, or docx",
    )
    data_scenarios_advise.add_argument(
        "--field",
        action="append",
        default=[],
        help="Source field or comma-separated field names (repeatable)",
    )
    data_scenarios_advise.add_argument(
        "--include-unavailable",
        action="store_true",
        help="Include close matches that are unavailable on the active runtime",
    )

    def add_guided_source_arguments(parser, *, allow_managed: bool = False):
        parser.add_argument("--path", help="Local dataset file or folder on the workstation")
        parser.add_argument("--hf-id", help="Hugging Face dataset repository id")
        parser.add_argument("--import-id", help="Resume an existing persistent import")
        parser.add_argument("--name", help="Display name for a new import")
        parser.add_argument("--config", help="Hugging Face dataset configuration")
        parser.add_argument("--split", default="train", help="Hugging Face split")
        parser.add_argument("--revision", help="Pinned Hugging Face revision")
        parser.add_argument(
            "--scenario", help="Optional scenario id or immutable revision to test first"
        )
        parser.add_argument(
            "--force", action="store_true", help="Retry a failed or interrupted inspection"
        )
        parser.add_argument(
            "--capacity-override-reason",
            help=(
                "Reviewed reason for proceeding when the import would cross the "
                "workstation disk reserve"
            ),
        )
        if allow_managed:
            parser.add_argument(
                "--managed",
                action="store_true",
                help="Copy a local source into checksummed managed storage",
            )

    data_inspect_parser = data_subparsers.add_parser(
        "inspect", help="Persistently inspect and infer a local or Hugging Face source"
    )
    add_dataset_lab_root(data_inspect_parser)
    add_guided_source_arguments(data_inspect_parser)

    data_import_parser = data_subparsers.add_parser(
        "import", help="Create a persistent own-data import and inspect it"
    )
    add_dataset_lab_root(data_import_parser)
    add_guided_source_arguments(data_import_parser, allow_managed=True)
    data_import_parser.add_argument(
        "--no-inspect",
        action="store_true",
        help="Create or upload the import without starting inspection",
    )

    data_extract_parser = data_subparsers.add_parser(
        "extract",
        help="Extract canonical documents from text, Markdown, HTML, PDF, DOCX, or structured rows",
    )
    add_dataset_lab_root(data_extract_parser)
    data_extract_parser.add_argument("--path", help="Document file or directory")
    data_extract_parser.add_argument("--import-id", help="Persistent import source")
    data_extract_parser.add_argument("--source-id", help="Registered dataset source")
    data_extract_parser.add_argument(
        "--extraction", help="Existing extraction id to show or manage"
    )
    data_extract_parser.add_argument(
        "--list", action="store_true", help="List extraction records"
    )
    data_extract_parser.add_argument("--status", help="Filter listed extractions")
    data_extract_parser.add_argument(
        "--preview", action="store_true", help="Preview extracted documents"
    )
    data_extract_parser.add_argument(
        "--verify", action="store_true", help="Verify published bundle checksums"
    )
    data_extract_parser.add_argument(
        "--cancel", action="store_true", help="Cancel the selected extraction"
    )
    data_extract_parser.add_argument(
        "--retry", action="store_true", help="Retry the selected extraction"
    )
    data_extract_parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Return after durable scheduling instead of executing locally",
    )
    data_extract_parser.add_argument(
        "--text-column",
        action="append",
        default=[],
        help="Structured-source text field (repeatable)",
    )
    data_extract_parser.add_argument("--title-column")
    data_extract_parser.add_argument("--id-column")
    data_extract_parser.add_argument(
        "--metadata-column", action="append", default=[]
    )
    data_extract_parser.add_argument("--min-text-chars", type=int, default=1)
    data_extract_parser.add_argument("--include-hidden", action="store_true")
    data_extract_parser.add_argument(
        "--single-pdf-document",
        action="store_true",
        help="Combine a PDF text layer instead of emitting one document per page",
    )
    data_extract_parser.add_argument("--limit", type=int, default=20)
    data_extract_parser.add_argument("--offset", type=int, default=0)
    data_extract_parser.add_argument(
        "--no-text", action="store_true", help="Omit full text from previews"
    )

    data_corpus_profile_parser = data_subparsers.add_parser(
        "corpus-profile",
        help="Show exact document, character, byte, paragraph, and split statistics",
    )
    add_dataset_lab_root(data_corpus_profile_parser)
    data_corpus_profile_parser.add_argument("version", help="Corpus version id")

    data_add_parser = data_subparsers.add_parser(
        "add", help="Register a local or pinned Hugging Face dataset source"
    )
    add_dataset_lab_root(data_add_parser)
    data_add_parser.add_argument("--name", required=True)
    data_add_parser.add_argument("--path", help="Local JSON/JSONL/CSV/Parquet file or folder")
    data_add_parser.add_argument("--hf-id", help="Hugging Face dataset repository id")
    data_add_parser.add_argument("--config", help="Hugging Face dataset configuration")
    data_add_parser.add_argument("--split", default="train")
    data_add_parser.add_argument("--revision", help="Pinned Hugging Face revision")
    data_add_parser.add_argument(
        "--scenario",
        help="Verified guided scenario id or immutable revision id",
    )
    data_add_parser.add_argument(
        "--kind",
        choices=[
            "sft",
            "chat",
            "preference",
            "prompt",
            "rlvr",
            "reasoning",
            "tool",
            "vlm",
            "audio",
            "corpus",
        ],
        help="Canonical record kind (inferred when omitted)",
    )
    data_add_parser.add_argument(
        "--modality",
        choices=["text", "image", "vlm", "audio"],
        help="Dataset modality (inferred when omitted)",
    )
    data_add_parser.add_argument("--mapping", help="Field mapping as a JSON object")
    data_add_parser.add_argument(
        "--map",
        dest="map_fields",
        action="append",
        default=[],
        metavar="TARGET=SOURCE",
        help="Map a canonical field to a source field (repeatable)",
    )
    data_add_parser.add_argument(
        "--accept-recommended",
        action="store_true",
        help="Explicitly accept the recommended detected scenario and mapping",
    )

    data_list_parser = data_subparsers.add_parser("list", help="List Dataset Lab sources")
    add_dataset_lab_root(data_list_parser)

    data_show_parser = data_subparsers.add_parser("show", help="Show one Dataset Lab source")
    add_dataset_lab_root(data_show_parser)
    data_show_parser.add_argument("dataset", help="Dataset/source id")
    data_show_parser.add_argument("--profile", action="store_true", help="Include a fresh profile")

    data_build_parser = data_subparsers.add_parser(
        "build", help="Build an immutable Dataset Lab version from a recipe"
    )
    add_dataset_lab_root(data_build_parser)
    data_build_parser.add_argument("dataset", help="Dataset/source id")
    data_build_recipe = data_build_parser.add_mutually_exclusive_group(required=True)
    data_build_recipe.add_argument("--recipe", help="YAML or JSON recipe path")
    data_build_recipe.add_argument(
        "--recommended-recipe",
        action="store_true",
        help="Use the explicitly confirmed guided preparation plan",
    )
    data_build_parser.add_argument("--materialize-assets", action="store_true")

    data_versions_parser = data_subparsers.add_parser("versions", help="List dataset versions")
    add_dataset_lab_root(data_versions_parser)
    data_versions_parser.add_argument("dataset", nargs="?", help="Optional dataset id filter")

    data_preview_parser = data_subparsers.add_parser(
        "preview", help="Preview canonical source records"
    )
    add_dataset_lab_root(data_preview_parser)
    data_preview_parser.add_argument("dataset", help="Dataset/source id")
    data_preview_parser.add_argument("--offset", type=int, default=0)
    data_preview_parser.add_argument("--limit", type=int, default=20)
    data_preview_parser.add_argument("--split", default="train", help="Version split to preview")

    data_export_parser = data_subparsers.add_parser("export", help="Export a dataset version")
    add_dataset_lab_root(data_export_parser)
    data_export_parser.add_argument("version", help="Dataset version id")
    data_export_parser.add_argument("--output", "-o", required=True)
    data_export_parser.add_argument("--split", help="Export only one split")
    data_export_parser.add_argument(
        "--format", choices=["jsonl", "csv", "parquet"], default="jsonl"
    )

    data_materialize_parser = data_subparsers.add_parser(
        "materialize", help="Copy referenced media assets into managed storage"
    )
    add_dataset_lab_root(data_materialize_parser)
    data_materialize_parser.add_argument("version", help="Dataset version id")

    data_jobs_parser = data_subparsers.add_parser("jobs", help="List or manage Dataset Lab jobs")
    add_dataset_lab_root(data_jobs_parser)
    data_jobs_parser.add_argument("--job", help="Show one job")
    data_jobs_parser.add_argument("--cancel", help="Cancel a queued/running job")
    data_jobs_parser.add_argument("--retry", help="Retry a failed/interrupted job")

    data_render_parser = data_subparsers.add_parser(
        "render", help="Render a content-addressed trainer dataset artifact"
    )
    add_dataset_lab_root(data_render_parser)
    data_render_parser.add_argument("version", nargs="?", help="Train-role version shorthand")
    data_render_parser.add_argument("--split", default="train")
    data_render_parser.add_argument(
        "--binding", action="append", default=[], metavar="ROLE=VERSION:SPLIT"
    )
    data_render_parser.add_argument(
        "--trainer",
        required=True,
        choices=[
            "sft",
            "raft",
            "dpo",
            "orpo",
            "rm",
            "grpo",
            "vlm",
            "audio",
            "reasoning",
            "agentic",
            "cpt",
        ],
    )
    data_render_parser.add_argument("--adapter")
    data_render_parser.add_argument("--model")
    data_render_parser.add_argument("--tokenizer-revision")
    data_render_parser.add_argument("--model-revision")
    data_render_parser.add_argument("--model-hash")
    data_render_parser.add_argument("--tokenizer-hash")
    data_render_parser.add_argument("--validation-fraction", type=float, default=0.05)
    data_render_parser.add_argument("--seed", type=int, default=42)
    data_render_parser.add_argument("--max-seq-length", type=int, default=2048)
    data_render_parser.add_argument(
        "--packing", default="paragraph_eos_non_overlap_v1"
    )
    data_render_parser.add_argument(
        "--budget-mode", choices=["tokens", "passes"], default="passes"
    )
    data_render_parser.add_argument("--target-tokens", type=_positive_int_arg)
    data_render_parser.add_argument("--corpus-passes", type=float, default=1.0)
    data_render_parser.add_argument("--effective-batch-size", type=int, default=1)

    data_compare_parser = data_subparsers.add_parser(
        "compare", help="Compare two immutable dataset versions by record identity"
    )
    add_dataset_lab_root(data_compare_parser)
    data_compare_parser.add_argument("left")
    data_compare_parser.add_argument("right")

    data_mine_parser = data_subparsers.add_parser(
        "mine", help="Preview or build a reviewed child version from evaluation failures"
    )
    add_dataset_lab_root(data_mine_parser)
    data_mine_parser.add_argument(
        "--candidate", required=True, help="Completed candidate evaluation id"
    )
    data_mine_parser.add_argument("--base", help="Completed base evaluation id")
    data_mine_parser.add_argument(
        "--selector",
        choices=["candidate_failure", "regression", "improvement", "verifier_disagreement"],
        default="candidate_failure",
    )
    data_mine_parser.add_argument("--task")
    data_mine_parser.add_argument("--category")
    data_mine_parser.add_argument("--failure-reason")
    data_mine_parser.add_argument("--min-score", type=float)
    data_mine_parser.add_argument("--max-score", type=float)
    data_mine_parser.add_argument("--min-reward", type=float)
    data_mine_parser.add_argument("--max-reward", type=float)
    data_mine_parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Explicit record/selection id to exclude (repeatable)",
    )
    data_mine_parser.add_argument("--dataset", help="Target dataset id; omit for preview only")
    data_mine_parser.add_argument(
        "--parent-version", help="Parent version id; omit for preview only"
    )
    data_mine_parser.add_argument("--target-split", default="train")
    data_mine_parser.add_argument("--mode", choices=["append", "replace"], default="append")
    data_mine_parser.add_argument("--materialize-assets", action="store_true", default=None)

    # Human Feedback and Active Data Studio.
    review_parser = subparsers.add_parser(
        "review", help="Acquire, review, label, and publish training evidence"
    )
    review_subparsers = review_parser.add_subparsers(dest="review_command", required=True)

    def add_review_runtime(command_parser):
        command_parser.add_argument("--database", help="Override the Halo Forge SQLite path")
        command_parser.add_argument(
            "--root", help="Review Studio storage root (default: ~/.halo-forge/reviews)"
        )
        command_parser.add_argument("--json", action="store_true", help="Emit JSON")

    review_capabilities = review_subparsers.add_parser(
        "capabilities", help="Show supported review tasks, strategies, and output adapters"
    )
    add_review_runtime(review_capabilities)

    review_schema = review_subparsers.add_parser("schema", help="Annotation schemas")
    review_schema_actions = review_schema.add_subparsers(dest="schema_action", required=True)
    review_schema_list = review_schema_actions.add_parser("list")
    add_review_runtime(review_schema_list)
    review_schema_list.add_argument("--limit", type=int, default=100)
    review_schema_list.add_argument("--offset", type=int, default=0)
    review_schema_show = review_schema_actions.add_parser("show")
    add_review_runtime(review_schema_show)
    review_schema_show.add_argument("identifier")
    review_schema_show.add_argument(
        "--revision", action="store_true", help="Treat identifier as an immutable revision id"
    )

    def add_review_schema_fields(value_parser, *, creating: bool):
        if creating:
            value_parser.add_argument("--schema-id")
            value_parser.add_argument("--name", required=True)
            value_parser.add_argument("--description")
        else:
            value_parser.add_argument("--name")
        value_parser.add_argument(
            "--modality",
            choices=["text", "preference", "tool", "vlm", "audio"],
            required=creating,
        )
        value_parser.add_argument(
            "--task-type",
            choices=[
                "binary",
                "categorical",
                "multi_label",
                "scalar",
                "text_correction",
                "structured_correction",
                "pairwise",
                "ranking",
            ],
            required=creating,
        )
        value_parser.add_argument(
            "--definition", required=True, help="JSON/YAML path or inline annotation definition"
        )

    review_schema_validate = review_schema_actions.add_parser("validate")
    add_review_runtime(review_schema_validate)
    add_review_schema_fields(review_schema_validate, creating=False)
    review_schema_create = review_schema_actions.add_parser("create")
    add_review_runtime(review_schema_create)
    add_review_schema_fields(review_schema_create, creating=True)
    review_schema_revise = review_schema_actions.add_parser("revise")
    add_review_runtime(review_schema_revise)
    review_schema_revise.add_argument("schema_id")
    add_review_schema_fields(review_schema_revise, creating=False)

    review_acquire = review_subparsers.add_parser("acquire", help="Acquisition proposals")
    review_acquire_actions = review_acquire.add_subparsers(dest="acquire_action", required=True)
    review_acquire_list = review_acquire_actions.add_parser("list")
    add_review_runtime(review_acquire_list)
    review_acquire_list.add_argument("--status")
    review_acquire_list.add_argument("--limit", type=int, default=100)
    review_acquire_list.add_argument("--offset", type=int, default=0)
    review_acquire_create = review_acquire_actions.add_parser("create")
    add_review_runtime(review_acquire_create)
    review_acquire_create.add_argument(
        "--spec", help="Full JSON/YAML acquisition spec; overrides simple flags"
    )
    review_acquire_create.add_argument("--records", help="JSON/JSONL records path or inline array")
    review_acquire_create.add_argument(
        "--source-kind",
        choices=[
            "evaluation",
            "evaluation_comparison",
            "verifier_calibration",
            "run_samples",
            "dataset_version",
            "playground_session",
            "jsonl",
        ],
    )
    review_acquire_create.add_argument("--source-ref")
    review_acquire_create.add_argument("--source-split", default="train")
    review_acquire_create.add_argument("--base-evaluation")
    review_acquire_create.add_argument("--candidate-evaluation")
    review_acquire_create.add_argument(
        "--failure-selector",
        choices=[
            "false_accept",
            "false_reject",
            "high_confidence_disagreement",
            "repeat_instability",
            "order_flip",
            "ranking_inversion",
            "threshold_adjacent",
            "parser_runtime",
            "subgroup",
            "chain_component",
        ],
        help="Reviewed failure selector for a verifier_calibration source",
    )
    review_acquire_create.add_argument(
        "--selector-options",
        help="JSON/YAML path or inline options for the verifier failure selector",
    )
    review_acquire_create.add_argument(
        "--strategy",
        action="append",
        default=[],
        help="Strategy name or inline JSON/YAML; repeat for ordered quota strata",
    )
    review_acquire_create.add_argument("--quota", type=int)
    review_acquire_create.add_argument(
        "--embedding-revision",
        help=(
            "Pinned diversity model as modality:model@revision; missing vectors are "
            "generated by durable workstation work"
        ),
    )
    review_acquire_create.add_argument("--score-direction", choices=["maximize", "minimize"])
    review_acquire_create.add_argument("--seed", type=int, default=0)
    review_acquire_create.add_argument("--name")
    review_acquire_show = review_acquire_actions.add_parser("show")
    add_review_runtime(review_acquire_show)
    review_acquire_show.add_argument("batch_id")
    review_acquire_show.add_argument("--candidates", action="store_true")
    review_acquire_show.add_argument("--limit", type=int, default=100)
    review_acquire_show.add_argument("--offset", type=int, default=0)
    for action_name in ("cancel", "retry"):
        value_parser = review_acquire_actions.add_parser(action_name)
        add_review_runtime(value_parser)
        value_parser.add_argument("batch_id")

    review_queue = review_subparsers.add_parser("queue", help="Review queues")
    review_queue_actions = review_queue.add_subparsers(dest="queue_action", required=True)
    review_queue_list = review_queue_actions.add_parser("list")
    add_review_runtime(review_queue_list)
    review_queue_list.add_argument("--status")
    review_queue_list.add_argument("--limit", type=int, default=100)
    review_queue_list.add_argument("--offset", type=int, default=0)
    review_queue_show = review_queue_actions.add_parser("show")
    add_review_runtime(review_queue_show)
    review_queue_show.add_argument("queue_id")
    review_queue_create = review_queue_actions.add_parser("create")
    add_review_runtime(review_queue_create)
    review_queue_create.add_argument("--batch", required=True)
    review_queue_create.add_argument("--schema", required=True)
    review_queue_create.add_argument("--name")
    review_queue_create.add_argument("--policy", help="JSON/YAML review policy")
    review_queue_clone = review_queue_actions.add_parser("clone")
    add_review_runtime(review_queue_clone)
    review_queue_clone.add_argument("queue_id")
    review_queue_clone.add_argument("--batch")
    review_queue_clone.add_argument("--schema")
    review_queue_clone.add_argument("--name")
    review_queue_clone.add_argument("--policy")
    for action_name in ("pause", "resume", "archive", "start-second-pass"):
        value_parser = review_queue_actions.add_parser(action_name)
        add_review_runtime(value_parser)
        value_parser.add_argument("queue_id")
        value_parser.add_argument("--reason")

    review_items = review_subparsers.add_parser("items", help="List review items")
    add_review_runtime(review_items)
    review_items.add_argument("queue_id")
    review_items.add_argument("--status")
    review_items.add_argument("--pass-number", type=int)
    review_items.add_argument("--limit", type=int, default=100)
    review_items.add_argument("--offset", type=int, default=0)
    review_item = review_subparsers.add_parser("item", help="Show one review item")
    add_review_runtime(review_item)
    review_item.add_argument("item_id")

    def add_review_event_fields(value_parser):
        add_review_runtime(value_parser)
        value_parser.add_argument("item_id")
        value_parser.add_argument("--idempotency-key")
        value_parser.add_argument("--expected-active-event")
        value_parser.add_argument("--pass-number", type=int)

    review_submit = review_subparsers.add_parser("submit", help="Submit a review decision")
    add_review_event_fields(review_submit)
    review_submit.add_argument("--label", required=True, help="JSON/YAML label")
    review_submit.add_argument("--event-type", default="label")
    review_correct = review_subparsers.add_parser("correct", help="Supersede a decision")
    add_review_event_fields(review_correct)
    review_correct.add_argument("--label", required=True)
    review_correct.add_argument("--reason", required=True)
    review_correct.add_argument("--supersedes-event", required=True)
    for command_name in ("exclude", "flag"):
        value_parser = review_subparsers.add_parser(command_name)
        add_review_event_fields(value_parser)
        value_parser.add_argument("--reason", required=True)
    review_adjudicate = review_subparsers.add_parser(
        "adjudicate", help="Resolve a two-pass disagreement"
    )
    add_review_event_fields(review_adjudicate)
    review_adjudicate.add_argument("--label", required=True)
    review_adjudicate.add_argument("--reason", required=True)

    review_suggestions = review_subparsers.add_parser(
        "suggestions", help="Optional model suggestions"
    )
    review_suggestion_actions = review_suggestions.add_subparsers(
        dest="suggestions_action", required=True
    )
    review_suggestion_generate = review_suggestion_actions.add_parser("generate")
    add_review_runtime(review_suggestion_generate)
    review_suggestion_generate.add_argument("item_id")
    review_suggestion_generate.add_argument("--provider", default="openai_compatible")
    review_suggestion_generate.add_argument("--model", required=True)
    review_suggestion_generate.add_argument("--prompt")
    review_suggestion_generate.add_argument("--pass-number", type=int, choices=[1, 2])
    review_suggestion_generate.add_argument(
        "--output", help="Persist an existing JSON/YAML suggestion instead of calling a provider"
    )
    review_suggestion_generate.add_argument("--parameters")
    review_suggestion_generate.add_argument(
        "--verifier-profile-revision",
        help="Exact verifier revision used to check the suggestion",
    )
    review_suggestion_show = review_suggestion_actions.add_parser("show")
    add_review_runtime(review_suggestion_show)
    review_suggestion_show.add_argument("item_id")
    review_suggestion_show.add_argument("--pass-number", type=int, choices=[1, 2])
    review_suggestion_show.add_argument("--limit", type=int, default=100)
    review_suggestion_show.add_argument("--offset", type=int, default=0)

    review_stats = review_subparsers.add_parser("stats", help="Queue coverage and agreement")
    add_review_runtime(review_stats)
    review_stats.add_argument("queue_id")

    review_label_set = review_subparsers.add_parser("label-set", help="Immutable label sets")
    review_label_actions = review_label_set.add_subparsers(dest="label_set_action", required=True)
    review_label_list = review_label_actions.add_parser("list")
    add_review_runtime(review_label_list)
    review_label_list.add_argument("--limit", type=int, default=100)
    review_label_list.add_argument("--offset", type=int, default=0)
    review_label_show = review_label_actions.add_parser("show")
    add_review_runtime(review_label_show)
    review_label_show.add_argument("identifier")
    review_label_publish = review_label_actions.add_parser("publish")
    add_review_runtime(review_label_publish)
    review_label_publish.add_argument("--queue", required=True)
    review_label_publish.add_argument("--name")
    review_label_publish.add_argument("--adapter")
    review_label_publish.add_argument(
        "--mode", choices=["filter", "replace_by_record_id", "append", "annotate"]
    )
    review_label_verify = review_label_actions.add_parser("verify")
    add_review_runtime(review_label_verify)
    review_label_verify.add_argument("revision_id")

    def add_label_dataset_fields(value_parser, *, build: bool):
        add_review_runtime(value_parser)
        value_parser.add_argument("revision_id")
        value_parser.add_argument("--adapter")
        value_parser.add_argument(
            "--mode", choices=["filter", "replace_by_record_id", "append", "annotate"]
        )
        value_parser.add_argument("--dataset")
        value_parser.add_argument("--parent-version")
        value_parser.add_argument("--target-split", default="train")
        if build:
            value_parser.add_argument("--name")
            value_parser.add_argument("--materialize-assets", action="store_true")

    review_label_preview = review_label_actions.add_parser("preview")
    add_label_dataset_fields(review_label_preview, build=False)
    review_label_build = review_label_actions.add_parser("build-dataset")
    add_label_dataset_fields(review_label_build, build=True)

    # Immutable checkpoint schedules, development-only evidence gates, and
    # reviewed continuation policies shared by dashboard and orchestration.
    checkpoint_policy_parser = subparsers.add_parser(
        "checkpoint-policy",
        help="Create and inspect immutable adaptive-training checkpoint policies",
    )
    checkpoint_policy_subparsers = checkpoint_policy_parser.add_subparsers(
        dest="checkpoint_policy_command", required=True
    )

    def add_checkpoint_policy_runtime_flags(command_parser):
        command_parser.add_argument(
            "--database", help="Override the Halo Forge SQLite database path"
        )
        command_parser.add_argument("--json", action="store_true", help="Emit JSON")

    def add_checkpoint_policy_definition_flags(command_parser):
        command_parser.add_argument(
            "--spec", help="YAML/JSON policy revision path or inline object"
        )
        command_parser.add_argument("--policy-id")
        command_parser.add_argument("--name")
        command_parser.add_argument("--description")
        command_parser.add_argument("--suite-revision", help="Pinned development-suite revision id")
        command_parser.add_argument("--metric", help="Primary development metric")
        command_parser.add_argument("--direction", choices=["maximize", "minimize"])
        command_parser.add_argument(
            "--schedule",
            help="YAML/JSON ordered schedule (final, interval, percentages, or explicit)",
        )
        command_parser.add_argument(
            "--rules", help="JSON array of objective, guardrail, and plateau rules"
        )
        command_parser.add_argument(
            "--automatic-actions",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Allow declared boundary actions without operator review",
        )
        command_parser.add_argument(
            "--capability",
            action="append",
            default=[],
            help="Compatible trainer capability id (repeatable)",
        )

    checkpoint_policy_list = checkpoint_policy_subparsers.add_parser(
        "list", help="List the latest immutable policy revisions"
    )
    add_checkpoint_policy_runtime_flags(checkpoint_policy_list)
    checkpoint_policy_list.add_argument("--trainer", help="Filter by trainer mode/capability")
    checkpoint_policy_list.add_argument("--limit", type=int, default=100)
    checkpoint_policy_list.add_argument("--offset", type=int, default=0)

    checkpoint_policy_show = checkpoint_policy_subparsers.add_parser(
        "show", help="Show a policy revision or the latest revision for a policy"
    )
    add_checkpoint_policy_runtime_flags(checkpoint_policy_show)
    checkpoint_policy_show.add_argument("policy")

    checkpoint_policy_create = checkpoint_policy_subparsers.add_parser(
        "create", help="Create a named policy and immutable revision"
    )
    add_checkpoint_policy_runtime_flags(checkpoint_policy_create)
    add_checkpoint_policy_definition_flags(checkpoint_policy_create)

    checkpoint_policy_validate = checkpoint_policy_subparsers.add_parser(
        "validate", help="Validate and normalize a policy without saving it"
    )
    add_checkpoint_policy_runtime_flags(checkpoint_policy_validate)
    add_checkpoint_policy_definition_flags(checkpoint_policy_validate)

    # Durable repeat/sweep experiment operations. These commands share the
    # same orchestration service and SQLite catalog as the dashboard/API.
    sweep_parser = subparsers.add_parser(
        "sweep", help="Create and operate repeat/sweep experiment groups"
    )
    sweep_subparsers = sweep_parser.add_subparsers(dest="sweep_command", required=True)

    def add_experiment_output_flags(command_parser):
        command_parser.add_argument(
            "--database", help="Override the Halo Forge SQLite database path"
        )
        command_parser.add_argument("--json", action="store_true", help="Emit JSON")

    sweep_create_parser = sweep_subparsers.add_parser(
        "create", help="Materialize and queue a repeat or parameter sweep"
    )
    add_experiment_output_flags(sweep_create_parser)
    sweep_create_parser.add_argument(
        "--spec", help="YAML/JSON run-group spec path or inline object"
    )
    sweep_create_parser.add_argument("--name", help="Experiment group name")
    sweep_create_parser.add_argument("--kind", choices=["repeat", "sweep"])
    sweep_create_parser.add_argument("--trainer", help="Trainer mode (sft, dpo, raft, ...)")
    sweep_create_parser.add_argument(
        "--suite", help="Pinned development benchmark suite revision id"
    )
    sweep_create_parser.add_argument(
        "--holdout-suite", help="Optional pinned holdout suite revision id"
    )
    sweep_create_parser.add_argument("--model", help="Pinned base model name/revision")
    sweep_create_parser.add_argument("--backend", help="Training backend (hf, mlx, ...)")
    sweep_create_parser.add_argument("--output-root", help="Managed run output root")
    sweep_create_parser.add_argument(
        "--max-steps",
        type=_positive_int_arg,
        help="Verified step budget for checkpoint-gated HF trainers",
    )
    sweep_create_parser.add_argument(
        "--cycles",
        type=_positive_int_arg,
        help="Verified cycle budget for RAFT and multimodal trainers",
    )
    sweep_create_parser.add_argument(
        "--base-config", help="YAML/JSON base launch config path or inline object"
    )
    sweep_create_parser.add_argument(
        "--dataset-version",
        "--dataset",
        dest="dataset_version",
        help="Dataset Lab version id bound to the train split",
    )
    sweep_create_parser.add_argument(
        "--dataset-binding",
        action="append",
        default=[],
        metavar="ROLE=VERSION:SPLIT",
        help="Repeatable explicit Dataset Lab role binding",
    )
    sweep_create_parser.add_argument(
        "--seeds",
        nargs="+",
        metavar="SEED",
        help="Seed cohort (space- or comma-separated integers)",
    )
    sweep_create_parser.add_argument("--trials", type=int, help="Number of sweep trials")
    sweep_create_parser.add_argument(
        "--search-space", help="YAML/JSON search-space path or inline object"
    )
    sweep_create_parser.add_argument(
        "--sampler",
        choices=["random", "grid"],
        help="Parameter sampler (TPE requires programmatic sampled_params)",
    )
    sweep_create_parser.add_argument("--sampler-seed", type=int)
    sweep_create_parser.add_argument(
        "--pruning", help="YAML/JSON successive-halving policy path or inline object"
    )
    sweep_create_parser.add_argument(
        "--prune",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable synchronous successive halving (off by default)",
    )
    sweep_create_parser.add_argument(
        "--budgets",
        nargs="+",
        metavar="BUDGET",
        help="Increasing successive-halving step/cycle budgets",
    )
    sweep_create_parser.add_argument("--reduction-factor", type=int)
    sweep_create_parser.add_argument("--priority", type=int, default=None)
    sweep_create_parser.add_argument("--max-retries", type=int, default=None)
    sweep_create_parser.add_argument(
        "--checkpoint-policy",
        help="Immutable checkpoint-policy revision id (requires max-steps or cycles)",
    )

    sweep_list_parser = sweep_subparsers.add_parser("list", help="List experiment groups")
    add_experiment_output_flags(sweep_list_parser)
    sweep_list_parser.add_argument("--kind", choices=["repeat", "sweep"])
    sweep_list_parser.add_argument("--status")
    sweep_list_parser.add_argument("--limit", type=int, default=100)

    sweep_show_parser = sweep_subparsers.add_parser("show", help="Show one group and trials")
    add_experiment_output_flags(sweep_show_parser)
    sweep_show_parser.add_argument("group_id")

    sweep_cancel_parser = sweep_subparsers.add_parser("cancel", help="Cancel group work")
    add_experiment_output_flags(sweep_cancel_parser)
    sweep_cancel_parser.add_argument("group_id")

    sweep_resume_parser = sweep_subparsers.add_parser(
        "resume", help="Retry interrupted, failed, or cancelled group work"
    )
    add_experiment_output_flags(sweep_resume_parser)
    sweep_resume_parser.add_argument("group_id")
    sweep_resume_parser.add_argument(
        "--reason", help="Recorded operator reason for resuming paused group work"
    )

    sweep_checkpoints_parser = sweep_subparsers.add_parser(
        "checkpoints", help="Show checkpoint, evaluation, and gate trajectory"
    )
    add_experiment_output_flags(sweep_checkpoints_parser)
    sweep_checkpoints_parser.add_argument("group_id")

    sweep_analyze_parser = sweep_subparsers.add_parser(
        "analyze", help="Create a deterministic matched-seed cohort snapshot"
    )
    add_experiment_output_flags(sweep_analyze_parser)
    sweep_analyze_parser.add_argument("group_id")
    sweep_analyze_parser.add_argument("--baseline", help="Baseline trial/subject id")
    sweep_analyze_parser.add_argument("--practical-delta", type=float, default=0.0)
    sweep_analyze_parser.add_argument("--equivalence-delta", type=float)
    sweep_analyze_parser.add_argument("--confidence", type=float, default=0.95)
    sweep_analyze_parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    sweep_analyze_parser.add_argument("--bootstrap-seed", type=int, default=42)

    sweep_decide_parser = sweep_subparsers.add_parser(
        "decide", help="Record an append-only reviewed research decision"
    )
    add_experiment_output_flags(sweep_decide_parser)
    sweep_decide_parser.add_argument("group_id")
    sweep_decide_parser.add_argument("--analysis", help="Analysis snapshot id (latest by default)")
    sweep_decide_parser.add_argument("--select", required=True, help="Selected trial/subject id")
    sweep_decide_parser.add_argument("--reject", action="append", default=[])
    sweep_decide_parser.add_argument("--exclude", action="append", default=[])
    sweep_decide_parser.add_argument("--rationale", required=True)
    sweep_decide_parser.add_argument("--override-reason")
    sweep_decide_parser.add_argument(
        "--fork", action="store_true", help="Capture a reviewed fork specification"
    )

    sweep_report_parser = sweep_subparsers.add_parser(
        "report", help="Queue an immutable reproducibility evidence bundle"
    )
    add_experiment_output_flags(sweep_report_parser)
    sweep_report_parser.add_argument("group_id")
    sweep_report_parser.add_argument("--analysis", help="Analysis snapshot id (latest by default)")
    sweep_report_parser.add_argument("--decision", help="Reviewed research decision id")
    sweep_report_parser.add_argument(
        "--format",
        action="append",
        default=None,
        choices=["markdown", "html", "json", "csv", "svg"],
    )

    sweep_compare_parser = sweep_subparsers.add_parser(
        "compare", help="Compare best seed-complete cohorts on one suite revision"
    )
    add_experiment_output_flags(sweep_compare_parser)
    sweep_compare_parser.add_argument("left_group_id")
    sweep_compare_parser.add_argument("right_group_id")

    sweep_fork_parser = sweep_subparsers.add_parser(
        "fork-best", help="Emit (without launching) a repeat spec from the best trial"
    )
    add_experiment_output_flags(sweep_fork_parser)
    sweep_fork_parser.add_argument("group_id")
    sweep_fork_parser.add_argument("--name", help="Name for the proposed child group")
    sweep_fork_parser.add_argument(
        "--seeds",
        nargs="+",
        metavar="SEED",
        help="Override child seed cohort (space- or comma-separated)",
    )

    # One unified queue for training/evaluation and other accelerator-heavy work.
    jobs_parser = subparsers.add_parser(
        "jobs", help="Inspect and run the durable workstation work queue"
    )
    jobs_subparsers = jobs_parser.add_subparsers(dest="jobs_command", required=True)

    jobs_list_parser = jobs_subparsers.add_parser("list", help="List queued work")
    add_experiment_output_flags(jobs_list_parser)
    jobs_list_parser.add_argument("--status")
    jobs_list_parser.add_argument("--kind")
    jobs_list_parser.add_argument("--run-id", help="Filter by canonical run id")
    jobs_list_parser.add_argument("--limit", type=int, default=200)
    jobs_list_parser.add_argument("--offset", type=int, default=0)

    jobs_show_parser = jobs_subparsers.add_parser("show", help="Show queue position/blockers")
    add_experiment_output_flags(jobs_show_parser)
    jobs_show_parser.add_argument("work_item_id")

    jobs_cancel_parser = jobs_subparsers.add_parser("cancel", help="Cancel queued/running work")
    add_experiment_output_flags(jobs_cancel_parser)
    jobs_cancel_parser.add_argument("work_item_id")

    jobs_retry_parser = jobs_subparsers.add_parser(
        "retry", help="Retry failed, interrupted, or cancelled work"
    )
    add_experiment_output_flags(jobs_retry_parser)
    jobs_retry_parser.add_argument("work_item_id")

    jobs_worker_parser = jobs_subparsers.add_parser(
        "worker", help="Run the durable workstation queue worker"
    )
    add_experiment_output_flags(jobs_worker_parser)
    jobs_worker_parser.add_argument(
        "--once", action="store_true", help="Run at most one ready work item"
    )
    jobs_worker_parser.add_argument("--poll-interval", type=float, default=0.25)
    jobs_worker_parser.add_argument("--heartbeat-interval", type=float, default=None)
    jobs_worker_parser.add_argument("--terminate-timeout", type=float, default=10.0)
    jobs_worker_parser.add_argument(
        "--recover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recover/adopt interrupted worker state before processing (default: on)",
    )

    # Artifact Studio — one content-addressed library for trained and derived models.
    artifact_parser = subparsers.add_parser(
        "artifact", help="Browse, transform, qualify, promote, serve, and export artifacts"
    )
    artifact_subparsers = artifact_parser.add_subparsers(dest="artifact_command", required=True)

    def add_artifact_runtime_flags(command_parser):
        command_parser.add_argument(
            "--database", help="Override the Halo Forge SQLite database path"
        )
        command_parser.add_argument(
            "--artifact-root",
            "--root",
            dest="artifact_root",
            help="Artifact library root (default: ~/.halo-forge/artifacts)",
        )
        command_parser.add_argument("--json", action="store_true", help="Emit JSON")

    def add_artifact_queue_flags(command_parser):
        command_parser.add_argument("--priority", type=int, default=0)
        command_parser.add_argument("--max-retries", type=int, default=1)

    def add_cleanup_flags(command_parser):
        cleanup_action = command_parser.add_mutually_exclusive_group()
        cleanup_action.add_argument(
            "--apply", metavar="PLAN_ID", help="Apply a previously previewed cleanup plan"
        )
        cleanup_action.add_argument(
            "--restore", metavar="CONTENT_HASH", help="Restore an artifact from seven-day trash"
        )
        cleanup_action.add_argument(
            "--purge", action="store_true", help="Permanently delete expired trash only"
        )
        command_parser.add_argument(
            "--review-note", help="Required operator reason when applying a cleanup plan"
        )

    artifact_list_parser = artifact_subparsers.add_parser(
        "list", help="List artifact occurrences in the local library"
    )
    add_artifact_runtime_flags(artifact_list_parser)
    artifact_list_parser.add_argument("--kind", help="Filter by artifact kind")
    artifact_list_parser.add_argument(
        "--pinned",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Filter to pinned or unpinned occurrences",
    )
    artifact_list_parser.add_argument("--run-id")
    artifact_list_parser.add_argument("--limit", type=int, default=100)
    artifact_list_parser.add_argument("--offset", type=int, default=0)

    artifact_show_parser = artifact_subparsers.add_parser(
        "show", help="Show an occurrence, alias, blob id, or content hash"
    )
    add_artifact_runtime_flags(artifact_show_parser)
    artifact_show_parser.add_argument("artifact")

    artifact_import_parser = artifact_subparsers.add_parser(
        "import", help="Reference or adopt a local checkpoint/model into the library"
    )
    add_artifact_runtime_flags(artifact_import_parser)
    artifact_import_parser.add_argument("source")
    artifact_import_parser.add_argument(
        "--kind",
        required=True,
        help="checkpoint, adapter, final, merged, converted, quantized, or export_bundle",
    )
    artifact_import_parser.add_argument(
        "--format", required=True, help="Source format, such as hf, mlx, gguf, or raw"
    )
    artifact_import_parser.add_argument("--model-id")
    artifact_import_parser.add_argument("--backend", default="local")
    artifact_import_parser.add_argument(
        "--managed",
        action="store_true",
        help="Adopt verified bytes into the managed content-addressed library",
    )
    artifact_import_parser.add_argument("--dtype")
    artifact_import_parser.add_argument("--quantization")
    artifact_import_parser.add_argument("--occurrence-id")
    artifact_import_parser.add_argument("--run-id")
    artifact_import_parser.add_argument("--tokenizer-revision")
    artifact_import_parser.add_argument("--chat-template-hash")
    artifact_import_parser.add_argument(
        "--metadata", help="JSON/YAML object or path with occurrence metadata"
    )

    artifact_lineage_parser = artifact_subparsers.add_parser(
        "lineage", help="Show ordered occurrence and content lineage"
    )
    add_artifact_runtime_flags(artifact_lineage_parser)
    artifact_lineage_parser.add_argument("artifact")

    artifact_verify_parser = artifact_subparsers.add_parser(
        "verify", help="Verify immutable hashes and library integrity"
    )
    add_artifact_runtime_flags(artifact_verify_parser)
    artifact_verify_parser.add_argument("artifact")

    artifact_merge_parser = artifact_subparsers.add_parser(
        "merge", help="Queue adapter baking or a multi-adapter merge"
    )
    add_artifact_runtime_flags(artifact_merge_parser)
    add_artifact_queue_flags(artifact_merge_parser)
    artifact_merge_parser.add_argument(
        "artifacts", nargs="+", help="Input artifact occurrence ids or aliases, in order"
    )
    artifact_merge_parser.add_argument("--base-model", required=True)
    artifact_merge_parser.add_argument("--mode", choices=["bake", "combine"], default="combine")
    artifact_merge_parser.add_argument("--method", default="dare_ties")
    artifact_merge_parser.add_argument("--weights", nargs="+", type=float)
    artifact_merge_parser.add_argument("--bake-after-merge", action="store_true")

    artifact_convert_parser = artifact_subparsers.add_parser(
        "convert", help="Queue verified conversion or post-training quantization"
    )
    add_artifact_runtime_flags(artifact_convert_parser)
    add_artifact_queue_flags(artifact_convert_parser)
    artifact_convert_parser.add_argument("artifact")
    artifact_convert_parser.add_argument(
        "--format", required=True, help="hf, mlx, gguf, or onnx when supported"
    )
    artifact_convert_parser.add_argument(
        "--quantization",
        "--quant",
        dest="quantization",
        default="fp16",
        help="q4/q8 post-training quantization or fp16/bf16/fp32 dtype conversion; not QAT",
    )
    artifact_convert_parser.add_argument(
        "--allow-unquantized-fallback",
        action="store_true",
        help="Permit an explicitly reported unquantized output if the backend lacks quantization",
    )

    artifact_qualify_parser = artifact_subparsers.add_parser(
        "qualify", help="Queue evidence-backed qualification under an immutable profile revision"
    )
    add_artifact_runtime_flags(artifact_qualify_parser)
    add_artifact_queue_flags(artifact_qualify_parser)
    artifact_qualify_parser.add_argument("artifact")
    artifact_qualify_parser.add_argument("--profile", required=True)
    artifact_qualify_parser.add_argument("--parent")
    artifact_qualify_parser.add_argument(
        "--request", help="JSON/YAML object or path with evaluator execution settings"
    )

    artifact_compare_parser = artifact_subparsers.add_parser(
        "compare", help="Compare parent and candidate qualification evidence"
    )
    add_artifact_runtime_flags(artifact_compare_parser)
    artifact_compare_parser.add_argument("parent")
    artifact_compare_parser.add_argument("candidate")
    artifact_compare_parser.add_argument(
        "--profile", help="Require this exact qualification profile revision"
    )

    for command_name, command_help in (
        ("pin", "Protect an artifact from cleanup"),
        ("unpin", "Remove an artifact pin"),
    ):
        command_parser = artifact_subparsers.add_parser(command_name, help=command_help)
        add_artifact_runtime_flags(command_parser)
        command_parser.add_argument("artifact")

    artifact_tag_parser = artifact_subparsers.add_parser("tag", help="Add or replace artifact tags")
    add_artifact_runtime_flags(artifact_tag_parser)
    artifact_tag_parser.add_argument("artifact")
    artifact_tag_parser.add_argument("tags", nargs="+")
    artifact_tag_parser.add_argument("--replace", action="store_true")

    artifact_promote_parser = artifact_subparsers.add_parser(
        "promote", help="Explicitly move the candidate or approved alias"
    )
    add_artifact_runtime_flags(artifact_promote_parser)
    artifact_promote_parser.add_argument("artifact")
    artifact_promote_parser.add_argument("alias", nargs="?", choices=["candidate", "approved"])
    artifact_promote_parser.add_argument(
        "--to", "--alias", dest="alias_option", choices=["candidate", "approved"]
    )
    artifact_promote_parser.add_argument(
        "--override-note", help="Recorded reason for overriding failed or missing gates"
    )

    artifact_serve_parser = artifact_subparsers.add_parser(
        "serve", help="Create a verified managed-serving reservation"
    )
    add_artifact_runtime_flags(artifact_serve_parser)
    add_artifact_queue_flags(artifact_serve_parser)
    artifact_serve_parser.add_argument("artifact")
    artifact_serve_parser.add_argument("--name", default="Local serving profile")
    artifact_serve_parser.add_argument("--backend", default="local")
    artifact_serve_parser.add_argument(
        "--endpoint", help="JSON/YAML endpoint settings object or path"
    )
    artifact_serve_parser.add_argument(
        "--generation", help="JSON/YAML generation defaults object or path"
    )
    artifact_serve_parser.add_argument(
        "--resources", help="JSON/YAML resource requirements object or path"
    )
    artifact_serve_parser.add_argument("--serving-id")

    artifact_export_parser = artifact_subparsers.add_parser(
        "export", help="Create a portable verified local bundle"
    )
    add_artifact_runtime_flags(artifact_export_parser)
    add_artifact_queue_flags(artifact_export_parser)
    artifact_export_parser.add_argument("artifact")
    artifact_export_parser.add_argument("destination")
    artifact_export_parser.add_argument("--replay-identity")
    artifact_export_parser.add_argument("--dataset-identity")
    artifact_export_parser.add_argument("--license-metadata")
    artifact_export_parser.add_argument(
        "--model-card", help="Model-card text or a path containing model-card text"
    )

    artifact_cleanup_parser = artifact_subparsers.add_parser(
        "cleanup", help="Preview, apply, restore, or purge reviewed artifact cleanup"
    )
    add_artifact_runtime_flags(artifact_cleanup_parser)
    add_artifact_queue_flags(artifact_cleanup_parser)
    add_cleanup_flags(artifact_cleanup_parser)

    storage_parser = subparsers.add_parser(
        "storage", help="Inspect Artifact Studio storage and reviewed cleanup"
    )
    storage_subparsers = storage_parser.add_subparsers(dest="storage_command", required=True)
    storage_status_parser = storage_subparsers.add_parser(
        "status", help="Show content-addressed storage inventory and disk forecast inputs"
    )
    add_artifact_runtime_flags(storage_status_parser)
    storage_cleanup_parser = storage_subparsers.add_parser(
        "cleanup", help="Preview or execute a reviewed seven-day-trash cleanup plan"
    )
    add_artifact_runtime_flags(storage_cleanup_parser)
    add_artifact_queue_flags(storage_cleanup_parser)
    add_cleanup_flags(storage_cleanup_parser)

    # Continued causal pretraining on immutable document corpora.
    cpt_parser = subparsers.add_parser(
        "cpt", help="Continued causal pretraining on a document corpus"
    )
    cpt_subparsers = cpt_parser.add_subparsers(
        dest="cpt_command", required=True
    )
    cpt_train_parser = cpt_subparsers.add_parser(
        "train", help="Run corpus continued pretraining"
    )
    add_apple_runtime_flags(cpt_train_parser)
    add_dataset_training_flags(cpt_train_parser)
    cpt_train_parser.add_argument("--config", "-c", help="CPT YAML config")
    cpt_train_parser.add_argument(
        "--model", "-m", help="Causal language model or local model path"
    )
    cpt_train_parser.add_argument("--model-revision")
    cpt_train_parser.add_argument("--model-hash")
    cpt_train_parser.add_argument("--tokenizer-revision")
    cpt_train_parser.add_argument("--tokenizer-hash")
    cpt_train_parser.add_argument(
        "--training-artifact-id",
        help="Verified Dataset Lab training-artifact identity",
    )
    cpt_train_parser.add_argument(
        "--training-artifact-hash",
        help="Verified Dataset Lab training-artifact content hash",
    )
    cpt_train_parser.add_argument(
        "--expected-packing-plan-hash",
        help="Reject execution if repacking differs from the rendered artifact",
    )
    cpt_train_parser.add_argument(
        "--train-file",
        help="Canonical corpus JSONL with non-empty text fields",
    )
    cpt_train_parser.add_argument(
        "--validation-file",
        help="Supplied canonical validation JSONL; never exposed as training data",
    )
    cpt_train_parser.add_argument(
        "--adaptation",
        choices=["lora", "full"],
        help="Required explicit weight-update mode",
    )
    cpt_train_parser.add_argument(
        "--output", "-o", default=None, help="Output directory"
    )
    cpt_train_parser.add_argument("--resume", help="Resume a supported checkpoint")
    cpt_train_parser.add_argument(
        "--dry-run", action="store_true", help="Validate corpus and configuration"
    )
    cpt_train_parser.add_argument("--seed", type=int, default=None)
    cpt_train_parser.add_argument(
        "--max-seq-length", type=int, default=None
    )
    cpt_train_parser.add_argument(
        "--packing",
        choices=[
            "paragraph_eos_non_overlap_v1",
            "paragraph_eos_non_overlap",
        ],
        default=None,
    )
    cpt_train_parser.add_argument(
        "--budget-mode", choices=["tokens", "passes"], default=None
    )
    cpt_train_parser.add_argument("--target-tokens", type=_positive_int_arg)
    cpt_train_parser.add_argument("--corpus-passes", type=float, default=None)
    cpt_train_parser.add_argument("--batch-size", type=int, default=None)
    cpt_train_parser.add_argument(
        "--gradient-accumulation", type=int, default=None
    )
    cpt_train_parser.add_argument("--learning-rate", type=float, default=None)
    cpt_train_parser.add_argument("--warmup-ratio", type=float, default=None)
    cpt_train_parser.add_argument("--weight-decay", type=float, default=None)
    cpt_train_parser.add_argument("--max-grad-norm", type=float, default=None)
    cpt_train_parser.add_argument("--max-steps", type=_positive_int_arg)
    cpt_train_parser.add_argument("--optim", default=None)
    cpt_train_parser.add_argument("--lora-rank", type=int, default=None)
    cpt_train_parser.add_argument("--lora-alpha", type=int, default=None)
    cpt_train_parser.add_argument("--lora-dropout", type=float, default=None)
    cpt_train_parser.add_argument(
        "--target-modules",
        help="Comma-separated LoRA module names; omit for reviewed defaults",
    )
    cpt_train_parser.add_argument("--use-dora", action="store_true", default=None)
    cpt_train_parser.add_argument("--use-rslora", action="store_true", default=None)
    cpt_train_parser.add_argument("--init-lora-weights", default=None)
    cpt_train_parser.add_argument("--load-in-4bit", action="store_true", default=None)
    cpt_train_parser.add_argument("--no-bf16", action="store_true", default=None)
    cpt_train_parser.add_argument(
        "--no-gradient-checkpointing", action="store_true", default=None
    )
    cpt_train_parser.add_argument("--validation-split", type=float, default=None)
    cpt_train_parser.add_argument("--save-steps", type=int, default=None)
    cpt_train_parser.add_argument("--eval-steps", type=int, default=None)
    cpt_train_parser.add_argument("--save-total-limit", type=int, default=None)
    cpt_train_parser.add_argument("--logging-steps", type=int, default=None)

    # sft command
    sft_parser = subparsers.add_parser("sft", help="SFT training")
    sft_subparsers = sft_parser.add_subparsers(dest="sft_command", required=True)

    # sft train
    sft_train_parser = sft_subparsers.add_parser("train", help="Run SFT training")
    add_apple_runtime_flags(sft_train_parser)
    add_dataset_training_flags(sft_train_parser)
    sft_train_parser.add_argument("--config", "-c", help="Config file path")
    sft_train_parser.add_argument(
        "--model", "-m", default="Qwen/Qwen2.5-Coder-7B", help="Base model"
    )
    sft_train_parser.add_argument(
        "--dataset", "-d", help="HuggingFace dataset ID or short name (e.g., codealpaca, metamath)"
    )
    sft_train_parser.add_argument("--data", help="Local training data file (JSONL)")
    sft_train_parser.add_argument(
        "--validation-data", help="Prepared validation JSONL; never re-split from train"
    )
    sft_train_parser.add_argument("--output", "-o", default="models/sft", help="Output directory")
    sft_train_parser.add_argument("--resume", help="Resume from checkpoint")
    sft_train_parser.add_argument(
        "--dry-run", action="store_true", help="Validate config without training"
    )
    sft_train_parser.add_argument(
        "--seed", type=int, default=42, help="Training seed (default: 42)"
    )
    sft_train_parser.add_argument(
        "--max-steps",
        type=_positive_int_arg,
        help="Stop after this many optimizer steps (enables bounded sweep segments)",
    )
    sft_train_parser.add_argument(
        "--capture-parameter-hashes",
        action="store_true",
        help="Record exact before/after trainable-tensor hashes for certification evidence",
    )

    # Training hyperparameters
    sft_train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    sft_train_parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    sft_train_parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    sft_train_parser.add_argument(
        "--warmup-ratio", type=float, default=0.03, help="Warmup ratio for LR scheduler"
    )
    sft_train_parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay for regularization"
    )
    sft_train_parser.add_argument(
        "--max-grad-norm", type=float, default=0.3, help="Max gradient norm for clipping"
    )
    sft_train_parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=16,
        help="Gradient accumulation steps (effective batch = batch_size * accum)",
    )

    # LoRA options
    sft_train_parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    sft_train_parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    sft_train_parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    sft_train_parser.add_argument(
        "--no-lora", action="store_true", help="Disable LoRA (full fine-tuning)"
    )
    # Track T5 — PEFT additions. Vanilla LoRA stays the default; opt in.
    sft_train_parser.add_argument(
        "--use-dora",
        action="store_true",
        help="Use DoRA (decomposed magnitude+direction); slightly slower but typically matches LoRA at lower rank",
    )
    sft_train_parser.add_argument(
        "--use-rslora",
        action="store_true",
        help="Use rank-stabilized LoRA scaling (alpha/sqrt(r) instead of alpha/r)",
    )
    sft_train_parser.add_argument(
        "--init-lora-weights",
        default="true",
        help="LoRA initialization: true (default), pissa, pissa_niter_4, loftq, olora, gaussian, false",
    )
    # Track T4 — optimizer choice.
    sft_train_parser.add_argument(
        "--optim",
        default="adamw_torch",
        help="Optimizer (adamw_torch, adamw_bnb_8bit, lion_8bit, paged_adamw_8bit, ...)",
    )

    # Checkpointing
    sft_train_parser.add_argument(
        "--save-steps", type=int, default=500, help="Save checkpoint every N steps"
    )
    sft_train_parser.add_argument(
        "--eval-steps", type=int, default=250, help="Evaluate every N steps"
    )
    sft_train_parser.add_argument(
        "--save-total-limit", type=int, default=3, help="Max checkpoints to keep"
    )

    # Early stopping
    sft_train_parser.add_argument(
        "--early-stopping-patience", type=int, default=5, help="Stop if no improvement for N evals"
    )

    # Data options
    sft_train_parser.add_argument(
        "--max-samples", type=int, help="Limit number of training samples"
    )
    sft_train_parser.add_argument(
        "--validation-split", type=float, default=0.05, help="Validation set fraction"
    )
    sft_train_parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Maximum sequence length"
    )

    # Hardware options
    sft_train_parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (uses more memory)",
    )

    # sft datasets
    sft_datasets_parser = sft_subparsers.add_parser("datasets", help="List available SFT datasets")

    # dpo command (Track T1 / phase Q1) - Direct Preference Optimization
    dpo_parser = subparsers.add_parser("dpo", help="DPO (Direct Preference Optimization) training")
    dpo_subparsers = dpo_parser.add_subparsers(dest="dpo_command", required=True)

    # dpo train
    dpo_train_parser = dpo_subparsers.add_parser("train", help="Run DPO training")
    add_apple_runtime_flags(dpo_train_parser)
    add_dataset_training_flags(dpo_train_parser)
    dpo_train_parser.add_argument("--config", "-c", help="Config file path")
    dpo_train_parser.add_argument(
        "--model", "-m", default="Qwen/Qwen2.5-3B-Instruct", help="Base / SFT-tuned model to align"
    )
    dpo_train_parser.add_argument(
        "--dataset",
        "-d",
        help="HuggingFace dataset id or short name (ultrafeedback, orca_dpo, hh_rlhf, py_dpo)",
    )
    dpo_train_parser.add_argument(
        "--data", help="Local JSONL file with prompt/chosen/rejected rows"
    )
    dpo_train_parser.add_argument("--validation-data", help="Prepared validation JSONL")
    dpo_train_parser.add_argument("--output", "-o", default="models/dpo", help="Output directory")
    dpo_train_parser.add_argument("--resume", help="Resume from checkpoint")
    dpo_train_parser.add_argument(
        "--dry-run", action="store_true", help="Validate config without training"
    )
    dpo_train_parser.add_argument(
        "--seed", type=int, default=42, help="Training seed (default: 42)"
    )
    dpo_train_parser.add_argument(
        "--max-steps",
        type=_positive_int_arg,
        help="Stop after this many optimizer steps (enables bounded sweep segments)",
    )

    # DPO algorithm knobs
    dpo_train_parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="KL-regularization strength against the reference model (default: 0.1)",
    )
    dpo_train_parser.add_argument(
        "--loss-type",
        default="sigmoid",
        choices=["sigmoid", "ipo", "hinge", "kto_pair", "rpo"],
        help="DPO loss variant (default: sigmoid)",
    )
    dpo_train_parser.add_argument(
        "--reference-free",
        action="store_true",
        help="Skip the reference model (uses policy at step 0); saves memory",
    )
    dpo_train_parser.add_argument(
        "--label-smoothing", type=float, default=0.0, help="cDPO label smoothing (default: 0.0)"
    )

    # Training hyperparameters
    dpo_train_parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    dpo_train_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size (DPO doubles memory: chosen+rejected)",
    )
    dpo_train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate (DPO needs much smaller LR than SFT)",
    )
    dpo_train_parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    dpo_train_parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    dpo_train_parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Max gradient norm"
    )
    dpo_train_parser.add_argument(
        "--gradient-accumulation", type=int, default=16, help="Gradient accumulation steps"
    )

    # LoRA
    dpo_train_parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    dpo_train_parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    dpo_train_parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    dpo_train_parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="QLoRA: load base model in 4-bit (CUDA/ROCm only)",
    )
    # Track T5 — PEFT additions.
    dpo_train_parser.add_argument(
        "--use-dora", action="store_true", help="Use DoRA (decomposed magnitude+direction)"
    )
    dpo_train_parser.add_argument(
        "--use-rslora", action="store_true", help="Use rank-stabilized LoRA scaling"
    )
    dpo_train_parser.add_argument(
        "--init-lora-weights",
        default="true",
        help="LoRA initialization: true, pissa, loftq, olora, gaussian, false",
    )
    # Track T4 — optimizer choice.
    dpo_train_parser.add_argument(
        "--optim",
        default="adamw_torch",
        help="Optimizer (adamw_torch, adamw_bnb_8bit, lion_8bit, ...)",
    )

    # Checkpointing
    dpo_train_parser.add_argument("--save-steps", type=int, default=200, help="Save every N steps")
    dpo_train_parser.add_argument("--eval-steps", type=int, default=100, help="Eval every N steps")
    dpo_train_parser.add_argument(
        "--save-total-limit", type=int, default=3, help="Max checkpoints to keep"
    )

    # Data options
    dpo_train_parser.add_argument("--max-samples", type=int, help="Limit number of training pairs")
    dpo_train_parser.add_argument(
        "--validation-split", type=float, default=0.05, help="Validation fraction"
    )
    dpo_train_parser.add_argument(
        "--max-seq-length", type=int, default=1024, help="Combined prompt+response length cap"
    )
    dpo_train_parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=512,
        help="Prompt length cap (DPO truncates from the left after this)",
    )

    # Hardware
    dpo_train_parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (uses more memory)",
    )

    # dpo datasets
    dpo_datasets_parser = dpo_subparsers.add_parser(
        "datasets", help="List available preference datasets"
    )

    # orpo command (Track T17b) — Odds-Ratio Preference Optimization.
    # Same input shape as DPO (prompt/chosen/rejected); reference-free,
    # single-pass — typically half the wall-time of DPO at similar quality.
    orpo_parser = subparsers.add_parser(
        "orpo", help="ORPO (Odds-Ratio Preference Optimization) training"
    )
    orpo_subparsers = orpo_parser.add_subparsers(dest="orpo_command", required=True)

    orpo_train_parser = orpo_subparsers.add_parser("train", help="Run ORPO training")
    add_apple_runtime_flags(orpo_train_parser)
    add_dataset_training_flags(orpo_train_parser)
    orpo_train_parser.add_argument("--config", "-c", help="Config file path")
    orpo_train_parser.add_argument(
        "--model", "-m", default="Qwen/Qwen2.5-3B-Instruct", help="Base / SFT-tuned model to align"
    )
    orpo_train_parser.add_argument(
        "--dataset",
        "-d",
        help="HuggingFace dataset id or short name (ultrafeedback, orca_dpo, hh_rlhf, py_dpo)",
    )
    orpo_train_parser.add_argument("--data", help="Local JSONL with prompt/chosen/rejected rows")
    orpo_train_parser.add_argument("--validation-data", help="Prepared validation JSONL")
    orpo_train_parser.add_argument("--output", "-o", default="models/orpo", help="Output directory")
    orpo_train_parser.add_argument("--resume", help="Resume from checkpoint")
    orpo_train_parser.add_argument(
        "--dry-run", action="store_true", help="Validate config without training"
    )
    orpo_train_parser.add_argument(
        "--seed", type=int, default=42, help="Training seed (default: 42)"
    )
    orpo_train_parser.add_argument(
        "--max-steps",
        type=_positive_int_arg,
        help="Stop after this many optimizer steps (enables bounded sweep segments)",
    )

    orpo_train_parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Relative weight of preference (log-odds) term vs NLL (default: 0.1)",
    )

    orpo_train_parser.add_argument("--epochs", type=int, default=1)
    orpo_train_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size (ORPO sees chosen+rejected per row)",
    )
    orpo_train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=8e-6,
        help="Learning rate (default: 8e-6 — between SFT and DPO)",
    )
    orpo_train_parser.add_argument("--warmup-ratio", type=float, default=0.1)
    orpo_train_parser.add_argument("--weight-decay", type=float, default=0.0)
    orpo_train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    orpo_train_parser.add_argument("--gradient-accumulation", type=int, default=16)

    orpo_train_parser.add_argument("--lora-rank", type=int, default=16)
    orpo_train_parser.add_argument("--lora-alpha", type=int, default=32)
    orpo_train_parser.add_argument("--lora-dropout", type=float, default=0.05)
    orpo_train_parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="QLoRA: load base model in 4-bit (CUDA/ROCm only)",
    )
    orpo_train_parser.add_argument("--use-dora", action="store_true")
    orpo_train_parser.add_argument("--use-rslora", action="store_true")
    orpo_train_parser.add_argument("--init-lora-weights", default="true")
    orpo_train_parser.add_argument("--optim", default="adamw_torch")

    orpo_train_parser.add_argument("--save-steps", type=int, default=200)
    orpo_train_parser.add_argument("--eval-steps", type=int, default=100)
    orpo_train_parser.add_argument("--save-total-limit", type=int, default=3)
    orpo_train_parser.add_argument("--max-samples", type=int)
    orpo_train_parser.add_argument("--validation-split", type=float, default=0.05)
    orpo_train_parser.add_argument("--max-seq-length", type=int, default=1024)
    orpo_train_parser.add_argument("--max-prompt-length", type=int, default=512)
    orpo_train_parser.add_argument("--no-gradient-checkpointing", action="store_true")

    # rm command (Track T3) — Bradley-Terry reward model trainer.
    rm_parser = subparsers.add_parser("rm", help="Reward model training (Bradley-Terry)")
    rm_subparsers = rm_parser.add_subparsers(dest="rm_command", required=True)

    rm_train_parser = rm_subparsers.add_parser("train", help="Run reward-model training")
    add_apple_runtime_flags(rm_train_parser)
    add_dataset_training_flags(rm_train_parser)
    rm_train_parser.add_argument("--config", "-c", help="Config file path")
    rm_train_parser.add_argument(
        "--model", "-m", default="Qwen/Qwen2.5-3B-Instruct", help="Base / SFT-tuned model"
    )
    rm_train_parser.add_argument(
        "--dataset",
        "-d",
        help="HF preference dataset id (ultrafeedback, orca_dpo, hh_rlhf, py_dpo)",
    )
    rm_train_parser.add_argument("--data", help="Local JSONL with prompt/chosen/rejected rows")
    rm_train_parser.add_argument("--validation-data", help="Prepared validation JSONL")
    rm_train_parser.add_argument("--output", "-o", default="models/rm", help="Output directory")
    rm_train_parser.add_argument("--resume", help="Resume from checkpoint")
    rm_train_parser.add_argument("--dry-run", action="store_true")
    rm_train_parser.add_argument("--seed", type=int, default=42, help="Training seed (default: 42)")
    rm_train_parser.add_argument(
        "--max-steps",
        type=_positive_int_arg,
        help="Stop after this many optimizer steps (enables bounded sweep segments)",
    )

    rm_train_parser.add_argument("--epochs", type=int, default=1)
    rm_train_parser.add_argument("--batch-size", type=int, default=4)
    rm_train_parser.add_argument("--learning-rate", type=float, default=1e-5)
    rm_train_parser.add_argument("--warmup-ratio", type=float, default=0.05)
    rm_train_parser.add_argument("--weight-decay", type=float, default=0.0)
    rm_train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    rm_train_parser.add_argument("--gradient-accumulation", type=int, default=4)
    rm_train_parser.add_argument("--max-length", type=int, default=1024)
    rm_train_parser.add_argument("--max-samples", type=int)
    rm_train_parser.add_argument(
        "--center-rewards-coefficient",
        type=float,
        default=0.01,
        help="Centering regularizer (default 0.01; 0 disables)",
    )

    rm_train_parser.add_argument("--lora-rank", type=int, default=8)
    rm_train_parser.add_argument("--lora-alpha", type=int, default=16)
    rm_train_parser.add_argument("--lora-dropout", type=float, default=0.05)
    rm_train_parser.add_argument("--load-in-4bit", action="store_true")
    rm_train_parser.add_argument("--use-dora", action="store_true")
    rm_train_parser.add_argument("--use-rslora", action="store_true")
    rm_train_parser.add_argument("--init-lora-weights", default="true")
    rm_train_parser.add_argument("--optim", default="adamw_torch")
    rm_train_parser.add_argument(
        "--save-steps", type=int, default=200, help="Save every N optimizer steps"
    )
    rm_train_parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum checkpoints retained by the trainer",
    )
    rm_train_parser.add_argument("--no-gradient-checkpointing", action="store_true")

    # grpo command (Track T2 / phase Q1) — Group Relative Policy Optimization
    grpo_parser = subparsers.add_parser("grpo", help="GRPO training (verifier-grounded RL)")
    grpo_subparsers = grpo_parser.add_subparsers(dest="grpo_command", required=True)

    grpo_train_parser = grpo_subparsers.add_parser("train", help="Run GRPO training")
    add_apple_runtime_flags(grpo_train_parser)
    add_dataset_training_flags(grpo_train_parser, consumes_verifier=True)
    grpo_train_parser.add_argument("--config", "-c", help="Config file path")
    grpo_train_parser.add_argument(
        "--model", "-m", default="Qwen/Qwen2.5-3B-Instruct", help="Base / SFT-tuned model"
    )
    grpo_train_parser.add_argument(
        "--dataset", "-d", help='HuggingFace dataset id (must have a "prompt" column)'
    )
    grpo_train_parser.add_argument("--data", help='Local JSONL with "prompt" rows')
    grpo_train_parser.add_argument("--output", "-o", default="models/grpo", help="Output directory")
    grpo_train_parser.add_argument("--resume", help="Resume from checkpoint")
    grpo_train_parser.add_argument(
        "--dry-run", action="store_true", help="Validate config without training"
    )
    grpo_train_parser.add_argument(
        "--seed", type=int, default=42, help="Training seed (default: 42)"
    )
    grpo_train_parser.add_argument(
        "--max-steps",
        type=_positive_int_arg,
        help="Stop after this many optimizer steps (enables bounded sweep segments)",
    )

    # GRPO algorithm
    grpo_train_parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Group size: completions sampled per prompt (default: 4)",
    )
    grpo_train_parser.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="KL-regularization strength (default: 0.04, DeepSeek-R1)",
    )
    grpo_train_parser.add_argument(
        "--epsilon", type=float, default=0.2, help="PPO ratio clip (default: 0.2)"
    )
    grpo_train_parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Rollout temperature (default: 0.9 — diverse groups)",
    )
    grpo_train_parser.add_argument(
        "--no-scale-rewards",
        action="store_true",
        help="Skip dividing advantages by std(group); RLOO-flavored",
    )
    grpo_train_parser.add_argument(
        "--reference-free", action="store_true", help="Skip reference model; saves memory"
    )
    grpo_train_parser.add_argument(
        "--verifier",
        default="execution",
        help="Verifier short-name from the V1 plugin registry "
        "(execution, llm_judge, ...). Run halo-forge sft datasets "
        "to see registered verifiers.",
    )
    grpo_train_parser.add_argument(
        "--reward-threshold", type=float, default=0.0, help="Below this, advantage is forced to 0"
    )

    # Hyperparameters
    grpo_train_parser.add_argument("--epochs", type=int, default=1)
    grpo_train_parser.add_argument("--batch-size", type=int, default=1)
    grpo_train_parser.add_argument(
        "--learning-rate", type=float, default=1e-6, help="GRPO LR (much smaller than SFT)"
    )
    grpo_train_parser.add_argument("--warmup-ratio", type=float, default=0.1)
    grpo_train_parser.add_argument("--weight-decay", type=float, default=0.0)
    grpo_train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    grpo_train_parser.add_argument("--gradient-accumulation", type=int, default=16)

    # LoRA
    grpo_train_parser.add_argument("--lora-rank", type=int, default=16)
    grpo_train_parser.add_argument("--lora-alpha", type=int, default=32)
    grpo_train_parser.add_argument("--lora-dropout", type=float, default=0.05)
    grpo_train_parser.add_argument("--load-in-4bit", action="store_true")
    grpo_train_parser.add_argument("--use-dora", action="store_true")
    grpo_train_parser.add_argument("--use-rslora", action="store_true")
    grpo_train_parser.add_argument("--init-lora-weights", default="true")
    grpo_train_parser.add_argument("--optim", default="adamw_torch")
    grpo_train_parser.add_argument(
        "--save-steps", type=int, default=100, help="Save every N optimizer steps"
    )
    grpo_train_parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum checkpoints retained by the trainer",
    )

    # Data lengths
    grpo_train_parser.add_argument("--max-samples", type=int)
    grpo_train_parser.add_argument("--max-prompt-length", type=int, default=512)
    grpo_train_parser.add_argument("--max-completion-length", type=int, default=512)

    # Rollout engine (Track I6)
    grpo_train_parser.add_argument(
        "--rollout-engine",
        default="auto",
        choices=["auto", "torch", "vllm", "mlx"],
        help="Generation backend for the rollout stage",
    )

    grpo_train_parser.add_argument("--no-gradient-checkpointing", action="store_true")

    # probe command (Track V9) — mid-training general-benchmark probe.
    probe_parser = subparsers.add_parser(
        "probe",
        help="Run a small held-out benchmark + diff vs baseline (catastrophic-forgetting safeguard)",
    )
    probe_parser.add_argument(
        "--model", "-m", required=True, help="Model id, mlx-community id, or local path"
    )
    probe_parser.add_argument(
        "--tasks", "-t", help="Comma-separated task names (default: small probe set)"
    )
    probe_parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Samples per task (default 100; smaller = faster probe)",
    )
    probe_parser.add_argument(
        "--baseline", help="Path to baseline.json. First run writes; subsequent runs diff."
    )
    probe_parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Regression triggered when Δ < -tolerance (default 0.05)",
    )
    probe_parser.add_argument("--backend", default="hf", choices=["hf", "vllm", "mlx"])
    probe_parser.add_argument(
        "--cycle", type=int, help="Tag this probe with the cycle number it ran at"
    )
    probe_parser.add_argument(
        "--notes", help='Free-form annotation for this probe (e.g. "after SFT")'
    )

    # eval command (Track V8) — lm-evaluation-harness wrapper.
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run academic benchmarks via lm-evaluation-harness",
    )
    eval_parser.add_argument("--model", "-m", help="Model id, mlx-community id, or local path")
    eval_parser.add_argument(
        "--tasks",
        "-t",
        default="core",
        help="Comma-separated task names or curated group "
        "(core, reasoning, code, instruction_following, knowledge)",
    )
    eval_parser.add_argument("--limit", type=int, help="Cap samples per task (smoke-test mode)")
    eval_parser.add_argument(
        "--batch-size", type=int, help="Per-step batch size (lm-eval default if omitted)"
    )
    eval_parser.add_argument(
        "--backend",
        default="hf",
        choices=["hf", "vllm", "mlx"],
        help="lm-eval model adapter (hf works on every backend; "
        "vllm faster on CUDA/ROCm; mlx for Apple Silicon)",
    )
    eval_parser.add_argument(
        "--output", "-o", help="Directory to write lm_eval_summary.json + raw results"
    )
    eval_parser.add_argument(
        "--list-tasks", action="store_true", help="Print curated task groups and exit"
    )
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command")

    eval_suite_parser = eval_subparsers.add_parser(
        "suite", help="Create, revise, list, or inspect persistent benchmark suites"
    )
    eval_suite_actions = eval_suite_parser.add_subparsers(dest="suite_action", required=True)
    eval_suite_actions.add_parser("list")
    eval_suite_show = eval_suite_actions.add_parser("show")
    eval_suite_show.add_argument("suite_id")

    def add_suite_revision_fields(value_parser):
        value_parser.add_argument(
            "--items", required=True, help="JSON file or inline ordered item array"
        )
        value_parser.add_argument("--primary-metric", default="score")
        value_parser.add_argument(
            "--direction", choices=["maximize", "minimize"], default="maximize"
        )
        value_parser.add_argument("--generation-settings", help="JSON file or inline object")
        value_parser.add_argument("--evaluator-versions", help="JSON file or inline object")

    eval_suite_create = eval_suite_actions.add_parser("create")
    eval_suite_create.add_argument("--name", required=True)
    eval_suite_create.add_argument("--description")
    eval_suite_create.add_argument(
        "--purpose",
        choices=["development", "holdout", "unspecified"],
        default="unspecified",
        help="How this suite may be used: development guides selection; holdout is confirmation-only",
    )
    add_suite_revision_fields(eval_suite_create)
    eval_suite_revise = eval_suite_actions.add_parser("revise")
    eval_suite_revise.add_argument("suite_id")
    add_suite_revision_fields(eval_suite_revise)

    eval_run_parser = eval_subparsers.add_parser(
        "run", help="Launch a persistent evaluation against an immutable suite revision"
    )
    eval_run_parser.add_argument("--suite-revision", required=True)
    eval_run_parser.add_argument(
        "--adapter",
        help="Evaluation adapter override (default: infer from the immutable suite revision)",
    )
    eval_run_parser.add_argument(
        "--subject-type", choices=["model", "run", "final_model", "checkpoint"], default="model"
    )
    eval_run_parser.add_argument("--subject", required=True)
    eval_run_parser.add_argument(
        "--subject-revision",
        help="Pinned model/tokenizer revision included in subject identity",
    )
    eval_run_parser.add_argument(
        "--run-id",
        help="Resolve a final model or checkpoint relative to this completed run",
    )
    eval_run_parser.add_argument("--request", help="Adapter request JSON file or inline object")
    eval_run_parser.add_argument(
        "--verifier-profile-revision",
        help="Exact qualified verifier revision consumed by verifier-backed adapters",
    )
    eval_run_parser.add_argument(
        "--wait", action="store_true", help="Wait and print the completed immutable result"
    )

    eval_jobs_parser = eval_subparsers.add_parser("jobs", help="Inspect or manage evaluation jobs")
    eval_jobs_parser.add_argument("--evaluation-id")
    eval_jobs_parser.add_argument("--status")
    eval_jobs_parser.add_argument("--limit", type=int, default=100)
    eval_jobs_parser.add_argument("--cancel")
    eval_jobs_parser.add_argument("--retry")

    eval_compare_parser = eval_subparsers.add_parser(
        "compare", help="Direction-aware comparison over the same suite revision"
    )
    eval_compare_parser.add_argument("base")
    eval_compare_parser.add_argument("candidate")

    eval_history_parser = eval_subparsers.add_parser(
        "history", help="List immutable longitudinal evaluation results"
    )
    eval_history_parser.add_argument("--database")
    eval_history_parser.add_argument("--subject", help="Pinned subject reference")
    eval_history_parser.add_argument("--suite-revision")
    eval_history_parser.add_argument("--limit", type=int, default=100)

    eval_drift_parser = eval_subparsers.add_parser(
        "drift", help="Compare compatible points in an evaluation history"
    )
    eval_drift_parser.add_argument("--database")
    eval_drift_parser.add_argument("--base", help="Base evaluation id")
    eval_drift_parser.add_argument("--candidate", help="Candidate evaluation id")
    eval_drift_parser.add_argument("--subject", help="Select the two latest for this subject")
    eval_drift_parser.add_argument("--suite-revision")
    eval_drift_parser.add_argument("--practical-delta", type=float, default=0.0)

    # token command (Track P1) — API token lifecycle.
    token_parser = subparsers.add_parser(
        "token",
        help="Manage API tokens for the public API (auto-required when bound to non-loopback)",
    )
    token_subparsers = token_parser.add_subparsers(dest="token_command", required=True)

    token_create = token_subparsers.add_parser("create", help="Create a new bearer token")
    token_create.add_argument("name", help='Friendly name (e.g. "dashboard", "ci")')
    token_create.add_argument("--note", help="Free-form annotation")

    token_subparsers.add_parser("list", help="List existing tokens (no secrets shown)")

    token_revoke = token_subparsers.add_parser("revoke", help="Revoke a token by name")
    token_revoke.add_argument("name", help="Name of the token to revoke")

    # replay command (Track T15) — deterministic-replay manifest tools.
    replay_parser = subparsers.add_parser(
        "replay",
        help="Show or relaunch a captured run from its replay.json manifest",
    )
    replay_parser.add_argument("source", help="Path to a run directory or replay.json file")
    replay_parser.add_argument(
        "--launch",
        action="store_true",
        help="Actually relaunch (subprocess) instead of just printing the command",
    )
    replay_parser.add_argument(
        "--force", action="store_true", help="[--launch] Launch even if the env fingerprint differs"
    )
    replay_parser.add_argument(
        "--allow-dataset-drift",
        action="store_true",
        help="[--launch] Launch even if a captured local dataset hash differs",
    )
    replay_parser.add_argument(
        "--allow-verifier-drift",
        action="store_true",
        help="[--launch] Override a missing or drifted immutable verifier identity",
    )
    replay_parser.add_argument(
        "--verifier-drift-reason",
        help="Required recorded operator reason when --allow-verifier-drift is used",
    )
    replay_parser.add_argument(
        "--allow-reward-drift",
        action="store_true",
        help="[--launch] Override a missing or drifted immutable reward-system identity",
    )
    replay_parser.add_argument(
        "--reward-drift-reason",
        help="Required recorded operator reason when --allow-reward-drift is used",
    )

    # merge command (Tracks T12 + T13) — adapter bake / multi-adapter combine.
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge LoRA adapters (bake into base, or combine multiple via TIES/DARE)",
    )
    merge_parser.add_argument(
        "--mode",
        "-m",
        choices=["bake", "combine"],
        default="bake",
        help="bake = single adapter into base; combine = N adapters into one",
    )
    merge_parser.add_argument("--base", "-b", help="Base model (HF id or local path). Required.")
    merge_parser.add_argument("--output", "-o", help="Output directory.")
    # bake mode
    merge_parser.add_argument("--adapter", "-a", help="[bake] Adapter directory to merge")
    # combine mode
    merge_parser.add_argument("--adapters", help="[combine] Comma-separated adapter paths")
    merge_parser.add_argument(
        "--weights",
        help='[combine] Comma-separated weights (e.g. "0.5,0.3,0.2"). Defaults to uniform.',
    )
    merge_parser.add_argument(
        "--method",
        default="dare_ties",
        help="[combine] Merge method: linear / ties / dare_linear / dare_ties / magnitude_prune",
    )
    merge_parser.add_argument(
        "--bake-after-merge",
        action="store_true",
        help="[combine] Also bake the combined adapter into the base; output is a merged checkpoint",
    )
    merge_parser.add_argument(
        "--svd-rank", type=int, help="[combine] Override SVD rank for ties / dare_ties methods"
    )
    merge_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Opt into executing remote model code while loading merge inputs",
    )
    merge_parser.add_argument(
        "--list", action="store_true", help="Print supported operations / methods and exit"
    )

    # convert command (Track I5) — unified format conversion.
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert a model between formats (HF → MLX / GGUF / HF dtype recast)",
    )
    convert_parser.add_argument(
        "--source", "-s", help="HuggingFace id, mlx-community id, or local path of source model"
    )
    convert_parser.add_argument(
        "--output", "-o", help="Output path (file for GGUF, directory for MLX/HF)"
    )
    convert_parser.add_argument(
        "--format",
        "-f",
        default="mlx",
        choices=["mlx", "gguf", "hf"],
        help="Target format (default: mlx)",
    )
    convert_parser.add_argument(
        "--quant",
        "-q",
        default="q4",
        help="Normalized quant: q4, q8, fp16, bf16, fp32 (default: q4)",
    )
    convert_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Opt into executing remote model code while loading/converting",
    )
    convert_parser.add_argument(
        "--allow-unquantized-fallback",
        action="store_true",
        help="For GGUF only: allow FP16 output if requested quantization cannot run",
    )
    convert_parser.add_argument(
        "--list", action="store_true", help="Print supported formats / quants and exit"
    )
    convert_parser.add_argument(
        "--verify",
        action="store_true",
        help="Track I4: after conversion, run a fixed prompt set "
        "through both source and exported and flag drift. "
        "Adds ~30s; catches silently-broken exports.",
    )

    # serve command (Track I1) — OpenAI-compatible serving endpoint.
    serve_parser = subparsers.add_parser(
        "serve",
        help="Run an OpenAI-compatible serving endpoint for a trained model",
    )
    serve_parser.add_argument(
        "--model", "-m", required=True, help="Model id, mlx-community id, or local path to serve"
    )
    serve_parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1; loopback only)"
    )
    serve_parser.add_argument("--port", type=int, default=8001, help="Bind port (default: 8001)")
    serve_parser.add_argument(
        "--backend",
        help="Force a backend (mlx, mps, cuda, rocm_gfx1151, cpu); " "defaults to autodetect",
    )
    serve_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Opt into executing remote model code while loading the served model",
    )
    serve_parser.add_argument(
        "--check",
        action="store_true",
        help="Validate serving configuration and print endpoints without binding a port",
    )

    # dashboard command — user-facing app, API + built React dashboard on one origin.
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        aliases=["app"],
        help="Run the Halo Forge dashboard app",
    )
    dashboard_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1; use 0.0.0.0 for trusted-network access)",
    )
    dashboard_parser.add_argument(
        "--port", type=int, default=8000, help="Bind port (default: 8000)"
    )
    dashboard_parser.add_argument(
        "--check",
        action="store_true",
        help="Print dashboard startup details without binding a port",
    )
    dashboard_parser.add_argument(
        "--no-build",
        action="store_true",
        help="Do not auto-build public_app/dist when dashboard assets are missing",
    )
    dashboard_parser.add_argument(
        "--open",
        action="store_true",
        help="Open the dashboard URL in the default browser for loopback launches",
    )

    # serve-public — dashboard FastAPI (the API the public_app SPA talks to)
    serve_public_parser = subparsers.add_parser(
        "serve-public",
        help="Run only the dashboard FastAPI for frontend development",
    )
    serve_public_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1; loopback skips bearer auth)",
    )
    serve_public_parser.add_argument(
        "--port", type=int, default=8000, help="Bind port (default: 8000)"
    )
    serve_public_parser.add_argument(
        "--check",
        action="store_true",
        help="Print dashboard API startup details without binding a port",
    )

    # raft command
    raft_parser = subparsers.add_parser("raft", help="RAFT training")
    raft_subparsers = raft_parser.add_subparsers(dest="raft_command", required=True)

    # raft train
    raft_train_parser = raft_subparsers.add_parser("train", help="Run RAFT training")
    add_apple_runtime_flags(raft_train_parser, neural_accelerators=False)
    add_dataset_training_flags(raft_train_parser, consumes_verifier=True)
    raft_train_parser.add_argument("--config", "-c", help="Config file path")
    raft_train_parser.add_argument(
        "--seed", type=int, default=42, help="Training seed (default: 42)"
    )
    raft_train_parser.add_argument(
        "--model", "-m", default="Qwen/Qwen2.5-Coder-3B", help="Base model"
    )
    # Phase 5a: when --accelerator mlx is set, rollouts run on MLX while
    # the policy update stays on PyTorch. --rollout-model lets you point at
    # an MLX-format weight set distinct from the torch base. If omitted, we
    # use --model for both (only works if it happens to be MLX-loadable).
    raft_train_parser.add_argument(
        "--rollout-model",
        help="MLX-format model used for rollouts when --accelerator mlx is set "
        "(e.g. mlx-community/Qwen2.5-3B-Instruct-bf16). Defaults to --model.",
    )
    # Phase 5a hybrid: --accelerator mlx --rollout-only keeps the PyTorch
    # RAFT trainer in charge but swaps in MLX-fast rollouts. Without this
    # flag, --accelerator mlx selects the full MLX-native RAFT trainer
    # (Phase 5b).
    raft_train_parser.add_argument(
        "--rollout-only",
        action="store_true",
        help="[--accelerator mlx] Hybrid mode: MLX rollouts + PyTorch policy update. "
        "Without this, --accelerator mlx runs RAFT entirely on MLX.",
    )
    # Track I6 — vLLM rollouts for CUDA/ROCm. Largest single throughput
    # win available for RAFT without changing the algorithm.
    raft_train_parser.add_argument(
        "--rollout-engine",
        default="auto",
        choices=["auto", "torch", "vllm", "mlx"],
        help='Generation engine for the rollout stage. "auto" picks torch '
        '(default; HF generate). "vllm" uses continuous-batched '
        'inference (CUDA / ROCm only). "mlx" uses mlx_lm.generate '
        "(Apple Silicon; equivalent throughput story to vllm on its "
        "native hardware).",
    )
    raft_train_parser.add_argument("--checkpoint", help="SFT checkpoint path (optional)")
    raft_train_parser.add_argument("--prompts", "-p", help="Prompts file")
    raft_train_parser.add_argument(
        "--max-prompts",
        type=int,
        help="Use only the first N prompts (used by deterministic proof runs)",
    )
    raft_train_parser.add_argument("--output", "-o", default="models/raft", help="Output directory")
    raft_train_parser.add_argument("--cycles", type=int, help="Number of RAFT cycles")
    raft_train_parser.add_argument(
        "--verifier",
        default="gcc",
        choices=list(RAFT_TRAIN_SUPPORTED_VERIFIERS),
        help="Verifier type (parser matches runtime-supported options)",
    )
    raft_train_parser.add_argument(
        "--keep-percent",
        type=float,
        default=0.5,
        help="Keep top X%% of passing samples (0.0-1.0, default: 0.5 = 50%%)",
    )
    raft_train_parser.add_argument(
        "--reward-threshold",
        type=float,
        default=0.5,
        help="Minimum reward to consider sample passing (default: 0.5)",
    )
    raft_train_parser.add_argument(
        "--allow-compile-only-training",
        action="store_true",
        help="Allow compile-only verifier results to train RAFT samples. Disabled by default.",
    )
    raft_train_parser.add_argument(
        "--unsafe-verifier-execution",
        action="store_true",
        help="Run generated-code verifiers directly on the host instead of the sandbox. Dangerous; disabled by default.",
    )
    raft_train_parser.add_argument(
        "--curriculum",
        default="none",
        choices=["none", "complexity", "progressive", "adaptive", "historical"],
        help="Curriculum learning strategy (default: none)",
    )
    raft_train_parser.add_argument(
        "--curriculum-stats",
        type=str,
        default=None,
        help="Path to historical stats JSON for historical curriculum",
    )
    raft_train_parser.add_argument(
        "--curriculum-start",
        type=float,
        default=0.2,
        help="Progressive curriculum: start with this fraction of prompts (default: 0.2)",
    )
    raft_train_parser.add_argument(
        "--curriculum-increment",
        type=float,
        default=0.2,
        help="Progressive curriculum: add this fraction each cycle (default: 0.2)",
    )
    raft_train_parser.add_argument(
        "--reward-shaping",
        default="fixed",
        choices=["fixed", "annealing", "adaptive", "warmup"],
        help="Reward shaping strategy (default: fixed)",
    )
    raft_train_parser.add_argument(
        "--lr-decay", type=float, default=0.85, help="Learning rate decay per cycle (default: 0.85)"
    )
    raft_train_parser.add_argument(
        "--min-lr", type=float, default=1e-6, help="Minimum learning rate floor (default: 1e-6)"
    )
    raft_train_parser.add_argument(
        "--experimental-attention",
        action="store_true",
        help="Enable experimental ROCm attention (needed for LFM2.5, etc.)",
    )
    raft_train_parser.add_argument(
        "--system-prompt",
        default="You are an expert Windows systems programmer.",
        help="System prompt for generation",
    )
    raft_train_parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=8,
        help="Samples to generate per prompt (default: 8)",
    )
    raft_train_parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)"
    )
    raft_train_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024)",
    )
    raft_train_parser.add_argument(
        "--min-samples",
        type=int,
        help="Minimum samples per cycle (auto-adjusts threshold if needed)",
    )
    # Training hyperparameters
    raft_train_parser.add_argument(
        "--learning-rate", type=float, help="Base learning rate (default: 5e-5)"
    )
    raft_train_parser.add_argument(
        "--batch-size", type=int, help="Per-device batch size (default: 2)"
    )
    raft_train_parser.add_argument(
        "--gradient-accumulation", type=int, help="Gradient accumulation steps (default: 16)"
    )
    raft_train_parser.add_argument("--warmup-steps", type=int, help="LR warmup steps (default: 10)")
    # LoRA configuration
    raft_train_parser.add_argument("--lora-rank", type=int, help="LoRA rank (default: 16)")
    raft_train_parser.add_argument("--lora-alpha", type=int, help="LoRA alpha (default: 32)")
    # Verifier options
    raft_train_parser.add_argument("--host", help="MSVC verifier host")
    raft_train_parser.add_argument("--user", help="MSVC verifier user")
    raft_train_parser.add_argument("--ssh-key", help="MSVC verifier SSH key")

    # benchmark command (for reporting, not training)
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark reporting (compare to papers). For training verification, use RAFT.",
    )
    bench_subparsers = bench_parser.add_subparsers(dest="bench_command", required=True)

    # benchmark run (legacy pass@k benchmark)
    bench_run_parser = bench_subparsers.add_parser("run", help="Run pass@k benchmark")
    bench_run_parser.add_argument("--model", "-m", required=True, help="Model path")
    bench_run_parser.add_argument("--prompts", "-p", required=True, help="Prompts file")
    bench_run_parser.add_argument("--output", "-o", help="Output file path")
    bench_run_parser.add_argument("--samples", type=int, default=10, help="Samples per prompt")
    bench_run_parser.add_argument("--k", default="1,5,10", help="k values (comma-separated)")
    bench_run_parser.add_argument("--max-prompts", type=int, help="Max prompts to evaluate")
    bench_run_parser.add_argument(
        "--verifier",
        default="gcc",
        choices=[
            "gcc",
            "mingw",
            "msvc",
            "rust",
            "go",
            "dotnet",
            "powershell",
            "auto",
            "humaneval",
            "mbpp",
            "python",
        ],
        help="Verifier type (humaneval/mbpp/python for Python, auto=multi-language)",
    )
    bench_run_parser.add_argument(
        "--base-model", default="Qwen/Qwen2.5-Coder-7B", help="Base model"
    )
    bench_run_parser.add_argument(
        "--system-prompt",
        default="You are an expert Windows systems programmer.",
        help="System prompt",
    )
    bench_run_parser.add_argument("--host", help="MSVC host")
    bench_run_parser.add_argument("--user", help="MSVC user")
    bench_run_parser.add_argument("--ssh-key", help="MSVC SSH key")
    bench_run_parser.add_argument(
        "--cross-compile", action="store_true", help="Enable Windows cross-compilation for rust/go"
    )
    bench_run_parser.add_argument(
        "--run-after-compile", action="store_true", help="Run compiled code after compile"
    )
    bench_run_parser.add_argument(
        "--unsafe-verifier-execution",
        action="store_true",
        help="Run generated-code verifiers directly on the host instead of the sandbox. Dangerous; disabled by default.",
    )
    bench_run_parser.add_argument(
        "--experimental-attention",
        action="store_true",
        help="Enable experimental ROCm attention (needed for LFM2.5, etc.)",
    )

    # benchmark full (comprehensive RAFT benchmark with hardware metrics)
    bench_full_parser = bench_subparsers.add_parser("full", help="Run comprehensive RAFT benchmark")
    bench_full_parser.add_argument(
        "--model", "-m", help="Model to benchmark (e.g., Qwen/Qwen2.5-Coder-0.5B)"
    )
    bench_full_parser.add_argument(
        "--suite",
        "-s",
        choices=["all", "small", "medium"],
        help="Run predefined suite: all (0.5B, 1.5B, 3B), small (0.5B), medium (0.5B, 1.5B)",
    )
    bench_full_parser.add_argument(
        "--cycles", "-c", type=int, default=2, help="Number of RAFT cycles (default: 2)"
    )
    bench_full_parser.add_argument(
        "--output", "-o", default="results/benchmarks", help="Output directory"
    )
    bench_full_parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    # benchmark eval (simple code evaluation on standard datasets)
    bench_eval_parser = bench_subparsers.add_parser(
        "eval", help="Evaluate model on standard code benchmarks"
    )
    bench_eval_parser.add_argument("--model", "-m", required=True, help="Model name or path")
    bench_eval_parser.add_argument(
        "--benchmark",
        "-b",
        default="humaneval",
        choices=["humaneval", "mbpp", "livecodebench", "cpp", "rust", "go"],
        help="Benchmark dataset (default: humaneval)",
    )
    bench_eval_parser.add_argument("--limit", type=int, help="Max samples to evaluate")
    bench_eval_parser.add_argument("--output", "-o", help="Output file path")
    bench_eval_parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=5,
        help="Samples per prompt for pass@k (default: 5)",
    )
    bench_eval_parser.add_argument(
        "--run-after-compile",
        action="store_true",
        help="Run compiled code (MVR mode). Default: compile-only (MVP)",
    )
    bench_eval_parser.add_argument(
        "--language",
        choices=["cpp", "rust", "go", "python"],
        help="Target language for native benchmarks",
    )
    bench_eval_parser.add_argument(
        "--verifier",
        choices=["gcc", "mingw", "clang", "rust", "go", "humaneval", "mbpp"],
        help="Verifier type",
    )

    # inference command
    inference_parser = subparsers.add_parser("inference", help="Inference optimization")
    inference_subparsers = inference_parser.add_subparsers(dest="inference_command", required=True)

    # inference optimize
    inf_optimize_parser = inference_subparsers.add_parser(
        "optimize", help="Optimize model for inference"
    )
    inf_optimize_parser.add_argument("--model", "-m", required=True, help="Model path")
    inf_optimize_parser.add_argument(
        "--target-precision",
        default="int4",
        choices=["int4", "int8", "fp16"],
        help="Target precision (default: int4)",
    )
    inf_optimize_parser.add_argument(
        "--target-latency", type=float, default=50.0, help="Target latency in ms (default: 50)"
    )
    inf_optimize_parser.add_argument("--calibration-data", help="Path to calibration data JSONL")
    inf_optimize_parser.add_argument(
        "--output", "-o", default="models/optimized", help="Output directory"
    )
    inf_optimize_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dependencies without running optimization",
    )

    # inference export
    inf_export_parser = inference_subparsers.add_parser(
        "export", help="Export model to deployment format"
    )
    inf_export_parser.add_argument("--model", "-m", required=True, help="Model path")
    inf_export_parser.add_argument(
        "--format", "-f", required=True, choices=["gguf", "onnx"], help="Export format"
    )
    inf_export_parser.add_argument(
        "--quantization", "-q", default="Q4_K_M", help="GGUF quantization type (default: Q4_K_M)"
    )
    inf_export_parser.add_argument("--output", "-o", required=True, help="Output path")

    # inference benchmark
    inf_bench_parser = inference_subparsers.add_parser(
        "benchmark", help="Benchmark inference latency"
    )
    inf_bench_parser.add_argument("--model", "-m", required=True, help="Model path")
    inf_bench_parser.add_argument("--prompts", "-p", help="Test prompts JSONL")
    inf_bench_parser.add_argument(
        "--num-prompts", type=int, default=10, help="Number of prompts to test"
    )
    inf_bench_parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens to generate"
    )
    inf_bench_parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    inf_bench_parser.add_argument(
        "--measure-memory", action="store_true", help="Measure memory usage"
    )

    # vlm command
    vlm_parser = subparsers.add_parser("vlm", help="Vision-Language Model training")
    vlm_subparsers = vlm_parser.add_subparsers(dest="vlm_command", required=True)

    # vlm train
    vlm_train_parser = vlm_subparsers.add_parser("train", help="Train VLM with RAFT")
    add_apple_runtime_flags(vlm_train_parser)
    add_dataset_training_flags(vlm_train_parser, consumes_verifier=True)
    vlm_train_parser.add_argument(
        "--model", "-m", default="Qwen/Qwen2-VL-7B-Instruct", help="VLM model name"
    )
    vlm_train_parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Dataset name (textvqa, docvqa, chartqa) or JSONL path",
    )
    vlm_train_parser.add_argument(
        "--output", "-o", default="models/vlm_raft", help="Output directory"
    )
    vlm_train_parser.add_argument("--cycles", type=int, default=6, help="Number of RAFT cycles")
    vlm_train_parser.add_argument(
        "--samples-per-prompt", type=int, default=4, help="Samples per prompt (default: 4)"
    )
    vlm_train_parser.add_argument(
        "--perception-weight",
        type=float,
        default=0.3,
        help="Weight for perception verification (default: 0.3)",
    )
    vlm_train_parser.add_argument(
        "--reasoning-weight",
        type=float,
        default=0.4,
        help="Weight for reasoning verification (default: 0.4)",
    )
    vlm_train_parser.add_argument(
        "--output-weight",
        type=float,
        default=0.3,
        help="Weight for output verification (default: 0.3)",
    )
    vlm_train_parser.add_argument(
        "--lr-decay", type=float, default=0.85, help="Learning rate decay per cycle (default: 0.85)"
    )
    vlm_train_parser.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature (default: 0.7)"
    )
    vlm_train_parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Maximum tokens to generate (default: 512)"
    )
    vlm_train_parser.add_argument(
        "--keep-percent",
        type=float,
        default=0.5,
        help="Keep top X%% of passing samples (default: 0.5)",
    )
    vlm_train_parser.add_argument(
        "--reward-threshold",
        type=float,
        default=0.5,
        help="Minimum reward to consider passing (default: 0.5)",
    )
    vlm_train_parser.add_argument("--limit", type=int, help="Limit dataset samples")
    vlm_train_parser.add_argument(
        "--resume-from-cycle",
        type=int,
        default=0,
        help="Resume training from this cycle index (default: 0)",
    )
    vlm_train_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic runs (default: 42)"
    )
    vlm_train_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and datasets without running training",
    )
    vlm_train_parser.add_argument(
        "--allow-prototype-train",
        action="store_true",
        help="Required while VLM training capability is prototype-gated",
    )

    # vlm benchmark
    vlm_bench_parser = vlm_subparsers.add_parser("benchmark", help="Benchmark VLM")
    vlm_bench_parser.add_argument("--model", "-m", required=True, help="VLM model path")
    vlm_bench_parser.add_argument(
        "--dataset", "-d", default="textvqa", help="Dataset name (default: textvqa)"
    )
    vlm_bench_parser.add_argument("--split", default="validation", help="Dataset split")
    vlm_bench_parser.add_argument(
        "--limit", type=int, default=100, help="Limit samples (default: 100)"
    )
    vlm_bench_parser.add_argument("--output", "-o", help="Output file for results")

    # vlm datasets
    vlm_datasets_parser = vlm_subparsers.add_parser("datasets", help="List available VLM datasets")

    # vlm sft
    vlm_sft_parser = vlm_subparsers.add_parser("sft", help="SFT training for VLM")
    vlm_sft_parser.add_argument(
        "--model", "-m", default="Qwen/Qwen2-VL-2B-Instruct", help="VLM model name"
    )
    vlm_sft_parser.add_argument(
        "--dataset", "-d", default="llava", help="Dataset name (default: llava)"
    )
    vlm_sft_parser.add_argument("--max-samples", type=int, help="Limit training samples")
    vlm_sft_parser.add_argument("--output", "-o", default="models/vlm_sft", help="Output directory")
    vlm_sft_parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    vlm_sft_parser.add_argument("--dry-run", action="store_true", help="Validate config only")

    # audio command
    audio_parser = subparsers.add_parser("audio", help="Audio-language training")
    audio_subparsers = audio_parser.add_subparsers(dest="audio_command", required=True)

    # audio datasets
    audio_datasets_parser = audio_subparsers.add_parser(
        "datasets", help="List available audio datasets"
    )

    # audio benchmark
    audio_bench_parser = audio_subparsers.add_parser("benchmark", help="Benchmark audio model")
    audio_bench_parser.add_argument(
        "--model",
        "-m",
        default="openai/whisper-small",
        help="Audio model (default: openai/whisper-small)",
    )
    audio_bench_parser.add_argument(
        "--dataset", "-d", default="librispeech", help="Dataset name (default: librispeech)"
    )
    audio_bench_parser.add_argument(
        "--task",
        "-t",
        default="asr",
        choices=["asr", "tts", "classification"],
        help="Task type (default: asr)",
    )
    audio_bench_parser.add_argument(
        "--limit", type=int, default=100, help="Limit samples (default: 100)"
    )
    audio_bench_parser.add_argument("--output", "-o", help="Output file for results")

    # audio train
    audio_train_parser = audio_subparsers.add_parser("train", help="Train audio model with RAFT")
    add_apple_runtime_flags(audio_train_parser)
    add_dataset_training_flags(audio_train_parser, consumes_verifier=True)
    audio_train_parser.add_argument(
        "--model",
        "-m",
        default="openai/whisper-small",
        help="Audio model (default: openai/whisper-small)",
    )
    audio_train_parser.add_argument(
        "--dataset", "-d", default="librispeech", help="Dataset name or path (default: librispeech)"
    )
    audio_train_parser.add_argument(
        "--task",
        "-t",
        default="asr",
        choices=["asr", "tts", "classification"],
        help="Task type (default: asr)",
    )
    audio_train_parser.add_argument(
        "--cycles", type=int, default=6, help="Number of RAFT cycles (default: 6)"
    )
    audio_train_parser.add_argument(
        "--limit",
        type=int,
        help="Limit training records (used by deterministic proof runs)",
    )
    audio_train_parser.add_argument(
        "--lr", type=float, default=5e-5, help="Initial learning rate (default: 5e-5)"
    )
    audio_train_parser.add_argument(
        "--lr-decay", type=float, default=0.85, help="Learning rate decay per cycle (default: 0.85)"
    )
    audio_train_parser.add_argument(
        "--samples-per-prompt", type=int, default=4, help="Samples per prompt (default: 4)"
    )
    audio_train_parser.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature (default: 0.7)"
    )
    audio_train_parser.add_argument(
        "--keep-percent",
        type=float,
        default=0.5,
        help="Keep top X%% of passing samples (default: 0.5)",
    )
    audio_train_parser.add_argument(
        "--reward-threshold",
        type=float,
        default=0.5,
        help="Minimum reward to consider passing (default: 0.5)",
    )
    audio_train_parser.add_argument(
        "--output",
        "-o",
        default="models/audio_raft",
        help="Output directory (default: models/audio_raft)",
    )
    audio_train_parser.add_argument(
        "--resume-from-cycle",
        type=int,
        default=0,
        help="Resume training from this cycle index (default: 0)",
    )
    audio_train_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic runs (default: 42)"
    )
    audio_train_parser.add_argument(
        "--dry-run", action="store_true", help="Validate config without running training"
    )
    audio_train_parser.add_argument(
        "--allow-prototype-train",
        action="store_true",
        help="Required while audio training capability is prototype-gated",
    )

    # audio sft
    audio_sft_parser = audio_subparsers.add_parser("sft", help="SFT training for audio")
    audio_sft_parser.add_argument(
        "--model",
        "-m",
        default="openai/whisper-small",
        help="Audio model (default: openai/whisper-small)",
    )
    audio_sft_parser.add_argument(
        "--dataset", "-d", default="librispeech_sft", help="Dataset name (default: librispeech_sft)"
    )
    audio_sft_parser.add_argument("--max-samples", type=int, help="Limit training samples")
    audio_sft_parser.add_argument(
        "--output", "-o", default="models/audio_sft", help="Output directory"
    )
    audio_sft_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    audio_sft_parser.add_argument("--dry-run", action="store_true", help="Validate config only")

    # reasoning command
    reasoning_parser = subparsers.add_parser("reasoning", help="Math/Reasoning training")
    reasoning_subparsers = reasoning_parser.add_subparsers(dest="reasoning_command", required=True)

    # reasoning datasets
    reasoning_datasets_parser = reasoning_subparsers.add_parser(
        "datasets", help="List available math datasets"
    )

    # reasoning benchmark
    reasoning_bench_parser = reasoning_subparsers.add_parser(
        "benchmark", help="Benchmark math reasoning"
    )
    reasoning_bench_parser.add_argument(
        "--model",
        "-m",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    reasoning_bench_parser.add_argument(
        "--dataset", "-d", default="gsm8k", help="Dataset name (default: gsm8k)"
    )
    reasoning_bench_parser.add_argument(
        "--split", default="test", help="Dataset split (default: test)"
    )
    reasoning_bench_parser.add_argument(
        "--limit", type=int, default=100, help="Limit samples (default: 100)"
    )
    reasoning_bench_parser.add_argument("--output", "-o", help="Output file for results")

    # reasoning train
    reasoning_train_parser = reasoning_subparsers.add_parser("train", help="Train with RAFT")
    add_apple_runtime_flags(reasoning_train_parser)
    add_dataset_training_flags(reasoning_train_parser, consumes_verifier=True)
    reasoning_train_parser.add_argument(
        "--model",
        "-m",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    reasoning_train_parser.add_argument(
        "--dataset", "-d", default="gsm8k", help="Dataset name (default: gsm8k)"
    )
    reasoning_train_parser.add_argument(
        "--cycles", type=int, default=4, help="Number of RAFT cycles (default: 4)"
    )
    reasoning_train_parser.add_argument(
        "--lr", type=float, default=1e-5, help="Initial learning rate (default: 1e-5)"
    )
    reasoning_train_parser.add_argument(
        "--lr-decay", type=float, default=0.85, help="Learning rate decay per cycle (default: 0.85)"
    )
    reasoning_train_parser.add_argument(
        "--samples-per-prompt", type=int, default=4, help="Samples per prompt (default: 4)"
    )
    reasoning_train_parser.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature (default: 0.7)"
    )
    reasoning_train_parser.add_argument(
        "--keep-percent",
        type=float,
        default=0.5,
        help="Keep top X%% of passing samples (default: 0.5)",
    )
    reasoning_train_parser.add_argument(
        "--output",
        "-o",
        default="models/reasoning_raft",
        help="Output directory (default: models/reasoning_raft)",
    )
    reasoning_train_parser.add_argument("--limit", type=int, help="Limit dataset samples")
    reasoning_train_parser.add_argument(
        "--resume-from-cycle",
        type=int,
        default=0,
        help="Resume training from this cycle index (default: 0)",
    )
    reasoning_train_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic runs (default: 42)"
    )
    reasoning_train_parser.add_argument(
        "--dry-run", action="store_true", help="Validate config without running training"
    )
    reasoning_train_parser.add_argument(
        "--allow-prototype-train",
        action="store_true",
        help="Required while reasoning training capability is prototype-gated",
    )

    # reasoning sft
    reasoning_sft_parser = reasoning_subparsers.add_parser("sft", help="SFT training for reasoning")
    reasoning_sft_parser.add_argument(
        "--model",
        "-m",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model name (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    reasoning_sft_parser.add_argument(
        "--dataset", "-d", default="metamath", help="Dataset name (default: metamath)"
    )
    reasoning_sft_parser.add_argument("--max-samples", type=int, help="Limit training samples")
    reasoning_sft_parser.add_argument(
        "--output", "-o", default="models/reasoning_sft", help="Output directory"
    )
    reasoning_sft_parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    reasoning_sft_parser.add_argument("--dry-run", action="store_true", help="Validate config only")

    # agentic command (tool calling)
    agentic_parser = subparsers.add_parser(
        "agentic", help="Tool calling / function calling training"
    )
    agentic_subparsers = agentic_parser.add_subparsers(dest="agentic_command", required=True)

    # agentic datasets
    agentic_datasets_parser = agentic_subparsers.add_parser(
        "datasets", help="List available tool calling datasets"
    )

    # agentic benchmark
    agentic_bench_parser = agentic_subparsers.add_parser(
        "benchmark", help="Benchmark tool calling model"
    )
    agentic_bench_parser.add_argument(
        "--model",
        "-m",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    agentic_bench_parser.add_argument(
        "--dataset", "-d", default="xlam", help="Dataset name: xlam, glaive (default: xlam)"
    )
    agentic_bench_parser.add_argument(
        "--limit", type=int, default=100, help="Limit samples (default: 100)"
    )
    agentic_bench_parser.add_argument("--output", "-o", help="Output file for results")

    # agentic train
    agentic_train_parser = agentic_subparsers.add_parser(
        "train", help="Train tool calling with RAFT"
    )
    add_apple_runtime_flags(agentic_train_parser)
    add_dataset_training_flags(agentic_train_parser, consumes_verifier=True)
    agentic_train_parser.add_argument(
        "--model",
        "-m",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    agentic_train_parser.add_argument(
        "--dataset", "-d", default="xlam", help="Dataset name: xlam, glaive (default: xlam)"
    )
    agentic_train_parser.add_argument(
        "--cycles", type=int, default=5, help="Number of RAFT cycles (default: 5)"
    )
    agentic_train_parser.add_argument(
        "--lr", type=float, default=5e-5, help="Initial learning rate (default: 5e-5)"
    )
    agentic_train_parser.add_argument(
        "--lr-decay", type=float, default=0.85, help="Learning rate decay per cycle (default: 0.85)"
    )
    agentic_train_parser.add_argument(
        "--samples-per-prompt", type=int, default=4, help="Samples per prompt (default: 4)"
    )
    agentic_train_parser.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature (default: 0.7)"
    )
    agentic_train_parser.add_argument(
        "--keep-percent",
        type=float,
        default=0.5,
        help="Keep top X%% of passing samples (default: 0.5)",
    )
    agentic_train_parser.add_argument(
        "--output",
        "-o",
        default="models/agentic_raft",
        help="Output directory (default: models/agentic_raft)",
    )
    agentic_train_parser.add_argument("--limit", type=int, help="Limit dataset samples")
    agentic_train_parser.add_argument(
        "--resume-from-cycle",
        type=int,
        default=0,
        help="Resume training from this cycle index (default: 0)",
    )
    agentic_train_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic runs (default: 42)"
    )
    agentic_train_parser.add_argument(
        "--dry-run", action="store_true", help="Validate config without running training"
    )
    agentic_train_parser.add_argument(
        "--allow-prototype-train",
        action="store_true",
        help="Required while agentic training capability is prototype-gated",
    )

    # agentic sft
    agentic_sft_parser = agentic_subparsers.add_parser("sft", help="SFT training for tool calling")
    agentic_sft_parser.add_argument(
        "--model",
        "-m",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    agentic_sft_parser.add_argument(
        "--dataset", "-d", default="xlam_sft", help="Dataset name (default: xlam_sft)"
    )
    agentic_sft_parser.add_argument("--max-samples", type=int, help="Limit training samples")
    agentic_sft_parser.add_argument(
        "--output", "-o", default="models/agentic_sft", help="Output directory"
    )
    agentic_sft_parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    agentic_sft_parser.add_argument("--dry-run", action="store_true", help="Validate config only")

    # info command
    info_parser = subparsers.add_parser("info", help="Show hardware info")

    # doctor command
    doctor_parser = subparsers.add_parser("doctor", help="Run environment readiness checks")
    doctor_subparsers = doctor_parser.add_subparsers(dest="doctor_command", required=True)
    doctor_mlx_parser = doctor_subparsers.add_parser(
        "mlx", help="Check Apple MLX package and Metal runtime readiness"
    )
    doctor_mlx_parser.add_argument("--json", action="store_true", help="Emit JSON")

    # models command
    models_parser = subparsers.add_parser("models", help="Browse curated base-model catalog")
    models_subparsers = models_parser.add_subparsers(dest="models_command", required=True)
    models_list_parser = models_subparsers.add_parser(
        "list", help="List recommended and compatible models"
    )
    models_list_parser.add_argument(
        "--mode", help="Filter by trainer/mode (sft, raft, dpo, grpo, vlm, audio, ...)"
    )
    models_list_parser.add_argument(
        "--backend", help="Filter by backend (cuda, rocm, mps, mlx, cpu)"
    )
    models_list_parser.add_argument(
        "--modality", help="Filter by modality (text, code, vision, audio)"
    )
    models_list_parser.add_argument(
        "--provider", help="Filter by provider (Qwen, Liquid AI, Meta, ...)"
    )
    models_list_parser.add_argument(
        "--status", help="Filter by status (recommended, compatible, experimental)"
    )
    models_list_parser.add_argument(
        "--memory-tier", help="Filter by memory tier (tiny, small, medium, large)"
    )
    models_list_parser.add_argument("--json", action="store_true", help="Emit JSON")
    models_show_parser = models_subparsers.add_parser("show", help="Show one model catalog entry")
    models_show_parser.add_argument("model_id", help="Model id, e.g. Qwen/Qwen2.5-Coder-3B")
    models_show_parser.add_argument("--json", action="store_true", help="Emit JSON")

    # plot command - visualization tools
    plot_parser = subparsers.add_parser("plot", help="Generate training/benchmark visualizations")
    plot_subparsers = plot_parser.add_subparsers(dest="plot_command")

    # plot training
    plot_training_parser = plot_subparsers.add_parser(
        "training", help="Generate charts from TensorBoard training logs"
    )
    plot_training_parser.add_argument(
        "log_dirs", nargs="+", help="TensorBoard log directory (e.g., models/code_sft/logs)"
    )
    plot_training_parser.add_argument(
        "--output", "-o", default=None, help="Output directory for charts"
    )
    plot_training_parser.add_argument(
        "--compare", action="store_true", help="Generate comparison charts for multiple runs"
    )
    plot_training_parser.add_argument(
        "--only",
        choices=["loss", "lr", "grad", "summary"],
        help="Generate only specific chart type",
    )
    plot_training_parser.add_argument(
        "--name", default=None, help="Override run name in chart titles"
    )

    # plot benchmarks
    plot_benchmarks_parser = plot_subparsers.add_parser(
        "benchmarks", help="Generate charts from benchmark results"
    )
    plot_benchmarks_parser.add_argument(
        "results_dir", help="Directory containing benchmark results"
    )
    plot_benchmarks_parser.add_argument(
        "--output", "-o", default=None, help="Output directory for charts"
    )

    # test command
    test_parser = subparsers.add_parser("test", help="Run pipeline validation tests")
    test_parser.add_argument(
        "--level",
        "-l",
        default="standard",
        choices=[
            "smoke",
            "standard",
            "full",
            "modality",
            "ops-e2e",
            "ops-burnin",
            "all-modules",
            'walkthroughs',
            "all-module-qualification",
            "all-module-bootstrap",
            "all-module-live",
        ],
        help="Test level: smoke, standard, full, modality, ops-e2e, ops-burnin, all-modules, walkthroughs, all-module-qualification, all-module-bootstrap, all-module-live",
    )
    test_parser.add_argument(
        "--model",
        "-m",
        default="Qwen/Qwen2.5-Coder-0.5B",
        help="Model to use for testing (default: Qwen2.5-Coder-0.5B)",
    )
    test_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output with detailed logging"
    )
    test_parser.add_argument(
        "--baseline-file",
        default="tests/baselines/modality_runtime_baseline.v1.json",
        help="Baseline JSON path for modality/ops-burnin/all-module-qualification drift checks",
    )
    test_parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write/overwrite modality, ops-burnin, or all-module-qualification baseline snapshot",
    )
    test_parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare modality, ops-burnin, or all-module-qualification run against baseline and fail on hard drift",
    )
    test_parser.add_argument(
        "--report-file",
        default="results/readiness/ops_e2e_launch_reliability.v1.json",
        help="Output report path for --level ops-e2e, ops-burnin, all-modules, walkthroughs, all-module-qualification, all-module-bootstrap, or all-module-live",
    )
    test_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail non-zero when module status is fail (used with --level ops-e2e, ops-burnin, all-modules, walkthroughs, all-module-qualification, all-module-bootstrap, all-module-live)",
    )
    test_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for ops-e2e/ops-burnin/all-modules/walkthroughs/all-module-qualification/all-module-bootstrap/all-module-live checks (default: 42)",
    )
    test_parser.add_argument(
        "--burnin-profile",
        default="tiny-v1",
        help="Burn-in profile for --level ops-burnin (default: tiny-v1)",
    )
    test_parser.add_argument(
        "--profile",
        default="bounded-v1",
        help="Readiness profile for --level all-modules (default: bounded-v1); walkthroughs uses contract-v1/live-local",
    )
    test_parser.add_argument(
        "--qualification-profile",
        default="contract-v1",
        choices=["contract-v1", "fixture-v1", "live-local"],
        help="Qualification profile for --level all-module-qualification (default: contract-v1)",
    )
    test_parser.add_argument(
        "--module",
        action="append",
        default=[],
        help="Filter module(s) for --level all-modules, walkthroughs, all-module-qualification, all-module-bootstrap, or all-module-live (repeatable)",
    )
    test_parser.add_argument(
        "--fixture-pack",
        default="",
        help="Fixture pack for ops-e2e/all-modules/all-module-qualification checks (e.g., v1 or tests/fixtures/.../v1)",
    )
    test_parser.add_argument(
        "--show-fix-commands",
        action="store_true",
        help="Emit parseable remediation command lines for qualification issues (--level all-module-qualification)",
    )
    test_parser.add_argument(
        "--bootstrap-profile",
        default="contract-v1",
        choices=["contract-v1", "live-local"],
        help="Bootstrap profile for --level all-module-bootstrap (default: contract-v1)",
    )
    test_parser.add_argument(
        "--output-root",
        default="results/bootstrap",
        help="Evidence output root for --level all-module-bootstrap or --level all-module-live (default: results/bootstrap)",
    )
    test_parser.add_argument(
        "--live-profile",
        default="live-smoke-v1",
        choices=["live-smoke-v1", "live-local"],
        help="Live execution profile for --level all-module-live (default: live-smoke-v1)",
    )
    test_parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute bounded command probes (used with --level walkthroughs and profile=live-local)",
    )

    # The legacy `halo-forge ui` command launched a NiceGUI web app —
    # retired in favor of the Vite + React frontend at `public_app/`.
    # If you got here from an old script: `cd public_app && npm run dev`.

    # Parse arguments and dispatch
    args = parser.parse_args()

    # Plumb --accelerator into HALOFORGE_BACKEND so every downstream
    # halo_forge.backend.get_backend() call (in trainers, inference,
    # public_api) sees the user's choice without requiring each subcommand
    # handler to thread the flag through manually. Note: distinct from
    # `args.backend` on the `data generate` subcommand (LLM API selection).
    accelerator_choice = getattr(args, "accelerator", "auto")
    if accelerator_choice and accelerator_choice != "auto":
        os.environ["HALOFORGE_BACKEND"] = accelerator_choice

    _dispatch_commands(args)


# =============================================================================
# Reasoning Commands
# =============================================================================


def cmd_reasoning_datasets(args):
    """List available math datasets."""
    from halo_forge.reasoning.data import list_math_datasets

    print_banner()

    print(f"\n{GREEN}Available Math/Reasoning Datasets{NC}")
    print("=" * 60)

    dataset_info = {
        "gsm8k": ("Grade School", "8.5K problems, 2-8 step solutions"),
        "math": ("Competition", "12.5K problems, 7 subjects, 5 levels"),
        "aime": ("Competition", "AIME problems (hard)"),
    }

    datasets = list_math_datasets()

    for name in datasets:
        level, desc = dataset_info.get(name, ("Unknown", "Math dataset"))
        print(f"  {name:12} [{level:12}] - {desc}")

    print()
    print("Usage:")
    print("  halo-forge reasoning benchmark --dataset gsm8k")
    print("  halo-forge reasoning train --dataset gsm8k --cycles 4 --seed 42")


def cmd_reasoning_sft(args):
    """SFT training for reasoning."""
    from halo_forge.sft.trainer import SFTTrainer, SFTConfig

    print_banner()
    print(f"{GREEN}Reasoning SFT Training{NC}")
    print("=" * 60)

    dataset = getattr(args, "dataset", "metamath")
    max_samples = getattr(args, "max_samples", None)
    dry_run = getattr(args, "dry_run", False)

    print(f"Model: {args.model}")
    print(f"Dataset: {dataset}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print()

    if dry_run:
        print(f"{YELLOW}Dry run mode - validating configuration only{NC}")
        from halo_forge.sft.datasets import get_sft_dataset_spec, is_huggingface_id

        spec = get_sft_dataset_spec(dataset)
        if spec:
            print(f"{GREEN}✓{NC} Dataset: {spec.name} ({spec.huggingface_id})")
        elif is_huggingface_id(dataset):
            print(f"{GREEN}✓{NC} HuggingFace dataset: {dataset}")
        else:
            print(f"{RED}✗{NC} Unknown dataset: {dataset}")
            sys.exit(1)
        print(f"{GREEN}Configuration valid!{NC}")
        return

    config = SFTConfig(
        model_name=args.model,
        dataset=dataset,
        max_samples=max_samples,
        output_dir=args.output,
        num_epochs=args.epochs,
    )

    trainer = SFTTrainer(config)
    summary = trainer.train()
    _print_completed_training_summary("reasoning_sft", args.output, summary)


def cmd_reasoning_benchmark(args):
    """Benchmark math reasoning model."""
    from halo_forge.reasoning import MathVerifier, ReasoningRAFTConfig
    from halo_forge.reasoning.data import load_math_dataset

    print_banner()

    print(f"\n{GREEN}Reasoning Benchmark{NC}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Limit: {args.limit}")

    # Load dataset
    try:
        dataset = load_math_dataset(args.dataset, split=args.split, limit=args.limit)
        print(f"\nLoaded {len(dataset)} samples from {args.dataset}")
    except Exception as e:
        print(f"\n{RED}Error loading dataset: {e}{NC}")
        sys.exit(1)

    # Load model
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"\nLoading model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=recommended_dtype(),
            device_map=get_device_map(),
            trust_remote_code=True,
        )
        print(f"Model loaded on {model.device}")
    except Exception as e:
        print(f"\n{RED}Error loading model: {e}{NC}")
        sys.exit(1)

    # Run benchmark
    verifier = MathVerifier()
    correct = 0
    total = 0
    total_reward = 0

    print(f"\nRunning benchmark...")
    from tqdm import tqdm

    for sample in tqdm(dataset, desc="Evaluating", unit="sample"):
        # Format prompt
        prompt = (
            f"Solve the following math problem step by step. "
            f"Put your final answer in \\boxed{{}}.\n\n"
            f"Problem: {sample.question}\n\nSolution:"
        )

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        completion = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        # Verify
        result = verifier.verify(sample.question, completion, sample.answer)

        total += 1
        total_reward += result.reward
        if result.success:
            correct += 1

    # Results
    accuracy = correct / total if total > 0 else 0
    avg_reward = total_reward / total if total > 0 else 0

    print(f"\n{GREEN}Results:{NC}")
    print(f"  Samples: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Average reward: {avg_reward:.3f}")

    if args.output:
        results = {
            "model": args.model,
            "dataset": args.dataset,
            "samples": total,
            "correct": correct,
            "accuracy": accuracy,
            "avg_reward": avg_reward,
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


def cmd_reasoning_train(args):
    """Train reasoning model with RAFT."""
    _prepare_reward_integrity(args, trainer="reasoning")
    if _enqueue_managed_reward_training(args, trainer="reasoning"):
        return
    profile_verifier = _prepare_profile_verifier(
        args, consumer="direct", modality="text", training=True
    )
    _apply_managed_dataset_args(args, "reasoning", "dataset")
    from halo_forge.reasoning import ReasoningRAFTTrainer, ReasoningRAFTConfig
    from halo_forge.reasoning.data import load_math_dataset

    print_banner()

    print(f"\n{GREEN}Reasoning RAFT Training{NC}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Cycles: {args.cycles}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")

    _enforce_modality_train_contract("reasoning", args)

    if args.dry_run:
        print(f"\n{YELLOW}Dry run mode - validating configuration only{NC}")

        # Check dependencies
        try:
            import sympy

            print(f"\n{GREEN}✓{NC} sympy installed")
        except ImportError:
            print(f"\n{RED}✗{NC} sympy not installed (pip install sympy)")

        # Check dataset
        try:
            from halo_forge.reasoning.data import list_math_datasets

            if args.dataset in list_math_datasets():
                print(f"{GREEN}✓{NC} Dataset: {args.dataset}")
            else:
                print(f"{RED}✗{NC} Unknown dataset: {args.dataset}")
        except Exception as e:
            print(f"{RED}✗{NC} Error: {e}")

        print(f"\n{GREEN}Configuration valid!{NC}")
        return

    # Create config
    config = ReasoningRAFTConfig(
        model_name=args.model,
        num_cycles=args.cycles,
        reward_threshold=getattr(args, "reward_threshold", 0.0),
        learning_rate=args.lr,
        lr_decay_per_cycle=args.lr_decay,
        output_dir=args.output,
        enable_neural_accelerators=getattr(args, "enable_neural_accelerators", False),
        seed=args.seed,
    )
    if profile_verifier is not None:
        contract = dict(getattr(args, "_verifier_profile_contract", {}) or {})
        config.reward_threshold = float(
            contract.get("threshold")
            if contract.get("threshold") is not None
            else contract.get("minimum", config.reward_threshold)
        )

    # Load dataset
    dataset = load_math_dataset(args.dataset, split="train", limit=args.limit)
    print(f"\nLoaded {len(dataset)} samples from {args.dataset}")

    # Train
    signal_session = _build_training_signal_session(
        args,
        trainer="reasoning",
        output_dir=config.output_dir,
        total_boundaries=config.num_cycles,
        reward_threshold=config.reward_threshold,
    )
    trainer = ReasoningRAFTTrainer(config, signal_sink=signal_session)
    if profile_verifier is not None:
        trainer.verifier = profile_verifier
    try:
        summary = trainer.train(
            list(dataset),
            resume_from_cycle=getattr(args, "resume_from_cycle", 0),
        )
    except ValueError as e:
        print(f"{RED}Training error: {e}{NC}")
        sys.exit(2)
    _enforce_training_outcome_or_exit("reasoning", summary)

    print(f"\n{GREEN}Training complete!{NC}")
    print(f"Final accuracy: {summary.get('final_accuracy', 0):.1%}")
    if summary.get("final_model_path"):
        print(f"Final model: {summary['final_model_path']}")
    _print_training_run_metadata(summary)
    total_steps = sum(int(c.get("train_steps_executed", 0)) for c in summary.get("cycles", []))
    print(f"Train steps executed: {total_steps}")
    print(f"Results saved to: {args.output}")
    _finalize_managed_training_replay(args, "reasoning", args.output, config, summary)


# =============================================================================
# Agentic / Tool Calling Commands
# =============================================================================


def cmd_agentic_datasets(args):
    """List available agentic/tool calling datasets."""
    from halo_forge.agentic.data import list_agentic_datasets

    print_banner()

    datasets = list_agentic_datasets()

    print(f"\n{GREEN}Available Agentic / Tool Calling Datasets{NC}")
    print("=" * 60)

    for key, info in datasets.items():
        print(f"\n  {CYAN}{key:<12}{NC} [{YELLOW}Tool Calling{NC}]")
        print(f"               {info['description']}")
        print(f"               HuggingFace: {info['hf_path']}")
        print(f"               Size: {info['size']}")

    print(f"\n{YELLOW}Note:{NC} Datasets are downloaded on first use via HuggingFace.")


def cmd_agentic_sft(args):
    """SFT training for tool calling."""
    from halo_forge.sft.trainer import SFTTrainer, SFTConfig

    print_banner()
    print(f"{GREEN}Agentic SFT Training{NC}")
    print("=" * 60)

    dataset = getattr(args, "dataset", "xlam_sft")
    max_samples = getattr(args, "max_samples", None)
    dry_run = getattr(args, "dry_run", False)

    print(f"Model: {args.model}")
    print(f"Dataset: {dataset}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print()

    if dry_run:
        print(f"{YELLOW}Dry run mode - validating configuration only{NC}")
        from halo_forge.sft.datasets import get_sft_dataset_spec, is_huggingface_id

        spec = get_sft_dataset_spec(dataset)
        if spec:
            print(f"{GREEN}✓{NC} Dataset: {spec.name} ({spec.huggingface_id})")
        elif is_huggingface_id(dataset):
            print(f"{GREEN}✓{NC} HuggingFace dataset: {dataset}")
        else:
            print(f"{RED}✗{NC} Unknown dataset: {dataset}")
            sys.exit(1)
        print(f"{GREEN}Configuration valid!{NC}")
        return

    dataset_path = Path(dataset).expanduser()
    config = SFTConfig(
        model_name=args.model,
        dataset=None if dataset_path.is_file() else dataset,
        train_file=str(dataset_path.resolve()) if dataset_path.is_file() else None,
        max_samples=max_samples,
        output_dir=args.output,
        num_epochs=args.epochs,
    )

    trainer = SFTTrainer(config)
    summary = trainer.train()
    _print_completed_training_summary("agentic_sft", args.output, summary)


def cmd_agentic_benchmark(args):
    """Run agentic/tool calling benchmark."""
    from halo_forge.agentic import AgenticRAFTTrainer, AgenticRAFTConfig
    from halo_forge.agentic.data import XLAMLoader, GlaiveLoader, LocalToolCallingLoader

    print_banner()

    print(f"\n{GREEN}Agentic / Tool Calling Benchmark{NC}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Limit: {args.limit}")

    # Load dataset
    dataset_path = Path(args.dataset).expanduser()
    if dataset_path.is_file():
        loader = LocalToolCallingLoader(dataset_path)
    elif args.dataset == "xlam":
        loader = XLAMLoader()
    elif args.dataset == "glaive":
        loader = GlaiveLoader()
    else:
        print(f"{RED}Unknown dataset: {args.dataset}{NC}")
        print("Available: xlam, glaive")
        sys.exit(1)

    print(f"\n{YELLOW}Loading dataset...{NC}")
    samples = loader.load(limit=args.limit)
    print(f"Loaded {len(samples)} samples")

    # Create trainer for benchmark
    config = AgenticRAFTConfig(
        model_name=args.model,
    )
    trainer = AgenticRAFTTrainer(config)

    print(f"\n{YELLOW}Loading model...{NC}")
    trainer.load_model()

    print(f"\n{YELLOW}Running benchmark...{NC}")
    results = trainer.benchmark(samples, limit=args.limit)

    print(f"\n{GREEN}Benchmark Results{NC}")
    print("=" * 60)
    print(f"  Total samples:     {results['total']}")
    print(f"  Correct:           {results['correct']} ({results['accuracy']:.1%})")
    print(f"  JSON valid:        {results['json_valid']} ({results['json_valid_rate']:.1%})")
    print(
        f"  Function correct:  {results['function_correct']} ({results['function_accuracy']:.1%})"
    )
    print(f"  Average reward:    {results['avg_reward']:.3f}")
    print(f"  False positives:   {results['false_positives']}")

    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


def cmd_agentic_train(args):
    """Train agentic/tool calling model with RAFT."""
    _prepare_reward_integrity(args, trainer="agentic")
    if _enqueue_managed_reward_training(args, trainer="agentic"):
        return
    profile_verifier = _prepare_profile_verifier(
        args, consumer="direct", modality="tool", training=True
    )
    _apply_managed_dataset_args(args, "agentic", "dataset")
    from halo_forge.agentic import AgenticRAFTTrainer, AgenticRAFTConfig
    from halo_forge.agentic.data import XLAMLoader, GlaiveLoader, LocalToolCallingLoader

    print_banner()

    print(f"\n{GREEN}Agentic / Tool Calling RAFT Training{NC}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Cycles: {args.cycles}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")

    _enforce_modality_train_contract("agentic", args)

    if args.dry_run:
        print(f"\n{YELLOW}Dry run mode - validating configuration only{NC}")

        # Check dependencies
        print(f"\n{GREEN}✓{NC} agentic module available")

        # Check dataset
        from halo_forge.agentic.data import list_agentic_datasets

        if Path(args.dataset).expanduser().is_file():
            print(f"{GREEN}✓{NC} Local Dataset Lab artifact: {args.dataset}")
        elif args.dataset in list_agentic_datasets():
            print(f"{GREEN}✓{NC} Dataset: {args.dataset}")
        else:
            print(f"{RED}✗{NC} Unknown dataset: {args.dataset}")

        print(f"\n{GREEN}Configuration valid!{NC}")
        return

    # Load dataset
    dataset_path = Path(args.dataset).expanduser()
    if dataset_path.is_file():
        loader = LocalToolCallingLoader(dataset_path)
    elif args.dataset == "xlam":
        loader = XLAMLoader()
    elif args.dataset == "glaive":
        loader = GlaiveLoader()
    else:
        print(f"{RED}Unknown dataset: {args.dataset}{NC}")
        sys.exit(1)

    print(f"\n{YELLOW}Loading dataset...{NC}")
    samples = loader.load(limit=args.limit)
    print(f"Loaded {len(samples)} samples")

    # Create config
    config = AgenticRAFTConfig(
        model_name=args.model,
        num_cycles=args.cycles,
        reward_threshold=getattr(args, "reward_threshold", 0.5),
        learning_rate=args.lr,
        lr_decay_per_cycle=args.lr_decay,
        output_dir=args.output,
        enable_neural_accelerators=getattr(args, "enable_neural_accelerators", False),
        seed=args.seed,
    )
    if profile_verifier is not None:
        contract = dict(getattr(args, "_verifier_profile_contract", {}) or {})
        config.reward_threshold = float(
            contract.get("threshold")
            if contract.get("threshold") is not None
            else contract.get("minimum", config.reward_threshold)
        )

    # Train
    signal_session = _build_training_signal_session(
        args,
        trainer="agentic",
        output_dir=config.output_dir,
        total_boundaries=config.num_cycles,
        reward_threshold=config.reward_threshold,
    )
    trainer = AgenticRAFTTrainer(config, signal_sink=signal_session)
    if profile_verifier is not None:
        trainer.verifier = profile_verifier
    try:
        results = trainer.train(
            samples,
            resume_from_cycle=getattr(args, "resume_from_cycle", 0),
        )
    except ValueError as e:
        print(f"{RED}Training error: {e}{NC}")
        sys.exit(2)
    total_steps = int(results.get("total_train_steps_executed", 0))
    final_loss = results.get("final_train_loss")
    _enforce_training_outcome_or_exit("agentic", results)

    print(f"\n{GREEN}Training complete!{NC}")
    print(f"Final accuracy: {results.get('final_success_rate', 0):.1%}")
    print(f"Final avg reward: {results.get('final_avg_reward', 0):.3f}")
    if results.get("final_model_path"):
        print(f"Final model: {results['final_model_path']}")
    _print_training_run_metadata(results)
    print(f"Train steps executed: {total_steps}")
    if isinstance(final_loss, (int, float)):
        print(f"Final train loss: {final_loss:.4f}")
    print(f"Results saved to: {args.output}")
    _finalize_managed_training_replay(args, "agentic", args.output, config, results)


# The test parser and dispatch logic is inside main() at line 1598
# These are the remaining handler functions that were placed after main()


def _dispatch_commands(args):
    """Dispatch to appropriate command handler."""

    from halo_forge.lab_v11_v15.cli import dispatch_future_lab
    from halo_forge.product_cli import dispatch_product_v17
    from halo_forge.training_plan.cli import dispatch_training_plan
    from halo_forge.managed_runtime.cli import dispatch_managed_runtime

    if dispatch_managed_runtime(args):
        return

    if dispatch_training_plan(args):
        return

    if dispatch_product_v17(args):
        return

    plan_revision_id = str(
        getattr(args, "training_plan_revision", None) or ""
    ).strip()
    if plan_revision_id:
        from halo_forge.run_db import get_database
        from halo_forge.training_plan import TrainingPlanService

        plan_service = TrainingPlanService(
            get_database(_verifier_database_path(args))
        )
        resolved_plan = plan_service.resolved_launch_payload(plan_revision_id)
        argv = tuple(sys.argv[1:])

        def explicitly_supplied(*flags):
            return any(
                token == flag or token.startswith(flag + "=")
                for token in argv
                for flag in flags
            )

        def same_plan_value(actual, expected):
            if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                return float(actual) == float(expected)
            return str(actual).strip().lower() == str(expected).strip().lower()

        explicit_contract = (
            (("--model", "-m"), "model", "training_plan_model_id"),
            (("--seed",), "seed", "seed"),
            (("--batch-size",), "batch_size", "batch_size"),
            (("--gradient-accumulation",), "gradient_accumulation", "gradient_accumulation_steps"),
            (("--learning-rate",), "learning_rate", "learning_rate"),
            (("--lr",), "lr", "learning_rate"),
            (("--max-steps",), "max_steps", "max_steps"),
            (("--max-samples",), "max_samples", "max_samples"),
            (("--max-seq-length",), "max_seq_length", "max_sequence_length"),
            (("--adaptation",), "adaptation", "adaptation"),
            (("--epochs",), "epochs", "epochs"),
            (("--cycles",), "cycles", "cycles"),
            (("--verifier-profile-revision",), "verifier_profile_revision", "verifier_profile_revision_id"),
            (("--reward-system-revision",), "reward_system_revision", "reward_system_revision_id"),
        )
        contradictory = []
        for flags, attribute, plan_key in explicit_contract:
            if not explicitly_supplied(*flags):
                continue
            actual = getattr(args, attribute, None)
            expected = resolved_plan.get(plan_key)
            if expected is None or not same_plan_value(actual, expected):
                contradictory.append(flags[0])
        if explicitly_supplied("--no-gradient-checkpointing") and bool(
            resolved_plan.get("gradient_checkpointing", True)
        ):
            contradictory.append("--no-gradient-checkpointing")
        if explicitly_supplied("--dataset", "-d", "--data", "--prompts", "--train-file"):
            contradictory.append("dataset input")
        if contradictory:
            raise ValueError(
                "--training-plan-revision conflicts with explicit options: "
                + ", ".join(sorted(set(contradictory)))
            )
        for key, value in resolved_plan.items():
            attribute = {
                "dataset_version_id": "dataset_version",
                "gradient_accumulation_steps": "gradient_accumulation",
                "max_sequence_length": "max_seq_length",
            }.get(key, key)
            if hasattr(args, attribute) or attribute in {
                "training_capacity_check_id",
                "model_preparation_id",
                "training_compute_shape_hash",
                "training_capacity_adjustment",
                "training_plan_content_hash",
                "training_plan_runtime_hash",
                "training_plan_recommendation_reasons",
                "training_plan_forecast",
                "resolved_model_commit",
                "model_preparation_manifest_hash",
                "training_plan_model_id",
                "training_path_revision_id",
                "training_path_certification_id",
                "proof_run",
            }:
                setattr(args, attribute, value)
        readiness = plan_service.readiness(plan_revision_id)
        setattr(
            args,
            "model_preparation_id",
            readiness.model_preparation.id if readiness.model_preparation else None,
        )
        setattr(
            args,
            "training_capacity_check_id",
            readiness.capacity_check.id if readiness.capacity_check else None,
        )
        revision = plan_service.get_revision(plan_revision_id)
        setattr(
            args,
            "training_compute_shape_hash",
            revision.compute_shape_hash if revision else None,
        )
        setattr(
            args,
            "training_capacity_adjustment",
            readiness.capacity_check.selected_adjustment
            if readiness.capacity_check
            else None,
        )
        setattr(
            args,
            "training_path_revision_id",
            revision.training_path_revision_id if revision else None,
        )
        setattr(
            args,
            "training_path_certification_id",
            revision.training_path_certification_id if revision else None,
        )

    if dispatch_future_lab(args):
        return

    # Commands that should have auto-logging enabled
    logged_commands = {
        ("raft", "train"): "raft_train",
        ("sft", "train"): "sft_train",
        ("cpt", "train"): "cpt_train",
        ("dpo", "train"): "dpo_train",
        ("orpo", "train"): "orpo_train",
        ("grpo", "train"): "grpo_train",
        ("rm", "train"): "rm_train",
        ("vlm", "train"): "vlm_train",
        ("audio", "train"): "audio_train",
        ("reasoning", "train"): "reasoning_train",
        ("agentic", "train"): "agentic_train",
        ("benchmark", "run"): "benchmark_run",
        ("benchmark", "full"): "benchmark_full",
        ("benchmark", "eval"): "benchmark_eval",
    }

    # Setup auto-logging for training/benchmark commands
    quiet = getattr(args, "quiet", False)
    subcommand = None
    if args.command == "raft":
        subcommand = getattr(args, "raft_command", None)
    elif args.command == "sft":
        subcommand = getattr(args, "sft_command", None)
    elif args.command == "cpt":
        subcommand = getattr(args, "cpt_command", None)
    elif args.command == "dpo":
        subcommand = getattr(args, "dpo_command", None)
    elif args.command == "orpo":
        subcommand = getattr(args, "orpo_command", None)
    elif args.command == "grpo":
        subcommand = getattr(args, "grpo_command", None)
    elif args.command == "rm":
        subcommand = getattr(args, "rm_command", None)
    elif args.command == "vlm":
        subcommand = getattr(args, "vlm_command", None)
    elif args.command == "audio":
        subcommand = getattr(args, "audio_command", None)
    elif args.command == "reasoning":
        subcommand = getattr(args, "reasoning_command", None)
    elif args.command == "agentic":
        subcommand = getattr(args, "agentic_command", None)
    elif args.command == "benchmark":
        subcommand = getattr(args, "bench_command", None)

    log_key = (args.command, subcommand) if subcommand else None
    if log_key in logged_commands:
        log_name = logged_commands[log_key]
        log_path = setup_auto_logging(log_name, quiet=quiet)
        if not quiet:
            print(f"Logging to: {log_path}")

    # Route to handler
    if args.command == "config":
        if args.config_command == "validate":
            cmd_config_validate(args)
    elif args.command == "data":
        if args.data_command == "prepare":
            cmd_data_prepare(args)
        elif args.data_command == "generate":
            cmd_data_generate(args)
        elif args.data_command == "validate":
            cmd_data_validate(args)
        elif args.data_command == "dedup":
            cmd_data_dedup(args)
        elif args.data_command == "score":
            cmd_data_score(args)
        elif args.data_command == "synthesize":
            cmd_data_synthesize(args)
        elif args.data_command == "scenarios":
            cmd_data_lab_scenarios(args)
        elif args.data_command == "inspect":
            cmd_data_lab_inspect(args)
        elif args.data_command == "import":
            cmd_data_lab_import(args)
        elif args.data_command == "extract":
            cmd_data_lab_extract(args)
        elif args.data_command == "corpus-profile":
            cmd_data_lab_corpus_profile(args)
        elif args.data_command == "add":
            cmd_data_lab_add(args)
        elif args.data_command == "list":
            cmd_data_lab_list(args)
        elif args.data_command == "show":
            cmd_data_lab_show(args)
        elif args.data_command == "build":
            cmd_data_lab_build(args)
        elif args.data_command == "versions":
            cmd_data_lab_versions(args)
        elif args.data_command == "preview":
            cmd_data_lab_preview(args)
        elif args.data_command == "export":
            cmd_data_lab_export(args)
        elif args.data_command == "materialize":
            cmd_data_lab_materialize(args)
        elif args.data_command == "jobs":
            cmd_data_lab_jobs(args)
        elif args.data_command == "render":
            cmd_data_lab_render(args)
        elif args.data_command == "compare":
            cmd_data_lab_compare(args)
        elif args.data_command == "mine":
            cmd_data_lab_mine(args)
    elif args.command == "sweep":
        cmd_sweep(args)
    elif args.command == "checkpoint-policy":
        cmd_checkpoint_policy(args)
    elif args.command == "jobs":
        cmd_jobs(args)
    elif args.command == "artifact":
        cmd_artifact(args)
    elif args.command == "storage":
        cmd_storage(args)
    elif args.command == "review":
        cmd_review(args)
    elif args.command == "verifier":
        from halo_forge.verifier_cli import cmd_verifier

        cmd_verifier(args)
    elif args.command == "reward":
        from halo_forge.reward_cli import cmd_reward

        cmd_reward(args)
    elif args.command == "sft":
        if args.sft_command == "train":
            cmd_sft_train(args)
        elif args.sft_command == "datasets":
            cmd_sft_datasets(args)
    elif args.command == "cpt":
        if args.cpt_command == "train":
            cmd_cpt_train(args)
    elif args.command == "dpo":
        if args.dpo_command == "train":
            cmd_dpo_train(args)
        elif args.dpo_command == "datasets":
            cmd_dpo_datasets(args)
    elif args.command == "orpo":
        if args.orpo_command == "train":
            cmd_orpo_train(args)
    elif args.command == "grpo":
        if args.grpo_command == "train":
            cmd_grpo_train(args)
    elif args.command == "rm":
        if args.rm_command == "train":
            cmd_rm_train(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command in {"dashboard", "app"}:
        cmd_dashboard(args)
    elif args.command == "serve-public":
        cmd_serve_public(args)
    elif args.command == "convert":
        cmd_convert(args)
    elif args.command == "merge":
        cmd_merge(args)
    elif args.command == "replay":
        cmd_replay(args)
    elif args.command == "token":
        cmd_token(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "probe":
        cmd_probe(args)
    elif args.command == "raft":
        if args.raft_command == "train":
            cmd_raft_train(args)
    elif args.command == "benchmark":
        if args.bench_command == "run":
            cmd_benchmark(args)
        elif args.bench_command == "full":
            if not args.model and not args.suite:
                print("Error: Either --model or --suite is required")
                print("Examples:")
                print("  halo-forge benchmark full --model Qwen/Qwen2.5-Coder-0.5B")
                print("  halo-forge benchmark full --suite all")
                sys.exit(1)
            cmd_benchmark_full(args)
        elif args.bench_command == "eval":
            cmd_benchmark_eval(args)
    elif args.command == "inference":
        if args.inference_command == "optimize":
            cmd_inference_optimize(args)
        elif args.inference_command == "export":
            cmd_inference_export(args)
        elif args.inference_command == "benchmark":
            cmd_inference_benchmark(args)
    elif args.command == "vlm":
        if args.vlm_command == "train":
            cmd_vlm_train(args)
        elif args.vlm_command == "benchmark":
            cmd_vlm_benchmark(args)
        elif args.vlm_command == "datasets":
            cmd_vlm_datasets(args)
        elif args.vlm_command == "sft":
            cmd_vlm_sft(args)
    elif args.command == "audio":
        if args.audio_command == "datasets":
            cmd_audio_datasets(args)
        elif args.audio_command == "benchmark":
            cmd_audio_benchmark(args)
        elif args.audio_command == "train":
            cmd_audio_train(args)
        elif args.audio_command == "sft":
            cmd_audio_sft(args)
    elif args.command == "reasoning":
        if args.reasoning_command == "datasets":
            cmd_reasoning_datasets(args)
        elif args.reasoning_command == "benchmark":
            cmd_reasoning_benchmark(args)
        elif args.reasoning_command == "train":
            cmd_reasoning_train(args)
        elif args.reasoning_command == "sft":
            cmd_reasoning_sft(args)
    elif args.command == "agentic":
        if args.agentic_command == "datasets":
            cmd_agentic_datasets(args)
        elif args.agentic_command == "benchmark":
            cmd_agentic_benchmark(args)
        elif args.agentic_command == "train":
            cmd_agentic_train(args)
        elif args.agentic_command == "sft":
            cmd_agentic_sft(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "models":
        cmd_models(args)
    elif args.command == "plot":
        if not hasattr(args, "plot_command") or not args.plot_command:
            print("Usage: halo-forge plot {training|benchmarks} ...")
            print("\nAvailable commands:")
            print("  training    Generate charts from TensorBoard training logs")
            print("  benchmarks  Generate charts from benchmark results")
        elif args.plot_command == "training":
            cmd_plot_training(args)
        elif args.plot_command == "benchmarks":
            cmd_plot_benchmarks(args)
    elif args.command == "test":
        cmd_test(args)


if __name__ == "__main__":
    main()
