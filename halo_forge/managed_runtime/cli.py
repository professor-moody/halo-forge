"""CLI parity for managed runtimes and accelerator availability."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping


def add_managed_runtime_parsers(subparsers: Any) -> None:
    runtime = subparsers.add_parser(
        "runtime", help="Prepare and qualify managed accelerator runtimes"
    )
    actions = runtime.add_subparsers(dest="runtime_action", required=True)
    listing = actions.add_parser("list", help="List managed runtime profiles")
    show = actions.add_parser("show", help="Show a runtime profile or revision")
    show.add_argument("identifier")
    prepare = actions.add_parser("prepare", help="Confirm, download, and build a pinned runtime")
    prepare.add_argument("revision_id")
    prepare.add_argument("--wait", action="store_true")
    qualify = actions.add_parser("qualify", help="Run the hardware qualification ladder")
    qualify.add_argument("revision_id")
    qualify.add_argument("--wait", action="store_true")
    verify = actions.add_parser("verify", help="Verify qualification checksums and runtime identity")
    verify.add_argument("qualification_id")
    paths = actions.add_parser("paths", help="Show real trainer-path certification states")
    paths.add_argument("--family", choices=("rocm", "cuda"))
    certify = actions.add_parser("certify", help="Verify one path through its real trainer")
    certify.add_argument("path_revision_id")
    certify.add_argument("--runtime-revision")
    certify.add_argument("--wait", action="store_true")
    certification = actions.add_parser("certification", help="Inspect or recover a path certification")
    certification_actions = certification.add_subparsers(dest="certification_action", required=True)
    cert_show = certification_actions.add_parser("show")
    cert_steps = certification_actions.add_parser("steps")
    cert_verify = certification_actions.add_parser("verify")
    cert_retry = certification_actions.add_parser("retry")
    for parser in (cert_show, cert_steps, cert_verify, cert_retry):
        parser.add_argument("certification_id")
    cert_retry.add_argument("--reason", required=True)
    for parser in (listing, show, prepare, qualify, verify, paths, certify, cert_show, cert_steps, cert_verify, cert_retry):
        parser.add_argument("--database")
        parser.add_argument("--root")
        parser.add_argument("--json", action="store_true")

    accelerator = subparsers.add_parser(
        "accelerator", help="Inspect or wait for safe accelerator availability"
    )
    accelerator_actions = accelerator.add_subparsers(
        dest="accelerator_action", required=True
    )
    status = accelerator_actions.add_parser("status", help="Inspect external compute occupancy")
    wait = accelerator_actions.add_parser("wait", help="Wait for three verified idle samples")
    for parser in (status, wait):
        parser.add_argument("--family", choices=("rocm", "cuda"))
        parser.add_argument("--json", action="store_true")
    wait.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Seconds to wait; zero performs one three-sample check",
    )


def _serializable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _serializable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def _emit(args: Any, value: Any) -> None:
    value = _serializable(value)
    if getattr(args, "json", False):
        print(json.dumps(value, indent=2, sort_keys=True, default=str))
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(item, (dict, list, tuple)):
                print(f"{key.replace('_', ' ').title()}: {item}")
    elif isinstance(value, (list, tuple)):
        for item in value:
            payload = _serializable(item)
            print(
                f"{payload.get('name') or payload.get('id')} — "
                f"{payload.get('accelerator_family') or payload.get('status')}"
            )


def _family(args: Any) -> str:
    if getattr(args, "family", None):
        return str(args.family)
    try:
        from halo_forge.backend import get_backend

        backend = str(get_backend().name).lower()
    except Exception:
        backend = ""
    if backend.startswith("rocm"):
        return "rocm"
    if backend == "cuda":
        return "cuda"
    raise ValueError("--family is required when ROCm or CUDA is not the active backend")


def dispatch_managed_runtime(args: Any) -> bool:
    if getattr(args, "command", None) == "accelerator":
        from .occupancy import probe_accelerator, wait_for_stable_idle

        family = _family(args)
        if args.accelerator_action == "status":
            value = probe_accelerator(family)
        else:
            deadline = time.monotonic() + max(0.0, float(args.timeout))
            while True:
                idle, samples = wait_for_stable_idle(family)
                if idle or time.monotonic() >= deadline:
                    value = {
                        "idle": idle,
                        "samples": [sample.to_dict() for sample in samples],
                    }
                    break
                time.sleep(min(15.0, max(0.0, deadline - time.monotonic())))
        _emit(args, value)
        return True
    if getattr(args, "command", None) != "runtime":
        return False
    from halo_forge.run_db import get_database
    from halo_forge.workstation_jobs import WorkstationScheduler
    from .service import ManagedRuntimeService

    database = get_database(getattr(args, "database", None))
    scheduler = WorkstationScheduler(database)
    service = ManagedRuntimeService(
        database,
        root=Path(
            getattr(args, "root", None)
            or Path.home() / ".halo-forge" / "runtimes"
        ),
        scheduler=scheduler,
    )
    action = args.runtime_action
    if action == "list":
        value = service.list_profiles()
    elif action == "show":
        value = service.get_revision(args.identifier) or service.get_profile(args.identifier)
        if value is None:
            raise KeyError(args.identifier)
    elif action == "prepare":
        value = service.prepare(args.revision_id, confirmed=True, enqueue=not args.wait)
        if args.wait:
            value = service.run_preparation(value.id)
    elif action == "qualify":
        value = service.qualify(args.revision_id, enqueue=not args.wait)
        if args.wait:
            value = service.run_qualification(value.id)
    elif action == "verify":
        value = service.verify(args.qualification_id)
    else:
        from halo_forge.training_path_certification import TrainingPathCertificationService

        configured_runtime_root = Path(
            getattr(args, "root", None)
            or Path.home() / ".halo-forge" / "runtimes"
        ).expanduser()
        certification_service = TrainingPathCertificationService(
            database,
            root=configured_runtime_root.parent / "certifications",
            runtime_service=service,
            scheduler=scheduler,
        )
        if action == "paths":
            value = certification_service.capabilities(args.family or _family(args))
        elif action == "certify":
            path = certification_service.get_revision(args.path_revision_id)
            if path is None:
                raise KeyError(args.path_revision_id)
            runtime_revision_id = str(args.runtime_revision or "").strip()
            if not runtime_revision_id:
                matrix = certification_service.capabilities(path.runtime_family)
                candidate = next((item.runtime_revision_id for item in matrix.paths if item.path_revision_id == path.id), None)
                if not candidate:
                    raise ValueError("Prepare and qualify the matching runtime first")
                runtime_revision_id = candidate
            value = certification_service.certify(
                path.id, runtime_revision_id, enqueue=not args.wait
            )
            if args.wait:
                value = certification_service.run_certification(value.id)
        else:
            subaction = args.certification_action
            if subaction == "show":
                value = certification_service.get_certification(args.certification_id)
            elif subaction == "steps":
                certification_value = certification_service.get_certification(args.certification_id)
                if certification_value is None:
                    raise KeyError(args.certification_id)
                value = certification_value.steps
            elif subaction == "verify":
                value = certification_service.verify(args.certification_id)
            else:
                value = certification_service.retry(args.certification_id, reason=args.reason)
            if value is None:
                raise KeyError(args.certification_id)
    _emit(args, value)
    return True


__all__ = ["add_managed_runtime_parsers", "dispatch_managed_runtime"]
