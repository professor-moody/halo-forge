"""CLI parity for V17 readiness, guided repairs, support, and qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--database", help="Override the Halo Forge SQLite path")
    parser.add_argument("--root", help="Managed root (default: ~/.halo-forge)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")


def add_product_v17_parsers(
    subparsers: Any, data_subparsers: Any
) -> None:
    setup = subparsers.add_parser("setup", help="Check or safely repair workstation setup")
    setup_actions = setup.add_subparsers(dest="setup_action", required=True)
    setup_check = setup_actions.add_parser("check", help="Check workstation readiness")
    _common(setup_check)
    setup_fix = setup_actions.add_parser("fix", help="Apply one safe setup remediation")
    setup_fix.add_argument(
        "action",
        nargs="?",
        default="create_managed_directories",
        help="Remediation action (default: create_managed_directories)",
    )
    _common(setup_fix)

    repair = data_subparsers.add_parser(
        "repair", help="Inspect and publish non-destructive dataset repairs"
    )
    repair_actions = repair.add_subparsers(dest="repair_action", required=True)
    inspect = repair_actions.add_parser("inspect", help="Scan a source for repairable issues")
    inspect.add_argument("source", nargs="?", help="Local source file or manifest")
    inspect.add_argument("--source-id")
    inspect.add_argument("--inspection-id")
    inspect.add_argument("--dataset-version")
    inspect.add_argument("--scenario-revision")
    _common(inspect)

    preview = repair_actions.add_parser("preview", help="Run an exact reviewed repair preview")
    preview.add_argument("session_id")
    preview.add_argument("--spec", help="JSON/YAML repair plan with an actions list")
    preview.add_argument(
        "--quarantine", action="append", default=[], metavar="RECORD_ID[@SOURCE_INDEX]"
    )
    preview.add_argument(
        "--exclude", action="append", default=[], metavar="RECORD_ID[@SOURCE_INDEX]"
    )
    preview.add_argument("--trim", action="store_true", help="Trim strings and empty chat turns")
    preview.add_argument("--normalize-roles", action="store_true")
    preview.add_argument("--media-root")
    preview.add_argument("--reason", default="Operator-reviewed deterministic repair")
    _common(preview)

    apply = repair_actions.add_parser("apply", help="Publish an exact preview immutably")
    apply.add_argument("preview_id")
    _common(apply)
    rebase = repair_actions.add_parser("rebase", help="Rebase a repair after source changes")
    rebase.add_argument("session_id")
    _common(rebase)

    support = subparsers.add_parser("support", help="Create privacy-safe support artifacts")
    support_group = support.add_subparsers(dest="support_group", required=True)
    bundle = support_group.add_parser("bundle", help="Preview, create, or verify a support bundle")
    bundle_actions = bundle.add_subparsers(dest="bundle_action", required=True)
    for action in ("preview", "create"):
        command = bundle_actions.add_parser(action, help=f"{action.title()} a support bundle")
        command.add_argument("--category", action="append", default=[])
        _common(command)
    verify = bundle_actions.add_parser("verify", help="Verify bundle checksums")
    verify.add_argument("bundle_id")
    _common(verify)

    release = subparsers.add_parser("release", help="Record release qualification evidence")
    release_actions = release.add_subparsers(dest="release_action", required=True)
    qualify = release_actions.add_parser("qualify", help="Qualify a platform package candidate")
    qualify.add_argument("--package", required=True, help="Candidate package path")
    qualify.add_argument("--package-type", required=True)
    qualify.add_argument("--signature-state", default="unsigned")
    qualify.add_argument("--smoke-status", choices=["passed", "failed"], required=True)
    qualify.add_argument("--backend", action="append", default=[])
    _common(qualify)
    workstation = release_actions.add_parser(
        "workstation-certify", help="Snapshot real beta-qualification evidence"
    )
    workstation.add_argument("--runtime-revision", required=True)
    workstation.add_argument("--evidence", help="JSON file containing completed evidence IDs")
    workstation.add_argument(
        "--wait", action="store_true", help="Run the durable report job in this process"
    )
    _common(workstation)
    report = release_actions.add_parser(
        "workstation-report", help="Show a workstation certification report"
    )
    report.add_argument("certification_id")
    _common(report)


def _service(args: Any):
    from halo_forge.product_lab import ProductLabService
    from halo_forge.run_db import get_database

    root = Path(str(getattr(args, "root", None) or Path.home() / ".halo-forge")).expanduser()
    return ProductLabService(get_database(getattr(args, "database", None)), root=root)


def _load_spec(path: str) -> Mapping[str, Any]:
    source = Path(path).expanduser()
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional CLI dependency
            raise RuntimeError("PyYAML is required for YAML repair plans") from exc
        value = yaml.safe_load(text)
    else:
        value = json.loads(text)
    if not isinstance(value, Mapping):
        raise ValueError("Repair plan must be an object")
    return value


def _emit(args: Any, value: Any, *, headline: str | None = None) -> None:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if getattr(args, "json", False):
        print(json.dumps(value, indent=2, sort_keys=True, default=str))
        return
    if headline:
        print(headline)
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"checks", "remediations", "sample", "manifest", "actions"}:
                continue
            print(f"{key.replace('_', ' ').title()}: {item}")
    else:
        print(value)


def _repair_preview_spec(args: Any) -> Mapping[str, Any]:
    if args.spec:
        return _load_spec(args.spec)
    actions: list[dict[str, Any]] = []
    def selector(value: str) -> dict[str, Any]:
        record_id, separator, raw_index = value.rpartition("@")
        if separator and raw_index.isdigit():
            return {"record_id": record_id, "source_index": int(raw_index)}
        return {"record_id": value}

    for record_id in args.quarantine:
        actions.append(
            {"action_kind": "quarantine", **selector(record_id), "reason": args.reason}
        )
    for record_id in args.exclude:
        actions.append(
            {"action_kind": "exclude", **selector(record_id), "reason": args.reason}
        )
    if args.trim:
        actions.append({"action_kind": "trim", "reason": args.reason})
    if args.normalize_roles:
        actions.append({"action_kind": "normalize_roles", "reason": args.reason})
    if args.media_root:
        actions.append(
            {"action_kind": "media_root", "value": args.media_root, "reason": args.reason}
        )
    if not actions:
        raise ValueError("Choose at least one repair action or pass --spec")
    return {"actions": actions}


def dispatch_product_v17(args: Any) -> bool:
    command = getattr(args, "command", None)
    if command == "setup":
        service = _service(args)
        if args.setup_action == "check":
            value = service.assess_readiness()
        else:
            value = service.apply_setup_remediation(args.action)
        _emit(args, value, headline="Workstation readiness")
        return True

    if command == "data" and getattr(args, "data_command", None) == "repair":
        service = _service(args)
        if args.repair_action == "inspect":
            payload = {
                key: value
                for key, value in {
                    "source_uri": args.source,
                    "source_id": args.source_id,
                    "inspection_id": args.inspection_id,
                    "dataset_version_id": args.dataset_version,
                    "scenario_revision_id": args.scenario_revision,
                    "scan": True,
                }.items()
                if value not in (None, "")
            }
            if not any(
                payload.get(key)
                for key in ("source_uri", "source_id", "inspection_id", "dataset_version_id")
            ):
                raise ValueError("Provide a source path or managed source/inspection/version")
            value = service.create_repair_session(payload, enqueue=False)
        elif args.repair_action == "preview":
            plan = service.create_repair_plan(args.session_id, _repair_preview_spec(args))
            value = service.prepare_repair_preview(
                args.session_id, plan.id, enqueue=False
            )
        elif args.repair_action == "apply":
            value = service.publish_repair_revision(args.preview_id)
        else:
            value = service.rebase_repair_session(args.session_id)
        _emit(args, value, headline="Dataset repair")
        return True

    if command == "support":
        service = _service(args)
        if args.bundle_action == "preview":
            value = service.support_bundle_preview(args.category or None)
        elif args.bundle_action == "create":
            value = service.create_support_bundle(args.category or None, enqueue=False)
        else:
            value = service.verify_support_bundle(args.bundle_id)
        _emit(args, value, headline="Support bundle")
        return True

    if command == "release":
        if args.release_action in {"workstation-certify", "workstation-report"}:
            from halo_forge.managed_runtime import ManagedRuntimeService
            from halo_forge.run_db import get_database
            from halo_forge.training_path_certification import TrainingPathCertificationService

            root = Path(str(getattr(args, "root", None) or Path.home() / ".halo-forge")).expanduser()
            database = get_database(getattr(args, "database", None))
            certification = TrainingPathCertificationService(
                database,
                root=root / "certifications",
                runtime_service=ManagedRuntimeService(database, root=root / "runtimes"),
            )
            if args.release_action == "workstation-certify":
                evidence = (
                    json.loads(Path(args.evidence).expanduser().read_text(encoding="utf-8"))
                    if args.evidence
                    else {}
                )
                value = certification.workstation_certify(
                    args.runtime_revision,
                    evidence=evidence,
                    enqueue=not args.wait,
                )
            else:
                value = certification.get_workstation_certification(args.certification_id)
                if value is None:
                    raise KeyError(args.certification_id)
            _emit(args, value, headline="Workstation certification")
            return True
        service = _service(args)
        value = service.qualify_release(
            {
                "package_path": args.package,
                "package_type": args.package_type,
                "signature_state": args.signature_state,
                "smoke_status": args.smoke_status,
                "supported_backends": args.backend,
            }
        )
        _emit(args, value, headline="Release qualification")
        return True
    return False


__all__: Sequence[str] = ("add_product_v17_parsers", "dispatch_product_v17")
