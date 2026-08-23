"""CLI transport for Reward Integrity and Training Signal Studio.

The handlers stay deliberately thin: immutable validation, audit decisions,
reuse, and evidence verification live in :mod:`halo_forge.reward_integrity` so
the dashboard, API, and command line cannot develop different policies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional


def _json_value(value: Optional[str], *, default: Any) -> Any:
    if value in {None, ""}:
        return default
    path = Path(str(value)).expanduser()
    text = path.read_text(encoding="utf-8") if path.is_file() else str(value)
    return json.loads(text)


def _leaf(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--database", help="SQLite catalog path")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def add_reward_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "reward", help="Capture and audit verifier-guided training signals"
    )
    commands = parser.add_subparsers(dest="reward_command", required=True)
    _leaf(commands.add_parser("capabilities", help="Show trainer signal capabilities"))

    system = commands.add_parser("system", help="Manage immutable reward systems")
    system_commands = system.add_subparsers(dest="reward_system_action", required=True)
    s_list = _leaf(system_commands.add_parser("list"))
    s_list.add_argument("--modality")
    s_list.add_argument("--task-type")
    s_list.add_argument("--limit", type=int, default=100)
    s_list.add_argument("--offset", type=int, default=0)
    s_show = _leaf(system_commands.add_parser("show"))
    s_show.add_argument("system_or_revision_id")
    s_validate = _leaf(system_commands.add_parser("validate"))
    s_validate.add_argument("--spec", required=True)

    def system_fields(value: argparse.ArgumentParser, *, revise: bool = False) -> None:
        if revise:
            value.add_argument("system_id")
        else:
            value.add_argument("--name", required=True)
            value.add_argument("--description")
        value.add_argument("--spec", help="JSON file or inline revision definition")
        value.add_argument("--optimizer-verifier-revision")
        value.add_argument("--primary-sentinel-revision")
        value.add_argument(
            "--diagnostic-auditor-revision", action="append", default=[]
        )
        value.add_argument("--modality", default="text")
        value.add_argument("--task-type", default="binary")
        value.add_argument("--reward-mapping", help="JSON reward normalization/shaping")
        value.add_argument("--input-mapping", help="JSON input/context mapping")
        _leaf(value)

    system_fields(system_commands.add_parser("create"))
    system_fields(system_commands.add_parser("revise"), revise=True)

    protocol = commands.add_parser("protocol", help="Manage signal-retention protocols")
    protocol_commands = protocol.add_subparsers(dest="reward_protocol_action", required=True)
    p_list = _leaf(protocol_commands.add_parser("list"))
    p_list.add_argument("--limit", type=int, default=100)
    p_list.add_argument("--offset", type=int, default=0)
    p_show = _leaf(protocol_commands.add_parser("show"))
    p_show.add_argument("protocol_or_revision_id")
    for action in ("create", "revise"):
        value = _leaf(protocol_commands.add_parser(action))
        if action == "create":
            value.add_argument("--name", required=True)
            value.add_argument("--description")
        else:
            value.add_argument("protocol_id")
        value.add_argument("--spec", required=True)

    profile = commands.add_parser(
        "integrity-profile", help="Manage pass/warn/fail audit policies"
    )
    profile_commands = profile.add_subparsers(
        dest="reward_integrity_profile_action", required=True
    )
    ip_list = _leaf(profile_commands.add_parser("list"))
    ip_list.add_argument("--limit", type=int, default=100)
    ip_list.add_argument("--offset", type=int, default=0)
    ip_show = _leaf(profile_commands.add_parser("show"))
    ip_show.add_argument("profile_or_revision_id")
    for action in ("create", "revise"):
        value = _leaf(profile_commands.add_parser(action))
        if action == "create":
            value.add_argument("--name", required=True)
            value.add_argument("--description")
        else:
            value.add_argument("profile_id")
        value.add_argument(
            "--template",
            choices=["strict_integrity", "human_aligned_integrity", "exploratory", "custom"],
            default="human_aligned_integrity",
        )
        value.add_argument("--spec", help="Custom requirements JSON")

    trace = commands.add_parser("trace", help="Inspect sealed training-signal shards")
    trace_commands = trace.add_subparsers(dest="reward_trace_action", required=True)
    t_list = _leaf(trace_commands.add_parser("list"))
    t_list.add_argument("--run-id")
    t_list.add_argument("--limit", type=int, default=100)
    t_list.add_argument("--offset", type=int, default=0)
    for action in ("show", "verify"):
        value = _leaf(trace_commands.add_parser(action))
        value.add_argument("trace_id")

    audit = commands.add_parser("audit", help="Run same-output sentinel audits")
    audit_commands = audit.add_subparsers(dest="reward_audit_action", required=True)
    a_list = _leaf(audit_commands.add_parser("list"))
    a_list.add_argument("--run-id")
    a_list.add_argument("--status")
    a_list.add_argument("--limit", type=int, default=100)
    a_list.add_argument("--offset", type=int, default=0)
    a_create = _leaf(audit_commands.add_parser("create"))
    a_create.add_argument("--trace", required=True)
    a_create.add_argument("--reward-system-revision", required=True)
    a_create.add_argument("--protocol-revision", required=True)
    a_create.add_argument("--integrity-profile-revision", required=True)
    a_create.add_argument("--development-suite-revision")
    a_create.add_argument("--runtime", help="Runtime identity JSON")
    a_create.add_argument("--wait", action="store_true")
    for action in ("show", "verify", "cancel"):
        value = _leaf(audit_commands.add_parser(action))
        value.add_argument("audit_id")
    a_retry = _leaf(audit_commands.add_parser("retry"))
    a_retry.add_argument("audit_id")
    a_retry.add_argument(
        "--reason", required=True, help="Recorded reason for this forced retry"
    )
    for action in ("samples", "metrics"):
        value = _leaf(audit_commands.add_parser(action))
        value.add_argument("audit_id")
        value.add_argument("--limit", type=int, default=100)
        value.add_argument("--offset", type=int, default=0)
        if action == "samples":
            value.add_argument("--population")
            value.add_argument("--outcome")
            value.add_argument("--query")
        else:
            value.add_argument("--subgroup")
    a_compare = _leaf(audit_commands.add_parser("compare"))
    a_compare.add_argument("base_id")
    a_compare.add_argument("candidate_id")
    a_compare.add_argument("--limit", type=int, default=100)
    a_compare.add_argument("--offset", type=int, default=0)
    a_review = _leaf(audit_commands.add_parser("review"))
    a_review.add_argument("audit_id")
    a_review.add_argument(
        "--action", choices=["continue", "stop", "fork", "create_review_proposal"], required=True
    )
    a_review.add_argument("--reason", required=True)
    a_review.add_argument("--checkpoint")
    return parser


def _service(args: Any):
    from halo_forge.reward_integrity import RewardIntegrityService
    from halo_forge.run_db import RunDatabase, get_database

    db = RunDatabase(str(Path(args.database).expanduser())) if args.database else get_database()
    return RewardIntegrityService(db)


def _value(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    return value


def _emit(value: Any, *, as_json: bool) -> None:
    payload = _value(value)
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    if isinstance(payload, Mapping) and isinstance(payload.get("items"), list):
        for item in payload["items"]:
            if isinstance(item, Mapping):
                identity = item.get("id") or item.get("name") or item.get("key")
                state = item.get("status") or item.get("capture_fidelity") or ""
                print(f"{identity}\t{state}".rstrip())
            else:
                print(item)
        return
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _system_definition(args: Any) -> dict[str, Any]:
    if args.spec:
        return dict(_json_value(args.spec, default={}))
    auditors = [
        {
            "role": "primary_sentinel",
            "verifier_revision_id": args.primary_sentinel_revision,
        }
    ]
    auditors.extend(
        {
            "role": "diagnostic",
            "verifier_revision_id": revision,
        }
        for revision in args.diagnostic_auditor_revision
    )
    return {
        "optimizer_verifier_revision_id": args.optimizer_verifier_revision,
        "modality": args.modality,
        "task_type": args.task_type,
        "input_mapping": _json_value(args.input_mapping, default={}),
        "reward_mapping": _json_value(args.reward_mapping, default={}),
        "auditors": auditors,
    }


def _validated_system_definition(service: Any, args: Any) -> dict[str, Any]:
    """Reject malformed identities before creating the catalog head.

    Correlated auditors are intentionally still publishable for inspection;
    the reward service marks them non-gating.  Missing or unknown immutable
    verifier revisions, however, must not leave an orphan reward-system row.
    """

    definition = _system_definition(args)
    validation = service.validate_system_definition(definition)
    blockers = [
        str(value)
        for value in validation.get("blockers", [])
        if not str(value).startswith("primary_sentinel_correlated:")
    ]
    if blockers:
        raise ValueError("invalid reward system: " + "; ".join(blockers))
    return definition


def _revise_system(service: Any, system_id: str, definition: Mapping[str, Any]) -> Any:
    """Translate the public ordered spec into the core revision arguments."""

    raw = dict(definition)
    return service.create_system_revision(
        system_id,
        optimizer_verifier_revision_id=str(
            raw.pop("optimizer_verifier_revision_id")
        ),
        modality=str(raw.pop("modality", "text")),
        task_type=str(raw.pop("task_type", "binary")),
        auditors=list(raw.pop("auditors", [])),
        input_mapping=dict(raw.pop("input_mapping", {})),
        reward_mapping=dict(raw.pop("reward_mapping", {})),
        definition=raw,
    )


def _detail(service: Any, noun: str, identifier: str) -> Any:
    detail = getattr(service, f"get_{noun}_detail", None)
    if detail is not None:
        return detail(identifier)
    try:
        return getattr(service, f"get_{noun}")(identifier)
    except KeyError:
        return getattr(service, f"get_{noun}_revision")(identifier)


def cmd_reward(args: Any) -> None:
    service = _service(args)
    command = args.reward_command
    if command == "capabilities":
        result = service.capabilities()
        from halo_forge.training_signal import TRAINING_SIGNAL_CAPABILITIES

        result["training_signal_capabilities"] = [
            item.to_dict() for item in TRAINING_SIGNAL_CAPABILITIES.list()
        ]
    elif command == "system":
        action = args.reward_system_action
        if action == "list":
            result = service.list_systems(
                modality=args.modality,
                task_type=args.task_type,
                limit=args.limit,
                offset=args.offset,
            )
        elif action == "show":
            result = _detail(service, "system", args.system_or_revision_id)
        elif action == "validate":
            result = service.validate_system_definition(
                dict(_json_value(args.spec, default={}))
            )
        elif action == "create":
            result = service.create_system(
                name=args.name,
                description=args.description,
                definition=_validated_system_definition(service, args),
            )
        else:
            result = _revise_system(
                service,
                args.system_id,
                _validated_system_definition(service, args),
            )
    elif command == "protocol":
        action = args.reward_protocol_action
        if action == "list":
            result = service.list_protocols(limit=args.limit, offset=args.offset)
        elif action == "show":
            result = _detail(service, "protocol", args.protocol_or_revision_id)
        elif action == "create":
            result = service.create_protocol(
                name=args.name,
                description=args.description,
                definition=dict(_json_value(args.spec, default={})),
            )
        else:
            result = service.create_protocol_revision(
                args.protocol_id,
                definition=dict(_json_value(args.spec, default={})),
            )
    elif command == "integrity-profile":
        action = args.reward_integrity_profile_action
        if action == "list":
            result = service.list_integrity_profiles(limit=args.limit, offset=args.offset)
        elif action == "show":
            result = _detail(
                service, "integrity_profile", args.profile_or_revision_id
            )
        elif action == "create":
            requirements = _json_value(args.spec, default=None)
            if requirements is None:
                if args.template == "custom":
                    raise ValueError("--template custom requires --spec")
                from halo_forge.reward_integrity import PROFILE_DEFAULTS

                requirements = PROFILE_DEFAULTS[args.template]
            result = service.create_integrity_profile(
                name=args.name,
                description=args.description,
                template_kind=args.template,
                requirements=requirements,
            )
        else:
            requirements = _json_value(args.spec, default=None)
            if requirements is None:
                if args.template == "custom":
                    raise ValueError("--template custom requires --spec")
                from halo_forge.reward_integrity import PROFILE_DEFAULTS

                requirements = PROFILE_DEFAULTS[args.template]
            result = service.create_integrity_profile_revision(
                args.profile_id,
                template_kind=args.template,
                requirements=requirements,
            )
    elif command == "trace":
        action = args.reward_trace_action
        if action == "list":
            result = service.list_signal_shards(
                run_id=args.run_id, limit=args.limit, offset=args.offset
            )
        elif action == "show":
            result = service.get_signal_shard(args.trace_id)
        else:
            result = service.verify_signal_shard(args.trace_id)
    else:
        action = args.reward_audit_action
        if action == "list":
            result = service.list_audits(
                run_id=args.run_id,
                status=args.status,
                limit=args.limit,
                offset=args.offset,
            )
        elif action == "create":
            shard = service.get_signal_shard(args.trace)
            if shard.reward_system_revision_id != args.reward_system_revision:
                raise ValueError("--reward-system-revision conflicts with the sealed trace")
            if shard.protocol_revision_id != args.protocol_revision:
                raise ValueError("--protocol-revision conflicts with the sealed trace")
            result = service.create_audit(
                signal_shard_id=args.trace,
                integrity_profile_revision_id=args.integrity_profile_revision,
                development_suite_revision_id=args.development_suite_revision,
                runtime_identity=dict(_json_value(args.runtime, default={})),
                request={"source": "cli"},
                submit=not args.wait,
            )
            if args.wait:
                from halo_forge.reward_integrity.runtime import execute_pinned_audit

                result = execute_pinned_audit(service.db, _value(result)["id"])
        elif action == "show":
            result = _detail(service, "audit", args.audit_id)
        elif action == "samples":
            result = service.list_audit_samples(
                args.audit_id,
                population=args.population,
                outcome=args.outcome,
                query=args.query,
                limit=args.limit,
                offset=args.offset,
            )
        elif action == "metrics":
            result = service.list_audit_metrics(
                args.audit_id,
                subgroup=args.subgroup,
                limit=args.limit,
                offset=args.offset,
            )
        elif action == "compare":
            result = service.compare_audits(
                args.base_id,
                args.candidate_id,
                limit=args.limit,
                offset=args.offset,
            )
        elif action == "verify":
            result = service.verify_audit_bundle(args.audit_id)
        elif action == "cancel":
            result = service.cancel_audit(args.audit_id)
        elif action == "retry":
            result = service.retry_audit(args.audit_id, reason=args.reason)
        else:
            audit = service.get_audit(args.audit_id)
            reviewed = service.review_audit(
                args.audit_id,
                action=args.action,
                reason=args.reason,
                checkpoint=args.checkpoint,
            )
            result = reviewed
            if args.action in {"continue", "stop", "fork"}:
                try:
                    replay_sync = service.sync_audit_replay(
                        args.audit_id, decision=reviewed
                    )
                except Exception as exc:
                    replay_sync = {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                result = {**_value(reviewed), "replay_sync": replay_sync}
                if args.action == "continue" and audit.direct_run_segment_id:
                    from halo_forge.reward_integrity.direct_segments import (
                        enqueue_next_direct_segment,
                    )

                    next_work = enqueue_next_direct_segment(
                        service.db,
                        service.scheduler,
                        current_segment_id=audit.direct_run_segment_id,
                        dependency_work_item_id=audit.work_item_id,
                    )
                    result["next_work_item_id"] = (
                        next_work.id if next_work is not None else None
                    )
    if result is None:
        raise SystemExit("Reward integrity object not found")
    _emit(result, as_json=bool(args.json))


__all__ = ["add_reward_parser", "cmd_reward"]
