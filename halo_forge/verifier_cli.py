"""CLI transport for Verifier Reliability and Reward Studio.

The command handlers are intentionally thin.  Dashboard, API, and CLI all
delegate to :class:`halo_forge.verifier_lab.VerifierLabService` so immutable
identity, eligibility, scheduling, and qualification rules cannot drift.
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


def add_verifier_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "verifier", help="Create, calibrate, qualify, and bind immutable verifiers"
    )
    commands = parser.add_subparsers(dest="verifier_command", required=True)
    _leaf(commands.add_parser("catalog", help="List verifier reliability capabilities"))

    profile = commands.add_parser("profile", help="Manage immutable verifier profiles")
    profile_commands = profile.add_subparsers(dest="verifier_profile_action", required=True)
    p_list = _leaf(profile_commands.add_parser("list"))
    p_list.add_argument("--family")
    p_list.add_argument("--modality")
    p_list.add_argument("--task-type")
    p_list.add_argument(
        "--qualified-only",
        action="store_true",
        help="Show only candidate/approved revisions used by guided pickers",
    )
    p_show = _leaf(profile_commands.add_parser("show"))
    p_show.add_argument("profile_or_revision_id")
    p_validate = _leaf(profile_commands.add_parser("validate"))
    p_validate.add_argument("--spec", required=True, help="JSON file or inline object")

    def profile_fields(value: argparse.ArgumentParser, *, revise: bool = False) -> None:
        if revise:
            value.add_argument("profile_id")
        else:
            value.add_argument("--name", required=True)
            value.add_argument("--description")
        value.add_argument("--spec", help="JSON file or inline revision definition")
        value.add_argument(
            "--family",
            choices=["deterministic", "llm_judge", "reward_model", "chain"],
        )
        value.add_argument("--implementation", help="Registry name, plugin, or artifact id")
        value.add_argument("--modality", default="text")
        value.add_argument("--task-type", default="binary")
        value.add_argument("--reward-min", type=float, default=0.0)
        value.add_argument("--reward-max", type=float, default=1.0)
        value.add_argument("--direction", choices=["maximize", "minimize"], default="maximize")
        value.add_argument("--threshold", type=float, default=0.5)
        value.add_argument("--tie-policy", choices=["pass", "fail", "tie"], default="fail")
        value.add_argument("--configuration", help="Credential-free JSON configuration")
        value.add_argument("--runtime", help="Runtime contract JSON")
        _leaf(value)

    profile_fields(profile_commands.add_parser("create"))
    profile_fields(profile_commands.add_parser("revise"), revise=True)

    protocol = commands.add_parser("protocol", help="Manage calibration protocols")
    protocol_commands = protocol.add_subparsers(dest="verifier_protocol_action", required=True)
    _leaf(protocol_commands.add_parser("list"))
    protocol_show = _leaf(protocol_commands.add_parser("show"))
    protocol_show.add_argument("protocol_or_revision_id")
    protocol_create = _leaf(protocol_commands.add_parser("create"))
    protocol_create.add_argument("--name", required=True)
    protocol_create.add_argument("--description")
    protocol_create.add_argument("--spec", required=True)
    protocol_revise = _leaf(protocol_commands.add_parser("revise"))
    protocol_revise.add_argument("protocol_id")
    protocol_revise.add_argument("--spec", required=True)

    qualification = commands.add_parser(
        "qualification-profile", help="Manage verifier qualification policies"
    )
    qualification_commands = qualification.add_subparsers(
        dest="verifier_qualification_profile_action", required=True
    )
    _leaf(qualification_commands.add_parser("list"))
    qualification_show = _leaf(qualification_commands.add_parser("show"))
    qualification_show.add_argument("profile_or_revision_id")
    qualification_create = _leaf(qualification_commands.add_parser("create"))
    qualification_create.add_argument("--name", required=True)
    qualification_create.add_argument("--description")
    qualification_create.add_argument(
        "--template",
        choices=["strict_oracle", "human_aligned", "exploratory", "custom"],
        default="human_aligned",
    )
    qualification_create.add_argument("--spec", help="Custom requirements JSON")
    qualification_revise = _leaf(qualification_commands.add_parser("revise"))
    qualification_revise.add_argument("profile_id")
    qualification_revise.add_argument(
        "--template",
        choices=["strict_oracle", "human_aligned", "exploratory", "custom"],
        default="human_aligned",
    )
    qualification_revise.add_argument("--spec", help="Custom requirements JSON")

    calibration = commands.add_parser("calibration", help="Run replicated calibrations")
    calibration_commands = calibration.add_subparsers(
        dest="verifier_calibration_action", required=True
    )
    c_list = _leaf(calibration_commands.add_parser("list"))
    c_list.add_argument("--status")
    c_list.add_argument("--verifier-profile-revision")
    c_create = _leaf(calibration_commands.add_parser("create"))
    c_create.add_argument("--verifier-profile-revision", required=True)
    c_create.add_argument("--source-kind", choices=["label_set", "benchmark_suite"], required=True)
    c_create.add_argument("--source-revision", required=True)
    c_create.add_argument("--protocol-revision", required=True)
    c_create.add_argument("--qualification-profile-revision", required=True)
    c_create.add_argument("--confirmation", action="store_true")
    c_create.add_argument("--runtime", help="Runtime identity/requirements JSON")
    c_create.add_argument("--wait", action="store_true")
    for action in ("show", "samples", "cancel", "retry"):
        value = _leaf(calibration_commands.add_parser(action))
        value.add_argument("calibration_id")
        if action == "samples":
            value.add_argument("--partition")
            value.add_argument("--outcome")
            value.add_argument("--offset", type=int, default=0)
            value.add_argument("--limit", type=int, default=100)
    c_compare = _leaf(calibration_commands.add_parser("compare"))
    c_compare.add_argument("base_id")
    c_compare.add_argument("candidate_id")
    c_compare.add_argument("--offset", type=int, default=0)
    c_compare.add_argument("--limit", type=int, default=100)

    qualify = _leaf(commands.add_parser("qualify", help="Record a qualification decision"))
    qualify.add_argument("calibration_id")
    qualify.add_argument(
        "--scope",
        choices=["development", "operational", "confirmation"],
        default="development",
    )
    qualify.add_argument("--override-note")

    promote = _leaf(commands.add_parser("promote", help="Move candidate/approved alias"))
    promote.add_argument("verifier_profile_revision_id")
    promote.add_argument("--alias", choices=["candidate", "approved"], required=True)
    promote.add_argument("--override-note")

    usage = _leaf(commands.add_parser("usage", help="Inspect exact downstream bindings"))
    usage.add_argument("verifier_profile_revision_id")
    usage.add_argument("--limit", type=int, default=100)
    usage.add_argument("--offset", type=int, default=0)
    return parser


def _service(args: Any):
    from halo_forge.run_db import RunDatabase, get_database
    from halo_forge.verifier_lab import VerifierLabService

    db = RunDatabase(str(Path(args.database).expanduser())) if args.database else get_database()
    return VerifierLabService(db)


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
                state = item.get("status") or item.get("family") or item.get("decision") or ""
                print(f"{identity}\t{state}".rstrip())
            else:
                print(item)
        return
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _profile_definition(args: Any) -> dict[str, Any]:
    if args.spec:
        return dict(_json_value(args.spec, default={}))
    return {
        "family": args.family,
        "implementation": {"ref": args.implementation},
        "modality": args.modality,
        "task_type": args.task_type,
        "reward_contract": {
            "minimum": args.reward_min,
            "maximum": args.reward_max,
            "direction": args.direction,
            "threshold": args.threshold,
            "tie_policy": args.tie_policy,
        },
        "configuration": _json_value(args.configuration, default={}),
        "runtime_contract": _json_value(args.runtime, default={}),
    }


def cmd_verifier(args: Any) -> None:
    service = _service(args)
    command = args.verifier_command
    result: Any
    if command == "catalog":
        result = service.capabilities()
    elif command == "profile":
        action = args.verifier_profile_action
        if action == "list":
            result = service.list_profiles(
                family=args.family,
                modality=args.modality,
                task_type=args.task_type,
                qualified_only=bool(args.qualified_only),
            )
        elif action == "show":
            result = service.get_profile_detail(args.profile_or_revision_id)
        elif action == "validate":
            result = service.validate_profile_definition(
                dict(_json_value(args.spec, default={}))
            )
        elif action == "create":
            result = service.create_profile(
                name=args.name,
                description=args.description,
                definition=_profile_definition(args),
            )
        else:
            result = service.revise_profile(
                args.profile_id, definition=_profile_definition(args)
            )
    elif command == "protocol":
        action = args.verifier_protocol_action
        if action == "list":
            result = service.list_protocols()
        elif action == "show":
            result = service.get_protocol_detail(args.protocol_or_revision_id)
        elif action == "create":
            result = service.create_protocol(
                name=args.name,
                description=args.description,
                definition=dict(_json_value(args.spec, default={})),
            )
        else:
            result = service.revise_protocol(
                args.protocol_id, definition=dict(_json_value(args.spec, default={}))
            )
    elif command == "qualification-profile":
        action = args.verifier_qualification_profile_action
        if action == "list":
            result = service.list_qualification_profiles()
        elif action == "show":
            result = service.get_qualification_profile_detail(args.profile_or_revision_id)
        elif action == "create":
            result = service.create_qualification_profile(
                name=args.name,
                description=args.description,
                template_kind=args.template,
                requirements=_json_value(args.spec, default=None),
            )
        else:
            result = service.revise_qualification_profile(
                args.profile_id,
                template_kind=args.template,
                requirements=_json_value(args.spec, default=None),
            )
    elif command == "calibration":
        action = args.verifier_calibration_action
        if action == "list":
            result = service.list_calibrations(
                status=args.status,
                verifier_revision_id=args.verifier_profile_revision,
            )
        elif action == "create":
            result = service.launch_calibration(
                verifier_revision_id=args.verifier_profile_revision,
                source_kind=args.source_kind,
                source_revision_id=args.source_revision,
                protocol_revision_id=args.protocol_revision,
                qualification_profile_revision_id=args.qualification_profile_revision,
                confirmation=bool(args.confirmation),
                runtime_identity=_json_value(args.runtime, default={}),
            )
            if args.wait:
                result = service.wait_for_calibration(_value(result)["id"])
        elif action == "show":
            result = service.get_calibration_detail(args.calibration_id)
        elif action == "samples":
            result = service.list_calibration_samples(
                args.calibration_id,
                partition=args.partition,
                outcome=args.outcome,
                offset=args.offset,
                limit=args.limit,
            )
        elif action == "cancel":
            result = service.cancel_calibration(args.calibration_id)
        elif action == "retry":
            result = service.retry_calibration(args.calibration_id)
        else:
            result = service.compare_calibrations(
                args.base_id, args.candidate_id, offset=args.offset, limit=args.limit
            )
    elif command == "qualify":
        result = service.qualify_calibration(
            args.calibration_id,
            scope=args.scope,
            override_note=args.override_note,
        )
    elif command == "promote":
        result = service.promote_revision(
            args.verifier_profile_revision_id,
            alias=args.alias,
            override_note=args.override_note,
        )
    else:
        result = service.list_usage(
            args.verifier_profile_revision_id,
            offset=args.offset,
            limit=args.limit,
        )
    if result is None:
        raise SystemExit("Verifier Reliability object not found")
    _emit(result, as_json=bool(args.json))


__all__ = ["add_verifier_parser", "cmd_verifier"]
