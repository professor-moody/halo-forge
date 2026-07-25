"""CLI parity for Halo Forge Labs V11-V15."""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from halo_forge.run_db import RunDatabase, get_database

from .service import FutureLabService


def _shared(parser: ArgumentParser) -> None:
    parser.add_argument("--database", help="SQLite catalog path")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")


def _spec(parser: ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        "--spec",
        required=required,
        help="JSON or YAML specification file",
    )


def add_future_lab_parsers(subparsers: Any) -> None:
    outcome = subparsers.add_parser(
        "outcome", help="Assess proof-run technical and quality outcomes"
    )
    outcome_actions = outcome.add_subparsers(dest="outcome_command", required=True)
    profile = outcome_actions.add_parser("profile", help="Outcome profile catalog")
    profile_actions = profile.add_subparsers(dest="outcome_profile_action", required=True)
    for action in ("list", "show"):
        parser = profile_actions.add_parser(action)
        if action == "show":
            parser.add_argument("profile_id")
        parser.add_argument("--scenario-revision")
        _shared(parser)
    assess = outcome_actions.add_parser("assess")
    assess.add_argument("proof_run_id")
    assess.add_argument("--base-evaluation")
    assess.add_argument("--candidate-evaluation")
    assess.add_argument("--scenario-revision")
    _shared(assess)
    prepare = outcome_actions.add_parser(
        "prepare", help="Check a proof run using matched development evidence"
    )
    prepare.add_argument("proof_run_id")
    prepare.add_argument("--suite-revision")
    prepare.add_argument("--scenario-revision")
    prepare.add_argument("--wait", action="store_true")
    _shared(prepare)
    show = outcome_actions.add_parser("show")
    show.add_argument("assessment_id")
    _shared(show)
    findings = outcome_actions.add_parser("findings")
    findings.add_argument("assessment_id")
    findings.add_argument("--limit", type=int, default=100)
    findings.add_argument("--offset", type=int, default=0)
    _shared(findings)
    review = outcome_actions.add_parser("review")
    review.add_argument("proof_run_id")
    review.add_argument(
        "--decision",
        required=True,
        choices=["evaluate", "repair", "retry", "fork", "start_full_run", "override"],
    )
    review.add_argument("--assessment")
    review.add_argument("--reason", default="")
    review.add_argument("--full-run-id")
    _shared(review)
    full_context = outcome_actions.add_parser("full-run-context")
    full_context.add_argument("proof_run_id")
    full_context.add_argument("--assessment")
    full_context.add_argument("--override-reason")
    _shared(full_context)

    study = subparsers.add_parser(
        "study", help="Controlled adaptation validity and ablation studies"
    )
    study_actions = study.add_subparsers(dest="study_command", required=True)
    for action in (
        "list",
        "show",
        "create",
        "revise",
        "validate",
        "materialize",
        "launch",
        "status",
        "analyze",
        "deviate",
        "decide",
        "export",
    ):
        parser = study_actions.add_parser(action)
        if action in {"show"}:
            parser.add_argument("study_id")
        elif action in {"revise"}:
            parser.add_argument("study_id")
            _spec(parser)
        elif action in {"materialize", "launch", "status", "analyze", "deviate", "decide", "export"}:
            parser.add_argument("revision_id")
            if action in {"analyze", "deviate", "decide"}:
                _spec(parser)
            if action == "launch":
                parser.add_argument("--wait", action="store_true")
        elif action in {"create", "validate"}:
            _spec(parser)
        parser.add_argument("--limit", type=int, default=100)
        parser.add_argument("--offset", type=int, default=0)
        _shared(parser)

    for command, help_text in (
        ("classify", "Train, evaluate, and predict with classification models"),
        ("embed", "Train, evaluate, and serve embedding models"),
        ("rerank", "Train, evaluate, and serve rerankers"),
    ):
        parser = subparsers.add_parser(command, help=help_text)
        actions = parser.add_subparsers(dest=f"{command}_command", required=True)
        action_names = (
            ("train", "evaluate", "predict")
            if command == "classify"
            else ("train", "evaluate", "serve")
        )
        for action in action_names:
            child = actions.add_parser(action)
            if action == "train":
                _spec(child, required=False)
                child.add_argument("--model")
                child.add_argument("--dataset")
                child.add_argument("--output")
                child.add_argument("--validation-file")
                child.add_argument("--epochs", type=int, default=1)
                child.add_argument("--batch-size", type=int, default=4)
                child.add_argument("--learning-rate", type=float, default=2e-5)
                child.add_argument("--max-samples", type=int)
                child.add_argument("--seed", type=int, default=42)
                child.add_argument("--multi-label", action="store_true")
                child.add_argument("--proof-run", action="store_true")
                child.add_argument("--training-plan-revision")
                child.add_argument("--no-caffeinate", action="store_true")
                child.add_argument("--label-schema-revision")
                child.add_argument("--retrieval-corpus")
            else:
                _spec(child)
            _shared(child)

    env = subparsers.add_parser(
        "env", help="Deterministic local agent environments and trajectories"
    )
    env_actions = env.add_subparsers(dest="env_command", required=True)
    for action in (
        "list",
        "show",
        "validate",
        "create",
        "revise",
        "run",
        "rerun",
        "episodes",
        "episode",
        "replay",
        "compare",
    ):
        parser = env_actions.add_parser(action)
        if action in {"show", "revise"}:
            parser.add_argument("environment_id")
        elif action in {"episode", "replay", "rerun"}:
            parser.add_argument("episode_id")
        if action in {"validate", "create", "revise", "compare"}:
            _spec(parser)
        if action == "run":
            _spec(parser, required=False)
            parser.add_argument("--suite")
            parser.add_argument("--item")
            parser.add_argument("--subject")
            parser.add_argument("--serve-url")
            parser.add_argument("--wait", action="store_true")
        if action == "rerun":
            parser.add_argument("--subject")
            parser.add_argument("--serve-url")
            parser.add_argument("--wait", action="store_true")
        parser.add_argument("--limit", type=int, default=100)
        parser.add_argument("--offset", type=int, default=0)
        _shared(parser)
    suite = env_actions.add_parser("suite")
    suite_actions = suite.add_subparsers(dest="env_suite_action", required=True)
    for action in ("list", "show", "create", "revise"):
        parser = suite_actions.add_parser(action)
        if action == "create":
            parser.add_argument("environment_revision_id")
            _spec(parser)
        elif action == "revise":
            parser.add_argument("suite_id")
            _spec(parser)
        elif action == "show":
            parser.add_argument("suite_revision_id")
        _shared(parser)
    trajectory = env_actions.add_parser("trajectory")
    trajectory_actions = trajectory.add_subparsers(
        dest="env_trajectory_action", required=True
    )
    for action in ("publish", "verify", "review-proposal", "build-dataset"):
        parser = trajectory_actions.add_parser(action)
        _spec(parser)
        _shared(parser)


def add_ground_parser(data_subparsers: Any) -> None:
    ground = data_subparsers.add_parser(
        "ground", help="Generate citation-grounded reviewed data from immutable corpora"
    )
    actions = ground.add_subparsers(dest="ground_command", required=True)
    profile = actions.add_parser("profile")
    profile_actions = profile.add_subparsers(dest="ground_profile_action", required=True)
    for action in ("list", "show", "create", "revise"):
        parser = profile_actions.add_parser(action)
        if action == "show":
            parser.add_argument("profile_id")
        elif action == "revise":
            parser.add_argument("profile_id")
            _spec(parser)
        elif action == "create":
            _spec(parser)
        _shared(parser)
    for action in (
        "generate",
        "show",
        "candidates",
        "verify",
        "review-proposal",
        "build-dataset",
        "build-suite",
    ):
        parser = actions.add_parser(action)
        if action == "generate":
            parser.add_argument("profile_revision_id")
            _spec(parser, required=False)
            parser.add_argument(
                "--preset",
                choices=["quick", "standard", "thorough"],
                default="standard",
            )
            parser.add_argument("--source-version")
            parser.add_argument("--wait", action="store_true")
        else:
            parser.add_argument("batch_id")
            if action in {"build-dataset", "build-suite"}:
                _spec(parser)
        parser.add_argument("--limit", type=int, default=100)
        parser.add_argument("--offset", type=int, default=0)
        _shared(parser)


def _load_spec(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read YAML specs") from exc
        value = yaml.safe_load(text)
    else:
        value = json.loads(text)
    if not isinstance(value, Mapping):
        raise ValueError("specification must be an object")
    return dict(value)


def _engine(args: Namespace) -> FutureLabService:
    path = getattr(args, "database", None)
    db = RunDatabase(str(Path(path).expanduser())) if path else get_database()
    return FutureLabService(db)


def _public_service(engine: FutureLabService):
    from halo_forge.public_api.service import PublicApiService
    from halo_forge.workstation_jobs import WorkstationScheduler

    scheduler = WorkstationScheduler(engine.db)
    return PublicApiService(
        database=engine.db,
        workstation_scheduler=scheduler,
        future_lab=engine,
        future_lab_storage_root=engine.root,
        dataset_storage_root=engine.root / "datasets",
        evaluation_storage_root=engine.root / "evaluations",
    )


def _wait_for_work(service: Any, work_item_id: str) -> Dict[str, Any]:
    import time

    from halo_forge.workstation_jobs import WorkstationWorker

    worker = WorkstationWorker(service._scheduler())
    while True:
        current = service.get_work_item(work_item_id)
        if current is None:
            raise RuntimeError(f"work item disappeared: {work_item_id}")
        if current["status"] in {
            "completed",
            "failed",
            "cancelled",
            "interrupted",
            "needs_reconciliation",
        }:
            return current
        executed = worker.run_once()
        if executed is None:
            time.sleep(0.1)


def _emit(args: Namespace, value: Any) -> None:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def dispatch_future_lab(args: Namespace) -> bool:
    command = getattr(args, "command", "")
    if command not in {"outcome", "study", "classify", "embed", "rerank", "env"} and not (
        command == "data" and getattr(args, "data_command", "") == "ground"
    ):
        return False
    engine = _engine(args)

    if command == "outcome":
        action = args.outcome_command
        if action == "profile":
            values = engine.list_outcome_profiles(
                scenario_revision_id=args.scenario_revision
            )
            if args.outcome_profile_action == "show":
                values = [value for value in values if value.id == args.profile_id]
                _emit(args, values[0] if values else None)
            else:
                _emit(args, {"items": [value.to_dict() for value in values]})
        elif action == "assess":
            _emit(
                args,
                engine.assess_outcome(
                    args.proof_run_id,
                    {
                        "base_evaluation_id": args.base_evaluation,
                        "candidate_evaluation_id": args.candidate_evaluation,
                        "scenario_revision_id": args.scenario_revision,
                    },
                ),
            )
        elif action == "prepare":
            public = _public_service(engine)
            value = public.prepare_training_outcome(
                args.proof_run_id,
                {
                    "suite_revision_id": args.suite_revision,
                    "scenario_revision_id": args.scenario_revision,
                },
            )
            if args.wait and value.get("work_item_id"):
                value["work_item"] = _wait_for_work(
                    public, str(value["work_item_id"])
                )
                assessment = value.get("assessment") or {}
                if assessment.get("id"):
                    value["assessment"] = public.get_training_outcome(
                        str(assessment["id"])
                    )
            _emit(args, value)
        elif action == "show":
            _emit(args, engine.get_outcome_assessment(args.assessment_id))
        elif action == "findings":
            _emit(
                args,
                engine.list_outcome_findings(
                    args.assessment_id, limit=args.limit, offset=args.offset
                ),
            )
        elif action == "review":
            _emit(
                args,
                engine.create_outcome_decision(
                    args.proof_run_id,
                    {
                        "decision": args.decision,
                        "assessment_id": args.assessment,
                        "reason": args.reason,
                        "full_run_id": args.full_run_id,
                    },
                ),
            )
        else:
            _emit(
                args,
                engine.full_run_context(
                    args.proof_run_id,
                    assessment_id=args.assessment,
                    override_reason=args.override_reason,
                ),
            )
        return True

    if command == "study":
        action = args.study_command
        spec = _load_spec(getattr(args, "spec", None))
        if action == "list":
            value = engine.list_studies(limit=args.limit, offset=args.offset)
        elif action == "show":
            value = engine.get_study(args.study_id)
        elif action == "create":
            value = engine.create_study(spec)
        elif action == "revise":
            value = engine.create_study_protocol(args.study_id, spec)
        elif action == "validate":
            value = engine.validate_study_protocol(spec)
        elif action == "materialize":
            value = engine.materialize_study(args.revision_id)
        elif action == "launch":
            public = _public_service(engine)
            value = public.launch_adaptation_study(args.revision_id)
            if args.wait and value.get("work_item_id"):
                value["work_item"] = _wait_for_work(
                    public, str(value["work_item_id"])
                )
                value["protocol"] = public.get_adaptation_study_protocol(
                    args.revision_id
                )
        elif action == "status":
            value = engine.get_study_protocol(args.revision_id)
        elif action == "analyze":
            value = engine.analyze_study(args.revision_id, spec)
        elif action == "deviate":
            value = engine.create_study_deviation(args.revision_id, spec)
        elif action == "decide":
            value = engine.create_study_decision(args.revision_id, spec)
        else:
            value = engine.get_study_protocol(args.revision_id)
        _emit(args, value)
        return True

    if command == "data":
        action = args.ground_command
        spec = _load_spec(getattr(args, "spec", None))
        if action == "profile":
            profile_action = args.ground_profile_action
            if profile_action == "list":
                value = engine.list_grounding_profiles()
            elif profile_action == "show":
                value = engine.get_grounding_profile(args.profile_id)
            elif profile_action == "create":
                value = engine.create_grounding_profile(spec)
            else:
                value = engine.create_grounding_profile_revision(args.profile_id, spec)
        elif action == "generate":
            public = _public_service(engine)
            value = public.launch_grounded_generation(
                args.profile_revision_id,
                {
                    **spec,
                    "preset": args.preset,
                    "source_version_id": args.source_version
                    or spec.get("source_version_id"),
                },
            )
            if args.wait and value.get("work_item_id"):
                value["work_item"] = _wait_for_work(
                    public, str(value["work_item_id"])
                )
                value["batch"] = public.get_grounded_generation(str(value["id"]))
        elif action == "show":
            value = engine.get_grounded_batch(args.batch_id)
        elif action == "candidates":
            value = engine.list_grounded_candidates(
                args.batch_id, limit=args.limit, offset=args.offset
            )
        elif action == "review-proposal":
            value = engine.grounding_review_proposal(args.batch_id)
        else:
            value = {
                "batch_id": args.batch_id,
                "action": action,
                "status": "review_required",
                "message": "Publish reviewed labels through Review Studio before building.",
            }
        _emit(args, value)
        return True

    if command in {"classify", "embed", "rerank"}:
        spec = _load_spec(args.spec)
        action = getattr(args, f"{command}_command")
        if action == "train":
            from .specialized import SpecializedTrainConfig, run_specialized_training

            resolved = {
                **spec,
                **{
                    key: value
                    for key, value in {
                        "task": command,
                        "model": getattr(args, "model", None),
                        "dataset": getattr(args, "dataset", None),
                        "output_dir": getattr(args, "output", None),
                        "validation_file": getattr(args, "validation_file", None),
                        "epochs": getattr(args, "epochs", None),
                        "batch_size": getattr(args, "batch_size", None),
                        "learning_rate": getattr(args, "learning_rate", None),
                        "max_samples": getattr(args, "max_samples", None),
                        "seed": getattr(args, "seed", None),
                        "multi_label": getattr(args, "multi_label", False),
                        "proof_run": getattr(args, "proof_run", False),
                        "label_schema_revision_id": getattr(
                            args, "label_schema_revision", None
                        ),
                        "retrieval_corpus_id": getattr(args, "retrieval_corpus", None),
                    }.items()
                    if value not in {None, ""}
                },
            }
            for required in ("model", "dataset", "output_dir"):
                if not str(resolved.get(required) or "").strip():
                    raise ValueError(
                        f"{required.replace('_', '-')} is required for {command} train"
                    )
            value = run_specialized_training(SpecializedTrainConfig(**resolved))
        elif command == "classify" and action == "evaluate":
            value = engine.classification_metrics(
                spec.get("expected") or [],
                spec.get("predicted") or [],
                multilabel=bool(spec.get("multi_label", False)),
            )
        elif command == "classify" and action == "predict":
            value = engine.predict_classification(spec)
        elif command == "embed" and action == "evaluate":
            value = engine.retrieval_metrics(
                spec.get("rankings") or [],
                spec.get("relevant") or [],
                k=int(spec.get("k") or 10),
            )
        elif command == "rerank" and action == "evaluate":
            value = engine.retrieval_metrics(
                spec.get("rankings") or [],
                spec.get("relevant") or [],
                k=int(spec.get("k") or 10),
            )
        else:
            value = {
                "task": command,
                "action": action,
                "status": "validated",
                "spec_hash": __import__("hashlib").sha256(
                    json.dumps(spec, sort_keys=True).encode()
                ).hexdigest(),
                "message": "The resolved task contract is ready for the managed trainer.",
            }
        _emit(args, value)
        return True

    if command == "env":
        action = args.env_command
        spec = _load_spec(getattr(args, "spec", None))
        if action == "list":
            value = engine.list_environments(limit=args.limit, offset=args.offset)
        elif action == "show":
            value = engine.get_environment(args.environment_id)
        elif action == "validate":
            value = {"valid": True, "capabilities": engine.capabilities()["environments"]}
        elif action == "create":
            value = engine.create_environment(spec)
        elif action == "revise":
            value = engine.create_environment_revision(args.environment_id, spec)
        elif action == "run":
            public = _public_service(engine)
            suite_revision_id = str(
                args.suite or spec.get("suite_revision_id") or ""
            )
            if not suite_revision_id:
                raise ValueError("--suite is required for env run")
            value = public.launch_agent_episode(
                suite_revision_id,
                {
                    **spec,
                    "suite_item_id": args.item or spec.get("suite_item_id"),
                    "subject_ref": args.subject or spec.get("subject_ref"),
                    "serve_url": args.serve_url or spec.get("serve_url"),
                },
            )
            if args.wait and value.get("work_item_id"):
                value["work_item"] = _wait_for_work(
                    public, str(value["work_item_id"])
                )
                value["episode"] = public.get_agent_episode(str(value["id"]))
        elif action == "rerun":
            public = _public_service(engine)
            value = public.rerun_agent_episode(
                args.episode_id,
                {
                    "subject_ref": args.subject,
                    "serve_url": args.serve_url,
                },
            )
            if args.wait and value.get("work_item_id"):
                value["work_item"] = _wait_for_work(
                    public, str(value["work_item_id"])
                )
                value["episode"] = public.get_agent_episode(str(value["id"]))
        elif action == "episodes":
            value = engine.list_episodes(limit=args.limit, offset=args.offset)
        elif action == "episode":
            value = engine.get_episode(args.episode_id)
        elif action == "replay":
            value = engine.replay_episode(args.episode_id)
        elif action == "compare":
            base = engine.get_episode(str(spec.get("base_episode_id") or ""))
            candidate = engine.get_episode(
                str(spec.get("candidate_episode_id") or "")
            )
            if base is None or candidate is None:
                raise ValueError("base and candidate episodes are required")
            value = engine.compare_environment_subjects(
                suite_revision_id=base.suite_revision_id,
                base_subject_hash=base.subject_hash,
                candidate_subject_hash=candidate.subject_hash,
            )
        elif action == "suite":
            suite_action = args.env_suite_action
            if suite_action == "create":
                value = engine.create_episode_suite(_load_spec(args.spec))
            elif suite_action == "revise":
                value = engine.create_episode_suite_revision(
                    args.suite_id, _load_spec(args.spec)
                )
            else:
                value = {"status": "use the revision identifier with the API"}
        else:
            value = engine.publish_trajectory_set(_load_spec(args.spec))
        _emit(args, value)
        return True
    return False


__all__ = ["add_future_lab_parsers", "add_ground_parser", "dispatch_future_lab"]
