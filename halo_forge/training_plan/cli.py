"""CLI parity for V18 guided training plans."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Mapping


def add_training_plan_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "train-plan", help="Recommend, prepare, check, and launch a guided proof plan"
    )
    actions = parser.add_subparsers(dest="train_plan_action", required=True)

    recommend = actions.add_parser("recommend", help="Recommend one safe training plan")
    recommend.add_argument("--dataset-version", required=True)
    recommend.add_argument("--scenario-revision")
    recommend.add_argument("--trainer-mode")
    recommend.add_argument("--model")

    show = actions.add_parser("show", help="Show a plan or immutable revision")
    show.add_argument("identifier")

    alternatives = actions.add_parser("alternatives", help="List compatible model alternatives")
    alternatives.add_argument("revision_id")

    prepare = actions.add_parser("prepare", help="Resolve and prepare the exact model")
    prepare.add_argument("revision_id")
    prepare.add_argument("--wait", action="store_true")
    prepare.add_argument("--offline", action="store_true", help="Use cached files only")

    check = actions.add_parser("check", help="Run the disposable capacity check")
    check.add_argument("revision_id")
    check.add_argument("--wait", action="store_true")

    proof = actions.add_parser("proof", help="Launch the confirmed bounded proof run")
    proof.add_argument("revision_id")
    proof.add_argument("--output-root")
    proof.add_argument("--wait", action="store_true")

    for command in (recommend, show, alternatives, prepare, check, proof):
        command.add_argument("--database")
        command.add_argument("--root")
        command.add_argument("--json", action="store_true")


def _service(args: Any):
    from halo_forge.run_db import get_database
    from halo_forge.training_plan import TrainingPlanService
    from halo_forge.workstation_jobs import WorkstationScheduler

    database = get_database(getattr(args, "database", None))
    root = Path(getattr(args, "root", None) or Path.home() / ".halo-forge").expanduser()
    scheduler = WorkstationScheduler(database)
    return TrainingPlanService(database, root=root, scheduler=scheduler), scheduler, root


def _emit(args: Any, value: Any) -> None:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if getattr(args, "json", False):
        print(json.dumps(value, indent=2, sort_keys=True, default=str))
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(item, (dict, list, tuple)):
                continue
            print(f"{key.replace('_', ' ').title()}: {item}")
    else:
        print(value)


def _wait_for_item(scheduler: Any, work_item_id: str) -> Any:
    from halo_forge.workstation_jobs.worker import WorkstationWorker

    worker = WorkstationWorker(scheduler)
    worker.run_once(work_item_id=work_item_id)
    return scheduler.database.get_work_item(work_item_id)


def dispatch_training_plan(args: Any) -> bool:
    if getattr(args, "command", None) != "train-plan":
        return False
    service, scheduler, root = _service(args)
    action = args.train_plan_action
    if action == "recommend":
        value = service.recommend(
            {
                "dataset_version_id": args.dataset_version,
                "scenario_revision_id": args.scenario_revision,
                "trainer_mode": args.trainer_mode,
                "model": args.model,
            }
        )
    elif action == "show":
        value = service.get_revision(args.identifier) or service.get_plan(args.identifier)
        if value is None:
            raise KeyError(args.identifier)
    elif action == "alternatives":
        value = service.alternatives(args.revision_id)
    elif action == "prepare":
        service.record_decision(
            args.revision_id,
            "confirmed",
            details={
                "download_confirmed": not args.offline,
                "offline_cached_operation": bool(args.offline),
                "confirmation_surface": "cli",
            },
        )
        value = service.prepare_model(
            args.revision_id, enqueue=not args.wait, allow_download=not args.offline
        )
        if args.wait and value.status != "completed":
            value = service.run_model_preparation(value.id, allow_download=not args.offline)
        elif args.wait and value.work_item_id:
            _wait_for_item(scheduler, value.work_item_id)
            value = service.get_model_preparation(value.id)
    elif action == "check":
        value = service.create_capacity_check(args.revision_id, enqueue=not args.wait)
        if args.wait and value.status not in {"ready", "ready_with_adjustment", "blocked"}:
            value = service.run_capacity_check(value.id)
        elif args.wait and value.work_item_id:
            _wait_for_item(scheduler, value.work_item_id)
            value = service.get_capacity_check(value.id)
    else:
        from halo_forge.public_api.service import PublicApiService

        public = PublicApiService(
            database=service.database,
            workstation_scheduler=scheduler,
            product_lab_storage_root=root,
            training_plan=service,
            training_plan_storage_root=root,
        )
        value = asyncio.run(
            public.launch_training_plan_proof(
                args.revision_id,
                {"output_root": args.output_root} if args.output_root else {},
            )
        )
        if args.wait and isinstance(value, Mapping) and value.get("work_item_id"):
            _wait_for_item(scheduler, str(value["work_item_id"]))
    _emit(args, value)
    return True


__all__ = ["add_training_plan_parser", "dispatch_training_plan"]
