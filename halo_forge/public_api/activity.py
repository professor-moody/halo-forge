"""Normalized Activity Center projections over durable workstation records."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


def _as_utc(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _tail_log(path_value: Optional[str], *, limit: int = 80) -> list[str]:
    if not path_value:
        return []
    try:
        path = Path(path_value).expanduser()
        if not path.is_file():
            return []
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except OSError:
        return []


def _progress_values(progress: Mapping[str, Any]) -> tuple[Optional[float], Optional[float]]:
    current = progress.get(
        "current", progress.get("processed", progress.get("processed_records"))
    )
    total = progress.get("total", progress.get("total_records"))
    current_value = (
        float(current)
        if isinstance(current, (int, float)) and not isinstance(current, bool)
        else None
    )
    total_value = (
        float(total) if isinstance(total, (int, float)) and not isinstance(total, bool) else None
    )
    return current_value, total_value


def _eta_seconds(item: Any, current: Optional[float], total: Optional[float]) -> Optional[int]:
    if current is None or total is None or current <= 0 or total <= current:
        return 0 if current is not None and total is not None and current >= total else None
    started = _as_utc(item.started_at)
    if started is None:
        return None
    elapsed = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
    if elapsed <= 0:
        return None
    return max(0, int(round((total - current) * elapsed / current)))


def normalize_work_event(event: Any) -> dict[str, Any]:
    value = event.to_dict()
    payload = dict(value.get("payload") or {})
    event_type = str(value.pop("event_type", "event"))
    message = payload.get("message") or payload.get("reason") or payload.get("error")
    return {
        **value,
        "type": event_type,
        "status": payload.get("status"),
        "message": str(message) if message is not None else None,
        "payload": payload,
    }


def normalize_work_attempt(attempt: Any, *, retry_reason: Optional[str] = None) -> dict[str, Any]:
    value = attempt.to_dict()
    return {
        "id": value["id"],
        "work_item_id": value["work_item_id"],
        "attempt": int(value.get("ordinal") or 0),
        "status": value.get("status"),
        "output_dir": value.get("output_dir"),
        "worker_id": value.get("worker_id"),
        "pid": value.get("worker_pid"),
        "process_start_identity": value.get("worker_pid_started_at"),
        "started_at": value.get("started_at"),
        "completed_at": value.get("completed_at"),
        "error": value.get("error"),
        "retry_reason": retry_reason,
    }


def blocker_messages(blockers: Mapping[str, Any]) -> list[str]:
    messages: list[str] = []
    for dependency in blockers.get("dependencies") or ():
        identifier = dependency.get("depends_on_work_item_id") or "dependency"
        status = dependency.get("dependency_status") or "unfinished"
        messages.append(f"Waiting for {identifier} ({status})")
    lease = blockers.get("resource_lease")
    if isinstance(lease, Mapping):
        owner = lease.get("holder_id") or lease.get("holder_type") or "another operation"
        messages.append(f"Workstation resource is held by {owner}")
    return messages


def activity_item_view(database: Any, catalog: Any, item: Any) -> dict[str, Any]:
    """Build the one stable ActivityItem contract used by HTTP and SSE views."""

    launch_spec = dict(item.launch_spec)
    progress = dict(item.progress)
    current, total = _progress_values(progress)
    progress_percent = (
        min(100.0, max(0.0, current / total * 100.0))
        if current is not None and total is not None and total > 0
        else None
    )
    raw_events = catalog.list_events(work_item_id=item.id, limit=100)
    events = [normalize_work_event(value) for value in raw_events]
    retry_reasons = [
        str(value["message"])
        for value in events
        if value["type"] in {"retry", "retry_requested", "forced_retry"} and value.get("message")
    ]
    raw_attempts = catalog.list_attempts(item.id)
    attempts = [
        normalize_work_attempt(
            value,
            retry_reason=(
                retry_reasons[index - 1] if index > 0 and index <= len(retry_reasons) else None
            ),
        )
        for index, value in enumerate(raw_attempts)
    ]
    telemetry_rollup = catalog.get_telemetry_rollup(
        item.id,
        attempt_id=raw_attempts[-1].id if raw_attempts else None,
    )
    blockers = blocker_messages(database.work_item_blockers(item.id))
    logs = _tail_log(item.log_path)
    latest_log = progress.get("latest_log")
    if latest_log and str(latest_log) not in logs:
        logs.append(str(latest_log))
    payload_title = launch_spec.get("name") or launch_spec.get("model")
    title = str(payload_title or item.stage or item.kind).replace("_", " ")
    next_actions: list[str] = []
    if item.status in {"failed", "interrupted", "needs_reconciliation", "cancelled"}:
        next_actions.append("retry")
    if item.status in {"queued", "running", "blocked", "preparing"}:
        next_actions.append("cancel")
    domain_type = item.domain_kind or (
        "run" if item.canonical_run_id else "run_group" if item.run_group_id else None
    )
    domain_id = item.domain_id or item.canonical_run_id or item.run_group_id
    display_status = item.status
    action_links: list[dict[str, str]] = []
    summary_metrics: dict[str, Optional[float]] = {}
    if domain_type == "dataset_inspection" and domain_id:
        next_actions.append("open_source")
        action_links.append(
            {
                "id": "open_source",
                "label": "Resume import",
                "href": f"/datasets/new?inspection={domain_id}",
            }
        )
    elif domain_type == "document_extraction" and domain_id:
        extraction = database.get_document_extraction(str(domain_id))
        import_id = str(
            (extraction.import_id if extraction is not None else "")
            or launch_spec.get("import_id")
            or ""
        )
        import_record = (
            database.get_dataset_import(import_id) if import_id else None
        )
        inspection_id = str(
            (
                import_record.latest_inspection_id
                if import_record is not None
                else ""
            )
            or launch_spec.get("inspection_id")
            or ""
        )
        next_actions.append("open_source")
        action_links.append(
            {
                "id": "open_source",
                "label": (
                    "Review extraction"
                    if item.status == "completed"
                    else "Resume corpus import"
                ),
                "href": (
                    f"/datasets/new?inspection={inspection_id}"
                    if inspection_id
                    else "/datasets/new"
                ),
            }
        )
    elif domain_type == "dataset_registration" and domain_id:
        import_record = database.get_dataset_import(str(domain_id))
        dataset_id = str(
            (import_record.published_dataset_id if import_record is not None else "")
            or launch_spec.get("dataset_id")
            or ""
        )
        inspection_id = str(launch_spec.get("inspection_id") or "")
        next_actions.append("open_source")
        action_links.append(
            {
                "id": "open_source",
                "label": "Open dataset" if item.status == "completed" else "Resume import",
                "href": (
                    f"/datasets/{dataset_id}"
                    if item.status == "completed" and dataset_id
                    else f"/datasets/new?inspection={inspection_id}"
                ),
            }
        )
    elif domain_type == "dataset_source_refresh" and domain_id:
        source = database.get_dataset_source(str(domain_id))
        if source is not None:
            next_actions.append("open_source")
            action_links.append(
                {
                    "id": "open_source",
                    "label": "Open dataset",
                    "href": f"/datasets/{source.dataset_id}",
                }
            )
    elif domain_type == "acquisition_batch" and domain_id:
        next_actions.append("open_source")
        action_links.append(
            {
                "id": "open_source",
                "label": "Open Source",
                "href": f"/datasets/review?batch={domain_id}",
            }
        )
    elif domain_type == "review_suggestion":
        review_item_id = str(launch_spec.get("item_id") or "")
        if review_item_id:
            row = database._conn.execute(
                "SELECT queue_id FROM review_items WHERE id=?", (review_item_id,)
            ).fetchone()
            if row is not None:
                next_actions.append("resume_review")
                action_links.append(
                    {
                        "id": "resume_review",
                        "label": "Resume Review",
                        "href": f"/datasets/review/{row['queue_id']}?item={review_item_id}",
                    }
                )
    elif domain_type == "label_set_revision" and domain_id:
        row = database._conn.execute(
            """SELECT s.queue_id FROM label_set_revisions r
               JOIN label_sets s ON s.id=r.label_set_id WHERE r.id=?""",
            (domain_id,),
        ).fetchone()
        if row is not None:
            next_actions.append("resume_review")
            action_links.append(
                {
                    "id": "resume_review",
                    "label": "Resume Review",
                    "href": f"/datasets/review/{row['queue_id']}",
                }
            )
    elif domain_type == "dataset_job" and domain_id and item.status == "completed":
        job = database.get_dataset_job(str(domain_id))
        version_id = str((item.result or {}).get("version_id") or (job.version_id if job else ""))
        dataset_id = str((job.dataset_id if job else "") or "")
        if dataset_id and version_id:
            next_actions.append("open_child_version")
            action_links.append(
                {
                    "id": "open_child_version",
                    "label": "Open Child Version",
                    "href": f"/datasets/{dataset_id}/versions/{version_id}",
                }
            )
    elif domain_type == "reward_integrity_audit" and domain_id:
        audit_row = database._conn.execute(
            "SELECT run_id FROM reward_integrity_audits WHERE id=?", (domain_id,)
        ).fetchone()
        decision_row = database._conn.execute(
            "SELECT action FROM reward_integrity_decisions WHERE audit_id=? "
            "ORDER BY created_at DESC,id DESC LIMIT 1",
            (domain_id,),
        ).fetchone()
        coverage_row = database._conn.execute(
            "SELECT value FROM reward_integrity_metrics "
            "WHERE audit_id=? AND name='paired_coverage' AND subgroup='' "
            "AND population='uniform_core' AND available=1 LIMIT 1",
            (domain_id,),
        ).fetchone()
        if coverage_row is not None and coverage_row["value"] is not None:
            summary_metrics["paired_coverage"] = float(coverage_row["value"])
        requires_review = bool(
            decision_row is not None and str(decision_row["action"]) == "pause"
        )
        if requires_review:
            display_status = "awaiting_review"
            next_actions.extend(["continue", "stop", "fork"])
        next_actions.append("open_audit")
        run_id = str(audit_row["run_id"] if audit_row is not None else "")
        href = (
            f"/runs/{run_id}?tab=evaluation&evidence=training-audits&audit={domain_id}"
            if run_id
            else (
                "/eval?section=verifiers&verifierView=training-audits"
                f"&audit={domain_id}"
            )
        )
        action_links.append(
            {"id": "open_audit", "label": "Open Audit", "href": href}
        )
    elif domain_type in {"model_preparation", "training_capacity_check"} and domain_id:
        table = (
            "model_preparations"
            if domain_type == "model_preparation"
            else "training_capacity_checks"
        )
        row = database._conn.execute(
            f"SELECT plan_revision_id FROM {table} WHERE id=?",
            (domain_id,),
        ).fetchone()
        if row is not None:
            revision_id = str(row["plan_revision_id"])
            next_actions.append("return_to_training")
            action_links.append(
                {
                    "id": "return_to_training",
                    "label": "Return to training",
                    "href": (
                        "/datasets/new?trainingPlanRevision="
                        f"{revision_id}"
                    ),
                }
            )
            title = (
                "Prepare the recommended model"
                if domain_type == "model_preparation"
                else "Check workstation fit"
            )
    elif domain_type in {"runtime_preparation", "runtime_qualification"} and domain_id:
        next_actions.append("open_runtime")
        action_links.append(
            {
                "id": "open_runtime",
                "label": "Open runtime setup",
                "href": "/setup",
            }
        )
        title = (
            "Prepare training runtime"
            if domain_type == "runtime_preparation"
            else "Verify training runtime"
        )
    waiting_for_accelerator = (
        item.status == "queued" and item.stage == "waiting_for_accelerator"
    )
    if waiting_for_accelerator:
        display_status = "waiting_for_accelerator"
        title = "Waiting for accelerator"
        availability = dict((item.result or {}).get("availability") or {})
        for owner in availability.get("owners") or ():
            executable = owner.get("executable") or "external process"
            pid = owner.get("pid")
            elapsed = owner.get("elapsed_seconds")
            detail = f"{executable} · PID {pid}" if pid is not None else str(executable)
            if elapsed is not None:
                detail += f" · {int(elapsed)}s"
            blockers.append(detail)
    return {
        "id": item.id,
        "work_item_id": item.id,
        "domain_id": domain_id,
        "domain_type": domain_type,
        "kind": item.kind,
        "title": title,
        "status": display_status,
        "stage": item.stage,
        "priority": item.priority,
        "progress_current": current,
        "progress_total": total,
        "progress_percent": progress_percent,
        "queue_position": database.work_item_queue_position(item.id),
        "eta_seconds": _eta_seconds(item, current, total),
        "blockers": blockers,
        "resource_requirements": dict(item.resource_requirements),
        "worker_id": item.worker_id,
        "attempt": int(item.retry_count) + 1,
        "max_attempts": int(item.max_retries) + 1,
        "attempts": attempts,
        "events": events,
        "logs": logs,
        "error": None if waiting_for_accelerator else item.error,
        "created_at": item.created_at,
        "started_at": item.started_at,
        "completed_at": item.completed_at,
        "heartbeat_at": item.heartbeat_at,
        "telemetry_rollup": telemetry_rollup,
        "summary_metrics": summary_metrics,
        "next_actions": next_actions,
        "action_links": action_links,
    }


__all__ = [
    "activity_item_view",
    "blocker_messages",
    "normalize_work_attempt",
    "normalize_work_event",
]
