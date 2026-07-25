"""Durable direct-run segment planning for reward-audited managed training.

The public Train surface launches one canonical run, but resumable trainers
execute one bounded process per selected audit boundary.  This module keeps
the command transformation and next-segment enqueue path deterministic so an
automatic pass and an operator-reviewed Continue perform the same operation.
"""

from __future__ import annotations

import copy
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


SEGMENT_OPERATION = "managed_training_segment"


def resolve_boundary_values(
    boundaries: Sequence[str | int], *, total: int, resumable: bool
) -> list[int]:
    """Resolve an ordered audit schedule, always including the final bound."""

    total = max(1, int(total))
    raw_values = list(boundaries or ("final",))
    values: list[int] = []
    for raw_value in raw_values:
        raw = str(raw_value).strip().lower()
        value = total if raw in {"final", "last"} else int(raw.rsplit(":", 1)[-1])
        if not 1 <= value <= total:
            raise ValueError(f"reward audit boundary {raw_value!r} is outside 1..{total}")
        if value not in values:
            values.append(value)
    if values != sorted(values):
        raise ValueError("reward audit boundaries must be ordered")
    if total not in values:
        values.append(total)
    if not resumable and values != [total]:
        raise ValueError("this trainer supports a final-boundary audit only")
    return values


def _set_single_flag(command: list[str], flag: str, value: str) -> None:
    try:
        index = command.index(flag)
    except ValueError:
        command.extend([flag, value])
        return
    if index + 1 >= len(command):
        raise ValueError(f"managed training command has no value for {flag}")
    command[index + 1] = value


def _remove_repeated_flag(command: list[str], flag: str) -> None:
    while flag in command:
        index = command.index(flag)
        del command[index : min(len(command), index + 2)]


def command_for_segment(
    canonical_command: Sequence[str],
    *,
    mode: str,
    backend: str,
    start_value: int,
    end_value: int,
    unit: Optional[str] = None,
) -> list[str]:
    """Render a trainer argv that stops exactly at one selected boundary."""

    command = [str(value) for value in canonical_command]
    if unit == "final":
        _remove_repeated_flag(command, "--reward-audit-boundary")
        command.extend(["--reward-audit-boundary", "final"])
        return command
    if mode in {"raft", "vlm", "audio", "reasoning", "agentic"}:
        _set_single_flag(command, "--cycles", str(int(end_value)))
    elif mode == "grpo" and backend == "hf":
        _set_single_flag(command, "--max-steps", str(int(end_value)))
        _set_single_flag(command, "--save-steps", str(int(end_value)))
    else:
        _set_single_flag(command, "--epochs", str(int(end_value)))

    # Each process owns exactly one shard.  Future boundary flags would make
    # the first process attempt to create evidence for work it did not run.
    _remove_repeated_flag(command, "--reward-audit-boundary")
    command.extend(["--reward-audit-boundary", str(int(end_value))])

    if mode in {"vlm", "audio", "reasoning", "agentic"}:
        _set_single_flag(command, "--resume-from-cycle", str(int(start_value)))
    elif mode == "raft" and backend != "mlx" and start_value:
        raise ValueError("PyTorch RAFT cannot resume a managed boundary segment")
    elif mode == "grpo" and backend == "mlx" and start_value:
        raise ValueError("MLX GRPO supports a final-boundary audit only")
    return command


def segment_snapshot_path(
    canonical_output: str | Path, *, run_id: str, ordinal: int
) -> Path:
    output = Path(canonical_output).expanduser().resolve()
    return (
        output.parent
        / ".halo-forge-segments"
        / str(run_id)
        / f"segment-{int(ordinal):04d}"
    )


def build_segment_launch_spec(
    base_launch_spec: Mapping[str, Any],
    *,
    segment: Mapping[str, Any],
    previous_segment: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build one immutable scheduler launch spec from the canonical launch."""

    spec = copy.deepcopy(dict(base_launch_spec))
    config = dict(spec.get("resolved_launch_config") or {})
    mode = str(config.get("mode") or "")
    accelerator = str(config.get("accelerator") or "").strip().lower()
    model = str(config.get("model") or "")
    backend = str(
        config.get("_resolved_signal_backend")
        or ("mlx" if accelerator == "mlx" or model.startswith("mlx-community/") else "hf")
    )
    run_id = str(config.get("run_id") or "")
    output = str(spec.get("output_dir") or config.get("output_dir") or "")
    if not mode or not run_id or not output:
        raise ValueError("direct segment launch is missing mode, run ID, or output")
    canonical_command = list(spec.get("canonical_command") or spec.get("command") or ())
    spec.update(
        operation=SEGMENT_OPERATION,
        command=command_for_segment(
            canonical_command,
            mode=mode,
            backend=backend,
            start_value=int(segment["start_value"]),
            end_value=int(segment["end_value"]),
            unit=str(segment.get("unit") or ""),
        ),
        canonical_command=canonical_command,
        direct_run_segment_id=str(segment["id"]),
        direct_run_segment_ordinal=int(segment["ordinal"]),
        direct_run_segment_start=int(segment["start_value"]),
        direct_run_segment_end=int(segment["end_value"]),
        segment_output_dir=str(
            segment_snapshot_path(output, run_id=run_id, ordinal=int(segment["ordinal"]))
        ),
        previous_segment_output_dir=(
            str(
                segment_snapshot_path(
                    output,
                    run_id=run_id,
                    ordinal=int(previous_segment["ordinal"]),
                )
            )
            if previous_segment is not None
            else None
        ),
        final_segment=bool(segment.get("is_final", False)),
        resume_checkpoint_pattern=(
            f"checkpoint-{int(segment['start_value'])}"
            if mode == "grpo" and backend == "hf" and int(segment["start_value"])
            else None
        ),
    )
    return spec


def enqueue_next_direct_segment(
    database: Any,
    scheduler: Any,
    *,
    current_segment_id: str,
    dependency_work_item_id: Optional[str],
) -> Optional[Any]:
    """Idempotently enqueue the segment after an accepted audit decision."""

    current = database.get_direct_run_segment(str(current_segment_id))
    if current is None:
        raise KeyError(f"unknown direct-run segment: {current_segment_id}")
    segments = database.list_direct_run_segments(str(current["run_id"]))
    next_segment = next(
        (value for value in segments if int(value["ordinal"]) == int(current["ordinal"]) + 1),
        None,
    )
    if next_segment is None:
        return None
    if next_segment.get("work_item_id"):
        return database.get_work_item(str(next_segment["work_item_id"]))
    source_work_id = str(current.get("work_item_id") or "")
    source_work = database.get_work_item(source_work_id) if source_work_id else None
    if source_work is None:
        raise ValueError("current direct-run segment has no durable training work item")
    launch_spec = build_segment_launch_spec(
        source_work.launch_spec,
        segment={
            **next_segment,
            "is_final": int(next_segment["ordinal"]) == len(segments) - 1,
        },
        previous_segment=current,
    )
    work_id = f"training-segment-work-{next_segment['id']}"
    dependencies = [str(dependency_work_item_id)] if dependency_work_item_id else []
    try:
        item = scheduler.enqueue(
            kind="training",
            launch_spec=launch_spec,
            resource_class="accelerator",
            resource_requirements=dict(source_work.resource_requirements),
            domain_kind="run",
            domain_id=str(current["run_id"]),
            canonical_run_id=str(current["run_id"]),
            log_path=source_work.log_path,
            dependencies=dependencies,
            max_retries=source_work.max_retries,
            work_item_id=work_id,
        )
    except sqlite3.IntegrityError:
        item = database.get_work_item(work_id)
        if item is None:
            raise
    database.update_direct_run_segment(
        str(next_segment["id"]),
        status="queued",
        work_item_id=item.id,
    )
    run = database.get_run(str(current["run_id"]))
    if run is not None and run.status not in {"stopped", "cancelled", "failed"}:
        with database._lock:
            database._conn.execute(
                "UPDATE runs SET status='queued' WHERE run_id=?",
                (str(current["run_id"]),),
            )
            database._conn.commit()
    return item


__all__ = [
    "SEGMENT_OPERATION",
    "build_segment_launch_spec",
    "command_for_segment",
    "enqueue_next_direct_segment",
    "resolve_boundary_values",
    "segment_snapshot_path",
]
