from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Mapping, Optional, Sequence

from halo_forge.public_api.service import PublicApiService
from halo_forge.run_db import RunDatabase
from halo_forge.runtime_determinism import RUN_ID_ENV
from halo_forge.workstation_jobs import WorkstationScheduler
from halo_forge.workstation_jobs.resources import (
    DiskCapacity,
    MemoryCapacity,
    WorkstationCapacity,
)
from halo_forge.workstation_jobs.worker import ProcessHandle, WorkstationWorker
from ui.services.results_service import ResultsService
from ui.state import AppState


class _CompletedProcess:
    pid = 4312

    def poll(self) -> Optional[int]:
        return 0

    def terminate(self) -> None:
        raise AssertionError("completed process should not be terminated")

    def kill(self) -> None:
        raise AssertionError("completed process should not be killed")


class _TrainingOutputRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Optional[str],
        env: Mapping[str, str],
        log_file: BinaryIO,
    ) -> ProcessHandle:
        argv = list(command)
        output_dir = Path(argv[argv.index("--output") + 1])
        final_model = output_dir / "final_model"
        final_model.mkdir(parents=True)
        (final_model / "adapter_config.json").write_text("{}", encoding="utf-8")
        (final_model / "adapter_model.safetensors").write_bytes(b"trained-adapter")
        (output_dir / "training_summary.json").write_text(
            json.dumps(
                {
                    "run_id": env[RUN_ID_ENV],
                    "modality": "sft",
                    "model_name": "local/model",
                    "training_steps": 2,
                    "total_train_steps_executed": 2,
                    "final_train_loss": 0.625,
                    "weights_updated": True,
                }
            ),
            encoding="utf-8",
        )
        log_file.write(b"managed trainer finished\n")
        self.calls.append({"command": argv, "cwd": cwd, "env": dict(env), "output_dir": output_dir})
        return _CompletedProcess()


def _ample_capacity(path: Path) -> WorkstationCapacity:
    gib = 1024**3
    return WorkstationCapacity(
        sampled_at=datetime.now(timezone.utc),
        disk=DiskCapacity(
            path=str(path), total_bytes=500 * gib, used_bytes=100 * gib, free_bytes=400 * gib
        ),
        memory=MemoryCapacity(
            total_bytes=64 * gib,
            used_bytes=8 * gib,
            available_bytes=56 * gib,
            source="test",
        ),
    )


def test_managed_api_training_is_scheduled_staged_published_and_cataloged(tmp_path):
    database = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(
        database,
        worker_id="managed-worker",
        capacity_probe=_ample_capacity,
    )
    state = AppState()
    service = PublicApiService(
        database=database,
        app_state=state,
        results_service=ResultsService(base_path=tmp_path),
        base_path=tmp_path,
        artifact_storage_root=tmp_path / "artifacts",
        workstation_scheduler=scheduler,
    )
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"prompt":"hello","response":"world"}\n', encoding="utf-8")
    run_root = tmp_path / "managed-runs"

    launched = asyncio.run(
        service.launch_training(
            {
                "mode": "sft",
                "model": "local/model",
                "dataset": str(dataset),
                "output_dir": str(tmp_path / "legacy-unused"),
                "output_root": str(run_root),
                "epochs": 1,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "seed": 17,
            }
        )
    )

    run_id = launched["run_id"]
    work_item_id = launched["work_item_id"]
    canonical_output = run_root / run_id
    assert launched["status"] == "pending"
    assert launched["managed"] is True
    assert not canonical_output.exists()
    queued = database.get_work_item(work_item_id)
    assert queued is not None and queued.canonical_run_id == run_id
    assert queued.launch_spec["resolved_launch_config"]["run_id"] == run_id
    assert queued.launch_spec["output_dir"] == str(canonical_output)

    runner = _TrainingOutputRunner()
    finished = WorkstationWorker(
        scheduler,
        runner=runner,
        process_identity=lambda _pid: 123.0,
        telemetry_sampler=lambda path, **_: _ample_capacity(Path(path)),
    ).run_once()

    assert finished is not None and finished.status == "completed", finished
    assert canonical_output.is_dir()
    attempt_output = Path(str(runner.calls[0]["output_dir"]))
    assert attempt_output != canonical_output
    assert ".halo-forge-attempts" in attempt_output.parts
    assert not attempt_output.exists()
    summary = json.loads((canonical_output / "training_summary.json").read_text())
    replay = json.loads((canonical_output / "replay.json").read_text())
    launch_context = json.loads((canonical_output / "launch_context.json").read_text())
    assert summary["run_id"] == replay["run_id"] == run_id
    assert summary["output_dir"] == str(canonical_output)
    assert replay["config"]["output_dir"] == str(canonical_output)
    assert launch_context["args"]["output_dir"] == str(canonical_output)
    assert launch_context["metadata"]["work_item_id"] == work_item_id

    run = database.get_run(run_id)
    assert run is not None and run.status == "completed"
    assert run.output_dir == str(canonical_output)
    assert run.weights_updated is True
    legacy_artifacts = database.list_model_artifacts(run_id=run_id)
    occurrences = service._artifact_studio_engine().catalog.list_occurrences(run_id=run_id)
    assert len(legacy_artifacts) == 1
    assert legacy_artifacts[0].artifact_kind == "adapter"
    assert len(occurrences) == 1
    assert occurrences[0].artifact_kind == "adapter"
    assert finished.result["canonical_run_id"] == run_id
    assert finished.result["artifact_occurrence_ids"] == [occurrences[0].id]

    detail = service.get_run_detail(run_id)
    assert detail["status"] == "completed"
    assert detail["metrics_summary"]["update_steps"] == 2
    assert detail["metrics_summary"]["final_train_loss"] == 0.625
    assert detail["work_items"][0]["id"] == work_item_id

    restarted = PublicApiService(
        database=database,
        app_state=AppState(),
        results_service=ResultsService(base_path=tmp_path),
        base_path=tmp_path,
        artifact_storage_root=tmp_path / "artifacts",
        workstation_scheduler=WorkstationScheduler(database, worker_id="restarted"),
    )
    recovered = restarted.get_run_detail(run_id)
    assert recovered["run_id"] == run_id
    assert recovered["status"] == "completed"
    assert recovered["work_items"][0]["id"] == work_item_id


def test_serving_work_waits_for_accelerator_and_runs_capacity_preflight(tmp_path):
    database = RunDatabase(str(tmp_path / "runs.db"))
    ample = WorkstationScheduler(
        database,
        worker_id="worker",
        capacity_probe=_ample_capacity,
    )
    heavy = ample.enqueue(kind="training", launch_spec={"command": ["unused"]})
    assert ample.claim(work_item_id=heavy.id) is not None
    serving = ample.enqueue(
        kind="artifact_serving",
        launch_spec={"command": ["unused"]},
        resource_class="none",
        resource_requirements={"lease_type": "serving", "output_path": str(tmp_path)},
    )
    assert ample.claim(work_item_id=serving.id) is None
    assert database.get_work_item(serving.id).status == "queued"  # type: ignore[union-attr]

    claimed_heavy = database.get_work_item(heavy.id)
    assert claimed_heavy is not None
    ample.complete(claimed_heavy)
    gib = 1024**3
    low_capacity = WorkstationScheduler(
        database,
        worker_id="low-capacity-worker",
        capacity_probe=lambda path: WorkstationCapacity(
            sampled_at=datetime.now(timezone.utc),
            disk=DiskCapacity(
                path=str(path), total_bytes=100 * gib, used_bytes=95 * gib, free_bytes=5 * gib
            ),
            memory=MemoryCapacity(
                total_bytes=16 * gib,
                used_bytes=15 * gib,
                available_bytes=1 * gib,
                source="test",
            ),
        ),
    )
    assert low_capacity.claim(work_item_id=serving.id) is None
    blocked = database.get_work_item(serving.id)
    assert blocked is not None and blocked.status == "blocked"
    assert blocked.stage == "blocked_capacity"
    assert "disk" in " ".join(blocked.result["preflight"]["blockers"]).lower()
