from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import BinaryIO, Mapping, Optional, Sequence

from halo_forge.run_db import LabV4Catalog, RunDatabase
from halo_forge.orchestration import OrchestrationService
from halo_forge.workstation_jobs import (
    DiskCapacity,
    MemoryCapacity,
    WorkstationCapacity,
    WorkstationScheduler,
    process_start_time,
)
from halo_forge.workstation_jobs.worker import (
    ProcessHandle,
    SubprocessRunner,
    WorkstationWorker,
    main,
)


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def _available_capacity(_path: Path) -> WorkstationCapacity:
    gib = 1024**3
    return WorkstationCapacity(
        sampled_at=datetime.now(timezone.utc),
        disk=DiskCapacity(
            path="/tmp/halo-forge-tests",
            total_bytes=200 * gib,
            used_bytes=100 * gib,
            free_bytes=100 * gib,
        ),
        memory=MemoryCapacity(
            total_bytes=32 * gib,
            used_bytes=8 * gib,
            available_bytes=24 * gib,
        ),
    )


class _StubbornProcess:
    pid = 9876

    def __init__(self, cancel) -> None:
        self._cancel = cancel
        self._cancelled = False
        self.terminated = 0
        self.killed = 0
        self.return_code: Optional[int] = None

    def poll(self) -> Optional[int]:
        if not self._cancelled:
            self._cancelled = True
            self._cancel()
        return self.return_code

    def terminate(self) -> None:
        self.terminated += 1

    def kill(self) -> None:
        self.killed += 1
        self.return_code = -9


class _FakeRunner:
    def __init__(self, process: ProcessHandle) -> None:
        self.process = process
        self.calls: list[dict[str, object]] = []

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Optional[str],
        env: Mapping[str, str],
        log_file: BinaryIO,
    ) -> ProcessHandle:
        self.calls.append({"command": list(command), "cwd": cwd, "env": dict(env)})
        log_file.write(b"fake process output\n")
        return self.process


def test_worker_executes_argv_captures_log_result_and_return_code(tmp_path):
    db = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(db, worker_id="worker", lease_ttl_seconds=2, capacity_probe=_available_capacity)
    log_path = tmp_path / "worker.log"
    result_path = tmp_path / "result.json"
    command = [
        sys.executable,
        "-c",
        (
            "import json, os; "
            "print(os.environ['WORKER_TEST_VALUE']); "
            f"json.dump({{'metric': 0.75}}, open({str(result_path)!r}, 'w'))"
        ),
    ]
    scheduler.enqueue(
        kind="train",
        launch_spec={
            "command": command,
            "cwd": str(tmp_path),
            "env": {"WORKER_TEST_VALUE": "ready"},
            "result_path": str(result_path),
        },
        log_path=str(log_path),
    )
    worker = WorkstationWorker(
        scheduler,
        runner=SubprocessRunner(),
        poll_interval=0.01,
        heartbeat_interval=0.05,
        process_identity=lambda _pid: 123.0,
    )

    finished = worker.run_once()

    assert finished is not None and finished.status == "completed"
    assert finished.result["return_code"] == 0
    assert finished.result["command"] == command
    assert finished.result["command_result"] == {"metric": 0.75}
    assert finished.result["result_path"] == str(result_path)
    assert finished.result["process_started_at"] == 123.0
    assert "ready" in log_path.read_text(encoding="utf-8")
    assert db.list_resource_leases() == []


def test_worker_honors_cancel_with_term_then_kill_and_finishes_cancelled(tmp_path):
    db = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(db, worker_id="worker", lease_ttl_seconds=2, capacity_probe=_available_capacity)
    queued = scheduler.enqueue(
        kind="evaluate",
        launch_spec={"command": ["fake", "--argument"]},
        log_path=str(tmp_path / "cancel.log"),
    )
    process = _StubbornProcess(lambda: scheduler.cancel(queued.id))
    runner = _FakeRunner(process)
    clock = _FakeClock()
    worker = WorkstationWorker(
        scheduler,
        runner=runner,
        poll_interval=0.1,
        heartbeat_interval=0.1,
        terminate_timeout=0.2,
        process_identity=lambda _pid: 456.0,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    finished = worker.run_once()

    assert finished is not None and finished.status == "cancelled"
    assert finished.result["return_code"] == -9
    assert finished.result["cancelled"] is True
    assert process.terminated == 1
    assert process.killed == 1
    assert runner.calls[0]["command"] == ["fake", "--argument"]
    assert "fake process output" in Path(finished.result["log_path"]).read_text()


def test_worker_rejects_string_commands_without_invoking_a_shell(tmp_path):
    db = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(db, worker_id="worker", capacity_probe=_available_capacity)
    scheduler.enqueue(
        kind="train",
        launch_spec={"command": "touch should-not-exist"},
        log_path=str(tmp_path / "invalid.log"),
    )

    finished = WorkstationWorker(scheduler).run_once()

    assert finished is not None and finished.status == "failed"
    assert "argv list" in (finished.error or "")
    assert not (tmp_path / "should-not-exist").exists()


def test_training_plan_domain_transaction_is_isolated_from_heartbeat(tmp_path, monkeypatch):
    from halo_forge.training_plan import TrainingPlanService

    db = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(
        db,
        worker_id="worker",
        lease_ttl_seconds=2,
        capacity_probe=_available_capacity,
    )
    scheduler.enqueue(
        kind="training_capacity_check",
        launch_spec={
            "handler": "training_plan.execute_work_item",
            "training_plan_root": str(tmp_path / "training-plan"),
        },
        domain_kind="transaction_probe",
        domain_id="probe",
    )

    def hold_domain_transaction(self, _item):
        self.conn.execute("BEGIN IMMEDIATE")
        self.conn.execute(
            "UPDATE schema_meta SET value=value WHERE key='schema_version'"
        )
        time.sleep(0.12)
        assert self.conn._connection.in_transaction, "heartbeat committed the domain transaction"
        self.conn.commit()
        return {"transaction_isolated": True}

    monkeypatch.setattr(TrainingPlanService, "execute_work_item", hold_domain_transaction)

    finished = WorkstationWorker(
        scheduler,
        heartbeat_interval=0.02,
        telemetry_sampler=lambda path, **_: _available_capacity(Path(path)),
    ).run_once()

    assert finished is not None and finished.status == "completed", finished
    assert finished.result["transaction_isolated"] is True


def test_worker_waits_when_serving_lease_blocks_claim(tmp_path):
    db = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(db, worker_id="worker", capacity_probe=_available_capacity)
    queued = scheduler.enqueue(kind="train", launch_spec={"command": ["unused"]})
    scheduler.start_serving(serving_id="model-server")

    assert WorkstationWorker(scheduler).run_once() is None
    assert db.get_work_item(queued.id).status == "queued"  # type: ignore[union-attr]


def test_module_main_once_accepts_database_and_empty_queue(tmp_path):
    assert (
        main(["--once", "--database", str(tmp_path / "empty.db"), "--poll-interval", "0.01"]) == 0
    )


def test_process_start_identity_has_a_standard_library_fallback():
    # psutil is optional; the worker must still be executable from the base
    # installation on Linux and macOS.
    assert process_start_time(os.getpid()) is not None


def test_worker_publishes_segment_artifacts_and_resolves_checkpoint_commands(tmp_path):
    db = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(db, worker_id="worker", capacity_probe=_available_capacity)
    service = OrchestrationService(db, scheduler=scheduler)
    suite = db.create_benchmark_suite(name="development", purpose="development")
    revision = db.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash="development-v1",
        items=[{"id": "one", "input": "x", "expected": "y"}],
        primary_metric="accuracy",
        direction="maximize",
    )
    created = service.create_group_from_payload(
        {
            "name": "gated",
            "kind": "sweep",
            "trainer_mode": "sft",
            "base_config": {
                "model": "local/model",
                "output_root": str(tmp_path / "outputs"),
                "data": str(tmp_path / "train.jsonl"),
            },
            "search_space": {"learning_rate": [0.1, 0.2, 0.3]},
            "n_trials": 3,
            "sampler": "grid",
            "seeds": [7],
            "pruning": {"enabled": True, "budgets": [3, 9]},
            "development_suite_revision_id": revision.id,
        }
    )
    worker = WorkstationWorker(scheduler)

    for index, trial in enumerate(created["trials"]):
        run = trial["runs"][0]
        training_data = next(value for value in run["work_items"] if value["kind"] == "training")
        evaluation_data = next(
            value for value in run["work_items"] if value["kind"] == "evaluation"
        )
        training = db.get_work_item(training_data["id"])
        assert training is not None
        output_dir = Path(training.launch_spec["output_dir"])
        checkpoint = output_dir / "checkpoint-3"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "adapter.bin").write_bytes(f"trial-{index}".encode())
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": 3}), encoding="utf-8"
        )

        publication = {}
        worker._publish_training_artifacts(training, training.launch_spec, publication)
        artifact = db.get_model_artifact(publication["model_artifact_id"])
        segment = db.get_trial_segment(training.launch_spec["trial_segment_id"])
        assert artifact is not None and artifact.step == 3
        managed_checkpoint = Path(publication["model_artifact_path"])
        assert artifact.path == str(managed_checkpoint)
        assert managed_checkpoint.is_dir()
        locations = LabV4Catalog(db).list_locations(publication["artifact_blob_id"])
        assert {value.storage_mode for value in locations} == {"managed", "referenced"}
        assert segment is not None and segment.checkpoint_artifact_id == artifact.id

        # Gated execution must survive trainer-owned checkpoint rotation.
        shutil.rmtree(checkpoint)

        claimed_training = scheduler.claim(work_item_id=training.id)
        assert claimed_training is not None
        scheduler.complete(claimed_training, result=publication)

        evaluation = db.get_work_item(evaluation_data["id"])
        assert evaluation is not None
        resolved_eval = worker._prepare_launch_spec(evaluation)
        assert resolved_eval["resolved_checkpoint_path"] == str(managed_checkpoint)
        assert resolved_eval["command"][-1] == "--wait"
        assert resolved_eval["command"][resolved_eval["command"].index("--subject") + 1] == str(
            managed_checkpoint
        )
        claimed_evaluation = scheduler.claim(work_item_id=evaluation.id)
        assert claimed_evaluation is not None
        scheduler.complete(
            claimed_evaluation,
            result={"metrics": {"accuracy": [0.1, 0.9, 0.4][index]}},
        )

    decision = service.advance_successive_halving(created["id"], rung_index=0)
    assert decision["ready"] is True
    promoted = db.get_run_group_trial(decision["promoted_trial_keys"][0])
    assert promoted is not None
    promoted_run = db.list_trial_runs(promoted.id)[0]
    segments = db.list_trial_segments(promoted_run.id)
    next_training = db.get_work_item(segments[-1].work_item_id)
    assert next_training is not None
    resolved_training = worker._prepare_launch_spec(next_training)
    assert resolved_training["command"][-2] == "--resume"
    assert Path(resolved_training["command"][-1]).name == "payload"
    assert Path(resolved_training["command"][-1]).is_dir()
    assert (
        resolved_training["command"][resolved_training["command"].index("--max-steps") + 1] == "9"
    )
