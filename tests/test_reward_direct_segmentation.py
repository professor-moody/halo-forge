from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

from halo_forge.cli import _enqueue_managed_reward_training
from pathlib import Path

from halo_forge.public_api.service import PublicApiService
from halo_forge.reward_integrity import RewardIntegrityService
from halo_forge.reward_integrity.direct_segments import (
    SEGMENT_OPERATION,
    build_segment_launch_spec,
    command_for_segment,
    enqueue_next_direct_segment,
    resolve_boundary_values,
)
from halo_forge.run_db import RunDatabase, RunRecord
from halo_forge.training_signal import TRAINING_SIGNAL_CAPABILITIES
from halo_forge.workstation_jobs import WorkstationScheduler
from halo_forge.workstation_jobs.worker import WorkstationWorker
from halo_forge.verifier_lab.store import VerifierLabStore
from ui.services.results_service import ResultsService
from ui.state import AppState


def _public_service(tmp_path: Path, database: RunDatabase) -> PublicApiService:
    return PublicApiService(
        database=database,
        app_state=AppState(),
        results_service=ResultsService(base_path=tmp_path),
        base_path=tmp_path,
        artifact_storage_root=tmp_path / "artifacts",
        workstation_scheduler=WorkstationScheduler(database),
    )


def _verifier_revision(
    database: RunDatabase, name: str, *, family: str, endpoint_type: str = ""
):
    store = VerifierLabStore(database)
    profile = store.create_profile(name=name)
    revision = store.create_profile_revision(
        profile.id,
        {
            "family": family,
            "implementation": {
                "kind": "builtin",
                "ref": "regex_format",
                "fingerprint": f"{name}-fingerprint",
            },
            "endpoint_type": endpoint_type,
            "modality": "text",
            "task_type": "binary",
            "reward_contract": {
                "minimum": 0,
                "maximum": 1,
                "direction": "maximize",
                "threshold": 0.5,
            },
            "runtime_requirements": {},
        },
    )
    database._conn.execute(
        "INSERT INTO verifier_aliases (profile_id,alias,revision_id,updated_at) "
        "VALUES (?,?,?,'now')",
        (profile.id, "candidate", revision.id),
    )
    database._conn.commit()
    return revision


def test_capabilities_only_claim_verified_process_resume():
    assert TRAINING_SIGNAL_CAPABILITIES.get("raft:hf").resumable is False
    assert TRAINING_SIGNAL_CAPABILITIES.get("raft:hf").available_boundaries == ("final",)
    assert TRAINING_SIGNAL_CAPABILITIES.get("raft:mlx").resumable is True
    assert TRAINING_SIGNAL_CAPABILITIES.get("grpo:hf").resumable is True
    assert TRAINING_SIGNAL_CAPABILITIES.get("grpo:hf").boundary_unit == "step"
    assert TRAINING_SIGNAL_CAPABILITIES.get("grpo:mlx").resumable is False
    for capability_id in ("vlm:hf", "audio:hf", "reasoning:hf", "agentic:hf"):
        assert TRAINING_SIGNAL_CAPABILITIES.get(capability_id).resumable is True


def test_top_level_audited_cli_launch_enqueues_managed_boundaries(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "runs.db"
    output = tmp_path / "managed-run"
    monkeypatch.setenv("HALOFORGE_RUN_DB_PATH", str(database_path))
    monkeypatch.setenv("HALO_FORGE_RUN_ID", "cli-managed-run")
    monkeypatch.delenv("HALOFORGE_WORK_ITEM_ID", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "halo-forge",
            "grpo",
            "train",
            "--model",
            "model",
            "--data",
            "prompts.jsonl",
            "--output",
            str(output),
            "--max-steps",
            "12",
        ],
    )
    args = Namespace(
        reward_system_revision="reward-revision",
        reward_audit_boundary=["3", "6", "final"],
        max_steps=12,
        output=str(output),
        model="model",
        accelerator="cuda",
        seed=42,
        wait=False,
        dry_run=False,
    )
    assert _enqueue_managed_reward_training(args, trainer="grpo") is True

    database = RunDatabase(str(database_path))
    run = database.get_run("cli-managed-run")
    assert run is not None and run.status == "queued"
    segments = database.list_direct_run_segments("cli-managed-run")
    assert [(value["start_value"], value["end_value"]) for value in segments] == [
        (0, 3),
        (3, 6),
        (6, 12),
    ]
    work = database.list_work_items(canonical_run_id="cli-managed-run", limit=10)
    assert len(work) == 1
    assert work[0].launch_spec["operation"] == SEGMENT_OPERATION
    assert work[0].launch_spec["command"][
        work[0].launch_spec["command"].index("--max-steps") + 1
    ] == "3"

    monkeypatch.setenv("HALOFORGE_WORK_ITEM_ID", work[0].id)
    assert _enqueue_managed_reward_training(args, trainer="grpo") is False


def test_managed_cli_setup_failure_is_projected_on_run_and_segments(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "runs.db"
    output = tmp_path / "managed-run"
    monkeypatch.setenv("HALOFORGE_RUN_DB_PATH", str(database_path))
    monkeypatch.setenv("HALO_FORGE_RUN_ID", "cli-managed-failed-setup")
    monkeypatch.delenv("HALOFORGE_WORK_ITEM_ID", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "halo-forge",
            "grpo",
            "train",
            "--model",
            "model",
            "--data",
            "prompts.jsonl",
            "--output",
            str(output),
            "--max-steps",
            "12",
        ],
    )

    def _fail_enqueue(*_args, **_kwargs):
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(WorkstationScheduler, "enqueue", _fail_enqueue)
    args = Namespace(
        reward_system_revision="reward-revision",
        reward_audit_boundary=["3", "final"],
        max_steps=12,
        output=str(output),
        model="model",
        accelerator="cuda",
        seed=42,
        wait=False,
        dry_run=False,
    )

    try:
        _enqueue_managed_reward_training(args, trainer="grpo")
    except RuntimeError as exc:
        assert "queue unavailable" in str(exc)
    else:
        raise AssertionError("managed setup failure was not propagated")

    database = RunDatabase(str(database_path))
    run = database.get_run("cli-managed-failed-setup")
    assert run is not None and run.status == "failed"
    assert "queue unavailable" in str(run.failure_reason)
    segments = database.list_direct_run_segments("cli-managed-failed-setup")
    assert segments and all(segment["status"] == "failed" for segment in segments)


def test_segment_command_stops_and_resumes_at_exact_cycle_boundary():
    base = [
        "python",
        "-m",
        "halo_forge.cli",
        "reasoning",
        "train",
        "--cycles",
        "6",
        "--resume-from-cycle",
        "0",
        "--reward-audit-boundary",
        "2",
        "--reward-audit-boundary",
        "final",
    ]
    command = command_for_segment(
        base, mode="reasoning", backend="hf", start_value=2, end_value=4
    )
    assert command[command.index("--cycles") + 1] == "4"
    assert command[command.index("--resume-from-cycle") + 1] == "2"
    assert command.count("--reward-audit-boundary") == 1
    assert command[command.index("--reward-audit-boundary") + 1] == "4"


def test_nonresumable_schedule_is_truthfully_final_only():
    assert resolve_boundary_values(["final"], total=6, resumable=False) == [6]
    try:
        resolve_boundary_values([2, "final"], total=6, resumable=False)
    except ValueError as exc:
        assert "final-boundary" in str(exc)
    else:
        raise AssertionError("non-resumable trainer accepted a mid-run boundary")


def test_hf_grpo_segment_pins_step_checkpoint_and_defers_resume_resolution():
    base = [
        "python",
        "-m",
        "halo_forge.cli",
        "grpo",
        "train",
        "--max-steps",
        "100",
        "--save-steps",
        "100",
        "--reward-audit-boundary",
        "final",
    ]
    command = command_for_segment(
        base, mode="grpo", backend="hf", start_value=25, end_value=50
    )
    assert command[command.index("--max-steps") + 1] == "50"
    assert command[command.index("--save-steps") + 1] == "50"
    assert "--resume" not in command


def test_mlx_raft_preserves_cycle_state_only_between_managed_segments(monkeypatch):
    from halo_forge.rlvr.mlx_raft_trainer import _preserve_managed_cycle_state

    monkeypatch.delenv("HALOFORGE_DIRECT_RUN_SEGMENT_ID", raising=False)
    monkeypatch.delenv("HALOFORGE_DIRECT_RUN_SEGMENT_FINAL", raising=False)
    assert _preserve_managed_cycle_state() is False
    monkeypatch.setenv("HALOFORGE_DIRECT_RUN_SEGMENT_ID", "segment-1")
    monkeypatch.setenv("HALOFORGE_DIRECT_RUN_SEGMENT_FINAL", "0")
    assert _preserve_managed_cycle_state() is True
    monkeypatch.setenv("HALOFORGE_DIRECT_RUN_SEGMENT_FINAL", "1")
    assert _preserve_managed_cycle_state() is False


def test_continue_enqueues_one_next_segment_with_audit_dependency(tmp_path):
    database = RunDatabase(":memory:")
    scheduler = WorkstationScheduler(database)
    database.upsert_run(
        RunRecord(
            run_id="segmented-run",
            modality="reasoning",
            model_name="model",
            status="running",
            output_dir=str(tmp_path / "run"),
            indexed_at="now",
        )
    )
    first = database.create_direct_run_segment(
        segment_id="segment-0",
        run_id="segmented-run",
        ordinal=0,
        unit="cycle",
        start_value=0,
        end_value=2,
    )
    second = database.create_direct_run_segment(
        segment_id="segment-1",
        run_id="segmented-run",
        ordinal=1,
        unit="cycle",
        start_value=2,
        end_value=4,
    )
    third = database.create_direct_run_segment(
        segment_id="segment-2",
        run_id="segmented-run",
        ordinal=2,
        unit="cycle",
        start_value=4,
        end_value=6,
    )
    canonical = [
        "python",
        "-m",
        "halo_forge.cli",
        "reasoning",
        "train",
        "--cycles",
        "6",
        "--resume-from-cycle",
        "0",
        "--reward-audit-boundary",
        "2",
        "--reward-audit-boundary",
        "4",
        "--reward-audit-boundary",
        "final",
    ]
    base = {
        "operation": "managed_training",
        "command": canonical,
        "canonical_command": canonical,
        "output_dir": str(tmp_path / "run"),
        "resolved_launch_config": {
            "mode": "reasoning",
            "run_id": "segmented-run",
            "output_dir": str(tmp_path / "run"),
            "_resolved_signal_backend": "hf",
        },
    }
    first_spec = build_segment_launch_spec(
        base, segment={**first, "is_final": False}
    )
    first_work = scheduler.enqueue(
        kind="training",
        launch_spec=first_spec,
        canonical_run_id="segmented-run",
        domain_kind="run",
        domain_id="segmented-run",
        work_item_id="training-0",
    )
    database.update_direct_run_segment(first["id"], work_item_id=first_work.id)
    audit_work = scheduler.enqueue(
        kind="reward_integrity_audit", work_item_id="audit-0", resource_class="cpu"
    )

    queued = enqueue_next_direct_segment(
        database,
        scheduler,
        current_segment_id=first["id"],
        dependency_work_item_id=audit_work.id,
    )
    assert queued is not None
    assert queued.launch_spec["operation"] == SEGMENT_OPERATION
    assert queued.launch_spec["direct_run_segment_id"] == second["id"]
    assert queued.launch_spec["final_segment"] is False
    assert queued.launch_spec["previous_segment_output_dir"].endswith("segment-0000")
    assert queued.launch_spec["segment_output_dir"].endswith("segment-0001")
    assert queued.launch_spec["command"][queued.launch_spec["command"].index("--cycles") + 1] == "4"
    dependencies = database._conn.execute(
        "SELECT depends_on_work_item_id FROM work_item_dependencies WHERE work_item_id=?",
        (queued.id,),
    ).fetchall()
    assert [row[0] for row in dependencies] == [audit_work.id]

    # Repeated auto/review callbacks are idempotent.
    repeated = enqueue_next_direct_segment(
        database,
        scheduler,
        current_segment_id=first["id"],
        dependency_work_item_id=audit_work.id,
    )
    assert repeated.id == queued.id
    assert database.get_direct_run_segment(third["id"])["work_item_id"] is None


def test_reward_cli_continue_enqueues_the_next_direct_segment_idempotently(
    monkeypatch, capsys
):
    from halo_forge import reward_cli
    from halo_forge.reward_integrity import direct_segments

    calls = []
    audit = SimpleNamespace(
        id="audit-paused",
        direct_run_segment_id="segment-paused",
        work_item_id="audit-work",
    )

    class FakeService:
        db = object()
        scheduler = object()

        @staticmethod
        def get_audit(audit_id):
            assert audit_id == audit.id
            return audit

        @staticmethod
        def review_audit(audit_id, *, action, reason, checkpoint):
            assert (audit_id, action, reason, checkpoint) == (
                audit.id,
                "continue",
                "reviewed the integrity evidence",
                None,
            )
            return {"id": "decision-continue", "action": "continue"}

        @staticmethod
        def sync_audit_replay(audit_id, *, decision):
            assert audit_id == audit.id
            assert decision["id"] == "decision-continue"
            return {"status": "updated"}

    def enqueue(database, scheduler, **values):
        calls.append((database, scheduler, values))
        return SimpleNamespace(id="next-segment-work")

    service = FakeService()
    monkeypatch.setattr(reward_cli, "_service", lambda _args: service)
    monkeypatch.setattr(direct_segments, "enqueue_next_direct_segment", enqueue)
    args = Namespace(
        database=None,
        json=True,
        reward_command="audit",
        reward_audit_action="review",
        audit_id=audit.id,
        action="continue",
        reason="reviewed the integrity evidence",
        checkpoint=None,
    )

    reward_cli.cmd_reward(args)
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["next_work_item_id"] == "next-segment-work"
    assert calls == [
        (
            service.db,
            service.scheduler,
            {
                "current_segment_id": "segment-paused",
                "dependency_work_item_id": "audit-work",
            },
        )
    ]


def test_managed_queue_launches_only_first_resumable_boundary(tmp_path):
    database = RunDatabase(str(tmp_path / "runs.db"))
    service = _public_service(tmp_path, database)
    payload = {
        "mode": "reasoning",
        "model": "local/model",
        "dataset": str(tmp_path / "data.jsonl"),
        "output_dir": str(tmp_path / "runs" / "run-segmented"),
        "cycles": 6,
        "seed": 42,
        "reward_system_revision_id": "reward-revision",
        "reward_audit_protocol_revision_id": "protocol-revision",
        "reward_integrity_profile_revision_id": "profile-revision",
        "reward_audit_boundaries": ["2", "4", "final"],
        "verifier_profile_revision_id": "optimizer-revision",
        "reward_integrity_binding": {
            "signal_capability": TRAINING_SIGNAL_CAPABILITIES.get(
                "reasoning:hf"
            ).to_dict()
        },
    }
    _, work = service._queue_managed_training(
        payload,
        canonical_run_id="run-segmented",
        dataset_version_metadata=None,
    )
    segments = database.list_direct_run_segments("run-segmented")
    assert [(row["start_value"], row["end_value"]) for row in segments] == [
        (0, 2),
        (2, 4),
        (4, 6),
    ]
    assert work.launch_spec["operation"] == SEGMENT_OPERATION
    assert work.launch_spec["direct_run_segment_id"] == segments[0]["id"]
    command = work.launch_spec["command"]
    assert command[command.index("--cycles") + 1] == "2"
    assert command[command.index("--resume-from-cycle") + 1] == "0"
    assert segments[0]["work_item_id"] == work.id
    assert segments[1]["status"] == "blocked"
    assert segments[1]["work_item_id"] is None


def test_reviewed_checkpoint_fork_starts_after_the_immutable_boundary(tmp_path):
    database = RunDatabase(str(tmp_path / "runs.db"))
    service = _public_service(tmp_path, database)
    snapshot = tmp_path / "parent-segments" / "segment-0000"
    snapshot.mkdir(parents=True)
    payload = {
        "mode": "reasoning",
        "model": "local/model",
        "dataset": str(tmp_path / "data.jsonl"),
        "output_dir": str(tmp_path / "runs" / "forked-run"),
        "cycles": 6,
        "seed": 42,
        "parent_run_id": "parent-run",
        "source_reward_integrity_audit_id": "audit-boundary-2",
        "source_reward_integrity_decision_id": "decision-fork",
        "fork_checkpoint_hash": "a" * 64,
        "fork_checkpoint_snapshot_path": str(snapshot),
        "fork_boundary_unit": "cycle",
        "fork_boundary_value": 2,
        "fork_resume_mode": "resume_boundary",
        "reward_system_revision_id": "reward-revision",
        "reward_audit_protocol_revision_id": "protocol-revision",
        "reward_integrity_profile_revision_id": "profile-revision",
        "reward_audit_boundaries": ["2", "4", "final"],
        "verifier_profile_revision_id": "optimizer-revision",
        "reward_integrity_binding": {
            "signal_capability": TRAINING_SIGNAL_CAPABILITIES.get(
                "reasoning:hf"
            ).to_dict()
        },
    }
    _, work = service._queue_managed_training(
        payload,
        canonical_run_id="forked-run",
        dataset_version_metadata=None,
    )
    segments = database.list_direct_run_segments("forked-run")
    assert [(row["start_value"], row["end_value"]) for row in segments] == [
        (2, 4),
        (4, 6),
    ]
    assert work.launch_spec["fork_checkpoint_snapshot_path"] == str(snapshot)
    assert work.launch_spec["fork_checkpoint_hash"] == "a" * 64
    command = work.launch_spec["command"]
    assert command[command.index("--resume-from-cycle") + 1] == "2"
    assert work.launch_spec["resolved_launch_config"][
        "source_reward_integrity_audit_id"
    ] == "audit-boundary-2"


def test_managed_hf_grpo_step_segment_has_resolved_budget(tmp_path):
    database = RunDatabase(str(tmp_path / "runs.db"))
    service = _public_service(tmp_path, database)
    payload = {
        "mode": "grpo",
        "model": "local/model",
        "dataset": str(tmp_path / "data.jsonl"),
        "output_dir": str(tmp_path / "runs" / "run-grpo"),
        "epochs": 1,
        "max_steps": 100,
        "seed": 42,
        "reward_system_revision_id": "reward-revision",
        "reward_audit_protocol_revision_id": "protocol-revision",
        "reward_integrity_profile_revision_id": "profile-revision",
        "reward_audit_boundaries": ["25", "50", "final"],
        "verifier_profile_revision_id": "optimizer-revision",
        "reward_integrity_binding": {
            "signal_capability": TRAINING_SIGNAL_CAPABILITIES.get("grpo:hf").to_dict()
        },
    }
    _, work = service._queue_managed_training(
        payload,
        canonical_run_id="run-grpo",
        dataset_version_metadata=None,
    )
    segments = database.list_direct_run_segments("run-grpo")
    assert [(row["unit"], row["end_value"]) for row in segments] == [
        ("step", 25),
        ("step", 50),
        ("step", 100),
    ]
    command = work.launch_spec["command"]
    assert command[command.index("--max-steps") + 1] == "25"
    assert command[command.index("--save-steps") + 1] == "25"


def test_audit_resource_class_distinguishes_hosted_and_local_judges(tmp_path):
    database = RunDatabase(":memory:")
    rewards = RewardIntegrityService(database, root=tmp_path / "reward")
    optimizer = _verifier_revision(
        database, "optimizer", family="deterministic"
    )
    for endpoint_type, expected in (("hosted", "cpu"), ("ollama", "accelerator")):
        sentinel = _verifier_revision(
            database,
            f"sentinel-{endpoint_type}",
            family="llm_judge",
            endpoint_type=endpoint_type,
        )
        system = rewards.create_system(name=f"system-{endpoint_type}")
        revision = rewards.create_system_revision(
            system.id,
            optimizer_verifier_revision_id=optimizer.id,
            modality="text",
            task_type="binary",
            auditors=[
                {
                    "role": "primary_sentinel",
                    "verifier_revision_id": sentinel.id,
                }
            ],
        )
        requirements = rewards.audit_resource_requirements(revision.id)
        assert requirements["resource_class"] == expected
        assert requirements["exclusive_heavy_operation"] is (
            expected == "accelerator"
        )


def test_local_diagnostic_auditor_requires_accelerator_lease(tmp_path):
    database = RunDatabase(":memory:")
    rewards = RewardIntegrityService(database, root=tmp_path / "reward")
    optimizer = _verifier_revision(database, "diag-optimizer", family="deterministic")
    primary = _verifier_revision(
        database,
        "hosted-primary",
        family="llm_judge",
        endpoint_type="hosted",
    )
    diagnostic = _verifier_revision(
        database,
        "local-diagnostic",
        family="reward_model",
        endpoint_type="local",
    )
    system = rewards.create_system(name="mixed-locality-system")
    revision = rewards.create_system_revision(
        system.id,
        optimizer_verifier_revision_id=optimizer.id,
        modality="text",
        task_type="binary",
        auditors=[
            {"role": "primary_sentinel", "verifier_revision_id": primary.id},
            {"role": "diagnostic", "verifier_revision_id": diagnostic.id},
        ],
    )
    requirements = rewards.audit_resource_requirements(revision.id)
    assert requirements["resource_class"] == "accelerator"
    assert requirements["family"] == "llm_judge"
    assert requirements["local_auditor_revision_ids"] == [diagnostic.id]


def test_worker_atomically_publishes_nonfinal_resume_snapshot(tmp_path):
    database = RunDatabase(":memory:")
    scheduler = WorkstationScheduler(database)
    canonical_output = tmp_path / "runs" / "snapshot-run"
    staging_output = tmp_path / "attempt"
    snapshot_output = tmp_path / "segments" / "segment-0000"
    database.upsert_run(
        RunRecord(
            run_id="snapshot-run",
            modality="reasoning",
            model_name="model",
            status="running",
            output_dir=str(canonical_output),
            indexed_at="now",
        )
    )
    segment = database.create_direct_run_segment(
        segment_id="snapshot-segment",
        run_id="snapshot-run",
        ordinal=0,
        unit="cycle",
        start_value=0,
        end_value=2,
    )
    work = scheduler.enqueue(
        kind="training",
        canonical_run_id="snapshot-run",
        domain_kind="run",
        domain_id="snapshot-run",
        launch_spec={
            "operation": SEGMENT_OPERATION,
            "resolved_launch_config": {"mode": "reasoning", "run_id": "snapshot-run"},
            "output_dir": str(canonical_output),
            "execution_output_dir": str(staging_output),
            "canonical_output_dir": str(canonical_output),
            "segment_output_dir": str(snapshot_output),
            "direct_run_segment_id": segment["id"],
            "final_segment": False,
        },
    )
    database.update_direct_run_segment(segment["id"], work_item_id=work.id)
    model = staging_output / "cycle_1" / "model"
    model.mkdir(parents=True)
    (model / "adapter_config.json").write_text("{}", encoding="utf-8")
    (model / "adapter_model.safetensors").write_bytes(b"checkpoint")
    (staging_output / "training_summary.json").write_text(
        '{"run_id":"snapshot-run","final_model_path":"'
        + str(model)
        + '"}',
        encoding="utf-8",
    )
    (staging_output / "_cycle_state.json").write_text(
        '{"next_cycle":2,"previous_adapter":"' + str(model) + '"}',
        encoding="utf-8",
    )

    result = {}
    WorkstationWorker(scheduler)._publish_managed_training(
        work, work.launch_spec, result
    )
    assert not staging_output.exists()
    assert snapshot_output.is_dir()
    state = __import__("json").loads(
        (snapshot_output / "_cycle_state.json").read_text(encoding="utf-8")
    )
    assert state["next_cycle"] == 2
    assert state["previous_adapter"].startswith(str(snapshot_output))
    assert database.get_direct_run_segment(segment["id"])["status"] == "audit_pending"
    assert database.get_run("snapshot-run").status == "audit_pending"
    assert result["awaiting_reward_audit"] is True
