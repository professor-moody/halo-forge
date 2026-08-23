from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from halo_forge.data_lab import DatasetLab
from halo_forge.product_lab import ProductLabService
from halo_forge.public_api.service import PublicApiService
from halo_forge.replay import MANIFEST_VERSION, capture_manifest
from halo_forge.run_db import RunDatabase
from halo_forge.run_db.schema import SCHEMA_VERSION
from halo_forge.workstation_jobs import WorkstationScheduler, WorkstationWorker


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_schema_v20_readiness_and_safe_setup_remediation(tmp_path: Path) -> None:
    database = RunDatabase(str(tmp_path / "catalog.sqlite"))
    assert SCHEMA_VERSION == 23
    tables = {
        row["name"]
        for row in database._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {
        "workstation_readiness_assessments",
        "dataset_repair_sessions",
        "dataset_repair_plan_revisions",
        "dataset_repair_revisions",
        "support_bundles",
        "release_qualifications",
    } <= tables

    root = tmp_path / "managed"
    service = ProductLabService(database, root=root)
    before = service.assess_readiness()
    assert any(item.id == "create_managed_directories" for item in before.remediations)
    after = service.apply_setup_remediation("create_managed_directories")
    assert (root / "datasets").is_dir()
    assert not any(item.id == "create_managed_directories" for item in after.remediations)
    assert {"browser", "cli"} <= set(after.capability.execution_surfaces)
    assert after.capability.desktop_status in {"preview", "supported", "unavailable"}


def test_schema_v19_to_v20_upgrade_is_additive(tmp_path: Path) -> None:
    path = tmp_path / "legacy-v19.sqlite"
    legacy = RunDatabase(str(path))
    dataset = legacy.create_dataset(
        name="Preserved source", modality="text", canonical_schema="sft"
    )
    v20_tables = (
        "release_qualifications",
        "support_bundles",
        "dataset_repair_revisions",
        "dataset_repair_previews",
        "dataset_repair_actions",
        "dataset_repair_issues",
        "dataset_repair_plan_revisions",
        "dataset_repair_sessions",
        "workstation_readiness_assessments",
    )
    with legacy._lock:
        for table in v20_tables:
            legacy._conn.execute(f"DROP TABLE IF EXISTS {table}")
        legacy._conn.execute(
            "UPDATE schema_meta SET value='19' WHERE key='schema_version'"
        )
        legacy._conn.commit()
    legacy.close()

    migrated = RunDatabase(str(path))
    assert migrated.get_dataset(dataset.id).name == "Preserved source"
    assert migrated._conn.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()[0] == "23"
    tables = {
        row["name"]
        for row in migrated._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert set(v20_tables) <= tables


def test_repair_overlay_is_exact_non_destructive_and_occurrence_aware(
    tmp_path: Path,
) -> None:
    rows = [
        {"prompt": "  hello  ", "response": "  world  "},
        {"prompt": "", "response": "bad"},
        {"prompt": "  hello  ", "response": "  world  "},
    ]
    source = _jsonl(tmp_path / "train.jsonl", rows)
    original = source.read_bytes()
    database = RunDatabase(str(tmp_path / "catalog.sqlite"))
    service = ProductLabService(database, root=tmp_path / "managed")
    session = service.create_repair_session(
        {
            "source_uri": str(source),
            "scenario_revision_id": "instruction-sft@1",
            "scan": True,
        },
        enqueue=False,
    )
    issues = service.list_repair_issues(session.id, limit=100)["items"]
    empty = next(item for item in issues if item["code"] == "empty_required_value")
    duplicate = next(item for item in issues if item["code"] == "duplicate_record")
    plan = service.create_repair_plan(
        session.id,
        {
            "actions": [
                {"action_kind": "trim", "reason": "Reviewed whitespace normalization"},
                {
                    "action_kind": "quarantine",
                    "record_id": empty["record_id"],
                    "source_index": empty["source_index"],
                    "reason": "Required prompt remains empty",
                },
                {
                    "action_kind": "exclude",
                    "record_id": duplicate["record_id"],
                    "source_index": duplicate["source_index"],
                    "reason": "Reviewed duplicate occurrence",
                },
            ]
        },
    )
    preview = service.prepare_repair_preview(session.id, plan.id, enqueue=False)
    assert preview.exact is True
    assert preview.counts["replace"] == 1
    assert preview.counts["quarantine"] == 1
    assert preview.counts["exclude"] == 1
    revision = service.publish_repair_revision(preview.id)
    assert source.read_bytes() == original
    assert revision["recipe_step"] == {
        "kind": "repair_overlay",
        "revision_id": revision["id"],
    }

    lab = DatasetLab(tmp_path / "datasets", database=database)
    registered = lab.add_source(
        {
            "kind": "local",
            "path": str(source),
            "canonical_kind": "sft",
            "modality": "text",
        }
    )
    version = lab.build(
        registered.id,
        {
            "schema": "sft",
            "steps": [
                revision["recipe_step"],
                {"kind": "validate", "on_error": "quarantine"},
            ],
        },
    )
    built_rows = [
        json.loads(line)
        for line in (Path(version.path) / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert built_rows == [{"prompt": "hello", "response": "world"}]
    lineage = [
        json.loads(line)
        for line in (Path(version.path) / "lineage.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert lineage[0]["operations"][0]["operation"] == "repair_overlay"


def test_source_drift_requires_rebase_and_reports_conflicts(tmp_path: Path) -> None:
    source = _jsonl(tmp_path / "chat.jsonl", [{"messages": [{"from": "human", "value": "Hi"}]}])
    service = ProductLabService(
        RunDatabase(str(tmp_path / "catalog.sqlite")), root=tmp_path / "managed"
    )
    session = service.create_repair_session(
        {"source_uri": str(source), "scenario_revision_id": "chat-sft@1"},
        enqueue=False,
    )
    issue = service.list_repair_issues(session.id)["items"][0]
    service.create_repair_plan(
        session.id,
        {
            "actions": [
                {
                    "action_kind": "edit",
                    "record_id": issue["record_id"],
                    "source_index": issue["source_index"],
                    "field_path": "messages.0.role",
                    "value": "user",
                    "reason": "Reviewed role correction",
                }
            ]
        },
    )
    _jsonl(source, [{"messages": [{"role": "user", "content": "Changed"}]}])
    stale = service.scan_repair_session(session.id)
    assert stale.status == "stale"
    rebased = service.rebase_repair_session(session.id)
    assert rebased.issue_summary["conflicts"]
    assert rebased.issue_summary["rebase_from"] == session.id


def test_repair_scan_reports_delimiter_and_field_type_problems_exactly(
    tmp_path: Path,
) -> None:
    source = tmp_path / "wrong.csv"
    source.write_text("question;answer\n42;usable\n", encoding="utf-8")
    service = ProductLabService(
        RunDatabase(str(tmp_path / "catalog.sqlite")), root=tmp_path / "managed"
    )
    session = service.create_repair_session(
        {"source_uri": str(source), "scenario_revision_id": "instruction-sft@1"},
        enqueue=False,
    )
    issues = service.list_repair_issues(session.id, limit=100)["items"]
    assert any(item["code"] == "possible_delimiter_mismatch" for item in issues)
    assert session.issue_summary["exact"] is True
    assert session.issue_summary["records_with_issues"] == 1
    assert session.issue_summary["clean_records_exact"] == 0

    typed = _jsonl(tmp_path / "typed.jsonl", [{"prompt": 42, "response": "answer"}])
    typed_session = service.create_repair_session(
        {"source_uri": str(typed), "scenario_revision_id": "instruction-sft@1"},
        enqueue=False,
    )
    typed_issues = service.list_repair_issues(typed_session.id, limit=100)["items"]
    assert any(
        item["code"] == "invalid_field_type" and item["field_path"] == "prompt"
        for item in typed_issues
    )


def test_support_bundle_redacts_content_paths_and_secrets_reproducibly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "managed"
    logs = root / "logs"
    logs.mkdir(parents=True)
    secret = "hf_abcdefghijklmnopqrstuvwxyz"
    (logs / "runtime.log").write_text(
        f"token={secret}\nprompt: private dataset sentence\nfailed at /Volumes/private/data.jsonl\n",
        encoding="utf-8",
    )
    service = ProductLabService(
        RunDatabase(str(tmp_path / "catalog.sqlite")), root=root
    )
    first = service.create_support_bundle(["versions", "logs"], enqueue=False)
    second = service.create_support_bundle(["versions", "logs"], enqueue=False)
    assert first.content_hash == second.content_hash
    assert Path(first.storage_path).read_bytes() == Path(second.storage_path).read_bytes()
    assert service.verify_support_bundle(first.id)["valid"] is True
    with zipfile.ZipFile(first.storage_path) as archive:
        rendered = "\n".join(
            archive.read(name).decode("utf-8") for name in archive.namelist()
        )
    assert secret not in rendered
    assert "private dataset sentence" not in rendered
    assert "/Volumes/private" not in rendered
    assert "automatic_upload" in rendered and "false" in rendered


def test_durable_worker_and_release_qualification_truthfulness(tmp_path: Path) -> None:
    database = RunDatabase(str(tmp_path / "catalog.sqlite"))
    scheduler = WorkstationScheduler(database, worker_id="v17-test")
    service = ProductLabService(database, root=tmp_path / "managed", scheduler=scheduler)
    source = _jsonl(tmp_path / "rows.jsonl", [{"prompt": "Q", "response": "A"}])
    session = service.create_repair_session(
        {"source_uri": str(source), "scenario_revision_id": "instruction-sft@1"}
    )
    terminal = WorkstationWorker(scheduler).run_once(work_item_id=session.work_item_id)
    assert terminal is not None and terminal.status == "completed"
    assert service.get_repair_session(session.id).status == "ready"

    package = tmp_path / "Halo-Forge-preview.bin"
    package.write_bytes(b"candidate")
    with pytest.raises(Exception, match="Unsigned desktop packages"):
        service.qualify_release(
            {
                "package_path": str(package),
                "signature_state": "unsigned",
                "distribution_status": "supported",
                "smoke_status": "passed",
            }
        )
    qualification = service.request_release_qualification(
        {
            "package_path": str(package),
            "package_type": "preview",
            "signature_state": "unsigned",
            "distribution_status": "preview",
            "smoke_status": "passed",
        }
    )
    terminal = WorkstationWorker(scheduler).run_once(
        work_item_id=qualification.work_item_id
    )
    assert terminal is not None and terminal.status == "completed"
    completed = service.get_release_qualification(qualification.id)
    assert completed.status == "completed"
    assert completed.evidence["package_sha256"]
    assert completed.signature_state == "unsigned"

    cancelled = service.request_release_qualification(
        {
            "package_path": str(package),
            "package_type": "preview",
            "signature_state": "unsigned",
            "distribution_status": "preview",
            "smoke_status": "passed",
            "evidence": {"attempt": "cancel-before-start"},
        }
    )
    cancelled = service.cancel_release_qualification(cancelled.id)
    assert cancelled.status == "cancelled"
    assert database.get_work_item(cancelled.work_item_id).status == "cancelled"


def test_replay_v11_captures_product_completion_identity() -> None:
    manifest = capture_manifest(
        run_id="v17-proof",
        modality="sft",
        model_name="fixture/model",
        seed=42,
        config={},
        product_completion_binding={
            "dataset_repair_revision_id": "repair-1",
            "repaired_record_set_hash": "abc",
            "workstation_readiness_id": "ready-1",
            "distribution_capability": {"platform": "linux"},
        },
    )
    assert MANIFEST_VERSION == manifest.manifest_version == 14
    assert manifest.product_completion["dataset_repair_revision_id"] == "repair-1"


def test_v17_public_api_routes_share_the_durable_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from halo_forge.auth.dependency import reset_store_for_tests
    from halo_forge.public_api import app as app_module

    database = RunDatabase(str(tmp_path / "catalog.sqlite"))
    scheduler = WorkstationScheduler(database, worker_id="v17-api")
    product = ProductLabService(database, root=tmp_path / "managed", scheduler=scheduler)
    public = PublicApiService(
        database=database,
        base_path=tmp_path,
        workstation_scheduler=scheduler,
        product_lab=product,
        product_lab_storage_root=tmp_path / "managed",
        dataset_storage_root=tmp_path / "managed" / "datasets",
    )
    monkeypatch.setattr(app_module, "PublicApiService", lambda: public)
    monkeypatch.setenv("HALOFORGE_DISABLE_AUTO_WORKER", "1")
    reset_store_for_tests(None)
    source = _jsonl(tmp_path / "api.jsonl", [{"prompt": "", "response": "A"}])

    with TestClient(app_module.create_app(serve_frontend=False)) as client:
        readiness = client.get("/api/public/setup/readiness")
        assert readiness.status_code == 200
        assert readiness.json()["capability"]["execution_surfaces"]

        created = client.post(
            "/api/public/dataset-repairs",
            json={
                "source_uri": str(source),
                "scenario_revision_id": "instruction-sft@1",
            },
        )
        assert created.status_code == 202, created.text
        session = created.json()
        assert session["work_item_id"]
        terminal = WorkstationWorker(scheduler).run_once(
            work_item_id=session["work_item_id"]
        )
        assert terminal is not None and terminal.status == "completed"
        issues = client.get(
            f"/api/public/dataset-repairs/{session['id']}/issues"
        ).json()
        assert issues["total"] == 1

        support_preview = client.post(
            "/api/public/support-bundles/preview",
            json={"categories": ["versions"]},
        )
        assert support_preview.status_code == 200
        assert "dataset records" in " ".join(
            support_preview.json()["excluded_by_default"]
        )
        support = client.post(
            "/api/public/support-bundles", json={"categories": ["versions"]}
        )
        assert support.status_code == 202
        assert support.json()["work_item_id"]
