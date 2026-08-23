"""CLI contracts for the v4 Artifact Studio and storage surface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from halo_forge import cli
from halo_forge.run_db import LabV4Catalog, get_database


def _run_cli(monkeypatch, capsys, *args: str):
    monkeypatch.setattr(sys, "argv", ["halo-forge", *args])
    cli.main()
    return json.loads(capsys.readouterr().out)


def _runtime_args(tmp_path: Path) -> tuple[str, ...]:
    return (
        "--database",
        str(tmp_path / "runs.db"),
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--json",
    )


def _source(tmp_path: Path, name: str, value: str = "weights") -> Path:
    path = tmp_path / name
    path.mkdir()
    (path / "weights.bin").write_text(value, encoding="utf-8")
    return path


def _import(monkeypatch, capsys, tmp_path: Path, name: str, value: str = "weights"):
    return _run_cli(
        monkeypatch,
        capsys,
        "artifact",
        "import",
        str(_source(tmp_path, name, value)),
        "--kind",
        "final",
        "--format",
        "raw",
        *_runtime_args(tmp_path),
    )


def test_artifact_cli_import_browse_verify_and_annotations(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    imported = _import(monkeypatch, capsys, tmp_path, "candidate")
    occurrence_id = imported["occurrence"]["id"]
    content_hash = imported["blob"]["content_hash"]

    listed = _run_cli(
        monkeypatch,
        capsys,
        "artifact",
        "list",
        *_runtime_args(tmp_path),
    )
    assert listed["items"][0]["occurrence"]["id"] == occurrence_id

    shown = _run_cli(
        monkeypatch,
        capsys,
        "artifact",
        "show",
        content_hash,
        *_runtime_args(tmp_path),
    )
    assert shown["occurrence"]["id"] == occurrence_id

    verified = _run_cli(
        monkeypatch,
        capsys,
        "artifact",
        "verify",
        occurrence_id,
        *_runtime_args(tmp_path),
    )
    assert verified["passed"] is True

    pinned = _run_cli(
        monkeypatch,
        capsys,
        "artifact",
        "pin",
        occurrence_id,
        *_runtime_args(tmp_path),
    )
    assert pinned["occurrence"]["pinned"] is True
    tagged = _run_cli(
        monkeypatch,
        capsys,
        "artifact",
        "tag",
        occurrence_id,
        "research",
        "candidate-source",
        *_runtime_args(tmp_path),
    )
    assert tagged["occurrence"]["tags"] == ["candidate-source", "research"]
    unpinned = _run_cli(
        monkeypatch,
        capsys,
        "artifact",
        "unpin",
        occurrence_id,
        *_runtime_args(tmp_path),
    )
    assert unpinned["occurrence"]["pinned"] is False

    lineage = _run_cli(
        monkeypatch,
        capsys,
        "artifact",
        "lineage",
        occurrence_id,
        *_runtime_args(tmp_path),
    )
    assert lineage["catalog"]["occurrence_id"] == occurrence_id

    storage = _run_cli(
        monkeypatch,
        capsys,
        "storage",
        "status",
        *_runtime_args(tmp_path),
    )
    assert storage["blob_count"] == 1
    assert storage["free_bytes"] > 0


def test_artifact_convert_and_qualify_return_domain_and_work_ids(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    imported = _import(monkeypatch, capsys, tmp_path, "source")
    occurrence_id = imported["occurrence"]["id"]
    converted = _run_cli(
        monkeypatch,
        capsys,
        "artifact",
        "convert",
        occurrence_id,
        "--format",
        "hf",
        "--quant",
        "fp16",
        *_runtime_args(tmp_path),
    )
    assert converted["domain_kind"] == "artifact_operation"
    assert converted["domain_id"]
    assert converted["work_item_id"]
    assert converted["quantization_method"] == "dtype_conversion"
    assert converted["qat"] is False

    database = get_database(str(tmp_path / "runs.db"))
    development = database.create_benchmark_suite(
        name="CLI development", purpose="development"
    )
    operational = database.create_benchmark_suite(
        name="CLI operational", purpose="operational"
    )
    development_revision = database.create_benchmark_suite_revision(
        suite_id=development.id,
        content_hash="cli-dev",
        items=[{"id": "dev", "input": "hello"}],
        primary_metric="quality",
        direction="maximize",
    )
    operational_revision = database.create_benchmark_suite_revision(
        suite_id=operational.id,
        content_hash="cli-ops",
        items=[{"id": "ops", "input": "hello"}],
        primary_metric="latency_ms",
        direction="minimize",
    )
    profile = LabV4Catalog(database).create_qualification_profile_revision(
        name="CLI profile",
        content_hash="cli-profile",
        quality_suite_revision_id=development_revision.id,
        operational_suite_revision_id=operational_revision.id,
        thresholds=[
            {
                "stage": "development",
                "metric": "quality",
                "direction": "maximize",
                "pass_threshold": 0.8,
            },
            {
                "stage": "operational",
                "metric": "latency_ms",
                "direction": "minimize",
                "pass_threshold": 1000,
            },
        ],
        generation_settings={"seed": 42},
    )
    queued = _run_cli(
        monkeypatch,
        capsys,
        "artifact",
        "qualify",
        occurrence_id,
        "--profile",
        profile.id,
        *_runtime_args(tmp_path),
    )
    assert queued["domain_kind"] == "artifact_qualification"
    assert queued["domain_id"]
    assert queued["work_item_id"]
    assert "verified" in queued["execution_note"]


def test_artifact_compare_is_profile_and_direction_aware(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    parent = _import(monkeypatch, capsys, tmp_path, "parent", "parent")
    candidate = _import(monkeypatch, capsys, tmp_path, "child", "candidate")
    database = get_database(str(tmp_path / "runs.db"))
    development = database.create_benchmark_suite(
        name="Comparison development", purpose="development"
    )
    operational = database.create_benchmark_suite(
        name="Comparison operational", purpose="operational"
    )
    development_revision = database.create_benchmark_suite_revision(
        suite_id=development.id,
        content_hash="comparison-dev",
        items=[{"id": "dev", "input": "hello"}],
        primary_metric="quality",
        direction="maximize",
    )
    operational_revision = database.create_benchmark_suite_revision(
        suite_id=operational.id,
        content_hash="comparison-ops",
        items=[{"id": "ops", "input": "hello"}],
        primary_metric="latency_ms",
        direction="minimize",
    )
    catalog = LabV4Catalog(database)
    profile = catalog.create_qualification_profile_revision(
        name="Comparison profile",
        content_hash="comparison-profile",
        quality_suite_revision_id=development_revision.id,
        operational_suite_revision_id=operational_revision.id,
        thresholds=[
            {"stage": "development", "metric": "quality", "direction": "maximize"},
            {"stage": "operational", "metric": "latency_ms", "direction": "minimize"},
        ],
        generation_settings={"seed": 42},
    )
    values = (
        (parent["occurrence"]["id"], 0.8, 120.0),
        (candidate["occurrence"]["id"], 0.9, 100.0),
    )
    for occurrence_id, quality, latency in values:
        qualification = catalog.create_qualification(
            profile_revision_id=profile.id,
            occurrence_id=occurrence_id,
        )
        catalog.update_qualification(
            qualification.id,
            status="completed",
            decision="pass",
            metrics={
                "development": {"quality": quality},
                "operational": {"latency_ms": latency},
            },
        )

    comparison = _run_cli(
        monkeypatch,
        capsys,
        "artifact",
        "compare",
        parent["occurrence"]["id"],
        candidate["occurrence"]["id"],
        "--profile",
        profile.id,
        *_runtime_args(tmp_path),
    )
    deltas = {item["metric"]: item for item in comparison["deltas"]}
    assert deltas["development.quality"]["raw_delta"] == pytest.approx(0.1)
    assert deltas["development.quality"]["favorable_delta"] == pytest.approx(0.1)
    assert deltas["operational.latency_ms"]["raw_delta"] == pytest.approx(-20)
    assert deltas["operational.latency_ms"]["favorable_delta"] == pytest.approx(20)


def test_artifact_cleanup_is_previewed_and_requires_a_review_note(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _import(monkeypatch, capsys, tmp_path, "cleanup-source")
    preview = _run_cli(
        monkeypatch,
        capsys,
        "storage",
        "cleanup",
        *_runtime_args(tmp_path),
    )
    assert preview["id"]
    assert preview["status"] == "preview"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "halo-forge",
            "storage",
            "cleanup",
            "--apply",
            preview["id"],
            *_runtime_args(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 1
    error = json.loads(capsys.readouterr().out)
    assert "--review-note is required" in error["error"]

    queued = _run_cli(
        monkeypatch,
        capsys,
        "storage",
        "cleanup",
        "--apply",
        preview["id"],
        "--review-note",
        "Reviewed reclaim candidates",
        *_runtime_args(tmp_path),
    )
    assert queued["domain_kind"] == "artifact_cleanup"
    assert queued["domain_id"] == preview["id"]
    assert queued["work_item_id"]


def test_artifact_merge_promotion_serving_and_export_are_explicit(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    adapters = []
    for name, value in (("adapter-a", "a"), ("adapter-b", "b")):
        imported = _run_cli(
            monkeypatch,
            capsys,
            "artifact",
            "import",
            str(_source(tmp_path, name, value)),
            "--kind",
            "adapter",
            "--format",
            "raw",
            *_runtime_args(tmp_path),
        )
        adapters.append(imported["occurrence"]["id"])

    merged = _run_cli(
        monkeypatch,
        capsys,
        "artifact",
        "merge",
        *adapters,
        "--base-model",
        "local/base@revision",
        "--method",
        "ties",
        "--weights",
        "0.7",
        "0.3",
        *_runtime_args(tmp_path),
    )
    assert merged["domain_kind"] == "artifact_operation"
    assert merged["domain_id"] and merged["work_item_id"]

    promoted = _run_cli(
        monkeypatch,
        capsys,
        "artifact",
        "promote",
        adapters[0],
        "--alias",
        "candidate",
        "--override-note",
        "Reviewed local smoke candidate",
        *_runtime_args(tmp_path),
    )
    assert promoted["alias"] == "candidate"
    assert promoted["overridden"] is True

    serving = _run_cli(
        monkeypatch,
        capsys,
        "artifact",
        "serve",
        adapters[0],
        "--backend",
        "local",
        "--endpoint",
        '{"host":"127.0.0.1","port":8080}',
        *_runtime_args(tmp_path),
    )
    assert serving["domain_kind"] == "artifact_serving"
    assert serving["status"] == "queued"
    assert serving["work_item_id"]
    assert serving["server_started"] is False
    assert "supervised worker" in serving["next_action"]

    destination = tmp_path / "portable-bundle"
    exported = _run_cli(
        monkeypatch,
        capsys,
        "artifact",
        "export",
        adapters[0],
        str(destination),
        "--license-metadata",
        '{"license":"research-only"}',
        *_runtime_args(tmp_path),
    )
    assert exported["domain_kind"] == "artifact_operation"
    assert exported["domain_id"]
    assert exported["work_item_id"]
    assert not destination.exists()


def test_unsupported_conversion_is_not_queued(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    imported = _import(monkeypatch, capsys, tmp_path, "unsupported-source")
    occurrence_id = imported["occurrence"]["id"]
    database = get_database(str(tmp_path / "runs.db"))
    before = len(database.list_work_items())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "halo-forge",
            "artifact",
            "convert",
            occurrence_id,
            "--format",
            "onnx",
            "--quant",
            "fp16",
            *_runtime_args(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 1
    error = json.loads(capsys.readouterr().out)
    assert "No verified" in error["error"]
    assert len(database.list_work_items()) == before
