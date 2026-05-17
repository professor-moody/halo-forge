"""Diagnostics endpoint tests.

Cover the three things users actually need to debug "I launched a run
but nothing showed up":

  1. Inventory of every launch_context.json on disk, classified by
     whether it produced training_summary.json.
  2. logs/ directory listing.
  3. Safe path-bounded log tailing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_app_data_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("HALO_FORGE_RUN_ROOT", str(tmp_path / "halo-runs-default"))
    monkeypatch.setenv("HALO_FORGE_LOG_DIR", str(tmp_path / "halo-logs-default"))


def _write_launch(dir_path: Path, *, with_summary: bool = False) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "launch_context.json").write_text(json.dumps({
        "args": {"model": "test/model"},
        "command": ["python", "-m", "halo_forge.cli", "sft", "train"],
    }))
    if with_summary:
        (dir_path / "training_summary.json").write_text(json.dumps({
            "run_id": dir_path.name,
            "modality": "sft",
            "model_name": "test/model",
            "status": "completed",
            "timestamp": "2026-05-07T12:00:00+00:00",
        }))
    (dir_path / "abc_training.log").write_text("line1\nline2\nline3\n")


# ---------- inventory_launches -------------------------------------------


def test_inventory_launches_classifies_orphan_vs_completed(tmp_path):
    from halo_forge.public_api import diagnostics

    _write_launch(tmp_path / "models" / "good", with_summary=True)
    _write_launch(tmp_path / "models" / "aborted", with_summary=False)

    items = diagnostics.inventory_launches(tmp_path)
    by_status = {Path(e["output_dir"]).name: e["status"] for e in items}
    assert by_status["good"] == "completed"
    assert by_status["aborted"] == "orphan"


def test_inventory_launches_newest_first(tmp_path):
    """Launches sort newest first by mtime so the user's just-failed run
    surfaces at the top of the diagnostics list."""
    import time

    from halo_forge.public_api import diagnostics

    _write_launch(tmp_path / "models" / "old", with_summary=False)
    time.sleep(0.05)
    _write_launch(tmp_path / "models" / "new", with_summary=False)

    items = diagnostics.inventory_launches(tmp_path)
    names = [Path(e["output_dir"]).name for e in items]
    assert names.index("new") < names.index("old")


def test_inventory_launches_collects_log_files(tmp_path):
    from halo_forge.public_api import diagnostics

    run_dir = tmp_path / "models" / "logged"
    _write_launch(run_dir, with_summary=False)
    items = diagnostics.inventory_launches(tmp_path)
    assert items[0]["log_files"]
    assert items[0]["log_files"][0].endswith("abc_training.log")


def test_inventory_launches_returns_empty_when_no_runs(tmp_path):
    from halo_forge.public_api import diagnostics

    assert diagnostics.inventory_launches(tmp_path) == []


def test_inventory_launches_includes_dashboard_run_root(tmp_path, monkeypatch):
    from halo_forge.public_api import diagnostics

    run_root = tmp_path / "halo-runs"
    monkeypatch.setenv("HALO_FORGE_RUN_ROOT", str(run_root))
    _write_launch(run_root / "start-code", with_summary=False)

    items = diagnostics.inventory_launches(tmp_path / "readonly-app-cwd")

    assert len(items) == 1
    assert Path(items[0]["output_dir"]).name == "start-code"
    assert items[0]["status"] == "orphan"


# ---------- inventory_logs -----------------------------------------------


def test_inventory_logs_lists_size_and_mtime(tmp_path):
    from halo_forge.public_api import diagnostics

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "training.log").write_text("hello\nworld\n")

    items = diagnostics.inventory_logs(tmp_path)
    assert len(items) == 1
    entry = items[0]
    assert entry["name"] == "training.log"
    assert entry["size_bytes"] == len("hello\nworld\n")
    assert entry["mtime"] > 0


def test_inventory_logs_returns_empty_when_no_logs_dir(tmp_path):
    from halo_forge.public_api import diagnostics

    assert diagnostics.inventory_logs(tmp_path) == []


def test_inventory_logs_includes_dashboard_log_root(tmp_path, monkeypatch):
    from halo_forge.public_api import diagnostics

    log_root = tmp_path / "halo-logs"
    log_root.mkdir()
    monkeypatch.setenv("HALO_FORGE_LOG_DIR", str(log_root))
    (log_root / "sft_train.log").write_text("hello\n", encoding="utf-8")

    items = diagnostics.inventory_logs(tmp_path / "readonly-app-cwd")

    assert len(items) == 1
    assert items[0]["name"] == "sft_train.log"


# ---------- tail_log -----------------------------------------------------


def test_tail_log_returns_last_n_lines(tmp_path):
    from halo_forge.public_api import diagnostics

    logs = tmp_path / "logs"
    logs.mkdir()
    log = logs / "x.log"
    log.write_text("\n".join(f"line {i}" for i in range(50)))

    out = diagnostics.tail_log(
        base_path=tmp_path,
        requested_path=str(log),
        tail=5,
    )
    assert out["available"] is True
    assert out["lines"] == [f"line {i}" for i in range(45, 50)]


def test_tail_log_refuses_paths_outside_repo(tmp_path):
    from halo_forge.public_api import diagnostics

    # Try escaping via .. — must refuse.
    out = diagnostics.tail_log(
        base_path=tmp_path,
        requested_path="../../etc/passwd",
        tail=10,
    )
    assert out["available"] is False
    assert "outside" in out["reason"].lower() or "not found" in out["reason"].lower()


def test_tail_log_refuses_non_log_roots(tmp_path):
    """A file inside the repo but not under logs/models/outputs/results
    is still off-limits — the tail endpoint isn't a generic file reader."""
    from halo_forge.public_api import diagnostics

    naughty = tmp_path / "secrets.txt"
    naughty.write_text("don't read me")

    out = diagnostics.tail_log(
        base_path=tmp_path,
        requested_path=str(naughty),
        tail=10,
    )
    assert out["available"] is False
    assert "permitted roots" in out["reason"]


def test_tail_log_allows_dashboard_run_root_logs(tmp_path, monkeypatch):
    from halo_forge.public_api import diagnostics

    run_root = tmp_path / "halo-runs"
    monkeypatch.setenv("HALO_FORGE_RUN_ROOT", str(run_root))
    run_dir = run_root / "start-code"
    _write_launch(run_dir, with_summary=False)
    log = run_dir / "abc_training.log"

    out = diagnostics.tail_log(
        base_path=tmp_path / "readonly-app-cwd",
        requested_path=str(log),
        tail=2,
    )

    assert out["available"] is True
    assert out["lines"] == ["line2", "line3"]


def test_tail_log_handles_huge_file_via_max_bytes(tmp_path):
    from halo_forge.public_api import diagnostics

    logs = tmp_path / "logs"
    logs.mkdir()
    log = logs / "huge.log"
    # 5 MB of "x\n" lines.
    log.write_bytes(b"x\n" * (5 * 1024 * 1024 // 2))

    out = diagnostics.tail_log(
        base_path=tmp_path,
        requested_path=str(log),
        tail=10,
        max_bytes=1024,  # force the head-truncation branch
    )
    assert out["available"] is True
    assert out["truncated_head"] is True
    assert len(out["lines"]) <= 10


# ---------- summary ------------------------------------------------------


def test_summary_returns_counts(tmp_path):
    from halo_forge.public_api import diagnostics

    _write_launch(tmp_path / "models" / "a", with_summary=True)
    _write_launch(tmp_path / "models" / "b", with_summary=False)
    _write_launch(tmp_path / "models" / "c", with_summary=False)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "x.log").write_text("hi")

    out = diagnostics.summary(tmp_path)
    assert out["launches"]["total"] == 3
    assert out["launches"]["orphan"] == 2
    assert out["launches"]["completed"] == 1
    assert out["launches"]["most_recent_orphan"] is not None
    assert out["logs"]["total"] == 1


# ---------- end-to-end via TestClient -----------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HALOFORGE_RUN_DB_PATH", str(tmp_path / "runs.db"))
    from halo_forge.run_db import db as db_mod
    db_mod._GLOBAL_DB.clear()
    from halo_forge.auth.dependency import reset_store_for_tests
    reset_store_for_tests(None)

    # Seed an orphan + a completed launch under tmp_path.
    _write_launch(tmp_path / "models" / "good", with_summary=True)
    _write_launch(tmp_path / "models" / "aborted", with_summary=False)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "training.log").write_text("ok\n")

    monkeypatch.chdir(tmp_path)

    from fastapi.testclient import TestClient
    from halo_forge.public_api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c, tmp_path
    db_mod._GLOBAL_DB.clear()


def test_diagnostics_summary_endpoint(client):
    c, _ = client
    r = c.get("/api/public/diagnostics/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["launches"]["orphan"] == 1
    assert body["launches"]["completed"] == 1
    assert body["logs"]["total"] == 1


def test_diagnostics_launches_endpoint(client):
    c, _ = client
    r = c.get("/api/public/diagnostics/launches")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 2
    assert {e["status"] for e in items} == {"orphan", "completed"}


def test_diagnostics_log_tail_endpoint(client):
    c, base = client
    log_path = base / "logs" / "training.log"
    r = c.get(
        "/api/public/diagnostics/log",
        params={"path": str(log_path), "tail": 5},
    )
    assert r.status_code == 200
    assert r.json()["available"] is True
    assert r.json()["lines"] == ["ok"]


def test_diagnostics_log_tail_endpoint_refuses_traversal(client):
    c, base = client
    r = c.get(
        "/api/public/diagnostics/log",
        params={"path": "../../etc/passwd", "tail": 5},
    )
    assert r.status_code == 200
    assert r.json()["available"] is False
