"""Cohort eval endpoint tests (Track F-K backend).

Seeds a temporary results dir with synthetic `lm_eval_summary.json`
files, points the public API at it via env override, and exercises
the /eval/cohort endpoint end-to-end through the FastAPI TestClient.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def cohort_client(monkeypatch, tmp_path):
    """Create three fake "runs" each with a training_summary + an
    optional lm_eval_summary, and stand up the public-API FastAPI app
    pointed at this tree."""
    # Create three run directories with training_summary.json so
    # `_resolve_run_source` can find them through ResultsService.
    runs = {}
    for run_id, has_eval, scores in [
        ("run-a", True, {"mmlu": 0.65, "gsm8k": 0.42}),
        ("run-b", True, {"mmlu": 0.70, "gsm8k": 0.40}),
        ("run-c", False, None),  # no eval written
    ]:
        rd = tmp_path / run_id
        rd.mkdir()
        # training_summary.json — minimum fields for ResultsService to
        # treat this as a discoverable run.
        (rd / "training_summary.json").write_text(json.dumps({
            "id": run_id,
            "run_id": run_id,
            "modality": "sft",
            "model_name": f"test/{run_id}",
            "status": "completed",
            "timestamp": "2026-05-07T00:00:00+00:00",
            "cycles_executed": 1,
        }))
        if has_eval:
            (rd / "lm_eval_summary.json").write_text(json.dumps({
                "model_name": f"test/{run_id}",
                "tasks": list(scores.keys()),
                "task_results": [
                    {"task": t, "primary_metric": "acc", "value": v,
                     "n_samples": 100, "error": None,
                     "all_metrics": {"acc": v}}
                    for t, v in scores.items()
                ],
                "n_tasks_completed": len(scores),
                "n_tasks_failed": 0,
                "duration_seconds": 60.0,
                "backend": "hf",
            }))
        runs[run_id] = rd

    # Point ResultsService at our tmp tree.
    from ui.services.results_service import ResultsService

    monkeypatch.setattr(
        ResultsService, "TRAINING_DIRS", [tmp_path], raising=True,
    )

    # Reset run_db cache so the lazy sync picks our tmp paths instead of
    # whatever a previous test wired up.
    from halo_forge.run_db import db as db_mod
    db_mod._GLOBAL_DB.clear()
    monkeypatch.setenv("HALOFORGE_RUN_DB_PATH", str(tmp_path / "runs.db"))

    from fastapi.testclient import TestClient
    from halo_forge.public_api.app import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client


def test_run_eval_returns_summary_when_present(cohort_client):
    r = cohort_client.get("/api/public/runs/run-a/eval")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    by_task = {t["task"]: t for t in body["tasks"]}
    assert by_task["mmlu"]["value"] == 0.65
    assert by_task["mmlu"]["primary_metric"] == "acc"


def test_run_eval_unavailable_when_no_summary(cohort_client):
    r = cohort_client.get("/api/public/runs/run-c/eval")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert "halo-forge eval" in body["reason"]


def test_run_eval_missing_run_returns_unavailable_not_404(cohort_client):
    """The dashboard polls eval status across N runs — a 404 on one
    breaks the page. The honest contract is `available: false`."""
    r = cohort_client.get("/api/public/runs/does-not-exist/eval")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert "Run not found" in body["reason"]


def test_cohort_returns_grid_for_three_runs(cohort_client):
    r = cohort_client.get(
        "/api/public/eval/cohort"
        "?run_ids=run-a&run_ids=run-b&run_ids=run-c"
    )
    assert r.status_code == 200
    body = r.json()
    # 3 runs in order; all_tasks across all runs = {mmlu, gsm8k}.
    assert [run["run_id"] for run in body["runs"]] == ["run-a", "run-b", "run-c"]
    assert set(body["tasks"]) == {"mmlu", "gsm8k"}
    # run-c has no eval summary — entries available=False.
    by_run = {r["run_id"]: r for r in body["runs"]}
    assert by_run["run-c"]["available"] is False
    assert by_run["run-a"]["available"] is True


def test_cohort_cells_carry_per_task_value(cohort_client):
    r = cohort_client.get(
        "/api/public/eval/cohort?run_ids=run-a&run_ids=run-b"
    )
    body = r.json()
    cells = body["cells"]
    assert cells["run-a"]["mmlu"]["value"] == 0.65
    assert cells["run-b"]["mmlu"]["value"] == 0.70
    assert cells["run-a"]["gsm8k"]["value"] == 0.42
    assert cells["run-b"]["gsm8k"]["value"] == 0.40


def test_cohort_picks_best_per_task_higher_is_better(cohort_client):
    r = cohort_client.get(
        "/api/public/eval/cohort?run_ids=run-a&run_ids=run-b"
    )
    body = r.json()
    best = body["best_per_task_higher_is_better"]
    # mmlu: run-b at 0.70 > run-a at 0.65
    assert best["mmlu"] == "run-b"
    # gsm8k: run-a at 0.42 > run-b at 0.40
    assert best["gsm8k"] == "run-a"


def test_cohort_skips_runs_without_eval_in_best_calculation(cohort_client):
    r = cohort_client.get(
        "/api/public/eval/cohort"
        "?run_ids=run-a&run_ids=run-b&run_ids=run-c"
    )
    body = r.json()
    # run-c shouldn't poison the best calculation.
    assert body["best_per_task_higher_is_better"]["mmlu"] in {"run-a", "run-b"}


def test_cohort_empty_run_ids_returns_422(cohort_client):
    """Pydantic / FastAPI validates `run_ids` as min_length=1."""
    r = cohort_client.get("/api/public/eval/cohort")
    assert r.status_code == 422


def test_cohort_handles_unreadable_summary(cohort_client, tmp_path):
    """A corrupt lm_eval_summary.json shouldn't crash the cohort —
    that run shows up as unavailable, the others render normally."""
    # Corrupt run-a's summary.
    (tmp_path / "run-a" / "lm_eval_summary.json").write_text("{not valid json")

    r = cohort_client.get(
        "/api/public/eval/cohort?run_ids=run-a&run_ids=run-b"
    )
    body = r.json()
    by_run = {r["run_id"]: r for r in body["runs"]}
    assert by_run["run-a"]["available"] is False
    assert "unreadable" in (by_run["run-a"]["reason"] or "").lower()
    # run-b is unaffected.
    assert by_run["run-b"]["available"] is True
