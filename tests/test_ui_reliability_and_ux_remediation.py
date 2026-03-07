#!/usr/bin/env python3
"""UI reliability and targeted UX remediation regression tests."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

try:
    import nicegui  # noqa: F401
except ModuleNotFoundError:
    class _FakeUI:
        def page(self, *_args, **_kwargs):
            def _decorator(func):
                return func

            return _decorator

    class _FakeApp:
        @staticmethod
        def add_static_files(*_args, **_kwargs):
            return None

    sys.modules["nicegui"] = types.SimpleNamespace(ui=_FakeUI(), app=_FakeApp())

import ui.app as ui_app
import ui.pages.dashboard as dashboard_module
from ui.components.sidebar import _is_active_route
from ui.services.event_bus import Event, EventType
from ui.services.results_service import ResultsService


def test_results_service_sorts_mixed_aware_and_fallback_timestamps(tmp_path):
    """Mixed timestamp sources should normalize to UTC-aware datetimes and sort safely."""
    older = tmp_path / "results" / "benchmarks" / "aware" / "benchmark.json"
    older.parent.mkdir(parents=True, exist_ok=True)
    older.write_text(
        json.dumps(
            {
                "model": "org/aware-model",
                "dataset": "humaneval",
                "timestamp": "2026-01-01T00:00:00Z",
                "pass_at_1": 0.5,
            }
        ),
        encoding="utf-8",
    )

    newer = tmp_path / "results" / "benchmarks" / "fallback" / "benchmark.json"
    newer.parent.mkdir(parents=True, exist_ok=True)
    newer.write_text(
        json.dumps(
            {
                "model": "org/fallback-model",
                "dataset": "humaneval",
                "pass_at_1": 0.6,
            }
        ),
        encoding="utf-8",
    )
    newer_epoch = datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc).timestamp()
    os.utime(newer, (newer_epoch, newer_epoch))

    service = ResultsService(base_path=tmp_path)
    results = service.list_results(force_refresh=True)

    assert [result.model for result in results] == [
        "org/fallback-model",
        "org/aware-model",
    ]
    assert all(result.timestamp.tzinfo is not None for result in results)


def test_dashboard_job_change_updates_active_job_count(monkeypatch):
    """Dashboard job-change handler should refresh active job counts on JOB_STARTED."""
    class _FakeState:
        @staticmethod
        def get_active_jobs():
            return [object(), object()]

    class _FakeLabel:
        def __init__(self):
            self.value = None

        def set_text(self, value: str) -> None:
            self.value = value

    class _FakeContainer:
        def __init__(self):
            self.cleared = False

        def clear(self) -> None:
            self.cleared = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(dashboard_module, "state", _FakeState())
    monkeypatch.setattr(dashboard_module, "get_results_service", lambda: object())
    monkeypatch.setattr(dashboard_module, "get_dashboard_hub_service", lambda: object())
    monkeypatch.setattr(dashboard_module, "get_ops_readiness_service", lambda: object())

    dashboard = dashboard_module.Dashboard()
    dashboard._active_jobs_count_label = _FakeLabel()
    dashboard._active_jobs_container = _FakeContainer()
    rerendered = []
    dashboard._render_active_jobs_content = lambda: rerendered.append(True)

    dashboard._on_job_change(Event(type=EventType.JOB_STARTED, data={}))

    assert dashboard._active_jobs_count_label.value == "2"
    assert dashboard._active_jobs_container.cleared is True
    assert rerendered == [True]


def test_layout_hardware_monitor_bootstrap_schedules_once(monkeypatch):
    """Shared layout helper should schedule hardware monitor startup exactly once."""
    class _FakeMonitor:
        is_running = False

        async def start(self):
            return None

    scheduled = []
    fake_monitor = _FakeMonitor()

    def fake_create_task(coro):
        scheduled.append(coro)
        coro.close()

        class _FakeTask:
            pass

        return _FakeTask()

    monkeypatch.setattr(ui_app, "get_hardware_monitor", lambda: fake_monitor)
    monkeypatch.setattr(ui_app.asyncio, "create_task", fake_create_task)
    monkeypatch.setattr(ui_app, "_hardware_monitor_start_requested", False)

    async def _exercise() -> None:
        ui_app._ensure_hardware_monitor_started()
        ui_app._ensure_hardware_monitor_started()

    asyncio.run(_exercise())

    assert len(scheduled) == 1


def test_sidebar_active_route_is_exact_or_segment_scoped():
    """Sidebar route matching should avoid prefix collisions while supporting nested routes."""
    assert _is_active_route("/benchmark", "/benchmark")
    assert not _is_active_route("/benchmark", "/benchmark-advanced")
    assert _is_active_route("/monitor", "/monitor/abc123")
    assert not _is_active_route("/training", "/train")


def test_results_page_updates_in_place_without_self_navigation():
    """Results page sort/filter refresh should rerender in place instead of self-navigation."""
    source = Path("ui/pages/results.py").read_text(encoding="utf-8")
    assert "def _rerender(self)" in source
    assert "ui.navigate.to(\"/results\")" not in source
    assert "self._rerender()" in source


def test_storage_secret_uses_loopback_fallback_with_warning(monkeypatch, capsys):
    """Loopback bindings may use a warning-backed development storage secret fallback."""
    monkeypatch.delenv("HALO_UI_STORAGE_SECRET", raising=False)
    monkeypatch.setattr(ui_app, "_storage_secret_warning_emitted", False)

    secret = ui_app._resolve_storage_secret("127.0.0.1")

    assert secret == ui_app._DEV_STORAGE_SECRET
    assert "HALO_UI_STORAGE_SECRET" in capsys.readouterr().out


def test_storage_secret_requires_explicit_secret_for_non_loopback(monkeypatch):
    """Non-loopback bindings should fail fast without HALO_UI_STORAGE_SECRET."""
    monkeypatch.delenv("HALO_UI_STORAGE_SECRET", raising=False)
    monkeypatch.setattr(ui_app, "_storage_secret_warning_emitted", False)

    with pytest.raises(RuntimeError, match="HALO_UI_STORAGE_SECRET"):
        ui_app._resolve_storage_secret("0.0.0.0")


def test_storage_secret_prefers_explicit_env(monkeypatch):
    """Explicit HALO_UI_STORAGE_SECRET should be honored for any host."""
    monkeypatch.setenv("HALO_UI_STORAGE_SECRET", "top-secret")

    assert ui_app._resolve_storage_secret("0.0.0.0") == "top-secret"
