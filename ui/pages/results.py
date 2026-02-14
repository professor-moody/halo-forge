"""
Results Page

Domain-specific benchmark results built from canonical ResultsService DTOs.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from nicegui import ui, app

from ui.theme import COLORS
from ui.state import state
from ui.services import (
    BenchmarkResult,
    TrainingRunSummary,
    TrainingService,
    get_benchmark_service,
    get_results_service,
    read_launch_context,
)


class Results:
    """Benchmark results page component."""

    DOMAIN_ORDER = ["code", "reasoning", "vlm", "audio", "agentic"]

    def __init__(self):
        self.results_service = get_results_service()
        self.training_service = TrainingService(state)
        self.benchmark_service = get_benchmark_service(state)
        self.results: list[BenchmarkResult] = self.results_service.list_results(force_refresh=True)
        self.training_runs: list[TrainingRunSummary] = self.results_service.list_training_runs(force_refresh=False)
        self.grouped_results = self.results_service.get_results_grouped_by_domain(force_refresh=False)
        self.sort_by: str = app.storage.user.get("results_sort_by", "timestamp")
        self.sort_desc: bool = app.storage.user.get("results_sort_desc", True)

    def render(self):
        with ui.column().classes("page-content w-full gap-6 p-6"):
            with ui.row().classes("w-full items-center justify-between animate-in"):
                ui.label("Benchmark Results").classes(
                    f'text-2xl font-bold text-[{COLORS["text_primary"]}]'
                )
                with ui.row().classes("items-center gap-2"):
                    ui.button("Refresh", icon="refresh", on_click=self._refresh).props("flat")
                    ui.button("Export", icon="download", on_click=self._export).props("flat")

            if not self.results and not self.training_runs:
                self._render_empty_state()
                return

            self._render_summary()

            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Domain Views").classes(
                    f'text-base font-semibold text-[{COLORS["text_primary"]}]'
                )
                with ui.row().classes("items-center gap-2"):
                    ui.label("Sort:").classes(f'text-xs text-[{COLORS["text_muted"]}]')
                    ui.select(
                        options=["timestamp", "primary", "model"],
                        value=self.sort_by,
                        on_change=lambda e: self._sort_results(e.value),
                    ).props("outlined dense dark").classes("w-32")

            displayed_any = False
            for domain in self.DOMAIN_ORDER:
                domain_results = self.grouped_results.get(domain, [])
                if not domain_results:
                    continue
                displayed_any = True
                self._render_domain_table(domain, self._sorted(domain_results))

            if self.training_runs:
                displayed_any = True
                self._render_training_runs_table(self.training_runs)

            if not displayed_any:
                self._render_empty_state()

    def _render_summary(self):
        by_domain = self.results_service.get_summary().get("by_domain", {})
        unique_models = len({r.model for r in self.results})
        latest_timestamp = None
        if self.results:
            latest_timestamp = max(self.results, key=lambda r: r.timestamp).timestamp
        elif self.training_runs:
            latest_timestamp = max(self.training_runs, key=lambda r: r.timestamp).timestamp

        with ui.row().classes("w-full gap-4 animate-in"):
            self._stat_card("Total Runs", str(len(self.results)), "analytics")
            self._stat_card("Training Runs", str(len(self.training_runs)), "auto_awesome")
            self._stat_card("Unique Models", str(unique_models), "psychology")
            self._stat_card("Latest", latest_timestamp.strftime("%Y-%m-%d") if latest_timestamp else "--", "schedule")
            self._stat_card("Domains", str(len(by_domain)), "dashboard")

    def _stat_card(self, label: str, value: str, icon: str):
        with ui.column().classes(
            f'flex-1 min-w-[150px] gap-2 p-4 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c]'
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon(icon, size="20px").classes(f'text-[{COLORS["accent"]}]')
                ui.label(value).classes(
                    f'text-2xl font-bold text-[{COLORS["text_primary"]}]'
                )
            ui.label(label).classes(f'text-xs text-[{COLORS["text_muted"]}]')

    def _render_empty_state(self):
        with ui.column().classes(
            f'w-full gap-3 p-8 rounded-xl bg-[{COLORS["bg_card"]}] border border-[#2d343c] '
            "items-center justify-center"
        ):
            ui.icon("bar_chart", size="36px").classes(f'text-[{COLORS["text_muted"]}]')
            ui.label("No benchmark results found").classes(
                f'text-base font-semibold text-[{COLORS["text_primary"]}]'
            )
            ui.label("Run a benchmark from the Benchmark page to populate results.").classes(
                f'text-sm text-[{COLORS["text_muted"]}]'
            )
            ui.button("Go to Benchmark", on_click=lambda: ui.navigate.to("/benchmark")).props(
                "flat"
            ).classes(f'text-[{COLORS["accent"]}]')

    def _render_domain_table(self, domain: str, rows: list[BenchmarkResult]):
        columns = self.results_service.get_domain_metric_columns(domain)

        with ui.column().classes(
            f'w-full gap-3 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c] animate-in'
        ):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(f"{self._domain_title(domain)} ({len(rows)})").classes(
                    f'text-base font-semibold text-[{COLORS["text_primary"]}]'
                )
                ui.label(", ".join(label for _, label in columns)).classes(
                    f'text-xs text-[{COLORS["text_muted"]}]'
                )

            with ui.row().classes(
                f'w-full items-center gap-3 px-3 py-2 rounded-lg bg-[{COLORS["bg_secondary"]}]'
            ):
                ui.label("Model").classes(
                    f'flex-[2] text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                )
                ui.label("Benchmark").classes(
                    f'flex-1 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                )
                for _, label in columns:
                    ui.label(label).classes(
                        f'w-24 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}] text-right'
                    )
                ui.label("Samples").classes(
                    f'w-20 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}] text-right'
                )
                ui.label("Duration").classes(
                    f'w-20 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}] text-right'
                )
                ui.label("Time").classes(
                    f'w-28 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}] text-right'
                )
                ui.label("Actions").classes(
                    f'w-44 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}] text-right'
                )
                ui.label("Actions").classes(
                    f'w-36 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}] text-right'
                )

            for result in rows:
                with ui.row().classes(
                    f'w-full items-center gap-3 px-3 py-2 border-b border-[#2d343c] '
                    f'hover:bg-[{COLORS["bg_hover"]}]'
                ):
                    with ui.column().classes("flex-[2] gap-0"):
                        ui.label(Path(str(result.model)).name).classes(
                            f'text-sm text-[{COLORS["text_primary"]}]'
                        )
                        ui.label(str(result.model)).classes(
                            f'text-xs text-[{COLORS["text_muted"]}] truncate'
                        )
                    ui.label(str(result.benchmark)).classes(
                        f'flex-1 text-sm text-[{COLORS["text_secondary"]}]'
                    )
                    for key, _ in columns:
                        ui.label(self._format_metric(key, self._metric_value(result, key))).classes(
                            f'w-24 text-sm font-mono text-[{COLORS["primary"]}] text-right'
                        )
                    ui.label(str(result.samples)).classes(
                        f'w-20 text-sm font-mono text-[{COLORS["text_secondary"]}] text-right'
                    )
                    ui.label(f"{result.duration_seconds / 60:.1f}m").classes(
                        f'w-20 text-sm font-mono text-[{COLORS["text_muted"]}] text-right'
                    )
                    ui.label(result.timestamp.strftime("%m-%d %H:%M")).classes(
                        f'w-28 text-sm font-mono text-[{COLORS["text_muted"]}] text-right'
                    )
                    with ui.row().classes("w-36 justify-end gap-1"):
                        if result.has_relaunch_context and result.launch_context_path:
                            ui.button(
                                icon="replay",
                                on_click=lambda r=result: asyncio.create_task(self._relaunch_benchmark_result(r)),
                            ).props("flat round dense").classes(
                                f'text-[{COLORS["accent"]}]'
                            ).tooltip("Rerun")
                            ui.button(
                                icon="content_copy",
                                on_click=lambda r=result: self._clone_benchmark_to_form(r),
                            ).props("flat round dense").classes(
                                f'text-[{COLORS["text_secondary"]}]'
                            ).tooltip("Clone to Benchmark form")
                        else:
                            ui.label("--").classes(
                                f'w-full text-xs text-[{COLORS["text_muted"]}] text-right'
                            )

    def _render_training_runs_table(self, rows: list[TrainingRunSummary]):
        with ui.column().classes(
            f'w-full gap-3 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c] animate-in'
        ):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(f"Training Runs ({len(rows)})").classes(
                    f'text-base font-semibold text-[{COLORS["text_primary"]}]'
                )
                ui.label("Status metadata from training_summary.json").classes(
                    f'text-xs text-[{COLORS["text_muted"]}]'
                )

            with ui.row().classes(
                f'w-full items-center gap-3 px-3 py-2 rounded-lg bg-[{COLORS["bg_secondary"]}]'
            ):
                ui.label("Modality").classes(
                    f'w-28 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                )
                ui.label("Model").classes(
                    f'flex-[2] text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                )
                ui.label("Updated").classes(
                    f'w-20 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}] text-right'
                )
                ui.label("Steps").classes(
                    f'w-20 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}] text-right'
                )
                ui.label("Final Loss").classes(
                    f'w-24 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}] text-right'
                )
                ui.label("Reason").classes(
                    f'flex-1 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                )
                ui.label("Run").classes(
                    f'w-20 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}] text-right'
                )
                ui.label("Time").classes(
                    f'w-28 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}] text-right'
                )

            for run in rows[:20]:
                with ui.row().classes(
                    f'w-full items-center gap-3 px-3 py-2 border-b border-[#2d343c] '
                    f'hover:bg-[{COLORS["bg_hover"]}]'
                ):
                    ui.label(run.modality.upper()).classes(
                        f'w-28 text-sm text-[{COLORS["text_secondary"]}]'
                    )
                    with ui.column().classes("flex-[2] gap-0"):
                        ui.label(Path(str(run.model_name)).name).classes(
                            f'text-sm text-[{COLORS["text_primary"]}]'
                        )
                        ui.label(str(run.output_dir)).classes(
                            f'text-xs text-[{COLORS["text_muted"]}] truncate'
                        )
                    ui.label("yes" if run.weights_updated else "no").classes(
                        f'w-20 text-sm font-mono text-[{COLORS["primary"]}] text-right'
                    )
                    ui.label(str(run.total_train_steps_executed)).classes(
                        f'w-20 text-sm font-mono text-[{COLORS["text_secondary"]}] text-right'
                    )
                    ui.label(
                        f"{run.final_train_loss:.4f}" if isinstance(run.final_train_loss, (int, float)) else "--"
                    ).classes(
                        f'w-24 text-sm font-mono text-[{COLORS["text_secondary"]}] text-right'
                    )
                    ui.label(run.failure_reason or run.final_update_reason or "--").classes(
                        f'flex-1 text-sm text-[{COLORS["text_muted"]}] truncate'
                    )
                    ui.label(run.run_id or "--").classes(
                        f'w-20 text-xs font-mono text-[{COLORS["text_muted"]}] text-right'
                    )
                    ui.label(run.timestamp.strftime("%m-%d %H:%M")).classes(
                        f'w-28 text-sm font-mono text-[{COLORS["text_muted"]}] text-right'
                    )
                    with ui.row().classes("w-44 justify-end gap-1"):
                        if run.has_relaunch_context and run.launch_context_path:
                            ui.button(
                                icon="replay",
                                on_click=lambda r=run: asyncio.create_task(self._relaunch_training_run(r)),
                            ).props("flat round dense").classes(
                                f'text-[{COLORS["accent"]}]'
                            ).tooltip("Rerun")
                            if run.modality in {"raft", "vlm", "audio", "reasoning", "agentic"}:
                                ui.button(
                                    icon="history",
                                    on_click=lambda r=run: asyncio.create_task(
                                        self._relaunch_training_run(r, resume_latest=True)
                                    ),
                                ).props("flat round dense").classes(
                                    f'text-[{COLORS["info"]}]'
                                ).tooltip("Resume Latest")
                            ui.button(
                                icon="content_copy",
                                on_click=lambda r=run: self._clone_training_to_form(r),
                            ).props("flat round dense").classes(
                                f'text-[{COLORS["text_secondary"]}]'
                            ).tooltip("Clone to Training form")
                        else:
                            ui.label("--").classes(
                                f'w-full text-xs text-[{COLORS["text_muted"]}] text-right'
                            )

    def _metric_value(self, result: BenchmarkResult, key: str):
        if key in result.normalized_metrics:
            return result.normalized_metrics.get(key)
        if key == "pass_at_1":
            return result.pass_at_1
        if key == "pass_at_5":
            return result.pass_at_5
        if key == "pass_at_10":
            return result.pass_at_10
        if key == "accuracy":
            return result.accuracy
        return None

    def _format_metric(self, key: str, value):
        if value is None:
            return "--"

        numeric = float(value)
        if key in {
            "pass_at_1",
            "pass_at_5",
            "pass_at_10",
            "accuracy",
            "success_rate",
            "wer",
            "json_valid_rate",
            "function_correctness",
        }:
            return f"{numeric:.1%}"
        return f"{numeric:.3f}"

    def _sorted(self, rows: list[BenchmarkResult]) -> list[BenchmarkResult]:
        if self.sort_by == "model":
            return sorted(rows, key=lambda r: str(r.model).lower(), reverse=self.sort_desc)
        if self.sort_by == "primary":
            return sorted(
                rows,
                key=lambda r: (r.primary_metric is not None, r.primary_metric or 0.0),
                reverse=self.sort_desc,
            )
        return sorted(rows, key=lambda r: r.timestamp, reverse=self.sort_desc)

    def _sort_results(self, sort_by: str):
        if self.sort_by == sort_by:
            self.sort_desc = not self.sort_desc
        else:
            self.sort_by = sort_by
            self.sort_desc = True
        app.storage.user["results_sort_by"] = self.sort_by
        app.storage.user["results_sort_desc"] = self.sort_desc
        ui.navigate.to("/results")

    def _refresh(self):
        self.results = self.results_service.list_results(force_refresh=True)
        self.training_runs = self.results_service.list_training_runs(force_refresh=True)
        self.grouped_results = self.results_service.get_results_grouped_by_domain(force_refresh=False)
        ui.navigate.to("/results")

    def _export(self):
        payload = [result.to_dict() for result in self.results]
        print(json.dumps(payload, indent=2, default=str))
        ui.notify("Exported results payload to server console", type="positive")

    async def _relaunch_benchmark_result(self, result: BenchmarkResult):
        """Relaunch benchmark from durable launch context."""
        if not result.launch_context_path:
            ui.notify("No launch context found for this benchmark run", type="warning")
            return
        try:
            new_job_id = await self.benchmark_service.relaunch_from_context(
                result.launch_context_path,
                source_ui_page="/results",
            )
            ui.navigate.to(f"/monitor/{new_job_id}")
        except Exception as e:
            ui.notify(f"Benchmark relaunch failed: {e}", type="negative")

    async def _relaunch_training_run(self, run: TrainingRunSummary, resume_latest: bool = False):
        """Relaunch training run from durable launch context."""
        if not run.launch_context_path:
            ui.notify("No launch context found for this training run", type="warning")
            return
        try:
            new_job_id = await self.training_service.relaunch_from_context(
                run.launch_context_path,
                resume_latest=resume_latest,
                source_ui_page="/results",
            )
            ui.navigate.to(f"/monitor/{new_job_id}")
        except Exception as e:
            action = "Resume Latest" if resume_latest else "Relaunch"
            ui.notify(f"{action} failed: {e}", type="negative")

    def _clone_benchmark_to_form(self, result: BenchmarkResult):
        """Clone benchmark launch args into benchmark form."""
        if not result.launch_context_path:
            ui.notify("No launch context found for this benchmark run", type="warning")
            return
        try:
            context = read_launch_context(result.launch_context_path)
        except Exception as e:
            ui.notify(f"Launch context is invalid: {e}", type="negative")
            return
        app.storage.user["benchmark_clone_payload"] = {
            "launch_context_file": str(result.launch_context_path),
            "job_type": context.job_type,
            "args": context.args,
        }
        ui.navigate.to("/benchmark")

    def _clone_training_to_form(self, run: TrainingRunSummary):
        """Clone training launch args into training form."""
        if not run.launch_context_path:
            ui.notify("No launch context found for this training run", type="warning")
            return
        try:
            context = read_launch_context(run.launch_context_path)
        except Exception as e:
            ui.notify(f"Launch context is invalid: {e}", type="negative")
            return
        app.storage.user["training_clone_payload"] = {
            "launch_context_file": str(run.launch_context_path),
            "job_type": context.job_type,
            "args": context.args,
        }
        ui.navigate.to("/training")

    def _domain_title(self, domain: str) -> str:
        if domain == "vlm":
            return "VLM"
        return domain.capitalize()
