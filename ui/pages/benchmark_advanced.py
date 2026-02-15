"""
Benchmark Advanced Page

Batch orchestration across non-code benchmark domains.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from nicegui import ui

from ui.components.notifications import notify_job_failed, notify_job_started
from ui.services import BenchmarkType, get_benchmark_service, get_presets_for_type
from ui.services.ops_readiness_service import get_ops_readiness_service
from ui.state import state
from ui.theme import COLORS


@dataclass
class BenchmarkAdvancedData:
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    output_root: str = "results/benchmarks"
    limit: int = 100
    run_vlm: bool = True
    run_audio: bool = True
    run_reasoning: bool = True
    run_agentic: bool = True
    vlm_dataset: str = "textvqa"
    audio_dataset: str = "librispeech"
    reasoning_dataset: str = "gsm8k"
    agentic_dataset: str = "xlam"


class BenchmarkAdvanced:
    """Multi-run benchmark orchestrator for ops testing."""

    def __init__(self):
        self.data = BenchmarkAdvancedData()
        self.benchmark_service = get_benchmark_service(state)
        self.ops_readiness_service = get_ops_readiness_service()
        self.is_running = False
        self._init_defaults()

    def _init_defaults(self) -> None:
        def _first_dataset(benchmark_type: BenchmarkType, fallback: str) -> str:
            presets = get_presets_for_type(benchmark_type)
            return presets[0].dataset if presets else fallback

        self.data.vlm_dataset = _first_dataset(BenchmarkType.VLM, self.data.vlm_dataset)
        self.data.audio_dataset = _first_dataset(BenchmarkType.AUDIO, self.data.audio_dataset)
        self.data.reasoning_dataset = _first_dataset(
            BenchmarkType.REASONING,
            self.data.reasoning_dataset,
        )
        self.data.agentic_dataset = _first_dataset(BenchmarkType.AGENTIC, self.data.agentic_dataset)

    def render(self) -> None:
        with ui.column().classes("page-content w-full gap-6 p-6"):
            with ui.row().classes("w-full items-center justify-between animate-in"):
                ui.label("Benchmark Advanced").classes(
                    f"text-2xl font-bold text-[{COLORS['text_primary']}]"
                )
                ui.label("Batch non-code benchmark orchestration").classes(
                    f"text-sm text-[{COLORS['text_secondary']}]"
                )

            self._render_all_module_readiness_banner()

            with ui.row().classes("w-full gap-6 flex-wrap"):
                with ui.column().classes(
                    f"flex-1 min-w-[350px] gap-4 p-5 rounded-xl bg-[{COLORS['bg_card']}] border border-[#2d343c]"
                ):
                    self._render_core_fields()
                with ui.column().classes(
                    f"flex-1 min-w-[350px] gap-4 p-5 rounded-xl bg-[{COLORS['bg_card']}] border border-[#2d343c]"
                ):
                    self._render_domain_selection()

            with ui.column().classes(
                f"w-full gap-4 p-5 rounded-xl bg-[{COLORS['bg_card']}] border border-[#2d343c]"
            ):
                ui.label("Launch").classes(f"text-base font-semibold text-[{COLORS['text_primary']}]")
                ui.label(
                    "This launches one benchmark job per selected domain."
                ).classes(f"text-xs text-[{COLORS['text_muted']}]")
                ui.button(
                    "Launch Benchmark Batch",
                    icon="playlist_play",
                    on_click=lambda: asyncio.create_task(self._launch_batch()),
                ).props("unelevated").classes(
                    f"w-full bg-[{COLORS['primary']}] text-white"
                )

    def _render_all_module_readiness_banner(self) -> None:
        """Render all-module readiness status for advanced benchmark surface."""
        try:
            report = self.ops_readiness_service.get_effective_all_module_readiness()
            output_map = self.ops_readiness_service.resolve_effective_output_map(
                include_all_modules=True
            )
        except Exception as e:
            ui.label(f"All-module readiness unavailable: {e}").classes(
                f"text-xs text-[{COLORS['warning']}]"
            )
            return

        entry = report.modules.get("benchmark_non_code")
        if entry is None:
            return

        status = entry.status.lower()
        if status == "pass":
            color = COLORS["success"]
            icon = "check_circle"
        elif status == "warn":
            color = COLORS["warning"]
            icon = "warning"
        else:
            color = COLORS["error"]
            icon = "error"

        with ui.column().classes(
            f"w-full gap-2 p-3 rounded-lg border border-[{color}]/30 bg-[{color}]/10"
        ):
            with ui.row().classes("items-center justify-between gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon(icon, size="16px").classes(f"text-[{color}]")
                    source = f"{report.source}"
                    if report.stale:
                        source += " (stale)"
                    ui.label(
                        f"All-module readiness {status.upper()} • module=benchmark_non_code • source={source}"
                    ).classes(f"text-xs text-[{color}] font-medium")
                ui.button(
                    "Run contract probe",
                    icon="play_arrow",
                    on_click=self._run_contract_probe,
                ).props("flat dense size=sm").classes(
                    f"text-[{COLORS['text_secondary']}]"
                )
            if entry.errors:
                if entry.launch_blocked:
                    ui.label(f"Launch blocked: {entry.errors[0]}").classes(
                        f"text-xs text-[{COLORS['error']}]"
                    )
                else:
                    ui.label(f"Evidence missing (non-blocking): {entry.errors[0]}").classes(
                        f"text-xs text-[{COLORS['warning']}]"
                    )
            elif entry.warnings:
                ui.label(f"Evidence missing (non-blocking): {entry.warnings[0]}").classes(
                    f"text-xs text-[{COLORS['warning']}]"
                )
            if entry.action_hint:
                ui.label(f"Action: {entry.action_hint}").classes(
                    f"text-xs text-[{COLORS['text_secondary']}]"
                )
            expected_path = output_map.get("benchmark_non_code") or entry.last_output_dir
            if expected_path:
                ui.label(f"What is missing? Expected evidence root: {expected_path}").classes(
                    f"text-xs font-mono text-[{COLORS['text_muted']}]"
                )

    def _run_contract_probe(self) -> None:
        ok, message = self.ops_readiness_service.run_contract_probe(
            module="benchmark_non_code",
            include_all_modules=True,
        )
        ui.notify(message, type="positive" if ok else "warning", timeout=1800)
        self._refresh_view()

    def _render_core_fields(self) -> None:
        ui.label("Batch Config").classes(f"text-base font-semibold text-[{COLORS['text_primary']}]")
        ui.input(
            value=self.data.model,
            on_change=lambda e: setattr(self.data, "model", e.value.strip()),
            placeholder="Model ID or local path",
            label="Model",
        ).classes("w-full").props("outlined")
        ui.input(
            value=self.data.output_root,
            on_change=lambda e: setattr(self.data, "output_root", e.value.strip()),
            label="Output Root",
        ).classes("w-full").props("outlined")
        ui.number(
            value=self.data.limit,
            min=1,
            step=1,
            on_change=lambda e: setattr(self.data, "limit", int(e.value or 100)),
            label="Limit per Benchmark",
        ).classes("w-full").props("outlined")

    def _render_domain_selection(self) -> None:
        ui.label("Domains").classes(f"text-base font-semibold text-[{COLORS['text_primary']}]")
        self._domain_row(
            "VLM",
            "run_vlm",
            "vlm_dataset",
            BenchmarkType.VLM,
        )
        self._domain_row(
            "Audio",
            "run_audio",
            "audio_dataset",
            BenchmarkType.AUDIO,
        )
        self._domain_row(
            "Reasoning",
            "run_reasoning",
            "reasoning_dataset",
            BenchmarkType.REASONING,
        )
        self._domain_row(
            "Agentic",
            "run_agentic",
            "agentic_dataset",
            BenchmarkType.AGENTIC,
        )

    def _domain_row(
        self,
        label: str,
        enabled_key: str,
        dataset_key: str,
        benchmark_type: BenchmarkType,
    ) -> None:
        presets = get_presets_for_type(benchmark_type)
        options = {preset.dataset: preset.name for preset in presets}
        with ui.row().classes("w-full items-center gap-3"):
            ui.checkbox(
                label,
                value=bool(getattr(self.data, enabled_key)),
                on_change=lambda e, key=enabled_key: setattr(self.data, key, bool(e.value)),
            ).classes(f"text-[{COLORS['text_secondary']}]")
            ui.select(
                options=options or {getattr(self.data, dataset_key): getattr(self.data, dataset_key)},
                value=getattr(self.data, dataset_key),
                on_change=lambda e, key=dataset_key: setattr(self.data, key, str(e.value)),
            ).classes("flex-1").props("outlined dense")

    async def _launch_batch(self) -> None:
        if self.is_running:
            return
        model = self.data.model.strip()
        output_root = self.data.output_root.strip()
        if not model:
            notify_job_failed("Benchmark Batch", "Please provide a model")
            return
        if not output_root:
            notify_job_failed("Benchmark Batch", "Please provide an output root")
            return

        requested = []
        if self.data.run_vlm:
            requested.append((BenchmarkType.VLM, self.data.vlm_dataset, {}))
        if self.data.run_audio:
            requested.append((BenchmarkType.AUDIO, self.data.audio_dataset, {"task": "asr"}))
        if self.data.run_reasoning:
            requested.append((BenchmarkType.REASONING, self.data.reasoning_dataset, {"split": "test"}))
        if self.data.run_agentic:
            requested.append((BenchmarkType.AGENTIC, self.data.agentic_dataset, {}))

        if not requested:
            notify_job_failed("Benchmark Batch", "Select at least one domain")
            return

        self.is_running = True
        launched_jobs: list[str] = []
        try:
            for benchmark_type, dataset, extra_args in requested:
                output_path = (
                    Path(output_root)
                    / f"{Path(model).name}-{dataset}"
                    / "benchmark.json"
                )
                job_id = await self.benchmark_service.launch_benchmark(
                    model=model,
                    benchmark_type=benchmark_type,
                    benchmark_name=dataset,
                    limit=int(self.data.limit),
                    output_path=str(output_path),
                    source_ui_page="/benchmark-advanced",
                    **extra_args,
                )
                launched_jobs.append(job_id)
            notify_job_started(f"Benchmark batch launched ({len(launched_jobs)} jobs)")
            ui.navigate.to(f"/monitor/{launched_jobs[0]}")
        except Exception as e:
            notify_job_failed("Benchmark Batch", str(e))
        finally:
            self.is_running = False

    def _refresh_view(self) -> None:
        ui.navigate.reload()
