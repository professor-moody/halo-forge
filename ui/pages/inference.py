"""
Inference Launch Page

Configure and launch inference optimize/benchmark jobs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nicegui import app, ui

from ui.components.file_picker import FilePicker
from ui.components.notifications import notify_job_failed, notify_job_started
from ui.services import get_inference_service
from ui.services.ops_readiness_service import get_ops_readiness_service
from ui.state import state
from ui.theme import COLORS


InferenceMode = Literal["optimize", "benchmark"]


@dataclass
class InferenceFormData:
    model: str = "Qwen/Qwen2.5-Coder-3B"
    mode: InferenceMode = "optimize"
    output_dir: str = "models/optimized"
    target_precision: str = "int4"
    target_latency: float = 50.0
    calibration_data: str = ""
    dry_run: bool = False
    prompts: str = ""
    num_prompts: int = 10
    max_tokens: int = 100
    warmup: int = 3
    measure_memory: bool = False


class Inference:
    """Inference command launcher page."""

    def __init__(self):
        self.data = InferenceFormData()
        self.is_running = False
        self.inference_service = get_inference_service(state)
        self.ops_readiness_service = get_ops_readiness_service()
        self._tabs_container = None
        self._form_container = None
        self._consume_clone_payload()

    def _consume_clone_payload(self) -> None:
        payload = app.storage.user.pop("inference_clone_payload", None)
        if not isinstance(payload, dict):
            return
        args = payload.get("args")
        if not isinstance(args, dict):
            return
        mode = str(args.get("mode") or self.data.mode).strip().lower()
        if mode in {"optimize", "benchmark"}:
            self.data.mode = mode  # type: ignore[assignment]
        self.data.model = str(args.get("model") or self.data.model)
        self.data.output_dir = str(args.get("output_dir") or self.data.output_dir)
        self.data.target_precision = str(args.get("target_precision") or self.data.target_precision)
        if args.get("target_latency") is not None:
            try:
                self.data.target_latency = float(args.get("target_latency"))
            except (TypeError, ValueError):
                pass
        self.data.calibration_data = str(args.get("calibration_data") or self.data.calibration_data)
        self.data.dry_run = bool(args.get("dry_run", self.data.dry_run))
        self.data.prompts = str(args.get("prompts") or self.data.prompts)
        for key in ("num_prompts", "max_tokens", "warmup"):
            value = args.get(key)
            if value is None:
                continue
            try:
                setattr(self.data, key, int(value))
            except (TypeError, ValueError):
                pass
        self.data.measure_memory = bool(args.get("measure_memory", self.data.measure_memory))

    def render(self) -> None:
        with ui.column().classes("page-content w-full gap-6 p-6"):
            with ui.row().classes("w-full items-center justify-between animate-in"):
                ui.label("Inference").classes(
                    f"text-2xl font-bold text-[{COLORS['text_primary']}]"
                )
                ui.label("Optimize and benchmark inference runtime").classes(
                    f"text-sm text-[{COLORS['text_secondary']}]"
                )

            self._render_all_module_readiness_banner()

            with ui.row().classes(
                f"w-full gap-2 p-2 rounded-xl bg-[{COLORS['bg_card']}] "
                f"border border-[#2d343c] animate-in stagger-1"
            ) as self._tabs_container:
                self._render_mode_tabs()

            with ui.column().classes("w-full gap-6") as self._form_container:
                self._render_form()

    def _render_mode_tabs(self) -> None:
        self._mode_button("Optimize", "optimize", "tune")
        self._mode_button("Benchmark", "benchmark", "speed")

    def _mode_button(self, label: str, mode: InferenceMode, icon: str) -> None:
        is_active = self.data.mode == mode
        with ui.element("div").classes(
            "flex-1 flex items-center justify-center gap-3 py-4 rounded-lg cursor-pointer transition-all "
            + (
                f"bg-[{COLORS['primary']}]/20 border border-[{COLORS['primary']}]"
                if is_active
                else f"bg-transparent border border-transparent hover:bg-[{COLORS['bg_hover']}]"
            )
        ).on("click", lambda m=mode: self._set_mode(m)):
            ui.icon(icon, size="24px").classes(
                f"text-[{COLORS['primary']}]" if is_active else f"text-[{COLORS['text_secondary']}]"
            )
            ui.label(label).classes(
                "text-base font-medium "
                + (
                    f"text-[{COLORS['primary']}]"
                    if is_active
                    else f"text-[{COLORS['text_secondary']}]"
                )
            )

    def _set_mode(self, mode: InferenceMode) -> None:
        self.data.mode = mode
        if mode == "benchmark" and self.data.output_dir == "models/optimized":
            self.data.output_dir = "results/inference_benchmarks"
        if mode == "optimize" and self.data.output_dir == "results/inference_benchmarks":
            self.data.output_dir = "models/optimized"
        self._tabs_container.clear()
        with self._tabs_container:
            self._render_mode_tabs()
        self._form_container.clear()
        with self._form_container:
            self._render_form()

    def _render_form(self) -> None:
        with ui.row().classes("w-full gap-6"):
            with ui.column().classes("flex-1 gap-6"):
                self._render_model_section()
                self._render_mode_specific_section()
            with ui.column().classes("flex-1 gap-6"):
                self._render_output_section()
                self._render_launch_section()

    def _render_model_section(self) -> None:
        with ui.column().classes(
            f"w-full gap-4 p-5 rounded-xl bg-[{COLORS['bg_card']}] border border-[#2d343c]"
        ):
            ui.label("Model").classes(f"text-base font-semibold text-[{COLORS['text_primary']}]")
            ui.input(
                value=self.data.model,
                placeholder="Model ID or local path",
                on_change=lambda e: setattr(self.data, "model", e.value.strip()),
            ).classes("w-full").props("outlined")

    def _render_mode_specific_section(self) -> None:
        with ui.column().classes(
            f"w-full gap-4 p-5 rounded-xl bg-[{COLORS['bg_card']}] border border-[#2d343c]"
        ):
            ui.label("Settings").classes(f"text-base font-semibold text-[{COLORS['text_primary']}]")
            if self.data.mode == "optimize":
                ui.select(
                    options={"int4": "INT4", "int8": "INT8", "fp16": "FP16"},
                    value=self.data.target_precision,
                    on_change=lambda e: setattr(self.data, "target_precision", e.value),
                    label="Target Precision",
                ).classes("w-full").props("outlined")
                ui.number(
                    value=self.data.target_latency,
                    min=1,
                    step=1,
                    label="Target Latency (ms)",
                    on_change=lambda e: setattr(self.data, "target_latency", float(e.value or 50.0)),
                ).classes("w-full").props("outlined")
                with ui.row().classes("w-full gap-2 items-center"):
                    ui.input(
                        value=self.data.calibration_data,
                        placeholder="Calibration JSONL path (optional)",
                        on_change=lambda e: setattr(self.data, "calibration_data", e.value.strip()),
                    ).classes("flex-1").props("outlined")
                    ui.button(
                        icon="folder_open",
                        on_click=lambda: self._browse_file("calibration"),
                    ).props("flat")
                ui.checkbox(
                    "Dry run",
                    value=self.data.dry_run,
                    on_change=lambda e: setattr(self.data, "dry_run", bool(e.value)),
                ).classes(f"text-[{COLORS['text_secondary']}]")
            else:
                with ui.row().classes("w-full gap-2 items-center"):
                    ui.input(
                        value=self.data.prompts,
                        placeholder="Prompt JSONL path (optional)",
                        on_change=lambda e: setattr(self.data, "prompts", e.value.strip()),
                    ).classes("flex-1").props("outlined")
                    ui.button(
                        icon="folder_open",
                        on_click=lambda: self._browse_file("prompts"),
                    ).props("flat")
                ui.number(
                    value=self.data.num_prompts,
                    min=1,
                    step=1,
                    label="Num Prompts",
                    on_change=lambda e: setattr(self.data, "num_prompts", int(e.value or 10)),
                ).classes("w-full").props("outlined")
                ui.number(
                    value=self.data.max_tokens,
                    min=1,
                    step=1,
                    label="Max Tokens",
                    on_change=lambda e: setattr(self.data, "max_tokens", int(e.value or 100)),
                ).classes("w-full").props("outlined")
                ui.number(
                    value=self.data.warmup,
                    min=1,
                    step=1,
                    label="Warmup Iterations",
                    on_change=lambda e: setattr(self.data, "warmup", int(e.value or 3)),
                ).classes("w-full").props("outlined")
                ui.checkbox(
                    "Measure Memory",
                    value=self.data.measure_memory,
                    on_change=lambda e: setattr(self.data, "measure_memory", bool(e.value)),
                ).classes(f"text-[{COLORS['text_secondary']}]")

    def _render_output_section(self) -> None:
        with ui.column().classes(
            f"w-full gap-4 p-5 rounded-xl bg-[{COLORS['bg_card']}] border border-[#2d343c]"
        ):
            ui.label("Output").classes(f"text-base font-semibold text-[{COLORS['text_primary']}]")
            with ui.row().classes("w-full items-center gap-2"):
                ui.input(
                    value=self.data.output_dir,
                    on_change=lambda e: setattr(self.data, "output_dir", e.value.strip()),
                ).classes("flex-1").props("outlined")
                ui.button(
                    icon="folder_open",
                    on_click=self._browse_output,
                ).props("flat")

    def _render_launch_section(self) -> None:
        with ui.column().classes(
            f"w-full gap-4 p-5 rounded-xl bg-[{COLORS['bg_card']}] border border-[#2d343c]"
        ):
            ui.label("Launch").classes(f"text-base font-semibold text-[{COLORS['text_primary']}]")
            ui.button(
                f"Launch {self.data.mode.title()}",
                icon="play_arrow",
                on_click=lambda: asyncio.create_task(self._launch()),
            ).props("unelevated").classes(
                f"w-full bg-[{COLORS['primary']}] text-white"
            )

    async def _launch(self) -> None:
        if self.is_running:
            return
        model = self.data.model.strip()
        output_dir = self.data.output_dir.strip()
        if not model:
            notify_job_failed("Inference", "Please provide a model")
            return
        if not output_dir:
            notify_job_failed("Inference", "Please provide an output directory")
            return

        self.is_running = True
        try:
            if self.data.mode == "optimize":
                job_id = await self.inference_service.launch_optimize(
                    model=model,
                    output_dir=output_dir,
                    target_precision=self.data.target_precision,
                    target_latency=float(self.data.target_latency),
                    calibration_data=self.data.calibration_data or None,
                    dry_run=self.data.dry_run,
                    source_ui_page="/inference",
                )
            else:
                job_id = await self.inference_service.launch_benchmark(
                    model=model,
                    output_dir=output_dir,
                    prompts=self.data.prompts or None,
                    num_prompts=int(self.data.num_prompts),
                    max_tokens=int(self.data.max_tokens),
                    warmup=int(self.data.warmup),
                    measure_memory=self.data.measure_memory,
                    source_ui_page="/inference",
                )

            notify_job_started(f"Inference {self.data.mode}: {Path(model).name}")
            ui.navigate.to(f"/monitor/{job_id}")
        except Exception as e:
            notify_job_failed("Inference", str(e))
        finally:
            self.is_running = False

    def _browse_output(self) -> None:
        FilePicker(
            start_path=".",
            on_select=lambda path: setattr(self.data, "output_dir", path),
            directories_only=True,
        )

    def _browse_file(self, target: Literal["calibration", "prompts"]) -> None:
        def _set_path(path: str) -> None:
            if target == "calibration":
                self.data.calibration_data = path
            else:
                self.data.prompts = path
            self._form_container.clear()
            with self._form_container:
                self._render_form()

    def _render_all_module_readiness_banner(self) -> None:
        """Render all-module readiness status for inference surface."""
        try:
            report = self.ops_readiness_service.get_effective_all_module_readiness()
        except Exception as e:
            ui.label(f"All-module readiness unavailable: {e}").classes(
                f"text-xs text-[{COLORS['warning']}]"
            )
            return

        entry = report.modules.get("inference")
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
            with ui.row().classes("items-center gap-2"):
                ui.icon(icon, size="16px").classes(f"text-[{color}]")
                source = f"{report.source}"
                if report.stale:
                    source += " (stale)"
                ui.label(
                    f"All-module readiness {status.upper()} • module=inference • source={source}"
                ).classes(f"text-xs text-[{color}] font-medium")
            if entry.errors:
                ui.label(f"Blocking reason: {entry.errors[0]}").classes(
                    f"text-xs text-[{COLORS['error']}]"
                )
            elif entry.warnings:
                ui.label(f"Warning: {entry.warnings[0]}").classes(
                    f"text-xs text-[{COLORS['warning']}]"
                )

        FilePicker(
            start_path=".",
            on_select=_set_path,
            directories_only=False,
        )
