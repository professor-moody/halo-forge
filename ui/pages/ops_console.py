"""
Ops Console Page

Contract-first utility module execution for config/data/info/plot.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from nicegui import app, ui

from ui.components.notifications import notify_job_failed, notify_job_started
from ui.components.diagnostic_panel import render_readiness_diagnostic_panel
from ui.query_params import get_query_param
from ui.services import get_module_ops_service
from ui.services.ops_readiness_service import get_ops_readiness_service
from ui.state import state
from ui.theme import COLORS


@dataclass
class OpsConsoleData:
    module: str = "config"
    execution_mode: str = "contract"  # contract | live
    output_root: str = "results/ops"

    config_path: str = "configs/sft_example.yaml"
    config_type: str = "auto"
    config_verbose: bool = False

    data_action: str = "validate"  # validate | prepare | generate | list
    data_file: str = "data/rlvr/humaneval_prompts.jsonl"
    dataset: str = "codeforces_cpp"
    topic: str = "rust_async"
    backend: str = "deepseek"
    backend_model: str = ""
    data_output: str = ""
    template: str = "qwen"
    system_prompt: str = ""

    plot_action: str = "benchmarks"  # training | benchmarks
    plot_input: str = "results/benchmarks"
    plot_output: str = ""
    plot_compare: bool = False


class OpsConsole:
    """Utility module launcher for operational testing and parity checks."""

    MODULES: tuple[tuple[str, str, str], ...] = (
        ("config", "Config", "settings"),
        ("data", "Data", "dataset"),
        ("info", "Info", "memory"),
        ("plot", "Plot", "show_chart"),
    )

    def __init__(self):
        self.data = OpsConsoleData()
        self.is_running = False
        self.module_ops_service = get_module_ops_service(state)
        self.ops_readiness_service = get_ops_readiness_service()
        self._banner_container = None
        self._tabs_container = None
        self._form_container = None
        self._query_warnings: list[str] = []
        self._consume_clone_payload()
        self._consume_query_params()

    def _consume_clone_payload(self) -> None:
        payload = app.storage.user.pop("ops_clone_payload", None)
        if not isinstance(payload, dict):
            return
        args = payload.get("args")
        if not isinstance(args, dict):
            return

        module = str(args.get("module") or payload.get("job_type") or self.data.module).strip().lower()
        if module in {"config", "data", "info", "plot"}:
            self.data.module = module

        execution_mode = str(args.get("execution_mode") or self.data.execution_mode).strip().lower()
        if execution_mode in {"contract", "live"}:
            self.data.execution_mode = execution_mode

        for key in (
            "output_root",
            "config_path",
            "config_type",
            "data_action",
            "data_file",
            "dataset",
            "topic",
            "backend",
            "backend_model",
            "data_output",
            "template",
            "system_prompt",
            "plot_action",
            "plot_input",
            "plot_output",
        ):
            if key in args and args.get(key) is not None:
                setattr(self.data, key, str(args.get(key)))

        for key in ("config_verbose", "plot_compare"):
            if key in args:
                setattr(self.data, key, bool(args.get(key)))

    def _consume_query_params(self) -> None:
        """Apply explicit query-param preselection (overrides clone payload)."""
        module = get_query_param("module", "").lower()
        if module:
            if module in {"config", "data", "info", "plot"}:
                self.data.module = module
            else:
                self._query_warnings.append(f"ignored invalid ops module query param: {module}")

        execution_mode = get_query_param("execution_mode", "").lower()
        if execution_mode:
            if execution_mode in {"contract", "live"}:
                self.data.execution_mode = execution_mode
            else:
                self._query_warnings.append(
                    f"ignored invalid ops execution_mode query param: {execution_mode}"
                )

    def render(self) -> None:
        with ui.column().classes("page-content w-full gap-6 p-6"):
            with ui.row().classes("w-full items-center justify-between animate-in"):
                ui.label("Ops Console").classes(
                    f"text-2xl font-bold text-[{COLORS['text_primary']}]"
                )
                ui.label("Utility module execution (contract-first, live optional)").classes(
                    f"text-sm text-[{COLORS['text_secondary']}]"
                )
            for warning in self._query_warnings:
                ui.label(warning).classes(f"text-xs text-[{COLORS['warning']}]")

            self._banner_container = ui.column().classes("w-full")
            with self._banner_container:
                self._render_readiness_banner()

            with ui.row().classes(
                f"w-full gap-2 p-2 rounded-xl bg-[{COLORS['bg_card']}] "
                f"border border-[#2d343c] animate-in stagger-1"
            ) as self._tabs_container:
                self._render_module_tabs()

            with ui.column().classes("w-full gap-6") as self._form_container:
                self._render_form()

    def _render_module_tabs(self) -> None:
        for module, label, icon in self.MODULES:
            is_active = self.data.module == module
            with ui.element("div").classes(
                "flex-1 flex items-center justify-center gap-3 py-4 rounded-lg cursor-pointer transition-all "
                + (
                    f"bg-[{COLORS['primary']}]/20 border border-[{COLORS['primary']}]"
                    if is_active
                    else f"bg-transparent border border-transparent hover:bg-[{COLORS['bg_hover']}]"
                )
            ).on("click", lambda m=module: self._set_module(m)):
                ui.icon(icon, size="20px").classes(
                    f"text-[{COLORS['primary']}]"
                    if is_active
                    else f"text-[{COLORS['text_secondary']}]"
                )
                ui.label(label).classes(
                    "text-base font-medium "
                    + (
                        f"text-[{COLORS['primary']}]"
                        if is_active
                        else f"text-[{COLORS['text_secondary']}]"
                    )
                )

    def _set_module(self, module: str) -> None:
        self.data.module = module
        self._refresh_form()

    def _refresh_form(self) -> None:
        if self._banner_container is not None:
            self._banner_container.clear()
            with self._banner_container:
                self._render_readiness_banner()

        if self._tabs_container is not None:
            self._tabs_container.clear()
            with self._tabs_container:
                self._render_module_tabs()

        if self._form_container is not None:
            self._form_container.clear()
            with self._form_container:
                self._render_form()

    def _render_form(self) -> None:
        with ui.column().classes(
            f"w-full gap-4 p-5 rounded-xl bg-[{COLORS['bg_card']}] border border-[#2d343c]"
        ):
            ui.label("Execution Settings").classes(
                f"text-base font-semibold text-[{COLORS['text_primary']}]"
            )

            ui.select(
                options={"contract": "Contract (default)", "live": "Live"},
                value=self.data.execution_mode,
                on_change=lambda e: self._set_execution_mode(str(e.value)),
                label="Execution Mode",
            ).classes("w-full").props("outlined")

            ui.input(
                value=self.data.output_root,
                on_change=lambda e: setattr(self.data, "output_root", e.value.strip()),
                label="Output Root",
            ).classes("w-full").props("outlined")

            mode_help = (
                "Contract mode runs safe bounded command-shape checks."
                if self.data.execution_mode == "contract"
                else "Live mode runs explicit utility commands with your provided args."
            )
            ui.label(mode_help).classes(f"text-xs text-[{COLORS['text_muted']}]")

        with ui.column().classes(
            f"w-full gap-4 p-5 rounded-xl bg-[{COLORS['bg_card']}] border border-[#2d343c]"
        ):
            ui.label("Module Arguments").classes(
                f"text-base font-semibold text-[{COLORS['text_primary']}]"
            )
            self._render_module_inputs()

        with ui.column().classes(
            f"w-full gap-4 p-5 rounded-xl bg-[{COLORS['bg_card']}] border border-[#2d343c]"
        ):
            ui.label("Launch").classes(f"text-base font-semibold text-[{COLORS['text_primary']}]")
            ui.button(
                f"Run {self.data.module.upper()} ({self.data.execution_mode})",
                icon="play_arrow",
                on_click=lambda: asyncio.create_task(self._launch()),
            ).props("unelevated").classes(
                f"w-full bg-[{COLORS['primary']}] text-white"
            )

    def _set_execution_mode(self, mode: str) -> None:
        if mode in {"contract", "live"}:
            self.data.execution_mode = mode
        self._refresh_form()

    def _render_module_inputs(self) -> None:
        if self.data.module == "config":
            ui.input(
                value=self.data.config_path,
                on_change=lambda e: setattr(self.data, "config_path", e.value.strip()),
                label="Config Path",
            ).classes("w-full").props("outlined")
            ui.select(
                options={"auto": "Auto", "sft": "SFT", "raft": "RAFT"},
                value=self.data.config_type,
                on_change=lambda e: setattr(self.data, "config_type", str(e.value)),
                label="Config Type",
            ).classes("w-full").props("outlined")
            ui.checkbox(
                "Verbose output",
                value=self.data.config_verbose,
                on_change=lambda e: setattr(self.data, "config_verbose", bool(e.value)),
            ).classes(f"text-[{COLORS['text_secondary']}]")
            return

        if self.data.module == "data":
            ui.select(
                options={
                    "validate": "Validate",
                    "prepare": "Prepare",
                    "generate": "Generate",
                    "list": "List",
                },
                value=self.data.data_action,
                on_change=lambda e: self._set_data_action(str(e.value)),
                label="Data Action",
            ).classes("w-full").props("outlined")

            if self.data.data_action == "validate":
                ui.input(
                    value=self.data.data_file,
                    on_change=lambda e: setattr(self.data, "data_file", e.value.strip()),
                    label="Data File",
                ).classes("w-full").props("outlined")
            elif self.data.data_action == "prepare":
                ui.input(
                    value=self.data.dataset,
                    on_change=lambda e: setattr(self.data, "dataset", e.value.strip()),
                    label="Dataset",
                ).classes("w-full").props("outlined")
                ui.input(
                    value=self.data.data_output,
                    on_change=lambda e: setattr(self.data, "data_output", e.value.strip()),
                    label="Output File (optional)",
                ).classes("w-full").props("outlined")
                ui.input(
                    value=self.data.template,
                    on_change=lambda e: setattr(self.data, "template", e.value.strip()),
                    label="Template",
                ).classes("w-full").props("outlined")
                ui.input(
                    value=self.data.system_prompt,
                    on_change=lambda e: setattr(self.data, "system_prompt", e.value),
                    label="System Prompt (optional)",
                ).classes("w-full").props("outlined")
            elif self.data.data_action == "generate":
                ui.input(
                    value=self.data.topic,
                    on_change=lambda e: setattr(self.data, "topic", e.value.strip()),
                    label="Topic",
                ).classes("w-full").props("outlined")
                ui.input(
                    value=self.data.backend,
                    on_change=lambda e: setattr(self.data, "backend", e.value.strip()),
                    label="Backend",
                ).classes("w-full").props("outlined")
                ui.input(
                    value=self.data.backend_model,
                    on_change=lambda e: setattr(self.data, "backend_model", e.value.strip()),
                    label="Backend Model (optional)",
                ).classes("w-full").props("outlined")
                ui.input(
                    value=self.data.data_output,
                    on_change=lambda e: setattr(self.data, "data_output", e.value.strip()),
                    label="Output File (optional)",
                ).classes("w-full").props("outlined")
                ui.input(
                    value=self.data.template,
                    on_change=lambda e: setattr(self.data, "template", e.value.strip()),
                    label="Template",
                ).classes("w-full").props("outlined")
            return

        if self.data.module == "info":
            ui.label("No additional arguments required.").classes(
                f"text-sm text-[{COLORS['text_secondary']}]"
            )
            return

        if self.data.module == "plot":
            ui.select(
                options={"benchmarks": "Benchmarks", "training": "Training"},
                value=self.data.plot_action,
                on_change=lambda e: self._set_plot_action(str(e.value)),
                label="Plot Action",
            ).classes("w-full").props("outlined")
            ui.input(
                value=self.data.plot_input,
                on_change=lambda e: setattr(self.data, "plot_input", e.value.strip()),
                label="Input Path",
            ).classes("w-full").props("outlined")
            ui.input(
                value=self.data.plot_output,
                on_change=lambda e: setattr(self.data, "plot_output", e.value.strip()),
                label="Output Path (optional)",
            ).classes("w-full").props("outlined")
            if self.data.plot_action == "training":
                ui.checkbox(
                    "Compare multiple runs",
                    value=self.data.plot_compare,
                    on_change=lambda e: setattr(self.data, "plot_compare", bool(e.value)),
                ).classes(f"text-[{COLORS['text_secondary']}]")

    def _set_data_action(self, action: str) -> None:
        self.data.data_action = action
        self._refresh_form()

    def _set_plot_action(self, action: str) -> None:
        self.data.plot_action = action
        self._refresh_form()

    def _render_readiness_banner(self) -> None:
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

        entry = report.modules.get(self.data.module)
        if entry is None:
            return
        render_readiness_diagnostic_panel(
            module=self.data.module,
            entry=entry,
            source=report.source,
            stale=bool(report.stale),
            expected_path=str(output_map.get(self.data.module) or entry.last_output_dir or ""),
            on_probe=self._run_contract_probe,
        )

    def _run_contract_probe(self) -> None:
        ok, message = self.ops_readiness_service.run_contract_probe(
            module=self.data.module,
            include_all_modules=True,
        )
        ui.notify(message, type="positive" if ok else "warning", timeout=1800)
        self._refresh_form()

    async def _launch(self) -> None:
        if self.is_running:
            return

        self.is_running = True
        try:
            launch_kwargs = self._build_launch_kwargs()
            job_id = await self.module_ops_service.launch_module_op(
                module=self.data.module,
                execution_mode=self.data.execution_mode,
                output_root=self.data.output_root,
                source_ui_page="/ops-console",
                **launch_kwargs,
            )
            notify_job_started(f"{self.data.module.upper()} job started")
            ui.navigate.to(f"/monitor/{job_id}")
        except Exception as e:
            notify_job_failed("Ops Console", str(e))
        finally:
            self.is_running = False

    def _build_launch_kwargs(self) -> dict:
        if self.data.module == "config":
            return {
                "config_path": self.data.config_path,
                "config_type": self.data.config_type,
                "verbose": self.data.config_verbose,
            }

        if self.data.module == "data":
            return {
                "data_action": self.data.data_action,
                "data_file": self.data.data_file,
                "dataset": self.data.dataset,
                "topic": self.data.topic,
                "backend": self.data.backend,
                "backend_model": self.data.backend_model,
                "data_output": self.data.data_output,
                "template": self.data.template,
                "system_prompt": self.data.system_prompt,
            }

        if self.data.module == "plot":
            return {
                "plot_action": self.data.plot_action,
                "plot_input": self.data.plot_input,
                "plot_output": self.data.plot_output,
                "plot_compare": self.data.plot_compare,
            }

        return {}
