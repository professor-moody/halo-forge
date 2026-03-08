"""
Monitor Page

Real-time job monitoring with loss charts and log viewer.
"""

from nicegui import ui, app
from datetime import datetime
from typing import Optional, Callable, List, Dict, Any
from pathlib import Path
import asyncio
import json

from halo_forge.training_recovery import build_recovery_guidance
from ui.theme import COLORS
from ui.state import state, JobState
from ui.services.training_presentation import (
    TrainingAction,
    TrainingPresentation,
    build_training_run_presentation,
)
from ui.services import (
    TrainingService,
    get_benchmark_service,
    get_bootstrap_service,
    get_inference_service,
    get_live_probe_service,
    get_module_ops_service,
    get_qualification_service,
    get_event_bus,
    Event,
    EventType,
    read_launch_context,
)
from ui.components.notifications import (
    notify_training_stopped,
    notify_job_started,
    notify_job_completed,
    notify_job_failed,
)

CYCLE_BASED_JOB_TYPES = {"raft", "vlm", "audio", "reasoning", "agentic"}
TRAINING_JOB_TYPES = {"sft"} | CYCLE_BASED_JOB_TYPES
UTILITY_JOB_TYPES = {"config", "data", "info", "plot"}
QUALIFICATION_JOB_TYPES = {"qualification"}
BOOTSTRAP_JOB_TYPES = {"bootstrap"}
LIVE_PROBE_JOB_TYPES = {"live_probe"}
DIAGNOSTIC_JOB_TYPES = QUALIFICATION_JOB_TYPES | BOOTSTRAP_JOB_TYPES | LIVE_PROBE_JOB_TYPES
INDETERMINATE_PROGRESS_JOB_TYPES = {"inference"} | UTILITY_JOB_TYPES | DIAGNOSTIC_JOB_TYPES

TRAINING_FIX_ROUTES = {
    "sft": "/training?mode=sft&ui_mode=quickstart&preset=sft_fast_local",
    "raft": "/training?mode=raft&ui_mode=quickstart&preset=raft_safe_default",
    "vlm": "/training?mode=vlm&ui_mode=quickstart&preset=vlm_tiny",
    "audio": "/training?mode=audio&ui_mode=quickstart&preset=audio_whisper_tiny",
    "reasoning": "/training?mode=reasoning&ui_mode=quickstart&preset=reasoning_small",
    "agentic": "/training?mode=agentic&ui_mode=quickstart&preset=agentic_small",
}


class Monitor:
    """Real-time job monitoring page component."""
    
    def __init__(self, job_id: Optional[str] = None):
        self.job_id = job_id
        self.job: Optional[JobState] = None
        self.update_timer = None
        self.log_lines: list[str] = []  # Legacy - kept for event handler compatibility
        self.training_service = TrainingService(state)
        self.benchmark_service = get_benchmark_service(state)
        self.inference_service = get_inference_service(state)
        self.module_ops_service = get_module_ops_service(state)
        self.qualification_service = get_qualification_service(state)
        self.bootstrap_service = get_bootstrap_service(state)
        self.live_probe_service = get_live_probe_service(state)
        self._update_task: Optional[asyncio.Task] = None
        self._unsubscribe_callbacks: List[Callable[[], None]] = []
        
        # Persistent log handling
        self._all_log_lines: list[str] = []  # Full log history
        self._displayed_log_count: int = 0   # Track what's already displayed
        
        # References to dynamic UI elements for live updates
        self._duration_label = None
        self._status_label = None
        self._duration_timer = None
        self._progress_percent_label = None
        self._progress_bar = None
        self._epoch_label = None
        self._step_label = None
        self._loss_label = None
        self._lr_label = None
        self._grad_norm_label = None
        self._verification_label = None
        self._weights_updated_label = None
        self._update_steps_label = None
        self._update_reason_label = None
        self._final_loss_label = None
        self._run_id_label = None
        self._seed_label = None
        self._resume_label = None
        self._failure_reason_label = None
        self._quality_status_label = None
        self._quality_keep_rate_label = None
        self._quality_drop_reason_label = None
        self._quality_action_label = None
        self._recovery_action_label = None
        self._quality_headline_label = None
        self._quality_summary_note_label = None
        self._benchmark_metric_labels: Dict[str, Any] = {}
        self._inference_metric_labels: Dict[str, Any] = {}
        self._utility_metric_labels: Dict[str, Any] = {}
        self._diagnostic_metric_labels: Dict[str, Any] = {}
        
        if job_id:
            self.job = state.get_job(job_id)
            # Load existing logs from file on mount
            self._load_logs_from_file()
    
    def _load_logs_from_file(self):
        """Load existing logs from persistent log file."""
        if not self.job:
            return
        
        # Try job's log_file_path first
        log_file_path = self._get_log_file_path()
        if log_file_path and log_file_path.exists():
            try:
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.rstrip('\n')
                        if line:
                            self._all_log_lines.append(line)
            except Exception as e:
                print(f"[Monitor] Error loading log file: {e}")
    
    def _get_log_file_path(self) -> Optional[Path]:
        """Get the path to the persistent log file."""
        if not self.job:
            return None
        
        if self.job.log_file_path:
            return Path(self.job.log_file_path)
        
        if self.job.output_dir:
            output_dir = Path(self.job.output_dir)
            candidates = [
                output_dir / f"{self.job_id}_training.log",
                output_dir / f"{self.job_id}_benchmark.log",
                output_dir / f"{self.job_id}_inference.log",
                output_dir / f"{self.job_id}_{self.job.type}.log",
                output_dir / "stdout.log",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return candidates[0]
        
        return None

    def _resolve_benchmark_progress_counts(self) -> tuple[int, int]:
        """Resolve benchmark evaluated/total counts from state, result payload, and lifecycle metadata."""
        payload = self._read_benchmark_result_payload() or {}
        lifecycle = (
            self.job.lifecycle_metadata
            if self.job and isinstance(self.job.lifecycle_metadata, dict)
            else {}
        )

        def _to_int(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        total = (
            _to_int(self.job.total_steps if self.job else 0)
            or _to_int(payload.get("total_prompts"))
            or _to_int(payload.get("samples"))
            or _to_int(lifecycle.get("benchmark_total_prompts"))
        )
        current = (
            _to_int(self.job.current_step if self.job else 0)
            or _to_int(payload.get("samples_evaluated"))
            or _to_int(payload.get("passed"))
            or _to_int(lifecycle.get("benchmark_samples_evaluated"))
            or _to_int(lifecycle.get("benchmark_passed"))
        )

        if total > 0 and current > total:
            total = current
        return current, total

    def _resolve_progress_display(self) -> tuple[str, str]:
        """Resolve progress percent and counter labels for the current job."""
        if not self.job:
            return "--", "--"
        if self.job.type == "benchmark":
            current, total = self._resolve_benchmark_progress_counts()
            if total > 0:
                return f"{(float(current) / float(total) * 100.0):.1f}%", f"{current}/{total}"
            return "--", f"{current}/?"
        if self.job.type in INDETERMINATE_PROGRESS_JOB_TYPES:
            current = int(self.job.current_step or 0)
            total = int(self.job.total_steps or 0)
            if total > 0:
                return f"{(float(current) / float(total) * 100.0):.1f}%", f"{current}/{total}"
            return "—", "indeterminate"
        return f"{self.job.progress_percent:.1f}%", f"{self.job.current_step}/{self.job.total_steps or '?'}"

    def _get_launch_context_path(self) -> Optional[Path]:
        """Resolve persisted launch context for the current job."""
        if not self.job:
            return None
        if self.job.launch_context_file and Path(self.job.launch_context_file).exists():
            return Path(self.job.launch_context_file)
        if self.job.output_dir:
            candidate = Path(self.job.output_dir) / "launch_context.json"
            if candidate.exists():
                return candidate
        return None

    def _get_launch_context(self) -> Optional[Any]:
        """Read launch context safely for action availability checks."""
        path = self._get_launch_context_path()
        if not path:
            return None
        try:
            return read_launch_context(path)
        except Exception:
            return None

    def _tone_color(self, tone: str) -> str:
        return {
            "success": COLORS["success"],
            "warning": COLORS["warning"],
            "danger": COLORS["error"],
            "neutral": COLORS["text_secondary"],
        }.get(str(tone or "").strip().lower(), COLORS["text_secondary"])

    def _current_training_presentation(self) -> TrainingPresentation:
        summary = self._derive_training_outcome()
        context = self._get_launch_context()
        recovery = self._current_recovery_guidance()
        return build_training_run_presentation(
            job_status=self.job.status if self.job else "",
            quality_status=summary.get("quality_status"),
            quality_summary=summary.get("quality_summary"),
            recovery_status=str(recovery.get("status") or ""),
            recovery_action=summary.get("recovery_action"),
            recovery_summary=str(recovery.get("evidence_summary") or summary.get("quality_summary") or ""),
            failure_reason=summary.get("failure_reason"),
            final_reason=summary.get("final_reason"),
            has_launch_context=context is not None,
            can_resume_latest=bool(
                self.job
                and self.job.type in CYCLE_BASED_JOB_TYPES
                and context
                and context.relaunch_capabilities.get("can_resume_latest", False)
            ),
            weights_updated=(summary.get("weights_updated") == "yes"),
        )

    def _render_secondary_monitor_actions(self, actions: list[TrainingAction]) -> None:
        if not actions:
            return
        with ui.row().classes(
            f'items-center gap-1 px-2 py-1 rounded-lg bg-[{COLORS["bg_secondary"]}] border border-[#2d343c]'
        ):
            for action in actions:
                ui.button(
                    icon=action.icon,
                    on_click=lambda a=action: self._trigger_monitor_action(a),
                ).props("flat round dense").classes(
                    f'text-[{self._tone_color(action.tone)}]'
                ).tooltip(action.label)

    def _trigger_monitor_action(self, action: TrainingAction) -> None:
        if action.id == "guided_fix":
            self._open_recovery_review_dialog()
        elif action.id == "review_details":
            self._open_quality_review_dialog()
        elif action.id == "run_again":
            asyncio.create_task(self._rerun_job())
        elif action.id == "resume_latest":
            asyncio.create_task(self._resume_latest_job())
        elif action.id == "edit_config":
            if self._get_launch_context_path():
                self._clone_to_form()
            else:
                ui.navigate.to(self._recovery_route())
    
    def render(self):
        """Render the monitor page."""
        with ui.column().classes('page-content w-full gap-6 p-6'):
            if not self.job:
                self._render_no_job()
                return
            
            # Header with job info
            with ui.row().classes(
                'w-full items-center justify-between animate-in'
            ):
                with ui.column().classes('gap-1'):
                    ui.label(self.job.name).classes(
                        f'text-2xl font-bold text-[{COLORS["text_primary"]}]'
                    )
                    with ui.row().classes('items-center gap-3'):
                        # Status badge
                        self._render_status_badge()
                        if self.job.type in (QUALIFICATION_JOB_TYPES | BOOTSTRAP_JOB_TYPES | LIVE_PROBE_JOB_TYPES):
                            ui.label("Advanced Diagnostics").classes(
                                f'text-xs px-2 py-0.5 rounded-full bg-[{COLORS["warning"]}]/20 text-[{COLORS["warning"]}]'
                            )
                        # Duration (stored for live updates)
                        self._duration_label = ui.label(f'Duration: {self.job.duration_str}').classes(
                            f'text-sm text-[{COLORS["text_secondary"]}]'
                        )
                        # Job ID
                        ui.label(f'ID: {self.job.id}').classes(
                            f'text-sm text-[{COLORS["text_muted"]}] font-mono'
                        )
                
                # Controls
                with ui.row().classes('items-center gap-2'):
                    if self.job.status == 'running':
                        ui.button('Stop', icon='stop', on_click=self._stop_job).props(
                            'flat'
                        ).classes(f'text-[{COLORS["error"]}]')
                    else:
                        context = self._get_launch_context()
                        context_path = self._get_launch_context_path()
                        if context:
                            presentation = self._current_training_presentation()
                            if presentation.primary_action:
                                ui.button(
                                    presentation.primary_action.label,
                                    icon=presentation.primary_action.icon,
                                    on_click=lambda a=presentation.primary_action: self._trigger_monitor_action(a),
                                ).props("unelevated").classes(
                                    f'bg-[{self._tone_color(presentation.primary_action.tone)}] text-white'
                                )
                            self._render_secondary_monitor_actions(presentation.secondary_actions)
                        elif context_path:
                            ui.label("launch context unavailable (invalid JSON)").classes(
                                f'text-xs text-[{COLORS["warning"]}]'
                            )

                    ui.button(icon='refresh', on_click=self._refresh).props(
                        'flat round'
                    ).classes(f'text-[{COLORS["text_secondary"]}]')
                    if self.job.output_dir:
                        ui.button(
                            icon='folder',
                            on_click=self._copy_artifacts_path,
                        ).props('flat round').classes(
                            f'text-[{COLORS["text_secondary"]}]'
                        ).tooltip(str(self.job.output_dir))

            if self.job.type in TRAINING_JOB_TYPES:
                self._render_training_decision_card()
            
            # Progress section
            with ui.column().classes(
                f'w-full gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                f'border border-[#2d343c] animate-in stagger-1'
            ):
                self._render_progress()
            
            # Main content grid
            with ui.row().classes('w-full gap-6 flex-wrap'):
                # Loss chart
                with ui.column().classes(
                    f'flex-[2] min-w-[400px] gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                    f'border border-[#2d343c] animate-in stagger-2'
                ):
                    self._render_chart_section()
                
                # Metrics panel
                with ui.column().classes(
                    f'flex-1 min-w-[250px] gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                    f'border border-[#2d343c] animate-in stagger-3'
                ):
                    self._render_metrics_panel()
            
            # Log viewer
            with ui.column().classes(
                f'w-full gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                f'border border-[#2d343c] animate-in stagger-4'
            ):
                self._render_log_viewer()
            
            # Start live updates if job is running
            if self.job and self.job.status == 'running':
                self._start_live_updates()
                # Timer for duration updates (duration changes every second)
                self._duration_timer = ui.timer(1.0, self._tick_duration)
            
            # Register cleanup on client disconnect (handles navigation away)
            ui.context.client.on_disconnect(self._cleanup_subscriptions)
    
    def _render_no_job(self):
        """Render when no job is selected."""
        with ui.column().classes(
            'w-full items-center justify-center py-16 gap-4 animate-in'
        ):
            ui.icon('hourglass_empty', size='64px').classes(
                f'text-[{COLORS["text_muted"]}]'
            )
            ui.label('No job selected').classes(
                f'text-xl text-[{COLORS["text_secondary"]}]'
            )
            ui.label('Start a training run or select a job from the dashboard.').classes(
                f'text-sm text-[{COLORS["text_muted"]}]'
            )
            ui.button('Go to Dashboard', on_click=lambda: ui.navigate.to('/')).props(
                'flat'
            ).classes(f'text-[{COLORS["accent"]}]')
    
    def _render_status_badge(self):
        """Render the job status badge."""
        status = self.job.status
        color = COLORS.get(status, COLORS['text_secondary'])
        
        with ui.row().classes(
            f'items-center gap-1.5 px-2.5 py-1 rounded-full bg-[{color}]/10'
        ):
            # Animated dot for running
            if status == 'running':
                ui.element('div').classes(
                    f'w-2 h-2 rounded-full bg-[{color}] running-glow'
                )
            else:
                ui.element('div').classes(
                    f'w-2 h-2 rounded-full bg-[{color}]'
                )
            self._status_label = ui.label(status.capitalize()).classes(
                f'text-xs font-medium text-[{color}]'
            )

    def _recovery_route(self) -> str:
        """Return best-effort route to fix current job launch inputs."""
        if not self.job:
            return "/training"
        job_type = str(self.job.type or "").strip().lower()
        if job_type in TRAINING_FIX_ROUTES:
            return TRAINING_FIX_ROUTES[job_type]
        if job_type == "benchmark":
            return "/benchmark?ui_mode=quickstart"
        if job_type == "inference":
            return "/inference?mode=optimize&ui_mode=quickstart&preset=optimize_int4_smoke"
        if job_type in UTILITY_JOB_TYPES:
            return f"/ops-console?module={job_type}&execution_mode=contract"
        return "/training"

    def _failure_recovery_message(self) -> str:
        """Derive concise recovery summary for failed/stopped jobs."""
        if not self.job:
            return "Run did not complete. Review inputs and retry."
        if self.job.error_message:
            return self.job.error_message
        if self.job.status == "stopped":
            return "Run was stopped before completion."
        summary = self._derive_training_outcome()
        failure_reason = summary.get("failure_reason")
        if failure_reason and failure_reason != "--":
            return failure_reason
        return "Run did not complete. Review inputs and retry."

    def _render_training_decision_card(self) -> None:
        """Render a layered summary card with the recommended next step."""
        if not self.job:
            return
        summary = self._derive_training_outcome()
        presentation = self._current_training_presentation()
        tone_color = self._tone_color(presentation.confidence_tone)
        with ui.column().classes(
            f'w-full gap-3 p-4 rounded-xl bg-[{COLORS["bg_card"]}] border border-[{tone_color}]/35 animate-in'
        ):
            with ui.row().classes("w-full items-start justify-between gap-4 flex-wrap"):
                with ui.column().classes("gap-1"):
                    ui.label("Training Decision").classes(
                        f'text-sm font-semibold text-[{tone_color}]'
                    )
                    ui.label(presentation.headline_status).classes(
                        f'text-base font-semibold text-[{COLORS["text_primary"]}]'
                    )
                    ui.label(presentation.supporting_summary).classes(
                        f'text-sm text-[{COLORS["text_secondary"]}]'
                    )
                if self.job.status == "running":
                    ui.label(summary.get("quality_action", "--")).classes(
                        f'px-2 py-1 rounded text-[11px] uppercase tracking-wider bg-[{COLORS["bg_secondary"]}] text-[{tone_color}]'
                    )
            with ui.row().classes("w-full gap-3 flex-wrap"):
                for label, value in (
                    ("Quality", summary.get("quality_status", "--")),
                    ("Keep Rate", summary.get("quality_keep_rate", "--")),
                    ("Top Issue", summary.get("quality_drop_reason", "--")),
                    (
                        "Recommended Next Step",
                        presentation.primary_action.label if presentation.primary_action else summary.get("quality_action", "--"),
                    ),
                ):
                    with ui.column().classes(
                        f'flex-1 min-w-[160px] gap-1 p-3 rounded-lg bg-[{COLORS["bg_secondary"]}] border border-[#2d343c]'
                    ):
                        ui.label(label).classes(
                            f'text-[11px] uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                        )
                        ui.label(str(value or "--")).classes(
                            f'text-xs text-[{COLORS["text_primary"]}]'
                        )

    def _render_failure_recovery_panel(self) -> None:
        """Render concise, actionable recovery controls for failed starts."""
        context_exists = self._get_launch_context_path() is not None
        fix_route = self._recovery_route()
        message = self._failure_recovery_message()

        with ui.column().classes(
            f'w-full gap-3 p-4 rounded-xl bg-[{COLORS["error"]}]/10 border border-[{COLORS["error"]}]/30'
        ):
            ui.label("Recovery Actions").classes(
                f'text-sm font-semibold text-[{COLORS["error"]}]'
            )
            ui.label(message).classes(
                f'text-xs text-[{COLORS["text_secondary"]}]'
            )
            with ui.row().classes("w-full gap-2 flex-wrap"):
                recovery = self._current_recovery_guidance()
                if recovery.get("status") == "ready":
                    ui.button(
                        "Apply Suggested Fix",
                        icon="auto_fix_high",
                        on_click=lambda: self._open_recovery_review_dialog(),
                    ).props("dense unelevated").classes(
                        f'bg-[{COLORS["success"]}] text-white'
                    )
                ui.button(
                    "Fix input",
                    icon="edit",
                    on_click=lambda: ui.navigate.to(fix_route),
                ).props("dense unelevated").classes(
                    f'bg-[{COLORS["primary"]}] text-white'
                )
                reopen_btn = ui.button(
                    "Re-open launch form",
                    icon="open_in_new",
                    on_click=self._clone_to_form,
                ).props("dense flat").classes(f'text-[{COLORS["text_secondary"]}]')
                retry_btn = ui.button(
                    "Retry with same config",
                    icon="replay",
                    on_click=lambda: asyncio.create_task(self._rerun_job()),
                ).props("dense flat").classes(f'text-[{COLORS["text_secondary"]}]')
                if not context_exists:
                    reopen_btn.disable()
                    retry_btn.disable()

            with ui.expansion(
                text="Technical details",
                icon="terminal",
                value=False,
            ).classes(
                f'w-full rounded-lg bg-[{COLORS["bg_card"]}] border border-[#2d343c]'
            ).props('dense dark'):
                ui.label(self._failure_recovery_message()).classes(
                    f'text-xs font-mono text-[{COLORS["text_muted"]}] p-2 break-all'
                )
    
    def _render_progress(self):
        """Render the progress section."""
        progress_text, counter_text = self._resolve_progress_display()
        with ui.row().classes('w-full items-center justify-between'):
            ui.label('Progress').classes(
                f'text-sm font-semibold text-[{COLORS["text_primary"]}]'
            )
            # Store reference for live updates
            self._progress_percent_label = ui.label(progress_text).classes(
                f'text-sm font-mono text-[{COLORS["accent"]}]'
            )
        
        # Progress bar (store reference)
        with ui.element('div').classes(
            f'w-full h-2 rounded-full bg-[{COLORS["bg_secondary"]}] overflow-hidden'
        ):
            initial_width = "0%"
            if self.job and self.job.type not in INDETERMINATE_PROGRESS_JOB_TYPES:
                initial_width = f"{self.job.progress_percent}%"
            elif self.job and self.job.type == "benchmark" and "/" in counter_text:
                try:
                    current, total = counter_text.split("/", 1)
                    if total not in {"?", "0"} and int(total) > 0:
                        initial_width = f"{(int(current) / int(total)) * 100.0}%"
                except Exception:
                    initial_width = "0%"
            self._progress_bar = ui.element('div').classes(
                f'h-full bg-[{COLORS["primary"]}] progress-fill rounded-full'
            ).style(f'width: {initial_width}')
        
        # Progress details (store references)
        with ui.row().classes('w-full gap-6 mt-2'):
            if self.job.type == "benchmark":
                with ui.row().classes('items-center gap-2'):
                    ui.label('Evaluated:').classes(
                        f'text-xs text-[{COLORS["text_muted"]}]'
                    )
                    self._step_label = ui.label(counter_text).classes(
                        f'text-sm font-mono text-[{COLORS["text_secondary"]}]'
                    )
                return
            if self.job.type in INDETERMINATE_PROGRESS_JOB_TYPES:
                with ui.row().classes('items-center gap-2'):
                    ui.label('Progress:').classes(
                        f'text-xs text-[{COLORS["text_muted"]}]'
                    )
                    self._step_label = ui.label(counter_text).classes(
                        f'text-sm font-mono text-[{COLORS["text_secondary"]}]'
                    )
                return

            # Epoch/Cycle
            with ui.row().classes('items-center gap-2'):
                is_cycle_job = self.job.type in CYCLE_BASED_JOB_TYPES
                epoch_label = 'Cycle' if is_cycle_job else 'Epoch'
                if is_cycle_job:
                    current = self.job.current_cycle
                else:
                    # Display epoch as float if fractional, int otherwise
                    current = self.job.current_epoch
                    if isinstance(current, float) and current % 1 != 0:
                        current = f'{current:.1f}'
                    else:
                        current = int(current)
                total = self.job.total_cycles if is_cycle_job else self.job.total_epochs
                ui.label(f'{epoch_label}:').classes(
                    f'text-xs text-[{COLORS["text_muted"]}]'
                )
                self._epoch_label = ui.label(f'{current}/{total}').classes(
                    f'text-sm font-mono text-[{COLORS["text_secondary"]}]'
                )
            
            # Step
            with ui.row().classes('items-center gap-2'):
                ui.label('Step:').classes(
                    f'text-xs text-[{COLORS["text_muted"]}]'
                )
                self._step_label = ui.label(f'{self.job.current_step}/{self.job.total_steps or "?"}').classes(
                    f'text-sm font-mono text-[{COLORS["text_secondary"]}]'
                )
    
    def _render_chart_section(self):
        """Render the loss chart section."""
        if self.job and self.job.type not in TRAINING_JOB_TYPES:
            title = "Benchmark Timeline" if self.job.type == "benchmark" else "Run Timeline"
            ui.label(title).classes(
                f'text-base font-semibold text-[{COLORS["text_primary"]}]'
            )
            with ui.column().classes(
                f'w-full h-64 items-center justify-center gap-3 rounded-lg '
                f'bg-[{COLORS["bg_primary"]}] border border-[#2d343c]'
            ):
                ui.icon("timeline", size="28px").classes(f'text-[{COLORS["text_muted"]}]')
                ui.label("This run does not emit training-loss curves.").classes(
                    f'text-sm text-[{COLORS["text_secondary"]}]'
                )
                ui.label("Use the metrics panel and logs for live execution status.").classes(
                    f'text-xs text-[{COLORS["text_muted"]}]'
                )
            return

        with ui.row().classes('w-full items-center justify-between'):
            ui.label('Training Loss').classes(
                f'text-base font-semibold text-[{COLORS["text_primary"]}]'
            )
            
            with ui.row().classes('gap-2'):
                ui.button('Loss', on_click=lambda: None).props(
                    'flat dense size=sm'
                ).classes(f'text-[{COLORS["primary"]}] bg-[{COLORS["primary"]}]/10')
                ui.button('LR', on_click=lambda: None).props(
                    'flat dense size=sm'
                ).classes(f'text-[{COLORS["text_muted"]}]')
        
        # Chart container
        self.chart = ui.echart({
            'backgroundColor': 'transparent',
            'grid': {
                'top': 30,
                'right': 20,
                'bottom': 30,
                'left': 50,
            },
            'xAxis': {
                'type': 'value',
                'name': 'Step',
                'nameLocation': 'middle',
                'nameGap': 25,
                'axisLine': {'lineStyle': {'color': COLORS['text_muted']}},
                'axisLabel': {'color': COLORS['text_muted']},
            },
            'yAxis': {
                'type': 'value',
                'name': 'Loss',
                'axisLine': {'lineStyle': {'color': COLORS['text_muted']}},
                'axisLabel': {'color': COLORS['text_muted']},
                'splitLine': {'lineStyle': {'color': '#2d343c'}},
            },
            'tooltip': {
                'trigger': 'axis',
                'backgroundColor': COLORS['bg_card'],
                'borderColor': '#2d343c',
                'textStyle': {'color': COLORS['text_primary']},
            },
            'series': [{
                'type': 'line',
                'smooth': True,
                'symbol': 'none',
                'lineStyle': {
                    'width': 2,
                    'color': COLORS['primary'],
                },
                'areaStyle': {
                    'opacity': 0.1,
                    'color': COLORS['primary'],
                },
                'data': self._get_loss_data(),
            }],
        }).classes('w-full h-64')
    
    def _get_loss_data(self) -> list:
        """Get loss data for the chart."""
        if not self.job_id or self.job_id not in state.metrics_history:
            return []
        
        loss_points = state.metrics_history[self.job_id].get('loss', [])
        return [[p.step, p.value] for p in loss_points]
    
    def _render_metrics_panel(self):
        """Render the current metrics panel with stored references for live updates."""
        if self.job and self.job.type == "benchmark":
            self._render_benchmark_metrics_panel()
            return
        if self.job and self.job.type == "inference":
            self._render_inference_metrics_panel()
            return
        if self.job and self.job.type in UTILITY_JOB_TYPES:
            self._render_utility_metrics_panel()
            return
        if self.job and self.job.type in DIAGNOSTIC_JOB_TYPES:
            self._render_diagnostics_metrics_panel()
            return

        ui.label('Training Metrics').classes(
            f'text-base font-semibold text-[{COLORS["text_primary"]}]'
        )
        summary = self._derive_training_outcome()
        presentation = self._current_training_presentation()
        tone_color = self._tone_color(presentation.confidence_tone)
        with ui.column().classes(
            f'w-full gap-2 p-3 rounded-lg bg-[{COLORS["bg_secondary"]}] border border-[{tone_color}]/35'
        ):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                self._quality_headline_label = ui.label(presentation.headline_status).classes(
                    f'text-sm font-semibold text-[{COLORS["text_primary"]}]'
                )
                self._quality_status_label = ui.label(summary["quality_status"]).classes(
                    f'px-2 py-1 rounded text-[11px] uppercase tracking-wider bg-[{COLORS["bg_card"]}] text-[{tone_color}]'
                )
            self._quality_summary_note_label = ui.label(presentation.supporting_summary).classes(
                f'text-xs text-[{COLORS["text_secondary"]}]'
            )
            with ui.row().classes("w-full gap-3 flex-wrap"):
                for label, value, attr, color in (
                    ("Keep Rate", summary["quality_keep_rate"], "_quality_keep_rate_label", COLORS["text_primary"]),
                    ("Top Issue", summary["quality_drop_reason"], "_quality_drop_reason_label", COLORS["text_primary"]),
                    ("Recommended Next Step", summary["quality_action"], "_quality_action_label", COLORS["accent"]),
                    ("Recovery", summary["recovery_action"], "_recovery_action_label", COLORS["success"]),
                ):
                    with ui.column().classes("flex-1 min-w-[120px] gap-1"):
                        ui.label(label).classes(
                            f'text-[11px] uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                        )
                        setattr(
                            self,
                            attr,
                            ui.label(value).classes(
                                f'text-sm font-mono text-[{color}]'
                            ),
                        )

        ui.label('Detailed Metrics').classes(
            f'text-sm font-semibold text-[{COLORS["text_secondary"]}] mt-1'
        )

        with ui.column().classes('w-full gap-3 mt-2'):
            # Loss
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Loss').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                val = f'{self.job.latest_loss:.4f}' if self.job.latest_loss is not None else '--'
                self._loss_label = ui.label(val).classes(
                    f'text-sm font-mono text-[{COLORS["text_primary"]}]'
                )
            
            # Learning Rate
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Learning Rate').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                val = f'{self.job.latest_lr:.2e}' if self.job.latest_lr is not None else '--'
                self._lr_label = ui.label(val).classes(
                    f'text-sm font-mono text-[{COLORS["text_primary"]}]'
                )
            
            # Grad Norm
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Grad Norm').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                val = f'{self.job.latest_grad_norm:.4f}' if self.job.latest_grad_norm is not None else '--'
                self._grad_norm_label = ui.label(val).classes(
                    f'text-sm font-mono text-[{COLORS["text_primary"]}]'
                )
            
            # Verification (available when parser captured it)
            if self.job.verification_rate is not None:
                ui.separator().classes('my-2')
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label('Verification').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                    val = f'{self.job.verification_rate:.1%}' if self.job.verification_rate is not None else '--'
                    self._verification_label = ui.label(val).classes(
                        f'text-sm font-mono text-[{COLORS["text_primary"]}]'
                    )

            ui.separator().classes('my-2')
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Weights Updated').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                self._weights_updated_label = ui.label(summary["weights_updated"]).classes(
                    f'text-sm font-mono text-[{COLORS["text_primary"]}]'
                )
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Update Steps').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                self._update_steps_label = ui.label(summary["update_steps"]).classes(
                    f'text-sm font-mono text-[{COLORS["text_primary"]}]'
                )
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Final Loss').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                self._final_loss_label = ui.label(summary["final_loss"]).classes(
                    f'text-sm font-mono text-[{COLORS["text_primary"]}]'
                )
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Final Reason').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                self._update_reason_label = ui.label(summary["final_reason"]).classes(
                    f'text-sm font-mono text-[{COLORS["text_primary"]}]'
                )
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Failure Reason').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                self._failure_reason_label = ui.label(summary["failure_reason"]).classes(
                    f'text-sm font-mono text-[{COLORS["text_primary"]}]'
                )
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Run ID').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                self._run_id_label = ui.label(summary["run_id"]).classes(
                    f'text-sm font-mono text-[{COLORS["text_primary"]}]'
                )
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Seed').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                self._seed_label = ui.label(summary["seed"]).classes(
                    f'text-sm font-mono text-[{COLORS["text_primary"]}]'
                )
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Resume').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                self._resume_label = ui.label(summary["resume"]).classes(
                    f'text-sm font-mono text-[{COLORS["text_primary"]}]'
                )
            if summary["quality_summary"] != "--":
                ui.label(summary["quality_summary"]).classes(
                    f'text-xs text-[{COLORS["text_muted"]}]'
                )

    def _render_inference_metrics_panel(self) -> None:
        """Render inference-specific metrics summary."""
        ui.label('Inference Metrics').classes(
            f'text-base font-semibold text-[{COLORS["text_primary"]}]'
        )
        summary = self._derive_inference_outcome()
        self._inference_metric_labels = {}
        with ui.column().classes('w-full gap-3 mt-2'):
            for label, key in (
                ("Status", "status"),
                ("Mode", "mode"),
                ("Model", "model"),
                ("Output", "output_dir"),
                ("Target", "target"),
            ):
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label(label).classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                    value_label = ui.label(summary[key]).classes(
                        f'text-sm font-mono text-[{COLORS["text_primary"]}] break-all text-right'
                    )
                    self._inference_metric_labels[key] = value_label

    def _render_utility_metrics_panel(self) -> None:
        """Render utility run metrics summary."""
        ui.label('Utility Run Metrics').classes(
            f'text-base font-semibold text-[{COLORS["text_primary"]}]'
        )
        summary = self._derive_utility_outcome()
        self._utility_metric_labels = {}
        with ui.column().classes('w-full gap-3 mt-2'):
            for label, key in (
                ("Status", "status"),
                ("Module", "module"),
                ("Execution Mode", "execution_mode"),
                ("Output", "output_dir"),
                ("Command", "command"),
            ):
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label(label).classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                    value_label = ui.label(summary[key]).classes(
                        f'text-sm font-mono text-[{COLORS["text_primary"]}] break-all text-right'
                    )
                    self._utility_metric_labels[key] = value_label

    def _render_qualification_metrics_panel(self) -> None:
        """Render qualification-specific metrics summary."""
        ui.label('Qualification Metrics').classes(
            f'text-base font-semibold text-[{COLORS["text_primary"]}]'
        )
        summary = self._derive_qualification_outcome()
        with ui.column().classes('w-full gap-3 mt-2'):
            for label, key in (
                ("Status", "status"),
                ("Pass Count", "pass_count"),
                ("Warn Count", "warn_count"),
                ("Fail Count", "fail_count"),
                ("Top Issue", "top_issue"),
                ("Fix Now", "fix_now"),
                ("Profile", "profile"),
            ):
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label(label).classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                    ui.label(summary[key]).classes(
                        f'text-sm font-mono text-[{COLORS["text_primary"]}]'
                    )

    def _render_benchmark_metrics_panel(self) -> None:
        """Render benchmark-specific metrics summary."""
        ui.label('Benchmark Metrics').classes(
            f'text-base font-semibold text-[{COLORS["text_primary"]}]'
        )
        summary = self._derive_benchmark_outcome()
        self._benchmark_metric_labels = {}
        with ui.column().classes('w-full gap-3 mt-2'):
            for label, key in (
                ("Status", "status"),
                ("Evaluated", "evaluated"),
                ("Pass@1", "pass_at_1"),
                ("Pass@5", "pass_at_5"),
                ("Pass@10", "pass_at_10"),
                ("Pass Rate", "pass_rate"),
                ("Output", "output_file"),
            ):
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label(label).classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                    value_label = ui.label(summary[key]).classes(
                        f'text-sm font-mono text-[{COLORS["text_primary"]}] break-all text-right'
                    )
                    self._benchmark_metric_labels[key] = value_label

    def _render_diagnostics_metrics_panel(self) -> None:
        """Render diagnostics-specific metrics summary for qualification/bootstrap/live jobs."""
        self._diagnostic_metric_labels = {}
        if self.job and self.job.type in QUALIFICATION_JOB_TYPES:
            title = "Setup Check Metrics"
            summary = self._derive_qualification_outcome()
            fields = (
                ("Status", "status"),
                ("Pass Count", "pass_count"),
                ("Warn Count", "warn_count"),
                ("Fail Count", "fail_count"),
                ("Top Issue", "top_issue"),
                ("Profile", "profile"),
            )
        elif self.job and self.job.type in BOOTSTRAP_JOB_TYPES:
            title = "Setup Files Metrics"
            summary = self._derive_bootstrap_outcome()
            fields = (
                ("Status", "status"),
                ("Pass Count", "pass_count"),
                ("Warn Count", "warn_count"),
                ("Fail Count", "fail_count"),
                ("Top Issue", "top_issue"),
                ("Profile", "profile"),
            )
        else:
            title = "System Health Metrics"
            summary = self._derive_live_probe_outcome()
            fields = (
                ("Status", "status"),
                ("Pass Count", "pass_count"),
                ("Warn Count", "warn_count"),
                ("Fail Count", "fail_count"),
                ("Top Issue", "top_issue"),
                ("Profile", "profile"),
            )

        ui.label(title).classes(
            f'text-base font-semibold text-[{COLORS["text_primary"]}]'
        )
        with ui.column().classes('w-full gap-3 mt-2'):
            for label, key in fields:
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label(label).classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                    value_label = ui.label(summary[key]).classes(
                        f'text-sm font-mono text-[{COLORS["text_primary"]}] break-all text-right'
                    )
                    self._diagnostic_metric_labels[key] = value_label

    def _render_live_probe_metrics_panel(self) -> None:
        """Render live-probe-specific metrics summary."""
        ui.label('Live Probe Metrics').classes(
            f'text-base font-semibold text-[{COLORS["text_primary"]}]'
        )
        summary = self._derive_live_probe_outcome()
        with ui.column().classes('w-full gap-3 mt-2'):
            for label, key in (
                ("Status", "status"),
                ("Pass Count", "pass_count"),
                ("Warn Count", "warn_count"),
                ("Fail Count", "fail_count"),
                ("Top Issue", "top_issue"),
                ("Profile", "profile"),
            ):
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label(label).classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                    ui.label(summary[key]).classes(
                        f'text-sm font-mono text-[{COLORS["text_primary"]}]'
                    )

    def _read_training_summary_payload(self) -> Optional[Dict[str, Any]]:
        """Load canonical training summary payload if present."""
        if not self.job or not self.job.output_dir:
            return None
        output_dir = Path(self.job.output_dir)
        for filename in ("training_summary.json", "training_metrics.json"):
            candidate = output_dir / filename
            if not candidate.exists():
                continue
            try:
                with open(candidate, encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def _derive_training_outcome(self) -> Dict[str, str]:
        """Compute display-safe training outcome fields from summary payloads."""
        payload = self._read_training_summary_payload()
        live_yield = (
            self.job.latest_yield_snapshot
            if self.job and isinstance(self.job.latest_yield_snapshot, dict)
            else None
        )
        if not payload:
            recovery = self._current_recovery_guidance(yield_diagnostics=live_yield or {})
            return {
                "weights_updated": "--",
                "update_steps": "--",
                "final_loss": "--",
                "final_reason": "--",
                "failure_reason": "--",
                "run_id": "--",
                "seed": "--",
                "resume": "--",
                "quality_status": str(
                    ((live_yield or {}).get("summary") or {}).get("status") or "--"
                ),
                "quality_keep_rate": self._format_percent(
                    ((live_yield or {}).get("rates") or {}).get("keep_rate")
                ),
                "quality_drop_reason": str(
                    ((live_yield or {}).get("summary") or {}).get("dominant_rejection_reason") or "--"
                ).replace("_", " "),
                "quality_action": self._yield_action_hint(live_yield),
                "quality_summary": str(
                    ((live_yield or {}).get("summary") or {}).get("text") or "--"
                ),
                "recovery_action": str(recovery.get("recommended_action") or "--"),
            }

        total_steps = int(payload.get("total_train_steps_executed", 0) or 0)
        weights_updated = payload.get("weights_updated")
        final_loss = payload.get("final_train_loss")
        final_reason = payload.get("final_update_reason")
        run_id = payload.get("run_id")
        seed = payload.get("seed")
        failure_reason = payload.get("failure_reason")
        resume_cycle = payload.get("resume_from_cycle")
        resumed_checkpoint = payload.get("resumed_from_checkpoint")
        yield_diagnostics = (
            payload.get("yield_diagnostics")
            if isinstance(payload.get("yield_diagnostics"), dict)
            else live_yield or {}
        )
        yield_summary = (
            yield_diagnostics.get("summary")
            if isinstance(yield_diagnostics.get("summary"), dict)
            else {}
        )
        yield_rates = (
            yield_diagnostics.get("rates")
            if isinstance(yield_diagnostics.get("rates"), dict)
            else {}
        )
        recovery = self._current_recovery_guidance(payload=payload, yield_diagnostics=yield_diagnostics)

        cycle_entries = payload.get("cycles") or payload.get("cycle_results") or []
        if isinstance(cycle_entries, list):
            if weights_updated is None:
                weights_updated = False
                for entry in cycle_entries:
                    if not isinstance(entry, dict):
                        continue
                    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else entry
                    if metrics.get("weights_updated") is True:
                        weights_updated = True
                        break
            if total_steps == 0:
                total_steps = sum(
                    int(
                        (
                            entry.get("metrics", entry)
                            if isinstance(entry, dict)
                            else {}
                        ).get("train_steps_executed", 0)
                    )
                    for entry in cycle_entries
                    if isinstance(entry, dict)
                )
            if final_loss is None and cycle_entries:
                last = cycle_entries[-1]
                if isinstance(last, dict):
                    metrics = last.get("metrics") if isinstance(last.get("metrics"), dict) else last
                    final_loss = metrics.get("train_loss")
                    if not final_reason:
                        final_reason = metrics.get("update_reason")

        if weights_updated is True:
            weights_label = "yes"
        elif weights_updated is False:
            weights_label = "no"
        else:
            weights_label = "--"

        if isinstance(final_loss, (int, float)):
            final_loss_label = f"{float(final_loss):.4f}"
        else:
            final_loss_label = "--"

        return {
            "weights_updated": weights_label,
            "update_steps": str(total_steps),
            "final_loss": final_loss_label,
            "final_reason": str(final_reason or "--"),
            "failure_reason": str(failure_reason or "--"),
            "run_id": str(run_id or "--"),
            "seed": str(seed if seed is not None else "--"),
            "resume": (
                f"cycle {resume_cycle} ({resumed_checkpoint.get('cycle_dir', 'checkpoint')})"
                if isinstance(resumed_checkpoint, dict) and (resume_cycle or 0) > 0
                else (f"cycle {resume_cycle}" if (resume_cycle or 0) > 0 else "none")
            ),
            "quality_status": str(yield_summary.get("status") or "--"),
            "quality_keep_rate": self._format_percent(yield_rates.get("keep_rate")),
            "quality_drop_reason": str(
                yield_summary.get("dominant_rejection_reason") or "--"
            ).replace("_", " "),
            "quality_action": self._yield_action_hint(yield_diagnostics),
            "quality_summary": str(yield_summary.get("text") or "--"),
            "recovery_action": str(recovery.get("recommended_action") or "--"),
        }

    def _format_percent(self, value: Any) -> str:
        try:
            return f"{float(value):.0%}"
        except (TypeError, ValueError):
            return "--"

    def _yield_action_hint(self, diagnostics: Optional[Dict[str, Any]]) -> str:
        if not isinstance(diagnostics, dict):
            return "--"
        summary = diagnostics.get("summary") if isinstance(diagnostics.get("summary"), dict) else {}
        reason = str(summary.get("dominant_rejection_reason") or "").strip().lower()
        if reason == "below_reward_threshold":
            return "Lower threshold"
        if reason == "dropped_by_keep_percent":
            return "Increase keep percent"
        if reason in {"missing_text", "empty_target"}:
            return "Inspect dataset formatting"
        if reason == "verification_failed":
            return "Inspect verifier failures"
        status = str(summary.get("status") or "").strip().lower()
        if status in {"low_yield", "no_signal"}:
            return "Increase sample budget"
        return "Settings look balanced"

    def _current_recovery_guidance(
        self,
        *,
        payload: Optional[Dict[str, Any]] = None,
        yield_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        summary_payload = payload or self._read_training_summary_payload() or {}
        if isinstance(summary_payload.get("recovery_guidance"), dict):
            return dict(summary_payload.get("recovery_guidance") or {})
        launch_args = {}
        context = self._get_launch_context()
        if context:
            launch_args = dict(context.args)
        elif self.job and isinstance(self.job.launch_args, dict):
            launch_args = dict(self.job.launch_args)
        return build_recovery_guidance(
            modality=str((summary_payload.get("modality") if isinstance(summary_payload, dict) else None) or (self.job.type if self.job else "")).strip().lower(),
            yield_diagnostics=yield_diagnostics or (
                summary_payload.get("yield_diagnostics")
                if isinstance(summary_payload.get("yield_diagnostics"), dict)
                else {}
            ),
            effectiveness=(
                summary_payload.get("effectiveness")
                if isinstance(summary_payload.get("effectiveness"), dict)
                else {}
            ),
            launch_args=launch_args,
            representative_examples=[],
        )

    def _preferred_recovery_resume_latest(self) -> bool:
        context = self._get_launch_context()
        return bool(
            self.job
            and self.job.type in CYCLE_BASED_JOB_TYPES
            and context
            and context.relaunch_capabilities.get("can_resume_latest", False)
        )

    def _open_quality_review_dialog(self) -> None:
        summary = self._derive_training_outcome()
        presentation = self._current_training_presentation()
        dialog = ui.dialog()
        with dialog, ui.card().classes(
            f'w-[720px] max-w-[95vw] gap-4 bg-[{COLORS["bg_card"]}] text-[{COLORS["text_primary"]}]'
        ):
            ui.label("Training Quality Review").classes("text-lg font-semibold")
            ui.label(presentation.supporting_summary).classes(
                f'text-sm text-[{COLORS["text_secondary"]}]'
            )
            with ui.row().classes("w-full gap-3 flex-wrap"):
                for label, value in (
                    ("Status", presentation.headline_status),
                    ("Quality", summary.get("quality_status", "--")),
                    ("Keep Rate", summary.get("quality_keep_rate", "--")),
                    ("Top Issue", summary.get("quality_drop_reason", "--")),
                    ("Next Step", presentation.primary_action.label if presentation.primary_action else summary.get("quality_action", "--")),
                ):
                    with ui.column().classes(
                        f'flex-1 min-w-[140px] gap-1 p-3 rounded-lg bg-[{COLORS["bg_secondary"]}] border border-[#2d343c]'
                    ):
                        ui.label(label).classes(
                            f'text-[11px] uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                        )
                        ui.label(str(value or "--")).classes(
                            f'text-sm text-[{COLORS["text_primary"]}]'
                        )
            with ui.row().classes("w-full justify-end"):
                ui.button("Close", on_click=dialog.close).props("flat")
        dialog.open()

    def _open_recovery_review_dialog(self) -> None:
        recovery = self._current_recovery_guidance()
        if recovery.get("status") != "ready":
            return
        dialog = ui.dialog()
        launch_mode = "resume latest" if self._preferred_recovery_resume_latest() else "relaunch"
        with dialog, ui.card().classes(
            f'w-[720px] max-w-[95vw] gap-4 bg-[{COLORS["bg_card"]}] text-[{COLORS["text_primary"]}]'
        ):
            ui.label("Review Suggested Fix").classes("text-lg font-semibold")
            ui.label(str(recovery.get("evidence_summary") or "--")).classes(
                f'text-sm text-[{COLORS["text_secondary"]}]'
            )
            with ui.column().classes(
                f'w-full gap-2 p-3 rounded-lg bg-[{COLORS["bg_secondary"]}] border border-[#2d343c]'
            ):
                ui.label(f"Launch mode: {launch_mode}").classes(f'text-sm text-[{COLORS["text_primary"]}]')
                ui.label(f"Why this fix: {recovery.get('recommended_action') or '--'}").classes(
                    f'text-xs text-[{COLORS["success"]}]'
                )
                for key, value in dict(recovery.get("suggested_overrides") or {}).items():
                    ui.label(f"{key}: {value}").classes(f'text-xs font-mono text-[{COLORS["accent"]}]')
            examples = [dict(example) for example in recovery.get("representative_examples", []) if isinstance(example, dict)]
            if examples:
                with ui.expansion(text="Representative evidence", icon="fact_check", value=False).classes(
                    f'w-full rounded-lg bg-[{COLORS["bg_secondary"]}] border border-[#2d343c]'
                ).props('dense dark'):
                    with ui.column().classes("w-full gap-2 p-3"):
                        for example in examples:
                            ui.label(f"{example.get('label') or example.get('reason')}: {example.get('preview') or '--'}").classes(
                                f'text-xs text-[{COLORS["text_secondary"]}]'
                            )
                            if example.get("context"):
                                ui.label(str(example.get("context"))).classes(
                                    f'text-[11px] text-[{COLORS["text_muted"]}]'
                                )
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(
                    "Launch with suggested fix",
                    icon="play_arrow",
                    on_click=lambda: asyncio.create_task(
                        self._apply_recovery_guidance(dialog, recovery)
                    ),
                ).props("unelevated").classes(
                    f'bg-[{COLORS["primary"]}] text-white'
                )
        dialog.open()

    async def _apply_recovery_guidance(self, dialog, recovery: Dict[str, Any]) -> None:
        if not self.job:
            dialog.close()
            return
        context_path = self._get_launch_context_path()
        if not context_path:
            notify_job_failed(self.job.name, "No launch context found for guided recovery")
            dialog.close()
            return
        try:
            new_job_id = await self.training_service.relaunch_from_context(
                context_path,
                origin_job_id=self.job.id,
                resume_latest=self._preferred_recovery_resume_latest(),
                override_args=dict(recovery.get("suggested_overrides") or {}),
                guided_recovery=recovery,
                source_ui_page="/monitor",
            )
            dialog.close()
            notify_job_started(f"Guided recovery: {self.job.type.upper()}")
            ui.navigate.to(f"/monitor/{new_job_id}")
        except Exception as e:
            notify_job_failed(self.job.name, f"Guided recovery failed: {e}")

    def _derive_qualification_outcome(self) -> Dict[str, str]:
        """Compute display-safe qualification summary fields from report payload."""
        payload = self._read_qualification_report_payload()
        if not payload:
            return {
                "status": "--",
                "pass_count": "--",
                "warn_count": "--",
                "fail_count": "--",
                "top_issue": "--",
                "fix_now": "--",
                "profile": "--",
            }

        modules = payload.get("modules") if isinstance(payload.get("modules"), dict) else {}
        pass_count = 0
        warn_count = 0
        fail_count = 0
        top_issue = "--"
        fix_now = "--"
        for _, entry in modules.items():
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status") or "warn").strip().lower()
            if status == "pass":
                pass_count += 1
            elif status == "fail":
                fail_count += 1
            else:
                warn_count += 1
            if top_issue == "--" and entry.get("issue_code"):
                top_issue = str(entry.get("issue_code"))
            if fix_now == "--" and entry.get("fix_now"):
                fix_now = str(entry.get("fix_now"))

        overall_status = "pass"
        if fail_count > 0:
            overall_status = "fail"
        elif warn_count > 0:
            overall_status = "warn"

        return {
            "status": overall_status,
            "pass_count": str(pass_count),
            "warn_count": str(warn_count),
            "fail_count": str(fail_count),
            "top_issue": top_issue,
            "fix_now": fix_now,
            "profile": str(payload.get("profile") or "--"),
        }

    def _read_qualification_report_payload(self) -> Optional[Dict[str, Any]]:
        """Load qualification report payload if present in job output."""
        if not self.job or not self.job.output_dir:
            return None
        candidate = Path(self.job.output_dir) / "all_module_qualification.v1.json"
        if not candidate.exists():
            return None
        try:
            with open(candidate, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _read_live_probe_report_payload(self) -> Optional[Dict[str, Any]]:
        """Load live probe report payload if present in job output."""
        if not self.job or not self.job.output_dir:
            return None
        candidate = Path(self.job.output_dir) / "all_module_live_execution.v1.json"
        if not candidate.exists():
            return None
        try:
            with open(candidate, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _read_benchmark_result_payload(self) -> Optional[Dict[str, Any]]:
        """Load benchmark result payload if present in job output."""
        if not self.job or not self.job.output_dir:
            return None
        candidate = Path(self.job.output_dir) / "benchmark.json"
        if not candidate.exists():
            return None
        try:
            with open(candidate, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _derive_benchmark_outcome(self) -> Dict[str, str]:
        """Compute display-safe benchmark summary fields."""
        payload = self._read_benchmark_result_payload() or {}
        lifecycle = (
            self.job.lifecycle_metadata
            if self.job and isinstance(self.job.lifecycle_metadata, dict)
            else {}
        )

        pass_at_k = payload.get("pass_at_k")
        if isinstance(pass_at_k, dict):
            pass_1 = pass_at_k.get("1", pass_at_k.get(1))
            pass_5 = pass_at_k.get("5", pass_at_k.get(5))
            pass_10 = pass_at_k.get("10", pass_at_k.get(10))
        else:
            pass_1 = payload.get("pass_at_1")
            pass_5 = payload.get("pass_at_5")
            pass_10 = payload.get("pass_at_10")

        if pass_1 is None:
            pass_1 = lifecycle.get("benchmark_pass_at_1")
        if pass_5 is None:
            pass_5 = lifecycle.get("benchmark_pass_at_5")
        if pass_10 is None:
            pass_10 = lifecycle.get("benchmark_pass_at_10")
        if pass_1 is None and self.job and self.job.verification_rate is not None:
            pass_1 = self.job.verification_rate

        pass_rate = payload.get("pass_rate")
        if pass_rate is None:
            pass_rate = lifecycle.get("benchmark_pass_rate")
        if pass_rate is None and pass_1 is not None:
            pass_rate = pass_1

        current, total = self._resolve_benchmark_progress_counts()

        output_file = "--"
        if self.job and self.job.output_dir:
            output_file = str(Path(self.job.output_dir) / "benchmark.json")

        def _fmt(value):
            if value is None:
                return "--"
            try:
                parsed = float(value)
                return f"{parsed:.4f}"
            except (TypeError, ValueError):
                return str(value)

        return {
            "status": str(self.job.status if self.job else "--"),
            "evaluated": f"{current}/{total or '?'}",
            "pass_at_1": _fmt(pass_1),
            "pass_at_5": _fmt(pass_5),
            "pass_at_10": _fmt(pass_10),
            "pass_rate": _fmt(pass_rate),
            "output_file": output_file,
        }

    def _derive_inference_outcome(self) -> Dict[str, str]:
        """Compute display-safe inference summary fields."""
        mode = "--"
        model = "--"
        target = "--"
        output_dir = str(self.job.output_dir) if self.job and self.job.output_dir else "--"
        if self.job and isinstance(self.job.launch_args, dict):
            mode = str(self.job.launch_args.get("mode") or "--")
            model = str(self.job.launch_args.get("model") or "--")
            if mode == "optimize":
                precision = str(self.job.launch_args.get("target_precision") or "--")
                latency = self.job.launch_args.get("target_latency")
                target = f"{precision} @ {latency}ms" if latency is not None else precision
            elif mode == "benchmark":
                prompts = self.job.launch_args.get("num_prompts")
                target = f"{prompts} prompts" if prompts is not None else "--"
        return {
            "status": str(self.job.status if self.job else "--"),
            "mode": mode,
            "model": Path(model).name if model not in {"", "--"} else "--",
            "output_dir": output_dir,
            "target": str(target),
        }

    def _derive_utility_outcome(self) -> Dict[str, str]:
        """Compute display-safe utility run summary fields."""
        execution_mode = "--"
        command = "--"
        if self.job and isinstance(self.job.launch_args, dict):
            execution_mode = str(self.job.launch_args.get("execution_mode") or "--")
        launch_context = self._get_launch_context()
        if launch_context and launch_context.command:
            command = " ".join(str(part) for part in launch_context.command[:4])
            if len(launch_context.command) > 4:
                command += " ..."
        return {
            "status": str(self.job.status if self.job else "--"),
            "module": str(self.job.type if self.job else "--"),
            "execution_mode": execution_mode,
            "output_dir": str(self.job.output_dir) if self.job and self.job.output_dir else "--",
            "command": command,
        }

    def _read_bootstrap_report_payload(self) -> Optional[Dict[str, Any]]:
        """Load bootstrap report payload if present in job output."""
        if not self.job or not self.job.output_dir:
            return None
        candidate = Path(self.job.output_dir) / "all_module_bootstrap.v1.json"
        if not candidate.exists():
            return None
        try:
            with open(candidate, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _derive_bootstrap_outcome(self) -> Dict[str, str]:
        """Compute display-safe bootstrap summary fields from report payload."""
        payload = self._read_bootstrap_report_payload()
        if not payload:
            return {
                "status": "--",
                "pass_count": "--",
                "warn_count": "--",
                "fail_count": "--",
                "top_issue": "--",
                "profile": "--",
            }

        modules = payload.get("modules") if isinstance(payload.get("modules"), dict) else {}
        pass_count = 0
        warn_count = 0
        fail_count = 0
        top_issue = "--"
        for _, entry in modules.items():
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status") or "warn").strip().lower()
            if status == "pass":
                pass_count += 1
            elif status == "fail":
                fail_count += 1
                if top_issue == "--" and entry.get("errors"):
                    top_issue = str(entry.get("errors")[0])
            else:
                warn_count += 1
                if top_issue == "--" and entry.get("warnings"):
                    top_issue = str(entry.get("warnings")[0])

        overall_status = "pass"
        if fail_count > 0:
            overall_status = "fail"
        elif warn_count > 0:
            overall_status = "warn"

        return {
            "status": overall_status,
            "pass_count": str(pass_count),
            "warn_count": str(warn_count),
            "fail_count": str(fail_count),
            "top_issue": top_issue,
            "profile": str(payload.get("profile") or "--"),
        }

    def _derive_live_probe_outcome(self) -> Dict[str, str]:
        """Compute display-safe live probe summary fields from report payload."""
        payload = self._read_live_probe_report_payload()
        if not payload:
            return {
                "status": "--",
                "pass_count": "--",
                "warn_count": "--",
                "fail_count": "--",
                "top_issue": "--",
                "profile": "--",
            }

        modules = payload.get("modules") if isinstance(payload.get("modules"), dict) else {}
        pass_count = 0
        warn_count = 0
        fail_count = 0
        top_issue = "--"
        for _, entry in modules.items():
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status") or "warn").strip().lower()
            if status == "pass":
                pass_count += 1
            elif status == "fail":
                fail_count += 1
                if top_issue == "--" and entry.get("errors"):
                    top_issue = str(entry.get("errors")[0])
            else:
                warn_count += 1
                if top_issue == "--" and entry.get("warnings"):
                    top_issue = str(entry.get("warnings")[0])

        overall_status = "pass"
        if fail_count > 0:
            overall_status = "fail"
        elif warn_count > 0:
            overall_status = "warn"

        return {
            "status": overall_status,
            "pass_count": str(pass_count),
            "warn_count": str(warn_count),
            "fail_count": str(fail_count),
            "top_issue": top_issue,
            "profile": str(payload.get("profile") or "--"),
        }
    
    def _render_log_viewer(self):
        """Render the log viewer section."""
        title = "Run Logs"
        if self.job:
            if self.job.type in TRAINING_JOB_TYPES:
                title = "Training Logs"
            elif self.job.type == "benchmark":
                title = "Benchmark Logs"
            elif self.job.type == "inference":
                title = "Inference Logs"
            elif self.job.type in DIAGNOSTIC_JOB_TYPES:
                title = "Diagnostics Logs"
        with ui.row().classes('w-full items-center justify-between'):
            ui.label(title).classes(
                f'text-base font-semibold text-[{COLORS["text_primary"]}]'
            )
            
            with ui.row().classes('gap-2'):
                ui.button(icon='vertical_align_bottom', on_click=self._scroll_to_bottom).props(
                    'flat round dense size=sm'
                ).classes(f'text-[{COLORS["text_muted"]}]').tooltip('Scroll to bottom')
                
                ui.button(icon='content_copy', on_click=self._copy_logs).props(
                    'flat round dense size=sm'
                ).classes(f'text-[{COLORS["text_muted"]}]').tooltip('Copy logs')
                
                ui.button(icon='download', on_click=self._download_logs).props(
                    'flat round dense size=sm'
                ).classes(f'text-[{COLORS["text_muted"]}]').tooltip('Download logs')
        
        # Log container - with data attribute for scroll targeting, increased height
        self.log_container = ui.column().classes(
            f'w-full h-96 overflow-y-auto p-4 rounded-lg bg-[{COLORS["bg_primary"]}] '
            f'font-mono text-xs leading-relaxed'
        ).props('data-log-container')
        
        with self.log_container:
            # Display existing logs (loaded from file or empty)
            if self._all_log_lines:
                for line in self._all_log_lines:
                    color = self._get_log_color(line)
                    ui.label(line).classes(f'text-[{color}]')
                    self._displayed_log_count += 1
            else:
                # Show placeholder if no logs yet
                ui.label('Waiting for logs...').classes(
                    f'text-[{COLORS["text_muted"]}] italic'
                )
    
    async def _refresh(self):
        """Refresh the monitor data."""
        if self.job_id:
            self.job = state.get_job(self.job_id)
            self._update_metrics_display()
            self._update_logs_display()
        ui.notify('Refreshed', type='info', timeout=1000)
    
    def _start_live_updates(self):
        """Start live updates for running job using event subscriptions."""
        if not self.job_id:
            return
        
        bus = get_event_bus()
        
        # Subscribe to metrics updates
        unsub_metrics = bus.subscribe(EventType.METRICS_UPDATE, self._on_metrics_event)
        self._unsubscribe_callbacks.append(unsub_metrics)
        
        # Subscribe to log lines
        unsub_logs = bus.subscribe(EventType.LOG_LINE, self._on_log_event)
        self._unsubscribe_callbacks.append(unsub_logs)
        
        # Subscribe to job completion
        unsub_completed = bus.subscribe(EventType.JOB_COMPLETED, self._on_job_completed)
        self._unsubscribe_callbacks.append(unsub_completed)
        
        # Subscribe to job failed
        unsub_failed = bus.subscribe(EventType.JOB_FAILED, self._on_job_failed)
        self._unsubscribe_callbacks.append(unsub_failed)
        
        # Subscribe to job stopped
        unsub_stopped = bus.subscribe(EventType.JOB_STOPPED, self._on_job_stopped)
        self._unsubscribe_callbacks.append(unsub_stopped)
        
        # Subscribe to checkpoint saves
        unsub_checkpoint = bus.subscribe(EventType.CHECKPOINT_SAVED, self._on_checkpoint)
        self._unsubscribe_callbacks.append(unsub_checkpoint)
    
    def _on_metrics_event(self, event: Event):
        """Handle metrics update event."""
        if event.job_id != self.job_id:
            return
        
        # Update job state
        self.job = state.get_job(self.job_id)
        
        # Update UI (wrap in try/except for background context)
        try:
            self._update_metrics_display()
            self._update_chart()
        except Exception:
            pass  # UI context may be invalid
    
    def _on_log_event(self, event: Event):
        """Handle new log line event."""
        if event.job_id != self.job_id:
            return
        
        line = event.data.get('line', '')
        timestamp = event.data.get('timestamp', '')
        if line:
            # Format with timestamp for consistency with file logs
            formatted_line = f"[{timestamp}] {line}" if timestamp else line
            self._all_log_lines.append(formatted_line)
            self.log_lines.append(line)  # Keep legacy for compatibility
            self._update_logs_display()
    
    def _on_job_completed(self, event: Event):
        """Handle job completion event."""
        if event.job_id != self.job_id:
            return
        
        self.job = state.get_job(self.job_id)
        try:
            notify_job_completed(self.job.name if self.job else "Job")
        except Exception:
            pass  # Notification failed due to context
        self._update_metrics_display()
        self._cleanup_subscriptions()
    
    def _on_job_failed(self, event: Event):
        """Handle job failed event."""
        if event.job_id != self.job_id:
            return
        
        self.job = state.get_job(self.job_id)
        error_msg = event.data.get('error', 'Unknown error')
        try:
            notify_job_failed(self.job.name if self.job else "Job", error_msg)
        except Exception:
            pass  # Notification failed due to context
        self._update_metrics_display()
        self._cleanup_subscriptions()
    
    def _on_job_stopped(self, event: Event):
        """Handle job stopped event."""
        if event.job_id != self.job_id:
            return
        
        self.job = state.get_job(self.job_id)
        try:
            notify_training_stopped(self.job.name if self.job else "Job")
        except Exception:
            pass  # Notification failed due to context
        self._update_metrics_display()
        self._cleanup_subscriptions()
    
    def _on_checkpoint(self, event: Event):
        """Handle checkpoint saved event."""
        if event.job_id != self.job_id:
            return
        
        # Checkpoint notification is already handled by TrainingService
        pass
    
    def _cleanup_subscriptions(self):
        """Unsubscribe from all events and stop timers."""
        for unsub in self._unsubscribe_callbacks:
            try:
                unsub()
            except Exception:
                pass
        self._unsubscribe_callbacks.clear()
        
        # Stop duration timer
        if self._duration_timer:
            self._duration_timer.cancel()
            self._duration_timer = None
    
    def _tick_duration(self):
        """Timer callback to update duration every second."""
        if not self.job_id:
            return
        
        # Always fetch fresh job state to get accurate duration
        self.job = state.get_job(self.job_id)
        if self.job:
            self._update_metrics_display()
        
        # Stop timer if job is no longer running
        if self.job and self.job.status not in ('running', 'pending'):
            if self._duration_timer:
                try:
                    self._duration_timer.cancel()
                    self._duration_timer = None
                except Exception:
                    pass
    
    def _update_metrics_display(self):
        """Update all dynamic UI elements with current job state."""
        if not self.job:
            return
        
        # Refresh job state
        self.job = state.get_job(self.job_id)
        if not self.job:
            return
        
        try:
            # Duration
            if self._duration_label:
                self._duration_label.set_text(f'Duration: {self.job.duration_str}')
            if self._status_label:
                self._status_label.set_text(self.job.status.capitalize())
            
            # Progress
            progress_text, counter_text = self._resolve_progress_display()
            if self._progress_percent_label:
                self._progress_percent_label.set_text(progress_text)
            if self._progress_bar:
                if progress_text.endswith("%"):
                    self._progress_bar.style(f'width: {progress_text.rstrip("%")}%')
                else:
                    self._progress_bar.style('width: 0%')
            if self._step_label:
                self._step_label.set_text(counter_text)
            
            # Epoch/Cycle
            if self._epoch_label:
                if self.job.type in CYCLE_BASED_JOB_TYPES:
                    self._epoch_label.set_text(f'{self.job.current_cycle}/{self.job.total_cycles}')
                else:
                    # Display epoch as float if fractional, int otherwise
                    current = self.job.current_epoch
                    if isinstance(current, float) and current % 1 != 0:
                        epoch_str = f'{current:.1f}'
                    else:
                        epoch_str = str(int(current))
                    self._epoch_label.set_text(f'{epoch_str}/{self.job.total_epochs}')
            
            # Step
            if self._step_label and self.job.type in TRAINING_JOB_TYPES:
                self._step_label.set_text(f'{self.job.current_step}/{self.job.total_steps or "?"}')
            
            # Metrics
            if self._loss_label:
                val = f'{self.job.latest_loss:.4f}' if self.job.latest_loss is not None else '--'
                self._loss_label.set_text(val)
            
            if self._lr_label:
                val = f'{self.job.latest_lr:.2e}' if self.job.latest_lr is not None else '--'
                self._lr_label.set_text(val)
            
            if self._grad_norm_label:
                val = f'{self.job.latest_grad_norm:.4f}' if self.job.latest_grad_norm is not None else '--'
                self._grad_norm_label.set_text(val)
            
            if self._verification_label and self.job.verification_rate is not None:
                self._verification_label.set_text(f'{self.job.verification_rate:.1%}')

            summary = self._derive_training_outcome()
            if self._weights_updated_label:
                self._weights_updated_label.set_text(summary["weights_updated"])
            if self._update_steps_label:
                self._update_steps_label.set_text(summary["update_steps"])
            if self._final_loss_label:
                self._final_loss_label.set_text(summary["final_loss"])
            if self._update_reason_label:
                self._update_reason_label.set_text(summary["final_reason"])
            if self._failure_reason_label:
                self._failure_reason_label.set_text(summary["failure_reason"])
            if self._run_id_label:
                self._run_id_label.set_text(summary["run_id"])
            if self._seed_label:
                self._seed_label.set_text(summary["seed"])
            if self._resume_label:
                self._resume_label.set_text(summary["resume"])
            if self._quality_status_label:
                self._quality_status_label.set_text(summary["quality_status"])
            if self._quality_headline_label or self._quality_summary_note_label:
                presentation = self._current_training_presentation()
                if self._quality_headline_label:
                    self._quality_headline_label.set_text(presentation.headline_status)
                if self._quality_summary_note_label:
                    self._quality_summary_note_label.set_text(presentation.supporting_summary)
            if self._quality_keep_rate_label:
                self._quality_keep_rate_label.set_text(summary["quality_keep_rate"])
            if self._quality_drop_reason_label:
                self._quality_drop_reason_label.set_text(summary["quality_drop_reason"])
            if self._quality_action_label:
                self._quality_action_label.set_text(summary["quality_action"])
            if self._recovery_action_label:
                self._recovery_action_label.set_text(summary["recovery_action"])

            if self._benchmark_metric_labels:
                benchmark_summary = self._derive_benchmark_outcome()
                for key, label_ref in self._benchmark_metric_labels.items():
                    if label_ref:
                        label_ref.set_text(benchmark_summary.get(key, "--"))
            if self._inference_metric_labels:
                inference_summary = self._derive_inference_outcome()
                for key, label_ref in self._inference_metric_labels.items():
                    if label_ref:
                        label_ref.set_text(inference_summary.get(key, "--"))
            if self._utility_metric_labels:
                utility_summary = self._derive_utility_outcome()
                for key, label_ref in self._utility_metric_labels.items():
                    if label_ref:
                        label_ref.set_text(utility_summary.get(key, "--"))
            if self._diagnostic_metric_labels:
                if self.job.type in QUALIFICATION_JOB_TYPES:
                    diagnostics_summary = self._derive_qualification_outcome()
                elif self.job.type in BOOTSTRAP_JOB_TYPES:
                    diagnostics_summary = self._derive_bootstrap_outcome()
                else:
                    diagnostics_summary = self._derive_live_probe_outcome()
                for key, label_ref in self._diagnostic_metric_labels.items():
                    if label_ref:
                        label_ref.set_text(diagnostics_summary.get(key, "--"))
        except Exception:
            pass  # UI context may be invalid
    
    def _update_chart(self):
        """Update the loss chart with new data."""
        try:
            if not self.job or self.job.type not in TRAINING_JOB_TYPES:
                return
            if hasattr(self, 'chart') and self.chart:
                loss_data = self._get_loss_data()
                self.chart.options['series'][0]['data'] = loss_data
                self.chart.update()
        except Exception:
            pass  # UI context may be invalid
    
    def _update_logs_display(self):
        """Update the logs display with new entries (append-only for performance)."""
        if not hasattr(self, 'log_container') or not self.log_container:
            return
        
        # On first render, load from _all_log_lines (includes file-loaded logs)
        # Then append only new lines
        new_lines = self._all_log_lines[self._displayed_log_count:]
        
        if not new_lines and self._displayed_log_count == 0:
            # Fallback: Get logs from in-memory service buffer if no file logs loaded yet.
            logs = self._service_logs_for_job()
            for entry in logs:
                line = entry.get('line', '')
                timestamp = entry.get('timestamp', '')
                formatted = f"[{timestamp}] {line}" if timestamp else line
                if formatted not in self._all_log_lines:
                    self._all_log_lines.append(formatted)
            new_lines = self._all_log_lines[self._displayed_log_count:]
        
        if new_lines:
            try:
                # Wrap in try/except to handle background task context issues
                with self.log_container:
                    for line in new_lines:
                        color = self._get_log_color(line)
                        ui.label(line).classes(f'text-[{color}]')
                        self._displayed_log_count += 1
                
                # Auto-scroll to bottom
                self._scroll_to_bottom()
            except Exception:
                # UI context may be invalid (e.g., called from background task or user navigated away)
                pass
    
    def _get_log_color(self, line: str) -> str:
        """Determine log line color."""
        line_lower = line.lower()
        if 'error' in line_lower or 'failed' in line_lower:
            return COLORS['error']
        elif 'warning' in line_lower:
            return COLORS['warning']
        elif 'saved' in line_lower or 'checkpoint' in line_lower:
            return COLORS['success']
        elif 'loading' in line_lower or 'starting' in line_lower:
            return COLORS['info']
        return COLORS['text_secondary']
    
    async def _stop_job(self):
        """Stop the current job."""
        if not self.job:
            return
        
        with ui.dialog() as dialog, ui.card().classes(f'bg-[{COLORS["bg_card"]}] p-6'):
            ui.label('Stop Training?').classes(
                f'text-lg font-semibold text-[{COLORS["text_primary"]}]'
            )
            ui.label('This will terminate the training process. Progress may be lost.').classes(
                f'text-sm text-[{COLORS["text_secondary"]}] mt-2'
            )
            
            with ui.row().classes('w-full justify-end gap-2 mt-4'):
                ui.button('Cancel', on_click=dialog.close).props('flat').classes(
                    f'text-[{COLORS["text_secondary"]}]'
                )
                ui.button('Stop', on_click=lambda: self._confirm_stop(dialog)).props(
                    'unelevated'
                ).classes(f'bg-[{COLORS["error"]}] text-white')
        
        dialog.open()
    
    async def _confirm_stop(self, dialog):
        """Confirm stopping the job."""
        dialog.close()
        
        if self.job:
            success = False
            # Route stop calls by job type.
            try:
                if self.job.type == "benchmark":
                    success = await self.benchmark_service.stop_job(self.job.id)
                elif self.job.type == "inference":
                    success = await self.inference_service.stop_job(self.job.id)
                elif self.job.type in UTILITY_JOB_TYPES:
                    success = await self.module_ops_service.stop_job(self.job.id)
                elif self.job.type in QUALIFICATION_JOB_TYPES:
                    success = await self.qualification_service.stop_job(self.job.id)
                elif self.job.type in BOOTSTRAP_JOB_TYPES:
                    success = await self.bootstrap_service.stop_job(self.job.id)
                elif self.job.type in LIVE_PROBE_JOB_TYPES:
                    success = await self.live_probe_service.stop_job(self.job.id)
                else:
                    success = await self.training_service.stop_job(self.job.id)
            except Exception as e:
                try:
                    notify_job_failed(self.job.name, f"Stop failed: {e}")
                except Exception:
                    pass
                return

            refreshed_job = state.get_job(self.job_id)
            if refreshed_job and refreshed_job.status in {"stopped", "completed", "failed"}:
                success = True
            
            try:
                if success:
                    notify_training_stopped(self.job.name)
                    self.job = refreshed_job or state.get_job(self.job_id)
                    # Re-render route so header controls match terminal status.
                    ui.navigate.to(f"/monitor/{self.job_id}")
                else:
                    notify_job_failed(self.job.name, "Failed to stop job")
            except Exception:
                pass  # Notification failed due to context

    async def _rerun_job(self):
        """Relaunch the current job from persisted launch context."""
        if not self.job:
            return
        context_path = self._get_launch_context_path()
        if not context_path:
            notify_job_failed(self.job.name, "No launch context found for rerun")
            return

        try:
            context = read_launch_context(context_path)
            if self.job.type == "benchmark":
                new_job_id = await self.benchmark_service.relaunch_from_context(
                    context_path,
                    origin_job_id=self.job.id,
                    source_ui_page="/monitor",
                )
            elif self.job.type == "inference":
                new_job_id = await self.inference_service.relaunch_from_context(
                    context_path,
                    origin_job_id=self.job.id,
                    source_ui_page="/monitor",
                )
            elif self.job.type in UTILITY_JOB_TYPES:
                new_job_id = await self.module_ops_service.relaunch_from_context(
                    context_path,
                    origin_job_id=self.job.id,
                    source_ui_page="/monitor",
                )
            elif self.job.type in QUALIFICATION_JOB_TYPES:
                new_job_id = await self.qualification_service.relaunch_from_context(
                    context_path,
                    origin_job_id=self.job.id,
                    source_ui_page="/monitor",
                )
            elif self.job.type in BOOTSTRAP_JOB_TYPES:
                new_job_id = await self.bootstrap_service.relaunch_from_context(
                    context_path,
                    origin_job_id=self.job.id,
                    source_ui_page="/monitor",
                )
            elif self.job.type in LIVE_PROBE_JOB_TYPES:
                new_job_id = await self.live_probe_service.relaunch_from_context(
                    context_path,
                    origin_job_id=self.job.id,
                    source_ui_page="/monitor",
                )
            else:
                new_job_id = await self.training_service.relaunch_from_context(
                    context_path,
                    origin_job_id=self.job.id,
                    source_ui_page="/monitor",
                )
            notify_job_started(f"Rerun: {context.job_type}")
            ui.navigate.to(f"/monitor/{new_job_id}")
        except Exception as e:
            notify_job_failed(self.job.name, f"Rerun failed: {e}")

    async def _resume_latest_job(self):
        """Resume from latest cycle checkpoint when supported."""
        if not self.job:
            return
        if self.job.type not in CYCLE_BASED_JOB_TYPES:
            notify_job_failed(self.job.name, "Resume Latest is not supported for this job type")
            return
        context_path = self._get_launch_context_path()
        if not context_path:
            notify_job_failed(self.job.name, "No launch context found for resume")
            return

        try:
            new_job_id = await self.training_service.relaunch_from_context(
                context_path,
                origin_job_id=self.job.id,
                resume_latest=True,
                source_ui_page="/monitor",
            )
            notify_job_started(f"Resume Latest: {self.job.type.upper()}")
            ui.navigate.to(f"/monitor/{new_job_id}")
        except Exception as e:
            notify_job_failed(self.job.name, f"Resume Latest failed: {e}")

    def _clone_to_form(self):
        """Clone persisted launch args into the matching launch page form."""
        if not self.job:
            return
        context_path = self._get_launch_context_path()
        if not context_path:
            notify_job_failed(self.job.name, "No launch context found to clone")
            return
        try:
            context = read_launch_context(context_path)
            payload = {
                "launch_context_file": str(context_path),
                "job_type": context.job_type,
                "args": context.args,
            }
            recovery = self._current_recovery_guidance()
            if recovery.get("status") in {"ready", "advisory_only"}:
                payload["suggested_overrides"] = dict(recovery.get("suggested_overrides") or {})
                payload["recovery_reason_code"] = recovery.get("reason_code")
                payload["recovery_summary"] = recovery.get("evidence_summary")
            if self.job.type == "benchmark":
                app.storage.user["benchmark_clone_payload"] = payload
                ui.navigate.to("/benchmark")
            elif self.job.type == "inference":
                app.storage.user["inference_clone_payload"] = payload
                ui.navigate.to("/inference")
            elif self.job.type in UTILITY_JOB_TYPES:
                app.storage.user["ops_clone_payload"] = payload
                ui.navigate.to("/ops-console")
            elif self.job.type in QUALIFICATION_JOB_TYPES:
                app.storage.user["qualification_clone_payload"] = payload
                ui.navigate.to("/research-hub")
            elif self.job.type in BOOTSTRAP_JOB_TYPES:
                app.storage.user["bootstrap_clone_payload"] = payload
                ui.navigate.to("/research-hub")
            elif self.job.type in LIVE_PROBE_JOB_TYPES:
                app.storage.user["live_probe_clone_payload"] = payload
                ui.navigate.to("/research-hub")
            else:
                app.storage.user["training_clone_payload"] = payload
                ui.navigate.to("/training")
        except Exception as e:
            notify_job_failed(self.job.name, f"Clone to form failed: {e}")

    def _service_logs_for_job(self) -> list[dict]:
        if not self.job:
            return []
        if self.job.type == "benchmark":
            return self.benchmark_service.get_logs(self.job_id, last_n=1000)
        if self.job.type == "inference":
            return self.inference_service.get_logs(self.job_id, last_n=1000)
        if self.job.type in UTILITY_JOB_TYPES:
            return self.module_ops_service.get_logs(self.job_id, last_n=1000)
        if self.job.type in QUALIFICATION_JOB_TYPES:
            return self.qualification_service.get_logs(self.job_id, last_n=1000)
        if self.job.type in BOOTSTRAP_JOB_TYPES:
            return self.bootstrap_service.get_logs(self.job_id, last_n=1000)
        if self.job.type in LIVE_PROBE_JOB_TYPES:
            return self.live_probe_service.get_logs(self.job_id, last_n=1000)
        return self.training_service.get_logs(self.job_id, last_n=1000)

    def _copy_artifacts_path(self):
        """Copy artifact output directory to clipboard."""
        if not self.job or not self.job.output_dir:
            ui.notify("No artifact path available", type="warning", timeout=1500)
            return
        output_path = str(self.job.output_dir)
        ui.run_javascript(
            f"navigator.clipboard.writeText({json.dumps(output_path)});"
        )
        ui.notify("Copied artifact path", type="positive", timeout=1200)
    
    def _scroll_to_bottom(self):
        """Scroll log viewer to bottom."""
        if hasattr(self, 'log_container') and self.log_container:
            ui.run_javascript(f'document.querySelector("[data-log-container]").scrollTop = 999999')
    
    def _copy_logs(self):
        """Copy ALL logs to clipboard from persistent file or memory."""
        log_text = ""
        
        # Try to read from persistent log file first (has full history)
        log_file = self._get_log_file_path()
        if log_file and log_file.exists():
            try:
                log_text = log_file.read_text(encoding='utf-8').strip()
            except Exception:
                pass
        
        # Fallback to in-memory logs (use strip() check, not truthiness)
        if not log_text:
            log_text = '\n'.join(self._all_log_lines).strip()
        
        # Final fallback to service buffer
        if not log_text:
            logs = self._service_logs_for_job()
            log_text = '\n'.join([entry.get('line', '') for entry in logs]).strip()
        
        if not log_text:
            ui.notify('No logs available to copy', type='warning', timeout=1500)
            return
        
        # Use json.dumps for proper JS string escaping with error handling
        ui.run_javascript(f'''
            navigator.clipboard.writeText({json.dumps(log_text)})
                .then(() => {{ /* success */ }})
                .catch((err) => {{ 
                    console.error("Clipboard copy failed:", err);
                    // Fallback: create a temporary textarea
                    const ta = document.createElement('textarea');
                    ta.value = {json.dumps(log_text)};
                    document.body.appendChild(ta);
                    ta.select();
                    document.execCommand('copy');
                    document.body.removeChild(ta);
                }});
        ''')
        line_count = len(log_text.strip().split('\n'))
        ui.notify(f'Copied {line_count} log lines to clipboard', type='positive', timeout=1500)
    
    async def _download_logs(self):
        """Download ALL logs as a text file from persistent file or memory."""
        log_text = ""
        
        # Try to read from persistent log file first (has full history)
        log_file = self._get_log_file_path()
        if log_file and log_file.exists():
            try:
                log_text = log_file.read_text(encoding='utf-8').strip()
            except Exception:
                pass
        
        # Fallback to in-memory logs (use strip() check, not truthiness)
        if not log_text:
            log_text = '\n'.join(self._all_log_lines).strip()
        
        # Final fallback to service buffer
        if not log_text:
            logs = self._service_logs_for_job()
            log_text = '\n'.join([entry.get('line', '') for entry in logs]).strip()
        
        if not log_text:
            ui.notify('No logs available to download', type='warning', timeout=1500)
            return
        
        job_name = self.job.name.replace(' ', '_').replace(':', '-') if self.job else 'training'
        filename = f"{job_name}_logs.txt"
        
        # Trigger browser download
        line_count = len(log_text.strip().split('\n'))
        ui.download(log_text.encode('utf-8'), filename)
        ui.notify(f'Downloading {line_count} log lines', type='positive', timeout=1500)


class MonitorList:
    """List of all jobs for monitoring."""

    def __init__(self) -> None:
        self.show_advanced_diagnostics = bool(
            app.storage.user.get("monitor_show_advanced_diagnostics", False)
        )

    def _is_advanced_diagnostics_job(self, job: JobState) -> bool:
        return job.type in DIAGNOSTIC_JOB_TYPES

    def _on_toggle_advanced(self, value: bool) -> None:
        self.show_advanced_diagnostics = bool(value)
        app.storage.user["monitor_show_advanced_diagnostics"] = self.show_advanced_diagnostics
        ui.navigate.to("/monitor")

    def render(self):
        """Render the job list page."""
        with ui.column().classes('page-content w-full gap-6 p-6'):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label('Monitor Jobs').classes(
                    f'text-2xl font-bold text-[{COLORS["text_primary"]}] animate-in'
                )
                ui.checkbox(
                    "Show advanced diagnostics runs",
                    value=self.show_advanced_diagnostics,
                    on_change=lambda e: self._on_toggle_advanced(bool(e.value)),
                ).classes(f'text-sm text-[{COLORS["text_secondary"]}]')
            
            # Active jobs
            active_jobs = state.get_active_jobs()
            if not self.show_advanced_diagnostics:
                active_jobs = [
                    job for job in active_jobs if not self._is_advanced_diagnostics_job(job)
                ]
            if active_jobs:
                with ui.column().classes(
                    f'w-full gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                    f'border border-[#2d343c] animate-in stagger-1'
                ):
                    ui.label('Active Jobs').classes(
                        f'text-base font-semibold text-[{COLORS["text_primary"]}]'
                    )
                    
                    for job in active_jobs:
                        self._render_job_card(job)
            
            # Recent jobs
            recent = state.get_recent_jobs(10)
            completed = [j for j in recent if j.status != 'running']
            if not self.show_advanced_diagnostics:
                completed = [
                    job for job in completed if not self._is_advanced_diagnostics_job(job)
                ]
            hidden_count = 0
            if not self.show_advanced_diagnostics:
                hidden_count = sum(
                    1 for job in state.get_recent_jobs(50) if self._is_advanced_diagnostics_job(job)
                )
                hidden_count += sum(
                    1 for job in state.get_active_jobs() if self._is_advanced_diagnostics_job(job)
                )
            
            with ui.column().classes(
                f'w-full gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                f'border border-[#2d343c] animate-in stagger-2'
            ):
                ui.label('Recent Jobs').classes(
                    f'text-base font-semibold text-[{COLORS["text_primary"]}]'
                )
                if hidden_count:
                    ui.label(
                        f"{hidden_count} advanced diagnostics run(s) hidden. Enable the toggle to view them."
                    ).classes(f'text-xs text-[{COLORS["text_muted"]}]')
                
                if not completed and not active_jobs:
                    with ui.column().classes('w-full items-center py-8 gap-2'):
                        ui.icon('inbox', size='48px').classes(
                            f'text-[{COLORS["text_muted"]}]'
                        )
                        ui.label('No jobs yet').classes(
                            f'text-sm text-[{COLORS["text_muted"]}]'
                        )
                else:
                    for job in completed:
                        self._render_job_card(job)
    
    def _render_job_card(self, job: JobState):
        """Render a job card in the list."""
        status_color = COLORS.get(job.status, COLORS['text_secondary'])
        
        with ui.row().classes(
            f'w-full items-center gap-4 p-4 rounded-lg bg-[{COLORS["bg_secondary"]}] '
            f'hover:bg-[{COLORS["bg_hover"]}] transition-colors cursor-pointer'
        ).on('click', lambda j=job: ui.navigate.to(f'/monitor/{j.id}')):
            # Status indicator
            if job.status == 'running':
                ui.element('div').classes(
                    f'w-3 h-3 rounded-full bg-[{status_color}] running-glow'
                )
            else:
                ui.element('div').classes(
                    f'w-3 h-3 rounded-full bg-[{status_color}]'
                )
            
            # Job info
            with ui.column().classes('flex-1 gap-0.5'):
                ui.label(job.name).classes(
                    f'text-sm font-medium text-[{COLORS["text_primary"]}]'
                )
                ui.label(f'{job.type.upper()} • {job.duration_str}').classes(
                    f'text-xs text-[{COLORS["text_muted"]}]'
                )
            
            # Progress or status
            if job.status == 'running':
                if job.type in INDETERMINATE_PROGRESS_JOB_TYPES and job.total_steps == 0:
                    ui.label('—').classes(
                        f'text-sm font-mono text-[{COLORS["text_muted"]}]'
                    )
                else:
                    ui.label(f'{job.progress_percent:.0f}%').classes(
                        f'text-sm font-mono text-[{COLORS["primary"]}]'
                    )
            else:
                ui.label(job.status.capitalize()).classes(
                    f'text-xs text-[{status_color}]'
                )
            
            # Arrow
            ui.icon('chevron_right', size='20px').classes(
                f'text-[{COLORS["text_muted"]}]'
            )
