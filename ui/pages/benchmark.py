"""
Benchmark Launch Page

Configure and launch Code, VLM, Audio, and Agentic benchmarks.
"""

from nicegui import ui
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field

from ui.theme import COLORS
from ui.state import state
from ui.services.benchmark_service import (
    BenchmarkService,
    BenchmarkType,
    BenchmarkPreset,
    get_presets_for_type,
    CODE_PRESETS,
    VLM_PRESETS,
    AUDIO_PRESETS,
    AGENTIC_PRESETS,
    get_benchmark_service,
)
from ui.components.notifications import notify_job_started, notify_job_failed
from ui.components.file_picker import FilePicker


ModelSource = Literal["preset", "local", "custom"]


@dataclass
class BenchmarkFormData:
    """Benchmark form data."""
    model: str = "Qwen/Qwen2.5-Coder-3B"
    model_source: ModelSource = "preset"
    custom_model_path: str = ""
    benchmark_type: BenchmarkType = BenchmarkType.CODE
    preset: Optional[BenchmarkPreset] = None
    limit: int = 500
    output_dir: str = "results/benchmarks"
    # Code benchmark specific
    samples_per_prompt: int = 5  # For pass@k calculation
    verifier: str = "humaneval"  # Code verifier type


class Benchmark:
    """Benchmark launch page component."""
    
    # Popular models by type
    CODE_MODELS = [
        "Qwen/Qwen2.5-Coder-0.5B",
        "Qwen/Qwen2.5-Coder-1.5B",
        "Qwen/Qwen2.5-Coder-3B",
        "Qwen/Qwen2.5-Coder-7B",
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
    ]
    
    VLM_MODELS = [
        "Qwen/Qwen2-VL-2B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
        "llava-hf/llava-1.5-7b-hf",
        "microsoft/Phi-3-vision-128k-instruct",
    ]
    
    AUDIO_MODELS = [
        "openai/whisper-tiny",
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-large-v3",
    ]
    
    AGENTIC_MODELS = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]
    
    def __init__(self):
        self.data = BenchmarkFormData()
        self.data.preset = CODE_PRESETS[0] if CODE_PRESETS else None
        self.is_running = False
        self.benchmark_service = get_benchmark_service(state)
        self._tabs_container = None
        self._config_container = None
        # Cache for discovered local models
        self._local_models_cache: list[tuple[str, str]] = []
        self._refresh_local_models()
    
    def _refresh_local_models(self):
        """Refresh the cache of locally trained models."""
        self._local_models_cache = self._discover_local_models()
    
    def _discover_local_models(self) -> list[tuple[str, str]]:
        """
        Discover locally trained models in models/ directory.
        
        Returns:
            List of (path, display_name) tuples for local models.
        """
        local_models = []
        models_dir = Path("models")
        
        if not models_dir.exists():
            return local_models
        
        try:
            for run_dir in sorted(models_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if not run_dir.is_dir():
                    continue
                
                # Skip hidden directories
                if run_dir.name.startswith('.'):
                    continue
                
                # Look for final_model directory (SFT/RAFT output)
                final_model = run_dir / "final_model"
                if final_model.exists() and final_model.is_dir():
                    display_name = f"{run_dir.name} (local)"
                    local_models.append((str(final_model), display_name))
                
                # Look for checkpoints (show last 2)
                checkpoints = sorted(
                    run_dir.glob("checkpoint-*"),
                    key=lambda p: int(p.name.split('-')[-1]) if p.name.split('-')[-1].isdigit() else 0,
                    reverse=True
                )
                for ckpt in checkpoints[:2]:
                    if ckpt.is_dir():
                        display_name = f"{run_dir.name}/{ckpt.name}"
                        local_models.append((str(ckpt), display_name))
                
                # Also check for direct model files (adapter_config.json indicates LoRA)
                if (run_dir / "adapter_config.json").exists() or (run_dir / "config.json").exists():
                    if str(run_dir) not in [m[0] for m in local_models]:
                        display_name = f"{run_dir.name} (local)"
                        local_models.append((str(run_dir), display_name))
        except Exception as e:
            print(f"[Benchmark] Error scanning models directory: {e}")
        
        return local_models
    
    def render(self):
        """Render the benchmark page."""
        with ui.column().classes('page-content w-full gap-6 p-6'):
            # Header
            with ui.row().classes('w-full items-center justify-between animate-in'):
                ui.label('Benchmark').classes(
                    f'text-2xl font-bold text-[{COLORS["text_primary"]}]'
                )
                
                # Quick status
                with ui.row().classes('items-center gap-2'):
                    ui.icon('speed', size='20px').classes(f'text-[{COLORS["accent"]}]')
                    ui.label('Compare model to published benchmarks').classes(
                        f'text-sm text-[{COLORS["text_muted"]}]'
                    )
            
            # Benchmark type tabs - in container for refresh
            with ui.row().classes(
                f'w-full gap-2 p-2 rounded-xl bg-[{COLORS["bg_card"]}] '
                f'border border-[#2d343c] animate-in stagger-1'
            ) as self._tabs_container:
                self._render_type_tabs()
            
            # Main form
            with ui.column().classes('w-full gap-6') as self._config_container:
                self._render_form()
    
    def _render_type_tabs(self):
        """Render the benchmark type tab buttons."""
        self._type_button("Code", BenchmarkType.CODE, "code")
        self._type_button("VLM", BenchmarkType.VLM, "image")
        self._type_button("Audio", BenchmarkType.AUDIO, "mic")
        self._type_button("Agentic", BenchmarkType.AGENTIC, "smart_toy")
    
    def _type_button(self, label: str, btype: BenchmarkType, icon: str):
        """Render a benchmark type toggle button."""
        is_active = self.data.benchmark_type == btype
        
        with ui.element('div').classes(
            f'flex-1 flex items-center justify-center gap-3 py-4 rounded-lg cursor-pointer transition-all '
            + (f'bg-[{COLORS["primary"]}]/20 border border-[{COLORS["primary"]}]' if is_active 
               else f'bg-transparent border border-transparent hover:bg-[{COLORS["bg_hover"]}]')
        ).on('click', lambda t=btype: self._set_type(t)):
            ui.icon(icon, size='24px').classes(
                f'text-[{COLORS["primary"]}]' if is_active else f'text-[{COLORS["text_secondary"]}]'
            )
            ui.label(label).classes(
                f'text-base font-medium '
                + (f'text-[{COLORS["primary"]}]' if is_active else f'text-[{COLORS["text_secondary"]}]')
            )
    
    def _set_type(self, btype: BenchmarkType):
        """Switch benchmark type."""
        self.data.benchmark_type = btype
        
        # Update preset to first of this type
        presets = get_presets_for_type(btype)
        self.data.preset = presets[0] if presets else None
        
        # Reset model source to preset and select first base model for this type
        self.data.model_source = "preset"
        self.data.custom_model_path = ""
        models = self._get_models_for_type(btype)
        if models:
            self.data.model = models[0]
        
        # Refresh BOTH tabs and form
        self._tabs_container.clear()
        with self._tabs_container:
            self._render_type_tabs()
        
        self._config_container.clear()
        with self._config_container:
            self._render_form()
    
    def _get_models_for_type(self, btype: BenchmarkType) -> list[str]:
        """Get suggested models for benchmark type."""
        if btype == BenchmarkType.CODE:
            return self.CODE_MODELS
        elif btype == BenchmarkType.VLM:
            return self.VLM_MODELS
        elif btype == BenchmarkType.AUDIO:
            return self.AUDIO_MODELS
        elif btype == BenchmarkType.AGENTIC:
            return self.AGENTIC_MODELS
        return self.CODE_MODELS
    
    def _render_form(self):
        """Render the benchmark configuration form."""
        # Two-column layout
        with ui.row().classes('w-full gap-6'):
            # Left column - Model & Benchmark Selection
            with ui.column().classes('flex-1 gap-6'):
                self._render_model_section()
                self._render_preset_section()
            
            # Right column - Configuration & Launch
            with ui.column().classes('flex-1 gap-6'):
                self._render_config_section()
                self._render_launch_section()
    
    def _render_model_section(self):
        """Render model selection section with dropdown for local and base models."""
        with ui.column().classes(
            f'w-full gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c] animate-in stagger-2'
        ):
            with ui.row().classes('w-full items-center justify-between'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('psychology', size='20px').classes(f'text-[{COLORS["primary"]}]')
                    ui.label('Model').classes(f'text-base font-semibold text-[{COLORS["text_primary"]}]')
                
                # Refresh local models button
                ui.button(icon='refresh', on_click=self._on_refresh_local_models).props(
                    'flat round dense size=sm'
                ).classes(f'text-[{COLORS["text_muted"]}]').tooltip('Refresh local models')
            
            # Build dropdown options
            model_options = {}
            
            # Add local models first (most relevant for benchmarking trained models)
            if self._local_models_cache:
                for path, display_name in self._local_models_cache:
                    model_options[path] = f"📁 {display_name}"
            
            # Add base models for current benchmark type
            base_models = self._get_models_for_type(self.data.benchmark_type)
            for model in base_models:
                short_name = Path(model).name
                model_options[model] = f"🤗 {short_name}"
            
            # Add custom option
            model_options["custom"] = "📂 Custom path..."
            
            # Determine current value for dropdown
            if self.data.model_source == "custom":
                current_value = "custom"
            else:
                current_value = self.data.model
            
            with ui.column().classes('w-full gap-2'):
                ui.label('Select Model').classes(f'text-xs text-[{COLORS["text_muted"]}]')
                
                def on_model_change(e):
                    selected = e.value
                    if selected == "custom":
                        self.data.model_source = "custom"
                        # Don't change model yet, wait for custom input
                    else:
                        self.data.model_source = "local" if selected.startswith("models/") else "preset"
                        self.data.model = selected
                    # Refresh form to show/hide custom input
                    self._config_container.clear()
                    with self._config_container:
                        self._render_form()
                
                ui.select(
                    options=model_options,
                    value=current_value,
                    on_change=on_model_change
                ).classes('w-full').props('outlined dense dark color=grey-7')
                
                # Show custom input when "custom" is selected
                if self.data.model_source == "custom":
                    with ui.column().classes('w-full gap-2 mt-2'):
                        ui.label('Custom Model Path').classes(f'text-xs text-[{COLORS["text_muted"]}]')
                        with ui.row().classes('w-full gap-2'):
                            custom_input = ui.input(
                                placeholder='HuggingFace model ID or local path...',
                                value=self.data.custom_model_path
                            ).classes('flex-1').props('outlined dense dark color=grey-7')
                            
                            def on_custom_input(e):
                                self.data.custom_model_path = e.value
                                self.data.model = e.value
                            
                            custom_input.on('update:model-value', on_custom_input)
                            
                            ui.button(
                                icon='folder_open',
                                on_click=self._browse_model
                            ).props('flat dense').classes(
                                f'text-[{COLORS["text_muted"]}]'
                            ).tooltip('Browse local models...')
                        
                        ui.label('Enter a HuggingFace model ID (e.g., Qwen/Qwen2.5-Coder-7B) or local path').classes(
                            f'text-xs text-[{COLORS["text_muted"]}]'
                        )
                
                # Quick select buttons for popular base models
                ui.label('Quick select:').classes(f'text-xs text-[{COLORS["text_muted"]}] mt-3')
                
                with ui.row().classes('w-full flex-wrap gap-2'):
                    for model in base_models[:4]:
                        short_name = Path(model).name
                        is_selected = self.data.model == model and self.data.model_source != "custom"
                        
                        ui.button(
                            short_name,
                            on_click=lambda m=model: self._select_model(m, "preset")
                        ).props(
                            f'{"" if is_selected else "outline"} dense size=sm'
                        ).classes(
                            f'text-xs {"bg-[" + COLORS["primary"] + "]/20" if is_selected else ""}'
                        )
    
    def _select_model(self, model: str, source: ModelSource = "preset"):
        """Select a model from presets or local models."""
        self.data.model = model
        self.data.model_source = source
        if source != "custom":
            self.data.custom_model_path = ""
        # Refresh form to update UI
        self._config_container.clear()
        with self._config_container:
            self._render_form()
    
    def _browse_model(self):
        """Open file picker for local model directory."""
        FilePicker(
            title="Select Model Directory",
            path_type="directory",
            start_path="models/",
            on_select=self._set_custom_model
        )
    
    def _set_custom_model(self, path: str):
        """Set a custom model path from file picker."""
        self.data.model = path
        self.data.model_source = "custom"
        self.data.custom_model_path = path
        # Refresh form to update UI
        self._config_container.clear()
        with self._config_container:
            self._render_form()
    
    def _on_refresh_local_models(self):
        """Refresh local models list and update UI."""
        self._refresh_local_models()
        ui.notify(f'Found {len(self._local_models_cache)} local models', type='info', timeout=1500)
        # Refresh form to update dropdown
        self._config_container.clear()
        with self._config_container:
            self._render_form()
    
    def _render_preset_section(self):
        """Render benchmark preset selection."""
        presets = get_presets_for_type(self.data.benchmark_type)
        
        with ui.column().classes(
            f'w-full gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c] animate-in stagger-3'
        ):
            with ui.row().classes('w-full items-center gap-2'):
                ui.icon('playlist_add_check', size='20px').classes(f'text-[{COLORS["secondary"]}]')
                ui.label('Benchmark').classes(f'text-base font-semibold text-[{COLORS["text_primary"]}]')
            
            # Preset cards
            with ui.column().classes('w-full gap-2'):
                for preset in presets:
                    self._render_preset_card(preset)
    
    def _render_preset_card(self, preset: BenchmarkPreset):
        """Render a single preset card."""
        is_selected = self.data.preset and self.data.preset.name == preset.name
        
        with ui.element('div').classes(
            f'w-full p-3 rounded-lg cursor-pointer transition-all '
            + (f'bg-[{COLORS["primary"]}]/10 border border-[{COLORS["primary"]}]' if is_selected
               else f'bg-[{COLORS["bg_primary"]}] border border-transparent hover:border-[{COLORS["text_muted"]}]')
        ).on('click', lambda p=preset: self._select_preset(p)):
            with ui.row().classes('w-full items-center justify-between'):
                with ui.column().classes('gap-1'):
                    ui.label(preset.name).classes(
                        f'text-sm font-medium text-[{COLORS["text_primary"]}]'
                    )
                    ui.label(preset.description).classes(
                        f'text-xs text-[{COLORS["text_muted"]}]'
                    )
                
                if is_selected:
                    ui.icon('check_circle', size='20px').classes(f'text-[{COLORS["primary"]}]')
    
    def _select_preset(self, preset: BenchmarkPreset):
        """Select a benchmark preset."""
        self.data.preset = preset
        self.data.limit = preset.default_limit
        
        # Refresh form
        self._config_container.clear()
        with self._config_container:
            self._render_form()
    
    def _render_config_section(self):
        """Render configuration options."""
        with ui.column().classes(
            f'w-full gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c] animate-in stagger-4'
        ):
            with ui.row().classes('w-full items-center gap-2'):
                ui.icon('tune', size='20px').classes(f'text-[{COLORS["info"]}]')
                ui.label('Configuration').classes(f'text-base font-semibold text-[{COLORS["text_primary"]}]')
            
            # Sample limit
            with ui.column().classes('w-full gap-2'):
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label('Sample Limit').classes(f'text-xs text-[{COLORS["text_muted"]}]')
                    self._limit_label = ui.label(str(self.data.limit)).classes(
                        f'text-sm font-mono text-[{COLORS["text_secondary"]}]'
                    )
                
                limit_slider = ui.slider(min=10, max=1000, step=10, value=self.data.limit).classes('w-full')
                limit_slider.on('update:model-value', lambda e: self._update_limit(e.args))
            
            # Code benchmark specific options
            if self.data.benchmark_type == BenchmarkType.CODE:
                with ui.row().classes('w-full gap-4 flex-wrap'):
                    # Samples per prompt (for pass@k)
                    with ui.column().classes('flex-1 min-w-[200px] gap-2'):
                        with ui.row().classes('w-full items-center justify-between'):
                            ui.label('Samples per Prompt').classes(f'text-xs text-[{COLORS["text_muted"]}]')
                            self._samples_label = ui.label(str(self.data.samples_per_prompt)).classes(
                                f'text-sm font-mono text-[{COLORS["text_secondary"]}]'
                            )
                        samples_slider = ui.slider(
                            min=1, max=50, step=1, value=self.data.samples_per_prompt
                        ).classes('w-full')
                        samples_slider.on('update:model-value', lambda e: self._update_samples(e.args))
                        ui.label('More samples = more accurate pass@k').classes(
                            f'text-xs text-[{COLORS["text_muted"]}]'
                        )
                    
                    # Verifier selection
                    with ui.column().classes('flex-1 min-w-[200px] gap-2'):
                        ui.label('Verifier').classes(f'text-xs text-[{COLORS["text_muted"]}]')
                        ui.select(
                            options={
                                'humaneval': 'HumanEval (Python tests)',
                                'mbpp': 'MBPP (Python tests)',
                                'gcc': 'GCC (C/C++ compile)',
                                'python': 'Python (generic)',
                                'auto': 'Auto-detect',
                            },
                            value=self.data.verifier
                        ).classes('w-full').props(
                            'outlined dense dark color=grey-7'
                        ).bind_value(self.data, 'verifier')
            
            # Output directory
            with ui.column().classes('w-full gap-2'):
                ui.label('Output Directory').classes(f'text-xs text-[{COLORS["text_muted"]}]')
                output_input = ui.input(
                    value=self.data.output_dir,
                    placeholder='results/benchmarks',
                ).classes('w-full').props('outlined dense')
                output_input.bind_value(self.data, 'output_dir')
            
            # Info box
            with ui.row().classes(
                f'w-full p-3 rounded-lg gap-3 bg-[{COLORS["bg_primary"]}]'
            ):
                ui.icon('info', size='18px').classes(f'text-[{COLORS["info"]}]')
                with ui.column().classes('gap-1'):
                    ui.label('Results will be saved as JSON').classes(
                        f'text-xs text-[{COLORS["text_secondary"]}]'
                    )
                    ui.label('View results on the Results page after completion').classes(
                        f'text-xs text-[{COLORS["text_muted"]}]'
                    )
    
    def _update_samples(self, value):
        """Update samples per prompt value."""
        self.data.samples_per_prompt = int(value)
        if hasattr(self, '_samples_label'):
            self._samples_label.set_text(str(self.data.samples_per_prompt))
    
    def _update_limit(self, value):
        """Update limit value."""
        self.data.limit = int(value)
        if hasattr(self, '_limit_label'):
            self._limit_label.set_text(str(self.data.limit))
    
    def _render_launch_section(self):
        """Render the launch section."""
        with ui.column().classes(
            f'w-full gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c] animate-in stagger-5'
        ):
            with ui.row().classes('w-full items-center gap-2'):
                ui.icon('rocket_launch', size='20px').classes(f'text-[{COLORS["success"]}]')
                ui.label('Launch').classes(f'text-base font-semibold text-[{COLORS["text_primary"]}]')
            
            # Summary - determine effective model for display
            if self.data.model_source == "custom":
                display_model = self.data.custom_model_path or self.data.model
            else:
                display_model = self.data.model
            display_model_name = Path(display_model).name if display_model else "-"
            
            with ui.column().classes(f'w-full gap-2 p-4 rounded-lg bg-[{COLORS["bg_primary"]}]'):
                with ui.row().classes('w-full justify-between'):
                    ui.label('Model').classes(f'text-xs text-[{COLORS["text_muted"]}]')
                    ui.label(display_model_name).classes(
                        f'text-xs font-mono text-[{COLORS["text_secondary"]}]'
                    )
                
                with ui.row().classes('w-full justify-between'):
                    ui.label('Benchmark').classes(f'text-xs text-[{COLORS["text_muted"]}]')
                    ui.label(self.data.preset.name if self.data.preset else '-').classes(
                        f'text-xs font-mono text-[{COLORS["text_secondary"]}]'
                    )
                
                with ui.row().classes('w-full justify-between'):
                    ui.label('Samples').classes(f'text-xs text-[{COLORS["text_muted"]}]')
                    ui.label(str(self.data.limit)).classes(
                        f'text-xs font-mono text-[{COLORS["text_secondary"]}]'
                    )
                
                with ui.row().classes('w-full justify-between'):
                    ui.label('Type').classes(f'text-xs text-[{COLORS["text_muted"]}]')
                    ui.label(self.data.benchmark_type.value.upper()).classes(
                        f'text-xs font-mono text-[{COLORS["text_secondary"]}]'
                    )
            
            # Launch button
            with ui.row().classes('w-full gap-3'):
                ui.button(
                    'Launch Benchmark',
                    icon='play_arrow',
                    on_click=self._launch_benchmark
                ).props('unelevated color=positive').classes('flex-1')
                
                ui.button(
                    icon='open_in_new',
                    on_click=lambda: ui.navigate.to('/results')
                ).props('outline').classes(f'text-[{COLORS["text_secondary"]}]').tooltip('View Results')
    
    async def _launch_benchmark(self):
        """Launch the benchmark."""
        # Get the effective model path
        if self.data.model_source == "custom":
            model = self.data.custom_model_path or self.data.model
        else:
            model = self.data.model
        
        if not model:
            notify_job_failed("Benchmark", "Please select a model")
            return
        
        if not self.data.preset:
            notify_job_failed("Benchmark", "Please select a benchmark")
            return
        
        self.is_running = True
        
        try:
            # Build output path
            model_name = Path(model).name
            output_dir = f"{self.data.output_dir}/{model_name}-{self.data.preset.dataset}"
            
            # Merge preset CLI args with form values
            extra_args = dict(self.data.preset.cli_args)
            
            # Launch benchmark
            job_id = await self.benchmark_service.launch_benchmark(
                model=model,
                benchmark_type=self.data.benchmark_type,
                benchmark_name=self.data.preset.dataset,
                limit=self.data.limit,
                output_dir=output_dir,
                samples_per_prompt=self.data.samples_per_prompt,
                verifier=self.data.verifier if self.data.benchmark_type == BenchmarkType.CODE else None,
                **extra_args,
            )
            
            notify_job_started(f"Benchmark: {self.data.preset.name}")
            
            # Navigate to monitor
            ui.navigate.to(f'/monitor/{job_id}')
            
        except Exception as e:
            notify_job_failed("Benchmark", str(e))
        finally:
            self.is_running = False
