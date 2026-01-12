"""
Training Launch Page

Configure and launch SFT/RAFT training runs.
"""

from nicegui import ui
from pathlib import Path
from typing import Literal, Optional
from dataclasses import dataclass, field

from ui.theme import COLORS
from ui.state import state
from ui.services import TrainingService
from ui.components.notifications import notify_job_started, notify_job_failed


# =============================================================================
# Recommended Models (from research docs)
# =============================================================================

RECOMMENDED_MODELS = {
    "code": [
        ("Qwen/Qwen2.5-Coder-0.5B", "Qwen2.5-Coder-0.5B (Quick testing)"),
        ("Qwen/Qwen2.5-Coder-1.5B", "Qwen2.5-Coder-1.5B (Fast iteration)"),
        ("Qwen/Qwen2.5-Coder-3B", "Qwen2.5-Coder-3B (Balanced)"),
        ("Qwen/Qwen2.5-Coder-7B", "Qwen2.5-Coder-7B (High quality)"),
        ("Qwen/Qwen2.5-Coder-14B", "Qwen2.5-Coder-14B (Best quality)"),
        ("deepseek-ai/deepseek-coder-1.3b-instruct", "DeepSeek-Coder-1.3B"),
        ("deepseek-ai/deepseek-coder-6.7b-instruct", "DeepSeek-Coder-6.7B"),
        ("LiquidAI/LFM2.5-1.2B-Instruct", "LiquidAI LFM2.5-1.2B (Experimental)"),
        ("LiquidAI/LFM2.5-3.2B-Instruct", "LiquidAI LFM2.5-3.2B (Experimental)"),
    ],
    "general": [
        ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"),
        ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct"),
        ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct"),
        ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"),
        ("Qwen/Qwen2.5-14B-Instruct", "Qwen2.5-14B-Instruct"),
    ],
}


# =============================================================================
# Datasets (expanded from datasets page)
# =============================================================================

SFT_DATASETS = [
    # Code datasets
    ("codealpaca", "CodeAlpaca (20K code instructions)"),
    ("evol_instruct_code", "Evol-Instruct-Code (100K evolved)"),
    # Math datasets
    ("metamath", "MetaMath (395K math problems)"),
    ("gsm8k", "GSM8K (8.5K grade-school math)"),
    ("math", "MATH (12.5K competition math)"),
    # General instruction datasets  
    ("alpaca", "Alpaca (52K instruction-following)"),
    ("dolly", "Databricks Dolly (15K human-written)"),
    ("oasst", "OpenAssistant (161K conversations)"),
    # Function calling
    ("xlam", "xLAM (60K function calling)"),
    ("glaive", "Glaive (100K function calling)"),
    # Custom
    ("custom", "Custom JSONL file..."),
]

RAFT_PROMPT_PRESETS = [
    ("data/rlvr/humaneval_prompts.jsonl", "HumanEval (164 Python problems)"),
    ("data/rlvr/mbpp_prompts.jsonl", "MBPP (974 Python basics)"),
    ("data/rlvr/livecodebench_prompts.jsonl", "LiveCodeBench (Multi-language)"),
    ("custom", "Custom prompts file..."),
]


@dataclass
class SFTFormData:
    """SFT training form data."""
    model: str = "Qwen/Qwen2.5-Coder-3B"
    model_source: str = "preset"  # "preset" or "custom"
    custom_model: str = ""
    dataset: str = "codealpaca"
    dataset_source: str = "preset"  # "preset" or "custom"
    custom_dataset: str = ""
    output_dir: str = "models/sft_run"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    lora_rank: int = 32
    lora_alpha: int = 64
    max_length: int = 2048
    use_lora: bool = True
    gradient_checkpointing: bool = True


@dataclass
class RAFTFormData:
    """RAFT training form data."""
    model: str = "Qwen/Qwen2.5-Coder-3B"
    model_source: str = "preset"  # "preset" or "custom"
    custom_model: str = ""
    prompts: str = "data/rlvr/humaneval_prompts.jsonl"
    prompts_source: str = "preset"  # "preset" or "custom"
    custom_prompts: str = ""
    output_dir: str = "models/raft_run"
    cycles: int = 5
    samples_per_prompt: int = 8
    temperature: float = 0.7
    keep_percent: float = 0.5
    reward_threshold: float = 0.5
    learning_rate: float = 1e-5
    min_samples: int = 64
    max_new_tokens: int = 1024
    verifier: str = "humaneval"


class Training:
    """Training launch page component."""
    
    # Available presets
    RAFT_PRESETS = {
        "conservative": {
            "samples_per_prompt": 8,
            "keep_percent": 0.8,
            "reward_threshold": 0.5,
            "temperature": 0.7,
        },
        "aggressive": {
            "samples_per_prompt": 16,
            "keep_percent": 0.3,
            "reward_threshold": 0.7,
            "temperature": 0.8,
        },
        "custom": {},
    }
    
    VERIFIERS = [
        ("humaneval", "HumanEval (Python coding)"),
        ("mbpp", "MBPP (Python basics)"),
        ("livecodebench", "LiveCodeBench (Multi-language)"),
        ("math", "Math (Numerical verification)"),
        ("gcc", "GCC (C/C++ compilation)"),
        ("pytest", "Pytest (Python tests)"),
    ]
    
    def __init__(self):
        self.mode: Literal["sft", "raft"] = "sft"
        self.sft_data = SFTFormData()
        self.raft_data = RAFTFormData()
        self.selected_preset = "conservative"
        self.is_running = False
        self.training_service = TrainingService(state)
        
        # Container references for dynamic updates
        self._toggle_container = None
        self.form_container = None
    
    def render(self):
        """Render the training page."""
        with ui.column().classes('page-content w-full gap-6 p-6'):
            # Header
            with ui.row().classes('w-full items-center justify-between animate-in'):
                ui.label('Training').classes(
                    f'text-2xl font-bold text-[{COLORS["text_primary"]}]'
                )
            
            # Mode toggle - store reference for re-rendering
            self._toggle_container = ui.row().classes(
                f'w-full gap-2 p-2 rounded-xl bg-[{COLORS["bg_card"]}] '
                f'border border-[#2d343c] animate-in stagger-1'
            )
            with self._toggle_container:
                self._render_mode_buttons()
            
            # Main form container
            self.form_container = ui.column().classes('w-full gap-6')
            with self.form_container:
                self._render_form()
    
    def _render_mode_buttons(self):
        """Render the SFT/RAFT mode toggle buttons."""
        self._mode_button("SFT", "sft", "school")
        self._mode_button("RAFT", "raft", "autorenew")
    
    def _mode_button(self, label: str, mode: str, icon: str):
        """Render a mode toggle button."""
        is_active = self.mode == mode
        
        with ui.element('div').classes(
            f'flex-1 flex items-center justify-center gap-3 py-4 rounded-lg cursor-pointer transition-all '
            + (f'bg-[{COLORS["primary"]}]/20 border border-[{COLORS["primary"]}]' if is_active 
               else f'bg-transparent border border-transparent hover:bg-[{COLORS["bg_hover"]}]')
        ).on('click', lambda m=mode: self._set_mode(m)):
            ui.icon(icon, size='24px').classes(
                f'text-[{COLORS["primary"]}]' if is_active else f'text-[{COLORS["text_secondary"]}]'
            )
            ui.label(label).classes(
                f'text-base font-medium '
                + (f'text-[{COLORS["primary"]}]' if is_active else f'text-[{COLORS["text_secondary"]}]')
            )
    
    def _set_mode(self, mode: str):
        """Switch between SFT and RAFT mode."""
        self.mode = mode
        
        # Re-render BOTH toggle buttons and form
        self._toggle_container.clear()
        with self._toggle_container:
            self._render_mode_buttons()
        
        self.form_container.clear()
        with self.form_container:
            self._render_form()
    
    def _render_form(self):
        """Render the current form based on mode."""
        if self.mode == "sft":
            self._render_sft_form()
        else:
            self._render_raft_form()
    
    def _render_sft_form(self):
        """Render the SFT training form."""
        # Model & Dataset section
        with ui.column().classes(
            f'w-full gap-5 p-6 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c] animate-in stagger-2'
        ):
            self._section_header("Model & Dataset", "database")
            
            # Model selection
            with ui.column().classes('w-full gap-4'):
                self._render_model_selector(
                    "Base Model",
                    self.sft_data,
                    model_type="code"
                )
            
            # Dataset selection
            with ui.column().classes('w-full gap-4 mt-4'):
                self._render_dataset_selector()
            
            # Output directory
            with ui.column().classes('w-full gap-2 mt-4'):
                ui.label('Output Directory').classes(
                    f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                )
                with ui.row().classes('w-full gap-2'):
                    ui.input(value=self.sft_data.output_dir).classes('flex-1').props(
                        'outlined dense dark color=grey-7'
                    ).bind_value(self.sft_data, 'output_dir')
                    ui.button(icon='folder_open', on_click=lambda: self._browse_directory('sft_output')).props(
                        'flat dense'
                    ).classes(f'text-[{COLORS["text_muted"]}]').tooltip('Browse...')
        
        # Training Parameters section
        with ui.column().classes(
            f'w-full gap-5 p-6 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c] animate-in stagger-3'
        ):
            self._section_header("Training Parameters", "tune")
            
            with ui.row().classes('w-full gap-4 flex-wrap'):
                self._number_input("Epochs", self.sft_data.epochs, 
                                   lambda v: setattr(self.sft_data, 'epochs', int(v)),
                                   min_val=1, max_val=20)
                
                self._number_input("Batch Size", self.sft_data.batch_size,
                                   lambda v: setattr(self.sft_data, 'batch_size', int(v)),
                                   min_val=1, max_val=32)
                
                self._number_input("Learning Rate", self.sft_data.learning_rate,
                                   lambda v: setattr(self.sft_data, 'learning_rate', float(v)),
                                   format_val="2e-5")
                
                self._number_input("Max Length", self.sft_data.max_length,
                                   lambda v: setattr(self.sft_data, 'max_length', int(v)),
                                   min_val=512, max_val=8192)
        
        # LoRA Settings section
        with ui.column().classes(
            f'w-full gap-5 p-6 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c] animate-in stagger-4'
        ):
            with ui.row().classes('w-full items-center justify-between'):
                self._section_header("LoRA Settings", "layers")
                ui.switch(value=self.sft_data.use_lora).props(
                    f'color=primary'
                ).bind_value(self.sft_data, 'use_lora')
            
            with ui.row().classes('w-full gap-4 flex-wrap'):
                self._number_input("LoRA Rank", self.sft_data.lora_rank,
                                   lambda v: setattr(self.sft_data, 'lora_rank', int(v)),
                                   min_val=4, max_val=256)
                
                self._number_input("LoRA Alpha", self.sft_data.lora_alpha,
                                   lambda v: setattr(self.sft_data, 'lora_alpha', int(v)),
                                   min_val=4, max_val=512)
        
        # Launch button
        self._render_launch_button("Start SFT Training", self._launch_sft)
    
    def _render_raft_form(self):
        """Render the RAFT training form."""
        # Preset selector
        with ui.column().classes(
            f'w-full gap-4 p-6 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c] animate-in stagger-2'
        ):
            self._section_header("Training Preset", "auto_awesome")
            
            with ui.row().classes('w-full gap-3'):
                for preset_name in self.RAFT_PRESETS.keys():
                    is_selected = self.selected_preset == preset_name
                    with ui.element('div').classes(
                        f'flex-1 flex items-center justify-center py-3 rounded-lg cursor-pointer transition-all '
                        + (f'bg-[{COLORS["accent"]}]/20 border border-[{COLORS["accent"]}]' if is_selected
                           else f'bg-[{COLORS["bg_secondary"]}] border border-[#2d343c] hover:bg-[{COLORS["bg_hover"]}]')
                    ).on('click', lambda p=preset_name: self._apply_preset(p)):
                        ui.label(preset_name.capitalize()).classes(
                            f'text-sm font-medium '
                            + (f'text-[{COLORS["accent"]}]' if is_selected else f'text-[{COLORS["text_secondary"]}]')
                        )
        
        # Model & Prompts section
        with ui.column().classes(
            f'w-full gap-5 p-6 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c] animate-in stagger-3'
        ):
            self._section_header("Model & Data", "database")
            
            # Model selection
            with ui.column().classes('w-full gap-4'):
                self._render_model_selector(
                    "Base Model / Checkpoint",
                    self.raft_data,
                    model_type="code"
                )
            
            with ui.row().classes('w-full gap-4 flex-wrap mt-4'):
                # Verifier selection
                with ui.column().classes('flex-1 min-w-[280px] gap-2'):
                    ui.label('Verifier').classes(
                        f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                    )
                    ui.select(
                        options={k: v for k, v in self.VERIFIERS},
                        value=self.raft_data.verifier
                    ).classes('w-full').props(
                        'outlined dense dark color=grey-7'
                    ).bind_value(self.raft_data, 'verifier')
            
            # Prompts file selection
            with ui.column().classes('w-full gap-4 mt-4'):
                self._render_prompts_selector()
            
            # Output directory
            with ui.column().classes('w-full gap-2 mt-4'):
                ui.label('Output Directory').classes(
                    f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                )
                with ui.row().classes('w-full gap-2'):
                    ui.input(value=self.raft_data.output_dir).classes('flex-1').props(
                        'outlined dense dark color=grey-7'
                    ).bind_value(self.raft_data, 'output_dir')
                    ui.button(icon='folder_open', on_click=lambda: self._browse_directory('raft_output')).props(
                        'flat dense'
                    ).classes(f'text-[{COLORS["text_muted"]}]').tooltip('Browse...')
        
        # RAFT Parameters section
        with ui.column().classes(
            f'w-full gap-5 p-6 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c] animate-in stagger-4'
        ):
            self._section_header("RAFT Parameters", "settings")
            
            with ui.row().classes('w-full gap-4 flex-wrap'):
                self._number_input("Cycles", self.raft_data.cycles,
                                   lambda v: setattr(self.raft_data, 'cycles', int(v)),
                                   min_val=1, max_val=20)
                
                self._number_input("Samples/Prompt", self.raft_data.samples_per_prompt,
                                   lambda v: setattr(self.raft_data, 'samples_per_prompt', int(v)),
                                   min_val=1, max_val=32)
                
                self._number_input("Temperature", self.raft_data.temperature,
                                   lambda v: setattr(self.raft_data, 'temperature', float(v)),
                                   min_val=0.0, max_val=2.0)
                
                self._number_input("Keep %", self.raft_data.keep_percent,
                                   lambda v: setattr(self.raft_data, 'keep_percent', float(v)),
                                   min_val=0.1, max_val=1.0)
            
            with ui.row().classes('w-full gap-4 flex-wrap'):
                self._number_input("Reward Threshold", self.raft_data.reward_threshold,
                                   lambda v: setattr(self.raft_data, 'reward_threshold', float(v)),
                                   min_val=0.0, max_val=1.0)
                
                self._number_input("Min Samples", self.raft_data.min_samples,
                                   lambda v: setattr(self.raft_data, 'min_samples', int(v)),
                                   min_val=8, max_val=512)
                
                self._number_input("Learning Rate", self.raft_data.learning_rate,
                                   lambda v: setattr(self.raft_data, 'learning_rate', float(v)),
                                   format_val="1e-5")
                
                self._number_input("Max New Tokens", self.raft_data.max_new_tokens,
                                   lambda v: setattr(self.raft_data, 'max_new_tokens', int(v)),
                                   min_val=256, max_val=4096)
        
        # Launch button
        self._render_launch_button("Start RAFT Training", self._launch_raft)
    
    def _render_model_selector(self, label: str, data_obj, model_type: str = "code"):
        """Render model selection with dropdown + custom option."""
        ui.label(label).classes(
            f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
        )
        
        models = RECOMMENDED_MODELS.get(model_type, RECOMMENDED_MODELS["code"])
        
        # Add custom option
        model_options = {k: v for k, v in models}
        model_options["custom"] = "Custom HuggingFace model or local path..."
        
        with ui.row().classes('w-full gap-2'):
            # Main dropdown
            model_select = ui.select(
                options=model_options,
                value=data_obj.model if data_obj.model_source == "preset" else "custom"
            ).classes('flex-1').props('outlined dense dark color=grey-7')
            
            def on_model_change(e):
                val = e.value if hasattr(e, 'value') else e.args
                if val == "custom":
                    data_obj.model_source = "custom"
                else:
                    data_obj.model_source = "preset"
                    data_obj.model = val
                # Refresh form to show/hide custom input
                self.form_container.clear()
                with self.form_container:
                    self._render_form()
            
            model_select.on('update:model-value', on_model_change)
            
            # Browse button for local models
            ui.button(icon='folder_open', on_click=lambda: self._browse_model(data_obj)).props(
                'flat dense'
            ).classes(f'text-[{COLORS["text_muted"]}]').tooltip('Browse local model...')
        
        # Show custom input if custom is selected
        if data_obj.model_source == "custom":
            with ui.row().classes('w-full gap-2 mt-2'):
                ui.input(
                    placeholder='Enter HuggingFace model ID or local path...',
                    value=data_obj.custom_model or data_obj.model
                ).classes('flex-1').props('outlined dense dark color=grey-7').bind_value(data_obj, 'custom_model')
    
    def _render_dataset_selector(self):
        """Render dataset selection with dropdown + custom option."""
        ui.label('Dataset').classes(
            f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
        )
        
        dataset_options = {k: v for k, v in SFT_DATASETS}
        
        with ui.row().classes('w-full gap-2'):
            dataset_select = ui.select(
                options=dataset_options,
                value=self.sft_data.dataset if self.sft_data.dataset_source == "preset" else "custom"
            ).classes('flex-1').props('outlined dense dark color=grey-7')
            
            def on_dataset_change(e):
                val = e.value if hasattr(e, 'value') else e.args
                if val == "custom":
                    self.sft_data.dataset_source = "custom"
                else:
                    self.sft_data.dataset_source = "preset"
                    self.sft_data.dataset = val
                # Refresh form
                self.form_container.clear()
                with self.form_container:
                    self._render_form()
            
            dataset_select.on('update:model-value', on_dataset_change)
            
            ui.button(icon='folder_open', on_click=self._browse_dataset).props(
                'flat dense'
            ).classes(f'text-[{COLORS["text_muted"]}]').tooltip('Browse JSONL file...')
        
        # Show custom input if custom is selected
        if self.sft_data.dataset_source == "custom":
            with ui.row().classes('w-full gap-2 mt-2'):
                ui.input(
                    placeholder='Path to JSONL file (e.g., data/my_dataset.jsonl)',
                    value=self.sft_data.custom_dataset
                ).classes('flex-1').props('outlined dense dark color=grey-7').bind_value(self.sft_data, 'custom_dataset')
    
    def _render_prompts_selector(self):
        """Render prompts file selection with dropdown + custom option."""
        ui.label('Prompts File').classes(
            f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
        )
        
        prompts_options = {k: v for k, v in RAFT_PROMPT_PRESETS}
        
        with ui.row().classes('w-full gap-2'):
            prompts_select = ui.select(
                options=prompts_options,
                value=self.raft_data.prompts if self.raft_data.prompts_source == "preset" else "custom"
            ).classes('flex-1').props('outlined dense dark color=grey-7')
            
            def on_prompts_change(e):
                val = e.value if hasattr(e, 'value') else e.args
                if val == "custom":
                    self.raft_data.prompts_source = "custom"
                else:
                    self.raft_data.prompts_source = "preset"
                    self.raft_data.prompts = val
                # Refresh form
                self.form_container.clear()
                with self.form_container:
                    self._render_form()
            
            prompts_select.on('update:model-value', on_prompts_change)
            
            ui.button(icon='folder_open', on_click=self._browse_prompts).props(
                'flat dense'
            ).classes(f'text-[{COLORS["text_muted"]}]').tooltip('Browse JSONL file...')
        
        # Show custom input if custom is selected
        if self.raft_data.prompts_source == "custom":
            with ui.row().classes('w-full gap-2 mt-2'):
                ui.input(
                    placeholder='Path to prompts JSONL file',
                    value=self.raft_data.custom_prompts
                ).classes('flex-1').props('outlined dense dark color=grey-7').bind_value(self.raft_data, 'custom_prompts')
    
    def _browse_model(self, data_obj):
        """Open file picker for local model."""
        self._open_file_picker(
            title="Select Local Model Directory",
            path_type="directory",
            start_path="models/",
            callback=lambda path: self._set_custom_model(data_obj, path)
        )
    
    def _browse_dataset(self):
        """Open file picker for dataset."""
        self._open_file_picker(
            title="Select Dataset File",
            path_type="file",
            file_filter="*.jsonl",
            start_path="data/",
            callback=self._set_custom_dataset
        )
    
    def _browse_prompts(self):
        """Open file picker for prompts file."""
        self._open_file_picker(
            title="Select Prompts File",
            path_type="file",
            file_filter="*.jsonl",
            start_path="data/",
            callback=self._set_custom_prompts
        )
    
    def _browse_directory(self, target: str):
        """Open directory picker for output."""
        self._open_file_picker(
            title="Select Output Directory",
            path_type="directory",
            start_path="models/",
            callback=lambda path: self._set_output_dir(target, path)
        )
    
    def _set_custom_model(self, data_obj, path: str):
        """Set custom model path."""
        data_obj.model_source = "custom"
        data_obj.custom_model = path
        data_obj.model = path
        self.form_container.clear()
        with self.form_container:
            self._render_form()
    
    def _set_custom_dataset(self, path: str):
        """Set custom dataset path."""
        self.sft_data.dataset_source = "custom"
        self.sft_data.custom_dataset = path
        self.sft_data.dataset = path
        self.form_container.clear()
        with self.form_container:
            self._render_form()
    
    def _set_custom_prompts(self, path: str):
        """Set custom prompts path."""
        self.raft_data.prompts_source = "custom"
        self.raft_data.custom_prompts = path
        self.raft_data.prompts = path
        self.form_container.clear()
        with self.form_container:
            self._render_form()
    
    def _set_output_dir(self, target: str, path: str):
        """Set output directory."""
        if target == "sft_output":
            self.sft_data.output_dir = path
        elif target == "raft_output":
            self.raft_data.output_dir = path
        self.form_container.clear()
        with self.form_container:
            self._render_form()
    
    def _open_file_picker(self, title: str, path_type: str, callback, file_filter: str = None, start_path: str = "."):
        """Open a file/directory picker dialog."""
        from ui.components.file_picker import FilePicker
        FilePicker(
            title=title,
            path_type=path_type,
            file_filter=file_filter,
            start_path=start_path,
            on_select=callback
        ).open()
    
    def _section_header(self, title: str, icon: str):
        """Render a form section header."""
        with ui.row().classes('items-center gap-2 mb-2'):
            ui.icon(icon, size='18px').classes(f'text-[{COLORS["accent"]}]')
            ui.label(title).classes(
                f'text-sm font-semibold text-[{COLORS["text_primary"]}]'
            )
    
    def _number_input(self, label: str, value, on_change, min_val=None, max_val=None, format_val=None):
        """Render a number input field."""
        with ui.column().classes('flex-1 min-w-[140px] gap-2'):
            ui.label(label).classes(
                f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
            )
            display_value = format_val if format_val else str(value)
            inp = ui.input(value=display_value).classes('w-full').props(
                'outlined dense dark color=grey-7'
            )
            inp.on('update:model-value', lambda e: on_change(e.args))
    
    def _render_launch_button(self, label: str, on_click):
        """Render the launch training button."""
        with ui.row().classes('w-full justify-end pt-4'):
            with ui.button(on_click=on_click).props('unelevated').classes(
                f'btn-hover px-8 py-3 bg-[{COLORS["primary"]}] text-white rounded-lg'
            ):
                if self.is_running:
                    ui.spinner('dots', size='20px').classes('mr-2')
                else:
                    ui.icon('play_arrow', size='20px').classes('mr-2')
                ui.label(label).classes('text-sm font-medium')
    
    def _apply_preset(self, preset_name: str):
        """Apply a RAFT preset configuration."""
        self.selected_preset = preset_name
        preset = self.RAFT_PRESETS.get(preset_name, {})
        
        for key, value in preset.items():
            if hasattr(self.raft_data, key):
                setattr(self.raft_data, key, value)
        
        # Notify before clearing (context is still valid)
        ui.notify(f'Applied "{preset_name}" preset', type='positive', timeout=1500)
        
        # Re-render form to show updated values
        self.form_container.clear()
        with self.form_container:
            self._render_form()
    
    def _get_effective_model(self, data_obj) -> str:
        """Get the effective model path (handles custom vs preset)."""
        if data_obj.model_source == "custom" and data_obj.custom_model:
            return data_obj.custom_model
        return data_obj.model
    
    def _get_effective_dataset(self) -> str:
        """Get the effective dataset path (handles custom vs preset)."""
        if self.sft_data.dataset_source == "custom" and self.sft_data.custom_dataset:
            return self.sft_data.custom_dataset
        return self.sft_data.dataset
    
    def _get_effective_prompts(self) -> str:
        """Get the effective prompts path (handles custom vs preset)."""
        if self.raft_data.prompts_source == "custom" and self.raft_data.custom_prompts:
            return self.raft_data.custom_prompts
        return self.raft_data.prompts
    
    async def _launch_sft(self):
        """Launch SFT training."""
        if self.is_running:
            return
        
        self.is_running = True
        
        try:
            model = self._get_effective_model(self.sft_data)
            dataset = self._get_effective_dataset()
            
            # Launch actual training subprocess via TrainingService
            job_id = await self.training_service.launch_sft(
                model=model,
                dataset=dataset,
                output_dir=self.sft_data.output_dir,
                epochs=self.sft_data.epochs,
                batch_size=self.sft_data.batch_size,
                learning_rate=self.sft_data.learning_rate,
                use_lora=self.sft_data.use_lora,
                lora_rank=self.sft_data.lora_rank,
            )
            
            notify_job_started(f"SFT: {Path(dataset).stem if '/' in dataset else dataset}")
            
            # Navigate to monitor
            ui.navigate.to(f'/monitor/{job_id}')
            
        except Exception as e:
            notify_job_failed("SFT Training", str(e))
        finally:
            self.is_running = False
    
    async def _launch_raft(self):
        """Launch RAFT training."""
        if self.is_running:
            return
        
        self.is_running = True
        
        try:
            model = self._get_effective_model(self.raft_data)
            prompts = self._get_effective_prompts()
            
            # Launch actual training subprocess via TrainingService
            job_id = await self.training_service.launch_raft(
                model=model,
                prompts=prompts,
                output_dir=self.raft_data.output_dir,
                verifier=self.raft_data.verifier,
                cycles=self.raft_data.cycles,
                samples_per_prompt=self.raft_data.samples_per_prompt,
                temperature=self.raft_data.temperature,
                keep_percent=self.raft_data.keep_percent,
                reward_threshold=self.raft_data.reward_threshold,
            )
            
            notify_job_started(f"RAFT: {self.raft_data.verifier}")
            
            # Navigate to monitor
            ui.navigate.to(f'/monitor/{job_id}')
            
        except Exception as e:
            notify_job_failed("RAFT Training", str(e))
        finally:
            self.is_running = False
