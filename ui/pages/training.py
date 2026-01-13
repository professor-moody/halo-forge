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
# All Available Datasets
# =============================================================================

SFT_DATASETS = [
    # --- Code Datasets ---
    ("codealpaca", "CodeAlpaca (20K code instructions)"),
    ("evol_instruct_code", "Evol-Instruct-Code (100K evolved)"),
    ("codeforces_cpp", "CodeForces C++ (~5000 competitive)"),
    ("codeforces_python", "CodeForces Python (~1000 competitive)"),
    ("codeforces_rust", "CodeForces Rust (~500 competitive)"),
    # --- Python Benchmarks ---
    ("mbpp", "MBPP (974 Python basics)"),
    ("humaneval", "HumanEval (164 Python problems)"),
    ("humaneval_plus", "HumanEval+ (164 + extended tests)"),
    # --- Math Datasets ---
    ("metamath", "MetaMath (395K math problems)"),
    ("gsm8k", "GSM8K (8.5K grade-school math)"),
    ("math", "MATH (12.5K competition math)"),
    # --- General Instruction ---
    ("alpaca", "Alpaca (52K instruction-following)"),
    ("dolly", "Databricks Dolly (15K human-written)"),
    ("oasst", "OpenAssistant (161K conversations)"),
    # --- Function Calling ---
    ("xlam", "xLAM (60K function calling)"),
    ("glaive", "Glaive (100K function calling)"),
    # --- Multi-language ---
    ("livecodebench", "LiveCodeBench (contamination-free)"),
    # --- Custom ---
    ("custom", "Custom JSONL file..."),
]

# =============================================================================
# RAFT Prompt Files (available in data/rlvr/)
# =============================================================================

RAFT_PROMPT_PRESETS = [
    # HumanEval variants
    ("data/rlvr/humaneval_prompts.jsonl", "HumanEval Prompts (164 Python)"),
    ("data/rlvr/humaneval_full.jsonl", "HumanEval Full (with solutions)"),
    ("data/rlvr/humaneval_validation.jsonl", "HumanEval Validation"),
    # MBPP variants
    ("data/rlvr/mbpp_train_prompts.jsonl", "MBPP Train Prompts"),
    ("data/rlvr/mbpp_train_full.jsonl", "MBPP Train Full"),
    ("data/rlvr/mbpp_sanitized_prompts.jsonl", "MBPP Sanitized Prompts"),
    ("data/rlvr/mbpp_sanitized_full.jsonl", "MBPP Sanitized Full"),
    ("data/rlvr/mbpp_validation.jsonl", "MBPP Validation"),
    # Custom
    ("custom", "Custom prompts file..."),
]

# =============================================================================
# All Available Verifiers (grouped by category)
# =============================================================================

VERIFIERS = [
    # --- Python Test Verifiers ---
    ("humaneval", "HumanEval (Python test execution)"),
    ("mbpp", "MBPP (Python test execution)"),
    ("pytest", "Pytest (generic Python tests)"),
    ("unittest", "Unittest (Python unittest)"),
    # --- Compile Verifiers (C/C++) ---
    ("gcc", "GCC (C/C++ compile)"),
    ("clang", "Clang (C/C++ compile)"),
    ("mingw", "MinGW (Windows cross-compile)"),
    ("msvc", "MSVC Remote (Windows via SSH)"),
    # --- Execution Verifiers (compile + run) ---
    ("execution", "Execution (compile + test I/O)"),
    ("gcc_execution", "GCC Execution (compile + run)"),
    ("clang_execution", "Clang Execution (compile + run)"),
    ("mingw_execution", "MinGW Execution (cross-compile + run)"),
    # --- Other Language Verifiers ---
    ("rust", "Rust (Cargo build + run)"),
    ("go", "Go (go build + run)"),
    ("dotnet", "DotNet (C# compile)"),
    ("powershell", "PowerShell (syntax check)"),
    # --- Multi-Language ---
    ("auto", "Auto-detect (route to appropriate verifier)"),
    # --- Math ---
    ("math", "Math (numerical answer verification)"),
    # --- Chained ---
    ("chained", "Chained (multi-stage verification)"),
]


@dataclass
class SFTFormData:
    """SFT training form data."""
    # Model selection
    model: str = "Qwen/Qwen2.5-Coder-3B"
    model_source: str = "preset"  # "preset" or "custom"
    custom_model: str = ""
    
    # Dataset selection
    dataset: str = "codealpaca"
    dataset_source: str = "preset"  # "preset" or "custom"
    custom_dataset: str = ""
    
    # Output
    output_dir: str = "models/sft_run"
    
    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    
    # LoRA configuration
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Data options
    max_length: int = 2048
    validation_split: float = 0.05
    max_samples: Optional[int] = None
    
    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 250
    
    # Early stopping
    early_stopping_patience: int = 5
    
    # Hardware
    gradient_checkpointing: bool = True


@dataclass
class RAFTFormData:
    """RAFT training form data."""
    # Model selection
    model: str = "Qwen/Qwen2.5-Coder-3B"
    model_source: str = "preset"  # "preset" or "custom"
    custom_model: str = ""
    
    # Checkpoint resume (optional SFT checkpoint)
    checkpoint: str = ""
    use_checkpoint: bool = False
    
    # Prompts selection
    prompts: str = "data/rlvr/humaneval_prompts.jsonl"
    prompts_source: str = "preset"  # "preset" or "custom"
    custom_prompts: str = ""
    
    # Output
    output_dir: str = "models/raft_run"
    
    # RAFT parameters
    cycles: int = 5
    samples_per_prompt: int = 8
    temperature: float = 0.7
    keep_percent: float = 0.5
    reward_threshold: float = 0.5
    min_samples: int = 64
    max_new_tokens: int = 1024
    verifier: str = "humaneval"
    
    # Learning rate schedule
    learning_rate: float = 1e-5
    lr_decay: float = 0.85
    min_lr: float = 1e-6
    
    # Advanced strategies
    curriculum: str = "none"  # none, complexity, progressive, adaptive, historical
    curriculum_stats_path: str = ""  # Path to stats JSON for historical curriculum
    curriculum_start: float = 0.2   # Progressive: start with this fraction
    curriculum_increment: float = 0.2  # Progressive: add this fraction per cycle
    reward_shaping: str = "fixed"  # fixed, annealing, adaptive, warmup
    
    # Generation options
    system_prompt: str = "You are an expert programmer."
    
    # Hardware/experimental
    experimental_attention: bool = False


class Training:
    """Training launch page component."""
    
    # SFT presets
    SFT_PRESETS = {
        "quick_test": {
            "epochs": 1,
            "max_samples": 100,
            "save_steps": 50,
            "eval_steps": 25,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
        },
        "standard": {
            "epochs": 3,
            "batch_size": 2,
            "gradient_accumulation_steps": 16,
            "learning_rate": 2e-4,
            "warmup_ratio": 0.03,
            "lora_rank": 16,
            "lora_alpha": 32,
        },
        "quality": {
            "epochs": 5,
            "learning_rate": 1e-4,
            "warmup_ratio": 0.1,
            "weight_decay": 0.05,
            "early_stopping_patience": 10,
            "lora_rank": 32,
            "lora_alpha": 64,
        },
        "large_model": {
            "batch_size": 1,
            "gradient_accumulation_steps": 32,
            "gradient_checkpointing": True,
            "max_length": 1024,
            "lora_rank": 8,
            "lora_alpha": 16,
        },
        "custom": {},
    }
    
    # RAFT presets
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
    
    def __init__(self):
        self.mode: Literal["sft", "raft"] = "sft"
        self.sft_data = SFTFormData()
        self.raft_data = RAFTFormData()
        self.selected_sft_preset = "standard"
        self.selected_raft_preset = "conservative"
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
        """Render the SFT training form with presets and organized sections."""
        # Presets section
        with ui.column().classes(
            f'w-full gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c] animate-in stagger-2'
        ):
            self._section_header("Presets", "flash_on")
            with ui.row().classes('w-full gap-2 flex-wrap'):
                for preset_name in self.SFT_PRESETS.keys():
                    is_selected = self.selected_sft_preset == preset_name
                    with ui.button().props('flat').classes(
                        'px-4 py-2 rounded-lg transition-all '
                        + (f'bg-[{COLORS["accent"]}]/20 border border-[{COLORS["accent"]}]' if is_selected
                           else f'bg-[{COLORS["bg_secondary"]}] border border-[#2d343c] hover:bg-[{COLORS["bg_hover"]}]')
                    ).on('click', lambda p=preset_name: self._apply_sft_preset(p)):
                        ui.label(preset_name.replace('_', ' ').title()).classes(
                            f'text-sm font-medium '
                            + (f'text-[{COLORS["accent"]}]' if is_selected else f'text-[{COLORS["text_secondary"]}]')
                        )
        
        # Model & Dataset section
        with ui.column().classes(
            f'w-full gap-5 p-6 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c] animate-in stagger-3'
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
            f'border border-[#2d343c] animate-in stagger-4'
        ):
            self._section_header("Training Parameters", "tune")
            
            with ui.row().classes('w-full gap-4 flex-wrap'):
                self._number_input("Epochs", self.sft_data.epochs, 
                                   lambda v: setattr(self.sft_data, 'epochs', int(v)),
                                   min_val=1, max_val=20)
                
                self._number_input("Batch Size", self.sft_data.batch_size,
                                   lambda v: setattr(self.sft_data, 'batch_size', int(v)),
                                   min_val=1, max_val=32)
                
                self._number_input("Grad Accum", self.sft_data.gradient_accumulation_steps,
                                   lambda v: setattr(self.sft_data, 'gradient_accumulation_steps', int(v)),
                                   min_val=1, max_val=64)
                
                self._number_input("Learning Rate", self.sft_data.learning_rate,
                                   lambda v: setattr(self.sft_data, 'learning_rate', float(v)),
                                   format_val="2e-4")
            
            with ui.row().classes('w-full gap-4 flex-wrap mt-2'):
                self._number_input("Warmup Ratio", self.sft_data.warmup_ratio,
                                   lambda v: setattr(self.sft_data, 'warmup_ratio', float(v)),
                                   format_val="0.03")
                
                self._number_input("Max Grad Norm", self.sft_data.max_grad_norm,
                                   lambda v: setattr(self.sft_data, 'max_grad_norm', float(v)),
                                   format_val="0.3")
                
                self._number_input("Max Length", self.sft_data.max_length,
                                   lambda v: setattr(self.sft_data, 'max_length', int(v)),
                                   min_val=512, max_val=8192)
            
            # Show effective batch size
            effective_batch = self.sft_data.batch_size * self.sft_data.gradient_accumulation_steps
            ui.label(f'Effective batch size: {effective_batch}').classes(
                f'text-xs text-[{COLORS["text_muted"]}] mt-2'
            )
        
        # LoRA Settings section (collapsible)
        with ui.expansion(
            text='LoRA Configuration',
            icon='layers',
            value=True  # Open by default
        ).classes(
            f'w-full rounded-xl bg-[{COLORS["bg_card"]}] border border-[#2d343c] animate-in stagger-5'
        ).props('dense dark'):
            with ui.column().classes('w-full gap-4 p-4'):
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label('Enable LoRA').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                    ui.switch(value=self.sft_data.use_lora).props(
                        f'color=primary'
                    ).bind_value(self.sft_data, 'use_lora')
                
                with ui.row().classes('w-full gap-4 flex-wrap'):
                    self._number_input("Rank", self.sft_data.lora_rank,
                                       lambda v: setattr(self.sft_data, 'lora_rank', int(v)),
                                       min_val=4, max_val=256)
                    
                    self._number_input("Alpha", self.sft_data.lora_alpha,
                                       lambda v: setattr(self.sft_data, 'lora_alpha', int(v)),
                                       min_val=4, max_val=512)
                    
                    self._number_input("Dropout", self.sft_data.lora_dropout,
                                       lambda v: setattr(self.sft_data, 'lora_dropout', float(v)),
                                       format_val="0.05")
        
        # Advanced Settings section (collapsible)
        with ui.expansion(
            text='Advanced Settings',
            icon='settings',
            value=False  # Collapsed by default
        ).classes(
            f'w-full rounded-xl bg-[{COLORS["bg_card"]}] border border-[#2d343c] animate-in stagger-6'
        ).props('dense dark'):
            with ui.column().classes('w-full gap-4 p-4'):
                with ui.row().classes('w-full gap-4 flex-wrap'):
                    self._number_input("Weight Decay", self.sft_data.weight_decay,
                                       lambda v: setattr(self.sft_data, 'weight_decay', float(v)),
                                       format_val="0.01")
                    
                    self._number_input("Val Split", self.sft_data.validation_split,
                                       lambda v: setattr(self.sft_data, 'validation_split', float(v)),
                                       format_val="0.05")
                    
                    self._number_input("Max Samples", self.sft_data.max_samples or 0,
                                       lambda v: setattr(self.sft_data, 'max_samples', int(v) if int(v) > 0 else None),
                                       min_val=0)
                
                with ui.row().classes('w-full gap-4 flex-wrap mt-2'):
                    self._number_input("Save Steps", self.sft_data.save_steps,
                                       lambda v: setattr(self.sft_data, 'save_steps', int(v)),
                                       min_val=50, max_val=5000)
                    
                    self._number_input("Eval Steps", self.sft_data.eval_steps,
                                       lambda v: setattr(self.sft_data, 'eval_steps', int(v)),
                                       min_val=25, max_val=2500)
                    
                    self._number_input("Early Stop", self.sft_data.early_stopping_patience,
                                       lambda v: setattr(self.sft_data, 'early_stopping_patience', int(v)),
                                       min_val=1, max_val=20)
                
                # Hardware options
                with ui.row().classes('w-full items-center gap-4 mt-2'):
                    ui.label('Gradient Checkpointing').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                    ui.switch(value=self.sft_data.gradient_checkpointing).props(
                        f'color=primary'
                    ).bind_value(self.sft_data, 'gradient_checkpointing')
        
        # Launch button
        self._render_launch_button("Start SFT Training", self._launch_sft)
    
    def _apply_sft_preset(self, preset_name: str):
        """Apply an SFT preset configuration."""
        self.selected_sft_preset = preset_name
        preset = self.SFT_PRESETS.get(preset_name, {})
        
        for key, value in preset.items():
            if hasattr(self.sft_data, key):
                setattr(self.sft_data, key, value)
        
        # Notify before clearing
        ui.notify(f'Applied "{preset_name.replace("_", " ").title()}" preset', type='positive', timeout=1500)
        
        # Re-render form
        self.form_container.clear()
        with self.form_container:
            self._render_form()
    
    def _render_raft_form(self):
        """Render the RAFT training form with comprehensive options."""
        # Preset selector
        with ui.column().classes(
            f'w-full gap-4 p-6 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c] animate-in stagger-2'
        ):
            self._section_header("Training Preset", "auto_awesome")
            
            with ui.row().classes('w-full gap-3'):
                for preset_name in self.RAFT_PRESETS.keys():
                    is_selected = self.selected_raft_preset == preset_name
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
                    "Base Model",
                    self.raft_data,
                    model_type="code"
                )
            
            # Checkpoint resume (optional)
            with ui.column().classes('w-full gap-2 mt-4'):
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label('Resume from SFT Checkpoint').classes(
                        f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                    )
                    ui.switch(value=self.raft_data.use_checkpoint).props(
                        'color=primary'
                    ).bind_value(self.raft_data, 'use_checkpoint')
                
                if self.raft_data.use_checkpoint:
                    with ui.row().classes('w-full gap-2'):
                        ui.input(
                            value=self.raft_data.checkpoint,
                            placeholder='Path to SFT checkpoint (e.g., models/sft/final_model)'
                        ).classes('flex-1').props(
                            'outlined dense dark color=grey-7'
                        ).bind_value(self.raft_data, 'checkpoint')
                        ui.button(icon='folder_open', on_click=lambda: self._browse_checkpoint()).props(
                            'flat dense'
                        ).classes(f'text-[{COLORS["text_muted"]}]').tooltip('Browse checkpoint...')
            
            with ui.row().classes('w-full gap-4 flex-wrap mt-4'):
                # Verifier selection
                with ui.column().classes('flex-1 min-w-[280px] gap-2'):
                    ui.label('Verifier').classes(
                        f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                    )
                    verifier_options = {k: v for k, v in VERIFIERS}
                    ui.select(
                        options=verifier_options,
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
                
                self._number_input("Max New Tokens", self.raft_data.max_new_tokens,
                                   lambda v: setattr(self.raft_data, 'max_new_tokens', int(v)),
                                   min_val=256, max_val=4096)
            
            # Learning rate section
            with ui.row().classes('w-full gap-4 flex-wrap mt-2'):
                self._number_input("Learning Rate", self.raft_data.learning_rate,
                                   lambda v: setattr(self.raft_data, 'learning_rate', float(v)),
                                   format_val="1e-5")
                
                self._number_input("LR Decay", self.raft_data.lr_decay,
                                   lambda v: setattr(self.raft_data, 'lr_decay', float(v)),
                                   format_val="0.85")
                
                self._number_input("Min LR", self.raft_data.min_lr,
                                   lambda v: setattr(self.raft_data, 'min_lr', float(v)),
                                   format_val="1e-6")
            
            ui.label(f'LR decays by {self.raft_data.lr_decay:.0%} each cycle, floor at {self.raft_data.min_lr:.0e}').classes(
                f'text-xs text-[{COLORS["text_muted"]}] mt-1'
            )
        
        # Generation Settings section (collapsible)
        with ui.expansion(
            text='Generation Settings',
            icon='edit',
            value=False
        ).classes(
            f'w-full rounded-xl bg-[{COLORS["bg_card"]}] border border-[#2d343c] animate-in stagger-5'
        ).props('dense dark'):
            with ui.column().classes('w-full gap-4 p-4'):
                ui.label('System Prompt').classes(
                    f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                )
                ui.textarea(
                    value=self.raft_data.system_prompt
                ).classes('w-full').props(
                    'outlined dense dark color=grey-7 rows=3'
                ).bind_value(self.raft_data, 'system_prompt')
        
        # Advanced Strategies section (collapsible)
        with ui.expansion(
            text='Advanced Strategies',
            icon='psychology',
            value=False
        ).classes(
            f'w-full rounded-xl bg-[{COLORS["bg_card"]}] border border-[#2d343c] animate-in stagger-6'
        ).props('dense dark'):
            with ui.column().classes('w-full gap-4 p-4'):
                with ui.row().classes('w-full gap-4 flex-wrap'):
                    # Curriculum learning
                    with ui.column().classes('flex-1 min-w-[200px] gap-2'):
                        ui.label('Curriculum').classes(
                            f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                        )
                        
                        def on_curriculum_change(e):
                            self.raft_data.curriculum = e.value
                            # Refresh form to show/hide conditional fields
                            self.form_container.clear()
                            with self.form_container:
                                self._render_form()
                        
                        ui.select(
                            options={
                                'none': 'None (default)',
                                'complexity': 'Complexity (easy → hard)',
                                'progressive': 'Progressive (gradual)',
                                'adaptive': 'Adaptive (performance-based)',
                                'historical': 'Historical (from past runs)',
                            },
                            value=self.raft_data.curriculum,
                            on_change=on_curriculum_change
                        ).classes('w-full').props(
                            'outlined dense dark color=grey-7'
                        )
                    
                    # Reward shaping
                    with ui.column().classes('flex-1 min-w-[200px] gap-2'):
                        ui.label('Reward Shaping').classes(
                            f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                        )
                        ui.select(
                            options={
                                'fixed': 'Fixed (default)',
                                'annealing': 'Annealing',
                                'adaptive': 'Adaptive',
                                'warmup': 'Warmup',
                            },
                            value=self.raft_data.reward_shaping
                        ).classes('w-full').props(
                            'outlined dense dark color=grey-7'
                        ).bind_value(self.raft_data, 'reward_shaping')
                
                # Conditional fields based on curriculum strategy
                if self.raft_data.curriculum == 'historical':
                    with ui.column().classes('w-full gap-2 mt-2'):
                        ui.label('Historical Stats File').classes(
                            f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                        )
                        with ui.row().classes('w-full gap-2'):
                            ui.input(
                                value=self.raft_data.curriculum_stats_path,
                                placeholder='Path to curriculum_stats.json from previous run'
                            ).classes('flex-1').props(
                                'outlined dense dark color=grey-7'
                            ).bind_value(self.raft_data, 'curriculum_stats_path')
                            ui.button(icon='folder_open', on_click=lambda: self._browse_curriculum_stats()).props(
                                'flat dense'
                            ).classes(f'text-[{COLORS["text_muted"]}]').tooltip('Browse...')
                        ui.label('Uses success rates from previous training runs to order prompts').classes(
                            f'text-xs text-[{COLORS["text_muted"]}]'
                        )
                
                elif self.raft_data.curriculum == 'progressive':
                    with ui.row().classes('w-full gap-4 mt-2'):
                        with ui.column().classes('flex-1 gap-2'):
                            ui.label('Start Fraction').classes(
                                f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                            )
                            ui.number(
                                value=self.raft_data.curriculum_start,
                                min=0.1, max=0.9, step=0.1,
                                format='%.1f'
                            ).classes('w-full').props(
                                'outlined dense dark color=grey-7'
                            ).bind_value(self.raft_data, 'curriculum_start')
                        
                        with ui.column().classes('flex-1 gap-2'):
                            ui.label('Increment per Cycle').classes(
                                f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                            )
                            ui.number(
                                value=self.raft_data.curriculum_increment,
                                min=0.05, max=0.5, step=0.05,
                                format='%.2f'
                            ).classes('w-full').props(
                                'outlined dense dark color=grey-7'
                            ).bind_value(self.raft_data, 'curriculum_increment')
                    
                    # Calculate when 100% will be reached
                    cycles_to_full = int((1.0 - self.raft_data.curriculum_start) / self.raft_data.curriculum_increment) + 1
                    ui.label(f'Reaches 100% of prompts at cycle {cycles_to_full}').classes(
                        f'text-xs text-[{COLORS["text_muted"]}] mt-1'
                    )
        
        # Hardware Options section (collapsible)
        with ui.expansion(
            text='Hardware Options',
            icon='memory',
            value=False
        ).classes(
            f'w-full rounded-xl bg-[{COLORS["bg_card"]}] border border-[#2d343c] animate-in stagger-7'
        ).props('dense dark'):
            with ui.column().classes('w-full gap-4 p-4'):
                with ui.row().classes('w-full items-center gap-4'):
                    ui.label('Experimental Attention').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                    ui.switch(value=self.raft_data.experimental_attention).props(
                        'color=primary'
                    ).bind_value(self.raft_data, 'experimental_attention')
                ui.label('Enable for LFM2.5 and other models requiring ROCm experimental attention').classes(
                    f'text-xs text-[{COLORS["text_muted"]}]'
                )
        
        # Launch button
        self._render_launch_button("Start RAFT Training", self._launch_raft)
    
    def _browse_checkpoint(self):
        """Open file picker for checkpoint directory."""
        self._open_file_picker(
            title="Select SFT Checkpoint",
            path_type="directory",
            start_path="models/",
            on_select=lambda path: setattr(self.raft_data, 'checkpoint', path)
        )
    
    def _browse_curriculum_stats(self):
        """Open file picker for curriculum stats JSON file."""
        def on_select(path):
            self.raft_data.curriculum_stats_path = path
            # Refresh form
            self.form_container.clear()
            with self.form_container:
                self._render_form()
        
        self._open_file_picker(
            title="Select Curriculum Stats File",
            path_type="file",
            start_path="models/",
            extensions=[".json"],
            on_select=on_select
        )
    
    def _render_model_selector(self, label: str, data_obj, model_type: str = "code"):
        """Render model selection with dropdown + custom option."""
        ui.label(label).classes(
            f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
        )
        
        models = RECOMMENDED_MODELS.get(model_type, RECOMMENDED_MODELS["code"])
        
        # Build options dict
        model_options = {k: v for k, v in models}
        model_options["custom"] = "Custom HuggingFace model or local path..."
        
        # Determine select value - MUST be a valid option key
        if data_obj.model_source == "preset" and data_obj.model in model_options:
            select_value = data_obj.model
        else:
            select_value = "custom"
        
        def on_model_change(e):
            """Handle model selection change."""
            val = e.value
            if val == "custom":
                data_obj.model_source = "custom"
            else:
                data_obj.model_source = "preset"
                data_obj.model = val
            # Refresh form to show/hide custom input
            self.form_container.clear()
            with self.form_container:
                self._render_form()
        
        with ui.row().classes('w-full gap-2'):
            # Main dropdown - use on_change with e.value (standard NiceGUI pattern)
            ui.select(
                options=model_options,
                value=select_value,
                on_change=on_model_change
            ).classes('flex-1').props('outlined dense dark color=grey-7')
            
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
        
        # Determine select value - MUST be a valid option key
        if self.sft_data.dataset_source == "preset" and self.sft_data.dataset in dataset_options:
            select_value = self.sft_data.dataset
        else:
            select_value = "custom"
        
        def on_dataset_change(e):
            """Handle dataset selection change."""
            val = e.value
            if val == "custom":
                self.sft_data.dataset_source = "custom"
            else:
                self.sft_data.dataset_source = "preset"
                self.sft_data.dataset = val
            # Refresh form
            self.form_container.clear()
            with self.form_container:
                self._render_form()
        
        with ui.row().classes('w-full gap-2'):
            ui.select(
                options=dataset_options,
                value=select_value,
                on_change=on_dataset_change
            ).classes('flex-1').props('outlined dense dark color=grey-7')
            
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
        
        # Determine select value - MUST be a valid option key
        if self.raft_data.prompts_source == "preset" and self.raft_data.prompts in prompts_options:
            select_value = self.raft_data.prompts
        else:
            select_value = "custom"
        
        def on_prompts_change(e):
            """Handle prompts selection change."""
            val = e.value
            if val == "custom":
                self.raft_data.prompts_source = "custom"
            else:
                self.raft_data.prompts_source = "preset"
                self.raft_data.prompts = val
            # Refresh form
            self.form_container.clear()
            with self.form_container:
                self._render_form()
        
        with ui.row().classes('w-full gap-2'):
            ui.select(
                options=prompts_options,
                value=select_value,
                on_change=on_prompts_change
            ).classes('flex-1').props('outlined dense dark color=grey-7')
            
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
        self.selected_raft_preset = preset_name
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
                # Training hyperparameters
                epochs=self.sft_data.epochs,
                batch_size=self.sft_data.batch_size,
                gradient_accumulation_steps=self.sft_data.gradient_accumulation_steps,
                learning_rate=self.sft_data.learning_rate,
                warmup_ratio=self.sft_data.warmup_ratio,
                weight_decay=self.sft_data.weight_decay,
                max_grad_norm=self.sft_data.max_grad_norm,
                # LoRA config
                use_lora=self.sft_data.use_lora,
                lora_rank=self.sft_data.lora_rank,
                lora_alpha=self.sft_data.lora_alpha,
                lora_dropout=self.sft_data.lora_dropout,
                # Data options
                max_seq_length=self.sft_data.max_length,
                validation_split=self.sft_data.validation_split,
                max_samples=self.sft_data.max_samples,
                # Checkpointing
                save_steps=self.sft_data.save_steps,
                eval_steps=self.sft_data.eval_steps,
                early_stopping_patience=self.sft_data.early_stopping_patience,
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
            
            # Get checkpoint if enabled
            checkpoint = None
            if self.raft_data.use_checkpoint and self.raft_data.checkpoint:
                checkpoint = self.raft_data.checkpoint
            
            # Launch actual training subprocess via TrainingService
            job_id = await self.training_service.launch_raft(
                model=model,
                prompts=prompts,
                output_dir=self.raft_data.output_dir,
                verifier=self.raft_data.verifier,
                # RAFT parameters
                cycles=self.raft_data.cycles,
                samples_per_prompt=self.raft_data.samples_per_prompt,
                temperature=self.raft_data.temperature,
                keep_percent=self.raft_data.keep_percent,
                reward_threshold=self.raft_data.reward_threshold,
                min_samples=self.raft_data.min_samples,
                max_new_tokens=self.raft_data.max_new_tokens,
                # Learning rate schedule
                learning_rate=self.raft_data.learning_rate,
                lr_decay=self.raft_data.lr_decay,
                min_lr=self.raft_data.min_lr,
                # Checkpoint resume
                checkpoint=checkpoint,
                # Advanced strategies
                curriculum=self.raft_data.curriculum,
                curriculum_stats=self.raft_data.curriculum_stats_path if self.raft_data.curriculum == "historical" else None,
                curriculum_start=self.raft_data.curriculum_start,
                curriculum_increment=self.raft_data.curriculum_increment,
                reward_shaping=self.raft_data.reward_shaping,
                # Generation options
                system_prompt=self.raft_data.system_prompt,
                # Hardware options
                experimental_attention=self.raft_data.experimental_attention,
            )
            
            notify_job_started(f"RAFT: {self.raft_data.verifier}")
            
            # Navigate to monitor
            ui.navigate.to(f'/monitor/{job_id}')
            
        except Exception as e:
            notify_job_failed("RAFT Training", str(e))
        finally:
            self.is_running = False
