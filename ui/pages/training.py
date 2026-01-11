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


@dataclass
class SFTFormData:
    """SFT training form data."""
    model: str = "Qwen/Qwen2.5-Coder-3B"
    dataset: str = "alpaca"
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
    prompts: str = "data/rlvr/humaneval_prompts.jsonl"
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
    
    SFT_DATASETS = [
        ("alpaca", "Alpaca (52K instruction-following)"),
        ("metamath", "MetaMath (395K math problems)"),
        ("gsm8k", "GSM8K (8.5K grade-school math)"),
        ("xlam", "xLAM (60K function calling)"),
    ]
    
    VERIFIERS = [
        ("humaneval", "HumanEval (Python coding)"),
        ("mbpp", "MBPP (Python basics)"),
        ("livecodebench", "LiveCodeBench (Multi-language)"),
        ("math", "Math (Numerical verification)"),
    ]
    
    def __init__(self):
        self.mode: Literal["sft", "raft"] = "sft"
        self.sft_data = SFTFormData()
        self.raft_data = RAFTFormData()
        self.selected_preset = "conservative"
        self.is_running = False
        self.training_service = TrainingService(state)
    
    def render(self):
        """Render the training page."""
        with ui.column().classes('page-content w-full gap-6 p-6'):
            # Header
            with ui.row().classes('w-full items-center justify-between animate-in'):
                ui.label('Training').classes(
                    f'text-2xl font-bold text-[{COLORS["text_primary"]}]'
                )
            
            # Mode toggle
            with ui.row().classes(
                f'w-full gap-2 p-2 rounded-xl bg-[{COLORS["bg_card"]}] '
                f'border border-[#2d343c] animate-in stagger-1'
            ):
                self._mode_button("SFT", "sft", "school")
                self._mode_button("RAFT", "raft", "autorenew")
            
            # Main form container
            self.form_container = ui.column().classes('w-full gap-6')
            with self.form_container:
                self._render_form()
    
    def _mode_button(self, label: str, mode: str, icon: str):
        """Render a mode toggle button."""
        is_active = self.mode == mode
        
        # Use a container div instead of nesting inside button
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
            
            with ui.row().classes('w-full gap-4 flex-wrap'):
                with ui.column().classes('flex-1 min-w-[280px] gap-2'):
                    ui.label('Base Model').classes(
                        f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                    )
                    ui.input(value=self.sft_data.model).classes('w-full').props(
                        f'outlined dense dark color=grey-7'
                    ).on('change', lambda e: setattr(self.sft_data, 'model', e.value))
                
                with ui.column().classes('flex-1 min-w-[280px] gap-2'):
                    ui.label('Dataset').classes(
                        f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                    )
                    ui.select(
                        options={k: v for k, v in self.SFT_DATASETS},
                        value=self.sft_data.dataset
                    ).classes('w-full').props(
                        'outlined dense dark color=grey-7'
                    ).on('change', lambda e: setattr(self.sft_data, 'dataset', e.value))
            
            with ui.column().classes('w-full gap-2'):
                ui.label('Output Directory').classes(
                    f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                )
                ui.input(value=self.sft_data.output_dir).classes('w-full').props(
                    'outlined dense dark color=grey-7'
                ).on('change', lambda e: setattr(self.sft_data, 'output_dir', e.value))
        
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
                                   format_val="1e-5")
                
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
                ).on('change', lambda e: setattr(self.sft_data, 'use_lora', e.value))
            
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
            
            with ui.row().classes('w-full gap-4 flex-wrap'):
                with ui.column().classes('flex-1 min-w-[280px] gap-2'):
                    ui.label('Base Model / Checkpoint').classes(
                        f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                    )
                    ui.input(value=self.raft_data.model).classes('w-full').props(
                        'outlined dense dark color=grey-7'
                    ).on('change', lambda e: setattr(self.raft_data, 'model', e.value))
                
                with ui.column().classes('flex-1 min-w-[280px] gap-2'):
                    ui.label('Verifier').classes(
                        f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                    )
                    ui.select(
                        options={k: v for k, v in self.VERIFIERS},
                        value=self.raft_data.verifier
                    ).classes('w-full').props(
                        'outlined dense dark color=grey-7'
                    ).on('change', lambda e: setattr(self.raft_data, 'verifier', e.value))
            
            with ui.row().classes('w-full gap-4 flex-wrap'):
                with ui.column().classes('flex-1 min-w-[280px] gap-2'):
                    ui.label('Prompts File').classes(
                        f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                    )
                    ui.input(value=self.raft_data.prompts).classes('w-full').props(
                        'outlined dense dark color=grey-7'
                    ).on('change', lambda e: setattr(self.raft_data, 'prompts', e.value))
                
                with ui.column().classes('flex-1 min-w-[280px] gap-2'):
                    ui.label('Output Directory').classes(
                        f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                    )
                    ui.input(value=self.raft_data.output_dir).classes('w-full').props(
                        'outlined dense dark color=grey-7'
                    ).on('change', lambda e: setattr(self.raft_data, 'output_dir', e.value))
        
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
            inp.on('change', lambda e: on_change(e.value))
    
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
        
        # Re-render form to show updated values
        self.form_container.clear()
        with self.form_container:
            self._render_form()
        
        ui.notify(f'Applied "{preset_name}" preset', type='positive', timeout=1500)
    
    async def _launch_sft(self):
        """Launch SFT training."""
        if self.is_running:
            return
        
        self.is_running = True
        ui.notify('Starting SFT training...', type='info')
        
        try:
            # Launch actual training subprocess via TrainingService
            job_id = await self.training_service.launch_sft(
                model=self.sft_data.model,
                dataset=self.sft_data.dataset,
                output_dir=self.sft_data.output_dir,
                epochs=self.sft_data.epochs,
                batch_size=self.sft_data.batch_size,
                learning_rate=self.sft_data.learning_rate,
                use_lora=self.sft_data.use_lora,
                lora_rank=self.sft_data.lora_rank,
            )
            
            ui.notify(f'Training started! Job ID: {job_id}', type='positive')
            
            # Navigate to monitor
            ui.navigate.to(f'/monitor/{job_id}')
            
        except Exception as e:
            ui.notify(f'Failed to start training: {e}', type='negative')
        finally:
            self.is_running = False
    
    async def _launch_raft(self):
        """Launch RAFT training."""
        if self.is_running:
            return
        
        self.is_running = True
        ui.notify('Starting RAFT training...', type='info')
        
        try:
            # Launch actual training subprocess via TrainingService
            job_id = await self.training_service.launch_raft(
                model=self.raft_data.model,
                prompts=self.raft_data.prompts,
                output_dir=self.raft_data.output_dir,
                verifier=self.raft_data.verifier,
                cycles=self.raft_data.cycles,
                samples_per_prompt=self.raft_data.samples_per_prompt,
                temperature=self.raft_data.temperature,
                keep_percent=self.raft_data.keep_percent,
                reward_threshold=self.raft_data.reward_threshold,
            )
            
            ui.notify(f'RAFT training started! Job ID: {job_id}', type='positive')
            
            # Navigate to monitor
            ui.navigate.to(f'/monitor/{job_id}')
            
        except Exception as e:
            ui.notify(f'Failed to start RAFT: {e}', type='negative')
        finally:
            self.is_running = False
