"""
Config Editor Page

YAML configuration editor with syntax highlighting and validation.
"""

from nicegui import ui
from pathlib import Path
from typing import Optional
import yaml

from ui.theme import COLORS


class ConfigEditor:
    """Configuration editor page component."""
    
    CONFIG_DIR = Path("configs")
    
    # Template configs
    TEMPLATES = {
        "sft_basic": {
            "name": "SFT Basic",
            "content": """# Basic SFT Configuration
model:
  name: "Qwen/Qwen2.5-Coder-3B"
  
training:
  epochs: 3
  batch_size: 4
  learning_rate: 2e-5
  max_length: 2048
  
lora:
  enabled: true
  rank: 32
  alpha: 64
  dropout: 0.05
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

output:
  dir: "models/sft_output"
  save_steps: 500
"""
        },
        "raft_conservative": {
            "name": "RAFT Conservative",
            "content": """# Conservative RAFT Configuration
model:
  name: "Qwen/Qwen2.5-Coder-3B"
  
raft:
  cycles: 5
  samples_per_prompt: 8
  keep_top_percent: 0.8
  reward_threshold: 0.5
  temperature: 0.7
  
training:
  learning_rate: 1e-5
  lr_decay_per_cycle: 0.9
  
verifier: "humaneval"
prompts: "data/rlvr/humaneval_prompts.jsonl"
output_dir: "models/raft_output"
"""
        },
        "raft_aggressive": {
            "name": "RAFT Aggressive", 
            "content": """# Aggressive RAFT Configuration
model:
  name: "Qwen/Qwen2.5-Coder-3B"
  
raft:
  cycles: 10
  samples_per_prompt: 16
  keep_top_percent: 0.3
  reward_threshold: 0.7
  temperature: 0.8
  min_samples_per_cycle: 64
  
training:
  learning_rate: 2e-5
  lr_decay_per_cycle: 0.85
  
verifier: "humaneval"
prompts: "data/rlvr/humaneval_prompts.jsonl"
output_dir: "models/raft_aggressive"
"""
        }
    }
    
    def __init__(self):
        self.current_file: Optional[Path] = None
        self.editor_content: str = ""
        self.is_modified: bool = False
        self.validation_error: Optional[str] = None
    
    def render(self):
        """Render the config editor page."""
        with ui.column().classes('page-content w-full gap-6 p-6'):
            # Header
            with ui.row().classes('w-full items-center justify-between animate-in'):
                ui.label('Configuration').classes(
                    f'text-2xl font-bold text-[{COLORS["text_primary"]}]'
                )
                
                with ui.row().classes('gap-2'):
                    ui.button('New', icon='add', on_click=self._new_config).props(
                        'flat'
                    ).classes(f'text-[{COLORS["text_secondary"]}]')
                    
                    ui.button('Save', icon='save', on_click=self._save_config).props(
                        'unelevated'
                    ).classes(f'bg-[{COLORS["primary"]}] text-white')
            
            # Main layout
            with ui.row().classes('w-full gap-6 flex-wrap'):
                # File browser
                with ui.column().classes(
                    f'w-64 gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                    f'border border-[#2d343c] animate-in stagger-1'
                ):
                    self._render_file_browser()
                
                # Editor
                with ui.column().classes(
                    f'flex-1 min-w-[400px] gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                    f'border border-[#2d343c] animate-in stagger-2'
                ):
                    self._render_editor()
            
            # Templates section
            with ui.column().classes(
                f'w-full gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                f'border border-[#2d343c] animate-in stagger-3'
            ):
                self._render_templates()
    
    def _render_file_browser(self):
        """Render the config file browser."""
        ui.label('Config Files').classes(
            f'text-sm font-semibold text-[{COLORS["text_primary"]}]'
        )
        
        with ui.column().classes('w-full gap-1'):
            # List existing configs
            if self.CONFIG_DIR.exists():
                for config_file in sorted(self.CONFIG_DIR.glob("*.yaml")) + \
                                   sorted(self.CONFIG_DIR.glob("*.yml")):
                    self._file_item(config_file)
            
            # Empty state
            if not self.CONFIG_DIR.exists() or not any(self.CONFIG_DIR.glob("*.y*ml")):
                ui.label('No config files found').classes(
                    f'text-xs text-[{COLORS["text_muted"]}] py-4'
                )
    
    def _file_item(self, path: Path):
        """Render a file item in the browser."""
        is_selected = self.current_file == path
        
        with ui.row().classes(
            f'w-full items-center gap-2 px-3 py-2 rounded-lg cursor-pointer '
            + (f'bg-[{COLORS["primary"]}]/10' if is_selected else f'hover:bg-[{COLORS["bg_hover"]}]')
        ).on('click', lambda p=path: self._open_file(p)):
            ui.icon('description', size='16px').classes(
                f'text-[{COLORS["accent"]}]' if is_selected else f'text-[{COLORS["text_muted"]}]'
            )
            ui.label(path.name).classes(
                f'text-sm text-[{COLORS["text_primary"]}]' if is_selected 
                else f'text-sm text-[{COLORS["text_secondary"]}]'
            )
    
    def _render_editor(self):
        """Render the YAML editor."""
        with ui.row().classes('w-full items-center justify-between'):
            if self.current_file:
                ui.label(self.current_file.name).classes(
                    f'text-sm font-medium text-[{COLORS["text_primary"]}]'
                )
                if self.is_modified:
                    ui.label('(modified)').classes(
                        f'text-xs text-[{COLORS["accent"]}]'
                    )
            else:
                ui.label('New Configuration').classes(
                    f'text-sm font-medium text-[{COLORS["text_secondary"]}]'
                )
            
            # Validate button
            ui.button('Validate', icon='check_circle', on_click=self._validate).props(
                'flat dense size=sm'
            ).classes(f'text-[{COLORS["info"]}]')
        
        # Editor textarea
        self.editor = ui.textarea(
            value=self.editor_content,
            on_change=self._on_edit
        ).classes(
            f'w-full font-mono text-sm'
        ).props(
            f'outlined autogrow dark rows=20 '
            f'input-class="font-mono leading-relaxed"'
        ).style(
            f'background: {COLORS["bg_primary"]};'
        )
        
        # Validation error display
        if self.validation_error:
            with ui.row().classes(
                f'w-full items-center gap-2 p-3 rounded-lg bg-[{COLORS["error"]}]/10 '
                f'border border-[{COLORS["error"]}]/30'
            ):
                ui.icon('error', size='18px').classes(f'text-[{COLORS["error"]}]')
                ui.label(self.validation_error).classes(
                    f'text-sm text-[{COLORS["error"]}]'
                )
    
    def _render_templates(self):
        """Render template configs section."""
        ui.label('Templates').classes(
            f'text-sm font-semibold text-[{COLORS["text_primary"]}]'
        )
        
        with ui.row().classes('w-full gap-3 flex-wrap'):
            for template_id, template in self.TEMPLATES.items():
                with ui.button(
                    on_click=lambda t=template: self._load_template(t)
                ).props('flat').classes(
                    f'px-4 py-3 bg-[{COLORS["bg_secondary"]}] '
                    f'border border-[#2d343c] rounded-lg card-hover'
                ):
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('content_paste', size='18px').classes(
                            f'text-[{COLORS["accent"]}]'
                        )
                        ui.label(template['name']).classes(
                            f'text-sm text-[{COLORS["text_primary"]}]'
                        )
    
    def _open_file(self, path: Path):
        """Open a config file for editing."""
        try:
            self.editor_content = path.read_text()
            self.current_file = path
            self.is_modified = False
            self.validation_error = None
            
            if hasattr(self, 'editor'):
                self.editor.value = self.editor_content
            
            ui.notify(f'Opened {path.name}', type='positive', timeout=1500)
        except Exception as e:
            ui.notify(f'Failed to open file: {e}', type='negative')
    
    def _on_edit(self, e):
        """Handle editor content changes."""
        self.editor_content = e.value
        self.is_modified = True
        self.validation_error = None
    
    def _validate(self):
        """Validate the current YAML content."""
        try:
            yaml.safe_load(self.editor_content)
            self.validation_error = None
            ui.notify('Valid YAML ✓', type='positive', timeout=1500)
        except yaml.YAMLError as e:
            self.validation_error = str(e)
            ui.notify('Invalid YAML', type='negative')
    
    def _save_config(self):
        """Save the current config."""
        if not self.editor_content.strip():
            ui.notify('Nothing to save', type='warning')
            return
        
        # Validate first
        try:
            yaml.safe_load(self.editor_content)
        except yaml.YAMLError as e:
            self.validation_error = str(e)
            ui.notify('Fix YAML errors before saving', type='negative')
            return
        
        if self.current_file:
            self.current_file.write_text(self.editor_content)
            self.is_modified = False
            ui.notify(f'Saved {self.current_file.name}', type='positive')
        else:
            # Show save dialog
            self._show_save_dialog()
    
    def _show_save_dialog(self):
        """Show the save as dialog."""
        with ui.dialog() as dialog, ui.card().classes(
            f'bg-[{COLORS["bg_card"]}] p-6 min-w-[300px]'
        ):
            ui.label('Save Configuration').classes(
                f'text-lg font-semibold text-[{COLORS["text_primary"]}]'
            )
            
            filename_input = ui.input(
                'Filename',
                value='config.yaml'
            ).classes('w-full mt-4').props('outlined dense dark')
            
            with ui.row().classes('w-full justify-end gap-2 mt-4'):
                ui.button('Cancel', on_click=dialog.close).props('flat').classes(
                    f'text-[{COLORS["text_secondary"]}]'
                )
                ui.button('Save', on_click=lambda: self._do_save(filename_input.value, dialog)).props(
                    'unelevated'
                ).classes(f'bg-[{COLORS["primary"]}] text-white')
        
        dialog.open()
    
    def _do_save(self, filename: str, dialog):
        """Perform the save operation."""
        if not filename.endswith(('.yaml', '.yml')):
            filename += '.yaml'
        
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        path = self.CONFIG_DIR / filename
        path.write_text(self.editor_content)
        
        self.current_file = path
        self.is_modified = False
        
        dialog.close()
        ui.notify(f'Saved {filename}', type='positive')
    
    def _new_config(self):
        """Create a new config."""
        self.current_file = None
        self.editor_content = "# New Configuration\n\n"
        self.is_modified = False
        self.validation_error = None
        
        if hasattr(self, 'editor'):
            self.editor.value = self.editor_content
    
    def _load_template(self, template: dict):
        """Load a template into the editor."""
        self.editor_content = template['content']
        self.current_file = None
        self.is_modified = True
        self.validation_error = None
        
        if hasattr(self, 'editor'):
            self.editor.value = self.editor_content
        
        ui.notify(f'Loaded {template["name"]} template', type='positive', timeout=1500)
