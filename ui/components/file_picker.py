"""
File Picker Component

A reusable file/directory picker dialog for the UI.
"""

from nicegui import ui
from pathlib import Path
from typing import Callable, Optional, Literal
import os

from ui.theme import COLORS


class FilePicker:
    """File/directory picker dialog component."""
    
    def __init__(
        self,
        title: str = "Select File",
        path_type: Literal["file", "directory"] = "file",
        file_filter: Optional[str] = None,
        start_path: str = ".",
        on_select: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the file picker.
        
        Args:
            title: Dialog title
            path_type: "file" or "directory"
            file_filter: Glob pattern for files (e.g., "*.jsonl")
            start_path: Starting directory
            on_select: Callback when file/directory is selected
        """
        self.title = title
        self.path_type = path_type
        self.file_filter = file_filter
        self.start_path = start_path
        self.on_select = on_select
        
        self.current_path = Path(start_path).resolve()
        self.selected_item: Optional[str] = None
        self.dialog = None
        self._items_container = None
        self._path_input = None
    
    def open(self):
        """Open the file picker dialog."""
        with ui.dialog() as self.dialog:
            with ui.card().classes(
                f'w-[600px] max-h-[80vh] bg-[{COLORS["bg_card"]}] p-0 overflow-hidden'
            ):
                # Header
                with ui.row().classes(
                    f'w-full items-center justify-between p-4 border-b border-[#2d343c]'
                ):
                    ui.label(self.title).classes(
                        f'text-lg font-semibold text-[{COLORS["text_primary"]}]'
                    )
                    ui.button(icon='close', on_click=self.dialog.close).props(
                        'flat round dense'
                    ).classes(f'text-[{COLORS["text_muted"]}]')
                
                # Path bar
                with ui.row().classes('w-full items-center gap-2 px-4 py-2'):
                    ui.button(icon='arrow_upward', on_click=self._go_up).props(
                        'flat round dense'
                    ).classes(f'text-[{COLORS["text_muted"]}]').tooltip('Go up')
                    
                    ui.button(icon='home', on_click=self._go_home).props(
                        'flat round dense'
                    ).classes(f'text-[{COLORS["text_muted"]}]').tooltip('Home')
                    
                    self._path_input = ui.input(
                        value=str(self.current_path)
                    ).classes('flex-1').props('outlined dense dark')
                    self._path_input.on('keydown.enter', self._navigate_to_input)
                    
                    ui.button(icon='refresh', on_click=self._refresh).props(
                        'flat round dense'
                    ).classes(f'text-[{COLORS["text_muted"]}]').tooltip('Refresh')
                
                # File list
                with ui.scroll_area().classes('w-full h-80'):
                    self._items_container = ui.column().classes('w-full gap-0 p-2')
                    with self._items_container:
                        self._render_items()
                
                # Quick path input for manual entry
                with ui.row().classes('w-full items-center gap-2 px-4 py-2 border-t border-[#2d343c]'):
                    ui.label('Path:').classes(f'text-xs text-[{COLORS["text_muted"]}]')
                    self._selected_display = ui.input(
                        placeholder='Enter path or select from list above...',
                        value=self.selected_item or ''
                    ).classes('flex-1').props('outlined dense dark')
                
                # Actions
                with ui.row().classes(
                    f'w-full items-center justify-end gap-2 p-4 border-t border-[#2d343c]'
                ):
                    ui.button('Cancel', on_click=self.dialog.close).props(
                        'flat'
                    ).classes(f'text-[{COLORS["text_secondary"]}]')
                    
                    ui.button(
                        'Select',
                        on_click=self._confirm_selection
                    ).props('unelevated').classes(
                        f'bg-[{COLORS["primary"]}] text-white'
                    )
        
        self.dialog.open()
    
    def _render_items(self):
        """Render the file/directory list."""
        try:
            items = []
            
            # Get directories
            for item in sorted(self.current_path.iterdir()):
                if item.name.startswith('.'):
                    continue  # Skip hidden files
                
                if item.is_dir():
                    items.append((item, 'directory'))
                elif self.path_type == "file":
                    # Check file filter
                    if self.file_filter:
                        if item.match(self.file_filter):
                            items.append((item, 'file'))
                    else:
                        items.append((item, 'file'))
            
            # Sort: directories first, then files
            items.sort(key=lambda x: (x[1] != 'directory', x[0].name.lower()))
            
            if not items:
                ui.label('Empty directory').classes(
                    f'text-sm text-[{COLORS["text_muted"]}] italic p-4'
                )
                return
            
            for item_path, item_type in items:
                self._render_item(item_path, item_type)
                
        except PermissionError:
            ui.label('Permission denied').classes(
                f'text-sm text-[{COLORS["error"]}] p-4'
            )
        except Exception as e:
            ui.label(f'Error: {e}').classes(
                f'text-sm text-[{COLORS["error"]}] p-4'
            )
    
    def _render_item(self, item_path: Path, item_type: str):
        """Render a single file/directory item."""
        is_selected = self.selected_item == str(item_path)
        
        icon_name = 'folder' if item_type == 'directory' else 'description'
        icon_color = COLORS['accent'] if item_type == 'directory' else COLORS['text_muted']
        
        with ui.row().classes(
            f'w-full items-center gap-3 px-3 py-2 rounded-lg cursor-pointer '
            + (f'bg-[{COLORS["primary"]}]/20' if is_selected
               else f'hover:bg-[{COLORS["bg_hover"]}]')
        ).on('click', lambda p=item_path, t=item_type: self._on_item_click(p, t)).on(
            'dblclick', lambda p=item_path, t=item_type: self._on_item_double_click(p, t)
        ):
            ui.icon(icon_name, size='20px').classes(f'text-[{icon_color}]')
            
            with ui.column().classes('flex-1 gap-0'):
                ui.label(item_path.name).classes(
                    f'text-sm text-[{COLORS["text_primary"]}]'
                )
                
                if item_type == 'file':
                    try:
                        size = item_path.stat().st_size
                        size_str = self._format_size(size)
                        ui.label(size_str).classes(
                            f'text-xs text-[{COLORS["text_muted"]}]'
                        )
                    except:
                        pass
    
    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f'{size:.1f} {unit}'
            size /= 1024
        return f'{size:.1f} TB'
    
    def _on_item_click(self, item_path: Path, item_type: str):
        """Handle single click on item."""
        # For directories in directory mode, or files in file mode, allow selection
        if self.path_type == "directory" and item_type == "directory":
            self.selected_item = str(item_path)
        elif self.path_type == "file" and item_type == "file":
            self.selected_item = str(item_path)
        elif self.path_type == "file" and item_type == "directory":
            # In file mode, clicking directory just highlights it (for navigation)
            pass
        
        # Update display
        if self.selected_item and hasattr(self, '_selected_display'):
            self._selected_display.value = self.selected_item
        
        # Refresh to update highlighting
        self._items_container.clear()
        with self._items_container:
            self._render_items()
    
    def _on_item_double_click(self, item_path: Path, item_type: str):
        """Handle double click on item."""
        if item_type == 'directory':
            # Navigate into directory
            self.current_path = item_path
            self._path_input.value = str(self.current_path)
            self.selected_item = None
            self._items_container.clear()
            with self._items_container:
                self._render_items()
        else:
            # For files, confirm selection
            self.selected_item = str(item_path)
            self._confirm_selection()
    
    def _go_up(self):
        """Navigate to parent directory."""
        parent = self.current_path.parent
        if parent != self.current_path:
            self.current_path = parent
            self._path_input.value = str(self.current_path)
            self.selected_item = None
            self._items_container.clear()
            with self._items_container:
                self._render_items()
    
    def _go_home(self):
        """Navigate to home directory."""
        self.current_path = Path.home()
        self._path_input.value = str(self.current_path)
        self.selected_item = None
        self._items_container.clear()
        with self._items_container:
            self._render_items()
    
    def _navigate_to_input(self):
        """Navigate to the path in the input field."""
        path_str = self._path_input.value
        path = Path(path_str)
        
        if path.is_dir():
            self.current_path = path.resolve()
            self._path_input.value = str(self.current_path)
            self.selected_item = None
            self._items_container.clear()
            with self._items_container:
                self._render_items()
        else:
            ui.notify(f'Directory not found: {path_str}', type='negative', timeout=2000)
    
    def _refresh(self):
        """Refresh the current directory."""
        self._items_container.clear()
        with self._items_container:
            self._render_items()
    
    def _confirm_selection(self):
        """Confirm the selection and close dialog."""
        # Get selection from the input field (allows manual entry)
        selected = self._selected_display.value if hasattr(self, '_selected_display') else self.selected_item
        
        if not selected:
            # For directory mode, use current path if nothing selected
            if self.path_type == "directory":
                selected = str(self.current_path)
            else:
                ui.notify('Please select a file', type='warning', timeout=2000)
                return
        
        if self.on_select:
            self.on_select(selected)
        
        self.dialog.close()
