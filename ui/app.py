"""
halo-forge Web UI Application

NiceGUI-based web interface with routing and layout.
"""

from nicegui import ui, app
from pathlib import Path

from ui.theme import apply_theme, COLORS
from ui.components.sidebar import Sidebar
from ui.components.header import Header


def create_layout(page_title: str = "Dashboard"):
    """Create the base page layout with sidebar and header."""
    apply_theme()
    
    # Header
    with ui.header().classes(
        f'bg-[{COLORS["bg_secondary"]}] border-b border-[#2d343c] h-14'
    ):
        header = Header(title=page_title)
        header.register_cleanup()
    
    # Left drawer (sidebar)
    with ui.left_drawer(value=True, fixed=True).classes(
        f'bg-[{COLORS["bg_secondary"]}] w-56 p-0 border-r border-[#2d343c]'
    ).props('behavior=desktop bordered'):
        Sidebar()
    
    return header


@ui.page('/')
def dashboard_page():
    """Dashboard page."""
    create_layout("Dashboard")
    with ui.column().classes('w-full h-full'):
        from ui.pages.dashboard import Dashboard
        Dashboard().render()


@ui.page('/training')
def training_page():
    """Training configuration and launch page."""
    create_layout("Training")
    with ui.column().classes('w-full h-full'):
        from ui.pages.training import Training
        Training().render()


@ui.page('/monitor')
def monitor_list_page():
    """Job monitor list page."""
    create_layout("Monitor")
    with ui.column().classes('w-full h-full'):
        from ui.pages.monitor import MonitorList
        MonitorList().render()


@ui.page('/monitor/{job_id}')
def monitor_page(job_id: str):
    """Job monitor detail page."""
    create_layout("Job Monitor")
    with ui.column().classes('w-full h-full'):
        from ui.pages.monitor import Monitor
        Monitor(job_id=job_id).render()


@ui.page('/config')
def config_page():
    """Configuration editor page."""
    create_layout("Configuration")
    with ui.column().classes('w-full h-full'):
        from ui.pages.config import ConfigEditor
        ConfigEditor().render()


@ui.page('/verifiers')
def verifiers_page():
    """Verifier management page."""
    create_layout("Verifiers")
    with ui.column().classes('w-full h-full'):
        from ui.pages.verifiers import Verifiers
        Verifiers().render()


@ui.page('/datasets')
def datasets_page():
    """Dataset browser page."""
    create_layout("Datasets")
    with ui.column().classes('w-full h-full'):
        from ui.pages.datasets import Datasets
        Datasets().render()


@ui.page('/results')
def results_page():
    """Benchmark results page."""
    create_layout("Results")
    with ui.column().classes('w-full h-full'):
        from ui.pages.results import Results
        Results().render()


@ui.page('/benchmark')
def benchmark_page():
    """Benchmark launcher page."""
    create_layout("Benchmark")
    with ui.column().classes('w-full h-full'):
        from ui.pages.benchmark import Benchmark
        Benchmark().render()


def run(host: str = "127.0.0.1", port: int = 8080, reload: bool = False):
    """Run the halo-forge web UI."""
    from pathlib import Path
    
    # Serve favicon from ui/static
    favicon_path = Path(__file__).parent / "static" / "favicon.svg"
    
    ui.run(
        host=host,
        port=port,
        reload=reload,
        title="halo-forge",
        favicon=favicon_path if favicon_path.exists() else "🔥",
        dark=True,
        binding_refresh_interval=0.1,
        storage_secret='halo-forge-storage-secret',  # Required for app.storage.user
    )
