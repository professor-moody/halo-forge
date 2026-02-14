"""
halo-forge Web UI Application

NiceGUI-based web interface with routing and layout.
"""

from nicegui import ui, app
from pathlib import Path

from ui.theme import apply_theme, COLORS
from ui.components.sidebar import Sidebar
from ui.components.header import Header
from ui.feature_flags import get_ui_feature_flags


# Serve static files from ui/static/ at /static URL path
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.add_static_files('/static', _static_dir)


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


def _render_feature_disabled(feature_name: str, env_var: str) -> None:
    """Render explicit disabled feature state instead of 404."""
    with ui.column().classes(
        f"w-full max-w-2xl mx-auto mt-10 gap-4 p-6 rounded-xl bg-[{COLORS['bg_card']}] "
        f"border border-[#2d343c]"
    ):
        ui.label(f"{feature_name} is disabled").classes(
            f"text-lg font-semibold text-[{COLORS['text_primary']}]"
        )
        ui.label(
            "This route is behind a feature flag."
        ).classes(f"text-sm text-[{COLORS['text_secondary']}]")
        ui.label(
            f"Set `{env_var}=1` before starting the UI to enable it."
        ).classes(f"text-xs text-[{COLORS['text_muted']}] font-mono")


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


@ui.page('/inference')
def inference_page():
    """Inference launch page (feature-flagged)."""
    create_layout("Inference")
    flags = get_ui_feature_flags()
    with ui.column().classes('w-full h-full'):
        if not flags.enable_inference_page:
            _render_feature_disabled("Inference", "HALO_UI_ENABLE_INFERENCE_PAGE")
            return
        from ui.pages.inference import Inference
        Inference().render()


@ui.page('/benchmark-advanced')
def benchmark_advanced_page():
    """Advanced benchmark orchestration page (feature-flagged)."""
    create_layout("Benchmark Advanced")
    flags = get_ui_feature_flags()
    with ui.column().classes('w-full h-full'):
        if not flags.enable_benchmark_advanced_page:
            _render_feature_disabled(
                "Benchmark Advanced",
                "HALO_UI_ENABLE_BENCHMARK_ADVANCED_PAGE",
            )
            return
        from ui.pages.benchmark_advanced import BenchmarkAdvanced
        BenchmarkAdvanced().render()


@ui.page('/research-hub')
def research_hub_page():
    """Ops readiness and research hub page (feature-flagged)."""
    create_layout("Research Hub")
    flags = get_ui_feature_flags()
    with ui.column().classes('w-full h-full'):
        if not flags.enable_research_hub_page:
            _render_feature_disabled(
                "Research Hub",
                "HALO_UI_ENABLE_RESEARCH_HUB_PAGE",
            )
            return
        from ui.pages.research_hub import ResearchHub
        ResearchHub().render()


def run(host: str = "127.0.0.1", port: int = 8080, reload: bool = False):
    """Run the halo-forge web UI."""
    static_dir = Path(__file__).parent / "static"
    
    # Prefer SVG favicon, fall back to PNG, then emoji
    favicon_svg = static_dir / "favicon.svg"
    favicon_png = static_dir / "favicon.png"
    
    if favicon_svg.exists():
        favicon = favicon_svg
    elif favicon_png.exists():
        favicon = favicon_png
    else:
        favicon = "🔥"
    
    ui.run(
        host=host,
        port=port,
        reload=reload,
        title="halo-forge",
        favicon=favicon,
        dark=True,
        binding_refresh_interval=0.1,
        storage_secret='halo-forge-storage-secret',  # Required for app.storage.user
    )
