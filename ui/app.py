"""
halo-forge Web UI Application

NiceGUI-based web interface with routing and layout.
"""

import asyncio
import ipaddress
import os
from nicegui import ui, app
from pathlib import Path

from ui.theme import apply_theme, COLORS
from ui.components.sidebar import Sidebar
from ui.components.header import Header
from ui.components.page_guard import render_guarded_page
from ui.feature_flags import get_ui_feature_flags
from ui.services.hardware_monitor import get_hardware_monitor


# Serve static files from ui/static/ at /static URL path
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.add_static_files('/static', _static_dir)

_DEV_STORAGE_SECRET = "halo-forge-dev-storage-secret"
_storage_secret_warning_emitted = False
_hardware_monitor_start_requested = False


def _is_loopback_host(host: str) -> bool:
    """Return True when the configured bind host is loopback-only."""
    normalized = str(host or "").strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _resolve_storage_secret(host: str) -> str:
    """Resolve the storage secret with an explicit non-loopback safety gate."""
    global _storage_secret_warning_emitted

    configured = os.getenv("HALO_UI_STORAGE_SECRET", "").strip()
    if configured:
        return configured
    if _is_loopback_host(host):
        if not _storage_secret_warning_emitted:
            print(
                "UI_WARN HALO_UI_STORAGE_SECRET is not set; using a development fallback "
                "because the UI is bound to a loopback host."
            )
            _storage_secret_warning_emitted = True
        return _DEV_STORAGE_SECRET
    raise RuntimeError(
        "HALO_UI_STORAGE_SECRET must be set when binding the UI to a non-loopback host."
    )


def _ensure_hardware_monitor_started() -> None:
    """Start the singleton hardware monitor once for the UI process lifetime."""
    global _hardware_monitor_start_requested

    monitor = get_hardware_monitor()
    if monitor.is_running or _hardware_monitor_start_requested:
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    asyncio.create_task(monitor.start())
    _hardware_monitor_start_requested = True


def create_layout(page_title: str = "Dashboard"):
    """Create the base page layout with sidebar and header."""
    apply_theme()
    _ensure_hardware_monitor_started()
    
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
        render_guarded_page(
            "Dashboard",
            lambda: __import__("ui.pages.dashboard", fromlist=["Dashboard"]).Dashboard().render(),
        )


@ui.page('/training')
def training_page():
    """Training configuration and launch page."""
    create_layout("Training")
    with ui.column().classes('w-full h-full'):
        render_guarded_page(
            "Training",
            lambda: __import__("ui.pages.training", fromlist=["Training"]).Training().render(),
        )


@ui.page('/monitor')
def monitor_list_page():
    """Job monitor list page."""
    create_layout("Monitor")
    with ui.column().classes('w-full h-full'):
        render_guarded_page(
            "Monitor",
            lambda: __import__("ui.pages.monitor", fromlist=["MonitorList"]).MonitorList().render(),
        )


@ui.page('/monitor/{job_id}')
def monitor_page(job_id: str):
    """Job monitor detail page."""
    create_layout("Job Monitor")
    with ui.column().classes('w-full h-full'):
        render_guarded_page(
            "Job Monitor",
            lambda: __import__("ui.pages.monitor", fromlist=["Monitor"]).Monitor(job_id=job_id).render(),
        )


@ui.page('/config')
def config_page():
    """Configuration editor page."""
    create_layout("Configuration")
    with ui.column().classes('w-full h-full'):
        render_guarded_page(
            "Configuration",
            lambda: __import__("ui.pages.config", fromlist=["ConfigEditor"]).ConfigEditor().render(),
        )


@ui.page('/verifiers')
def verifiers_page():
    """Verifier management page."""
    create_layout("Verifiers")
    with ui.column().classes('w-full h-full'):
        render_guarded_page(
            "Verifiers",
            lambda: __import__("ui.pages.verifiers", fromlist=["Verifiers"]).Verifiers().render(),
        )


@ui.page('/datasets')
def datasets_page():
    """Dataset browser page."""
    create_layout("Datasets")
    with ui.column().classes('w-full h-full'):
        render_guarded_page(
            "Datasets",
            lambda: __import__("ui.pages.datasets", fromlist=["Datasets"]).Datasets().render(),
        )


@ui.page('/results')
def results_page():
    """Benchmark results page."""
    create_layout("Results")
    with ui.column().classes('w-full h-full'):
        render_guarded_page(
            "Results",
            lambda: __import__("ui.pages.results", fromlist=["Results"]).Results().render(),
        )


@ui.page('/benchmark')
def benchmark_page():
    """Benchmark launcher page."""
    create_layout("Benchmark")
    with ui.column().classes('w-full h-full'):
        render_guarded_page(
            "Benchmark",
            lambda: __import__("ui.pages.benchmark", fromlist=["Benchmark"]).Benchmark().render(),
        )


@ui.page('/ops-console')
def ops_console_page():
    """Utility module operations console."""
    create_layout("Ops Console")
    with ui.column().classes('w-full h-full'):
        render_guarded_page(
            "Ops Console",
            lambda: __import__("ui.pages.ops_console", fromlist=["OpsConsole"]).OpsConsole().render(),
        )


@ui.page('/inference')
def inference_page():
    """Inference launch page (feature-flagged)."""
    create_layout("Inference")
    flags = get_ui_feature_flags()
    with ui.column().classes('w-full h-full'):
        if not flags.enable_inference_page:
            _render_feature_disabled("Inference", "HALO_UI_ENABLE_INFERENCE_PAGE")
            return
        render_guarded_page(
            "Inference",
            lambda: __import__("ui.pages.inference", fromlist=["Inference"]).Inference().render(),
        )


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
        render_guarded_page(
            "Benchmark Advanced",
            lambda: __import__("ui.pages.benchmark_advanced", fromlist=["BenchmarkAdvanced"]).BenchmarkAdvanced().render(),
        )


@ui.page('/research-hub')
def research_hub_page():
    """Ops readiness and research hub page (feature-flagged)."""
    create_layout("Advanced Diagnostics Tools")
    flags = get_ui_feature_flags()
    with ui.column().classes('w-full h-full'):
        if not flags.enable_research_hub_page:
            _render_feature_disabled(
                "Advanced Diagnostics Tools",
                "HALO_UI_ENABLE_RESEARCH_HUB_PAGE",
            )
            return
        render_guarded_page(
            "Advanced Diagnostics Tools",
            lambda: __import__("ui.pages.research_hub", fromlist=["ResearchHub"]).ResearchHub().render(),
        )


def run(
    host: str = "127.0.0.1",
    port: int = 8080,
    reload: bool = False,
    open_browser: bool = False,
):
    """Run the halo-forge web UI."""
    static_dir = Path(__file__).parent / "static"
    storage_secret = _resolve_storage_secret(host)
    
    # Prefer SVG favicon, fall back to PNG, then emoji
    favicon_svg = static_dir / "favicon.svg"
    favicon_png = static_dir / "favicon.png"
    
    if favicon_svg.exists():
        favicon = favicon_svg
    elif favicon_png.exists():
        favicon = favicon_png
    else:
        favicon = "🔥"

    base_url = f"http://{host}:{port}"
    print(f"UI_START base_url={base_url} open_browser={1 if open_browser else 0}")
    print(f"UI_ROUTE root={base_url}/")
    print(f"UI_ROUTE training={base_url}/training")
    print(f"UI_ROUTE benchmark={base_url}/benchmark")
    print(f"UI_ROUTE inference={base_url}/inference")
    
    ui.run(
        host=host,
        port=port,
        reload=reload,
        show=open_browser,
        title="halo-forge",
        favicon=favicon,
        dark=True,
        binding_refresh_interval=0.1,
        storage_secret=storage_secret,
    )
