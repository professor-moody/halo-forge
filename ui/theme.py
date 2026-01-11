"""
halo-forge UI Theme

Sage/gold color palette with rich animations.
Based on the existing halo_forge/ui.py terminal colors.
"""

from nicegui import ui

# =============================================================================
# Color Palette
# =============================================================================

COLORS = {
    # Backgrounds
    "bg_primary": "#0f1419",       # Warm near-black
    "bg_secondary": "#1a1f25",     # Slightly lighter
    "bg_card": "#242a31",          # Card backgrounds
    "bg_hover": "#2d343c",         # Hover state
    
    # Brand colors (from halo_forge/ui.py)
    "primary": "#7C9885",          # Sage green
    "secondary": "#A8B5A2",        # Light sage
    "accent": "#C9A959",           # Muted gold
    
    # Text
    "text_primary": "#e8ebe4",     # Warm white
    "text_secondary": "#8a9182",   # Muted sage text
    "text_muted": "#5a6356",       # Very muted
    
    # Status colors
    "success": "#7C9885",          # Sage green
    "error": "#B85C5C",            # Muted red
    "warning": "#C9A959",          # Muted gold
    "info": "#6B8E9B",             # Steel blue
    
    # Status-specific
    "running": "#7C9885",
    "pending": "#8a9182",
    "completed": "#7C9885",
    "failed": "#B85C5C",
    "stopped": "#C9A959",
}

# =============================================================================
# CSS Animations
# =============================================================================

CSS_ANIMATIONS = """
/* Keyframe animations */
@keyframes fadeSlideIn {
    from { 
        opacity: 0; 
        transform: translateY(20px); 
    }
    to { 
        opacity: 1; 
        transform: translateY(0); 
    }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes pulseGlow {
    0%, 100% { 
        box-shadow: 0 0 0 0 rgba(124, 152, 133, 0.4); 
    }
    50% { 
        box-shadow: 0 0 20px 4px rgba(124, 152, 133, 0.2); 
    }
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* Animation utility classes */
.animate-in {
    opacity: 0;
    animation: fadeSlideIn 0.4s ease-out forwards;
}

.animate-fade {
    opacity: 0;
    animation: fadeIn 0.3s ease-out forwards;
}

/* Staggered delays for card reveals */
.stagger-1 { animation-delay: 0.05s; }
.stagger-2 { animation-delay: 0.1s; }
.stagger-3 { animation-delay: 0.15s; }
.stagger-4 { animation-delay: 0.2s; }
.stagger-5 { animation-delay: 0.25s; }
.stagger-6 { animation-delay: 0.3s; }

/* Interactive hover effects */
.card-hover {
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.card-hover:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}

/* Button hover */
.btn-hover {
    transition: all 0.2s ease;
}
.btn-hover:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(124, 152, 133, 0.3);
}

/* Running job glow effect */
.running-glow {
    animation: pulseGlow 2s infinite;
}

/* Spinner */
.spinner {
    animation: spin 1s linear infinite;
}

/* Skeleton loading */
.skeleton {
    background: linear-gradient(
        90deg,
        #242a31 25%,
        #2d343c 50%,
        #242a31 75%
    );
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
}

/* Sidebar nav item - use box-shadow instead of border to avoid layout shift */
.nav-item {
    transition: all 0.15s ease;
    box-shadow: inset 3px 0 0 0 transparent;
    padding-left: 16px !important;
}
.nav-item:hover {
    background: rgba(124, 152, 133, 0.1);
    box-shadow: inset 3px 0 0 0 rgba(124, 152, 133, 0.5);
}
.nav-item.active {
    background: rgba(124, 152, 133, 0.15);
    box-shadow: inset 3px 0 0 0 #7C9885;
}

/* Page transition container */
.page-content {
    opacity: 0;
    animation: fadeSlideIn 0.35s ease-out 0.1s forwards;
}

/* Progress bar fill */
.progress-fill {
    transition: width 0.5s ease-out;
}

/* Tooltip */
.tooltip-hover {
    position: relative;
}
.tooltip-hover::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%) translateY(-4px);
    padding: 4px 8px;
    background: #1a1f25;
    border: 1px solid #2d343c;
    border-radius: 4px;
    font-size: 12px;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s ease;
}
.tooltip-hover:hover::after {
    opacity: 1;
}

/* Grid utilities for consistent layouts */
.grid-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
}

.grid-panels {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
}

/* Equal height cards */
.equal-height {
    display: flex;
    flex-direction: column;
}
.equal-height > * {
    flex: 1;
}
"""

# =============================================================================
# Custom Fonts
# =============================================================================

FONT_IMPORTS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
"""

FONT_STYLES = """
body {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

.font-mono, code, pre, .metrics, .log-viewer {
    font-family: 'JetBrains Mono', monospace;
}
"""

# =============================================================================
# Global Styles
# =============================================================================

GLOBAL_STYLES = f"""
{FONT_IMPORTS}
{FONT_STYLES}
{CSS_ANIMATIONS}

/* Scrollbar styling */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}
::-webkit-scrollbar-track {{
    background: {COLORS['bg_primary']};
}}
::-webkit-scrollbar-thumb {{
    background: {COLORS['bg_hover']};
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: {COLORS['text_muted']};
}}

/* Selection */
::selection {{
    background: rgba(124, 152, 133, 0.3);
}}

/* Focus ring */
*:focus-visible {{
    outline: 2px solid {COLORS['primary']};
    outline-offset: 2px;
}}

/* Base body */
body {{
    background: {COLORS['bg_primary']};
    color: {COLORS['text_primary']};
}}
"""


def apply_theme():
    """Apply the halo-forge theme to the page."""
    ui.add_head_html(f'<style>{GLOBAL_STYLES}</style>')
    ui.query('body').classes(f'bg-[{COLORS["bg_primary"]}]')


def get_status_color(status: str) -> str:
    """Get the appropriate color for a job status."""
    return COLORS.get(status, COLORS['text_secondary'])
