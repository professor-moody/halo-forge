#!/usr/bin/env bash
# Auto-install halo-forge in editable mode when entering the toolbox
# This ensures code changes are picked up immediately

_halo_forge_install() {
    local HALO_FORGE_DIR="${HOME}/projects/halo-forge"
    local MARKER_FILE="/tmp/.halo-forge-installed-${USER}"
    
    # Skip if already installed this session
    [[ -f "$MARKER_FILE" ]] && return 0
    
    # Skip if directory doesn't exist
    [[ ! -d "$HALO_FORGE_DIR" ]] && return 0
    
    # Skip if not in venv
    [[ -z "$VIRTUAL_ENV" ]] && return 0
    
    # Check if halo-forge needs install/reinstall
    # We always reinstall to pick up code changes
    if [[ -f "$HALO_FORGE_DIR/pyproject.toml" ]]; then
        echo -e "\033[0;34m>\033[0m Installing halo-forge from source..."
        if pip install -q -e "$HALO_FORGE_DIR" 2>/dev/null; then
            echo -e "\033[0;32m✓\033[0m halo-forge installed (editable mode)"
            touch "$MARKER_FILE"
        else
            echo -e "\033[0;31m✗\033[0m Failed to install halo-forge"
        fi
    fi
}

# Run on first shell prompt
if [[ -z "$_HALO_FORGE_INIT_DONE" ]]; then
    export _HALO_FORGE_INIT_DONE=1
    _halo_forge_install
fi
