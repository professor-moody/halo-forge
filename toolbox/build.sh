#!/bin/bash
# =============================================================================
# halo-forge Toolbox Build Script
# =============================================================================
# Builds the halo-forge container image optimized for AMD Strix Halo (gfx1151)
#
# Usage:
#   ./build.sh              # Build with cache
#   ./build.sh --no-cache   # Build from scratch
#   ./build.sh --help       # Show help
#
# After building:
#   toolbox create halo-forge --image localhost/halo-forge:latest
#   toolbox enter halo-forge
# =============================================================================

set -e

# Configuration
IMAGE_NAME="halo-forge"
IMAGE_TAG="latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors (if terminal supports them)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    DIM='\033[0;90m'
    BOLD='\033[1m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    DIM=''
    BOLD=''
    NC=''
fi

# Functions
print_header() {
    echo ""
    echo -e "${BOLD}╭──────────────────────────────────────────────────────────────╮${NC}"
    echo -e "${BOLD}│${NC}                                                              ${BOLD}│${NC}"
    echo -e "${BOLD}│${NC}   ${GREEN}HALO-FORGE${NC} Toolbox Builder                                ${BOLD}│${NC}"
    echo -e "${BOLD}│${NC}                                                              ${BOLD}│${NC}"
    echo -e "${BOLD}│${NC}   ${DIM}RLVR Training Framework for AMD Strix Halo${NC}                 ${BOLD}│${NC}"
    echo -e "${BOLD}│${NC}                                                              ${BOLD}│${NC}"
    echo -e "${BOLD}╰──────────────────────────────────────────────────────────────╯${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
}

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Build the halo-forge toolbox container image."
    echo ""
    echo "Options:"
    echo "  --no-cache    Build without using cached layers"
    echo "  --help        Show this help message"
    echo ""
    echo "After building:"
    echo "  toolbox create halo-forge --image localhost/${IMAGE_NAME}:${IMAGE_TAG}"
    echo "  toolbox enter halo-forge"
    echo ""
}

check_prerequisites() {
    print_step "Checking prerequisites..."
    
    # Check for podman
    if ! command -v podman &> /dev/null; then
        print_error "podman not found. Please install podman first."
        echo "  On Fedora: sudo dnf install podman"
        exit 1
    fi
    print_success "podman found: $(podman --version | head -1)"
    
    # Check for Dockerfile
    if [[ ! -f "${SCRIPT_DIR}/Dockerfile" ]]; then
        print_error "Dockerfile not found in ${SCRIPT_DIR}"
        exit 1
    fi
    print_success "Dockerfile found"
    
    # Check disk space (need at least 20GB for build)
    available_gb=$(df -BG "${SCRIPT_DIR}" | tail -1 | awk '{print $4}' | tr -d 'G')
    if [[ "${available_gb}" -lt 20 ]]; then
        print_warning "Low disk space: ${available_gb}GB available (recommended: 20GB+)"
    else
        print_success "Disk space: ${available_gb}GB available"
    fi
    
    echo ""
}

build_image() {
    local build_args=""
    
    if [[ "${NO_CACHE}" == "true" ]]; then
        build_args="--no-cache"
        print_step "Building without cache (this will take longer)..."
    else
        print_step "Building with cache..."
    fi
    
    echo -e "${DIM}Image: localhost/${IMAGE_NAME}:${IMAGE_TAG}${NC}"
    echo ""
    
    # Record start time
    start_time=$(date +%s)
    
    # Build
    cd "${SCRIPT_DIR}"
    if podman build ${build_args} -t "${IMAGE_NAME}:${IMAGE_TAG}" -f Dockerfile .; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        minutes=$((duration / 60))
        seconds=$((duration % 60))
        
        echo ""
        print_success "Build complete in ${minutes}m ${seconds}s"
    else
        echo ""
        print_error "Build failed!"
        exit 1
    fi
}

print_next_steps() {
    echo ""
    echo -e "${BOLD}Next Steps${NC}"
    echo -e "${DIM}─────────────────────────────────────────────────────────────${NC}"
    echo ""
    echo "1. Create the toolbox:"
    echo -e "   ${GREEN}toolbox create halo-forge --image localhost/${IMAGE_NAME}:${IMAGE_TAG}${NC}"
    echo ""
    echo "2. Enter the toolbox:"
    echo -e "   ${GREEN}toolbox enter halo-forge${NC}"
    echo ""
    echo "3. Verify setup:"
    echo -e "   ${GREEN}halo-forge test --level smoke${NC}"
    echo ""
}

# =============================================================================
# Main
# =============================================================================

# Parse arguments
NO_CACHE="false"
for arg in "$@"; do
    case $arg in
        --no-cache)
            NO_CACHE="true"
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $arg"
            show_help
            exit 1
            ;;
    esac
done

# Run
print_header
check_prerequisites
build_image
print_next_steps
