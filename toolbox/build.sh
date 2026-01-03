#!/bin/bash
# =============================================================================
# halo-forge Toolbox Build Script
# =============================================================================
# Builds the halo-forge container image optimized for AMD Strix Halo (gfx1151)
#
# Usage:
#   ./build.sh              # Build with cache
#   ./build.sh --no-cache   # Build from scratch (recommended for release)
#   ./build.sh --quick      # Quick build for testing (skip heavy deps)
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
LOG_FILE="${SCRIPT_DIR}/build.log"

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
    echo "  --no-cache    Build without using cached layers (recommended for release)"
    echo "  --core        Build core image without TUI (uses Dockerfile.core)"
    echo "  --tag TAG     Use custom tag (default: latest)"
    echo "  --help        Show this help message"
    echo ""
    echo "Build variants:"
    echo "  Default:      Full build with TUI support (textual, rich)"
    echo "  --core:       Minimal build for CLI-only usage (no textual)"
    echo ""
    echo "Build Requirements:"
    echo "  - podman installed"
    echo "  - ~25GB disk space"
    echo "  - Internet connection (downloads ~5GB of data)"
    echo ""
    echo "Build time estimates:"
    echo "  - With cache:    5-10 minutes"
    echo "  - Without cache: 20-40 minutes (depends on network)"
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
    if [[ ! -f "${SCRIPT_DIR}/${DOCKERFILE}" ]]; then
        print_error "${DOCKERFILE} not found in ${SCRIPT_DIR}"
        exit 1
    fi
    print_success "${DOCKERFILE} found"
    
    # Check for scripts directory
    if [[ ! -d "${SCRIPT_DIR}/scripts" ]]; then
        print_error "scripts/ directory not found in ${SCRIPT_DIR}"
        echo "  Expected files: scripts/01-triton-env.sh, scripts/99-halo-forge-banner.sh, scripts/zz-venv-path-fix.sh"
        exit 1
    fi
    
    # Check required scripts exist
    local required_scripts=("01-triton-env.sh" "99-halo-forge-banner.sh" "zz-venv-path-fix.sh")
    for script in "${required_scripts[@]}"; do
        if [[ ! -f "${SCRIPT_DIR}/scripts/${script}" ]]; then
            print_error "Missing script: scripts/${script}"
            exit 1
        fi
    done
    print_success "All required scripts found"
    
    # Check disk space (need at least 25GB for build)
    available_gb=$(df -BG "${SCRIPT_DIR}" | tail -1 | awk '{print $4}' | tr -d 'G')
    if [[ "${available_gb}" -lt 25 ]]; then
        print_warning "Low disk space: ${available_gb}GB available (recommended: 25GB+)"
        echo "         Build may fail if space runs out"
    else
        print_success "Disk space: ${available_gb}GB available"
    fi
    
    # Check network connectivity
    if ! curl -s --connect-timeout 5 https://therock-nightly-tarball.s3.amazonaws.com > /dev/null 2>&1; then
        print_warning "Cannot reach ROCm nightly server - build may fail"
    else
        print_success "Network connectivity OK"
    fi
    
    echo ""
}

cleanup_old_images() {
    print_step "Checking for existing images..."
    
    # Check if image exists
    if podman image exists "${IMAGE_NAME}:${IMAGE_TAG}" 2>/dev/null; then
        print_warning "Existing image found: ${IMAGE_NAME}:${IMAGE_TAG}"
        echo -e "         ${DIM}Will be replaced after successful build${NC}"
    fi
    
    # Check for dangling images
    dangling=$(podman images -f "dangling=true" -q 2>/dev/null | wc -l)
    if [[ "${dangling}" -gt 0 ]]; then
        print_warning "${dangling} dangling images found"
        echo -e "         ${DIM}Run 'podman image prune' to clean up${NC}"
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
    
    if [[ "${CORE_BUILD}" == "true" ]]; then
        echo -e "${DIM}Build type: Core (CLI only, no TUI)${NC}"
    else
        echo -e "${DIM}Build type: Full (with TUI support)${NC}"
    fi
    echo -e "${DIM}Image: localhost/${IMAGE_NAME}:${IMAGE_TAG}${NC}"
    echo -e "${DIM}Log:   ${LOG_FILE}${NC}"
    echo ""
    
    # Record start time
    start_time=$(date +%s)
    
    # Build
    cd "${SCRIPT_DIR}"
    
    echo -e "${DIM}─────────────────────────────────────────────────────────────${NC}"
    
    if podman build ${build_args} -t "${IMAGE_NAME}:${IMAGE_TAG}" -f "${DOCKERFILE}" . 2>&1 | tee "${LOG_FILE}"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        minutes=$((duration / 60))
        seconds=$((duration % 60))
        
        echo -e "${DIM}─────────────────────────────────────────────────────────────${NC}"
        echo ""
        print_success "Build complete in ${minutes}m ${seconds}s"
        
        # Show image info
        echo ""
        print_step "Image details:"
        podman image inspect "${IMAGE_NAME}:${IMAGE_TAG}" --format '  Size: {{.Size}}' 2>/dev/null | \
            awk '{printf "  Size: %.2f GB\n", $2/1024/1024/1024}'
        podman image inspect "${IMAGE_NAME}:${IMAGE_TAG}" --format '  Created: {{.Created}}' 2>/dev/null
    else
        echo ""
        print_error "Build failed!"
        echo ""
        echo "Check the log file for details:"
        echo "  ${LOG_FILE}"
        echo ""
        echo "Common issues:"
        echo "  - Network timeout: Try again or use a VPN"
        echo "  - Disk space: Free up space and retry"
        echo "  - ROCm tarball not found: Check https://therock-nightly-tarball.s3.amazonaws.com"
        exit 1
    fi
}

verify_build() {
    print_step "Verifying build..."
    
    # Quick verification - check that key packages are installed
    if podman run --rm "${IMAGE_NAME}:${IMAGE_TAG}" python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        print_success "PyTorch installed correctly"
    else
        print_warning "Could not verify PyTorch (may be OK in toolbox context)"
    fi
    
    if podman run --rm "${IMAGE_NAME}:${IMAGE_TAG}" python -c "import transformers; print(f'Transformers {transformers.__version__}')" 2>/dev/null; then
        print_success "Transformers installed correctly"
    else
        print_warning "Could not verify Transformers"
    fi
    
    # Only check Textual for full builds
    if [[ "${CORE_BUILD}" != "true" ]]; then
        if podman run --rm "${IMAGE_NAME}:${IMAGE_TAG}" python -c "import textual; print(f'Textual {textual.__version__}')" 2>/dev/null; then
            print_success "Textual (TUI) installed correctly"
        else
            print_warning "Could not verify Textual"
        fi
    else
        print_success "Core build (TUI not included)"
    fi
    
    echo ""
}

print_next_steps() {
    echo ""
    echo -e "${BOLD}Next Steps${NC}"
    echo -e "${DIM}─────────────────────────────────────────────────────────────${NC}"
    echo ""
    echo "1. Remove old toolbox (if exists):"
    echo -e "   ${GREEN}toolbox rm -f halo-forge || true${NC}"
    echo ""
    echo "2. Create the toolbox:"
    echo -e "   ${GREEN}toolbox create halo-forge --image localhost/${IMAGE_NAME}:${IMAGE_TAG}${NC}"
    echo ""
    echo "3. Enter the toolbox:"
    echo -e "   ${GREEN}toolbox enter halo-forge${NC}"
    echo ""
    echo "4. Install halo-forge package:"
    echo -e "   ${GREEN}cd /home/\$USER/projects/halo-forge && pip install -e .${NC}"
    echo ""
    echo "5. Verify setup:"
    echo -e "   ${GREEN}halo-forge test --level smoke${NC}"
    echo ""
    echo "6. Start training:"
    echo -e "   ${GREEN}halo-forge raft train --help${NC}"
    echo ""
}

# =============================================================================
# Main
# =============================================================================

# Parse arguments
NO_CACHE="false"
CORE_BUILD="false"
DOCKERFILE="Dockerfile"

for arg in "$@"; do
    case $arg in
        --no-cache)
            NO_CACHE="true"
            ;;
        --core)
            CORE_BUILD="true"
            DOCKERFILE="Dockerfile.core"
            IMAGE_NAME="halo-forge-core"
            ;;
        --tag)
            shift
            IMAGE_TAG="$1"
            ;;
        --tag=*)
            IMAGE_TAG="${arg#*=}"
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
cleanup_old_images
build_image
verify_build
print_next_steps
