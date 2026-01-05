#!/bin/bash
# =============================================================================
# halo-forge Ubuntu/Docker Build Script
# =============================================================================
# WARNING: EXPERIMENTAL - Ubuntu/Docker support is less tested than Fedora.
# The Fedora toolbox (build.sh) is the primary supported platform.
#
# Builds the halo-forge container image for Ubuntu using Docker.
# Reference: https://github.com/kyuz0/amd-strix-halo-llm-finetuning
#
# Usage:
#   ./build-ubuntu.sh              # Build with cache
#   ./build-ubuntu.sh --no-cache   # Build from scratch (recommended for release)
#   ./build-ubuntu.sh --tag TAG    # Use custom tag
#   ./build-ubuntu.sh --help       # Show help
#
# After building:
#   docker run -it --device=/dev/kfd --device=/dev/dri \
#     --security-opt seccomp=unconfined \
#     -v ~/projects:/workspace halo-forge:ubuntu
# =============================================================================

set -e

# Configuration
IMAGE_NAME="halo-forge"
IMAGE_TAG="ubuntu"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/build-ubuntu.log"
DOCKERFILE="Dockerfile.ubuntu"

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
    echo -e "${BOLD}│${NC}   ${GREEN}HALO-FORGE${NC} Ubuntu/Docker Builder                           ${BOLD}│${NC}"
    echo -e "${BOLD}│${NC}                                                              ${BOLD}│${NC}"
    echo -e "${BOLD}│${NC}   ${DIM}RLVR Training Framework for AMD Strix Halo${NC}                 ${BOLD}│${NC}"
    echo -e "${BOLD}│${NC}                                                              ${BOLD}│${NC}"
    echo -e "${BOLD}│${NC}   ${YELLOW}⚠ EXPERIMENTAL - Fedora toolbox is primary platform${NC}       ${BOLD}│${NC}"
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
    echo "Build the halo-forge Ubuntu/Docker container image."
    echo ""
    echo "Options:"
    echo "  --no-cache    Build without using cached layers (recommended for release)"
    echo "  --tag TAG     Use custom tag (default: ubuntu)"
    echo "  --help        Show this help message"
    echo ""
    echo "Build Requirements:"
    echo "  - docker installed"
    echo "  - ~25GB disk space"
    echo "  - Internet connection (downloads ~5GB of data)"
    echo ""
    echo "Build time estimates:"
    echo "  - With cache:    5-10 minutes"
    echo "  - Without cache: 20-40 minutes (depends on network)"
    echo ""
    echo "After building:"
    echo "  docker run -it --device=/dev/kfd --device=/dev/dri \\"
    echo "    -v ~/projects:/workspace ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
}

check_prerequisites() {
    print_step "Checking prerequisites..."
    
    # Check for docker
    if ! command -v docker &> /dev/null; then
        print_error "docker not found. Please install docker first."
        echo "  On Ubuntu: sudo apt-get install docker.io"
        echo "  Or install Docker Desktop: https://www.docker.com/products/docker-desktop"
        exit 1
    fi
    print_success "docker found: $(docker --version | head -1)"
    
    # Check docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon not running. Start it with: sudo systemctl start docker"
        exit 1
    fi
    print_success "Docker daemon is running"
    
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
    if docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" &> /dev/null; then
        print_warning "Existing image found: ${IMAGE_NAME}:${IMAGE_TAG}"
        echo -e "         ${DIM}Will be replaced after successful build${NC}"
    fi
    
    # Check for dangling images
    dangling=$(docker images -f "dangling=true" -q 2>/dev/null | wc -l)
    if [[ "${dangling}" -gt 0 ]]; then
        print_warning "${dangling} dangling images found"
        echo -e "         ${DIM}Run 'docker image prune' to clean up${NC}"
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
    
    echo -e "${DIM}Image: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
    echo -e "${DIM}Log:   ${LOG_FILE}${NC}"
    echo ""
    
    # Record start time
    start_time=$(date +%s)
    
    # Build
    cd "${SCRIPT_DIR}"
    
    echo -e "${DIM}─────────────────────────────────────────────────────────────${NC}"
    
    if docker build ${build_args} -t "${IMAGE_NAME}:${IMAGE_TAG}" -f "${DOCKERFILE}" . 2>&1 | tee "${LOG_FILE}"; then
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
        docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" --format '  Size: {{.Size}}' 2>/dev/null | \
            awk '{printf "  Size: %.2f GB\n", $2/1024/1024/1024}'
        docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" --format '  Created: {{.Created}}' 2>/dev/null
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
    if docker run --rm "${IMAGE_NAME}:${IMAGE_TAG}" python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        print_success "PyTorch installed correctly"
    else
        print_warning "Could not verify PyTorch (may need GPU to fully test)"
    fi
    
    if docker run --rm "${IMAGE_NAME}:${IMAGE_TAG}" python -c "import transformers; print(f'Transformers {transformers.__version__}')" 2>/dev/null; then
        print_success "Transformers installed correctly"
    else
        print_warning "Could not verify Transformers"
    fi
    
    echo ""
}

print_next_steps() {
    echo ""
    echo -e "${BOLD}Next Steps${NC}"
    echo -e "${DIM}─────────────────────────────────────────────────────────────${NC}"
    echo ""
    echo -e "${YELLOW}Note: Ubuntu/Docker support is experimental. Use Fedora toolbox for production.${NC}"
    echo ""
    echo "1. If GPU devices are not visible, add udev rules:"
    echo -e "   ${DIM}sudo tee /etc/udev/rules.d/99-amd-kfd.rules >/dev/null <<'EOF'${NC}"
    echo -e "   ${DIM}SUBSYSTEM==\"kfd\", GROUP=\"render\", MODE=\"0666\"${NC}"
    echo -e "   ${DIM}SUBSYSTEM==\"drm\", KERNEL==\"card[0-9]*\", GROUP=\"render\", MODE=\"0666\"${NC}"
    echo -e "   ${DIM}EOF${NC}"
    echo -e "   ${DIM}sudo udevadm control --reload-rules && sudo udevadm trigger${NC}"
    echo ""
    echo "2. Run the container with GPU access:"
    echo -e "   ${GREEN}docker run -it --device=/dev/kfd --device=/dev/dri \\${NC}"
    echo -e "   ${GREEN}  --security-opt seccomp=unconfined \\${NC}"
    echo -e "   ${GREEN}  -v ~/projects:/workspace \\${NC}"
    echo -e "   ${GREEN}  ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
    echo ""
    echo "3. Or with user mapping:"
    echo -e "   ${GREEN}docker run -it --device=/dev/kfd --device=/dev/dri \\${NC}"
    echo -e "   ${GREEN}  --security-opt seccomp=unconfined \\${NC}"
    echo -e "   ${GREEN}  -v ~/projects:/workspace \\${NC}"
    echo -e "   ${GREEN}  -u \$(id -u):\$(id -g) \\${NC}"
    echo -e "   ${GREEN}  ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
    echo ""
    echo "4. Install halo-forge package:"
    echo -e "   ${GREEN}cd /workspace/halo-forge && pip install -e .${NC}"
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

for arg in "$@"; do
    case $arg in
        --no-cache)
            NO_CACHE="true"
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

