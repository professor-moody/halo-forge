#!/bin/bash
# =============================================================================
# halo-forge Environment Validation Script
# =============================================================================
# Validates that the toolbox has all required dependencies for all phases.
#
# Usage:
#   ./scripts/validate_environment.sh
#
# Run inside the halo-forge toolbox:
#   toolbox enter halo-forge
#   cd /path/to/halo-forge
#   ./scripts/validate_environment.sh
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           halo-forge Environment Validation                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

FAILED=0

check_python() {
    local module=$1
    local name=${2:-$1}
    if python -c "import $module" 2>/dev/null; then
        version=$(python -c "import $module; print(getattr($module, '__version__', 'OK'))" 2>/dev/null || echo "OK")
        echo -e "  ${GREEN}✓${NC} $name: $version"
    else
        echo -e "  ${RED}✗${NC} $name: NOT FOUND"
        FAILED=1
    fi
}

check_command() {
    local cmd=$1
    if command -v "$cmd" &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $cmd: $(command -v $cmd)"
    else
        echo -e "  ${RED}✗${NC} $cmd: NOT FOUND"
        FAILED=1
    fi
}

check_halo_forge() {
    local module=$1
    local name=$2
    if python -c "from halo_forge.$module import *" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} halo_forge.$name"
    else
        echo -e "  ${RED}✗${NC} halo_forge.$name: NOT FOUND"
        FAILED=1
    fi
}

# =============================================================================
# Core Dependencies
# =============================================================================
echo -e "\n${YELLOW}=== Core Dependencies ===${NC}"

check_python "torch" "PyTorch"
check_python "transformers" "Transformers"
check_python "peft" "PEFT"
check_python "accelerate" "Accelerate"
check_python "datasets" "Datasets"
check_python "trl" "TRL"

# Check ROCm/HIP
echo ""
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo -e "  ${GREEN}✓${NC} GPU Available: $GPU_NAME"
else
    echo -e "  ${YELLOW}⚠${NC} GPU: Not available (CPU-only mode)"
fi

# =============================================================================
# Phase 1-2: Code + Inference
# =============================================================================
echo -e "\n${YELLOW}=== Phase 1-2: Code + Inference ===${NC}"

check_python "bitsandbytes" "bitsandbytes"
check_command "gcc"
check_command "x86_64-w64-mingw32-g++"
check_command "rustc"
check_command "go"
check_command "dotnet"

# =============================================================================
# Phase 3: VLM
# =============================================================================
echo -e "\n${YELLOW}=== Phase 3: Vision-Language ===${NC}"

check_python "ultralytics" "ultralytics (YOLO)"
check_python "easyocr" "easyocr"
check_python "PIL" "Pillow"
check_halo_forge "vlm" "vlm"

# =============================================================================
# Phase 4: Audio
# =============================================================================
echo -e "\n${YELLOW}=== Phase 4: Audio-Language ===${NC}"

check_python "torchaudio" "torchaudio"
check_python "librosa" "librosa"
check_python "jiwer" "jiwer (WER)"
check_halo_forge "audio" "audio"

# =============================================================================
# Phase 5: Reasoning
# =============================================================================
echo -e "\n${YELLOW}=== Phase 5: Reasoning ===${NC}"

check_python "sympy" "sympy"
check_halo_forge "reasoning" "reasoning"

# =============================================================================
# Phase 6: Agentic
# =============================================================================
echo -e "\n${YELLOW}=== Phase 6: Agentic ===${NC}"

check_halo_forge "agentic" "agentic"

# Check specific agentic imports
if python -c "from halo_forge.agentic import ToolCallingVerifier, AgenticRAFTTrainer" 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} ToolCallingVerifier"
    echo -e "  ${GREEN}✓${NC} AgenticRAFTTrainer"
else
    echo -e "  ${RED}✗${NC} Agentic verifier/trainer import failed"
    FAILED=1
fi

# =============================================================================
# Training Utilities
# =============================================================================
echo -e "\n${YELLOW}=== Training Utilities ===${NC}"

check_python "tensorboard" "TensorBoard"
check_python "rich" "Rich"
check_python "pytest" "pytest"

# =============================================================================
# halo-forge CLI
# =============================================================================
echo -e "\n${YELLOW}=== halo-forge CLI ===${NC}"

if python -m halo_forge.cli --help &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} halo-forge CLI available"
else
    echo -e "  ${RED}✗${NC} halo-forge CLI not working"
    FAILED=1
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All dependencies validated successfully!${NC}"
    echo ""
    echo "Environment is ready for:"
    echo "  - Code training (gcc, mingw, rust, go)"
    echo "  - VLM training (ultralytics, easyocr)"
    echo "  - Audio training (torchaudio, librosa)"
    echo "  - Reasoning training (sympy)"
    echo "  - Agentic training (tool calling)"
    echo ""
    exit 0
else
    echo -e "${RED}Some dependencies are missing!${NC}"
    echo ""
    echo "Please ensure you are running inside the halo-forge toolbox:"
    echo "  toolbox enter halo-forge"
    echo ""
    exit 1
fi
