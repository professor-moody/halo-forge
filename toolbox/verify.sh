#!/bin/bash
# =============================================================================
# halo-forge Post-Build Verification Script
# =============================================================================
# Validates that the toolbox environment is correctly configured for training.
#
# Usage:
#   ./verify.sh              # Run all checks
#   ./verify.sh --quick      # Quick checks only (no GPU test)
#   ./verify.sh --gpu        # Include GPU memory test
# =============================================================================

# Don't use set -e as we want to continue through failures and report at the end

# Colors
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    DIM='\033[0;90m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    DIM=''
    BOLD=''
    NC=''
fi

# Test results
PASSED=0
FAILED=0
WARNINGS=0

print_header() {
    echo ""
    echo -e "${BOLD}╭──────────────────────────────────────────────────────────────╮${NC}"
    echo -e "${BOLD}│${NC}   ${GREEN}HALO-FORGE${NC} Environment Verification                       ${BOLD}│${NC}"
    echo -e "${BOLD}╰──────────────────────────────────────────────────────────────╯${NC}"
    echo ""
}

pass() {
    echo -e "  ${GREEN}✓${NC} $1"
    ((PASSED++))
}

fail() {
    echo -e "  ${RED}✗${NC} $1"
    ((FAILED++))
}

warn() {
    echo -e "  ${YELLOW}!${NC} $1"
    ((WARNINGS++))
}

section() {
    echo ""
    echo -e "${BLUE}$1${NC}"
    echo -e "${DIM}─────────────────────────────────────────────────────────────${NC}"
}

# =============================================================================
# Tests
# =============================================================================

test_python() {
    section "Python Environment"
    
    # Check Python version
    if python3 --version &>/dev/null; then
        ver=$(python3 --version 2>&1)
        pass "Python installed: ${ver}"
    else
        fail "Python not found"
        return 1
    fi
    
    # Check venv is active
    if [[ -n "$VIRTUAL_ENV" ]]; then
        pass "Virtual environment active: ${VIRTUAL_ENV}"
    else
        warn "Virtual environment not detected"
    fi
}

test_pytorch() {
    section "PyTorch & ROCm"
    
    # Check PyTorch import
    if python3 -c "import torch" 2>/dev/null; then
        ver=$(python3 -c "import torch; print(torch.__version__)")
        pass "PyTorch installed: ${ver}"
    else
        fail "PyTorch not installed"
        return 1
    fi
    
    # Check CUDA/ROCm availability
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        gpu=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
        pass "GPU detected: ${gpu}"
    else
        fail "GPU not available (torch.cuda.is_available() = False)"
    fi
    
    # Check ROCm environment
    if [[ -n "$ROCM_PATH" ]]; then
        pass "ROCM_PATH set: ${ROCM_PATH}"
    else
        warn "ROCM_PATH not set"
    fi
}

test_ml_stack() {
    section "ML Libraries"
    
    local libs=("transformers" "peft" "accelerate" "datasets" "trl" "bitsandbytes")
    
    for lib in "${libs[@]}"; do
        ver=$(python3 -c "from importlib.metadata import version; print(version('${lib}'))" 2>/dev/null)
        if [[ -n "$ver" ]]; then
            pass "${lib}: ${ver}"
        else
            fail "${lib} not installed"
        fi
    done
}

test_tui() {
    section "TUI & CLI"
    
    # Check Rich (optional but recommended)
    if python3 -c "import rich" 2>/dev/null; then
        ver=$(python3 -c "from importlib.metadata import version; print(version('rich'))" 2>/dev/null || echo "installed")
        pass "Rich installed: ${ver}"
    else
        warn "Rich not installed (optional, provides nicer CLI output)"
    fi
    
    # Check Textual (optional, for TUI)
    if python3 -c "import textual" 2>/dev/null; then
        ver=$(python3 -c "import textual; print(textual.__version__)")
        pass "Textual installed: ${ver}"
    else
        warn "Textual not installed (TUI disabled, CLI still works)"
    fi
    
    # Check halo-forge CLI
    if command -v halo-forge &>/dev/null; then
        pass "halo-forge CLI available"
    else
        warn "halo-forge CLI not in PATH (run: pip install -e .)"
    fi
}

test_verifiers() {
    section "Verifiers"
    
    # Check GCC
    if command -v g++ &>/dev/null; then
        ver=$(g++ --version | head -1)
        pass "GCC: ${ver}"
    else
        fail "GCC (g++) not found"
    fi
    
    # Check Clang
    if command -v clang++ &>/dev/null; then
        ver=$(clang++ --version | head -1)
        pass "Clang: ${ver}"
    else
        warn "Clang not found (optional)"
    fi
    
    # Check MinGW (for Windows cross-compile)
    if command -v x86_64-w64-mingw32-g++ &>/dev/null; then
        pass "MinGW cross-compiler available"
    else
        warn "MinGW not found (Windows verification disabled)"
    fi
    
    # Check pytest
    if python3 -c "import pytest" 2>/dev/null; then
        pass "Pytest available"
    else
        fail "Pytest not installed"
    fi
}

test_flash_attention() {
    section "Flash Attention"
    
    if python3 -c "import flash_attn" 2>/dev/null; then
        ver=$(python3 -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null || echo "unknown")
        pass "Flash Attention installed: ${ver}"
    else
        warn "Flash Attention not installed (will use eager attention)"
    fi
    
    if [[ "$FLASH_ATTENTION_TRITON_AMD_ENABLE" == "TRUE" ]]; then
        pass "FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE"
    else
        warn "FLASH_ATTENTION_TRITON_AMD_ENABLE not set"
    fi
}

test_gpu_memory() {
    section "GPU Memory Test"
    
    echo -e "  ${DIM}Allocating test tensor...${NC}"
    
    result=$(python3 -c "
import torch
import warnings
warnings.filterwarnings('ignore')
try:
    if not torch.cuda.is_available():
        print('FAIL:GPU not available')
    else:
        x = torch.zeros(256, 1024, 1024, dtype=torch.float32, device='cuda')
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        del x
        torch.cuda.empty_cache()
        print(f'PASS:Allocated 1GB tensor successfully (peak: {mem_gb:.2f}GB)')
except Exception as e:
    print(f'FAIL:{e}')
" 2>/dev/null)
    
    if [[ "$result" == PASS:* ]]; then
        pass "${result#PASS:}"
    else
        fail "${result#FAIL:}"
    fi
}

test_halo_forge_modules() {
    section "halo-forge Modules"
    
    # Core modules (required)
    local core_modules=(
        "halo_forge.rlvr.raft_trainer:RAFTTrainer"
        "halo_forge.rlvr.verifiers:GCCVerifier"
        "halo_forge.sft.trainer:SFTTrainer"
        "halo_forge.cli:main"
    )
    
    for mod in "${core_modules[@]}"; do
        module="${mod%:*}"
        cls="${mod#*:}"
        if python3 -c "from ${module} import ${cls}" 2>/dev/null; then
            pass "${module}.${cls}"
        else
            warn "${module}.${cls} not importable (run: pip install -e .)"
        fi
    done
    
    # TUI module (optional)
    if python3 -c "from halo_forge.tui.app import HaloForgeApp" 2>/dev/null; then
        pass "halo_forge.tui.app.HaloForgeApp"
    else
        warn "halo_forge.tui.app.HaloForgeApp not available (TUI disabled)"
    fi
}

print_summary() {
    echo ""
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}Summary${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${GREEN}Passed:${NC}   ${PASSED}"
    echo -e "  ${YELLOW}Warnings:${NC} ${WARNINGS}"
    echo -e "  ${RED}Failed:${NC}   ${FAILED}"
    echo ""
    
    if [[ ${FAILED} -eq 0 ]]; then
        if [[ ${WARNINGS} -eq 0 ]]; then
            echo -e "${GREEN}All checks passed! Environment is ready for training.${NC}"
        else
            echo -e "${YELLOW}Environment ready with warnings. Some optional features may be unavailable.${NC}"
        fi
        return 0
    else
        echo -e "${RED}Some checks failed. Please review and fix before training.${NC}"
        return 1
    fi
}

# =============================================================================
# Main
# =============================================================================

QUICK=false
GPU_TEST=false

for arg in "$@"; do
    case $arg in
        --quick)
            QUICK=true
            ;;
        --gpu)
            GPU_TEST=true
            ;;
        --help|-h)
            echo "Usage: $0 [--quick] [--gpu]"
            echo ""
            echo "Options:"
            echo "  --quick    Skip GPU and module tests"
            echo "  --gpu      Include GPU memory allocation test"
            exit 0
            ;;
    esac
done

print_header

test_python
test_pytorch
test_ml_stack
test_tui
test_verifiers
test_flash_attention

if [[ "$QUICK" != "true" ]]; then
    test_halo_forge_modules
fi

if [[ "$GPU_TEST" == "true" ]]; then
    test_gpu_memory
fi

print_summary

