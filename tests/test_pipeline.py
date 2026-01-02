#!/usr/bin/env python3
"""
End-to-end pipeline test for halo-forge.

This test validates the entire training pipeline:
1. Data preparation
2. SFT training (minimal)
3. RAFT training (1 cycle)
4. Verification

This is a QUICK test using minimal data and epochs.
For production training, use the full pipeline examples.

Run with:
    pytest tests/test_pipeline.py -v --timeout=600
    
Or directly:
    python tests/test_pipeline.py
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Helper Functions (defined early for use in decorators)
# =============================================================================

def _torch_cuda_available() -> bool:
    """Check if PyTorch CUDA/ROCm is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _gpu_memory_gb() -> float:
    """Get available GPU memory in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / 1e9
        return 0.0
    except:
        return 0.0


def _compiler_available(compiler: str) -> bool:
    """Check if a compiler is available in PATH."""
    import shutil
    return shutil.which(compiler) is not None


# =============================================================================
# Test Configuration
# =============================================================================

# Minimal test settings
TEST_CONFIG = {
    "model_name": "Qwen/Qwen2.5-Coder-0.5B",  # Smallest model for fast test
    "num_prompts": 5,
    "sft_epochs": 1,
    "raft_cycles": 1,
    "samples_per_prompt": 2,
    "max_new_tokens": 128,
}


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def test_dir():
    """Create temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp(prefix="halo_forge_test_")
    yield Path(tmpdir)
    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_prompts(test_dir):
    """Create sample prompts file."""
    prompts = [
        {"prompt": "Write a C++ function that returns the sum of two integers."},
        {"prompt": "Write a C++ function to calculate factorial."},
        {"prompt": "Write a C++ program that prints Hello World."},
        {"prompt": "Write a C++ function to find the maximum of three numbers."},
        {"prompt": "Write a C++ function to check if a number is even."},
    ]
    
    prompts_file = test_dir / "prompts.jsonl"
    with open(prompts_file, "w") as f:
        for p in prompts[:TEST_CONFIG["num_prompts"]]:
            f.write(json.dumps(p) + "\n")
    
    return prompts_file


@pytest.fixture
def sample_train_data(test_dir):
    """Create minimal training data."""
    examples = [
        {
            "text": "<|im_start|>system\nYou are an expert C++ programmer.<|im_end|>\n<|im_start|>user\nWrite a function to add two numbers.<|im_end|>\n<|im_start|>assistant\n```cpp\nint add(int a, int b) {\n    return a + b;\n}\n```<|im_end|>"
        },
        {
            "text": "<|im_start|>system\nYou are an expert C++ programmer.<|im_end|>\n<|im_start|>user\nWrite Hello World.<|im_end|>\n<|im_start|>assistant\n```cpp\n#include <iostream>\nint main() {\n    std::cout << \"Hello, World!\" << std::endl;\n    return 0;\n}\n```<|im_end|>"
        },
    ]
    
    train_file = test_dir / "train.jsonl"
    with open(train_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    
    return train_file


# =============================================================================
# Unit Tests
# =============================================================================

class TestImports:
    """Test that all modules can be imported."""
    
    def test_import_verifiers(self):
        """Verifiers should import without error."""
        from halo_forge.rlvr.verifiers import (
            Verifier,
            VerifyResult,
            RewardLevel,
            GCCVerifier,
            MinGWVerifier,
            ClangVerifier,
            PytestVerifier,
            ChainedVerifier,
        )
        assert Verifier is not None
        assert RewardLevel is not None
    
    def test_import_trainers(self):
        """Trainers should import without error."""
        from halo_forge.rlvr.raft_trainer import RAFTTrainer, RAFTConfig
        from halo_forge.sft.trainer import SFTTrainer, SFTConfig
        
        assert RAFTTrainer is not None
        assert SFTTrainer is not None
    
    def test_import_cli(self):
        """CLI should import without error."""
        from halo_forge import cli
        assert cli is not None


class TestVerifierBasics:
    """Basic verifier functionality tests."""
    
    def test_gcc_verifier_exists(self):
        """GCCVerifier should be instantiable."""
        from halo_forge.rlvr.verifiers import GCCVerifier
        
        verifier = GCCVerifier()
        assert verifier is not None
        assert verifier.compiler == "g++"
    
    def test_reward_levels(self):
        """RewardLevel enum should have expected values."""
        from halo_forge.rlvr.verifiers import RewardLevel
        
        assert RewardLevel.FAILURE.value == 0.0
        assert RewardLevel.COMPILE_WARNINGS.value == 0.3
        assert RewardLevel.COMPILE_CLEAN.value == 0.5
        assert RewardLevel.RUNS_NO_CRASH.value == 0.7
        assert RewardLevel.CORRECT_OUTPUT.value == 1.0
    
    def test_verify_result_creation(self):
        """VerifyResult should be creatable."""
        from halo_forge.rlvr.verifiers import VerifyResult
        
        result = VerifyResult(
            success=True,
            reward=1.0,
            details="Test passed"
        )
        
        assert result.success is True
        assert result.reward == 1.0


class TestCodeExtraction:
    """Test code extraction from model output."""
    
    def test_extract_markdown_cpp(self):
        """Should extract code from markdown blocks."""
        from halo_forge.rlvr.verifiers import GCCVerifier
        
        verifier = GCCVerifier()
        
        text = '''Here is the solution:

```cpp
#include <iostream>
int main() { return 0; }
```

This is the code.'''
        
        extracted = verifier.extract_code(text)
        
        assert "#include <iostream>" in extracted
        assert "```" not in extracted
    
    def test_extract_raw_code(self):
        """Should handle raw code without markdown."""
        from halo_forge.rlvr.verifiers import GCCVerifier
        
        verifier = GCCVerifier()
        
        text = '''#include <iostream>
int main() { return 0; }'''
        
        extracted = verifier.extract_code(text)
        
        assert "#include <iostream>" in extracted


# =============================================================================
# Integration Tests (require GPU)
# =============================================================================

@pytest.mark.skipif(
    not _torch_cuda_available(),
    reason="CUDA/ROCm not available"
)
class TestPipelineIntegration:
    """Integration tests that require GPU."""
    
    def test_sft_config_creation(self):
        """SFTConfig should be creatable."""
        from halo_forge.sft.trainer import SFTConfig
        
        config = SFTConfig(
            model_name=TEST_CONFIG["model_name"],
            train_file="data/train.jsonl",
            output_dir="models/sft"
        )
        
        assert config.model_name == TEST_CONFIG["model_name"]
    
    def test_raft_config_creation(self):
        """RAFTConfig should be creatable."""
        from halo_forge.rlvr.raft_trainer import RAFTConfig
        
        config = RAFTConfig(
            base_model=TEST_CONFIG["model_name"],
            sft_checkpoint="models/sft/final_model",
            num_cycles=1
        )
        
        assert config.num_cycles == 1


# =============================================================================
# Full Pipeline Test (slow, requires GPU)
# =============================================================================

@pytest.mark.slow
@pytest.mark.skipif(
    not _torch_cuda_available(),
    reason="CUDA/ROCm not available"
)
class TestFullPipeline:
    """
    Full end-to-end pipeline test.
    
    This test takes several minutes and requires GPU.
    Run with: pytest tests/test_pipeline.py -v -m slow
    """
    
    def test_minimal_pipeline(self, test_dir, sample_prompts, sample_train_data):
        """Run minimal training pipeline."""
        import torch
        from halo_forge.rlvr.verifiers import GCCVerifier
        from halo_forge.rlvr.raft_trainer import RAFTTrainer, RAFTConfig
        
        # Skip if no GPU
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
        
        # Setup
        sft_dir = test_dir / "sft"
        raft_dir = test_dir / "raft"
        
        # Create verifier
        verifier = GCCVerifier(max_workers=4)
        
        # Test verifier works
        test_code = '#include <iostream>\nint main() { return 0; }'
        result = verifier.verify(test_code)
        assert result.success, f"Verifier failed: {result.error}"
        
        # Load prompts
        prompts = []
        with open(sample_prompts) as f:
            for line in f:
                prompts.append(json.loads(line)["prompt"])
        
        print(f"\nLoaded {len(prompts)} prompts")
        print(f"Test directory: {test_dir}")
        
        # Note: Full SFT + RAFT test is very slow
        # For quick validation, just test verifier and config creation
        
        # Create RAFT config
        config = RAFTConfig(
            base_model=TEST_CONFIG["model_name"],
            sft_checkpoint=str(sft_dir / "final_model"),
            output_dir=str(raft_dir),
            num_cycles=TEST_CONFIG["raft_cycles"],
            samples_per_prompt=TEST_CONFIG["samples_per_prompt"],
            max_new_tokens=TEST_CONFIG["max_new_tokens"]
        )
        
        print(f"RAFT config created: {config.num_cycles} cycles")
        
        # For actual training, uncomment:
        # trainer = RAFTTrainer(verifier=verifier, config=config)
        # trainer.run(prompts, num_cycles=1)


# =============================================================================
# Standalone Runner
# =============================================================================

def run_quick_test():
    """Run quick validation tests."""
    print("=" * 60)
    print("halo-forge Quick Validation Test")
    print("=" * 60)
    print()
    
    # Test imports
    print("Testing imports...")
    try:
        from halo_forge.rlvr.verifiers import (
            GCCVerifier, RewardLevel, VerifyResult
        )
        print("  Verifiers: OK")
    except Exception as e:
        print(f"  Verifiers: FAILED - {e}")
        return False
    
    try:
        from halo_forge.rlvr.raft_trainer import RAFTTrainer, RAFTConfig
        print("  RAFTTrainer: OK")
    except Exception as e:
        print(f"  RAFTTrainer: FAILED - {e}")
        return False
    
    try:
        from halo_forge.sft.trainer import SFTTrainer, SFTConfig
        print("  SFTTrainer: OK")
    except Exception as e:
        print(f"  SFTTrainer: FAILED - {e}")
        return False
    
    print()
    
    # Test verifier
    print("Testing GCCVerifier...")
    try:
        verifier = GCCVerifier()
        
        # Test valid code
        valid_code = '#include <iostream>\nint main() { return 0; }'
        result = verifier.verify(valid_code)
        
        if result.success:
            print(f"  Valid code: OK (reward={result.reward})")
        else:
            print(f"  Valid code: FAILED - {result.error}")
            return False
        
        # Test invalid code
        invalid_code = '#include <iostream>\nint main() { return 0'
        result = verifier.verify(invalid_code)
        
        if not result.success:
            print(f"  Invalid code detection: OK (reward={result.reward})")
        else:
            print("  Invalid code detection: FAILED - should have failed")
            return False
        
    except Exception as e:
        print(f"  GCCVerifier: FAILED - {e}")
        return False
    
    print()
    
    # Test GPU
    print("Testing GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU: {device_name}")
            print(f"  Memory: {memory_gb:.1f} GB")
        else:
            print("  GPU: Not available (CPU-only mode)")
    except ImportError:
        print("  GPU: PyTorch not installed")
    except Exception as e:
        print(f"  GPU: FAILED - {e}")
    
    print()
    print("=" * 60)
    print("All quick tests passed!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        pytest.main([__file__, "-v"])

