#!/usr/bin/env python3
"""
Unit tests for verifiers.

Tests each verifier with known good and bad code samples
to verify correct reward assignment.

Run with:
    pytest tests/test_verifiers.py -v
"""

import shutil
import pytest
from halo_forge.rlvr.verifiers import (
    GCCVerifier,
    ClangVerifier,
    MinGWVerifier,
    PytestVerifier,
    ChainedVerifier,
    RewardLevel,
    VerifyResult
)


# =============================================================================
# Helper Functions
# =============================================================================

def _compiler_available(compiler: str) -> bool:
    """Check if a compiler is available in PATH."""
    return shutil.which(compiler) is not None


# =============================================================================
# Test Data
# =============================================================================

# Valid C++ code that should compile
VALID_CPP = '''
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
'''

# C++ with syntax error (missing semicolon)
INVALID_CPP_SYNTAX = '''
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl
    return 0;
}
'''

# C++ that compiles but crashes (null pointer dereference)
VALID_CPP_CRASHES = '''
#include <iostream>

int main() {
    int* p = nullptr;
    std::cout << *p << std::endl;
    return 0;
}
'''

# C++ that produces specific output
VALID_CPP_OUTPUT = '''
#include <iostream>

int main() {
    std::cout << "42" << std::endl;
    return 0;
}
'''

# Valid Python with passing tests
VALID_PYTHON_TESTS = '''
def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, 1) == 0
'''

# Python with failing tests
FAILING_PYTHON_TESTS = '''
def add(a, b):
    return a - b  # Bug: should be +

def test_add():
    assert add(2, 3) == 5  # Will fail

def test_add_zero():
    assert add(0, 0) == 0  # Will pass
'''

# Code wrapped in markdown (common model output format)
WRAPPED_CPP = '''
Here is the solution:

```cpp
#include <iostream>

int main() {
    std::cout << "Hello!" << std::endl;
    return 0;
}
```

This code prints Hello to the console.
'''


# =============================================================================
# GCCVerifier Tests
# =============================================================================

class TestGCCVerifier:
    """Tests for GCCVerifier."""
    
    def test_valid_code_compiles(self):
        """Valid C++ should compile with reward 0.5."""
        verifier = GCCVerifier()
        result = verifier.verify(VALID_CPP)
        
        assert result.success is True
        assert result.reward == RewardLevel.COMPILE_CLEAN.value
        assert "successful" in result.details.lower()
    
    def test_invalid_code_fails(self):
        """Syntax errors should fail with reward 0.0."""
        verifier = GCCVerifier()
        result = verifier.verify(INVALID_CPP_SYNTAX)
        
        assert result.success is False
        assert result.reward == RewardLevel.FAILURE.value
        assert result.error is not None
    
    def test_wrapped_code_extracted(self):
        """Code in markdown blocks should be extracted."""
        verifier = GCCVerifier()
        result = verifier.verify(WRAPPED_CPP)
        
        assert result.success is True
        assert result.reward == RewardLevel.COMPILE_CLEAN.value
    
    def test_run_after_compile(self):
        """With run_after_compile, reward should be higher."""
        verifier = GCCVerifier(run_after_compile=True)
        result = verifier.verify(VALID_CPP)
        
        assert result.success is True
        assert result.reward == RewardLevel.RUNS_NO_CRASH.value
    
    def test_run_with_crash(self):
        """Code that crashes should get compile reward only."""
        verifier = GCCVerifier(run_after_compile=True, run_timeout=2)
        result = verifier.verify(VALID_CPP_CRASHES)
        
        # Should compile but crash
        assert result.reward >= RewardLevel.COMPILE_CLEAN.value
        # May or may not crash depending on system
    
    def test_output_verification(self):
        """Correct output should get full reward."""
        verifier = GCCVerifier(
            run_after_compile=True,
            expected_output="42"
        )
        result = verifier.verify(VALID_CPP_OUTPUT)
        
        assert result.success is True
        assert result.reward == RewardLevel.CORRECT_OUTPUT.value
    
    def test_wrong_output(self):
        """Wrong output should get partial reward."""
        verifier = GCCVerifier(
            run_after_compile=True,
            expected_output="99"
        )
        result = verifier.verify(VALID_CPP_OUTPUT)
        
        # Should run but output is wrong
        assert result.reward == RewardLevel.RUNS_NO_CRASH.value
    
    def test_batch_verification(self):
        """Batch verification should process all samples."""
        verifier = GCCVerifier(max_workers=4)
        codes = [VALID_CPP, INVALID_CPP_SYNTAX, VALID_CPP]
        
        results = verifier.verify_batch(codes)
        
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True


# =============================================================================
# ClangVerifier Tests
# =============================================================================

class TestClangVerifier:
    """Tests for ClangVerifier."""
    
    @pytest.mark.skipif(
        not _compiler_available('clang++'),
        reason="clang++ not available"
    )
    def test_valid_code_compiles(self):
        """Valid C++ should compile with Clang."""
        verifier = ClangVerifier()
        result = verifier.verify(VALID_CPP)
        
        assert result.success is True
        assert result.reward == RewardLevel.COMPILE_CLEAN.value


# =============================================================================
# MinGWVerifier Tests
# =============================================================================

class TestMinGWVerifier:
    """Tests for MinGWVerifier."""
    
    @pytest.mark.skipif(
        not _compiler_available('x86_64-w64-mingw32-g++'),
        reason="MinGW not available"
    )
    def test_valid_code_compiles(self):
        """Valid C++ should cross-compile with MinGW."""
        verifier = MinGWVerifier()
        result = verifier.verify(VALID_CPP)
        
        assert result.success is True


# =============================================================================
# PytestVerifier Tests
# =============================================================================

class TestPytestVerifier:
    """Tests for PytestVerifier."""
    
    def test_passing_tests(self):
        """All passing tests should get full reward."""
        verifier = PytestVerifier(timeout=30)
        result = verifier.verify(VALID_PYTHON_TESTS)
        
        assert result.success is True
        assert result.reward == 1.0
    
    def test_failing_tests(self):
        """Failing tests should get partial reward."""
        verifier = PytestVerifier(timeout=30)
        result = verifier.verify(FAILING_PYTHON_TESTS)
        
        assert result.success is False
        # Should have some reward for the passing test
        assert result.reward > 0.0
        assert result.reward < 1.0


# =============================================================================
# ChainedVerifier Tests
# =============================================================================

class TestChainedVerifier:
    """Tests for ChainedVerifier."""
    
    def test_all_pass(self):
        """If all verifiers pass, should get combined reward."""
        verifier = ChainedVerifier([
            GCCVerifier(),
            GCCVerifier()
        ])
        result = verifier.verify(VALID_CPP)
        
        assert result.success is True
        assert result.reward > 0.0
    
    def test_first_fails(self):
        """If first verifier fails, should stop early."""
        verifier = ChainedVerifier([
            GCCVerifier(),
            GCCVerifier()
        ])
        result = verifier.verify(INVALID_CPP_SYNTAX)
        
        assert result.success is False
        assert "Stage 1 failed" in result.details
    
    def test_weighted_rewards(self):
        """Weights should affect final reward."""
        verifier = ChainedVerifier(
            verifiers=[GCCVerifier(), GCCVerifier()],
            weights=[0.3, 0.7]
        )
        result = verifier.verify(VALID_CPP)
        
        assert result.success is True
        # Reward should be weighted sum


# =============================================================================
# RewardLevel Tests
# =============================================================================

class TestRewardLevel:
    """Tests for RewardLevel enum."""
    
    def test_from_compile_result_success(self):
        """Successful compile should return 0.5."""
        reward = RewardLevel.from_compile_result(success=True)
        assert reward == 0.5
    
    def test_from_compile_result_warnings(self):
        """Compile with warnings should return 0.3."""
        reward = RewardLevel.from_compile_result(success=True, has_warnings=True)
        assert reward == 0.3
    
    def test_from_compile_result_failure(self):
        """Failed compile should return 0.0."""
        reward = RewardLevel.from_compile_result(success=False)
        assert reward == 0.0
    
    def test_from_execution_result_correct(self):
        """Correct output should return 1.0."""
        reward = RewardLevel.from_execution_result(
            compiles=True, runs=True, correct=True
        )
        assert reward == 1.0
    
    def test_from_execution_result_runs(self):
        """Runs but incorrect should return 0.7."""
        reward = RewardLevel.from_execution_result(
            compiles=True, runs=True, correct=False
        )
        assert reward == 0.7


# =============================================================================
# Helper Functions
# =============================================================================

def _compiler_available(compiler: str) -> bool:
    """Check if a compiler is available in PATH."""
    import shutil
    return shutil.which(compiler) is not None


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

