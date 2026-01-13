"""
Test Graduated Rewards

Verifies that the reward levels in our verifiers work correctly:
- 0.0: Doesn't compile (syntax error)
- 0.3: Compiles with warnings
- 0.5: Compiles clean (MVP mode stops here)
- 0.7: Runs without crash
- 1.0: Produces correct output

These graduated rewards are critical for RAFT training as they provide
learning signal even from imperfect samples.
"""

import pytest
from halo_forge.rlvr.verifiers import GCCVerifier, RustVerifier, GoVerifier
from halo_forge.rlvr.verifiers.base import RewardLevel


class TestRewardLevels:
    """Test the RewardLevel enum and helpers."""
    
    def test_reward_level_values(self):
        """Verify standard reward level values."""
        assert RewardLevel.FAILURE.value == 0.0
        assert RewardLevel.COMPILE_WARNINGS.value == 0.3
        assert RewardLevel.COMPILE_CLEAN.value == 0.5
        assert RewardLevel.RUNS_NO_CRASH.value == 0.7
        assert RewardLevel.CORRECT_OUTPUT.value == 1.0
    
    def test_from_compile_result(self):
        """Test RewardLevel.from_compile_result helper."""
        assert RewardLevel.from_compile_result(False) == 0.0
        assert RewardLevel.from_compile_result(True, has_warnings=True) == 0.3
        assert RewardLevel.from_compile_result(True, has_warnings=False) == 0.5
    
    def test_from_execution_result(self):
        """Test RewardLevel.from_execution_result helper."""
        assert RewardLevel.from_execution_result(False, False, False) == 0.0
        assert RewardLevel.from_execution_result(True, False, False) == 0.5
        assert RewardLevel.from_execution_result(True, True, False) == 0.7
        assert RewardLevel.from_execution_result(True, True, True) == 1.0


class TestGCCGraduatedRewards:
    """Test graduated rewards from GCC verifier."""
    
    @pytest.fixture
    def verifier_mvp(self):
        """MVP mode verifier (compile-only)."""
        return GCCVerifier(run_after_compile=False, timeout=10)
    
    @pytest.fixture
    def verifier_mvr(self):
        """MVR mode verifier (full verification)."""
        return GCCVerifier(run_after_compile=True, timeout=10, run_timeout=5)
    
    def test_mvp_syntax_error_returns_0(self, verifier_mvp):
        """Syntax error should return 0.0 reward."""
        code = "int main() { this is not valid C++ syntax }"
        result = verifier_mvp.verify(code)
        assert result.reward == 0.0
        assert not result.success
        assert "compile" in result.details.lower() or "failed" in result.details.lower()
    
    def test_mvp_compiles_clean_returns_05(self, verifier_mvp):
        """Clean compilation should return 0.5 reward in MVP mode."""
        code = """
        #include <stdio.h>
        int main() {
            return 0;
        }
        """
        result = verifier_mvp.verify(code)
        assert result.reward == 0.5
        assert result.success
    
    def test_mvr_syntax_error_returns_0(self, verifier_mvr):
        """Syntax error should return 0.0 reward in MVR mode."""
        code = "int main() { this is not valid C++ }"
        result = verifier_mvr.verify(code)
        assert result.reward == 0.0
        assert not result.success
    
    def test_mvr_compiles_and_runs_returns_07(self, verifier_mvr):
        """Code that compiles and runs without crash returns 0.7."""
        code = """
        #include <stdio.h>
        int main() {
            printf("Hello");
            return 0;
        }
        """
        result = verifier_mvr.verify(code)
        # Without expected_output, runs without crash = 0.7
        assert result.reward == 0.7
        assert result.success
    
    def test_mvr_correct_output_returns_10(self):
        """Correct output should return 1.0 reward."""
        verifier = GCCVerifier(
            run_after_compile=True,
            timeout=10,
            run_timeout=5,
            expected_output="42"
        )
        code = """
        #include <stdio.h>
        int main() {
            printf("42");
            return 0;
        }
        """
        result = verifier.verify(code)
        assert result.reward == 1.0
        assert result.success
    
    def test_mvr_wrong_output_returns_07(self):
        """Wrong output should return 0.7 (runs without crash)."""
        verifier = GCCVerifier(
            run_after_compile=True,
            timeout=10,
            run_timeout=5,
            expected_output="42"
        )
        code = """
        #include <stdio.h>
        int main() {
            printf("99");  // Wrong output
            return 0;
        }
        """
        result = verifier.verify(code)
        assert result.reward == 0.7
        # Not a success because output doesn't match
        assert not result.success
    
    def test_mvr_crash_returns_05(self, verifier_mvr):
        """Code that compiles but crashes returns 0.5."""
        code = """
        #include <stdio.h>
        int main() {
            int *p = 0;
            *p = 42;  // Null pointer dereference - should crash
            return 0;
        }
        """
        result = verifier_mvr.verify(code)
        # Crash should return compile reward (0.5)
        assert result.reward == 0.5


class TestRustGraduatedRewards:
    """Test graduated rewards from Rust verifier."""
    
    @pytest.fixture
    def verifier(self):
        """Rust verifier with run enabled."""
        return RustVerifier(run_after_compile=True, timeout=30, run_timeout=5)
    
    def test_syntax_error_returns_0(self, verifier):
        """Rust syntax error should return 0.0."""
        code = "fn main() { this is not valid rust }"
        result = verifier.verify(code)
        assert result.reward == 0.0
        assert not result.success
    
    def test_compiles_and_runs_returns_07(self, verifier):
        """Rust code that compiles and runs returns 0.7."""
        code = """
        fn main() {
            println!("Hello, World!");
        }
        """
        result = verifier.verify(code)
        assert result.reward >= 0.5  # At least compiles


class TestGoGraduatedRewards:
    """Test graduated rewards from Go verifier."""
    
    @pytest.fixture
    def verifier(self):
        """Go verifier with run enabled."""
        return GoVerifier(run_after_compile=True, timeout=30, run_timeout=5)
    
    def test_syntax_error_returns_0(self, verifier):
        """Go syntax error should return 0.0."""
        code = "func main() { this is not valid go }"
        result = verifier.verify(code)
        assert result.reward == 0.0
        assert not result.success
    
    def test_compiles_and_runs(self, verifier):
        """Go code that compiles and runs should get positive reward."""
        code = """
        package main
        
        import "fmt"
        
        func main() {
            fmt.Println("Hello, World!")
        }
        """
        result = verifier.verify(code)
        assert result.reward >= 0.5  # At least compiles


class TestMVPvsMVR:
    """Test the difference between MVP (compile-only) and MVR (full) modes."""
    
    def test_mvp_max_reward_is_05(self):
        """MVP mode should cap at 0.5 reward."""
        verifier = GCCVerifier(run_after_compile=False, timeout=10)
        
        code = """
        #include <stdio.h>
        int main() {
            printf("Hello");
            return 0;
        }
        """
        result = verifier.verify(code)
        
        # MVP mode: max is 0.5 (compile clean)
        assert result.reward == 0.5
        assert result.success
    
    def test_mvr_can_reach_10(self):
        """MVR mode can reach 1.0 reward with correct output."""
        verifier = GCCVerifier(
            run_after_compile=True,
            timeout=10,
            expected_output="Hello"
        )
        
        code = """
        #include <stdio.h>
        int main() {
            printf("Hello");
            return 0;
        }
        """
        result = verifier.verify(code)
        
        # MVR mode: can reach 1.0
        assert result.reward == 1.0
        assert result.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
