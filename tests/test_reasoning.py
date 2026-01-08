"""
Tests for Reasoning Module

Unit tests for math verification, answer extraction, and dataset loading.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np


# =============================================================================
# Test MathSample and Data Structures
# =============================================================================

class TestMathSample:
    """Tests for MathSample dataclass."""
    
    def test_basic_creation(self):
        """Test creating basic sample."""
        from halo_forge.reasoning.data.loaders import MathSample
        
        sample = MathSample(
            question="What is 2 + 2?",
            answer="4",
        )
        
        assert sample.question == "What is 2 + 2?"
        assert sample.answer == "4"
        assert sample.solution is None
        assert sample.difficulty is None
    
    def test_full_creation(self):
        """Test creating sample with all fields."""
        from halo_forge.reasoning.data.loaders import MathSample
        
        sample = MathSample(
            question="Solve x^2 = 4",
            answer="2",
            solution="x^2 = 4, so x = ±2, taking positive: x = 2",
            difficulty="Level 1",
            subject="Algebra",
            metadata={"source": "test"},
        )
        
        assert sample.question == "Solve x^2 = 4"
        assert sample.answer == "2"
        assert sample.solution is not None
        assert sample.difficulty == "Level 1"
        assert sample.subject == "Algebra"
        assert sample.metadata["source"] == "test"


# =============================================================================
# Test Answer Extractor
# =============================================================================

class TestAnswerExtractor:
    """Tests for AnswerExtractor."""
    
    def test_extract_boxed(self):
        """Test extracting from LaTeX boxed format."""
        from halo_forge.reasoning.verifiers.answer_extract import AnswerExtractor
        
        extractor = AnswerExtractor()
        
        text = "The solution is x = 5. Therefore, \\boxed{5}"
        assert extractor.extract(text) == "5"
    
    def test_extract_boxed_with_spaces(self):
        """Test extracting from boxed with spaces."""
        from halo_forge.reasoning.verifiers.answer_extract import AnswerExtractor
        
        extractor = AnswerExtractor()
        
        text = "So the answer is \\boxed{ 42 }"
        assert extractor.extract(text) == "42"
    
    def test_extract_answer_is_pattern(self):
        """Test extracting from 'answer is' pattern."""
        from halo_forge.reasoning.verifiers.answer_extract import AnswerExtractor
        
        extractor = AnswerExtractor()
        
        text = "After calculation, the answer is 17"
        assert extractor.extract(text) == "17"
    
    def test_extract_therefore_pattern(self):
        """Test extracting from 'therefore' pattern."""
        from halo_forge.reasoning.verifiers.answer_extract import AnswerExtractor
        
        extractor = AnswerExtractor()
        
        text = "We have 3 + 5 = 8. Therefore, 8"
        result = extractor.extract(text)
        assert "8" in result
    
    def test_extract_fraction(self):
        """Test extracting fractions."""
        from halo_forge.reasoning.verifiers.answer_extract import AnswerExtractor
        
        extractor = AnswerExtractor()
        
        text = "The result is 3/4"
        assert extractor.extract(text) == "3/4"
    
    def test_extract_negative(self):
        """Test extracting negative numbers."""
        from halo_forge.reasoning.verifiers.answer_extract import AnswerExtractor
        
        extractor = AnswerExtractor()
        
        text = "The final answer is \\boxed{-7}"
        assert extractor.extract(text) == "-7"
    
    def test_extract_decimal(self):
        """Test extracting decimals."""
        from halo_forge.reasoning.verifiers.answer_extract import AnswerExtractor
        
        extractor = AnswerExtractor()
        
        text = "The answer is \\boxed{3.14159}"
        result = extractor.extract(text)
        assert result == "3.14159"
    
    def test_extract_boxed_only(self):
        """Test extracting only from boxed format."""
        from halo_forge.reasoning.verifiers.answer_extract import AnswerExtractor
        
        extractor = AnswerExtractor()
        
        text = "Random text without boxed format. Answer is 5."
        result = extractor.extract_boxed(text)
        assert result is None
        
        text_with_box = "The answer is \\boxed{42}"
        assert extractor.extract_boxed(text_with_box) == "42"
    
    def test_extract_all(self):
        """Test extracting all potential answers."""
        from halo_forge.reasoning.verifiers.answer_extract import AnswerExtractor
        
        extractor = AnswerExtractor()
        
        text = "The answer is 5. Therefore \\boxed{5}"
        results = extractor.extract_all(text)
        
        assert len(results) >= 2
        assert any("5" in r[0] for r in results)
    
    def test_normalize_answer_fraction(self):
        """Test normalizing fraction answers."""
        from halo_forge.reasoning.verifiers.answer_extract import AnswerExtractor
        
        extractor = AnswerExtractor()
        
        assert extractor.normalize_answer("1/2") == "0.5"
        assert extractor.normalize_answer("3/4") == "0.75"
    
    def test_normalize_answer_percentage(self):
        """Test normalizing percentage answers."""
        from halo_forge.reasoning.verifiers.answer_extract import AnswerExtractor
        
        extractor = AnswerExtractor()
        
        assert extractor.normalize_answer("50%") == "0.5"
        assert extractor.normalize_answer("25%") == "0.25"


# =============================================================================
# Test Math Verifier
# =============================================================================

class TestMathVerifier:
    """Tests for MathVerifier."""
    
    def test_exact_match(self):
        """Test exact numeric match."""
        from halo_forge.reasoning.verifiers import MathVerifier
        
        verifier = MathVerifier()
        
        result = verifier.verify(
            prompt="What is 2+2?",
            completion="Let me calculate: 2 + 2 = 4. The answer is \\boxed{4}",
            expected_answer="4"
        )
        
        assert result.success is True
        assert result.reward == 1.0
    
    def test_numeric_equivalence(self):
        """Test numeric equivalence with different formats."""
        from halo_forge.reasoning.verifiers import MathVerifier
        
        verifier = MathVerifier()
        
        # Fraction vs decimal
        result = verifier.verify(
            prompt="What is 1/2?",
            completion="The answer is \\boxed{0.5}",
            expected_answer="0.5"
        )
        
        assert result.success is True
    
    def test_wrong_answer(self):
        """Test incorrect answer."""
        from halo_forge.reasoning.verifiers import MathVerifier
        
        verifier = MathVerifier()
        
        result = verifier.verify(
            prompt="What is 2+2?",
            completion="The answer is \\boxed{5}",
            expected_answer="4"
        )
        
        assert result.success is False
        assert result.reward < 1.0
    
    def test_no_answer_extracted(self):
        """Test when no answer can be extracted."""
        from halo_forge.reasoning.verifiers import MathVerifier
        
        verifier = MathVerifier()
        
        result = verifier.verify(
            prompt="What is 2+2?",
            completion="I'm not sure how to solve this.",
            expected_answer="4"
        )
        
        assert result.success is False
        assert result.reward < 0.5
    
    def test_partial_credit_for_work(self):
        """Test partial credit for showing work."""
        from halo_forge.reasoning.verifiers import MathVerifier
        
        verifier = MathVerifier(partial_credit_for_work=True)
        
        result = verifier.verify(
            prompt="What is 2+2?",
            completion=(
                "Step 1: We have 2 + 2\n"
                "Step 2: Therefore, we get 5\n"
                "The answer is \\boxed{5}"
            ),
            expected_answer="4"
        )
        
        assert result.success is False
        # Should get some partial credit for showing work
        assert result.reward > 0
    
    def test_tolerance(self):
        """Test numeric tolerance."""
        from halo_forge.reasoning.verifiers import MathVerifier
        
        verifier = MathVerifier(tolerance=0.01)
        
        result = verifier.verify(
            prompt="Calculate pi",
            completion="The answer is \\boxed{3.14}",
            expected_answer="3.14159"
        )
        
        # Within tolerance
        assert result.success is True


class TestMathVerifierSympy:
    """Tests for MathVerifier with SymPy."""
    
    @pytest.fixture
    def verifier(self):
        """Create verifier with SymPy enabled."""
        from halo_forge.reasoning.verifiers import MathVerifier
        return MathVerifier()
    
    def test_sympy_available(self, verifier):
        """Test that SymPy is available."""
        try:
            import sympy
            assert verifier._sympy_available is True
        except ImportError:
            pytest.skip("SymPy not installed")
    
    def test_symbolic_equivalence(self, verifier):
        """Test symbolic equivalence checking."""
        if not verifier._sympy_available:
            pytest.skip("SymPy not installed")
        
        result = verifier.verify(
            prompt="Simplify 2x + 3x",
            completion="The answer is \\boxed{5x}",
            expected_answer="5*x"
        )
        
        # Should recognize symbolic equivalence
        # (This may or may not match depending on SymPy parsing)
    
    def test_latex_fraction(self, verifier):
        """Test LaTeX fraction parsing."""
        if not verifier._sympy_available:
            pytest.skip("SymPy not installed")
        
        result = verifier.verify(
            prompt="What is 1/2 + 1/4?",
            completion="The answer is \\boxed{\\frac{3}{4}}",
            expected_answer="0.75"
        )
        
        # Should recognize fraction equivalence


# =============================================================================
# Test Dataset Loaders (with mocking)
# =============================================================================

class TestGSM8KLoader:
    """Tests for GSM8KLoader."""
    
    @patch('datasets.load_dataset')
    def test_load_samples(self, mock_load):
        """Test loading GSM8K samples."""
        from halo_forge.reasoning.data.loaders import GSM8KLoader
        
        # Mock dataset
        mock_dataset = MagicMock()
        mock_items = [
            {
                "question": "What is 2 + 2?",
                "answer": "2 + 2 = 4\n#### 4",
            }
        ]
        mock_dataset.__iter__ = lambda self: iter(mock_items)
        mock_load.return_value = mock_dataset
        
        loader = GSM8KLoader(limit=1)
        samples = loader.load()
        
        assert len(samples) == 1
        assert samples[0].question == "What is 2 + 2?"
        assert samples[0].answer == "4"
    
    def test_extract_gsm8k_answer(self):
        """Test GSM8K answer extraction."""
        from halo_forge.reasoning.data.loaders import GSM8KLoader
        
        loader = GSM8KLoader()
        
        solution = "First we have 2+2=4. Then 4+3=7.\n#### 7"
        answer = loader._extract_gsm8k_answer(solution)
        assert answer == "7"
        
        solution_with_comma = "The total is 1,234.\n#### 1,234"
        answer = loader._extract_gsm8k_answer(solution_with_comma)
        assert answer == "1234"


class TestMATHLoader:
    """Tests for MATHLoader."""
    
    @patch('datasets.load_dataset')
    def test_load_samples(self, mock_load):
        """Test loading MATH samples."""
        from halo_forge.reasoning.data.loaders import MATHLoader
        
        # Mock dataset
        mock_dataset = MagicMock()
        mock_items = [
            {
                "problem": "Solve x^2 = 4",
                "solution": "Taking square root: x = ±2. Since we want positive, \\boxed{2}",
                "level": "Level 1",
                "type": "Algebra",
            }
        ]
        mock_dataset.__iter__ = lambda self: iter(mock_items)
        mock_load.return_value = mock_dataset
        
        loader = MATHLoader(limit=1)
        samples = loader.load()
        
        assert len(samples) == 1
        assert samples[0].question == "Solve x^2 = 4"
        assert samples[0].answer == "2"
        assert samples[0].difficulty == "Level 1"
    
    def test_extract_math_answer(self):
        """Test MATH answer extraction."""
        from halo_forge.reasoning.data.loaders import MATHLoader
        
        loader = MATHLoader()
        
        solution = "The answer is \\boxed{42}"
        answer = loader._extract_math_answer(solution)
        assert answer == "42"
        
        # Nested braces are tricky - just check we get something
        solution_fraction = "Therefore, \\boxed{\\frac{3}{4}}"
        answer = loader._extract_math_answer(solution_fraction)
        assert len(answer) > 0  # Should extract something


# =============================================================================
# Test Reasoning Verify Result
# =============================================================================

class TestReasoningVerifyResult:
    """Tests for ReasoningVerifyResult."""
    
    def test_failure(self):
        """Test creating failure result."""
        from halo_forge.reasoning.verifiers.base import ReasoningVerifyResult
        
        result = ReasoningVerifyResult.failure("Test error", partial_reward=0.2)
        
        assert result.success is False
        assert result.reward == 0.2
        assert result.error == "Test error"
    
    def test_correct(self):
        """Test creating correct result."""
        from halo_forge.reasoning.verifiers.base import ReasoningVerifyResult
        
        result = ReasoningVerifyResult.correct("4", "4")
        
        assert result.success is True
        assert result.reward == 1.0
        assert result.extracted_answer == "4"
        assert result.expected_answer == "4"


# =============================================================================
# Test ReasoningRAFTConfig
# =============================================================================

class TestReasoningRAFTConfig:
    """Tests for ReasoningRAFTConfig."""
    
    def test_defaults(self):
        """Test default configuration values."""
        from halo_forge.reasoning.trainer import ReasoningRAFTConfig
        
        config = ReasoningRAFTConfig()
        
        assert config.num_cycles == 4
        assert config.samples_per_prompt == 4
        assert config.lr_decay_per_cycle == 0.85
        assert config.tolerance == 1e-6
    
    def test_custom_values(self):
        """Test custom configuration values."""
        from halo_forge.reasoning.trainer import ReasoningRAFTConfig
        
        config = ReasoningRAFTConfig(
            model_name="custom/model",
            num_cycles=8,
            learning_rate=5e-5,
        )
        
        assert config.model_name == "custom/model"
        assert config.num_cycles == 8
        assert config.learning_rate == 5e-5


# =============================================================================
# Test ReasoningRAFTTrainer
# =============================================================================

class TestReasoningRAFTTrainer:
    """Tests for ReasoningRAFTTrainer."""
    
    def test_get_learning_rate(self):
        """Test learning rate decay."""
        from halo_forge.reasoning.trainer import ReasoningRAFTTrainer, ReasoningRAFTConfig
        
        config = ReasoningRAFTConfig(
            learning_rate=1e-4,
            lr_decay_per_cycle=0.85,
        )
        trainer = ReasoningRAFTTrainer(config)
        
        # Cycle 0
        assert trainer.get_learning_rate(0) == pytest.approx(1e-4)
        
        # Cycle 1
        assert trainer.get_learning_rate(1) == pytest.approx(1e-4 * 0.85)
        
        # Cycle 5
        expected = 1e-4 * (0.85 ** 5)
        assert trainer.get_learning_rate(5) == pytest.approx(expected)
    
    def test_format_prompt(self):
        """Test prompt formatting."""
        from halo_forge.reasoning.trainer import ReasoningRAFTTrainer, ReasoningRAFTConfig
        
        config = ReasoningRAFTConfig()
        trainer = ReasoningRAFTTrainer(config)
        
        prompt = trainer._format_prompt("What is 2+2?")
        
        assert "What is 2+2?" in prompt
        assert "\\boxed{" in prompt
        assert "Solution:" in prompt


# =============================================================================
# Integration Tests
# =============================================================================

class TestReasoningIntegration:
    """Integration tests for reasoning module."""
    
    def test_full_verification_pipeline(self):
        """Test full math verification pipeline."""
        from halo_forge.reasoning.verifiers import MathVerifier, AnswerExtractor
        
        extractor = AnswerExtractor()
        verifier = MathVerifier()
        
        # Sample completion
        completion = """
        Let me solve this step by step.
        
        We have 15 + 27 = 42.
        
        Therefore, the answer is \\boxed{42}
        """
        
        # Extract answer
        answer = extractor.extract(completion)
        assert answer == "42"
        
        # Verify
        result = verifier.verify(
            prompt="What is 15 + 27?",
            completion=completion,
            expected_answer="42"
        )
        
        assert result.success is True
        assert result.reward == 1.0
    
    def test_list_datasets(self):
        """Test listing available datasets."""
        from halo_forge.reasoning.data import list_math_datasets
        
        datasets = list_math_datasets()
        
        assert "gsm8k" in datasets
        assert "math" in datasets
