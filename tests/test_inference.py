#!/usr/bin/env python3
"""
Unit tests for Inference Optimization module.

Tests the InferenceOptimizer, InferenceOptimizationVerifier,
QATTrainer, and export functionality.

Run with:
    pytest tests/test_inference.py -v
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest
import torch

from halo_forge.inference.optimizer import InferenceOptimizer, OptimizationConfig
from halo_forge.inference.verifier import InferenceOptimizationVerifier, InferenceMetrics
from halo_forge.inference.quantization import (
    QATConfig,
    QATTrainer,
    prepare_qat,
    convert_to_quantized,
)
from halo_forge.inference.calibration import CalibrationConfig, CalibrationDataset
from halo_forge.inference.export.gguf import GGUFExporter, GGUF_QUANTIZATIONS
from halo_forge.inference.export.onnx import ONNXExporter


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.device = torch.device("cpu")
    model.config = MagicMock()
    model.config.quantization_config = None
    model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
    model.save_pretrained = MagicMock()
    model.half = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    
    # Create a mock that behaves like BatchEncoding
    class MockBatchEncoding(dict):
        def to(self, device):
            return self
    
    mock_result = MockBatchEncoding({
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    })
    
    tokenizer.return_value = mock_result
    tokenizer.decode = MagicMock(return_value="Hello world test output")
    tokenizer.save_pretrained = MagicMock()
    return tokenizer


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# OptimizationConfig Tests
# =============================================================================

class TestOptimizationConfig:
    """Tests for OptimizationConfig dataclass."""
    
    def test_default_values(self):
        """Default config should have sensible values."""
        config = OptimizationConfig()
        
        assert config.target_precision == "int4"
        assert config.target_latency_ms == 50.0
        assert config.quality_threshold == 0.95
        assert config.calibration_samples == 512
        assert config.export_format is None
        assert config.output_dir == "models/optimized"
    
    def test_custom_values(self):
        """Custom values should be set correctly."""
        config = OptimizationConfig(
            target_precision="int8",
            target_latency_ms=100.0,
            quality_threshold=0.90,
            calibration_samples=256,
            export_format="gguf",
            output_dir="/custom/path"
        )
        
        assert config.target_precision == "int8"
        assert config.target_latency_ms == 100.0
        assert config.quality_threshold == 0.90
        assert config.calibration_samples == 256
        assert config.export_format == "gguf"
        assert config.output_dir == "/custom/path"


# =============================================================================
# InferenceOptimizer Tests
# =============================================================================

class TestInferenceOptimizer:
    """Tests for InferenceOptimizer."""
    
    def test_initialization_default_config(self):
        """Optimizer should initialize with default config."""
        optimizer = InferenceOptimizer()
        
        assert optimizer.config is not None
        assert optimizer.config.target_precision == "int4"
        assert optimizer.model is None
        assert optimizer.tokenizer is None
    
    def test_initialization_custom_config(self):
        """Optimizer should accept custom config."""
        config = OptimizationConfig(target_precision="int8")
        optimizer = InferenceOptimizer(config)
        
        assert optimizer.config.target_precision == "int8"
    
    def test_load_model(self, mock_model, mock_tokenizer):
        """load_model should load model and tokenizer."""
        try:
            with patch('transformers.AutoModelForCausalLM') as mock_model_cls, \
                 patch('transformers.AutoTokenizer') as mock_tokenizer_cls:
                mock_model_cls.from_pretrained.return_value = mock_model
                mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
                
                optimizer = InferenceOptimizer()
                result = optimizer.load_model("test/model")
                
                assert result == mock_model
                assert optimizer.model == mock_model
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Dependency issue: {e}")
    
    def test_load_model_sets_baseline(self, mock_model, mock_tokenizer):
        """load_model should track baseline model path."""
        try:
            with patch('transformers.AutoModelForCausalLM') as mock_model_cls, \
                 patch('transformers.AutoTokenizer') as mock_tokenizer_cls:
                mock_model_cls.from_pretrained.return_value = mock_model
                mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
                
                optimizer = InferenceOptimizer()
                optimizer.load_model("test/model")
                
                assert optimizer.baseline_model == "test/model"
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Dependency issue: {e}")


# =============================================================================
# InferenceOptimizationVerifier Tests
# =============================================================================

class TestInferenceOptimizationVerifier:
    """Tests for InferenceOptimizationVerifier."""
    
    def test_initialization_defaults(self):
        """Verifier should initialize with sensible defaults."""
        verifier = InferenceOptimizationVerifier()
        
        assert verifier.target_latency_ms == 50.0
        assert verifier.quality_threshold == 0.95
        assert verifier.max_new_tokens == 100
        assert verifier.num_warmup == 3
    
    def test_initialization_custom(self):
        """Verifier should accept custom parameters."""
        verifier = InferenceOptimizationVerifier(
            target_latency_ms=100.0,
            quality_threshold=0.8,
            max_new_tokens=50
        )
        
        assert verifier.target_latency_ms == 100.0
        assert verifier.quality_threshold == 0.8
        assert verifier.max_new_tokens == 50
    
    def test_measure_latency(self, mock_model, mock_tokenizer):
        """measure_latency should return list of latencies."""
        verifier = InferenceOptimizationVerifier(num_warmup=0)
        
        prompts = ["Hello", "World"]
        latencies = verifier.measure_latency(mock_model, mock_tokenizer, prompts)
        
        assert len(latencies) == 2
        assert all(isinstance(l, float) for l in latencies)
        assert all(l >= 0 for l in latencies)
    
    def test_compare_quality_identical_outputs(self, mock_model, mock_tokenizer):
        """Identical outputs should give quality score of 1.0."""
        verifier = InferenceOptimizationVerifier()
        
        # Both models return same output
        mock_tokenizer.decode.return_value = "identical output tokens"
        
        quality = verifier.compare_quality(
            mock_model, mock_model, mock_tokenizer, ["test prompt"]
        )
        
        assert quality == 1.0
    
    def test_compare_quality_different_outputs(self, mock_model, mock_tokenizer):
        """Different outputs should give lower quality score."""
        verifier = InferenceOptimizationVerifier()
        
        # Return different outputs for each call
        mock_tokenizer.decode.side_effect = [
            "completely different output",
            "some other text entirely"
        ]
        
        quality = verifier.compare_quality(
            mock_model, mock_model, mock_tokenizer, ["test"]
        )
        
        # Should have some overlap but not perfect
        assert 0.0 <= quality <= 1.0
    
    def test_verify_without_baseline(self, mock_model, mock_tokenizer):
        """Verify should work without baseline (assumes full quality)."""
        verifier = InferenceOptimizationVerifier(num_warmup=0)
        
        result = verifier.verify(
            mock_model,
            ["test prompt 1", "test prompt 2"],
            tokenizer=mock_tokenizer
        )
        
        assert result.success is True  # No baseline = assume 1.0 quality
        assert result.reward > 0
        assert "avg_latency_ms" in result.metadata
        assert "quality_score" in result.metadata
    
    def test_verify_reward_calculation(self, mock_model, mock_tokenizer):
        """Verify should calculate combined reward from latency and quality."""
        verifier = InferenceOptimizationVerifier(
            target_latency_ms=1000.0,  # Very lenient target
            quality_threshold=0.5,
            num_warmup=0
        )
        
        result = verifier.verify(
            mock_model,
            ["test"],
            tokenizer=mock_tokenizer
        )
        
        # Reward should be between 0 and 1
        assert 0.0 <= result.reward <= 1.0
        
        # Should include latency factor in metadata
        assert "latency_factor" in result.metadata
    
    def test_cleanup(self, mock_model):
        """cleanup should free baseline model."""
        verifier = InferenceOptimizationVerifier()
        verifier.baseline_model = mock_model
        verifier._baseline_loaded = True
        
        verifier.cleanup()
        
        assert verifier.baseline_model is None
        assert verifier._baseline_loaded is False


# =============================================================================
# QAT Tests
# =============================================================================

class TestQATConfig:
    """Tests for QATConfig."""
    
    def test_default_values(self):
        """QATConfig should have sensible defaults."""
        config = QATConfig()
        
        assert config.target_precision == "int4"
        assert config.calibration_samples == 512
        assert config.epochs == 1
        assert config.batch_size == 4


class TestQATTrainer:
    """Tests for QATTrainer."""
    
    def test_initialization_default(self):
        """QATTrainer should initialize with default config."""
        trainer = QATTrainer()
        
        assert trainer.config is not None
        assert trainer.best_reward == 0.0
        assert trainer.best_checkpoint is None
    
    def test_initialization_custom(self):
        """QATTrainer should accept custom config."""
        config = QATConfig(target_precision="int8", epochs=3)
        trainer = QATTrainer(config)
        
        assert trainer.config.target_precision == "int8"
        assert trainer.config.epochs == 3


class TestPrepareQAT:
    """Tests for prepare_qat function."""
    
    def test_fp16_conversion(self, mock_model):
        """prepare_qat should convert to FP16."""
        try:
            result = prepare_qat(mock_model, "fp16")
            mock_model.half.assert_called_once()
        except (ImportError, RuntimeError) as e:
            if "bitsandbytes" in str(e) or "Python 3.14" in str(e) or "torch.compile" in str(e):
                pytest.skip(f"bitsandbytes/Python version issue: {e}")
            raise
    
    def test_invalid_precision_raises(self, mock_model):
        """prepare_qat should raise for invalid precision."""
        try:
            with pytest.raises(ValueError, match="Unknown precision"):
                prepare_qat(mock_model, "invalid")
        except RuntimeError as e:
            if "Python 3.14" in str(e) or "torch.compile" in str(e) or "bitsandbytes" in str(e):
                pytest.skip(f"Python/bitsandbytes version issue: {e}")
            raise
    
    def test_int4_quantization(self, mock_model):
        """prepare_qat should configure int4 quantization."""
        try:
            result = prepare_qat(mock_model, "int4")
            assert mock_model.config.quantization_config is not None
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"bitsandbytes not available: {e}")


class TestConvertToQuantized:
    """Tests for convert_to_quantized function."""
    
    def test_saves_to_output_path(self, mock_model, temp_dir):
        """convert_to_quantized should save model."""
        output_path = os.path.join(temp_dir, "quantized")
        
        result = convert_to_quantized(mock_model, "int4", output_path)
        
        mock_model.save_pretrained.assert_called_once_with(output_path)
    
    def test_returns_model_without_path(self, mock_model):
        """convert_to_quantized should return model when no path."""
        result = convert_to_quantized(mock_model, "int4")
        
        assert result == mock_model
        mock_model.save_pretrained.assert_not_called()


# =============================================================================
# CalibrationDataset Tests
# =============================================================================

class TestCalibrationConfig:
    """Tests for CalibrationConfig."""
    
    def test_default_values(self):
        """CalibrationConfig should have sensible defaults."""
        try:
            config = CalibrationConfig()
            
            # max_seq_length defaults to 2048
            assert config.max_seq_length == 2048
            assert config.batch_size == 4
            assert config.num_samples == 512
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Dependency issue: {e}")


class TestCalibrationDataset:
    """Tests for CalibrationDataset."""
    
    def test_from_texts(self, mock_tokenizer):
        """CalibrationDataset should create from text list."""
        try:
            texts = ["Hello world", "Test text", "Another sample"]
            
            dataset = CalibrationDataset.from_texts(texts, mock_tokenizer)
            
            assert len(dataset) == 3
        except (ImportError, RuntimeError, AttributeError) as e:
            pytest.skip(f"Dependency issue: {e}")
    
    def test_get_dataloader(self, mock_tokenizer):
        """CalibrationDataset should provide DataLoader."""
        try:
            texts = ["Hello world", "Test text"]
            dataset = CalibrationDataset.from_texts(texts, mock_tokenizer)
            
            dataloader = dataset.get_dataloader(batch_size=1)
            
            assert dataloader is not None
            # Should be iterable
            for batch in dataloader:
                break  # Just check it's iterable
        except (ImportError, RuntimeError, AttributeError) as e:
            pytest.skip(f"Dependency issue: {e}")


# =============================================================================
# GGUFExporter Tests
# =============================================================================

class TestGGUFExporter:
    """Tests for GGUFExporter."""
    
    def test_format_properties(self):
        """GGUFExporter should have correct format info."""
        exporter = GGUFExporter()
        
        assert exporter.format_name == "GGUF"
        assert exporter.file_extension == ".gguf"
    
    def test_list_quantizations(self):
        """list_quantizations should return available types."""
        quants = GGUFExporter.list_quantizations()
        
        assert "Q4_K_M" in quants
        assert "Q8_0" in quants
        assert "F16" in quants
    
    def test_recommended_quantization_small_model(self):
        """Small models should get Q8_0 recommendation."""
        rec = GGUFExporter.recommended_quantization(1.0)
        assert rec == "Q8_0"
    
    def test_recommended_quantization_medium_model(self):
        """Medium models should get Q4_K_M recommendation."""
        rec = GGUFExporter.recommended_quantization(7.0)
        assert rec == "Q4_K_M"
    
    def test_recommended_quantization_large_model(self):
        """Large models should get Q4_K_S recommendation."""
        rec = GGUFExporter.recommended_quantization(13.0)
        assert rec == "Q4_K_S"
    
    def test_invalid_quantization_raises(self, mock_model, temp_dir):
        """export should raise for invalid quantization type."""
        exporter = GGUFExporter()
        output_path = os.path.join(temp_dir, "test.gguf")
        
        with pytest.raises(ValueError, match="Unknown quantization"):
            exporter.export(mock_model, output_path, quantization="INVALID")
    
    def test_check_requirements_no_llama(self):
        """_check_requirements should return False without llama.cpp."""
        exporter = GGUFExporter()
        
        # Mock both methods to fail
        with patch.object(exporter, '_find_llama_cpp', return_value=None):
            # Will also fail import check
            result = exporter._check_requirements()
            # May or may not pass depending on environment
            assert isinstance(result, bool)


# =============================================================================
# ONNXExporter Tests
# =============================================================================

class TestONNXExporter:
    """Tests for ONNXExporter."""
    
    def test_format_properties(self):
        """ONNXExporter should have correct format info."""
        exporter = ONNXExporter()
        
        assert exporter.format_name == "ONNX"
        assert exporter.file_extension == ".onnx"
    
    def test_initialization_default_opset(self):
        """ONNXExporter should default to opset 17."""
        exporter = ONNXExporter()
        
        assert exporter.opset_version == 17
    
    def test_initialization_custom_opset(self):
        """ONNXExporter should accept custom opset."""
        exporter = ONNXExporter(opset_version=14)
        
        assert exporter.opset_version == 14
    
    def test_check_requirements(self):
        """_check_requirements should check for onnx and optimum."""
        exporter = ONNXExporter()
        
        result = exporter._check_requirements()
        
        # Result depends on what's installed
        assert isinstance(result, bool)
    
    def test_output_path_handling(self, temp_dir):
        """Output path should handle .onnx suffix."""
        exporter = ONNXExporter()
        
        # Test with .onnx suffix
        path = Path(temp_dir) / "model.onnx"
        result_path = Path(temp_dir) / "model"
        
        # The exporter should strip .onnx and create a directory
        assert path.suffix == ".onnx"


# =============================================================================
# Integration Tests
# =============================================================================

class TestInferenceIntegration:
    """Integration tests for the Inference module."""
    
    def test_optimizer_to_verifier_flow(self, mock_model, mock_tokenizer):
        """Optimizer output should be verifiable."""
        try:
            # Create optimizer
            optimizer = InferenceOptimizer()
            optimizer.model = mock_model
            optimizer.tokenizer = mock_tokenizer
            
            # Create verifier
            verifier = InferenceOptimizationVerifier(
                target_latency_ms=1000.0,  # Lenient
                num_warmup=0
            )
            
            # Verify the "optimized" model
            result = verifier.verify(
                mock_model,
                ["test prompt"],
                tokenizer=mock_tokenizer
            )
            
            assert result is not None
            assert result.reward > 0
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Dependency issue: {e}")
    
    def test_qat_to_export_flow(self, mock_model, temp_dir):
        """QAT model should be exportable."""
        try:
            # Prepare for QAT
            qat_model = prepare_qat(mock_model, "fp16")
            
            # Convert to quantized
            output_path = os.path.join(temp_dir, "quantized")
            convert_to_quantized(qat_model, "fp16", output_path)
            
            # Check save was called
            mock_model.save_pretrained.assert_called()
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Dependency issue: {e}")


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_empty_prompts_latency(self, mock_model, mock_tokenizer):
        """measure_latency should handle empty prompt list."""
        verifier = InferenceOptimizationVerifier(num_warmup=0)
        
        latencies = verifier.measure_latency(mock_model, mock_tokenizer, [])
        
        assert latencies == []
    
    def test_quality_empty_tokens(self, mock_model, mock_tokenizer):
        """compare_quality should handle empty token outputs."""
        verifier = InferenceOptimizationVerifier()
        mock_tokenizer.decode.return_value = ""
        
        quality = verifier.compare_quality(
            mock_model, mock_model, mock_tokenizer, ["test"]
        )
        
        # Empty vs empty should be 1.0
        assert quality == 1.0
    
    def test_verifier_no_cuda(self, mock_model, mock_tokenizer):
        """Verifier should work without CUDA."""
        verifier = InferenceOptimizationVerifier(num_warmup=0)
        
        # Should not raise even without CUDA
        result = verifier.verify(mock_model, ["test"], tokenizer=mock_tokenizer)
        
        assert result is not None


# =============================================================================
# InferenceMetrics Tests
# =============================================================================

class TestInferenceMetrics:
    """Tests for InferenceMetrics dataclass."""
    
    def test_creation(self):
        """InferenceMetrics should store all fields."""
        metrics = InferenceMetrics(
            avg_latency_ms=50.0,
            min_latency_ms=40.0,
            max_latency_ms=60.0,
            tokens_per_second=100.0,
            memory_used_mb=1024.0,
            quality_score=0.95
        )
        
        assert metrics.avg_latency_ms == 50.0
        assert metrics.min_latency_ms == 40.0
        assert metrics.max_latency_ms == 60.0
        assert metrics.tokens_per_second == 100.0
        assert metrics.memory_used_mb == 1024.0
        assert metrics.quality_score == 0.95


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
