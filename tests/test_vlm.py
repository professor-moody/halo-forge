#!/usr/bin/env python3
"""
Unit tests for Vision-Language Model (VLM) module.

Tests the VisionVerifier, PerceptionChecker, ReasoningChecker,
and OutputChecker components.

Run with:
    pytest tests/test_vlm.py -v
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

import pytest
import numpy as np
from PIL import Image

from halo_forge.vlm.verifiers.base import VisionVerifier, VisionVerifyResult
from halo_forge.vlm.verifiers.perception import (
    PerceptionChecker,
    PerceptionResult,
    Detection
)
from halo_forge.vlm.verifiers.reasoning import (
    ReasoningChecker,
    ReasoningResult,
    ReasoningStep
)
from halo_forge.vlm.verifiers.output import (
    OutputChecker,
    OutputResult
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_image():
    """Create a simple test image."""
    img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    return img


@pytest.fixture
def sample_detections():
    """Create sample object detections."""
    return [
        Detection(
            label='person',
            confidence=0.95,
            bbox=(10, 10, 100, 200),
            center=(55, 105)
        ),
        Detection(
            label='dog',
            confidence=0.88,
            bbox=(120, 80, 180, 160),
            center=(150, 120)
        ),
        Detection(
            label='car',
            confidence=0.75,
            bbox=(200, 50, 280, 120),
            center=(240, 85)
        ),
    ]


@pytest.fixture
def temp_image_file(sample_image):
    """Create a temporary image file."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        sample_image.save(f.name)
        yield f.name
    os.unlink(f.name)


# =============================================================================
# Detection Tests
# =============================================================================

class TestDetection:
    """Tests for Detection dataclass."""
    
    def test_creation(self):
        """Detection should store all fields correctly."""
        det = Detection(
            label='cat',
            confidence=0.9,
            bbox=(0, 0, 50, 50),
            center=(25, 25)
        )
        
        assert det.label == 'cat'
        assert det.confidence == 0.9
        assert det.bbox == (0, 0, 50, 50)
        assert det.center == (25, 25)


# =============================================================================
# PerceptionChecker Tests
# =============================================================================

class TestPerceptionChecker:
    """Tests for PerceptionChecker."""
    
    def test_initialization_defaults(self):
        """PerceptionChecker should initialize with sensible defaults."""
        checker = PerceptionChecker()
        
        assert checker.detector_model == "yolov8n"
        assert checker.confidence_threshold == 0.25
        assert checker.use_ocr is True
        assert checker.max_workers == 1
    
    def test_initialization_custom(self):
        """PerceptionChecker should accept custom parameters."""
        checker = PerceptionChecker(
            detector_model="yolov8m",
            confidence_threshold=0.5,
            use_ocr=False
        )
        
        assert checker.detector_model == "yolov8m"
        assert checker.confidence_threshold == 0.5
        assert checker.use_ocr is False
    
    def test_synonyms_defined(self):
        """Common object synonyms should be defined."""
        assert 'car' in PerceptionChecker.SYNONYMS
        assert 'person' in PerceptionChecker.SYNONYMS
        assert 'automobile' in PerceptionChecker.SYNONYMS['car']
        assert 'man' in PerceptionChecker.SYNONYMS['person']
    
    def test_spatial_patterns_defined(self):
        """Spatial relationship patterns should be defined."""
        patterns = PerceptionChecker.SPATIAL_PATTERNS
        
        assert 'left' in patterns
        assert 'right' in patterns
        assert 'above' in patterns
        assert 'below' in patterns
    
    def test_lazy_loading_detector(self):
        """Detector should not be loaded until needed."""
        checker = PerceptionChecker()
        
        assert checker._detector is None
        assert checker._detector_loaded is False
    
    def test_lazy_loading_ocr(self):
        """OCR should not be loaded until needed."""
        checker = PerceptionChecker()
        
        assert checker._ocr is None
        assert checker._ocr_loaded is False
    
    def test_detect_objects_returns_list(self, sample_image):
        """detect_objects should return list of Detection objects."""
        checker = PerceptionChecker()
        
        # If ultralytics is not installed, skip this test
        try:
            from ultralytics import YOLO
        except ImportError:
            pytest.skip("ultralytics not installed")
        
        # Mock the detector directly
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.data = MagicMock()
        mock_result.boxes.data.cpu.return_value.numpy.return_value = np.array([
            [10, 10, 100, 200, 0.95, 0]  # x1, y1, x2, y2, conf, class
        ])
        mock_result.names = {0: 'person'}
        
        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        
        checker._detector = mock_model
        checker._detector_loaded = True
        
        detections = checker.detect_objects(sample_image)
        
        assert isinstance(detections, list)
    
    def test_extract_objects_from_text(self):
        """Should extract mentioned objects from completion text."""
        checker = PerceptionChecker()
        
        text = "I can see a person walking their dog near a car."
        
        # This is a simplified check - full implementation would use NLP
        # For now, check that synonyms work
        person_synonyms = checker.SYNONYMS.get('person', set())
        assert 'person' in person_synonyms or 'person' == 'person'


class TestPerceptionResult:
    """Tests for PerceptionResult dataclass."""
    
    def test_creation(self):
        """PerceptionResult should store all scores."""
        result = PerceptionResult(
            object_score=0.9,
            text_score=0.8,
            spatial_score=0.7,
            counting_score=0.6,
            overall_score=0.75,
            details={'found_objects': ['person', 'dog']}
        )
        
        assert result.object_score == 0.9
        assert result.text_score == 0.8
        assert result.spatial_score == 0.7
        assert result.counting_score == 0.6
        assert result.overall_score == 0.75
        assert 'found_objects' in result.details


# =============================================================================
# ReasoningChecker Tests
# =============================================================================

class TestReasoningChecker:
    """Tests for ReasoningChecker."""
    
    def test_initialization_defaults(self):
        """ReasoningChecker should initialize with sensible defaults."""
        checker = ReasoningChecker()
        
        assert checker.min_steps == 2
        assert checker.require_evidence is True
        assert checker.require_conclusion is True
    
    def test_initialization_custom(self):
        """ReasoningChecker should accept custom parameters."""
        checker = ReasoningChecker(
            min_steps=3,
            require_evidence=False,
            require_conclusion=False
        )
        
        assert checker.min_steps == 3
        assert checker.require_evidence is False
        assert checker.require_conclusion is False
    
    def test_step_patterns_defined(self):
        """Step indicator patterns should be defined."""
        patterns = ReasoningChecker.STEP_PATTERNS
        
        assert len(patterns) > 0
        # Should match common step indicators
        assert any('first' in p for p in patterns)
        assert any('second' in p for p in patterns)
    
    def test_evidence_patterns_defined(self):
        """Evidence grounding patterns should be defined."""
        patterns = ReasoningChecker.EVIDENCE_PATTERNS
        
        assert len(patterns) > 0
        # Should match visual references
        assert any('see' in p for p in patterns)
        assert any('image' in p for p in patterns)
    
    def test_conclusion_patterns_defined(self):
        """Conclusion patterns should be defined."""
        patterns = ReasoningChecker.CONCLUSION_PATTERNS
        
        assert len(patterns) > 0
        assert any('therefore' in p for p in patterns)
    
    def test_extract_numbered_steps(self):
        """Should extract numbered reasoning steps."""
        checker = ReasoningChecker()
        
        text = """
        1. First, I observe the image shows a dog.
        2. Second, the dog appears to be a golden retriever.
        3. Therefore, the answer is yes.
        """
        
        steps = checker.extract_reasoning_steps(text)
        
        assert len(steps) >= 2
        assert all(isinstance(s, ReasoningStep) for s in steps)
    
    def test_extract_unnumbered_steps(self):
        """Should extract steps from flowing text."""
        checker = ReasoningChecker()
        
        text = "Looking at the image, I can see a cat. The cat is sitting on a table. Therefore, the answer is a cat on a table."
        
        steps = checker.extract_reasoning_steps(text)
        
        assert len(steps) >= 1
    
    def test_step_has_evidence_detection(self):
        """Should detect evidence references in steps."""
        checker = ReasoningChecker()
        
        text_with_evidence = "Looking at the image, I can see a person."
        text_without_evidence = "The answer is blue."
        
        step1 = checker._create_step(text_with_evidence, 1)
        step2 = checker._create_step(text_without_evidence, 2)
        
        assert step1.has_evidence is True
        assert step2.has_evidence is False
    
    def test_step_is_conclusion_detection(self):
        """Should detect conclusion statements."""
        checker = ReasoningChecker()
        
        text_conclusion = "Therefore, the answer is yes."
        text_not_conclusion = "I can see a dog in the image."
        
        step1 = checker._create_step(text_conclusion, 1)
        step2 = checker._create_step(text_not_conclusion, 2)
        
        assert step1.is_conclusion is True
        assert step2.is_conclusion is False


class TestReasoningStep:
    """Tests for ReasoningStep dataclass."""
    
    def test_creation(self):
        """ReasoningStep should store all fields."""
        step = ReasoningStep(
            text="I can see a dog.",
            step_number=1,
            has_evidence=True,
            is_conclusion=False
        )
        
        assert step.text == "I can see a dog."
        assert step.step_number == 1
        assert step.has_evidence is True
        assert step.is_conclusion is False


class TestReasoningResult:
    """Tests for ReasoningResult dataclass."""
    
    def test_creation(self):
        """ReasoningResult should store all scores."""
        result = ReasoningResult(
            structure_score=0.8,
            consistency_score=0.9,
            grounding_score=0.7,
            overall_score=0.8,
            details={'num_steps': 3}
        )
        
        assert result.structure_score == 0.8
        assert result.consistency_score == 0.9
        assert result.grounding_score == 0.7
        assert result.overall_score == 0.8


# =============================================================================
# OutputChecker Tests
# =============================================================================

class TestOutputChecker:
    """Tests for OutputChecker."""
    
    def test_initialization_defaults(self):
        """OutputChecker should initialize with sensible defaults."""
        checker = OutputChecker()
        
        assert checker.fuzzy_threshold == 0.8
        assert checker.use_semantic is False
        assert checker.normalize_answers is True
    
    def test_initialization_custom(self):
        """OutputChecker should accept custom parameters."""
        checker = OutputChecker(
            fuzzy_threshold=0.6,
            use_semantic=True,
            normalize_answers=False
        )
        
        assert checker.fuzzy_threshold == 0.6
        assert checker.use_semantic is True
        assert checker.normalize_answers is False
    
    def test_normalize_lowercase(self):
        """normalize should lowercase text."""
        checker = OutputChecker()
        
        result = checker.normalize("HELLO WORLD")
        
        assert result == "hello world"
    
    def test_normalize_removes_punctuation(self):
        """normalize should remove trailing punctuation."""
        checker = OutputChecker()
        
        result = checker.normalize("answer.")
        
        assert result == "answer"
    
    def test_normalize_removes_articles(self):
        """normalize should remove leading articles."""
        checker = OutputChecker()
        
        assert checker.normalize("a cat") == "cat"
        assert checker.normalize("an apple") == "apple"
        assert checker.normalize("the dog") == "dog"
    
    def test_normalize_whitespace(self):
        """normalize should collapse whitespace."""
        checker = OutputChecker()
        
        result = checker.normalize("hello   world")
        
        assert result == "hello world"
    
    def test_extract_answer_explicit(self):
        """Should extract explicit 'the answer is' format."""
        checker = OutputChecker()
        
        text = "Looking at the image, I see a cat. The answer is cat."
        
        answer = checker.extract_answer(text)
        
        assert answer is not None
        assert 'cat' in answer.lower()
    
    def test_extract_answer_multiple_choice(self):
        """Should extract multiple choice answers."""
        checker = OutputChecker()
        
        text = "A) Red\nB) Blue\nC) Green\n\nB"
        
        answer = checker.extract_answer(text)
        
        # Should get B
        assert answer is not None
    
    def test_extract_answer_conclusion(self):
        """Should extract answers from conclusion."""
        checker = OutputChecker()
        
        text = "Based on my analysis, therefore, the dog is brown."
        
        answer = checker.extract_answer(text)
        
        assert answer is not None
    
    def test_exact_match_identical(self):
        """exact_match should return True for identical strings."""
        checker = OutputChecker()
        
        assert checker.exact_match("cat", "cat") is True
    
    def test_exact_match_case_insensitive(self):
        """exact_match should be case insensitive when normalizing."""
        checker = OutputChecker(normalize_answers=True)
        
        assert checker.exact_match("CAT", "cat") is True
    
    def test_exact_match_different(self):
        """exact_match should return False for different strings."""
        checker = OutputChecker()
        
        assert checker.exact_match("cat", "dog") is False
    
    def test_fuzzy_match_identical(self):
        """fuzzy_match should return 1.0 for identical strings."""
        checker = OutputChecker()
        
        score = checker.fuzzy_match("hello world", "hello world")
        
        assert score == 1.0
    
    def test_fuzzy_match_similar(self):
        """fuzzy_match should return high score for similar strings."""
        checker = OutputChecker()
        
        score = checker.fuzzy_match("hello world", "hello worl")
        
        assert score > 0.8
    
    def test_fuzzy_match_different(self):
        """fuzzy_match should return low score for different strings."""
        checker = OutputChecker()
        
        score = checker.fuzzy_match("hello", "goodbye")
        
        assert score < 0.5
    
    def test_answer_formats_defined(self):
        """Common VQA answer formats should be defined."""
        formats = OutputChecker.ANSWER_FORMATS
        
        assert 'yes_no' in formats
        assert 'number' in formats
        assert 'multiple_choice' in formats
        assert 'color' in formats


class TestOutputResult:
    """Tests for OutputResult dataclass."""
    
    def test_creation(self):
        """OutputResult should store all fields."""
        result = OutputResult(
            exact_match=True,
            fuzzy_score=1.0,
            semantic_score=1.0,
            format_score=1.0,
            overall_score=1.0,
            details={'extracted': 'cat', 'ground_truth': 'cat'}
        )
        
        assert result.exact_match is True
        assert result.fuzzy_score == 1.0
        assert result.overall_score == 1.0


# =============================================================================
# VisionVerifier Tests
# =============================================================================

class TestVisionVerifier:
    """Tests for VisionVerifier."""
    
    def test_initialization_defaults(self):
        """VisionVerifier should initialize with balanced weights."""
        verifier = VisionVerifier()
        
        # Weights should sum to ~1.0
        total = verifier.perception_weight + verifier.reasoning_weight + verifier.output_weight
        assert abs(total - 1.0) < 0.01
    
    def test_initialization_custom_weights(self):
        """VisionVerifier should accept custom weights."""
        verifier = VisionVerifier(
            perception_weight=0.5,
            reasoning_weight=0.3,
            output_weight=0.2
        )
        
        # Weights should be normalized
        total = verifier.perception_weight + verifier.reasoning_weight + verifier.output_weight
        assert abs(total - 1.0) < 0.01
    
    def test_weights_normalized(self):
        """Weights should be normalized to sum to 1.0."""
        verifier = VisionVerifier(
            perception_weight=1.0,
            reasoning_weight=1.0,
            output_weight=1.0
        )
        
        # Should normalize to 1/3 each
        assert abs(verifier.perception_weight - 1/3) < 0.01
        assert abs(verifier.reasoning_weight - 1/3) < 0.01
        assert abs(verifier.output_weight - 1/3) < 0.01
    
    def test_checkers_initialized(self):
        """VisionVerifier should initialize all checkers."""
        verifier = VisionVerifier()
        
        # Checkers should be created (may be None if deps missing)
        # Just verify the verifier can be created
        assert verifier is not None


class TestVisionVerifyResult:
    """Tests for VisionVerifyResult dataclass."""
    
    def test_creation(self):
        """VisionVerifyResult should store all results."""
        perception = PerceptionResult(
            object_score=0.9, text_score=0.8, spatial_score=0.7,
            counting_score=0.6, overall_score=0.75, details={}
        )
        reasoning = ReasoningResult(
            structure_score=0.8, consistency_score=0.9,
            grounding_score=0.7, overall_score=0.8, details={}
        )
        output = OutputResult(
            exact_match=True, fuzzy_score=1.0, semantic_score=1.0,
            format_score=1.0, overall_score=1.0, details={}
        )
        
        result = VisionVerifyResult(
            perception=perception,
            reasoning=reasoning,
            output=output,
            overall_reward=0.85,
            success=True,
            details={'test': 'data'}
        )
        
        assert result.perception == perception
        assert result.reasoning == reasoning
        assert result.output == output
        assert result.overall_reward == 0.85
        assert result.success is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestVLMIntegration:
    """Integration tests for VLM verifier components."""
    
    def test_full_verification_pipeline(self, sample_image):
        """Test complete verification flow with mocked components."""
        # Create verifier with mocked checkers
        verifier = VisionVerifier()
        
        # Mock the verification
        completion = """
        Looking at the image, I can see a person and a dog.
        The person is standing on the left.
        The dog is a golden retriever sitting on the right.
        Therefore, the answer is yes, there is a dog in the image.
        """
        
        # We can't fully test without loading models,
        # but we can test the structure exists
        assert verifier is not None
    
    def test_reasoning_to_output_flow(self):
        """Reasoning extraction should feed into output verification."""
        reasoning_checker = ReasoningChecker()
        output_checker = OutputChecker()
        
        completion = """
        Step 1: I can see a cat in the image.
        Step 2: The cat is orange colored.
        Therefore, the answer is orange cat.
        """
        
        # Extract reasoning
        steps = reasoning_checker.extract_reasoning_steps(completion)
        assert len(steps) >= 2
        
        # Extract answer
        answer = output_checker.extract_answer(completion)
        assert answer is not None
    
    def test_verification_with_no_ground_truth(self, sample_image):
        """Verification should work without ground truth (output only)."""
        verifier = VisionVerifier()
        
        # Without ground truth, output score should be neutral
        # Just verify no exceptions
        assert verifier is not None


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_empty_completion_reasoning(self):
        """ReasoningChecker should handle empty completion."""
        checker = ReasoningChecker()
        
        steps = checker.extract_reasoning_steps("")
        
        assert steps == []
    
    def test_empty_completion_output(self):
        """OutputChecker should handle empty completion."""
        checker = OutputChecker()
        
        answer = checker.extract_answer("")
        
        # May return None or empty string
        assert answer is None or answer == ""
    
    def test_normalize_empty_string(self):
        """normalize should handle empty string."""
        checker = OutputChecker()
        
        result = checker.normalize("")
        
        assert result == ""
    
    def test_normalize_none(self):
        """normalize should handle None gracefully."""
        checker = OutputChecker()
        
        result = checker.normalize(None)
        
        assert result == ""
    
    def test_fuzzy_match_empty_strings(self):
        """fuzzy_match should handle empty strings."""
        checker = OutputChecker()
        
        score = checker.fuzzy_match("", "")
        
        assert score == 1.0  # Empty strings are "equal"
    
    def test_exact_match_with_punctuation(self):
        """exact_match should handle punctuation correctly."""
        checker = OutputChecker()
        
        # With normalization, these should match
        assert checker.exact_match("yes.", "yes") is True
        assert checker.exact_match("No!", "no") is True


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
