#!/usr/bin/env python3
"""
Unit tests for VLM data loaders and processors.

Tests dataset loaders (TextVQA, DocVQA, ChartQA) and image processors.

Run with:
    pytest tests/test_vlm_data.py -v
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from io import BytesIO

import pytest
import numpy as np
import torch
from PIL import Image

from halo_forge.vlm.data.loaders import (
    VLMSample,
    VLMDataset,
    TextVQALoader,
    DocVQALoader,
    ChartQALoader,
    RealWorldQALoader,
)
from halo_forge.vlm.data.processors import (
    ProcessedImage,
    ImageProcessor,
    VLMPreprocessor,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_image():
    """Create a simple test image."""
    return Image.new('RGB', (224, 224), color=(128, 128, 128))


@pytest.fixture
def sample_image_large():
    """Create a larger test image."""
    return Image.new('RGB', (640, 480), color=(255, 0, 0))


@pytest.fixture
def temp_image_file(sample_image):
    """Create a temporary image file."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        sample_image.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# VLMSample Tests
# =============================================================================

class TestVLMSample:
    """Tests for VLMSample dataclass."""
    
    def test_creation_minimal(self, sample_image):
        """VLMSample should create with minimal args."""
        sample = VLMSample(
            image=sample_image,
            prompt="What is in the image?"
        )
        
        assert sample.image == sample_image
        assert sample.prompt == "What is in the image?"
        assert sample.ground_truth is None
        assert sample.alternatives is None
        assert sample.metadata is None
    
    def test_creation_full(self, sample_image):
        """VLMSample should store all fields."""
        sample = VLMSample(
            image=sample_image,
            prompt="What color is the car?",
            ground_truth="red",
            alternatives=["crimson", "scarlet"],
            metadata={'source': 'test', 'category': 'color'}
        )
        
        assert sample.ground_truth == "red"
        assert sample.alternatives == ["crimson", "scarlet"]
        assert sample.metadata == {'source': 'test', 'category': 'color'}
    
    def test_load_image_pil(self, sample_image):
        """load_image should return PIL Image directly."""
        sample = VLMSample(image=sample_image, prompt="test")
        
        loaded = sample.load_image()
        
        assert loaded == sample_image
    
    def test_load_image_path(self, temp_image_file):
        """load_image should load from file path."""
        sample = VLMSample(image=temp_image_file, prompt="test")
        
        loaded = sample.load_image()
        
        assert isinstance(loaded, Image.Image)
    
    def test_load_image_invalid_type(self):
        """load_image should raise for invalid type."""
        sample = VLMSample(image=12345, prompt="test")
        
        with pytest.raises(ValueError, match="Unknown image type"):
            sample.load_image()


# =============================================================================
# VLMDataset Tests
# =============================================================================

class TestVLMDataset:
    """Tests for VLMDataset base class."""
    
    def test_abstract_methods(self):
        """VLMDataset should define abstract methods."""
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            VLMDataset()
    
    def test_cache_dir_creation(self, temp_dir):
        """Dataset should create cache directory."""
        # Create a concrete implementation for testing
        class TestDataset(VLMDataset):
            @property
            def name(self):
                return "test"
            
            def load(self):
                return []
        
        cache_path = os.path.join(temp_dir, "test_cache")
        dataset = TestDataset(cache_dir=cache_path)
        
        assert dataset.cache_dir == Path(cache_path)
    
    def test_len_empty(self, temp_dir):
        """Empty dataset should have length 0."""
        class TestDataset(VLMDataset):
            @property
            def name(self):
                return "test"
            
            def load(self):
                return []
        
        dataset = TestDataset(cache_dir=temp_dir)
        
        assert len(dataset) == 0
    
    def test_iteration(self, sample_image, temp_dir):
        """Dataset should be iterable."""
        class TestDataset(VLMDataset):
            @property
            def name(self):
                return "test"
            
            def load(self):
                self.samples = [
                    VLMSample(image=sample_image, prompt="test1"),
                    VLMSample(image=sample_image, prompt="test2"),
                ]
                return self.samples
        
        dataset = TestDataset(cache_dir=temp_dir)
        dataset.load()
        
        samples = list(dataset)
        assert len(samples) == 2
    
    def test_indexing(self, sample_image, temp_dir):
        """Dataset should support indexing."""
        class TestDataset(VLMDataset):
            @property
            def name(self):
                return "test"
            
            def load(self):
                self.samples = [
                    VLMSample(image=sample_image, prompt="first"),
                    VLMSample(image=sample_image, prompt="second"),
                ]
                return self.samples
        
        dataset = TestDataset(cache_dir=temp_dir)
        dataset.load()
        
        assert dataset[0].prompt == "first"
        assert dataset[1].prompt == "second"
    
    def test_to_rlvr_format(self, temp_image_file, temp_dir):
        """to_rlvr_format should export to JSONL."""
        class TestDataset(VLMDataset):
            @property
            def name(self):
                return "test"
            
            def load(self):
                self.samples = [
                    VLMSample(
                        image=temp_image_file,
                        prompt="What is this?",
                        ground_truth="a test",
                        metadata={'id': 1}
                    ),
                ]
                return self.samples
        
        dataset = TestDataset(cache_dir=temp_dir)
        dataset.load()
        
        output_path = os.path.join(temp_dir, "output.jsonl")
        dataset.to_rlvr_format(output_path)
        
        assert os.path.exists(output_path)
        
        with open(output_path) as f:
            line = f.readline()
            record = json.loads(line)
        
        assert record['prompt'] == "What is this?"
        assert record['ground_truth'] == "a test"
        assert 'metadata' in record
    
    def test_to_sft_format_qwen(self, temp_image_file, temp_dir):
        """to_sft_format should export with Qwen template."""
        class TestDataset(VLMDataset):
            @property
            def name(self):
                return "test"
            
            def load(self):
                self.samples = [
                    VLMSample(
                        image=temp_image_file,
                        prompt="What is this?",
                        ground_truth="a cat"
                    ),
                ]
                return self.samples
        
        dataset = TestDataset(cache_dir=temp_dir)
        dataset.load()
        
        output_path = os.path.join(temp_dir, "sft.jsonl")
        dataset.to_sft_format(output_path, template="qwen")
        
        with open(output_path) as f:
            record = json.loads(f.readline())
        
        assert '<|im_start|>user' in record['text']
        assert '<|im_start|>assistant' in record['text']
        assert 'What is this?' in record['text']
        assert 'a cat' in record['text']
    
    def test_to_sft_format_default(self, temp_image_file, temp_dir):
        """to_sft_format should handle default template."""
        class TestDataset(VLMDataset):
            @property
            def name(self):
                return "test"
            
            def load(self):
                self.samples = [
                    VLMSample(
                        image=temp_image_file,
                        prompt="What is this?",
                        ground_truth="a cat"
                    ),
                ]
                return self.samples
        
        dataset = TestDataset(cache_dir=temp_dir)
        dataset.load()
        
        output_path = os.path.join(temp_dir, "sft.jsonl")
        dataset.to_sft_format(output_path, template="default")
        
        with open(output_path) as f:
            record = json.loads(f.readline())
        
        assert '### Question' in record['text']
        assert '### Answer' in record['text']


# =============================================================================
# Dataset Loader Tests
# =============================================================================

class TestTextVQALoader:
    """Tests for TextVQA dataset loader."""
    
    def test_name(self, temp_dir):
        """TextVQALoader should have correct name."""
        loader = TextVQALoader(cache_dir=temp_dir)
        
        assert loader.name == "textvqa"
    
    def test_initialization(self, temp_dir):
        """TextVQALoader should initialize correctly."""
        loader = TextVQALoader(split="validation", cache_dir=temp_dir, limit=100)
        
        assert loader.split == "validation"
        assert loader.limit == 100
    
    def test_load_calls_huggingface(self, temp_dir):
        """load should attempt to call HuggingFace datasets."""
        # Just verify the loader can be created
        loader = TextVQALoader(cache_dir=temp_dir)
        
        # Don't actually call load() as it requires network access
        # Just verify the interface
        assert hasattr(loader, 'load')
        assert hasattr(loader, 'name')
        assert loader.name == 'textvqa'


class TestDocVQALoader:
    """Tests for DocVQA dataset loader."""
    
    def test_name(self, temp_dir):
        """DocVQALoader should have correct name."""
        loader = DocVQALoader(cache_dir=temp_dir)
        
        assert loader.name == "docvqa"


class TestChartQALoader:
    """Tests for ChartQA dataset loader."""
    
    def test_name(self, temp_dir):
        """ChartQALoader should have correct name."""
        loader = ChartQALoader(cache_dir=temp_dir)
        
        assert loader.name == "chartqa"


class TestRealWorldQALoader:
    """Tests for RealWorldQA dataset loader."""
    
    def test_name(self, temp_dir):
        """RealWorldQALoader should have correct name."""
        loader = RealWorldQALoader(cache_dir=temp_dir)
        
        assert loader.name == "realworldqa"


# =============================================================================
# ProcessedImage Tests
# =============================================================================

class TestProcessedImage:
    """Tests for ProcessedImage dataclass."""
    
    def test_creation(self):
        """ProcessedImage should store all fields."""
        pixel_values = torch.randn(3, 224, 224)
        
        result = ProcessedImage(
            pixel_values=pixel_values,
            image_size=(224, 224),
            original_size=(640, 480),
            metadata={'resized': True}
        )
        
        assert result.pixel_values.shape == (3, 224, 224)
        assert result.image_size == (224, 224)
        assert result.original_size == (640, 480)
        assert result.metadata == {'resized': True}


# =============================================================================
# VLMPreprocessor Tests
# =============================================================================

class TestVLMPreprocessor:
    """Tests for VLMPreprocessor."""
    
    def test_initialization_defaults(self):
        """VLMPreprocessor should have sensible defaults."""
        processor = VLMPreprocessor()
        
        assert processor.image_size == (336, 336)
        # Mean should be ImageNet standard
        assert len(processor.mean) == 3
        assert len(processor.std) == 3
    
    def test_initialization_custom(self):
        """VLMPreprocessor should accept custom parameters."""
        processor = VLMPreprocessor(
            image_size=(224, 224),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
        
        assert processor.image_size == (224, 224)
        assert processor.mean == (0.5, 0.5, 0.5)
        assert processor.std == (0.5, 0.5, 0.5)
    
    def test_load_image_pil(self, sample_image):
        """load_image should return PIL Image directly."""
        processor = VLMPreprocessor()
        
        loaded = processor.load_image(sample_image)
        
        assert isinstance(loaded, Image.Image)
    
    def test_load_image_path(self, temp_image_file):
        """load_image should load from file path."""
        processor = VLMPreprocessor()
        
        loaded = processor.load_image(temp_image_file)
        
        assert isinstance(loaded, Image.Image)
    
    def test_load_image_invalid_type(self):
        """load_image should raise for invalid type."""
        processor = VLMPreprocessor()
        
        with pytest.raises(ValueError, match="Unknown image type"):
            processor.load_image(12345)
    
    def test_resize_image_exact(self, sample_image):
        """resize_image should resize to exact size when not keeping aspect."""
        processor = VLMPreprocessor()
        
        resized = processor.resize_image(sample_image, (100, 100), keep_aspect=False)
        
        assert resized.size == (100, 100)
    
    def test_resize_image_keep_aspect(self, sample_image_large):
        """resize_image should maintain aspect ratio when requested."""
        processor = VLMPreprocessor()
        
        # 640x480 resized to 224x224 should maintain aspect
        resized = processor.resize_image(sample_image_large, (224, 224), keep_aspect=True)
        
        # Result should be padded to target size
        assert resized.size == (224, 224)
    
    def test_call_returns_processed_image(self, sample_image):
        """__call__ should return ProcessedImage."""
        processor = VLMPreprocessor()
        
        result = processor(sample_image)
        
        assert isinstance(result, ProcessedImage)
        assert isinstance(result.pixel_values, torch.Tensor)
        # Can be 3D (C, H, W) or 4D (B, C, H, W) depending on implementation
        assert result.pixel_values.ndim in (3, 4)
    
    def test_call_normalizes_values(self, sample_image):
        """__call__ should normalize pixel values."""
        processor = VLMPreprocessor()
        
        result = processor(sample_image)
        
        # Normalized values should be roughly in [-3, 3] range
        assert result.pixel_values.min() >= -5
        assert result.pixel_values.max() <= 5
    
    def test_call_preserves_original_size(self, sample_image_large):
        """__call__ should record original image size."""
        processor = VLMPreprocessor(image_size=(336, 336))
        
        result = processor(sample_image_large)
        
        assert result.original_size == (640, 480)
        assert result.image_size == (336, 336)


# =============================================================================
# Integration Tests
# =============================================================================

class TestDataIntegration:
    """Integration tests for VLM data pipeline."""
    
    def test_sample_to_processed(self, temp_image_file):
        """VLMSample should work with VLMPreprocessor."""
        sample = VLMSample(
            image=temp_image_file,
            prompt="What is in this image?",
            ground_truth="a test image"
        )
        
        # Load image from sample
        image = sample.load_image()
        
        # Process image
        processor = VLMPreprocessor()
        processed = processor(image)
        
        assert isinstance(processed, ProcessedImage)
        # Shape can be (C, H, W) or (B, C, H, W) depending on implementation
        # Check that channels dimension is 3 (RGB)
        if processed.pixel_values.ndim == 4:
            assert processed.pixel_values.shape[1] == 3  # B, C, H, W
        else:
            assert processed.pixel_values.shape[0] == 3  # C, H, W
    
    def test_dataset_export_roundtrip(self, temp_image_file, temp_dir):
        """Exported dataset should be loadable as JSON."""
        class TestDataset(VLMDataset):
            @property
            def name(self):
                return "test"
            
            def load(self):
                self.samples = [
                    VLMSample(
                        image=temp_image_file,
                        prompt="Q1",
                        ground_truth="A1",
                        metadata={'id': 1}
                    ),
                    VLMSample(
                        image=temp_image_file,
                        prompt="Q2",
                        ground_truth="A2",
                        metadata={'id': 2}
                    ),
                ]
                return self.samples
        
        dataset = TestDataset(cache_dir=temp_dir)
        dataset.load()
        
        output_path = os.path.join(temp_dir, "export.jsonl")
        dataset.to_rlvr_format(output_path)
        
        # Verify all records are valid JSON
        records = []
        with open(output_path) as f:
            for line in f:
                records.append(json.loads(line))
        
        assert len(records) == 2
        assert records[0]['prompt'] == "Q1"
        assert records[1]['prompt'] == "Q2"


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_load_nonexistent_image(self):
        """load_image should raise for nonexistent file."""
        sample = VLMSample(image="/nonexistent/path.jpg", prompt="test")
        
        with pytest.raises(FileNotFoundError):
            sample.load_image()
    
    def test_processor_empty_image(self):
        """Processor should handle 1x1 pixel images."""
        processor = VLMPreprocessor()
        tiny_image = Image.new('RGB', (1, 1), color=(128, 128, 128))
        
        result = processor(tiny_image)
        
        assert isinstance(result, ProcessedImage)
    
    def test_dataset_limit(self, sample_image, temp_dir):
        """Dataset should respect limit parameter."""
        class TestDataset(VLMDataset):
            @property
            def name(self):
                return "test"
            
            def load(self):
                all_samples = [
                    VLMSample(image=sample_image, prompt=f"test{i}")
                    for i in range(100)
                ]
                
                if self.limit:
                    self.samples = all_samples[:self.limit]
                else:
                    self.samples = all_samples
                    
                return self.samples
        
        dataset = TestDataset(cache_dir=temp_dir, limit=10)
        dataset.load()
        
        assert len(dataset) == 10


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
