"""
Unit tests for the Agentic / Tool Calling module.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock

# Test imports
from halo_forge.agentic.data.loaders import (
    ToolCallSample,
    XLAMLoader,
    GlaiveLoader,
    list_agentic_datasets,
    AGENTIC_DATASETS,
)
from halo_forge.agentic.data.formatters import (
    HermesFormatter,
    format_to_hermes,
    create_training_sample,
)
from halo_forge.agentic.verifiers.base import (
    ToolCallingVerifier,
    ToolCallVerifyResult,
    ToolCallingVerifyConfig,
)
from halo_forge.agentic.trainer import (
    AgenticRAFTTrainer,
    AgenticRAFTConfig,
)


# =============================================================================
# Data Loaders Tests
# =============================================================================

class TestToolCallSample:
    """Test ToolCallSample dataclass."""
    
    def test_creation(self):
        """Test creating a ToolCallSample."""
        sample = ToolCallSample(
            messages=[{"role": "user", "content": "Get weather"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            expected_calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}],
        )
        
        assert len(sample.messages) == 1
        assert len(sample.tools) == 1
        assert len(sample.expected_calls) == 1
        assert sample.is_irrelevant is False
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        sample = ToolCallSample(
            messages=[{"role": "user", "content": "Test"}],
            tools=[],
            expected_calls=[],
            is_irrelevant=True,
            metadata={"source": "test"},
        )
        
        d = sample.to_dict()
        assert d["is_irrelevant"] is True
        assert d["metadata"]["source"] == "test"


class TestListAgenticDatasets:
    """Test list_agentic_datasets function."""
    
    def test_returns_datasets(self):
        """Test that datasets are returned."""
        datasets = list_agentic_datasets()
        
        assert "xlam" in datasets
        assert "glaive" in datasets
        assert "toolbench" in datasets
    
    def test_dataset_info(self):
        """Test dataset info structure."""
        datasets = list_agentic_datasets()
        
        xlam = datasets["xlam"]
        assert "name" in xlam
        assert "hf_path" in xlam
        assert "description" in xlam
        assert "size" in xlam


class TestXLAMLoader:
    """Test XLAMLoader class."""
    
    def test_initialization(self):
        """Test loader initialization."""
        loader = XLAMLoader(split="train")
        
        assert loader.split == "train"
        assert loader.dataset_info == AGENTIC_DATASETS["xlam"]
    
    @patch("datasets.load_dataset")
    def test_load_samples(self, mock_load_dataset):
        """Test loading samples."""
        # Mock dataset
        mock_dataset = [
            {
                "query": "What's the weather?",
                "tools": '[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]',
                "answers": '[{"name": "get_weather", "arguments": {"city": "Paris"}}]',
            },
        ]
        mock_load_dataset.return_value = mock_dataset
        
        loader = XLAMLoader()
        samples = loader.load(limit=1)
        
        assert len(samples) == 1
        assert samples[0].messages[0]["content"] == "What's the weather?"
        assert samples[0].expected_calls[0]["name"] == "get_weather"
    
    def test_normalize_tools_flat_format(self):
        """Test normalizing flat tool format."""
        loader = XLAMLoader()
        
        tools = [{"name": "test", "parameters": {}}]
        normalized = loader._normalize_tools(tools)
        
        assert normalized[0]["type"] == "function"
        assert normalized[0]["function"]["name"] == "test"
    
    def test_normalize_tools_already_normalized(self):
        """Test normalizing already normalized tools."""
        loader = XLAMLoader()
        
        tools = [{"type": "function", "function": {"name": "test"}}]
        normalized = loader._normalize_tools(tools)
        
        assert normalized == tools


class TestGlaiveLoader:
    """Test GlaiveLoader class."""
    
    def test_initialization(self):
        """Test loader initialization."""
        loader = GlaiveLoader(split="train")
        
        assert loader.split == "train"
        assert loader.dataset_info == AGENTIC_DATASETS["glaive"]


# =============================================================================
# Formatter Tests
# =============================================================================

class TestHermesFormatter:
    """Test HermesFormatter class."""
    
    def test_format_sample(self):
        """Test formatting a sample to Hermes format."""
        formatter = HermesFormatter()
        
        sample = ToolCallSample(
            messages=[{"role": "user", "content": "Get weather in Paris"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            expected_calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}],
        )
        
        formatted = formatter.format(sample)
        
        assert "<|im_start|>system" in formatted
        assert "<tools>" in formatted
        assert "</tools>" in formatted
        assert "<|im_start|>user" in formatted
        assert "Get weather in Paris" in formatted
        assert "<tool_call>" in formatted
    
    def test_format_prompt(self):
        """Test formatting a prompt only."""
        formatter = HermesFormatter()
        
        sample = ToolCallSample(
            messages=[{"role": "user", "content": "Test prompt"}],
            tools=[],
            expected_calls=[],
        )
        
        prompt = formatter.format_prompt(sample)
        
        assert "<|im_start|>system" in prompt
        assert "<|im_start|>user" in prompt
        assert "Test prompt" in prompt
        assert "<|im_start|>assistant" in prompt
    
    def test_compact_json(self):
        """Test compact JSON formatting."""
        formatter = HermesFormatter(compact_json=True)
        
        tools = [{"name": "test", "param": "value"}]
        result = formatter._format_tools(tools)
        
        # Compact format has no spaces after : or ,
        assert " " not in result or result.count(" ") == 0


class TestFormatToHermes:
    """Test format_to_hermes convenience function."""
    
    def test_formats_correctly(self):
        """Test convenience function."""
        sample = ToolCallSample(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[],
            expected_calls=[],
        )
        
        result = format_to_hermes(sample)
        
        assert "<|im_start|>" in result


class TestCreateTrainingSample:
    """Test create_training_sample function."""
    
    def test_creates_dict(self):
        """Test creating training sample dict."""
        sample = ToolCallSample(
            messages=[{"role": "user", "content": "Test"}],
            tools=[],
            expected_calls=[{"name": "test", "arguments": {}}],
        )
        
        result = create_training_sample(sample)
        
        assert "input" in result
        assert "target" in result
        assert "full" in result


# =============================================================================
# Verifier Tests
# =============================================================================

class TestToolCallingVerifyConfig:
    """Test ToolCallingVerifyConfig."""
    
    def test_default_values(self):
        """Test default config values."""
        config = ToolCallingVerifyConfig()
        
        assert config.reward_correct == 1.0
        assert config.reward_false_positive == -0.25
        assert config.max_workers == 8


class TestToolCallVerifyResult:
    """Test ToolCallVerifyResult dataclass."""
    
    def test_creation(self):
        """Test creating a result."""
        result = ToolCallVerifyResult(
            success=True,
            reward=1.0,
            details="Test",
            json_valid=True,
            called_correct_function=True,
        )
        
        assert result.success is True
        assert result.reward == 1.0
        assert result.json_valid is True


class TestToolCallingVerifier:
    """Test ToolCallingVerifier class."""
    
    def test_initialization(self):
        """Test verifier initialization."""
        verifier = ToolCallingVerifier()
        
        assert verifier.available_tools == {}
        assert verifier.executor is None
    
    def test_initialization_with_tools(self):
        """Test verifier with tools."""
        tools = [
            {"type": "function", "function": {"name": "get_weather"}},
        ]
        verifier = ToolCallingVerifier(available_tools=tools)
        
        assert "get_weather" in verifier.available_tools
    
    def test_verify_correct_call(self):
        """Test verifying a correct tool call."""
        verifier = ToolCallingVerifier()
        
        output = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
        expected = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
        
        result = verifier.verify(output, expected_calls=expected)
        
        assert result.success is True
        assert result.reward == 0.75  # No execution test
        assert result.json_valid is True
    
    def test_verify_wrong_function(self):
        """Test verifying with wrong function name."""
        verifier = ToolCallingVerifier()
        
        output = '<tool_call>{"name": "wrong_func", "arguments": {}}</tool_call>'
        expected = [{"name": "get_weather", "arguments": {}}]
        
        result = verifier.verify(output, expected_calls=expected)
        
        assert result.success is False
        assert result.reward == 0.25  # Valid JSON, wrong function
    
    def test_verify_no_tool_call(self):
        """Test verifying when no tool call is found."""
        verifier = ToolCallingVerifier()
        
        output = "I cannot help with that."
        expected = [{"name": "get_weather", "arguments": {}}]
        
        result = verifier.verify(output, expected_calls=expected)
        
        assert result.success is False
        assert result.reward == 0.0
        assert result.json_valid is False
    
    def test_verify_irrelevant_correct(self):
        """Test verifying correctly not calling a tool."""
        verifier = ToolCallingVerifier()
        
        output = "I cannot help with that using the available tools."
        
        result = verifier.verify(output, expected_calls=[], is_irrelevant=True)
        
        assert result.success is True
        assert result.reward == 1.0
    
    def test_verify_false_positive(self):
        """Test verifying false positive (called when shouldn't)."""
        verifier = ToolCallingVerifier()
        
        output = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
        
        result = verifier.verify(output, expected_calls=[], is_irrelevant=True)
        
        assert result.success is False
        assert result.reward == -0.25
    
    def test_extract_tool_calls(self):
        """Test extracting tool calls from output."""
        verifier = ToolCallingVerifier()
        
        output = '<tool_call>{"name": "test", "arguments": {"a": 1}}</tool_call>'
        calls = verifier._extract_tool_calls(output)
        
        assert len(calls) == 1
        assert calls[0]["name"] == "test"
    
    def test_extract_tool_calls_alternative_format(self):
        """Test extracting from alternative format."""
        verifier = ToolCallingVerifier()
        
        output = '```json\n{"name": "test", "arguments": {}}\n```'
        calls = verifier._extract_tool_calls(output)
        
        assert len(calls) == 1
        assert calls[0]["name"] == "test"
    
    def test_verify_single_call_args_mismatch(self):
        """Test verifying with argument mismatch."""
        verifier = ToolCallingVerifier()
        
        parsed = {"name": "get_weather", "arguments": {"city": "London"}}
        expected = {"name": "get_weather", "arguments": {"city": "Paris"}}
        
        result = verifier._verify_single_call(parsed, expected)
        
        assert result.success is False
        assert result.reward >= 0.5  # Correct function
        assert result.reward < 0.75  # Wrong args


# =============================================================================
# Trainer Tests
# =============================================================================

class TestAgenticRAFTConfig:
    """Test AgenticRAFTConfig."""
    
    def test_default_values(self):
        """Test default config values."""
        config = AgenticRAFTConfig()
        
        assert config.model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert config.num_cycles == 5
        assert config.keep_top_percent == 0.25
        assert config.bf16 is True


class TestAgenticRAFTTrainer:
    """Test AgenticRAFTTrainer class."""
    
    def test_initialization(self):
        """Test trainer initialization."""
        config = AgenticRAFTConfig(output_dir="/tmp/test_agentic")
        trainer = AgenticRAFTTrainer(config)
        
        assert trainer.config == config
        assert trainer.verifier is not None
        assert trainer.formatter is not None
        assert trainer.model is None  # Not loaded yet
    
    def test_get_learning_rate(self):
        """Test learning rate decay."""
        config = AgenticRAFTConfig(
            learning_rate=1e-4,
            lr_decay_per_cycle=0.5,
        )
        trainer = AgenticRAFTTrainer(config)
        
        # Cycle 0: 1e-4
        assert trainer.get_learning_rate(0) == 1e-4
        
        # Cycle 1: 1e-4 * 0.5 = 5e-5
        assert trainer.get_learning_rate(1) == 5e-5
        
        # Cycle 2: 1e-4 * 0.25 = 2.5e-5
        assert trainer.get_learning_rate(2) == 2.5e-5
    
    def test_filter_completions(self):
        """Test filtering completions."""
        config = AgenticRAFTConfig(
            reward_threshold=0.5,
            keep_top_percent=0.5,
        )
        trainer = AgenticRAFTTrainer(config)
        
        # Create mock completions
        from halo_forge.agentic.trainer import AgenticCompletion
        completions = [
            AgenticCompletion(prompt="", output="", reward=1.0),
            AgenticCompletion(prompt="", output="", reward=0.8),
            AgenticCompletion(prompt="", output="", reward=0.6),
            AgenticCompletion(prompt="", output="", reward=0.3),  # Below threshold
        ]
        
        filtered = trainer._filter_completions(completions)
        
        # Should keep top 50% of those above 0.5 threshold (3 above, keep 1-2)
        assert len(filtered) >= 1
        assert all(c.reward >= 0.5 for c in filtered)


# =============================================================================
# Integration Tests
# =============================================================================

class TestAgenticIntegration:
    """Integration tests for agentic module."""
    
    def test_verifier_to_trainer_flow(self):
        """Test verifier and trainer work together."""
        # Create sample
        sample = ToolCallSample(
            messages=[{"role": "user", "content": "Get weather"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            expected_calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}],
        )
        
        # Format
        formatter = HermesFormatter()
        prompt = formatter.format_prompt(sample)
        
        # Mock output
        mock_output = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
        
        # Verify
        verifier = ToolCallingVerifier()
        result = verifier.verify(
            output=mock_output,
            expected_calls=sample.expected_calls,
        )
        
        assert result.success is True
        assert result.reward > 0.5
    
    def test_full_sample_formatting(self):
        """Test complete sample formatting pipeline."""
        sample = ToolCallSample(
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                            "required": ["location"],
                        },
                    },
                },
            ],
            expected_calls=[
                {"name": "get_weather", "arguments": {"location": "Paris"}},
            ],
        )
        
        # Create training sample
        training = create_training_sample(sample)
        
        assert "get_weather" in training["full"]
        assert "Paris" in training["target"]
