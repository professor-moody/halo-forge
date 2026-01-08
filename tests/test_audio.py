"""
Tests for Audio Module

Unit tests for audio processing, verifiers, and training components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict
import numpy as np


# =============================================================================
# Test AudioSample and Data Structures
# =============================================================================

class TestAudioSample:
    """Tests for AudioSample dataclass."""
    
    def test_basic_creation(self):
        """Test creating basic sample."""
        from halo_forge.audio.data.loaders import AudioSample
        
        sample = AudioSample(
            audio_path="test.wav",
            text="hello world",
            duration=2.5,
        )
        
        assert sample.audio_path == "test.wav"
        assert sample.text == "hello world"
        assert sample.duration == 2.5
        assert sample.task == "asr"
    
    def test_with_audio_array(self):
        """Test sample with in-memory audio."""
        from halo_forge.audio.data.loaders import AudioSample
        
        audio = np.random.randn(16000)  # 1 second at 16kHz
        
        sample = AudioSample(
            audio_path="",
            text="test",
            duration=1.0,
            audio_array=audio,
            sample_rate=16000,
        )
        
        assert sample.audio_array is not None
        assert sample.sample_rate == 16000
    
    def test_classification_sample(self):
        """Test classification sample."""
        from halo_forge.audio.data.loaders import AudioSample
        
        sample = AudioSample(
            audio_path="dog_bark.wav",
            text="dog",
            duration=1.5,
            task="classification",
            metadata={"source": "audioset"},
        )
        
        assert sample.task == "classification"
        assert sample.metadata["source"] == "audioset"


# =============================================================================
# Test AudioProcessor
# =============================================================================

class TestAudioProcessor:
    """Tests for AudioProcessor."""
    
    def test_initialization(self):
        """Test processor initialization."""
        from halo_forge.audio.data.processors import AudioProcessor
        
        processor = AudioProcessor(
            sample_rate=16000,
            normalize=True,
            mono=True,
        )
        
        assert processor.sample_rate == 16000
        assert processor.normalize is True
        assert processor.mono is True
    
    def test_load_array(self):
        """Test loading from numpy array."""
        from halo_forge.audio.data.processors import AudioProcessor
        
        processor = AudioProcessor(sample_rate=16000)
        
        # Create test audio
        audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)  # 1s at 440Hz
        
        with patch.object(processor, '_torchaudio') as mock_ta:
            mock_ta.transforms.Resample = Mock(return_value=Mock(return_value=Mock()))
            
            result = processor.load_array(audio, original_sr=16000)
            
            assert result.sample_rate == 16000
            assert result.duration == pytest.approx(1.0, rel=0.01)
    
    def test_normalization(self):
        """Test audio normalization."""
        from halo_forge.audio.data.processors import AudioProcessor
        import torch
        
        processor = AudioProcessor(normalize=True)
        
        # Create unnormalized audio
        audio = np.array([0.0, 0.5, 1.0, 0.5, 0.0]) * 2.0  # Peaks at 2.0
        
        with patch.object(processor, '_torchaudio') as mock_ta:
            mock_ta.transforms.Resample = Mock(return_value=Mock())
            result = processor.load_array(audio, 16000)
            
            # After normalization, max should be 1.0
            assert result.waveform.abs().max() <= 1.0


# =============================================================================
# Test ASRChecker
# =============================================================================

class TestASRChecker:
    """Tests for ASRChecker."""
    
    def test_perfect_match(self):
        """Test perfect transcription."""
        from halo_forge.audio.verifiers.asr import ASRChecker
        
        checker = ASRChecker(wer_threshold=0.3)
        
        result = checker.verify(
            prediction="hello world",
            ground_truth="hello world",
        )
        
        assert result.success is True
        assert result.reward == pytest.approx(1.0)
        assert result.wer == pytest.approx(0.0)
    
    def test_partial_match(self):
        """Test partial match with some errors."""
        from halo_forge.audio.verifiers.asr import ASRChecker
        
        checker = ASRChecker(wer_threshold=0.3)
        
        result = checker.verify(
            prediction="hello word",  # 1 error
            ground_truth="hello world",
        )
        
        # WER = 1/2 = 0.5
        assert result.wer == pytest.approx(0.5, rel=0.1)
        assert result.reward == pytest.approx(0.5, rel=0.1)
        assert result.success is False  # Above threshold
    
    def test_complete_mismatch(self):
        """Test complete mismatch."""
        from halo_forge.audio.verifiers.asr import ASRChecker
        
        checker = ASRChecker()
        
        result = checker.verify(
            prediction="foo bar baz",
            ground_truth="hello world",
        )
        
        # All words different
        assert result.wer >= 1.0
        assert result.reward <= 0.0
    
    def test_normalization(self):
        """Test text normalization."""
        from halo_forge.audio.verifiers.asr import ASRChecker
        
        checker = ASRChecker(normalize_text=True)
        
        result = checker.verify(
            prediction="HELLO WORLD!",
            ground_truth="hello world",
        )
        
        # Should match after normalization
        assert result.success is True
        assert result.wer == pytest.approx(0.0)
    
    def test_cer_calculation(self):
        """Test character error rate."""
        from halo_forge.audio.verifiers.asr import ASRChecker
        
        checker = ASRChecker(use_cer=True)
        
        result = checker.verify(
            prediction="hello",
            ground_truth="hallo",  # 1 char different
        )
        
        assert result.cer is not None
        assert result.cer == pytest.approx(0.2, rel=0.1)  # 1/5


# =============================================================================
# Test AudioClassificationChecker
# =============================================================================

class TestAudioClassificationChecker:
    """Tests for AudioClassificationChecker."""
    
    def test_exact_match(self):
        """Test exact label match."""
        from halo_forge.audio.verifiers.classification import AudioClassificationChecker
        
        checker = AudioClassificationChecker()
        
        result = checker.verify(
            prediction="dog",
            ground_truth="dog",
        )
        
        assert result.success is True
        assert result.reward == 1.0
        assert result.is_exact_match is True
    
    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        from halo_forge.audio.verifiers.classification import AudioClassificationChecker
        
        checker = AudioClassificationChecker(case_sensitive=False)
        
        result = checker.verify(
            prediction="DOG",
            ground_truth="dog",
        )
        
        assert result.success is True
        assert result.reward == 1.0
    
    def test_mismatch(self):
        """Test label mismatch."""
        from halo_forge.audio.verifiers.classification import AudioClassificationChecker
        
        checker = AudioClassificationChecker()
        
        result = checker.verify(
            prediction="cat",
            ground_truth="dog",
        )
        
        assert result.success is False
        assert result.reward == 0.0
    
    def test_label_aliases(self):
        """Test label alias resolution."""
        from halo_forge.audio.verifiers.classification import AudioClassificationChecker
        
        checker = AudioClassificationChecker(
            label_aliases={"dog": ["canine", "puppy", "hound"]}
        )
        
        result = checker.verify(
            prediction="canine",
            ground_truth="dog",
        )
        
        assert result.success is True
        assert result.reward == 1.0


# =============================================================================
# Test TTSChecker
# =============================================================================

class TestTTSChecker:
    """Tests for TTSChecker."""
    
    def test_initialization(self):
        """Test TTS checker initialization."""
        from halo_forge.audio.verifiers.tts import TTSChecker
        
        checker = TTSChecker(
            intelligibility_weight=0.4,
            quality_weight=0.4,
            consistency_weight=0.2,
        )
        
        assert checker.weights["intelligibility"] == 0.4
        assert checker.weights["quality"] == 0.4
        assert checker.weights["consistency"] == 0.2
    
    def test_quality_check_silent(self):
        """Test quality check on silent audio."""
        from halo_forge.audio.verifiers.tts import TTSChecker
        
        checker = TTSChecker()
        
        # Silent audio
        audio = np.zeros(16000)
        
        score, details = checker._check_quality(audio)
        
        # Silent audio should have low quality score
        assert score < 0.5
    
    def test_quality_check_normal(self):
        """Test quality check on normal audio."""
        from halo_forge.audio.verifiers.tts import TTSChecker
        
        checker = TTSChecker()
        
        # Generate reasonable audio signal
        t = np.linspace(0, 1, 16000)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
        
        score, details = checker._check_quality(audio)
        
        # Normal audio should have decent quality
        assert score > 0.3


# =============================================================================
# Test AudioVerifier
# =============================================================================

class TestAudioVerifier:
    """Tests for AudioVerifier base class."""
    
    def test_asr_task(self):
        """Test verifier with ASR task."""
        from halo_forge.audio.verifiers.base import AudioVerifier, AudioVerifyConfig
        
        config = AudioVerifyConfig(task="asr", wer_threshold=0.3)
        verifier = AudioVerifier(config)
        
        result = verifier.verify(
            prediction="hello world",
            ground_truth="hello world",
        )
        
        assert result.task == "asr"
        assert result.success is True
    
    def test_classification_task(self):
        """Test verifier with classification task."""
        from halo_forge.audio.verifiers.base import AudioVerifier, AudioVerifyConfig
        
        config = AudioVerifyConfig(task="classification")
        verifier = AudioVerifier(config)
        
        result = verifier.verify(
            prediction="dog",
            ground_truth="dog",
        )
        
        assert result.task == "classification"
        assert result.success is True
    
    def test_batch_verification(self):
        """Test batch verification."""
        from halo_forge.audio.verifiers.base import AudioVerifier, AudioVerifyConfig
        
        config = AudioVerifyConfig(task="asr")
        verifier = AudioVerifier(config)
        
        predictions = ["hello", "world", "test"]
        ground_truths = ["hello", "word", "best"]
        
        results = verifier.verify_batch(predictions, ground_truths)
        
        assert len(results) == 3
        assert results[0].success is True  # Exact match
        assert results[1].success is False  # Different


# =============================================================================
# Test Model Adapters
# =============================================================================

class TestWhisperAdapter:
    """Tests for WhisperAdapter."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        from halo_forge.audio.models.adapters import WhisperAdapter
        
        adapter = WhisperAdapter(
            model_name="openai/whisper-tiny",
            device="cpu",
        )
        
        assert adapter.model_name == "openai/whisper-tiny"
        assert adapter.device == "cpu"
        assert adapter.model is None  # Not loaded yet
    
    def test_transcribe_calls_load(self):
        """Test that transcribe loads model if needed."""
        from halo_forge.audio.models.adapters import WhisperAdapter
        
        adapter = WhisperAdapter()
        
        # Patch the load method
        with patch.object(adapter, 'load') as mock_load:
            # Set up model after load would be called
            def set_model():
                adapter.model = Mock()
                adapter.processor = Mock()
                adapter.processor.return_value = Mock(input_features=Mock(to=Mock(return_value=Mock())))
                adapter.model.generate = Mock(return_value=[[1, 2, 3]])
                adapter.processor.batch_decode = Mock(return_value=["test"])
            
            mock_load.side_effect = set_model
            
            audio = np.zeros(16000)
            result = adapter.transcribe(audio)
            
            # Model should be loaded
            mock_load.assert_called_once()


class TestWav2VecAdapter:
    """Tests for Wav2VecAdapter."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        from halo_forge.audio.models.adapters import Wav2VecAdapter
        
        adapter = Wav2VecAdapter(
            model_name="facebook/wav2vec2-base-960h",
            device="cpu",
        )
        
        assert "wav2vec" in adapter.model_name.lower()
        assert adapter.sample_rate == 16000


class TestGetAudioAdapter:
    """Tests for get_audio_adapter factory."""
    
    def test_whisper_model(self):
        """Test Whisper model detection."""
        from halo_forge.audio.models.adapters import get_audio_adapter, WhisperAdapter
        
        adapter = get_audio_adapter("openai/whisper-small")
        assert isinstance(adapter, WhisperAdapter)
    
    def test_wav2vec_model(self):
        """Test Wav2Vec model detection."""
        from halo_forge.audio.models.adapters import get_audio_adapter, Wav2VecAdapter
        
        adapter = get_audio_adapter("facebook/wav2vec2-base-960h")
        assert isinstance(adapter, Wav2VecAdapter)


# =============================================================================
# Test AudioRAFTTrainer
# =============================================================================

class TestAudioRAFTConfig:
    """Tests for AudioRAFTConfig."""
    
    def test_defaults(self):
        """Test default configuration."""
        from halo_forge.audio.trainer import AudioRAFTConfig
        
        config = AudioRAFTConfig()
        
        assert config.model_name == "openai/whisper-small"
        assert config.task == "asr"
        assert config.num_cycles == 6
        assert config.learning_rate == 5e-5
    
    def test_custom_config(self):
        """Test custom configuration."""
        from halo_forge.audio.trainer import AudioRAFTConfig
        
        config = AudioRAFTConfig(
            model_name="openai/whisper-tiny",
            task="classification",
            num_cycles=3,
            learning_rate=1e-4,
        )
        
        assert config.model_name == "openai/whisper-tiny"
        assert config.task == "classification"
        assert config.num_cycles == 3


class TestAudioRAFTTrainer:
    """Tests for AudioRAFTTrainer."""
    
    def test_initialization(self):
        """Test trainer initialization."""
        from halo_forge.audio.trainer import AudioRAFTTrainer, AudioRAFTConfig
        
        config = AudioRAFTConfig(output_dir="/tmp/test_audio_raft")
        trainer = AudioRAFTTrainer(config)
        
        assert trainer.config == config
        assert trainer.adapter is None  # Not loaded yet
    
    def test_learning_rate_decay(self):
        """Test learning rate decay calculation."""
        from halo_forge.audio.trainer import AudioRAFTTrainer, AudioRAFTConfig
        
        config = AudioRAFTConfig(
            learning_rate=1e-4,
            lr_decay_per_cycle=0.85,
            min_lr=1e-6,
        )
        trainer = AudioRAFTTrainer(config)
        
        # Cycle 0
        assert trainer.get_learning_rate(0) == pytest.approx(1e-4)
        
        # Cycle 1
        assert trainer.get_learning_rate(1) == pytest.approx(1e-4 * 0.85)
        
        # Cycle 5
        expected = 1e-4 * (0.85 ** 5)
        assert trainer.get_learning_rate(5) == pytest.approx(expected)


# =============================================================================
# Test Dataset Loaders (with mocking)
# =============================================================================

class TestLibriSpeechLoader:
    """Tests for LibriSpeechLoader."""
    
    @patch('datasets.load_dataset')
    def test_load_samples(self, mock_load):
        """Test loading LibriSpeech samples."""
        from halo_forge.audio.data.loaders import LibriSpeechLoader
        
        # Mock dataset
        mock_dataset = [
            {
                "audio": {
                    "array": np.zeros(16000),
                    "sampling_rate": 16000,
                    "path": "test.wav",
                },
                "text": "hello world",
                "speaker_id": 1,
                "chapter_id": 1,
            }
        ]
        mock_load.return_value = mock_dataset
        
        loader = LibriSpeechLoader(limit=1)
        samples = loader.load()
        
        assert len(samples) == 1
        assert samples[0].text == "hello world"
        assert samples[0].task == "asr"


class TestSpeechCommandsLoader:
    """Tests for SpeechCommandsLoader."""
    
    @patch('datasets.load_dataset')
    def test_load_samples(self, mock_load):
        """Test loading Speech Commands samples."""
        from halo_forge.audio.data.loaders import SpeechCommandsLoader
        
        # Mock dataset
        mock_dataset = [
            {
                "audio": {
                    "array": np.zeros(16000),
                    "sampling_rate": 16000,
                },
                "label": "yes",
            }
        ]
        mock_load.return_value = mock_dataset
        
        loader = SpeechCommandsLoader(limit=1)
        samples = loader.load()
        
        assert len(samples) == 1
        assert samples[0].text == "yes"
        assert samples[0].task == "classification"


# =============================================================================
# Integration Tests
# =============================================================================

class TestAudioIntegration:
    """Integration tests for audio module."""
    
    def test_asr_pipeline(self):
        """Test full ASR verification pipeline."""
        from halo_forge.audio.verifiers import AudioVerifier, ASRChecker
        from halo_forge.audio.verifiers.base import AudioVerifyConfig
        
        # Create verifier
        config = AudioVerifyConfig(task="asr", wer_threshold=0.3)
        verifier = AudioVerifier(config)
        
        # Test cases
        test_cases = [
            ("hello world", "hello world", True),
            ("hello", "hello world", False),  # Missing word
            ("foo bar", "hello world", False),  # Wrong words
        ]
        
        for pred, gt, expected_success in test_cases:
            result = verifier.verify(pred, gt)
            assert result.success == expected_success, f"Failed for {pred} vs {gt}"
    
    def test_classification_pipeline(self):
        """Test full classification pipeline."""
        from halo_forge.audio.verifiers import AudioVerifier, AudioClassificationChecker
        from halo_forge.audio.verifiers.base import AudioVerifyConfig
        
        # Create verifier
        config = AudioVerifyConfig(task="classification")
        verifier = AudioVerifier(config)
        
        # Test correct classification
        result = verifier.verify("dog", "dog")
        assert result.success is True
        assert result.reward == 1.0
        
        # Test incorrect classification
        result = verifier.verify("cat", "dog")
        assert result.success is False
        assert result.reward == 0.0
    
    def test_list_datasets(self):
        """Test listing available datasets."""
        from halo_forge.audio.data import list_audio_datasets
        
        datasets = list_audio_datasets()
        
        assert "librispeech" in datasets
        assert "speech_commands" in datasets
        assert "audioset" in datasets
        assert "common_voice" in datasets
