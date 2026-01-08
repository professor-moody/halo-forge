---
title: "Changelog"
description: "All notable changes to halo forge"
---

## [1.0.0] - 2026-01-08

### Added

#### Audio Training (Phase 4)
- New `halo_forge/audio/` module for audio-language RLVR training
- `AudioRAFTTrainer` for RAFT training on audio models
- Multi-task verification: ASR, TTS, Audio Classification

#### Audio Verifiers
- `AudioVerifier` base class inheriting from core `Verifier`
- `ASRChecker` for speech-to-text with WER/CER metrics
- `TTSChecker` for text-to-speech quality (UTMOS-based)
- `AudioClassificationChecker` for sound event detection

#### Audio Model Adapters
- `WhisperAdapter` for OpenAI Whisper models
- `Wav2VecAdapter` for wav2vec2 models
- Automatic dtype handling and attention mask generation

#### Audio Dataset Loaders
- `LibriSpeechLoader` - 960h clean audiobook speech
- `CommonVoiceLoader` - Multilingual crowdsourced audio
- `AudioSetLoader` - 5M clips for classification
- `SpeechCommandsLoader` - Keyword spotting dataset

#### Math/Reasoning Training (Phase 5)
- New `halo_forge/reasoning/` module for mathematical reasoning
- `ReasoningRAFTTrainer` for reasoning task training
- SymPy-based answer verification

#### Reasoning Verifiers
- `ReasoningVerifier` base class inheriting from core `Verifier`
- `MathVerifier` with numeric and symbolic comparison
- `AnswerExtractor` for parsing answers from completions
- Support for `\boxed{}`, "The answer is", and numeric formats
- Partial credit for showing reasoning steps

#### Reasoning Dataset Loaders
- `GSM8KLoader` - 8.5K grade school math problems
- `MATHLoader` - 12.5K competition math problems
- Support for difficulty levels and subject filtering

#### CLI Commands
- `halo-forge audio train` - Train audio models with RAFT
- `halo-forge audio benchmark` - Benchmark on audio datasets
- `halo-forge audio datasets` - List audio datasets
- `halo-forge reasoning train` - Train on math datasets
- `halo-forge reasoning benchmark` - Math benchmarking
- `halo-forge reasoning datasets` - List reasoning datasets

#### Architecture Improvements
- All verifiers now inherit from base `Verifier` class
- Consistent `verify() -> VerifyResult` interface across domains
- Unified `VerifyResult` dataclass with `success`, `reward`, `error`

### Changed
- Updated all containers to v1.0.0
- Removed `torchcodec` dependency (using torchaudio/librosa directly)
- Improved audio loading with graceful fallback to librosa
- Consistent CLI banner and colors across all commands

### Fixed
- CLI subcommand dispatch issue causing empty output
- Build script argument parsing for `--tag` option
- Whisper dtype mismatch causing float/half errors
- VLM preprocessor returning 4D tensors instead of 3D

---

## [0.5.0] - 2026-01-07

### Added

#### Vision-Language Model Training (Phase 3)
- New `halo_forge/vlm/` module for VLM RLVR training
- `VLMRAFTTrainer` for RAFT training on VLMs
- Multi-stage verification pipeline for VLM outputs

#### VLM Verifiers
- `VisionVerifier` combining perception, reasoning, and output verification
- `PerceptionChecker` with YOLOv8 object detection and EasyOCR
- `ReasoningChecker` for chain-of-thought validation
- `OutputChecker` for answer matching (exact, fuzzy, semantic)
- Specialized verifiers: `VQAVerifier`, `DocVQAVerifier`, `ChartQAVerifier`

#### VLM Model Adapters
- `QwenVLAdapter` for Qwen-VL and Qwen2-VL models
- `LLaVAAdapter` for LLaVA model family
- `GenericVLMAdapter` for other HuggingFace VLMs
- Auto-detection of appropriate adapter from model name

#### VLM Dataset Loaders
- `TextVQALoader` - Text reading in natural images
- `DocVQALoader` - Document understanding
- `ChartQALoader` - Chart interpretation
- `RealWorldQALoader` - Real-world reasoning
- `MathVistaLoader` - Mathematical reasoning with visuals
- Export to RLVR and SFT formats

#### Image Processing
- `VLMPreprocessor` for generic image preprocessing
- `QwenVLProcessor` for Qwen-VL models
- `LLaVAProcessor` for LLaVA models

#### CLI Commands
- `halo-forge vlm train` - Train VLM with RAFT
- `halo-forge vlm benchmark` - Benchmark VLM on datasets
- `halo-forge vlm datasets` - List available VLM datasets

### Changed
- Updated changelog with Phase 3 features
- Added VLM documentation pages to website

---

## [0.4.0] - 2026-01-06

### Added

#### Inference Optimization Mode
- New `halo_forge/inference/` module for model optimization
- `InferenceOptimizationVerifier` for quality verification
- `InferenceOptimizer` for end-to-end optimization pipeline
- `QATTrainer` for quantization-aware training

#### Model Export
- `GGUFExporter` for llama.cpp/Ollama deployment
- `ONNXExporter` for cross-platform inference
- Support for Q4_K_M, Q8_0, F16 quantization types

#### CLI Commands
- `halo-forge inference optimize` - Optimize for deployment
- `halo-forge inference export` - Export to GGUF/ONNX
- `halo-forge inference benchmark` - Measure latency

#### Calibration
- `CalibrationDataset` for calibration data handling
- Support for synthetic calibration data generation

### Changed
- Updated CLI reference with inference commands
- Added inference section to website documentation

---

## [0.3.0] - 2026-01-06

### Added

#### Learning Rate Decay
- `--lr-decay` flag for exponential LR decay across RAFT cycles (default: 0.85)
- `--min-lr` flag to set learning rate floor (default: 1e-6)
- Prevents training degradation at cycles 7-8

#### Execution Verifier
- New `ExecutionVerifier` for test case-based verification
- Supports multiple test cases with input/output pairs
- Graduated rewards: 0.5 + 0.5 × pass_rate
- Match modes: exact, contains, regex, numeric
- Pre-configured variants: `GCCExecutionVerifier`, `ClangExecutionVerifier`, `MinGWExecutionVerifier`

#### Multi-Language Support
- New `MultiLanguageVerifier` with auto-detection
- Detects: C++, C, Python, Rust, Go, C#, PowerShell
- Use `--verifier auto` for automatic language detection
- `AutoVerifier` alias for CLI convenience

#### New Verifiers
- `RustVerifier` with Windows cross-compilation support
- `GoVerifier` with Windows cross-compilation support
- `DotNetVerifier` for C# compilation to Windows PE
- `PowerShellVerifier` for script syntax validation

#### Metrics Tracking
- `MetricsTracker` with TensorBoard integration
- JSON logging for all cycle metrics
- `TrainingMonitor` for early stopping detection
- Automatic `metrics.jsonl` generation

#### Dataset Loaders
- `HumanEvalPlusLoader` - 80x more test cases per problem
- `LiveCodeBenchLoader` - Contamination-free benchmark

#### CLI Enhancements
- `halo-forge config validate` command
- `--system-prompt` flag for custom prompts
- MSVC verifier validation with helpful error messages

### Changed
- Default system prompt updated to "You are an expert Windows systems programmer"
- Improved PEFT adapter handling to prevent stacking
- Category tracking now supports root-level fields in datasets

### Fixed
- PEFT adapter stacking bug in `_reload_model()`
- "Unknown" category issue in benchmark results
- MSVC verifier parameter validation

## [0.2.0] - 2025-01-01

### Added
- `halo-forge test` command for pipeline validation
  - `--level smoke`: Quick imports/compiler check (no GPU)
  - `--level standard`: Model loading, generation, verification
  - `--level full`: Complete mini-RAFT cycle with training
- `halo-forge benchmark full` command for comprehensive benchmarks
- Graduated rewards (`RewardLevel`) for partial credit
- Runtime verification (`run_after_compile`) for compile verifiers
- Comprehensive verifier unit tests
- Chunked verification in RAFT trainer to prevent OOM

### Changed
- Optimized for BF16 (4-bit quantization removed from defaults)
- Updated all docs to reflect 128GB unified memory
- Improved error messages in verifiers
- SFT trainer now uses `device_map="auto"`

### Fixed
- Memory leak during RAFT verification
- Gradient checkpointing warning during benchmark training

## [0.1.0] - 2024-12-28

### Added
- Initial release
- Custom toolbox with ROCm 7 nightly for gfx1151
- Data generation module (public datasets + LLM generation)
- SFT training with LoRA/BF16 support
- RAFT training with pluggable verifiers
- Benchmarking with pass@k metrics
- Built-in verifiers: GCC, Clang, MinGW, MSVC, pytest, unittest
- CLI with subcommands
- Documentation
