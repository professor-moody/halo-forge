---
title: "Changelog"
description: "All notable changes to halo forge"
---

## [1.4.0] - 2026-05-14

Major release. Halo-forge becomes a cross-vendor local finetuning workstation
with an acceptance-backed Apple Silicon MLX path.

### Added

#### Trainers
- **DPO** (`halo-forge dpo train`) — preference optimization via TRL on PyTorch backends; MLX-native DPO on Apple Silicon supports sigmoid, IPO, hinge, and KTO-pair in reference-free and reference-model modes. RPO remains a PyTorch path.
- **GRPO** (`halo-forge grpo train`) — verifier-grounded policy gradient with group-relative advantages. PyTorch via TRL; MLX-native reference-free and reference-model paths. DeepSeek-R1 / Tülu 3 family.
- **Reward Model** (`halo-forge rm train`) — Bradley-Terry RM from preference pairs. Closes the RLHF loop for non-code modalities.
- **MLX-native DPO + GRPO** — DPO supports sigmoid, IPO, hinge, and KTO-pair in reference-free and reference-model modes; GRPO supports reference-free and reference-model eager updates with rollouts via `mlx_lm.generate`.

#### PEFT + optimizers
- **DoRA / rsLoRA / PiSSA / LoftQ / OLoRA** — `--use-dora`, `--use-rslora`, `--init-lora-weights pissa` etc. on every torch backend.
- **bitsandbytes optimizers** — `--optim adamw_bnb_8bit`, `lion_8bit`, `paged_adamw_8bit`. CUDA / ROCm only.

#### Verifiers
- **Plugin registry** (V1) — `@register_verifier` decorator + `~/.halo-forge/verifiers/*.py` discovery + entry-point packages.
- **LLM-as-judge** (V2) — rubric-graded with pluggable judge callable. Default targets a local `halo-forge serve` endpoint.
- **Schema verifiers** (V3) — `json_structure`, `json_schema`, `regex_format`.
- **Reference-metric verifiers** (V4) — `bleu`, `rouge`, `chrf`.

#### Data pipeline
- **Synthesize** (`halo-forge data synthesize`) — teacher → verifier → filter pipeline. SFT or preference data shape.
- **Dedup** (`halo-forge data dedup`) — exact (SHA-256) + fuzzy (MinHash + LSH).
- **Quality scoring** (`halo-forge data score`) — five-component heuristic + threshold / top-K filter.

#### Inference + serving
- **OpenAI-compatible serving** — `halo-forge serve --model X` exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and a Halo Forge `/health` status endpoint.
- **Streaming responses** — chat and text completions support OpenAI-shaped `text/event-stream` chunks ending in `data: [DONE]`.
- **Unified convert** — `halo-forge convert --format mlx|gguf|hf --quant q4|q8|fp16|bf16|fp32`.
- **Round-trip verify** — `halo-forge convert --verify` catches silently-broken exports.
- **vLLM rollout** — `--rollout-engine vllm` for CUDA / ROCm.
- **MLX rollout** — `--rollout-engine mlx` for Apple Silicon parity.
- **Adapter merge** — `halo-forge merge --mode bake|combine` (linear / ties / dare_linear / dare_ties / magnitude_prune).

#### Evaluation
- **lm-evaluation-harness** — `halo-forge eval --tasks core` with curated task groups (core / reasoning / code / instruction_following / knowledge).
- **Mid-training probe** — `halo-forge probe --baseline ./baseline.json` runs a small held-out benchmark + reports deltas; exits 2 on regression.

#### Reproducibility + management
- **Replay manifests** — `halo-forge replay <run_dir>` regenerates the launch command. Captures seed bundle, dataset hash, env fingerprint, full config.
- **Hyperparameter sweeps** — programmatic library with random / TPE / grid samplers, `Uniform` / `LogUniform` / `Choice` distributions, sweep-level early stopping.
- **SQLite run database** — `/runs/search` endpoint with filter / sort / paginate / facet support.
- **Cohort eval dashboard** — `/eval` route renders runs × tasks grid; best-per-task highlighted.
- **Cost rollup** — per-run kWh + $ from wall-clock × backend nominal power. UI panel + `HALOFORGE_COST_PER_KWH` env var.
- **API token auth** — bearer-token gate that turns on automatically when bound to non-loopback. `halo-forge token create / list / revoke`.

#### Frontend
- **Vite + React 19 + Tanstack Router** — replaces the retired NiceGUI surface.
- **First Run Experience v2** — guided `/start` flow with backend detection, MLX readiness, safe model recommendation, preflight, launch, and route-to-run behavior.
- **Model Catalog v2** — curated static catalog with first-run ranking, memory estimates, license/download notes, fit notes, and risk levels.
- **Live run view** — cycle-by-cycle loss + reward charts, scrubber, log tail, sample inspector, cancel button.
- **Multi-run comparison** — pin up to 6 runs, overlay loss / reward, side-by-side config diff.
- **Run search** — DB-backed filter chips for modality / status / model / has-eval.
- **Cohort eval** — runs × tasks grid over pinned runs.
- **Energy & spend card** — kWh + $ per run.

### Removed

- **NiceGUI web UI** — retired in favor of `public_app/`. The `ui/services/` and `ui/state.py` modules survive as the service layer the public_api consumes.
- `halo-forge ui` CLI subcommand.
- `nicegui` dependency.

### Changed

- **Documentation** — README rewritten; new feature-area docs (TRAINERS, DATA, EVAL, SERVING, REPLAY); per-backend feature × backend matrix in HARDWARE_NOTES.
- **Backend matrix** — every shipped feature is documented with its actual backend support status (✅ / ⚠️ / ❌). Silent-failure paths for MLX-with-PEFT-flags now warn loudly at trainer init.
- **MLX productization** — `halo-forge doctor mlx`, Terminal smoke validation, and acceptance evidence document the supported Apple Silicon path for SFT, RAFT, DPO, and GRPO.

---

## [1.3.0] - 2026-01-21

### Added
- **Web UI Verifier Integration** - Verifier test page now calls real backend verifiers instead of returning hardcoded results
- **Branding** - Halo-forge favicon and sidebar logo integrated into web UI
- **Static Asset Serving** - UI properly serves static files from `ui/static/`
- **SFT `--no-gradient-checkpointing` CLI flag** - Control gradient checkpointing from UI and CLI

### Fixed
- **SFT Dataset Routing** - Local `.jsonl` files now correctly use `--data` flag; HuggingFace IDs use `--dataset`
- **RAFT Verifier Alignment** - UI verifier choices now match CLI `--verifier` options exactly
- **MBPP Verifier** - Natural language prompts no longer cause syntax errors during execution

### Changed
- Removed unused RAFT learning rate UI field (CLI uses lr-decay schedule, not initial LR)

---

## [1.2.0] - 2026-01-10

### Added

#### Auto-Logging
- **Automatic log capture** - All training and benchmark commands now automatically log to `logs/` with timestamped filenames
- **`--quiet` flag** - Suppress terminal output while still writing to log file
- No more need for manual `tee` or `PYTHONUNBUFFERED=1`

#### New RAFT CLI Flags
- `--samples-per-prompt` - Control samples generated per prompt (default: 8)
- `--temperature` - Set generation temperature (default: 0.7)
- `--max-new-tokens` - Limit generation length (default: 1024)
- `--min-samples` - Auto-adjust threshold if too few samples pass filtering

#### Preset Config Files
- `configs/raft_conservative.yaml` - Safe training: 80% keep, slow LR decay, min 200 samples
- `configs/raft_aggressive.yaml` - Strict filtering: 30% keep, 16 samples/prompt, 0.8 temp
- `configs/vlm_example.yaml` - VLM RAFT with perception/reasoning/output weights
- `configs/audio_example.yaml` - Audio RAFT for ASR/TTS
- `configs/reasoning_example.yaml` - Math/reasoning RAFT

#### Module Consistency
- Added missing flags to all domain modules (VLM, Audio, Reasoning, Agentic)
- Consistent `--samples-per-prompt`, `--temperature`, `--keep-percent`, `--reward-threshold` across all `train` commands

### Changed
- Added `humaneval`, `mbpp`, `python` to verifier choices in CLI
- Improved base model loading for LoRA checkpoints (reads from `adapter_config.json`)
- Fixed code extraction to strip input tokens from generated completions

---

## [1.1.0] - 2026-01-08

### Added

#### Unified SFT Pipeline
- **`halo-forge sft train --dataset`** - Load HuggingFace datasets directly
- **`halo-forge sft datasets`** - List all available SFT datasets
- Domain-specific SFT commands for all modules:
  - `halo-forge vlm sft`
  - `halo-forge audio sft`
  - `halo-forge reasoning sft`
  - `halo-forge agentic sft`

#### SFT Datasets Module
- New `halo_forge/sft/datasets.py` with dataset registry
- Short name support (e.g., `codealpaca`, `metamath`, `llava`)
- Auto-formatting to ChatML for HuggingFace datasets
- `--max-samples` flag to limit dataset size
- `--dry-run` for validation

#### Supported SFT Datasets
| Domain | Dataset | HuggingFace ID | Size |
|--------|---------|----------------|------|
| Code | `codealpaca` | sahil2801/CodeAlpaca-20k | 20K |
| Code | `code_instructions_122k` | TokenBender/code_instructions_122k | 122K |
| Reasoning | `metamath` | meta-math/MetaMathQA | 395K |
| Reasoning | `gsm8k_sft` | gsm8k | 8.5K |
| VLM | `llava` | liuhaotian/LLaVA-Instruct-150K | 150K |
| Audio | `librispeech_sft` | librispeech_asr | 100h |
| Agentic | `xlam_sft` | Salesforce/xlam-function-calling-60k | 60K |
| Agentic | `glaive_sft` | glaiveai/glaive-function-calling-v2 | 113K |

#### Agentic / Tool Calling Training (Phase 6)
- New `halo_forge/agentic/` module for tool calling RLVR training
- `AgenticRAFTTrainer` for RAFT training on function calling
- Hermes format support (Qwen2.5, NousHermes compatible)
- TensorBoard integration via MetricsTracker

#### Tool Calling Verifier
- `ToolCallingVerifier` with graduated reward structure
- JSON validation and schema compliance
- Function name and argument matching
- Irrelevance detection (penalizes false positives)
- Support for parallel and multi-turn tool calls

#### Tool Calling Dataset Loaders
- `XLAMLoader` - 60k verified samples, 3,673 APIs
- `GlaiveLoader` - 113k samples with irrelevance detection
- `HermesFormatter` for converting to standard format

#### CLI Commands
- `halo-forge agentic train` - Train tool calling with RAFT
- `halo-forge agentic benchmark` - Benchmark on tool calling
- `halo-forge agentic datasets` - List available datasets

### Improved
- Consistent SFT → RAFT → Benchmark pipeline for ALL modules
- Consistent CLI banner and colors across all modules
- MetricsTracker integration for TensorBoard logging
- 32 new unit tests for agentic module

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
