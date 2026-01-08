# Changelog

All notable changes to halo forge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `halo-forge test` command for pipeline validation
  - `--level smoke`: Quick imports/compiler check (no GPU)
  - `--level standard`: Model loading, generation, verification
  - `--level full`: Complete mini-RAFT cycle with training
- `halo-forge benchmark full` command for comprehensive RAFT benchmarks
  - Model scaling tests (0.5B, 1.5B, 3B)
  - Hardware metrics collection (GPU utilization, memory, power)
  - JSON/CSV output artifacts
- Graduated rewards (`RewardLevel`) for partial credit
- Runtime verification (`run_after_compile`) for compile verifiers
- Comprehensive verifier unit tests
- Chunked verification in RAFT trainer to prevent OOM on large batches
- Reward distribution tracking in verification stats

### Changed
- Optimized for BF16 (4-bit quantization removed from defaults)
- Updated all docs to reflect 128GB unified memory
- Improved error messages in verifiers
- SFT trainer now uses `device_map="auto"` for unified memory optimization
- Replaced deprecated `tokenizer` param with `processing_class` in all trainers
- RAFT trainer now includes `dataloader_num_workers=0` and `dataloader_pin_memory=False`
- Added `gradient_checkpointing_kwargs={"use_reentrant": False}` for cleaner checkpointing

### Fixed
- Memory leak during RAFT verification with large sample batches
- Gradient checkpointing warning during benchmark training

## [0.2.0] - 2025-01-01

### Added
- Initial public release
- Custom toolbox with ROCm 7 nightly + PyTorch nightly for gfx1151
- Data generation module (public datasets + LLM generation)
- SFT training with LoRA/BF16 support
- RAFT training with pluggable verifiers
- Benchmarking with pass@k metrics
- Built-in verifiers: GCC, Clang, MinGW, MSVC (SSH), pytest, unittest
- Built-in dataset specs: CodeForces, MBPP, HumanEval
- CLI with subcommands for data/sft/raft/benchmark/info
- Documentation: README, QUICKSTART, FULL_PIPELINE, THEORY, VERIFIERS, HARDWARE_NOTES
- Example configurations and training scripts

## [0.1.0] - 2025-12-28

### Added
- Initial project structure
- Core framework components

