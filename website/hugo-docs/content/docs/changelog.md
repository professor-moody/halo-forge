---
title: "Changelog"
description: "All notable changes to halo-forge"
---

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
