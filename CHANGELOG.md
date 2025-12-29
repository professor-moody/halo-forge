# Changelog

All notable changes to halo-forge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release
- Custom toolbox with ROCm 7 + PyTorch 2.7 for gfx1151
- Data generation module (public datasets + LLM generation)
- SFT training with QLoRA support
- RAFT training with pluggable verifiers
- GRPO training (experimental)
- Benchmarking with pass@k metrics
- Built-in verifiers: GCC, MinGW, MSVC (SSH), pytest
- Built-in dataset specs: CodeForces, MBPP, HumanEval
- CLI with subcommands for data/sft/raft/benchmark
- Documentation and examples

## [0.1.0] - 2025-12-28

### Added
- Initial project structure
- Core framework components

