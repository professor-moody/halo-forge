# Benchmarking in halo-forge

halo-forge uses a **dual evaluation architecture** that separates training verification from benchmark reporting. Understanding this distinction is key to using the platform effectively.

## Two Evaluation Modes

### 1. Training Verification (Native Verifiers)

Used **during RAFT training** to provide graduated reward signals. These are training infrastructure, not benchmarks.

| Verifier | Languages | Graduated Rewards |
|----------|-----------|-------------------|
| GCC/Clang | C/C++ | compile → syntax → partial → full |
| MinGW/MSVC | Windows C++ | cross-compile from Linux |
| RustVerifier | Rust | cargo check → cargo test |
| GoVerifier | Go | go build → go test |
| PytestVerifier | Python | syntax → unittest → pass |
| MathVerifier | Math reasoning | format → numeric → exact |
| HumanEvalVerifier | Python code | HumanEval test suite |
| MBPPVerifier | Python code | MBPP test suite |

**Key Characteristics:**
- Graduated rewards (0.0 → 0.3 → 0.5 → 0.7 → 1.0) enable learning
- Multi-step verification (compile → run → test) provides rich feedback
- Tightly integrated with RAFTTrainer
- Provides training signal, not final scores

### 2. Benchmark Reporting (Community Tools)

Used for **final model evaluation**, producing results comparable to published papers.

| Domain | Tool | Benchmarks |
|--------|------|------------|
| VLM | VLMEvalKit | MMStar, TextVQA, DocVQA, ChartQA, MMMU, etc. |
| Code | Native pass@k | HumanEval, MBPP, LiveCodeBench |
| Audio | Standard WER | LibriSpeech, CommonVoice |
| Agentic | Native / xLAM | Function calling, tool use |

**Key Characteristics:**
- Standard metrics comparable to published results
- Uses community tools where available (VLMEvalKit)
- Produces numbers for papers and leaderboards
- Run after training is complete

---

## When to Use Which

### Use Training Verifiers When:
- Running RAFT training loops
- Need graduated reward feedback
- Want compile/execution verification
- Training on Windows-specific code (MinGW/MSVC)

### Use Benchmark Reporting When:
- Evaluating a trained model
- Comparing to published baselines
- Preparing results for papers
- Need leaderboard-comparable metrics

---

## CLI Commands

### Training (uses native verifiers)
```bash
# RAFT training with verification feedback
halo-forge raft train --model Qwen/Qwen2.5-Coder-0.5B \
  --prompts data/prompts.jsonl \
  --verifier gcc

# Available verifiers: gcc, clang, mingw, msvc, rust, go, python, humaneval, mbpp
```

### Benchmark Reporting
```bash
# Code benchmarks (HumanEval, MBPP, LiveCodeBench)
halo-forge benchmark eval --model models/raft/final \
  --benchmark humaneval --limit 164

# VLM benchmarks (uses VLMEvalKit when available)
halo-forge vlm benchmark --model Qwen/Qwen2-VL-2B-Instruct \
  --dataset textvqa --limit 500

# Audio benchmarks
halo-forge audio benchmark --model openai/whisper-small \
  --dataset librispeech --limit 500

# Agentic benchmarks
halo-forge agentic benchmark --model Qwen/Qwen2.5-7B-Instruct \
  --dataset xlam --limit 500
```

---

## VLMEvalKit Integration

For VLM benchmarks, halo-forge integrates with [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), the community standard for vision-language model evaluation.

### Supported VLM Benchmarks
- MMStar, MMBench, MMMU
- TextVQA, DocVQA, ChartQA, InfoVQA
- RealWorldQA, OCRBench, MathVista
- ScienceQA, AI2D, GQA, VQA, OKVQA
- POPE, SeedBench, BLINK

### Installation
```bash
pip install halo-forge[vlm]  # Includes vlmeval dependency
```

### Usage
VLMEvalKit is automatically used when:
- Running VLM benchmarks via CLI
- The model is detected as a vision-language model
- The benchmark is in the VLM benchmark list

---

## Graduated Reward System

Native verifiers use a graduated reward ladder that provides learning signal beyond binary pass/fail:

```
Reward Level    Value    Meaning
────────────────────────────────────────
NONE            0.0      Failed to generate valid output
SYNTAX          0.3      Valid syntax, failed to compile
COMPILE         0.5      Compiled successfully, failed tests
PARTIAL         0.7      Passed some tests
FULL            1.0      Passed all tests
```

This gradient enables RAFT to learn from partial successes, not just perfect solutions.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────┐
│  Final Model Evaluation                         │
│  ─────────────────────                          │
│  VLMEvalKit, pass@k, WER                        │
│  → Papers, leaderboards, comparison             │
└─────────────────────────────────────────────────┘
                    ▲
                    │ trained model
                    │
┌─────────────────────────────────────────────────┐
│  RAFT Training Loop                             │
│  ─────────────────                              │
│  Native Verifiers with Graduated Rewards        │
│  → Training signal, compile verification        │
└─────────────────────────────────────────────────┘
                    ▲
                    │ base model
                    │
┌─────────────────────────────────────────────────┐
│  Pre-trained Model                              │
│  ─────────────────                              │
│  Qwen, DeepSeek, Whisper, etc.                  │
└─────────────────────────────────────────────────┘
```

---

## See Also

- [VERIFIERS.md](VERIFIERS.md) - Detailed verifier documentation
- [TRAINING.md](TRAINING.md) - RAFT training guide
- [WEB_UI.md](WEB_UI.md) - GUI benchmark and verifier pages
