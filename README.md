<p align="center">
  <img src="halo-forge.png" alt="halo-forge logo" width="350">
</p>

<h1 align="center">halo-forge</h1>

<p align="center">
  A complete RLVR (Reinforcement Learning from Verifier Rewards) training framework for AMD Strix Halo.<br>
  Train language models to generate verified code through iterative refinement using automated verification.
</p>

---

## Table of Contents

1. [What is halo-forge](#what-is-halo-forge)
2. [How It Works](#how-it-works)
3. [Key Concepts](#key-concepts)
4. [Hardware Requirements](#hardware-requirements)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Testing Your Setup](#testing-your-setup)
8. [Verifiers](#verifiers)
9. [Configuration Reference](#configuration-reference)
10. [Performance Expectations](#performance-expectations)
11. [Benchmark Results](#benchmark-results)
12. [Troubleshooting](#troubleshooting)
13. [Research Background](#research-background)
14. [Extensibility](#extensibility)
15. [Project Structure](#project-structure)
16. [License](#license)

---

## What's New in v0.5.0

### Vision-Language Model Training (Phase 3)

| Feature | Description | CLI Command |
|---------|-------------|-------------|
| **VLM RAFT Training** | Train VLMs with perception-aware verification | `halo-forge vlm train` |
| **VisionVerifier** | Multi-stage verification (perception + reasoning + output) | Built-in |
| **Perception Checker** | YOLOv8 object detection + EasyOCR text extraction | Built-in |
| **VLM Datasets** | TextVQA, DocVQA, ChartQA, RealWorldQA, MathVista | `halo-forge vlm datasets` |
| **Model Adapters** | Qwen-VL, LLaVA, and generic VLM support | Auto-detected |

```bash
# Train VLM on TextVQA
halo-forge vlm train \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --dataset textvqa \
  --cycles 6 \
  --output models/vlm_raft

# Benchmark VLM
halo-forge vlm benchmark \
  --model models/vlm_raft/cycle_6 \
  --dataset docvqa
```

### Previous Release (v0.3.0)

| Feature | Description | CLI Flag |
|---------|-------------|----------|
| **LR Decay** | Prevents training degradation at cycles 7-8 | `--lr-decay 0.85` |
| **Execution Verifier** | Run code with test cases for graduated rewards | `--verifier execution` |
| **Multi-Language** | Auto-detect language from code (C++, Python, Rust, Go, C#, PowerShell) | `--verifier auto` |
| **Metrics Tracking** | TensorBoard integration + JSON logs | Automatic |
| **HumanEval+** | Extended test cases for better evaluation | Dataset loader |
| **LiveCodeBench** | Contamination-free benchmark | Dataset loader |

---

## What is halo-forge

halo-forge implements **RAFT (Reward-Ranked Fine-Tuning)**, a technique for improving code generation models by training on verified outputs rather than relying solely on human preferences or model self-evaluation.

The core insight: **A compiler is a perfect reward signal.** It provides unambiguous, deterministic feedback about code correctness that cannot be gamed.

### The Problem with Traditional Approaches

| Approach | Problem |
|----------|---------|
| **SFT only** | Distribution mismatch - model outputs differ from training data |
| **RLHF** | Expensive human labeling, inconsistent judgments, doesn't scale |
| **Self-evaluation** | Models hallucinate correctness, can be gamed |

### The RLVR Solution

Replace human feedback with automated verification:

```
Model generates code --> Verifier checks it --> Train on verified outputs --> Repeat
```

This creates a self-improving loop where the model learns from its own successful outputs.

---

## How It Works

```
                         halo-forge Pipeline

    +------------------+     +------------------+     +------------------+
    |  DATA GENERATION |     |   SFT TRAINING   |     |  RAFT TRAINING   |
    |                  |     |                  |     |                  |
    | - Public datasets| --> | - LoRA adapters  | --> | - Generate       |
    | - LLM generation |     | - BF16 precision |     | - Verify         |
    |                  |     | - Checkpointing  |     | - Filter         |
    +------------------+     +------------------+     | - Train          |
                                                      | - Repeat         |
                                                      +------------------+
                                                              |
                                                              v
                                                      +------------------+
                                                      |   BENCHMARKING   |
                                                      |                  |
                                                      | - pass@k metrics |
                                                      | - Compare models |
                                                      +------------------+
```

### Pipeline Stages

**Stage 1: Data Generation**
- Download public datasets (CodeForces, MBPP, HumanEval)
- Or generate domain-specific data with LLMs (DeepSeek, Ollama)
- Format into training-ready JSONL

**Stage 2: SFT (Supervised Fine-Tuning)**
- Train base model on your dataset
- Uses LoRA for memory efficiency
- BF16 precision (optimal for Strix Halo)
- Produces initial fine-tuned checkpoint

**Stage 3: RAFT (Reward-Ranked Fine-Tuning)**
- Generate multiple samples per prompt
- Verify each sample with your chosen verifier
- Filter to keep only verified samples
- Train on filtered samples
- Repeat for N cycles

**Stage 4: Benchmarking**
- Evaluate final model on held-out prompts
- Compute pass@1, pass@5, pass@10 metrics
- Compare across training stages

---

## Key Concepts

### SFT (Supervised Fine-Tuning)

Initial training phase where the model learns from human-written examples. This establishes baseline capabilities before RAFT refinement.

### RAFT (Reward-Ranked Fine-Tuning)

Iterative training loop:
1. Generate N samples per prompt using current model
2. Verify all samples with chosen verifier
3. Assign rewards based on verification result
4. Filter to keep top K% of samples above reward threshold
5. SFT on filtered samples
6. Repeat with updated model

### Verifiers

Automated systems that check if generated code meets requirements. Examples:
- **Compile verifiers**: Does the code compile without errors?
- **Test verifiers**: Do unit tests pass?
- **Chained verifiers**: Multiple stages (compile, then test)

### Rewards

Numeric scores assigned by verifiers:
- `0.0`: Complete failure (syntax errors, doesn't compile)
- `0.5`: Partial success (compiles but crashes)
- `1.0`: Full success (compiles, runs, produces correct output)

### pass@k

Standard metric for code generation. Given k attempts per problem:
- **pass@1**: Probability of solving on first try
- **pass@5**: Probability of solving within 5 tries
- **pass@10**: Probability of solving within 10 tries

---

## Hardware Requirements

### AMD Strix Halo (Primary Target)

| Component | Specification |
|-----------|---------------|
| GPU | AMD Strix Halo (RDNA 3.5, gfx1151) |
| Memory | 128GB unified LPDDR5X |
| Compute Units | 40 CUs (2560 shaders) |
| TDP | 120W |

### Key Hardware Characteristics

**Unified Memory Architecture**
- GPU and CPU share the same 128GB memory pool
- No discrete VRAM limitation
- Enables training 7B+ models without memory constraints

**Compute-Bound Workload**
- GPU runs at 96-99% compute utilization during training
- Memory bandwidth is NOT the bottleneck
- This is why 4-bit quantization is slower (dequantization overhead with no memory benefit)

### Optimal Settings

```yaml
# Use BF16, NOT 4-bit quantization
bf16: true

# Critical for unified memory - prevents GPU hangs
dataloader_num_workers: 0
dataloader_pin_memory: false

# Enable gradient checkpointing for large models
gradient_checkpointing: true
```

### Software Requirements

- Fedora 42+ (for toolbox support)
- Kernel 6.16+ recommended
- 100GB+ storage for models and datasets

---

## Installation

### Step 1: Build the Toolbox

The toolbox provides a self-contained environment with ROCm 7, PyTorch, and all dependencies.

```bash
cd halo-forge/toolbox
chmod +x build.sh
./build.sh
```

This builds a container with:
- ROCm 7 nightly from TheRock
- PyTorch nightly from AMD gfx1151 nightlies
- bitsandbytes built from upstream with gfx1151 target
- Flash Attention from ROCm fork

### Step 2: Create and Enter Toolbox

```bash
toolbox create halo-forge --image localhost/halo-forge:latest
toolbox enter halo-forge
```

### Step 3: Verify Setup

```bash
# Check GPU detection
python3 -c "import torch; print(f'ROCm: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Check halo-forge
halo-forge info
```

Expected output:
```
ROCm: True
GPU: AMD Radeon(TM) Graphics
```

---

## Quick Start

### Validate Your Setup First

```bash
# Enter toolbox
toolbox enter halo-forge

# Run quick validation (5 seconds, no GPU)
halo-forge test --level smoke

# Run standard validation (2-3 minutes, loads model)
halo-forge test --level standard

# Run full validation (5 minutes, includes training step)
halo-forge test --level full
```

### Minimal Working Example

```bash
# Enter toolbox
toolbox enter halo-forge
cd ~/training

# 1. Download dataset (MBPP Python examples)
halo-forge data prepare --dataset mbpp --output data/train.jsonl

# 2. Create prompts file
head -50 data/train.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    if 'text' in d:
        print(json.dumps({'prompt': 'Write a Python function: ' + d.get('prompt', 'example')}))
" > data/prompts.jsonl

# 3. Run SFT (1 epoch for quick test)
halo-forge sft train --data data/train.jsonl --output models/sft --epochs 1

# 4. Run RAFT (1 cycle for quick test)
halo-forge raft train \
  --checkpoint models/sft/final_model \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --cycles 1 \
  --output models/raft

# 5. Benchmark
halo-forge benchmark run \
  --model models/raft/cycle_1_final \
  --prompts data/prompts.jsonl \
  --verifier gcc \
  --samples 5
```

---

## Testing Your Setup

halo-forge includes a built-in test command to validate your installation at various levels.

### Test Levels

| Level | Time | GPU Required | What It Tests |
|-------|------|--------------|---------------|
| `smoke` | 5s | No | Imports, compiler availability, verifier logic |
| `standard` | 2-3 min | Yes | Model loading, code generation, verification |
| `full` | 5 min | Yes | Complete mini-RAFT cycle with training step |

### Running Tests

```bash
# Quick smoke test - validates environment without GPU
halo-forge test --level smoke

# Standard test - loads model, generates code, verifies
halo-forge test --level standard

# Full test - includes SFT training step
halo-forge test --level full

# With verbose output for debugging
halo-forge test --level standard --verbose

# Use a different model
halo-forge test --level standard --model Qwen/Qwen2.5-Coder-7B
```

### Expected Output

```
============================================================
halo-forge Standard Test
Model: Qwen/Qwen2.5-Coder-0.5B
============================================================

  [OK] Import modules (0.0s)
  [OK] Compiler available (0.0s)
  [OK] GPU available (0.0s)
  [OK] Model loading (1.2s)
  [OK] Code generation (21.6s)
  [OK] Code verification (0.3s)

============================================================
Test Results: 6/6 passed
============================================================
```

---

## Verifiers

Verifiers are the core of RLVR training. They provide the reward signal that guides model improvement.

### Built-in Verifiers

| Verifier | Language | Use Case | Reward Model |
|----------|----------|----------|--------------|
| `GCCVerifier` | C/C++ | Linux compilation | Graduated: 0.0→0.5→0.7→1.0 |
| `ClangVerifier` | C/C++ | Alternative compiler | Graduated: 0.0→0.5→0.7→1.0 |
| `MinGWVerifier` | C/C++ | Cross-compile to Windows | Graduated: 0.0→0.5→0.7→1.0 |
| `RemoteMSVCVerifier` | C/C++ | Remote Windows MSVC | Graduated: 0.0→0.5→0.7→1.0 |
| `ExecutionVerifier` | C/C++ | **NEW** Test cases with I/O | 0.5 + 0.5 × pass_rate |
| `MultiLanguageVerifier` | Multi | **NEW** Auto-detect language | Delegates to language verifier |
| `RustVerifier` | Rust | Rust compilation | Graduated rewards |
| `GoVerifier` | Go | Go compilation | Graduated rewards |
| `DotNetVerifier` | C# | .NET compilation | Graduated rewards |
| `PowerShellVerifier` | PS1 | PowerShell syntax | Syntax check |
| `PytestVerifier` | Python | Test-driven development | Partial credit for tests |
| `UnittestVerifier` | Python | Built-in unittest | 1.0 if all pass |
| `ChainedVerifier` | Any | Multi-stage | Weighted sum of stages |
| `SubprocessVerifier` | Any | Custom CLI tool | Based on return code |

### Choosing a Verifier

```
Use Case                           Recommended Verifier
-----------------------------------------------------------------
C++ competitive programming   -->  GCCVerifier or ExecutionVerifier
Windows API development       -->  MinGWVerifier or RemoteMSVCVerifier
Python with tests             -->  PytestVerifier
Rust code                     -->  RustVerifier
Go code                       -->  GoVerifier
C#/.NET code                  -->  DotNetVerifier
PowerShell scripts            -->  PowerShellVerifier
Mixed language dataset        -->  MultiLanguageVerifier (--verifier auto)
Multi-stage (compile + test)  -->  ChainedVerifier
Custom domain                 -->  SubprocessVerifier or custom class
```

### Using Verifiers in Code

```python
from halo_forge.rlvr.verifiers import GCCVerifier, PytestVerifier, ChainedVerifier

# Simple compilation check
verifier = GCCVerifier(max_workers=8)
result = verifier.verify(code)
print(f"Success: {result.success}, Reward: {result.reward}")

# Python tests
verifier = PytestVerifier(timeout=60)

# Multi-stage: compile then test
verifier = ChainedVerifier([
    GCCVerifier(),
    PytestVerifier()
])
```

### Creating Custom Verifiers

```python
from halo_forge.rlvr.verifiers import Verifier, VerifyResult

class MyVerifier(Verifier):
    def verify(self, code: str) -> VerifyResult:
        # Extract code from model output
        extracted = self.extract_code(code)
        
        # Your verification logic
        success = your_check(extracted)
        
        return VerifyResult(
            success=success,
            reward=1.0 if success else 0.0,
            details="Verification passed" if success else "Failed",
            error=None if success else "Error message"
        )
```

See [docs/VERIFIERS.md](docs/VERIFIERS.md) for detailed documentation.

---

## Configuration Reference

### SFT Configuration

```yaml
# configs/sft_example.yaml

model:
  name: Qwen/Qwen2.5-Coder-7B      # Base model from HuggingFace
  trust_remote_code: true          # Required for Qwen models
  attn_implementation: eager       # Or flash_attention_2 with ROCm fork

data:
  train_file: data/train.jsonl     # Training data in JSONL format
  validation_split: 0.05           # 5% held out for validation
  max_seq_length: 2048             # Maximum sequence length

# LoRA configuration (BF16 is optimal for Strix Halo)
lora:
  r: 16                            # LoRA rank
  alpha: 32                        # LoRA alpha (typically 2x rank)
  dropout: 0.05                    # Dropout rate
  target_modules:                  # Modules to apply LoRA
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  output_dir: models/sft           # Where to save checkpoints
  num_train_epochs: 3              # Number of epochs
  per_device_train_batch_size: 2   # Batch size per GPU
  gradient_accumulation_steps: 16  # Effective batch = 2 * 16 = 32
  learning_rate: 2e-4              # Learning rate
  warmup_ratio: 0.03               # Warmup fraction
  gradient_checkpointing: true     # Save memory
  bf16: true                       # Use BF16 precision
  
  # Critical for Strix Halo unified memory
  dataloader_num_workers: 0
  dataloader_pin_memory: false
```

### RAFT Configuration

```yaml
# configs/raft_example.yaml

# Model paths
sft_checkpoint: models/sft/final_model   # SFT checkpoint to start from
output_dir: models/raft                   # Where to save RAFT checkpoints
prompts: data/prompts.jsonl               # Training prompts

# RAFT parameters
raft:
  num_cycles: 5                    # Number of generate-verify-train cycles
  samples_per_prompt: 8            # Samples to generate per prompt
  reward_threshold: 0.5            # Minimum reward to keep sample
  keep_top_percent: 0.5            # Keep top 50% above threshold

# Generation settings
generation:
  max_new_tokens: 1024             # Max tokens to generate
  temperature: 0.7                 # Sampling temperature
  batch_size: 8                    # Generation batch size

# Training settings (per cycle)
training:
  epochs: 1                        # Epochs per cycle
  batch_size: 2                    # Training batch size
  gradient_accumulation_steps: 16  # Gradient accumulation
  learning_rate: 5e-5              # Learning rate (lower than SFT)
  
  # Critical for Strix Halo
  dataloader_num_workers: 0
  dataloader_pin_memory: false

# Hardware settings
hardware:
  bf16: true                       # Use BF16 precision
  gradient_checkpointing: true     # Enable checkpointing
  attn_implementation: eager       # Attention implementation

# Verifier settings
verifier:
  type: gcc                        # gcc, mingw, msvc, pytest
```

---

## Performance Expectations

### Training Times (Strix Halo, 7B model)

| Operation | Time | Notes |
|-----------|------|-------|
| Load model | 2-3 min | First load, cached afterward |
| Generate 100 samples | 10-15 min | batch_size=8 |
| Verify 100 (GCC) | 30 sec | 8 workers parallel |
| Verify 100 (MSVC/SSH) | 2-3 min | Network overhead |
| Train 1 epoch on 100 | 5-10 min | Depends on seq_len |
| **Full RAFT cycle** | **30-45 min** | Generate + verify + train |

### Compilation Rate Progression

Typical improvement over RAFT cycles for C++ code:

| Stage | Compile Rate | Notes |
|-------|--------------|-------|
| SFT only | 20-30% | Many syntax errors |
| Cycle 1 | 35-45% | Basic structure improved |
| Cycle 2 | 45-55% | Fewer missing semicolons |
| Cycle 3 | 55-65% | Better type handling |
| Cycle 4+ | 60-70% | Diminishing returns |

### Resource Usage

During training on Strix Halo:
- **GPU utilization**: 95-99% (compute-bound)
- **GTT memory**: 40-60GB typical
- **Power consumption**: ~100W sustained

### Automatic Resume

RAFT automatically caches progress at each stage. If a run crashes, simply re-run the same command:

```bash
# If this crashes during cycle 3...
halo-forge raft train --cycles 5 --output models/raft

# Just run it again - it auto-resumes:
halo-forge raft train --cycles 5 --output models/raft
# Output: "Cycle 1 already complete, skipping..."
# Output: "Cycle 2 already complete, skipping..."
# Output: "Loading cached samples..." (resumes cycle 3)
```

Cache files per cycle:
- `cycle_N_samples.jsonl` - Generated completions (skip generation on resume)
- `cycle_N_verified.jsonl` - Verification results (skip verification on resume)
- `cycle_N_final/` - Trained checkpoint (skip entire cycle on resume)

---

## Benchmark Results

### Training Results

Results will vary based on model, dataset, hardware, and configuration. RAFT training typically shows:

| Observation | Notes |
|-------------|-------|
| **Early cycles (1-3)** | Largest gains as model learns basic patterns |
| **Mid cycles (4-5)** | Continued improvement, slower rate |
| **Late cycles (6+)** | Diminishing returns; monitor for degradation |

**Training tips:**
- Monitor compile/pass rate each cycle
- Stop when improvement plateaus or reverses
- BF16 precision is optimal for Strix Halo (4-bit is slower due to dequantization overhead)
- In our testing, 5-6 cycles often worked well before diminishing returns

### Demo Benchmark Results

Quick validation benchmarks on built-in prompts (16 prompts, 2 cycles):

| Model | Baseline | After 2 Cycles | Time |
|-------|----------|----------------|------|
| Qwen2.5-Coder-0.5B | 32.0% | 32.0% | 41 min |
| Qwen2.5-Coder-1.5B | 67.2% | 67.2% | 56 min |
| Qwen2.5-Coder-3B | TBD | TBD | ~150 min |

Demo benchmarks use a small prompt set (16 prompts, 2 cycles) to quickly validate the pipeline works on your hardware. With such small datasets, minimal or no improvement is expected and normal.

### Running Your Own Benchmark

```bash
# Quick smoke test (no GPU, ~10 sec)
halo-forge test --level smoke

# Demo benchmark with GPU (~40 min for 0.5B)
halo-forge benchmark full --model Qwen/Qwen2.5-Coder-0.5B --cycles 2

# Full benchmark suite (all models, ~5 hours)
halo-forge benchmark full --suite all
```

---

## Troubleshooting

### GPU not detected

```bash
# Check device access
ls -l /dev/dri /dev/kfd

# If missing, add udev rules
sudo tee /etc/udev/rules.d/99-amd-kfd.rules >/dev/null <<'EOF'
SUBSYSTEM=="kfd", GROUP="render", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="card[0-9]*", GROUP="render", MODE="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### GPU hang during training

Ensure these settings in your config:
```yaml
training:
  dataloader_num_workers: 0    # Must be 0
  dataloader_pin_memory: false # Must be false
```

Also try:
```bash
export HSA_ENABLE_SDMA=0
```

### Out of memory

Unlikely with 128GB, but if it happens:
1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Enable `gradient_checkpointing: true`
4. Reduce `max_seq_length`

### Slow generation

1. Verify using BF16 (not 4-bit quantization)
2. Check GPU utilization with `rocm-smi` or `radeontop`
3. Ensure only one training process running

### Import errors

Make sure you're inside the toolbox:
```bash
toolbox enter halo-forge
```

---

## Research Background

### RAFT: Reward-Ranked Fine-Tuning

halo-forge implements RAFT as described in:

> **RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment**
> Hanze Dong, Wei Xiong, Deepanshu Goyal, et al.
> Published: TMLR, 23 Nov 2023
> [Paper (OpenReview)](https://openreview.net/forum?id=m7p5O7zblY)

Key insight from the paper:
> "Utilizing a reward model and a sufficient number of samples, our approach selects the high-quality samples, discarding those that exhibit undesired behavior, and subsequently enhancing the model by fine-tuning on these filtered samples."

### Why RAFT Over PPO/GRPO

| Approach | Complexity | Memory | Stability |
|----------|------------|--------|-----------|
| PPO | High | 4x model | Requires careful tuning |
| GRPO | Medium | 2x model | Better but still tricky |
| RAFT | Low | 1x model | Simple and stable |

RAFT is essentially "iterated rejection sampling" - simple to implement, stable to train, and produces comparable results with less engineering effort.

### What Models Learn During RAFT

Through verification-based training, models learn:
1. **Syntax correctness**: Matching braces, semicolons
2. **Include statements**: Right headers for functions used
3. **Type consistency**: Matching function signatures
4. **Error handling**: Adding null checks, return values

### Limitations

RAFT with compile verification only ensures code compiles, not that it's correct:
- Code may compile but produce wrong output
- Edge cases may not be handled
- Efficiency is not optimized

For correctness verification, use test-based verifiers (PytestVerifier) or execution-based verification.

### Additional References

- **STaR: Self-Taught Reasoner** (Zelikman et al., 2022) - Bootstrap reasoning from correct outputs
- **HumanEval** (Chen et al., 2021) - Standard code evaluation benchmark
- **kyuz0's AMD Strix Halo Guide** - Community resource for Strix Halo training

---

## Extensibility

halo-forge is designed as a generic RLVR framework that can be extended to domains beyond code generation.

### Custom Verifier Template

Any domain with deterministic verification can use halo-forge:

```python
from halo_forge.rlvr.verifiers import Verifier, VerifyResult

class MyDomainVerifier(Verifier):
    def __init__(self, config, max_workers=8):
        super().__init__(max_workers=max_workers)
        self.config = config
    
    def verify(self, output: str) -> VerifyResult:
        # Your domain-specific verification
        is_valid = your_verification_logic(output)
        
        return VerifyResult(
            success=is_valid,
            reward=1.0 if is_valid else 0.0,
            details="Verification details"
        )
    
    def cleanup(self):
        # Optional: cleanup resources
        pass
```

### Example Domain Applications

| Domain | Verifier Approach |
|--------|-------------------|
| **Security research** | Detection testing, static analysis |
| **Formal verification** | Theorem provers (Coq, Lean, Z3) |
| **Multi-language** | Additional compilers (Rust, Go, Zig) |
| **Execution testing** | I/O comparison for algorithm correctness |
| **API compliance** | Check generated code against specifications |
| **Documentation** | Verify generated docs compile/render |

### Roadmap

- **v0.3.0**: Multi-language support, execution verifier, LR decay, metrics tracking
- **v0.4.0**: Inference optimization mode, GGUF/ONNX export, QAT training
- **v0.5.0** (Current): Vision-language training (VLM RLVR)
- **v0.6.0** (Planned): Audio-language training (ASR/TTS)
- **v1.0.0** (Planned): Cross-platform GUI, full multi-modal support

---

## Benchmark Results

Demo benchmarks on AMD Strix Halo (128GB unified memory) with 16 prompts, 2 cycles:

| Model | Baseline Compile | Final Compile | Improvement | Time |
|-------|-----------------|---------------|-------------|------|
| Qwen2.5-Coder-0.5B | 32.0% | 32.0% | +0.0% | 41 min |
| Qwen2.5-Coder-1.5B | 67.2% | 67.2% | +0.0% | 52 min |
| Qwen2.5-Coder-3B | 97.7% | 99.2% | **+1.6%** | 79 min |

**Note**: Demo benchmarks validate the pipeline works. Larger datasets and more cycles are needed for meaningful improvement.

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for detailed methodology and results.

---

## Project Structure

```
halo-forge/
├── toolbox/              # ROCm toolbox (Dockerfile, build scripts)
│   ├── Dockerfile        # Container definition
│   ├── build.sh          # Build script
│   └── TOOLBOX_SETUP.md  # Setup guide
├── halo_forge/           # Main Python package
│   ├── data/             # Data generation
│   │   ├── public_datasets.py   # Download CodeForces, MBPP, etc.
│   │   └── llm_generate.py      # Generate with LLMs
│   ├── sft/              # SFT training
│   │   ├── trainer.py    # SFTTrainer class
│   │   └── config.py     # SFTConfig dataclass
│   ├── rlvr/             # RAFT training + verifiers
│   │   ├── raft_trainer.py      # RAFTTrainer class
│   │   └── verifiers/           # Verifier implementations
│   │       ├── base.py          # Abstract Verifier class
│   │       ├── compile.py       # GCC, MinGW, Clang
│   │       ├── remote_compile.py # Remote MSVC
│   │       ├── test_runner.py   # Pytest, Unittest
│   │       └── custom.py        # SubprocessVerifier
│   ├── benchmark/        # Evaluation
│   │   └── pass_at_k.py  # pass@k computation
│   ├── utils/            # Utilities
│   │   └── hardware.py   # Hardware detection
│   └── cli.py            # Command-line interface
├── configs/              # Example configurations
│   ├── sft_example.yaml
│   └── raft_example.yaml
├── docs/                 # Documentation
│   ├── QUICKSTART.md
│   ├── FULL_PIPELINE.md
│   ├── THEORY.md
│   ├── VERIFIERS.md
│   ├── HARDWARE_NOTES.md
│   └── experimental/     # Research & experimental features
│       ├── LEARNING_RATE_THEORY.md
│       ├── LR_QUICK_REFERENCE.md
│       └── configs/      # Experimental config templates
├── examples/             # Working examples
│   ├── full_pipeline/
│   ├── compile_verified/
│   └── test_verified/
└── tests/                # Test suite
```

---

## License

Apache 2.0

---

## Acknowledgments

- AMD for Strix Halo hardware
- [kyuz0](https://github.com/kyuz0/amd-strix-halo-llm-finetuning) for the original fine-tuning toolbox
- TheRock project for ROCm nightlies
- The Strix Halo community for testing and feedback
- RAFT paper authors for the foundational research
