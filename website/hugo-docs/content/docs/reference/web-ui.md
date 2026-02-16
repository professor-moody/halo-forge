---
title: "Web UI"
description: "Dashboard for training, benchmarking, and monitoring"
weight: 5
---

The halo-forge web interface provides a modern dashboard for training, benchmarking, and monitoring LLM fine-tuning jobs.

## Quick Start

```bash
# Launch the UI
halo-forge ui

# Custom host/port
halo-forge ui --host 0.0.0.0 --port 8888

# With auto-reload for development
halo-forge ui --reload

# Headless-safe (default behavior)
halo-forge ui --no-browser

# Explicit browser auto-open
halo-forge ui --open-browser
```

The UI will be available at `http://127.0.0.1:8080` by default.
Startup logs print canonical route URLs (`/`, `/training`, `/benchmark`, `/inference`).

## Pages Overview

### Dashboard (`/`)

The main landing page showing:
- **Training Launcher**: Primary start controls for SFT/RAFT/modality training
- **GPU Status**: Real-time GPU utilization
- **Active Jobs**: Currently running training/benchmark jobs
- **Completed/Failed Counts**: Job statistics
- **Training History Chart**: Loss curves from recent runs
- **Benchmark Scores Chart**: Pass@1 comparisons across models
- **Recent Runs**: Quick access to completed jobs
- **Advanced Diagnostics (Optional)**: grouped coding/non-coding/ops setup and system checks
- **Advanced Actions**: `Run Setup Check`, `Generate Setup Artifacts`, and `Run System Health Check`

### Training (`/training`)

Configure and launch training jobs:
- **Training-first flow**: model + dataset + output + start controls are primary
- **Advanced setup diagnostics**: optional collapsed panel for troubleshooting checks

**SFT (Supervised Fine-Tuning)**
- Model selection (HuggingFace or local path)
- Dataset selection (Alpaca, MetaMath, GSM8K, xLAM)
- Training hyperparameters (epochs, batch size, learning rate)
- LoRA configuration (rank, alpha)
- Gradient checkpointing toggle

**RAFT (Reward-Ranked Fine-Tuning)**
- Preset configurations: Conservative, Aggressive, Custom
- Verifier selection (HumanEval, MBPP, LiveCodeBench, Math)
- RAFT-specific parameters (cycles, samples per prompt, temperature, keep percent)
- Reward threshold configuration

### Benchmark (`/benchmark`)

**Purpose**: Run standardized benchmarks for **model comparison** (not training).

> **Note**: This page is for benchmark **reporting** — comparing your trained model to published results. For training verification, use the Verifiers page to test the native training verifiers.

| Type | Benchmarks | Models |
|------|------------|--------|
| **Code** | HumanEval, MBPP, LiveCodeBench | Qwen2.5-Coder, DeepSeek-Coder |
| **VLM** | TextVQA, DocVQA, ChartQA | Qwen2-VL, LLaVA, Phi-3-Vision |
| **Audio** | LibriSpeech, CommonVoice | Whisper (tiny to large-v3) |
| **Reasoning** | GSM8K, MATH | Qwen2.5-Instruct, Mistral-Instruct |
| **Agentic** | xLAM Function Calling | Qwen2.5-Instruct, Mistral-Instruct |

**Uses community tools**: VLM benchmarks use VLMEvalKit when available for standardized, comparable results.

Features:
- Model autocomplete with popular presets
- Sample limit slider
- Custom output directory
- One-click launch with redirect to Monitor

### Monitor (`/monitor`)

Real-time job monitoring with:
- **Live Duration Counter**: Updates every second
- **Progress Bar**: Visual step/epoch progress
- **Loss Chart**: ECharts line graph with real-time updates
- **Metrics Panel**: Loss, learning rate, gradient norm, verification rate
- **Log Viewer**: Streaming logs with syntax highlighting
- **Stop Button**: Graceful job termination with confirmation

### Config (`/config`)

YAML configuration editor:
- Syntax highlighting
- Schema validation (checks for valid halo-forge config keys)
- Save to file
- Template presets
- All-module readiness banner for config contract status

### Verifiers (`/verifiers`)

**Purpose**: Test **native training verifiers** — these provide reward signals for RAFT, not benchmark scores.

> **Note**: Verifiers are **training infrastructure**. They provide graduated rewards (0.0 to 1.0) for the RAFT training loop. For final model evaluation with comparable metrics, use the Benchmark page.

Available verifiers:
- **HumanEval** (Python): HumanEval test suite verification
- **MBPP** (Python basics): Mostly Basic Python Problems tests
- **Execution** (Multi-language): Compile + run verification
- **Math** (Numerical): Answer extraction and numeric comparison
- **GSM8K** (Grade-school math): Math reasoning verification

Interactive testing:
1. Select a verifier
2. Enter code snippet
3. Click "Run Verification"
4. See graduated reward result (not just pass/fail)

### Datasets (`/datasets`)

Browse available datasets:
- Public datasets from HuggingFace
- Local JSONL files
- Preview samples
- Filter by type/source

### Results (`/results`)

Benchmark results table:
- Model, Benchmark, Pass@1, Pass@5, Duration
- Multi-select for comparison
- Sort by any column
- Export to JSON/CSV

### Inference (`/inference`)

Launch inference optimize and benchmark jobs from the UI:
- `inference optimize` launch contract (precision/latency/calibration)
- `inference benchmark` launch contract (prompts, tokens, warmup)
- Durable launch context + monitor/relaunch parity
- All-module readiness banner for inference contract status

### Benchmark Advanced (`/benchmark-advanced`)

Batch orchestration for non-code benchmark runs:
- VLM/audio/reasoning/agentic batch launch
- Per-domain dataset selection
- Monitor handoff to first launched job
- All-module readiness banner for non-code benchmark contract status

### Research Hub (`/research-hub`)

Cross-module ops readiness visibility:
- Reads canonical ops readiness report when available
- Reads canonical all-module readiness report when available:
  - `results/readiness/all_modules_readiness.v1.json`
- Reads canonical all-module qualification report when available:
  - `results/readiness/all_module_qualification.v1.json`
- Reads canonical all-module bootstrap report when available:
  - `results/readiness/all_module_bootstrap.v1.json`
- Reads canonical all-module live execution report when available:
  - `results/readiness/all_module_live_execution.v1.json`
- Falls back to live contract checks when report missing/corrupt
- Shows actionable pass/warn/fail evidence by module
- Shows optional dataset burn-in provenance when available:
  - `burnin_report_present`
  - `burnin_generated_at`
  - `burnin_status`
- Shows qualification lifecycle provenance when available:
  - `qualification_report_present`
  - `qualification_generated_at`
  - `qualification_status`
  - `qualification_profile`
- Shows bootstrap evidence-generation provenance when available:
  - `bootstrap_report_present`
  - `bootstrap_generated_at`
  - `bootstrap_status`
  - `bootstrap_profile`
- Shows live execution provenance when available:
  - `live_report_present`
  - `live_generated_at`
  - `live_status`
  - `live_profile`
- Supports `Generate setup artifacts` action (tracked, non-blocking job in Monitor)
- Supports `Run system health check` action (tracked, non-blocking job in Monitor)
- Supports module-level `Generate Setup Artifacts (Advanced)` actions to bootstrap evidence roots on demand
- Supports module-level `Run System Health Check (Advanced)` actions for bounded per-module execution checks
- Supports `Run setup check` action (tracked, non-blocking job in Monitor)

## Deep-Link Query Parameters

UI routes support deterministic preselection via query params:

- `/training?mode=<sft|raft|vlm|audio|reasoning|agentic>`
- `/benchmark?view=<code|non_code>`
- `/benchmark-advanced?domains=<csv>` (example: `domains=vlm,audio`)
- `/inference?mode=<optimize|benchmark>`
- `/ops-console?module=<config|data|info|plot>&execution_mode=<contract|live>`
- `/research-hub?module=<module_key>`

Unknown values are ignored safely and pages fall back to default selections.

## Readiness Semantics (Warn-and-Launch)

UI readiness is contract-based and **does not block launches** for missing historical evidence.

- **PASS**: Required contracts are healthy.
- **WARN**: Evidence missing/stale; launch is still allowed.
- **FAIL**: Contract/preflight issue. Launch may be blocked when `launch_blocked=true`.

Banner wording:
- `Evidence missing (non-blocking)` means files like prior `training_summary.json` or benchmark outputs were not found yet.
- `Setup check not satisfied (advanced diagnostics)` means setup checks found issues; users can still correct form inputs and run training.
- `Qualification issue` means an explicit lifecycle check failed in qualification mode (separate from normal launch readiness).
- `Bootstrap issue` means evidence generation encountered a contract/probe failure.
- `Live probe issue` means bounded live probe execution failed for the module in the selected profile.

Each readiness banner includes:
- Stable issue metadata (`issue_code`, `severity`, `issue_scope`).
- `Fix now` remediation text and fix options.
- `What is missing?` evidence gaps or expected evidence root path.
- `Run setup check` button to refresh module setup diagnostics.
- `Generate setup artifacts` button to create bounded bootstrap artifacts for the selected module.
- `Run system health check` button to execute bounded live command probes for the selected module.

## Architecture

### Services Layer

The UI uses a services architecture that connects NiceGUI pages to halo-forge backends:

```
UI Pages (Dashboard, Training, Monitor, ...)
            │
            ▼
UI Services (TrainingService, BenchmarkService, HardwareMonitor, ...)
            │
            ▼
halo-forge Core (CLI Commands, Trainers, Verifiers)
```

### Event Bus

Real-time updates are powered by an event bus system:

- `JOB_CREATED`, `JOB_STARTED`, `JOB_COMPLETED`, `JOB_FAILED`, `JOB_STOPPED`
- `METRICS_UPDATE`: Loss, learning rate, step progress
- `LOG_LINE`: Streaming log output
- `GPU_UPDATE`: Real-time GPU utilization
- `CHECKPOINT_SAVED`: Checkpoint save notifications

Pages subscribe to events and update UI elements without polling.

### State Management

Job state is managed centrally in `ui/state.py`:
- Job creation and tracking
- Metrics history for charts
- Progress tracking (epoch, step, cycle)

## AMD Strix Halo Optimization

The UI automatically applies optimized environment variables for AMD Strix Halo:

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.1
PYTORCH_ROCM_ARCH=gfx1151
HIP_VISIBLE_DEVICES=0
PYTORCH_HIP_ALLOC_CONF=backend:native,expandable_segments:True,...
HSA_ENABLE_SDMA=0
```

These are set when launching any training or benchmark subprocess.

## Customization

### Theme Colors

Colors are defined in `ui/theme.py`:

```python
COLORS = {
    "bg_primary": "#0f1318",
    "bg_secondary": "#161b22",
    "bg_card": "#1c2128",
    "primary": "#7C9885",      # Sage green
    "secondary": "#8BA888",
    "accent": "#9BC4A8",
    "success": "#7C9885",
    "running": "#4C9AFF",
    "error": "#F85149",
    ...
}
```

### Adding New Pages

1. Create page component in `ui/pages/`
2. Add route in `ui/app.py`
3. Add navigation item in `ui/components/sidebar.py`

## Feature Flags (Default-On Kill Switches)

The following pages are enabled by default:
- `/inference`
- `/benchmark-advanced`
- `/research-hub`

Disable any page with env vars:

```bash
HALO_UI_ENABLE_INFERENCE_PAGE=0
HALO_UI_ENABLE_BENCHMARK_ADVANCED_PAGE=0
HALO_UI_ENABLE_RESEARCH_HUB_PAGE=0
```

Accepted false values: `0`, `false`, `no`, `off`.

## Troubleshooting

### "gio: Operation not supported"

Use `halo-forge ui --no-browser` (default) in headless environments.  
Only use `--open-browser` when desktop browser integration is available.

### Burn-in provenance unavailable

If burn-in status is unavailable in Dashboard or Research Hub, generate the report:

```bash
python3 scripts/run_ops_dataset_burnin.py \
  --burnin-profile tiny-v1 \
  --write-report \
  --report-file results/readiness/ops_dataset_burnin.v1.json
```

### All-module readiness unavailable

If coding/non-coding readiness is unavailable in Dashboard or Research Hub, generate the canonical report:

```bash
python3 scripts/run_all_module_matrix.py \
  --fixture-pack v1 \
  --write-report \
  --report-file results/readiness/all_modules_readiness.v1.json
```

### All-module qualification unavailable

If qualification status is unavailable in Dashboard or Research Hub, generate the canonical qualification report:

```bash
python3 scripts/run_all_module_qualification.py \
  --qualification-profile fixture-v1 \
  --fixture-pack v1 \
  --write-report \
  --report-file results/readiness/all_module_qualification.v1.json
```

### All-module live execution unavailable

If live execution status is unavailable in Dashboard or Research Hub, generate the canonical live report:

```bash
python3 scripts/run_all_module_live_matrix.py \
  --live-profile live-smoke-v1 \
  --write-report \
  --report-file results/readiness/all_module_live_execution.v1.json
```

### Duration/Progress not updating

Ensure the training process is emitting progress to stdout. The MetricsParser looks for patterns like:
- `Epoch X/Y`
- `Step X/Y`
- `loss: X.XXX`
- `lr: X.XXe-XX`

### GPU not detected

Check that ROCm is properly installed and `rocm-smi` is accessible.
