# halo-forge Web UI

The halo-forge web interface provides a modern dashboard for training, benchmarking, and monitoring LLM fine-tuning jobs.

## Quick Start

```bash
# Launch the UI
halo-forge ui

# Custom host/port
halo-forge ui --host 0.0.0.0 --port 8888

# With auto-reload for development
halo-forge ui --reload
```

The UI will be available at `http://127.0.0.1:8080` by default.

## Pages Overview

### Dashboard (`/`)

The main landing page showing:
- **GPU Status**: Real-time GPU utilization
- **Active Jobs**: Currently running training/benchmark jobs
- **Completed/Failed Counts**: Job statistics
- **Training History Chart**: Loss curves from recent runs
- **Benchmark Scores Chart**: Pass@1 comparisons across models
- **Recent Runs**: Quick access to completed jobs

### Training (`/training`)

Configure and launch training jobs:

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

Run standardized benchmarks:

| Type | Benchmarks | Models |
|------|------------|--------|
| **Code** | HumanEval, MBPP, LiveCodeBench | Qwen2.5-Coder, DeepSeek-Coder |
| **VLM** | TextVQA, DocVQA, MMStar, ChartQA | Qwen2-VL, LLaVA, Phi-3-Vision |
| **Audio** | LibriSpeech, CommonVoice | Whisper (tiny → large-v3) |
| **Agentic** | xLAM Function Calling | Qwen2.5-Instruct, Mistral-Instruct |

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

### Verifiers (`/verifiers`)

Test code verification backends:
- HumanEval (Python)
- MBPP (Python basics)
- LiveCodeBench (Multi-language)
- Math (Numerical)
- GSM8K (Grade-school math)

Interactive testing with code input and verification results.

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

## Architecture

### Services Layer

The UI uses a services architecture that connects NiceGUI pages to halo-forge backends:

```
┌─────────────────────────────────────────────┐
│                 UI Pages                     │
│  Dashboard │ Training │ Monitor │ ...       │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│              UI Services                     │
│  TrainingService    │ BenchmarkService      │
│  HardwareMonitor    │ ResultsService        │
│  VerifierService    │ DatasetsService       │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│            halo-forge Core                   │
│  CLI Commands │ Trainers │ Verifiers        │
└─────────────────────────────────────────────┘
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

## Troubleshooting

### "gio: Operation not supported"

This harmless warning appears when running on headless systems where the browser can't auto-open.

### Duration/Progress not updating

Ensure the training process is emitting progress to stdout. The MetricsParser looks for patterns like:
- `Epoch X/Y`
- `Step X/Y`
- `loss: X.XXX`
- `lr: X.XXe-XX`

### GPU not detected

Check that ROCm is properly installed and `rocm-smi` is accessible.
