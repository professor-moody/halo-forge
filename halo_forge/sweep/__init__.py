"""Hyperparameter sweep infrastructure (Track T10).

Sweeps three things at once:

  1. **Search space**: per-knob distributions (log-uniform LR, choice
     of LoRA rank, integer batch size). Defines the universe.
  2. **Sampler**: random / TPE / grid. Picks the next trial config.
  3. **Pruner**: ASHA-style early stopping that halts under-performing
     trials before they finish so the budget concentrates on
     promising regions.

Halo-forge owns the integration (config templating, trial running,
result aggregation, CLI). Optuna is the optional brain — when it's
installed we route sampling through it; otherwise we fall back to
random sampling so the surface keeps working without the dep.

Sweep results are JSONL-shaped per-trial so the upcoming F-P sweep
dashboard can stream them as they complete instead of waiting for
the full run to finish.
"""

from halo_forge.sweep.config import (
    Choice,
    LogUniform,
    SearchSpace,
    SweepConfig,
    Uniform,
)
from halo_forge.sweep.runner import (
    SweepResult,
    TrialResult,
    run_sweep,
)

__all__ = [
    "Choice",
    "LogUniform",
    "SearchSpace",
    "SweepConfig",
    "SweepResult",
    "TrialResult",
    "Uniform",
    "run_sweep",
]
