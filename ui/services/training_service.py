"""
Training Service

Manages training job execution via subprocess, log streaming, and job control.
This is the bridge between the UI and actual training processes.
"""

import asyncio
import os
import re
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any
from collections import deque

from .metrics_parser import MetricsParser, ParsedMetrics
from .event_bus import (
    get_event_bus,
    Event,
    EventType,
    build_transition_payload,
)
from .launch_contracts import (
    SFT_LAUNCH_CONTRACT,
    RAFT_LAUNCH_CONTRACT,
    MODALITY_TRAIN_LAUNCH_CONTRACTS,
    validate_launch_payload,
    ensure_local_path_exists_if_pathlike,
)
from .launch_context import (
    CYCLE_BASED_TRAINING_JOB_TYPES,
    persist_launch_context,
    read_launch_context,
)
from halo_forge.capabilities import check_modality_train_capability
from halo_forge.utils.macos_runtime import caffeinate_command

# Import notification helpers (only used when UI is running)
try:
    from ui.components.notifications import notify_checkpoint_saved
    HAS_UI_NOTIFICATIONS = True
except ImportError:
    HAS_UI_NOTIFICATIONS = False


@dataclass
class TrainingMetrics:
    """Current training metrics."""
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: float = 0.0
    step: int = 0
    total_steps: int = 0
    cycle: int = 0  # RAFT
    total_cycles: int = 0  # RAFT
    compile_rate: Optional[float] = None  # RAFT
    grad_norm: Optional[float] = None


@dataclass
class TrainingLaunchPreflight:
    """Structured preflight result for training launch preparation."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    resolved_paths: dict[str, str] = field(default_factory=dict)
    suggested_fixes: list[str] = field(default_factory=list)
    quality_outlook: dict[str, Any] = field(default_factory=dict)


class TrainingService:
    """
    Service for launching and managing training processes.
    
    This service:
    - Spawns training as subprocess with proper env vars
    - Streams stdout/stderr and parses metrics
    - Updates job state in real-time
    - Handles graceful and forced termination
    
    Usage:
        from ui.state import state
        service = TrainingService(state)
        
        job_id = await service.launch_sft(
            model="Qwen/Qwen2.5-Coder-3B-Instruct",
            dataset="humaneval",
            output_dir="./outputs/sft",
            epochs=3,
        )
        
        # Later...
        await service.stop_job(job_id)
    """
    
    def __init__(self, state):
        """
        Initialize training service.
        
        Args:
            state: AppState instance for job tracking
        """
        self.state = state
        self._parsers: dict[str, MetricsParser] = {}
        self._log_buffers: dict[str, deque] = {}
        self._callbacks: dict[str, list[Callable]] = {}
        self._last_progress_line_by_job: dict[str, str] = {}

    @staticmethod
    def _normalize_stream_chunk(raw_text: str) -> list[str]:
        """Collapse carriage-return progress redraws into stable logical lines."""
        normalized_lines: list[str] = []
        for newline_segment in raw_text.replace("\r\n", "\n").split("\n"):
            if not newline_segment:
                continue
            line = newline_segment
            if "\r" in line:
                redraw_segments = [segment.strip() for segment in line.split("\r") if segment.strip()]
                if not redraw_segments:
                    continue
                line = redraw_segments[-1]
            line = line.strip()
            if line:
                normalized_lines.append(line)
        return normalized_lines

    @staticmethod
    def _is_progress_only_line(line: str) -> bool:
        """Return True for bare tqdm-style progress lines without extra signal."""
        return bool(
            MetricsParser.PATTERNS["tqdm_progress"].search(line)
            and not re.search(r"\{.*\}|loss|lr|learning_rate|epoch|cycle|HALO_YIELD", line, re.IGNORECASE)
        )

    def _should_skip_stream_line(self, job_id: str, line: str) -> bool:
        """Drop repeated progress redraws while preserving normal logs."""
        if not self._is_progress_only_line(line):
            self._last_progress_line_by_job.pop(job_id, None)
            return False
        if self._last_progress_line_by_job.get(job_id) == line:
            return True
        self._last_progress_line_by_job[job_id] = line
        return False
    
    def _get_strix_halo_env(self) -> dict[str, str]:
        """Get environment variables optimized for AMD Strix Halo."""
        env = os.environ.copy()
        
        # GPU architecture
        env.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.5.1')
        env.setdefault('PYTORCH_ROCM_ARCH', 'gfx1151')
        env.setdefault('HIP_VISIBLE_DEVICES', '0')
        
        # Memory management for unified memory
        env.setdefault(
            'PYTORCH_HIP_ALLOC_CONF',
            'backend:native,expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512'
        )
        
        # Stability settings
        env.setdefault('HSA_ENABLE_SDMA', '0')
        
        # Dataloader settings (critical for unified memory)
        env.setdefault('OMP_NUM_THREADS', '1')
        
        return env

    def _validate_sft_launch_payload(
        self,
        model: str,
        dataset: str,
        output_dir: str,
        epochs: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        max_samples: Optional[int],
    ) -> tuple[str, str, str]:
        """Validate user inputs before creating an SFT job."""
        normalized = validate_launch_payload(
            {
                "model": model,
                "dataset": dataset,
                "output_dir": output_dir,
                "epochs": epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            },
            SFT_LAUNCH_CONTRACT,
        )
        model = normalized["model"]
        dataset = normalized["dataset"]
        output_dir = normalized["output_dir"]
        ensure_local_path_exists_if_pathlike(dataset, "dataset")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be greater than 0 when provided")
        return model, dataset, output_dir

    def _validate_raft_launch_payload(
        self,
        model: str,
        prompts: str,
        output_dir: str,
        cycles: int,
        samples_per_prompt: int,
        keep_percent: float,
        reward_threshold: float,
        min_samples: int,
        max_new_tokens: int,
        checkpoint: Optional[str] = None,
    ) -> tuple[str, str, str]:
        """Validate user inputs before creating a RAFT job."""
        normalized = validate_launch_payload(
            {
                "model": model,
                "prompts": prompts,
                "output_dir": output_dir,
                "cycles": cycles,
                "samples_per_prompt": samples_per_prompt,
                "keep_percent": keep_percent,
                "reward_threshold": reward_threshold,
                "min_samples": min_samples,
                "max_new_tokens": max_new_tokens,
                "checkpoint": checkpoint,
            },
            RAFT_LAUNCH_CONTRACT,
        )
        model = normalized["model"]
        prompts = normalized["prompts"]
        output_dir = normalized["output_dir"]
        return model, prompts, output_dir

    def _validate_modality_launch_payload(
        self,
        *,
        modality: str,
        model: str,
        dataset: str,
        output_dir: str,
        cycles: int,
        resume_from_cycle: int = 0,
        seed: int = 42,
        limit: Optional[int] = None,
        task: Optional[str] = None,
        samples_per_prompt: Optional[int] = None,
    ) -> tuple[str, str, str, int]:
        """Validate user inputs for modality-specific train launches."""
        contract = MODALITY_TRAIN_LAUNCH_CONTRACTS[modality]
        if modality == "vlm" and not samples_per_prompt:
            samples_per_prompt = 4
        payload = {
            "model": model,
            "dataset": dataset,
            "output_dir": output_dir,
            "cycles": cycles,
            "resume_from_cycle": resume_from_cycle,
            "seed": seed,
            "limit": limit,
            "task": task,
            "samples_per_prompt": samples_per_prompt,
        }
        normalized = validate_launch_payload(payload, contract)
        if resume_from_cycle < 0:
            raise ValueError("resume_from_cycle must be >= 0")
        try:
            parsed_seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError("seed must be >= 0")
        if parsed_seed < 0:
            raise ValueError("seed must be >= 0")
        return (
            normalized["model"],
            normalized["dataset"],
            normalized["output_dir"],
            parsed_seed,
        )

    def _build_lifecycle_metadata(
        self,
        *,
        origin_job_id: Optional[str],
        relaunch: bool,
        launch_context_file: Optional[Path],
        resume_strategy: Optional[str],
        guided_recovery: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if origin_job_id:
            metadata["origin_job_id"] = origin_job_id
        if relaunch:
            metadata["relaunch"] = True
        if launch_context_file:
            metadata["launch_context_file"] = str(launch_context_file)
        if resume_strategy:
            metadata["resume_strategy"] = resume_strategy
        if guided_recovery:
            metadata["guided_recovery"] = dict(guided_recovery)
        return metadata

    def _build_launch_context_metadata(
        self,
        guided_recovery: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {"guided_recovery": dict(guided_recovery or {})}
        system_info = self._system_info_metadata()
        if system_info:
            metadata["system_info"] = system_info
        return metadata

    @staticmethod
    def _system_info_metadata() -> dict[str, Any]:
        data: dict[str, Any] = {}
        try:
            from halo_forge.backend import get_backend

            backend = get_backend()
            data["backend"] = backend.name
            data["supports_neural_accelerators"] = bool(
                getattr(backend.capabilities, "supports_neural_accelerators", False)
            )
        except Exception:
            pass
        try:
            from halo_forge.telemetry.apple_silicon import AppleSiliconTelemetry

            chip = AppleSiliconTelemetry._detect_chip_info(
                AppleSiliconTelemetry._detect_device_name()
            )
            if chip is not None:
                data["chip"] = chip.to_dict()
        except Exception:
            pass
        return data

    def _apply_launch_overrides(
        self,
        args: dict[str, Any],
        override_args: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        resolved = dict(args)
        for key, value in (override_args or {}).items():
            if value is None:
                continue
            resolved[key] = value
        return resolved

    def _preflight_output_dir(
        self,
        output_dir: str,
        *,
        resolved_paths: dict[str, str],
        warnings: list[str],
        errors: list[str],
        suggested_fixes: list[str],
    ) -> None:
        """Validate output directory shape and writability for launch preflight."""
        raw = str(output_dir or "").strip()
        if not raw:
            errors.append("output_dir is required")
            suggested_fixes.append("Set an output directory before launch.")
            return

        output_path = Path(raw).expanduser()
        resolved_paths["output_dir"] = str(output_path)

        if output_path.exists():
            if not output_path.is_dir():
                errors.append(f"output_dir exists but is not a directory: {output_path}")
                suggested_fixes.append("Choose a directory path for output_dir.")
                return
            if not os.access(output_path, os.W_OK):
                errors.append(f"output_dir is not writable: {output_path}")
                suggested_fixes.append("Grant write permissions on output_dir or choose a writable location.")
            return

        parent = output_path.parent if str(output_path.parent) else Path(".")
        if parent.exists():
            if not parent.is_dir():
                errors.append(f"output_dir parent is not a directory: {parent}")
                suggested_fixes.append("Choose an output_dir with a valid parent directory.")
                return
            if not os.access(parent, os.W_OK):
                errors.append(f"output_dir parent is not writable: {parent}")
                suggested_fixes.append("Grant write permissions on the output parent directory.")
                return
            warnings.append(f"output_dir does not exist yet: {output_path}")
            suggested_fixes.append("Use 'Create output scaffold' or launch to create output_dir.")
            return

        ancestor = parent
        while not ancestor.exists() and ancestor.parent != ancestor:
            ancestor = ancestor.parent
        if not ancestor.exists() or not os.access(ancestor, os.W_OK):
            errors.append(f"output_dir cannot be created from current permissions: {output_path}")
            suggested_fixes.append("Choose an output_dir under a writable path.")
            return

        warnings.append(f"output_dir parent does not exist yet and will be created: {parent}")
        suggested_fixes.append("Use 'Create output scaffold' to pre-create output directories.")

    def expected_output_artifacts(self, mode_key: str) -> list[str]:
        """Return canonical artifact expectations for a training mode."""
        key = str(mode_key or "").strip().lower()
        if key == "sft":
            return [
                "training_summary.json",
                "final_model/",
                "launch_context.json",
                "<job_id>_training.log",
            ]
        return [
            "training_summary.json",
            "latest_checkpoint.json",
            "cycle_<n>/model/",
            "final_model/",
            "launch_context.json",
            "<job_id>_training.log",
        ]

    def scaffold_output_dir(self, output_dir: str, *, mode_key: str) -> Path:
        """Create a minimal output scaffold for first-run launch success."""
        output_path = Path(str(output_dir or "").strip()).expanduser()
        if not str(output_path):
            raise ValueError("output_dir is required")

        output_path.mkdir(parents=True, exist_ok=True)
        marker_path = output_path / ".halo_forge_output_scaffold.json"
        if not marker_path.exists():
            import json

            marker = {
                "created_at": datetime.now().isoformat(),
                "mode": str(mode_key or "").strip().lower(),
                "expected_artifacts": self.expected_output_artifacts(mode_key),
            }
            marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
        return output_path

    def _count_jsonl_rows(self, path_value: str) -> Optional[int]:
        path = Path(str(path_value or "").strip()).expanduser()
        if not path.exists() or not path.is_file():
            return None
        if path.suffix.lower() != ".jsonl":
            return None
        count = 0
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        count += 1
        except Exception:
            return None
        return count

    def _read_previous_quality_status(self, output_dir: str) -> Optional[str]:
        candidate = Path(str(output_dir or "").strip()).expanduser() / "training_summary.json"
        if not candidate.exists():
            return None
        try:
            import json

            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            return None
        yield_diagnostics = (
            payload.get("yield_diagnostics")
            if isinstance(payload, dict)
            else None
        )
        if not isinstance(yield_diagnostics, dict):
            return None
        summary = yield_diagnostics.get("summary")
        if not isinstance(summary, dict):
            return None
        status = str(summary.get("status") or "").strip().lower()
        return status or None

    def _build_quality_outlook(
        self,
        *,
        mode_key: str,
        output_dir: str,
        warnings: list[str],
        suggestions: list[str],
        artifact_notes: list[str],
    ) -> dict[str, Any]:
        status = "healthy"
        if (
            len(warnings) >= 2
            or any(
                marker in warning.lower()
                for warning in warnings
                for marker in ("low signal", "too small", "starve", "aggressive")
            )
        ):
            status = "low_yield"
        elif warnings:
            status = "caution"
        summary = {
            "healthy": "Current settings should produce enough signal for a first useful run.",
            "caution": "This run may complete, but some settings could reduce usable training signal.",
            "low_yield": "This run is at risk of producing very little usable training signal.",
        }[status]
        return {
            "status": status,
            "summary": summary,
            "warnings": warnings[:3],
            "suggested_adjustments": suggestions[:3],
            "artifact_notes": artifact_notes,
            "yield_safety_note": (
                "Balanced defaults for first useful updates."
                if status == "healthy"
                else (
                    "Watch sample budget and thresholds to avoid starving updates."
                    if mode_key != "sft"
                    else "Use enough examples to avoid a no-signal SFT run."
                )
            ),
        }

    def _quality_outlook_for_sft(
        self,
        *,
        output_dir: str,
        epochs: int,
        batch_size: int,
        max_samples: Optional[int],
    ) -> dict[str, Any]:
        warnings: list[str] = []
        suggestions: list[str] = []
        if max_samples is not None and max_samples < 32:
            warnings.append("Sample budget is very small for SFT and may produce low signal.")
            suggestions.append("Increase max_samples to at least 32-64 for a more representative update.")
        if max_samples is not None and max_samples < 64 and epochs > 1:
            warnings.append("Tiny dataset limits with multiple epochs may overfit before learning stabilizes.")
            suggestions.append("Use one epoch for smoke runs or raise max_samples before repeating epochs.")
        if (
            max_samples is not None
            and max_samples > 0
            and batch_size * max(1, epochs) > max_samples
        ):
            suggestions.append("If loss looks noisy, increase max_samples or lower epochs before relaunch.")
        previous_status = self._read_previous_quality_status(output_dir)
        if previous_status in {"low_yield", "no_signal"}:
            warnings.append("This output directory previously ended with low signal.")
            suggestions.append("Use a fresh output directory or raise sample supply before resuming this run.")
        return self._build_quality_outlook(
            mode_key="sft",
            output_dir=output_dir,
            warnings=warnings,
            suggestions=suggestions,
            artifact_notes=self.expected_output_artifacts("sft"),
        )

    def _quality_outlook_for_raft(
        self,
        *,
        prompts: str,
        output_dir: str,
        cycles: int,
        samples_per_prompt: int,
        keep_percent: float,
        reward_threshold: float,
        min_samples: int,
    ) -> dict[str, Any]:
        warnings: list[str] = []
        suggestions: list[str] = []
        prompt_count = self._count_jsonl_rows(prompts)
        sample_budget = (prompt_count or 0) * max(1, samples_per_prompt)
        if prompt_count and sample_budget < 32:
            warnings.append("Prompt/sample budget is small for RAFT and may not produce enough kept examples.")
            suggestions.append("Increase prompts or samples_per_prompt so each cycle sees at least 32 candidates.")
        if keep_percent <= 0.2:
            warnings.append("Keep percent is very selective and may starve updates.")
            suggestions.append("Raise keep percent closer to 0.4-0.6 for exploratory runs.")
        if reward_threshold >= 0.85:
            warnings.append("Reward threshold is aggressive and may drop nearly all verified samples.")
            suggestions.append("Lower reward_threshold for first-pass runs, then tighten once yield is stable.")
        if prompt_count and min_samples > sample_budget:
            warnings.append("min_samples exceeds the likely sample supply for a single cycle.")
            suggestions.append("Lower min_samples or increase prompts/sample budget so the floor is reachable.")
        if prompt_count and cycles > 1 and sample_budget < max(min_samples, 24):
            warnings.append("Multi-cycle training with a tiny sample pool is likely to recycle weak signal.")
            suggestions.append("Increase prompt variety before running multiple cycles.")
        previous_status = self._read_previous_quality_status(output_dir)
        if previous_status in {"low_yield", "no_signal"}:
            warnings.append("The prior run in this output directory ended with low signal.")
            suggestions.append("Adjust thresholds or sample supply before resuming the same run directory.")
        return self._build_quality_outlook(
            mode_key="raft",
            output_dir=output_dir,
            warnings=warnings,
            suggestions=suggestions,
            artifact_notes=self.expected_output_artifacts("raft"),
        )

    def _quality_outlook_for_modality(
        self,
        *,
        modality: str,
        output_dir: str,
        cycles: int,
        limit: Optional[int],
        samples_per_prompt: Optional[int],
        keep_percent: Optional[float],
        reward_threshold: Optional[float],
        resume_from_cycle: int,
    ) -> dict[str, Any]:
        warnings: list[str] = []
        suggestions: list[str] = []
        effective_limit = int(limit) if limit is not None else 0
        spp = int(samples_per_prompt) if samples_per_prompt is not None else 1
        cycle_supply = effective_limit * max(1, spp) if effective_limit > 0 else 0
        if effective_limit and cycle_supply < 24:
            warnings.append("Sample budget is small for this modality and may produce low signal.")
            suggestions.append("Raise the dataset limit or samples-per-prompt before long runs.")
        if keep_percent is not None and keep_percent <= 0.2:
            warnings.append("Keep percent is very selective and may leave too little supervision.")
            suggestions.append("Increase keep percent for first runs, then tighten after yield stabilizes.")
        if reward_threshold is not None and reward_threshold >= 0.85:
            warnings.append("Reward threshold is aggressive for a first run.")
            suggestions.append("Lower the threshold slightly to confirm the verifier and data pipeline first.")
        if effective_limit and cycles > 1 and cycle_supply < 16:
            warnings.append("Tiny dataset limits paired with multiple cycles can recycle weak signal.")
            suggestions.append("Use more source samples or fewer cycles for the first pass.")
        if resume_from_cycle > 0:
            previous_status = self._read_previous_quality_status(output_dir)
            if previous_status in {"low_yield", "no_signal"}:
                warnings.append("Resume target previously produced low signal.")
                suggestions.append("Consider adjusting thresholds or using a fresh output directory before resuming.")
        return self._build_quality_outlook(
            mode_key=modality,
            output_dir=output_dir,
            warnings=warnings,
            suggestions=suggestions,
            artifact_notes=self.expected_output_artifacts(modality),
        )

    def preflight_sft_launch(
        self,
        *,
        model: str,
        dataset: str,
        output_dir: str,
        epochs: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        max_samples: Optional[int] = None,
    ) -> TrainingLaunchPreflight:
        """Run structured preflight checks for SFT launch."""
        errors: list[str] = []
        warnings: list[str] = []
        resolved_paths: dict[str, str] = {}
        suggested_fixes: list[str] = []

        try:
            model, dataset, output_dir = self._validate_sft_launch_payload(
                model=model,
                dataset=dataset,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_samples=max_samples,
            )
        except ValueError as e:
            errors.append(str(e))
            return TrainingLaunchPreflight(
                ok=False,
                errors=errors,
                warnings=warnings,
                resolved_paths=resolved_paths,
                suggested_fixes=["Fix required inputs before launch."],
                quality_outlook={},
            )

        resolved_paths["model"] = model
        resolved_paths["dataset"] = dataset
        if Path(dataset).expanduser().exists():
            resolved_paths["dataset"] = str(Path(dataset).expanduser())
        self._preflight_output_dir(
            output_dir,
            resolved_paths=resolved_paths,
            warnings=warnings,
            errors=errors,
            suggested_fixes=suggested_fixes,
        )
        return TrainingLaunchPreflight(
            ok=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            resolved_paths=resolved_paths,
            suggested_fixes=suggested_fixes,
            quality_outlook=self._quality_outlook_for_sft(
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                max_samples=max_samples,
            ),
        )

    def preflight_raft_launch(
        self,
        *,
        model: str,
        prompts: str,
        output_dir: str,
        cycles: int,
        samples_per_prompt: int,
        keep_percent: float,
        reward_threshold: float,
        min_samples: int,
        max_new_tokens: int,
        checkpoint: Optional[str] = None,
    ) -> TrainingLaunchPreflight:
        """Run structured preflight checks for RAFT launch."""
        errors: list[str] = []
        warnings: list[str] = []
        resolved_paths: dict[str, str] = {}
        suggested_fixes: list[str] = []

        try:
            model, prompts, output_dir = self._validate_raft_launch_payload(
                model=model,
                prompts=prompts,
                output_dir=output_dir,
                cycles=cycles,
                samples_per_prompt=samples_per_prompt,
                keep_percent=keep_percent,
                reward_threshold=reward_threshold,
                min_samples=min_samples,
                max_new_tokens=max_new_tokens,
                checkpoint=checkpoint,
            )
        except ValueError as e:
            errors.append(str(e))
            return TrainingLaunchPreflight(
                ok=False,
                errors=errors,
                warnings=warnings,
                resolved_paths=resolved_paths,
                suggested_fixes=["Fix required inputs before launch."],
                quality_outlook={},
            )

        resolved_paths["model"] = model
        resolved_paths["prompts"] = str(Path(prompts).expanduser())
        if checkpoint:
            resolved_paths["checkpoint"] = str(Path(checkpoint).expanduser())
        self._preflight_output_dir(
            output_dir,
            resolved_paths=resolved_paths,
            warnings=warnings,
            errors=errors,
            suggested_fixes=suggested_fixes,
        )
        return TrainingLaunchPreflight(
            ok=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            resolved_paths=resolved_paths,
            suggested_fixes=suggested_fixes,
            quality_outlook=self._quality_outlook_for_raft(
                prompts=prompts,
                output_dir=output_dir,
                cycles=cycles,
                samples_per_prompt=samples_per_prompt,
                keep_percent=keep_percent,
                reward_threshold=reward_threshold,
                min_samples=min_samples,
            ),
        )

    def preflight_modality_train_launch(
        self,
        *,
        modality: str,
        model: str,
        dataset: str,
        output_dir: str,
        cycles: int,
        resume_from_cycle: int = 0,
        seed: int = 42,
        allow_prototype_train: bool = False,
        limit: Optional[int] = None,
        task: Optional[str] = None,
        samples_per_prompt: Optional[int] = None,
        keep_percent: Optional[float] = None,
        reward_threshold: Optional[float] = None,
    ) -> TrainingLaunchPreflight:
        """Run structured preflight checks for modality training launch."""
        errors: list[str] = []
        warnings: list[str] = []
        resolved_paths: dict[str, str] = {}
        suggested_fixes: list[str] = []

        try:
            model, dataset, output_dir, seed = self._validate_modality_launch_payload(
                modality=modality,
                model=model,
                dataset=dataset,
                output_dir=output_dir,
                cycles=cycles,
                resume_from_cycle=resume_from_cycle,
                seed=seed,
                limit=limit,
                task=task,
                samples_per_prompt=samples_per_prompt,
            )
        except ValueError as e:
            errors.append(str(e))
            return TrainingLaunchPreflight(
                ok=False,
                errors=errors,
                warnings=warnings,
                resolved_paths=resolved_paths,
                suggested_fixes=["Fix required inputs before launch."],
                quality_outlook={},
            )

        capability = check_modality_train_capability(
            modality=modality,
            model_name=model,
            allow_prototype_train=allow_prototype_train,
            dry_run=False,
        )
        if not capability.allowed:
            errors.append(capability.message.splitlines()[-1])
            suggested_fixes.append(
                "Use a supported model family or enable prototype override if the capability is still gated."
            )

        resolved_paths["model"] = model
        resolved_paths["dataset"] = dataset
        resolved_paths["seed"] = str(seed)
        self._preflight_output_dir(
            output_dir,
            resolved_paths=resolved_paths,
            warnings=warnings,
            errors=errors,
            suggested_fixes=suggested_fixes,
        )
        return TrainingLaunchPreflight(
            ok=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            resolved_paths=resolved_paths,
            suggested_fixes=suggested_fixes,
            quality_outlook=self._quality_outlook_for_modality(
                modality=modality,
                output_dir=output_dir,
                cycles=cycles,
                limit=limit,
                samples_per_prompt=samples_per_prompt,
                keep_percent=keep_percent,
                reward_threshold=reward_threshold,
                resume_from_cycle=resume_from_cycle,
            ),
        )

    def _event_extra_fields(self, job) -> dict[str, Any]:
        if not job or not job.lifecycle_metadata:
            return {}
        return {
            "origin_job_id": job.lifecycle_metadata.get("origin_job_id"),
            "relaunch": job.lifecycle_metadata.get("relaunch"),
            "launch_context_file": job.lifecycle_metadata.get("launch_context_file"),
            "resume_strategy": job.lifecycle_metadata.get("resume_strategy"),
        }

    def _merge_transition_metadata(self, job, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if job and job.lifecycle_metadata:
            metadata.update(job.lifecycle_metadata)
        if extra:
            metadata.update(extra)
        return metadata

    def resolve_resume_latest_cycle(self, output_dir: str | Path) -> int:
        """
        Resolve next resume cycle from latest checkpoint metadata.

        Returns:
            resume_from_cycle value (checkpoint cycle + 1)
        """
        output_path = Path(output_dir)
        latest_path = output_path / "latest_checkpoint.json"
        if not latest_path.exists():
            raise ValueError(
                f"latest checkpoint metadata not found: {latest_path}"
            )

        import json

        try:
            with latest_path.open(encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            raise ValueError(f"failed to parse {latest_path}: {e}") from e

        cycle_value = payload.get("cycle")
        try:
            cycle = int(cycle_value)
        except (TypeError, ValueError):
            raise ValueError(f"latest checkpoint cycle is invalid: {cycle_value}")
        if cycle < 0:
            raise ValueError(f"latest checkpoint cycle must be >= 0: {cycle}")

        return cycle + 1
    
    async def launch_sft(
        self,
        model: str,
        dataset: str,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 16,
        learning_rate: float = 2e-4,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.01,
        max_grad_norm: float = 0.3,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        max_seq_length: int = 2048,
        validation_split: float = 0.05,
        max_samples: Optional[int] = None,
        save_steps: int = 500,
        eval_steps: int = 250,
        early_stopping_patience: int = 5,
        gradient_checkpointing: bool = True,
        on_log: Optional[Callable[[str], None]] = None,
        source_ui_page: str = "/training",
        origin_job_id: Optional[str] = None,
        relaunch: bool = False,
        resume_strategy: Optional[str] = None,
        guided_recovery: Optional[dict[str, Any]] = None,
        no_caffeinate: bool = False,
        **kwargs
    ) -> str:
        """
        Launch SFT training as subprocess.
        
        Args:
            model: Model name or path
            dataset: Dataset name or path
            output_dir: Output directory for checkpoints
            epochs: Number of training epochs
            batch_size: Per-device batch size
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            warmup_ratio: Warmup ratio for LR scheduler
            weight_decay: Weight decay for regularization
            max_grad_norm: Max gradient norm for clipping
            use_lora: Whether to use LoRA
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            max_seq_length: Maximum sequence length
            validation_split: Validation set fraction
            max_samples: Limit training samples (None = all)
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            early_stopping_patience: Stop if no improvement for N evals
            on_log: Optional callback for log lines
            **kwargs: Additional CLI arguments
            
        Returns:
            Job ID
        """
        model, dataset, output_dir = self._validate_sft_launch_payload(
            model=model,
            dataset=dataset,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_samples=max_samples,
        )

        # Create job in state
        job = self.state.create_job(
            job_type="sft",
            name=f"SFT: {Path(model).name} on {dataset}",
            output_dir=Path(output_dir),
        )
        job.total_epochs = epochs
        
        # Build command
        cmd = [
            sys.executable, "-m", "halo_forge.cli", "sft", "train",
            "--model", model,
        ]
        
        # Route dataset: use --data for local files, --dataset for HuggingFace IDs
        # Check if it's an actual local file (HF IDs like "user/dataset" also contain "/")
        if dataset and Path(dataset).exists():
            cmd.extend(["--data", dataset])
        else:
            cmd.extend(["--dataset", dataset])
        
        cmd.extend([
            "--output", output_dir,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--gradient-accumulation", str(gradient_accumulation_steps),
            "--learning-rate", str(learning_rate),
            "--warmup-ratio", str(warmup_ratio),
            "--weight-decay", str(weight_decay),
            "--max-grad-norm", str(max_grad_norm),
            "--max-seq-length", str(max_seq_length),
            "--validation-split", str(validation_split),
            "--save-steps", str(save_steps),
            "--eval-steps", str(eval_steps),
            "--early-stopping-patience", str(early_stopping_patience),
        ])
        
        # LoRA options
        if use_lora:
            cmd.extend([
                "--lora-rank", str(lora_rank),
                "--lora-alpha", str(lora_alpha),
                "--lora-dropout", str(lora_dropout),
            ])
        else:
            cmd.append("--no-lora")
        
        # Optional max samples limit
        if max_samples is not None and max_samples > 0:
            cmd.extend(["--max-samples", str(max_samples)])
        
        # Hardware options
        if not gradient_checkpointing:
            cmd.append("--no-gradient-checkpointing")
        if no_caffeinate:
            cmd.append("--no-caffeinate")
        
        # Add any extra arguments
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        launch_args = {
            "model": model,
            "dataset": dataset,
            "output_dir": output_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
            "use_lora": use_lora,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "max_seq_length": max_seq_length,
            "validation_split": validation_split,
            "max_samples": max_samples,
            "save_steps": save_steps,
            "eval_steps": eval_steps,
            "early_stopping_patience": early_stopping_patience,
            "gradient_checkpointing": gradient_checkpointing,
            "no_caffeinate": no_caffeinate,
        }
        launch_args.update({k: v for k, v in kwargs.items() if v is not None})
        launch_context_file = None
        try:
            launch_context_file = persist_launch_context(
                output_dir=output_dir,
                job_type="sft",
                service="training",
                source_ui_page=source_ui_page,
                command=cmd,
                args=launch_args,
                relaunch_capabilities={
                    "can_relaunch": True,
                    "can_clone": True,
                    "can_resume_latest": False,
                },
                metadata=self._build_launch_context_metadata(guided_recovery),
            )
        except Exception as e:
            print(f"[TrainingService] Failed to persist launch context: {e}")
        lifecycle_metadata = self._build_lifecycle_metadata(
            origin_job_id=origin_job_id,
            relaunch=relaunch,
            launch_context_file=launch_context_file,
            resume_strategy=resume_strategy,
            guided_recovery=guided_recovery,
        )
        job.launch_context_file = launch_context_file
        job.launch_args = launch_args
        job.lifecycle_metadata = lifecycle_metadata

        # Emit job created event
        created_transition = {
            "from_status": None,
            "to_status": "pending",
            "applied": True,
            "source": "training_service.launch_sft",
            "reason": "job_created",
            "timestamp": datetime.now().isoformat(),
            "metadata": self._merge_transition_metadata(
                job,
                {"job_type": "sft"},
            ),
        }
        get_event_bus().emit_sync(Event(
            type=EventType.JOB_CREATED,
            job_id=job.id,
            data=build_transition_payload(
                created_transition,
                name=job.name,
                type="sft",
                **self._event_extra_fields(job),
            ),
        ))
        
        # Launch subprocess
        await self._launch_process_with_runtime_options(
            job.id,
            cmd,
            on_log,
            no_caffeinate=no_caffeinate,
        )
        
        return job.id
    
    async def launch_raft(
        self,
        model: str,
        prompts: str,
        output_dir: str,
        verifier: str = "execution",
        cycles: int = 5,
        samples_per_prompt: int = 8,
        temperature: float = 0.7,
        keep_percent: float = 0.5,
        reward_threshold: float = 0.5,
        min_samples: int = 64,
        max_new_tokens: int = 1024,
        lr_decay: float = 0.85,
        min_lr: float = 1e-6,
        checkpoint: Optional[str] = None,
        curriculum: str = "none",
        curriculum_stats: Optional[str] = None,
        curriculum_start: float = 0.2,
        curriculum_increment: float = 0.2,
        reward_shaping: str = "fixed",
        system_prompt: str = "You are an expert programmer.",
        experimental_attention: bool = False,
        on_log: Optional[Callable[[str], None]] = None,
        source_ui_page: str = "/training",
        origin_job_id: Optional[str] = None,
        relaunch: bool = False,
        resume_strategy: Optional[str] = None,
        guided_recovery: Optional[dict[str, Any]] = None,
        no_caffeinate: bool = False,
        **kwargs
    ) -> str:
        """
        Launch RAFT training as subprocess.
        
        Args:
            model: Model name or path
            prompts: Path to prompts file
            output_dir: Output directory
            verifier: Verifier type
            cycles: Number of RAFT cycles
            samples_per_prompt: Samples per prompt
            temperature: Sampling temperature
            keep_percent: Percentage of samples to keep
            reward_threshold: Minimum reward threshold
            min_samples: Minimum samples per cycle
            max_new_tokens: Maximum tokens to generate
            lr_decay: Learning rate decay per cycle
            min_lr: Minimum learning rate floor
            checkpoint: Optional SFT checkpoint path
            curriculum: Curriculum learning strategy
            reward_shaping: Reward shaping strategy
            system_prompt: System prompt for generation
            experimental_attention: Enable experimental ROCm attention
            on_log: Optional callback for log lines
            **kwargs: Additional CLI arguments
            
        Returns:
            Job ID
        """
        model, prompts, output_dir = self._validate_raft_launch_payload(
            model=model,
            prompts=prompts,
            output_dir=output_dir,
            cycles=cycles,
            samples_per_prompt=samples_per_prompt,
            keep_percent=keep_percent,
            reward_threshold=reward_threshold,
            min_samples=min_samples,
            max_new_tokens=max_new_tokens,
            checkpoint=checkpoint,
        )
        if curriculum == "historical":
            ensure_local_path_exists_if_pathlike(curriculum_stats, "curriculum_stats")

        # Create job in state
        job = self.state.create_job(
            job_type="raft",
            name=f"RAFT: {Path(model).name}",
            output_dir=Path(output_dir),
        )
        job.total_cycles = cycles
        
        # Build command
        cmd = [
            sys.executable, "-m", "halo_forge.cli", "raft", "train",
            "--model", model,
            "--prompts", prompts,
            "--output", output_dir,
            "--verifier", verifier,
            "--cycles", str(cycles),
            "--samples-per-prompt", str(samples_per_prompt),
            "--temperature", str(temperature),
            "--keep-percent", str(keep_percent),
            "--reward-threshold", str(reward_threshold),
            "--min-samples", str(min_samples),
            "--max-new-tokens", str(max_new_tokens),
            "--lr-decay", str(lr_decay),
            "--min-lr", str(min_lr),
            "--curriculum", curriculum,
            "--reward-shaping", reward_shaping,
            "--system-prompt", system_prompt,
        ]
        
        # Curriculum-specific options
        if curriculum == "historical" and curriculum_stats:
            cmd.extend(["--curriculum-stats", curriculum_stats])
        elif curriculum == "progressive":
            cmd.extend([
                "--curriculum-start", str(curriculum_start),
                "--curriculum-increment", str(curriculum_increment),
            ])
        
        # Optional checkpoint for resume
        if checkpoint:
            cmd.extend(["--checkpoint", checkpoint])
        
        # Experimental attention flag
        if experimental_attention:
            cmd.append("--experimental-attention")
        if no_caffeinate:
            cmd.append("--no-caffeinate")
        
        # Add any extra arguments
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        launch_args = {
            "model": model,
            "prompts": prompts,
            "output_dir": output_dir,
            "verifier": verifier,
            "cycles": cycles,
            "samples_per_prompt": samples_per_prompt,
            "temperature": temperature,
            "keep_percent": keep_percent,
            "reward_threshold": reward_threshold,
            "min_samples": min_samples,
            "max_new_tokens": max_new_tokens,
            "lr_decay": lr_decay,
            "min_lr": min_lr,
            "checkpoint": checkpoint,
            "curriculum": curriculum,
            "curriculum_stats": curriculum_stats,
            "curriculum_start": curriculum_start,
            "curriculum_increment": curriculum_increment,
            "reward_shaping": reward_shaping,
            "system_prompt": system_prompt,
            "experimental_attention": experimental_attention,
            "no_caffeinate": no_caffeinate,
        }
        launch_args.update({k: v for k, v in kwargs.items() if v is not None})
        launch_context_file = None
        try:
            launch_context_file = persist_launch_context(
                output_dir=output_dir,
                job_type="raft",
                service="training",
                source_ui_page=source_ui_page,
                command=cmd,
                args=launch_args,
                relaunch_capabilities={
                    "can_relaunch": True,
                    "can_clone": True,
                    "can_resume_latest": True,
                },
                metadata=self._build_launch_context_metadata(guided_recovery),
            )
        except Exception as e:
            print(f"[TrainingService] Failed to persist launch context: {e}")
        lifecycle_metadata = self._build_lifecycle_metadata(
            origin_job_id=origin_job_id,
            relaunch=relaunch,
            launch_context_file=launch_context_file,
            resume_strategy=resume_strategy,
            guided_recovery=guided_recovery,
        )
        job.launch_context_file = launch_context_file
        job.launch_args = launch_args
        job.lifecycle_metadata = lifecycle_metadata

        # Emit job created event
        created_transition = {
            "from_status": None,
            "to_status": "pending",
            "applied": True,
            "source": "training_service.launch_raft",
            "reason": "job_created",
            "timestamp": datetime.now().isoformat(),
            "metadata": self._merge_transition_metadata(
                job,
                {"job_type": "raft"},
            ),
        }
        get_event_bus().emit_sync(Event(
            type=EventType.JOB_CREATED,
            job_id=job.id,
            data=build_transition_payload(
                created_transition,
                name=job.name,
                type="raft",
                **self._event_extra_fields(job),
            ),
        ))
        
        # Launch subprocess
        await self._launch_process_with_runtime_options(
            job.id,
            cmd,
            on_log,
            no_caffeinate=no_caffeinate,
        )
        
        return job.id

    async def launch_modality_train(
        self,
        *,
        modality: str,
        model: str,
        dataset: str,
        output_dir: str,
        cycles: int,
        learning_rate: Optional[float] = None,
        lr_decay: Optional[float] = None,
        samples_per_prompt: Optional[int] = None,
        temperature: Optional[float] = None,
        keep_percent: Optional[float] = None,
        reward_threshold: Optional[float] = None,
        task: Optional[str] = None,
        limit: Optional[int] = None,
        resume_from_cycle: int = 0,
        seed: int = 42,
        allow_prototype_train: bool = False,
        on_log: Optional[Callable[[str], None]] = None,
        source_ui_page: str = "/training",
        origin_job_id: Optional[str] = None,
        relaunch: bool = False,
        resume_strategy: Optional[str] = None,
        guided_recovery: Optional[dict[str, Any]] = None,
        no_caffeinate: bool = False,
    ) -> str:
        """Launch modality-specific train command (vlm/audio/reasoning/agentic)."""
        if modality not in MODALITY_TRAIN_LAUNCH_CONTRACTS:
            raise ValueError(f"Unsupported modality: {modality}")

        model, dataset, output_dir, seed = self._validate_modality_launch_payload(
            modality=modality,
            model=model,
            dataset=dataset,
            output_dir=output_dir,
            cycles=cycles,
            resume_from_cycle=resume_from_cycle,
            seed=seed,
            limit=limit,
            task=task,
            samples_per_prompt=samples_per_prompt,
        )

        capability = check_modality_train_capability(
            modality=modality,
            model_name=model,
            allow_prototype_train=allow_prototype_train,
            dry_run=False,
        )
        if not capability.allowed:
            raise ValueError(capability.message.splitlines()[-1])

        job = self.state.create_job(
            job_type=modality,
            name=f"{modality.upper()} Train: {Path(model).name} on {dataset}",
            output_dir=Path(output_dir),
        )
        job.total_cycles = cycles

        cmd = [
            sys.executable, "-m", "halo_forge.cli",
            modality, "train",
            "--model", model,
            "--dataset", dataset,
            "--output", output_dir,
            "--cycles", str(cycles),
            "--resume-from-cycle", str(max(0, resume_from_cycle)),
            "--seed", str(seed),
        ]
        if learning_rate is not None and modality in {"audio", "reasoning", "agentic"}:
            cmd.extend(["--lr", str(learning_rate)])
        if lr_decay is not None:
            cmd.extend(["--lr-decay", str(lr_decay)])
        if samples_per_prompt is not None and modality in {"vlm", "audio"}:
            cmd.extend(["--samples-per-prompt", str(samples_per_prompt)])
        if temperature is not None:
            cmd.extend(["--temperature", str(temperature)])
        if keep_percent is not None:
            cmd.extend(["--keep-percent", str(keep_percent)])
        if reward_threshold is not None and modality in {"vlm", "audio"}:
            cmd.extend(["--reward-threshold", str(reward_threshold)])
        if modality == "audio" and task:
            cmd.extend(["--task", task])
        if limit is not None and modality in {"vlm", "reasoning", "agentic"}:
            cmd.extend(["--limit", str(limit)])
        if capability.capability.status == "prototype" and allow_prototype_train:
            cmd.append("--allow-prototype-train")
        if no_caffeinate:
            cmd.append("--no-caffeinate")

        launch_args = {
            "model": model,
            "dataset": dataset,
            "output_dir": output_dir,
            "cycles": cycles,
            "learning_rate": learning_rate,
            "lr_decay": lr_decay,
            "samples_per_prompt": samples_per_prompt,
            "temperature": temperature,
            "keep_percent": keep_percent,
            "reward_threshold": reward_threshold,
            "task": task,
            "limit": limit,
            "resume_from_cycle": max(0, resume_from_cycle),
            "seed": seed,
            "allow_prototype_train": (
                capability.capability.status == "prototype" and allow_prototype_train
            ),
            "no_caffeinate": no_caffeinate,
        }
        launch_context_file = None
        try:
            launch_context_file = persist_launch_context(
                output_dir=output_dir,
                job_type=modality,
                service="training",
                source_ui_page=source_ui_page,
                command=cmd,
                args=launch_args,
                relaunch_capabilities={
                    "can_relaunch": True,
                    "can_clone": True,
                    "can_resume_latest": modality in CYCLE_BASED_TRAINING_JOB_TYPES,
                },
                metadata=self._build_launch_context_metadata(guided_recovery),
            )
        except Exception as e:
            print(f"[TrainingService] Failed to persist launch context: {e}")
        lifecycle_metadata = self._build_lifecycle_metadata(
            origin_job_id=origin_job_id,
            relaunch=relaunch,
            launch_context_file=launch_context_file,
            resume_strategy=resume_strategy,
            guided_recovery=guided_recovery,
        )
        job.launch_context_file = launch_context_file
        job.launch_args = launch_args
        job.lifecycle_metadata = lifecycle_metadata

        created_transition = {
            "from_status": None,
            "to_status": "pending",
            "applied": True,
            "source": f"training_service.launch_{modality}_train",
            "reason": "job_created",
            "timestamp": datetime.now().isoformat(),
            "metadata": self._merge_transition_metadata(
                job,
                {"job_type": modality},
            ),
        }
        get_event_bus().emit_sync(Event(
            type=EventType.JOB_CREATED,
            job_id=job.id,
            data=build_transition_payload(
                created_transition,
                name=job.name,
                type=modality,
                **self._event_extra_fields(job),
            ),
        ))

        await self._launch_process_with_runtime_options(
            job.id,
            cmd,
            on_log,
            no_caffeinate=no_caffeinate,
        )
        return job.id

    async def relaunch_from_context(
        self,
        launch_context_file: str | Path,
        *,
        origin_job_id: Optional[str] = None,
        resume_latest: bool = False,
        override_args: Optional[dict[str, Any]] = None,
        guided_recovery: Optional[dict[str, Any]] = None,
        source_ui_page: str = "/monitor",
        on_log: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Relaunch a training job from persisted launch context."""
        context = read_launch_context(launch_context_file)
        if context.service != "training":
            raise ValueError("launch context does not belong to training service")

        args = self._apply_launch_overrides(dict(context.args), override_args)
        job_type = context.job_type
        resume_strategy = "resume_latest" if resume_latest else "relaunch"
        recovery_payload = None
        if guided_recovery or override_args:
            recovery_payload = {
                "applied_overrides": dict(override_args or {}),
                "reason_code": str((guided_recovery or {}).get("reason_code") or ""),
                "summary": str((guided_recovery or {}).get("evidence_summary") or ""),
            }

        if resume_latest:
            if job_type not in CYCLE_BASED_TRAINING_JOB_TYPES:
                raise ValueError(f"{job_type} does not support resume_latest")
            output_dir = args.get("output_dir")
            if not output_dir:
                raise ValueError("resume_latest requires output_dir in launch context")
            next_cycle = self.resolve_resume_latest_cycle(str(output_dir))
            total_cycles = int(args.get("cycles") or 0)
            if total_cycles and next_cycle > total_cycles:
                raise ValueError(
                    f"resume_from_cycle {next_cycle} exceeds configured cycles {total_cycles}"
                )
            if job_type == "raft":
                checkpoint_cycle = max(0, next_cycle - 1)
                checkpoint_path = Path(output_dir) / f"cycle_{checkpoint_cycle}_final"
                if not checkpoint_path.exists():
                    raise ValueError(f"RAFT checkpoint not found for resume_latest: {checkpoint_path}")
                args["checkpoint"] = str(checkpoint_path)
            else:
                args["resume_from_cycle"] = next_cycle

        if job_type == "sft":
            return await self.launch_sft(
                on_log=on_log,
                source_ui_page=source_ui_page,
                origin_job_id=origin_job_id,
                relaunch=True,
                resume_strategy=resume_strategy,
                guided_recovery=recovery_payload,
                **args,
            )
        if job_type == "raft":
            return await self.launch_raft(
                on_log=on_log,
                source_ui_page=source_ui_page,
                origin_job_id=origin_job_id,
                relaunch=True,
                resume_strategy=resume_strategy,
                guided_recovery=recovery_payload,
                **args,
            )
        if job_type in {"vlm", "audio", "reasoning", "agentic"}:
            return await self.launch_modality_train(
                modality=job_type,
                on_log=on_log,
                source_ui_page=source_ui_page,
                origin_job_id=origin_job_id,
                relaunch=True,
                resume_strategy=resume_strategy,
                guided_recovery=recovery_payload,
                **args,
            )

        raise ValueError(f"Unsupported training relaunch job_type: {job_type}")
    
    async def _launch_process_with_runtime_options(
        self,
        job_id: str,
        cmd: list[str],
        on_log: Optional[Callable[[str], None]] = None,
        *,
        no_caffeinate: bool = False,
    ):
        """Launch with optional runtime wrappers while preserving the default call contract."""
        if no_caffeinate:
            await self._launch_process(job_id, cmd, on_log, no_caffeinate=True)
            return
        await self._launch_process(job_id, cmd, on_log)

    async def _launch_process(
        self,
        job_id: str,
        cmd: list[str],
        on_log: Optional[Callable[[str], None]] = None,
        *,
        no_caffeinate: bool = False,
    ):
        """Launch subprocess and start log streaming."""
        job = self.state.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        # Set up parser and log buffer
        self._parsers[job_id] = MetricsParser()
        self._log_buffers[job_id] = deque(maxlen=1000)
        
        if on_log:
            if job_id not in self._callbacks:
                self._callbacks[job_id] = []
            self._callbacks[job_id].append(on_log)
        
        # Get optimized environment
        env = self._get_strix_halo_env()
        
        exec_cmd = caffeinate_command(cmd, enabled=not no_caffeinate)

        # Launch subprocess
        process = await asyncio.create_subprocess_exec(
            *exec_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=Path.cwd(),
        )
        
        job.process = process
        job.started_at = datetime.now()
        transitioned = self.state.update_job_status(
            job_id,
            "running",
            source="training_service._launch_process",
            reason="process_started",
            metadata=self._merge_transition_metadata(
                job,
                {
                    "command": cmd,
                    "executed_command": exec_cmd,
                    "caffeinate": exec_cmd != cmd,
                },
            ),
        )

        # Emit job started event
        if transitioned:
            transition = self.state.get_last_transition(job_id)
            await get_event_bus().emit(Event(
                type=EventType.JOB_STARTED,
                job_id=job_id,
                data=build_transition_payload(
                    transition,
                    name=job.name,
                    type=job.type,
                    **self._event_extra_fields(job),
                ),
            ))
        
        # Start log streaming task
        asyncio.create_task(self._stream_logs(job_id))
    
    async def _stream_logs(self, job_id: str):
        """Stream subprocess output and parse metrics."""
        job = self.state.get_job(job_id)
        if not job or not job.process:
            return
        
        parser = self._parsers.get(job_id)
        log_buffer = self._log_buffers.get(job_id, deque(maxlen=1000))
        callbacks = self._callbacks.get(job_id, [])
        event_bus = get_event_bus()
        
        # Set up persistent log file
        log_file_path = None
        log_file = None
        if job.output_dir:
            log_file_path = Path(job.output_dir) / f"{job_id}_training.log"
            job.log_file_path = log_file_path
            try:
                log_file_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = open(log_file_path, 'a', encoding='utf-8')
            except Exception as e:
                print(f"[TrainingService] Could not open log file: {e}")
                log_file = None
        
        try:
            async for line_bytes in job.process.stdout:
                for line in self._normalize_stream_chunk(
                    line_bytes.decode('utf-8', errors='replace')
                ):
                    if self._should_skip_stream_line(job_id, line):
                        continue

                    timestamp = datetime.now().isoformat()
                    try:
                        from halo_forge.telemetry.apple_silicon import get_mps_fallback_counter

                        get_mps_fallback_counter().record_warning_line(line)
                    except Exception:
                        pass

                    # Store log line in memory buffer
                    log_buffer.append({
                        'timestamp': timestamp,
                        'line': line,
                    })

                    # Write to persistent log file
                    if log_file:
                        try:
                            log_file.write(f"[{timestamp}] {line}\n")
                            log_file.flush()
                        except Exception:
                            pass

                    # Emit log line event
                    await event_bus.emit(Event(
                        type=EventType.LOG_LINE,
                        job_id=job_id,
                        data={'line': line, 'timestamp': timestamp}
                    ))

                    # Call legacy callbacks
                    for callback in callbacks:
                        try:
                            callback(line)
                        except Exception:
                            pass

                    # Parse metrics
                    if parser:
                        metrics = parser.parse_line(line)
                        if metrics:
                            self._update_job_metrics(job_id, metrics)

                            # Emit metrics update event
                            await event_bus.emit(Event(
                                type=EventType.METRICS_UPDATE,
                                job_id=job_id,
                                data={
                                    'loss': metrics.loss,
                                    'learning_rate': metrics.learning_rate,
                                    'epoch': metrics.epoch,
                                    'step': metrics.step,
                                    'total_steps': metrics.total_steps,
                                    'cycle': metrics.cycle,
                                    'total_cycles': metrics.total_cycles,
                                    'compile_rate': metrics.compile_rate,
                                    'grad_norm': metrics.grad_norm,
                                    'yield_snapshot': metrics.yield_snapshot,
                                }
                            ))

                    # Detect checkpoint saves and notify
                    line_lower = line.lower()
                    if ('checkpoint' in line_lower and 'saved' in line_lower) or \
                       ('saving' in line_lower and 'checkpoint' in line_lower):
                        checkpoint_path = str(job.output_dir) if job.output_dir else "checkpoint"

                        # Emit checkpoint event
                        await event_bus.emit(Event(
                            type=EventType.CHECKPOINT_SAVED,
                            job_id=job_id,
                            data={'path': checkpoint_path}
                        ))

                        if HAS_UI_NOTIFICATIONS:
                            notify_checkpoint_saved(checkpoint_path)
        
        except Exception as e:
            job.error_message = str(e)
        finally:
            # Close persistent log file
            self._last_progress_line_by_job.pop(job_id, None)
            if log_file:
                try:
                    log_file.close()
                except Exception:
                    pass
        
        # Process completed
        return_code = await job.process.wait()

        if job.stop_requested:
            transitioned = self.state.update_job_status(
                job_id,
                "stopped",
                source="training_service._stream_logs",
                reason="stop_requested",
                metadata=self._merge_transition_metadata(
                    job,
                    {"return_code": return_code},
                ),
            )
            if transitioned:
                transition = self.state.get_last_transition(job_id)
                await event_bus.emit(Event(
                    type=EventType.JOB_STOPPED,
                    job_id=job_id,
                    data=build_transition_payload(
                        transition,
                        return_code=return_code,
                        **self._event_extra_fields(job),
                    ),
                ))
        elif return_code == 0:
            transitioned = self.state.update_job_status(
                job_id,
                "completed",
                source="training_service._stream_logs",
                reason="process_exit_ok",
                metadata=self._merge_transition_metadata(
                    job,
                    {"return_code": return_code},
                ),
            )
            if not transitioned:
                return
            transition = self.state.get_last_transition(job_id)
            await event_bus.emit(Event(
                type=EventType.JOB_COMPLETED,
                job_id=job_id,
                data=build_transition_payload(
                    transition,
                    return_code=return_code,
                    **self._event_extra_fields(job),
                ),
            ))
        elif return_code == -signal.SIGTERM or return_code == -signal.SIGKILL:
            transitioned = self.state.update_job_status(
                job_id,
                "stopped",
                source="training_service._stream_logs",
                reason="terminated_signal",
                metadata=self._merge_transition_metadata(
                    job,
                    {"return_code": return_code},
                ),
            )
            if not transitioned:
                return
            transition = self.state.get_last_transition(job_id)
            await event_bus.emit(Event(
                type=EventType.JOB_STOPPED,
                job_id=job_id,
                data=build_transition_payload(
                    transition,
                    return_code=return_code,
                    **self._event_extra_fields(job),
                ),
            ))
        else:
            job.error_message = f"Process exited with code {return_code}"
            transitioned = self.state.update_job_status(
                job_id,
                "failed",
                source="training_service._stream_logs",
                reason="process_exit_error",
                metadata=self._merge_transition_metadata(
                    job,
                    {"return_code": return_code, "error": job.error_message},
                ),
            )
            if not transitioned:
                return
            transition = self.state.get_last_transition(job_id)
            await event_bus.emit(Event(
                type=EventType.JOB_FAILED,
                job_id=job_id,
                data=build_transition_payload(
                    transition,
                    return_code=return_code,
                    error=job.error_message,
                    **self._event_extra_fields(job),
                ),
            ))
    
    def _update_job_metrics(self, job_id: str, metrics: ParsedMetrics):
        """Update job state with parsed metrics."""
        job = self.state.get_job(job_id)
        if not job:
            return
        
        if metrics.loss is not None:
            job.latest_loss = metrics.loss
            step = metrics.step or job.current_step
            self.state.add_metric(job_id, 'loss', step, metrics.loss)
        
        if metrics.learning_rate is not None:
            job.latest_lr = metrics.learning_rate
        
        if metrics.epoch is not None:
            job.current_epoch = metrics.epoch  # Keep as float for accurate progress
        
        if metrics.step is not None:
            job.current_step = metrics.step
        
        if metrics.total_steps is not None:
            job.total_steps = metrics.total_steps
        
        if metrics.cycle is not None:
            job.current_cycle = metrics.cycle
        
        if metrics.total_cycles is not None:
            job.total_cycles = metrics.total_cycles
        
        if metrics.compile_rate is not None:
            job.verification_rate = metrics.compile_rate
        
        if metrics.grad_norm is not None:
            job.latest_grad_norm = metrics.grad_norm
            self.state.add_metric(job_id, 'grad_norm', job.current_step, metrics.grad_norm)
        
        if metrics.yield_snapshot is not None:
            job.latest_yield_snapshot = dict(metrics.yield_snapshot)
            job.yield_history.append(dict(metrics.yield_snapshot))
    
    async def stop_job(self, job_id: str, timeout: float = 30.0) -> bool:
        """
        Stop a running training job.
        
        Sends SIGTERM first to allow graceful shutdown (checkpoint saving),
        then SIGKILL if the process doesn't exit within timeout.
        
        Args:
            job_id: Job ID to stop
            timeout: Seconds to wait for graceful shutdown
            
        Returns:
            True if job was stopped, False if not found or not running
        """
        job = self.state.get_job(job_id)
        if not job:
            return False
        if job.status in {"stopped", "completed", "failed"}:
            return True
        if not job.process:
            return False
        
        if job.status != "running":
            return job.status in {"stopped", "completed", "failed"}

        job.stop_requested = True
        
        # Send SIGTERM first (allows checkpoint saving)
        try:
            job.process.terminate()
        except ProcessLookupError:
            # Process already dead
            transitioned = self.state.update_job_status(
                job_id,
                "stopped",
                source="training_service.stop_job",
                reason="process_missing",
                metadata=self._merge_transition_metadata(job),
            )
            if transitioned:
                transition = self.state.get_last_transition(job_id)
                await get_event_bus().emit(Event(
                    type=EventType.JOB_STOPPED,
                    job_id=job_id,
                    data=build_transition_payload(
                        transition,
                        **self._event_extra_fields(job),
                    ),
                ))
            return True
        
        try:
            # Wait for graceful shutdown
            await asyncio.wait_for(job.process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # Force kill if still running
            try:
                job.process.kill()
                await job.process.wait()
            except ProcessLookupError:
                pass
        
        return True
    
    def get_logs(self, job_id: str, last_n: Optional[int] = None) -> list[dict]:
        """
        Get log entries for a job.
        
        Args:
            job_id: Job ID
            last_n: Only return last N entries
            
        Returns:
            List of log entries with timestamp and line
        """
        buffer = self._log_buffers.get(job_id, deque())
        logs = list(buffer)
        
        if last_n is not None:
            logs = logs[-last_n:]
        
        return logs
    
    def get_metrics(self, job_id: str) -> TrainingMetrics:
        """
        Get current metrics for a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            TrainingMetrics with current values
        """
        job = self.state.get_job(job_id)
        if not job:
            return TrainingMetrics()
        
        return TrainingMetrics(
            loss=job.latest_loss,
            learning_rate=job.latest_lr,
            epoch=float(job.current_epoch),
            step=job.current_step,
            total_steps=job.total_steps,
            cycle=job.current_cycle,
            total_cycles=job.total_cycles,
            compile_rate=job.verification_rate,
            grad_norm=job.latest_grad_norm,
        )
    
    def add_log_callback(self, job_id: str, callback: Callable[[str], None]):
        """Add a callback for log lines."""
        if job_id not in self._callbacks:
            self._callbacks[job_id] = []
        self._callbacks[job_id].append(callback)
    
    def remove_log_callback(self, job_id: str, callback: Callable[[str], None]):
        """Remove a log callback."""
        if job_id in self._callbacks:
            try:
                self._callbacks[job_id].remove(callback)
            except ValueError:
                pass
