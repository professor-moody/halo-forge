"""Shared public product service built on top of internal halo-forge services."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from halo_forge.training_recovery import build_recovery_guidance
from ui.services.ops_readiness_service import OpsReadinessService, get_ops_readiness_service
from ui.services.quickstart_presets import list_quickstart_presets
from ui.services.results_service import ResultsService, TrainingRunSummary, get_results_service
from ui.services.training_presentation import build_launch_presentation
from ui.services.training_service import TrainingLaunchPreflight, TrainingService
from ui.state import AppState, JobState, state as default_state

from .views import (
    ActiveRunRowView,
    AttentionItemView,
    DashboardSummaryView,
    DocsCapabilitySummaryView,
    ModalityReadinessView,
    ProductUserSummaryView,
    PublicActionView,
    ResearchSectionView,
    RunMetricsSummaryView,
    TrainingLaunchPreflightView,
    TrainingRecoveryView,
    TrainingRunDetailView,
    TrainingRunListItemView,
    TrainingRunLiveView,
    build_user_summary,
    to_dict,
)


TRAINING_MODALITIES = ("sft", "raft", "vlm", "audio", "reasoning", "agentic")

PUBLIC_TRAIN_ALLOWED_FIELDS: dict[str, set[str]] = {
    "sft": {
        "mode",
        "model",
        "dataset",
        "output_dir",
        "epochs",
        "max_samples",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "no_caffeinate",
    },
    "raft": {
        "mode",
        "model",
        "prompts",
        "output_dir",
        "cycles",
        "samples_per_prompt",
        "keep_percent",
        "reward_threshold",
        "temperature",
        "no_caffeinate",
    },
    "vlm": {
        "mode",
        "model",
        "dataset",
        "output_dir",
        "cycles",
        "limit",
        "samples_per_prompt",
        "keep_percent",
        "reward_threshold",
        "temperature",
        "no_caffeinate",
    },
    "audio": {
        "mode",
        "model",
        "dataset",
        "output_dir",
        "cycles",
        "samples_per_prompt",
        "keep_percent",
        "reward_threshold",
        "temperature",
        "task",
        "no_caffeinate",
    },
    "reasoning": {
        "mode",
        "model",
        "dataset",
        "output_dir",
        "cycles",
        "limit",
        "keep_percent",
        "temperature",
        "learning_rate",
        "no_caffeinate",
    },
    "agentic": {
        "mode",
        "model",
        "dataset",
        "output_dir",
        "cycles",
        "limit",
        "keep_percent",
        "temperature",
        "learning_rate",
        "no_caffeinate",
    },
}

PUBLIC_TRAIN_REQUIRED_TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "sft": ("model", "dataset", "output_dir"),
    "raft": ("model", "prompts", "output_dir"),
    "vlm": ("model", "dataset", "output_dir"),
    "audio": ("model", "dataset", "output_dir", "task"),
    "reasoning": ("model", "dataset", "output_dir"),
    "agentic": ("model", "dataset", "output_dir"),
}


@dataclass(frozen=True)
class _DocsSource:
    slug: str
    path: Path
    audience: str
    doc_url: str


class PublicApiService:
    """Pure service layer shared by the public API and public frontend."""

    DOC_SOURCES = (
        _DocsSource(
            slug="public-frontend",
            path=Path("website/hugo-docs/content/docs/reference/public-frontend.md"),
            audience="product",
            doc_url="/docs/public-frontend",
        ),
        _DocsSource(
            slug="web-ui-console",
            path=Path("website/hugo-docs/content/docs/reference/web-ui.md"),
            audience="research",
            doc_url="/docs/reference/web-ui",
        ),
        _DocsSource(
            slug="modality-readiness",
            path=Path("website/hugo-docs/content/docs/experimental.md"),
            audience="product",
            doc_url="/docs/experimental",
        ),
        _DocsSource(
            slug="local-docs-index",
            path=Path("docs/README.md"),
            audience="research",
            doc_url="/docs/local",
        ),
    )

    def __init__(
        self,
        *,
        app_state: AppState | None = None,
        results_service: ResultsService | None = None,
        readiness_service: OpsReadinessService | None = None,
        training_service: TrainingService | None = None,
        base_path: Path | None = None,
    ) -> None:
        self.app_state = app_state or default_state
        self.base_path = (base_path or Path.cwd()).resolve()
        self.results_service = results_service or get_results_service()
        self.readiness_service = readiness_service or get_ops_readiness_service()
        self.training_service = training_service or TrainingService(self.app_state)

    def _active_backend_name(self) -> str:
        """Return the active accelerator-kind name for cost / display use.

        Cached per service instance after the first probe — backend
        detection is cheap but not free, and the run-detail endpoint
        fires it on every request.
        """
        cached = getattr(self, "_cached_backend_name", None)
        if cached:
            return cached
        try:
            from halo_forge.backend import get_backend

            name = get_backend().name
        except Exception:
            name = "unknown"
        self._cached_backend_name = name
        return name

    def get_backend_info(self) -> Dict[str, Any]:
        """Return the active compute backend and its capabilities.

        Used by the frontend to render "Running on Apple Silicon (MPS)" /
        "Running on AMD ROCm" badges and to gate UI affordances (e.g. hide
        4-bit quantization toggles on backends that can't honor them).
        """
        from halo_forge.backend import get_backend
        from dataclasses import asdict

        backend = get_backend()
        chip = None
        try:
            from halo_forge.telemetry.apple_silicon import AppleSiliconTelemetry

            provider = AppleSiliconTelemetry(backend_name=backend.name)
            chip = provider.sample().chip
        except Exception:
            chip = None
        return {
            "name": backend.name,
            "device": backend.device(),
            "chip": chip,
            "capabilities": asdict(backend.capabilities),
            "training_defaults": backend.training_defaults(),
            "inference_defaults": backend.inference_defaults(),
        }

    async def cancel_run(self, run_identifier: str) -> Dict[str, Any]:
        """Cancel a running training job.

        Only valid for active jobs (`_resolve_run_source` returns
        kind="job"); completed runs in the results service have no
        process to stop. Returns a stable envelope so the frontend can
        render result-or-reason without branching on HTTP status.

        Backed by `TrainingService.stop_job` which sends SIGTERM, waits
        for graceful shutdown (so the trainer can save a checkpoint),
        then SIGKILLs on timeout.
        """
        try:
            source = self._resolve_run_source(run_identifier)
        except KeyError as exc:
            return {
                "ok": False,
                "reason": f"Run not found: {exc}",
                "run_id": run_identifier,
                "status": None,
            }

        if source.get("kind") != "job":
            return {
                "ok": False,
                "reason": "Run is not active; only running jobs can be cancelled.",
                "run_id": run_identifier,
                "status": "completed",
            }

        job = source["job"]
        try:
            stopped = await self.training_service.stop_job(job.id)
        except Exception as exc:
            return {
                "ok": False,
                "reason": f"stop_job failed: {exc}",
                "run_id": job.id,
                "status": job.status,
            }

        return {
            "ok": bool(stopped),
            "reason": None if stopped else "Job was not running.",
            "run_id": job.id,
            "status": job.status,
        }

    def get_run_logs(
        self,
        run_identifier: str,
        *,
        tail: int = 200,
    ) -> Dict[str, Any]:
        """Return the tail of training logs for a run.

        Looks for `run.log` (or `train.log`) inside the run's output_dir
        first; falls back to scanning `logs/` for the newest log file
        whose basename references this run. Honest about unavailability
        — returns `{"available": False, "lines": [], "reason": "..."}`
        rather than 5xx-ing.

        Phase D v2 contract: the frontend polls this every few seconds
        for active runs and renders the last N lines in a virtual-scroll
        panel.
        """
        try:
            source = self._resolve_run_source(run_identifier)
        except Exception as exc:
            return {
                "available": False,
                "lines": [],
                "reason": f"Run not found: {exc}",
                "log_path": None,
                "tail": int(tail),
            }

        from pathlib import Path

        # _resolve_run_source returns either a job (active) or a summary
        # (completed). Both expose an output_dir, but the field lives in
        # different places. Normalize them here.
        output_dir, run_id = _extract_output_dir_and_run_id(source)

        candidates: list[Path] = []
        if output_dir:
            for name in ("run.log", "train.log", "training.log"):
                candidate = output_dir / name
                if candidate.exists():
                    candidates.append(candidate)

        # Fall back to logs/ scan — newest file whose basename mentions
        # the run_id or output_dir basename.
        if not candidates:
            logs_dir = self.base_path / "logs"
            if logs_dir.is_dir():
                tokens = [t for t in (run_id, output_dir.name if output_dir else "") if t]
                matches = []
                for log_file in logs_dir.glob("*.log"):
                    if any(tok in log_file.name for tok in tokens):
                        matches.append(log_file)
                # Newest first
                matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                candidates.extend(matches[:1])

        if not candidates:
            return {
                "available": False,
                "lines": [],
                "reason": "No log file found alongside this run.",
                "log_path": None,
                "tail": int(tail),
            }

        log_path = candidates[0]
        # Defensive line-by-line read with a soft cap so an enormous log
        # never blows the API memory budget.
        max_tail = max(1, min(int(tail), 5000))
        try:
            with log_path.open(encoding="utf-8", errors="replace") as f:
                buf: list[str] = []
                for line in f:
                    buf.append(line.rstrip("\n"))
                    if len(buf) > max_tail * 2:
                        # Keep the trailing window only; deque-like prune
                        buf = buf[-max_tail:]
                lines = buf[-max_tail:] if len(buf) > max_tail else buf
        except OSError as exc:
            return {
                "available": False,
                "lines": [],
                "reason": f"Could not read {log_path.name}: {exc}",
                "log_path": str(log_path),
                "tail": max_tail,
            }

        return {
            "available": True,
            "lines": lines,
            "reason": None,
            "log_path": str(log_path),
            "tail": max_tail,
            "total_lines_returned": len(lines),
        }

    def get_run_samples(
        self,
        run_identifier: str,
        *,
        cycle: Optional[int] = None,
        kind: str = "samples",
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Return RAFT-style sample artifacts for a cycle.

        kind="samples"  -> all generated samples for the cycle
                           (cycle_{N}_samples.jsonl)
        kind="accepted" -> the post-filter set fed to SFT
                           (cycle_{N}/accepted.jsonl)

        Returns a stable JSON envelope so the frontend can render an
        "available: false" placeholder when the trainer didn't write
        these artifacts (older summaries, SFT-only runs, or local-only
        files that never reached this host).
        """
        try:
            source = self._resolve_run_source(run_identifier)
        except Exception as exc:
            return {
                "available": False,
                "samples": [],
                "reason": f"Run not found: {exc}",
                "cycle": cycle,
                "kind": kind,
            }

        from pathlib import Path

        out, _ = _extract_output_dir_and_run_id(source)
        if out is None:
            return {
                "available": False,
                "samples": [],
                "reason": "Run has no recorded output_dir.",
                "cycle": cycle,
                "kind": kind,
            }

        # Discover cycles by scanning the output dir for cycle_N folders.
        if cycle is None:
            available_cycles = sorted(
                int(p.name.split("_")[1])
                for p in out.glob("cycle_*")
                if p.name.split("_", 1)[1].isdigit()
            )
            if available_cycles:
                cycle = available_cycles[-1]
            else:
                return {
                    "available": False,
                    "samples": [],
                    "reason": "No cycle artifacts found.",
                    "cycle": None,
                    "kind": kind,
                    "available_cycles": [],
                }
        else:
            cycle = int(cycle)

        if kind == "accepted":
            jsonl_path = out / f"cycle_{cycle}" / "accepted.jsonl"
        else:
            jsonl_path = out / f"cycle_{cycle}_samples.jsonl"

        if not jsonl_path.exists():
            return {
                "available": False,
                "samples": [],
                "reason": f"{jsonl_path.name} not found.",
                "cycle": cycle,
                "kind": kind,
            }

        import json as _json

        samples: list[dict[str, Any]] = []
        max_limit = max(1, min(int(limit), 500))
        try:
            with jsonl_path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    samples.append(record)
                    if len(samples) >= max_limit:
                        break
        except OSError as exc:
            return {
                "available": False,
                "samples": [],
                "reason": f"Could not read {jsonl_path.name}: {exc}",
                "cycle": cycle,
                "kind": kind,
            }

        # Discover all cycles for the scrubber.
        available_cycles = sorted(
            int(p.name.split("_")[1])
            for p in out.glob("cycle_*")
            if p.name.split("_", 1)[1].isdigit()
        )

        return {
            "available": True,
            "samples": samples,
            "reason": None,
            "cycle": cycle,
            "kind": kind,
            "available_cycles": available_cycles,
            "limit": max_limit,
            "total_returned": len(samples),
            "source_path": str(jsonl_path),
        }

    def get_telemetry(self) -> Dict[str, Any]:
        """Live hardware telemetry — the data behind the public_app's
        telemetry strip.

        Polled by the frontend at ~3s intervals. The provider's own
        cache (1s TTL on rocm-smi/nvidia-smi subprocess output) keeps
        the cost bounded even if a flood of clients polls in lockstep.

        Failures inside the provider are caught and surfaced as a
        `note` field on the sample rather than 5xx responses, because
        the strip is meant to *always* render; missing values render
        as "—" but the contract stays stable.
        """
        from halo_forge.telemetry import (
            TelemetryUnavailableError,
            get_telemetry_provider,
        )

        try:
            provider = get_telemetry_provider()
        except TelemetryUnavailableError as exc:
            # Should not happen — the registry falls back to CPU — but
            # we shape the response identically so the frontend never
            # sees an undefined payload.
            return {
                "timestamp": 0.0,
                "backend": "unknown",
                "device_name": None,
                "note": f"Telemetry unavailable: {exc}",
            }
        sample = provider.sample()
        return sample.to_dict()

    def list_training_datasets(self) -> list[dict[str, Any]]:
        """Catalog of known training datasets for the launch configurator.

        Reads `halo_forge.sft.datasets.SFT_DATASETS` (the same registry the
        CLI uses) and projects it down to a JSON-shaped list. Domain
        ('code', 'vlm', 'audio', 'reasoning', 'agentic') is included so
        the frontend can group + filter by modality without re-deriving
        the mapping client-side.
        """
        from halo_forge.sft.datasets import SFT_DATASETS

        items: list[dict[str, Any]] = []
        for spec in SFT_DATASETS.values():
            items.append(
                {
                    "key": spec.name,
                    "huggingface_id": spec.huggingface_id,
                    "description": spec.description,
                    "domain": spec.domain,
                    "size_hint": spec.size_hint,
                    "default_split": spec.default_split,
                }
            )
        return items

    def list_training_verifiers(self) -> list[dict[str, Any]]:
        """Verifiers available for the RAFT loop. Each entry exposes the
        toolchain dependency the user needs locally so the configurator
        can preflight whether the binary is reachable.
        """
        from halo_forge.cli import RAFT_TRAIN_SUPPORTED_VERIFIERS

        # Hand-curated metadata that the CLI surface already understands.
        # Kept here (not in cli.py) so the API doesn't pull cli imports
        # at module load.
        catalog: dict[str, dict[str, Any]] = {
            "gcc": {
                "label": "GCC (Linux/POSIX C/C++)",
                "toolchain": "gcc",
                "modality": "code",
                "platforms": ["linux", "macos"],
            },
            "mingw": {
                "label": "MinGW (Windows cross-compile)",
                "toolchain": "x86_64-w64-mingw32-g++",
                "modality": "code",
                "platforms": ["linux", "macos"],
            },
            "msvc": {
                "label": "MSVC (remote Windows host)",
                "toolchain": "remote-msvc",
                "modality": "code",
                "platforms": ["any"],
            },
            "humaneval": {
                "label": "HumanEval (Python)",
                "toolchain": "python",
                "modality": "code",
                "platforms": ["any"],
            },
            "mbpp": {
                "label": "MBPP (Python)",
                "toolchain": "python",
                "modality": "code",
                "platforms": ["any"],
            },
            "rust": {
                "label": "Rust (rustc)",
                "toolchain": "rustc",
                "modality": "code",
                "platforms": ["any"],
            },
            "go": {
                "label": "Go (go build)",
                "toolchain": "go",
                "modality": "code",
                "platforms": ["any"],
            },
            "auto": {
                "label": "Auto-detect",
                "toolchain": "any",
                "modality": "code",
                "platforms": ["any"],
            },
            "execution": {
                "label": "Execution (sandboxed runtime)",
                "toolchain": "sandbox",
                "modality": "code",
                "platforms": ["any"],
            },
        }
        return [
            {"key": k, **catalog.get(k, {"label": k, "toolchain": k, "modality": "code"})}
            for k in RAFT_TRAIN_SUPPORTED_VERIFIERS
        ]

    def list_verifier_catalog(self) -> dict[str, Any]:
        """Inventory of every verifier the runtime can resolve (Track F-O).

        Wraps `halo_forge.rlvr.verifiers.registry.inventory()` and adds
        the plugin directory path so the UI can tell users *where* to
        drop a new `.py` to register one.

        Origin counts are also returned to keep the UI from filtering
        the items list just to render headline metrics.
        """
        from halo_forge.rlvr.verifiers.registry import _plugin_dir, inventory

        items = inventory()
        counts = {"builtin": 0, "user_plugin": 0, "entry_point": 0}
        for entry in items:
            origin = str(entry.get("origin", "builtin"))
            counts[origin] = counts.get(origin, 0) + 1
        return {
            "items": items,
            "counts": counts,
            "plugin_dir": str(_plugin_dir()),
            "total": len(items),
        }

    def list_suggested_models(self) -> list[dict[str, Any]]:
        """Backend-aware base-model suggestions.

        On torch backends we surface the HF ids the CLI uses by default;
        on MLX we list a few `mlx-community/...` variants known to load
        cleanly. The frontend renders this as a quick-pick list inside
        the model dropdown so users don't have to remember the canonical
        repo names.
        """
        from halo_forge.backend import get_backend

        backend = get_backend()
        backend_name = backend.name

        if backend_name == "mlx":
            suggestions = [
                "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
                "mlx-community/Qwen2.5-3B-Instruct-bf16",
                "mlx-community/Qwen2.5-7B-Instruct-bf16",
                "mlx-community/Llama-3.2-3B-Instruct-4bit",
            ]
        else:
            suggestions = [
                "Qwen/Qwen2.5-Coder-0.5B",
                "Qwen/Qwen2.5-Coder-3B",
                "Qwen/Qwen2.5-Coder-7B",
                "Qwen/Qwen2.5-3B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
            ]

        return [
            {
                "id": m,
                "for_backend": backend_name,
            }
            for m in suggestions
        ]

    def list_training_presets(self) -> list[dict[str, Any]]:
        """Return public-safe quickstart presets for training."""
        items: list[dict[str, Any]] = []
        for preset in list_quickstart_presets("training"):
            items.append(
                {
                    "key": preset.key,
                    "mode": preset.target,
                    "label": preset.label,
                    "description": preset.description,
                    "when_to_use": preset.recommendation.when_to_use,
                    "expected_runtime": preset.recommendation.expected_runtime,
                    "yield_safety": preset.recommendation.yield_safety,
                    "required_fields": list(preset.field_set.required_fields),
                    "optional_fields": list(preset.field_set.optional_fields),
                    "values": dict(preset.values),
                }
            )
        return items

    def preflight_training(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run launch preflight for the requested training mode."""
        payload = self._sanitize_public_training_payload(payload)
        mode = str(payload["mode"])

        if mode == "sft":
            preflight = self.training_service.preflight_sft_launch(
                model=str(payload.get("model") or ""),
                dataset=str(payload.get("dataset") or ""),
                output_dir=str(payload.get("output_dir") or ""),
                epochs=int(payload.get("epochs") or 1),
                batch_size=int(payload.get("batch_size") or 2),
                gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps") or 4),
                max_samples=self._optional_int(payload.get("max_samples")),
            )
        elif mode == "raft":
            preflight = self.training_service.preflight_raft_launch(
                model=str(payload.get("model") or ""),
                prompts=str(payload.get("prompts") or ""),
                output_dir=str(payload.get("output_dir") or ""),
                cycles=int(payload.get("cycles") or 1),
                samples_per_prompt=int(payload.get("samples_per_prompt") or 4),
                keep_percent=float(payload.get("keep_percent") or 0.5),
                reward_threshold=float(payload.get("reward_threshold") or 0.5),
                min_samples=int(payload.get("min_samples") or 1),
                max_new_tokens=int(payload.get("max_new_tokens") or 512),
                checkpoint=self._optional_str(payload.get("checkpoint")),
            )
        else:
            preflight = self.training_service.preflight_modality_train_launch(
                modality=mode,
                model=str(payload.get("model") or ""),
                dataset=str(payload.get("dataset") or ""),
                output_dir=str(payload.get("output_dir") or ""),
                cycles=int(payload.get("cycles") or 1),
                resume_from_cycle=int(payload.get("resume_from_cycle") or 0),
                seed=int(payload.get("seed") or 42),
                allow_prototype_train=bool(payload.get("allow_prototype_train", False)),
                limit=self._optional_int(payload.get("limit")),
                task=self._optional_str(payload.get("task")),
                samples_per_prompt=self._optional_int(payload.get("samples_per_prompt")),
                keep_percent=self._optional_float(payload.get("keep_percent")),
                reward_threshold=self._optional_float(payload.get("reward_threshold")),
            )

        return to_dict(self._build_preflight_view(mode=mode, preflight=preflight))

    async def launch_training(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Launch training from the public API."""
        payload = self._sanitize_public_training_payload(payload)
        mode = str(payload["mode"])
        if mode == "sft":
            job_id = await self.training_service.launch_sft(
                model=str(payload.get("model") or ""),
                dataset=str(payload.get("dataset") or ""),
                output_dir=str(payload.get("output_dir") or ""),
                epochs=int(self._value_or_default(payload.get("epochs"), 1)),
                batch_size=int(self._value_or_default(payload.get("batch_size"), 2)),
                gradient_accumulation_steps=int(self._value_or_default(payload.get("gradient_accumulation_steps"), 4)),
                max_samples=self._optional_int(payload.get("max_samples")),
                learning_rate=float(self._value_or_default(payload.get("learning_rate"), 2e-4)),
                no_caffeinate=bool(payload.get("no_caffeinate", False)),
                source_ui_page="/public/train",
            )
        elif mode == "raft":
            job_id = await self.training_service.launch_raft(
                model=str(payload.get("model") or ""),
                prompts=str(payload.get("prompts") or ""),
                output_dir=str(payload.get("output_dir") or ""),
                verifier=str(payload.get("verifier") or "humaneval"),
                cycles=int(self._value_or_default(payload.get("cycles"), 1)),
                samples_per_prompt=int(self._value_or_default(payload.get("samples_per_prompt"), 4)),
                temperature=float(self._value_or_default(payload.get("temperature"), 0.7)),
                keep_percent=float(self._value_or_default(payload.get("keep_percent"), 0.5)),
                reward_threshold=float(self._value_or_default(payload.get("reward_threshold"), 0.5)),
                min_samples=int(self._value_or_default(payload.get("min_samples"), 1)),
                max_new_tokens=int(self._value_or_default(payload.get("max_new_tokens"), 512)),
                no_caffeinate=bool(payload.get("no_caffeinate", False)),
                source_ui_page="/public/train",
            )
        elif mode in {"vlm", "audio", "reasoning", "agentic"}:
            job_id = await self.training_service.launch_modality_train(
                modality=mode,
                model=str(payload.get("model") or ""),
                dataset=str(payload.get("dataset") or ""),
                output_dir=str(payload.get("output_dir") or ""),
                cycles=int(payload.get("cycles") or 1),
                learning_rate=self._optional_float(payload.get("learning_rate")),
                lr_decay=self._optional_float(payload.get("lr_decay")),
                samples_per_prompt=self._optional_int(payload.get("samples_per_prompt")),
                temperature=self._optional_float(payload.get("temperature")),
                keep_percent=self._optional_float(payload.get("keep_percent")),
                reward_threshold=self._optional_float(payload.get("reward_threshold")),
                task=self._optional_str(payload.get("task")),
                limit=self._optional_int(payload.get("limit")),
                resume_from_cycle=int(payload.get("resume_from_cycle") or 0),
                seed=int(payload.get("seed") or 42),
                allow_prototype_train=bool(payload.get("allow_prototype_train", False)),
                no_caffeinate=bool(payload.get("no_caffeinate", False)),
                source_ui_page="/public/train",
            )
        else:
            raise ValueError(f"Unsupported training mode: {mode}")
        return self.get_run_detail(job_id, include_research=True, include_internal=False)

    def _sanitize_public_training_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        mode = str(payload.get("mode") or "").strip().lower()
        if mode not in TRAINING_MODALITIES:
            raise ValueError(f"Unsupported training mode: {mode}")

        allowed_fields = PUBLIC_TRAIN_ALLOWED_FIELDS[mode]
        unsupported_fields = sorted(
            key
            for key, value in payload.items()
            if key not in allowed_fields and self._has_public_value(value)
        )
        if unsupported_fields:
            raise ValueError(
                f"Unsupported fields for {mode}: {', '.join(unsupported_fields)}"
            )

        sanitized: Dict[str, Any] = {"mode": mode}
        for field_name in PUBLIC_TRAIN_REQUIRED_TEXT_FIELDS[mode]:
            text = str(payload.get(field_name) or "").strip()
            if not text:
                raise ValueError(f"{field_name} is required")
            sanitized[field_name] = text

        optional_fields = allowed_fields - {"mode"} - set(PUBLIC_TRAIN_REQUIRED_TEXT_FIELDS[mode])
        for field_name in sorted(optional_fields):
            value = payload.get(field_name)
            if self._has_public_value(value):
                sanitized[field_name] = value
        return sanitized

    async def apply_guided_recovery(
        self,
        run_identifier: str,
        *,
        resume_latest: bool = False,
    ) -> Dict[str, Any]:
        """Apply guided recovery using the stored launch context."""
        detail = self._resolve_run_source(run_identifier)
        launch_context = detail["launch_context_path"]
        recovery = detail["recovery"]
        if not launch_context:
            raise ValueError("This run does not have relaunch context.")
        if recovery.status != "ready":
            raise ValueError("Guided recovery is not available for this run.")

        job_id = await self.training_service.relaunch_from_context(
            launch_context,
            resume_latest=resume_latest,
            override_args=recovery.suggested_overrides,
            guided_recovery={
                "reason_code": recovery.reason_code,
                "evidence_summary": recovery.evidence_summary,
            },
            source_ui_page="/public/results",
        )
        return self.get_run_detail(job_id, include_research=True, include_internal=False)

    def search_runs(
        self,
        *,
        modalities: Optional[List[str]] = None,
        statuses: Optional[List[str]] = None,
        model_substring: Optional[str] = None,
        since_iso: Optional[str] = None,
        until_iso: Optional[str] = None,
        has_eval: Optional[bool] = None,
        weights_updated: Optional[bool] = None,
        sort_by: str = "timestamp",
        sort_dir: str = "desc",
        limit: Optional[int] = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """DB-backed run search (Track F-G commit 2).

        Lazily ensures the SQLite index is in sync with the filesystem,
        then queries it with the supplied filters. The existing
        ``list_runs`` surface keeps its filesystem-walk behavior so the
        run-list page is untouched; this endpoint is what the cohort /
        comparison / search-bar surfaces in the upcoming F-J / F-K
        items will target.

        Returns:
            ``{"items": [...], "total": N, "filters": {...},
              "facets": {"modalities": [...], "models": [...]}}``

            ``items`` is the paginated row list, ``total`` is the
            unpaginated match count, and ``facets`` is the distinct
            modality / model values present in the index — useful for
            the filter-chip UI without an extra round trip.
        """
        from halo_forge.run_db import RunFilter, get_database, sync_from_filesystem

        db = get_database()
        # Lazy sync. Cheap if the DB already mirrors the FS (incremental
        # by mtime); the first call after a fresh install pays the full
        # walk once.
        try:
            sync_from_filesystem(db)
        except Exception as exc:  # pragma: no cover - logged at runtime
            self._cached_backend_name = self._cached_backend_name  # touch attr to silence linters
            # Soft failure: serve what's already indexed rather than 5xx.
            # The sync is idempotent so the next call retries.
            import logging

            logging.getLogger(__name__).warning(
                "run_db sync failed; serving cached index: %s", exc
            )

        filt = RunFilter(
            modalities=list(modalities) if modalities else None,
            statuses=list(statuses) if statuses else None,
            model_substring=model_substring,
            since_iso=since_iso,
            until_iso=until_iso,
            has_eval=has_eval,
            weights_updated=weights_updated,
            sort_by=sort_by,
            sort_dir=sort_dir,
            limit=limit,
            offset=offset,
        )

        records = db.list_runs(filt)
        total = db.count_runs(filt)

        items = [self._db_record_to_list_item(record) for record in records]
        return {
            "items": items,
            "total": total,
            "filters": {
                "modalities": filt.modalities,
                "statuses": filt.statuses,
                "model_substring": filt.model_substring,
                "since_iso": filt.since_iso,
                "until_iso": filt.until_iso,
                "has_eval": filt.has_eval,
                "weights_updated": filt.weights_updated,
                "sort_by": filt.sort_by,
                "sort_dir": filt.sort_dir,
                "limit": filt.limit,
                "offset": filt.offset,
            },
            "facets": {
                "modalities": db.distinct_modalities(),
                "modality_counts": db.modality_counts(),
                "models": db.distinct_models(),
            },
        }

    def get_run_eval(self, run_identifier: str) -> Dict[str, Any]:
        """Return the lm_eval_summary.json for a run if present.

        Track F-K building block. Looks for `lm_eval_summary.json` inside
        the run's output_dir; honest unavailable shape on miss so the
        cohort dashboard can render a missing-eval column without 5xx.
        """
        try:
            source = self._resolve_run_source(run_identifier)
        except Exception as exc:
            return {
                "available": False,
                "reason": f"Run not found: {exc}",
                "tasks": [],
            }

        if source["kind"] == "summary":
            output_dir = source["summary"].output_dir
        else:
            job = source["job"]
            output_dir = Path(str(job.output_dir)) if job.output_dir else None

        if output_dir is None:
            return {
                "available": False,
                "reason": "Run has no output_dir to inspect",
                "tasks": [],
            }

        eval_path = Path(output_dir) / "lm_eval_summary.json"
        if not eval_path.exists():
            return {
                "available": False,
                "reason": f"No eval summary at {eval_path.name} — run "
                          f"`halo-forge eval --output {output_dir}` to populate.",
                "tasks": [],
            }

        try:
            data = json.loads(eval_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "available": False,
                "reason": f"Eval summary unreadable: {exc}",
                "tasks": [],
            }

        return {
            "available": True,
            "model_name": data.get("model_name"),
            "tasks": [
                {
                    "task": tr.get("task"),
                    "primary_metric": tr.get("primary_metric"),
                    "value": tr.get("value"),
                    "n_samples": tr.get("n_samples"),
                    "error": tr.get("error"),
                }
                for tr in data.get("task_results", [])
            ],
            "n_tasks_completed": data.get("n_tasks_completed"),
            "duration_seconds": data.get("duration_seconds"),
            "backend": data.get("backend"),
            "summary_path": str(eval_path),
        }

    def get_eval_cohort(
        self,
        run_ids: List[str],
    ) -> Dict[str, Any]:
        """Aggregate eval summaries across N runs into a cohort table.

        Track F-K. Returns ``{"runs": [{run_id, ...}], "tasks": [task_name],
        "cells": {run_id: {task: value}}}`` so the frontend can render
        a sortable runs-×-tasks grid without per-task fetching.

        Missing eval summaries surface as `available: False` on the run
        entry; the cohort table renders those rows with em-dashes.
        """
        run_entries: List[Dict[str, Any]] = []
        cells: Dict[str, Dict[str, Any]] = {}
        all_tasks: List[str] = []
        seen_tasks: set[str] = set()

        for raw_id in run_ids:
            run_id = str(raw_id or "").strip()
            if not run_id:
                continue
            eval_data = self.get_run_eval(run_id)

            entry = {
                "run_id": run_id,
                "available": eval_data.get("available", False),
                "reason": eval_data.get("reason"),
                "model_name": eval_data.get("model_name"),
                "duration_seconds": eval_data.get("duration_seconds"),
                "backend": eval_data.get("backend"),
            }
            run_entries.append(entry)

            cells[run_id] = {}
            for task in eval_data.get("tasks") or []:
                name = task.get("task")
                if not name:
                    continue
                if name not in seen_tasks:
                    seen_tasks.add(name)
                    all_tasks.append(name)
                cells[run_id][name] = {
                    "primary_metric": task.get("primary_metric"),
                    "value": task.get("value"),
                    "n_samples": task.get("n_samples"),
                    "error": task.get("error"),
                }

        # Per-task best so the UI can highlight winners. Higher is
        # better for accuracy-shaped metrics; lower is better for loss.
        # We can't always tell, so we surface both and let the UI decide
        # based on the metric name (acc / acc_norm / pass@1 / exact_match
        # all higher-is-better; metrics ending in _stderr or _loss don't
        # apply to this dashboard).
        best_per_task_high: Dict[str, Optional[str]] = {}
        for task in all_tasks:
            best_run, best_val = None, None
            for run_id in cells:
                cell = cells[run_id].get(task)
                if cell is None or cell.get("error"):
                    continue
                v = cell.get("value")
                if not isinstance(v, (int, float)):
                    continue
                if best_val is None or v > best_val:
                    best_val = v
                    best_run = run_id
            best_per_task_high[task] = best_run

        return {
            "runs": run_entries,
            "tasks": all_tasks,
            "cells": cells,
            "best_per_task_higher_is_better": best_per_task_high,
        }

    # ----- run stats (Track P3) -------------------------------------------

    def get_run_stats(self) -> Dict[str, Any]:
        """Aggregate counts for the Prometheus `/metrics` endpoint.

        Cheap to compute — single SQLite scan + a dict over the
        in-memory job table. Sub-millisecond on any reasonable run-DB
        size.
        """
        from halo_forge.run_db import get_database

        db = get_database()
        by_modality: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        try:
            cur = db._conn.execute(  # noqa: SLF001 — internal optimization, intentional
                "SELECT modality, status, COUNT(*) AS c FROM runs GROUP BY modality, status"
            )
            for row in cur.fetchall():
                modality = str(row["modality"] or "unknown")
                status = str(row["status"] or "unknown")
                count = int(row["c"])
                by_modality[modality] = by_modality.get(modality, 0) + count
                by_status[status] = by_status.get(status, 0) + count
        except Exception:
            # Empty DB / first call before any sync — leave dicts empty.
            pass

        total = sum(by_modality.values())

        # Active runs come from the in-memory job table; the DB
        # doesn't track runs that are still streaming.
        active = sum(
            1 for job in self.app_state.jobs.values()
            if job.status in {"pending", "running"}
        )

        return {
            "total_runs": total,
            "by_modality": by_modality,
            "by_status": by_status,
            "active_runs": active,
        }

    # ----- playground proxy (Track F-S) -------------------------------------

    def playground_chat(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        serve_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 120.0,
    ) -> Dict[str, Any]:
        """Forward a chat request to a `halo-forge serve`-style endpoint.

        Avoids CORS by routing through the public API; lets the frontend
        chat UI hit any OpenAI-compatible endpoint (local serve, remote
        host, hosted API) under one auth + origin model. Returns the
        upstream response body verbatim so the UI gets the OpenAI shape
        it expects.

        Defaults the serve URL to `http://127.0.0.1:8001/v1` — exactly
        what `halo-forge serve` exposes locally.
        """
        import os
        import httpx

        resolved_url = (
            serve_url
            or os.environ.get("HALOFORGE_PLAYGROUND_BASE_URL")
            or "http://127.0.0.1:8001/v1"
        )
        resolved_key = (
            api_key
            or os.environ.get("HALOFORGE_PLAYGROUND_API_KEY")
            or "EMPTY"
        )

        body: Dict[str, Any] = {
            "model": model or "halo-forge",
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        if stop:
            body["stop"] = list(stop)

        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(
                f"{resolved_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {resolved_key}"},
                json=body,
            )
            # Pass upstream errors through so the UI can render the
            # actual problem (model not loaded, OOM, etc.) instead of
            # a generic 500.
            if resp.status_code >= 400:
                try:
                    detail = resp.json()
                except Exception:
                    detail = {"error": resp.text}
                return {"upstream_error": True, "status": resp.status_code, "detail": detail}
            return resp.json()

    # ----- run lineage (Track F-Q) -----------------------------------------

    def get_run_lineage(self, run_id: str) -> Dict[str, Any]:
        from halo_forge.run_db import get_database

        db = get_database()
        return db.get_lineage(run_id)

    def record_run_fork(
        self,
        *,
        child_run_id: str,
        parent_run_id: str,
        forked_at_cycle: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        from halo_forge.run_db import get_database

        db = get_database()
        db.record_fork(
            child_run_id=child_run_id,
            parent_run_id=parent_run_id,
            forked_at_cycle=forked_at_cycle,
            notes=notes,
        )
        return db.get_lineage(child_run_id)

    def remove_run_fork(
        self, *, child_run_id: str, parent_run_id: str,
    ) -> bool:
        from halo_forge.run_db import get_database

        db = get_database()
        return db.remove_fork(
            child_run_id=child_run_id, parent_run_id=parent_run_id,
        )

    # ----- model registry (Track F-J) ---------------------------------------

    def list_registry_entries(self) -> List[Dict[str, Any]]:
        from halo_forge.run_db import get_database

        db = get_database()
        return [e.to_dict() for e in db.list_registry_entries()]

    def get_registry_entry(self, entry_id: int) -> Optional[Dict[str, Any]]:
        from halo_forge.run_db import get_database

        db = get_database()
        entry = db.get_registry_entry(entry_id)
        return entry.to_dict() if entry else None

    def create_registry_entry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from halo_forge.run_db import get_database

        db = get_database()
        entry = db.create_registry_entry(
            name=str(payload.get("name") or "").strip(),
            description=payload.get("description"),
            base_model=payload.get("base_model"),
            run_ids=payload.get("run_ids") or [],
            tags=payload.get("tags") or [],
        )
        return entry.to_dict()

    def update_registry_entry(
        self, entry_id: int, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        from halo_forge.run_db import get_database

        db = get_database()
        # Only forward the keys the user actually sent so missing keys
        # mean "leave alone", not "set to None".
        kwargs: Dict[str, Any] = {}
        for k in ("description", "base_model", "run_ids", "tags"):
            if k in payload:
                kwargs[k] = payload[k]
        entry = db.update_registry_entry(entry_id, **kwargs)
        return entry.to_dict() if entry else None

    def delete_registry_entry(self, entry_id: int) -> bool:
        from halo_forge.run_db import get_database

        db = get_database()
        return db.delete_registry_entry(entry_id)

    def _db_record_to_list_item(self, record) -> Dict[str, Any]:
        """Project a `RunRecord` to the list-item dict shape the
        frontend already consumes from /runs.

        Keeps the wire shape stable across the two endpoints so the
        run-list components don't have to branch on which fetched them.
        """
        return {
            "id": record.fs_id or record.run_id,
            "run_id": record.run_id,
            "modality": record.modality,
            "model_name": record.model_name,
            "status": record.status,
            "timestamp": record.timestamp,
            "cycles_executed": record.cycles_executed,
            "weights_updated": record.weights_updated,
            "final_train_loss": record.final_train_loss,
            "effectiveness": (
                {"verdict": record.effectiveness_verdict}
                if record.effectiveness_verdict
                else None
            ),
            "quality_status": record.quality_status,
            "keep_rate": record.keep_rate,
            "top_issue": record.dominant_rejection_reason,
            "output_dir": record.output_dir,
        }

    def list_runs(
        self,
        *,
        include_completed: bool = True,
        active_only: bool = False,
        include_research: bool = False,
    ) -> Dict[str, Any]:
        """List training runs for public monitor and results pages."""
        items: list[TrainingRunListItemView] = []
        seen_keys: set[str] = set()

        summaries = self.results_service.list_training_runs(force_refresh=True) if include_completed else []
        for summary in summaries:
            item = self._summary_to_list_item(summary, include_research=include_research)
            items.append(item)
            seen_keys.add(str(summary.output_dir.resolve()))

        if not include_completed or active_only:
            items = []

        active_jobs = sorted(
            [job for job in self.app_state.jobs.values() if job.type in TRAINING_MODALITIES],
            key=lambda job: job.created_at,
            reverse=True,
        )
        for job in active_jobs:
            output_key = str(job.output_dir.resolve()) if job.output_dir else ""
            if job.status == "completed" and output_key in seen_keys:
                continue
            if active_only and job.status not in {"pending", "running"}:
                continue
            items.append(self._job_to_list_item(job, include_research=include_research))

        items.sort(key=lambda item: item.timestamp, reverse=True)
        return {"items": [to_dict(item) for item in items]}

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Return the workstation dashboard summary."""
        readiness = self.list_readiness()
        active_rows = [
            self._to_active_row(self._job_to_list_item(job, include_research=False))
            for job in sorted(
                [job for job in self.app_state.jobs.values() if job.type in TRAINING_MODALITIES and job.status in {"pending", "running"}],
                key=lambda current: current.created_at,
                reverse=True,
            )
        ]
        completed_runs = [
            self._summary_to_list_item(summary, include_research=False)
            for summary in self.results_service.list_training_runs(force_refresh=True)
        ]
        completed_runs.sort(key=lambda item: item.timestamp, reverse=True)
        attention_source = active_rows[:]
        attention_source.extend(
            self._to_active_row(item)
            for item in completed_runs
            if item.user_summary.confidence_tone in {"warning", "danger"}
        )
        attention_items = [
            AttentionItemView(
                id=row.id,
                modality=row.modality,
                headline=row.headline,
                why_it_matters=row.metrics_summary.eval_metric_name or row.next_step,
                next_step=row.next_step,
                confidence_tone=row.primary_action.tone if row.primary_action else "warning",
                primary_action=row.primary_action,
            )
            for row in attention_source[:5]
        ]
        dashboard = DashboardSummaryView(
            readiness_tier=str(readiness.get("aggregate_tier") or "experimental"),
            generated_at=readiness.get("generated_at"),
            active_runs_count=len(active_rows),
            attention_count=len(attention_items),
            production_ready_count=sum(
                1 for item in readiness.get("items", []) if bool(item.get("production_ready"))
            ),
            modality_count=len(readiness.get("items", [])),
            active_runs=active_rows[:6],
            attention_items=attention_items,
            recent_outcomes=completed_runs[:6],
        )
        return to_dict(dashboard)

    def get_run_detail(
        self,
        run_identifier: str,
        *,
        include_research: bool = True,
        include_internal: bool = False,
    ) -> Dict[str, Any]:
        """Resolve a run from active job state or persisted training summaries."""
        resolved = self._resolve_run_source(run_identifier)
        if resolved["kind"] == "job":
            view = self._job_to_detail_view(
                resolved["job"],
                include_research=include_research,
                include_internal=include_internal,
            )
        else:
            view = self._summary_to_detail_view(
                resolved["summary"],
                include_research=include_research,
                include_internal=include_internal,
            )
        return to_dict(view)

    def get_run_live(
        self,
        run_identifier: str,
        *,
        include_research: bool = True,
    ) -> Dict[str, Any]:
        """Return a polling-friendly live view for a run."""
        resolved = self._resolve_run_source(run_identifier)
        if resolved["kind"] == "job":
            return to_dict(
                self._job_to_live_view(
                    resolved["job"],
                    include_research=include_research,
                )
            )

        summary = resolved["summary"]
        detail = self._summary_to_detail_view(
            summary,
            include_research=include_research,
            include_internal=False,
        )
        return to_dict(
            TrainingRunLiveView(
                id=detail.id,
                status=detail.status,
                progress_percent=100.0 if detail.status == "completed" else 0.0,
                current_step=int(detail.details.get("update_steps") or 0),
                total_steps=int(detail.details.get("update_steps") or 0),
                current_epoch=0.0,
                total_epochs=0,
                current_cycle=int(detail.details.get("cycles_executed") or 0),
                total_cycles=int(detail.details.get("cycles_executed") or 0),
                latest_loss=detail.details.get("final_train_loss"),
                latest_learning_rate=None,
                latest_grad_norm=None,
                headline=detail.headline,
                next_step=detail.next_step,
                top_issue=detail.top_issue,
                user_summary=detail.user_summary,
                metrics_summary=detail.metrics_summary,
                primary_action=detail.primary_action,
                research_sections=detail.research_sections,
            )
        )

    async def stream_run(self, run_identifier: str, *, include_research: bool = True):
        """Stream polling snapshots as server-sent events."""
        while True:
            payload = self.get_run_live(run_identifier, include_research=include_research)
            yield f"data: {json.dumps(payload)}\n\n"
            status = str(payload.get("status") or "").lower()
            if status in {"completed", "failed", "stopped"}:
                break
            await asyncio.sleep(1.0)

    async def stream_telemetry(self, *, interval_seconds: float = 2.0):
        """Push hardware telemetry as server-sent events.

        Replaces the 3s polling on the public_app's TelemetryStrip.
        Yields one `data: <json>\\n\\n` event per `interval_seconds`.

        Each event is the same shape `GET /api/public/telemetry` would
        return — the frontend EventSource can parse it identically and
        feed it into the same render path. The provider's own internal
        cache (1s on rocm-smi/nvidia-smi) keeps the actual subprocess
        cost bounded regardless of how aggressive the interval is.

        Streams until the client disconnects (FastAPI raises
        asyncio.CancelledError, which propagates out cleanly).
        """
        from halo_forge.telemetry import (
            TelemetryUnavailableError,
            get_telemetry_provider,
        )

        try:
            provider = get_telemetry_provider()
        except TelemetryUnavailableError as exc:
            # Emit a single error event then exit so the client gets
            # a clear signal instead of a silent hang.
            yield f"data: {json.dumps({'error': f'Telemetry unavailable: {exc}'})}\n\n"
            return

        # SSE retry hint — if the connection drops, the browser will
        # re-open after this many milliseconds. Keep it short so a
        # network blip doesn't leave the strip stale for long.
        yield f"retry: 3000\n\n"

        while True:
            try:
                sample = provider.sample()
                yield f"data: {json.dumps(sample.to_dict())}\n\n"
            except Exception as exc:
                # Don't crash the stream on a single sample failure;
                # surface the error in the event payload and keep
                # the connection alive so the next interval recovers.
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            await asyncio.sleep(max(0.5, float(interval_seconds)))

    async def stream_run_logs(
        self,
        run_identifier: str,
        *,
        initial_tail: int = 200,
        poll_seconds: float = 1.0,
    ):
        """Tail a run's log file and emit new lines as SSE events.

        The first event carries the `initial_tail` last lines so the
        frontend renders content immediately; subsequent events carry
        only newly-appended lines. Each event payload is
        `{"lines": [...], "log_path": "...", "appended_at": ts}`.

        Stops cleanly when the run reaches a terminal status (the file
        won't grow further) or when the client disconnects.
        """
        try:
            source = self._resolve_run_source(run_identifier)
        except Exception as exc:
            yield f"data: {json.dumps({'error': f'Run not found: {exc}'})}\n\n"
            return

        from pathlib import Path

        out_dir, run_id = _extract_output_dir_and_run_id(source)
        log_path: Optional[Path] = None
        if out_dir:
            for name in ("run.log", "train.log", "training.log"):
                if (out_dir / name).exists():
                    log_path = out_dir / name
                    break

        if log_path is None:
            # Fall back to logs/ scan once at start; if nothing matches,
            # emit a single "unavailable" event and exit.
            logs_dir = self.base_path / "logs"
            if logs_dir.is_dir():
                tokens = [t for t in (run_id, out_dir.name if out_dir else "") if t]
                matches = [
                    p
                    for p in logs_dir.glob("*.log")
                    if any(tok in p.name for tok in tokens)
                ]
                matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                if matches:
                    log_path = matches[0]

        if log_path is None or not log_path.exists():
            yield f"data: {json.dumps({'error': 'No log file alongside this run.'})}\n\n"
            return

        yield "retry: 3000\n\n"

        # Send the initial tail up front so the user sees something
        # without waiting for new lines.
        try:
            with log_path.open(encoding="utf-8", errors="replace") as f:
                buf: list[str] = []
                for line in f:
                    buf.append(line.rstrip("\n"))
                    if len(buf) > initial_tail * 2:
                        buf = buf[-initial_tail:]
                initial = buf[-initial_tail:] if len(buf) > initial_tail else buf
        except OSError as exc:
            yield f"data: {json.dumps({'error': f'Cannot read log: {exc}'})}\n\n"
            return

        yield f"data: {json.dumps({'lines': initial, 'log_path': str(log_path), 'reset': True})}\n\n"

        # Now follow the file for newly-appended bytes. We track byte
        # offset rather than line count so a partial line on disk
        # doesn't get duplicated when its remainder lands.
        offset = log_path.stat().st_size
        while True:
            try:
                size = log_path.stat().st_size
            except OSError:
                # File rotated/deleted — stop the stream gracefully.
                yield f"data: {json.dumps({'error': 'Log file disappeared'})}\n\n"
                return

            if size > offset:
                try:
                    with log_path.open(encoding="utf-8", errors="replace") as f:
                        f.seek(offset)
                        chunk = f.read(size - offset)
                except OSError:
                    chunk = ""
                if chunk:
                    new_lines = chunk.splitlines()
                    if new_lines:
                        yield f"data: {json.dumps({'lines': new_lines, 'log_path': str(log_path)})}\n\n"
                offset = size
            elif size < offset:
                # File was truncated — re-emit the head so the client
                # doesn't render stale state.
                offset = 0
                continue

            # If the underlying job is in a terminal state, stop here
            # so we don't hold the connection open forever after the
            # log stops growing. Re-resolve cheaply each iteration.
            try:
                refreshed = self._resolve_run_source(run_identifier)
                if refreshed.get("kind") == "summary":
                    # Completed run — give one final flush, then close.
                    await asyncio.sleep(poll_seconds)
                    return
            except Exception:
                pass

            await asyncio.sleep(max(0.25, float(poll_seconds)))

    def list_training_results(self, *, include_research: bool = False) -> Dict[str, Any]:
        """Return completed training results for the public results page."""
        items = [
            self._summary_to_list_item(summary, include_research=include_research)
            for summary in self.results_service.list_training_runs(force_refresh=True)
        ]
        items.sort(key=lambda item: item.timestamp, reverse=True)
        return {"items": [to_dict(item) for item in items]}

    def list_readiness(self) -> Dict[str, Any]:
        """Return public-safe readiness for training modalities."""
        report = self.readiness_service.load_qualification_report(force_refresh=True)
        items: list[ModalityReadinessView] = []
        counts = {"experimental": 0, "qualified": 0, "production_ready": 0}
        for module in TRAINING_MODALITIES:
            entry = report.modules.get(module)
            if entry is None:
                continue
            tier = str(getattr(entry, "readiness_tier", "") or "experimental")
            if tier in counts:
                counts[tier] += 1
            items.append(
                ModalityReadinessView(
                    modality=module,
                    readiness_tier=tier,
                    production_ready=bool(getattr(entry, "production_ready", False)),
                    status=str(getattr(entry, "status", "warn") or "warn"),
                    caveat=self._readiness_caveat(entry),
                    next_step=str(getattr(entry, "fix_now", "") or "Review readiness details before wider rollout."),
                    eval_metric_name=str(getattr(entry, "eval_metric_name", "") or ""),
                    baseline_value=getattr(entry, "baseline_value", None),
                    final_value=getattr(entry, "final_value", None),
                    delta=getattr(entry, "delta", None),
                    details={
                        "errors": list(getattr(entry, "errors", []) or []),
                        "warnings": list(getattr(entry, "warnings", []) or []),
                        "weights_updated": bool(getattr(entry, "weights_updated", False)),
                        "optimizer_steps": int(getattr(entry, "optimizer_steps", 0) or 0),
                        "samples_kept": int(getattr(entry, "samples_kept", 0) or 0),
                    },
                )
            )
        aggregate_tier = "experimental"
        if counts["production_ready"] == len(items) and items:
            aggregate_tier = "production_ready"
        elif counts["production_ready"] > 0 or counts["qualified"] > 0:
            aggregate_tier = "qualified"
        return {
            "generated_at": getattr(report, "generated_at", None),
            "aggregate_tier": aggregate_tier,
            "items": [to_dict(item) for item in items],
        }

    def list_docs_capabilities(self) -> Dict[str, Any]:
        """Return curated docs summaries for the public docs page."""
        items: list[DocsCapabilitySummaryView] = []
        for source in self.DOC_SOURCES:
            if not source.path.exists():
                continue
            title, summary = self._markdown_title_and_summary(source.path)
            items.append(
                DocsCapabilitySummaryView(
                    slug=source.slug,
                    title=title,
                    summary=summary,
                    source_path=str(source.path),
                    doc_url=source.doc_url,
                    audience=source.audience,
                )
            )
        return {"items": [to_dict(item) for item in items]}

    def _resolve_run_source(self, run_identifier: str) -> Dict[str, Any]:
        identifier = str(run_identifier or "").strip()
        job = self.app_state.get_job(identifier)
        if job is not None and job.type in TRAINING_MODALITIES:
            return {
                "kind": "job",
                "job": job,
                "launch_context_path": str(job.launch_context_file) if job.launch_context_file else None,
                "recovery": self._job_recovery(job),
            }

        summaries = self.results_service.list_training_runs(force_refresh=True)
        for summary in summaries:
            if identifier in {
                summary.id,
                str(summary.run_id or ""),
                summary.output_dir.name,
            }:
                return {
                    "kind": "summary",
                    "summary": summary,
                    "launch_context_path": str(summary.launch_context_path) if summary.launch_context_path else None,
                    "recovery": self._summary_recovery(summary),
                }
        raise KeyError(f"Training run not found: {identifier}")

    def _build_preflight_view(
        self,
        *,
        mode: str,
        preflight: TrainingLaunchPreflight,
    ) -> TrainingLaunchPreflightView:
        outlook = dict(preflight.quality_outlook or {})
        launch_presentation = build_launch_presentation(
            mode_label=mode.upper(),
            quality_status=str(outlook.get("status") or "healthy"),
            quality_summary=str(outlook.get("summary") or ""),
            suggested_adjustments=[
                str(item)
                for item in outlook.get("suggested_adjustments", [])
                if item is not None
            ],
            yield_safety_note=str(outlook.get("yield_safety_note") or ""),
        )
        return TrainingLaunchPreflightView(
            mode=mode,
            ok=bool(preflight.ok),
            resolved_paths=dict(preflight.resolved_paths),
            errors=list(preflight.errors),
            warnings=list(preflight.warnings),
            suggested_fixes=list(preflight.suggested_fixes),
            user_summary=ProductUserSummaryView(
                headline=launch_presentation.headline_status,
                why_it_matters=launch_presentation.supporting_summary,
                next_step=(
                    "Fix required inputs before launch"
                    if preflight.errors
                    else "Launch run when ready"
                ),
                confidence_tone=launch_presentation.confidence_tone,
            ),
            details={
                "quality_outlook": outlook,
                "recommended_adjustment": launch_presentation.recommended_adjustment,
            },
        )

    def _summary_to_list_item(
        self,
        summary: TrainingRunSummary,
        *,
        include_research: bool,
    ) -> TrainingRunListItemView:
        status = "failed" if summary.failure_reason else "completed"
        user_summary = build_user_summary(
            job_status=status,
            quality_status=summary.quality_status,
            quality_summary=summary.quality_summary,
            recovery_status=summary.recovery_status,
            recovery_action=summary.recovery_recommended_action,
            recovery_summary=summary.recovery_summary,
            failure_reason=summary.failure_reason,
            final_reason=summary.final_update_reason,
            has_launch_context=summary.has_relaunch_context,
            can_resume_latest=summary.modality in {"raft", "vlm", "audio", "reasoning", "agentic"},
            weights_updated=summary.weights_updated,
        )
        metrics_summary = self._metrics_summary(
            progress_percent=100.0 if status == "completed" else 0.0,
            keep_rate=summary.keep_rate,
            update_steps=summary.total_train_steps_executed,
            final_train_loss=summary.final_train_loss,
            effectiveness=dict(summary.raw_data.get("effectiveness") or {}),
        )
        research_sections = (
            self._build_research_sections(
                yield_diagnostics=summary.yield_diagnostics,
                effectiveness=dict(summary.raw_data.get("effectiveness") or {}),
                recovery=self._summary_recovery(summary),
                representative_examples=list(summary.representative_examples),
                lineage={
                    "run_id": summary.run_id,
                    "resume_from_cycle": summary.resume_from_cycle,
                    "final_model_available": bool(summary.final_model_path),
                },
            )
            if include_research
            else []
        )
        return TrainingRunListItemView(
            id=summary.id,
            run_id=str(summary.run_id or summary.id),
            modality=summary.modality,
            model_name=summary.model_name,
            status=status,
            timestamp=self._isoformat(summary.timestamp),
            headline=user_summary.headline,
            next_step=user_summary.next_step,
            top_issue=summary.dominant_rejection_reason,
            user_summary=user_summary,
            metrics_summary=metrics_summary,
            primary_action=user_summary.primary_action,
            details={
                "verdict": summary.effectiveness_verdict,
                "keep_rate": summary.keep_rate,
                "quality_status": summary.quality_status,
                "top_issue": summary.dominant_rejection_reason,
                "update_steps": summary.total_train_steps_executed,
                "final_train_loss": summary.final_train_loss,
            },
            research_sections=research_sections,
        )

    def _summary_to_detail_view(
        self,
        summary: TrainingRunSummary,
        *,
        include_research: bool,
        include_internal: bool,
    ) -> TrainingRunDetailView:
        item = self._summary_to_list_item(summary, include_research=include_research)
        recovery = self._summary_recovery(summary)
        return TrainingRunDetailView(
            id=item.id,
            run_id=item.run_id,
            modality=item.modality,
            model_name=item.model_name,
            status=item.status,
            timestamp=item.timestamp,
            headline=item.headline,
            next_step=item.next_step,
            top_issue=item.top_issue,
            user_summary=item.user_summary,
            metrics_summary=item.metrics_summary,
            recovery=recovery,
            primary_action=item.primary_action,
            details={
                **item.details,
                "cycles_executed": summary.cycles_executed,
                "seed": summary.seed,
                "resume_from_cycle": summary.resume_from_cycle,
                "final_model_available": bool(summary.final_model_path),
                # Phase D: per-cycle metric series for the live run view.
                # Flat plot-friendly entries so the frontend chart code
                # can hand them straight to recharts without re-shaping.
                "cycle_metrics": _project_cycles_for_charts(summary.raw_data),
                "cycle_losses": list(summary.cycle_losses),
                "yield_diagnostics": summary.yield_diagnostics,
                # Track P2 — energy/cost rollup. Estimated from wall-clock
                # + active backend's nominal training power; `source` flags
                # the provenance for the UI.
                "cost": _project_run_cost(
                    summary.raw_data,
                    backend_name=self._active_backend_name(),
                ),
            },
            research_sections=item.research_sections,
            internal_details=(
                {
                    "output_dir": str(summary.output_dir),
                    "final_model_path": summary.final_model_path,
                    "launch_context_path": (
                        str(summary.launch_context_path)
                        if summary.launch_context_path
                        else None
                    ),
                }
                if include_internal
                else {}
            ),
        )

    def _job_to_list_item(
        self,
        job: JobState,
        *,
        include_research: bool,
    ) -> TrainingRunListItemView:
        live_yield = dict(job.latest_yield_snapshot or {})
        yield_summary = live_yield.get("summary") if isinstance(live_yield.get("summary"), dict) else {}
        yield_rates = live_yield.get("rates") if isinstance(live_yield.get("rates"), dict) else {}
        recovery = self._job_recovery(job)
        user_summary = build_user_summary(
            job_status=job.status,
            quality_status=str(yield_summary.get("status") or ""),
            quality_summary=str(yield_summary.get("text") or ""),
            recovery_status=recovery.status,
            recovery_action=recovery.recommended_action,
            recovery_summary=recovery.evidence_summary,
            failure_reason=job.error_message,
            final_reason=job.lifecycle_metadata.get("resume_strategy") if job.lifecycle_metadata else "",
            has_launch_context=bool(job.launch_context_file),
            can_resume_latest=job.type in {"raft", "vlm", "audio", "reasoning", "agentic"},
            weights_updated=job.current_step > 0 or job.current_cycle > 0,
        )
        metrics_summary = self._metrics_summary(
            progress_percent=job.progress_percent,
            keep_rate=self._coerce_float(yield_rates.get("keep_rate")),
            update_steps=job.current_step,
            final_train_loss=job.latest_loss,
            effectiveness={},
        )
        research_sections = (
            self._build_research_sections(
                yield_diagnostics=live_yield,
                effectiveness={},
                recovery=recovery,
                representative_examples=list(recovery.representative_examples),
                lineage={
                    "run_id": job.id,
                    "current_epoch": job.current_epoch,
                    "current_cycle": job.current_cycle,
                    "output_dir": str(job.output_dir) if job.output_dir else "",
                },
            )
            if include_research
            else []
        )
        return TrainingRunListItemView(
            id=job.id,
            run_id=job.id,
            modality=job.type,
            model_name=job.name,
            status=job.status,
            timestamp=self._isoformat(job.created_at),
            headline=user_summary.headline,
            next_step=user_summary.next_step,
            top_issue=(
                str(yield_summary.get("dominant_rejection_reason"))
                if yield_summary.get("dominant_rejection_reason") not in (None, "")
                else None
            ),
            user_summary=user_summary,
            metrics_summary=metrics_summary,
            primary_action=user_summary.primary_action,
            details={
                "quality_status": yield_summary.get("status"),
                "keep_rate": yield_rates.get("keep_rate"),
                "top_issue": yield_summary.get("dominant_rejection_reason"),
                "update_steps": job.current_step,
                "final_train_loss": job.latest_loss,
            },
            research_sections=research_sections,
        )

    def _job_to_detail_view(
        self,
        job: JobState,
        *,
        include_research: bool,
        include_internal: bool,
    ) -> TrainingRunDetailView:
        item = self._job_to_list_item(job, include_research=include_research)
        recovery = self._job_recovery(job)
        return TrainingRunDetailView(
            id=item.id,
            run_id=item.run_id,
            modality=item.modality,
            model_name=item.model_name,
            status=item.status,
            timestamp=item.timestamp,
            headline=item.headline,
            next_step=item.next_step,
            top_issue=item.top_issue,
            user_summary=item.user_summary,
            metrics_summary=item.metrics_summary,
            recovery=recovery,
            primary_action=item.primary_action,
            details={
                **item.details,
                "current_epoch": job.current_epoch,
                "total_epochs": job.total_epochs,
                "current_cycle": job.current_cycle,
                "total_cycles": job.total_cycles,
                "verification_rate": job.verification_rate,
            },
            research_sections=item.research_sections,
            internal_details=(
                {
                    "output_dir": str(job.output_dir) if job.output_dir else None,
                    "launch_context_path": (
                        str(job.launch_context_file) if job.launch_context_file else None
                    ),
                    "lifecycle_metadata": dict(job.lifecycle_metadata),
                }
                if include_internal
                else {}
            ),
        )

    def _job_to_live_view(
        self,
        job: JobState,
        *,
        include_research: bool,
    ) -> TrainingRunLiveView:
        detail = self._job_to_detail_view(
            job,
            include_research=include_research,
            include_internal=False,
        )
        return TrainingRunLiveView(
            id=detail.id,
            status=detail.status,
            progress_percent=job.progress_percent,
            current_step=job.current_step,
            total_steps=job.total_steps,
            current_epoch=job.current_epoch,
            total_epochs=job.total_epochs,
            current_cycle=job.current_cycle,
            total_cycles=job.total_cycles,
            latest_loss=job.latest_loss,
            latest_learning_rate=job.latest_lr,
            latest_grad_norm=job.latest_grad_norm,
            headline=detail.headline,
            next_step=detail.next_step,
            top_issue=detail.top_issue,
            user_summary=detail.user_summary,
            metrics_summary=detail.metrics_summary,
            primary_action=detail.primary_action,
            research_sections=detail.research_sections,
        )

    def _summary_recovery(self, summary: TrainingRunSummary) -> TrainingRecoveryView:
        return TrainingRecoveryView(
            status=str(summary.recovery_status or "unavailable"),
            reason_code=str(summary.recovery_reason_code or ""),
            recommended_action=str(summary.recovery_recommended_action or ""),
            evidence_summary=str(summary.recovery_summary or ""),
            suggested_overrides=dict(summary.recovery_suggested_overrides),
            representative_examples=list(summary.representative_examples),
        )

    def _job_recovery(self, job: JobState) -> TrainingRecoveryView:
        live_yield = dict(job.latest_yield_snapshot or {})
        guidance = build_recovery_guidance(
            modality=str(job.type or "unknown"),
            yield_diagnostics=live_yield,
            effectiveness={
                "verdict": "pass" if job.current_step > 0 else "warn",
                "reasons": [],
            },
            launch_args=dict(job.launch_args),
        )
        return TrainingRecoveryView(
            status=str(guidance.get("status") or "unavailable"),
            reason_code=str(guidance.get("reason_code") or ""),
            recommended_action=str(guidance.get("recommended_action") or ""),
            evidence_summary=str(guidance.get("evidence_summary") or ""),
            suggested_overrides=(
                dict(guidance.get("suggested_overrides"))
                if isinstance(guidance.get("suggested_overrides"), dict)
                else {}
            ),
            representative_examples=[
                dict(example)
                for example in guidance.get("representative_examples", [])
                if isinstance(example, dict)
            ],
        )

    def _readiness_caveat(self, entry: Any) -> str:
        if bool(getattr(entry, "production_ready", False)):
            return "Deterministic launch, updates, artifacts, resume, and eval checks are currently passing."
        warnings = [str(item) for item in getattr(entry, "warnings", []) or [] if str(item).strip()]
        errors = [str(item) for item in getattr(entry, "errors", []) or [] if str(item).strip()]
        if errors:
            return errors[0]
        if warnings:
            return warnings[0]
        return str(getattr(entry, "fix_now", "") or "Qualification evidence is incomplete for this modality.")

    def _markdown_title_and_summary(self, path: Path) -> tuple[str, str]:
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        title = path.stem.replace("-", " ").title()
        summary = "Documentation summary unavailable."
        index = 0
        if lines and lines[0].strip() == "---":
            index = 1
            while index < len(lines) and lines[index].strip() != "---":
                line = lines[index].strip()
                if line.startswith("title:"):
                    title = line.split(":", 1)[1].strip().strip('"')
                if line.startswith("description:"):
                    summary = line.split(":", 1)[1].strip().strip('"')
                index += 1
            index += 1
        for line in lines[index:]:
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                continue
            if stripped and not stripped.startswith(("-", "*", "|", "`")):
                summary = stripped
                break
        return title, summary

    def _build_research_sections(
        self,
        *,
        yield_diagnostics: Dict[str, Any],
        effectiveness: Dict[str, Any],
        recovery: TrainingRecoveryView,
        representative_examples: list[dict[str, Any]],
        lineage: Dict[str, Any],
    ) -> list[ResearchSectionView]:
        sections: list[ResearchSectionView] = []
        yield_summary = yield_diagnostics.get("summary") if isinstance(yield_diagnostics.get("summary"), dict) else {}
        yield_rates = yield_diagnostics.get("rates") if isinstance(yield_diagnostics.get("rates"), dict) else {}
        sections.append(
            ResearchSectionView(
                key="data_yield",
                title="Data yield",
                summary=str(yield_summary.get("text") or "Yield details unavailable."),
                items=[
                    {"label": "Quality", "value": yield_summary.get("status")},
                    {"label": "Keep rate", "value": yield_rates.get("keep_rate")},
                    {"label": "Top issue", "value": yield_summary.get("dominant_rejection_reason")},
                ],
            )
        )
        update_quality = effectiveness.get("update_quality") if isinstance(effectiveness.get("update_quality"), dict) else {}
        sections.append(
            ResearchSectionView(
                key="update_quality",
                title="Update quality",
                summary=str(effectiveness.get("verdict") or "No effectiveness verdict."),
                items=[
                    {"label": "Optimizer steps", "value": update_quality.get("optimizer_steps") or update_quality.get("train_steps_executed")},
                    {"label": "Final loss", "value": update_quality.get("final_train_loss") or update_quality.get("loss_delta")},
                    {"label": "Weights updated", "value": update_quality.get("weights_updated")},
                ],
            )
        )
        evaluation = effectiveness.get("evaluation") if isinstance(effectiveness.get("evaluation"), dict) else {}
        sections.append(
            ResearchSectionView(
                key="eval_outcome",
                title="Eval outcome",
                summary=str(evaluation.get("status") or evaluation.get("metric_name") or "Eval unavailable."),
                items=[
                    {"label": "Metric", "value": evaluation.get("metric_name")},
                    {"label": "Current", "value": evaluation.get("final_value")},
                    {"label": "Delta", "value": evaluation.get("delta")},
                ],
            )
        )
        sections.append(
            ResearchSectionView(
                key="recovery_reasoning",
                title="Recovery reasoning",
                summary=recovery.evidence_summary or "No guided recovery recommendation.",
                items=[
                    {"label": "Status", "value": recovery.status},
                    {"label": "Recommended action", "value": recovery.recommended_action},
                    {"label": "Reason", "value": recovery.reason_code},
                ],
            )
        )
        if representative_examples:
            sections.append(
                ResearchSectionView(
                    key="representative_examples",
                    title="Representative examples",
                    summary="Representative evidence from dropped or weak samples.",
                    items=[dict(example) for example in representative_examples[:3]],
                )
            )
        sections.append(
            ResearchSectionView(
                key="artifact_lineage",
                title="Artifact lineage",
                summary="Artifact and resume lineage for this run.",
                items=[
                    {"label": str(key).replace("_", " ").title(), "value": value}
                    for key, value in lineage.items()
                    if value not in (None, "", False)
                ],
            )
        )
        return sections

    def _metrics_summary(
        self,
        *,
        progress_percent: float,
        keep_rate: Optional[float],
        update_steps: int,
        final_train_loss: Optional[float],
        effectiveness: Dict[str, Any],
    ) -> RunMetricsSummaryView:
        evaluation = effectiveness.get("evaluation") if isinstance(effectiveness.get("evaluation"), dict) else {}
        return RunMetricsSummaryView(
            progress_percent=progress_percent,
            keep_rate=keep_rate,
            update_steps=update_steps,
            final_train_loss=final_train_loss,
            eval_metric_name=str(evaluation.get("metric_name") or ""),
            eval_metric_value=self._coerce_float(evaluation.get("final_value")),
            eval_delta=self._coerce_float(evaluation.get("delta")),
        )

    def _to_active_row(self, item: TrainingRunListItemView) -> ActiveRunRowView:
        return ActiveRunRowView(
            id=item.id,
            modality=item.modality,
            model_name=item.model_name,
            status=item.status,
            headline=item.headline,
            next_step=item.next_step,
            primary_action=item.primary_action,
            metrics_summary=item.metrics_summary,
        )

    @staticmethod
    def _isoformat(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        return int(value)

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        return float(value)

    @staticmethod
    def _value_or_default(value: Any, default: Any) -> Any:
        if value in (None, ""):
            return default
        return value

    @staticmethod
    def _optional_str(value: Any) -> Optional[str]:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _has_public_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            if value in (None, ""):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Phase D helpers — used by `_summary_to_detail_view` to surface chart-ready
# per-cycle data without leaking the entire training_summary.json across the
# wire. Defined at module scope so tests can exercise it without standing up
# the full PublicApiService.
# ---------------------------------------------------------------------------


def _project_run_cost(raw_data: Dict[str, Any], *, backend_name: str) -> Dict[str, Any]:
    """Roll up wall-clock + nominal-power into a cost estimate (Track P2).

    Sums `cycle_duration_seconds` across the run's cycles and hands them
    to `telemetry.cost.estimate_run_cost`. The backend name comes from
    the *currently active host* — the training_summary doesn't carry the
    backend at write time, so for completed runs displayed on a different
    host the cost is "what would this run cost *here*". Same-host case
    is accurate; cross-host is an honest estimate. The frontend renders
    the `source` field so users know it's an estimate, not a meter
    reading.
    """
    from halo_forge.telemetry.cost import estimate_run_cost

    duration = 0.0
    if isinstance(raw_data, dict):
        cycles = raw_data.get("cycles")
        if isinstance(cycles, list):
            for entry in cycles:
                if isinstance(entry, dict):
                    v = _coerce_optional_float(entry.get("cycle_duration_seconds"))
                    if v:
                        duration += v
    cost = estimate_run_cost(
        duration_seconds=duration,
        backend_name=backend_name or "unknown",
    )
    return cost.to_dict()


def _project_cycles_for_charts(raw_data: Dict[str, Any]) -> list[dict[str, Any]]:
    """Project the cycles array from a training_summary.json payload to a
    flat plot-friendly shape.

    The raw `cycles` list contains everything the trainer emitted, including
    `yield_diagnostics` sub-objects, which are not useful for charts and
    inflate the wire size. We extract just the scalar per-cycle metrics
    that the live run view actually charts: train/eval loss, reward
    averages, success rate, and sample counts.

    Tolerates missing fields (older trainers, partial summaries) by
    returning None for any absent value — the frontend renders gaps as
    breaks in the line, not as zeros.
    """
    if not isinstance(raw_data, dict):
        return []
    cycles = raw_data.get("cycles")
    if not isinstance(cycles, list):
        return []
    projected: list[dict[str, Any]] = []
    for entry in cycles:
        if not isinstance(entry, dict):
            continue
        projected.append(
            {
                "cycle": int(entry.get("cycle") or 0),
                "train_loss": _coerce_optional_float(entry.get("train_loss")),
                "initial_train_loss": _coerce_optional_float(entry.get("initial_train_loss")),
                "eval_loss": _coerce_optional_float(entry.get("eval_loss")),
                "avg_reward": _coerce_optional_float(entry.get("avg_reward")),
                "avg_kept_reward": _coerce_optional_float(entry.get("avg_kept_reward")),
                "success_rate": _coerce_optional_float(entry.get("success_rate")),
                "samples_seen": _coerce_optional_int(entry.get("samples_seen")),
                "samples_kept": _coerce_optional_int(entry.get("samples_kept")),
                "train_steps_executed": _coerce_optional_int(entry.get("train_steps_executed")),
                "cycle_duration_seconds": _coerce_optional_float(
                    entry.get("cycle_duration_seconds")
                ),
                "learning_rate": _coerce_optional_float(entry.get("learning_rate")),
            }
        )
    return projected


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if not (result != result) else None  # filter NaN


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_output_dir_and_run_id(
    source: Dict[str, Any],
) -> tuple[Optional["Path"], str]:
    """Normalize the (output_dir, run_id) pair across the two run-source
    flavors `_resolve_run_source` returns.

    For active jobs the output_dir lives on `job.output_dir` and the
    identifier is `job.id`. For completed summaries it's
    `summary.output_dir` (already a Path) and `summary.run_id` (or the
    summary id when run_id wasn't recorded). Both are mapped to the
    same shape so the logs/samples endpoints can read uniformly.
    """
    from pathlib import Path

    kind = source.get("kind")
    if kind == "job":
        job = source.get("job")
        out = getattr(job, "output_dir", None)
        out_path = Path(out) if out else None
        run_id = str(getattr(job, "id", "") or "")
        return out_path, run_id

    if kind == "summary":
        summary = source.get("summary")
        out = getattr(summary, "output_dir", None)
        out_path = Path(out) if out else None
        run_id = str(getattr(summary, "run_id", "") or getattr(summary, "id", "") or "")
        return out_path, run_id

    return None, ""
