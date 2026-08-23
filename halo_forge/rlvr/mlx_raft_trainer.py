"""Pure-MLX RAFT trainer (Phase 5b).

Where Phase 5a ran rollouts on MLX while keeping the policy update on
PyTorch, this trainer keeps everything on MLX:

  rollout (MLXRolloutGenerator)
    -> verify (cross-platform sandbox-exec verifiers, unchanged)
    -> filter (halo_forge.rlvr._shared.verify_and_filter_samples)
    -> SFT-on-accepted (MLXSFTTrainer)
    -> reload base + LoRA (handled inside MLXSFTTrainer per cycle)

Phase 5c will port the curriculum + recovery management layer from
`raft_trainer.py`. For now we run a fixed number of cycles with the
config's static `samples_per_prompt` / `temperature` / `learning_rate`,
which is the minimum useful surface.

Limits vs. PyTorch RAFT trainer (intentional, scoped):
- No per-cycle curriculum schedule (Phase 5c).
- No `_reload_model` ceremony — MLX has no PEFT-config accumulation
  problem because each cycle's MLXSFTTrainer instance loads the base from
  scratch via `mlx_lm.load`.
- No streaming-checkpoint resume mid-cycle (the rollout cache *does*
  resume; only the trainer's outer cycle loop restarts a partial cycle).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from halo_forge.rlvr._shared import (
    filtered_to_jsonl_records,
    print_filtering_summary,
    verify_and_filter_samples,
)


def _preserve_managed_cycle_state() -> bool:
    """Whether a bounded managed process must leave its resume sidecar."""

    return bool(os.environ.get("HALOFORGE_DIRECT_RUN_SEGMENT_ID")) and os.environ.get(
        "HALOFORGE_DIRECT_RUN_SEGMENT_FINAL", "1"
    ) != "1"
from halo_forge.rlvr.mlx_rollout import MLXRolloutGenerator, MLXRolloutUnavailable
from halo_forge.runtime_determinism import build_run_id, set_global_seed
from halo_forge.sft.config import SFTConfig
from halo_forge.training_contracts import (
    attach_effectiveness_contract,
    build_cycle_summary,
    build_training_summary,
    write_json_atomic,
)
from halo_forge.training_recovery import attach_recovery_guidance
from halo_forge.training_signal import complete_signal_boundary


class MLXRAFTUnavailable(RuntimeError):
    """Raised when MLX RAFT is requested but `[mlx]` extra isn't installed."""


class MLXRAFTTrainer:
    """RAFT loop running entirely on MLX.

    Args:
        verifier: same `Verifier` interface as the PyTorch RAFT trainer.
            The sandbox runner is already cross-platform (sandbox-exec on
            macOS), so verifier code does not need backend awareness.
        config: a `RAFTConfig` instance. Duck-typed — only attributes the
            shared helpers consult are required (`samples_per_prompt`,
            `verification_chunk_size`, `reward_threshold`, etc.).
        rollout_model: HF id / local path of MLX-format weights for both
            rollout and SFT. Defaults to `config.base_model` if it's
            already MLX-loadable (i.e. an `mlx-community/...` repo).
        adapter_dir: where to write LoRA adapters between cycles. Defaults
            to `config.output_dir/cycle_<n>_final`.

    Cross-cycle weight handling: MLXSFTTrainer loads the base from scratch
    each cycle, then applies LoRA. Cycle N+1 starts from cycle N's adapter
    if `chain_adapters` is True (default) — the trainer passes the previous
    cycle's `adapters.safetensors` path as `extra["adapter_file"]` on the
    next cycle's load.
    """

    def __init__(
        self,
        verifier: Any,
        config: Any,
        *,
        rollout_model: Optional[str] = None,
        sft_config: Optional[SFTConfig] = None,
        chain_adapters: bool = True,
        signal_sink: Optional[Any] = None,
    ) -> None:
        self.verifier = verifier
        self.signal_sink = signal_sink
        self.config = config
        self.rollout_model = rollout_model or getattr(config, "base_model", None)
        if not self.rollout_model:
            raise ValueError(
                "MLX RAFT requires a model name — pass rollout_model= or set "
                "config.base_model to an MLX-format weight set "
                "(e.g. mlx-community/Qwen2.5-3B-Instruct-bf16)."
            )

        self.output_dir = Path(getattr(config, "output_dir", "models/raft_mlx"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default SFT config inherits from the RAFT config's training fields
        # where they overlap; users can pass a fully-customized SFTConfig
        # for fine-grained control over the per-cycle SFT step.
        self.sft_config_template = sft_config or self._default_sft_config()
        self.chain_adapters = chain_adapters

        # Owned components
        self.rollout_generator = MLXRolloutGenerator(self.rollout_model)
        self.cycle_stats: List[Dict[str, Any]] = []
        self.run_id: str = ""
        self.training_summary: Dict[str, Any] = {}
        self.representative_examples: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, prompts: List[str], num_cycles: Optional[int] = None) -> Dict[str, Any]:
        """Run the RAFT loop and return the canonical training summary.

        Phase 5c: cycle-level resume. After each cycle the trainer writes a
        small `_cycle_state.json` sidecar with the cycle index, run_id, seed,
        and per-cycle summaries built so far. On re-launch with the same
        `output_dir`, the trainer picks up at the next cycle and reuses the
        previous adapter — useful for long runs where a single hardware
        glitch shouldn't redo days of work.

        Args:
            prompts: prompt list (raw user-message strings).
            num_cycles: override config.num_cycles.

        Returns:
            Same dict shape as `RAFTTrainer.run` so the public API and
            `_print_completed_training_summary` consume it identically.
        """
        cfg = self.config
        total_cycles = int(num_cycles or getattr(cfg, "num_cycles", 1))

        resumed = self._load_cycle_state()
        if resumed is not None:
            self.run_id = resumed["run_id"]
            seed = int(resumed["seed"])
            cycle_summaries: List[Dict[str, Any]] = list(resumed.get("cycle_summaries", []))
            start_cycle = int(resumed["next_cycle"])
            previous_adapter: Optional[Path] = (
                Path(resumed["previous_adapter"])
                if resumed.get("previous_adapter")
                else None
            )
            self.representative_examples = list(resumed.get("representative_examples", []))
            self._log(
                f"Resumed run {self.run_id}: starting at cycle {start_cycle + 1}/{total_cycles}",
                "success",
            )
        else:
            self.run_id = build_run_id("raft_mlx")
            seed = set_global_seed(getattr(cfg, "seed", 42))
            cycle_summaries = []
            start_cycle = 0
            previous_adapter = None

        run_start = time.time()

        for cycle in range(start_cycle, total_cycles):
            cycle_start = time.time()
            print(f"\n{'='*70}\nMLX RAFT cycle {cycle + 1} / {total_cycles}\n{'='*70}")

            # 1. Rollout (cached per-cycle so partial runs resume cleanly).
            cache_path = self.output_dir / f"cycle_{cycle}_samples.jsonl"
            samples = self.rollout_generator.generate_samples(
                prompts,
                num_samples=getattr(cfg, "samples_per_prompt", 8),
                max_new_tokens=getattr(cfg, "max_new_tokens", 512),
                temperature=self._temperature_for_cycle(cycle),
                batch_size=getattr(cfg, "generation_batch_size", 1),
                system_prompt=getattr(cfg, "system_prompt", ""),
                cache_path=cache_path,
                log=self._log,
            )

            # 2. Verify + filter (shared with PyTorch trainer).
            filtered, stats, reps = verify_and_filter_samples(
                samples,
                self.verifier,
                chunk_size=getattr(cfg, "verification_chunk_size", 200),
                reward_threshold=getattr(cfg, "reward_threshold", 0.5),
                keep_top_percent=getattr(cfg, "keep_top_percent", 0.5),
                min_samples_per_cycle=getattr(cfg, "min_samples_per_cycle", None),
                allow_compile_only_training=getattr(cfg, "allow_compile_only_training", False),
                progress_logger=self._log,
                signal_sink=self.signal_sink,
                signal_source={"cycle": cycle + 1, "backend": "mlx"},
                samples_per_prompt=getattr(cfg, "samples_per_prompt", 8),
                generation_settings={
                    "temperature": self._temperature_for_cycle(cycle),
                    "max_new_tokens": getattr(cfg, "max_new_tokens", 512),
                    "samples_per_prompt": getattr(cfg, "samples_per_prompt", 8),
                },
                producer_model_hash=self.rollout_model,
            )
            print_filtering_summary(stats)
            if not self.representative_examples:
                self.representative_examples = reps

            # 3. SFT on accepted samples — RAFT *is* repeated SFT.
            sft_summary = self._sft_on_filtered(
                filtered=filtered,
                cycle=cycle,
                previous_adapter=previous_adapter if self.chain_adapters else None,
            )
            cycle_adapter = Path(sft_summary.get("final_model_path", ""))
            if cycle_adapter.exists():
                previous_adapter = cycle_adapter

            # 4. Build cycle summary.
            cycle_summaries.append(self._build_cycle_summary(
                cycle=cycle,
                stats=stats,
                samples=samples,
                filtered=filtered,
                sft_summary=sft_summary,
                cycle_duration=time.time() - cycle_start,
            ))

            # 5. Phase 5c: persist cycle state so a re-launch resumes here.
            self._save_cycle_state(
                run_id=self.run_id,
                seed=seed,
                next_cycle=cycle + 1,
                cycle_summaries=cycle_summaries,
                previous_adapter=previous_adapter,
            )
            checkpoint_path = cycle_adapter if cycle_adapter.exists() else self.output_dir / f"cycle_{cycle}"
            complete_signal_boundary(
                self.signal_sink,
                boundary_value=cycle + 1,
                checkpoint_path=checkpoint_path,
            )

        run_duration = time.time() - run_start

        # A managed non-final audit segment is a deliberately bounded
        # successful process, not the end of the canonical run.  Preserve its
        # verified cycle-state sidecar so the next scheduler attempt resumes
        # at ``next_cycle`` instead of replaying cycles 1..N.  Normal/direct
        # launches and the final managed segment retain the historical cleanup.
        if not _preserve_managed_cycle_state():
            self._clear_cycle_state()

        summary = self._build_run_summary(
            cycle_summaries=cycle_summaries,
            seed=seed,
            total_cycles_planned=total_cycles,
            run_duration=run_duration,
            final_adapter=previous_adapter,
        )
        write_json_atomic(self.output_dir / "training_summary.json", summary)
        self.training_summary = summary
        return summary

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sft_on_filtered(
        self,
        *,
        filtered: List[Dict[str, Any]],
        cycle: int,
        previous_adapter: Optional[Path],
    ) -> Dict[str, Any]:
        """Run one MLX SFT cycle on the accepted samples.

        Materializes the filtered set as a JSONL and invokes
        `MLXSFTTrainer.train(train_file=...)`. Routes through the deferred
        import so a missing `[mlx]` extra raises one clean error per
        process instead of multiple stack traces.
        """
        from halo_forge.sft.mlx_trainer import MLXSFTTrainer

        cycle_dir = self.output_dir / f"cycle_{cycle}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        records = filtered_to_jsonl_records(
            filtered, system_prompt=getattr(self.config, "system_prompt", "")
        )
        sft_data = cycle_dir / "accepted.jsonl"
        with sft_data.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        sft_cfg = SFTConfig(**asdict(self.sft_config_template))
        sft_cfg.train_file = str(sft_data)
        sft_cfg.dataset = None
        sft_cfg.output_dir = str(cycle_dir / "sft")
        sft_cfg.model_name = self.rollout_model
        sft_cfg.learning_rate = self._learning_rate_for_cycle(cycle)

        # Phase 5c: chain from the previous cycle's adapter. MLXSFTTrainer
        # accepts an explicit `resume_adapter_path` argument that gets
        # forwarded to mlx_lm.load via the LoadSpec extras dict. Each
        # cycle resumes from the prior cycle's `adapters.safetensors` so
        # RAFT's "repeated SFT on filtered samples" actually accumulates
        # weight changes across cycles.
        resume_adapter: Optional[str] = None
        if previous_adapter is not None and previous_adapter.exists():
            print(f"[mlx-raft] chaining from previous adapter: {previous_adapter}")
            (cycle_dir / "previous_adapter.txt").write_text(str(previous_adapter))
            resume_adapter = str(previous_adapter)

        trainer = MLXSFTTrainer(sft_cfg)
        if resume_adapter is not None:
            trainer.resume_adapter_path = resume_adapter
        try:
            return trainer.train(train_file=str(sft_data))
        except Exception as exc:
            # Surface MLX-specific install failures with a clean message;
            # other exceptions bubble unchanged so the user sees the
            # original stack trace.
            from halo_forge.sft.mlx_trainer import MLXTrainerUnavailable

            if isinstance(exc, MLXTrainerUnavailable):
                raise MLXRAFTUnavailable(str(exc)) from exc
            raise

    def _temperature_for_cycle(self, cycle: int) -> float:
        """Honor a `get_temperature_for_cycle` config method if present;
        otherwise use the static `temperature`. Phase 5c will plug a
        proper curriculum schedule here.
        """
        getter = getattr(self.config, "get_temperature_for_cycle", None)
        if callable(getter):
            return float(getter(cycle))
        return float(getattr(self.config, "temperature", 0.7))

    def _learning_rate_for_cycle(self, cycle: int) -> float:
        getter = getattr(self.config, "get_learning_rate_for_cycle", None)
        if callable(getter):
            return float(getter(cycle))
        return float(getattr(self.config, "learning_rate", 1e-5))

    def _default_sft_config(self) -> SFTConfig:
        cfg = self.config
        return SFTConfig(
            model_name=self.rollout_model,
            num_epochs=int(getattr(cfg, "sft_epochs_per_cycle", 1)),
            batch_size=int(getattr(cfg, "sft_batch_size", 2)),
            gradient_accumulation_steps=int(getattr(cfg, "sft_gradient_accumulation", 8)),
            learning_rate=float(getattr(cfg, "learning_rate", 1e-5)),
            lora_r=int(getattr(cfg, "lora_r", 16)),
            lora_alpha=int(getattr(cfg, "lora_alpha", 32)),
            lora_dropout=float(getattr(cfg, "lora_dropout", 0.05)),
            output_dir=str(self.output_dir / "sft_default"),
            seed=int(getattr(cfg, "seed", 42)),
        )

    # ------------------------------------------------------------------
    # Cycle-state checkpoint (Phase 5c)
    # ------------------------------------------------------------------

    def _cycle_state_path(self) -> Path:
        return self.output_dir / "_cycle_state.json"

    def _load_cycle_state(self) -> Optional[Dict[str, Any]]:
        """Read the cycle-state sidecar if present.

        Returns None on the first launch (or after a successful run, when
        the sidecar has been cleared). Schema is internal — bumping it is
        fine, missing fields are handled by callers via .get with defaults.
        """
        path = self._cycle_state_path()
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            self._log(f"Could not read cycle state at {path}: {exc}; starting fresh", "warning")
            return None

    def _save_cycle_state(
        self,
        *,
        run_id: str,
        seed: int,
        next_cycle: int,
        cycle_summaries: List[Dict[str, Any]],
        previous_adapter: Optional[Path],
    ) -> None:
        state = {
            "run_id": run_id,
            "seed": seed,
            "next_cycle": next_cycle,
            "cycle_summaries": cycle_summaries,
            "previous_adapter": str(previous_adapter) if previous_adapter else None,
            "representative_examples": self.representative_examples,
        }
        write_json_atomic(self._cycle_state_path(), state)

    def _clear_cycle_state(self) -> None:
        path = self._cycle_state_path()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            self._log(f"Could not clear cycle state at {path}: {exc}", "warning")

    def _log(self, msg: str, level: str = "info") -> None:
        """Plain-text logging; matches RAFTTrainer._log shape so the
        rollout generator's `log` callback sees the same interface.
        """
        prefix = {
            "warning": "[!]",
            "success": "[OK]",
            "dim": "...",
            "error": "[X]",
        }.get(level, "")
        print(f"{prefix} {msg}".strip())

    # ------------------------------------------------------------------
    # Summary builders
    # ------------------------------------------------------------------

    def _build_cycle_summary(
        self,
        *,
        cycle: int,
        stats: Dict[str, Any],
        samples: List[Tuple[str, str]],
        filtered: List[Dict[str, Any]],
        sft_summary: Dict[str, Any],
        cycle_duration: float,
    ) -> Dict[str, Any]:
        cfg = self.config
        train_loss = sft_summary.get("final_train_loss")
        steps = int(sft_summary.get("total_train_steps_executed", 0) or 0)
        return build_cycle_summary(
            cycle=cycle,
            learning_rate=self._learning_rate_for_cycle(cycle),
            samples_seen=len(samples),
            samples_kept=len(filtered),
            cycle_duration_seconds=cycle_duration,
            update_metrics={
                "train_steps_executed": steps,
                "train_loss": train_loss,
                "weights_updated": steps > 0,
                "update_reason": "updated" if steps > 0 else "no_optimizer_steps",
                "optimizer_steps": steps,
                "skipped_batches_non_finite": 0,
            },
            extra={
                "backend": "mlx",
                "avg_reward": stats.get("avg_reward"),
                "avg_kept_reward": stats.get("avg_kept_reward"),
                "success_rate": stats.get("success_rate"),
                "effective_threshold": stats.get("effective_threshold"),
                "threshold_adjusted": stats.get("threshold_adjusted"),
            },
        )

    def _build_run_summary(
        self,
        *,
        cycle_summaries: List[Dict[str, Any]],
        seed: int,
        total_cycles_planned: int,
        run_duration: float,
        final_adapter: Optional[Path],
    ) -> Dict[str, Any]:
        cfg = self.config
        summary = build_training_summary(
            modality="raft",
            model_name=self.rollout_model,
            total_cycles_planned=total_cycles_planned,
            cycles=cycle_summaries,
            run_id=self.run_id,
            seed=seed,
            base_model_name=self.rollout_model,
            active_model_name=self.rollout_model,
            extra={
                "backend": "mlx",
                "run_duration_seconds": run_duration,
            },
        )
        if final_adapter is not None:
            summary["final_model_path"] = str(final_adapter)
        attach_effectiveness_contract(
            summary,
            minimum_samples_kept=1,
            minimum_optimizer_steps=1,
            evaluation={
                "metric_name": "avg_kept_reward",
                "baseline_value": cycle_summaries[0].get("avg_kept_reward") if cycle_summaries else None,
                "final_value": cycle_summaries[-1].get("avg_kept_reward") if cycle_summaries else None,
                "higher_is_better": True,
                "tolerance": 0.0,
            },
            evaluation_required=False,
            checkpoint_written=final_adapter is not None and final_adapter.exists(),
            final_model_path=str(final_adapter) if final_adapter else "",
            training_summary_path=self.output_dir / "training_summary.json",
        )
        attach_recovery_guidance(
            summary,
            modality="raft",
            launch_args={
                "model": self.rollout_model,
                "output_dir": str(self.output_dir),
                "cycles": total_cycles_planned,
                "samples_per_prompt": getattr(cfg, "samples_per_prompt", 8),
                "reward_threshold": getattr(cfg, "reward_threshold", 0.5),
                "backend": "mlx",
            },
            representative_examples=self.representative_examples,
        )
        return summary


__all__ = [
    "MLXRAFTTrainer",
    "MLXRAFTUnavailable",
]
