#!/usr/bin/env python3
"""Training pipeline contract hardening regression tests."""

from types import SimpleNamespace

import pytest

from halo_forge.cli import _enforce_training_outcome_or_exit
from halo_forge.training_contracts import (
    build_cycle_summary,
    build_training_summary,
    normalize_update_metrics,
)


def test_training_contract_helpers_emit_stable_shapes():
    """Shared helpers should always produce canonical keys."""
    update = normalize_update_metrics({"weights_updated": False}, default_reason="no_filtered_samples")
    assert update == {
        "train_steps_executed": 0,
        "train_loss": None,
        "weights_updated": False,
        "update_reason": "no_filtered_samples",
        "optimizer_steps": 0,
        "skipped_batches_non_finite": 0,
    }

    cycle = build_cycle_summary(
        cycle=0,
        learning_rate=1e-5,
        samples_seen=12,
        samples_kept=3,
        cycle_duration_seconds=2.5,
        update_metrics=update,
        extra={"avg_reward": 0.4},
    )
    assert {
        "cycle",
        "learning_rate",
        "samples_seen",
        "samples_kept",
        "cycle_duration_seconds",
        "train_steps_executed",
        "train_loss",
        "weights_updated",
        "update_reason",
    }.issubset(cycle.keys())

    summary = build_training_summary(
        modality="reasoning",
        model_name="org/model",
        total_cycles_planned=2,
        cycles=[cycle],
    )
    assert summary["total_train_steps_executed"] == 0
    assert summary["weights_updated"] is False
    assert summary["final_update_reason"] == "no_filtered_samples"
    assert summary["failure_reason"] == "no_filtered_samples"


def test_reasoning_cycle_metrics_include_canonical_training_contract_keys(monkeypatch, tmp_path):
    """Reasoning cycle payload should include canonical update telemetry keys."""
    try:
        from halo_forge.reasoning.trainer import (
            ReasoningCompletion,
            ReasoningRAFTConfig,
            ReasoningRAFTTrainer,
        )
        from halo_forge.reasoning.data import MathSample
    except ModuleNotFoundError as e:
        if e.name and not e.name.startswith("halo_forge"):
            pytest.skip(f"optional dependency missing ({e.name})")
        raise

    trainer = ReasoningRAFTTrainer(
        ReasoningRAFTConfig(num_cycles=1, output_dir=str(tmp_path / "reasoning"))
    )
    sample = MathSample(question="1+1?", answer="2")
    completion = ReasoningCompletion(sample=sample, completion="2", reward=1.0, verified=True)

    monkeypatch.setattr(trainer, "generate_completions", lambda _: [completion])
    monkeypatch.setattr(trainer, "verify_completions", lambda completions: completions)
    monkeypatch.setattr(trainer, "filter_completions", lambda completions: completions)
    monkeypatch.setattr(
        trainer,
        "_train_on_filtered",
        lambda completions, cycle: {
            "train_steps_executed": 1,
            "train_loss": 0.2,
            "weights_updated": True,
            "update_reason": "updated",
        },
    )

    metrics = trainer.train_cycle([sample], 0)
    assert metrics["samples_seen"] == 1
    assert metrics["samples_kept"] == 1
    assert metrics["train_steps_executed"] == 1
    assert metrics["weights_updated"] is True


def test_audio_train_writes_canonical_training_summary(monkeypatch, tmp_path):
    """Audio train should persist canonical summary even though return type remains cycle list."""
    try:
        from halo_forge.audio.trainer import AudioRAFTConfig, AudioRAFTCycleResult, AudioRAFTTrainer
    except ModuleNotFoundError as e:
        if e.name and not e.name.startswith("halo_forge"):
            pytest.skip(f"optional dependency missing ({e.name})")
        raise

    trainer = AudioRAFTTrainer(
        AudioRAFTConfig(
            num_cycles=1,
            output_dir=str(tmp_path / "audio"),
            save_every_cycle=False,
        )
    )

    monkeypatch.setattr(trainer, "_init_adapter", lambda: None)
    monkeypatch.setattr(trainer, "_init_verifier", lambda: None)
    monkeypatch.setattr(
        trainer,
        "_train_cycle",
        lambda cycle, samples: AudioRAFTCycleResult(
            cycle=cycle,
            samples_generated=4,
            samples_verified=4,
            samples_kept=2,
            average_reward=0.5,
            learning_rate=1e-5,
            metrics=build_cycle_summary(
                cycle=cycle,
                learning_rate=1e-5,
                samples_seen=4,
                samples_kept=2,
                cycle_duration_seconds=0.5,
                update_metrics={
                    "train_steps_executed": 2,
                    "train_loss": 0.3,
                    "weights_updated": True,
                    "update_reason": "updated",
                },
                extra={"avg_reward": 0.5},
            ),
        ),
    )
    monkeypatch.setattr(trainer, "_save_checkpoint", lambda *args, **kwargs: None)

    results = trainer.train(samples=[object()])
    assert len(results) == 1
    summary = trainer.training_summary
    assert summary["modality"] == "audio"
    assert summary["weights_updated"] is True
    assert summary["total_train_steps_executed"] == 2
    assert (tmp_path / "audio" / "training_summary.json").exists()


def test_agentic_resume_requires_previous_checkpoint(tmp_path):
    """Agentic resume should fail fast when checkpoint lineage is missing."""
    try:
        from halo_forge.agentic.trainer import AgenticRAFTConfig, AgenticRAFTTrainer
    except ModuleNotFoundError as e:
        if e.name and not e.name.startswith("halo_forge"):
            pytest.skip(f"optional dependency missing ({e.name})")
        raise

    trainer = AgenticRAFTTrainer(
        AgenticRAFTConfig(num_cycles=3, output_dir=str(tmp_path / "agentic"))
    )
    trainer.model = object()
    trainer.tokenizer = object()

    with pytest.raises(ValueError, match="requires checkpoint"):
        trainer.train(samples=[], resume_from_cycle=1)


def test_vlm_resume_requires_history_and_checkpoint(tmp_path):
    """VLM resume should reject incomplete checkpoint/history state."""
    try:
        from halo_forge.vlm.trainer import VLMRAFTConfig, VLMRAFTTrainer
    except ModuleNotFoundError as e:
        if e.name and not e.name.startswith("halo_forge"):
            pytest.skip(f"optional dependency missing ({e.name})")
        raise

    trainer = VLMRAFTTrainer(
        VLMRAFTConfig(num_cycles=3, output_dir=str(tmp_path / "vlm"))
    )
    trainer._setup = lambda: None
    trainer.run_cycle = lambda prompts, cycle: {}

    with pytest.raises(ValueError, match="requires checkpoint"):
        trainer.train(prompts=[object()], resume_from=1)


def test_cli_enforces_non_zero_exit_when_no_training_updates():
    """CLI helper should hard-fail commands that never updated weights."""
    with pytest.raises(SystemExit) as exc:
        _enforce_training_outcome_or_exit(
            "reasoning",
            {
                "weights_updated": False,
                "final_update_reason": "no_filtered_samples",
                "total_train_steps_executed": 0,
            },
        )
    assert exc.value.code == 2
