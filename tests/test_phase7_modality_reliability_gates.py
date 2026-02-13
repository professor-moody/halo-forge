#!/usr/bin/env python3
"""Phase 7A modality reliability gate tests."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from halo_forge.modality_artifacts import resolve_resume_checkpoint
from halo_forge.training_contracts import build_training_summary
from halo_forge.training_updates import run_text_supervised_updates
from ui.services.training_service import TrainingService
from ui.state import AppState


def test_cli_modality_parsers_expose_seed_flag():
    """Each modality train parser should expose deterministic seed controls."""
    cli_source = Path("halo_forge/cli.py").read_text(encoding="utf-8")
    assert cli_source.count("--seed") >= 4


def test_training_service_modality_launch_includes_seed(monkeypatch, tmp_path):
    """TrainingService command builders should pass --seed through to CLI."""
    state = AppState()
    service = TrainingService(state)
    captured = {}

    async def _fake_launch(job_id, cmd, on_log=None):
        captured["cmd"] = cmd

    monkeypatch.setattr(service, "_launch_process", _fake_launch)

    asyncio.run(
        service.launch_modality_train(
            modality="reasoning",
            model="Qwen/Qwen2.5-7B-Instruct",
            dataset="gsm8k",
            output_dir=str(tmp_path / "reasoning"),
            cycles=1,
            seed=7,
        )
    )

    cmd = captured["cmd"]
    assert "--seed" in cmd
    assert cmd[cmd.index("--seed") + 1] == "7"


def test_trainer_preflight_rejects_unsupported_reasoning_model(tmp_path):
    """Trainer entrypoint should enforce model-family policy without CLI gate."""
    from halo_forge.reasoning.trainer import ReasoningRAFTConfig, ReasoningRAFTTrainer

    trainer = ReasoningRAFTTrainer(
        ReasoningRAFTConfig(
            model_name="org/unsupported-model",
            num_cycles=1,
            output_dir=str(tmp_path / "reasoning"),
        )
    )
    with pytest.raises(ValueError, match="Unsupported model family"):
        trainer.train(samples=[SimpleNamespace()])


def test_trainer_preflight_rejects_empty_vlm_samples(tmp_path):
    """VLM trainer should fail fast on empty input payloads."""
    pytest.importorskip("torch")
    from halo_forge.vlm.trainer import VLMRAFTConfig, VLMRAFTTrainer

    trainer = VLMRAFTTrainer(
        VLMRAFTConfig(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            num_cycles=1,
            output_dir=str(tmp_path / "vlm"),
        )
    )
    with pytest.raises(ValueError, match="at least one prompt"):
        trainer.train(prompts=[])


def test_resume_checkpoint_validation_rejects_modality_mismatch(tmp_path):
    """Resume resolver should reject mismatched checkpoint modality metadata."""
    cycle_dir = tmp_path / "cycle_0"
    model_dir = cycle_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    state_path = cycle_dir / "checkpoint_state.json"
    state_path.write_text(
        json.dumps(
            {
                "contract_version": 1,
                "modality": "audio",
                "model_dir": str(model_dir),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected modality=vlm"):
        resolve_resume_checkpoint(
            output_dir=tmp_path,
            resume_from_cycle=1,
            max_cycles=3,
            modality="vlm",
        )


def test_training_summary_includes_phase7_run_metadata():
    """Canonical training summary should include run/replay provenance fields."""
    summary = build_training_summary(
        modality="agentic",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        total_cycles_planned=1,
        cycles=[
            {
                "train_steps_executed": 0,
                "train_loss": None,
                "weights_updated": False,
                "update_reason": "no_filtered_samples",
            }
        ],
        run_id="agentic-123",
        seed=17,
        resume_from_cycle=1,
        resumed_from_checkpoint={"cycle": 0, "model_dir": "/tmp/cycle_0/model"},
        base_model_name="Qwen/Qwen2.5-7B-Instruct",
        active_model_name="/tmp/cycle_0/model",
    )
    assert summary["run_id"] == "agentic-123"
    assert summary["seed"] == 17
    assert summary["resume_from_cycle"] == 1
    assert summary["resumed_from_checkpoint"]["cycle"] == 0
    assert summary["base_model_name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert summary["active_model_name"] == "/tmp/cycle_0/model"
    assert summary["failure_reason"] == "no_filtered_samples"


def test_non_finite_update_batches_are_skipped():
    """Non-finite loss batches should be counted and excluded from optimizer steps."""
    torch = pytest.importorskip("torch")

    class _Tokenizer:
        def __call__(self, *args, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            }

    class _NanModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            _ = self.proj(torch.zeros((1, 2), dtype=torch.float32))
            return SimpleNamespace(loss=torch.tensor(float("nan"), requires_grad=True))

    metrics = run_text_supervised_updates(
        model=_NanModel(),
        tokenizer=_Tokenizer(),
        texts=["sample"],
        learning_rate=1e-4,
        batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=1,
    )
    assert metrics["weights_updated"] is False
    assert metrics["optimizer_steps"] == 0
    assert metrics["skipped_batches_non_finite"] == 1


def test_training_page_only_shows_prototype_override_when_needed():
    """Training UI should gate prototype override controls by capability status."""
    source = Path("ui/pages/training.py").read_text(encoding="utf-8")
    assert 'capability.capability.status == "prototype"' in source
