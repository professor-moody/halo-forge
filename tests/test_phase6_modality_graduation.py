#!/usr/bin/env python3
"""Phase 6 modality graduation regression tests."""

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from halo_forge.capabilities import (
    CAPABILITY_STATUS_PROTOTYPE,
    CAPABILITY_STATUS_REAL_TRAINING,
    MODALITY_TRAIN_CAPABILITIES,
    check_modality_train_capability,
)
from halo_forge.training_contracts import build_cycle_summary
from ui.services.training_service import TrainingService
from ui.state import AppState


class _FakeSaveComponent:
    def __init__(self, marker: str):
        self.marker = marker

    def save_pretrained(self, target_dir: str):
        path = Path(target_dir) / f"{self.marker}.txt"
        path.write_text(self.marker, encoding="utf-8")


def test_vlm_trainer_writes_cycle_and_final_artifacts(monkeypatch, tmp_path):
    """VLM trainer should persist canonical cycle and final artifacts."""
    torch = pytest.importorskip("torch")
    from halo_forge.vlm.trainer import VLMRAFTConfig, VLMRAFTTrainer, VLMSampleResult

    class _FakeAdapter:
        def __init__(self):
            self.model = _FakeSaveComponent("vlm_model")
            self.tokenizer = _FakeSaveComponent("vlm_tokenizer")
            self.processor = _FakeSaveComponent("vlm_processor")

        def cleanup(self):
            return None

    trainer = VLMRAFTTrainer(
        VLMRAFTConfig(num_cycles=1, output_dir=str(tmp_path / "vlm"))
    )

    monkeypatch.setattr(
        trainer,
        "_setup",
        lambda: setattr(trainer, "adapter", _FakeAdapter()) or setattr(
            trainer,
            "verifier",
            SimpleNamespace(cleanup=lambda: None),
        ),
    )
    monkeypatch.setattr(
        trainer,
        "generate_samples",
        lambda prompts, spp: [
            VLMSampleResult(
                image="fake.png",
                prompt="p",
                completion="c",
                ground_truth="c",
                reward=1.0,
                success=True,
                details={},
            )
        ],
    )
    monkeypatch.setattr(trainer, "filter_samples", lambda samples: samples)
    monkeypatch.setattr(
        trainer,
        "train_on_samples",
        lambda samples, cycle: {
            "train_steps_executed": 1,
            "train_loss": 0.2,
            "weights_updated": True,
            "update_reason": "updated",
        },
    )

    summary = trainer.train(prompts=[SimpleNamespace()])
    assert (tmp_path / "vlm" / "cycle_0" / "model").exists()
    assert (tmp_path / "vlm" / "cycle_0" / "checkpoint_state.json").exists()
    assert (tmp_path / "vlm" / "latest_checkpoint.json").exists()
    assert (tmp_path / "vlm" / "final_model").exists()
    assert summary["final_model_path"].endswith("final_model")


def test_vlm_resume_from_cycle_uses_previous_checkpoint_model(monkeypatch, tmp_path):
    """VLM resume should resolve cycle_(resume-1)/model as baseline checkpoint."""
    pytest.importorskip("torch")
    from halo_forge.vlm.trainer import VLMRAFTConfig, VLMRAFTTrainer, VLMSampleResult

    class _FakeAdapter:
        def __init__(self):
            self.model = _FakeSaveComponent("vlm_model")
            self.tokenizer = _FakeSaveComponent("vlm_tokenizer")
            self.processor = _FakeSaveComponent("vlm_processor")

        def cleanup(self):
            return None

    def _patch_trainer(trainer):
        monkeypatch.setattr(
            trainer,
            "_setup",
            lambda: setattr(trainer, "adapter", _FakeAdapter()) or setattr(
                trainer,
                "verifier",
                SimpleNamespace(cleanup=lambda: None),
            ),
        )
        monkeypatch.setattr(
            trainer,
            "generate_samples",
            lambda prompts, spp: [
                VLMSampleResult(
                    image="fake.png",
                    prompt="p",
                    completion="c",
                    ground_truth="c",
                    reward=1.0,
                    success=True,
                    details={},
                )
            ],
        )
        monkeypatch.setattr(trainer, "filter_samples", lambda samples: samples)
        monkeypatch.setattr(
            trainer,
            "train_on_samples",
            lambda samples, cycle: {
                "train_steps_executed": 1,
                "train_loss": 0.2,
                "weights_updated": True,
                "update_reason": "updated",
            },
        )

    output_dir = tmp_path / "vlm_resume"
    trainer1 = VLMRAFTTrainer(VLMRAFTConfig(num_cycles=2, output_dir=str(output_dir)))
    _patch_trainer(trainer1)
    trainer1.train(prompts=[SimpleNamespace()])

    resume_model_path = {}
    trainer2 = VLMRAFTTrainer(VLMRAFTConfig(num_cycles=3, output_dir=str(output_dir)))
    original_setup = trainer2._setup

    def _capture_setup():
        resume_model_path["value"] = trainer2.config.model_name
        return original_setup()

    _patch_trainer(trainer2)
    monkeypatch.setattr(trainer2, "_setup", _capture_setup)
    summary = trainer2.train(prompts=[SimpleNamespace()], resume_from=2)
    assert str(Path(resume_model_path["value"])).endswith("cycle_1/model")
    assert summary["cycles_executed"] == 3


def test_audio_reasoning_agentic_write_artifact_contract(monkeypatch, tmp_path):
    """Audio, reasoning, and agentic trainers should write cycle/final artifacts."""
    pytest.importorskip("torch")

    from halo_forge.audio.trainer import AudioRAFTConfig, AudioRAFTCycleResult, AudioRAFTTrainer
    from halo_forge.reasoning.data import MathSample
    from halo_forge.reasoning.trainer import (
        ReasoningCompletion,
        ReasoningRAFTConfig,
        ReasoningRAFTTrainer,
    )
    from halo_forge.agentic.trainer import (
        AgenticRAFTConfig,
        AgenticRAFTCycleResult,
        AgenticRAFTTrainer,
    )

    audio = AudioRAFTTrainer(AudioRAFTConfig(num_cycles=1, output_dir=str(tmp_path / "audio")))
    audio.adapter = SimpleNamespace(
        model=_FakeSaveComponent("audio_model"),
        tokenizer=_FakeSaveComponent("audio_tokenizer"),
        processor=_FakeSaveComponent("audio_processor"),
    )
    monkeypatch.setattr(audio, "_init_adapter", lambda: None)
    monkeypatch.setattr(audio, "_init_verifier", lambda: None)
    monkeypatch.setattr(
        audio,
        "_train_cycle",
        lambda cycle, samples: AudioRAFTCycleResult(
            cycle=cycle,
            samples_generated=1,
            samples_verified=1,
            samples_kept=1,
            average_reward=1.0,
            learning_rate=1e-5,
            metrics=build_cycle_summary(
                cycle=cycle,
                learning_rate=1e-5,
                samples_seen=1,
                samples_kept=1,
                cycle_duration_seconds=0.1,
                update_metrics={
                    "train_steps_executed": 1,
                    "train_loss": 0.1,
                    "weights_updated": True,
                    "update_reason": "updated",
                },
            ),
        ),
    )
    audio.train(samples=[object()])
    assert (tmp_path / "audio" / "cycle_0" / "model").exists()
    assert (tmp_path / "audio" / "final_model").exists()

    reasoning = ReasoningRAFTTrainer(
        ReasoningRAFTConfig(num_cycles=1, output_dir=str(tmp_path / "reasoning"))
    )
    reasoning.model = _FakeSaveComponent("reasoning_model")
    reasoning.tokenizer = _FakeSaveComponent("reasoning_tokenizer")
    sample = MathSample(question="1+1?", answer="2")
    completion = ReasoningCompletion(sample=sample, completion="2")
    monkeypatch.setattr(reasoning, "generate_completions", lambda samples: [completion])
    monkeypatch.setattr(
        reasoning,
        "verify_completions",
        lambda comps: [
            ReasoningCompletion(
                sample=sample,
                completion="2",
                reward=1.0,
                verified=True,
                result=SimpleNamespace(extracted_answer="2"),
            )
        ],
    )
    monkeypatch.setattr(reasoning, "filter_completions", lambda comps: comps)
    monkeypatch.setattr(
        reasoning,
        "_train_on_filtered",
        lambda comps, cycle: {
            "train_steps_executed": 1,
            "train_loss": 0.1,
            "weights_updated": True,
            "update_reason": "updated",
        },
    )
    reasoning.train([sample])
    assert (tmp_path / "reasoning" / "cycle_0" / "model").exists()
    assert (tmp_path / "reasoning" / "final_model").exists()

    agentic = AgenticRAFTTrainer(
        AgenticRAFTConfig(num_cycles=1, output_dir=str(tmp_path / "agentic"))
    )
    agentic.model = _FakeSaveComponent("agentic_model")
    agentic.tokenizer = _FakeSaveComponent("agentic_tokenizer")
    monkeypatch.setattr(
        agentic,
        "_run_cycle",
        lambda samples, cycle: AgenticRAFTCycleResult(
            cycle=cycle,
            total_samples=1,
            verified_samples=1,
            avg_reward=1.0,
            success_rate=1.0,
            training_samples=1,
            metrics=build_cycle_summary(
                cycle=cycle,
                learning_rate=1e-5,
                samples_seen=1,
                samples_kept=1,
                cycle_duration_seconds=0.1,
                update_metrics={
                    "train_steps_executed": 1,
                    "train_loss": 0.1,
                    "weights_updated": True,
                    "update_reason": "updated",
                },
            ),
        ),
    )
    agentic.train(samples=[object()])
    assert (tmp_path / "agentic" / "cycle_0" / "model").exists()
    assert (tmp_path / "agentic" / "final_model").exists()


def test_capability_registry_is_graduated_and_strict():
    """Capability matrix should be real_training with strict allowlists."""
    for modality in ("vlm", "audio", "reasoning", "agentic"):
        capability = MODALITY_TRAIN_CAPABILITIES[modality]
        assert capability.status == CAPABILITY_STATUS_REAL_TRAINING
        assert "*" not in capability.supported_model_families

    reasoning_fail = check_modality_train_capability(
        modality="reasoning",
        model_name="org/not-supported-family",
        allow_prototype_train=True,
        dry_run=False,
    )
    assert reasoning_fail.allowed is False
    assert reasoning_fail.reason == "unsupported_model"

    agentic_ok = check_modality_train_capability(
        modality="agentic",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        allow_prototype_train=True,
        dry_run=False,
    )
    assert agentic_ok.allowed is True


def test_training_service_modality_command_respects_prototype_flag(monkeypatch, tmp_path):
    """UI service should include prototype flag only when required by capability status."""
    state = AppState()
    service = TrainingService(state)
    captured_cmd = {}

    async def _fake_launch(job_id, cmd, on_log=None):
        captured_cmd["value"] = cmd

    monkeypatch.setattr(service, "_launch_process", _fake_launch)

    original = MODALITY_TRAIN_CAPABILITIES["vlm"]
    monkeypatch.setitem(
        MODALITY_TRAIN_CAPABILITIES,
        "vlm",
        replace(original, status=CAPABILITY_STATUS_PROTOTYPE),
    )

    with pytest.raises(ValueError, match="rollout-gated"):
        asyncio.run(
            service.launch_modality_train(
                modality="vlm",
                model="Qwen/Qwen2-VL-7B-Instruct",
                dataset="textvqa",
                output_dir=str(tmp_path / "vlm"),
                cycles=1,
            )
        )

    asyncio.run(
        service.launch_modality_train(
            modality="vlm",
            model="Qwen/Qwen2-VL-7B-Instruct",
            dataset="textvqa",
            output_dir=str(tmp_path / "vlm"),
            cycles=1,
            allow_prototype_train=True,
        )
    )
    assert "--allow-prototype-train" in captured_cmd["value"]
    assert "--seed" in captured_cmd["value"]


def test_app_state_cycle_progress_for_modality_jobs():
    """Cycle-based modality jobs should report progress percent via cycle counters."""
    app_state = AppState()
    job = app_state.create_job(
        job_type="audio",
        name="Audio",
        output_dir=Path("models/audio"),
    )
    job.total_cycles = 4
    job.current_cycle = 2
    assert job.progress_percent == 50.0
