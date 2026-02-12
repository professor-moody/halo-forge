#!/usr/bin/env python3
"""Phase 3 truth-in-advertising regression tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from halo_forge.capabilities import (
    MODALITY_TRAIN_CAPABILITIES,
    check_modality_train_capability,
)


def test_capability_registry_covers_all_modalities():
    """Capability registry should define all modality train commands."""
    assert set(MODALITY_TRAIN_CAPABILITIES.keys()) == {
        "vlm",
        "audio",
        "reasoning",
        "agentic",
    }


def test_prototype_gating_requires_explicit_flag():
    """Prototype-gated train calls must fail without explicit override flag."""
    check = check_modality_train_capability(
        modality="vlm",
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        allow_prototype_train=False,
        dry_run=False,
    )
    assert check.allowed is False
    assert check.reason == "prototype_flag_required"
    assert "CAPABILITY_ERROR" in check.message
    assert "--allow-prototype-train" in check.message


def test_dry_run_bypasses_prototype_gate():
    """Dry runs should remain available without explicit prototype override."""
    check = check_modality_train_capability(
        modality="audio",
        model_name="openai/whisper-small",
        allow_prototype_train=False,
        dry_run=True,
    )
    assert check.allowed is True


def test_unsupported_audio_model_fails_capability_check():
    """Audio training should reject unsupported model families."""
    check = check_modality_train_capability(
        modality="audio",
        model_name="facebook/wav2vec2-base-960h",
        allow_prototype_train=True,
        dry_run=False,
    )
    assert check.allowed is False
    assert check.reason == "unsupported_model"
    assert "supported_families=whisper" in check.message


def test_supported_audio_local_checkpoint_uses_metadata_family_hint(tmp_path):
    """Local checkpoints should pass when metadata shows a supported family."""
    adapter_dir = tmp_path / "audio_run" / "final_model"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"openai/whisper-small"}',
        encoding="utf-8",
    )

    check = check_modality_train_capability(
        modality="audio",
        model_name=str(adapter_dir),
        allow_prototype_train=True,
        dry_run=False,
    )
    assert check.allowed is True


def test_reasoning_train_updates_reported(monkeypatch, tmp_path):
    """Reasoning trainer should report explicit weight-update telemetry."""
    try:
        from halo_forge.reasoning.trainer import (
            ReasoningCompletion,
            ReasoningRAFTConfig,
            ReasoningRAFTTrainer,
        )
        from halo_forge.reasoning.data import MathSample
    except ModuleNotFoundError as e:
        pytest.skip(f"reasoning module unavailable ({e})")

    trainer = ReasoningRAFTTrainer(
        ReasoningRAFTConfig(output_dir=str(tmp_path / "reasoning"))
    )
    trainer.model = object()
    trainer.tokenizer = object()

    monkeypatch.setattr(
        "halo_forge.reasoning.trainer.run_text_supervised_updates",
        lambda **kwargs: {
            "train_steps_executed": 3,
            "train_loss": 0.12,
            "weights_updated": True,
            "update_reason": "updated",
        },
    )

    metrics = trainer._train_on_filtered(
        [
            ReasoningCompletion(
                sample=MathSample(question="1+1?", answer="2"),
                completion="The answer is 2.",
                reward=1.0,
            )
        ],
        cycle=0,
    )
    assert metrics["train_steps_executed"] == 3
    assert metrics["weights_updated"] is True


def test_agentic_train_updates_reported(monkeypatch, tmp_path):
    """Agentic trainer should report explicit weight-update telemetry."""
    try:
        from halo_forge.agentic.trainer import (
            AgenticCompletion,
            AgenticRAFTConfig,
            AgenticRAFTTrainer,
        )
    except ModuleNotFoundError as e:
        pytest.skip(f"agentic module unavailable ({e})")

    trainer = AgenticRAFTTrainer(
        AgenticRAFTConfig(output_dir=str(tmp_path / "agentic"))
    )
    trainer.model = object()
    trainer.tokenizer = object()

    monkeypatch.setattr(
        "halo_forge.agentic.trainer.run_text_supervised_updates",
        lambda **kwargs: {
            "train_steps_executed": 2,
            "train_loss": 0.34,
            "weights_updated": True,
            "update_reason": "updated",
        },
    )

    metrics = trainer._train_on_samples(
        [AgenticCompletion(prompt="p", output="o", reward=1.0)],
        cycle=0,
    )
    assert metrics["train_steps_executed"] == 2
    assert metrics["weights_updated"] is True


def test_audio_train_cycle_surfaces_update_metrics(monkeypatch, tmp_path):
    """Audio cycle metrics should carry explicit training update fields."""
    try:
        from halo_forge.audio.trainer import AudioRAFTConfig, AudioRAFTTrainer
    except ModuleNotFoundError as e:
        pytest.skip(f"audio module unavailable ({e})")

    trainer = AudioRAFTTrainer(AudioRAFTConfig(output_dir=str(tmp_path / "audio")))

    monkeypatch.setattr(trainer, "_generate_predictions", lambda samples: ["ok"])
    monkeypatch.setattr(
        trainer,
        "_verify_predictions",
        lambda predictions, samples: [
            {"reward": 1.0, "success": True, "details": {}, "sample": object()}
        ],
    )
    monkeypatch.setattr(trainer, "_filter_samples", lambda verified: verified)
    monkeypatch.setattr(
        trainer,
        "_train_on_samples",
        lambda kept, lr: {
            "train_steps_executed": 1,
            "train_loss": 0.5,
            "weights_updated": True,
            "update_reason": "updated",
        },
    )

    result = trainer._train_cycle(0, samples=[object()])
    assert result.metrics["train_steps_executed"] == 1
    assert result.metrics["weights_updated"] is True


def test_vlm_train_updates_reported(monkeypatch, tmp_path):
    """VLM trainer should expose non-placeholder update telemetry."""
    torch = pytest.importorskip("torch")
    try:
        from halo_forge.vlm.trainer import VLMRAFTConfig, VLMRAFTTrainer, VLMSampleResult
    except ModuleNotFoundError as e:
        pytest.skip(f"vlm module unavailable ({e})")

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        @property
        def device(self):
            return self.weight.device

        def forward(self, input_ids=None, labels=None, attention_mask=None, **kwargs):
            loss = (self.weight * input_ids.float().mean()) ** 2
            return SimpleNamespace(loss=loss)

    class FakeProcessor:
        def __call__(self, **kwargs):
            ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    trainer = VLMRAFTTrainer(VLMRAFTConfig(output_dir=str(tmp_path / "vlm")))
    trainer.adapter = SimpleNamespace(
        model=FakeModel(),
        tokenizer=SimpleNamespace(),
        processor=FakeProcessor(),
        _load_image=lambda image: image,
    )

    metrics = trainer.train_on_samples(
        [
            VLMSampleResult(
                image=SimpleNamespace(),
                prompt="Describe image",
                completion="A red car.",
                ground_truth="A red car",
                reward=1.0,
                success=True,
                details={},
            )
        ],
        cycle=0,
    )

    assert metrics["train_steps_executed"] >= 1
    assert metrics["weights_updated"] is True


def test_docs_and_cli_reflect_capability_gating_contract():
    """Docs and CLI parser should both expose the prototype gating flag contract."""
    cli_source = Path("halo_forge/cli.py").read_text(encoding="utf-8")
    assert cli_source.count("--allow-prototype-train") >= 4

    experimental_doc = Path(
        "website/hugo-docs/content/docs/experimental.md"
    ).read_text(encoding="utf-8")
    assert "Modality Training Capability Matrix" in experimental_doc
    assert "--allow-prototype-train" in experimental_doc

    command_index = Path(
        "website/hugo-docs/content/docs/reference/command-index.md"
    ).read_text(encoding="utf-8")
    assert command_index.count("--allow-prototype-train") >= 4
