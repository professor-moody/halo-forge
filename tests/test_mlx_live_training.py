"""Full live MLX training smoke tests.

These are intentionally gated behind ``requires_mlx``. They exercise the real
MLX model/load/train/save path on Apple Silicon, while remaining tiny enough for
local release acceptance runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.requires_mlx


MLX_TINY_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"


def _write_tiny_preference_file(path: Path) -> Path:
    rows = [
        {
            "prompt": "Write a Python function named add that returns a + b.",
            "chosen": "def add(a, b):\n    return a + b\n",
            "rejected": "def add(a, b):\n    return a - b\n",
        },
        {
            "prompt": "Write a Python function named square that squares x.",
            "chosen": "def square(x):\n    return x * x\n",
            "rejected": "def square(x):\n    return x + x\n",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _run_mlx_dpo_live_sigmoid(tmp_path, *, reference_free: bool, label: str) -> dict:
    from halo_forge.dpo.config import DPOConfig
    from halo_forge.dpo.mlx_trainer import MLXDPOTrainer

    train_file = _write_tiny_preference_file(tmp_path / f"dpo_{label}.jsonl")
    cfg = DPOConfig(
        model_name=MLX_TINY_MODEL,
        train_file=str(train_file),
        output_dir=str(tmp_path / f"dpo_{label}_out"),
        num_epochs=1,
        batch_size=1,
        max_samples=2,
        validation_split=0.0,
        max_seq_length=64,
        max_prompt_length=48,
        beta=0.1,
        loss_type="sigmoid",
        reference_free=reference_free,
        learning_rate=5e-6,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        logging_steps=1,
        save_steps=10,
        eval_steps=10,
        gradient_checkpointing=False,
    )

    return MLXDPOTrainer(cfg).train()


def test_mlx_dpo_live_reference_free_sigmoid_runs_one_cycle(tmp_path):
    summary = _run_mlx_dpo_live_sigmoid(
        tmp_path,
        reference_free=True,
        label="reference_free",
    )

    assert summary["modality"] == "dpo"
    assert summary["model_name"] == MLX_TINY_MODEL
    assert summary["weights_updated"] is True
    assert summary["total_train_steps_executed"] > 0
    assert summary["reference_free"] is True
    assert summary["reference_model_loaded"] is False
    assert summary["backend"] == "mlx"
    final_path = Path(summary["final_model_path"])
    assert final_path.exists()
    assert list(final_path.glob("*.safetensors"))


def test_mlx_dpo_live_reference_model_sigmoid_runs_one_cycle(tmp_path):
    summary = _run_mlx_dpo_live_sigmoid(
        tmp_path,
        reference_free=False,
        label="reference_model",
    )

    assert summary["modality"] == "dpo"
    assert summary["model_name"] == MLX_TINY_MODEL
    assert summary["weights_updated"] is True
    assert summary["total_train_steps_executed"] > 0
    assert summary["reference_free"] is False
    assert summary["reference_model_loaded"] is True
    assert summary["backend"] == "mlx"
    final_path = Path(summary["final_model_path"])
    assert final_path.exists()
    assert list(final_path.glob("*.safetensors"))


def test_mlx_grpo_live_reference_free_runs_one_cycle(monkeypatch, tmp_path):
    from halo_forge.grpo.config import GRPOConfig
    from halo_forge.grpo.mlx_trainer import MLXGRPOTrainer

    class CountingVerifier:
        def __init__(self):
            self.calls = 0

        def verify(self, completion):
            self.calls += 1
            reward = 1.0 if self.calls % 2 == 0 else 0.0
            return SimpleNamespace(
                reward=reward,
                success=bool(reward),
                details={"call": self.calls},
            )

    monkeypatch.setattr(
        "halo_forge.rlvr.verifiers.get_verifier",
        lambda name: CountingVerifier,
    )

    train_file = tmp_path / "grpo_prompts.jsonl"
    train_file.write_text(
        json.dumps({"prompt": "Give a concise answer: what is 2 + 2?"}) + "\n"
    )
    cfg = GRPOConfig(
        model_name=MLX_TINY_MODEL,
        train_file=str(train_file),
        output_dir=str(tmp_path / "grpo_out"),
        reference_free=True,
        num_generations=2,
        max_samples=1,
        max_prompt_length=48,
        max_completion_length=8,
        num_epochs=1,
        batch_size=1,
        temperature=0.7,
        verifier_name="counting_smoke",
        reward_threshold=0.0,
        learning_rate=1e-6,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        logging_steps=1,
        gradient_checkpointing=False,
    )

    summary = MLXGRPOTrainer(cfg).train()

    assert summary["modality"] == "grpo"
    assert summary["model_name"] == MLX_TINY_MODEL
    assert summary["weights_updated"] is True
    assert summary["total_train_steps_executed"] > 0
    assert summary["reference_free"] is True
    assert summary["backend"] == "mlx"
    assert summary["n_prompts"] == 1
    assert summary["n_completions"] == 2
    final_path = Path(summary["final_model_path"])
    assert final_path.exists()
    assert list(final_path.glob("*.safetensors"))
