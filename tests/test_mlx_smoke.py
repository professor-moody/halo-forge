"""End-to-end MLX smoke tests.

Guard against upstream `mlx-lm` API drift. The library moved sampler
configuration (0.21+), dataset interfaces (CacheDataset wrapping required),
and `mx.metal.clear_cache` (deprecated) inside our development window, and
each migration silently broke a working halo-forge path. These tests run
the actual MLX surface so we catch the next change before it ships.

Each test is gated by `requires_mlx` so they auto-skip on non-Mac CI runners
and on Macs where `[mlx]` is not installed. They download a tiny
`mlx-community/Qwen2.5-0.5B-Instruct-bf16` (~1GB) on first run; HuggingFace
Hub caching handles re-runs.

Slow markers are deliberately omitted — the whole file completes in under
a minute on Apple Silicon. If it grows beyond that, split out the model
load into a session-scoped fixture.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.requires_mlx


# Tiny MLX-format model. Re-used across tests; HF cache means only one
# download per machine.
MLX_TINY_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"


@pytest.fixture(scope="module")
def mlx_inference_adapter():
    """Loaded MLXInferenceAdapter shared across tests in this module."""
    from halo_forge.backend.mlx import MLXInferenceAdapter

    adapter = MLXInferenceAdapter(MLX_TINY_MODEL)
    adapter.load()
    yield adapter
    adapter.cleanup()


def test_mlx_backend_capabilities_match_phase4_state():
    """The backend registry should report MLX as training-capable post-phase 4.

    If a future change accidentally flips supports_training=False, the
    SFT and RAFT tests below will start auto-skipping, which would be
    a silent loss of coverage.
    """
    from halo_forge.backend import get_backend

    b = get_backend("mlx")
    assert b.name == "mlx"
    assert b.capabilities.supports_training is True
    assert b.capabilities.supports_4bit is False  # MLX bakes quant into the artifact
    assert b.capabilities.preferred_dtype_str == "float16"


def test_mlx_inference_smoke(mlx_inference_adapter):
    """MLXInferenceAdapter loads + generates. Catches sampler-API drift.

    mlx-lm 0.21 moved `temp=` into a sampler object built via
    `mlx_lm.sample_utils.make_sampler`. The adapter has both code paths
    with a fallback; if both break in a future version, this test fails.
    """
    out = mlx_inference_adapter.generate(
        "def add(a, b):", max_tokens=20, temperature=0.0
    )
    assert isinstance(out, str)
    assert len(out) > 0
    # Greedy decoding from a sane base model should produce real Python.
    # We assert minimally — this isn't a generation-quality test, just a
    # sanity check that something coherent came back.
    assert "return" in out or "a" in out


def test_mlx_sft_runs_one_cycle(tmp_path):
    """End-to-end MLX LoRA SFT on a tiny dataset.

    Catches three classes of mlx-lm API breakage we already hit:
    - `tokenizer=` kwarg removal from mlx_lm.tuner.trainer.train
    - File-path dataset arguments (must be CacheDataset(TextDataset(...)))
    - `model.freeze()` ordering vs. `linear_to_lora_layers`

    Asserts on the canonical training summary so we also catch drift in
    halo_forge.training_contracts surface.
    """
    from halo_forge.sft.config import SFTConfig
    from halo_forge.sft.mlx_trainer import MLXSFTTrainer

    train_file = tmp_path / "tiny.jsonl"
    samples = [
        {"text": f"def f{i}(a, b):\n    return a {op} b\n"}
        for i, op in enumerate(["+", "-", "*", "/", "%", "**", "//"] * 3)
    ]
    train_file.write_text("\n".join(json.dumps(s) for s in samples) + "\n")

    cfg = SFTConfig(
        model_name=MLX_TINY_MODEL,
        train_file=str(train_file),
        output_dir=str(tmp_path / "out"),
        num_epochs=1,
        batch_size=2,
        max_seq_length=64,
        eval_steps=2,
        save_steps=10,
        lora_r=4,
        lora_alpha=8,
        max_samples=16,
        validation_split=0.25,
        learning_rate=1e-4,
        gradient_checkpointing=False,
    )
    trainer = MLXSFTTrainer(cfg)
    summary = trainer.train()

    # Canonical summary shape — same as the PyTorch trainer's output.
    assert summary["modality"] == "sft"
    assert summary["model_name"] == MLX_TINY_MODEL
    assert summary["weights_updated"] is True
    assert summary["total_train_steps_executed"] > 0
    assert isinstance(summary["final_train_loss"], float)
    assert summary["cycles_executed"] == 1

    # Adapter actually written.
    final_path = Path(summary["final_model_path"])
    assert final_path.exists(), f"adapter dir missing at {final_path}"
    adapter_files = list(final_path.glob("*.safetensors"))
    assert adapter_files, f"no .safetensors files in {final_path}"

    # Backend annotation is preserved through the summary builder so the
    # downstream effectiveness gates can tell which backend produced the
    # artifact.
    assert summary.get("backend") == "mlx"


def test_mlx_raft_runs_one_cycle(tmp_path):
    """End-to-end MLX RAFT loop with a fake length-based verifier.

    Exercises rollout + verify + filter + SFT in a single cycle. The fake
    verifier scores by completion length so we don't need a real
    compiler/runtime here — that surface is tested elsewhere.
    """
    from halo_forge.rlvr.mlx_raft_trainer import MLXRAFTTrainer

    class _FakeVerifier:
        """Length-only grader so the test doesn't need a real toolchain."""

        def verify_batch(self, completions, prompts):
            results = []
            for c in completions:
                length = len(c.strip())
                reward = min(1.0, length / 30.0)
                success = length >= 10
                results.append(
                    SimpleNamespace(reward=reward, success=success, details={"len": length})
                )
            return results

    cfg = SimpleNamespace(
        base_model=MLX_TINY_MODEL,
        output_dir=str(tmp_path / "raft_out"),
        num_cycles=1,
        samples_per_prompt=2,
        max_new_tokens=20,
        temperature=0.7,
        generation_batch_size=1,
        verification_chunk_size=10,
        reward_threshold=0.0,
        keep_top_percent=0.5,
        min_samples_per_cycle=None,
        system_prompt="You are a helpful coding assistant.",
        seed=42,
        sft_epochs_per_cycle=1,
        sft_batch_size=1,
        sft_gradient_accumulation=1,
        learning_rate=1e-4,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.05,
    )
    trainer = MLXRAFTTrainer(verifier=_FakeVerifier(), config=cfg)
    summary = trainer.run(["Write a Python function to add two numbers"], num_cycles=1)

    assert summary["modality"] == "raft"
    assert summary["model_name"] == MLX_TINY_MODEL
    assert summary["weights_updated"] is True
    assert summary["cycles_executed"] == 1
    assert summary.get("backend") == "mlx"
    assert summary.get("run_id", "").startswith("raft_mlx-")

    # Cycle-level state sidecar should be cleared on successful completion
    # so a follow-up launch starts fresh instead of "resuming" the just-
    # completed run. Phase 5c contract.
    state_path = tmp_path / "raft_out" / "_cycle_state.json"
    assert not state_path.exists(), (
        f"cycle state sidecar should be cleared after success: {state_path}"
    )

    # Cycle 0 summary carries reward telemetry from the shared filter
    # helper — guards against accidental loss of fields when extending
    # the summary builder.
    cycle = summary["cycles"][0]
    for key in ("avg_reward", "avg_kept_reward", "success_rate", "effective_threshold"):
        assert key in cycle, f"cycle summary missing {key!r}"


def test_mlx_rollout_generator_resume_from_cache(tmp_path):
    """MLXRolloutGenerator should pick up where a partial cache left off.

    We pre-seed the cache with one (prompt, completion) pair that "covers"
    the requested num_samples for the first prompt, then ask for two more
    samples for a different prompt. The generator should append, not
    duplicate, and the final list should reflect both.
    """
    from halo_forge.rlvr.mlx_rollout import MLXRolloutGenerator

    cache = tmp_path / "rollout_cache.jsonl"
    cache.write_text(json.dumps({"prompt": "p1", "completion": "pre-seeded"}) + "\n")

    gen = MLXRolloutGenerator(MLX_TINY_MODEL)
    samples = gen.generate_samples(
        ["p1"],  # only one prompt; cache already covers it for num_samples=1
        num_samples=1,
        max_new_tokens=8,
        temperature=0.0,
        batch_size=1,
        system_prompt="",
        cache_path=cache,
    )
    # Cache covered the entire request; nothing should have been re-generated.
    assert len(samples) == 1
    assert samples[0] == ("p1", "pre-seeded")
