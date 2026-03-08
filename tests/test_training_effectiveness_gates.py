#!/usr/bin/env python3
"""Training effectiveness contract and deterministic regression gate tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from halo_forge.training_contracts import (
    attach_effectiveness_contract,
    build_cycle_summary,
    build_effectiveness_evaluation,
    build_training_summary,
)
from halo_forge.training_effectiveness_baseline import (
    build_actual_entry_from_effectiveness,
    compare_actuals_to_baseline,
    format_drift_lines,
    validate_baseline_payload,
)


def _load_baseline(name: str) -> dict:
    return json.loads(
        (Path("tests/baselines/training_effectiveness") / name).read_text(encoding="utf-8")
    )


def _assert_no_drifts(drifts: list[dict]) -> None:
    assert drifts == [], "\n".join(format_drift_lines(drifts))


class _FakeSaveComponent:
    def __init__(self, marker: str):
        self.marker = marker

    def save_pretrained(self, target_dir: str) -> None:
        path = Path(target_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{self.marker}.txt").write_text(self.marker, encoding="utf-8")


def test_effectiveness_contract_verdicts_cover_pass_warn_and_fail_cases():
    """Centralized effectiveness logic should distinguish pass/warn/fail outcomes."""
    cycle = build_cycle_summary(
        cycle=0,
        learning_rate=1e-5,
        samples_seen=4,
        samples_kept=2,
        cycle_duration_seconds=0.5,
        update_metrics={
            "train_steps_executed": 2,
            "train_loss": 0.4,
            "initial_train_loss": 0.7,
            "weights_updated": True,
            "update_reason": "updated",
            "optimizer_steps": 2,
        },
    )
    passing = build_training_summary(
        modality="reasoning",
        model_name="org/model",
        total_cycles_planned=1,
        cycles=[cycle],
    )
    attach_effectiveness_contract(
        passing,
        minimum_samples_kept=1,
        minimum_optimizer_steps=1,
        evaluation={
            "metric_name": "accuracy",
            "baseline_value": 0.5,
            "final_value": 0.75,
            "higher_is_better": True,
        },
        final_model_path="models/reasoning/final_model",
        training_summary_path=Path("models/reasoning/training_summary.json"),
    )
    assert passing["effectiveness"]["verdict"] == "pass"

    warned = build_training_summary(
        modality="reasoning",
        model_name="org/model",
        total_cycles_planned=1,
        cycles=[cycle],
    )
    attach_effectiveness_contract(
        warned,
        minimum_samples_kept=1,
        minimum_optimizer_steps=1,
        evaluation=None,
        final_model_path="models/reasoning/final_model",
        training_summary_path=Path("models/reasoning/training_summary.json"),
    )
    assert warned["effectiveness"]["verdict"] == "warn"
    assert "evaluation_not_available" in warned["effectiveness"]["reasons"]

    failed = build_training_summary(
        modality="reasoning",
        model_name="org/model",
        total_cycles_planned=1,
        cycles=[cycle],
    )
    attach_effectiveness_contract(
        failed,
        minimum_samples_kept=3,
        minimum_optimizer_steps=3,
        evaluation={
            "metric_name": "accuracy",
            "baseline_value": 0.8,
            "final_value": 0.5,
            "higher_is_better": True,
        },
        final_model_path="",
        training_summary_path=Path("models/reasoning/training_summary.json"),
        evaluation_required=True,
    )
    assert failed["effectiveness"]["verdict"] == "fail"
    assert "samples_kept_below_minimum" in failed["effectiveness"]["reasons"]
    assert "optimizer_steps_below_minimum" in failed["effectiveness"]["reasons"]
    assert "evaluation_regressed" in failed["effectiveness"]["reasons"]


def test_effectiveness_contract_treats_single_optimizer_step_loss_delta_as_informational():
    """A one-step update may still pass if all required contract fields succeed."""
    summary = build_training_summary(
        modality="audio",
        model_name="org/audio-model",
        total_cycles_planned=1,
        cycles=[
            build_cycle_summary(
                cycle=0,
                learning_rate=1e-5,
                samples_seen=2,
                samples_kept=1,
                cycle_duration_seconds=0.1,
                update_metrics={
                    "train_steps_executed": 1,
                    "train_loss": 0.4,
                    "initial_train_loss": 0.9,
                    "weights_updated": True,
                    "update_reason": "updated",
                    "optimizer_steps": 1,
                },
            )
        ],
    )
    attach_effectiveness_contract(
        summary,
        minimum_samples_kept=1,
        minimum_optimizer_steps=1,
        evaluation={
            "metric_name": "average_reward",
            "baseline_value": 0.2,
            "final_value": 0.2,
            "higher_is_better": True,
        },
        final_model_path="models/audio/final_model",
        training_summary_path=Path("models/audio/training_summary.json"),
    )

    update_quality = summary["effectiveness"]["update_quality"]
    assert update_quality["loss_delta"] == pytest.approx(-0.5)
    assert summary["effectiveness"]["verdict"] == "pass"


def test_effectiveness_contract_fails_when_gated_evaluation_is_unavailable():
    """Gated paths should fail when evaluation is missing even if updates ran."""
    summary = build_training_summary(
        modality="vlm",
        model_name="org/vlm",
        total_cycles_planned=1,
        cycles=[
            build_cycle_summary(
                cycle=0,
                learning_rate=1e-5,
                samples_seen=2,
                samples_kept=1,
                cycle_duration_seconds=0.1,
                update_metrics={
                    "train_steps_executed": 1,
                    "train_loss": 0.3,
                    "weights_updated": True,
                    "update_reason": "updated",
                    "optimizer_steps": 1,
                },
            )
        ],
    )
    attach_effectiveness_contract(
        summary,
        minimum_samples_kept=1,
        minimum_optimizer_steps=1,
        evaluation=build_effectiveness_evaluation(metric_name="avg_reward"),
        evaluation_required=True,
        final_model_path="models/vlm/final_model",
        training_summary_path=Path("models/vlm/training_summary.json"),
    )
    assert summary["effectiveness"]["verdict"] == "fail"
    assert "evaluation_unavailable" in summary["effectiveness"]["reasons"]


def test_training_effectiveness_baselines_validate_and_compare_cleanly():
    """Tracked baseline payloads should validate and compare without drift for matching outcomes."""
    baseline = _load_baseline("benchmark_code.v1.json")
    assert validate_baseline_payload(baseline) == []
    drifts = compare_actuals_to_baseline(
        expected=baseline,
        actual_entries={
            "benchmark_code": {
                "metric_name": "pass_at_1",
                "final_value": 0.5,
                "higher_is_better": True,
                "verdict": "pass",
                "samples_kept": 1,
                "optimizer_steps": 1,
                "evaluation_status": "available",
            }
        },
        fixture_paths=Path("tests/fixtures/training_effectiveness/benchmark_prompts.jsonl"),
    )
    _assert_no_drifts(drifts)


def test_sft_fixture_run_writes_canonical_summary_and_matches_baseline(monkeypatch, tmp_path):
    """SFT should persist canonical summary data and pass the deterministic baseline gate."""
    try:
        import halo_forge.sft.trainer as sft_module
    except ModuleNotFoundError as e:
        if e.name and not e.name.startswith("halo_forge"):
            pytest.skip(f"optional dependency missing ({e.name})")
        raise

    train_fixture = Path("tests/fixtures/training_effectiveness/sft_train.jsonl")
    eval_fixture = Path("tests/fixtures/training_effectiveness/sft_eval.jsonl")

    class _FakeTokenizer:
        pad_token = "<pad>"
        pad_token_id = 0
        eos_token = "</s>"
        eos_token_id = 1

        def save_pretrained(self, target_dir: str) -> None:
            path = Path(target_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "tokenizer.json").write_text("tokenizer", encoding="utf-8")

        def __len__(self) -> int:
            return 32000

    class _FakeTrainer:
        def __init__(self, *args, **kwargs):
            self.state = SimpleNamespace(
                log_history=[
                    {"loss": 0.8, "step": 1},
                    {"eval_loss": 0.9, "step": 1},
                    {"loss": 0.5, "step": 2},
                    {"eval_loss": 0.6, "step": 2},
                ]
            )

        def train(self, resume_from_checkpoint=None):
            return SimpleNamespace(
                global_step=2,
                training_loss=0.5,
                metrics={"train_runtime": 1.5, "global_step": 2},
            )

        def save_model(self, target_dir: str) -> None:
            path = Path(target_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter.bin").write_text("adapter", encoding="utf-8")

    monkeypatch.setattr(sft_module.SFTTrainer, "check_environment", lambda self: None)
    monkeypatch.setattr(
        sft_module.SFTTrainer,
        "_load_tokenizer",
        lambda self: setattr(self, "tokenizer", _FakeTokenizer()),
    )
    monkeypatch.setattr(
        sft_module.SFTTrainer,
        "setup_model",
        lambda self: setattr(self, "model", SimpleNamespace(device="cpu")),
    )
    monkeypatch.setattr(sft_module.SFTTrainer, "run_smoke_test", lambda self: None)
    monkeypatch.setattr(
        sft_module.SFTTrainer,
        "tokenize_function",
        lambda self, examples: {"input_ids": [[1, 2, 3] for _ in examples["text"]]},
    )
    monkeypatch.setattr(sft_module, "Trainer", _FakeTrainer)
    monkeypatch.setattr(sft_module, "DataCollatorForLanguageModeling", lambda **kwargs: object())
    monkeypatch.setattr(sft_module, "EarlyStoppingCallback", lambda **kwargs: object())

    config = sft_module.SFTConfig(
        model_name="org/sft-model",
        train_file=str(train_fixture),
        output_dir=str(tmp_path / "sft"),
        num_epochs=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        validation_split=0.25,
        seed=42,
    )
    trainer = sft_module.SFTTrainer(config)
    summary = trainer.train()

    training_summary_path = tmp_path / "sft" / "training_summary.json"
    assert training_summary_path.exists()
    assert summary["modality"] == "sft"
    assert summary["effectiveness"]["verdict"] == "pass"

    baseline = _load_baseline("sft.v1.json")
    drifts = compare_actuals_to_baseline(
        expected=baseline,
        actual_entries={"sft": build_actual_entry_from_effectiveness(summary["effectiveness"])},
        fixture_paths=[train_fixture, eval_fixture],
    )
    _assert_no_drifts(drifts)


def test_benchmark_result_emits_effectiveness_block_and_matches_baseline(tmp_path):
    """Code benchmark summaries should map into the shared effectiveness contract."""
    pytest.importorskip("torch")
    from halo_forge.benchmark.runner import BenchmarkResult, CycleResult, EvalResult

    result = BenchmarkResult(
        model_name="org/code-model",
        model_short="code-model",
        n_cycles=2,
        total_time_sec=2.0,
        baseline=EvalResult(total_prompts=2, total_samples=2, pass_at_1=0.5, compile_rate=0.5),
        cycles=[
            CycleResult(cycle=1, generated=2, verified=2, kept=1, training_loss=0.7, training_steps=1),
            CycleResult(cycle=2, generated=2, verified=2, kept=1, training_loss=0.4, training_steps=1),
        ],
        final=EvalResult(total_prompts=2, total_samples=2, pass_at_1=0.5, compile_rate=0.5),
        hardware_summary={"gpu": {}},
    )
    output_path = tmp_path / "benchmark" / "summary.json"
    result.save(output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["effectiveness"]["verdict"] == "pass"

    baseline = _load_baseline("benchmark_code.v1.json")
    drifts = compare_actuals_to_baseline(
        expected=baseline,
        actual_entries={
            "benchmark_code": build_actual_entry_from_effectiveness(payload["effectiveness"])
        },
        fixture_paths=Path("tests/fixtures/training_effectiveness/benchmark_prompts.jsonl"),
    )
    _assert_no_drifts(drifts)


def test_modality_smoke_summaries_match_effectiveness_baseline(monkeypatch, tmp_path):
    """Deterministic modality smoke runs should satisfy the tracked effectiveness baseline."""
    pytest.importorskip("torch")

    from halo_forge.agentic.trainer import AgenticRAFTConfig, AgenticRAFTCycleResult, AgenticRAFTTrainer
    from halo_forge.audio.trainer import AudioRAFTConfig, AudioRAFTCycleResult, AudioRAFTTrainer
    from halo_forge.reasoning.data import MathSample
    from halo_forge.reasoning.trainer import ReasoningRAFTConfig, ReasoningRAFTTrainer
    from halo_forge.vlm.trainer import VLMRAFTConfig, VLMRAFTTrainer, VLMSampleResult

    output_root = tmp_path / "modality"
    entries = {}

    class _FakeAdapter:
        def __init__(self, marker: str):
            self.model = _FakeSaveComponent(f"{marker}_model")
            self.tokenizer = _FakeSaveComponent(f"{marker}_tokenizer")
            self.processor = _FakeSaveComponent(f"{marker}_processor")

        def cleanup(self):
            return None

    vlm = VLMRAFTTrainer(VLMRAFTConfig(num_cycles=1, output_dir=str(output_root / "vlm"), seed=42))
    monkeypatch.setattr(
        vlm,
        "_setup",
        lambda: setattr(vlm, "adapter", _FakeAdapter("vlm")) or setattr(
            vlm, "verifier", SimpleNamespace(cleanup=lambda: None)
        ),
    )
    monkeypatch.setattr(
        vlm,
        "generate_samples",
        lambda prompts, spp: [
            VLMSampleResult(
                image="fixture.png",
                prompt="describe",
                completion="answer",
                ground_truth="answer",
                reward=1.0,
                success=True,
                details={},
            )
        ],
    )
    monkeypatch.setattr(vlm, "filter_samples", lambda samples: samples)
    monkeypatch.setattr(
        vlm,
        "train_on_samples",
        lambda samples, cycle: {
            "train_steps_executed": 1,
            "train_loss": 0.1,
            "initial_train_loss": 0.2,
            "weights_updated": True,
            "update_reason": "updated",
            "optimizer_steps": 1,
        },
    )
    vlm_summary = vlm.train(
        prompts=[SimpleNamespace(image="fixture.png", prompt="describe", ground_truth="answer")]
    )
    entries["vlm"] = build_actual_entry_from_effectiveness(vlm_summary["effectiveness"])

    audio = AudioRAFTTrainer(
        AudioRAFTConfig(num_cycles=1, output_dir=str(output_root / "audio"), seed=42)
    )
    audio.adapter = _FakeAdapter("audio")
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
                    "initial_train_loss": 0.2,
                    "weights_updated": True,
                    "update_reason": "updated",
                    "optimizer_steps": 1,
                },
                extra={"average_reward": 1.0},
            ),
        ),
    )
    audio.train(samples=[SimpleNamespace()])
    entries["audio"] = build_actual_entry_from_effectiveness(audio.training_summary["effectiveness"])

    reasoning = ReasoningRAFTTrainer(
        ReasoningRAFTConfig(num_cycles=1, output_dir=str(output_root / "reasoning"), seed=42)
    )
    reasoning.model = _FakeSaveComponent("reasoning_model")
    reasoning.tokenizer = _FakeSaveComponent("reasoning_tokenizer")
    monkeypatch.setattr(
        reasoning,
        "train_cycle",
        lambda samples, cycle: build_cycle_summary(
            cycle=cycle,
            learning_rate=1e-5,
            samples_seen=1,
            samples_kept=1,
            cycle_duration_seconds=0.1,
            update_metrics={
                "train_steps_executed": 1,
                "train_loss": 0.1,
                "initial_train_loss": 0.2,
                "weights_updated": True,
                "update_reason": "updated",
                "optimizer_steps": 1,
            },
            extra={"accuracy": 1.0, "avg_reward": 1.0},
        ),
    )
    reasoning_summary = reasoning.train(samples=[MathSample(question="1+1", answer="2")])
    entries["reasoning"] = build_actual_entry_from_effectiveness(reasoning_summary["effectiveness"])

    agentic = AgenticRAFTTrainer(
        AgenticRAFTConfig(num_cycles=1, output_dir=str(output_root / "agentic"), seed=42)
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
                    "initial_train_loss": 0.2,
                    "weights_updated": True,
                    "update_reason": "updated",
                    "optimizer_steps": 1,
                },
            ),
        ),
    )
    agentic_summary = agentic.train(
        samples=[SimpleNamespace(prompt="prompt", expected_calls=[], is_irrelevant=False)]
    )
    entries["agentic"] = build_actual_entry_from_effectiveness(agentic_summary["effectiveness"])

    baseline = _load_baseline("modality_pack.v1.json")
    drifts = compare_actuals_to_baseline(
        expected=baseline,
        actual_entries=entries,
        fixture_paths=Path("tests/fixtures/modality"),
    )
    _assert_no_drifts(drifts)


@pytest.mark.parametrize(
    ("summary_kwargs", "evaluation", "expected_reason"),
    [
        (
            {
                "samples_seen": 2,
                "samples_kept": 0,
                "train_steps_executed": 0,
                "train_loss": None,
                "weights_updated": False,
                "update_reason": "no_filtered_samples",
                "optimizer_steps": 0,
            },
            None,
            "weights_not_updated",
        ),
        (
            {
                "samples_seen": 2,
                "samples_kept": 1,
                "train_steps_executed": 1,
                "train_loss": 0.2,
                "weights_updated": True,
                "update_reason": "updated",
                "optimizer_steps": 1,
            },
            {
                "metric_name": "accuracy",
                "baseline_value": 0.8,
                "final_value": 0.5,
                "higher_is_better": True,
            },
            "evaluation_regressed",
        ),
    ],
)
def test_negative_effectiveness_cases_catch_false_success_runs(
    summary_kwargs, evaluation, expected_reason
):
    """False-success runs should fail the shared effectiveness contract."""
    summary = build_training_summary(
        modality="reasoning",
        model_name="org/model",
        total_cycles_planned=1,
        cycles=[
            build_cycle_summary(
                cycle=0,
                learning_rate=1e-5,
                samples_seen=summary_kwargs["samples_seen"],
                samples_kept=summary_kwargs["samples_kept"],
                cycle_duration_seconds=0.1,
                update_metrics=summary_kwargs,
            )
        ],
    )
    attach_effectiveness_contract(
        summary,
        minimum_samples_kept=1,
        minimum_optimizer_steps=1,
        evaluation=evaluation,
        evaluation_required=evaluation is not None,
        final_model_path="models/reasoning/final_model",
        training_summary_path=Path("models/reasoning/training_summary.json"),
    )
    assert summary["effectiveness"]["verdict"] == "fail"
    assert expected_reason in summary["effectiveness"]["reasons"]
