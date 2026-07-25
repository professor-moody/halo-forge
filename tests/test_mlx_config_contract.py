"""MLX trainers must honor (or loudly reject) training-critical config.

The May 2026 review found that both MLX trainers silently dropped
``gradient_accumulation_steps``, ``max_steps``, ``max_grad_norm`` and
``warmup_ratio``. With ``gradient_accumulation_steps`` defaulting to 16 the
MLX path ran at 16x the optimizer-step count and 1/16th the effective batch
size the user asked for — and then wrote the *requested* value into the
relaunch metadata as if it had been applied.

mlx / mlx-lm are not installed in CI, so every test here drives the trainers
through a stub ``_require_mlx_lm`` dependency bundle that mirrors the real
mlx-lm 0.31 surface (``TrainingArgs.grad_accumulation_steps``, ``iters``
counting *micro* batches, ``optimizer.update`` applied every
``grad_accumulation_steps`` iterations).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest


# ----------------------------------------------------------------------
# Stub mlx surface — mirrors mlx-lm 0.31 / mlx 0.31 API shapes.
# ----------------------------------------------------------------------


@dataclass
class _StubTrainingArgs:
    """Mirror of ``mlx_lm.tuner.trainer.TrainingArgs`` (mlx-lm 0.31)."""

    batch_size: int = 4
    iters: int = 100
    val_batches: int = 25
    steps_per_report: int = 10
    steps_per_eval: int = 200
    steps_per_save: int = 100
    max_seq_length: int = 2048
    adapter_file: str = "adapters.safetensors"
    grad_checkpoint: bool = False
    grad_accumulation_steps: int = 1


@dataclass
class _LegacyTrainingArgs:
    """Older mlx-lm: no gradient accumulation support at all."""

    batch_size: int = 4
    iters: int = 100
    val_batches: int = 25
    steps_per_report: int = 10
    steps_per_eval: int = 200
    steps_per_save: int = 100
    max_seq_length: int = 2048
    adapter_file: str = "adapters.safetensors"
    grad_checkpoint: bool = False


class _StubAdamW:
    """Stand-in for ``mlx.optimizers.AdamW``."""

    def __init__(self, learning_rate: Any = 1e-3, weight_decay: float = 0.0) -> None:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.applied_gradients: List[Any] = []
        self.state: Dict[str, Any] = {}

    def update(self, model: Any, gradients: Any) -> None:
        self.applied_gradients.append(gradients)


def _linear_schedule(init: float, end: float, steps: int):
    def schedule(step: float) -> float:
        frac = min(max(float(step), 0.0), float(steps)) / max(1.0, float(steps))
        return init + (end - init) * frac

    return schedule


def _cosine_decay(init: float, decay_steps: int):
    def schedule(step: float) -> float:
        frac = min(max(float(step), 0.0), float(decay_steps)) / max(1.0, float(decay_steps))
        return init * 0.5 * (1.0 + math.cos(math.pi * frac))

    return schedule


def _join_schedules(schedules, boundaries):
    def schedule(step: float) -> float:
        out = schedules[0](step)
        for boundary, nxt in zip(boundaries, schedules[1:]):
            if step >= boundary:
                out = nxt(step - boundary)
        return out

    return schedule


class _StubOptimModule:
    """Stand-in for ``mlx.optimizers``."""

    def __init__(self, *, with_clip: bool = True, with_schedules: bool = True) -> None:
        instances: List[_StubAdamW] = []
        self.instances = instances

        class _RecordingAdamW(_StubAdamW):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                instances.append(self)

        self.AdamW = _RecordingAdamW
        self.clip_calls: List[float] = []
        if with_clip:
            self.clip_grad_norm = self._clip_grad_norm
        if with_schedules:
            self.linear_schedule = staticmethod(_linear_schedule).__func__
            self.cosine_decay = staticmethod(_cosine_decay).__func__
            self.join_schedules = staticmethod(_join_schedules).__func__

    def _clip_grad_norm(self, gradients: Any, max_norm: float):
        self.clip_calls.append(max_norm)
        return gradients, 1.0


class _StubParam:
    size = 4


class _StubModel:
    def __init__(self) -> None:
        self.frozen = False
        self.layers = [object(), object()]

    def freeze(self) -> None:
        self.frozen = True

    def trainable_parameters(self) -> Dict[str, Any]:
        return {"lora_a": _StubParam()}

    def parameters(self) -> Dict[str, Any]:
        return {"lora_a": _StubParam()}


class _StubTokenizer:
    pad_token_id: Optional[int] = 0
    eos_token_id: int = 0

    def encode(self, text: str) -> List[int]:
        return [1] * max(1, len(text.split()))

    def save_pretrained(self, path: str) -> None:  # pragma: no cover - no-op
        return None


def _stub_tree_flatten(tree: Any):
    return list(tree.items()) if isinstance(tree, dict) else list(tree)


# ----------------------------------------------------------------------
# SFT: stub deps bundle + trainer driver
# ----------------------------------------------------------------------


class _SFTRecorder:
    def __init__(self) -> None:
        self.training_args_kwargs: Dict[str, Any] = {}
        self.train_calls: List[Dict[str, Any]] = []
        self.optimizer: Any = None


def _build_sft_deps(
    recorder: _SFTRecorder,
    *,
    optim_module: _StubOptimModule,
    training_args_cls: Any = _StubTrainingArgs,
) -> Dict[str, Any]:
    class _RecordingTrainingArgs(training_args_cls):  # type: ignore[misc, valid-type]
        """Subclass so capability detection still sees the dataclass fields."""

        def __init__(self, **kwargs: Any) -> None:
            recorder.training_args_kwargs = dict(kwargs)
            super().__init__(**kwargs)

    def _train(**kwargs: Any) -> None:
        recorder.train_calls.append(kwargs)
        recorder.optimizer = kwargs["optimizer"]
        callback = kwargs.get("training_callback")
        if callback is not None:
            callback.on_val_loss_report({"val_loss": 2.0})
            callback.on_train_loss_report({"train_loss": 1.5})
            callback.on_train_loss_report({"train_loss": 1.0})
        # mlx-lm applies the optimizer once per *accumulated* group.
        kwargs["optimizer"].update(kwargs["model"], {"lora_a": 1.0})

    return {
        "mx": object(),
        "nn": object(),
        "optim": optim_module,
        "tree_flatten": _stub_tree_flatten,
        "mlx_load": lambda name, **kw: (_StubModel(), _StubTokenizer()),
        "TextDataset": lambda records, tokenizer: list(records),
        "CacheDataset": lambda ds: list(ds),
        "TrainingArgs": _RecordingTrainingArgs,
        "mlx_tuner_train": _train,
        "linear_to_lora_layers": lambda model, num_layers, config: None,
    }


def _write_sft_jsonl(tmp_path, n: int = 64) -> str:
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(json.dumps({"text": f"sample {i} body"}) for i in range(n)) + "\n"
    )
    return str(path)


def _run_sft(
    tmp_path,
    monkeypatch,
    *,
    recorder: Optional[_SFTRecorder] = None,
    optim_module: Optional[_StubOptimModule] = None,
    training_args_cls: Any = _StubTrainingArgs,
    **config_overrides: Any,
):
    from halo_forge.sft import mlx_trainer as sft_mlx
    from halo_forge.sft.config import SFTConfig

    recorder = recorder or _SFTRecorder()
    optim_module = optim_module or _StubOptimModule()
    deps = _build_sft_deps(
        recorder, optim_module=optim_module, training_args_cls=training_args_cls
    )
    monkeypatch.setattr(sft_mlx, "_require_mlx_lm", lambda: deps)

    cfg = SFTConfig(
        model_name="stub-mlx-model",
        train_file=_write_sft_jsonl(tmp_path),
        output_dir=str(tmp_path / "out"),
        **config_overrides,
    )
    trainer = sft_mlx.MLXSFTTrainer(cfg)
    summary = trainer.train()
    return trainer, summary, recorder, optim_module


# ----------------------------------------------------------------------
# SFT tests
# ----------------------------------------------------------------------


def test_sft_forwards_gradient_accumulation_to_mlx_lm(tmp_path, monkeypatch):
    """The configured accumulation must reach mlx-lm, not be dropped."""
    _, _, recorder, _ = _run_sft(
        tmp_path, monkeypatch, batch_size=2, num_epochs=1, gradient_accumulation_steps=16
    )
    assert recorder.training_args_kwargs.get("grad_accumulation_steps") == 16
    # `iters` counts micro batches in mlx-lm; it must be a whole number of
    # optimizer steps or the trailing accumulation is silently discarded.
    assert recorder.training_args_kwargs["iters"] % 16 == 0


def test_sft_honors_max_steps(tmp_path, monkeypatch):
    """--max-steps is an optimizer-step ceiling, the standard smoke knob."""
    _, summary, recorder, _ = _run_sft(
        tmp_path,
        monkeypatch,
        batch_size=2,
        num_epochs=3,
        gradient_accumulation_steps=4,
        max_steps=3,
    )
    # 3 optimizer steps x 4 micro batches each.
    assert recorder.training_args_kwargs["iters"] == 12
    assert summary["cycles"][0]["optimizer_steps"] == 3


def test_sft_estimate_iters_respects_max_steps(tmp_path, monkeypatch):
    """Direct unit check on the step-budget helper (no mlx needed)."""
    from halo_forge.sft.config import SFTConfig
    from halo_forge.sft.mlx_trainer import MLXSFTTrainer

    cfg = SFTConfig(
        model_name="stub",
        train_file="unused.jsonl",
        batch_size=2,
        num_epochs=3,
        gradient_accumulation_steps=16,
        max_steps=5,
    )
    trainer = MLXSFTTrainer(cfg)
    iters = trainer._estimate_iters(1000)
    assert iters <= cfg.max_steps * cfg.gradient_accumulation_steps


def test_sft_summary_reports_optimizer_steps_not_micro_batches(tmp_path, monkeypatch):
    _, summary, recorder, _ = _run_sft(
        tmp_path, monkeypatch, batch_size=2, num_epochs=1, gradient_accumulation_steps=8
    )
    cycle = summary["cycles"][0]
    iters = recorder.training_args_kwargs["iters"]
    assert cycle["optimizer_steps"] == iters // 8
    assert cycle["optimizer_steps"] < iters


def _spy_recovery_launch_args(monkeypatch, module) -> Dict[str, Any]:
    """Capture the launch args the trainer writes into relaunch metadata."""
    captured: Dict[str, Any] = {}
    original = module.attach_recovery_guidance

    def _spy(summary, *, modality, launch_args=None, representative_examples=None):
        captured.update(launch_args or {})
        return original(
            summary,
            modality=modality,
            launch_args=launch_args,
            representative_examples=representative_examples,
        )

    monkeypatch.setattr(module, "attach_recovery_guidance", _spy)
    return captured


def test_sft_relaunch_metadata_reports_effective_accumulation(tmp_path, monkeypatch):
    """Relaunch metadata must not claim a value that was not applied."""
    from halo_forge.sft import mlx_trainer as sft_mlx

    monkeypatch.setenv("HALOFORGE_MLX_ALLOW_UNSUPPORTED_GRAD_ACCUM", "1")
    captured = _spy_recovery_launch_args(monkeypatch, sft_mlx)
    _, summary, _, _ = _run_sft(
        tmp_path,
        monkeypatch,
        batch_size=2,
        num_epochs=1,
        gradient_accumulation_steps=16,
        training_args_cls=_LegacyTrainingArgs,
    )
    assert captured["gradient_accumulation_steps"] == 1
    applied = summary["backend_config_applied"]
    assert applied["gradient_accumulation_steps"] == 1
    assert applied["gradient_accumulation_steps_requested"] == 16


def test_sft_hard_errors_when_backend_cannot_accumulate(tmp_path, monkeypatch):
    """A 16x silent divergence is worse than a failed launch."""
    from halo_forge.utils.backend_config import MLXUnsupportedConfigError

    monkeypatch.delenv("HALOFORGE_MLX_ALLOW_UNSUPPORTED_GRAD_ACCUM", raising=False)
    with pytest.raises(MLXUnsupportedConfigError, match="gradient_accumulation_steps"):
        _run_sft(
            tmp_path,
            monkeypatch,
            batch_size=2,
            num_epochs=1,
            gradient_accumulation_steps=16,
            training_args_cls=_LegacyTrainingArgs,
        )


def test_sft_applies_gradient_clipping(tmp_path, monkeypatch):
    optim_module = _StubOptimModule()
    _run_sft(
        tmp_path,
        monkeypatch,
        optim_module=optim_module,
        batch_size=2,
        num_epochs=1,
        gradient_accumulation_steps=1,
        max_grad_norm=0.3,
    )
    assert optim_module.clip_calls == [0.3]


def test_sft_applies_warmup_schedule(tmp_path, monkeypatch):
    _, _, recorder, _ = _run_sft(
        tmp_path,
        monkeypatch,
        batch_size=2,
        num_epochs=2,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        warmup_ratio=0.25,
    )
    lr = recorder.optimizer.learning_rate
    assert callable(lr), "warmup_ratio requires an LR schedule, not a constant"
    assert lr(0) < 2e-4
    steps = recorder.training_args_kwargs["iters"]
    warmup_steps = max(1, int(round(0.25 * steps)))
    assert lr(warmup_steps) == pytest.approx(2e-4, rel=1e-6)


def test_sft_warns_when_backend_cannot_clip_or_schedule(tmp_path, monkeypatch, capsys):
    """Whatever stays unsupported flows through the existing warn guard."""
    optim_module = _StubOptimModule(with_clip=False, with_schedules=False)
    _run_sft(
        tmp_path,
        monkeypatch,
        optim_module=optim_module,
        batch_size=2,
        num_epochs=1,
        gradient_accumulation_steps=1,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
    )
    out = capsys.readouterr().out
    assert "max_grad_norm" in out
    assert "warmup_ratio" in out


# ----------------------------------------------------------------------
# DPO: stub deps bundle + trainer driver
# ----------------------------------------------------------------------


class _FakeArray:
    def __init__(self, values: List[int]) -> None:
        self.values = list(values)

    @property
    def shape(self):
        return (len(self.values),)

    def __getitem__(self, item):
        return _FakeArray(self.values[item])


class _StubMX:
    def __init__(self) -> None:
        self.saved: Dict[str, Any] = {}

    def array(self, values: Any) -> _FakeArray:
        return _FakeArray(list(values))

    def eval(self, *args: Any, **kwargs: Any) -> None:
        return None

    def save_safetensors(self, path: str, weights: Dict[str, Any]) -> None:
        self.saved[path] = weights


class _StubNN:
    @staticmethod
    def value_and_grad(model: Any, fn: Any):
        def _call(model_arg: Any, *args: Any):
            return (0.5, (1.0, 0.0, 0.25)), {"lora_a": 1.0}

        return _call


def _build_dpo_deps(optim_module: _StubOptimModule) -> Dict[str, Any]:
    return {
        "mx": _StubMX(),
        "nn": _StubNN(),
        "optim": optim_module,
        "tree_flatten": _stub_tree_flatten,
        "mlx_load": lambda name, **kw: (_StubModel(), _StubTokenizer()),
        "linear_to_lora_layers": lambda model, num_layers, config: None,
    }


def _run_dpo(
    tmp_path,
    monkeypatch,
    *,
    n_rows: int = 8,
    optim_module: Optional[_StubOptimModule] = None,
    **config_overrides: Any,
):
    from halo_forge.dpo import datasets as dpo_datasets
    from halo_forge.dpo import mlx_trainer as dpo_mlx
    from halo_forge.dpo.config import DPOConfig

    optim_module = optim_module or _StubOptimModule()
    deps = _build_dpo_deps(optim_module)
    monkeypatch.setattr(dpo_mlx, "_require_mlx_lm", lambda: deps)

    rows = [
        {"prompt": f"prompt {i}", "chosen": "good answer", "rejected": "bad answer"}
        for i in range(n_rows)
    ]
    monkeypatch.setattr(
        dpo_datasets, "load_preference_dataset", lambda **kwargs: (rows, rows[:1])
    )

    cfg = DPOConfig(
        model_name="stub-mlx-model",
        train_file=str(tmp_path / "pairs.jsonl"),
        output_dir=str(tmp_path / "dpo_out"),
        reference_free=True,
        **config_overrides,
    )
    trainer = dpo_mlx.MLXDPOTrainer(cfg)
    summary = trainer.train()
    return trainer, summary, optim_module.instances[-1], optim_module


def test_dpo_accumulates_gradients(tmp_path, monkeypatch):
    """8 pairs with accumulation=4 must produce 2 optimizer updates."""
    _, _, optimizer, _ = _run_dpo(
        tmp_path,
        monkeypatch,
        n_rows=8,
        num_epochs=1,
        gradient_accumulation_steps=4,
        warmup_ratio=0.0,
        max_grad_norm=0.0,
    )
    assert len(optimizer.applied_gradients) == 2


def test_dpo_honors_max_steps(tmp_path, monkeypatch):
    _, summary, optimizer, _ = _run_dpo(
        tmp_path,
        monkeypatch,
        n_rows=8,
        num_epochs=3,
        gradient_accumulation_steps=1,
        max_steps=2,
        warmup_ratio=0.0,
        max_grad_norm=0.0,
    )
    assert len(optimizer.applied_gradients) == 2
    assert summary["cycles"][0]["optimizer_steps"] == 2


def test_dpo_applies_gradient_clipping(tmp_path, monkeypatch):
    optim_module = _StubOptimModule()
    _run_dpo(
        tmp_path,
        monkeypatch,
        n_rows=4,
        optim_module=optim_module,
        num_epochs=1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        warmup_ratio=0.0,
    )
    assert optim_module.clip_calls == [1.0, 1.0, 1.0, 1.0]


def test_dpo_relaunch_metadata_records_applied_knobs(tmp_path, monkeypatch):
    from halo_forge.dpo import mlx_trainer as dpo_mlx

    captured = _spy_recovery_launch_args(monkeypatch, dpo_mlx)
    _, summary, _, _ = _run_dpo(
        tmp_path,
        monkeypatch,
        n_rows=8,
        num_epochs=1,
        gradient_accumulation_steps=4,
        warmup_ratio=0.0,
        max_grad_norm=0.0,
    )
    assert captured["gradient_accumulation_steps"] == 4
    assert summary["backend_config_applied"]["gradient_accumulation_steps"] == 4


def test_dpo_reports_effective_batch_size_not_requested(tmp_path, monkeypatch):
    """The MLX DPO loop scores one pair at a time; batch_size is not applied.

    Recording the *requested* ``batch_size`` in relaunch metadata is the same
    lie this module exists to stop: a user asking for ``--batch-size 4``
    ``--gradient-accumulation 2`` gets an effective batch of 2 pairs, not 8,
    and relaunching with the recorded args would not reproduce the run.
    """
    from halo_forge.dpo import mlx_trainer as dpo_mlx

    captured = _spy_recovery_launch_args(monkeypatch, dpo_mlx)
    _, summary, _, _ = _run_dpo(
        tmp_path,
        monkeypatch,
        n_rows=8,
        num_epochs=1,
        batch_size=4,
        gradient_accumulation_steps=2,
        warmup_ratio=0.0,
        max_grad_norm=0.0,
    )
    applied = summary["backend_config_applied"]
    assert applied["batch_size"] == 1
    assert applied["batch_size_requested"] == 4
    # Effective pairs per optimizer step is accumulation alone, not
    # batch_size x accumulation.
    assert applied["effective_batch_size"] == 2
    assert captured["batch_size"] == 1


def test_dpo_warns_when_batch_size_is_not_applied(tmp_path, monkeypatch, capsys):
    _run_dpo(
        tmp_path,
        monkeypatch,
        n_rows=4,
        num_epochs=1,
        batch_size=4,
        gradient_accumulation_steps=1,
        warmup_ratio=0.0,
        max_grad_norm=0.0,
    )
    out = capsys.readouterr().out
    assert "batch_size" in out


def test_dpo_gradient_tree_helpers_are_pure():
    from halo_forge.dpo.mlx_trainer import _accumulate_gradients, _scale_gradients

    a = {"l1": {"w": 1.0, "b": [2.0, 3.0]}}
    b = {"l1": {"w": 3.0, "b": [4.0, 5.0]}}
    summed = _accumulate_gradients(a, b)
    assert summed == {"l1": {"w": 4.0, "b": [6.0, 8.0]}}
    assert _scale_gradients(summed, 0.5) == {"l1": {"w": 2.0, "b": [3.0, 4.0]}}
    # Accumulating onto nothing returns the incoming tree unchanged.
    assert _accumulate_gradients(None, b) == b


# ----------------------------------------------------------------------
# backend_config guard
# ----------------------------------------------------------------------


def test_capability_detection_reads_training_args_and_optim():
    from halo_forge.utils.backend_config import detect_mlx_capabilities

    modern = detect_mlx_capabilities(
        training_args_cls=_StubTrainingArgs, optim_module=_StubOptimModule()
    )
    assert modern.grad_accumulation is True
    assert modern.grad_accumulation_kwarg == "grad_accumulation_steps"
    assert modern.lr_schedule is True
    assert modern.grad_clipping is True

    legacy = detect_mlx_capabilities(
        training_args_cls=_LegacyTrainingArgs,
        optim_module=_StubOptimModule(with_clip=False, with_schedules=False),
    )
    assert legacy.grad_accumulation is False
    assert legacy.grad_accumulation_kwarg is None
    assert legacy.lr_schedule is False
    assert legacy.grad_clipping is False


def test_training_knob_gaps_flow_through_the_existing_warn_guard():
    from halo_forge.sft.config import SFTConfig
    from halo_forge.utils.backend_config import (
        MLXCapabilities,
        mlx_training_knob_gaps,
        warn_unsupported_for_mlx,
    )

    cfg = SFTConfig(gradient_accumulation_steps=8, warmup_ratio=0.1, max_grad_norm=0.3)
    gaps = mlx_training_knob_gaps(MLXCapabilities())
    emitted = warn_unsupported_for_mlx(cfg, trainer_label="MLX SFT", fields=gaps)
    joined = "\n".join(emitted)
    assert "gradient_accumulation_steps" in joined
    assert "warmup_ratio" in joined
    assert "max_grad_norm" in joined


def test_step_budget_is_whole_optimizer_steps():
    from halo_forge.utils.backend_config import resolve_mlx_step_budget

    micro, steps = resolve_mlx_step_budget(
        micro_batches_per_epoch=100, num_epochs=3, gradient_accumulation_steps=16
    )
    assert steps == 300 // 16
    assert micro == steps * 16

    micro, steps = resolve_mlx_step_budget(
        micro_batches_per_epoch=100,
        num_epochs=3,
        gradient_accumulation_steps=16,
        max_steps=4,
    )
    assert (micro, steps) == (64, 4)

    # Tiny datasets still get one real optimizer step.
    micro, steps = resolve_mlx_step_budget(
        micro_batches_per_epoch=1, num_epochs=1, gradient_accumulation_steps=16
    )
    assert (micro, steps) == (16, 1)


def test_peft_field_guard_surface_is_unchanged():
    """The pre-existing four-field guard must keep its exact surface."""
    from halo_forge.utils.backend_config import _MLX_UNSUPPORTED_SFT_FIELDS

    assert [spec[0] for spec in _MLX_UNSUPPORTED_SFT_FIELDS] == [
        "use_dora",
        "use_rslora",
        "init_lora_weights",
        "optim",
    ]
