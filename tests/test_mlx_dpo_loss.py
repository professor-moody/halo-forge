"""MLX DPO loss math tests (Track T17).

These exercise ``_sigmoid_dpo_loss`` and ``_response_logprobs`` directly
with stub ``mx`` and ``nn`` modules. Validates the loss math without
requiring mlx-lm to install (it doesn't on the CI Linux runners).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest


def _make_stub_mx_nn():
    """Minimal stand-in for mlx.core / mlx.nn that's just enough to run
    ``_sigmoid_dpo_loss`` and a stripped ``_response_logprobs`` for the
    math we want to verify."""
    import math as pymath

    class _Scalar:
        """Float-equivalent that supports the operators the loss uses."""

        def __init__(self, v):
            self.v = float(v)

        def __sub__(self, other):
            return _Scalar(self.v - (other.v if isinstance(other, _Scalar) else other))

        def __mul__(self, other):
            return _Scalar(self.v * (other.v if isinstance(other, _Scalar) else other))

        __rmul__ = __mul__

        def __truediv__(self, other):
            return _Scalar(self.v / (other.v if isinstance(other, _Scalar) else other))

        def __rtruediv__(self, other):
            return _Scalar((other.v if isinstance(other, _Scalar) else other) / self.v)

        def __neg__(self):
            return _Scalar(-self.v)

        def __add__(self, other):
            return _Scalar(self.v + (other.v if isinstance(other, _Scalar) else other))

        __radd__ = __add__

        def __rsub__(self, other):
            return _Scalar((other.v if isinstance(other, _Scalar) else other) - self.v)

        def __float__(self):
            return self.v

    nn = SimpleNamespace(
        log_sigmoid=lambda x: _Scalar(
            -pymath.log(1.0 + pymath.exp(-x.v)) if isinstance(x, _Scalar) else -pymath.log(1 + pymath.exp(-x))
        ),
    )
    mx = SimpleNamespace(
        array=lambda x: _Scalar(x),
        maximum=lambda a, b: _Scalar(max(float(a), float(b))),
        exp=lambda x: _Scalar(pymath.exp(float(x))),
    )
    return mx, nn, _Scalar


def test_sigmoid_dpo_loss_zero_when_chosen_equals_rejected():
    """With chosen_logp == rejected_logp the model is indifferent.
    loss = -log σ(0) = -log 0.5 = log 2 ≈ 0.693."""
    from halo_forge.dpo.mlx_trainer import _sigmoid_dpo_loss

    mx, nn, Scalar = _make_stub_mx_nn()
    loss = _sigmoid_dpo_loss(
        mx=mx, nn=nn,
        chosen_logp=Scalar(-5.0),
        rejected_logp=Scalar(-5.0),
        beta=0.1,
    )
    assert float(loss) == pytest.approx(math.log(2), rel=1e-9)


def test_sigmoid_dpo_loss_decreases_when_chosen_preferred():
    """Larger chosen-vs-rejected margin → smaller loss."""
    from halo_forge.dpo.mlx_trainer import _sigmoid_dpo_loss

    mx, nn, Scalar = _make_stub_mx_nn()
    small_margin = _sigmoid_dpo_loss(
        mx=mx, nn=nn,
        chosen_logp=Scalar(-2.0),
        rejected_logp=Scalar(-3.0),
        beta=1.0,
    )
    big_margin = _sigmoid_dpo_loss(
        mx=mx, nn=nn,
        chosen_logp=Scalar(0.0),
        rejected_logp=Scalar(-10.0),
        beta=1.0,
    )
    assert float(big_margin) < float(small_margin)
    assert float(big_margin) > 0  # always positive


def test_sigmoid_dpo_loss_higher_when_rejected_preferred():
    """If the model prefers rejected over chosen, the loss should be
    *higher* than when it prefers chosen — i.e. > log 2."""
    from halo_forge.dpo.mlx_trainer import _sigmoid_dpo_loss

    mx, nn, Scalar = _make_stub_mx_nn()
    loss = _sigmoid_dpo_loss(
        mx=mx, nn=nn,
        chosen_logp=Scalar(-5.0),
        rejected_logp=Scalar(-1.0),
        beta=1.0,
    )
    assert float(loss) > math.log(2)


def test_label_smoothing_pulls_toward_indifference():
    """cDPO label smoothing softens the loss so a confident-correct
    pair doesn't drive loss to ~0 — useful when the dataset has
    label noise."""
    from halo_forge.dpo.mlx_trainer import _sigmoid_dpo_loss

    mx, nn, Scalar = _make_stub_mx_nn()
    no_smooth = _sigmoid_dpo_loss(
        mx=mx, nn=nn,
        chosen_logp=Scalar(0.0), rejected_logp=Scalar(-10.0),
        beta=1.0, label_smoothing=0.0,
    )
    smooth = _sigmoid_dpo_loss(
        mx=mx, nn=nn,
        chosen_logp=Scalar(0.0), rejected_logp=Scalar(-10.0),
        beta=1.0, label_smoothing=0.1,
    )
    assert float(smooth) > float(no_smooth)
    # And smoothing 0 reduces to the standard formula.
    standard = _sigmoid_dpo_loss(
        mx=mx, nn=nn,
        chosen_logp=Scalar(-5.0), rejected_logp=Scalar(-5.0),
        beta=0.1, label_smoothing=0.0,
    )
    assert float(standard) == pytest.approx(math.log(2), rel=1e-9)


def test_reference_model_margin_subtracts_reference_preference():
    """Reference-model DPO should reward policy improvement over the
    reference, not just the raw policy chosen-vs-rejected gap."""
    from halo_forge.dpo.mlx_trainer import _dpo_margin

    _, _, Scalar = _make_stub_mx_nn()
    margin = _dpo_margin(
        chosen_logp=Scalar(-2.0),
        rejected_logp=Scalar(-5.0),
        reference_chosen_logp=Scalar(-1.0),
        reference_rejected_logp=Scalar(-5.0),
    )
    assert float(margin) == pytest.approx(-1.0)


def test_sigmoid_dpo_loss_accepts_reference_model_logps():
    from halo_forge.dpo.mlx_trainer import _sigmoid_dpo_loss

    mx, nn, Scalar = _make_stub_mx_nn()
    loss = _sigmoid_dpo_loss(
        mx=mx,
        nn=nn,
        chosen_logp=Scalar(-2.0),
        rejected_logp=Scalar(-5.0),
        reference_chosen_logp=Scalar(-1.0),
        reference_rejected_logp=Scalar(-5.0),
        beta=1.0,
    )
    assert float(loss) > math.log(2)


def test_reference_margin_requires_both_reference_logps():
    from halo_forge.dpo.mlx_trainer import _dpo_margin

    _, _, Scalar = _make_stub_mx_nn()
    with pytest.raises(ValueError, match="both reference"):
        _dpo_margin(
            chosen_logp=Scalar(-2.0),
            rejected_logp=Scalar(-5.0),
            reference_chosen_logp=Scalar(-1.0),
        )


def test_beta_amplifies_signal():
    """A larger β makes the same logp-margin produce a *smaller* loss
    — this is the KL-regularization knob in the DPO derivation."""
    from halo_forge.dpo.mlx_trainer import _sigmoid_dpo_loss

    mx, nn, Scalar = _make_stub_mx_nn()
    low_beta = _sigmoid_dpo_loss(
        mx=mx, nn=nn,
        chosen_logp=Scalar(0.0), rejected_logp=Scalar(-1.0),
        beta=0.01,
    )
    high_beta = _sigmoid_dpo_loss(
        mx=mx, nn=nn,
        chosen_logp=Scalar(0.0), rejected_logp=Scalar(-1.0),
        beta=1.0,
    )
    assert float(high_beta) < float(low_beta)


def test_ipo_loss_uses_per_token_delta_and_target():
    from halo_forge.dpo.mlx_trainer import _dpo_preference_loss

    mx, nn, Scalar = _make_stub_mx_nn()
    # chosen/rejected logratios divided by length produce delta=5, exactly
    # the IPO target for beta=0.1, so the loss should be zero.
    loss = _dpo_preference_loss(
        mx=mx,
        nn=nn,
        loss_type="ipo",
        chosen_logp=Scalar(10.0),
        rejected_logp=Scalar(0.0),
        beta=0.1,
        chosen_length=2,
        rejected_length=2,
    )
    assert float(loss) == pytest.approx(0.0)


def test_hinge_loss_clamps_positive_margin():
    from halo_forge.dpo.mlx_trainer import _dpo_preference_loss

    mx, nn, Scalar = _make_stub_mx_nn()
    satisfied = _dpo_preference_loss(
        mx=mx,
        nn=nn,
        loss_type="hinge",
        chosen_logp=Scalar(20.0),
        rejected_logp=Scalar(0.0),
        beta=0.1,
    )
    violated = _dpo_preference_loss(
        mx=mx,
        nn=nn,
        loss_type="hinge",
        chosen_logp=Scalar(0.0),
        rejected_logp=Scalar(0.0),
        beta=0.1,
    )
    assert float(satisfied) == pytest.approx(0.0)
    assert float(violated) == pytest.approx(1.0)


def test_kto_pair_loss_rewards_better_chosen_logratio():
    from halo_forge.dpo.mlx_trainer import _dpo_preference_loss

    mx, nn, Scalar = _make_stub_mx_nn()
    good = _dpo_preference_loss(
        mx=mx,
        nn=nn,
        loss_type="kto_pair",
        chosen_logp=Scalar(5.0),
        rejected_logp=Scalar(-5.0),
        beta=0.1,
    )
    bad = _dpo_preference_loss(
        mx=mx,
        nn=nn,
        loss_type="kto_pair",
        chosen_logp=Scalar(-5.0),
        rejected_logp=Scalar(5.0),
        beta=0.1,
    )
    assert float(good) < float(bad)


def test_mlx_dpo_trainer_module_imports_without_mlx():
    """The trainer module must load on non-MLX hosts so the dispatcher
    can import it; the actual trainer construction is what gates on
    mlx-lm being installed."""
    import halo_forge.dpo.mlx_trainer as mod

    assert hasattr(mod, "MLXDPOTrainer")
    assert hasattr(mod, "_sigmoid_dpo_loss")
    assert hasattr(mod, "_dpo_preference_loss")
    assert hasattr(mod, "_response_logprobs")


def test_mlx_dpo_trainer_init_allows_reference_model_sigmoid():
    """Reference-model sigmoid DPO is allowed at construction time.
    mlx-lm is still imported lazily only when training begins."""
    from halo_forge.dpo import DPOConfig
    from halo_forge.dpo.mlx_trainer import MLXDPOTrainer

    cfg = DPOConfig(reference_free=False)
    trainer = MLXDPOTrainer(cfg)
    assert trainer.config.reference_free is False


@pytest.mark.parametrize("loss_type", ["sigmoid", "ipo", "hinge", "kto_pair"])
def test_mlx_dpo_trainer_init_allows_supported_loss_types(loss_type):
    from halo_forge.dpo import DPOConfig
    from halo_forge.dpo.mlx_trainer import MLXDPOTrainer

    trainer = MLXDPOTrainer(DPOConfig(reference_free=True, loss_type=loss_type))
    assert trainer.config.loss_type == loss_type


def test_mlx_dpo_trainer_init_validates_loss_type():
    from halo_forge.dpo import DPOConfig
    from halo_forge.dpo.mlx_trainer import MLXDPOTrainer

    cfg = DPOConfig(reference_free=True, loss_type="rpo")
    with pytest.raises(NotImplementedError, match=r"rpo|loss_type"):
        MLXDPOTrainer(cfg)


def test_mlx_dpo_non_sigmoid_rejects_label_smoothing():
    from halo_forge.dpo import DPOConfig
    from halo_forge.dpo.mlx_trainer import MLXDPOTrainer

    cfg = DPOConfig(reference_free=True, loss_type="hinge", label_smoothing=0.1)
    with pytest.raises(NotImplementedError, match="label_smoothing"):
        MLXDPOTrainer(cfg)
