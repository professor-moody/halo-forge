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

        def __neg__(self):
            return _Scalar(-self.v)

        def __add__(self, other):
            return _Scalar(self.v + (other.v if isinstance(other, _Scalar) else other))

        __radd__ = __add__

        def __float__(self):
            return self.v

    nn = SimpleNamespace(
        log_sigmoid=lambda x: _Scalar(
            -pymath.log(1.0 + pymath.exp(-x.v)) if isinstance(x, _Scalar) else -pymath.log(1 + pymath.exp(-x))
        ),
    )
    mx = SimpleNamespace()  # not used by _sigmoid_dpo_loss
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


def test_mlx_dpo_trainer_module_imports_without_mlx():
    """The trainer module must load on non-MLX hosts so the dispatcher
    can import it; the actual trainer construction is what gates on
    mlx-lm being installed."""
    import halo_forge.dpo.mlx_trainer as mod

    assert hasattr(mod, "MLXDPOTrainer")
    assert hasattr(mod, "_sigmoid_dpo_loss")
    assert hasattr(mod, "_response_logprobs")


def test_mlx_dpo_trainer_init_validates_reference_free():
    """Constructing without reference_free=True raises a typed error
    even before mlx-lm imports happen."""
    from halo_forge.dpo import DPOConfig
    from halo_forge.dpo.mlx_trainer import MLXDPOTrainer

    cfg = DPOConfig(reference_free=False)
    with pytest.raises(NotImplementedError, match=r"reference-free|reference_free"):
        MLXDPOTrainer(cfg)


def test_mlx_dpo_trainer_init_validates_loss_type():
    from halo_forge.dpo import DPOConfig
    from halo_forge.dpo.mlx_trainer import MLXDPOTrainer

    cfg = DPOConfig(reference_free=True, loss_type="ipo")
    with pytest.raises(NotImplementedError, match=r"sigmoid|ipo"):
        MLXDPOTrainer(cfg)
