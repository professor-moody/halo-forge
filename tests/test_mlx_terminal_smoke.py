"""Bounded Terminal MLX smoke checks.

These checks exercise real MLX array execution without loading a model from
Hugging Face. The terminal smoke runner uses them after the readiness probe so
we can distinguish "MLX math path works" from "not wired yet" cheaply.
"""

from __future__ import annotations

import math

import pytest


pytestmark = pytest.mark.requires_mlx


def test_mlx_dpo_reference_model_terminal():
    import mlx.core as mx
    import mlx.nn as nn

    from halo_forge.dpo.mlx_trainer import _sigmoid_dpo_loss

    loss = _sigmoid_dpo_loss(
        mx=mx,
        nn=nn,
        chosen_logp=mx.array(-2.0),
        rejected_logp=mx.array(-5.0),
        reference_chosen_logp=mx.array(-1.0),
        reference_rejected_logp=mx.array(-5.0),
        beta=1.0,
    )
    mx.eval(loss)

    value = float(loss.item())
    assert math.isfinite(value)
    assert value > math.log(2)


def test_mlx_grpo_terminal():
    import mlx.core as mx
    import mlx.nn as nn

    from halo_forge.dpo.mlx_trainer import _response_logprobs
    from halo_forge.grpo.mlx_trainer import _grpo_policy_loss, _group_advantages

    class TinyUniformPolicy:
        def __call__(self, inputs):
            batch, sequence = inputs.shape
            return mx.zeros((batch, sequence, 8))

    advantages = _group_advantages([0.0, 1.0])
    prompt_tokens = mx.array([1, 2])
    completion_tokens = mx.array([3, 4])
    logp = _response_logprobs(
        mx=mx,
        nn=nn,
        model=TinyUniformPolicy(),
        prompt_tokens=prompt_tokens,
        response_tokens=completion_tokens,
    )
    loss = -advantages[1] * logp
    mx.eval(loss)

    value = float(loss.item())
    assert advantages == pytest.approx([-1.0, 1.0])
    assert math.isfinite(value)
    assert value == pytest.approx(2 * math.log(8), rel=1e-6)

    ref_loss = _grpo_policy_loss(
        mx.array(2.0),
        advantage=1.0,
        beta=0.1,
        reference_logp=mx.array(1.0),
    )
    mx.eval(ref_loss)
    assert float(ref_loss.item()) == pytest.approx(-1.9)
