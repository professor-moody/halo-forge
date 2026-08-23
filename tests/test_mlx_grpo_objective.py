"""Regression tests for the MLX GRPO objective.

The deep review found the MLX GRPO "KL penalty" was a signed log-ratio, which
is linear in ``log pi`` (no restoring force, unbounded below, zero curvature)
and which was additionally *reported* as a KL.  It also found ``epsilon`` echoed
into the run summary while no importance ratio or PPO clip existed anywhere in
the trainer.

These tests pin the objective itself, not the MLX runtime: the loss math is a
pure function of array-like inputs, so it is exercised here with plain floats
and numpy arrays.  The one end-to-end test drives ``MLXGRPOTrainer.train``
against a stub MLX dependency bundle so the *plumbing* (rollout-time log-probs,
clipping, KL reporting) is covered without importing mlx.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from halo_forge.grpo.config import GRPOConfig
from halo_forge.grpo.mlx_trainer import (
    MLXGRPOTrainer,
    _clip_binds,
    _grpo_policy_loss,
    _grpo_reference_kl,
)


def _implied_penalty(policy_logp: float, reference_logp: float) -> float:
    """Isolate the reference term: with advantage=0 and beta=1 the loss *is* it."""
    return float(
        _grpo_policy_loss(
            policy_logp,
            advantage=0.0,
            beta=1.0,
            reference_logp=reference_logp,
        )
    )


# ----------------------------------------------------------------------
# the reference term must be a divergence, not a signed log-ratio
# ----------------------------------------------------------------------


def test_reference_penalty_is_zero_only_when_policy_matches_reference():
    assert _implied_penalty(-2.0, -2.0) == pytest.approx(0.0, abs=1e-12)
    assert _grpo_reference_kl(-2.0, -2.0) == pytest.approx(0.0, abs=1e-12)


def test_reference_penalty_is_positive_in_both_directions():
    """This is the bug: the old term was negative when the policy fell below ref."""
    above = _implied_penalty(-1.5, -2.0)
    below = _implied_penalty(-2.5, -2.0)
    assert above > 0.0, f"penalty above reference must be positive, got {above}"
    assert below > 0.0, f"penalty below reference must be positive, got {below}"


def test_reference_penalty_matches_k3_estimator():
    policy, reference = -2.5, -2.0
    log_ratio = reference - policy
    expected = math.exp(log_ratio) - log_ratio - 1.0
    assert _implied_penalty(policy, reference) == pytest.approx(expected)
    assert _grpo_reference_kl(policy, reference) == pytest.approx(expected)


def test_reference_penalty_gradient_restores_toward_reference():
    """d penalty / d policy_logp must point back at the reference from both sides."""
    reference = -2.0
    step = 1e-6
    for policy, expected_sign in ((-1.5, +1.0), (-2.5, -1.0)):
        grad = (
            _implied_penalty(policy + step, reference)
            - _implied_penalty(policy - step, reference)
        ) / (2 * step)
        assert grad * expected_sign > 0.0, (
            f"gradient at policy={policy} was {grad}; expected sign {expected_sign}"
        )
        analytic = 1.0 - math.exp(reference - policy)
        assert grad == pytest.approx(analytic, abs=1e-5)


def test_reference_penalty_is_convex_and_vectorized_over_numpy():
    reference = np.full(5, -2.0)
    policy = np.array([-4.0, -3.0, -2.0, -1.0, 0.0])
    kl = _grpo_reference_kl(policy, reference, ops=np)
    assert np.all(kl >= 0.0)
    assert kl[2] == pytest.approx(0.0, abs=1e-12)
    curvature = kl[:-2] - 2 * kl[1:-1] + kl[2:]
    assert np.all(curvature > 0.0), f"k3 KL must be convex, got curvature {curvature}"


def test_array_inputs_do_not_require_an_explicit_ops_namespace():
    """Callers that pass arrays without ``ops`` (e.g. the MLX measure scripts) still work."""
    kl = _grpo_reference_kl(np.array([-1.0, -3.0]), np.array([-2.0, -2.0]))
    assert np.all(kl > 0.0)
    loss = _grpo_policy_loss(
        np.array([-1.0]),
        advantage=1.0,
        beta=0.1,
        reference_logp=np.array([-2.0]),
        old_policy_logp=np.array([-2.0]),
        epsilon=0.2,
    )
    assert float(loss[0]) == pytest.approx(-1.2 + 0.1 * (math.exp(-1.0) + 1.0 - 1.0))


def test_reference_free_loss_is_untouched_policy_gradient():
    assert _grpo_policy_loss(2.0, advantage=1.0, beta=0.04) == pytest.approx(-2.0)
    assert _grpo_policy_loss(2.0, advantage=-0.5, beta=0.04) == pytest.approx(1.0)


# ----------------------------------------------------------------------
# importance ratio + PPO clipping
# ----------------------------------------------------------------------


def test_ratio_is_one_when_policy_has_not_moved():
    loss = _grpo_policy_loss(
        -2.0,
        advantage=0.75,
        beta=0.0,
        old_policy_logp=-2.0,
        epsilon=0.2,
    )
    assert loss == pytest.approx(-0.75)


def test_positive_advantage_surrogate_is_clipped_above():
    """Once the ratio passes 1+eps the surrogate flattens — that is the whole point."""
    kwargs: Dict[str, Any] = {
        "advantage": 1.0,
        "beta": 0.0,
        "old_policy_logp": -2.0,
        "epsilon": 0.2,
    }
    inside = _grpo_policy_loss(-2.0 + 0.1, **kwargs)
    at_edge = _grpo_policy_loss(-2.0 + math.log(1.2), **kwargs)
    far = _grpo_policy_loss(-2.0 + 5.0, **kwargs)
    assert inside == pytest.approx(-math.exp(0.1))
    assert at_edge == pytest.approx(-1.2)
    assert far == pytest.approx(-1.2), "clipping must bound the surrogate from below"


def test_negative_advantage_surrogate_is_clipped_below():
    kwargs: Dict[str, Any] = {
        "advantage": -1.0,
        "beta": 0.0,
        "old_policy_logp": -2.0,
        "epsilon": 0.2,
    }
    far_below = _grpo_policy_loss(-2.0 - 5.0, **kwargs)
    assert far_below == pytest.approx(0.8), "loss must flatten at (1-eps)*|A| below"
    at_edge = _grpo_policy_loss(-2.0 + math.log(1.5), **kwargs)
    assert at_edge == pytest.approx(1.5), "pessimistic min keeps the unclipped ratio"


def test_clipped_surrogate_plus_kl_composes():
    policy, old, reference = -1.0, -2.0, -2.0
    loss = _grpo_policy_loss(
        policy,
        advantage=1.0,
        beta=0.1,
        reference_logp=reference,
        old_policy_logp=old,
        epsilon=0.2,
    )
    log_ratio = reference - policy
    expected = -1.2 + 0.1 * (math.exp(log_ratio) - log_ratio - 1.0)
    assert loss == pytest.approx(expected)


@pytest.mark.parametrize(
    "advantage,ratio",
    [(-1.0, 1.6), (1.0, 0.5), (1.0, 1.6), (-1.0, 0.5), (1.0, 1.0), (-1.0, 1.0)],
)
def test_clip_binds_agrees_with_the_actual_surrogate(advantage, ratio):
    """The clip-accounting predicate must match whether the loss really changed."""
    epsilon, old = 0.2, -2.0
    policy = old + math.log(ratio)
    kwargs: Dict[str, Any] = {
        "advantage": advantage,
        "beta": 0.0,
        "old_policy_logp": old,
    }
    clipped = _grpo_policy_loss(policy, epsilon=epsilon, **kwargs)
    unclipped = _grpo_policy_loss(policy, epsilon=None, **kwargs)
    assert _clip_binds(ratio, advantage, epsilon) is (abs(clipped - unclipped) > 1e-12)


def test_clipping_is_skipped_when_epsilon_is_none():
    loss = _grpo_policy_loss(
        -1.0,
        advantage=1.0,
        beta=0.0,
        old_policy_logp=-2.0,
        epsilon=None,
    )
    assert loss == pytest.approx(-math.e)


# ----------------------------------------------------------------------
# reporting: avg_kl must be the quantity it is named after
# ----------------------------------------------------------------------


def test_summary_reports_k3_kl_and_honest_clip_metadata():
    trainer = MLXGRPOTrainer(GRPOConfig(output_dir="unused", epsilon=0.2))
    trainer.run_id = "grpo_mlx_test"
    summary = trainer._build_summary(
        loss_history=[1.0, 0.5],
        kl_history=[0.25, 0.75],
        reward_history=[1.0, 0.0],
        advantage_history=[1.0, -1.0],
        scored_pairs=[("p", "c", 1.0, 1.0), ("p", "c2", 0.0, -1.0)],
        duration_seconds=1.0,
        n_prompts=1,
        train_file="prompts.jsonl",
        dataset=None,
        final_output=Path("out"),
        clipped_updates=1,
    )
    assert summary["avg_kl"] == pytest.approx(0.5)
    assert summary["kl_estimator"] == "k3"
    assert all(value >= 0.0 for value in summary["kl_history"])
    assert summary["epsilon"] == pytest.approx(0.2)
    assert summary["ratio_clipping_applied"] is True
    assert summary["clipped_update_fraction"] == pytest.approx(0.5)


# ----------------------------------------------------------------------
# end-to-end plumbing against a stub MLX dependency bundle
# ----------------------------------------------------------------------


class _StubModel:
    """Policy/reference stand-in whose log-prob drifts as the optimizer steps."""

    def __init__(self, base_logp: float, trainable: bool):
        self.base_logp = base_logp
        self.trainable = trainable
        self.drift = 0.0

    def freeze(self) -> None:
        pass

    def parameters(self) -> Dict[str, Any]:
        return {"w": np.zeros(1)}

    def trainable_parameters(self) -> Dict[str, Any]:
        return {"lora": np.zeros(1)}


class _StubTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str) -> List[int]:
        return [1 + (len(text) % 5), 2, 3]


class _StubOptimizer:
    """Each update nudges the policy so the importance ratio stops being 1."""

    def __init__(self, drift_per_step: float = 0.5):
        self.drift_per_step = drift_per_step
        self.state: Dict[str, Any] = {}
        self.steps = 0

    def update(self, model: _StubModel, grads: Any) -> None:
        model.drift += self.drift_per_step
        self.steps += 1


def _install_mlx_stubs(monkeypatch) -> Tuple[_StubModel, _StubOptimizer]:
    import halo_forge.dpo.mlx_trainer as dpo_mlx
    import halo_forge.grpo.mlx_trainer as grpo_mlx
    import halo_forge.rlvr.mlx_rollout as mlx_rollout
    import halo_forge.rlvr.verifiers as verifiers

    policy = _StubModel(base_logp=-2.0, trainable=True)
    reference = _StubModel(base_logp=-2.0, trainable=False)
    optimizer = _StubOptimizer()
    loaded: List[_StubModel] = [policy, reference]

    def value_and_grad(model, fn):
        def wrapped(*args):
            return fn(*args), {}

        return wrapped

    mx = type(
        "_StubMX",
        (),
        {
            "array": staticmethod(np.asarray),
            "exp": staticmethod(np.exp),
            "minimum": staticmethod(np.minimum),
            "clip": staticmethod(np.clip),
            "eval": staticmethod(lambda *args, **kwargs: None),
            "stop_gradient": staticmethod(lambda value: value),
            "save_safetensors": staticmethod(
                lambda path, arrays: Path(path).write_bytes(b"")
            ),
        },
    )
    nn = type("_StubNN", (), {"value_and_grad": staticmethod(value_and_grad)})

    deps = {
        "mx": mx,
        "nn": nn,
        "optim": type("_StubOptim", (), {"AdamW": staticmethod(lambda **kw: optimizer)}),
        "tree_flatten": lambda tree: list(tree.items()),
        "mlx_load": lambda name: (loaded.pop(0), _StubTokenizer()),
        "linear_to_lora_layers": lambda model, num_layers, config: None,
    }
    monkeypatch.setattr(grpo_mlx, "_require_mlx_lm", lambda: deps)

    def fake_response_logprobs(*, mx, nn, model, prompt_tokens, response_tokens):
        return np.float64(model.base_logp + model.drift)

    monkeypatch.setattr(dpo_mlx, "_response_logprobs", fake_response_logprobs)

    class _StubRollout:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def generate_samples(self, prompts, **kwargs):
            width = kwargs["num_samples"]
            return [(p, f"completion-{i}") for p in prompts for i in range(width)]

        def cleanup(self) -> None:
            pass

    monkeypatch.setattr(mlx_rollout, "MLXRolloutGenerator", _StubRollout)

    class _StubVerifier:
        def verify(self, completion: str):
            reward = 1.0 if completion.endswith("1") else 0.0
            return type("_Result", (), {"reward": reward, "success": reward > 0})()

    monkeypatch.setattr(verifiers, "get_verifier", lambda name: _StubVerifier)
    return policy, optimizer


def test_train_reports_nonnegative_kl_and_applies_the_ratio_clip(monkeypatch, tmp_path):
    _policy, optimizer = _install_mlx_stubs(monkeypatch)

    train_file = tmp_path / "prompts.jsonl"
    train_file.write_text(json.dumps({"prompt": "solve this"}) + "\n")

    cfg = GRPOConfig(
        model_name="stub-model",
        train_file=str(train_file),
        output_dir=str(tmp_path / "out"),
        num_generations=2,
        beta=0.1,
        epsilon=0.2,
        reference_free=False,
        num_epochs=1,
        logging_steps=1000,
    )
    summary = MLXGRPOTrainer(cfg).train()

    assert optimizer.steps == 2
    kl_history = summary["kl_history"]
    assert len(kl_history) == 2
    assert all(value >= 0.0 for value in kl_history), kl_history
    assert summary["kl_estimator"] == "k3"
    assert summary["avg_kl"] == pytest.approx(sum(kl_history) / 2)

    # Update 1: policy == rollout policy, so ratio == 1 and KL == 0.
    # advantage for "completion-0" (reward 0.0) is -1.0 → loss = +1.0.
    assert kl_history[0] == pytest.approx(0.0, abs=1e-12)
    assert summary["cycles"][0]["initial_train_loss"] == pytest.approx(1.0)

    # Update 2: the optimizer moved the policy by +0.5 nats, so the ratio is
    # exp(0.5)=1.649 > 1+eps. With advantage +1 the surrogate must clip at 1.2.
    log_ratio = -0.5
    expected_kl = math.exp(log_ratio) - log_ratio - 1.0
    assert kl_history[1] == pytest.approx(expected_kl)
    assert summary["cycles"][0]["train_loss"] == pytest.approx(-1.2 + 0.1 * expected_kl)
    assert summary["clipped_update_fraction"] == pytest.approx(0.5)
    assert summary["ratio_clipping_applied"] is True


def test_clipped_update_count_only_counts_updates_the_clip_actually_bound(
    monkeypatch, tmp_path
):
    """A ratio outside the trust region does not always mean the clip bound.

    ``loss = -min(rho*A, clip(rho)*A)`` selects the *unclipped* branch when
    ``A < 0`` and ``rho > 1+eps`` (and when ``A > 0`` and ``rho < 1-eps``), so
    counting ``abs(rho - 1) > eps`` overstates how often clipping changed the
    objective — the same class of dishonest metric this module exists to pin.
    """
    _policy, optimizer = _install_mlx_stubs(monkeypatch)

    train_file = tmp_path / "prompts.jsonl"
    train_file.write_text(
        json.dumps({"prompt": "solve this"}) + "\n"
        + json.dumps({"prompt": "and also solve that"}) + "\n"
    )

    cfg = GRPOConfig(
        model_name="stub-model",
        train_file=str(train_file),
        output_dir=str(tmp_path / "out"),
        num_generations=2,
        beta=0.1,
        epsilon=0.2,
        reference_free=False,
        num_epochs=1,
        logging_steps=1000,
    )
    summary = MLXGRPOTrainer(cfg).train()

    # Four updates; the stub optimizer drifts +0.5 nats each, so the ratios are
    # 1.0, e^0.5, e^1.0, e^1.5 against advantages -1, +1, -1, +1.
    #   u1  A=-1 rho=1.00  inside the band          -> no clip
    #   u2  A=+1 rho=1.65  above 1+eps, A>0         -> CLIPPED
    #   u3  A=-1 rho=2.72  above 1+eps but A<0      -> min picks unclipped
    #   u4  A=+1 rho=4.48  above 1+eps, A>0         -> CLIPPED
    assert optimizer.steps == 4
    assert summary["clipped_updates"] == 2, (
        "only updates where the clipped branch won may be counted; "
        f"got {summary['clipped_updates']}"
    )
    assert summary["clipped_update_fraction"] == pytest.approx(0.5)
