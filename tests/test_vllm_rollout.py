"""vLLM rollout generator tests (Track I6).

vLLM doesn't install on Apple Silicon (macOS isn't a target), so every
test injects a fake `vllm.LLM` via the constructor's ``llm=`` knob and
a stand-in ``vllm`` module (with the SamplingParams the rollout code
imports) via ``sys.modules``. This exercises the actual code path the
trainer uses without forcing a CUDA-only dependency on CI runners.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional

import pytest


@dataclass
class _StubSamplingParams:
    n: int = 1
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    seed: Optional[int] = None


@pytest.fixture(autouse=True)
def stub_vllm_module(monkeypatch):
    """Provide a `vllm` module with `SamplingParams` so `from vllm import
    SamplingParams` succeeds inside the rollout generator on Macs."""
    if "vllm" in sys.modules:
        # Real vllm is installed (CUDA host). Don't shadow it.
        yield
        return
    fake = ModuleType("vllm")
    fake.SamplingParams = _StubSamplingParams  # type: ignore[attr-defined]
    fake.LLM = object  # only used for failure paths if `_ensure_llm` is reached
    monkeypatch.setitem(sys.modules, "vllm", fake)
    yield
    # monkeypatch.setitem auto-removes on teardown.


# ----- fakes ----------------------------------------------------------------


class _FakeCompletionOutput:
    def __init__(self, text: str):
        self.text = text


class _FakeRequestOutput:
    def __init__(self, completions: list[str]):
        self.outputs = [_FakeCompletionOutput(c) for c in completions]


class _FakeTokenizer:
    """Minimal tokenizer with a chat_template attribute the rollout
    generator's `_format_chat` checks for."""

    chat_template = "yes"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        sys_msg = next(m["content"] for m in messages if m["role"] == "system")
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        return f"<sys>{sys_msg}</sys><usr>{user_msg}</usr>"


class _FakeLLM:
    """Records what was asked and returns deterministic fake completions."""

    def __init__(self, *, completion_template: str = "ANS:{i}"):
        self.completion_template = completion_template
        self.calls: list[tuple[list[str], object]] = []

    def get_tokenizer(self):
        return _FakeTokenizer()

    def generate(self, prompts, sampling_params):
        self.calls.append((list(prompts), sampling_params))
        # If sampling_params is a list (per-request), respect each n.
        if isinstance(sampling_params, list):
            ns = [sp.n for sp in sampling_params]
        else:
            ns = [sampling_params.n] * len(prompts)
        outputs = []
        for prompt, n in zip(prompts, ns):
            completions = [self.completion_template.format(i=i, prompt=prompt) for i in range(n)]
            outputs.append(_FakeRequestOutput(completions))
        return outputs


# ----- backend gating -------------------------------------------------------


def test_init_rejects_mlx_with_typed_error():
    from halo_forge.rlvr.vllm_rollout import VLLMRolloutGenerator, VLLMUnavailableError

    with pytest.raises(VLLMUnavailableError) as ei:
        VLLMRolloutGenerator("model", backend_name="mlx")
    assert "Apple Silicon" in str(ei.value) or "MLX" in str(ei.value)


def test_init_rejects_cpu_and_mps():
    from halo_forge.rlvr.vllm_rollout import VLLMRolloutGenerator, VLLMUnavailableError

    for backend in ("cpu", "mps"):
        with pytest.raises(VLLMUnavailableError):
            VLLMRolloutGenerator("model", backend_name=backend)


def test_init_accepts_cuda_and_rocm():
    """Constructor doesn't import vllm when an llm= is provided so we
    can validate the gating path without a CUDA host."""
    from halo_forge.rlvr.vllm_rollout import VLLMRolloutGenerator

    for backend in ("cuda", "rocm", "rocm_gfx1151"):
        # llm= bypasses the validate_backend call entirely; we still
        # supply the backend_name for storage.
        gen = VLLMRolloutGenerator("model", backend_name=backend, llm=_FakeLLM())
        assert gen.backend_name == backend


# ----- generate_samples -----------------------------------------------------


def test_generate_samples_returns_n_completions_per_prompt():
    from halo_forge.rlvr.vllm_rollout import VLLMRolloutGenerator

    fake = _FakeLLM()
    gen = VLLMRolloutGenerator("model", backend_name="cuda", llm=fake)
    samples = gen.generate_samples(
        ["What is 2+2?", "Who wrote Hamlet?"],
        num_samples=3,
        max_new_tokens=16,
        temperature=0.7,
        batch_size=2,
        system_prompt="be concise",
    )
    # 2 prompts × 3 samples = 6 (prompt, completion) pairs.
    assert len(samples) == 6
    # Both prompts represented evenly.
    counts = {p: sum(1 for pr, _ in samples if pr == p) for p in ["What is 2+2?", "Who wrote Hamlet?"]}
    assert counts == {"What is 2+2?": 3, "Who wrote Hamlet?": 3}
    # SamplingParams.n was 3 (uniform), so we sent one SamplingParams.
    _, sp = fake.calls[0]
    assert hasattr(sp, "n") and sp.n == 3
    # Chat template applied via the fake tokenizer surfaced both sys + user.
    sent_prompts = fake.calls[0][0]
    assert all("<sys>be concise</sys>" in p for p in sent_prompts)


def test_generate_samples_writes_streaming_cache(tmp_path: Path):
    from halo_forge.rlvr.vllm_rollout import VLLMRolloutGenerator

    cache = tmp_path / "samples.jsonl"
    gen = VLLMRolloutGenerator("m", backend_name="cuda", llm=_FakeLLM())
    gen.generate_samples(
        ["P1"],
        num_samples=2,
        max_new_tokens=8,
        temperature=0.5,
        batch_size=1,
        system_prompt="s",
        cache_path=cache,
    )
    lines = [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]
    assert len(lines) == 2
    assert all(row["prompt"] == "P1" for row in lines)


def test_generate_samples_resumes_from_cache(tmp_path: Path):
    """Pre-seeded cache means we ask vLLM for *fewer* completions per prompt."""
    from halo_forge.rlvr.vllm_rollout import VLLMRolloutGenerator

    cache = tmp_path / "samples.jsonl"
    cache.write_text(
        json.dumps({"prompt": "P", "completion": "cached0"}) + "\n"
    )

    fake = _FakeLLM(completion_template="new{i}")
    gen = VLLMRolloutGenerator("m", backend_name="cuda", llm=fake)
    samples = gen.generate_samples(
        ["P"],
        num_samples=3,
        max_new_tokens=8,
        temperature=0.5,
        batch_size=1,
        system_prompt="s",
        cache_path=cache,
    )

    assert len(samples) == 3
    # The fake should have been called with n=2 (one already cached).
    assert len(fake.calls) == 1
    sp = fake.calls[0][1]
    if isinstance(sp, list):
        assert sp[0].n == 2
    else:
        # Uniform path triggered when all needs match — but here only
        # one prompt needs 2, so generator should pick per-request.
        assert sp.n == 2


def test_no_remaining_prompts_skips_vllm_call(tmp_path: Path):
    """Cache full means we don't even instantiate vLLM."""
    from halo_forge.rlvr.vllm_rollout import VLLMRolloutGenerator

    cache = tmp_path / "samples.jsonl"
    rows = [
        json.dumps({"prompt": "P", "completion": f"c{i}"}) + "\n" for i in range(3)
    ]
    cache.write_text("".join(rows))

    fake = _FakeLLM()
    gen = VLLMRolloutGenerator("m", backend_name="cuda", llm=fake)
    samples = gen.generate_samples(
        ["P"],
        num_samples=3,
        max_new_tokens=8,
        temperature=0.5,
        batch_size=1,
        system_prompt="s",
        cache_path=cache,
    )

    assert len(samples) == 3
    assert fake.calls == []  # never called


def test_protocol_compatibility_with_torch_generator():
    """The two rollout generators must accept the same kwargs so the
    trainer is truly drop-in. Just inspecting signatures here."""
    import inspect
    from halo_forge.rlvr.rollout import TorchRolloutGenerator
    from halo_forge.rlvr.vllm_rollout import VLLMRolloutGenerator

    torch_sig = inspect.signature(TorchRolloutGenerator.generate_samples)
    vllm_sig = inspect.signature(VLLMRolloutGenerator.generate_samples)
    # Same parameter names in the same order. This catches accidental
    # protocol drift if either side adds a kwarg the other doesn't.
    assert list(torch_sig.parameters) == list(vllm_sig.parameters)


def test_cli_rollout_engine_choice_registers(monkeypatch, capsys):
    """`halo-forge raft train --rollout-engine vllm --help` must list the
    new choice."""
    import sys
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "raft", "train", "--help"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "rollout-engine" in out and "vllm" in out
