"""Contract tests for `halo-forge convert` / `--verify` (Tracks I4 + I5).

The convert/verify path is the one place where halo-forge claims to catch
silently-broken exports. That claim is only worth something if:

  - a failed round-trip verification produces a **nonzero exit code**, so CI
    and shell scripts actually stop;
  - the formats/quants that cannot work say so precisely, naming the tool to
    install rather than failing after a multi-GB model load;
  - the README's advertised commands and verifier names match the code.

These tests are deliberately import-light: no model is loaded and MLX is
never imported.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"


def _skip_on_unsupported_python() -> None:
    if sys.version_info >= (3, 14):
        pytest.skip("halo-forge CLI supports Python <3.14")


def _fake_convert_result(output_path: str, target_format: str) -> SimpleNamespace:
    """ConvertResult-shaped stand-in; enough for the CLI print path."""
    return SimpleNamespace(
        source="src",
        output_path=output_path,
        target_format=target_format,
        quantization="q4",
        bytes_written=1024,
        notes=None,
    )


def _fake_report(*, passed: bool) -> SimpleNamespace:
    """VerificationReport-shaped stand-in."""
    return SimpleNamespace(
        n_prompts=8,
        exact_match_rate=0.0 if not passed else 1.0,
        avg_char_overlap=0.11 if not passed else 0.99,
        avg_first_token_match=0.0 if not passed else 1.0,
        passed=passed,
        failures=[object()] * (8 if not passed else 0),
        samples=[],
        duration_seconds=1.5,
        notes=None,
    )


# ---------- exit-code contract ---------------------------------------------


def test_cli_convert_verify_failure_exits_nonzero(monkeypatch, tmp_path, capsys):
    """A failed round-trip must exit nonzero.

    Printing the failure in red and returning 0 makes `convert --verify`
    invisible to every caller that checks `$?` — which is the only reason
    the flag exists.
    """
    _skip_on_unsupported_python()
    import halo_forge.cli as cli_mod
    from halo_forge.inference import convert as convert_mod
    from halo_forge.inference import verify_export as verify_mod

    out = tmp_path / "out"
    monkeypatch.setattr(
        convert_mod,
        "convert",
        lambda **kw: _fake_convert_result(str(out), "hf"),
    )
    monkeypatch.setattr(verify_mod, "verify_export", lambda **kw: _fake_report(passed=False))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "halo-forge",
            "convert",
            "--source",
            "src",
            "--format",
            "hf",
            "--quant",
            "bf16",
            "--output",
            str(out),
            "--verify",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli_mod.main()
    assert excinfo.value.code not in (0, None), "failed verification must not exit 0"
    assert "verification failed" in capsys.readouterr().out.lower()


def test_cli_convert_verify_success_exits_zero(monkeypatch, tmp_path, capsys):
    """A passing round-trip must still return control normally."""
    _skip_on_unsupported_python()
    import halo_forge.cli as cli_mod
    from halo_forge.inference import convert as convert_mod
    from halo_forge.inference import verify_export as verify_mod

    out = tmp_path / "out"
    monkeypatch.setattr(
        convert_mod,
        "convert",
        lambda **kw: _fake_convert_result(str(out), "hf"),
    )
    monkeypatch.setattr(verify_mod, "verify_export", lambda **kw: _fake_report(passed=True))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "halo-forge",
            "convert",
            "--source",
            "src",
            "--format",
            "hf",
            "--quant",
            "bf16",
            "--output",
            str(out),
            "--verify",
        ],
    )

    cli_mod.main()  # must not raise SystemExit
    assert "verification passed" in capsys.readouterr().out.lower()


# ---------- format / quant validation --------------------------------------


def test_hf_quant_error_names_the_valid_values():
    """`--format hf --quant q4` is the README's own default combination, so
    the rejection has to say which values DO work, not just what doesn't."""
    from halo_forge.inference.convert import convert_to_hf

    with pytest.raises(ValueError) as excinfo:
        convert_to_hf(source="x", output_path="/tmp/does-not-matter", quantization="q4")
    message = str(excinfo.value)
    for valid in ("bf16", "fp16", "fp32"):
        assert valid in message, f"hf quant error must name {valid!r}: {message}"
    assert "--quant" in message and "--format hf" in message


def test_hf_format_default_quant_is_a_dtype():
    """The per-format default for hf must be usable as-is."""
    from halo_forge.inference.convert import default_quant_for_format

    assert default_quant_for_format("hf") in {"bf16", "fp16", "fp32"}
    assert default_quant_for_format("mlx") == "q4"
    assert default_quant_for_format("gguf") == "q4"


def test_list_supported_quants_is_filterable_per_format():
    from halo_forge.inference.convert import list_supported_quants

    assert set(list_supported_quants("hf")) == {"bf16", "fp16", "fp32"}
    assert "q4" in list_supported_quants("gguf")
    assert "q4" in list_supported_quants()


# ---------- GGUF: fail loudly and early ------------------------------------


def test_gguf_convert_fails_before_loading_weights_when_tooling_missing(monkeypatch):
    """GGUF export shells out to llama.cpp. When that is absent the run must
    abort *before* pulling a multi-GB checkpoint onto the CPU, and the error
    must name what to install."""
    from halo_forge.inference import convert as convert_mod

    loaded: list[str] = []

    class _Boom:
        @staticmethod
        def from_pretrained(*args, **kwargs):  # pragma: no cover - must not run
            loaded.append("model")
            raise AssertionError("model was loaded before the tooling preflight")

    monkeypatch.setattr(convert_mod, "_gguf_tooling_missing_reason", lambda: "no llama.cpp here")
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=_Boom, AutoTokenizer=_Boom),
    )

    with pytest.raises(RuntimeError) as excinfo:
        convert_mod.convert(
            source="x",
            output_path="/tmp/out.gguf",
            target_format="gguf",
            quantization="q4",
        )
    message = str(excinfo.value)
    assert "llama.cpp" in message
    assert "llama-cpp-python" in message
    assert not loaded, "preflight must run before the checkpoint is loaded"


def test_gguf_verification_error_names_the_missing_tooling():
    """`convert --format gguf --verify` cannot work today. The typed error
    has to tell the operator what to do instead of just 'roadmap'."""
    from halo_forge.inference.verify_export import verify_export

    with pytest.raises(NotImplementedError) as excinfo:
        verify_export(source_model="x", exported_path="/tmp/out.gguf", target_format="gguf")
    message = str(excinfo.value)
    assert "llama-cpp-python" in message
    assert "compare_generation" in message


# ---------- README claims ---------------------------------------------------


def _readme_text() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_marquee_convert_example_is_runnable_out_of_the_box():
    """The headline example must not be a command that always exits 1."""
    text = _readme_text()
    marquee = [
        line for line in text.splitlines()
        if line.strip().startswith("halo-forge convert")
    ]
    assert marquee, "README must still document `halo-forge convert`"
    for line in marquee:
        if "--verify" in line:
            assert "--format gguf" not in line, (
                "GGUF round-trip verification is not implemented; the README "
                f"must not advertise it as a working example: {line}"
            )
        if "--format hf" in line:
            assert "--quant q4" not in line, f"`--format hf --quant q4` always errors: {line}"


def test_readme_verifier_count_matches_registry():
    """README says N short names ship; the registry is the source of truth."""
    from halo_forge.rlvr.verifiers.registry import list_registered_verifiers

    text = _readme_text()
    match = re.search(r"(\d+)\s+short names ship out of the box", text)
    assert match, "README must state how many verifier short names ship"
    assert int(match.group(1)) == len(list_registered_verifiers())


def test_readme_zero_config_verifiers_are_actually_zero_arg_constructible(monkeypatch):
    """Every name the README lists under the plain `--verifier <name>` path
    must construct with no arguments; ones that need constructor arguments
    must be documented separately.

    Pinned to the repo root on purpose: `humaneval` and `mbpp` resolve their
    bundled task files through the cwd-relative `data/rlvr/*.jsonl` default and
    raise `FileNotFoundError` from `__init__` anywhere else, which would
    otherwise make this a lottery on the directory pytest was invoked from.
    The README documents that same working-directory requirement.
    """
    from halo_forge.rlvr.verifiers.registry import get_verifier, list_registered_verifiers

    monkeypatch.chdir(REPO_ROOT)
    text = _readme_text()
    start = text.index("### Verifiers")
    section = text[start : text.index("### ", start + 1)]

    needs_args_marker = "Need constructor arguments"
    assert needs_args_marker in section, (
        "README must call out the verifiers that cannot be used via a bare "
        "`--verifier <name>`"
    )
    zero_config_block, needs_args_block = section.split(needs_args_marker, 1)

    zero_config = set(re.findall(r"`([a-z0-9_]+)`", zero_config_block))
    documented_needs_args = set(re.findall(r"`([a-z0-9_]+)`", needs_args_block))
    registered = set(list_registered_verifiers())

    unknown = (zero_config | documented_needs_args) - registered
    assert not unknown, f"README lists unregistered verifier names: {sorted(unknown)}"

    actually_zero_arg = set()
    for name in registered:
        try:
            get_verifier(name)()
        except Exception:
            continue
        actually_zero_arg.add(name)

    assert zero_config == actually_zero_arg, (
        "README's zero-config verifier list disagrees with the registry: "
        f"missing={sorted(actually_zero_arg - zero_config)} "
        f"bogus={sorted(zero_config - actually_zero_arg)}"
    )
    assert documented_needs_args == registered - actually_zero_arg


def test_readme_does_not_claim_inference_extra_ships_gguf_tooling():
    """`.[inference]` installs bitsandbytes/scipy/sentencepiece/protobuf/einops
    — none of which can write a GGUF file."""
    text = _readme_text()
    for line in text.splitlines():
        if '".[inference]"' in line:
            assert "GGUF" not in line, (
                f"`.[inference]` does not provide GGUF export tooling: {line}"
            )
