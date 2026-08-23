"""Unified model conversion (Track I5).

Wraps the format-specific conversion tools behind one ``halo-forge convert``
command so users don't have to remember whether they need ``mlx_lm.convert``,
``llama.cpp/quantize``, or transformers' export. Three target formats:

    --format mlx   — HF safetensors → MLX (with optional quantization).
                     Routes to `mlx_lm.convert`.
    --format gguf  — HF safetensors → GGUF for llama.cpp / Ollama.
                     Routes to the existing `GGUFExporter`.
    --format hf    — pass-through HF safetensors. Optionally re-saves
                     in a different dtype (bf16/fp16/fp32).

The MLX path is the new contribution; GGUF and HF are surfaced under
the unified CLI for parity. Quantization knobs are normalized — `--quant
q4` means "4-bit affine quantization with group size 64" regardless of
format, and the converter translates to the underlying tool's vocabulary.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Normalized quantization name → format-specific arg map. Adding a new
# format means adding a column to this table; the CLI vocabulary stays
# stable.
_QUANT_TABLE: Dict[str, Dict[str, Any]] = {
    "q4": {
        "mlx": {"quantize": True, "q_bits": 4, "q_group_size": 64},
        "gguf": {"quantization": "q4_k_m"},
    },
    "q8": {
        "mlx": {"quantize": True, "q_bits": 8, "q_group_size": 64},
        "gguf": {"quantization": "q8_0"},
    },
    "fp16": {
        "mlx": {"quantize": False, "dtype": "float16"},
        "gguf": {"quantization": "f16"},
    },
    "bf16": {
        "mlx": {"quantize": False, "dtype": "bfloat16"},
        "gguf": {"quantization": "bf16"},
    },
    "fp32": {
        "mlx": {"quantize": False, "dtype": "float32"},
        "gguf": {"quantization": "f32"},
    },
}


@dataclass
class ConvertResult:
    """Outcome of a conversion run, returned to the CLI for printing."""

    source: str
    output_path: str
    target_format: str
    quantization: str
    actual_quantization: Optional[str] = None
    requested_backend_quantization: Optional[str] = None
    actual_backend_quantization: Optional[str] = None
    unquantized_fallback_used: bool = False
    bytes_written: Optional[int] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "output_path": self.output_path,
            "target_format": self.target_format,
            "quantization": self.quantization,
            "actual_quantization": self.actual_quantization or self.quantization,
            "requested_backend_quantization": self.requested_backend_quantization,
            "actual_backend_quantization": self.actual_backend_quantization,
            "unquantized_fallback_used": self.unquantized_fallback_used,
            "bytes_written": self.bytes_written,
            "notes": self.notes,
        }


# `--format hf` is a dtype recast, not a quantizer: there is no 4-bit or
# 8-bit HF pass-through, so asking for one is a category error rather than a
# missing feature.
_HF_SUPPORTED_QUANTS: tuple[str, ...] = ("bf16", "fp16", "fp32")

# Per-format default quantization. A single global default cannot be right:
# `q4` is the sane default for the two real quantizers and is invalid for hf.
_DEFAULT_QUANT_BY_FORMAT: Dict[str, str] = {
    "mlx": "q4",
    "gguf": "q4",
    "hf": "bf16",
}


def list_supported_formats() -> list[str]:
    return ["mlx", "gguf", "hf"]


def list_supported_quants(target_format: Optional[str] = None) -> list[str]:
    """Normalized quant names, optionally narrowed to one target format.

    Args:
        target_format: When given, return only the quants that format can
            actually produce. Without it, return the full vocabulary.
    """
    if target_format is None:
        return list(_QUANT_TABLE)
    canonical = target_format.strip().lower()
    if canonical == "hf":
        return list(_HF_SUPPORTED_QUANTS)
    return [q for q, table in _QUANT_TABLE.items() if canonical in table]


def default_quant_for_format(target_format: str) -> str:
    """Return the quantization to use when the caller didn't pick one.

    ``--format hf`` cannot do true quantization, so it defaults to a bf16
    dtype recast instead of the ``q4`` the quantizing formats default to.
    """
    canonical = target_format.strip().lower()
    if canonical not in _DEFAULT_QUANT_BY_FORMAT:
        raise ValueError(
            f"Unknown target format {target_format!r}. "
            f"Choose from: {', '.join(list_supported_formats())}"
        )
    return _DEFAULT_QUANT_BY_FORMAT[canonical]


def _resolve_quant(quant: str, target_format: str) -> Dict[str, Any]:
    canonical = quant.strip().lower()
    if canonical not in _QUANT_TABLE:
        raise ValueError(
            f"Unknown quantization {quant!r}. " f"Choose from: {', '.join(_QUANT_TABLE)}"
        )
    table = _QUANT_TABLE[canonical]
    if target_format not in table:
        raise ValueError(
            f"Quantization {canonical!r} is not supported for format {target_format!r}. "
            f"Supported quants for this format: "
            f"{', '.join(q for q, t in _QUANT_TABLE.items() if target_format in t)}"
        )
    return dict(table[target_format])


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _canonical_gguf_quantization(backend_quantization: str) -> str:
    normalized = backend_quantization.strip().lower()
    for canonical, formats in _QUANT_TABLE.items():
        candidate = formats.get("gguf", {}).get("quantization")
        if candidate is not None and str(candidate).strip().lower() == normalized:
            return canonical
    raise ValueError(f"GGUF exporter reported unknown quantization {backend_quantization!r}")


def convert_to_mlx(
    *,
    source: str,
    output_path: str,
    quantization: str = "q4",
    trust_remote_code: bool = False,
) -> ConvertResult:
    """Convert a HuggingFace model to MLX format (Apple Silicon)."""
    try:
        from mlx_lm import convert as _mlx_convert
    except ImportError as exc:
        raise ImportError(
            "MLX conversion requires `mlx-lm`. Install with `pip install '.[mlx]'`."
        ) from exc

    args = _resolve_quant(quantization, "mlx")
    logger.info("Converting %s -> %s (mlx, %s, args=%s)", source, output_path, quantization, args)
    out_dir = Path(output_path)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    _mlx_convert(
        hf_path=source,
        mlx_path=str(out_dir),
        trust_remote_code=trust_remote_code,
        **args,
    )

    return ConvertResult(
        source=source,
        output_path=str(out_dir),
        target_format="mlx",
        quantization=quantization,
        bytes_written=_dir_size(out_dir),
    )


GGUF_TOOLING_INSTALL_HINT = (
    "GGUF export is performed by llama.cpp's `convert_hf_to_gguf.py`, which "
    "halo-forge does not vendor and no halo-forge extra installs. Install it "
    "with either:\n"
    "  git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp\n"
    "  pip install llama-cpp-python   (also needs the llama.cpp checkout for "
    "the converter script)\n"
    "halo-forge searches ~/llama.cpp, /opt/llama.cpp, and ./llama.cpp."
)


def _gguf_tooling_missing_reason() -> Optional[str]:
    """Return None when llama.cpp GGUF tooling is usable, else why it isn't.

    `GGUFExporter` only ever succeeds through the llama.cpp convert script —
    its transformers fallback is a stub that always returns False — so we can
    decide this before spending minutes downloading and materializing weights.
    """
    from halo_forge.inference.export import GGUFExporter

    probe = GGUFExporter()
    try:
        probe._check_requirements()
    except Exception as exc:  # pragma: no cover - defensive; probe is pure I/O
        return f"llama.cpp tooling probe failed: {exc}"
    script = getattr(probe, "_convert_script", None)
    if script is None:
        return "no llama.cpp checkout containing convert_hf_to_gguf.py was found"
    if not Path(script).exists():
        return f"llama.cpp converter script {script} is missing"
    return None


def convert_to_gguf(
    *,
    source: str,
    output_path: str,
    quantization: str = "q4",
    trust_remote_code: bool = False,
    allow_unquantized_fallback: bool = False,
) -> ConvertResult:
    """Convert a HuggingFace model to GGUF format (llama.cpp / Ollama).

    Wraps the existing `GGUFExporter` so we have one entry point but
    don't duplicate the conversion logic. Requires a llama.cpp checkout —
    `convert()` preflights that before any weights are loaded, and a direct
    call re-raises the exporter's failure with the install steps attached.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from halo_forge.inference.export import GGUFExporter

    args = _resolve_quant(quantization, "gguf")
    if "quantization" in args:
        args["quantization"] = str(args["quantization"]).upper()

    logger.info("Loading source model on CPU for GGUF export: %s (quant=%s)", source, quantization)
    model = AutoModelForCausalLM.from_pretrained(
        source, trust_remote_code=trust_remote_code, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)

    exporter = GGUFExporter()
    try:
        out_file = exporter.export(
            model,
            output_path,
            tokenizer=tokenizer,
            allow_unquantized_fallback=allow_unquantized_fallback,
            **args,
        )
    except RuntimeError as exc:
        # The exporter's own message says "install llama.cpp" without saying
        # where halo-forge looks or that no extra ships it. Restate it fully.
        raise RuntimeError(f"{exc}\n{GGUF_TOOLING_INSTALL_HINT}") from exc
    evidence = dict(exporter.last_export_evidence)
    requested_backend = str(
        evidence.get("requested_backend_quantization") or args.get("quantization") or ""
    )
    actual_backend = str(evidence.get("actual_backend_quantization") or "")
    if not actual_backend:
        raise RuntimeError("GGUF exporter completed without reporting actual quantization")
    actual_quantization = _canonical_gguf_quantization(actual_backend)
    fallback_used = bool(evidence.get("unquantized_fallback_used", False))

    return ConvertResult(
        source=source,
        output_path=str(out_file),
        target_format="gguf",
        quantization=quantization,
        actual_quantization=actual_quantization,
        requested_backend_quantization=requested_backend,
        actual_backend_quantization=actual_backend,
        unquantized_fallback_used=fallback_used,
        bytes_written=_dir_size(Path(out_file)),
        notes=(
            f"Requested {quantization}; exported {actual_quantization} because "
            f"{evidence.get('fallback_reason', 'an explicit fallback was used')}"
            if fallback_used
            else None
        ),
    )


def convert_to_hf(
    *,
    source: str,
    output_path: str,
    quantization: str = "bf16",
    trust_remote_code: bool = False,
) -> ConvertResult:
    """Pass-through HF re-export, optionally re-cast to a different dtype.

    Useful for "I have a fp32 checkpoint; give me a bf16 one I can serve".
    """
    canonical = quantization.strip().lower()
    if canonical not in _HF_SUPPORTED_QUANTS:
        raise ValueError(
            f"HF re-export only supports dtype changes; got quantization={quantization!r}. "
            f"Valid --quant values for --format hf: {', '.join(_HF_SUPPORTED_QUANTS)} "
            f"(default: {_DEFAULT_QUANT_BY_FORMAT['hf']}). "
            f"Use --format mlx or --format gguf for true quantization."
        )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[canonical]

    logger.info("Re-saving HF model %s as %s -> %s", source, canonical, output_path)
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        source, trust_remote_code=trust_remote_code, device_map="cpu", dtype=dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    return ConvertResult(
        source=source,
        output_path=str(out_dir),
        target_format="hf",
        quantization=canonical,
        bytes_written=_dir_size(out_dir),
    )


# Captured before any caller can rebind the module global. `convert` only
# preflights llama.cpp when it is about to call *this* implementation — the
# toolchain requirement belongs to the shipped GGUF backend, not to the
# dispatch, so an embedder that substitutes its own converter is left alone.
_SHIPPED_GGUF_CONVERTER = convert_to_gguf


def convert(
    *,
    source: str,
    output_path: str,
    target_format: str,
    quantization: Optional[str] = None,
    trust_remote_code: bool = False,
    allow_unquantized_fallback: bool = False,
) -> ConvertResult:
    """Dispatch entry point — picks the right converter for ``target_format``.

    ``quantization=None`` resolves to `default_quant_for_format`, which is
    ``q4`` for the quantizing formats and ``bf16`` for the hf dtype recast.
    GGUF is preflighted here so a missing llama.cpp fails in milliseconds
    instead of after the source checkpoint has been materialized.
    """
    target_format = target_format.strip().lower()
    if quantization is None:
        quantization = default_quant_for_format(target_format)
    if target_format == "gguf" and convert_to_gguf is _SHIPPED_GGUF_CONVERTER:
        missing_reason = _gguf_tooling_missing_reason()
        if missing_reason is not None:
            raise RuntimeError(
                f"Cannot export GGUF: {missing_reason}.\n{GGUF_TOOLING_INSTALL_HINT}"
            )
    if target_format == "mlx":
        return convert_to_mlx(
            source=source,
            output_path=output_path,
            quantization=quantization,
            trust_remote_code=trust_remote_code,
        )
    if target_format == "gguf":
        return convert_to_gguf(
            source=source,
            output_path=output_path,
            quantization=quantization,
            trust_remote_code=trust_remote_code,
            allow_unquantized_fallback=allow_unquantized_fallback,
        )
    if target_format == "hf":
        return convert_to_hf(
            source=source,
            output_path=output_path,
            quantization=quantization,
            trust_remote_code=trust_remote_code,
        )
    raise ValueError(
        f"Unknown target format {target_format!r}. "
        f"Choose from: {', '.join(list_supported_formats())}"
    )


__all__ = [
    "GGUF_TOOLING_INSTALL_HINT",
    "ConvertResult",
    "convert",
    "default_quant_for_format",
    "convert_to_mlx",
    "convert_to_gguf",
    "convert_to_hf",
    "list_supported_formats",
    "list_supported_quants",
]
