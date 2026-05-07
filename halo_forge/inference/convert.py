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
    bytes_written: Optional[int] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "output_path": self.output_path,
            "target_format": self.target_format,
            "quantization": self.quantization,
            "bytes_written": self.bytes_written,
            "notes": self.notes,
        }


def list_supported_formats() -> list[str]:
    return ["mlx", "gguf", "hf"]


def list_supported_quants() -> list[str]:
    return list(_QUANT_TABLE)


def _resolve_quant(quant: str, target_format: str) -> Dict[str, Any]:
    canonical = quant.strip().lower()
    if canonical not in _QUANT_TABLE:
        raise ValueError(
            f"Unknown quantization {quant!r}. "
            f"Choose from: {', '.join(_QUANT_TABLE)}"
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


def convert_to_mlx(
    *,
    source: str,
    output_path: str,
    quantization: str = "q4",
    trust_remote_code: bool = True,
) -> ConvertResult:
    """Convert a HuggingFace model to MLX format (Apple Silicon)."""
    try:
        from mlx_lm import convert as _mlx_convert
    except ImportError as exc:
        raise ImportError(
            "MLX conversion requires `mlx-lm`. Install with `pip install '.[mlx]'`."
        ) from exc

    args = _resolve_quant(quantization, "mlx")
    logger.info(
        "Converting %s -> %s (mlx, %s, args=%s)", source, output_path, quantization, args
    )
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


def convert_to_gguf(
    *,
    source: str,
    output_path: str,
    quantization: str = "q4",
    trust_remote_code: bool = True,
) -> ConvertResult:
    """Convert a HuggingFace model to GGUF format (llama.cpp / Ollama).

    Wraps the existing `GGUFExporter` so we have one entry point but
    don't duplicate the conversion logic.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from halo_forge.inference.export import GGUFExporter

    args = _resolve_quant(quantization, "gguf")

    logger.info(
        "Loading source model on CPU for GGUF export: %s (quant=%s)", source, quantization
    )
    model = AutoModelForCausalLM.from_pretrained(
        source, trust_remote_code=trust_remote_code, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)

    exporter = GGUFExporter()
    out_file = exporter.export(
        model, output_path, tokenizer=tokenizer, **args
    )

    return ConvertResult(
        source=source,
        output_path=str(out_file),
        target_format="gguf",
        quantization=quantization,
        bytes_written=_dir_size(Path(out_file)),
    )


def convert_to_hf(
    *,
    source: str,
    output_path: str,
    quantization: str = "bf16",
    trust_remote_code: bool = True,
) -> ConvertResult:
    """Pass-through HF re-export, optionally re-cast to a different dtype.

    Useful for "I have a fp32 checkpoint; give me a bf16 one I can serve".
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    canonical = quantization.strip().lower()
    if canonical not in {"bf16", "fp16", "fp32"}:
        raise ValueError(
            f"HF re-export only supports dtype changes; got quantization={quantization!r}. "
            f"Use --format mlx or --format gguf for true quantization."
        )
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


def convert(
    *,
    source: str,
    output_path: str,
    target_format: str,
    quantization: str = "q4",
    trust_remote_code: bool = True,
) -> ConvertResult:
    """Dispatch entry point — picks the right converter for ``target_format``."""
    target_format = target_format.strip().lower()
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
    "ConvertResult",
    "convert",
    "convert_to_mlx",
    "convert_to_gguf",
    "convert_to_hf",
    "list_supported_formats",
    "list_supported_quants",
]
