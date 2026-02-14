"""
Non-code modality research matrix and runtime evidence validators.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED, normalize_seed


NON_CODE_MODALITIES: Tuple[str, ...] = ("vlm", "audio", "reasoning", "agentic")

CYCLE_BASED_MODALITIES = set(NON_CODE_MODALITIES)

POSITIVE_CASES: Dict[str, Dict[str, Any]] = {
    "vlm": {
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "dataset": "textvqa",
        "train_extra": [],
        "benchmark_extra": [],
    },
    "audio": {
        "model": "openai/whisper-small",
        "dataset": "librispeech",
        "train_extra": ["--task", "asr"],
        "benchmark_extra": ["--task", "asr"],
    },
    "reasoning": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "dataset": "gsm8k",
        "train_extra": ["--limit", "64"],
        "benchmark_extra": ["--limit", "20"],
    },
    "agentic": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "dataset": "xlam",
        "train_extra": ["--limit", "64"],
        "benchmark_extra": ["--limit", "20"],
    },
}

NEGATIVE_CASES: Dict[str, Dict[str, Any]] = {
    "vlm": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "dataset": "textvqa",
    },
    "audio": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "dataset": "librispeech",
        "extra": ["--task", "asr"],
    },
    "reasoning": {
        "model": "openai/whisper-small",
        "dataset": "gsm8k",
    },
    "agentic": {
        "model": "openai/whisper-small",
        "dataset": "xlam",
    },
}


@dataclass(frozen=True)
class ModalityScenario:
    """Single research/testing scenario command definition."""

    modality: str
    scenario: str
    description: str
    command: List[str]
    expects_non_zero: bool = False
    output_dir: str | None = None
    output_file: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactValidationResult:
    """Validation result for a modality training output directory."""

    modality: str
    output_dir: str
    ok: bool
    errors: List[str]
    warnings: List[str]
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "output_dir": self.output_dir,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "evidence": dict(self.evidence),
        }


def build_non_code_modality_matrix(
    *,
    seed: int = DEFAULT_TRAINING_SEED,
    train_output_root: Path | str = Path("models/phase7d"),
    benchmark_output_root: Path | str = Path("results/phase7d"),
    cycles: int = 2,
) -> Dict[str, List[ModalityScenario]]:
    """
    Build deterministic CLI scenario matrix for non-code modality research/testing.
    """
    normalized_seed = normalize_seed(seed)
    train_root = Path(train_output_root)
    benchmark_root = Path(benchmark_output_root)
    matrix: Dict[str, List[ModalityScenario]] = {}

    for modality in NON_CODE_MODALITIES:
        positive = POSITIVE_CASES[modality]
        negative = NEGATIVE_CASES[modality]
        train_output = train_root / f"{modality}_phase7d"
        benchmark_output = benchmark_root / f"{modality}_phase7d" / "benchmark.json"

        scenarios = [
            ModalityScenario(
                modality=modality,
                scenario="train_positive",
                description="Allowlisted family should train and produce artifacts",
                command=_train_command(
                    modality=modality,
                    model=positive["model"],
                    dataset=positive["dataset"],
                    output_dir=train_output,
                    cycles=cycles,
                    seed=normalized_seed,
                    extra=positive.get("train_extra", []),
                ),
                expects_non_zero=False,
                output_dir=str(train_output),
            ),
            ModalityScenario(
                modality=modality,
                scenario="train_negative_unsupported_family",
                description="Unsupported family should fail non-zero with guidance",
                command=_train_command(
                    modality=modality,
                    model=negative["model"],
                    dataset=negative["dataset"],
                    output_dir=train_root / f"{modality}_negative",
                    cycles=1,
                    seed=normalized_seed,
                    extra=negative.get("extra", []),
                ),
                expects_non_zero=True,
                output_dir=str(train_root / f"{modality}_negative"),
            ),
            ModalityScenario(
                modality=modality,
                scenario="benchmark_smoke",
                description="Benchmark output should be discoverable in results views",
                command=_benchmark_command(
                    modality=modality,
                    model_path=train_output / "final_model",
                    dataset=positive["dataset"],
                    output_file=benchmark_output,
                    extra=positive.get("benchmark_extra", []),
                ),
                expects_non_zero=False,
                output_file=str(benchmark_output),
            ),
        ]
        matrix[modality] = scenarios

    return matrix


def matrix_as_json_serializable(
    matrix: Mapping[str, Iterable[ModalityScenario]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Convert matrix object to JSON-serializable payload."""
    payload: Dict[str, List[Dict[str, Any]]] = {}
    for modality, scenarios in matrix.items():
        payload[modality] = [scenario.to_dict() for scenario in scenarios]
    return payload


def validate_modality_training_artifacts(
    *,
    modality: str,
    output_dir: Path | str,
    expected_seed: int = DEFAULT_TRAINING_SEED,
) -> ArtifactValidationResult:
    """
    Validate canonical training/relaunch artifacts for a non-code modality run.
    """
    modality_key = str(modality).strip().lower()
    path = Path(output_dir)
    errors: List[str] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {}
    normalized_seed = normalize_seed(expected_seed)

    if modality_key not in NON_CODE_MODALITIES:
        errors.append(f"unsupported modality: {modality_key}")
        return ArtifactValidationResult(
            modality=modality_key,
            output_dir=str(path),
            ok=False,
            errors=errors,
            warnings=warnings,
            evidence=evidence,
        )

    summary_path = path / "training_summary.json"
    launch_context_path = path / "launch_context.json"
    latest_checkpoint_path = path / "latest_checkpoint.json"
    final_model_path = path / "final_model"

    evidence["training_summary"] = str(summary_path)
    evidence["launch_context"] = str(launch_context_path)
    evidence["latest_checkpoint"] = str(latest_checkpoint_path)
    evidence["final_model"] = str(final_model_path)

    summary_payload: Dict[str, Any] | None = None
    if not summary_path.exists():
        errors.append(f"missing training_summary.json: {summary_path}")
    else:
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(summary_payload, dict):
                errors.append("training_summary.json must contain an object")
                summary_payload = None
        except Exception as e:
            errors.append(f"failed to parse training_summary.json: {e}")

    if not launch_context_path.exists():
        errors.append(f"missing launch_context.json: {launch_context_path}")
    else:
        try:
            launch_context_payload = json.loads(launch_context_path.read_text(encoding="utf-8"))
            if not isinstance(launch_context_payload, dict):
                errors.append("launch_context.json must contain an object")
            else:
                evidence["launch_context_contract_version"] = launch_context_payload.get(
                    "contract_version"
                )
                evidence["launch_context_job_type"] = launch_context_payload.get("job_type")
        except Exception as e:
            errors.append(f"failed to parse launch_context.json: {e}")

    if modality_key in CYCLE_BASED_MODALITIES:
        if not latest_checkpoint_path.exists():
            errors.append(f"missing latest_checkpoint.json: {latest_checkpoint_path}")
        else:
            try:
                checkpoint_payload = json.loads(latest_checkpoint_path.read_text(encoding="utf-8"))
                if not isinstance(checkpoint_payload, dict):
                    errors.append("latest_checkpoint.json must contain an object")
                else:
                    checkpoint_cycle = checkpoint_payload.get("cycle")
                    evidence["checkpoint_cycle"] = checkpoint_cycle
                    try:
                        if int(checkpoint_cycle) < 0:
                            errors.append("latest_checkpoint.json cycle must be >= 0")
                    except Exception:
                        errors.append("latest_checkpoint.json cycle must be an integer")
            except Exception as e:
                errors.append(f"failed to parse latest_checkpoint.json: {e}")

    if not final_model_path.exists():
        warnings.append(f"final_model directory not found: {final_model_path}")

    if summary_payload:
        _validate_summary_payload(
            summary_payload,
            modality=modality_key,
            expected_seed=normalized_seed,
            errors=errors,
            warnings=warnings,
            evidence=evidence,
        )

    return ArtifactValidationResult(
        modality=modality_key,
        output_dir=str(path),
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
    )


def build_validation_report_markdown(
    results: Iterable[ArtifactValidationResult],
) -> str:
    """Render artifact validation results as markdown report."""
    lines: List[str] = ["# Non-Code Modality Artifact Validation Report", ""]
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        lines.append(f"## {result.modality.upper()} - {status}")
        lines.append(f"- Output: `{result.output_dir}`")
        if result.errors:
            lines.append("- Errors:")
            for err in result.errors:
                lines.append(f"  - {err}")
        if result.warnings:
            lines.append("- Warnings:")
            for warn in result.warnings:
                lines.append(f"  - {warn}")
        if result.evidence:
            lines.append("- Evidence:")
            for key, value in sorted(result.evidence.items()):
                lines.append(f"  - {key}: `{value}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _train_command(
    *,
    modality: str,
    model: str,
    dataset: str,
    output_dir: Path,
    cycles: int,
    seed: int,
    extra: Iterable[str],
) -> List[str]:
    cmd = [
        "halo-forge",
        modality,
        "train",
        "--model",
        model,
        "--dataset",
        dataset,
        "--cycles",
        str(cycles),
        "--output",
        str(output_dir),
        "--seed",
        str(seed),
    ]
    cmd.extend(list(extra))
    return cmd


def _benchmark_command(
    *,
    modality: str,
    model_path: Path,
    dataset: str,
    output_file: Path,
    extra: Iterable[str],
) -> List[str]:
    cmd = [
        "halo-forge",
        modality,
        "benchmark",
        "--model",
        str(model_path),
        "--dataset",
        dataset,
        "--output",
        str(output_file),
    ]
    cmd.extend(list(extra))
    return cmd


def _validate_summary_payload(
    payload: Mapping[str, Any],
    *,
    modality: str,
    expected_seed: int,
    errors: List[str],
    warnings: List[str],
    evidence: Dict[str, Any],
) -> None:
    run_id = payload.get("run_id")
    if not run_id:
        errors.append("training_summary missing run_id")
    else:
        evidence["run_id"] = str(run_id)

    summary_seed = payload.get("seed")
    try:
        summary_seed_int = int(summary_seed)
        evidence["seed"] = summary_seed_int
        if summary_seed_int != expected_seed:
            warnings.append(
                f"training_summary seed differs from expected {expected_seed}: {summary_seed_int}"
            )
    except Exception:
        errors.append("training_summary seed must be an integer")

    summary_modality = str(payload.get("modality") or "").strip().lower()
    if summary_modality and summary_modality != modality:
        errors.append(
            f"training_summary modality mismatch: expected {modality}, found {summary_modality}"
        )

    for field_name in ("weights_updated", "total_train_steps_executed", "final_update_reason"):
        if field_name not in payload:
            errors.append(f"training_summary missing {field_name}")

    try:
        evidence["total_train_steps_executed"] = int(
            payload.get("total_train_steps_executed", 0) or 0
        )
    except Exception:
        errors.append("training_summary total_train_steps_executed must be an integer")

    evidence["weights_updated"] = bool(payload.get("weights_updated", False))
    evidence["failure_reason"] = payload.get("failure_reason")


def build_matrix_markdown(matrix: Mapping[str, Iterable[ModalityScenario]]) -> str:
    """Render matrix as markdown for research reports."""
    lines: List[str] = ["# Non-Code Modality CLI Matrix", ""]
    for modality in NON_CODE_MODALITIES:
        lines.append(f"## {modality.upper()}")
        scenarios = list(matrix.get(modality, []))
        if not scenarios:
            lines.append("- No scenarios defined")
            lines.append("")
            continue
        for scenario in scenarios:
            lines.append(f"- `{scenario.scenario}`: {scenario.description}")
            lines.append(f"  - Command: `{_shell_join(scenario.command)}`")
            lines.append(f"  - Expects non-zero: `{str(scenario.expects_non_zero).lower()}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _shell_join(argv: Iterable[str]) -> str:
    escaped: List[str] = []
    for part in argv:
        token = str(part)
        if any(ch in token for ch in (" ", "\t", '"', "'")):
            token = '"' + token.replace('"', '\\"') + '"'
        escaped.append(token)
    return " ".join(escaped)


def parse_validation_targets(values: Iterable[str]) -> List[Tuple[str, Path]]:
    """Parse --validate-training modality=path arguments."""
    targets: List[Tuple[str, Path]] = []
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(
                f"Invalid validation target '{text}'. Expected format: modality=/path/to/output"
            )
        modality, path_text = text.split("=", 1)
        modality_key = modality.strip().lower()
        if modality_key not in NON_CODE_MODALITIES:
            raise ValueError(f"Unsupported modality in validation target: {modality_key}")
        output_path = Path(path_text.strip())
        if not output_path:
            raise ValueError(f"Validation target path is empty: {text}")
        targets.append((modality_key, output_path))
    return targets


def matrix_command_for_docs() -> List[str]:
    """Return canonical script command used in docs/tooling references."""
    return [
        sys.executable,
        "scripts/run_non_code_modality_matrix.py",
        "--print-matrix",
        "--matrix-format",
        "markdown",
    ]
