"""
Dataset-backed burn-in contracts for non-code operational modules.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from halo_forge.ops_e2e_reliability import validate_ops_e2e_module
from halo_forge.ops_module_readiness import OPS_MODULES, validate_ops_module
from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED, normalize_seed


OPS_BURNIN_CONTRACT_VERSION = 1
OPS_BURNIN_GENERATOR_VERSION = "phase7i.v1"
OPS_BURNIN_STATUSES = ("pass", "warn", "fail")
OPS_BURNIN_SOURCES = ("script", "cli_test", "ui_live_compute")
DEFAULT_BURNIN_PROFILE = "tiny-v1"
DEFAULT_OPS_BURNIN_REPORT_FILE = Path("results/readiness/ops_dataset_burnin.v1.json")
DEFAULT_OPS_BURNIN_BASELINE_FILE = Path("tests/baselines/ops_dataset_burnin_baseline.v1.json")
CYCLE_BASED_MODALITIES = {"vlm", "audio", "reasoning", "agentic"}


@dataclass
class OpsDatasetBurninModuleResult:
    """Per-module burn-in contract result."""

    module: str
    status: str
    contract_checks: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    artifacts: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    commands: List[str] = field(default_factory=list)
    duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "contract_checks": dict(self.contract_checks),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "artifacts": dict(self.artifacts),
            "metrics": dict(self.metrics),
            "commands": list(self.commands),
            "duration_ms": int(self.duration_ms),
        }

    @staticmethod
    def from_dict(module: str, payload: Mapping[str, Any]) -> "OpsDatasetBurninModuleResult":
        return OpsDatasetBurninModuleResult(
            module=module,
            status=str(payload.get("status", "fail")),
            contract_checks={
                str(k): bool(v)
                for k, v in (payload.get("contract_checks") or {}).items()
            }
            if isinstance(payload.get("contract_checks"), Mapping)
            else {},
            errors=[str(v) for v in payload.get("errors", []) if v is not None],
            warnings=[str(v) for v in payload.get("warnings", []) if v is not None],
            artifacts={
                str(k): bool(v) for k, v in (payload.get("artifacts") or {}).items()
            }
            if isinstance(payload.get("artifacts"), Mapping)
            else {},
            metrics=dict(payload.get("metrics", {}))
            if isinstance(payload.get("metrics"), Mapping)
            else {},
            commands=[str(v) for v in payload.get("commands", []) if v is not None],
            duration_ms=int(payload.get("duration_ms", 0) or 0),
        )


@dataclass
class OpsDatasetBurninReportV1:
    """Canonical report for dataset-backed non-code ops burn-in."""

    contract_version: int
    generator_version: str
    profile: str
    seed: int
    source: str
    generated_at: str
    modules: Dict[str, OpsDatasetBurninModuleResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": int(self.contract_version),
            "generator_version": self.generator_version,
            "profile": self.profile,
            "seed": int(self.seed),
            "source": self.source,
            "generated_at": self.generated_at,
            "modules": {
                module: entry.to_dict()
                for module, entry in sorted(self.modules.items())
            },
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "OpsDatasetBurninReportV1":
        modules_raw = payload.get("modules") if isinstance(payload.get("modules"), Mapping) else {}
        modules: Dict[str, OpsDatasetBurninModuleResult] = {}
        for module in OPS_MODULES:
            entry_payload = modules_raw.get(module, {})
            if not isinstance(entry_payload, Mapping):
                entry_payload = {}
            modules[module] = OpsDatasetBurninModuleResult.from_dict(module, entry_payload)
        return OpsDatasetBurninReportV1(
            contract_version=int(payload.get("contract_version", 0) or 0),
            generator_version=str(payload.get("generator_version", "")),
            profile=str(payload.get("profile", DEFAULT_BURNIN_PROFILE)),
            seed=int(payload.get("seed", DEFAULT_TRAINING_SEED) or DEFAULT_TRAINING_SEED),
            source=str(payload.get("source", "")),
            generated_at=str(payload.get("generated_at", "")),
            modules=modules,
        )


def validate_ops_burnin_payload(payload: Mapping[str, Any]) -> List[str]:
    """Validate burn-in report payload schema."""
    errors: List[str] = []
    required_top = (
        "contract_version",
        "generator_version",
        "profile",
        "seed",
        "source",
        "generated_at",
        "modules",
    )
    for key in required_top:
        if key not in payload:
            errors.append(f"missing top-level key: {key}")

    try:
        version = int(payload.get("contract_version", 0))
        if version != OPS_BURNIN_CONTRACT_VERSION:
            errors.append(
                f"unsupported contract_version: {version} (expected {OPS_BURNIN_CONTRACT_VERSION})"
            )
    except Exception:
        errors.append("contract_version must be an integer")

    if str(payload.get("generator_version", "")) != OPS_BURNIN_GENERATOR_VERSION:
        errors.append(
            "unsupported generator_version: "
            f"{payload.get('generator_version')} (expected {OPS_BURNIN_GENERATOR_VERSION})"
        )

    source = str(payload.get("source", ""))
    if source and source not in OPS_BURNIN_SOURCES:
        errors.append(f"invalid source value: {source}")

    modules = payload.get("modules")
    if not isinstance(modules, Mapping):
        errors.append("modules must be an object")
        return errors

    generated_at = str(payload.get("generated_at") or "")
    if generated_at:
        try:
            datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        except Exception:
            errors.append("generated_at must be ISO-8601 timestamp")

    for module in OPS_MODULES:
        entry = modules.get(module)
        if not isinstance(entry, Mapping):
            errors.append(f"module entry must be object: {module}")
            continue
        status = str(entry.get("status", ""))
        if status not in OPS_BURNIN_STATUSES:
            errors.append(f"invalid status for {module}: {status}")
        if not isinstance(entry.get("contract_checks", {}), Mapping):
            errors.append(f"contract_checks must be object for {module}")
        if not isinstance(entry.get("artifacts", {}), Mapping):
            errors.append(f"artifacts must be object for {module}")
        if not isinstance(entry.get("metrics", {}), Mapping):
            errors.append(f"metrics must be object for {module}")
        if not isinstance(entry.get("commands", []), list):
            errors.append(f"commands must be list for {module}")
        if not isinstance(entry.get("errors", []), list):
            errors.append(f"errors must be list for {module}")
        if not isinstance(entry.get("warnings", []), list):
            errors.append(f"warnings must be list for {module}")
        try:
            int(entry.get("duration_ms", 0) or 0)
        except Exception:
            errors.append(f"duration_ms must be integer for {module}")

    return errors


def build_ops_burnin_report(
    *,
    module_entries: Mapping[str, OpsDatasetBurninModuleResult],
    profile: str = DEFAULT_BURNIN_PROFILE,
    seed: int = DEFAULT_TRAINING_SEED,
    source: str = "script",
    generated_at: Optional[str] = None,
) -> OpsDatasetBurninReportV1:
    """Build full burn-in report with all expected modules."""
    source_key = str(source or "").strip()
    if source_key not in OPS_BURNIN_SOURCES:
        raise ValueError(f"invalid source '{source_key}'")
    normalized_seed = normalize_seed(seed)
    modules: Dict[str, OpsDatasetBurninModuleResult] = {}
    for module in OPS_MODULES:
        entry = module_entries.get(module)
        if entry is not None:
            modules[module] = entry
            continue
        modules[module] = OpsDatasetBurninModuleResult(
            module=module,
            status="fail",
            contract_checks={},
            errors=[f"missing burn-in module entry: {module}"],
            warnings=[],
            artifacts={},
            metrics={},
            commands=[],
            duration_ms=0,
        )

    return OpsDatasetBurninReportV1(
        contract_version=OPS_BURNIN_CONTRACT_VERSION,
        generator_version=OPS_BURNIN_GENERATOR_VERSION,
        profile=profile,
        seed=normalized_seed,
        source=source_key,
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
        modules=modules,
    )


def write_ops_burnin_report(path: Path, report: OpsDatasetBurninReportV1) -> None:
    """Write burn-in report atomically."""
    payload = report.to_dict()
    schema_errors = validate_ops_burnin_payload(payload)
    if schema_errors:
        raise ValueError("Cannot write invalid burn-in payload: " + "; ".join(schema_errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def load_ops_burnin_report(path: Path) -> OpsDatasetBurninReportV1:
    """Load burn-in report from disk."""
    payload_raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_raw, Mapping):
        raise ValueError("Burn-in report must be JSON object")
    schema_errors = validate_ops_burnin_payload(payload_raw)
    if schema_errors:
        raise ValueError("Invalid burn-in report: " + "; ".join(schema_errors))
    return OpsDatasetBurninReportV1.from_dict(payload_raw)


def normalize_ops_burnin_payload(report: OpsDatasetBurninReportV1) -> Dict[str, Any]:
    """Normalize non-deterministic fields for drift comparisons."""
    payload = report.to_dict()
    payload["generated_at"] = "<normalized>"
    for module in OPS_MODULES:
        entry = payload["modules"][module]
        entry["duration_ms"] = 0
        entry["errors"] = [_normalize_text(v) for v in entry.get("errors", [])]
        entry["warnings"] = [_normalize_text(v) for v in entry.get("warnings", [])]
        entry["commands"] = [_normalize_text(v) for v in entry.get("commands", [])]
        entry["metrics"] = _normalize_mapping(entry.get("metrics", {}))
    return payload


def build_burnin_baseline_payload(report: OpsDatasetBurninReportV1) -> Dict[str, Any]:
    """Build baseline payload from report."""
    normalized = normalize_ops_burnin_payload(report)
    return {
        "contract_version": OPS_BURNIN_CONTRACT_VERSION,
        "generator_version": OPS_BURNIN_GENERATOR_VERSION,
        "profile": normalized["profile"],
        "seed": int(normalized["seed"]),
        "modules": normalized["modules"],
    }


def validate_burnin_baseline_payload(payload: Mapping[str, Any]) -> List[str]:
    """Validate baseline payload schema."""
    errors: List[str] = []
    for key in ("contract_version", "generator_version", "profile", "seed", "modules"):
        if key not in payload:
            errors.append(f"missing top-level key: {key}")
    try:
        version = int(payload.get("contract_version", 0))
        if version != OPS_BURNIN_CONTRACT_VERSION:
            errors.append(
                f"unsupported contract_version: {version} (expected {OPS_BURNIN_CONTRACT_VERSION})"
            )
    except Exception:
        errors.append("contract_version must be integer")

    if str(payload.get("generator_version", "")) != OPS_BURNIN_GENERATOR_VERSION:
        errors.append(
            "unsupported generator_version: "
            f"{payload.get('generator_version')} (expected {OPS_BURNIN_GENERATOR_VERSION})"
        )

    modules = payload.get("modules")
    if not isinstance(modules, Mapping):
        errors.append("modules must be object")
        return errors

    for module in OPS_MODULES:
        entry = modules.get(module)
        if not isinstance(entry, Mapping):
            errors.append(f"missing module baseline entry: {module}")
            continue
        if str(entry.get("status", "")) not in OPS_BURNIN_STATUSES:
            errors.append(f"invalid baseline status for {module}")
        if not isinstance(entry.get("contract_checks", {}), Mapping):
            errors.append(f"contract_checks must be object for {module}")
        if not isinstance(entry.get("artifacts", {}), Mapping):
            errors.append(f"artifacts must be object for {module}")
    return errors


def write_burnin_baseline_file(path: Path, payload: Mapping[str, Any]) -> None:
    """Write baseline JSON atomically."""
    errors = validate_burnin_baseline_payload(payload)
    if errors:
        raise ValueError("Cannot write invalid burn-in baseline: " + "; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def load_burnin_baseline_file(path: Path) -> Dict[str, Any]:
    """Load baseline file."""
    payload_raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_raw, Mapping):
        raise ValueError("Burn-in baseline must be JSON object")
    payload = dict(payload_raw)
    errors = validate_burnin_baseline_payload(payload)
    if errors:
        raise ValueError("Invalid burn-in baseline: " + "; ".join(errors))
    return payload


def compare_burnin_baselines(
    expected: Mapping[str, Any],
    current: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """
    Compare burn-in baselines.

    Severity:
    - hard: lifecycle/contract checks, required artifacts, parser/structure drift
    - warn: status/metrics drift outside hard contract fields
    """
    drifts: List[Dict[str, Any]] = []

    for key in ("contract_version", "generator_version", "profile", "seed"):
        if expected.get(key) != current.get(key):
            drifts.append(
                {
                    "module": "_meta",
                    "path": key,
                    "expected": expected.get(key),
                    "actual": current.get(key),
                    "severity": "hard",
                    "reason": "metadata_mismatch",
                }
            )

    expected_modules = expected.get("modules", {})
    current_modules = current.get("modules", {})
    if not isinstance(expected_modules, Mapping) or not isinstance(current_modules, Mapping):
        drifts.append(
            {
                "module": "_meta",
                "path": "modules",
                "expected": type(expected_modules).__name__,
                "actual": type(current_modules).__name__,
                "severity": "hard",
                "reason": "modules_shape_invalid",
            }
        )
        return drifts

    for module in OPS_MODULES:
        expected_entry = expected_modules.get(module)
        current_entry = current_modules.get(module)
        if not isinstance(expected_entry, Mapping) or not isinstance(current_entry, Mapping):
            drifts.append(
                {
                    "module": module,
                    "path": "entry",
                    "expected": "object",
                    "actual": "missing_or_invalid",
                    "severity": "hard",
                    "reason": "module_entry_missing",
                }
            )
            continue

        _compare_mapping_values(
            module=module,
            namespace="contract_checks",
            expected=expected_entry.get("contract_checks", {}),
            current=current_entry.get("contract_checks", {}),
            severity="hard",
            reason="contract_check_changed",
            drifts=drifts,
        )

        required_artifacts = _required_artifact_keys(module)
        expected_artifacts = expected_entry.get("artifacts", {})
        current_artifacts = current_entry.get("artifacts", {})
        for key in required_artifacts:
            if not isinstance(expected_artifacts, Mapping) or not isinstance(current_artifacts, Mapping):
                drifts.append(
                    {
                        "module": module,
                        "path": f"artifacts.{key}",
                        "expected": "bool",
                        "actual": "missing_or_invalid",
                        "severity": "hard",
                        "reason": "required_artifact_changed",
                    }
                )
                continue
            if bool(expected_artifacts.get(key, False)) != bool(current_artifacts.get(key, False)):
                drifts.append(
                    {
                        "module": module,
                        "path": f"artifacts.{key}",
                        "expected": bool(expected_artifacts.get(key, False)),
                        "actual": bool(current_artifacts.get(key, False)),
                        "severity": "hard",
                        "reason": "required_artifact_changed",
                    }
                )

        if str(expected_entry.get("status", "")) != str(current_entry.get("status", "")):
            drifts.append(
                {
                    "module": module,
                    "path": "status",
                    "expected": expected_entry.get("status"),
                    "actual": current_entry.get("status"),
                    "severity": "warn",
                    "reason": "status_changed",
                }
            )

        _compare_mapping_values(
            module=module,
            namespace="metrics",
            expected=expected_entry.get("metrics", {}),
            current=current_entry.get("metrics", {}),
            severity="warn",
            reason="metric_changed",
            drifts=drifts,
        )

    return drifts


def format_burnin_drift_lines(drifts: Sequence[Mapping[str, Any]]) -> List[str]:
    """Format drifts as parseable lines."""
    lines: List[str] = []
    for drift in drifts:
        lines.append(
            "BURNIN_DRIFT "
            f"severity={drift.get('severity', 'hard')} "
            f"module={drift.get('module', '_meta')} "
            f"path={drift.get('path', '')} "
            f"reason={drift.get('reason', '')} "
            f"expected={drift.get('expected')} "
            f"actual={drift.get('actual')}"
        )
    return lines


def compute_ops_dataset_burnin(
    *,
    profile: str = DEFAULT_BURNIN_PROFILE,
    seed: int = DEFAULT_TRAINING_SEED,
    source: str = "script",
    output_map: Optional[Mapping[str, str]] = None,
    execute_commands: bool = False,
    command_timeout_sec: float = 45.0,
    fixture_pack: str = "v1",
) -> OpsDatasetBurninReportV1:
    """Compute dataset-backed burn-in report."""
    profile_key = _validate_profile(profile)
    resolved_output_map = _resolve_output_map(output_map=output_map, fixture_pack=fixture_pack)

    entries: Dict[str, OpsDatasetBurninModuleResult] = {}
    for module in OPS_MODULES:
        entries[module] = _compute_module_entry(
            module=module,
            output_dir=Path(resolved_output_map[module]),
            seed=seed,
            profile=profile_key,
            execute_commands=execute_commands,
            command_timeout_sec=command_timeout_sec,
        )
    return build_ops_burnin_report(
        module_entries=entries,
        profile=profile_key,
        seed=seed,
        source=source,
    )


def _compute_module_entry(
    *,
    module: str,
    output_dir: Path,
    seed: int,
    profile: str,
    execute_commands: bool,
    command_timeout_sec: float,
) -> OpsDatasetBurninModuleResult:
    start = time.perf_counter()

    e2e = validate_ops_e2e_module(module=module, output_dir=output_dir, seed=seed)
    readiness = validate_ops_module(module=module, output_dir=output_dir, seed=seed)
    errors = _dedupe(list(e2e.errors) + list(readiness.errors))
    warnings = _dedupe(list(e2e.warnings) + list(readiness.warnings))

    contract_checks: Dict[str, bool] = {
        "launch_ok": bool(e2e.launch_ok),
        "stop_ok": bool(e2e.stop_ok),
        "relaunch_ok": bool(e2e.relaunch_ok),
        "readiness_parser_ok": readiness.status in {"pass", "warn"},
    }
    if module in CYCLE_BASED_MODALITIES:
        contract_checks["resume_latest_ok"] = bool(e2e.resume_latest_ok)

    artifacts = _module_artifacts(module=module, output_dir=output_dir, e2e=e2e)
    commands = _profile_commands(module=module, profile=profile, seed=seed)

    command_results: List[int] = []
    if execute_commands:
        for cmd in commands:
            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=command_timeout_sec,
                )
                command_results.append(int(completed.returncode))
                if completed.returncode != 0:
                    warnings.append(
                        f"command returned non-zero ({completed.returncode}): {' '.join(cmd)}"
                    )
            except Exception as exc:
                warnings.append(f"command execution unavailable for {' '.join(cmd)}: {exc}")
                command_results.append(-1)

    # Hard contract checks: lifecycle + required artifacts.
    for check_name, check_value in contract_checks.items():
        if not check_value:
            errors.append(f"required contract check failed: {check_name}")

    for artifact_name in _required_artifact_keys(module):
        if not artifacts.get(artifact_name, False):
            errors.append(f"required artifact missing: {artifact_name}")

    errors = _dedupe(errors)
    warnings = _dedupe(warnings)
    status = "fail" if errors else ("warn" if warnings else "pass")

    metrics: Dict[str, Any] = {
        "seed": int(seed),
        "profile": profile,
        "execute_commands": bool(execute_commands),
        "command_count": len(commands),
        "command_non_zero_count": len([code for code in command_results if code > 0]),
        "command_unavailable_count": len([code for code in command_results if code < 0]),
        "e2e_status": e2e.status,
        "readiness_status": readiness.status,
    }
    metrics.update(_summary_metrics(output_dir))

    duration_ms = int((time.perf_counter() - start) * 1000)
    return OpsDatasetBurninModuleResult(
        module=module,
        status=status,
        contract_checks=contract_checks,
        errors=errors,
        warnings=warnings,
        artifacts=artifacts,
        metrics=metrics,
        commands=[" ".join(cmd) for cmd in commands],
        duration_ms=duration_ms,
    )


def _resolve_output_map(
    *,
    output_map: Optional[Mapping[str, str]],
    fixture_pack: str,
) -> Dict[str, str]:
    default_map = {
        "vlm": str(Path.cwd() / "models/phase7d/vlm_phase7d"),
        "audio": str(Path.cwd() / "models/phase7d/audio_phase7d"),
        "reasoning": str(Path.cwd() / "models/phase7d/reasoning_phase7d"),
        "agentic": str(Path.cwd() / "models/phase7d/agentic_phase7d"),
        "inference": str(Path.cwd() / "models/optimized"),
        "benchmark": str(Path.cwd() / "results/benchmarks"),
        "ui_ops": str(Path.cwd()),
    }

    if fixture_pack:
        pack_root = _resolve_fixture_pack_path(fixture_pack)
        if pack_root.exists() and pack_root.is_dir():
            for module in OPS_MODULES:
                if module == "ui_ops":
                    default_map[module] = str(Path.cwd())
                else:
                    default_map[module] = str(pack_root / module)

    if output_map:
        for module, value in output_map.items():
            if module in OPS_MODULES and value:
                default_map[module] = str(value)
    return default_map


def _resolve_fixture_pack_path(pack: str) -> Path:
    text = str(pack or "").strip()
    if not text:
        return Path("")
    if "/" in text or text.startswith("."):
        path = Path(text).expanduser()
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()
    return (Path.cwd() / "tests" / "fixtures" / "ops_dataset_burnin" / text).resolve()


def _validate_profile(profile: str) -> str:
    profile_key = str(profile or DEFAULT_BURNIN_PROFILE).strip().lower()
    if profile_key != DEFAULT_BURNIN_PROFILE:
        raise ValueError(f"Unsupported burn-in profile: {profile_key}")
    return profile_key


def _profile_commands(module: str, *, profile: str, seed: int) -> List[List[str]]:
    if profile != DEFAULT_BURNIN_PROFILE:
        raise ValueError(f"Unsupported burn-in profile: {profile}")

    base = [sys.executable, "-m", "halo_forge.cli"]
    seed_text = str(seed)
    commands: Dict[str, List[List[str]]] = {
        "vlm": [
            base
            + [
                "vlm",
                "train",
                "--dataset",
                "textvqa",
                "--limit",
                "24",
                "--cycles",
                "1",
                "--samples-per-prompt",
                "1",
                "--seed",
                seed_text,
                "--dry-run",
            ]
        ],
        "audio": [
            base
            + [
                "audio",
                "train",
                "--dataset",
                "librispeech",
                "--task",
                "asr",
                "--cycles",
                "1",
                "--samples-per-prompt",
                "1",
                "--seed",
                seed_text,
                "--dry-run",
            ]
        ],
        "reasoning": [
            base
            + [
                "reasoning",
                "train",
                "--dataset",
                "gsm8k",
                "--limit",
                "32",
                "--cycles",
                "1",
                "--seed",
                seed_text,
                "--dry-run",
            ]
        ],
        "agentic": [
            base
            + [
                "agentic",
                "train",
                "--dataset",
                "xlam",
                "--limit",
                "32",
                "--cycles",
                "1",
                "--seed",
                seed_text,
                "--dry-run",
            ]
        ],
        "inference": [
            base
            + [
                "inference",
                "optimize",
                "--model",
                "Qwen/Qwen2.5-Coder-0.5B",
                "--target-precision",
                "int4",
                "--dry-run",
            ]
        ],
        "benchmark": [
            base
            + [
                "reasoning",
                "benchmark",
                "--model",
                "Qwen/Qwen2.5-7B-Instruct",
                "--dataset",
                "gsm8k",
                "--limit",
                "32",
                "--output",
                "results/readiness/ops_dataset_burnin_benchmark.json",
            ]
        ],
        "ui_ops": [base + ["ui", "--no-browser"]],
    }
    return commands[module]


def _module_artifacts(
    *,
    module: str,
    output_dir: Path,
    e2e: Any,
) -> Dict[str, bool]:
    if module in CYCLE_BASED_MODALITIES:
        return {
            "launch_context": (output_dir / "launch_context.json").exists(),
            "training_summary": (output_dir / "training_summary.json").exists(),
            "latest_checkpoint": (output_dir / "latest_checkpoint.json").exists(),
            "final_model": (output_dir / "final_model").exists(),
        }
    if module == "inference":
        return {
            "launch_context": (output_dir / "launch_context.json").exists(),
            "quantized_or_exported": any(
                [
                    (output_dir / "quantized").exists(),
                    (output_dir / "model.gguf").exists(),
                    (output_dir / "model.onnx").exists(),
                ]
            ),
        }
    if module == "benchmark":
        benchmark_files = list(output_dir.glob("**/benchmark.json"))
        return {
            "benchmark_results": len(benchmark_files) > 0,
            "launch_context": any(output_dir.glob("**/launch_context.json")),
        }
    # ui_ops
    return {
        "ui_routes_contract": bool(e2e.launch_ok),
        "ui_service_contract": bool(e2e.artifacts_ok),
    }


def _required_artifact_keys(module: str) -> Tuple[str, ...]:
    if module in CYCLE_BASED_MODALITIES:
        return ("launch_context", "training_summary", "latest_checkpoint")
    if module == "inference":
        return ("launch_context",)
    if module == "benchmark":
        return ("benchmark_results", "launch_context")
    return ("ui_routes_contract", "ui_service_contract")


def _summary_metrics(output_dir: Path) -> Dict[str, Any]:
    summary_path = output_dir / "training_summary.json"
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, Mapping):
        return {}

    metrics: Dict[str, Any] = {}
    for key in (
        "weights_updated",
        "total_train_steps_executed",
        "final_update_reason",
        "failure_reason",
    ):
        if key in payload:
            metrics[key] = payload[key]
    return metrics


def _compare_mapping_values(
    *,
    module: str,
    namespace: str,
    expected: Any,
    current: Any,
    severity: str,
    reason: str,
    drifts: List[Dict[str, Any]],
) -> None:
    if not isinstance(expected, Mapping) or not isinstance(current, Mapping):
        drifts.append(
            {
                "module": module,
                "path": namespace,
                "expected": type(expected).__name__,
                "actual": type(current).__name__,
                "severity": severity,
                "reason": reason,
            }
        )
        return
    keys = sorted(set(expected.keys()) | set(current.keys()))
    for key in keys:
        expected_value = expected.get(key)
        current_value = current.get(key)
        if expected_value != current_value:
            drifts.append(
                {
                    "module": module,
                    "path": f"{namespace}.{key}",
                    "expected": expected_value,
                    "actual": current_value,
                    "severity": severity,
                    "reason": reason,
                }
            )


def _normalize_mapping(payload: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in sorted(payload.items()):
        if isinstance(value, Mapping):
            normalized[key] = _normalize_mapping(value)
        elif isinstance(value, list):
            normalized[key] = [_normalize_scalar(v) for v in value]
        else:
            normalized[key] = _normalize_scalar(value, key_hint=str(key))
    return normalized


def _normalize_scalar(value: Any, key_hint: str = "") -> Any:
    if isinstance(value, str):
        if key_hint.endswith("_id") or "run_id" in key_hint:
            return "<normalized>"
        return _normalize_text(value)
    return value


def _normalize_text(value: str) -> str:
    text = str(value)
    text = re.sub(r"/tmp/[^\s]+", "/tmp/<normalized>", text)
    text = re.sub(r"/var/folders/[^\s]+", "/var/folders/<normalized>", text)
    return text


def _dedupe(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered
