"""
Deterministic E2E launch reliability contracts for non-code modules.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from halo_forge.modality_research import NON_CODE_MODALITIES, validate_modality_training_artifacts
from halo_forge.ops_module_readiness import OPS_MODULES, default_output_map as default_ops_output_map
from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED, normalize_seed
from ui.services.launch_context import read_launch_context


OPS_E2E_CONTRACT_VERSION = 1
OPS_E2E_STATUSES = ("pass", "warn", "fail")
OPS_E2E_SOURCES = ("script", "cli_test", "ui_live_compute")
DEFAULT_OPS_E2E_REPORT_FILE = Path("results/readiness/ops_e2e_launch_reliability.v1.json")
CYCLE_BASED_MODULES = set(NON_CODE_MODALITIES)


def _status(errors: List[str], warnings: List[str]) -> str:
    if errors:
        return "fail"
    if warnings:
        return "warn"
    return "pass"


@dataclass
class OpsE2EModuleResult:
    """Per-module E2E lifecycle contract result."""

    module: str
    status: str
    launch_ok: bool
    stop_ok: bool
    relaunch_ok: bool
    resume_latest_ok: Optional[bool]
    artifacts_ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    last_output_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "launch_ok": bool(self.launch_ok),
            "stop_ok": bool(self.stop_ok),
            "relaunch_ok": bool(self.relaunch_ok),
            "resume_latest_ok": self.resume_latest_ok,
            "artifacts_ok": bool(self.artifacts_ok),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "evidence": dict(self.evidence),
            "last_output_dir": self.last_output_dir,
        }

    @staticmethod
    def from_dict(module: str, payload: Mapping[str, Any]) -> "OpsE2EModuleResult":
        resume_latest_ok = payload.get("resume_latest_ok")
        parsed_resume: Optional[bool]
        if resume_latest_ok is None:
            parsed_resume = None
        else:
            parsed_resume = bool(resume_latest_ok)
        return OpsE2EModuleResult(
            module=module,
            status=str(payload.get("status", "fail")),
            launch_ok=bool(payload.get("launch_ok", False)),
            stop_ok=bool(payload.get("stop_ok", False)),
            relaunch_ok=bool(payload.get("relaunch_ok", False)),
            resume_latest_ok=parsed_resume,
            artifacts_ok=bool(payload.get("artifacts_ok", False)),
            errors=[str(v) for v in payload.get("errors", []) if v is not None],
            warnings=[str(v) for v in payload.get("warnings", []) if v is not None],
            evidence=dict(payload.get("evidence", {}))
            if isinstance(payload.get("evidence"), Mapping)
            else {},
            last_output_dir=str(payload.get("last_output_dir") or ""),
        )


@dataclass
class OpsE2EReliabilityReport:
    """Canonical E2E launch reliability report."""

    contract_version: int
    generated_at: str
    seed: int
    source: str
    modules: Dict[str, OpsE2EModuleResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": int(self.contract_version),
            "generated_at": self.generated_at,
            "seed": int(self.seed),
            "source": self.source,
            "modules": {
                module: result.to_dict()
                for module, result in sorted(self.modules.items())
            },
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "OpsE2EReliabilityReport":
        modules_raw = payload.get("modules") if isinstance(payload.get("modules"), Mapping) else {}
        modules: Dict[str, OpsE2EModuleResult] = {}
        for module in OPS_MODULES:
            module_payload = modules_raw.get(module, {})
            if not isinstance(module_payload, Mapping):
                module_payload = {}
            modules[module] = OpsE2EModuleResult.from_dict(module, module_payload)
        return OpsE2EReliabilityReport(
            contract_version=int(payload.get("contract_version", 0) or 0),
            generated_at=str(payload.get("generated_at") or ""),
            seed=int(payload.get("seed", DEFAULT_TRAINING_SEED) or DEFAULT_TRAINING_SEED),
            source=str(payload.get("source") or ""),
            modules=modules,
        )


def validate_ops_e2e_payload(payload: Mapping[str, Any]) -> List[str]:
    """Validate report schema."""
    errors: List[str] = []
    required_top_level = ("contract_version", "generated_at", "seed", "source", "modules")
    for key in required_top_level:
        if key not in payload:
            errors.append(f"missing top-level key: {key}")

    try:
        version = int(payload.get("contract_version", 0))
        if version != OPS_E2E_CONTRACT_VERSION:
            errors.append(
                f"unsupported contract_version: {version} (expected {OPS_E2E_CONTRACT_VERSION})"
            )
    except Exception:
        errors.append("contract_version must be an integer")

    source = str(payload.get("source") or "")
    if source and source not in OPS_E2E_SOURCES:
        errors.append(f"invalid source value: {source}")

    generated_at = str(payload.get("generated_at") or "")
    if generated_at:
        try:
            datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        except Exception:
            errors.append("generated_at must be ISO-8601 timestamp")

    modules = payload.get("modules")
    if not isinstance(modules, Mapping):
        errors.append("modules must be an object")
        return errors

    for module in OPS_MODULES:
        entry = modules.get(module)
        if not isinstance(entry, Mapping):
            errors.append(f"module entry must be an object: {module}")
            continue
        status = str(entry.get("status") or "")
        if status not in OPS_E2E_STATUSES:
            errors.append(f"invalid status for {module}: {status}")
        for field_name in ("launch_ok", "stop_ok", "relaunch_ok", "artifacts_ok"):
            if not isinstance(entry.get(field_name), bool):
                errors.append(f"{field_name} must be boolean for {module}")
        resume_latest_ok = entry.get("resume_latest_ok")
        if resume_latest_ok is not None and not isinstance(resume_latest_ok, bool):
            errors.append(f"resume_latest_ok must be boolean|null for {module}")
        if not isinstance(entry.get("errors", []), list):
            errors.append(f"errors must be list for {module}")
        if not isinstance(entry.get("warnings", []), list):
            errors.append(f"warnings must be list for {module}")
        evidence = entry.get("evidence")
        if evidence is not None and not isinstance(evidence, Mapping):
            errors.append(f"evidence must be object for {module}")

    return errors


def build_ops_e2e_report(
    *,
    module_entries: Mapping[str, OpsE2EModuleResult],
    seed: int = DEFAULT_TRAINING_SEED,
    source: str = "script",
    generated_at: Optional[str] = None,
) -> OpsE2EReliabilityReport:
    """Build report ensuring all known modules are populated."""
    normalized_seed = normalize_seed(seed)
    source_key = str(source or "").strip()
    if source_key not in OPS_E2E_SOURCES:
        raise ValueError(f"Invalid report source '{source_key}'")

    modules: Dict[str, OpsE2EModuleResult] = {}
    for module in OPS_MODULES:
        entry = module_entries.get(module)
        if entry is not None:
            modules[module] = entry
            continue
        modules[module] = OpsE2EModuleResult(
            module=module,
            status="fail",
            launch_ok=False,
            stop_ok=False,
            relaunch_ok=False,
            resume_latest_ok=(False if module in CYCLE_BASED_MODULES else None),
            artifacts_ok=False,
            errors=[f"missing e2e module entry: {module}"],
            warnings=[],
            evidence={},
            last_output_dir="",
        )

    return OpsE2EReliabilityReport(
        contract_version=OPS_E2E_CONTRACT_VERSION,
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
        seed=normalized_seed,
        source=source_key,
        modules=modules,
    )


def write_ops_e2e_report(path: Path, report: OpsE2EReliabilityReport) -> None:
    """Write report atomically."""
    payload = report.to_dict()
    schema_errors = validate_ops_e2e_payload(payload)
    if schema_errors:
        raise ValueError("Cannot write invalid report: " + "; ".join(schema_errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def load_ops_e2e_report(path: Path) -> OpsE2EReliabilityReport:
    """Load and validate report from disk."""
    payload_raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_raw, Mapping):
        raise ValueError("E2E report must be a JSON object")
    schema_errors = validate_ops_e2e_payload(payload_raw)
    if schema_errors:
        raise ValueError("Invalid E2E report: " + "; ".join(schema_errors))
    return OpsE2EReliabilityReport.from_dict(payload_raw)


def normalize_ops_e2e_payload(report: OpsE2EReliabilityReport) -> Dict[str, Any]:
    """Return deterministic payload for drift comparisons."""
    payload = report.to_dict()
    payload["generated_at"] = "<normalized>"
    normalized_modules: Dict[str, Any] = {}
    for module, entry in payload["modules"].items():
        normalized_entry = dict(entry)
        normalized_entry["errors"] = [_normalize_text(v) for v in entry.get("errors", [])]
        normalized_entry["warnings"] = [_normalize_text(v) for v in entry.get("warnings", [])]
        evidence = entry.get("evidence", {})
        if isinstance(evidence, Mapping):
            normalized_entry["evidence"] = _normalize_mapping(evidence)
        normalized_modules[module] = normalized_entry
    payload["modules"] = normalized_modules
    return payload


def compute_ops_e2e_reliability(
    output_map: Mapping[str, str] | None = None,
    *,
    seed: int = DEFAULT_TRAINING_SEED,
    source: str = "script",
) -> OpsE2EReliabilityReport:
    """Compute E2E launch reliability contracts for all ops modules."""
    mapping = default_ops_output_map()
    if output_map:
        for key, value in output_map.items():
            if key in OPS_MODULES and value:
                mapping[key] = str(value)

    entries: Dict[str, OpsE2EModuleResult] = {}
    for module in OPS_MODULES:
        entries[module] = validate_ops_e2e_module(
            module=module,
            output_dir=Path(mapping[module]),
            seed=seed,
        )
    return build_ops_e2e_report(module_entries=entries, seed=seed, source=source)


def validate_ops_e2e_module(
    *,
    module: str,
    output_dir: Path,
    seed: int = DEFAULT_TRAINING_SEED,
) -> OpsE2EModuleResult:
    """Validate E2E launch lifecycle contract for one module."""
    key = str(module).strip().lower()
    if key not in OPS_MODULES:
        return OpsE2EModuleResult(
            module=key,
            status="fail",
            launch_ok=False,
            stop_ok=False,
            relaunch_ok=False,
            resume_latest_ok=None,
            artifacts_ok=False,
            errors=[f"unsupported module: {key}"],
            warnings=[],
            evidence={},
            last_output_dir=str(output_dir),
        )

    if key in NON_CODE_MODALITIES:
        return _validate_modality_module(key, output_dir, seed=seed)
    if key == "inference":
        return _validate_inference_module(output_dir)
    if key == "benchmark":
        return _validate_benchmark_module(output_dir)
    return _validate_ui_ops_module(output_dir)


def _validate_modality_module(module: str, output_dir: Path, *, seed: int) -> OpsE2EModuleResult:
    validation = validate_modality_training_artifacts(
        modality=module,
        output_dir=output_dir,
        expected_seed=seed,
    )
    errors = list(validation.errors)
    warnings = list(validation.warnings)
    evidence = dict(validation.evidence)

    launch_ok = False
    relaunch_ok = False
    resume_latest_ok = False
    stop_ok = False
    artifacts_ok = validation.ok

    launch_context_path = Path(str(evidence.get("launch_context", "")))
    latest_checkpoint_path = Path(str(evidence.get("latest_checkpoint", "")))
    summary_path = Path(str(evidence.get("training_summary", "")))

    if not launch_context_path.exists():
        errors.append(f"launch lifecycle contract missing launch_context: {launch_context_path}")
    else:
        try:
            context = read_launch_context(launch_context_path)
            launch_ok = context.job_type == module and context.service == "training"
            relaunch_ok = bool(context.relaunch_capabilities.get("can_relaunch", False))
            resume_latest_ok = bool(context.relaunch_capabilities.get("can_resume_latest", False))
            evidence["launch_context_job_type"] = context.job_type
            evidence["launch_context_service"] = context.service
        except Exception as exc:
            errors.append(f"failed to parse launch_context.json: {exc}")

    if not launch_ok:
        errors.append(f"launch contract invalid for module={module}")
    if not relaunch_ok:
        errors.append(f"relaunch capability missing for module={module}")
    if not resume_latest_ok:
        errors.append(f"resume_latest capability missing for module={module}")
    if not latest_checkpoint_path.exists():
        errors.append(f"resume_latest checkpoint missing: {latest_checkpoint_path}")

    if not summary_path.exists():
        errors.append(f"stop lifecycle evidence missing training_summary: {summary_path}")
    else:
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(summary_payload, Mapping):
                stop_ok = True
                evidence["summary_weights_updated"] = bool(summary_payload.get("weights_updated", False))
            else:
                errors.append("training_summary.json must be a JSON object")
        except Exception as exc:
            errors.append(f"failed to parse training_summary.json: {exc}")

    status = _status(errors, warnings)
    return OpsE2EModuleResult(
        module=module,
        status=status,
        launch_ok=launch_ok,
        stop_ok=stop_ok,
        relaunch_ok=relaunch_ok,
        resume_latest_ok=resume_latest_ok,
        artifacts_ok=artifacts_ok,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
        last_output_dir=str(output_dir),
    )


def _validate_inference_module(output_dir: Path) -> OpsE2EModuleResult:
    errors: List[str] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {"output_dir": str(output_dir)}

    launch_ok = False
    stop_ok = False
    relaunch_ok = False

    launch_context_path = output_dir / "launch_context.json"
    evidence["launch_context"] = str(launch_context_path)
    if not launch_context_path.exists():
        errors.append(f"missing launch_context.json: {launch_context_path}")
    else:
        try:
            context = read_launch_context(launch_context_path)
            launch_ok = context.job_type == "inference" and context.service == "inference"
            relaunch_ok = bool(context.relaunch_capabilities.get("can_relaunch", False))
            stop_ok = launch_ok
            evidence["launch_context_job_type"] = context.job_type
            evidence["launch_context_service"] = context.service
        except Exception as exc:
            errors.append(f"failed to parse launch_context.json: {exc}")

    has_artifacts = any(
        path.exists()
        for path in (
            output_dir / "quantized",
            output_dir / "model.gguf",
            output_dir / "model.onnx",
        )
    )
    evidence["has_optimized_artifact"] = has_artifacts
    if not has_artifacts:
        warnings.append("no optimized inference artifact found")
    if not launch_ok:
        errors.append("launch contract invalid for module=inference")
    if not relaunch_ok:
        errors.append("relaunch capability missing for module=inference")

    status = _status(errors, warnings)
    return OpsE2EModuleResult(
        module="inference",
        status=status,
        launch_ok=launch_ok,
        stop_ok=stop_ok,
        relaunch_ok=relaunch_ok,
        resume_latest_ok=None,
        artifacts_ok=has_artifacts,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
        last_output_dir=str(output_dir),
    )


def _validate_benchmark_module(output_dir: Path) -> OpsE2EModuleResult:
    errors: List[str] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {"output_dir": str(output_dir)}

    benchmark_files = sorted(output_dir.glob("**/benchmark.json"))
    evidence["benchmark_files"] = [str(path) for path in benchmark_files[:10]]
    artifacts_ok = bool(benchmark_files)
    if not benchmark_files:
        errors.append(f"no benchmark.json files found under: {output_dir}")

    parse_errors = 0
    for benchmark_path in benchmark_files:
        try:
            payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                parse_errors += 1
        except Exception:
            parse_errors += 1
    if parse_errors:
        errors.append(f"failed to parse {parse_errors} benchmark result file(s)")

    launch_context_paths = sorted(output_dir.glob("**/launch_context.json"))
    evidence["launch_context_files"] = [str(path) for path in launch_context_paths[:10]]
    launch_ok = bool(launch_context_paths)
    relaunch_ok = False
    if not launch_context_paths:
        errors.append(f"missing benchmark launch_context.json under: {output_dir}")
    else:
        for context_path in launch_context_paths:
            try:
                context = read_launch_context(context_path)
                if context.job_type == "benchmark" and context.service == "benchmark":
                    relaunch_ok = relaunch_ok or bool(
                        context.relaunch_capabilities.get("can_relaunch", False)
                    )
            except Exception as exc:
                warnings.append(f"failed to parse launch context {context_path}: {exc}")

    if not relaunch_ok:
        errors.append("relaunch capability missing for module=benchmark")

    stop_ok = artifacts_ok and parse_errors == 0
    status = _status(errors, warnings)
    return OpsE2EModuleResult(
        module="benchmark",
        status=status,
        launch_ok=launch_ok,
        stop_ok=stop_ok,
        relaunch_ok=relaunch_ok,
        resume_latest_ok=None,
        artifacts_ok=artifacts_ok,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
        last_output_dir=str(output_dir),
    )


def _validate_ui_ops_module(repo_root: Path) -> OpsE2EModuleResult:
    errors: List[str] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {"repo_root": str(repo_root)}

    app_path = repo_root / "ui/app.py"
    cli_path = repo_root / "halo_forge/cli.py"
    required_paths = (app_path, cli_path)
    for path in required_paths:
        if not path.exists():
            errors.append(f"missing ui ops contract file: {path}")
    if app_path.exists():
        app_source = app_path.read_text(encoding="utf-8")
        for route in ("/", "/inference", "/benchmark-advanced", "/research-hub"):
            if route not in app_source:
                errors.append(f"missing route in ui/app.py: {route}")
    if cli_path.exists():
        cli_source = cli_path.read_text(encoding="utf-8")
        if "ui_parser = subparsers.add_parser('ui'" not in cli_source:
            errors.append("missing ui parser command in halo_forge/cli.py")

    status = _status(errors, warnings)
    launch_ok = not errors
    return OpsE2EModuleResult(
        module="ui_ops",
        status=status,
        launch_ok=launch_ok,
        stop_ok=launch_ok,
        relaunch_ok=launch_ok,
        resume_latest_ok=None,
        artifacts_ok=launch_ok,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
        last_output_dir=str(repo_root),
    )


def default_output_map() -> Dict[str, str]:
    """Default output mapping for all modules."""
    return default_ops_output_map()


def _normalize_mapping(data: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in sorted(data.items()):
        if isinstance(value, Mapping):
            normalized[key] = _normalize_mapping(value)
        elif isinstance(value, list):
            normalized[key] = [_normalize_value(item) for item in value]
        else:
            normalized[key] = _normalize_value(value, key_hint=str(key))
    return normalized


def _normalize_value(value: Any, key_hint: str = "") -> Any:
    if isinstance(value, str):
        if "run_id" in key_hint or key_hint.endswith("_id"):
            return "<normalized>"
        return _normalize_text(value)
    return value


def _normalize_text(value: str) -> str:
    text = str(value)
    text = re.sub(r"/tmp/[^\s]+", "/tmp/<normalized>", text)
    text = re.sub(r"/var/folders/[^\s]+", "/var/folders/<normalized>", text)
    return text
