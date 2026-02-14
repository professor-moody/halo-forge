"""
Canonical readiness report schema/helpers for cross-module UI operations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from halo_forge.modality_readiness import ReadinessCheck
from halo_forge.modality_research import (
    NON_CODE_MODALITIES,
    validate_modality_training_artifacts,
)
from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED, normalize_seed


OPS_MODULES: tuple[str, ...] = (
    "vlm",
    "audio",
    "reasoning",
    "agentic",
    "inference",
    "benchmark",
    "ui_ops",
)
OPS_READINESS_CONTRACT_VERSION = 1
DEFAULT_OPS_READINESS_REPORT_FILE = Path(
    "results/readiness/ops_modules_readiness.v1.json"
)
OPS_READINESS_STALE_AFTER_SECONDS = 24 * 60 * 60
OPS_READINESS_STATUSES = ("pass", "warn", "fail")
OPS_READINESS_SOURCES = ("script", "ui_live_compute")


@dataclass
class OpsModuleReadiness:
    """Readiness payload for one ops module."""

    module: str
    status: str
    checks: Dict[str, ReadinessCheck] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    last_output_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "checks": {
                key: check.to_dict() for key, check in sorted(self.checks.items())
            },
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "evidence": dict(self.evidence),
            "last_output_dir": self.last_output_dir,
        }

    @staticmethod
    def from_dict(module: str, payload: Mapping[str, Any]) -> "OpsModuleReadiness":
        checks_raw = payload.get("checks") if isinstance(payload.get("checks"), Mapping) else {}
        checks: Dict[str, ReadinessCheck] = {}
        for check_name, check_payload in checks_raw.items():
            if not isinstance(check_payload, Mapping):
                continue
            checks[str(check_name)] = ReadinessCheck(
                name=str(check_name),
                status=str(check_payload.get("status", "fail")),
                required=bool(check_payload.get("required", True)),
                message=str(check_payload.get("message", "")),
            )
        return OpsModuleReadiness(
            module=module,
            status=str(payload.get("status", "fail")),
            checks=checks,
            errors=[str(v) for v in payload.get("errors", []) if v is not None],
            warnings=[str(v) for v in payload.get("warnings", []) if v is not None],
            evidence=dict(payload.get("evidence", {}))
            if isinstance(payload.get("evidence"), Mapping)
            else {},
            last_output_dir=str(payload.get("last_output_dir") or ""),
        )


@dataclass
class OpsReadinessReport:
    """Canonical ops readiness report."""

    contract_version: int
    generated_at: str
    seed: int
    source: str
    modules: Dict[str, OpsModuleReadiness]
    stale: bool = False
    age_seconds: Optional[int] = None

    def to_dict(self, include_runtime_fields: bool = False) -> Dict[str, Any]:
        payload = {
            "contract_version": int(self.contract_version),
            "generated_at": self.generated_at,
            "seed": int(self.seed),
            "source": self.source,
            "modules": {
                module: readiness.to_dict()
                for module, readiness in sorted(self.modules.items())
            },
        }
        if include_runtime_fields:
            payload["stale"] = bool(self.stale)
            payload["age_seconds"] = self.age_seconds
        return payload

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "OpsReadinessReport":
        modules_raw = payload.get("modules") if isinstance(payload.get("modules"), Mapping) else {}
        modules: Dict[str, OpsModuleReadiness] = {}
        for module in OPS_MODULES:
            module_payload = modules_raw.get(module, {})
            if not isinstance(module_payload, Mapping):
                module_payload = {}
            modules[module] = OpsModuleReadiness.from_dict(module, module_payload)
        return OpsReadinessReport(
            contract_version=int(payload.get("contract_version", 0) or 0),
            generated_at=str(payload.get("generated_at") or ""),
            seed=int(payload.get("seed", DEFAULT_TRAINING_SEED) or DEFAULT_TRAINING_SEED),
            source=str(payload.get("source") or ""),
            modules=modules,
        )


def readiness_status_from_lists(errors: List[str], warnings: List[str]) -> str:
    """Compute status from errors/warnings."""
    if errors:
        return "fail"
    if warnings:
        return "warn"
    return "pass"


def build_ops_readiness_report(
    *,
    module_entries: Mapping[str, OpsModuleReadiness],
    seed: int = DEFAULT_TRAINING_SEED,
    source: str = "script",
    generated_at: Optional[str] = None,
) -> OpsReadinessReport:
    """Build canonical ops readiness report covering all modules."""
    normalized_seed = normalize_seed(seed)
    source_key = str(source or "").strip()
    if source_key not in OPS_READINESS_SOURCES:
        raise ValueError(
            f"Invalid readiness source '{source_key}'. Expected one of: {OPS_READINESS_SOURCES}"
        )

    report_modules: Dict[str, OpsModuleReadiness] = {}
    for module in OPS_MODULES:
        entry = module_entries.get(module)
        if isinstance(entry, OpsModuleReadiness):
            report_modules[module] = entry
            continue
        report_modules[module] = OpsModuleReadiness(
            module=module,
            status="fail",
            errors=[f"no readiness entry available for module: {module}"],
        )

    return OpsReadinessReport(
        contract_version=OPS_READINESS_CONTRACT_VERSION,
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
        seed=normalized_seed,
        source=source_key,
        modules=report_modules,
    )


def validate_ops_readiness_payload(payload: Mapping[str, Any]) -> List[str]:
    """Validate report schema."""
    errors: List[str] = []
    required_top_level = ("contract_version", "generated_at", "seed", "source", "modules")
    for key in required_top_level:
        if key not in payload:
            errors.append(f"missing top-level key: {key}")

    try:
        version = int(payload.get("contract_version", 0))
        if version != OPS_READINESS_CONTRACT_VERSION:
            errors.append(
                f"unsupported contract_version: {version} (expected {OPS_READINESS_CONTRACT_VERSION})"
            )
    except Exception:
        errors.append("contract_version must be an integer")

    source = str(payload.get("source") or "")
    if source and source not in OPS_READINESS_SOURCES:
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
        if module not in modules:
            errors.append(f"missing module entry: {module}")
            continue
        entry = modules.get(module)
        if not isinstance(entry, Mapping):
            errors.append(f"module entry must be an object: {module}")
            continue
        status = str(entry.get("status") or "")
        if status not in OPS_READINESS_STATUSES:
            errors.append(f"invalid status for {module}: {status}")
        checks = entry.get("checks")
        if checks is not None and not isinstance(checks, Mapping):
            errors.append(f"checks must be an object for {module}")
        else:
            checks_obj = checks or {}
            for check_name, check_payload in checks_obj.items():
                if not isinstance(check_payload, Mapping):
                    errors.append(f"check must be an object for {module}.{check_name}")
                    continue
                check_status = str(check_payload.get("status") or "")
                if check_status not in OPS_READINESS_STATUSES:
                    errors.append(
                        f"invalid check status for {module}.{check_name}: {check_status}"
                    )
                if "required" not in check_payload:
                    errors.append(f"missing required flag for {module}.{check_name}")
        if not isinstance(entry.get("errors", []), list):
            errors.append(f"errors must be a list for {module}")
        if not isinstance(entry.get("warnings", []), list):
            errors.append(f"warnings must be a list for {module}")
        evidence = entry.get("evidence")
        if evidence is not None and not isinstance(evidence, Mapping):
            errors.append(f"evidence must be an object for {module}")

    return errors


def write_ops_readiness_report(path: Path, report: OpsReadinessReport) -> None:
    """Write report atomically."""
    payload = report.to_dict(include_runtime_fields=False)
    errors = validate_ops_readiness_payload(payload)
    if errors:
        raise ValueError("Cannot write invalid ops readiness payload: " + "; ".join(errors))

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def normalized_ops_readiness_payload(report: OpsReadinessReport) -> Dict[str, Any]:
    """Return deterministic payload shape for diffing/reporting."""
    payload = report.to_dict(include_runtime_fields=False)
    # Keep stable ordering and remove non-contract runtime fields if present.
    payload.pop("stale", None)
    payload.pop("age_seconds", None)
    return payload


def load_ops_readiness_report(path: Path) -> OpsReadinessReport:
    """Load report from disk and validate schema."""
    payload_raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_raw, Mapping):
        raise ValueError("Ops readiness report must be a JSON object")
    errors = validate_ops_readiness_payload(payload_raw)
    if errors:
        raise ValueError("Invalid ops readiness payload: " + "; ".join(errors))
    return OpsReadinessReport.from_dict(payload_raw)


def report_age_seconds(report: OpsReadinessReport) -> int:
    """Compute report age in seconds."""
    generated = datetime.fromisoformat(report.generated_at.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    return max(0, int((now - generated).total_seconds()))


def apply_staleness_policy(
    report: OpsReadinessReport,
    stale_after_seconds: int = OPS_READINESS_STALE_AFTER_SECONDS,
) -> OpsReadinessReport:
    """Annotate stale metadata and degrade pass->warn when stale."""
    age = report_age_seconds(report)
    stale = age > stale_after_seconds
    cloned = OpsReadinessReport.from_dict(report.to_dict())
    cloned.age_seconds = age
    cloned.stale = stale
    if not stale:
        return cloned

    for module in OPS_MODULES:
        entry = cloned.modules[module]
        stale_warning = (
            f"readiness report is stale ({age}s old; threshold {stale_after_seconds}s)"
        )
        if stale_warning not in entry.warnings:
            entry.warnings.append(stale_warning)
        if entry.status == "pass":
            entry.status = "warn"
    return cloned


def default_output_map() -> Dict[str, str]:
    """Default output map used for live readiness checks."""
    root = Path.cwd()
    mapping = {
        "vlm": str(root / "models/phase7d/vlm_phase7d"),
        "audio": str(root / "models/phase7d/audio_phase7d"),
        "reasoning": str(root / "models/phase7d/reasoning_phase7d"),
        "agentic": str(root / "models/phase7d/agentic_phase7d"),
        "inference": str(root / "models/optimized"),
        "benchmark": str(root / "results/benchmarks"),
        "ui_ops": str(root),
    }
    return mapping


def compute_ops_module_readiness(
    output_map: Mapping[str, str] | None = None,
    *,
    seed: int = DEFAULT_TRAINING_SEED,
    source: str = "ui_live_compute",
) -> OpsReadinessReport:
    """Compute live readiness for all ops modules."""
    mapping = default_output_map()
    if output_map:
        for key, value in output_map.items():
            if key in OPS_MODULES and value:
                mapping[key] = str(value)

    entries: Dict[str, OpsModuleReadiness] = {}
    for module in OPS_MODULES:
        entries[module] = validate_ops_module(
            module=module,
            output_dir=Path(mapping[module]),
            seed=seed,
        )

    return build_ops_readiness_report(
        module_entries=entries,
        seed=seed,
        source=source,
    )


def validate_ops_module(
    *,
    module: str,
    output_dir: Path,
    seed: int = DEFAULT_TRAINING_SEED,
) -> OpsModuleReadiness:
    """Validate one module readiness from its canonical output directory."""
    key = str(module).strip().lower()
    if key not in OPS_MODULES:
        return OpsModuleReadiness(
            module=key,
            status="fail",
            errors=[f"unsupported module: {key}"],
            last_output_dir=str(output_dir),
        )

    if key in NON_CODE_MODALITIES:
        return _modality_module_readiness(key, output_dir, seed=seed)
    if key == "inference":
        return _inference_module_readiness(output_dir)
    if key == "benchmark":
        return _benchmark_module_readiness(output_dir)
    return _ui_ops_module_readiness(output_dir)


def _modality_module_readiness(
    module: str,
    output_dir: Path,
    *,
    seed: int,
) -> OpsModuleReadiness:
    result = validate_modality_training_artifacts(
        modality=module,
        output_dir=output_dir,
        expected_seed=seed,
    )
    evidence = dict(result.evidence)
    checks = _build_path_checks(
        evidence=evidence,
        required_keys=("training_summary", "launch_context", "latest_checkpoint"),
        optional_keys=("final_model",),
    )
    status = readiness_status_from_lists(result.errors, result.warnings)
    return OpsModuleReadiness(
        module=module,
        status=status,
        checks=checks,
        errors=list(result.errors),
        warnings=list(result.warnings),
        evidence=evidence,
        last_output_dir=result.output_dir,
    )


def _inference_module_readiness(output_dir: Path) -> OpsModuleReadiness:
    errors: List[str] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "launch_context": str(output_dir / "launch_context.json"),
        "quantized_dir": str(output_dir / "quantized"),
        "model_gguf": str(output_dir / "model.gguf"),
        "model_onnx": str(output_dir / "model.onnx"),
    }

    if not output_dir.exists():
        errors.append(f"inference output directory not found: {output_dir}")

    launch_context_path = output_dir / "launch_context.json"
    if not launch_context_path.exists():
        errors.append(f"missing launch_context.json: {launch_context_path}")
    else:
        try:
            from ui.services.launch_context import read_launch_context

            context = read_launch_context(launch_context_path)
            if context.job_type != "inference":
                warnings.append(
                    f"launch_context job_type is '{context.job_type}', expected 'inference'"
                )
            evidence["launch_context_service"] = context.service
        except Exception as e:
            errors.append(f"failed to parse launch_context.json: {e}")

    has_optimized_artifact = any(
        path.exists()
        for path in (
            output_dir / "quantized",
            output_dir / "model.gguf",
            output_dir / "model.onnx",
        )
    )
    if not has_optimized_artifact:
        warnings.append(
            "no optimized inference artifacts detected (quantized/, model.gguf, model.onnx)"
        )
    evidence["has_optimized_artifact"] = has_optimized_artifact

    checks = _build_path_checks(
        evidence=evidence,
        required_keys=("launch_context",),
        optional_keys=("quantized_dir", "model_gguf", "model_onnx"),
    )
    status = readiness_status_from_lists(errors, warnings)
    return OpsModuleReadiness(
        module="inference",
        status=status,
        checks=checks,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
        last_output_dir=str(output_dir),
    )


def _benchmark_module_readiness(output_dir: Path) -> OpsModuleReadiness:
    errors: List[str] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {"output_dir": str(output_dir)}

    if not output_dir.exists():
        errors.append(f"benchmark output directory not found: {output_dir}")
        return OpsModuleReadiness(
            module="benchmark",
            status="fail",
            errors=errors,
            warnings=warnings,
            evidence=evidence,
            last_output_dir=str(output_dir),
        )

    benchmark_files = sorted(output_dir.glob("**/benchmark.json"))
    non_code_domains = {"vlm", "audio", "reasoning", "agentic"}
    discovered_domains: set[str] = set()
    parse_errors = 0

    for benchmark_file in benchmark_files:
        try:
            payload = json.loads(benchmark_file.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                parse_errors += 1
                continue
            search_text = json.dumps(payload).lower() + f" {benchmark_file.as_posix().lower()}"
            for domain in non_code_domains:
                if domain in search_text:
                    discovered_domains.add(domain)
        except Exception:
            parse_errors += 1

    evidence["benchmark_files"] = [str(path) for path in benchmark_files[:10]]
    evidence["benchmark_file_count"] = len(benchmark_files)
    evidence["domains_detected"] = sorted(discovered_domains)
    evidence["parse_errors"] = parse_errors

    if not benchmark_files:
        errors.append(f"no benchmark.json files found under: {output_dir}")
    if parse_errors:
        warnings.append(f"{parse_errors} benchmark result file(s) failed JSON parse")
    if not discovered_domains:
        warnings.append("no non-code benchmark domains detected in benchmark result payloads")

    checks = {
        "benchmark_results": ReadinessCheck(
            name="benchmark_results",
            status="pass" if benchmark_files else "fail",
            required=True,
            message="benchmark result files discovered" if benchmark_files else "missing benchmark results",
        ),
    }
    status = readiness_status_from_lists(errors, warnings)
    return OpsModuleReadiness(
        module="benchmark",
        status=status,
        checks=checks,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
        last_output_dir=str(output_dir),
    )


def _ui_ops_module_readiness(repo_root: Path) -> OpsModuleReadiness:
    errors: List[str] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {"repo_root": str(repo_root)}

    required_files = {
        "app": repo_root / "ui/app.py",
        "sidebar": repo_root / "ui/components/sidebar.py",
        "inference_service": repo_root / "ui/services/inference_service.py",
        "benchmark_service": repo_root / "ui/services/benchmark_service.py",
        "ops_readiness_service": repo_root / "ui/services/ops_readiness_service.py",
    }

    for label, path in required_files.items():
        if not path.exists():
            errors.append(f"missing required ui ops file: {path}")
        evidence[f"file_{label}"] = str(path)

    app_source = ""
    sidebar_source = ""
    if (repo_root / "ui/app.py").exists():
        app_source = (repo_root / "ui/app.py").read_text(encoding="utf-8")
    if (repo_root / "ui/components/sidebar.py").exists():
        sidebar_source = (repo_root / "ui/components/sidebar.py").read_text(encoding="utf-8")

    expected_routes = ("/inference", "/benchmark-advanced", "/research-hub")
    missing_routes = [route for route in expected_routes if route not in app_source]
    if missing_routes:
        errors.append("missing route wiring: " + ", ".join(missing_routes))

    expected_flags = (
        "HALO_UI_ENABLE_INFERENCE_PAGE",
        "HALO_UI_ENABLE_BENCHMARK_ADVANCED_PAGE",
        "HALO_UI_ENABLE_RESEARCH_HUB_PAGE",
    )
    missing_flags = [
        flag
        for flag in expected_flags
        if flag not in app_source and flag not in sidebar_source
    ]
    if missing_flags:
        errors.append("missing feature flag wiring: " + ", ".join(missing_flags))

    checks = {
        "routes_registered": ReadinessCheck(
            name="routes_registered",
            status="pass" if not missing_routes else "fail",
            required=True,
            message="feature routes present in ui/app.py"
            if not missing_routes
            else "missing route registration",
        ),
        "feature_flags": ReadinessCheck(
            name="feature_flags",
            status="pass" if not missing_flags else "fail",
            required=True,
            message="feature flags detected"
            if not missing_flags
            else "missing feature flag references",
        ),
    }
    status = readiness_status_from_lists(errors, warnings)
    return OpsModuleReadiness(
        module="ui_ops",
        status=status,
        checks=checks,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
        last_output_dir=str(repo_root),
    )


def _build_path_checks(
    *,
    evidence: Mapping[str, Any],
    required_keys: Iterable[str],
    optional_keys: Iterable[str],
) -> Dict[str, ReadinessCheck]:
    checks: Dict[str, ReadinessCheck] = {}

    def _add(key: str, *, required: bool) -> None:
        value = str(evidence.get(key) or "")
        exists = Path(value).exists() if value else False
        checks[key] = ReadinessCheck(
            name=key,
            status="pass" if exists else ("fail" if required else "warn"),
            required=required,
            message=f"{key} present" if exists else f"{key} missing",
        )

    for key in required_keys:
        _add(key, required=True)
    for key in optional_keys:
        _add(key, required=False)

    return checks
