"""All-module qualification orchestration contracts and helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from halo_forge.all_module_readiness import (
    ALL_MODULES,
    AllModuleReadiness,
    default_output_map as default_readiness_output_map,
    validate_all_module,
)
from halo_forge.diagnostics import (
    ISSUE_SCOPES,
    ISSUE_SEVERITIES,
    derive_issue_metadata,
    validate_issue_metadata_payload,
)
from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED, normalize_seed
from halo_forge.training_effectiveness_baseline import (
    build_actual_entry_from_effectiveness,
    validate_baseline_payload,
)

ALL_MODULE_QUALIFICATION_CONTRACT_VERSION = 1
ALL_MODULE_QUALIFICATION_GENERATOR_VERSION = "phase7m.v1"
ALL_MODULE_QUALIFICATION_STATUSES = ("pass", "warn", "fail")
ALL_MODULE_QUALIFICATION_PROFILES = ("contract-v1", "fixture-v1", "live-local")
ALL_MODULE_QUALIFICATION_SOURCES = ("script", "cli_test", "ui_live_compute")
ALL_MODULE_QUALIFICATION_READINESS_TIERS = (
    "",
    "experimental",
    "qualified",
    "production_ready",
)
DEFAULT_ALL_MODULE_QUALIFICATION_REPORT_FILE = Path(
    "results/readiness/all_module_qualification.v1.json"
)
DEFAULT_ALL_MODULE_QUALIFICATION_BASELINE_FILE = Path(
    "tests/baselines/all_module_qualification_baseline.v1.json"
)
DEFAULT_TRAINING_PRODUCTION_READINESS_BASELINE_DIR = Path(
    "tests/baselines/production_readiness"
)
CYCLE_BASED_MODULES = {"raft", "vlm", "audio", "reasoning", "agentic"}
UTILITY_MODULES = {"config", "data", "info", "plot"}
BENCHMARK_MODULES = {"benchmark_code", "benchmark_non_code"}
TRAINING_MODULES = {"sft", "raft", "vlm", "audio", "reasoning", "agentic"}
TRAINING_PRODUCTION_READINESS_BASELINES = {
    module: DEFAULT_TRAINING_PRODUCTION_READINESS_BASELINE_DIR / f"{module}.v1.json"
    for module in sorted(TRAINING_MODULES)
}
TRAINING_RESUME_REQUIRED = {module: module in CYCLE_BASED_MODULES for module in TRAINING_MODULES}


@dataclass
class AllModuleQualificationResult:
    """Qualification lifecycle result for one module."""

    module: str
    status: str
    launch_ok: bool
    monitor_ok: bool
    results_ingestion_ok: bool
    relaunch_ok: bool
    stop_ok: bool
    resume_latest_ok: bool
    artifacts_ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    rerun_commands: List[str] = field(default_factory=list)
    launch_blocked: bool = False
    issue_code: str = "UNKNOWN"
    issue_scope: str = "module"
    severity: str = "info"
    what_is_missing: List[str] = field(default_factory=list)
    fix_now: str = "No action needed."
    fix_options: List[str] = field(default_factory=list)
    eval_available: bool = False
    eval_metric_name: str = ""
    baseline_value: Optional[float] = None
    final_value: Optional[float] = None
    delta: Optional[float] = None
    weights_updated: bool = False
    optimizer_steps: int = 0
    samples_kept: int = 0
    production_ready: bool = False
    readiness_tier: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "launch_ok": bool(self.launch_ok),
            "monitor_ok": bool(self.monitor_ok),
            "results_ingestion_ok": bool(self.results_ingestion_ok),
            "relaunch_ok": bool(self.relaunch_ok),
            "stop_ok": bool(self.stop_ok),
            "resume_latest_ok": bool(self.resume_latest_ok),
            "artifacts_ok": bool(self.artifacts_ok),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "evidence": dict(self.evidence),
            "rerun_commands": list(self.rerun_commands),
            "launch_blocked": bool(self.launch_blocked),
            "issue_code": self.issue_code,
            "issue_scope": self.issue_scope,
            "severity": self.severity,
            "what_is_missing": list(self.what_is_missing),
            "fix_now": self.fix_now,
            "fix_options": list(self.fix_options),
            "eval_available": bool(self.eval_available),
            "eval_metric_name": self.eval_metric_name,
            "baseline_value": self.baseline_value,
            "final_value": self.final_value,
            "delta": self.delta,
            "weights_updated": bool(self.weights_updated),
            "optimizer_steps": int(self.optimizer_steps),
            "samples_kept": int(self.samples_kept),
            "production_ready": bool(self.production_ready),
            "readiness_tier": self.readiness_tier,
        }

    @staticmethod
    def from_dict(module: str, payload: Mapping[str, Any]) -> "AllModuleQualificationResult":
        result = AllModuleQualificationResult(
            module=module,
            status=str(payload.get("status", "fail")),
            launch_ok=bool(payload.get("launch_ok", False)),
            monitor_ok=bool(payload.get("monitor_ok", False)),
            results_ingestion_ok=bool(payload.get("results_ingestion_ok", False)),
            relaunch_ok=bool(payload.get("relaunch_ok", False)),
            stop_ok=bool(payload.get("stop_ok", False)),
            resume_latest_ok=bool(payload.get("resume_latest_ok", False)),
            artifacts_ok=bool(payload.get("artifacts_ok", False)),
            errors=[str(v) for v in payload.get("errors", []) if v is not None],
            warnings=[str(v) for v in payload.get("warnings", []) if v is not None],
            evidence=(
                dict(payload.get("evidence", {}))
                if isinstance(payload.get("evidence"), Mapping)
                else {}
            ),
            rerun_commands=[str(v) for v in payload.get("rerun_commands", []) if v is not None],
            launch_blocked=bool(payload.get("launch_blocked", False)),
            issue_code=str(payload.get("issue_code") or "UNKNOWN"),
            issue_scope=str(payload.get("issue_scope") or "module"),
            severity=str(payload.get("severity") or "info"),
            what_is_missing=[
                str(v) for v in payload.get("what_is_missing", []) if v is not None
            ],
            fix_now=str(payload.get("fix_now") or "No action needed."),
            fix_options=[str(v) for v in payload.get("fix_options", []) if v is not None],
            eval_available=bool(payload.get("eval_available", False)),
            eval_metric_name=str(payload.get("eval_metric_name") or ""),
            baseline_value=_coerce_float(payload.get("baseline_value")),
            final_value=_coerce_float(payload.get("final_value")),
            delta=_coerce_float(payload.get("delta")),
            weights_updated=bool(payload.get("weights_updated", False)),
            optimizer_steps=max(0, int(payload.get("optimizer_steps", 0) or 0)),
            samples_kept=max(0, int(payload.get("samples_kept", 0) or 0)),
            production_ready=bool(payload.get("production_ready", False)),
            readiness_tier=str(payload.get("readiness_tier") or ""),
        )
        _apply_issue_metadata(result)
        return result


@dataclass(frozen=True)
class TrainingProductionReadinessContract:
    module: str
    baseline_path: Path
    entry_id: str
    metric_name: str
    higher_is_better: bool
    tolerance: float
    minimum_samples_kept: int
    minimum_optimizer_steps: int
    expected_verdict: str
    resume_required: bool


@dataclass
class _TrainingProductionReadinessOutcome:
    eval_available: bool = False
    eval_metric_name: str = ""
    baseline_value: Optional[float] = None
    final_value: Optional[float] = None
    delta: Optional[float] = None
    weights_updated: bool = False
    optimizer_steps: int = 0
    samples_kept: int = 0
    production_ready: bool = False
    readiness_tier: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AllModuleQualificationReport:
    """Canonical all-module qualification report."""

    contract_version: int
    generator_version: str
    generated_at: str
    seed: int
    profile: str
    source: str
    modules: Dict[str, AllModuleQualificationResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": int(self.contract_version),
            "generator_version": self.generator_version,
            "generated_at": self.generated_at,
            "seed": int(self.seed),
            "profile": self.profile,
            "source": self.source,
            "modules": {
                module: entry.to_dict()
                for module, entry in sorted(self.modules.items())
            },
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "AllModuleQualificationReport":
        modules_raw = payload.get("modules") if isinstance(payload.get("modules"), Mapping) else {}
        modules: Dict[str, AllModuleQualificationResult] = {}
        for module in ALL_MODULES:
            module_payload = modules_raw.get(module, {})
            if not isinstance(module_payload, Mapping):
                module_payload = {}
            modules[module] = AllModuleQualificationResult.from_dict(module, module_payload)
        return AllModuleQualificationReport(
            contract_version=int(payload.get("contract_version", 0) or 0),
            generator_version=str(payload.get("generator_version") or ""),
            generated_at=str(payload.get("generated_at") or ""),
            seed=int(payload.get("seed", DEFAULT_TRAINING_SEED) or DEFAULT_TRAINING_SEED),
            profile=str(payload.get("profile") or "contract-v1"),
            source=str(payload.get("source") or ""),
            modules=modules,
        )


def _status(errors: List[str], warnings: List[str]) -> str:
    if errors:
        return "fail"
    if warnings:
        return "warn"
    return "pass"


def validate_all_module_qualification_payload(payload: Mapping[str, Any]) -> List[str]:
    """Validate qualification report schema."""
    errors: List[str] = []
    required_top_level = (
        "contract_version",
        "generator_version",
        "generated_at",
        "seed",
        "profile",
        "source",
        "modules",
    )
    for key in required_top_level:
        if key not in payload:
            errors.append(f"missing top-level key: {key}")

    try:
        version = int(payload.get("contract_version", 0))
        if version != ALL_MODULE_QUALIFICATION_CONTRACT_VERSION:
            errors.append(
                "unsupported contract_version: "
                f"{version} (expected {ALL_MODULE_QUALIFICATION_CONTRACT_VERSION})"
            )
    except Exception:
        errors.append("contract_version must be an integer")

    generator_version = str(payload.get("generator_version") or "")
    if generator_version and generator_version != ALL_MODULE_QUALIFICATION_GENERATOR_VERSION:
        errors.append(
            "unsupported generator_version: "
            f"{generator_version} (expected {ALL_MODULE_QUALIFICATION_GENERATOR_VERSION})"
        )

    profile = str(payload.get("profile") or "")
    if profile and profile not in ALL_MODULE_QUALIFICATION_PROFILES:
        errors.append(f"invalid profile value: {profile}")

    source = str(payload.get("source") or "")
    if source and source not in ALL_MODULE_QUALIFICATION_SOURCES:
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

    for module in ALL_MODULES:
        entry = modules.get(module)
        if not isinstance(entry, Mapping):
            errors.append(f"module entry must be an object: {module}")
            continue

        status = str(entry.get("status") or "")
        if status not in ALL_MODULE_QUALIFICATION_STATUSES:
            errors.append(f"invalid status for {module}: {status}")

        for field_name in (
            "launch_ok",
            "monitor_ok",
            "results_ingestion_ok",
            "relaunch_ok",
            "stop_ok",
            "resume_latest_ok",
            "artifacts_ok",
            "launch_blocked",
        ):
            if not isinstance(entry.get(field_name), bool):
                errors.append(f"{field_name} must be boolean for {module}")

        if not isinstance(entry.get("errors", []), list):
            errors.append(f"errors must be list for {module}")
        if not isinstance(entry.get("warnings", []), list):
            errors.append(f"warnings must be list for {module}")
        if not isinstance(entry.get("rerun_commands", []), list):
            errors.append(f"rerun_commands must be list for {module}")
        evidence = entry.get("evidence")
        if evidence is not None and not isinstance(evidence, Mapping):
            errors.append(f"evidence must be object for {module}")
        if not isinstance(entry.get("eval_available", False), bool):
            errors.append(f"eval_available must be boolean for {module}")
        eval_metric_name = entry.get("eval_metric_name", "")
        if eval_metric_name is not None and not isinstance(eval_metric_name, str):
            errors.append(f"eval_metric_name must be string for {module}")
        for field_name in ("baseline_value", "final_value", "delta"):
            value = entry.get(field_name)
            if value is not None and not isinstance(value, (int, float)):
                errors.append(f"{field_name} must be numeric or null for {module}")
        if not isinstance(entry.get("weights_updated", False), bool):
            errors.append(f"weights_updated must be boolean for {module}")
        for field_name in ("optimizer_steps", "samples_kept"):
            value = entry.get(field_name, 0)
            if not isinstance(value, int):
                errors.append(f"{field_name} must be integer for {module}")
        if not isinstance(entry.get("production_ready", False), bool):
            errors.append(f"production_ready must be boolean for {module}")
        readiness_tier = str(entry.get("readiness_tier") or "")
        if readiness_tier not in ALL_MODULE_QUALIFICATION_READINESS_TIERS:
            errors.append(f"invalid readiness_tier for {module}: {readiness_tier}")
        errors.extend(validate_issue_metadata_payload(entry, module=module))

    return errors


def build_all_module_qualification_report(
    *,
    module_entries: Mapping[str, AllModuleQualificationResult],
    seed: int = DEFAULT_TRAINING_SEED,
    profile: str = "contract-v1",
    source: str = "script",
    generated_at: Optional[str] = None,
) -> AllModuleQualificationReport:
    """Build qualification report while guaranteeing all known module keys."""
    normalized_seed = normalize_seed(seed)
    profile_key = str(profile or "").strip()
    if profile_key not in ALL_MODULE_QUALIFICATION_PROFILES:
        raise ValueError(
            f"Invalid profile '{profile_key}'. Expected one of {ALL_MODULE_QUALIFICATION_PROFILES}"
        )
    source_key = str(source or "").strip()
    if source_key not in ALL_MODULE_QUALIFICATION_SOURCES:
        raise ValueError(
            f"Invalid source '{source_key}'. Expected one of {ALL_MODULE_QUALIFICATION_SOURCES}"
        )

    modules: Dict[str, AllModuleQualificationResult] = {}
    for module in ALL_MODULES:
        entry = module_entries.get(module)
        if isinstance(entry, AllModuleQualificationResult):
            _apply_issue_metadata(entry)
            modules[module] = entry
            continue
        modules[module] = AllModuleQualificationResult(
            module=module,
            status="warn",
            launch_ok=False,
            monitor_ok=False,
            results_ingestion_ok=False,
            relaunch_ok=False,
            stop_ok=False,
            resume_latest_ok=False,
            artifacts_ok=False,
            errors=[],
            warnings=[f"module not evaluated in this qualification run: {module}"],
            evidence={"evaluated": False},
            rerun_commands=[" ".join(_module_command(module, normalized_seed))],
            readiness_tier="experimental" if module in TRAINING_MODULES else "",
        )
        _apply_issue_metadata(modules[module])

    return AllModuleQualificationReport(
        contract_version=ALL_MODULE_QUALIFICATION_CONTRACT_VERSION,
        generator_version=ALL_MODULE_QUALIFICATION_GENERATOR_VERSION,
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
        seed=normalized_seed,
        profile=profile_key,
        source=source_key,
        modules=modules,
    )


def write_all_module_qualification_report(path: Path, report: AllModuleQualificationReport) -> None:
    """Write qualification report atomically."""
    payload = report.to_dict()
    schema_errors = validate_all_module_qualification_payload(payload)
    if schema_errors:
        raise ValueError("Cannot write invalid qualification payload: " + "; ".join(schema_errors))

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def load_all_module_qualification_report(path: Path) -> AllModuleQualificationReport:
    """Load qualification report from disk and validate schema."""
    payload_raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_raw, Mapping):
        raise ValueError("Qualification report must be a JSON object")
    schema_errors = validate_all_module_qualification_payload(payload_raw)
    if schema_errors:
        raise ValueError("Invalid qualification report: " + "; ".join(schema_errors))
    return AllModuleQualificationReport.from_dict(payload_raw)


def default_output_map() -> Dict[str, str]:
    """Default output roots used for qualification contract checks."""
    return {
        module: str(path)
        for module, path in default_readiness_output_map().items()
    }


def compute_all_module_qualification(
    *,
    output_map: Optional[Mapping[str, str]] = None,
    seed: int = DEFAULT_TRAINING_SEED,
    profile: str = "contract-v1",
    source: str = "script",
    module_filters: Optional[Sequence[str]] = None,
) -> AllModuleQualificationReport:
    """Compute qualification report for all modules (or a filtered subset)."""
    profile_key = str(profile or "").strip()
    if profile_key not in ALL_MODULE_QUALIFICATION_PROFILES:
        raise ValueError(
            f"Invalid profile '{profile_key}'. Expected one of {ALL_MODULE_QUALIFICATION_PROFILES}"
        )

    require_artifacts = profile_key in {"fixture-v1", "live-local"}
    mapping = default_output_map()
    if output_map:
        for key, value in output_map.items():
            if key in ALL_MODULES and value:
                mapping[key] = str(value)

    selected = _resolve_selected_modules(module_filters)
    module_entries: Dict[str, AllModuleQualificationResult] = {}

    for module in ALL_MODULES:
        if module not in selected:
            module_entries[module] = AllModuleQualificationResult(
                module=module,
                status="warn",
                launch_ok=False,
                monitor_ok=False,
                results_ingestion_ok=False,
                relaunch_ok=False,
                stop_ok=False,
                resume_latest_ok=False,
                artifacts_ok=False,
                errors=[],
                warnings=[f"module not evaluated (filtered): {module}"],
                evidence={"evaluated": False},
                rerun_commands=[" ".join(_module_command(module, normalize_seed(seed)))],
                readiness_tier="experimental" if module in TRAINING_MODULES else "",
            )
            continue

        output_dir = Path(mapping[module])
        module_entries[module] = validate_all_module_qualification(
            module=module,
            output_dir=output_dir,
            seed=seed,
            require_artifacts=require_artifacts,
            profile=profile_key,
        )

    return build_all_module_qualification_report(
        module_entries=module_entries,
        seed=seed,
        profile=profile_key,
        source=source,
    )


def validate_all_module_qualification(
    *,
    module: str,
    output_dir: Path,
    seed: int = DEFAULT_TRAINING_SEED,
    require_artifacts: bool = False,
    profile: str = "contract-v1",
) -> AllModuleQualificationResult:
    """Validate one module lifecycle qualification contract."""
    module_key = str(module or "").strip().lower()
    if module_key not in ALL_MODULES:
        invalid_result = AllModuleQualificationResult(
            module=module_key,
            status="fail",
            launch_ok=False,
            monitor_ok=False,
            results_ingestion_ok=False,
            relaunch_ok=False,
            stop_ok=False,
            resume_latest_ok=False,
            artifacts_ok=False,
            errors=[f"unsupported module: {module_key}"],
            warnings=[],
            evidence={"output_dir": str(output_dir)},
            rerun_commands=[],
        )
        _apply_issue_metadata(invalid_result)
        return invalid_result

    readiness_entry: AllModuleReadiness = validate_all_module(
        module=module_key,
        output_dir=output_dir,
        seed=seed,
        require_artifacts=require_artifacts,
    )

    errors = list(readiness_entry.errors)
    warnings = list(readiness_entry.warnings)
    evidence = dict(readiness_entry.evidence)
    evidence["output_dir"] = str(output_dir)
    evidence["profile"] = profile

    artifacts_ok, artifact_errors, artifact_warnings = _artifact_contract(
        module=module_key,
        output_dir=output_dir,
        require_artifacts=require_artifacts,
    )
    errors.extend(artifact_errors)
    warnings.extend(artifact_warnings)

    monitor_ok = _monitor_contract_ok(module_key)
    stop_ok = _stop_contract_ok(module_key)
    relaunch_ok = _relaunch_contract_ok(module_key)
    results_ingestion_ok = _results_ingestion_contract_ok(
        module=module_key,
        output_dir=output_dir,
        require_artifacts=require_artifacts,
    )
    resume_latest_ok = _resume_latest_contract_ok(
        module=module_key,
        output_dir=output_dir,
        require_artifacts=require_artifacts,
    )

    if not monitor_ok:
        errors.append(f"monitor contract missing routing for module={module_key}")
    if not stop_ok:
        errors.append(f"stop contract missing routing for module={module_key}")
    if not relaunch_ok:
        errors.append(f"relaunch contract missing routing for module={module_key}")
    if not results_ingestion_ok:
        message = f"results ingestion contract missing/invalid for module={module_key}"
        if require_artifacts:
            errors.append(message)
        else:
            warnings.append(message)

    if module_key in CYCLE_BASED_MODULES and not resume_latest_ok:
        message = (
            f"resume_latest checkpoint evidence missing for module={module_key}: "
            f"{output_dir / 'latest_checkpoint.json'}"
        )
        if require_artifacts:
            errors.append(message)
        else:
            warnings.append(message)

    launch_ok = not readiness_entry.launch_blocked
    if not launch_ok and readiness_entry.errors:
        evidence["launch_blocked_reason"] = readiness_entry.errors[0]

    training_outcome = _training_production_readiness(
        module=module_key,
        output_dir=output_dir,
        require_artifacts=require_artifacts,
        launch_ok=launch_ok,
        monitor_ok=monitor_ok,
        relaunch_ok=relaunch_ok,
        results_ingestion_ok=results_ingestion_ok,
        resume_latest_ok=resume_latest_ok,
        artifacts_ok=artifacts_ok,
    )
    errors.extend(training_outcome.errors)
    warnings.extend(training_outcome.warnings)
    evidence.update(training_outcome.evidence)

    status = _status(errors, warnings)
    result = AllModuleQualificationResult(
        module=module_key,
        status=status,
        launch_ok=launch_ok,
        monitor_ok=monitor_ok,
        results_ingestion_ok=results_ingestion_ok,
        relaunch_ok=relaunch_ok,
        stop_ok=stop_ok,
        resume_latest_ok=resume_latest_ok,
        artifacts_ok=artifacts_ok,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
        rerun_commands=[" ".join(_module_command(module_key, normalize_seed(seed)))],
        launch_blocked=readiness_entry.launch_blocked,
        eval_available=training_outcome.eval_available,
        eval_metric_name=training_outcome.eval_metric_name,
        baseline_value=training_outcome.baseline_value,
        final_value=training_outcome.final_value,
        delta=training_outcome.delta,
        weights_updated=training_outcome.weights_updated,
        optimizer_steps=training_outcome.optimizer_steps,
        samples_kept=training_outcome.samples_kept,
        production_ready=training_outcome.production_ready,
        readiness_tier=training_outcome.readiness_tier,
    )
    _apply_issue_metadata(result)
    return result


def normalize_all_module_qualification_payload(
    report: AllModuleQualificationReport,
) -> Dict[str, Any]:
    """Normalize report payload for deterministic diff/baseline comparisons."""
    payload = report.to_dict()
    payload["generated_at"] = "<normalized>"

    for module in ALL_MODULES:
        entry = payload["modules"][module]
        entry["errors"] = [_normalize_text(v) for v in entry.get("errors", [])]
        entry["warnings"] = [_normalize_text(v) for v in entry.get("warnings", [])]
        entry["rerun_commands"] = [_normalize_text(v) for v in entry.get("rerun_commands", [])]
        entry["evidence"] = _normalize_mapping(entry.get("evidence", {}))

    return payload


def build_qualification_baseline_payload(report: AllModuleQualificationReport) -> Dict[str, Any]:
    """Build baseline payload from qualification report."""
    normalized = normalize_all_module_qualification_payload(report)
    return {
        "contract_version": ALL_MODULE_QUALIFICATION_CONTRACT_VERSION,
        "generator_version": ALL_MODULE_QUALIFICATION_GENERATOR_VERSION,
        "profile": str(normalized.get("profile") or "contract-v1"),
        "seed": int(normalized.get("seed", DEFAULT_TRAINING_SEED)),
        "modules": normalized.get("modules", {}),
    }


def validate_qualification_baseline_payload(payload: Mapping[str, Any]) -> List[str]:
    """Validate baseline payload schema."""
    errors: List[str] = []
    required_top = (
        "contract_version",
        "generator_version",
        "profile",
        "seed",
        "modules",
    )
    for key in required_top:
        if key not in payload:
            errors.append(f"missing baseline key: {key}")

    try:
        version = int(payload.get("contract_version", 0))
        if version != ALL_MODULE_QUALIFICATION_CONTRACT_VERSION:
            errors.append(
                "unsupported baseline contract_version: "
                f"{version} (expected {ALL_MODULE_QUALIFICATION_CONTRACT_VERSION})"
            )
    except Exception:
        errors.append("baseline contract_version must be integer")

    generator_version = str(payload.get("generator_version") or "")
    if generator_version != ALL_MODULE_QUALIFICATION_GENERATOR_VERSION:
        errors.append(
            "unsupported baseline generator_version: "
            f"{generator_version} (expected {ALL_MODULE_QUALIFICATION_GENERATOR_VERSION})"
        )

    profile = str(payload.get("profile") or "")
    if profile not in ALL_MODULE_QUALIFICATION_PROFILES:
        errors.append(f"invalid baseline profile: {profile}")

    try:
        int(payload.get("seed", DEFAULT_TRAINING_SEED))
    except Exception:
        errors.append("baseline seed must be integer")

    modules = payload.get("modules")
    if not isinstance(modules, Mapping):
        errors.append("baseline modules must be object")
        return errors

    for module in ALL_MODULES:
        if module not in modules:
            errors.append(f"missing baseline module entry: {module}")
            continue
        entry = modules.get(module)
        if not isinstance(entry, Mapping):
            errors.append(f"baseline module entry must be object: {module}")
            continue
        status = str(entry.get("status") or "")
        if status not in ALL_MODULE_QUALIFICATION_STATUSES:
            errors.append(f"invalid baseline status for {module}: {status}")

    return errors


def write_qualification_baseline_file(path: Path, payload: Mapping[str, Any]) -> None:
    """Write baseline payload atomically."""
    schema_errors = validate_qualification_baseline_payload(payload)
    if schema_errors:
        raise ValueError("Cannot write invalid qualification baseline: " + "; ".join(schema_errors))

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def load_qualification_baseline_file(path: Path) -> Dict[str, Any]:
    """Load baseline file from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Qualification baseline must be a JSON object")
    return dict(payload)


def compare_qualification_baselines(
    *,
    expected: Mapping[str, Any],
    current: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Compare expected/current qualification baseline payloads."""
    drifts: List[Dict[str, Any]] = []

    for key in ("contract_version", "generator_version", "profile", "seed"):
        if expected.get(key) != current.get(key):
            drifts.append(
                {
                    "severity": "hard",
                    "module": "_meta",
                    "field": key,
                    "expected": expected.get(key),
                    "actual": current.get(key),
                }
            )

    expected_modules = expected.get("modules") if isinstance(expected.get("modules"), Mapping) else {}
    current_modules = current.get("modules") if isinstance(current.get("modules"), Mapping) else {}

    hard_fields = (
        "status",
        "launch_ok",
        "monitor_ok",
        "results_ingestion_ok",
        "relaunch_ok",
        "stop_ok",
        "resume_latest_ok",
        "artifacts_ok",
        "eval_available",
        "eval_metric_name",
        "baseline_value",
        "final_value",
        "delta",
        "weights_updated",
        "optimizer_steps",
        "samples_kept",
        "production_ready",
        "readiness_tier",
    )

    for module in ALL_MODULES:
        exp_entry = expected_modules.get(module)
        cur_entry = current_modules.get(module)
        if not isinstance(exp_entry, Mapping):
            drifts.append(
                {
                    "severity": "hard",
                    "module": module,
                    "field": "module_entry",
                    "expected": "present",
                    "actual": "missing_in_expected",
                }
            )
            continue
        if not isinstance(cur_entry, Mapping):
            drifts.append(
                {
                    "severity": "hard",
                    "module": module,
                    "field": "module_entry",
                    "expected": "present",
                    "actual": "missing_in_current",
                }
            )
            continue

        for field_name in hard_fields:
            if exp_entry.get(field_name) != cur_entry.get(field_name):
                drifts.append(
                    {
                        "severity": "hard",
                        "module": module,
                        "field": field_name,
                        "expected": exp_entry.get(field_name),
                        "actual": cur_entry.get(field_name),
                    }
                )

        for field_name in ("errors", "warnings"):
            if _normalize_list(exp_entry.get(field_name, [])) != _normalize_list(
                cur_entry.get(field_name, [])
            ):
                drifts.append(
                    {
                        "severity": "warn",
                        "module": module,
                        "field": field_name,
                        "expected": exp_entry.get(field_name, []),
                        "actual": cur_entry.get(field_name, []),
                    }
                )

    return drifts


def format_qualification_drift_lines(drifts: Iterable[Mapping[str, Any]]) -> List[str]:
    """Format drift list into parseable lines."""
    lines: List[str] = []
    for drift in drifts:
        severity = str(drift.get("severity") or "warn")
        module = str(drift.get("module") or "unknown")
        field_name = str(drift.get("field") or "unknown")
        expected_value = json.dumps(drift.get("expected"), sort_keys=True)
        actual_value = json.dumps(drift.get("actual"), sort_keys=True)
        lines.append(
            "QUAL_DRIFT "
            f"severity={severity} module={module} field={field_name} "
            f"expected={expected_value} actual={actual_value}"
        )
    return lines


def format_qualification_issue_lines(
    entry: AllModuleQualificationResult,
    *,
    show_fix_commands: bool = False,
) -> List[str]:
    """Format module diagnostics into parseable qualification issue lines."""
    lines: List[str] = []
    if str(entry.status).lower() == "pass":
        return lines

    summary = ""
    if entry.errors:
        summary = entry.errors[0]
    elif entry.warnings:
        summary = entry.warnings[0]
    elif entry.what_is_missing:
        summary = entry.what_is_missing[0]
    else:
        summary = entry.fix_now

    lines.append(
        "ALL_QUAL_ISSUE "
        f"module={entry.module} "
        f"code={entry.issue_code} "
        f"scope={entry.issue_scope} "
        f"severity={entry.severity} "
        f"blocked={1 if entry.launch_blocked else 0} "
        f"summary={json.dumps(summary)} "
        f"fix_now={json.dumps(entry.fix_now)}"
    )

    if show_fix_commands:
        for option in entry.fix_options:
            lines.append(
                "ALL_QUAL_FIX "
                f"module={entry.module} "
                f"command={json.dumps(str(option))}"
            )
    return lines


@lru_cache(maxsize=16)
def _load_training_production_readiness_contract(
    module: str,
) -> Optional[TrainingProductionReadinessContract]:
    baseline_path = TRAINING_PRODUCTION_READINESS_BASELINES.get(module)
    if baseline_path is None or not baseline_path.exists():
        return None

    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    schema_errors = validate_baseline_payload(payload)
    if schema_errors:
        raise ValueError(
            f"Invalid production-readiness baseline for {module}: " + "; ".join(schema_errors)
        )

    entries = payload.get("entries")
    if not isinstance(entries, Mapping) or len(entries) != 1:
        raise ValueError(
            f"Production-readiness baseline for {module} must contain exactly one entry"
        )
    entry_id, raw_entry = next(iter(entries.items()))
    if not isinstance(raw_entry, Mapping):
        raise ValueError(f"Invalid production-readiness entry for {module}")

    metric_name = str(raw_entry.get("metric_name") or "")
    return TrainingProductionReadinessContract(
        module=module,
        baseline_path=baseline_path,
        entry_id=str(entry_id),
        metric_name=metric_name,
        higher_is_better=bool(raw_entry.get("higher_is_better", True)),
        tolerance=float(raw_entry.get("tolerance", 0.0) or 0.0),
        minimum_samples_kept=max(0, int(raw_entry.get("minimum_samples_kept", 0) or 0)),
        minimum_optimizer_steps=max(
            0, int(raw_entry.get("minimum_optimizer_steps", 0) or 0)
        ),
        expected_verdict=str(raw_entry.get("expected_verdict") or "pass"),
        resume_required=TRAINING_RESUME_REQUIRED.get(module, False),
    )


def _training_production_readiness(
    *,
    module: str,
    output_dir: Path,
    require_artifacts: bool,
    launch_ok: bool,
    monitor_ok: bool,
    relaunch_ok: bool,
    results_ingestion_ok: bool,
    resume_latest_ok: bool,
    artifacts_ok: bool,
) -> _TrainingProductionReadinessOutcome:
    if module not in TRAINING_MODULES:
        return _TrainingProductionReadinessOutcome(readiness_tier="")

    outcome = _TrainingProductionReadinessOutcome(readiness_tier="experimental")
    contract = _load_training_production_readiness_contract(module)
    if contract is None:
        message = f"production readiness contract missing for module={module}"
        if require_artifacts:
            outcome.errors.append(message)
        else:
            outcome.warnings.append(message)
        return outcome

    outcome.evidence["production_readiness_contract"] = {
        "baseline_file": str(contract.baseline_path),
        "entry_id": contract.entry_id,
        "metric_name": contract.metric_name,
        "minimum_samples_kept": contract.minimum_samples_kept,
        "minimum_optimizer_steps": contract.minimum_optimizer_steps,
        "expected_verdict": contract.expected_verdict,
        "resume_required": contract.resume_required,
    }

    summary_path = output_dir / "training_summary.json"
    if not summary_path.exists():
        message = f"production readiness evidence missing training_summary.json for module={module}"
        if require_artifacts:
            outcome.errors.append(message)
        else:
            outcome.warnings.append(message)
        outcome.readiness_tier = _derive_training_readiness_tier(
            production_ready=False,
            contract_ok=_base_training_contract_ok(
                module=module,
                launch_ok=launch_ok,
                monitor_ok=monitor_ok,
                relaunch_ok=relaunch_ok,
                results_ingestion_ok=results_ingestion_ok,
                resume_latest_ok=resume_latest_ok,
                artifacts_ok=artifacts_ok,
            ),
        )
        return outcome

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        message = f"production readiness could not parse training_summary.json for module={module}: {exc}"
        if require_artifacts:
            outcome.errors.append(message)
        else:
            outcome.warnings.append(message)
        return outcome

    if not isinstance(summary, Mapping):
        message = f"production readiness training_summary must be an object for module={module}"
        if require_artifacts:
            outcome.errors.append(message)
        else:
            outcome.warnings.append(message)
        return outcome

    effectiveness = summary.get("effectiveness")
    if not isinstance(effectiveness, Mapping):
        message = f"production readiness effectiveness block missing for module={module}"
        if require_artifacts:
            outcome.errors.append(message)
        else:
            outcome.warnings.append(message)
        outcome.weights_updated = bool(summary.get("weights_updated", False))
        outcome.optimizer_steps = max(0, int(summary.get("total_train_steps_executed", 0) or 0))
        outcome.readiness_tier = _derive_training_readiness_tier(
            production_ready=False,
            contract_ok=_base_training_contract_ok(
                module=module,
                launch_ok=launch_ok,
                monitor_ok=monitor_ok,
                relaunch_ok=relaunch_ok,
                results_ingestion_ok=results_ingestion_ok,
                resume_latest_ok=resume_latest_ok,
                artifacts_ok=artifacts_ok,
            ),
        )
        return outcome

    evaluation = (
        dict(effectiveness.get("evaluation") or {})
        if isinstance(effectiveness.get("evaluation"), Mapping)
        else {}
    )
    update_quality = (
        dict(effectiveness.get("update_quality") or {})
        if isinstance(effectiveness.get("update_quality"), Mapping)
        else {}
    )
    data_yield = (
        dict(effectiveness.get("data_yield") or {})
        if isinstance(effectiveness.get("data_yield"), Mapping)
        else {}
    )
    actual_entry = build_actual_entry_from_effectiveness(effectiveness)

    outcome.eval_available = actual_entry.get("evaluation_status") == "available"
    outcome.eval_metric_name = str(actual_entry.get("metric_name") or contract.metric_name)
    outcome.baseline_value = _coerce_float(
        evaluation.get("baseline_value", contract_metric_baseline(contract))
    )
    outcome.final_value = _coerce_float(evaluation.get("final_value"))
    outcome.delta = _coerce_float(evaluation.get("delta"))
    outcome.weights_updated = bool(update_quality.get("weights_updated", summary.get("weights_updated", False)))
    outcome.optimizer_steps = max(
        0,
        int(
            actual_entry.get("optimizer_steps", update_quality.get("optimizer_steps", 0)) or 0
        ),
    )
    outcome.samples_kept = max(
        0,
        int(
            actual_entry.get(
                "samples_kept",
                data_yield.get("samples_kept", _yield_stage_kept(summary)),
            )
            or 0
        ),
    )

    readiness_failures = _compare_training_readiness_contract(
        contract=contract,
        actual_entry=actual_entry,
        outcome=outcome,
    )
    base_contract_ok = _base_training_contract_ok(
        module=module,
        launch_ok=launch_ok,
        monitor_ok=monitor_ok,
        relaunch_ok=relaunch_ok,
        results_ingestion_ok=results_ingestion_ok,
        resume_latest_ok=resume_latest_ok,
        artifacts_ok=artifacts_ok,
    )
    outcome.production_ready = base_contract_ok and not readiness_failures
    outcome.readiness_tier = _derive_training_readiness_tier(
        production_ready=outcome.production_ready,
        contract_ok=base_contract_ok,
    )
    outcome.evidence["production_readiness_checks"] = {
        "base_contract_ok": base_contract_ok,
        "launch_ok": launch_ok,
        "monitor_ok": monitor_ok,
        "relaunch_ok": relaunch_ok,
        "results_ingestion_ok": results_ingestion_ok,
        "resume_latest_ok": resume_latest_ok,
        "artifacts_ok": artifacts_ok,
        "readiness_failures": list(readiness_failures),
    }
    if readiness_failures:
        message = (
            f"production readiness gate failed for module={module}: "
            + "; ".join(readiness_failures[:4])
        )
        if require_artifacts:
            outcome.errors.append(message)
        else:
            outcome.warnings.append(message)

    return outcome


def contract_metric_baseline(contract: TrainingProductionReadinessContract) -> Optional[float]:
    payload = json.loads(contract.baseline_path.read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload.get("entries"), Mapping) else {}
    entry = entries.get(contract.entry_id)
    if not isinstance(entry, Mapping):
        return None
    return _coerce_float(entry.get("baseline_value"))


def _compare_training_readiness_contract(
    *,
    contract: TrainingProductionReadinessContract,
    actual_entry: Mapping[str, Any],
    outcome: _TrainingProductionReadinessOutcome,
) -> List[str]:
    failures: List[str] = []
    if str(actual_entry.get("metric_name") or "") != contract.metric_name:
        failures.append(
            f"canonical metric mismatch (expected {contract.metric_name}, got {actual_entry.get('metric_name') or 'missing'})"
        )
    if str(actual_entry.get("evaluation_status") or "not_available") != "available":
        failures.append("post-train evaluation missing")
    if str(actual_entry.get("verdict") or "") != contract.expected_verdict:
        failures.append(
            f"effectiveness verdict expected {contract.expected_verdict}, got {actual_entry.get('verdict') or 'missing'}"
        )
    if outcome.samples_kept < contract.minimum_samples_kept:
        failures.append(
            f"samples_kept below minimum ({outcome.samples_kept} < {contract.minimum_samples_kept})"
        )
    if outcome.optimizer_steps < contract.minimum_optimizer_steps:
        failures.append(
            f"optimizer_steps below minimum ({outcome.optimizer_steps} < {contract.minimum_optimizer_steps})"
        )
    if not outcome.weights_updated:
        failures.append("weights were not updated")

    baseline_value = contract_metric_baseline(contract)
    final_value = _coerce_float(actual_entry.get("final_value"))
    if baseline_value is None or final_value is None:
        failures.append("baseline or final evaluation metric missing")
    elif _is_regressed(
        baseline_value=baseline_value,
        final_value=final_value,
        higher_is_better=contract.higher_is_better,
        tolerance=contract.tolerance,
    ):
        failures.append(
            f"evaluation regressed beyond tolerance ({final_value} vs baseline {baseline_value})"
        )
    return failures


def _base_training_contract_ok(
    *,
    module: str,
    launch_ok: bool,
    monitor_ok: bool,
    relaunch_ok: bool,
    results_ingestion_ok: bool,
    resume_latest_ok: bool,
    artifacts_ok: bool,
) -> bool:
    if not (launch_ok and monitor_ok and relaunch_ok and results_ingestion_ok and artifacts_ok):
        return False
    if TRAINING_RESUME_REQUIRED.get(module, False) and not resume_latest_ok:
        return False
    return True


def _derive_training_readiness_tier(*, production_ready: bool, contract_ok: bool) -> str:
    if production_ready:
        return "production_ready"
    if contract_ok:
        return "qualified"
    return "experimental"


def _yield_stage_kept(summary: Mapping[str, Any]) -> int:
    yield_diagnostics = summary.get("yield_diagnostics")
    if not isinstance(yield_diagnostics, Mapping):
        return 0
    stage_counts = yield_diagnostics.get("stage_counts")
    if not isinstance(stage_counts, Mapping):
        return 0
    return max(0, int(stage_counts.get("kept", 0) or 0))


def _is_regressed(
    *,
    baseline_value: float,
    final_value: float,
    higher_is_better: bool,
    tolerance: float,
) -> bool:
    if higher_is_better:
        return final_value + tolerance < baseline_value
    return final_value - tolerance > baseline_value


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_selected_modules(module_filters: Optional[Sequence[str]]) -> List[str]:
    selected: List[str] = []
    for module in module_filters or []:
        key = str(module or "").strip().lower()
        if not key:
            continue
        if key not in ALL_MODULES:
            raise ValueError(f"Unsupported module filter: {key}")
        if key not in selected:
            selected.append(key)
    if not selected:
        return list(ALL_MODULES)
    return selected


def _artifact_contract(
    *,
    module: str,
    output_dir: Path,
    require_artifacts: bool,
) -> tuple[bool, List[str], List[str]]:
    paths = _artifact_paths(module, output_dir)
    missing: List[str] = []
    errors: List[str] = []
    warnings: List[str] = []

    for label, path in paths.items():
        if isinstance(path, list):
            exists = any(Path(candidate).exists() for candidate in path)
            if not exists:
                missing.append(f"{label}: {', '.join(path)}")
        else:
            if not Path(path).exists():
                missing.append(f"{label}: {path}")

    artifacts_ok = len(missing) == 0
    if missing:
        message = "missing qualification evidence: " + "; ".join(missing)
        if require_artifacts:
            errors.append(message)
        else:
            warnings.append(message)

    return artifacts_ok, errors, warnings


def _artifact_paths(module: str, output_dir: Path) -> Dict[str, Any]:
    if module == "config":
        return {"config_file": str(output_dir / "base.yaml")}
    if module == "data":
        return {"dataset_file": str(output_dir / "sample.jsonl")}
    if module == "info":
        return {"hardware_snapshot": str(output_dir / "hardware_snapshot.json")}
    if module == "plot":
        return {
            "plot_artifact": [
                str(output_dir / "training_loss.png"),
                str(output_dir / "benchmark_comparison.png"),
            ]
        }
    if module == "sft":
        return {
            "training_summary": str(output_dir / "training_summary.json"),
            "launch_context": str(output_dir / "launch_context.json"),
            "final_model": str(output_dir / "final_model"),
        }
    if module == "inference":
        return {
            "launch_context": str(output_dir / "launch_context.json"),
        }
    if module in CYCLE_BASED_MODULES:
        return {
            "training_summary": str(output_dir / "training_summary.json"),
            "launch_context": str(output_dir / "launch_context.json"),
            "latest_checkpoint": str(output_dir / "latest_checkpoint.json"),
            "final_model": str(output_dir / "final_model"),
        }
    if module == "benchmark_code":
        return {"benchmark_json": [str(path) for path in output_dir.glob("**/benchmark.json")]}
    if module == "benchmark_non_code":
        return {"benchmark_json": [str(path) for path in output_dir.glob("**/benchmark.json")]}
    return {
        "ui_app": str(Path.cwd() / "ui" / "app.py"),
        "ui_sidebar": str(Path.cwd() / "ui" / "components" / "sidebar.py"),
    }


def _monitor_contract_ok(module: str) -> bool:
    source = _load_source(Path.cwd() / "ui" / "pages" / "monitor.py")
    if not source:
        return False

    if module in BENCHMARK_MODULES:
        return "self.benchmark_service.stop_job" in source
    if module == "inference":
        return "self.inference_service.stop_job" in source
    if module in UTILITY_MODULES:
        return "self.module_ops_service.stop_job" in source
    if module == "ui_ops":
        return "class Monitor" in source
    return "self.training_service.stop_job" in source


def _stop_contract_ok(module: str) -> bool:
    source = _load_source(Path.cwd() / "ui" / "pages" / "monitor.py")
    if not source:
        return False
    if module in BENCHMARK_MODULES:
        return "self.benchmark_service.stop_job" in source
    if module == "inference":
        return "self.inference_service.stop_job" in source
    if module in UTILITY_MODULES:
        return "self.module_ops_service.stop_job" in source
    if module == "ui_ops":
        return True
    return "self.training_service.stop_job" in source


def _relaunch_contract_ok(module: str) -> bool:
    source = _load_source(Path.cwd() / "ui" / "pages" / "monitor.py")
    if not source:
        return False
    if module in BENCHMARK_MODULES:
        return "self.benchmark_service.relaunch_from_context" in source
    if module == "inference":
        return "self.inference_service.relaunch_from_context" in source
    if module in UTILITY_MODULES:
        return "self.module_ops_service.relaunch_from_context" in source
    if module == "ui_ops":
        return True
    return "self.training_service.relaunch_from_context" in source


def _resume_latest_contract_ok(module: str, output_dir: Path, require_artifacts: bool) -> bool:
    if module not in CYCLE_BASED_MODULES:
        return False
    checkpoint_path = output_dir / "latest_checkpoint.json"
    if checkpoint_path.exists():
        return True
    return False if require_artifacts else False


def _results_ingestion_contract_ok(
    *,
    module: str,
    output_dir: Path,
    require_artifacts: bool,
) -> bool:
    results_source = _load_source(Path.cwd() / "ui" / "services" / "results_service.py")
    if not results_source:
        return False

    if module in BENCHMARK_MODULES:
        files = list(output_dir.glob("**/benchmark.json"))
        if files:
            for path in files[:5]:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, Mapping):
                        return True
                except Exception:
                    continue
        return False if require_artifacts else True

    if module in TRAINING_MODULES:
        summary_path = output_dir / "training_summary.json"
        if summary_path.exists():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(payload, Mapping):
                    return True
            except Exception:
                return False
        return False if require_artifacts else True

    if module == "inference":
        launch_context_path = output_dir / "launch_context.json"
        if launch_context_path.exists():
            try:
                payload = json.loads(launch_context_path.read_text(encoding="utf-8"))
                if isinstance(payload, Mapping):
                    return True
            except Exception:
                return False
        return False if require_artifacts else True

    if module in UTILITY_MODULES:
        return (
            "_parse_utility_run_summary_file" in results_source
            and "list_utility_runs" in results_source
        )

    if module == "ui_ops":
        return "get_latest_artifact_roots" in results_source

    return True


def _module_command(module: str, seed: int) -> List[str]:
    commands: Dict[str, List[str]] = {
        "config": ["halo-forge", "config", "validate", "configs/sft_example.yaml"],
        "data": ["halo-forge", "data", "prepare", "--list"],
        "info": ["halo-forge", "info"],
        "plot": ["halo-forge", "plot", "benchmarks", "results/benchmarks"],
        "sft": [
            "halo-forge",
            "sft",
            "train",
            "--dry-run",
            "--model",
            "Qwen/Qwen2.5-Coder-3B",
            "--dataset",
            "codealpaca",
        ],
        "raft": [
            "halo-forge",
            "raft",
            "train",
            "--dry-run",
            "--model",
            "Qwen/Qwen2.5-Coder-3B",
            "--prompts",
            "data/rlvr/humaneval_prompts.jsonl",
            "--seed",
            str(seed),
        ],
        "benchmark_code": [
            "halo-forge",
            "benchmark",
            "eval",
            "--benchmark",
            "humaneval",
            "--limit",
            "5",
        ],
        "benchmark_non_code": [
            "halo-forge",
            "reasoning",
            "benchmark",
            "--dataset",
            "gsm8k",
            "--limit",
            "20",
        ],
        "inference": [
            "halo-forge",
            "inference",
            "optimize",
            "--dry-run",
            "--model",
            "Qwen/Qwen2.5-Coder-3B",
        ],
        "vlm": [
            "halo-forge",
            "vlm",
            "train",
            "--dry-run",
            "--dataset",
            "textvqa",
            "--seed",
            str(seed),
        ],
        "audio": [
            "halo-forge",
            "audio",
            "train",
            "--dry-run",
            "--dataset",
            "librispeech",
            "--seed",
            str(seed),
        ],
        "reasoning": [
            "halo-forge",
            "reasoning",
            "train",
            "--dry-run",
            "--dataset",
            "gsm8k",
            "--seed",
            str(seed),
        ],
        "agentic": [
            "halo-forge",
            "agentic",
            "train",
            "--dry-run",
            "--dataset",
            "xlam",
            "--seed",
            str(seed),
        ],
        "ui_ops": ["halo-forge", "ui", "--no-browser"],
    }
    return commands[module]


def _load_source(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _normalize_text(value: Any) -> str:
    text = str(value)
    cwd = str(Path.cwd())
    text = text.replace(cwd, "<repo_root>")
    text = re.sub(r"/tmp/[^\s\"]+", "<tmp>", text)
    text = re.sub(r"[A-Za-z]:\\\\[^\s\"]+", "<win_path>", text)
    return text


def _normalize_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [_normalize_text(v) for v in values]


def _normalize_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_mapping(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, list):
        return [_normalize_mapping(item) for item in value]
    if isinstance(value, Path):
        return _normalize_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _normalize_text(value)


def _apply_issue_metadata(entry: AllModuleQualificationResult) -> None:
    metadata = derive_issue_metadata(
        module=entry.module,
        issue_class="preflight_blocker" if entry.launch_blocked else "evidence_gap",
        launch_blocked=entry.launch_blocked,
        errors=entry.errors,
        warnings=entry.warnings,
        action_hint="",
        evidence=entry.evidence,
        last_output_dir=str(entry.evidence.get("output_dir") or ""),
    )
    entry.issue_code = str(metadata["issue_code"])
    entry.issue_scope = str(metadata["issue_scope"])
    entry.severity = str(metadata["severity"])
    entry.what_is_missing = [str(v) for v in metadata["what_is_missing"]]
    entry.fix_now = str(metadata["fix_now"])
    merged_options = list(entry.fix_options) + [str(v) for v in metadata["fix_options"]]
    deduped: List[str] = []
    for option in merged_options:
        text = str(option).strip()
        if text and text not in deduped:
            deduped.append(text)
    entry.fix_options = deduped[:5]
