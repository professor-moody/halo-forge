"""
Canonical readiness report schema/helpers for all CLI modules (coding + non-coding).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from halo_forge.diagnostics import (
    ISSUE_SCOPES,
    ISSUE_SEVERITIES,
    derive_issue_metadata,
    validate_issue_metadata_payload,
)
from halo_forge.modality_readiness import ReadinessCheck
from halo_forge.ops_module_readiness import (
    OPS_READINESS_STALE_AFTER_SECONDS,
    OpsModuleReadiness,
    validate_ops_module,
)
from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED, normalize_seed


ALL_MODULES: tuple[str, ...] = (
    "config",
    "data",
    "info",
    "plot",
    "sft",
    "raft",
    "benchmark_code",
    "benchmark_non_code",
    "inference",
    "vlm",
    "audio",
    "reasoning",
    "agentic",
    "ui_ops",
)
ALL_MODULE_READINESS_CONTRACT_VERSION = 1
ALL_MODULE_READINESS_STATUSES = ("pass", "warn", "fail")
ALL_MODULE_READINESS_SOURCES = ("script", "ui_live_compute", "cli_test")
ALL_MODULE_ISSUE_CLASSES = ("none", "evidence_gap", "preflight_blocker", "contract_break")
ALL_MODULE_ISSUE_SCOPES = ISSUE_SCOPES
ALL_MODULE_ISSUE_SEVERITIES = ISSUE_SEVERITIES
DEFAULT_ALL_MODULE_READINESS_REPORT_FILE = Path(
    "results/readiness/all_modules_readiness.v1.json"
)


def readiness_status_from_lists(errors: List[str], warnings: List[str]) -> str:
    if errors:
        return "fail"
    if warnings:
        return "warn"
    return "pass"


@dataclass
class AllModuleReadiness:
    """Readiness payload for one CLI module surface."""

    module: str
    status: str
    checks: Dict[str, ReadinessCheck] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    last_output_dir: str = ""
    launch_blocked: bool = False
    issue_class: str = "none"
    action_hint: str = ""
    issue_code: str = "UNKNOWN"
    issue_scope: str = "module"
    severity: str = "info"
    what_is_missing: List[str] = field(default_factory=list)
    fix_now: str = "No action needed."
    fix_options: List[str] = field(default_factory=list)

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
            "launch_blocked": bool(self.launch_blocked),
            "issue_class": self.issue_class,
            "action_hint": self.action_hint,
            "issue_code": self.issue_code,
            "issue_scope": self.issue_scope,
            "severity": self.severity,
            "what_is_missing": list(self.what_is_missing),
            "fix_now": self.fix_now,
            "fix_options": list(self.fix_options),
        }

    @staticmethod
    def from_dict(module: str, payload: Mapping[str, Any]) -> "AllModuleReadiness":
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
        readiness = AllModuleReadiness(
            module=module,
            status=str(payload.get("status", "fail")),
            checks=checks,
            errors=[str(v) for v in payload.get("errors", []) if v is not None],
            warnings=[str(v) for v in payload.get("warnings", []) if v is not None],
            evidence=dict(payload.get("evidence", {}))
            if isinstance(payload.get("evidence"), Mapping)
            else {},
            last_output_dir=str(payload.get("last_output_dir") or ""),
            launch_blocked=bool(payload.get("launch_blocked", False)),
            issue_class=str(payload.get("issue_class") or "none"),
            action_hint=str(payload.get("action_hint") or ""),
            issue_code=str(payload.get("issue_code") or "UNKNOWN"),
            issue_scope=str(payload.get("issue_scope") or "module"),
            severity=str(payload.get("severity") or "info"),
            what_is_missing=[
                str(v) for v in payload.get("what_is_missing", []) if v is not None
            ],
            fix_now=str(payload.get("fix_now") or "No action needed."),
            fix_options=[
                str(v) for v in payload.get("fix_options", []) if v is not None
            ],
        )
        _apply_issue_metadata(readiness)
        return readiness


@dataclass
class AllModuleReadinessReport:
    """Canonical all-module readiness report."""

    contract_version: int
    generated_at: str
    seed: int
    source: str
    modules: Dict[str, AllModuleReadiness]
    stale: bool = False
    age_seconds: Optional[int] = None

    def to_dict(self, include_runtime_fields: bool = False) -> Dict[str, Any]:
        payload = {
            "contract_version": int(self.contract_version),
            "generated_at": self.generated_at,
            "seed": int(self.seed),
            "source": self.source,
            "modules": {
                module: entry.to_dict() for module, entry in sorted(self.modules.items())
            },
        }
        if include_runtime_fields:
            payload["stale"] = bool(self.stale)
            payload["age_seconds"] = self.age_seconds
        return payload

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "AllModuleReadinessReport":
        modules_raw = payload.get("modules") if isinstance(payload.get("modules"), Mapping) else {}
        modules: Dict[str, AllModuleReadiness] = {}
        for module in ALL_MODULES:
            module_payload = modules_raw.get(module, {})
            if not isinstance(module_payload, Mapping):
                module_payload = {}
            modules[module] = AllModuleReadiness.from_dict(module, module_payload)
        return AllModuleReadinessReport(
            contract_version=int(payload.get("contract_version", 0) or 0),
            generated_at=str(payload.get("generated_at") or ""),
            seed=int(payload.get("seed", DEFAULT_TRAINING_SEED) or DEFAULT_TRAINING_SEED),
            source=str(payload.get("source") or ""),
            modules=modules,
        )


def build_all_module_readiness_report(
    *,
    module_entries: Mapping[str, AllModuleReadiness],
    seed: int = DEFAULT_TRAINING_SEED,
    source: str = "script",
    generated_at: Optional[str] = None,
) -> AllModuleReadinessReport:
    """Build report with all required module entries populated."""
    normalized_seed = normalize_seed(seed)
    source_key = str(source or "").strip()
    if source_key not in ALL_MODULE_READINESS_SOURCES:
        raise ValueError(
            f"Invalid readiness source '{source_key}'. Expected {ALL_MODULE_READINESS_SOURCES}"
        )

    report_modules: Dict[str, AllModuleReadiness] = {}
    for module in ALL_MODULES:
        entry = module_entries.get(module)
        if isinstance(entry, AllModuleReadiness):
            _apply_issue_metadata(entry)
            report_modules[module] = entry
            continue
        report_modules[module] = AllModuleReadiness(
            module=module,
            status="fail",
            errors=[f"no readiness entry available for module: {module}"],
            last_output_dir="",
            launch_blocked=True,
            issue_class="contract_break",
            action_hint="Generate or compute readiness for this module before reviewing status.",
        )
        _apply_issue_metadata(report_modules[module])

    return AllModuleReadinessReport(
        contract_version=ALL_MODULE_READINESS_CONTRACT_VERSION,
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
        seed=normalized_seed,
        source=source_key,
        modules=report_modules,
    )


def validate_all_module_readiness_payload(payload: Mapping[str, Any]) -> List[str]:
    """Validate report schema."""
    errors: List[str] = []
    required_top_level = ("contract_version", "generated_at", "seed", "source", "modules")
    for key in required_top_level:
        if key not in payload:
            errors.append(f"missing top-level key: {key}")

    try:
        version = int(payload.get("contract_version", 0))
        if version != ALL_MODULE_READINESS_CONTRACT_VERSION:
            errors.append(
                "unsupported contract_version: "
                f"{version} (expected {ALL_MODULE_READINESS_CONTRACT_VERSION})"
            )
    except Exception:
        errors.append("contract_version must be an integer")

    source = str(payload.get("source") or "")
    if source and source not in ALL_MODULE_READINESS_SOURCES:
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
        if module not in modules:
            errors.append(f"missing module entry: {module}")
            continue
        entry = modules.get(module)
        if not isinstance(entry, Mapping):
            errors.append(f"module entry must be an object: {module}")
            continue
        status = str(entry.get("status") or "")
        if status not in ALL_MODULE_READINESS_STATUSES:
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
                if check_status not in ALL_MODULE_READINESS_STATUSES:
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
        launch_blocked = entry.get("launch_blocked")
        if launch_blocked is not None and not isinstance(launch_blocked, bool):
            errors.append(f"launch_blocked must be a boolean for {module}")
        issue_class = entry.get("issue_class")
        if issue_class is not None and str(issue_class) not in ALL_MODULE_ISSUE_CLASSES:
            errors.append(f"invalid issue_class for {module}: {issue_class}")
        action_hint = entry.get("action_hint")
        if action_hint is not None and not isinstance(action_hint, str):
            errors.append(f"action_hint must be a string for {module}")
        errors.extend(validate_issue_metadata_payload(entry, module=module))

    return errors


def write_all_module_readiness_report(path: Path, report: AllModuleReadinessReport) -> None:
    payload = report.to_dict(include_runtime_fields=False)
    errors = validate_all_module_readiness_payload(payload)
    if errors:
        raise ValueError("Cannot write invalid all-module readiness payload: " + "; ".join(errors))

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def load_all_module_readiness_report(path: Path) -> AllModuleReadinessReport:
    payload_raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_raw, Mapping):
        raise ValueError("All-module readiness report must be a JSON object")
    errors = validate_all_module_readiness_payload(payload_raw)
    if errors:
        raise ValueError("Invalid all-module readiness payload: " + "; ".join(errors))
    return AllModuleReadinessReport.from_dict(payload_raw)


def normalized_all_module_readiness_payload(report: AllModuleReadinessReport) -> Dict[str, Any]:
    payload = report.to_dict(include_runtime_fields=False)
    payload.pop("stale", None)
    payload.pop("age_seconds", None)
    return payload


def report_age_seconds(report: AllModuleReadinessReport) -> int:
    generated = datetime.fromisoformat(report.generated_at.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    return max(0, int((now - generated).total_seconds()))


def apply_staleness_policy(
    report: AllModuleReadinessReport,
    stale_after_seconds: int = OPS_READINESS_STALE_AFTER_SECONDS,
) -> AllModuleReadinessReport:
    age = report_age_seconds(report)
    stale = age > stale_after_seconds
    cloned = AllModuleReadinessReport.from_dict(report.to_dict())
    cloned.age_seconds = age
    cloned.stale = stale
    if not stale:
        return cloned

    stale_warning = f"readiness report is stale ({age}s old; threshold {stale_after_seconds}s)"
    for module in ALL_MODULES:
        entry = cloned.modules[module]
        if stale_warning not in entry.warnings:
            entry.warnings.append(stale_warning)
        if entry.status == "pass":
            entry.status = "warn"
            entry.issue_class = "evidence_gap"
            entry.action_hint = "Refresh or regenerate readiness evidence for this module."
            entry.launch_blocked = False
        _apply_issue_metadata(entry)
    return cloned


def default_output_map() -> Dict[str, str]:
    """Default output map used for live readiness checks."""
    root = Path.cwd()
    mapping = {
        "config": str(root / "configs"),
        "data": str(root / "data"),
        "info": str(root / "results" / "readiness"),
        "plot": str(root / "results"),
        "sft": str(root / "models" / "sft_run"),
        "raft": str(root / "models" / "raft_run"),
        "benchmark_code": str(root / "results" / "benchmarks"),
        "benchmark_non_code": str(root / "results" / "benchmarks"),
        "inference": str(root / "models" / "optimized"),
        "vlm": str(root / "models" / "phase7d" / "vlm_phase7d"),
        "audio": str(root / "models" / "phase7d" / "audio_phase7d"),
        "reasoning": str(root / "models" / "phase7d" / "reasoning_phase7d"),
        "agentic": str(root / "models" / "phase7d" / "agentic_phase7d"),
        "ui_ops": str(root),
    }
    return mapping


def compute_all_module_readiness(
    output_map: Mapping[str, str] | None = None,
    *,
    seed: int = DEFAULT_TRAINING_SEED,
    source: str = "ui_live_compute",
    require_artifacts: bool = False,
) -> AllModuleReadinessReport:
    """Compute readiness for all modules in scope."""
    mapping = default_output_map()
    if output_map:
        for key, value in output_map.items():
            if key in ALL_MODULES and value:
                mapping[key] = str(value)

    entries: Dict[str, AllModuleReadiness] = {}
    for module in ALL_MODULES:
        entries[module] = validate_all_module(
            module=module,
            output_dir=Path(mapping[module]),
            seed=seed,
            require_artifacts=require_artifacts,
        )

    return build_all_module_readiness_report(
        module_entries=entries,
        seed=seed,
        source=source,
    )


def validate_all_module(
    *,
    module: str,
    output_dir: Path,
    seed: int = DEFAULT_TRAINING_SEED,
    require_artifacts: bool = False,
) -> AllModuleReadiness:
    """Validate readiness contracts for one module."""
    key = str(module).strip().lower()
    if key not in ALL_MODULES:
        return _create_readiness(
            module=key,
            checks={},
            errors=[f"unsupported module: {key}"],
            warnings=[],
            evidence={},
            last_output_dir=str(output_dir),
            launch_blocked=True,
            issue_class="contract_break",
            action_hint=f"Use one of: {', '.join(ALL_MODULES)}.",
        )

    if key in {"vlm", "audio", "reasoning", "agentic", "inference", "ui_ops"}:
        return _from_ops_module(
            validate_ops_module(
                module=key,
                output_dir=output_dir,
                seed=seed,
                require_artifacts=require_artifacts,
            )
        )
    if key == "benchmark_non_code":
        return _from_ops_module(
            validate_ops_module(
                module="benchmark",
                output_dir=output_dir,
                seed=seed,
                require_artifacts=require_artifacts,
            ),
            module_name="benchmark_non_code",
        )

    if key == "config":
        return _validate_config_module(output_dir=output_dir, require_artifacts=require_artifacts)
    if key == "data":
        return _validate_data_module(output_dir=output_dir, require_artifacts=require_artifacts)
    if key == "info":
        return _validate_info_module(output_dir=output_dir, require_artifacts=require_artifacts)
    if key == "plot":
        return _validate_plot_module(output_dir=output_dir, require_artifacts=require_artifacts)
    if key == "sft":
        return _validate_sft_module(output_dir=output_dir, require_artifacts=require_artifacts)
    if key == "raft":
        return _validate_raft_module(output_dir=output_dir, require_artifacts=require_artifacts)
    if key == "benchmark_code":
        return _validate_benchmark_code_module(output_dir=output_dir, require_artifacts=require_artifacts)

    return _create_readiness(
        module=key,
        checks={},
        errors=[f"unhandled module: {key}"],
        warnings=[],
        evidence={},
        last_output_dir=str(output_dir),
        launch_blocked=True,
        issue_class="contract_break",
        action_hint="Add readiness validator wiring for this module.",
    )


def _from_ops_module(entry: OpsModuleReadiness, module_name: Optional[str] = None) -> AllModuleReadiness:
    module = module_name or entry.module
    readiness = AllModuleReadiness(
        module=module,
        status=entry.status,
        checks=entry.checks,
        errors=entry.errors,
        warnings=entry.warnings,
        evidence=entry.evidence,
        last_output_dir=entry.last_output_dir,
        launch_blocked=entry.launch_blocked,
        issue_class=entry.issue_class,
        action_hint=entry.action_hint,
    )
    _apply_issue_metadata(readiness)
    return readiness


def _load_cli_source() -> str:
    cli_path = Path.cwd() / "halo_forge" / "cli.py"
    if not cli_path.exists():
        return ""
    return cli_path.read_text(encoding="utf-8")


def _check_source_tokens(source: str, tokens: Iterable[str]) -> bool:
    # These readiness probes verify that CLI wiring is present, not which
    # quote style a formatter chose.  Accept equivalent Python string syntax
    # so a harmless formatter pass cannot downgrade production readiness.
    return all(
        token in source
        or token.replace("'", '"') in source
        or token.replace('"', "'") in source
        for token in tokens
    )


def _check_path(
    checks: Dict[str, ReadinessCheck],
    *,
    key: str,
    path: Path,
    required: bool,
) -> bool:
    exists = path.exists()
    checks[key] = ReadinessCheck(
        name=key,
        status="pass" if exists else ("fail" if required else "warn"),
        required=required,
        message=f"{key} present" if exists else f"{key} missing",
    )
    return exists


def _derive_issue_fields(
    *,
    errors: List[str],
    warnings: List[str],
    launch_blocked: bool,
) -> tuple[str, str]:
    if errors:
        if launch_blocked:
            return (
                "preflight_blocker",
                "Resolve blocking contract/preflight errors before launching this module.",
            )
        return (
            "contract_break",
            "Fix malformed contract artifacts and rerun readiness validation.",
        )
    if warnings:
        return (
            "evidence_gap",
            "Run a contract probe or launch flow to generate fresh evidence artifacts.",
        )
    return ("none", "No action needed.")


def _create_readiness(
    *,
    module: str,
    checks: Dict[str, ReadinessCheck],
    errors: List[str],
    warnings: List[str],
    evidence: Dict[str, Any],
    last_output_dir: str,
    launch_blocked: Optional[bool] = None,
    issue_class: Optional[str] = None,
    action_hint: Optional[str] = None,
) -> AllModuleReadiness:
    status = readiness_status_from_lists(errors, warnings)
    blocked = bool(errors) if launch_blocked is None else bool(launch_blocked)
    derived_issue, derived_hint = _derive_issue_fields(
        errors=errors,
        warnings=warnings,
        launch_blocked=blocked,
    )
    readiness = AllModuleReadiness(
        module=module,
        status=status,
        checks=checks,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
        last_output_dir=last_output_dir,
        launch_blocked=blocked,
        issue_class=issue_class or derived_issue,
        action_hint=action_hint or derived_hint,
    )
    _apply_issue_metadata(readiness)
    return readiness


def _apply_issue_metadata(entry: AllModuleReadiness) -> None:
    metadata = derive_issue_metadata(
        module=entry.module,
        issue_class=entry.issue_class,
        launch_blocked=entry.launch_blocked,
        errors=entry.errors,
        warnings=entry.warnings,
        action_hint=entry.action_hint,
        evidence=entry.evidence,
        last_output_dir=entry.last_output_dir,
    )
    entry.issue_code = str(metadata["issue_code"])
    entry.issue_scope = str(metadata["issue_scope"])
    entry.severity = str(metadata["severity"])
    entry.what_is_missing = [str(v) for v in metadata["what_is_missing"]]
    entry.fix_now = str(metadata["fix_now"])
    entry.fix_options = [str(v) for v in metadata["fix_options"]]


def _validate_config_module(*, output_dir: Path, require_artifacts: bool) -> AllModuleReadiness:
    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, ReadinessCheck] = {}
    evidence: Dict[str, Any] = {"output_dir": str(output_dir)}

    cli_source = _load_cli_source()
    has_contract = _check_source_tokens(cli_source, ("config_parser", "validate", "cmd_config_validate"))
    checks["cli_config_validate"] = ReadinessCheck(
        name="cli_config_validate",
        status="pass" if has_contract else "fail",
        required=True,
        message="config validate command wired" if has_contract else "config validate command missing",
    )
    if not has_contract:
        errors.append("missing config validate CLI contract")

    config_files = sorted(list(output_dir.glob("*.yaml")) + list(output_dir.glob("*.yml")))
    evidence["config_file_count"] = len(config_files)
    evidence["config_files"] = [str(path) for path in config_files[:10]]
    has_configs = len(config_files) > 0
    checks["config_examples"] = ReadinessCheck(
        name="config_examples",
        status="pass" if has_configs else ("fail" if require_artifacts else "warn"),
        required=require_artifacts,
        message="configuration files available" if has_configs else "no config files discovered",
    )
    if not has_configs:
        if require_artifacts:
            errors.append(f"no config files discovered in {output_dir}")
        else:
            warnings.append(f"no config files discovered in {output_dir}")

    return _create_readiness(
        module="config",
        checks=checks,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
        last_output_dir=str(output_dir),
    )


def _validate_data_module(*, output_dir: Path, require_artifacts: bool) -> AllModuleReadiness:
    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, ReadinessCheck] = {}
    evidence: Dict[str, Any] = {"output_dir": str(output_dir)}

    cli_source = _load_cli_source()
    has_prepare = _check_source_tokens(cli_source, ("data_subparsers.add_parser('prepare'", "cmd_data_prepare"))
    has_generate = _check_source_tokens(cli_source, ("data_subparsers.add_parser('generate'", "cmd_data_generate"))
    has_validate = _check_source_tokens(cli_source, ("data_subparsers.add_parser('validate'", "cmd_data_validate"))

    checks["cli_data_prepare"] = ReadinessCheck(
        name="cli_data_prepare",
        status="pass" if has_prepare else "fail",
        required=True,
        message="data prepare command wired" if has_prepare else "data prepare command missing",
    )
    checks["cli_data_generate"] = ReadinessCheck(
        name="cli_data_generate",
        status="pass" if has_generate else "fail",
        required=True,
        message="data generate command wired" if has_generate else "data generate command missing",
    )
    checks["cli_data_validate"] = ReadinessCheck(
        name="cli_data_validate",
        status="pass" if has_validate else "fail",
        required=True,
        message="data validate command wired" if has_validate else "data validate command missing",
    )
    if not (has_prepare and has_generate and has_validate):
        errors.append("missing one or more data CLI command contracts")

    jsonl_files = sorted(output_dir.glob("**/*.jsonl")) if output_dir.exists() else []
    evidence["jsonl_file_count"] = len(jsonl_files)
    evidence["jsonl_examples"] = [str(path) for path in jsonl_files[:10]]
    has_jsonl = len(jsonl_files) > 0
    checks["dataset_examples"] = ReadinessCheck(
        name="dataset_examples",
        status="pass" if has_jsonl else ("fail" if require_artifacts else "warn"),
        required=require_artifacts,
        message="dataset examples available" if has_jsonl else "no dataset examples discovered",
    )
    if not has_jsonl:
        if require_artifacts:
            errors.append(f"no .jsonl files discovered in {output_dir}")
        else:
            warnings.append(f"no .jsonl files discovered in {output_dir}")

    return _create_readiness(
        module="data",
        checks=checks,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
        last_output_dir=str(output_dir),
    )


def _validate_info_module(*, output_dir: Path, require_artifacts: bool) -> AllModuleReadiness:
    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, ReadinessCheck] = {}
    evidence: Dict[str, Any] = {"output_dir": str(output_dir)}

    cli_source = _load_cli_source()
    has_info = _check_source_tokens(cli_source, ("subparsers.add_parser('info'", "cmd_info"))
    checks["cli_info"] = ReadinessCheck(
        name="cli_info",
        status="pass" if has_info else "fail",
        required=True,
        message="info command wired" if has_info else "info command missing",
    )
    if not has_info:
        errors.append("missing info CLI contract")

    snapshot_path = output_dir / "hardware_snapshot.json"
    evidence["hardware_snapshot"] = str(snapshot_path)
    has_snapshot = _check_path(
        checks,
        key="hardware_snapshot",
        path=snapshot_path,
        required=require_artifacts,
    )
    if not has_snapshot:
        if require_artifacts:
            errors.append(f"missing hardware snapshot: {snapshot_path}")
        else:
            warnings.append(f"hardware snapshot not found: {snapshot_path}")

    return _create_readiness(
        module="info",
        checks=checks,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
        last_output_dir=str(output_dir),
    )


def _validate_plot_module(*, output_dir: Path, require_artifacts: bool) -> AllModuleReadiness:
    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, ReadinessCheck] = {}
    evidence: Dict[str, Any] = {"output_dir": str(output_dir)}

    cli_source = _load_cli_source()
    has_training = _check_source_tokens(
        cli_source,
        ("plot_subparsers.add_parser('training'", "cmd_plot_training"),
    )
    has_benchmarks = _check_source_tokens(
        cli_source,
        ("plot_subparsers.add_parser('benchmarks'", "cmd_plot_benchmarks"),
    )
    checks["cli_plot_training"] = ReadinessCheck(
        name="cli_plot_training",
        status="pass" if has_training else "fail",
        required=True,
        message="plot training command wired" if has_training else "plot training command missing",
    )
    checks["cli_plot_benchmarks"] = ReadinessCheck(
        name="cli_plot_benchmarks",
        status="pass" if has_benchmarks else "fail",
        required=True,
        message="plot benchmarks command wired" if has_benchmarks else "plot benchmarks command missing",
    )
    if not (has_training and has_benchmarks):
        errors.append("missing one or more plot CLI command contracts")

    chart_files = []
    if output_dir.exists():
        chart_files = sorted(output_dir.glob("**/*.png")) + sorted(output_dir.glob("**/*.svg"))
    evidence["chart_file_count"] = len(chart_files)
    evidence["chart_files"] = [str(path) for path in chart_files[:10]]
    has_chart = len(chart_files) > 0
    checks["chart_artifacts"] = ReadinessCheck(
        name="chart_artifacts",
        status="pass" if has_chart else ("fail" if require_artifacts else "warn"),
        required=require_artifacts,
        message="plot artifacts available" if has_chart else "no plot artifacts discovered",
    )
    if not has_chart:
        if require_artifacts:
            errors.append(f"no plot artifacts discovered in {output_dir}")
        else:
            warnings.append(f"no plot artifacts discovered in {output_dir}")

    return _create_readiness(
        module="plot",
        checks=checks,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
        last_output_dir=str(output_dir),
    )


def _validate_sft_module(*, output_dir: Path, require_artifacts: bool) -> AllModuleReadiness:
    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, ReadinessCheck] = {}
    evidence: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "training_summary": str(output_dir / "training_summary.json"),
        "launch_context": str(output_dir / "launch_context.json"),
    }

    cli_source = _load_cli_source()
    has_sft = _check_source_tokens(cli_source, ("sft_subparsers.add_parser('train'", "cmd_sft_train"))
    checks["cli_sft_train"] = ReadinessCheck(
        name="cli_sft_train",
        status="pass" if has_sft else "fail",
        required=True,
        message="sft train command wired" if has_sft else "sft train command missing",
    )
    if not has_sft:
        errors.append("missing sft train CLI contract")

    training_summary_exists = _check_path(
        checks,
        key="training_summary",
        path=output_dir / "training_summary.json",
        required=require_artifacts,
    )
    launch_context_exists = _check_path(
        checks,
        key="launch_context",
        path=output_dir / "launch_context.json",
        required=require_artifacts,
    )
    if require_artifacts:
        if not training_summary_exists:
            errors.append(f"missing training_summary.json: {output_dir / 'training_summary.json'}")
        if not launch_context_exists:
            errors.append(f"missing launch_context.json: {output_dir / 'launch_context.json'}")
    else:
        if not training_summary_exists:
            warnings.append(f"training_summary.json not found: {output_dir / 'training_summary.json'}")
        if not launch_context_exists:
            warnings.append(f"launch_context.json not found: {output_dir / 'launch_context.json'}")

    return _create_readiness(
        module="sft",
        checks=checks,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
        last_output_dir=str(output_dir),
    )


def _validate_raft_module(*, output_dir: Path, require_artifacts: bool) -> AllModuleReadiness:
    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, ReadinessCheck] = {}
    evidence: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "training_summary": str(output_dir / "training_summary.json"),
        "launch_context": str(output_dir / "launch_context.json"),
        "latest_checkpoint": str(output_dir / "latest_checkpoint.json"),
    }

    cli_source = _load_cli_source()
    has_raft = _check_source_tokens(cli_source, ("raft_subparsers.add_parser('train'", "cmd_raft_train"))
    checks["cli_raft_train"] = ReadinessCheck(
        name="cli_raft_train",
        status="pass" if has_raft else "fail",
        required=True,
        message="raft train command wired" if has_raft else "raft train command missing",
    )
    if not has_raft:
        errors.append("missing raft train CLI contract")

    training_summary_exists = _check_path(
        checks,
        key="training_summary",
        path=output_dir / "training_summary.json",
        required=require_artifacts,
    )
    launch_context_exists = _check_path(
        checks,
        key="launch_context",
        path=output_dir / "launch_context.json",
        required=require_artifacts,
    )
    checkpoint_exists = _check_path(
        checks,
        key="latest_checkpoint",
        path=output_dir / "latest_checkpoint.json",
        required=require_artifacts,
    )

    if require_artifacts:
        if not training_summary_exists:
            errors.append(f"missing training_summary.json: {output_dir / 'training_summary.json'}")
        if not launch_context_exists:
            errors.append(f"missing launch_context.json: {output_dir / 'launch_context.json'}")
        if not checkpoint_exists:
            errors.append(f"missing latest_checkpoint.json: {output_dir / 'latest_checkpoint.json'}")
    else:
        if not training_summary_exists:
            warnings.append(f"training_summary.json not found: {output_dir / 'training_summary.json'}")
        if not launch_context_exists:
            warnings.append(f"launch_context.json not found: {output_dir / 'launch_context.json'}")
        if not checkpoint_exists:
            warnings.append(f"latest_checkpoint.json not found: {output_dir / 'latest_checkpoint.json'}")

    return _create_readiness(
        module="raft",
        checks=checks,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
        last_output_dir=str(output_dir),
    )


def _validate_benchmark_code_module(*, output_dir: Path, require_artifacts: bool) -> AllModuleReadiness:
    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, ReadinessCheck] = {}
    evidence: Dict[str, Any] = {
        "output_dir": str(output_dir),
    }

    cli_source = _load_cli_source()
    has_run = _check_source_tokens(
        cli_source,
        ("bench_subparsers.add_parser('run'", "def cmd_benchmark(args):"),
    )
    has_full = _check_source_tokens(cli_source, ("bench_subparsers.add_parser('full'", "cmd_benchmark_full"))
    has_eval = _check_source_tokens(cli_source, ("bench_subparsers.add_parser('eval'", "cmd_benchmark_eval"))
    checks["cli_benchmark_run"] = ReadinessCheck(
        name="cli_benchmark_run",
        status="pass" if has_run else "fail",
        required=True,
        message="benchmark run command wired" if has_run else "benchmark run command missing",
    )
    checks["cli_benchmark_full"] = ReadinessCheck(
        name="cli_benchmark_full",
        status="pass" if has_full else "fail",
        required=True,
        message="benchmark full command wired" if has_full else "benchmark full command missing",
    )
    checks["cli_benchmark_eval"] = ReadinessCheck(
        name="cli_benchmark_eval",
        status="pass" if has_eval else "fail",
        required=True,
        message="benchmark eval command wired" if has_eval else "benchmark eval command missing",
    )
    if not (has_run and has_full and has_eval):
        errors.append("missing one or more benchmark CLI command contracts")

    benchmark_files = sorted(output_dir.glob("**/benchmark.json")) if output_dir.exists() else []
    code_keywords = ("humaneval", "mbpp", "livecodebench")
    code_files = []
    for path in benchmark_files:
        lower = path.as_posix().lower()
        if any(keyword in lower for keyword in code_keywords):
            code_files.append(path)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            blob = json.dumps(payload).lower()
            if any(keyword in blob for keyword in code_keywords):
                code_files.append(path)
        except Exception:
            continue

    evidence["benchmark_file_count"] = len(benchmark_files)
    evidence["code_benchmark_file_count"] = len(code_files)
    evidence["benchmark_files"] = [str(path) for path in benchmark_files[:10]]

    has_code_results = len(code_files) > 0
    checks["code_benchmark_results"] = ReadinessCheck(
        name="code_benchmark_results",
        status="pass" if has_code_results else ("fail" if require_artifacts else "warn"),
        required=require_artifacts,
        message="code benchmark results present" if has_code_results else "no code benchmark results detected",
    )
    if not has_code_results:
        if require_artifacts:
            errors.append(f"no code benchmark results detected under {output_dir}")
        else:
            warnings.append(f"no code benchmark results detected under {output_dir}")

    return _create_readiness(
        module="benchmark_code",
        checks=checks,
        errors=errors,
        warnings=warnings,
        evidence=evidence,
        last_output_dir=str(output_dir),
    )
