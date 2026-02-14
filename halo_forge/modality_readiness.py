"""
Canonical readiness report schema/helpers for non-code modality UI gating.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from halo_forge.modality_research import ArtifactValidationResult, NON_CODE_MODALITIES
from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED, normalize_seed


READINESS_CONTRACT_VERSION = 1
DEFAULT_READINESS_REPORT_FILE = Path(
    "results/readiness/non_code_modalities_readiness.v1.json"
)
READINESS_STALE_AFTER_SECONDS = 24 * 60 * 60
READINESS_STATUSES = ("pass", "warn", "fail")
READINESS_SOURCES = ("script", "ui_live_compute")


@dataclass(frozen=True)
class ReadinessCheck:
    """Single named readiness contract check."""

    name: str
    status: str
    required: bool
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "required": self.required,
            "message": self.message,
        }


@dataclass
class ModalityReadiness:
    """Readiness payload for one modality."""

    modality: str
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
    def from_dict(modality: str, payload: Mapping[str, Any]) -> "ModalityReadiness":
        checks_raw = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
        checks: Dict[str, ReadinessCheck] = {}
        for check_name, check_payload in checks_raw.items():
            if not isinstance(check_payload, Mapping):
                continue
            status = str(check_payload.get("status", "fail"))
            required = bool(check_payload.get("required", True))
            message = str(check_payload.get("message", ""))
            checks[str(check_name)] = ReadinessCheck(
                name=str(check_name),
                status=status,
                required=required,
                message=message,
            )
        return ModalityReadiness(
            modality=modality,
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
class ReadinessReport:
    """Canonical non-code modality readiness report."""

    contract_version: int
    generated_at: str
    seed: int
    source: str
    modalities: Dict[str, ModalityReadiness]
    stale: bool = False
    age_seconds: Optional[int] = None

    def to_dict(self, include_runtime_fields: bool = False) -> Dict[str, Any]:
        payload = {
            "contract_version": int(self.contract_version),
            "generated_at": self.generated_at,
            "seed": int(self.seed),
            "source": self.source,
            "modalities": {
                modality: readiness.to_dict()
                for modality, readiness in sorted(self.modalities.items())
            },
        }
        if include_runtime_fields:
            payload["stale"] = bool(self.stale)
            payload["age_seconds"] = self.age_seconds
        return payload

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "ReadinessReport":
        modalities_raw = payload.get("modalities") if isinstance(payload.get("modalities"), Mapping) else {}
        modalities: Dict[str, ModalityReadiness] = {}
        for modality in NON_CODE_MODALITIES:
            entry_raw = modalities_raw.get(modality, {})
            if not isinstance(entry_raw, Mapping):
                entry_raw = {}
            modalities[modality] = ModalityReadiness.from_dict(modality, entry_raw)
        return ReadinessReport(
            contract_version=int(payload.get("contract_version", 0) or 0),
            generated_at=str(payload.get("generated_at") or ""),
            seed=int(payload.get("seed", DEFAULT_TRAINING_SEED) or DEFAULT_TRAINING_SEED),
            source=str(payload.get("source") or ""),
            modalities=modalities,
        )


def readiness_status_from_lists(errors: List[str], warnings: List[str]) -> str:
    """Compute readiness status from collected errors/warnings."""
    if errors:
        return "fail"
    if warnings:
        return "warn"
    return "pass"


def readiness_from_validation_result(
    result: ArtifactValidationResult,
) -> ModalityReadiness:
    """Convert artifact validation result into canonical readiness shape."""
    modality = result.modality
    evidence = dict(result.evidence)
    checks = _build_checks_from_evidence(modality=modality, evidence=evidence)
    status = readiness_status_from_lists(result.errors, result.warnings)
    return ModalityReadiness(
        modality=modality,
        status=status,
        checks=checks,
        errors=list(result.errors),
        warnings=list(result.warnings),
        evidence=evidence,
        last_output_dir=result.output_dir,
    )


def build_readiness_report(
    *,
    readiness_entries: Mapping[str, ModalityReadiness],
    seed: int = DEFAULT_TRAINING_SEED,
    source: str = "script",
    generated_at: Optional[str] = None,
) -> ReadinessReport:
    """Build canonical readiness report for all non-code modalities."""
    normalized_seed = normalize_seed(seed)
    source_key = str(source or "").strip()
    if source_key not in READINESS_SOURCES:
        raise ValueError(
            f"Invalid readiness source '{source_key}'. Expected one of: {READINESS_SOURCES}"
        )
    report_modalities: Dict[str, ModalityReadiness] = {}
    for modality in NON_CODE_MODALITIES:
        entry = readiness_entries.get(modality)
        if isinstance(entry, ModalityReadiness):
            report_modalities[modality] = entry
            continue
        report_modalities[modality] = ModalityReadiness(
            modality=modality,
            status="fail",
            checks={},
            errors=[f"no readiness entry available for modality: {modality}"],
            warnings=[],
            evidence={},
            last_output_dir="",
        )
    return ReadinessReport(
        contract_version=READINESS_CONTRACT_VERSION,
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
        seed=normalized_seed,
        source=source_key,
        modalities=report_modalities,
    )


def validate_readiness_payload(payload: Mapping[str, Any]) -> List[str]:
    """Validate readiness payload shape for compatibility."""
    errors: List[str] = []
    required_top_level = (
        "contract_version",
        "generated_at",
        "seed",
        "source",
        "modalities",
    )
    for key in required_top_level:
        if key not in payload:
            errors.append(f"missing top-level key: {key}")

    contract_version = payload.get("contract_version")
    try:
        parsed_version = int(contract_version)
        if parsed_version != READINESS_CONTRACT_VERSION:
            errors.append(
                f"unsupported contract_version: {parsed_version} "
                f"(expected {READINESS_CONTRACT_VERSION})"
            )
    except Exception:
        errors.append("contract_version must be an integer")

    source = str(payload.get("source") or "")
    if source and source not in READINESS_SOURCES:
        errors.append(f"invalid source value: {source}")

    generated_at = str(payload.get("generated_at") or "")
    if generated_at:
        try:
            datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        except Exception:
            errors.append("generated_at must be ISO-8601 timestamp")

    modalities = payload.get("modalities")
    if not isinstance(modalities, Mapping):
        errors.append("modalities must be an object")
        return errors

    for modality in NON_CODE_MODALITIES:
        if modality not in modalities:
            errors.append(f"missing modality entry: {modality}")
            continue
        entry = modalities.get(modality)
        if not isinstance(entry, Mapping):
            errors.append(f"modality entry must be an object: {modality}")
            continue
        status = str(entry.get("status") or "")
        if status not in READINESS_STATUSES:
            errors.append(f"invalid status for {modality}: {status}")
        checks = entry.get("checks")
        if checks is not None and not isinstance(checks, Mapping):
            errors.append(f"checks must be an object for {modality}")
        else:
            checks_obj = checks or {}
            for check_name, check_payload in checks_obj.items():
                if not isinstance(check_payload, Mapping):
                    errors.append(f"check must be an object for {modality}.{check_name}")
                    continue
                check_status = str(check_payload.get("status") or "")
                if check_status not in READINESS_STATUSES:
                    errors.append(
                        f"invalid check status for {modality}.{check_name}: {check_status}"
                    )
                if "required" not in check_payload:
                    errors.append(f"missing required flag for {modality}.{check_name}")

        if not isinstance(entry.get("errors", []), list):
            errors.append(f"errors must be a list for {modality}")
        if not isinstance(entry.get("warnings", []), list):
            errors.append(f"warnings must be a list for {modality}")
        evidence = entry.get("evidence")
        if evidence is not None and not isinstance(evidence, Mapping):
            errors.append(f"evidence must be an object for {modality}")
    return errors


def write_readiness_report(path: Path, report: ReadinessReport) -> None:
    """Write readiness report atomically."""
    payload = report.to_dict(include_runtime_fields=False)
    errors = validate_readiness_payload(payload)
    if errors:
        raise ValueError("Cannot write invalid readiness payload: " + "; ".join(errors))

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def load_readiness_report(path: Path) -> ReadinessReport:
    """Load and validate readiness report from disk."""
    payload_raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_raw, Mapping):
        raise ValueError("Readiness report must be a JSON object")
    errors = validate_readiness_payload(payload_raw)
    if errors:
        raise ValueError("Invalid readiness payload: " + "; ".join(errors))
    return ReadinessReport.from_dict(payload_raw)


def report_age_seconds(report: ReadinessReport) -> int:
    """Return report age in seconds."""
    generated = datetime.fromisoformat(report.generated_at.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=timezone.utc)
    return max(0, int((now - generated).total_seconds()))


def apply_staleness_policy(
    report: ReadinessReport,
    stale_after_seconds: int = READINESS_STALE_AFTER_SECONDS,
) -> ReadinessReport:
    """Return copy of report with stale metadata and warning-level adjustments."""
    age = report_age_seconds(report)
    stale = age > stale_after_seconds
    cloned = ReadinessReport.from_dict(report.to_dict())
    cloned.age_seconds = age
    cloned.stale = stale
    if not stale:
        return cloned

    for modality in NON_CODE_MODALITIES:
        entry = cloned.modalities[modality]
        stale_warning = (
            f"readiness report is stale ({age}s old; threshold {stale_after_seconds}s)"
        )
        if stale_warning not in entry.warnings:
            entry.warnings.append(stale_warning)
        if entry.status == "pass":
            entry.status = "warn"
    return cloned


def build_readiness_report_from_validations(
    results: Iterable[ArtifactValidationResult],
    *,
    seed: int = DEFAULT_TRAINING_SEED,
    source: str = "script",
) -> ReadinessReport:
    """Build readiness report from artifact validation results."""
    entries: Dict[str, ModalityReadiness] = {}
    for result in results:
        entries[result.modality] = readiness_from_validation_result(result)
    return build_readiness_report(
        readiness_entries=entries,
        seed=seed,
        source=source,
    )


def _build_checks_from_evidence(
    *,
    modality: str,
    evidence: Mapping[str, Any],
) -> Dict[str, ReadinessCheck]:
    checks: Dict[str, ReadinessCheck] = {}

    def add_path_check(name: str, key: str, required: bool) -> None:
        path_text = str(evidence.get(key) or "")
        exists = Path(path_text).exists() if path_text else False
        if exists:
            status = "pass"
            message = f"{key} present"
        else:
            status = "fail" if required else "warn"
            message = f"{key} missing"
        checks[name] = ReadinessCheck(
            name=name,
            status=status,
            required=required,
            message=message,
        )

    add_path_check("training_summary", "training_summary", True)
    add_path_check("launch_context", "launch_context", True)
    if modality in NON_CODE_MODALITIES:
        add_path_check("latest_checkpoint", "latest_checkpoint", True)
    add_path_check("final_model", "final_model", False)
    return checks
