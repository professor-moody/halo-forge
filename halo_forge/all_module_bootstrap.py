"""All-module bootstrap contracts and evidence generation helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from halo_forge.all_module_readiness import ALL_MODULES, validate_all_module
from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED, normalize_seed

ALL_MODULE_BOOTSTRAP_CONTRACT_VERSION = 1
ALL_MODULE_BOOTSTRAP_STATUSES = ("pass", "warn", "fail")
ALL_MODULE_BOOTSTRAP_PROFILES = ("contract-v1", "live-local")
ALL_MODULE_BOOTSTRAP_SOURCES = ("script", "cli_test", "ui_live_compute")
DEFAULT_ALL_MODULE_BOOTSTRAP_REPORT_FILE = Path("results/readiness/all_module_bootstrap.v1.json")
DEFAULT_ALL_MODULE_BOOTSTRAP_OUTPUT_ROOT = Path("results/bootstrap")


@dataclass
class AllModuleBootstrapEntry:
    """Bootstrap result payload for one module."""

    module: str
    status: str
    bootstrap_attempted: bool
    artifacts_created: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    evidence_root: str = ""
    evidence_files: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "bootstrap_attempted": bool(self.bootstrap_attempted),
            "artifacts_created": list(self.artifacts_created),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "evidence_root": self.evidence_root,
            "evidence_files": list(self.evidence_files),
            "next_actions": list(self.next_actions),
        }

    @staticmethod
    def from_dict(module: str, payload: Mapping[str, Any]) -> "AllModuleBootstrapEntry":
        return AllModuleBootstrapEntry(
            module=module,
            status=str(payload.get("status", "fail")),
            bootstrap_attempted=bool(payload.get("bootstrap_attempted", False)),
            artifacts_created=[str(v) for v in payload.get("artifacts_created", []) if v is not None],
            errors=[str(v) for v in payload.get("errors", []) if v is not None],
            warnings=[str(v) for v in payload.get("warnings", []) if v is not None],
            evidence_root=str(payload.get("evidence_root") or ""),
            evidence_files=[str(v) for v in payload.get("evidence_files", []) if v is not None],
            next_actions=[str(v) for v in payload.get("next_actions", []) if v is not None],
        )


@dataclass
class AllModuleBootstrapReport:
    """Canonical all-module bootstrap report."""

    contract_version: int
    generated_at: str
    profile: str
    seed: int
    source: str
    modules: Dict[str, AllModuleBootstrapEntry]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": int(self.contract_version),
            "generated_at": self.generated_at,
            "profile": self.profile,
            "seed": int(self.seed),
            "source": self.source,
            "modules": {
                module: entry.to_dict()
                for module, entry in sorted(self.modules.items())
            },
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "AllModuleBootstrapReport":
        modules_raw = payload.get("modules") if isinstance(payload.get("modules"), Mapping) else {}
        modules: Dict[str, AllModuleBootstrapEntry] = {}
        for module in ALL_MODULES:
            module_payload = modules_raw.get(module, {})
            if not isinstance(module_payload, Mapping):
                module_payload = {}
            modules[module] = AllModuleBootstrapEntry.from_dict(module, module_payload)
        return AllModuleBootstrapReport(
            contract_version=int(payload.get("contract_version", 0) or 0),
            generated_at=str(payload.get("generated_at") or ""),
            profile=str(payload.get("profile") or "contract-v1"),
            seed=int(payload.get("seed", DEFAULT_TRAINING_SEED) or DEFAULT_TRAINING_SEED),
            source=str(payload.get("source") or "script"),
            modules=modules,
        )


def validate_all_module_bootstrap_payload(payload: Mapping[str, Any]) -> List[str]:
    """Validate all-module bootstrap report payload schema."""
    errors: List[str] = []
    for key in ("contract_version", "generated_at", "profile", "seed", "source", "modules"):
        if key not in payload:
            errors.append(f"missing top-level key: {key}")

    try:
        version = int(payload.get("contract_version", 0))
        if version != ALL_MODULE_BOOTSTRAP_CONTRACT_VERSION:
            errors.append(
                "unsupported contract_version: "
                f"{version} (expected {ALL_MODULE_BOOTSTRAP_CONTRACT_VERSION})"
            )
    except Exception:
        errors.append("contract_version must be an integer")

    profile = str(payload.get("profile") or "")
    if profile and profile not in ALL_MODULE_BOOTSTRAP_PROFILES:
        errors.append(f"invalid profile value: {profile}")

    source = str(payload.get("source") or "")
    if source and source not in ALL_MODULE_BOOTSTRAP_SOURCES:
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
        if status not in ALL_MODULE_BOOTSTRAP_STATUSES:
            errors.append(f"invalid status for {module}: {status}")

        if not isinstance(entry.get("bootstrap_attempted"), bool):
            errors.append(f"bootstrap_attempted must be boolean for {module}")
        if not isinstance(entry.get("artifacts_created", []), list):
            errors.append(f"artifacts_created must be list for {module}")
        if not isinstance(entry.get("errors", []), list):
            errors.append(f"errors must be list for {module}")
        if not isinstance(entry.get("warnings", []), list):
            errors.append(f"warnings must be list for {module}")
        if not isinstance(entry.get("evidence_files", []), list):
            errors.append(f"evidence_files must be list for {module}")
        if not isinstance(entry.get("next_actions", []), list):
            errors.append(f"next_actions must be list for {module}")
        if "evidence_root" in entry and not isinstance(entry.get("evidence_root"), str):
            errors.append(f"evidence_root must be string for {module}")

    return errors


def write_all_module_bootstrap_report(path: Path, report: AllModuleBootstrapReport) -> None:
    """Write bootstrap report atomically after schema validation."""
    payload = report.to_dict()
    errors = validate_all_module_bootstrap_payload(payload)
    if errors:
        raise ValueError("Cannot write invalid all-module bootstrap payload: " + "; ".join(errors))

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def load_all_module_bootstrap_report(path: Path) -> AllModuleBootstrapReport:
    """Load and validate all-module bootstrap report."""
    payload_raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_raw, Mapping):
        raise ValueError("All-module bootstrap report must be a JSON object")
    errors = validate_all_module_bootstrap_payload(payload_raw)
    if errors:
        raise ValueError("Invalid all-module bootstrap payload: " + "; ".join(errors))
    return AllModuleBootstrapReport.from_dict(payload_raw)


def normalize_all_module_bootstrap_payload(report: AllModuleBootstrapReport) -> Dict[str, Any]:
    """Normalize bootstrap payload for deterministic diffs."""
    payload = report.to_dict()
    payload["generated_at"] = "<normalized>"

    modules = payload.get("modules", {})
    if isinstance(modules, Mapping):
        for entry in modules.values():
            if not isinstance(entry, Mapping):
                continue
            if isinstance(entry.get("evidence_root"), str):
                entry["evidence_root"] = "<normalized_path>"
            evidence_files = entry.get("evidence_files")
            if isinstance(evidence_files, list):
                entry["evidence_files"] = ["<normalized_path>" for _ in evidence_files]
            artifacts_created = entry.get("artifacts_created")
            if isinstance(artifacts_created, list):
                entry["artifacts_created"] = ["<normalized_path>" for _ in artifacts_created]
    return payload


def default_bootstrap_output_map(*, output_root: Path | str = DEFAULT_ALL_MODULE_BOOTSTRAP_OUTPUT_ROOT) -> Dict[str, str]:
    """Default per-module evidence roots for bootstrap artifact generation."""
    root = Path(output_root)
    mapping: Dict[str, str] = {}
    for module in ALL_MODULES:
        if module == "ui_ops":
            mapping[module] = str(Path.cwd())
        else:
            mapping[module] = str(root / module)
    return mapping


def compute_all_module_bootstrap(
    *,
    bootstrap_profile: str = "contract-v1",
    seed: int = DEFAULT_TRAINING_SEED,
    source: str = "script",
    output_root: Path | str = DEFAULT_ALL_MODULE_BOOTSTRAP_OUTPUT_ROOT,
    output_map: Optional[Mapping[str, str]] = None,
    module_filters: Optional[Sequence[str]] = None,
    strict: bool = False,
) -> AllModuleBootstrapReport:
    """Generate bounded bootstrap evidence for all modules and return canonical report."""
    profile = str(bootstrap_profile or "contract-v1").strip()
    if profile not in ALL_MODULE_BOOTSTRAP_PROFILES:
        raise ValueError(
            f"Invalid bootstrap profile '{profile}'. Expected one of: {ALL_MODULE_BOOTSTRAP_PROFILES}"
        )

    source_key = str(source or "script").strip()
    if source_key not in ALL_MODULE_BOOTSTRAP_SOURCES:
        raise ValueError(
            f"Invalid bootstrap source '{source_key}'. Expected one of: {ALL_MODULE_BOOTSTRAP_SOURCES}"
        )

    selected_modules = _selected_modules(module_filters)
    normalized_seed = normalize_seed(seed)

    mapping = default_bootstrap_output_map(output_root=output_root)
    if output_map:
        for key, value in output_map.items():
            module = str(key or "").strip().lower()
            if module in ALL_MODULES and value:
                mapping[module] = str(value)

    entries: Dict[str, AllModuleBootstrapEntry] = {}
    for module in ALL_MODULES:
        if module not in selected_modules:
            entries[module] = AllModuleBootstrapEntry(
                module=module,
                status="warn",
                bootstrap_attempted=False,
                artifacts_created=[],
                errors=[],
                warnings=[f"module not selected for bootstrap run: {module}"],
                evidence_root=str(mapping[module]),
                evidence_files=[],
                next_actions=[
                    f"Run bootstrap for this module: halo-forge test --level all-module-bootstrap --module {module}"
                ],
            )
            continue

        module_root = Path(mapping[module])
        created_files: List[Path] = []
        errors: List[str] = []
        warnings: List[str] = []
        try:
            module_root.mkdir(parents=True, exist_ok=True)
            generated_files, generated_warnings = _bootstrap_module_evidence(
                module=module,
                output_dir=module_root,
                profile=profile,
                seed=normalized_seed,
                strict=strict,
            )
            created_files.extend(generated_files)
            warnings.extend(generated_warnings)
        except Exception as exc:
            errors.append(f"bootstrap generation failed: {exc}")

        readiness = validate_all_module(
            module=module,
            output_dir=module_root,
            seed=normalized_seed,
            require_artifacts=True,
        )
        errors.extend([str(v) for v in readiness.errors])
        warnings.extend([str(v) for v in readiness.warnings])

        status = "pass"
        if errors:
            status = "fail"
        elif warnings:
            status = "warn"

        next_actions = _next_actions_for_module(
            module=module,
            status=status,
            errors=errors,
            warnings=warnings,
            profile=profile,
            evidence_root=module_root,
        )

        entries[module] = AllModuleBootstrapEntry(
            module=module,
            status=status,
            bootstrap_attempted=True,
            artifacts_created=[str(path) for path in sorted(set(created_files))],
            errors=errors,
            warnings=warnings,
            evidence_root=str(module_root),
            evidence_files=[
                str(path)
                for path in sorted(module_root.glob("**/*"))
                if path.is_file()
            ][:50],
            next_actions=next_actions,
        )

    report = AllModuleBootstrapReport(
        contract_version=ALL_MODULE_BOOTSTRAP_CONTRACT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        profile=profile,
        seed=normalized_seed,
        source=source_key,
        modules=entries,
    )
    return report


def _selected_modules(module_filters: Optional[Sequence[str]]) -> List[str]:
    if not module_filters:
        return list(ALL_MODULES)

    selected: List[str] = []
    for module in module_filters:
        key = str(module or "").strip().lower()
        if not key:
            continue
        if key not in ALL_MODULES:
            raise ValueError(f"Unsupported bootstrap module filter: {key}")
        if key not in selected:
            selected.append(key)

    if not selected:
        return list(ALL_MODULES)
    return selected


def _bootstrap_module_evidence(
    *,
    module: str,
    output_dir: Path,
    profile: str,
    seed: int,
    strict: bool,
) -> tuple[List[Path], List[str]]:
    created: List[Path] = []
    warnings: List[str] = []

    if module == "ui_ops":
        return created, warnings

    if profile == "live-local":
        probe_files, probe_warnings = _attempt_live_probe(
            module=module,
            output_dir=output_dir,
            strict=strict,
        )
        created.extend(probe_files)
        warnings.extend(probe_warnings)

    if module in {"sft", "raft", "vlm", "audio", "reasoning", "agentic"}:
        created.extend(_write_training_evidence(module=module, output_dir=output_dir, seed=seed))
        return created, warnings

    if module in {"benchmark_code", "benchmark_non_code"}:
        created.extend(_write_benchmark_evidence(module=module, output_dir=output_dir))
        return created, warnings

    if module == "inference":
        created.extend(_write_inference_evidence(output_dir=output_dir))
        return created, warnings

    if module == "config":
        config_file = output_dir / "bootstrap_config.yaml"
        config_file.write_text("model: Qwen/Qwen2.5-Coder-0.5B\nepochs: 1\n", encoding="utf-8")
        created.append(config_file)
        return created, warnings

    if module == "data":
        dataset = output_dir / "bootstrap_dataset.jsonl"
        dataset.write_text('{"prompt":"hello","response":"world"}\n', encoding="utf-8")
        created.append(dataset)
        return created, warnings

    if module == "info":
        snapshot = output_dir / "hardware_snapshot.json"
        snapshot.write_text(
            json.dumps({"gpu": "simulated", "utilization_percent": 0, "timestamp": "bootstrap"}, indent=2) + "\n",
            encoding="utf-8",
        )
        created.append(snapshot)
        return created, warnings

    if module == "plot":
        chart = output_dir / "bootstrap_chart.png"
        chart.write_bytes(b"PNG_PLACEHOLDER")
        created.append(chart)
        return created, warnings

    return created, warnings


def _write_training_evidence(*, module: str, output_dir: Path, seed: int) -> List[Path]:
    created: List[Path] = []

    run_id = f"{module}-bootstrap"
    summary = output_dir / "training_summary.json"
    summary_payload: Dict[str, Any] = {
        "modality": module,
        "model_name": "bootstrap/model",
        "run_id": run_id,
        "seed": int(seed),
        "weights_updated": True,
        "total_train_steps_executed": 1,
        "final_update_reason": "bootstrap_contract",
        "cycles_executed": 1,
        "final_train_loss": 0.0,
        "base_model_name": "bootstrap/model",
        "active_model_name": "bootstrap/model",
        "resume_from_cycle": 0,
        "resumed_from_checkpoint": None,
        "failure_reason": None,
        "final_model_path": str(output_dir / "final_model"),
    }
    if module in {"sft", "raft"}:
        summary_payload["modality"] = module
    summary.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    created.append(summary)

    if module in {"raft", "vlm", "audio", "reasoning", "agentic"}:
        latest_checkpoint = output_dir / "latest_checkpoint.json"
        latest_checkpoint.write_text(
            json.dumps(
                {
                    "cycle": 0,
                    "model_dir": str(output_dir / "cycle_0" / "model"),
                    "checkpoint_state": str(output_dir / "cycle_0" / "checkpoint_state.json"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        created.append(latest_checkpoint)

    final_model_dir = output_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    (final_model_dir / "README.txt").write_text("bootstrap model placeholder\n", encoding="utf-8")
    created.append(final_model_dir / "README.txt")

    launch_context = output_dir / "launch_context.json"
    launch_context.write_text(
        json.dumps(
            {
                "contract_version": 1,
                "job_type": module,
                "service": "training",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_ui_page": "/training",
                "command": [sys.executable, "-m", "halo_forge.cli", module, "train"],
                "args": {
                    "model": "bootstrap/model",
                    "output_dir": str(output_dir),
                    "seed": int(seed),
                },
                "relaunch_capabilities": {
                    "can_relaunch": True,
                    "can_clone": True,
                    "can_resume_latest": module in {"raft", "vlm", "audio", "reasoning", "agentic"},
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    created.append(launch_context)

    return created


def _write_benchmark_evidence(*, module: str, output_dir: Path) -> List[Path]:
    created: List[Path] = []
    if module == "benchmark_code":
        target = output_dir / "humaneval-bootstrap"
        payload = {
            "model": "bootstrap/model",
            "dataset": "humaneval",
            "domain": "code",
            "pass_at_1": 0.5,
            "samples": 1,
        }
    else:
        target = output_dir / "non-code-bootstrap"
        payload = {
            "model": "bootstrap/model",
            "dataset": "textvqa",
            "domain": "vlm",
            "accuracy": 0.5,
            "samples": 1,
        }

    target.mkdir(parents=True, exist_ok=True)
    benchmark = target / "benchmark.json"
    benchmark.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    created.append(benchmark)

    launch_context = output_dir / "launch_context.json"
    launch_context.write_text(
        json.dumps(
            {
                "contract_version": 1,
                "job_type": "benchmark",
                "service": "benchmark",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_ui_page": "/benchmark",
                "command": [sys.executable, "-m", "halo_forge.cli", "benchmark", "eval"],
                "args": {
                    "benchmark_type": "code" if module == "benchmark_code" else "vlm",
                    "benchmark_name": payload["dataset"],
                    "output_path": str(benchmark),
                },
                "relaunch_capabilities": {
                    "can_relaunch": True,
                    "can_clone": True,
                    "can_resume_latest": False,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    created.append(launch_context)
    return created


def _write_inference_evidence(*, output_dir: Path) -> List[Path]:
    created: List[Path] = []

    quantized_dir = output_dir / "quantized"
    quantized_dir.mkdir(parents=True, exist_ok=True)
    quantized_marker = quantized_dir / "README.txt"
    quantized_marker.write_text("bootstrap inference artifact\n", encoding="utf-8")
    created.append(quantized_marker)

    launch_context = output_dir / "launch_context.json"
    launch_context.write_text(
        json.dumps(
            {
                "contract_version": 1,
                "job_type": "inference",
                "service": "inference",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_ui_page": "/inference",
                "command": [sys.executable, "-m", "halo_forge.cli", "inference", "optimize"],
                "args": {
                    "mode": "optimize",
                    "model": "bootstrap/model",
                    "output_dir": str(output_dir),
                    "dry_run": True,
                },
                "relaunch_capabilities": {
                    "can_relaunch": True,
                    "can_clone": True,
                    "can_resume_latest": False,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    created.append(launch_context)

    return created


def _attempt_live_probe(*, module: str, output_dir: Path, strict: bool) -> tuple[List[Path], List[str]]:
    """Run bounded help-level command probes for live-local profile."""
    created: List[Path] = []
    warnings: List[str] = []
    probe_log = output_dir / "live_probe.log"

    probe_cmd: List[str]
    if module == "config":
        probe_cmd = [sys.executable, "-m", "halo_forge.cli", "config", "validate", "--help"]
    elif module == "data":
        probe_cmd = [sys.executable, "-m", "halo_forge.cli", "data", "prepare", "--list"]
    elif module == "info":
        probe_cmd = [sys.executable, "-m", "halo_forge.cli", "info"]
    elif module == "plot":
        probe_cmd = [sys.executable, "-m", "halo_forge.cli", "plot", "benchmarks", "--help"]
    elif module == "inference":
        probe_cmd = [sys.executable, "-m", "halo_forge.cli", "inference", "optimize", "--help"]
    elif module.startswith("benchmark"):
        probe_cmd = [sys.executable, "-m", "halo_forge.cli", "benchmark", "eval", "--help"]
    else:
        probe_log.write_text(
            "live-local probe skipped for heavy module; using contract bootstrap artifacts\n",
            encoding="utf-8",
        )
        return [probe_log], [
            f"live-local probe skipped for heavy module '{module}'; using contract bootstrap artifacts"
        ]

    result = subprocess.run(
        probe_cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    probe_log.write_text(
        "command: " + " ".join(probe_cmd) + "\n"
        + f"returncode: {result.returncode}\n"
        + (result.stdout or "")
        + ("\n" + result.stderr if result.stderr else ""),
        encoding="utf-8",
    )
    created.append(probe_log)

    if result.returncode != 0:
        message = f"live probe returned non-zero for module={module}: returncode {result.returncode}"
        if strict:
            raise RuntimeError(message)
        warnings.append(message)

    return created, warnings


def _next_actions_for_module(
    *,
    module: str,
    status: str,
    errors: Sequence[str],
    warnings: Sequence[str],
    profile: str,
    evidence_root: Path,
) -> List[str]:
    if status == "pass":
        return [f"Evidence is ready for {module}. Run readiness/qualification checks to confirm."]

    actions = [
        f"Review evidence root: {evidence_root}",
        f"Re-run bootstrap: halo-forge test --level all-module-bootstrap --bootstrap-profile {profile} --module {module}",
    ]
    if errors:
        actions.append("Resolve hard contract errors before running strict qualification.")
    elif warnings:
        actions.append("Optional dependencies/evidence gaps detected; continue in non-strict mode if needed.")
    return actions
