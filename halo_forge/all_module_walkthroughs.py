"""All-module E2E walkthrough contracts and report helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from halo_forge.all_module_readiness import ALL_MODULES, default_output_map as default_readiness_output_map
from halo_forge.all_module_readiness import validate_all_module
from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED, normalize_seed

WALKTHROUGH_CONTRACT_VERSION = 1
WALKTHROUGH_STATUSES = ("pass", "warn", "fail")
WALKTHROUGH_PROFILES = ("contract-v1", "live-local")
DEFAULT_WALKTHROUGH_REPORT_FILE = Path(
    ".internal_docs/research_testing/walkthroughs/reports/all_module_e2e_walkthrough_report.v1.json"
)


@dataclass
class WalkthroughStep:
    """One step in a module walkthrough."""

    step_id: str
    title: str
    kind: str  # cli | ui | evidence
    instruction: str
    command: List[str] = field(default_factory=list)
    ui_route: str = ""
    expected_outcome: str = ""
    evidence_paths: List[str] = field(default_factory=list)
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "kind": self.kind,
            "instruction": self.instruction,
            "command": list(self.command),
            "ui_route": self.ui_route,
            "expected_outcome": self.expected_outcome,
            "evidence_paths": list(self.evidence_paths),
            "required": bool(self.required),
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "WalkthroughStep":
        return WalkthroughStep(
            step_id=str(payload.get("step_id") or ""),
            title=str(payload.get("title") or ""),
            kind=str(payload.get("kind") or ""),
            instruction=str(payload.get("instruction") or ""),
            command=[str(v) for v in payload.get("command", []) if v is not None],
            ui_route=str(payload.get("ui_route") or ""),
            expected_outcome=str(payload.get("expected_outcome") or ""),
            evidence_paths=[str(v) for v in payload.get("evidence_paths", []) if v is not None],
            required=bool(payload.get("required", True)),
        )


@dataclass
class ModuleWalkthrough:
    """Walkthrough result for one module."""

    module: str
    status: str
    steps: List[WalkthroughStep] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    evidence_required: List[str] = field(default_factory=list)
    evidence_found: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rerun_commands: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "steps": [step.to_dict() for step in self.steps],
            "expected_outputs": list(self.expected_outputs),
            "evidence_required": list(self.evidence_required),
            "evidence_found": dict(self.evidence_found),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "rerun_commands": list(self.rerun_commands),
        }

    @staticmethod
    def from_dict(module: str, payload: Mapping[str, Any]) -> "ModuleWalkthrough":
        steps_raw = payload.get("steps") if isinstance(payload.get("steps"), list) else []
        steps = [
            WalkthroughStep.from_dict(step)
            for step in steps_raw
            if isinstance(step, Mapping)
        ]
        evidence_found = payload.get("evidence_found") if isinstance(payload.get("evidence_found"), Mapping) else {}
        return ModuleWalkthrough(
            module=module,
            status=str(payload.get("status", "fail")),
            steps=steps,
            expected_outputs=[str(v) for v in payload.get("expected_outputs", []) if v is not None],
            evidence_required=[str(v) for v in payload.get("evidence_required", []) if v is not None],
            evidence_found={str(k): bool(v) for k, v in evidence_found.items()},
            errors=[str(v) for v in payload.get("errors", []) if v is not None],
            warnings=[str(v) for v in payload.get("warnings", []) if v is not None],
            rerun_commands=[str(v) for v in payload.get("rerun_commands", []) if v is not None],
        )


@dataclass
class WalkthroughReportV1:
    """Canonical walkthrough report payload."""

    contract_version: int
    generated_at: str
    seed: int
    profile: str
    modules: Dict[str, ModuleWalkthrough]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": int(self.contract_version),
            "generated_at": self.generated_at,
            "seed": int(self.seed),
            "profile": self.profile,
            "modules": {
                module: entry.to_dict()
                for module, entry in sorted(self.modules.items())
            },
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "WalkthroughReportV1":
        modules_raw = payload.get("modules") if isinstance(payload.get("modules"), Mapping) else {}
        modules: Dict[str, ModuleWalkthrough] = {}
        for module in ALL_MODULES:
            module_payload = modules_raw.get(module, {})
            if not isinstance(module_payload, Mapping):
                module_payload = {}
            modules[module] = ModuleWalkthrough.from_dict(module, module_payload)
        return WalkthroughReportV1(
            contract_version=int(payload.get("contract_version", 0) or 0),
            generated_at=str(payload.get("generated_at") or ""),
            seed=int(payload.get("seed", DEFAULT_TRAINING_SEED) or DEFAULT_TRAINING_SEED),
            profile=str(payload.get("profile") or ""),
            modules=modules,
        )


def _module_base_command(module: str, seed: int) -> List[str]:
    commands: Dict[str, List[str]] = {
        "config": ["halo-forge", "config", "validate", "configs/raft_windows.yaml"],
        "data": ["halo-forge", "data", "validate", "data/sample.jsonl"],
        "info": ["halo-forge", "info"],
        "plot": ["halo-forge", "plot", "benchmarks", "results/benchmarks"],
        "sft": ["halo-forge", "sft", "train", "--dry-run", "--model", "Qwen/Qwen2.5-Coder-3B", "--dataset", "codealpaca"],
        "raft": ["halo-forge", "raft", "train", "--dry-run", "--model", "Qwen/Qwen2.5-Coder-3B", "--prompts", "data/rlvr/humaneval_prompts.jsonl"],
        "benchmark_code": ["halo-forge", "benchmark", "eval", "--benchmark", "humaneval", "--limit", "5"],
        "benchmark_non_code": ["halo-forge", "reasoning", "benchmark", "--dataset", "gsm8k", "--limit", "20"],
        "inference": ["halo-forge", "inference", "optimize", "--dry-run", "--model", "Qwen/Qwen2.5-Coder-3B"],
        "vlm": ["halo-forge", "vlm", "train", "--dry-run", "--dataset", "textvqa", "--seed", str(seed)],
        "audio": ["halo-forge", "audio", "train", "--dry-run", "--dataset", "librispeech", "--seed", str(seed)],
        "reasoning": ["halo-forge", "reasoning", "train", "--dry-run", "--dataset", "gsm8k", "--seed", str(seed)],
        "agentic": ["halo-forge", "agentic", "train", "--dry-run", "--dataset", "xlam", "--seed", str(seed)],
        # NiceGUI `halo-forge ui` was retired with the SPA migration.
        # `halo-forge info` is the equivalent no-bind reachability probe.
        "ui_ops": ["halo-forge", "info"],
    }
    return commands[module]


def walkthrough_step_templates(seed: int = DEFAULT_TRAINING_SEED) -> Dict[str, List[WalkthroughStep]]:
    """Return fixed walkthrough step templates for each module."""
    templates: Dict[str, List[WalkthroughStep]] = {}
    for module in ALL_MODULES:
        base_cmd = _module_base_command(module, seed)
        templates[module] = [
            WalkthroughStep(
                step_id=f"{module}.cli.1",
                title="Run CLI contract command",
                kind="cli",
                instruction="Execute the canonical CLI command for this module.",
                command=base_cmd,
                expected_outcome="Command contract is accepted and surfaces expected arguments/output.",
                evidence_paths=[],
                required=True,
            ),
            WalkthroughStep(
                step_id=f"{module}.ui.1",
                title="Validate UI navigation path",
                kind="ui",
                instruction="Navigate through the UI route for this module and validate controls.",
                ui_route=_module_ui_route(module),
                expected_outcome="Operator can open the relevant UI surface and trigger launch actions.",
                evidence_paths=[],
                required=True,
            ),
            WalkthroughStep(
                step_id=f"{module}.evidence.1",
                title="Collect required evidence artifacts",
                kind="evidence",
                instruction="Confirm required artifacts/log evidence exist after execution.",
                expected_outcome="Required evidence files are present and parseable.",
                evidence_paths=_module_evidence_paths(module),
                required=True,
            ),
        ]
    return templates


def _module_ui_route(module: str) -> str:
    route_map = {
        "config": "/config",
        "data": "/datasets",
        "info": "/",
        "plot": "/results",
        "sft": "/training",
        "raft": "/training",
        "benchmark_code": "/benchmark",
        "benchmark_non_code": "/benchmark-advanced",
        "inference": "/inference",
        "vlm": "/training",
        "audio": "/training",
        "reasoning": "/training",
        "agentic": "/training",
        # The NiceGUI `/monitor` page was retired; the SPA's runs detail
        # (per-run-id) is the live equivalent. The walkthrough strings
        # are descriptive, not navigational, so this just needs to point
        # at the surface that fulfills the contract today.
        "ui_ops": "/runs",
    }
    return route_map[module]


def _module_evidence_paths(module: str) -> List[str]:
    evidence_map = {
        "config": ["{output_dir}/base.yaml"],
        "data": ["{output_dir}/sample.jsonl"],
        "info": ["{output_dir}/hardware_snapshot.json"],
        "plot": ["{output_dir}/training_loss.png"],
        "sft": ["{output_dir}/training_summary.json", "{output_dir}/launch_context.json"],
        "raft": [
            "{output_dir}/training_summary.json",
            "{output_dir}/launch_context.json",
            "{output_dir}/latest_checkpoint.json",
        ],
        "benchmark_code": ["{output_dir}/humaneval-fixture/benchmark.json"],
        "benchmark_non_code": ["{output_dir}/reasoning-fixture/benchmark.json"],
        "inference": ["{output_dir}/launch_context.json"],
        "vlm": [
            "{output_dir}/training_summary.json",
            "{output_dir}/launch_context.json",
            "{output_dir}/latest_checkpoint.json",
        ],
        "audio": [
            "{output_dir}/training_summary.json",
            "{output_dir}/launch_context.json",
            "{output_dir}/latest_checkpoint.json",
        ],
        "reasoning": [
            "{output_dir}/training_summary.json",
            "{output_dir}/launch_context.json",
            "{output_dir}/latest_checkpoint.json",
        ],
        "agentic": [
            "{output_dir}/training_summary.json",
            "{output_dir}/launch_context.json",
            "{output_dir}/latest_checkpoint.json",
        ],
        # SPA + FastAPI replaced the NiceGUI `ui/app.py` + `ui/components/`.
        # The walkthrough evidence is now the FastAPI surface and the
        # SPA entrypoint that exercises it.
        "ui_ops": [
            "{repo_root}/halo_forge/public_api/app.py",
            "{repo_root}/public_app/src/main.tsx",
        ],
    }
    return evidence_map[module]


def default_walkthrough_output_map() -> Dict[str, str]:
    """Default artifact roots used by walkthrough contracts."""
    base = default_readiness_output_map()
    return {
        "config": str(Path.cwd() / "configs"),
        "data": str(Path.cwd() / "data"),
        "info": str(Path.cwd() / "results" / "readiness"),
        "plot": str(Path.cwd() / "results"),
        "sft": str(Path.cwd() / "models" / "sft_run"),
        "raft": str(Path.cwd() / "models" / "raft_run"),
        "benchmark_code": str(Path.cwd() / "results" / "benchmarks"),
        "benchmark_non_code": str(Path.cwd() / "results" / "benchmarks"),
        "inference": str(base["inference"]),
        "vlm": str(base["vlm"]),
        "audio": str(base["audio"]),
        "reasoning": str(base["reasoning"]),
        "agentic": str(base["agentic"]),
        "ui_ops": str(Path.cwd()),
    }


def validate_walkthrough_payload(payload: Mapping[str, Any]) -> List[str]:
    """Validate walkthrough report schema."""
    errors: List[str] = []
    required_top = (
        "contract_version",
        "generated_at",
        "seed",
        "profile",
        "modules",
    )
    for key in required_top:
        if key not in payload:
            errors.append(f"missing top-level key: {key}")

    try:
        version = int(payload.get("contract_version", 0))
        if version != WALKTHROUGH_CONTRACT_VERSION:
            errors.append(
                f"unsupported contract_version: {version} (expected {WALKTHROUGH_CONTRACT_VERSION})"
            )
    except Exception:
        errors.append("contract_version must be an integer")

    profile = str(payload.get("profile") or "")
    if profile and profile not in WALKTHROUGH_PROFILES:
        errors.append(f"invalid profile value: {profile}")

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
        status = str(entry.get("status", ""))
        if status not in WALKTHROUGH_STATUSES:
            errors.append(f"invalid status for {module}: {status}")
        if not isinstance(entry.get("steps", []), list):
            errors.append(f"steps must be list for {module}")
        if not isinstance(entry.get("expected_outputs", []), list):
            errors.append(f"expected_outputs must be list for {module}")
        if not isinstance(entry.get("evidence_required", []), list):
            errors.append(f"evidence_required must be list for {module}")
        if not isinstance(entry.get("evidence_found", {}), Mapping):
            errors.append(f"evidence_found must be object for {module}")
        if not isinstance(entry.get("errors", []), list):
            errors.append(f"errors must be list for {module}")
        if not isinstance(entry.get("warnings", []), list):
            errors.append(f"warnings must be list for {module}")
        if not isinstance(entry.get("rerun_commands", []), list):
            errors.append(f"rerun_commands must be list for {module}")

    return errors


def build_walkthrough_report(
    *,
    module_entries: Mapping[str, ModuleWalkthrough],
    seed: int = DEFAULT_TRAINING_SEED,
    profile: str = "contract-v1",
    generated_at: Optional[str] = None,
) -> WalkthroughReportV1:
    """Build report ensuring all known modules are populated."""
    normalized_seed = normalize_seed(seed)
    profile_key = str(profile or "").strip()
    if profile_key not in WALKTHROUGH_PROFILES:
        raise ValueError(f"Invalid walkthrough profile '{profile_key}'")

    modules: Dict[str, ModuleWalkthrough] = {}
    for module in ALL_MODULES:
        entry = module_entries.get(module)
        if isinstance(entry, ModuleWalkthrough):
            modules[module] = entry
            continue
        modules[module] = ModuleWalkthrough(
            module=module,
            status="fail",
            errors=[f"missing walkthrough module entry: {module}"],
        )

    return WalkthroughReportV1(
        contract_version=WALKTHROUGH_CONTRACT_VERSION,
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
        seed=normalized_seed,
        profile=profile_key,
        modules=modules,
    )


def write_walkthrough_report(path: Path, report: WalkthroughReportV1) -> None:
    payload = report.to_dict()
    schema_errors = validate_walkthrough_payload(payload)
    if schema_errors:
        raise ValueError("Cannot write invalid walkthrough report: " + "; ".join(schema_errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def load_walkthrough_report(path: Path) -> WalkthroughReportV1:
    payload_raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_raw, Mapping):
        raise ValueError("Walkthrough report must be a JSON object")
    schema_errors = validate_walkthrough_payload(payload_raw)
    if schema_errors:
        raise ValueError("Invalid walkthrough report: " + "; ".join(schema_errors))
    return WalkthroughReportV1.from_dict(payload_raw)


def normalized_walkthrough_payload(report: WalkthroughReportV1) -> Dict[str, Any]:
    payload = report.to_dict()
    payload["generated_at"] = "<normalized>"
    for module_payload in payload["modules"].values():
        module_payload["errors"] = [_normalize_text(v) for v in module_payload.get("errors", [])]
        module_payload["warnings"] = [_normalize_text(v) for v in module_payload.get("warnings", [])]
    return payload


def compute_walkthroughs(
    *,
    output_map: Optional[Mapping[str, str]] = None,
    modules: Optional[Iterable[str]] = None,
    seed: int = DEFAULT_TRAINING_SEED,
    profile: str = "contract-v1",
    execute: bool = False,
    command_timeout_sec: float = 30.0,
) -> WalkthroughReportV1:
    """Compute walkthrough report for selected modules."""
    selected = _select_modules(modules)
    selected_set = set(selected)
    mapping = default_walkthrough_output_map()
    if output_map:
        for key, value in output_map.items():
            if key in ALL_MODULES and value:
                mapping[key] = str(value)

    entries: Dict[str, ModuleWalkthrough] = {}
    for module in ALL_MODULES:
        if module not in selected_set:
            entries[module] = ModuleWalkthrough(
                module=module,
                status="warn",
                warnings=[f"module skipped by filter selection: {module}"],
                rerun_commands=[
                    f"python3 scripts/run_all_module_walkthroughs.py --module {module} --profile {profile}"
                ],
            )
            continue
        entries[module] = evaluate_module_walkthrough(
            module=module,
            output_dir=Path(mapping[module]),
            seed=seed,
            profile=profile,
            execute=execute,
            command_timeout_sec=command_timeout_sec,
        )

    return build_walkthrough_report(
        module_entries=entries,
        seed=seed,
        profile=profile,
    )


def evaluate_module_walkthrough(
    *,
    module: str,
    output_dir: Path,
    seed: int = DEFAULT_TRAINING_SEED,
    profile: str = "contract-v1",
    execute: bool = False,
    command_timeout_sec: float = 30.0,
) -> ModuleWalkthrough:
    """Evaluate one module walkthrough contract."""
    key = str(module).strip().lower()
    if key not in ALL_MODULES:
        return ModuleWalkthrough(
            module=key,
            status="fail",
            errors=[f"unsupported module: {key}"],
        )

    profile_key = str(profile or "").strip()
    if profile_key not in WALKTHROUGH_PROFILES:
        return ModuleWalkthrough(
            module=key,
            status="fail",
            errors=[f"unsupported profile: {profile_key}"],
        )

    templates = walkthrough_step_templates(seed=seed)
    steps = templates[key]

    errors: List[str] = []
    warnings: List[str] = []

    # Contract evidence validation via all-module readiness validator.
    validation = validate_all_module(
        module=key,
        output_dir=output_dir,
        seed=seed,
        require_artifacts=(profile_key == "contract-v1"),
    )
    errors.extend(validation.errors)
    warnings.extend(validation.warnings)

    # Parse-time checks for step structure.
    for step in steps:
        if step.kind == "cli":
            if not step.command:
                errors.append(f"{step.step_id}: missing command")
                continue
            if step.command[0] != "halo-forge":
                errors.append(f"{step.step_id}: command must start with halo-forge")
        elif step.kind == "ui":
            if not step.ui_route.startswith("/"):
                errors.append(f"{step.step_id}: invalid ui route '{step.ui_route}'")

    if profile_key == "live-local":
        if execute:
            probe = _live_probe_command(key, seed)
            try:
                completed = subprocess.run(
                    probe,
                    capture_output=True,
                    text=True,
                    timeout=max(1.0, float(command_timeout_sec)),
                    check=False,
                )
                if completed.returncode != 0:
                    warnings.append(
                        "live-local probe returned non-zero "
                        f"({completed.returncode}) for module={key}"
                    )
            except Exception as exc:
                warnings.append(f"live-local probe unavailable for module={key}: {exc}")
        else:
            warnings.append("live-local profile selected without --execute; command probes skipped")

    evidence_required = _resolve_evidence_paths(key, output_dir)
    evidence_found = {path: Path(path).exists() for path in evidence_required}

    expected_outputs = [step.expected_outcome for step in steps if step.expected_outcome]
    rerun_commands = [
        " ".join(_module_base_command(key, seed)),
        f"python3 scripts/run_all_module_walkthroughs.py --module {key} --profile {profile_key}",
    ]

    status = _status(errors, warnings)
    return ModuleWalkthrough(
        module=key,
        status=status,
        steps=steps,
        expected_outputs=expected_outputs,
        evidence_required=evidence_required,
        evidence_found=evidence_found,
        errors=errors,
        warnings=warnings,
        rerun_commands=rerun_commands,
    )


def checklist_mapping_for_module(module: str, seed: int = DEFAULT_TRAINING_SEED) -> List[Dict[str, Any]]:
    """Return checklist coverage mapping used by dossiers/playbooks."""
    steps = walkthrough_step_templates(seed=seed)[module]
    step_ids = [step.step_id for step in steps]
    return [
        {
            "item": "Runtime contract checklist",
            "step_ids": step_ids,
            "coverage": "covered-by-step",
        },
        {
            "item": "CLI scenarios",
            "step_ids": [step.step_id for step in steps if step.kind == "cli"],
            "coverage": "covered-by-step",
        },
        {
            "item": "UI scenarios",
            "step_ids": [step.step_id for step in steps if step.kind == "ui"],
            "coverage": "covered-by-step",
        },
        {
            "item": "Required artifacts/evidence",
            "step_ids": [step.step_id for step in steps if step.kind == "evidence"],
            "coverage": "covered-by-step",
        },
        {
            "item": "Failure taxonomy + triage",
            "step_ids": [step_ids[-1]],
            "coverage": "partial",
        },
        {
            "item": "Weekly review/sign-off",
            "step_ids": [step_ids[-1]],
            "coverage": "partial",
        },
    ]


def _status(errors: List[str], warnings: List[str]) -> str:
    if errors:
        return "fail"
    if warnings:
        return "warn"
    return "pass"


def _resolve_evidence_paths(module: str, output_dir: Path) -> List[str]:
    repo_root = Path.cwd()
    resolved: List[str] = []
    for path_template in _module_evidence_paths(module):
        resolved.append(
            path_template.replace("{output_dir}", str(output_dir)).replace("{repo_root}", str(repo_root))
        )
    return resolved


def _live_probe_command(module: str, seed: int) -> List[str]:
    command = _module_base_command(module, seed)
    assert command[0] == "halo-forge"
    return [sys.executable, "-m", "halo_forge.cli", *command[1:], "--help"]


def _select_modules(modules: Optional[Iterable[str]]) -> List[str]:
    if not modules:
        return list(ALL_MODULES)
    selected: List[str] = []
    for module in modules:
        key = str(module or "").strip().lower()
        if not key:
            continue
        if key not in ALL_MODULES:
            raise ValueError(f"unsupported module selection: {key}")
        if key not in selected:
            selected.append(key)
    return selected


def _normalize_text(value: str) -> str:
    text = str(value)
    cwd = str(Path.cwd())
    normalized = text.replace(cwd, "<repo_root>")
    normalized = normalized.replace("/tmp/", "/tmp/<normalized>/")
    return normalized
