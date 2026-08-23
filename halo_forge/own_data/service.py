"""Transport-neutral service for guided imports, inspection, and mapping."""

from __future__ import annotations

import copy
import base64
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from halo_forge.data_lab import TRAINER_DATASET_ADAPTERS

from .guidance import (
    advise_scenarios,
    build_readiness_report,
    guided_example_descriptors,
    semantic_previews,
)
from .imports import DatasetImportManager
from .inspection import IMPORT_ADAPTER_VERSION, fingerprint_path, inspect_path
from .mapping import build_preparation_plan, preview_mapping
from .models import (
    FieldMappingPlan,
    InterfaceCapabilityDescriptor,
    ScenarioAdviceRequest,
    bounded_items,
)
from .registry import TRAINING_SCENARIOS, TrainingScenarioRegistry, interface_capabilities

_TRAINER_MODULES = {
    "sft": "halo_forge.sft.trainer",
    "raft": "halo_forge.rlvr.raft_trainer",
    "grpo": "halo_forge.grpo.trainer",
    "dpo": "halo_forge.dpo.trainer",
    "orpo": "halo_forge.orpo.trainer",
    "rm": "halo_forge.rm.trainer",
    "reasoning": "halo_forge.reasoning.trainer",
    "agentic": "halo_forge.agentic.trainer",
    "vlm": "halo_forge.vlm.trainer",
    "audio": "halo_forge.audio.trainer",
    "cpt": "halo_forge.cpt.trainer",
    "classify": "halo_forge.lab_v11_v15.specialized",
    "embed": "halo_forge.lab_v11_v15.specialized",
    "rerank": "halo_forge.lab_v11_v15.specialized",
}

_MLX_TRAINER_MODULES = {
    "sft": "halo_forge.sft.mlx_trainer",
    "dpo": "halo_forge.dpo.mlx_trainer",
    "grpo": "halo_forge.grpo.mlx_trainer",
    "raft": "halo_forge.rlvr.mlx_raft_trainer",
    "cpt": "halo_forge.cpt.mlx_trainer",
}

_TORCH_BACKENDS = {
    "auto",
    "cpu",
    "cuda",
    "hf",
    "huggingface",
    "mps",
    "rocm",
    "rocm_gfx1151",
    "torch",
    "torch_cpu",
    "torch_cuda",
    "torch_mps",
    "torch_rocm",
    "transformers",
    "trl",
}

# These probes describe what each shipped trainer actually imports on the
# selected execution path.  Keeping them beside the trainer-module map makes
# the capability response a runtime contract instead of a list of packages
# that happen to be installed in a developer environment.
_TORCH_TRAINER_REQUIREMENTS = {
    "sft": ("torch", "transformers", "peft", "datasets", "jsonlines"),
    "raft": ("torch", "transformers", "peft", "datasets"),
    "grpo": ("torch", "transformers", "peft", "trl"),
    "dpo": ("torch", "transformers", "peft", "trl"),
    "orpo": ("torch", "transformers", "peft", "trl"),
    "rm": ("torch", "transformers", "peft", "trl"),
    "reasoning": ("torch", "transformers", "peft"),
    "agentic": ("torch", "transformers", "peft"),
    "vlm": ("torch", "transformers", "peft", "PIL"),
    "audio": ("torch", "transformers", "peft", "numpy"),
    "cpt": ("torch", "transformers", "peft", "datasets", "jsonlines"),
    "classify": ("torch", "transformers"),
    "embed": ("torch", "transformers"),
    "rerank": ("torch", "transformers"),
}

_MLX_TRAINER_REQUIREMENTS = {
    "sft": ("mlx", "mlx_lm"),
    "dpo": ("mlx", "mlx_lm"),
    "grpo": ("mlx", "mlx_lm"),
    "raft": ("mlx", "mlx_lm"),
    "cpt": ("mlx", "mlx_lm"),
}

# Audio paths can be decoded either by torchaudio or by the deliberately
# supported CPU fallback.  A partial fallback install is not sufficient:
# librosa handles resampling while soundfile provides reliable file decoding.
_TRAINER_ALTERNATIVE_REQUIREMENTS = {
    "audio": (("torchaudio",), ("librosa", "soundfile")),
}

_RUNTIME_IMPORT_PROBES = {"torchaudio", "librosa", "soundfile", "pyarrow"}


@lru_cache(maxsize=8)
def _accelerator_runtime_probe(backend_name: str) -> tuple[bool, Optional[str]]:
    """Probe ROCm kernels in an isolated process before advertising training.

    A broken ROCm/PyTorch combination can segfault while importing or moving a
    model.  Running the smallest representative BF16 kernel out of process
    turns that crash into a truthful capability reason instead of taking down
    the dashboard.
    """

    backend = str(backend_name or "").strip().lower()
    if backend not in {"rocm", "rocm_gfx1151", "torch_rocm"}:
        return True, None
    probe = (
        "import torch; "
        "assert torch.cuda.is_available(), 'ROCm device unavailable'; "
        "x=torch.ones((16,16),device='cuda',dtype=torch.bfloat16); "
        "y=x@x; torch.cuda.synchronize(); "
        "assert float(y[0,0]) == 16.0"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "The ROCm runtime check timed out; guided GPU training is unavailable."
    except OSError as exc:
        return False, f"The ROCm runtime check could not start: {exc}"
    if completed.returncode == 0:
        return True, None
    if completed.returncode < 0 or completed.returncode in {134, 139}:
        return (
            False,
            "The ROCm runtime crashed during a small isolated GPU check; choose a qualified runtime before training.",
        )
    detail = completed.stderr.decode("utf-8", errors="replace").strip().splitlines()
    suffix = detail[-1] if detail else f"exit code {completed.returncode}"
    return (
        False,
        f"The ROCm runtime could not execute the required GPU kernel ({suffix[:240]}).",
    )


def _module_available(module_name: str) -> bool:
    try:
        if importlib.util.find_spec(module_name) is None:
            return False
        # Native optional packages can have a discoverable module while being
        # unusable because a shared library or ABI is missing.  These packages
        # are small enough to import for an honest active-runtime answer.
        if module_name in _RUNTIME_IMPORT_PROBES:
            importlib.import_module(module_name)
        return True
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError):
        return False


def _trainer_dependency_report(mode: str, backend: str) -> Dict[str, Any]:
    if backend == "mlx":
        required = _MLX_TRAINER_REQUIREMENTS.get(mode, ())
    elif backend in _TORCH_BACKENDS:
        required = _TORCH_TRAINER_REQUIREMENTS.get(mode, ())
    else:
        required = ()

    module_status = {name: _module_available(name) for name in required}
    missing = [name for name in required if not module_status[name]]
    alternatives = _TRAINER_ALTERNATIVE_REQUIREMENTS.get(mode, ()) if backend in _TORCH_BACKENDS else ()
    alternative_status = []
    for option in alternatives:
        status = {name: _module_available(name) for name in option}
        module_status.update(status)
        alternative_status.append(
            {
                "modules": list(option),
                "available": all(status.values()),
                "module_status": status,
            }
        )

    alternative_satisfied = not alternatives or any(
        option["available"] for option in alternative_status
    )
    requirement_labels = list(required)
    if alternatives:
        requirement_labels.append(
            " or ".join(
                option[0] if len(option) == 1 else " + ".join(option)
                for option in alternatives
            )
        )
    return {
        "available": not missing and alternative_satisfied,
        "requirements": requirement_labels,
        "module_status": module_status,
        "missing_dependencies": missing,
        "alternative_requirements": alternative_status,
        "alternative_satisfied": alternative_satisfied,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GuidedRegistrationConflict(ValueError):
    """A confirmed registration contains contradictory immutable identities."""


class GuidedOwnDataService:
    def __init__(
        self,
        database: Any,
        *,
        datasets_root: Path | str,
        imports_root: Optional[Path | str] = None,
        scheduler: Optional[Any] = None,
        capacity_probe: Optional[Any] = None,
        scenario_registry: TrainingScenarioRegistry = TRAINING_SCENARIOS,
    ) -> None:
        self.db = database
        self.datasets_root = Path(datasets_root).expanduser().resolve()
        self.imports_root = (
            Path(imports_root or self.datasets_root.parent / "imports").expanduser().resolve()
        )
        self.registry = scenario_registry
        self.scheduler = scheduler
        self.corpus_root = self.datasets_root.parent / "corpus"
        self.imports = DatasetImportManager(
            database,
            imports_root=self.imports_root,
            datasets_root=self.datasets_root,
            scenario_registry=scenario_registry,
            capacity_probe=(
                capacity_probe
                or (getattr(scheduler, "capacity_probe", None) if scheduler else None)
            ),
        )

    @staticmethod
    def runtime_trainer_compatibility(
        scenario: Any,
        backend_name: str,
        *,
        trainer_mode: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        """Resolve each advertised trainer against the active runtime.

        Dataset adapters and execution runtimes are separate contracts.  A
        preference adapter, for example, does not make ORPO or RM available
        on MLX.  Returning every declared mode (including blocked modes) lets
        clients explain a truthful unavailable reason without exposing it in
        a guided picker.
        """

        backend = str(backend_name or "").strip().lower()
        accelerator_ready, accelerator_reason = _accelerator_runtime_probe(backend)
        requested = str(trainer_mode or "").strip().lower()
        modes = [
            str(mode).strip().lower()
            for mode in scenario.trainer_modes
            if not requested or str(mode).strip().lower() == requested
        ]
        results: list[Dict[str, Any]] = []
        for mode in modes:
            adapters = TRAINER_DATASET_ADAPTERS.list(
                schema=scenario.canonical_schema,
                trainer_mode=mode,
            )
            adapter = adapters[0] if adapters else None
            reason: Optional[str] = None
            dependency_report: Dict[str, Any] = {
                "available": False,
                "requirements": [],
                "module_status": {},
                "missing_dependencies": [],
                "alternative_requirements": [],
                "alternative_satisfied": True,
            }
            if not scenario.available:
                reason = scenario.unavailable_reason or "This scenario has no verified trainer."
            elif backend in {"", "unknown"}:
                reason = "The active training backend could not be verified."
            elif not accelerator_ready:
                reason = accelerator_reason
            elif adapter is None:
                reason = (
                    f"No verified {mode} dataset adapter accepts "
                    f"{scenario.canonical_schema}."
                )
            else:
                if backend == "mlx":
                    module_name = _MLX_TRAINER_MODULES.get(mode)
                    if module_name is None:
                        reason = f"{mode.upper()} training is not implemented on MLX."
                elif backend in _TORCH_BACKENDS:
                    module_name = _TRAINER_MODULES.get(mode)
                else:
                    module_name = None
                    reason = f"The {backend_name} training runtime is not verified."

                if reason is None:
                    dependency_report = _trainer_dependency_report(mode, backend)
                    missing = dependency_report["missing_dependencies"]
                    if missing:
                        reason = (
                            f"The active {backend_name} runtime is missing "
                            f"{', '.join(missing)}."
                        )
                    elif not dependency_report["alternative_satisfied"]:
                        if mode == "audio":
                            reason = (
                                "Audio training requires torchaudio or both librosa "
                                "and soundfile in the active runtime."
                            )
                        else:
                            reason = (
                                f"The active {backend_name} runtime has no complete "
                                f"dependency option for {mode}."
                            )
                if reason is None:
                    module_present = bool(module_name) and _module_available(str(module_name))
                    if not module_present:
                        reason = (
                            f"The verified {mode} trainer implementation is not installed "
                            f"for {backend_name}."
                        )

            results.append(
                {
                    "adapter_id": adapter.id if adapter is not None else scenario.canonical_schema,
                    "adapter_version": adapter.version if adapter is not None else None,
                    "trainer_mode": mode,
                    "compatible": reason is None,
                    "reason": reason,
                    "required_schema": scenario.canonical_schema,
                    "backend": backend or "unknown",
                    "requirements": copy.deepcopy(dependency_report["requirements"]),
                    "module_status": copy.deepcopy(dependency_report["module_status"]),
                    "missing_dependencies": copy.deepcopy(
                        dependency_report["missing_dependencies"]
                    ),
                    "alternative_requirements": copy.deepcopy(
                        dependency_report["alternative_requirements"]
                    ),
                }
            )
        return results

    @classmethod
    def _runtime_scenario_status(
        cls, scenario: Any, backend_name: str
    ) -> tuple[bool, Optional[str]]:
        if not scenario.available:
            return False, scenario.unavailable_reason
        compatibility = cls.runtime_trainer_compatibility(scenario, backend_name)
        if any(item["compatible"] for item in compatibility):
            return True, None
        reasons = []
        for item in compatibility:
            reason = str(item.get("reason") or "").strip()
            if reason and reason not in reasons:
                reasons.append(reason)
        return (
            False,
            "; ".join(reasons) or "No verified trainer is available in this runtime.",
        )

    @staticmethod
    def _runtime_source_layouts(source_layouts: Any) -> list[Dict[str, Any]]:
        values: list[Dict[str, Any]] = []
        for source_layout in source_layouts:
            layout = str(source_layout).strip().lower()
            requirements: list[str] = []
            available = True
            reason: Optional[str] = None
            if layout == "parquet":
                requirements = ["pyarrow"]
                available = _module_available("pyarrow")
                if not available:
                    reason = (
                        "Parquet import is unavailable because pyarrow is not "
                        "installed in the active runtime."
                    )
            elif layout == "huggingface":
                requirements = ["datasets"]
                available = _module_available("datasets")
                if not available:
                    reason = (
                        "Hugging Face import is unavailable because datasets is "
                        "not installed in the active runtime."
                    )
            elif layout == "pdf":
                requirements = ["pypdf or pdftotext"]
                available = (
                    _module_available("pypdf")
                    or _module_available("PyPDF2")
                    or shutil.which("pdftotext") is not None
                )
                if not available:
                    reason = (
                        "Text-layer PDF extraction requires pypdf, PyPDF2, or "
                        "the pdftotext executable in the active runtime."
                    )
            values.append(
                {
                    "source_format": layout,
                    "available": available,
                    "status": "supported" if available else "unavailable",
                    "reason": reason,
                    "requirements": requirements,
                }
            )
        return values

    @classmethod
    def _runtime_scenario_value(
        cls,
        scenario: Any,
        backend_name: str,
        *,
        include_examples: bool = False,
    ) -> Dict[str, Any]:
        value = scenario.to_dict(include_examples=include_examples)
        compatibility = cls.runtime_trainer_compatibility(scenario, backend_name)
        compatible_modes = [
            str(item["trainer_mode"]) for item in compatibility if item["compatible"]
        ]
        available, reason = cls._runtime_scenario_status(scenario, backend_name)
        source_layout_capabilities = cls._runtime_source_layouts(scenario.source_layouts)
        value.update(
            available=available,
            verified=available,
            unavailable_reason=reason,
            compatible_trainers=compatibility,
            declared_trainer_modes=list(scenario.trainer_modes),
            # Normal-mode pickers consume only runtime-verified modes. The
            # complete declaration remains visible in compatible_trainers.
            trainer_modes=compatible_modes,
            declared_source_layouts=list(scenario.source_layouts),
            source_layouts=[
                item["source_format"]
                for item in source_layout_capabilities
                if item["available"]
            ],
            source_layout_capabilities=source_layout_capabilities,
            active_backend=str(backend_name or "unknown"),
        )
        return value

    @staticmethod
    def _orthogonal_runtime_capabilities(
        scenarios: list[Dict[str, Any]], backend_name: str
    ) -> list[InterfaceCapabilityDescriptor]:
        """Derive independent method/backend/model/format facts.

        Scenario cards remain useful workflow bundles, but consumers should
        not need to reverse-engineer them to answer whether a trainer method,
        backend, model family, or import format is available right now.
        """

        backend = str(backend_name or "unknown").strip().lower() or "unknown"
        descriptors: list[InterfaceCapabilityDescriptor] = []

        methods = sorted(
            {
                str(mode)
                for scenario in scenarios
                for mode in scenario.get("declared_trainer_modes", [])
            }
        )
        all_compatibility = [
            (scenario, compatibility)
            for scenario in scenarios
            for compatibility in scenario.get("compatible_trainers", [])
        ]
        for method in methods:
            relevant = [
                (scenario, item)
                for scenario, item in all_compatibility
                if item.get("trainer_mode") == method
            ]
            compatible = [pair for pair in relevant if pair[1].get("compatible")]
            reasons = list(
                dict.fromkeys(
                    str(item.get("reason"))
                    for _, item in relevant
                    if item.get("reason")
                )
            )
            requirements = sorted(
                {
                    str(requirement)
                    for _, item in relevant
                    for requirement in item.get("requirements", [])
                }
            )
            descriptors.append(
                InterfaceCapabilityDescriptor(
                    id=f"training-method:{method}",
                    kind="training_method",
                    label=method.upper(),
                    status="verified" if compatible else "unavailable",
                    available=bool(compatible),
                    reason=None if compatible else "; ".join(reasons),
                    requirements=tuple(requirements),
                    metadata={
                        "training_method": method,
                        "trainer_mode": method,
                        "backend": backend,
                        "backends": [backend],
                        "scenario_ids": [scenario["id"] for scenario, _ in relevant],
                        "canonical_shapes": sorted(
                            {
                                str(scenario["canonical_shape"])
                                for scenario, _ in relevant
                            }
                        ),
                        "model_families": sorted(
                            {
                                str(family)
                                for scenario, _ in relevant
                                for family in scenario.get("model_families", [])
                            }
                        ),
                    },
                )
            )

        backend_available = any(
            bool(item.get("compatible")) for _, item in all_compatibility
        )
        backend_reasons = list(
            dict.fromkeys(
                str(item.get("reason"))
                for _, item in all_compatibility
                if item.get("reason")
            )
        )
        descriptors.append(
            InterfaceCapabilityDescriptor(
                id=f"backend:{backend}",
                kind="backend",
                label=backend.upper(),
                status="verified" if backend_available else "unavailable",
                available=backend_available,
                reason=None if backend_available else "; ".join(backend_reasons),
                metadata={
                    "backend": backend,
                    "backends": [backend],
                    "training_methods": [
                        descriptor.metadata["training_method"]
                        for descriptor in descriptors
                        if descriptor.kind == "training_method" and descriptor.available
                    ],
                },
            )
        )

        model_families = sorted(
            {
                str(family)
                for scenario in scenarios
                for family in scenario.get("model_families", [])
            }
        )
        for family in model_families:
            relevant_scenarios = [
                scenario
                for scenario in scenarios
                if family in scenario.get("model_families", [])
            ]
            compatible_scenarios = [
                scenario for scenario in relevant_scenarios if scenario.get("available")
            ]
            reasons = list(
                dict.fromkeys(
                    str(scenario.get("unavailable_reason"))
                    for scenario in relevant_scenarios
                    if scenario.get("unavailable_reason")
                )
            )
            descriptors.append(
                InterfaceCapabilityDescriptor(
                    id=f"model-family:{family}",
                    kind="model_family",
                    label=family,
                    status="verified" if compatible_scenarios else "unavailable",
                    available=bool(compatible_scenarios),
                    reason=None if compatible_scenarios else "; ".join(reasons),
                    metadata={
                        "model_family": family,
                        "model_families": [family],
                        "backend": backend,
                        "backends": [backend],
                        "scenario_ids": [scenario["id"] for scenario in relevant_scenarios],
                        "training_methods": sorted(
                            {
                                str(mode)
                                for scenario in compatible_scenarios
                                for mode in scenario.get("trainer_modes", [])
                            }
                        ),
                    },
                )
            )

        source_layouts: Dict[str, list[Dict[str, Any]]] = {}
        for scenario in scenarios:
            for capability in scenario.get("source_layout_capabilities", []):
                source_layouts.setdefault(capability["source_format"], []).append(
                    capability
                )
        for source_format, values in sorted(source_layouts.items()):
            format_available = any(item["available"] for item in values)
            reasons = list(
                dict.fromkeys(
                    str(item.get("reason")) for item in values if item.get("reason")
                )
            )
            requirements = sorted(
                {
                    str(requirement)
                    for item in values
                    for requirement in item.get("requirements", [])
                }
            )
            descriptors.append(
                InterfaceCapabilityDescriptor(
                    id=f"source-format:{source_format}",
                    kind="source_format",
                    label=source_format.replace("_", " ").title(),
                    status="supported" if format_available else "unavailable",
                    available=format_available,
                    reason=None if format_available else "; ".join(reasons),
                    requirements=tuple(requirements),
                    metadata={"source_format": source_format},
                )
            )
        return descriptors

    def list_capabilities(self, *, backend_name: str) -> Dict[str, Any]:
        items = [value.to_dict() for value in interface_capabilities(backend_name=backend_name)]
        by_id = {item["id"]: item for item in items}
        runtime_scenarios: list[Dict[str, Any]] = []
        for scenario in self.registry.list(include_unavailable=True):
            scenario_value = self._runtime_scenario_value(scenario, backend_name)
            runtime_scenarios.append(scenario_value)
            item = by_id[f"scenario:{scenario.id}"]
            item.update(
                available=scenario_value["available"],
                status="verified" if scenario_value["available"] else "unavailable",
                reason=scenario_value["unavailable_reason"],
                compatible_trainers=copy.deepcopy(
                    scenario_value["compatible_trainers"]
                ),
                trainer_modes=list(scenario_value["trainer_modes"]),
            )
            item.setdefault("metadata", {})["trainer_modes"] = list(
                scenario_value["trainer_modes"]
            )
        items.extend(
            descriptor.to_dict()
            for descriptor in self._orthogonal_runtime_capabilities(
                runtime_scenarios, backend_name
            )
        )
        return {
            "items": items,
            "total": len(items),
            "limit": len(items),
            "offset": 0,
            "active_backend": backend_name,
            "scenario_registry_revision": self.registry.revision,
        }

    def list_scenarios(
        self,
        *,
        backend_name: str,
        include_unavailable: bool = False,
        modality: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        values = []
        for scenario in self.registry.list(include_unavailable=True):
            if modality and scenario.modality != modality:
                continue
            value = self._runtime_scenario_value(scenario, backend_name)
            if include_unavailable or value["available"]:
                values.append(value)
        result = bounded_items(values, limit=limit, offset=offset)
        result.update(
            registry_revision=self.registry.revision,
            scenario_registry_revision=self.registry.revision,
            active_backend=backend_name,
        )
        return result

    def get_scenario(
        self,
        identifier: str,
        *,
        backend_name: str,
        include_examples: bool = False,
    ) -> Dict[str, Any]:
        scenario = self.registry.get(identifier)
        return self._runtime_scenario_value(
            scenario,
            backend_name,
            include_examples=include_examples,
        )

    def list_examples(self, identifier: str) -> Dict[str, Any]:
        scenario = self.registry.get(identifier)
        descriptors = {
            item.id: item
            for item in guided_example_descriptors(registry=self.registry)
            if item.scenario_revision_id == scenario.revision_id
        }
        items = [
            {
                **item.to_dict(
                    include_records=True,
                    scenario_revision_id=scenario.revision_id,
                ),
                **(
                    descriptors[item.id].to_dict()
                    if item.id in descriptors
                    else {}
                ),
            }
            for item in scenario.examples
        ]
        return {"items": items, "total": len(items), "limit": len(items), "offset": 0}

    def list_guided_examples(self, *, backend_name: str) -> Dict[str, Any]:
        runtime_values = {
            scenario.revision_id: self._runtime_scenario_value(
                scenario, backend_name
            )
            for scenario in self.registry.list(include_unavailable=True)
        }
        items = [
            item.to_dict()
            for item in guided_example_descriptors(
                registry=self.registry, runtime_values=runtime_values
            )
        ]
        return {
            "items": items,
            "total": len(items),
            "limit": len(items),
            "offset": 0,
            "scenario_registry_revision": self.registry.revision,
            "active_backend": backend_name,
        }

    def scenario_advice(
        self, payload: Mapping[str, Any], *, backend_name: str
    ) -> Dict[str, Any]:
        runtime_values = {
            scenario.revision_id: self._runtime_scenario_value(
                scenario, backend_name
            )
            for scenario in self.registry.list(include_unavailable=True)
        }
        request = ScenarioAdviceRequest.from_value(payload)
        return advise_scenarios(
            request,
            registry=self.registry,
            runtime_values=runtime_values,
        ).to_dict()

    def scenario_template(
        self, identifier: str, example_id: Optional[str] = None
    ) -> Dict[str, Any]:
        scenario = self.registry.get(identifier)
        filename, files = self.registry.template_files(identifier, example_id)
        content = files[filename]
        example = next(item for item in scenario.examples if item.filename == filename)
        import hashlib

        value = example.to_dict(include_records=True, scenario_revision_id=scenario.revision_id)
        value.update(
            content=content.decode("utf-8"),
            size_bytes=len(content),
            checksum=hashlib.sha256(content).hexdigest(),
            files=[
                {
                    "path": path,
                    "size_bytes": len(file_content),
                    "checksum": hashlib.sha256(file_content).hexdigest(),
                    "content": (
                        file_content.decode("utf-8") if path == filename else None
                    ),
                    "content_base64": (
                        None
                        if path == filename
                        else base64.b64encode(file_content).decode("ascii")
                    ),
                }
                for path, file_content in files.items()
            ],
        )
        return value

    @staticmethod
    def _file_view(record: Any) -> Dict[str, Any]:
        value = record.to_dict()
        value.update(
            uploaded_bytes=value.pop("received_bytes"),
            content_hash=value.pop("content_sha256"),
        )
        if value["status"] == "complete":
            value["status"] = "verified" if value.get("content_hash") else "uploaded"
        value.pop("staging_path", None)
        value.pop("metadata", None)
        value.pop("expected_sha256", None)
        return value

    def import_view(self, record: Any) -> Dict[str, Any]:
        files = [self._file_view(value) for value in self.db.list_dataset_import_files(record.id)]
        value = record.to_dict()
        value.update(
            files=files,
            total_files=len(files),
            total_bytes=(
                value.get("expected_size_bytes")
                if value.get("expected_size_bytes") is not None
                else sum(item["size_bytes"] for item in files)
            ),
            uploaded_bytes=value.pop("received_size_bytes"),
            inspection_id=value.pop("latest_inspection_id"),
        )
        capacity = self.imports.capacity_status(record)
        value["disk_forecast"] = capacity
        value["readiness"] = {
            "ready": capacity["ready"],
            "requires_capacity_override": capacity["requires_override"],
            "blockers": list(capacity["blockers"]),
            "warnings": list(capacity["warnings"]),
            "remedy": capacity["remedy"],
        }
        for internal in ("staging_path", "managed_source_path", "metadata"):
            value.pop(internal, None)
        return value

    def create_import(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return self.import_view(self.imports.create(payload))

    def huggingface_options(self, repo_id: str, revision: str) -> Dict[str, Any]:
        if not repo_id.strip() or not revision.strip():
            raise ValueError("repo_id and pinned revision are required")
        try:
            from huggingface_hub import HfApi  # type: ignore
            from datasets import get_dataset_config_names, get_dataset_split_names  # type: ignore

            resolved = str(HfApi().dataset_info(repo_id, revision=revision).sha)
            configs = list(get_dataset_config_names(repo_id, revision=resolved))
            items = []
            for config in configs or [None]:
                splits = list(get_dataset_split_names(repo_id, config, revision=resolved))
                items.append({"config": config, "splits": splits})
        except Exception as exc:
            raise ValueError(f"could not inspect pinned Hugging Face dataset: {exc}") from exc
        return {
            "repo_id": repo_id,
            "requested_revision": revision,
            "resolved_revision": resolved,
            "items": items,
            "total": len(items),
            "limit": len(items),
            "offset": 0,
        }

    def get_import(self, import_id: str) -> Optional[Dict[str, Any]]:
        record = self.db.get_dataset_import(import_id)
        return self.import_view(record) if record else None

    def list_imports(self, *, status: Optional[str], limit: int, offset: int) -> Dict[str, Any]:
        all_values = self.db.list_dataset_imports(status=status, limit=10000, offset=0)
        items = [self.import_view(value) for value in all_values]
        return bounded_items(items, limit=limit, offset=offset)

    def create_import_file(self, import_id: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return self._file_view(self.imports.create_file(import_id, payload))

    def upload_chunk(
        self, import_id: str, file_id: str, content: bytes, **range_values: Any
    ) -> Dict[str, Any]:
        return self._file_view(
            self.imports.write_chunk(import_id, file_id, content, **range_values)
        )

    @staticmethod
    def inspection_view(record: Any) -> Dict[str, Any]:
        value = record.to_dict()
        preview_policy = value.get("statistics", {}).get("preview_policy") or {}
        if isinstance(preview_policy, Mapping):
            preview_policy_label = (
                f"first {int(preview_policy.get('head', 100))} + seed-"
                f"{int(preview_policy.get('seed', 42))} deterministic reservoir of "
                f"{int(preview_policy.get('reservoir', 900))}"
            )
        else:
            preview_policy_label = str(preview_policy) if preview_policy else None
        value.update(
            row_count=value.pop("total_records"),
            preview_records=value.pop("preview"),
            schema_candidates=value.pop("candidates"),
            parse_errors=value.pop("issues"),
            preview_policy=preview_policy_label,
            preview_policy_details=json_copy(preview_policy),
            media_summary=json_copy(value.get("statistics", {}).get("media_summary", {})),
            extraction_summary=json_copy(
                value.get("statistics", {}).get("extraction_summary", {})
            ),
            warnings=list(value.get("statistics", {}).get("warnings", [])),
            stage=value["status"],
            progress_percent=100.0 if value["status"] == "completed" else None,
        )
        value.pop("statistics", None)
        return value

    def request_inspection(
        self, import_id: str, *, scenario_revision_id: Optional[str] = None, force: bool = False
    ) -> Dict[str, Any]:
        session = self.db.get_dataset_import(import_id)
        if session is None:
            raise KeyError(import_id)
        if session.status not in {"ready", "completed", "failed", "cancelled", "published"}:
            raise ValueError("dataset import is not ready for inspection")
        if session.source_kind == "huggingface":
            # Do not download or scan a remote dataset on the HTTP request.
            # The immutable pin is sufficient to identify reusable inspection
            # work; materialization happens inside the durable worker attempt.
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "repo_id": session.source_uri,
                        "config": session.source_config,
                        "split": session.source_split or "train",
                        "commit": session.resolved_revision,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        else:
            source = self.imports.source_path(import_id)
            fingerprint, _, _ = fingerprint_path(source)
        exact = self.db.find_dataset_source_inspection(
            source_fingerprint=fingerprint,
            import_adapter_version=IMPORT_ADAPTER_VERSION,
            scenario_registry_revision=self.registry.revision,
        )
        matching_probe = next(
            (
                value
                for value in self.db.list_dataset_source_inspections(
                    status="completed", limit=10000
                )
                if value.import_adapter_version == IMPORT_ADAPTER_VERSION
                and value.scenario_registry_revision == self.registry.revision
                and str(
                    (value.statistics or {}).get("source_probe_fingerprint")
                    or value.source_fingerprint
                )
                == fingerprint
            ),
            None,
        )
        existing = exact or matching_probe
        existing_has_external_media = bool(
            existing
            and (
                (existing.statistics or {}).get("asset_fingerprints")
                or int(
                    ((existing.statistics or {}).get("media_summary") or {}).get(
                        "referenced", 0
                    )
                )
            )
        )
        if (
            existing is not None
            and existing.status == "completed"
            and not force
            and not existing_has_external_media
        ):
            self.db.link_dataset_import_inspection(import_id, existing.id)
            self.db.update_dataset_import(
                import_id,
                status="completed",
                fingerprint=fingerprint,
                latest_inspection_id=existing.id,
                completed_at=_now(),
                error=None,
            )
            inspection_value = self.inspection_view(existing)
            inspection_value["import_id"] = import_id
            return {
                "inspection": inspection_value,
                "work_item_id": existing.work_item_id,
                "reused": True,
            }
        if existing is None or existing.status == "completed":
            provisional_fingerprint = (
                fingerprint
                if exact is None
                else f"{fingerprint}:pending:{uuid.uuid4().hex}"
            )
            inspection = self.db.create_dataset_source_inspection(
                import_id=import_id,
                source_fingerprint=provisional_fingerprint,
                import_adapter_version=IMPORT_ADAPTER_VERSION,
                scenario_registry_revision=self.registry.revision,
                scenario_revision_id=scenario_revision_id,
            )
        else:
            inspection = self.db.update_dataset_source_inspection(
                existing.id,
                status="queued",
                scenario_revision_id=scenario_revision_id or existing.scenario_revision_id,
                error=None,
                completed_at=None,
            )
        assert inspection is not None
        self.db.link_dataset_import_inspection(import_id, inspection.id)
        work_item_id = None
        if self.scheduler is not None:
            resource_requirements = None
            if session.source_kind == "huggingface":
                capacity = dict(session.metadata.get("capacity") or {})
                resource_requirements = {
                    "output_path": str(self.imports_root),
                    "projected_disk_bytes": int(session.expected_size_bytes or 0),
                    "capacity_preflight": True,
                }
                override_reason = str(capacity.get("override_reason") or "").strip()
                if override_reason:
                    resource_requirements["capacity_override_reason"] = override_reason
            work = self.scheduler.enqueue(
                kind="dataset_inspection",
                launch_spec={
                    "handler": "own_data.inspect",
                    "inspection_id": inspection.id,
                    "import_id": import_id,
                    "dataset_root": str(self.datasets_root),
                    "imports_root": str(self.imports_root),
                },
                resource_class="cpu",
                resource_requirements=resource_requirements,
                domain_kind="dataset_inspection",
                domain_id=inspection.id,
                max_retries=2,
            )
            work_item_id = work.id
            inspection = (
                self.db.update_dataset_source_inspection(inspection.id, work_item_id=work.id)
                or inspection
            )
        self.db.update_dataset_import(
            import_id,
            status="inspecting",
            fingerprint=fingerprint,
            latest_inspection_id=inspection.id,
            work_item_id=work_item_id,
            error=None,
        )
        return {
            "inspection": self.inspection_view(inspection),
            "work_item_id": work_item_id,
            "reused": False,
        }

    def execute_inspection(self, inspection_id: str) -> Dict[str, Any]:
        inspection = self.db.get_dataset_source_inspection(inspection_id)
        if inspection is None:
            raise KeyError(inspection_id)
        if inspection.status == "completed":
            return self.inspection_view(inspection)
        if not inspection.import_id:
            raise ValueError("inspection has no import source")
        self.db.update_dataset_source_inspection(inspection_id, status="running", error=None)
        try:

            def work_record() -> Any:
                return (
                    self.db.get_work_item(inspection.work_item_id)
                    if inspection.work_item_id
                    else None
                )

            def cancelled() -> bool:
                work = work_record()
                return bool(
                    work
                    and (
                        work.cancel_requested
                        or work.status in {"cancelled", "failed", "interrupted"}
                    )
                )

            def progress(processed: int) -> None:
                work = work_record()
                if work is not None and self.scheduler is not None:
                    self.scheduler.heartbeat(
                        work,
                        stage="inspecting_source",
                        progress={"processed_records": int(processed), "total_records": None},
                    )

            source_path = self.imports.source_path(
                inspection.import_id,
                progress=progress,
                cancelled=cancelled,
            )
            session = self.db.get_dataset_import(inspection.import_id)
            if session is None:
                raise ValueError("inspection import no longer exists")
            scenario_revision_id = (
                inspection.scenario_revision_id or session.scenario_revision_id
            )
            scenario = (
                self.registry.get(scenario_revision_id)
                if scenario_revision_id
                else None
            )
            inspection_source = source_path
            extraction: Optional[Dict[str, Any]] = None
            extraction_quarantine: list[Dict[str, Any]] = []
            if scenario is not None and scenario.canonical_schema == "corpus":
                from halo_forge.corpus_lab import CorpusExtractionService

                extraction_service = CorpusExtractionService(
                    self.db,
                    root=self.corpus_root,
                    scheduler=None,
                )
                extracted = extraction_service.launch(
                    source_path,
                    import_id=inspection.import_id,
                    synchronous=True,
                )
                extraction = dict(extracted["extraction"])
                extraction_quarantine = [
                    copy.deepcopy(value)
                    for value in extracted.get("quarantine") or []
                    if isinstance(value, Mapping)
                ]
                bundle_path = str(extraction.get("bundle_path") or "").strip()
                if not bundle_path:
                    raise RuntimeError(
                        "corpus extraction completed without a published bundle"
                    )
                inspection_source = Path(bundle_path) / "documents.jsonl"

            result = inspect_path(
                inspection_source,
                registry=self.registry,
                progress=progress,
                cancelled=cancelled,
            )
            if extraction is not None:
                extracted_count = int(extraction.get("document_count") or 0)
                quarantined_count = int(
                    extraction.get("quarantined_count")
                    or len(extraction_quarantine)
                )
                result["total_records"] = extracted_count + quarantined_count
                result["valid_records"] = extracted_count
                result["invalid_records"] = quarantined_count
                result["size_bytes"] = int(
                    (extraction.get("provenance") or {}).get(
                        "source_size_bytes", result["size_bytes"]
                    )
                    or result["size_bytes"]
                )
                result["file_count"] = int(
                    (extraction.get("provenance") or {}).get(
                        "source_file_count", result["file_count"]
                    )
                    or result["file_count"]
                )
                extraction_issues = [
                    {
                        "code": str(
                            value.get("error_code")
                            or "document_extraction_failed"
                        ),
                        "index": value.get("ordinal"),
                        "path": value.get("relative_path")
                        or value.get("source_uri"),
                        "message": str(
                            value.get("error")
                            or "The document could not be extracted."
                        ),
                        "provenance": copy.deepcopy(
                            value.get("provenance") or {}
                        ),
                    }
                    for value in extraction_quarantine
                ]
                result["issues"] = (
                    list(result.get("issues") or []) + extraction_issues
                )[:500]
                result["statistics"]["extraction_summary"] = {
                    "extraction_id": extraction.get("id"),
                    "content_hash": extraction.get("content_hash"),
                    "bundle_path": extraction.get("bundle_path"),
                    "manifest_hash": extraction.get("manifest_hash"),
                    "extractor_version": extraction.get("extractor_version"),
                    "document_count": extracted_count + quarantined_count,
                    "extracted": extracted_count,
                    "failed": quarantined_count,
                    "quarantined": quarantined_count,
                    "quarantine_preview": extraction_quarantine[:20],
                }
                result["statistics"]["inspection_source_path"] = str(
                    inspection_source
                )
                result["statistics"]["raw_source_path"] = str(source_path)
            result["statistics"]["source_probe_fingerprint"] = str(
                session.fingerprint
                or result["statistics"].get("source_probe_fingerprint")
                or ""
            )
            canonical_identity = str(result["source_fingerprint"])
            identity_collision = self.db.find_dataset_source_inspection(
                source_fingerprint=canonical_identity,
                import_adapter_version=IMPORT_ADAPTER_VERSION,
                scenario_registry_revision=self.registry.revision,
            )
            stored_identity = (
                canonical_identity
                if identity_collision is None or identity_collision.id == inspection_id
                else f"{canonical_identity}:import:{inspection.import_id}"
            )
            result["statistics"]["content_identity"] = canonical_identity
            completed = self.db.update_dataset_source_inspection(
                inspection_id,
                status="completed",
                source_fingerprint=stored_identity,
                scenario_revision_id=(
                    inspection.scenario_revision_id or result["scenario_revision_id"]
                ),
                total_records=result["total_records"],
                valid_records=result["valid_records"],
                invalid_records=result["invalid_records"],
                sample_count=result["sample_count"],
                size_bytes=result["size_bytes"],
                fields=result["fields"],
                candidates=result["candidates"],
                preview=result["preview"],
                issues=result["issues"],
                statistics=result["statistics"],
                completed_at=_now(),
            )
            linked_import_ids = self.db.list_dataset_inspection_import_ids(inspection_id)
            for import_id in linked_import_ids or [inspection.import_id]:
                self.db.update_dataset_import(
                    import_id,
                    status="completed",
                    latest_inspection_id=inspection_id,
                    completed_at=_now(),
                    error=None,
                )
            assert completed is not None
            return self.inspection_view(completed)
        except Exception as exc:
            was_cancelled = "cancel" in str(exc).lower()
            self.db.update_dataset_source_inspection(
                inspection_id,
                status="cancelled" if was_cancelled else "failed",
                error=f"{type(exc).__name__}: {exc}",
                completed_at=_now(),
            )
            linked_import_ids = self.db.list_dataset_inspection_import_ids(inspection_id)
            for import_id in linked_import_ids or [inspection.import_id]:
                self.db.update_dataset_import(
                    import_id,
                    status="cancelled" if was_cancelled else "failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise

    def get_inspection(self, inspection_id: str) -> Optional[Dict[str, Any]]:
        record = self.db.get_dataset_source_inspection(inspection_id)
        return self.inspection_view(record) if record else None

    def cancel_inspection(self, inspection_id: str) -> Optional[Dict[str, Any]]:
        record = self.db.get_dataset_source_inspection(inspection_id)
        if record is None:
            return None
        if record.status == "completed":
            raise ValueError("completed inspection cannot be cancelled")
        if record.work_item_id and self.scheduler is not None:
            self.scheduler.cancel(record.work_item_id)
        updated = self.db.update_dataset_source_inspection(
            inspection_id, status="cancelled", completed_at=_now()
        )
        if record.import_id:
            self.db.update_dataset_import(record.import_id, status="cancelled")
        return self.inspection_view(updated)

    def cancel_import(self, import_id: str) -> Dict[str, Any]:
        session = self.db.get_dataset_import(import_id)
        if session is None:
            raise KeyError(import_id)
        if session.status in {"published", "expired", "completed"}:
            raise ValueError(f"dataset import in {session.status!r} state cannot be cancelled")
        if session.latest_inspection_id:
            inspection = self.db.get_dataset_source_inspection(session.latest_inspection_id)
            if inspection is not None and inspection.status not in {
                "completed",
                "cancelled",
                "failed",
                "interrupted",
            }:
                self.cancel_inspection(inspection.id)
        elif session.work_item_id and self.scheduler is not None:
            self.scheduler.cancel(session.work_item_id)
        updated = self.db.update_dataset_import(import_id, status="cancelled", error=None)
        assert updated is not None
        return {"dataset_import": self.import_view(updated), "work_item_id": updated.work_item_id}

    def retry_import(self, import_id: str) -> Dict[str, Any]:
        session = self.db.get_dataset_import(import_id)
        if session is None:
            raise KeyError(import_id)
        if session.status not in {"failed", "cancelled", "interrupted"}:
            raise ValueError(f"dataset import in {session.status!r} state cannot be retried")
        if session.latest_inspection_id:
            inspection = self.db.get_dataset_source_inspection(session.latest_inspection_id)
            if inspection is not None and inspection.status in {
                "failed",
                "cancelled",
                "interrupted",
            }:
                return self.retry_inspection(inspection.id)
        files = self.db.list_dataset_import_files(import_id)
        for file_record in files:
            if file_record.status == "failed":
                target = Path(file_record.staging_path)
                if target.is_symlink():
                    raise ValueError("upload staging file became a symbolic link")
                with target.open("wb"):
                    pass
                self.db.update_dataset_import_file(
                    file_record.id,
                    status="pending",
                    received_bytes=0,
                    content_sha256=None,
                    error=None,
                    completed_at=None,
                )
        refreshed_files = self.db.list_dataset_import_files(import_id)
        ready = bool(refreshed_files) and all(
            item.status == "complete" for item in refreshed_files
        )
        status = (
            "ready"
            if ready or session.source_kind != "upload"
            else "uploading" if refreshed_files else "draft"
        )
        updated = self.db.update_dataset_import(import_id, status=status, error=None)
        assert updated is not None
        return {"dataset_import": self.import_view(updated), "work_item_id": None}

    def retry_inspection(self, inspection_id: str) -> Dict[str, Any]:
        record = self.db.get_dataset_source_inspection(inspection_id)
        if record is None:
            raise KeyError(inspection_id)
        if record.status not in {"failed", "cancelled", "interrupted"}:
            raise ValueError(f"inspection in {record.status!r} state cannot be retried")
        return self.request_inspection(
            str(record.import_id), scenario_revision_id=record.scenario_revision_id, force=True
        )

    def mapping_preview(self, inspection_id: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        inspection = self.db.get_dataset_source_inspection(inspection_id)
        if inspection is None:
            raise KeyError(inspection_id)
        if inspection.status != "completed":
            raise ValueError("inspection must be complete before mapping preview")
        plan = FieldMappingPlan.from_value(payload.get("mapping_plan") or payload)
        return preview_mapping(inspection.to_dict(), plan).to_dict()

    def semantic_preview(
        self,
        inspection_id: str,
        payload: Mapping[str, Any],
        *,
        limit: int = 50,
    ) -> Dict[str, Any]:
        plan = FieldMappingPlan.from_value(payload.get("mapping_plan") or payload)
        scenario = self.registry.get(plan.scenario_revision_id)
        preview = self.mapping_preview(
            inspection_id, {"mapping_plan": plan.to_dict()}
        )
        return semantic_previews(
            preview,
            canonical_schema=scenario.canonical_schema,
            limit=limit,
        )

    def readiness_report(
        self, inspection_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        inspection = self.db.get_dataset_source_inspection(inspection_id)
        if inspection is None:
            raise KeyError(inspection_id)
        if inspection.status != "completed":
            raise ValueError("inspection must be complete before readiness analysis")
        raw = payload.get("preparation_plan") or payload
        mapping_value = raw.get("mapping_plan") if isinstance(raw, Mapping) else None
        if not isinstance(mapping_value, Mapping):
            mapping_value = payload.get("mapping_plan")
        if not isinstance(mapping_value, Mapping):
            raise ValueError("mapping_plan is required")
        plan = FieldMappingPlan.from_value(mapping_value)
        scenario = self.registry.get(plan.scenario_revision_id)
        mapping = self.mapping_preview(
            inspection_id, {"mapping_plan": plan.to_dict()}
        )
        preparation_input: Dict[str, Any]
        if isinstance(raw, Mapping) and raw.get("recipe") is not None:
            preparation_input = dict(raw)
        else:
            preparation_input = {"mapping_plan": plan.to_dict()}
        preparation = self.preparation_preview(
            inspection_id, {"preparation_plan": preparation_input}
        )
        inspection_value = inspection.to_dict()
        inspection_value.update(self.inspection_view(inspection))
        inspection_value["statistics"] = copy.deepcopy(
            getattr(inspection, "statistics", {}) or {}
        )
        return build_readiness_report(
            inspection_value,
            mapping,
            preparation,
            canonical_schema=scenario.canonical_schema,
            scenario_revision_id=scenario.revision_id,
        ).to_dict()

    def preparation_preview(self, inspection_id: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        from halo_forge.data_lab.recipe import Recipe

        inspection = self.db.get_dataset_source_inspection(inspection_id)
        if inspection is None:
            raise KeyError(inspection_id)
        if inspection.status != "completed":
            raise ValueError("inspection must be complete before preparation preview")
        raw = payload.get("preparation_plan") or payload
        mapping = raw.get("mapping_plan") if isinstance(raw, Mapping) else None
        if not isinstance(mapping, Mapping):
            raise ValueError("preparation_plan.mapping_plan is required")
        resolved = build_preparation_plan(inspection.to_dict(), mapping).to_dict()
        supplied_recipe = raw.get("recipe") if isinstance(raw, Mapping) else None
        if supplied_recipe is not None:
            recipe_value = copy.deepcopy(dict(supplied_recipe))
            scenario = self.registry.get(str(mapping.get("scenario_revision_id") or ""))
            recipe_value["schema"] = scenario.canonical_schema
            steps = recipe_value.get("steps")
            if not isinstance(steps, list) or not steps:
                raise ValueError("preparation recipe requires an ordered steps list")
            map_step = next(
                (
                    step
                    for step in steps
                    if isinstance(step, Mapping)
                    and str(step.get("kind") or step.get("type") or "").lower() == "map"
                ),
                None,
            )
            if map_step is None:
                map_step = {"kind": "map"}
                steps.insert(0, map_step)
            map_step["schema"] = scenario.canonical_schema
            map_step["fields"] = FieldMappingPlan.from_value(mapping).to_dict()["mappings"]
            resolved["recipe"] = Recipe.from_value(recipe_value).to_dict()
        return resolved

    def registration_payload(
        self, inspection_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Resolve a validated registration into its durable source location.

        This method may publish an uploaded source and is therefore intended
        for the workstation worker.  HTTP request handlers should call
        :meth:`registration_request` instead; that path validates only the
        bounded inspection sample and never copies or reloads the source.
        """

        request = self.registration_request(inspection_id, payload)
        inspection = self.db.get_dataset_source_inspection(inspection_id)
        assert inspection is not None
        scenario = self.registry.get(request["scenario_revision_id"])
        mapping = FieldMappingPlan.from_value(request["mapping_plan"])
        session = self.db.get_dataset_import(request["import_id"])
        assert session is not None
        if session.source_kind in {"upload", "huggingface"}:
            source_path, fingerprint = self.imports.publish(
                session.id,
                override_reason=(
                    str(payload.get("capacity_override_reason") or "").strip() or None
                ),
            )
            rewritten = {}
            for target, expression in mapping.mappings.items():
                if expression.kind == "media_root":
                    from .models import FieldMappingExpression

                    rewritten[target] = FieldMappingExpression(
                        kind="media_root",
                        source=expression.source,
                        media_root=str(source_path),
                    )
                else:
                    rewritten[target] = expression
            mapping = FieldMappingPlan(
                scenario_revision_id=mapping.scenario_revision_id,
                mappings=rewritten,
                confirmed=True,
                version=2,
            )
        else:
            source_path = self.imports.source_path(session.id)
            fingerprint = inspection.source_fingerprint
        extraction_summary: Dict[str, Any] = {}
        if scenario.canonical_schema == "corpus":
            extraction_summary = copy.deepcopy(
                dict(
                    (inspection.statistics or {}).get("extraction_summary")
                    or {}
                )
            )
            extraction_id = str(
                extraction_summary.get("extraction_id") or ""
            ).strip()
            extraction_record = (
                self.db.get_document_extraction(extraction_id)
                if extraction_id
                else None
            )
            if extraction_record is None or extraction_record.status != "completed":
                raise ValueError(
                    "the reviewed corpus extraction is missing or incomplete; "
                    "inspect the source again"
                )
            from halo_forge.corpus_lab import CorpusExtractionService

            extraction_service = CorpusExtractionService(
                self.db,
                root=self.corpus_root,
                scheduler=None,
            )
            verification = extraction_service.verify(extraction_id)
            if not verification.get("valid"):
                detail = "; ".join(
                    str(value) for value in verification.get("errors") or []
                )
                raise ValueError(
                    "the reviewed corpus extraction bundle failed checksum "
                    f"verification: {detail or 'unknown integrity error'}"
                )
            if (
                extraction_summary.get("content_hash")
                and str(extraction_summary["content_hash"])
                != str(extraction_record.content_hash)
            ):
                raise ValueError(
                    "the reviewed corpus extraction identity no longer matches "
                    "the inspection"
                )
            if (
                extraction_summary.get("manifest_hash")
                and str(extraction_summary["manifest_hash"])
                != str(extraction_record.manifest_hash)
            ):
                raise ValueError(
                    "the reviewed corpus extraction manifest no longer matches "
                    "the inspection"
                )
            bundle_path = Path(
                str(extraction_record.bundle_path or "")
            ).expanduser()
            documents_path = bundle_path / "documents.jsonl"
            if not documents_path.is_file():
                raise ValueError(
                    "the reviewed corpus extraction bundle is missing documents.jsonl; "
                    "verify or retry extraction"
                )
            source_path = documents_path.resolve()
        rewritten_preparation = copy.deepcopy(dict(request["preparation_plan_input"]))
        rewritten_preparation["mapping_plan"] = mapping.to_dict()
        preparation = self.preparation_preview(
            inspection_id, {"preparation_plan": rewritten_preparation}
        )
        return {
            "name": request["name"],
            "description": request.get("description"),
            "modality": scenario.modality,
            "canonical_schema": scenario.canonical_schema,
            "source": {
                "kind": "local",
                "uri": str(source_path),
                "field_mapping": mapping.to_dict()["mappings"],
            },
            "field_mapping": mapping.to_dict()["mappings"],
            "scenario_revision_id": scenario.revision_id,
            "preparation_plan": preparation,
            "import_id": session.id,
            "source_fingerprint": fingerprint,
            "extraction": extraction_summary or None,
        }

    def registration_request(
        self, inspection_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Validate a guided registration without touching the complete source."""

        inspection = self.db.get_dataset_source_inspection(inspection_id)
        if inspection is None or inspection.status != "completed":
            raise ValueError("inspection must be complete before dataset registration")
        linked_import_ids = self.db.list_dataset_inspection_import_ids(inspection_id)
        legacy_import_id = inspection.import_id
        requested_import_id = str(payload.get("import_id") or "").strip()
        if requested_import_id:
            if (
                requested_import_id not in linked_import_ids
                and requested_import_id != legacy_import_id
            ):
                raise GuidedRegistrationConflict(
                    "selected import and immutable inspection do not match"
                )
            import_id = requested_import_id
        elif linked_import_ids:
            # Reused immutable inspections are intentionally linked rather
            # than mutated.  The most recently linked import is the active
            # default for older clients that do not yet send import_id.
            import_id = linked_import_ids[0]
        elif legacy_import_id:
            import_id = legacy_import_id
        else:
            raise ValueError("inspection has no import source")
        scenario_id = str(payload.get("scenario_revision_id") or "").strip()
        scenario = self.registry.get(scenario_id)
        # The active backend is checked by PublicApiService.  This local check
        # prevents intentionally unavailable scenario revisions from entering
        # durable work through another transport.
        if not scenario.available:
            raise ValueError(
                scenario.unavailable_reason or "scenario is unavailable"
            )
        mapping = FieldMappingPlan.from_value(payload.get("mapping_plan") or {})
        if mapping.scenario_revision_id != scenario.revision_id:
            raise GuidedRegistrationConflict(
                "mapping plan and selected scenario revision do not match"
            )
        if not mapping.confirmed:
            raise ValueError("mapping plan must be explicitly confirmed")
        preparation_payload = payload.get("preparation_plan")
        if not isinstance(preparation_payload, Mapping):
            raise ValueError("preparation_plan is required")
        preparation_scenario = str(
            preparation_payload.get("scenario_revision_id") or scenario.revision_id
        ).strip()
        if preparation_scenario != scenario.revision_id:
            raise GuidedRegistrationConflict(
                "preparation plan and selected scenario revision do not match"
            )
        preparation_mapping_value = preparation_payload.get("mapping_plan")
        if not isinstance(preparation_mapping_value, Mapping):
            raise ValueError("preparation_plan.mapping_plan is required")
        preparation_mapping = FieldMappingPlan.from_value(preparation_mapping_value)
        if preparation_mapping.to_dict() != mapping.to_dict():
            raise GuidedRegistrationConflict(
                "preparation mapping and confirmed mapping plan do not match"
            )
        recipe = preparation_payload.get("recipe")
        if isinstance(recipe, Mapping):
            recipe_schema = str(recipe.get("schema") or scenario.canonical_schema).strip().lower()
            if recipe_schema != scenario.canonical_schema:
                raise GuidedRegistrationConflict(
                    "recipe schema and selected scenario canonical schema do not match"
                )
            recipe_scenario = str(
                recipe.get("scenario_revision_id") or scenario.revision_id
            ).strip()
            if recipe_scenario != scenario.revision_id:
                raise GuidedRegistrationConflict(
                    "recipe and selected scenario revision do not match"
                )
            for step in recipe.get("steps") or []:
                if not isinstance(step, Mapping):
                    continue
                if str(step.get("kind") or step.get("type") or "").lower() != "map":
                    continue
                step_schema = str(step.get("schema") or scenario.canonical_schema).lower()
                if step_schema != scenario.canonical_schema:
                    raise GuidedRegistrationConflict(
                        "recipe mapping schema and selected scenario do not match"
                    )
                step_fields = step.get("fields")
                if step_fields is not None and dict(step_fields) != mapping.to_dict()["mappings"]:
                    raise GuidedRegistrationConflict(
                        "recipe mapping fields and confirmed mapping plan do not match"
                    )
        # preparation_preview reads only the persisted, bounded inspection
        # sample.  It also validates custom Advanced recipes losslessly.
        preparation = self.preparation_preview(
            inspection_id, {"preparation_plan": preparation_payload}
        )
        session = self.db.get_dataset_import(import_id)
        if session is None:
            raise ValueError("inspection import no longer exists")
        return {
            "inspection_id": inspection.id,
            "import_id": session.id,
            "name": str(payload.get("name") or session.display_name or scenario.label),
            "description": payload.get("description"),
            "scenario_revision_id": scenario.revision_id,
            "canonical_schema": scenario.canonical_schema,
            "modality": scenario.modality,
            "mapping_plan": mapping.to_dict(),
            "preparation_plan": preparation,
            "preparation_plan_input": copy.deepcopy(dict(preparation_payload)),
            "source_fingerprint": inspection.source_fingerprint,
            "source_size_bytes": inspection.size_bytes,
            "source_kind": session.source_kind,
            "capacity_override_reason": (
                str(payload.get("capacity_override_reason") or "").strip() or None
            ),
        }

    def execute_registration(
        self,
        inspection_id: str,
        payload: Mapping[str, Any],
        *,
        dataset_lab: Any,
        dataset_id: str,
        source_id: str,
    ) -> Dict[str, Any]:
        """Publish and catalog one idempotent guided registration attempt.

        Deterministic dataset/source identities are minted before enqueueing.
        Consequently a worker crash after either catalog write can be retried
        without producing a second logical dataset or source occurrence.
        """

        inspection = self.db.get_dataset_source_inspection(inspection_id)
        if inspection is None:
            raise KeyError(inspection_id)
        request = self.registration_request(inspection_id, payload)
        session = self.db.get_dataset_import(request["import_id"])
        if session is None:
            raise KeyError(request["import_id"])

        existing_dataset = self.db.get_dataset(dataset_id)
        existing_source = self.db.get_dataset_source(source_id)
        if existing_dataset is not None and existing_source is not None:
            if existing_source.dataset_id != existing_dataset.id:
                raise ValueError("registration source identity belongs to another dataset")
            self.mark_registered(
                session.id, dataset_id=existing_dataset.id, source_id=existing_source.id
            )
            return {
                "dataset": {
                    **existing_dataset.to_dict(),
                    "sources": [existing_source.to_dict()],
                },
                "source": existing_source.to_dict(),
                "preparation_plan": self.registration_request(
                    inspection_id, payload
                )["preparation_plan"],
                "reused": True,
            }

        registration = self.registration_payload(inspection_id, payload)
        dataset = existing_dataset
        if dataset is None:
            dataset = self.db.create_dataset(
                dataset_id=dataset_id,
                name=registration["name"],
                description=registration.get("description"),
                modality=registration["modality"],
                canonical_schema=registration["canonical_schema"],
            )
        elif (
            dataset.modality != registration["modality"]
            or dataset.canonical_schema != registration["canonical_schema"]
        ):
            raise ValueError("registration dataset identity has incompatible metadata")

        source_payload = dict(registration["source"])
        engine_source = dataset_lab.register_source(
            {
                "kind": "local",
                "path": source_payload["uri"],
                "canonical_kind": registration["canonical_schema"],
                "modality": registration["modality"],
                "field_mapping": registration["field_mapping"],
            },
            dataset_id=dataset.id,
            name=dataset.name,
            source_id=source_id,
        )
        engine_data = (
            dict(engine_source.to_dict())
            if callable(getattr(engine_source, "to_dict", None))
            else dict(vars(engine_source))
        )
        source = self.db.get_dataset_source(source_id)
        if source is None:
            source = self.db.create_dataset_source(
                dataset_id=dataset.id,
                source_id=source_id,
                kind="local",
                uri=str(source_payload["uri"]),
                fingerprint=str(engine_data["fingerprint"]),
                size_bytes=int(engine_data.get("size_bytes") or 0),
                row_count=int(engine_data.get("row_count") or 0),
                metadata={
                    "file_count": int(engine_data.get("file_count") or 0),
                    "assets": copy.deepcopy(
                        engine_data.get("asset_fingerprints") or []
                    ),
                    "engine": engine_data,
                    "guided_own_data": {
                        "format_version": 1,
                        "scenario_revision_id": registration[
                            "scenario_revision_id"
                        ],
                        "import_id": registration["import_id"],
                        "inspection_id": inspection_id,
                        "source_fingerprint": registration[
                            "source_fingerprint"
                        ],
                        "field_mapping": registration["field_mapping"],
                        "preparation_plan": registration["preparation_plan"],
                        "corpus_extraction": copy.deepcopy(
                            registration.get("extraction") or {}
                        ),
                    },
                },
            )
        self.mark_registered(session.id, dataset_id=dataset.id, source_id=source.id)
        return {
            "dataset": {**dataset.to_dict(), "sources": [source.to_dict()]},
            "source": source.to_dict(),
            "preparation_plan": registration["preparation_plan"],
            "reused": False,
        }

    def mark_registered(self, import_id: str, *, dataset_id: str, source_id: str) -> None:
        self.db.update_dataset_import(
            import_id,
            status="published",
            published_dataset_id=dataset_id,
            published_source_id=source_id,
            completed_at=_now(),
        )

    def cleanup(self, *, approved: bool = False) -> Dict[str, Any]:
        return self.imports.cleanup_expired(apply=approved)


def json_copy(value: Any) -> Any:
    return copy.deepcopy(value)


__all__ = ["GuidedOwnDataService"]
