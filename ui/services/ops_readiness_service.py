"""
UI service for cross-module ops readiness (non-coding scope).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from halo_forge.diagnostics import derive_issue_metadata
from halo_forge.ops_module_readiness import (
    DEFAULT_OPS_READINESS_REPORT_FILE,
    OPS_MODULES,
    OPS_READINESS_STALE_AFTER_SECONDS,
    OpsReadinessReport,
    OpsModuleReadiness,
    apply_staleness_policy,
    build_ops_readiness_report,
    compute_ops_module_readiness,
    default_output_map,
    load_ops_readiness_report,
    validate_ops_module,
    write_ops_readiness_report,
)
from halo_forge.all_module_readiness import (
    ALL_MODULES,
    DEFAULT_ALL_MODULE_READINESS_REPORT_FILE,
    AllModuleReadiness,
    AllModuleReadinessReport,
    apply_staleness_policy as apply_all_module_staleness_policy,
    build_all_module_readiness_report,
    compute_all_module_readiness,
    default_output_map as default_all_module_output_map,
    load_all_module_readiness_report,
    validate_all_module,
    write_all_module_readiness_report,
)
from halo_forge.all_module_qualification import (
    DEFAULT_ALL_MODULE_QUALIFICATION_REPORT_FILE,
    load_all_module_qualification_report,
)
from halo_forge.all_module_bootstrap import (
    DEFAULT_ALL_MODULE_BOOTSTRAP_REPORT_FILE,
    load_all_module_bootstrap_report,
)
from .results_service import get_results_service
from .launch_context import find_latest_launch_context
from .qualification_service import get_qualification_service
from .bootstrap_service import get_bootstrap_service
from ui.state import state as app_state


class OpsReadinessService:
    """Loads and computes cross-module ops readiness for UI pages."""

    def __init__(
        self,
        *,
        base_path: Optional[Path] = None,
        report_path: Path = DEFAULT_OPS_READINESS_REPORT_FILE,
        all_module_report_path: Path = DEFAULT_ALL_MODULE_READINESS_REPORT_FILE,
        qualification_report_path: Path = DEFAULT_ALL_MODULE_QUALIFICATION_REPORT_FILE,
        bootstrap_report_path: Path = DEFAULT_ALL_MODULE_BOOTSTRAP_REPORT_FILE,
        burnin_report_path: Path = Path("results/readiness/ops_dataset_burnin.v1.json"),
        walkthrough_report_path: Path = Path(
            ".internal_docs/research_testing/walkthroughs/reports/all_module_e2e_walkthrough_report.v1.json"
        ),
    ):
        self.base_path = base_path or Path.cwd()
        self.report_path = report_path
        self.all_module_report_path = all_module_report_path
        self.qualification_report_path = qualification_report_path
        self.bootstrap_report_path = bootstrap_report_path
        self.burnin_report_path = burnin_report_path
        self.walkthrough_report_path = walkthrough_report_path
        self._cache: Optional[OpsReadinessReport] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 30
        self._all_module_cache: Optional[AllModuleReadinessReport] = None
        self._all_module_cache_time: Optional[datetime] = None
        self._burnin_cache = None
        self._burnin_cache_time: Optional[datetime] = None
        self._qualification_cache = None
        self._qualification_cache_time: Optional[datetime] = None
        self._bootstrap_cache = None
        self._bootstrap_cache_time: Optional[datetime] = None
        self._walkthrough_cache = None
        self._walkthrough_cache_time: Optional[datetime] = None

    def resolve_effective_output_map(
        self,
        *,
        include_all_modules: bool = False,
        force_refresh: bool = False,
    ) -> Dict[str, str]:
        """
        Resolve output roots using newest evidence first, then static defaults.

        Priority:
        1. Bootstrap report evidence roots (when available)
        2. Recent parsed results/training/utility artifacts
        3. Latest launch_context.json discovery
        4. Static default output map
        """
        if include_all_modules:
            resolved = self._default_all_module_output_map()
            valid_keys = set(ALL_MODULES)
        else:
            resolved = self._default_output_map()
            valid_keys = set(OPS_MODULES)

        # Prefer explicit bootstrap evidence roots over static defaults.
        try:
            bootstrap_report = self.load_bootstrap_report(force_refresh=force_refresh)
            for module, entry in bootstrap_report.modules.items():
                root = str(entry.evidence_root or "").strip()
                if not root:
                    continue
                if module in valid_keys:
                    resolved[module] = root
                elif module == "benchmark_non_code" and "benchmark" in valid_keys:
                    resolved["benchmark"] = root
        except Exception:
            pass

        try:
            results_service = get_results_service()
            artifact_roots = results_service.get_latest_artifact_roots()
            training_runs = results_service.list_training_runs(force_refresh=force_refresh)
            benchmark_results = results_service.list_results(force_refresh=force_refresh)
        except Exception:
            artifact_roots = {}
            training_runs = []
            benchmark_results = []

        for module, path in artifact_roots.items():
            if module in valid_keys and path:
                resolved[module] = str(path)

        for run in training_runs:
            module = str(run.modality or "").strip().lower()
            if module in valid_keys:
                resolved[module] = str(run.output_dir)

        # Prefer latest benchmark file parent discovered by domain.
        for result in benchmark_results:
            if not result.file_path:
                continue
            parent = str(result.file_path.parent)
            domain = str(result.domain or "").strip().lower()
            if include_all_modules:
                if domain == "code" and "benchmark_code" in valid_keys:
                    resolved["benchmark_code"] = parent
                if domain in {"vlm", "audio", "reasoning", "agentic"} and "benchmark_non_code" in valid_keys:
                    resolved["benchmark_non_code"] = parent
            if domain in {"vlm", "audio", "reasoning", "agentic"} and "benchmark" in valid_keys:
                resolved["benchmark"] = parent

        model_root = self.base_path / "models"
        results_root = self.base_path / "results"
        repo_root = self.base_path

        launch_lookup = {
            "sft": ("training", model_root),
            "raft": ("training", model_root),
            "vlm": ("training", model_root),
            "audio": ("training", model_root),
            "reasoning": ("training", model_root),
            "agentic": ("training", model_root),
            "inference": ("inference", model_root),
            "benchmark": ("benchmark", results_root),
        }

        for module, (service, root) in launch_lookup.items():
            if module not in valid_keys:
                continue
            context_path = find_latest_launch_context(
                root=root,
                job_type=module,
                service=service,
            )
            if context_path:
                resolved[module] = str(context_path.parent)

        if include_all_modules and "benchmark_code" in valid_keys:
            # Fallback: if only benchmark contexts exist and no code-domain parse yet.
            context_path = find_latest_launch_context(
                root=results_root,
                job_type="benchmark",
                service="benchmark",
            )
            if context_path and not Path(resolved["benchmark_code"]).exists():
                resolved["benchmark_code"] = str(context_path.parent)

        if "ui_ops" in valid_keys:
            resolved["ui_ops"] = str(repo_root)

        return resolved

    def load_readiness_report(self, force_refresh: bool = False) -> OpsReadinessReport:
        """Load canonical ops readiness report from disk."""
        if not force_refresh and self._cache and self._cache_time:
            age = (datetime.now() - self._cache_time).total_seconds()
            if age < self._cache_ttl_seconds:
                return self._cache

        report_file = self._resolve_report_path()
        if not report_file.exists():
            raise FileNotFoundError(f"Ops readiness report not found: {report_file}")

        report = load_ops_readiness_report(report_file)
        report = apply_staleness_policy(
            report,
            stale_after_seconds=OPS_READINESS_STALE_AFTER_SECONDS,
        )
        self._cache = report
        self._cache_time = datetime.now()
        return report

    def compute_live_readiness(
        self,
        output_map: Optional[Dict[str, str]] = None,
        seed: int = 42,
        force_refresh: bool = False,
    ) -> OpsReadinessReport:
        """Compute readiness directly from local contracts/artifacts."""
        merged_output_map = self.resolve_effective_output_map(
            include_all_modules=False,
            force_refresh=force_refresh,
        )
        if output_map:
            merged_output_map = self._apply_output_overrides(
                merged_output_map,
                output_map,
                valid_keys=set(OPS_MODULES),
            )

        report = compute_ops_module_readiness(
            output_map=merged_output_map,
            seed=seed,
            source="ui_live_compute",
            require_artifacts=False,
        )
        report = apply_staleness_policy(
            report,
            stale_after_seconds=OPS_READINESS_STALE_AFTER_SECONDS,
        )
        self._cache = report
        self._cache_time = datetime.now()
        return report

    def get_effective_readiness(
        self,
        output_map: Optional[Dict[str, str]] = None,
        seed: int = 42,
        force_refresh: bool = False,
    ) -> OpsReadinessReport:
        """
        Return effective readiness for UI.

        Preferred source: persisted canonical report.
        Fallback source: live contract compute.
        """
        try:
            report = self.load_readiness_report(force_refresh=force_refresh)
            if report.stale and not self._has_usable_entries(report):
                return self.compute_live_readiness(
                    output_map=output_map,
                    seed=seed,
                    force_refresh=force_refresh,
                )
            return report
        except Exception:
            return self.compute_live_readiness(
                output_map=output_map,
                seed=seed,
                force_refresh=force_refresh,
            )

    def _default_output_map(self) -> Dict[str, str]:
        defaults = default_output_map()
        return {
            module: str(Path(path))
            for module, path in defaults.items()
        }

    def _apply_output_overrides(
        self,
        base_map: Dict[str, str],
        overrides: Dict[str, str],
        *,
        valid_keys: set[str],
    ) -> Dict[str, str]:
        merged = dict(base_map)
        for key, value in overrides.items():
            if key in valid_keys and value:
                merged[key] = str(value)
        return merged

    def _has_usable_entries(self, report: OpsReadinessReport) -> bool:
        for module in OPS_MODULES:
            if report.modules[module].status in {"pass", "warn"}:
                return True
        return False

    def _resolve_report_path(self) -> Path:
        if self.report_path.is_absolute():
            return self.report_path
        return self.base_path / self.report_path

    def _resolve_all_module_report_path(self) -> Path:
        if self.all_module_report_path.is_absolute():
            return self.all_module_report_path
        return self.base_path / self.all_module_report_path

    def _resolve_burnin_report_path(self) -> Path:
        if self.burnin_report_path.is_absolute():
            return self.burnin_report_path
        return self.base_path / self.burnin_report_path

    def _resolve_qualification_report_path(self) -> Path:
        if self.qualification_report_path.is_absolute():
            return self.qualification_report_path
        return self.base_path / self.qualification_report_path

    def _resolve_bootstrap_report_path(self) -> Path:
        if self.bootstrap_report_path.is_absolute():
            return self.bootstrap_report_path
        return self.base_path / self.bootstrap_report_path

    def _resolve_walkthrough_report_path(self) -> Path:
        if self.walkthrough_report_path.is_absolute():
            return self.walkthrough_report_path
        return self.base_path / self.walkthrough_report_path

    def load_all_module_readiness_report(
        self,
        force_refresh: bool = False,
    ) -> AllModuleReadinessReport:
        """Load canonical all-module readiness report from disk."""
        if not force_refresh and self._all_module_cache and self._all_module_cache_time:
            age = (datetime.now() - self._all_module_cache_time).total_seconds()
            if age < self._cache_ttl_seconds:
                return self._all_module_cache

        report_file = self._resolve_all_module_report_path()
        if not report_file.exists():
            raise FileNotFoundError(f"All-module readiness report not found: {report_file}")

        report = load_all_module_readiness_report(report_file)
        self._normalize_all_module_report(report)
        report = apply_all_module_staleness_policy(
            report,
            stale_after_seconds=OPS_READINESS_STALE_AFTER_SECONDS,
        )
        self._normalize_all_module_report(report)
        self._all_module_cache = report
        self._all_module_cache_time = datetime.now()
        return report

    def compute_live_all_module_readiness(
        self,
        output_map: Optional[Dict[str, str]] = None,
        seed: int = 42,
        force_refresh: bool = False,
    ) -> AllModuleReadinessReport:
        """Compute all-module readiness directly from local contracts/artifacts."""
        merged_output_map = self.resolve_effective_output_map(
            include_all_modules=True,
            force_refresh=force_refresh,
        )
        if output_map:
            merged_output_map = self._apply_output_overrides(
                merged_output_map,
                output_map,
                valid_keys=set(ALL_MODULES),
            )

        report = compute_all_module_readiness(
            output_map=merged_output_map,
            seed=seed,
            source="ui_live_compute",
            require_artifacts=False,
        )
        self._normalize_all_module_report(report)
        report = apply_all_module_staleness_policy(
            report,
            stale_after_seconds=OPS_READINESS_STALE_AFTER_SECONDS,
        )
        self._normalize_all_module_report(report)
        self._all_module_cache = report
        self._all_module_cache_time = datetime.now()
        return report

    def get_effective_all_module_readiness(
        self,
        output_map: Optional[Dict[str, str]] = None,
        seed: int = 42,
        force_refresh: bool = False,
    ) -> AllModuleReadinessReport:
        """
        Return effective all-module readiness for UI.

        Preferred source: persisted canonical report.
        Fallback source: live contract compute.
        """
        try:
            report = self.load_all_module_readiness_report(force_refresh=force_refresh)
            if report.stale and not self._has_usable_all_module_entries(report):
                return self.compute_live_all_module_readiness(
                    output_map=output_map,
                    seed=seed,
                    force_refresh=force_refresh,
                )
            return report
        except Exception:
            return self.compute_live_all_module_readiness(
                output_map=output_map,
                seed=seed,
                force_refresh=force_refresh,
            )

    def _default_all_module_output_map(self) -> Dict[str, str]:
        defaults = default_all_module_output_map()
        return {
            module: str(Path(path))
            for module, path in defaults.items()
        }

    def run_contract_probe(
        self,
        *,
        module: str,
        seed: int = 42,
        include_all_modules: bool = True,
    ) -> tuple[bool, str]:
        """
        Recompute readiness for one module and persist updated report.

        Returns:
            (success, message)
        """
        module_key = str(module or "").strip().lower()
        if include_all_modules:
            if module_key not in ALL_MODULES:
                return False, f"Unsupported module: {module_key}"
            mapping = self.resolve_effective_output_map(include_all_modules=True, force_refresh=True)
            entry = validate_all_module(
                module=module_key,
                output_dir=Path(mapping[module_key]),
                seed=seed,
                require_artifacts=False,
            )
            existing: Dict[str, AllModuleReadiness] = {}
            try:
                report = self.load_all_module_readiness_report(force_refresh=True)
                existing = dict(report.modules)
            except Exception:
                existing = {}
            existing[module_key] = entry
            updated = build_all_module_readiness_report(
                module_entries=existing,
                seed=seed,
                source="ui_live_compute",
            )
            self._normalize_all_module_report(updated)
            path = self._resolve_all_module_report_path()
            write_all_module_readiness_report(path, updated)
            self._all_module_cache = apply_all_module_staleness_policy(
                updated,
                stale_after_seconds=OPS_READINESS_STALE_AFTER_SECONDS,
            )
            self._normalize_all_module_report(self._all_module_cache)
            self._all_module_cache_time = datetime.now()
            return True, f"Wrote contract probe readiness for {module_key} to {path}"

        if module_key not in OPS_MODULES:
            return False, f"Unsupported module: {module_key}"
        mapping = self.resolve_effective_output_map(include_all_modules=False, force_refresh=True)
        entry = validate_ops_module(
            module=module_key,
            output_dir=Path(mapping[module_key]),
            seed=seed,
            require_artifacts=False,
        )
        existing_ops: Dict[str, OpsModuleReadiness] = {}
        try:
            report = self.load_readiness_report(force_refresh=True)
            existing_ops = dict(report.modules)
        except Exception:
            existing_ops = {}
        existing_ops[module_key] = entry
        updated = build_ops_readiness_report(
            module_entries=existing_ops,
            seed=seed,
            source="ui_live_compute",
        )
        path = self._resolve_report_path()
        write_ops_readiness_report(path, updated)
        self._cache = apply_staleness_policy(
            updated,
            stale_after_seconds=OPS_READINESS_STALE_AFTER_SECONDS,
        )
        self._cache_time = datetime.now()
        return True, f"Wrote contract probe readiness for {module_key} to {path}"

    def _has_usable_all_module_entries(self, report: AllModuleReadinessReport) -> bool:
        for module in ALL_MODULES:
            if report.modules[module].status in {"pass", "warn"}:
                return True
        return False

    def _normalize_all_module_report(self, report: AllModuleReadinessReport) -> None:
        """Backfill additive diagnostic fields for older report payloads."""
        for module, entry in report.modules.items():
            metadata = derive_issue_metadata(
                module=module,
                issue_class=str(getattr(entry, "issue_class", "none") or "none"),
                launch_blocked=bool(getattr(entry, "launch_blocked", False)),
                errors=list(getattr(entry, "errors", []) or []),
                warnings=list(getattr(entry, "warnings", []) or []),
                action_hint=str(getattr(entry, "action_hint", "") or ""),
                evidence=dict(getattr(entry, "evidence", {}) or {}),
                last_output_dir=str(getattr(entry, "last_output_dir", "") or ""),
            )
            entry.issue_code = str(metadata["issue_code"])
            entry.issue_scope = str(metadata["issue_scope"])
            entry.severity = str(metadata["severity"])
            entry.what_is_missing = [str(v) for v in metadata["what_is_missing"]]
            entry.fix_now = str(metadata["fix_now"])
            entry.fix_options = [str(v) for v in metadata["fix_options"]]

    def load_burnin_report(self, force_refresh: bool = False):
        """Load dataset burn-in report if present and valid."""
        from halo_forge.ops_dataset_burnin import load_ops_burnin_report

        if not force_refresh and self._burnin_cache and self._burnin_cache_time:
            age = (datetime.now() - self._burnin_cache_time).total_seconds()
            if age < self._cache_ttl_seconds:
                return self._burnin_cache

        report_file = self._resolve_burnin_report_path()
        if not report_file.exists():
            raise FileNotFoundError(f"Ops dataset burn-in report not found: {report_file}")

        report = load_ops_burnin_report(report_file)
        self._burnin_cache = report
        self._burnin_cache_time = datetime.now()
        return report

    def get_burnin_provenance(self, force_refresh: bool = False) -> Dict[str, object]:
        """
        Return optional burn-in provenance metadata for UI surfaces.

        Keys:
        - burnin_report_present: bool
        - burnin_generated_at: Optional[str]
        - burnin_status: Optional[str]
        - burnin_source: Optional[str]
        """
        try:
            report = self.load_burnin_report(force_refresh=force_refresh)
        except Exception:
            return {
                "burnin_report_present": False,
                "burnin_generated_at": None,
                "burnin_status": None,
                "burnin_source": None,
            }

        statuses = [entry.status for entry in report.modules.values()]
        overall_status = "pass"
        if any(status == "fail" for status in statuses):
            overall_status = "fail"
        elif any(status == "warn" for status in statuses):
            overall_status = "warn"

        return {
            "burnin_report_present": True,
            "burnin_generated_at": report.generated_at,
            "burnin_status": overall_status,
            "burnin_source": report.source,
        }

    def load_qualification_report(self, force_refresh: bool = False):
        """Load all-module qualification report when present and schema-valid."""
        if not force_refresh and self._qualification_cache and self._qualification_cache_time:
            age = (datetime.now() - self._qualification_cache_time).total_seconds()
            if age < self._cache_ttl_seconds:
                return self._qualification_cache

        report_file = self._resolve_qualification_report_path()
        if not report_file.exists():
            raise FileNotFoundError(f"All-module qualification report not found: {report_file}")

        report = load_all_module_qualification_report(report_file)
        self._normalize_qualification_report(report)
        self._qualification_cache = report
        self._qualification_cache_time = datetime.now()
        return report

    def get_qualification_provenance(self, force_refresh: bool = False) -> Dict[str, object]:
        """
        Return optional qualification report provenance for dashboard/research surfaces.

        Keys:
        - qualification_report_present: bool
        - qualification_generated_at: Optional[str]
        - qualification_status: Optional[str]
        - qualification_source: Optional[str]
        - qualification_profile: Optional[str]
        - qualification_report_path: Optional[str]
        """
        try:
            report = self.load_qualification_report(force_refresh=force_refresh)
        except Exception:
            return {
                "qualification_report_present": False,
                "qualification_generated_at": None,
                "qualification_status": None,
                "qualification_source": None,
                "qualification_profile": None,
                "qualification_report_path": None,
            }

        statuses = [entry.status for entry in report.modules.values()]
        overall_status = "pass"
        if any(status == "fail" for status in statuses):
            overall_status = "fail"
        elif any(status == "warn" for status in statuses):
            overall_status = "warn"

        return {
            "qualification_report_present": True,
            "qualification_generated_at": report.generated_at,
            "qualification_status": overall_status,
            "qualification_source": report.source,
            "qualification_profile": report.profile,
            "qualification_report_path": str(self._resolve_qualification_report_path()),
        }

    def load_bootstrap_report(self, force_refresh: bool = False):
        """Load all-module bootstrap report when present and schema-valid."""
        if not force_refresh and self._bootstrap_cache and self._bootstrap_cache_time:
            age = (datetime.now() - self._bootstrap_cache_time).total_seconds()
            if age < self._cache_ttl_seconds:
                return self._bootstrap_cache

        report_file = self._resolve_bootstrap_report_path()
        if not report_file.exists():
            raise FileNotFoundError(f"All-module bootstrap report not found: {report_file}")

        report = load_all_module_bootstrap_report(report_file)
        self._bootstrap_cache = report
        self._bootstrap_cache_time = datetime.now()
        return report

    def get_bootstrap_provenance(self, force_refresh: bool = False) -> Dict[str, object]:
        """
        Return optional bootstrap report provenance for dashboard/research surfaces.

        Keys:
        - bootstrap_report_present: bool
        - bootstrap_generated_at: Optional[str]
        - bootstrap_status: Optional[str]
        - bootstrap_source: Optional[str]
        - bootstrap_profile: Optional[str]
        - bootstrap_report_path: Optional[str]
        """
        try:
            report = self.load_bootstrap_report(force_refresh=force_refresh)
        except Exception:
            return {
                "bootstrap_report_present": False,
                "bootstrap_generated_at": None,
                "bootstrap_status": None,
                "bootstrap_source": None,
                "bootstrap_profile": None,
                "bootstrap_report_path": None,
            }

        statuses = [entry.status for entry in report.modules.values()]
        overall_status = "pass"
        if any(status == "fail" for status in statuses):
            overall_status = "fail"
        elif any(status == "warn" for status in statuses):
            overall_status = "warn"

        return {
            "bootstrap_report_present": True,
            "bootstrap_generated_at": report.generated_at,
            "bootstrap_status": overall_status,
            "bootstrap_source": report.source,
            "bootstrap_profile": report.profile,
            "bootstrap_report_path": str(self._resolve_bootstrap_report_path()),
        }

    async def run_qualification_probe(
        self,
        *,
        qualification_profile: str = "contract-v1",
        strict: bool = False,
        module_filters: Optional[list[str]] = None,
        fixture_pack: str = "",
    ) -> tuple[bool, str, Optional[str]]:
        """
        Launch a tracked qualification probe job from UI.

        Returns:
            (success, message, job_id)
        """
        try:
            service = get_qualification_service(app_state)
            job_id = await service.launch_qualification(
                qualification_profile=qualification_profile,
                strict=strict,
                module_filters=module_filters or [],
                fixture_pack=fixture_pack,
                source_ui_page="/research-hub",
            )
            self._qualification_cache = None
            self._qualification_cache_time = None
            return True, f"Started qualification probe job {job_id}", job_id
        except Exception as exc:
            return False, f"Qualification probe failed: {exc}", None

    async def run_bootstrap_probe(
        self,
        *,
        bootstrap_profile: str = "contract-v1",
        strict: bool = False,
        modules: Optional[list[str]] = None,
        output_root: str = "results/readiness/bootstrap",
    ) -> tuple[bool, str, Optional[str]]:
        """
        Launch a tracked bootstrap probe job from UI.

        Returns:
            (success, message, job_id)
        """
        try:
            service = get_bootstrap_service(app_state)
            job_id = await service.launch_bootstrap(
                bootstrap_profile=bootstrap_profile,
                strict=strict,
                module_filters=modules or [],
                output_root=output_root,
                source_ui_page="/research-hub",
            )
            self._bootstrap_cache = None
            self._bootstrap_cache_time = None
            self._all_module_cache = None
            self._all_module_cache_time = None
            return True, f"Started bootstrap probe job {job_id}", job_id
        except Exception as exc:
            return False, f"Bootstrap probe failed: {exc}", None

    def load_walkthrough_report(self, force_refresh: bool = False):
        """Load internal all-module walkthrough report if present and valid."""
        from halo_forge.all_module_walkthroughs import load_walkthrough_report

        if not force_refresh and self._walkthrough_cache and self._walkthrough_cache_time:
            age = (datetime.now() - self._walkthrough_cache_time).total_seconds()
            if age < self._cache_ttl_seconds:
                return self._walkthrough_cache

        report_file = self._resolve_walkthrough_report_path()
        if not report_file.exists():
            raise FileNotFoundError(f"Walkthrough report not found: {report_file}")

        report = load_walkthrough_report(report_file)
        self._walkthrough_cache = report
        self._walkthrough_cache_time = datetime.now()
        return report

    def get_walkthrough_provenance(self, force_refresh: bool = False) -> Dict[str, object]:
        """
        Return optional internal walkthrough report metadata for UI read-only surfacing.

        Keys:
        - walkthrough_report_present: bool
        - walkthrough_generated_at: Optional[str]
        - walkthrough_profile: Optional[str]
        - walkthrough_status_summary: Optional[dict[str, int]]
        """
        try:
            report = self.load_walkthrough_report(force_refresh=force_refresh)
        except Exception:
            return {
                "walkthrough_report_present": False,
                "walkthrough_generated_at": None,
                "walkthrough_profile": None,
                "walkthrough_status_summary": None,
            }

        summary = {"pass": 0, "warn": 0, "fail": 0}
        for entry in report.modules.values():
            status = str(entry.status).lower()
            if status in summary:
                summary[status] += 1
        return {
            "walkthrough_report_present": True,
            "walkthrough_generated_at": getattr(report, "generated_at", None),
            "walkthrough_profile": getattr(report, "profile", None),
            "walkthrough_status_summary": summary,
        }

    def _normalize_qualification_report(self, report) -> None:
        """Backfill additive issue metadata for qualification payload compatibility."""
        for module, entry in report.modules.items():
            metadata = derive_issue_metadata(
                module=module,
                issue_class="preflight_blocker"
                if bool(getattr(entry, "launch_blocked", False))
                else "evidence_gap",
                launch_blocked=bool(getattr(entry, "launch_blocked", False)),
                errors=list(getattr(entry, "errors", []) or []),
                warnings=list(getattr(entry, "warnings", []) or []),
                action_hint="",
                evidence=dict(getattr(entry, "evidence", {}) or {}),
                last_output_dir=str(
                    (getattr(entry, "evidence", {}) or {}).get("output_dir") or ""
                ),
            )
            entry.issue_code = str(metadata["issue_code"])
            entry.issue_scope = str(metadata["issue_scope"])
            entry.severity = str(metadata["severity"])
            entry.what_is_missing = [str(v) for v in metadata["what_is_missing"]]
            entry.fix_now = str(metadata["fix_now"])
            entry.fix_options = [str(v) for v in metadata["fix_options"]]


_ops_readiness_service: Optional[OpsReadinessService] = None


def get_ops_readiness_service() -> OpsReadinessService:
    """Get singleton ops readiness service."""
    global _ops_readiness_service
    if _ops_readiness_service is None:
        _ops_readiness_service = OpsReadinessService()
    return _ops_readiness_service
