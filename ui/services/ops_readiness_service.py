"""
UI service for cross-module ops readiness (non-coding scope).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from halo_forge.ops_module_readiness import (
    DEFAULT_OPS_READINESS_REPORT_FILE,
    OPS_MODULES,
    OPS_READINESS_STALE_AFTER_SECONDS,
    OpsReadinessReport,
    apply_staleness_policy,
    compute_ops_module_readiness,
    default_output_map,
    load_ops_readiness_report,
)


class OpsReadinessService:
    """Loads and computes cross-module ops readiness for UI pages."""

    def __init__(
        self,
        *,
        base_path: Optional[Path] = None,
        report_path: Path = DEFAULT_OPS_READINESS_REPORT_FILE,
        burnin_report_path: Path = Path("results/readiness/ops_dataset_burnin.v1.json"),
    ):
        self.base_path = base_path or Path.cwd()
        self.report_path = report_path
        self.burnin_report_path = burnin_report_path
        self._cache: Optional[OpsReadinessReport] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 30
        self._burnin_cache = None
        self._burnin_cache_time: Optional[datetime] = None

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
    ) -> OpsReadinessReport:
        """Compute readiness directly from local contracts/artifacts."""
        merged_output_map = self._default_output_map()
        if output_map:
            for key, value in output_map.items():
                if key in OPS_MODULES and value:
                    merged_output_map[key] = value

        report = compute_ops_module_readiness(
            output_map=merged_output_map,
            seed=seed,
            source="ui_live_compute",
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
                return self.compute_live_readiness(output_map=output_map, seed=seed)
            return report
        except Exception:
            return self.compute_live_readiness(output_map=output_map, seed=seed)

    def _default_output_map(self) -> Dict[str, str]:
        defaults = default_output_map()
        return {
            module: str(Path(path))
            for module, path in defaults.items()
        }

    def _has_usable_entries(self, report: OpsReadinessReport) -> bool:
        for module in OPS_MODULES:
            if report.modules[module].status in {"pass", "warn"}:
                return True
        return False

    def _resolve_report_path(self) -> Path:
        if self.report_path.is_absolute():
            return self.report_path
        return self.base_path / self.report_path

    def _resolve_burnin_report_path(self) -> Path:
        if self.burnin_report_path.is_absolute():
            return self.burnin_report_path
        return self.base_path / self.burnin_report_path

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


_ops_readiness_service: Optional[OpsReadinessService] = None


def get_ops_readiness_service() -> OpsReadinessService:
    """Get singleton ops readiness service."""
    global _ops_readiness_service
    if _ops_readiness_service is None:
        _ops_readiness_service = OpsReadinessService()
    return _ops_readiness_service
