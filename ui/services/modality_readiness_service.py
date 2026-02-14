"""
UI readiness service for non-code modalities (vlm/audio/reasoning/agentic).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from halo_forge.modality_research import NON_CODE_MODALITIES, validate_modality_training_artifacts
from halo_forge.modality_readiness import (
    DEFAULT_READINESS_REPORT_FILE,
    READINESS_STALE_AFTER_SECONDS,
    ReadinessReport,
    apply_staleness_policy,
    build_readiness_report_from_validations,
    load_readiness_report,
)


class ModalityReadinessService:
    """Loads and computes non-code modality readiness state for UI pages."""

    def __init__(
        self,
        *,
        base_path: Optional[Path] = None,
        report_path: Path = DEFAULT_READINESS_REPORT_FILE,
    ):
        self.base_path = base_path or Path.cwd()
        self.report_path = report_path
        self._cache: Optional[ReadinessReport] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 30

    def load_readiness_report(self, force_refresh: bool = False) -> ReadinessReport:
        """Load canonical readiness report from disk."""
        if not force_refresh and self._cache and self._cache_time:
            age = (datetime.now() - self._cache_time).total_seconds()
            if age < self._cache_ttl_seconds:
                return self._cache

        report_file = self._resolve_report_path()
        if not report_file.exists():
            raise FileNotFoundError(f"Readiness report not found: {report_file}")

        report = load_readiness_report(report_file)
        report = apply_staleness_policy(report, stale_after_seconds=READINESS_STALE_AFTER_SECONDS)
        self._cache = report
        self._cache_time = datetime.now()
        return report

    def compute_live_readiness(
        self,
        output_map: Dict[str, str],
        seed: int = 42,
    ) -> ReadinessReport:
        """Compute readiness directly from local artifacts."""
        validations = []
        for modality in NON_CODE_MODALITIES:
            output_dir = output_map.get(modality) or str(
                self.base_path / "models" / "phase7d" / f"{modality}_phase7d"
            )
            validations.append(
                validate_modality_training_artifacts(
                    modality=modality,
                    output_dir=output_dir,
                    expected_seed=seed,
                )
            )
        report = build_readiness_report_from_validations(
            validations,
            seed=seed,
            source="ui_live_compute",
        )
        report = apply_staleness_policy(report, stale_after_seconds=READINESS_STALE_AFTER_SECONDS)
        self._cache = report
        self._cache_time = datetime.now()
        return report

    def get_effective_readiness(
        self,
        output_map: Optional[Dict[str, str]] = None,
        seed: int = 42,
        force_refresh: bool = False,
    ) -> ReadinessReport:
        """
        Return effective readiness for UI.

        Preferred source: canonical report file.
        Fallback source: live artifact compute.
        """
        resolved_output_map = output_map or self._default_output_map()
        try:
            report = self.load_readiness_report(force_refresh=force_refresh)
            if report.stale and not self._has_usable_entries(report):
                return self.compute_live_readiness(resolved_output_map, seed=seed)
            return report
        except Exception:
            return self.compute_live_readiness(resolved_output_map, seed=seed)

    def _default_output_map(self) -> Dict[str, str]:
        return {
            modality: str(self.base_path / "models" / "phase7d" / f"{modality}_phase7d")
            for modality in NON_CODE_MODALITIES
        }

    def _has_usable_entries(self, report: ReadinessReport) -> bool:
        for modality in NON_CODE_MODALITIES:
            status = report.modalities[modality].status
            if status in {"pass", "warn"}:
                return True
        return False

    def _resolve_report_path(self) -> Path:
        if self.report_path.is_absolute():
            return self.report_path
        return self.base_path / self.report_path


_readiness_service: Optional[ModalityReadinessService] = None


def get_modality_readiness_service() -> ModalityReadinessService:
    """Get singleton modality readiness service."""
    global _readiness_service
    if _readiness_service is None:
        _readiness_service = ModalityReadinessService()
    return _readiness_service
