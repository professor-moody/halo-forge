"""V17 cross-platform readiness, guided repair, and support services."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import tempfile
import uuid
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from halo_forge.data_lab.identity import deterministic_record_id, record_hash
from halo_forge.data_lab.models import get_field, set_field
from halo_forge.data_lab.sources import (
    SourceSpec,
    content_hash as dataset_content_hash,
    fingerprint_path as dataset_fingerprint_path,
    load_source,
)
from halo_forge.own_data.inspection import iter_source_records
from halo_forge.own_data.registry import TRAINING_SCENARIOS
from halo_forge.run_db.db import RunDatabase

from .models import (
    DatasetIssue,
    DatasetRepairAction,
    DatasetRepairPlanRevision,
    DatasetRepairPreview,
    DatasetRepairSession,
    DistributionCapability,
    ReleaseQualification,
    SetupRemediation,
    SupportBundle,
    SupportBundlePreview,
    WorkstationReadiness,
)


class ProductLabError(RuntimeError):
    """A V17 request cannot be completed without changing its inputs."""


_DEFAULT_SUPPORT_CATEGORIES = (
    "versions",
    "readiness",
    "scheduler",
    "logs",
    "integrity",
)
_SUPPORT_CATEGORIES = frozenset(_DEFAULT_SUPPORT_CATEGORIES)
_SECRET_KEYS = re.compile(
    r"(^|_)(api_?key|token|secret|password|authorization|cookie|credential|private_?key)($|_)",
    re.IGNORECASE,
)
_TOKEN_TEXT = re.compile(
    r"(?i)(bearer\s+)[A-Za-z0-9._~+\-/=]+|\b(?:hf|hfk|sk)[-_][A-Za-z0-9_-]{8,}\b"
)
_PATH_TEXT = re.compile(
    r"(?<![A-Za-z0-9:])(?:[A-Za-z]:\\[^\s\"']+|/(?:Users|home|var|tmp|opt|mnt|Volumes|private)/[^\s\"']+)"
)
_SENSITIVE_LOG_LINE = re.compile(
    r"(?i)\b(prompt|completion|response|transcript|caption|dataset[_ ]?(?:row|record|content)|media[_ ]?(?:bytes|content)|raw[_ ]?output)\b"
)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _loads(value: Any, default: Any) -> Any:
    try:
        return json.loads(value) if value else copy.deepcopy(default)
    except (TypeError, ValueError, json.JSONDecodeError):
        return copy.deepcopy(default)


def _bounded(limit: int, offset: int, maximum: int = 1000) -> tuple[int, int]:
    return max(1, min(int(limit), maximum)), max(0, int(offset))


def _source_path(value: str) -> Path:
    text = str(value or "").strip()
    if text.startswith("file://"):
        text = text[7:]
    return Path(text).expanduser().resolve()


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _repair_source_fingerprint(path: Path) -> str:
    """Use Dataset Lab's exact local-source identity for build compatibility."""

    try:
        return str(load_source(SourceSpec(kind="local", path=str(path))).fingerprint)
    except Exception:
        # A malformed source must still be inspectable. This is the same
        # Dataset Lab source envelope with no inferred assets; a repaired build
        # will only proceed once parsing is valid or malformed rows are handled
        # by a tolerant import adapter.
        return dataset_content_hash(
            {"source": dataset_fingerprint_path(path), "assets": []}
        )


def _socket_ready(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.08):
            return True
    except OSError:
        return False


def _platform_name() -> str:
    return {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(
        platform.system(), platform.system().lower() or "unknown"
    )


def _architecture() -> str:
    return {
        "arm64": "arm64",
        "aarch64": "arm64",
        "AMD64": "x86_64",
        "x86_64": "x86_64",
    }.get(platform.machine(), platform.machine().lower() or "unknown")


def _app_version() -> str:
    try:
        from halo_forge.version import DISPLAY_VERSION

        return str(DISPLAY_VERSION)
    except Exception:
        return "unknown"


def _active_backends() -> tuple[str, ...]:
    values: list[str] = []
    try:
        from halo_forge.backend import get_backend

        values.append(str(get_backend().name))
    except Exception:
        pass
    if not values:
        values.append("cpu")
    return tuple(dict.fromkeys(values))


def _distribution_capability() -> DistributionCapability:
    system = _platform_name()
    architecture = _architecture()
    package = {
        ("macos", "arm64"): "dmg",
        ("linux", "x86_64"): "appimage/deb",
        ("windows", "x86_64"): "nsis",
    }.get((system, architecture))
    supported = package is not None
    signature = str(os.environ.get("HALOFORGE_SIGNATURE_STATE") or "unsigned")
    desktop_status = "preview" if supported else "unavailable"
    if signature in {"signed", "signed_notarized"} and supported:
        desktop_status = "supported"
    reason = None
    if not supported:
        reason = f"No verified desktop package contract for {system} {architecture}."
    elif signature == "unsigned":
        reason = "The desktop package is an unsigned preview; browser and CLI remain supported."
    return DistributionCapability(
        platform=system,
        architecture=architecture,
        execution_surfaces=("browser", "cli", "desktop") if supported else ("browser", "cli"),
        desktop_package=package,
        desktop_status=desktop_status,
        signature_state=signature,
        runtime_version=_app_version(),
        supported_backends=_active_backends(),
        unavailable_reason=reason,
    )


class ProductLabService:
    """Transport-neutral V17 implementation shared by dashboard, API, and CLI."""

    def __init__(
        self,
        db: RunDatabase,
        *,
        root: str | Path | None = None,
        scheduler: Any = None,
    ) -> None:
        self.db = db
        self.root = Path(root or Path.home() / ".halo-forge").expanduser().resolve()
        self.scheduler = scheduler
        self.repair_root = self.root / "repairs"
        self.support_root = self.root / "support-bundles"

    @property
    def conn(self):
        return self.db._conn

    # ----- workstation readiness -------------------------------------

    def assess_readiness(self, *, persist: bool = True) -> WorkstationReadiness:
        capability = _distribution_capability()
        data_root = self.root
        output_root = self.root / "runs"
        managed_directories = (
            data_root,
            data_root / "datasets",
            output_root,
            self.repair_root,
            self.support_root,
        )
        managed_ready = all(path.is_dir() for path in managed_directories)
        parent = _nearest_existing_parent(data_root)
        writable = parent.exists() and os.access(parent, os.W_OK)
        try:
            disk = shutil.disk_usage(parent)
            free_bytes = int(disk.free)
            minimum_free = max(20 * 1024**3, int(disk.total * 0.10))
            disk_ready = free_bytes >= minimum_free
        except OSError:
            free_bytes = 0
            minimum_free = 20 * 1024**3
            disk_ready = False
        memory_total: Optional[int] = None
        memory_available: Optional[int] = None
        try:
            import psutil  # type: ignore

            memory = psutil.virtual_memory()
            memory_total = int(memory.total)
            memory_available = int(memory.available)
        except Exception:
            pass
        serving_ready = _socket_ready(8001)
        active_backends = list(capability.supported_backends)
        managed_family = next(
            (
                "rocm" if value.startswith("rocm") else "cuda"
                for value in active_backends
                if value.startswith("rocm") or value == "cuda"
            ),
            None,
        )
        runtime_capability = None
        if managed_family:
            try:
                from halo_forge.managed_runtime import ManagedRuntimeService

                runtime_capability = next(
                    (
                        value
                        for value in ManagedRuntimeService(
                            self.db,
                            root=self.root / "runtimes",
                            scheduler=self.scheduler,
                        ).capabilities()
                        if value.accelerator_family == managed_family
                    ),
                    None,
                )
            except Exception:
                runtime_capability = None
        runtime_verified = managed_family is None or bool(
            runtime_capability and runtime_capability.available
        )
        verified_scenarios = (
            [
                scenario.id
                for scenario in TRAINING_SCENARIOS.list(include_unavailable=False)
                if scenario.available
            ]
            if runtime_verified
            else []
        )
        checks: list[Dict[str, Any]] = [
            {
                "id": "managed_directories",
                "label": "Managed folders",
                "status": "ready" if managed_ready else "attention",
                "summary": (
                    "Dataset, run, repair, and support folders are prepared."
                    if managed_ready
                    else "Halo Forge can prepare its local folders automatically."
                ),
                "technical": {"paths": [str(value) for value in managed_directories]},
            },
            {
                "id": "runtime",
                "label": "Halo Forge runtime",
                "status": "ready",
                "summary": f"Halo Forge {_app_version()} is responding.",
            },
            {
                "id": "storage_root",
                "label": "Local storage",
                "status": "ready" if writable else "blocked",
                "summary": (
                    "The managed Halo Forge folder is writable."
                    if writable
                    else "Halo Forge cannot write to the selected managed folder."
                ),
                "technical": {"path": str(data_root)},
            },
            {
                "id": "disk",
                "label": "Free disk space",
                "status": "ready" if disk_ready else "blocked",
                "summary": (
                    "There is enough room for a small proof run."
                    if disk_ready
                    else "Free space is below the workstation safety minimum."
                ),
                "technical": {
                    "free_bytes": free_bytes,
                    "minimum_free_bytes": minimum_free,
                },
            },
            {
                "id": "memory",
                "label": "Memory",
                "status": "ready" if memory_available is not None else "attention",
                "summary": (
                    "Memory capacity was measured."
                    if memory_available is not None
                    else "Memory measurement is unavailable; preflight will check each run."
                ),
                "technical": {
                    "total_bytes": memory_total,
                    "available_bytes": memory_available,
                },
            },
            {
                "id": "backend",
                "label": "Training backend",
                "status": "ready" if verified_scenarios else "blocked",
                "summary": (
                    f"{len(verified_scenarios)} guided training scenarios are verified for this runtime."
                    if verified_scenarios
                    else "No guided training scenario is verified for the active runtime."
                ),
                "technical": {
                    "active_backends": active_backends,
                    "scenario_ids": verified_scenarios,
                    "managed_runtime": (
                        runtime_capability.to_dict() if runtime_capability else None
                    ),
                },
            },
            {
                "id": "model_access",
                "label": "Model access",
                "status": "ready" if os.environ.get("HF_TOKEN") else "attention",
                "summary": (
                    "Hugging Face access is configured."
                    if os.environ.get("HF_TOKEN")
                    else "Public models are available; gated models may require access."
                ),
            },
            {
                "id": "local_serving",
                "label": "Local model serving",
                "status": "ready" if serving_ready else "attention",
                "summary": (
                    "A local model endpoint is active."
                    if serving_ready
                    else "No local model is being served. This is optional until testing."
                ),
            },
        ]
        remediations: list[SetupRemediation] = []
        if not managed_ready or not writable:
            remediations.append(
                SetupRemediation(
                    id="create_managed_directories",
                    label="Prepare local storage",
                    description="Create the managed data, run, repair, and support folders.",
                    automatic=True,
                    action="create_managed_directories",
                )
            )
        if not disk_ready:
            remediations.append(
                SetupRemediation(
                    id="review_storage",
                    label="Review storage",
                    description="Open the reviewed cleanup flow before starting heavy work.",
                    automatic=False,
                    action="open_storage",
                    blocker="A proof run may fail or leave incomplete artifacts at this free-space level.",
                )
            )
        if managed_family and not runtime_verified:
            label = "AMD" if managed_family == "rocm" else "NVIDIA"
            remediations.insert(
                0,
                SetupRemediation(
                    id="prepare_accelerator_runtime",
                    label=f"Prepare {label} training",
                    description=(
                        "Download the pinned runtime, verify device access, run a real "
                        "optimizer update, and reload the saved artifact."
                    ),
                    automatic=False,
                    action="open_runtime",
                    blocker="Hardware detection alone is not enough to call this workstation ready to train.",
                ),
            )
        blocking = [value for value in checks if value["status"] == "blocked"]
        attention = [value for value in checks if value["status"] == "attention"]
        status = "blocked" if blocking else "attention" if attention else "ready"
        display = {
            "ready": "Ready to train",
            "attention": "Ready with optional setup",
            "blocked": "Setup needed",
        }[status]
        summary = (
            blocking[0]["summary"]
            if blocking
            else "The workstation can run a guided example. Optional connections can be added later."
        )
        payload = {
            "status": status,
            "platform": capability.to_dict(),
            "checks": checks,
            "remediations": [value.to_dict() for value in remediations],
        }
        content_hash = _hash(payload)
        identifier = f"ready-{content_hash[:24]}"
        created_at = _now()
        if persist:
            self.conn.execute(
                """INSERT OR IGNORE INTO workstation_readiness_assessments
                   (id,status,platform,architecture,app_version,content_hash,
                    checks_json,remediations_json,capability_json,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    identifier,
                    status,
                    capability.platform,
                    capability.architecture,
                    capability.runtime_version,
                    content_hash,
                    _json(checks),
                    _json([value.to_dict() for value in remediations]),
                    _json(capability.to_dict()),
                    created_at,
                ),
            )
            self.conn.commit()
        primary = (
            {
                "id": remediations[0].id,
                "label": remediations[0].label,
                "automatic": remediations[0].automatic,
            }
            if remediations
            else {"id": "working_example", "label": "Try a working example", "href": "/datasets/new?example=1"}
        )
        return WorkstationReadiness(
            id=identifier,
            status=status,
            display_status=display,
            summary=summary,
            checks=tuple(checks),
            remediations=tuple(remediations),
            capability=capability,
            content_hash=content_hash,
            created_at=created_at,
            primary_action=primary,
        )

    def apply_setup_remediation(self, action: str) -> WorkstationReadiness:
        if action != "create_managed_directories":
            raise ProductLabError(
                "This setup issue needs an external change; Halo Forge will not pretend it was repaired."
            )
        for path in (
            self.root,
            self.root / "datasets",
            self.root / "runs",
            self.repair_root,
            self.support_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self.assess_readiness()

    def distribution_capability(self) -> DistributionCapability:
        return _distribution_capability()

    def release_status(self) -> Dict[str, Any]:
        """Read a locally supplied, checksummed release manifest.

        Halo Forge never downloads or applies an update from this method. The
        desktop/runtime may place a manifest beside its resources, or an
        operator may point at one with ``HALOFORGE_RELEASE_MANIFEST``.
        """

        configured = os.environ.get("HALOFORGE_RELEASE_MANIFEST")
        path = Path(configured).expanduser() if configured else self.root / "release-manifest.json"
        if not path.is_file():
            return {
                "status": "unavailable",
                "current_version": _app_version(),
                "update_available": False,
                "automatic_update": False,
                "message": "No verified release manifest is installed. Halo Forge will not check or update silently.",
            }
        checksum_path = Path(str(path) + ".sha256")
        if not checksum_path.is_file():
            return {
                "status": "unverified",
                "current_version": _app_version(),
                "update_available": False,
                "automatic_update": False,
                "message": "The release manifest has no checksum and will not be trusted.",
            }
        expected = checksum_path.read_text(encoding="utf-8").strip().split()[0]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected != actual:
            return {
                "status": "invalid",
                "current_version": _app_version(),
                "update_available": False,
                "automatic_update": False,
                "message": "The release manifest checksum does not match.",
            }
        manifest = json.loads(path.read_text(encoding="utf-8"))
        latest = str(manifest.get("version") or "")
        update_available = False
        try:
            from packaging.version import Version

            update_available = bool(latest and Version(latest) > Version(_app_version()))
        except Exception:
            update_available = bool(latest and latest != _app_version())
        return {
            "status": "update_available" if update_available else "current",
            "current_version": _app_version(),
            "latest_version": latest or None,
            "update_available": update_available,
            "automatic_update": False,
            "message": (
                f"Halo Forge {latest} is available. Review the release before installing."
                if update_available
                else "This installation matches the verified release manifest."
            ),
            "release_url": manifest.get("release_url"),
            "manifest_sha256": actual,
        }

    # ----- repair session persistence --------------------------------

    def _resolve_repair_source(self, values: Mapping[str, Any]) -> Dict[str, Any]:
        source_id = str(values.get("source_id") or "").strip() or None
        inspection_id = str(values.get("inspection_id") or "").strip() or None
        version_id = str(values.get("dataset_version_id") or "").strip() or None
        inspection = self.db.get_dataset_source_inspection(inspection_id) if inspection_id else None
        if inspection_id and inspection is None:
            raise ProductLabError(f"Unknown dataset inspection: {inspection_id}")
        if inspection and not source_id:
            source_id = inspection.source_id
        source = self.db.get_dataset_source(source_id) if source_id else None
        version = self.db.get_dataset_version(version_id) if version_id else None
        if source_id and source is None:
            raise ProductLabError(f"Unknown dataset source: {source_id}")
        if version_id and version is None:
            raise ProductLabError(f"Unknown dataset version: {version_id}")
        import_record = None
        if inspection and inspection.import_id:
            import_record = self.db.get_dataset_import(inspection.import_id)
        uri = str(values.get("source_uri") or "").strip()
        if not uri and source:
            uri = source.uri
        if not uri and import_record:
            uri = str(import_record.managed_source_path or import_record.source_uri or "")
        if not uri and version:
            uri = str(Path(version.storage_path) / "records.jsonl")
        if not uri:
            raise ProductLabError("A local, managed, or version source is required for repair.")
        # A repair overlay is consumed by Dataset Lab, so local sources use
        # Dataset Lab's snapshot fingerprint rather than the inspection-only
        # byte fingerprint. Version linkage is retained separately while the
        # actual records file is still hashed so local mutation is detected.
        fingerprint = _repair_source_fingerprint(_source_path(uri))
        scenario = str(values.get("scenario_revision_id") or "").strip() or None
        if not scenario and inspection:
            scenario = inspection.scenario_revision_id
        if not scenario and source:
            metadata = source.metadata
            scenario = str(
                (metadata.get("guided_own_data") or {}).get("scenario_revision_id")
                or metadata.get("scenario_revision_id")
                or ""
            ).strip() or None
        return {
            "source_id": source_id,
            "inspection_id": inspection_id,
            "dataset_version_id": version_id,
            "source_uri": uri,
            "source_fingerprint": fingerprint,
            "scenario_revision_id": scenario,
        }

    def create_repair_session(
        self, values: Mapping[str, Any], *, enqueue: bool = True
    ) -> DatasetRepairSession:
        resolved = self._resolve_repair_source(values)
        identifier = f"repair-session-{uuid.uuid4().hex}"
        now = _now()
        work_item_id = f"product-v17-{uuid.uuid4().hex}" if enqueue and self.scheduler else None
        status = "scanning" if work_item_id else "draft"
        self.conn.execute(
            """INSERT INTO dataset_repair_sessions
               (id,source_id,inspection_id,dataset_version_id,source_uri,
                source_fingerprint,scenario_revision_id,status,stage,progress_json,
                issue_summary_json,work_item_id,created_at,updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                identifier,
                resolved["source_id"],
                resolved["inspection_id"],
                resolved["dataset_version_id"],
                resolved["source_uri"],
                resolved["source_fingerprint"],
                resolved["scenario_revision_id"],
                status,
                "scanning" if work_item_id else "draft",
                _json({"processed": 0}),
                "{}",
                None,
                now,
                now,
            ),
        )
        self.conn.commit()
        if work_item_id:
            try:
                self.scheduler.enqueue(
                    kind="product_v17_repair_scan",
                    launch_spec={
                        "handler": "product_v17.execute_work_item",
                        "action": "repair.scan",
                        "product_root": str(self.root),
                        "payload": {"session_id": identifier},
                    },
                    resource_class="none",
                    domain_kind="dataset_repair_session",
                    domain_id=identifier,
                    max_retries=2,
                    work_item_id=work_item_id,
                )
                self.conn.execute(
                    "UPDATE dataset_repair_sessions SET work_item_id=? WHERE id=?",
                    (work_item_id, identifier),
                )
                self.conn.commit()
            except Exception as exc:
                self.conn.execute(
                    "UPDATE dataset_repair_sessions SET status='failed',stage='failed',error=? WHERE id=?",
                    (str(exc), identifier),
                )
                self.conn.commit()
                raise
        elif bool(values.get("scan", True)):
            self.scan_repair_session(identifier)
        return self.get_repair_session(identifier)

    def get_repair_session(self, session_id: str) -> DatasetRepairSession:
        row = self.conn.execute(
            "SELECT * FROM dataset_repair_sessions WHERE id=?", (session_id,)
        ).fetchone()
        if row is None:
            raise ProductLabError(f"Unknown repair session: {session_id}")
        return DatasetRepairSession(
            id=str(row["id"]),
            source_id=row["source_id"],
            inspection_id=row["inspection_id"],
            dataset_version_id=row["dataset_version_id"],
            source_uri=str(row["source_uri"]),
            source_fingerprint=str(row["source_fingerprint"]),
            scenario_revision_id=row["scenario_revision_id"],
            status=str(row["status"]),
            stage=str(row["stage"]),
            progress=_loads(row["progress_json"], {}),
            issue_summary=_loads(row["issue_summary_json"], {}),
            latest_plan_revision_id=row["latest_plan_revision_id"],
            latest_preview_id=row["latest_preview_id"],
            published_repair_revision_id=row["published_repair_revision_id"],
            work_item_id=row["work_item_id"],
            error=row["error"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def list_repair_sessions(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        limit, offset = _bounded(limit, offset)
        total = int(self.conn.execute("SELECT COUNT(*) AS n FROM dataset_repair_sessions").fetchone()["n"])
        rows = self.conn.execute(
            "SELECT id FROM dataset_repair_sessions ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return {
            "items": [self.get_repair_session(str(row["id"])).to_dict() for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def _current_fingerprint(self, session: DatasetRepairSession) -> str:
        path = _source_path(session.source_uri)
        if not path.exists():
            raise ProductLabError(f"Repair source is missing: {path}")
        return _repair_source_fingerprint(path)

    def _scenario(self, session: DatasetRepairSession):
        if not session.scenario_revision_id:
            return None
        try:
            return TRAINING_SCENARIOS.get(session.scenario_revision_id)
        except KeyError:
            return None

    def _insert_issue(
        self,
        session_id: str,
        ordinal: int,
        *,
        record_id: Optional[str],
        source_index: Optional[int],
        code: str,
        category: str,
        severity: str,
        message: str,
        field_path: Optional[str] = None,
        suggested_actions: Sequence[str] = (),
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> None:
        identifier = f"issue-{_hash([session_id, ordinal, code, record_id, field_path])[:24]}"
        self.conn.execute(
            """INSERT INTO dataset_repair_issues
               (id,session_id,ordinal,record_id,source_index,code,category,severity,
                field_path,message,suggested_actions_json,evidence_json,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                identifier,
                session_id,
                ordinal,
                record_id,
                source_index,
                code,
                category,
                severity,
                field_path,
                message,
                _json(list(suggested_actions)),
                _json(dict(evidence or {})),
                _now(),
            ),
        )

    def _required_value(self, record: Mapping[str, Any], scenario: Any, target: str) -> Any:
        candidates = (target, *tuple(scenario.field_aliases.get(target) or ()))
        marker = object()
        for name in candidates:
            value = get_field(record, str(name), marker)
            if value is not marker:
                return value
        if target in scenario.safe_constants:
            return scenario.safe_constants[target]
        return None

    def _record_issues(
        self,
        record: Mapping[str, Any],
        *,
        session: DatasetRepairSession,
        source_index: int,
        duplicate_of: Optional[str],
    ) -> list[Dict[str, Any]]:
        issues: list[Dict[str, Any]] = []
        scenario = self._scenario(session)
        if scenario:
            for target in scenario.required_fields:
                value = self._required_value(record, scenario, target)
                if value is None:
                    issues.append(
                        {
                            "code": "missing_required_field",
                            "category": "schema",
                            "severity": "error",
                            "field_path": target,
                            "message": f"Required {target.replace('_', ' ')} is missing.",
                            "suggested_actions": ("map_field", "set_constant", "quarantine", "exclude"),
                        }
                    )
                elif isinstance(value, str) and not value.strip():
                    issues.append(
                        {
                            "code": "empty_required_value",
                            "category": "content",
                            "severity": "error",
                            "field_path": target,
                            "message": f"Required {target.replace('_', ' ')} is empty.",
                            "suggested_actions": ("edit", "quarantine", "exclude"),
                        }
                    )
                elif target != "messages":
                    expected: Optional[type | tuple[type, ...]] = None
                    if target in {"tools"}:
                        expected = (list, dict)
                    elif target in {"labels", "candidates"}:
                        expected = list
                    elif target in {
                        "prompt", "response", "chosen", "rejected", "question",
                        "answer", "reference_answer", "reasoning", "transcript",
                        "caption", "text", "input", "anchor", "positive", "query",
                        "document", "image", "audio",
                    }:
                        expected = str
                    if expected is not None and not isinstance(value, expected):
                        issues.append(
                            {
                                "code": "invalid_field_type",
                                "category": "schema",
                                "severity": "error",
                                "field_path": target,
                                "message": f"{target.replace('_', ' ').title()} has an incompatible type.",
                                "suggested_actions": (
                                    "coerce_scalar", "edit", "quarantine"
                                ) if expected is str else ("edit", "quarantine"),
                                "evidence": {"observed_type": type(value).__name__},
                            }
                        )
            messages = self._required_value(record, scenario, "messages")
            if messages is not None:
                if not isinstance(messages, list):
                    issues.append(
                        {
                            "code": "invalid_field_type",
                            "category": "schema",
                            "severity": "error",
                            "field_path": "messages",
                            "message": "Conversation messages must be a list.",
                            "suggested_actions": ("edit", "quarantine"),
                        }
                    )
                else:
                    allowed = {"system", "user", "assistant", "tool"}
                    previous_role: Optional[str] = None
                    for index, message in enumerate(messages):
                        role = message.get("role", message.get("from")) if isinstance(message, Mapping) else None
                        content = message.get("content", message.get("value")) if isinstance(message, Mapping) else None
                        if not isinstance(message, Mapping):
                            issues.append(
                                {
                                    "code": "invalid_message_type",
                                    "category": "conversation",
                                    "severity": "error",
                                    "field_path": f"messages.{index}",
                                    "message": "Each conversation turn must be an object.",
                                    "suggested_actions": ("edit", "quarantine"),
                                }
                            )
                            continue
                        if role not in allowed:
                            issues.append(
                                {
                                    "code": "invalid_conversation_role",
                                    "category": "conversation",
                                    "severity": "error",
                                    "field_path": f"messages.{index}.role",
                                    "message": f"Conversation role {role!r} is not recognized.",
                                    "suggested_actions": ("normalize_roles", "edit", "quarantine"),
                                }
                            )
                        if isinstance(content, str) and not content.strip():
                            issues.append(
                                {
                                    "code": "empty_conversation_turn",
                                    "category": "conversation",
                                    "severity": "error",
                                    "field_path": f"messages.{index}.content",
                                    "message": "A conversation turn is empty.",
                                    "suggested_actions": ("trim", "edit", "quarantine"),
                                }
                            )
                        normalized_role = {
                            "human": "user",
                            "gpt": "assistant",
                            "function": "tool",
                        }.get(str(role), str(role))
                        if index == 0 and normalized_role not in {"system", "user"}:
                            issues.append(
                                {
                                    "code": "invalid_conversation_order",
                                    "category": "conversation",
                                    "severity": "error",
                                    "field_path": f"messages.{index}.role",
                                    "message": "A conversation should begin with a system or user turn.",
                                    "suggested_actions": ("edit", "quarantine"),
                                }
                            )
                        if previous_role == normalized_role and normalized_role in {"user", "assistant"}:
                            issues.append(
                                {
                                    "code": "repeated_conversation_role",
                                    "category": "conversation",
                                    "severity": "warning",
                                    "field_path": f"messages.{index}.role",
                                    "message": f"Two {normalized_role} turns appear consecutively.",
                                    "suggested_actions": ("edit", "quarantine"),
                                }
                            )
                        if normalized_role in allowed:
                            previous_role = normalized_role
            if scenario.canonical_schema == "preference":
                chosen = self._required_value(record, scenario, "chosen")
                rejected = self._required_value(record, scenario, "rejected")
                if chosen is not None and rejected is not None and chosen == rejected:
                    issues.append(
                        {
                            "code": "identical_preference_sides",
                            "category": "preference",
                            "severity": "error",
                            "field_path": "chosen",
                            "message": "Chosen and rejected responses are identical.",
                            "suggested_actions": ("edit", "quarantine", "exclude"),
                        }
                    )
                chosen_score = record.get("chosen_score")
                rejected_score = record.get("rejected_score")
                if (
                    isinstance(chosen_score, (int, float))
                    and isinstance(rejected_score, (int, float))
                    and chosen_score < rejected_score
                ):
                    issues.append(
                        {
                            "code": "possible_swapped_preference",
                            "category": "preference",
                            "severity": "warning",
                            "field_path": "chosen",
                            "message": "The supplied scores rank the rejected response above the chosen response.",
                            "suggested_actions": ("edit", "quarantine", "exclude"),
                            "evidence": {"chosen_score": chosen_score, "rejected_score": rejected_score},
                        }
                    )
            media_target = "image" if scenario.modality == "image" else "audio" if scenario.modality == "audio" else None
            if media_target:
                value = self._required_value(record, scenario, media_target)
                if isinstance(value, str) and value.strip():
                    root = Path(str(record.get("_media_root") or _source_path(session.source_uri).parent))
                    candidate = Path(value).expanduser()
                    if not candidate.is_absolute():
                        candidate = root / candidate
                    try:
                        resolved = candidate.resolve()
                        resolved.relative_to(root.resolve())
                        safe = True
                    except ValueError:
                        safe = False
                        resolved = candidate.resolve()
                    if not safe:
                        issues.append(
                            {
                                "code": "unsafe_media_path",
                                "category": "media",
                                "severity": "error",
                                "field_path": media_target,
                                "message": "The media path escapes the selected source folder.",
                                "suggested_actions": ("media_root", "edit", "quarantine"),
                            }
                        )
                    elif not resolved.is_file():
                        issues.append(
                            {
                                "code": "missing_media",
                                "category": "media",
                                "severity": "error",
                                "field_path": media_target,
                                "message": "The referenced media file is missing.",
                                "suggested_actions": ("media_root", "edit", "quarantine"),
                                "evidence": {"reference": value},
                            }
                        )
                    else:
                        accepted = _IMAGE_SUFFIXES if media_target == "image" else _AUDIO_SUFFIXES
                        if resolved.suffix.lower() not in accepted:
                            issues.append(
                                {
                                    "code": "unsupported_media_type",
                                    "category": "media",
                                    "severity": "error",
                                    "field_path": media_target,
                                    "message": f"{resolved.suffix or 'Unknown'} media is not supported for this scenario.",
                                    "suggested_actions": ("edit", "quarantine"),
                                }
                            )
                        else:
                            metadata_error = self._validate_media_metadata(
                                resolved, media_target
                            )
                            if metadata_error:
                                issues.append(
                                    {
                                        "code": "unsupported_media_metadata",
                                        "category": "media",
                                        "severity": "error",
                                        "field_path": media_target,
                                        "message": metadata_error,
                                        "suggested_actions": ("quarantine", "exclude"),
                                        "evidence": {"reference": value},
                                    }
                                )
        if duplicate_of:
            issues.append(
                {
                    "code": "duplicate_record",
                    "category": "duplicate",
                    "severity": "warning",
                    "field_path": None,
                    "message": "This record duplicates an earlier source record.",
                    "suggested_actions": ("exclude", "quarantine"),
                    "evidence": {"duplicate_of": duplicate_of},
                }
            )
        return issues

    @staticmethod
    def _validate_media_metadata(path: Path, media_target: str) -> Optional[str]:
        """Return a truthful media validation error without changing the asset."""

        try:
            if media_target == "image":
                from PIL import Image

                with Image.open(path) as image:
                    width, height = image.size
                    image.verify()
                if width <= 0 or height <= 0:
                    return "The image has invalid dimensions."
                return None
            try:
                import soundfile as sf  # type: ignore

                info = sf.info(str(path))
                if int(info.samplerate) <= 0 or int(info.channels) <= 0 or int(info.frames) <= 0:
                    return "The audio has invalid duration, sample rate, or channel metadata."
                return None
            except ImportError:
                if path.suffix.lower() != ".wav":
                    return None
                import wave

                with wave.open(str(path), "rb") as handle:
                    if handle.getframerate() <= 0 or handle.getnchannels() <= 0 or handle.getnframes() <= 0:
                        return "The audio has invalid duration, sample rate, or channel metadata."
                return None
        except Exception as exc:
            return f"The {media_target} metadata could not be read: {exc}"

    def scan_repair_session(self, session_id: str) -> DatasetRepairSession:
        session = self.get_repair_session(session_id)
        current = self._current_fingerprint(session)
        if current != session.source_fingerprint:
            self.conn.execute(
                "UPDATE dataset_repair_sessions SET status='stale',stage='source_changed',error=?,updated_at=? WHERE id=?",
                ("The source changed after inspection. Rebase before applying repairs.", _now(), session_id),
            )
            self.conn.commit()
            return self.get_repair_session(session_id)
        self.conn.execute("DELETE FROM dataset_repair_issues WHERE session_id=?", (session_id,))
        self.conn.execute(
            "UPDATE dataset_repair_sessions SET status='scanning',stage='scanning',error=NULL,updated_at=? WHERE id=?",
            (_now(), session_id),
        )
        self.conn.commit()
        seen: Dict[str, str] = {}
        labels: Dict[str, set[str]] = defaultdict(set)
        counts: Counter[str] = Counter()
        records_with_issues: set[int] = set()
        processed = 0
        issue_ordinal = 0
        source = _source_path(session.source_uri)
        if source.suffix.lower() in {".csv", ".tsv"}:
            try:
                with source.open("r", encoding="utf-8-sig", newline="") as handle:
                    first_line = next(line.rstrip("\r\n") for line in handle if line.strip())
            except (OSError, StopIteration, UnicodeError):
                first_line = ""
            expected_delimiter = "," if source.suffix.lower() == ".csv" else "\t"
            alternate_delimiters = ("\t", ";", "|") if expected_delimiter == "," else (",", ";", "|")
            detected = next(
                (value for value in alternate_delimiters if first_line.count(value) >= 1),
                None,
            )
            if first_line and expected_delimiter not in first_line and detected:
                counts["parse"] += 1
                self._insert_issue(
                    session_id,
                    issue_ordinal,
                    record_id=None,
                    source_index=None,
                    code="possible_delimiter_mismatch",
                    category="parse",
                    severity="error",
                    message=(
                        f"The file extension expects {expected_delimiter!r}, but the header "
                        f"appears to use {detected!r}."
                    ),
                    suggested_actions=("quarantine", "exclude"),
                    evidence={"expected": expected_delimiter, "detected": detected},
                )
                issue_ordinal += 1
        for source_index, (record, parse_issue) in enumerate(iter_source_records(source)):
            state = self.conn.execute(
                "SELECT cancel_requested FROM dataset_repair_sessions WHERE id=?", (session_id,)
            ).fetchone()
            if state and bool(state["cancel_requested"]):
                self.conn.execute(
                    "UPDATE dataset_repair_sessions SET status='cancelled',stage='cancelled',updated_at=? WHERE id=?",
                    (_now(), session_id),
                )
                self.conn.commit()
                return self.get_repair_session(session_id)
            processed += 1
            if parse_issue is not None:
                counts["parse"] += 1
                records_with_issues.add(source_index)
                self._insert_issue(
                    session_id,
                    issue_ordinal,
                    record_id=None,
                    source_index=source_index,
                    code=str(parse_issue.get("code") or "parse_error"),
                    category="parse",
                    severity="error",
                    message=str(parse_issue.get("message") or "The record could not be parsed."),
                    suggested_actions=("quarantine", "exclude"),
                    evidence=parse_issue,
                )
                issue_ordinal += 1
                continue
            assert record is not None
            identity = deterministic_record_id(record)
            digest = record_hash(record)
            duplicate_of = seen.get(digest)
            seen.setdefault(digest, identity)
            scenario = self._scenario(session)
            if scenario and scenario.canonical_schema == "classification":
                label = self._required_value(record, scenario, "label")
                if label is not None:
                    labels[str(label).casefold().strip()].add(str(label))
            record_issues = self._record_issues(
                record,
                session=session,
                source_index=source_index,
                duplicate_of=duplicate_of,
            )
            if record_issues:
                records_with_issues.add(source_index)
            for issue in record_issues:
                counts[str(issue["category"])] += 1
                self._insert_issue(
                    session_id,
                    issue_ordinal,
                    record_id=identity,
                    source_index=source_index,
                    code=str(issue["code"]),
                    category=str(issue["category"]),
                    severity=str(issue["severity"]),
                    field_path=issue.get("field_path"),
                    message=str(issue["message"]),
                    suggested_actions=issue.get("suggested_actions") or (),
                    evidence=issue.get("evidence") or {},
                )
                issue_ordinal += 1
            if processed % 100 == 0:
                self.conn.execute(
                    "UPDATE dataset_repair_sessions SET progress_json=?,updated_at=? WHERE id=?",
                    (_json({"processed": processed, "issues": issue_ordinal}), _now(), session_id),
                )
                self.conn.commit()
        for normalized, variants in sorted(labels.items()):
            if len(variants) <= 1:
                continue
            counts["label"] += 1
            self._insert_issue(
                session_id,
                issue_ordinal,
                record_id=None,
                source_index=None,
                code="inconsistent_label_spelling",
                category="label",
                severity="warning",
                message=f"Equivalent label spellings were found: {', '.join(sorted(variants))}.",
                field_path="label",
                suggested_actions=("label_alias",),
                evidence={"normalized": normalized, "variants": sorted(variants)},
            )
            issue_ordinal += 1
        summary = {
            "records_scanned": processed,
            "issue_count": issue_ordinal,
            "by_category": dict(counts),
            "records_with_issues": len(records_with_issues),
            "clean_records_exact": max(0, processed - len(records_with_issues)),
            "exact": True,
        }
        self.conn.execute(
            """UPDATE dataset_repair_sessions
               SET status='ready',stage='issues_ready',progress_json=?,issue_summary_json=?,
                   cancel_requested=0,error=NULL,updated_at=? WHERE id=?""",
            (_json({"processed": processed, "total": processed}), _json(summary), _now(), session_id),
        )
        self.conn.commit()
        return self.get_repair_session(session_id)

    def list_repair_issues(
        self,
        session_id: str,
        *,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        self.get_repair_session(session_id)
        limit, offset = _bounded(limit, offset)
        clauses = ["session_id=?"]
        params: list[Any] = [session_id]
        if category:
            clauses.append("category=?")
            params.append(category)
        if severity:
            clauses.append("severity=?")
            params.append(severity)
        where = " AND ".join(clauses)
        total = int(self.conn.execute(f"SELECT COUNT(*) AS n FROM dataset_repair_issues WHERE {where}", params).fetchone()["n"])
        rows = self.conn.execute(
            f"SELECT * FROM dataset_repair_issues WHERE {where} ORDER BY ordinal LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        items = [
            DatasetIssue(
                id=str(row["id"]),
                session_id=str(row["session_id"]),
                ordinal=int(row["ordinal"]),
                record_id=row["record_id"],
                source_index=row["source_index"],
                code=str(row["code"]),
                category=str(row["category"]),
                severity=str(row["severity"]),
                field_path=row["field_path"],
                message=str(row["message"]),
                suggested_actions=tuple(_loads(row["suggested_actions_json"], [])),
                evidence=_loads(row["evidence_json"], {}),
            ).to_dict()
            for row in rows
        ]
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    # ----- immutable repair plans and overlays -----------------------

    @staticmethod
    def _normalize_action(value: Mapping[str, Any], ordinal: int) -> DatasetRepairAction:
        kind = str(value.get("action_kind") or value.get("kind") or "").strip().lower()
        allowed = {
            "map_field", "coerce_scalar", "normalize_roles", "trim", "label_alias",
            "media_root", "set_constant", "edit", "quarantine", "exclude",
        }
        if kind not in allowed:
            raise ProductLabError(f"Unsupported repair action: {kind}")
        reason = str(value.get("reason") or "").strip()
        if not reason:
            raise ProductLabError("Every repair action requires a retained reason.")
        field = str(value.get("field_path") or value.get("field") or "").strip() or None
        record_id = str(value.get("record_id") or "").strip() or None
        source_index = (
            int(value["source_index"])
            if value.get("source_index") is not None
            else None
        )
        if kind in {"edit", "coerce_scalar"} and not record_id:
            raise ProductLabError(f"{kind} requires record_id")
        if kind in {"map_field", "coerce_scalar", "label_alias", "set_constant", "edit"} and not field:
            raise ProductLabError(f"{kind} requires field_path")
        if kind in {"quarantine", "exclude"} and not record_id:
            raise ProductLabError(f"{kind} requires record_id")
        return DatasetRepairAction(
            ordinal=ordinal,
            issue_code=str(value.get("issue_code") or "operator_review"),
            action_kind=kind,
            reason=reason,
            record_id=record_id,
            source_index=source_index,
            field_path=field,
            value=copy.deepcopy(value.get("value")),
            before_hash=str(value.get("before_hash") or "").strip() or None,
            after_hash=str(value.get("after_hash") or "").strip() or None,
        )

    def create_repair_plan(
        self, session_id: str, values: Mapping[str, Any]
    ) -> DatasetRepairPlanRevision:
        session = self.get_repair_session(session_id)
        if session.status not in {"ready", "published"}:
            raise ProductLabError("Repair issues must finish scanning before a plan is saved.")
        actions = tuple(
            self._normalize_action(value, ordinal)
            for ordinal, value in enumerate(values.get("actions") or [])
        )
        if not actions:
            raise ProductLabError("A repair plan requires at least one reviewed action.")
        definition = {
            "source_fingerprint": session.source_fingerprint,
            "actions": [value.to_dict() for value in actions],
        }
        content_hash = _hash(definition)
        existing = self.conn.execute(
            "SELECT id FROM dataset_repair_plan_revisions WHERE content_hash=?", (content_hash,)
        ).fetchone()
        if existing:
            return self.get_repair_plan(str(existing["id"]))
        revision_number = int(
            self.conn.execute(
                "SELECT COALESCE(MAX(revision_number),0)+1 AS n FROM dataset_repair_plan_revisions WHERE session_id=?",
                (session_id,),
            ).fetchone()["n"]
        )
        identifier = f"repair-plan-{content_hash[:24]}"
        now = _now()
        self.conn.execute(
            """INSERT INTO dataset_repair_plan_revisions
               (id,session_id,revision_number,source_fingerprint,content_hash,definition_json,created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (identifier, session_id, revision_number, session.source_fingerprint, content_hash, _json(definition), now),
        )
        for action in actions:
            self.conn.execute(
                """INSERT INTO dataset_repair_actions
                   (revision_id,ordinal,record_id,source_index,issue_code,action_kind,field_path,
                    value_json,reason,before_hash,after_hash)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    identifier, action.ordinal, action.record_id, action.source_index, action.issue_code,
                    action.action_kind, action.field_path, _json(action.value), action.reason,
                    action.before_hash, action.after_hash,
                ),
            )
        self.conn.execute(
            "UPDATE dataset_repair_sessions SET latest_plan_revision_id=?,updated_at=? WHERE id=?",
            (identifier, now, session_id),
        )
        self.conn.commit()
        return self.get_repair_plan(identifier)

    def get_repair_plan(self, revision_id: str) -> DatasetRepairPlanRevision:
        row = self.conn.execute(
            "SELECT * FROM dataset_repair_plan_revisions WHERE id=?", (revision_id,)
        ).fetchone()
        if row is None:
            raise ProductLabError(f"Unknown repair plan revision: {revision_id}")
        actions = self.conn.execute(
            "SELECT * FROM dataset_repair_actions WHERE revision_id=? ORDER BY ordinal", (revision_id,)
        ).fetchall()
        return DatasetRepairPlanRevision(
            id=str(row["id"]),
            session_id=str(row["session_id"]),
            revision_number=int(row["revision_number"]),
            source_fingerprint=str(row["source_fingerprint"]),
            content_hash=str(row["content_hash"]),
            actions=tuple(
                DatasetRepairAction(
                    ordinal=int(action["ordinal"]),
                    issue_code=str(action["issue_code"]),
                    action_kind=str(action["action_kind"]),
                    reason=str(action["reason"]),
                    record_id=action["record_id"],
                    source_index=action["source_index"],
                    field_path=action["field_path"],
                    value=_loads(action["value_json"], None),
                    before_hash=action["before_hash"],
                    after_hash=action["after_hash"],
                )
                for action in actions
            ),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _coerce(value: Any, target: Any) -> Any:
        name = str(target.get("type") if isinstance(target, Mapping) else target).lower()
        if name in {"str", "string", "text"}:
            return str(value)
        if name in {"int", "integer"}:
            return int(value)
        if name in {"float", "number"}:
            return float(value)
        if name in {"bool", "boolean"}:
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes"}:
                    return True
                if lowered in {"false", "0", "no"}:
                    return False
                raise ValueError(f"cannot convert {value!r} to boolean")
            return bool(value)
        raise ValueError(f"unsupported scalar type: {name}")

    def _apply_actions(
        self,
        row: Mapping[str, Any],
        actions: Sequence[DatasetRepairAction],
        *,
        source_index: Optional[int] = None,
    ) -> tuple[str, Dict[str, Any], list[str]]:
        output = copy.deepcopy(dict(row))
        identity = deterministic_record_id(row)
        applied: list[str] = []
        for action in actions:
            if action.record_id and action.record_id != identity:
                continue
            if action.source_index is not None and action.source_index != source_index:
                continue
            kind = action.action_kind
            if kind in {"quarantine", "exclude"}:
                return kind, output, [*applied, kind]
            if kind == "map_field":
                set_field(output, str(action.field_path), copy.deepcopy(get_field(output, str(action.value))))
            elif kind == "coerce_scalar":
                set_field(output, str(action.field_path), self._coerce(get_field(output, str(action.field_path)), action.value))
            elif kind == "normalize_roles":
                field = action.field_path or "messages"
                messages = get_field(output, field)
                mapping = dict(action.value or {"human": "user", "gpt": "assistant", "function": "tool"})
                if isinstance(messages, list):
                    normalized = []
                    for message in messages:
                        if not isinstance(message, Mapping):
                            normalized.append(message)
                            continue
                        item = dict(message)
                        original = item.get("role", item.get("from"))
                        item["role"] = mapping.get(str(original), original)
                        if "content" not in item and "value" in item:
                            item["content"] = item["value"]
                        normalized.append(item)
                    set_field(output, field, normalized)
            elif kind == "trim":
                if action.field_path:
                    value = get_field(output, action.field_path)
                    if isinstance(value, str):
                        set_field(output, action.field_path, value.strip())
                else:
                    for key, value in list(output.items()):
                        if isinstance(value, str):
                            output[key] = value.strip()
                    messages = output.get("messages")
                    if isinstance(messages, list):
                        trimmed_messages = []
                        for message in messages:
                            if not isinstance(message, Mapping):
                                trimmed_messages.append(message)
                                continue
                            item = dict(message)
                            content_key = "content" if "content" in item else "value" if "value" in item else None
                            if content_key and isinstance(item.get(content_key), str):
                                item[content_key] = str(item[content_key]).strip()
                                if not item[content_key]:
                                    continue
                            trimmed_messages.append(item)
                        output["messages"] = trimmed_messages
            elif kind == "label_alias":
                value = get_field(output, str(action.field_path))
                aliases = dict(action.value or {})
                set_field(output, str(action.field_path), aliases.get(str(value), value))
            elif kind == "media_root":
                output["_media_root"] = str(action.value)
            elif kind in {"set_constant", "edit"}:
                set_field(output, str(action.field_path), copy.deepcopy(action.value))
            applied.append(kind)
        return "replace" if applied else "unchanged", output, applied

    def prepare_repair_preview(
        self, session_id: str, plan_revision_id: str, *, enqueue: bool = True
    ) -> DatasetRepairPreview:
        session = self.get_repair_session(session_id)
        plan = self.get_repair_plan(plan_revision_id)
        if plan.session_id != session_id:
            raise ProductLabError("Repair plan belongs to a different session.")
        identifier = f"repair-preview-{uuid.uuid4().hex}"
        work_item_id = f"product-v17-{uuid.uuid4().hex}" if enqueue and self.scheduler else None
        now = _now()
        self.conn.execute(
            """INSERT INTO dataset_repair_previews
               (id,session_id,plan_revision_id,source_fingerprint,status,work_item_id,created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (identifier, session_id, plan_revision_id, session.source_fingerprint, "queued" if work_item_id else "running", None, now),
        )
        self.conn.execute(
            "UPDATE dataset_repair_sessions SET status='previewing',stage='previewing',latest_preview_id=?,work_item_id=?,updated_at=? WHERE id=?",
            (identifier, None, now, session_id),
        )
        self.conn.commit()
        if work_item_id:
            try:
                self.scheduler.enqueue(
                    kind="product_v17_repair_preview",
                    launch_spec={
                        "handler": "product_v17.execute_work_item",
                        "action": "repair.preview",
                        "product_root": str(self.root),
                        "payload": {"preview_id": identifier},
                    },
                    resource_class="none",
                    domain_kind="dataset_repair_preview",
                    domain_id=identifier,
                    max_retries=2,
                    work_item_id=work_item_id,
                )
                self.conn.execute(
                    "UPDATE dataset_repair_previews SET work_item_id=? WHERE id=?",
                    (work_item_id, identifier),
                )
                self.conn.execute(
                    "UPDATE dataset_repair_sessions SET work_item_id=? WHERE id=?",
                    (work_item_id, session_id),
                )
                self.conn.commit()
            except Exception as exc:
                self.conn.execute(
                    "UPDATE dataset_repair_previews SET status='failed',error=? WHERE id=?",
                    (str(exc), identifier),
                )
                self.conn.commit()
                raise
            return self.get_repair_preview(identifier)
        return self.run_repair_preview(identifier)

    def run_repair_preview(self, preview_id: str) -> DatasetRepairPreview:
        preview = self.get_repair_preview(preview_id)
        session = self.get_repair_session(preview.session_id)
        plan = self.get_repair_plan(preview.plan_revision_id)
        current = self._current_fingerprint(session)
        if current != session.source_fingerprint or current != plan.source_fingerprint:
            self.conn.execute(
                "UPDATE dataset_repair_previews SET status='failed',error=?,completed_at=? WHERE id=?",
                ("The source changed. Rebase repairs before previewing again.", _now(), preview_id),
            )
            self.conn.execute(
                "UPDATE dataset_repair_sessions SET status='stale',stage='source_changed',updated_at=? WHERE id=?",
                (_now(), session.id),
            )
            self.conn.commit()
            return self.get_repair_preview(preview_id)
        self.conn.execute("UPDATE dataset_repair_previews SET status='running' WHERE id=?", (preview_id,))
        self.conn.commit()
        counts: Counter[str] = Counter()
        issue_counts: Counter[str] = Counter()
        samples: list[Dict[str, Any]] = []
        overlay_entries: list[Dict[str, Any]] = []
        source = _source_path(session.source_uri)
        valid_source_index = -1
        for index, (record, parse_issue) in enumerate(iter_source_records(source)):
            state = self.conn.execute(
                "SELECT cancel_requested FROM dataset_repair_sessions WHERE id=?",
                (session.id,),
            ).fetchone()
            if state and bool(state["cancel_requested"]):
                self.conn.execute(
                    "UPDATE dataset_repair_previews SET status='cancelled',error=?,completed_at=? WHERE id=?",
                    ("Preview cancelled by the operator", _now(), preview_id),
                )
                self.conn.execute(
                    "UPDATE dataset_repair_sessions SET status='cancelled',stage='cancelled',updated_at=? WHERE id=?",
                    (_now(), session.id),
                )
                self.conn.commit()
                return self.get_repair_preview(preview_id)
            if parse_issue is not None:
                counts["quarantined"] += 1
                issue_counts[str(parse_issue.get("code") or "parse_error")] += 1
                continue
            assert record is not None
            valid_source_index += 1
            identity = deterministic_record_id(record)
            before = record_hash(record)
            try:
                operation, repaired, applied = self._apply_actions(
                    record, plan.actions, source_index=index
                )
                after = record_hash(repaired)
            except (TypeError, ValueError) as exc:
                operation, repaired, applied, after = "quarantine", dict(record), ["repair_error"], before
                issue_counts["repair_error"] += 1
                repaired["_repair_error"] = str(exc)
            counts[operation] += 1
            if operation != "unchanged":
                overlay_entries.append(
                    {
                        "record_id": identity,
                        # Dataset Lab seeds lineage after parse failures have
                        # been quarantined, so this occurrence index is over
                        # valid source records. The reviewed action itself
                        # still uses the original source index shown in UI.
                        "source_index": valid_source_index,
                        "original_source_index": index,
                        "operation": operation,
                        "before_hash": before,
                        "after_hash": after,
                        "record": repaired if operation == "replace" else None,
                        "actions": applied,
                    }
                )
            if len(samples) < 100 and operation != "unchanged":
                samples.append(
                    {
                        "record_id": identity,
                        "operation": operation,
                        "original": record,
                        "repaired": repaired if operation == "replace" else None,
                        "actions": applied,
                    }
                )
            if index % 250 == 0:
                self.conn.execute(
                    "UPDATE dataset_repair_sessions SET progress_json=?,updated_at=? WHERE id=?",
                    (_json({"processed": index + 1, "changed": len(overlay_entries)}), _now(), session.id),
                )
                self.conn.commit()
        counts["total"] = sum(counts[key] for key in ("unchanged", "replace", "quarantine", "exclude"))
        counts["accepted"] = counts["unchanged"] + counts["replace"]
        counts["changed"] = counts["replace"]
        payload = {
            "source_fingerprint": session.source_fingerprint,
            "plan_hash": plan.content_hash,
            "entries": overlay_entries,
        }
        content_hash = _hash(payload)
        staging_parent = self.repair_root / session.id
        staging_parent.mkdir(parents=True, exist_ok=True)
        final = staging_parent / content_hash[:24]
        if not final.exists():
            temporary = Path(tempfile.mkdtemp(prefix=".preview-", dir=staging_parent))
            try:
                with (temporary / "overlay.jsonl").open("w", encoding="utf-8") as handle:
                    for entry in overlay_entries:
                        handle.write(_json(entry) + "\n")
                manifest = {
                    "format_version": 1,
                    "source_fingerprint": session.source_fingerprint,
                    "plan_revision_id": plan.id,
                    "plan_hash": plan.content_hash,
                    "content_hash": content_hash,
                    "counts": dict(counts),
                    "overlay_sha256": hashlib.sha256((temporary / "overlay.jsonl").read_bytes()).hexdigest(),
                }
                (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                os.replace(temporary, final)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        now = _now()
        self.conn.execute(
            """UPDATE dataset_repair_previews SET status='completed',exact=1,counts_json=?,
               issue_counts_json=?,split_impact_json=?,sample_json=?,content_hash=?,
               storage_path=?,error=NULL,completed_at=? WHERE id=?""",
            (
                _json(dict(counts)), _json(dict(issue_counts)),
                _json({"status": "recomputed_during_dataset_build", "protected_splits": True}),
                _json(samples), content_hash, str(final), now, preview_id,
            ),
        )
        self.conn.execute(
            "UPDATE dataset_repair_sessions SET status='ready',stage='preview_ready',progress_json=?,updated_at=? WHERE id=?",
            (_json({"processed": counts["total"], "total": counts["total"]}), now, session.id),
        )
        self.conn.commit()
        return self.get_repair_preview(preview_id)

    def get_repair_preview(self, preview_id: str) -> DatasetRepairPreview:
        row = self.conn.execute("SELECT * FROM dataset_repair_previews WHERE id=?", (preview_id,)).fetchone()
        if row is None:
            raise ProductLabError(f"Unknown repair preview: {preview_id}")
        return DatasetRepairPreview(
            id=str(row["id"]), session_id=str(row["session_id"]),
            plan_revision_id=str(row["plan_revision_id"]), source_fingerprint=str(row["source_fingerprint"]),
            status=str(row["status"]), exact=bool(row["exact"]), counts=_loads(row["counts_json"], {}),
            issue_counts=_loads(row["issue_counts_json"], {}), split_impact=_loads(row["split_impact_json"], {}),
            sample=tuple(_loads(row["sample_json"], [])), content_hash=row["content_hash"],
            storage_path=row["storage_path"], work_item_id=row["work_item_id"], error=row["error"],
            created_at=str(row["created_at"]), completed_at=row["completed_at"],
        )

    def publish_repair_revision(self, preview_id: str) -> Dict[str, Any]:
        preview = self.get_repair_preview(preview_id)
        if preview.status != "completed" or not preview.exact or not preview.storage_path:
            raise ProductLabError("An exact completed preview is required before publication.")
        session = self.get_repair_session(preview.session_id)
        if self._current_fingerprint(session) != preview.source_fingerprint:
            raise ProductLabError("The source changed after preview. Rebase before publication.")
        manifest_path = Path(preview.storage_path) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        overlay_hash = str(manifest["overlay_sha256"])
        identity = {
            "source_fingerprint": preview.source_fingerprint,
            "preview_content_hash": preview.content_hash,
            "plan_revision_id": preview.plan_revision_id,
            "overlay_hash": overlay_hash,
        }
        content_hash = _hash(identity)
        identifier = f"repair-{content_hash[:24]}"
        self.conn.execute(
            """INSERT OR IGNORE INTO dataset_repair_revisions
               (id,session_id,plan_revision_id,preview_id,source_fingerprint,content_hash,
                repaired_record_set_hash,storage_path,manifest_json,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                identifier, session.id, preview.plan_revision_id, preview.id,
                preview.source_fingerprint, content_hash, overlay_hash,
                preview.storage_path, _json({**manifest, "repair_revision_id": identifier}), _now(),
            ),
        )
        self.conn.execute(
            "UPDATE dataset_repair_sessions SET status='published',stage='published',published_repair_revision_id=?,updated_at=? WHERE id=?",
            (identifier, _now(), session.id),
        )
        self.conn.commit()
        return self.get_repair_revision(identifier)

    def get_repair_revision(self, revision_id: str) -> Dict[str, Any]:
        row = self.conn.execute("SELECT * FROM dataset_repair_revisions WHERE id=?", (revision_id,)).fetchone()
        if row is None:
            raise ProductLabError(f"Unknown dataset repair revision: {revision_id}")
        return {
            "id": str(row["id"]), "session_id": str(row["session_id"]),
            "plan_revision_id": str(row["plan_revision_id"]), "preview_id": str(row["preview_id"]),
            "source_fingerprint": str(row["source_fingerprint"]), "content_hash": str(row["content_hash"]),
            "repaired_record_set_hash": str(row["repaired_record_set_hash"]),
            "storage_path": str(row["storage_path"]), "manifest": _loads(row["manifest_json"], {}),
            "created_at": str(row["created_at"]),
            "recipe_step": {"kind": "repair_overlay", "revision_id": str(row["id"])},
        }

    def rebase_repair_session(self, session_id: str) -> DatasetRepairSession:
        old = self.get_repair_session(session_id)
        current = self._current_fingerprint(old)
        values = {
            "source_id": old.source_id,
            "inspection_id": old.inspection_id,
            "dataset_version_id": old.dataset_version_id,
            "source_uri": old.source_uri,
            "source_fingerprint": current,
            "scenario_revision_id": old.scenario_revision_id,
            "scan": True,
        }
        replacement = self.create_repair_session(values, enqueue=False)
        old_plan = self.get_repair_plan(old.latest_plan_revision_id) if old.latest_plan_revision_id else None
        if old_plan:
            unresolved_occurrences = {
                (action.record_id, action.source_index)
                for action in old_plan.actions
                if action.record_id
            }
            valid_occurrences: set[tuple[Optional[str], Optional[int]]] = set()
            if unresolved_occurrences:
                for current_index, (record, parse_issue) in enumerate(
                    iter_source_records(_source_path(replacement.source_uri))
                ):
                    if parse_issue is not None or record is None:
                        continue
                    identity = deterministic_record_id(record)
                    for candidate in ((identity, current_index), (identity, None)):
                        if candidate in unresolved_occurrences:
                            valid_occurrences.add(candidate)
                        if valid_occurrences == unresolved_occurrences:
                            break
            actions = []
            conflicts = []
            for action in old_plan.actions:
                if action.record_id and (action.record_id, action.source_index) not in valid_occurrences:
                    conflicts.append(action.to_dict())
                else:
                    actions.append(action.to_dict())
            if actions:
                self.create_repair_plan(replacement.id, {"actions": actions})
            self.conn.execute(
                "UPDATE dataset_repair_sessions SET issue_summary_json=? WHERE id=?",
                (_json({**replacement.issue_summary, "rebase_from": old.id, "conflicts": conflicts}), replacement.id),
            )
            self.conn.commit()
        return self.get_repair_session(replacement.id)

    def cancel_repair(self, session_id: str) -> DatasetRepairSession:
        session = self.get_repair_session(session_id)
        self.conn.execute(
            "UPDATE dataset_repair_sessions SET cancel_requested=1,updated_at=? WHERE id=?", (_now(), session_id)
        )
        self.conn.commit()
        if self.scheduler and session.work_item_id:
            self.scheduler.cancel(session.work_item_id)
        return self.get_repair_session(session_id)

    # ----- privacy-safe support bundles -------------------------------

    def support_bundle_preview(
        self, categories: Optional[Sequence[str]] = None
    ) -> SupportBundlePreview:
        selected = tuple(dict.fromkeys(str(value) for value in (categories or _DEFAULT_SUPPORT_CATEGORIES)))
        unknown = sorted(set(selected) - _SUPPORT_CATEGORIES)
        if unknown:
            raise ProductLabError("Unknown support categories: " + ", ".join(unknown))
        labels = {
            "versions": "Application, Python, platform, and runtime versions",
            "readiness": "Latest workstation checks without credentials",
            "scheduler": "Recent work status, failure classes, and retry counts",
            "logs": "Sanitized desktop and application log tails",
            "integrity": "Release and artifact integrity summaries",
        }
        return SupportBundlePreview(
            categories=selected,
            included=tuple({"id": key, "description": labels[key]} for key in selected),
            excluded_by_default=(
                "credentials and environment secrets", "dataset records", "prompts and outputs",
                "image and audio content", "model weights", "full local paths",
            ),
            redaction_policy="recursive-secret-redaction-and-stable-path-pseudonyms-v1",
        )

    def create_support_bundle(
        self, categories: Optional[Sequence[str]] = None, *, enqueue: bool = True
    ) -> SupportBundle:
        preview = self.support_bundle_preview(categories)
        identifier = f"support-{uuid.uuid4().hex}"
        work_item_id = f"product-v17-{uuid.uuid4().hex}" if enqueue and self.scheduler else None
        now = _now()
        self.conn.execute(
            """INSERT INTO support_bundles
               (id,status,categories_json,preview_json,work_item_id,created_at)
               VALUES (?,?,?,?,?,?)""",
            (identifier, "queued" if work_item_id else "draft", _json(list(preview.categories)), _json(preview.to_dict()), None, now),
        )
        self.conn.commit()
        if work_item_id:
            try:
                self.scheduler.enqueue(
                    kind="product_v17_support_bundle",
                    launch_spec={
                        "handler": "product_v17.execute_work_item", "action": "support.bundle",
                        "product_root": str(self.root), "payload": {"bundle_id": identifier},
                    },
                    resource_class="none", domain_kind="support_bundle", domain_id=identifier,
                    max_retries=1, work_item_id=work_item_id,
                )
                self.conn.execute(
                    "UPDATE support_bundles SET work_item_id=? WHERE id=?",
                    (work_item_id, identifier),
                )
                self.conn.commit()
            except Exception as exc:
                self.conn.execute(
                    "UPDATE support_bundles SET status='failed',error=? WHERE id=?",
                    (str(exc), identifier),
                )
                self.conn.commit()
                raise
            return self.get_support_bundle(identifier)
        return self.build_support_bundle(identifier)

    def _redact(self, value: Any, *, path_map: Dict[str, str]) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): ("<redacted>" if _SECRET_KEYS.search(str(key)) else self._redact(item, path_map=path_map))
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._redact(item, path_map=path_map) for item in value]
        if not isinstance(value, str):
            return value
        text = _TOKEN_TEXT.sub(lambda match: (match.group(1) if match.lastindex else "") + "<redacted>", value)
        candidates = [str(Path.home()), str(self.root), str(Path.cwd())]
        for candidate in sorted(set(candidates), key=len, reverse=True):
            if candidate and candidate in text:
                replacement = path_map.setdefault(candidate, f"<path-{len(path_map) + 1}>")
                text = text.replace(candidate, replacement)
        def replace_path(match: re.Match[str]) -> str:
            original = match.group(0)
            return path_map.setdefault(original, f"<path-{len(path_map) + 1}>")

        text = _PATH_TEXT.sub(replace_path, text)
        return text

    def _support_payload(self, categories: Sequence[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if "versions" in categories:
            payload["versions"] = {
                "halo_forge": _app_version(), "python": platform.python_version(),
                "platform": platform.platform(), "machine": platform.machine(),
            }
        if "readiness" in categories:
            payload["readiness"] = self.assess_readiness().to_dict()
        if "scheduler" in categories:
            payload["scheduler"] = [
                {
                    "id": item.id, "kind": item.kind, "status": item.status,
                    "domain_kind": item.domain_kind, "retry_count": item.retry_count,
                    "max_retries": item.max_retries, "error": item.error,
                    "created_at": item.created_at, "updated_at": item.updated_at,
                }
                for item in self.db.list_work_items(limit=200)
            ]
        if "logs" in categories:
            logs = []
            candidates = [self.root / "desktop" / "runtime.log"]
            candidates.extend(sorted((self.root / "logs").glob("*.log")) if (self.root / "logs").is_dir() else [])
            for path in candidates[:20]:
                if not path.is_file():
                    continue
                try:
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-300:]
                except OSError:
                    continue
                lines = [
                    "[record or model content omitted by support-bundle policy]"
                    if _SENSITIVE_LOG_LINE.search(line)
                    else line
                    for line in lines
                ]
                logs.append({"name": path.name, "lines": lines})
            payload["logs"] = logs
        if "integrity" in categories:
            rows = self.conn.execute(
                "SELECT * FROM release_qualifications ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
            payload["integrity"] = [
                {
                    "platform": row["platform"], "architecture": row["architecture"],
                    "package_type": row["package_type"], "signature_state": row["signature_state"],
                    "smoke_status": row["smoke_status"], "content_hash": row["content_hash"],
                }
                for row in rows
            ]
        return self._redact(payload, path_map={})

    def build_support_bundle(self, bundle_id: str) -> SupportBundle:
        bundle = self.get_support_bundle(bundle_id)
        categories = bundle.categories
        state = self.conn.execute(
            "SELECT cancel_requested FROM support_bundles WHERE id=?", (bundle_id,)
        ).fetchone()
        if state and bool(state["cancel_requested"]):
            self.conn.execute(
                "UPDATE support_bundles SET status='cancelled',error=?,completed_at=? WHERE id=?",
                ("Support bundle creation cancelled by the operator", _now(), bundle_id),
            )
            self.conn.commit()
            return self.get_support_bundle(bundle_id)
        self.conn.execute("UPDATE support_bundles SET status='running',error=NULL WHERE id=?", (bundle_id,))
        self.conn.commit()
        payload = self._support_payload(categories)
        content_hash = _hash(payload)
        self.support_root.mkdir(parents=True, exist_ok=True)
        final = self.support_root / f"halo-forge-support-{content_hash[:16]}.zip"
        manifest = {
            "format_version": 1, "content_hash": content_hash,
            "categories": list(categories), "redaction_policy": "v1",
            "automatic_upload": False,
            "files": {f"{name}.json": _hash(value) for name, value in sorted(payload.items())},
        }
        temporary = Path(tempfile.mkstemp(prefix=".support-", suffix=".zip", dir=self.support_root)[1])
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, value in sorted(payload.items()):
                    info = zipfile.ZipInfo(f"{name}.json", date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, json.dumps(value, indent=2, sort_keys=True) + "\n")
                info = zipfile.ZipInfo("manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            state = self.conn.execute(
                "SELECT cancel_requested FROM support_bundles WHERE id=?", (bundle_id,)
            ).fetchone()
            if state and bool(state["cancel_requested"]):
                temporary.unlink(missing_ok=True)
                self.conn.execute(
                    "UPDATE support_bundles SET status='cancelled',error=?,completed_at=? WHERE id=?",
                    ("Support bundle creation cancelled by the operator", _now(), bundle_id),
                )
                self.conn.commit()
                return self.get_support_bundle(bundle_id)
            os.replace(temporary, final)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        self.conn.execute(
            """UPDATE support_bundles SET status='completed',manifest_json=?,storage_path=?,
               content_hash=?,error=NULL,completed_at=? WHERE id=?""",
            (_json(manifest), str(final), content_hash, _now(), bundle_id),
        )
        self.conn.commit()
        return self.get_support_bundle(bundle_id)

    def get_support_bundle(self, bundle_id: str) -> SupportBundle:
        row = self.conn.execute("SELECT * FROM support_bundles WHERE id=?", (bundle_id,)).fetchone()
        if row is None:
            raise ProductLabError(f"Unknown support bundle: {bundle_id}")
        return SupportBundle(
            id=str(row["id"]), status=str(row["status"]), categories=tuple(_loads(row["categories_json"], [])),
            preview=_loads(row["preview_json"], {}), manifest=_loads(row["manifest_json"], {}),
            storage_path=row["storage_path"], content_hash=row["content_hash"], work_item_id=row["work_item_id"],
            error=row["error"], created_at=str(row["created_at"]), completed_at=row["completed_at"],
        )

    def verify_support_bundle(self, bundle_id: str) -> Dict[str, Any]:
        bundle = self.get_support_bundle(bundle_id)
        if bundle.status != "completed" or not bundle.storage_path:
            return {"bundle_id": bundle_id, "valid": False, "reason": "bundle_not_complete"}
        path = Path(bundle.storage_path)
        if not path.is_file():
            return {"bundle_id": bundle_id, "valid": False, "reason": "bundle_missing"}
        try:
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                files_valid = all(
                    _hash(json.loads(archive.read(name))) == expected
                    for name, expected in dict(manifest.get("files") or {}).items()
                )
        except (OSError, KeyError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
            return {"bundle_id": bundle_id, "valid": False, "reason": str(exc)}
        return {
            "bundle_id": bundle_id, "valid": bool(files_valid),
            "content_hash": bundle.content_hash, "automatic_upload": False,
        }

    def delete_support_bundle(self, bundle_id: str) -> bool:
        bundle = self.get_support_bundle(bundle_id)
        if bundle.storage_path:
            Path(bundle.storage_path).unlink(missing_ok=True)
        result = self.conn.execute("DELETE FROM support_bundles WHERE id=?", (bundle_id,))
        self.conn.commit()
        return bool(result.rowcount)

    # ----- release qualification -------------------------------------

    def request_release_qualification(
        self, values: Mapping[str, Any], *, enqueue: bool = True
    ) -> ReleaseQualification:
        capability = self.distribution_capability()
        package_type = str(values.get("package_type") or capability.desktop_package or "browser-cli")
        signature = str(values.get("signature_state") or capability.signature_state)
        smoke = str(values.get("smoke_status") or "not_run")
        if signature == "unsigned" and str(values.get("distribution_status") or "preview") == "supported":
            raise ProductLabError("Unsigned desktop packages cannot be qualified as normal supported installers.")
        package_path = str(values.get("package_path") or "").strip()
        if package_path:
            path = Path(package_path).expanduser().resolve()
            if not path.is_file():
                raise ProductLabError(f"Release package does not exist: {path}")
            package_path = str(path)
        request = {
            "platform": str(values.get("platform") or capability.platform),
            "architecture": str(values.get("architecture") or capability.architecture),
            "package_type": package_type, "signature_state": signature,
            "smoke_status": smoke,
            "supported_backends": list(
                values.get("supported_backends") or capability.supported_backends
            ),
            "package_path": package_path,
            "evidence": dict(values.get("evidence") or {}),
        }
        identifier = f"release-qualification-{uuid.uuid4().hex}"
        request_hash = _hash({"request": request, "request_id": identifier})
        work_item_id = f"product-v17-{uuid.uuid4().hex}" if enqueue and self.scheduler else None
        now = _now()
        self.conn.execute(
            """INSERT INTO release_qualifications
               (id,status,platform,architecture,package_type,signature_state,smoke_status,
                supported_backends_json,evidence_json,content_hash,work_item_id,progress_json,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                identifier, "queued" if work_item_id else "running", request["platform"],
                request["architecture"], package_type, signature, smoke,
                _json(request["supported_backends"]), _json({"request": request}),
                request_hash, None, _json({"stage": "waiting"}), now,
            ),
        )
        self.conn.commit()
        if work_item_id:
            try:
                self.scheduler.enqueue(
                    kind="product_v17_release_qualification",
                    launch_spec={
                        "handler": "product_v17.execute_work_item",
                        "action": "release.qualify",
                        "product_root": str(self.root),
                        "payload": {"qualification_id": identifier},
                    },
                    resource_class="none",
                    domain_kind="release_qualification",
                    domain_id=identifier,
                    max_retries=1,
                    work_item_id=work_item_id,
                )
                self.conn.execute(
                    "UPDATE release_qualifications SET work_item_id=? WHERE id=?",
                    (work_item_id, identifier),
                )
                self.conn.commit()
            except Exception as exc:
                self.conn.execute(
                    "UPDATE release_qualifications SET status='failed',error=? WHERE id=?",
                    (str(exc), identifier),
                )
                self.conn.commit()
                raise
            return self.get_release_qualification(identifier)
        return self.run_release_qualification(identifier)

    def qualify_release(self, values: Mapping[str, Any]) -> ReleaseQualification:
        """Synchronous compatibility entrypoint used by the CLI and tests."""

        return self.request_release_qualification(values, enqueue=False)

    def run_release_qualification(self, qualification_id: str) -> ReleaseQualification:
        row = self.conn.execute(
            "SELECT * FROM release_qualifications WHERE id=?", (qualification_id,)
        ).fetchone()
        if row is None:
            raise ProductLabError(f"Unknown release qualification: {qualification_id}")
        request = dict(_loads(row["evidence_json"], {}).get("request") or {})
        evidence = dict(request.get("evidence") or {})
        package_path = str(request.get("package_path") or "")
        self.conn.execute(
            "UPDATE release_qualifications SET status='running',progress_json=? WHERE id=?",
            (_json({"stage": "hashing", "processed_bytes": 0}), qualification_id),
        )
        self.conn.commit()
        if package_path:
            path = Path(package_path)
            digest = hashlib.sha256()
            processed = 0
            with path.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    state = self.conn.execute(
                        "SELECT cancel_requested FROM release_qualifications WHERE id=?",
                        (qualification_id,),
                    ).fetchone()
                    if state and bool(state["cancel_requested"]):
                        self.conn.execute(
                            "UPDATE release_qualifications SET status='cancelled',error=?,completed_at=? WHERE id=?",
                            ("Release qualification cancelled by the operator", _now(), qualification_id),
                        )
                        self.conn.commit()
                        return self.get_release_qualification(qualification_id)
                    digest.update(chunk)
                    processed += len(chunk)
                    self.conn.execute(
                        "UPDATE release_qualifications SET progress_json=? WHERE id=?",
                        (
                            _json(
                                {
                                    "stage": "hashing",
                                    "processed_bytes": processed,
                                    "total_bytes": path.stat().st_size,
                                }
                            ),
                            qualification_id,
                        ),
                    )
                    self.conn.commit()
            evidence.update(
                package_name=path.name,
                package_sha256=digest.hexdigest(),
                size_bytes=path.stat().st_size,
            )
        identity = {
            "platform": str(request["platform"]),
            "architecture": str(request["architecture"]),
            "package_type": str(request["package_type"]),
            "signature_state": str(request["signature_state"]),
            "smoke_status": str(request["smoke_status"]),
            "supported_backends": list(request.get("supported_backends") or []),
            "evidence": evidence,
        }
        content_hash = _hash(identity)
        self.conn.execute(
            """UPDATE release_qualifications
               SET status='completed',evidence_json=?,content_hash=?,progress_json=?,
                   error=NULL,completed_at=? WHERE id=?""",
            (
                _json(evidence), content_hash, _json({"stage": "verified", "complete": True}),
                _now(), qualification_id,
            ),
        )
        self.conn.commit()
        return self.get_release_qualification(qualification_id)

    def get_release_qualification(self, identifier: str) -> ReleaseQualification:
        row = self.conn.execute("SELECT * FROM release_qualifications WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ProductLabError(f"Unknown release qualification: {identifier}")
        return ReleaseQualification(
            id=identifier, platform=str(row["platform"]), architecture=str(row["architecture"]),
            package_type=str(row["package_type"]), signature_state=str(row["signature_state"]),
            smoke_status=str(row["smoke_status"]), supported_backends=tuple(_loads(row["supported_backends_json"], [])),
            evidence=_loads(row["evidence_json"], {}), content_hash=str(row["content_hash"]),
            work_item_id=row["work_item_id"], created_at=str(row["created_at"]),
            status=str(row["status"]), progress=_loads(row["progress_json"], {}),
            error=row["error"], completed_at=row["completed_at"],
        )

    def list_release_qualifications(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        limit, offset = _bounded(limit, offset)
        total = int(
            self.conn.execute(
                "SELECT COUNT(*) AS n FROM release_qualifications"
            ).fetchone()["n"]
        )
        rows = self.conn.execute(
            "SELECT id FROM release_qualifications ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return {
            "items": [
                self.get_release_qualification(str(row["id"])).to_dict()
                for row in rows
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def cancel_release_qualification(self, identifier: str) -> ReleaseQualification:
        value = self.get_release_qualification(identifier)
        self.conn.execute(
            "UPDATE release_qualifications SET cancel_requested=1 WHERE id=?",
            (identifier,),
        )
        self.conn.commit()
        if self.scheduler and value.work_item_id:
            self.scheduler.cancel(value.work_item_id)
        return self.get_release_qualification(identifier)

    # ----- worker entrypoint -----------------------------------------

    def execute_work_item(self, item: Any) -> Dict[str, Any]:
        action = str(item.launch_spec.get("action") or "")
        values = dict(item.launch_spec.get("payload") or {})
        if action == "repair.scan":
            return {"session": self.scan_repair_session(str(values["session_id"])).to_dict()}
        if action == "repair.preview":
            return {"preview": self.run_repair_preview(str(values["preview_id"])).to_dict()}
        if action == "support.bundle":
            return {"bundle": self.build_support_bundle(str(values["bundle_id"])).to_dict()}
        if action == "release.qualify":
            return {
                "qualification": self.run_release_qualification(
                    str(values["qualification_id"])
                ).to_dict()
            }
        raise ProductLabError(f"Unsupported V17 work action: {action}")
