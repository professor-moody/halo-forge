"""Truthful progressive certification of real Halo Forge training paths."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from halo_forge.managed_runtime import ManagedRuntimeService
from halo_forge.managed_runtime.occupancy import wait_for_stable_idle
from halo_forge.run_db import RunDatabase

from .models import (
    CertificationRecoveryAction,
    TrainingPathCapability,
    TrainingPathCertification,
    TrainingPathCertificationMatrix,
    TrainingPathCertificationStep,
    TrainingPathProfileRevision,
    WorkstationCertification,
)
from .registry import PATH_DEFINITIONS, normalized_definition


CERTIFICATION_STEPS: tuple[tuple[str, str], ...] = (
    ("fixture_dataset", "Build the certification dataset"),
    ("trainer_artifact", "Render the trainer-ready dataset"),
    ("model_preparation", "Prepare the exact model revision"),
    ("capacity_step", "Run a disposable capacity step"),
    ("optimizer_update", "Run the real trainer optimizer step"),
    ("parameter_delta", "Verify trained parameters changed"),
    ("artifact_files", "Verify the expected artifact files"),
    ("artifact_reload", "Reload the artifact and run fixed inference"),
    ("replay_lineage", "Verify replay and data lineage"),
    ("scratch_cleanup", "Verify disposable scratch cleanup"),
)


class TrainingPathCertificationError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _loads(value: Any, default: Any) -> Any:
    try:
        return json.loads(value) if value not in (None, "") else default
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    candidates = [root / "pyproject.toml", root / "requirements.txt"]
    candidates.extend((root / "halo_forge").rglob("*.py"))
    for path in sorted((p for p in candidates if p.is_file()), key=lambda p: str(p)):
        relative = str(path.relative_to(root)).encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


StepExecutor = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


class TrainingPathCertificationService:
    """Separate real-trainer proof from generic runtime diagnostics."""

    def __init__(
        self,
        database: RunDatabase,
        *,
        root: Optional[Path | str] = None,
        runtime_service: Optional[ManagedRuntimeService] = None,
        scheduler: Any = None,
        source_root: Optional[Path | str] = None,
        step_executor: Optional[StepExecutor] = None,
        occupancy_probe: Optional[Callable[[str], Any]] = None,
    ):
        self.database = database
        self.conn = database._conn
        self.root = Path(root or Path.home() / ".halo-forge" / "certifications").expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.source_root = Path(source_root or Path(__file__).resolve().parents[2]).resolve()
        self.scheduler = scheduler
        self.runtime = runtime_service or ManagedRuntimeService(
            database,
            root=Path.home() / ".halo-forge" / "runtimes",
            scheduler=scheduler,
            source_root=self.source_root,
        )
        self.step_executor = step_executor or self._execute_real_step
        self.occupancy_probe = occupancy_probe
        self._bootstrap_profiles()

    # ---- immutable path registry ----------------------------------

    def _bootstrap_profiles(self) -> None:
        now = _now()
        for family in ("rocm", "cuda"):
            for raw in PATH_DEFINITIONS:
                definition = normalized_definition(dict(raw), family)
                profile_id = f"{definition['profile_id']}-{family}"
                content = {
                    key: value
                    for key, value in definition.items()
                    if key not in {"label", "description", "recommended"}
                }
                content_hash = _hash(content)
                revision_id = f"training-path-revision-{content_hash[:24]}"
                self.conn.execute(
                    """INSERT OR IGNORE INTO training_path_profiles
                       (id,name,scenario_revision_id,trainer_mode,model_id,description,
                        latest_revision_id,created_at,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        profile_id,
                        f"{definition['label']} ({family.upper()})",
                        definition.get("scenario_revision_id"),
                        definition["trainer_mode"],
                        definition["model_id"],
                        "Real trainer-path certification; generic tensor diagnostics do not qualify.",
                        revision_id,
                        now,
                        now,
                    ),
                )
                self.conn.execute(
                    """INSERT OR IGNORE INTO training_path_profile_revisions
                       (id,profile_id,revision_number,content_hash,runtime_family,backend,
                        scenario_revision_id,trainer_mode,model_id,model_revision,
                        tokenizer_processor_hash,fixture_id,fixture_hash,
                        trainer_adapter_version,capacity_adapter_version,configuration_json,
                        expected_artifacts_json,created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        revision_id,
                        profile_id,
                        1,
                        content_hash,
                        family,
                        definition["backend"],
                        definition.get("scenario_revision_id"),
                        definition["trainer_mode"],
                        definition["model_id"],
                        definition["model_revision"],
                        definition["tokenizer_processor_hash"],
                        definition["fixture_id"],
                        definition["fixture_hash"],
                        definition["trainer_adapter_version"],
                        definition["capacity_adapter_version"],
                        _json(definition["configuration"]),
                        _json(list(definition["expected_artifacts"])),
                        now,
                    ),
                )
                self.conn.execute(
                    "UPDATE training_path_profiles SET latest_revision_id=?,updated_at=? WHERE id=?",
                    (revision_id, now, profile_id),
                )
        self.conn.commit()

    def list_revisions(self, *, runtime_family: Optional[str] = None) -> tuple[TrainingPathProfileRevision, ...]:
        if runtime_family:
            rows = self.conn.execute(
                "SELECT * FROM training_path_profile_revisions WHERE runtime_family=? ORDER BY trainer_mode,profile_id",
                (runtime_family,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM training_path_profile_revisions ORDER BY runtime_family,trainer_mode,profile_id"
            ).fetchall()
        return tuple(self._revision(row) for row in rows)

    def get_revision(self, identifier: str) -> Optional[TrainingPathProfileRevision]:
        row = self.conn.execute(
            "SELECT * FROM training_path_profile_revisions WHERE id=?", (identifier,)
        ).fetchone()
        return self._revision(row) if row else None

    # ---- certification lifecycle ---------------------------------

    def preview(self, path_revision_id: str, runtime_revision_id: str) -> dict[str, Any]:
        path = self.get_revision(path_revision_id)
        if path is None:
            raise KeyError(path_revision_id)
        runtime_revision = self.runtime.get_revision(runtime_revision_id)
        if runtime_revision is None:
            raise KeyError(runtime_revision_id)
        profile = self.runtime.get_profile(runtime_revision.profile_id)
        if profile is None or profile.accelerator_family != path.runtime_family:
            raise TrainingPathCertificationError("The training path and runtime accelerator families differ")
        qualification = self.runtime.latest_qualification(runtime_revision_id)
        runtime_ready = bool(
            qualification
            and qualification.status in {"local_verified", "vendor_supported"}
            and self.runtime.verify(qualification.id)["valid"]
        )
        return {
            "path_revision": path.to_dict(),
            "runtime_revision_id": runtime_revision_id,
            "runtime_ready": runtime_ready,
            "executor_available": bool(path.configuration.get("executor_available")),
            "unavailable_reason": (
                None
                if path.configuration.get("executor_available")
                else "This path has a pinned profile but no real certification executor yet. It remains unavailable in guided mode."
            ),
            "runtime_qualification_id": qualification.id if runtime_ready and qualification else None,
            "steps": [{"id": step, "label": label} for step, label in CERTIFICATION_STEPS],
            "estimates": {
                "model": path.model_id,
                "download_bytes": None,
                "storage_bytes": None,
                "duration_seconds": None,
                "provenance": "unavailable_until_measured",
            },
        }

    def certify(
        self,
        path_revision_id: str,
        runtime_revision_id: str,
        *,
        enqueue: bool = True,
    ) -> TrainingPathCertification:
        preview = self.preview(path_revision_id, runtime_revision_id)
        if not preview["runtime_ready"]:
            raise TrainingPathCertificationError("Complete core runtime qualification first")
        if not preview["executor_available"]:
            raise TrainingPathCertificationError(str(preview["unavailable_reason"]))
        qualification = self.runtime.get_qualification(preview["runtime_qualification_id"])
        assert qualification is not None
        current = self.conn.execute(
            """SELECT * FROM training_path_certifications
               WHERE path_revision_id=? AND runtime_revision_id=?
                 AND status IN ('queued','running','waiting_for_accelerator','verified')
               ORDER BY created_at DESC LIMIT 1""",
            (path_revision_id, runtime_revision_id),
        ).fetchone()
        if current is not None:
            value = self._certification(current)
            if value.status != "verified" or self.verify(value.id)["valid"]:
                return value
        source_identity = _source_hash(self.source_root)
        created = _now()
        identifier = f"training-path-certification-{_hash({'path': path_revision_id, 'runtime': runtime_revision_id, 'source': source_identity, 'created': created})[:24]}"
        self.conn.execute(
            """INSERT INTO training_path_certifications
               (id,path_revision_id,runtime_revision_id,runtime_qualification_id,status,
                stage,host_identity_hash,device_identity_hash,runtime_identity_hash,
                source_identity_hash,progress_json,resume_cursor_json,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                identifier,
                path_revision_id,
                runtime_revision_id,
                qualification.id,
                "queued" if enqueue else "running",
                "queued",
                qualification.host_identity_hash,
                qualification.device_identity_hash,
                qualification.runtime_identity_hash,
                source_identity,
                _json({"current": 0, "total": len(CERTIFICATION_STEPS)}),
                _json({"last_complete_step": 0}),
                created,
            ),
        )
        for ordinal, (step_id, label) in enumerate(CERTIFICATION_STEPS, 1):
            self.conn.execute(
                "INSERT INTO training_path_certification_steps (certification_id,ordinal,step_id,label,status) VALUES (?,?,?,?, 'pending')",
                (identifier, ordinal, step_id, label),
            )
        self.conn.commit()
        if enqueue:
            scheduler = self._scheduler()
            item = scheduler.enqueue(
                kind="training_path_certification",
                launch_spec={
                    "handler": "training_path_certification.execute_work_item",
                    "operation": "certify",
                    "certification_root": str(self.root),
                    "runtime_root": str(self.runtime.root),
                    "source_root": str(self.source_root),
                },
                resource_class="accelerator",
                resource_requirements={
                    "accelerator_family": self.get_revision(path_revision_id).runtime_family,  # type: ignore[union-attr]
                    "runtime_profile_revision_id": runtime_revision_id,
                    "output_path": str(self.root),
                },
                domain_kind="training_path_certification",
                domain_id=identifier,
                max_retries=1,
            )
            self.conn.execute(
                "UPDATE training_path_certifications SET work_item_id=? WHERE id=?",
                (item.id, identifier),
            )
            self.conn.commit()
        return self.get_certification(identifier)  # type: ignore[return-value]

    def run_certification(self, certification_id: str) -> TrainingPathCertification:
        value = self.get_certification(certification_id)
        if value is None:
            raise KeyError(certification_id)
        path = self.get_revision(value.path_revision_id)
        if path is None:
            raise TrainingPathCertificationError("Training path revision is missing")
        runtime_check = self.runtime.verify(value.runtime_qualification_id)
        if not runtime_check["valid"]:
            return self._finish_failure(value.id, "runtime", "Core runtime qualification is stale or corrupt", status="stale")
        stable, samples = wait_for_stable_idle(
            path.runtime_family,
            probe=self.occupancy_probe,
        ) if self.occupancy_probe else wait_for_stable_idle(path.runtime_family)
        if not stable:
            reason = samples[-1].reason or "accelerator is not independently verified idle"
            self.conn.execute(
                "UPDATE training_path_certifications SET status='waiting_for_accelerator',stage='waiting_for_accelerator',error=?,progress_json=? WHERE id=?",
                (reason, _json({"availability": samples[-1].to_dict()}), value.id),
            )
            self.conn.commit()
            return self.get_certification(value.id)  # type: ignore[return-value]

        attempt_count = self.conn.execute(
            "SELECT COUNT(*) AS count FROM training_path_certification_attempts WHERE certification_id=?",
            (value.id,),
        ).fetchone()
        attempt_ordinal = int(attempt_count["count"] or 0) + 1
        attempt_dir = self.root / "training-paths" / value.id / "attempts" / f"attempt-{attempt_ordinal}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt_id = f"training-path-attempt-{uuid.uuid4().hex}"
        resume_from = int(value.resume_cursor.get("last_complete_step") or 0)
        self.conn.execute(
            "INSERT INTO training_path_certification_attempts (id,certification_id,ordinal,status,resume_from_step,output_dir,created_at) VALUES (?,?,?,?,?,?,?)",
            (attempt_id, value.id, attempt_ordinal, "running", resume_from, str(attempt_dir), _now()),
        )
        self.conn.execute(
            "UPDATE training_path_certifications SET status='running',stage='starting',error=NULL WHERE id=?",
            (value.id,),
        )
        self.conn.commit()

        try:
            accumulated: dict[str, Any] = {}
            for step in value.steps:
                if step.ordinal <= resume_from and step.status == "passed":
                    accumulated[step.step_id] = step.result
                    continue
                cancel = self.conn.execute(
                    "SELECT cancel_requested FROM training_path_certifications WHERE id=?", (value.id,)
                ).fetchone()
                if cancel and bool(cancel["cancel_requested"]):
                    self.conn.execute(
                        "UPDATE training_path_certification_steps SET status='cancelled',completed_at=? WHERE certification_id=? AND ordinal=?",
                        (_now(), value.id, step.ordinal),
                    )
                    self.conn.execute(
                        "UPDATE training_path_certifications SET status='cancelled',stage='cancelled',completed_at=? WHERE id=?",
                        (_now(), value.id),
                    )
                    self.conn.execute(
                        "UPDATE training_path_certification_attempts SET status='cancelled',completed_at=? WHERE id=?",
                        (_now(), attempt_id),
                    )
                    self.conn.commit()
                    return self.get_certification(value.id)  # type: ignore[return-value]

                # Close the race before every step capable of starting GPU work.
                immediate = self.occupancy_probe(path.runtime_family) if self.occupancy_probe else self.runtime.availability(path.runtime_family)
                if not immediate.idle:
                    raise TrainingPathCertificationError(
                        immediate.reason or "accelerator became occupied before a certification child started"
                    )
                input_value = {
                    "certification_id": value.id,
                    "path_revision": path.to_dict(),
                    "runtime_revision_id": value.runtime_revision_id,
                    "runtime_qualification_id": value.runtime_qualification_id,
                    "source_identity_hash": value.source_identity_hash,
                    "attempt_dir": str(attempt_dir),
                    "evidence": accumulated,
                }
                input_hash = _hash({"step": step.step_id, "input": input_value})
                self.conn.execute(
                    "UPDATE training_path_certification_steps SET status='running',input_hash=?,started_at=? WHERE certification_id=? AND ordinal=?",
                    (input_hash, _now(), value.id, step.ordinal),
                )
                self.conn.execute(
                    "UPDATE training_path_certifications SET stage=?,progress_json=? WHERE id=?",
                    (step.step_id, _json({"current": step.ordinal - 1, "total": len(CERTIFICATION_STEPS), "label": step.label}), value.id),
                )
                self.conn.commit()
                result = dict(self.step_executor(step.step_id, input_value))
                after_child = (
                    self.occupancy_probe(path.runtime_family)
                    if self.occupancy_probe
                    else self.runtime.availability(path.runtime_family)
                )
                if not after_child.idle:
                    raise TrainingPathCertificationError(
                        "Accelerator contention appeared during certification; this measurement cannot be reused"
                    )
                self._validate_step(step.step_id, result, accumulated, path)
                result_hash = _hash(result)
                log_path = result.pop("_log_path", None)
                self.conn.execute(
                    """UPDATE training_path_certification_steps
                       SET status='passed',result_json=?,evidence_hash=?,log_path=?,completed_at=?
                       WHERE certification_id=? AND ordinal=?""",
                    (_json(result), result_hash, log_path, _now(), value.id, step.ordinal),
                )
                self.conn.execute(
                    "UPDATE training_path_certifications SET resume_cursor_json=?,progress_json=? WHERE id=?",
                    (_json({"last_complete_step": step.ordinal, "step_id": step.step_id}), _json({"current": step.ordinal, "total": len(CERTIFICATION_STEPS), "label": step.label}), value.id),
                )
                self.conn.commit()
                accumulated[step.step_id] = result

            return self._publish(value.id, path, accumulated, attempt_id)
        except Exception as exc:
            current = self.get_certification(value.id)
            stage = current.stage if current else "certification"
            self.conn.execute(
                "UPDATE training_path_certification_steps SET status='failed',result_json=?,completed_at=? WHERE certification_id=? AND step_id=? AND status='running'",
                (_json({"error": str(exc)}), _now(), value.id, stage),
            )
            self.conn.execute(
                "UPDATE training_path_certification_attempts SET status='failed',error=?,completed_at=? WHERE id=?",
                (str(exc)[-4000:], _now(), attempt_id),
            )
            self.conn.commit()
            return self._finish_failure(value.id, stage, str(exc))

    def _validate_step(
        self,
        step_id: str,
        result: Mapping[str, Any],
        prior: Mapping[str, Any],
        path: TrainingPathProfileRevision,
    ) -> None:
        if not bool(result.get("passed", True)):
            raise TrainingPathCertificationError(str(result.get("reason") or f"{step_id} failed"))
        if step_id == "fixture_dataset":
            if not result.get("dataset_version_id") or not result.get("dataset_content_hash"):
                raise TrainingPathCertificationError("Dataset Lab did not publish an immutable fixture version")
        elif step_id == "trainer_artifact":
            if int(result.get("format_version") or 0) < 3 or not result.get("artifact_hash"):
                raise TrainingPathCertificationError("The trainer artifact lacks v3 identity and lineage")
        elif step_id == "model_preparation":
            commit = str(result.get("resolved_model_commit") or "")
            if len(commit) < 7 or not result.get("tokenizer_processor_hash"):
                raise TrainingPathCertificationError("The model or tokenizer/processor was not pinned")
        elif step_id == "capacity_step":
            if not result.get("optimizer_step_executed") or not result.get("scratch_cleaned"):
                raise TrainingPathCertificationError("Disposable capacity execution did not complete and clean up")
        elif step_id == "optimizer_update":
            if not result.get("real_trainer_entrypoint") or not result.get("weights_updated"):
                raise TrainingPathCertificationError("The shipped trainer did not report a real optimizer update")
        elif step_id == "parameter_delta":
            before = str(result.get("before") or "")
            after = str(result.get("after") or "")
            if len(before) != 64 or len(after) != 64 or before == after or not result.get("changed"):
                raise TrainingPathCertificationError("Trainable parameter hashes do not prove a weight update")
        elif step_id == "artifact_files":
            missing = list(result.get("missing") or [])
            if missing or not result.get("verified"):
                raise TrainingPathCertificationError("Expected artifact files are missing: " + ", ".join(missing))
        elif step_id == "artifact_reload":
            if not result.get("reloaded") or not result.get("finite_output"):
                raise TrainingPathCertificationError("The saved artifact did not reload with finite output")
        elif step_id == "replay_lineage":
            if int(result.get("replay_version") or 0) < 14 or not result.get("lineage_complete"):
                raise TrainingPathCertificationError("Replay v14 and complete lineage are required")
        elif step_id == "scratch_cleanup" and not result.get("scratch_cleaned"):
            raise TrainingPathCertificationError("Certification scratch cleanup is incomplete")

    def _execute_real_step(self, step_id: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
        attempt_dir = Path(str(context["attempt_dir"])).resolve()
        request_path = attempt_dir / f"{step_id}.request.json"
        request_path.write_text(json.dumps(dict(context), indent=2, sort_keys=True), encoding="utf-8")
        command = (
            "python",
            "-m",
            "halo_forge.training_path_certification.certify",
            "--step",
            step_id,
            "--request",
            str(request_path),
        )
        argv, cwd, env, _ = self.runtime.wrap_execution(
            str(context["runtime_revision_id"]),
            command,
            cwd=str(attempt_dir),
            launch_spec={
                "output_dir": str(attempt_dir),
                "runtime_mounts": [str(attempt_dir), str(self.source_root)],
            },
        )
        result = subprocess.run(argv, cwd=cwd, env={**os.environ, **env}, capture_output=True, text=True, check=False, shell=False)
        log_path = attempt_dir / f"{step_id}.log"
        log_path.write_text((result.stdout or "") + ("\n" + result.stderr if result.stderr else ""), encoding="utf-8")
        if result.returncode:
            raise TrainingPathCertificationError(f"{step_id} failed; inspect {log_path}")
        try:
            payload = json.loads((result.stdout or "{}").splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise TrainingPathCertificationError(f"{step_id} returned no structured evidence") from exc
        payload["_log_path"] = str(log_path)
        return payload

    def _publish(
        self,
        certification_id: str,
        path: TrainingPathProfileRevision,
        evidence: Mapping[str, Any],
        attempt_id: str,
    ) -> TrainingPathCertification:
        target = self.root / "training-paths" / certification_id
        target.mkdir(parents=True, exist_ok=True)
        bundle = {
            "format_version": 1,
            "certification_id": certification_id,
            "path_revision": path.to_dict(),
            "runtime": self.runtime.verify(self.get_certification(certification_id).runtime_qualification_id),  # type: ignore[union-attr]
            "evidence": dict(evidence),
            "source_identity_hash": _source_hash(self.source_root),
            "published_at": _now(),
        }
        path_out = target / "certification.json"
        with tempfile.NamedTemporaryFile("w", dir=target, delete=False, encoding="utf-8") as handle:
            json.dump(bundle, handle, indent=2, sort_keys=True)
            temporary = Path(handle.name)
        os.replace(temporary, path_out)
        digest = _file_hash(path_out)
        completed = _now()
        self.conn.execute(
            "UPDATE training_path_certifications SET status='verified',stage='completed',certification_hash=?,evidence_path=?,progress_json=?,completed_at=? WHERE id=?",
            (digest, str(path_out), _json({"current": len(CERTIFICATION_STEPS), "total": len(CERTIFICATION_STEPS)}), completed, certification_id),
        )
        self.conn.execute(
            "UPDATE training_path_certification_attempts SET status='passed',evidence_json=?,completed_at=? WHERE id=?",
            (_json({"certification_hash": digest}), completed, attempt_id),
        )
        self.conn.commit()
        return self.get_certification(certification_id)  # type: ignore[return-value]

    def _finish_failure(self, identifier: str, stage: str, error: str, *, status: str = "failed") -> TrainingPathCertification:
        self.conn.execute(
            "UPDATE training_path_certifications SET status=?,stage=?,error=?,completed_at=? WHERE id=?",
            (status, stage, str(error)[-4000:], _now(), identifier),
        )
        self.conn.commit()
        return self.get_certification(identifier)  # type: ignore[return-value]

    # ---- truthfully derived capability matrix ---------------------

    def capabilities(self, runtime_family: str) -> TrainingPathCertificationMatrix:
        runtime_cap = next(
            (item for item in self.runtime.capabilities() if item.accelerator_family == runtime_family),
            None,
        )
        runtime_ready = bool(runtime_cap and runtime_cap.available)
        values: list[TrainingPathCapability] = []
        for revision in self.list_revisions(runtime_family=runtime_family):
            executor_available = bool(revision.configuration.get("executor_available"))
            certification = None
            latest = None
            if runtime_cap and runtime_cap.runtime_revision_id:
                row = self.conn.execute(
                    """SELECT * FROM training_path_certifications
                       WHERE path_revision_id=? AND runtime_revision_id=? AND status='verified'
                       ORDER BY completed_at DESC LIMIT 1""",
                    (revision.id, runtime_cap.runtime_revision_id),
                ).fetchone()
                if row:
                    candidate = self._certification(row, include_steps=False)
                    certification = candidate if self.verify(candidate.id)["valid"] else None
                latest_row = self.conn.execute(
                    """SELECT * FROM training_path_certifications
                       WHERE path_revision_id=? AND runtime_revision_id=?
                       ORDER BY created_at DESC LIMIT 1""",
                    (revision.id, runtime_cap.runtime_revision_id),
                ).fetchone()
                latest = self._certification(latest_row, include_steps=False) if latest_row else None
            if not executor_available:
                state, display, summary, blocker, action = (
                    "unavailable",
                    "Not available yet",
                    "This path has an immutable profile, but its real trainer certification executor has not shipped.",
                    "A generic tensor check cannot substitute for the missing real Dataset Lab and trainer-path executor.",
                    CertificationRecoveryAction(
                        "unavailable",
                        "Not available yet",
                        "Use a currently verified path or wait for this executor to be implemented and certified.",
                        enabled=False,
                    ),
                )
            elif certification:
                state, display, summary, blocker, action = (
                    "path_verified",
                    "Verified",
                    "This exact trainer path changed weights and produced a reloadable artifact.",
                    None,
                    None,
                )
            elif latest and latest.status in {"queued", "running", "waiting_for_accelerator"}:
                state, display, summary, blocker, action = (
                    "verification_in_progress",
                    "Verification in progress",
                    (
                        "Waiting for the accelerator without consuming a retry."
                        if latest.status == "waiting_for_accelerator"
                        else "Halo Forge is running the pinned fixture through the real trainer."
                    ),
                    latest.error,
                    CertificationRecoveryAction("open_activity", "View progress", "Open the current certification in Activity."),
                )
            elif runtime_ready:
                failed_reason = (
                    latest.error
                    if latest and latest.status in {"failed", "cancelled", "stale", "blocked"}
                    else None
                )
                state, display, summary, blocker, action = (
                    "runtime_ready",
                    "Needs attention" if failed_reason else "Needs verification",
                    (
                        "The previous real-trainer verification did not complete successfully."
                        if failed_reason
                        else "The accelerator runtime works, but this real trainer path has not been verified yet."
                    ),
                    failed_reason or "No current real-trainer certification exists for this exact path.",
                    CertificationRecoveryAction(
                        "retry" if failed_reason else "certify",
                        "Retry verification" if failed_reason else "Verify this training path",
                        failed_reason or "Run the pinned fixture through the real trainer.",
                    ),
                )
            else:
                state, display, summary, blocker, action = (
                    "unavailable",
                    "Needs attention",
                    "Prepare and verify the accelerator runtime before checking this path.",
                    runtime_cap.unavailable_reason if runtime_cap else "No managed runtime is available.",
                    CertificationRecoveryAction(
                        "prepare_runtime",
                        f"Prepare {'AMD' if runtime_family == 'rocm' else 'NVIDIA'} training",
                        "Core runtime qualification is required first.",
                    ),
                )
            values.append(
                TrainingPathCapability(
                    revision.id,
                    revision.profile_id,
                    str(self.conn.execute("SELECT name FROM training_path_profiles WHERE id=?", (revision.profile_id,)).fetchone()["name"]),
                    revision.scenario_revision_id,
                    revision.trainer_mode,
                    revision.model_id,
                    revision.runtime_family,
                    state,
                    display,
                    summary,
                    runtime_cap.runtime_revision_id if runtime_cap else None,
                    runtime_cap.qualification_id if runtime_cap else None,
                    certification.id if certification else latest.id if latest else None,
                    blocker,
                    action,
                )
            )
        beta = False
        beta_rows = self.conn.execute(
            "SELECT id,runtime_revision_id FROM workstation_certifications "
            "WHERE status='beta_qualified' ORDER BY completed_at DESC"
        ).fetchall()
        for beta_row in beta_rows:
            runtime_revision = self.runtime.get_revision(str(beta_row["runtime_revision_id"]))
            runtime_profile = (
                self.runtime.get_profile(runtime_revision.profile_id)
                if runtime_revision is not None
                else None
            )
            if (
                runtime_profile is not None
                and runtime_profile.accelerator_family == runtime_family
                and self.verify_workstation_certification(str(beta_row["id"]))["valid"]
            ):
                beta = True
                break
        recommended = next((value.path_revision_id for value in values if value.trainer_mode == "sft" and "instruction" in value.profile_id), None)
        return TrainingPathCertificationMatrix(runtime_family, runtime_ready, beta, tuple(values), recommended)

    def verify(self, certification_id: str) -> dict[str, Any]:
        value = self.get_certification(certification_id)
        if value is None:
            raise KeyError(certification_id)
        issues: list[str] = []
        if value.status != "verified":
            issues.append("certification is not complete")
        path = self.get_revision(value.path_revision_id)
        if path is None:
            issues.append("training path revision is missing")
        runtime = self.runtime.verify(value.runtime_qualification_id)
        if not runtime["valid"]:
            issues.append("runtime qualification is stale or corrupt")
        current_source = _source_hash(self.source_root)
        if current_source != value.source_identity_hash:
            issues.append("Halo Forge trainer source changed")
        if not value.evidence_path or not Path(value.evidence_path).is_file():
            issues.append("certification bundle is missing")
        elif value.certification_hash != _file_hash(Path(value.evidence_path)):
            issues.append("certification bundle checksum changed")
        required = {step for step, _ in CERTIFICATION_STEPS}
        passed = {step.step_id for step in value.steps if step.status == "passed"}
        if required - passed:
            issues.append("one or more real-path steps did not pass")
        return {"valid": not issues, "stale": any("changed" in issue or "stale" in issue for issue in issues), "issues": issues, "certification": value.to_dict()}

    # ---- beta snapshot / evidence report ---------------------------

    def workstation_certify(
        self,
        runtime_revision_id: str,
        *,
        evidence: Optional[Mapping[str, Any]] = None,
        enqueue: bool = False,
    ) -> WorkstationCertification:
        runtime_qualification = self.runtime.latest_qualification(runtime_revision_id)
        if runtime_qualification is None or not self.runtime.verify(runtime_qualification.id)["valid"]:
            raise TrainingPathCertificationError("A current core runtime qualification is required")
        runtime_revision = self.runtime.get_revision(runtime_revision_id)
        assert runtime_revision is not None
        family = self.runtime.get_profile(runtime_revision.profile_id).accelerator_family  # type: ignore[union-attr]
        matrix = self.capabilities(family)
        instruction = next((item for item in matrix.paths if item.path_revision_id == matrix.recommended_path_revision_id), None)
        if instruction is None:
            raise TrainingPathCertificationError("The recommended instruction path is missing")
        if instruction.state != "path_verified" or not instruction.certification_id:
            raise TrainingPathCertificationError(
                "A current real instruction-SFT path certification is required"
            )
        supplied = dict(evidence or {})
        created = _now()
        identifier = f"workstation-certification-{_hash({'runtime': runtime_revision_id, 'evidence': supplied, 'created': created})[:24]}"
        self.conn.execute(
            """INSERT INTO workstation_certifications
               (id,runtime_revision_id,runtime_qualification_id,instruction_path_revision_id,
                instruction_path_certification_id,status,stage,host_identity_hash,
                device_identity_hash,evidence_json,qualification_hash,report_path,
                progress_json,resume_cursor_json,created_at,completed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (identifier, runtime_revision_id, runtime_qualification.id, instruction.path_revision_id, instruction.certification_id, "queued", "queued", runtime_qualification.host_identity_hash, runtime_qualification.device_identity_hash, _json(supplied), None, None, _json({"requirements_complete": 0, "requirements_total": 11}), _json({}), created, None),
        )
        self.conn.commit()
        if enqueue:
            item = self._scheduler().enqueue(
                kind="workstation_certification",
                launch_spec={
                    "handler": "training_path_certification.execute_work_item",
                    "operation": "workstation_certify",
                    "certification_root": str(self.root),
                    "runtime_root": str(self.runtime.root),
                    "source_root": str(self.source_root),
                },
                resource_class="none",
                domain_kind="workstation_certification",
                domain_id=identifier,
                max_retries=1,
            )
            self.conn.execute(
                "UPDATE workstation_certifications SET work_item_id=? WHERE id=?",
                (item.id, identifier),
            )
            self.conn.commit()
            return self.get_workstation_certification(identifier)  # type: ignore[return-value]
        return self.run_workstation_certification(identifier)

    def _work_events(self, work_item_id: str) -> list[Any]:
        return self.conn.execute(
            "SELECT event_type,created_at FROM work_item_events WHERE work_item_id=? ORDER BY sequence",
            (work_item_id,),
        ).fetchall()

    def _validate_workstation_evidence(
        self,
        value: WorkstationCertification,
    ) -> tuple[dict[str, bool], dict[str, Any]]:
        supplied = dict(value.evidence or {})
        resolved: dict[str, Any] = {"submitted_ids": supplied}
        requirements: dict[str, bool] = {
            "runtime_core": bool(self.runtime.verify(value.runtime_qualification_id)["valid"]),
            "instruction_sft_path": False,
            "managed_capacity_check": False,
            "own_data_proof": False,
            "parameter_hash_delta": False,
            "artifact_reload": False,
            "outcome_assessment": False,
            "scheduler_restart_recovery": False,
            "external_workload_waiting": False,
            "twelve_hour_soak": False,
        }

        path = self.get_certification(str(value.instruction_path_certification_id or ""))
        if path is not None:
            path_check = self.verify(path.id)
            requirements["instruction_sft_path"] = bool(path_check["valid"])
            resolved["training_path"] = {
                "certification_id": path.id,
                "certification_hash": path.certification_hash,
                "valid": bool(path_check["valid"]),
            }

        proof_run_id = str(supplied.get("proof_run_id") or "").strip()
        run = self.database.get_run(proof_run_id) if proof_run_id else None
        plan_binding = None
        capacity = None
        dataset = None
        if run is not None:
            plan_binding = self.conn.execute(
                """SELECT rtp.capacity_check_id,rtp.role,tpr.dataset_version_id,
                          tpr.training_path_revision_id,tpr.training_path_certification_id
                   FROM run_training_plans rtp
                   JOIN training_plan_revisions tpr ON tpr.id=rtp.plan_revision_id
                   WHERE rtp.run_id=?""",
                (proof_run_id,),
            ).fetchone()
        if plan_binding is not None:
            capacity_id = str(
                supplied.get("capacity_check_id") or plan_binding["capacity_check_id"] or ""
            )
            capacity = self.conn.execute(
                "SELECT * FROM training_capacity_checks WHERE id=?", (capacity_id,)
            ).fetchone()
            dataset = self.conn.execute(
                "SELECT id,status,content_hash FROM dataset_versions WHERE id=?",
                (plan_binding["dataset_version_id"],),
            ).fetchone()
        requirements["managed_capacity_check"] = bool(
            capacity
            and str(capacity["status"]) in {"ready", "ready_with_adjustment"}
            and str(plan_binding["training_path_certification_id"] or "")
            == str(value.instruction_path_certification_id or "")
        )
        raw = run.raw if run is not None else {}
        parameter = dict(raw.get("parameter_evidence") or {})
        requirements["parameter_hash_delta"] = bool(
            parameter.get("algorithm") == "sha256-trainable-tensors-v1"
            and parameter.get("before")
            and parameter.get("after")
            and parameter.get("before") != parameter.get("after")
            and parameter.get("changed") is True
        )
        requirements["own_data_proof"] = bool(
            run
            and run.status == "completed"
            and run.weights_updated
            and plan_binding
            and str(plan_binding["role"]) == "proof"
            and dataset
            and str(dataset["status"]) == "completed"
            and requirements["managed_capacity_check"]
        )
        artifact = self.conn.execute(
            """SELECT id,artifact_hash,verification_status,path FROM model_artifacts
               WHERE run_id=? AND artifact_kind IN ('adapter','final_model')
               ORDER BY created_at DESC LIMIT 1""",
            (proof_run_id,),
        ).fetchone() if proof_run_id else None
        requirements["artifact_reload"] = bool(
            artifact
            and str(artifact["verification_status"]) in {"verified", "reload_verified"}
            and Path(str(artifact["path"])).exists()
        )
        if run is not None:
            resolved["proof_run"] = {
                "run_id": proof_run_id,
                "weights_updated": bool(run.weights_updated),
                "parameter_evidence": parameter,
                "dataset_version_id": str(plan_binding["dataset_version_id"]) if plan_binding else None,
                "dataset_content_hash": str(dataset["content_hash"]) if dataset else None,
                "capacity_check_id": str(capacity["id"]) if capacity else None,
                "artifact_id": str(artifact["id"]) if artifact else None,
                "artifact_hash": str(artifact["artifact_hash"]) if artifact else None,
            }

        assessment_id = str(supplied.get("outcome_assessment_id") or "").strip()
        if not assessment_id and proof_run_id:
            row = self.conn.execute(
                "SELECT id FROM training_outcome_assessments WHERE proof_run_id=? ORDER BY created_at DESC LIMIT 1",
                (proof_run_id,),
            ).fetchone()
            assessment_id = str(row["id"]) if row else ""
        assessment = self.conn.execute(
            "SELECT * FROM training_outcome_assessments WHERE id=?", (assessment_id,)
        ).fetchone() if assessment_id else None
        requirements["outcome_assessment"] = bool(
            assessment
            and str(assessment["proof_run_id"]) == proof_run_id
            and str(assessment["status"])
            in {"improved", "regressed", "mixed", "no_clear_change"}
            and assessment["base_evaluation_id"]
            and assessment["candidate_evaluation_id"]
        )
        resolved["outcome_assessment_id"] = assessment_id or None

        recovery_id = str(supplied.get("scheduler_recovery_work_item_id") or "").strip()
        recovery_events = self._work_events(recovery_id) if recovery_id else []
        recovery_types = {str(row["event_type"]) for row in recovery_events}
        requirements["scheduler_restart_recovery"] = bool(
            recovery_types.intersection({"interrupted", "needs_reconciliation"})
            and "completed" in recovery_types
        )
        wait_id = str(supplied.get("external_wait_work_item_id") or "").strip()
        preflights = self.conn.execute(
            "SELECT decision,evidence_hash FROM accelerator_preflight_decisions WHERE work_item_id=? ORDER BY created_at",
            (wait_id,),
        ).fetchall() if wait_id else []
        decisions = [str(row["decision"]) for row in preflights]
        requirements["external_workload_waiting"] = bool(
            any(item in {"waiting", "contention"} for item in decisions)
            and "idle" in decisions
            and decisions.index("idle") > min(
                index for index, item in enumerate(decisions) if item in {"waiting", "contention"}
            )
        ) if decisions and any(item in {"waiting", "contention"} for item in decisions) else False
        resolved["coexistence"] = {
            "scheduler_recovery_work_item_id": recovery_id or None,
            "recovery_events": sorted(recovery_types),
            "external_wait_work_item_id": wait_id or None,
            "preflight_decisions": decisions,
        }

        soak_ids = [str(item) for item in supplied.get("soak_work_item_ids") or [] if str(item)]
        proof_ids = [str(item) for item in supplied.get("sequential_proof_run_ids") or [] if str(item)]
        timestamps: list[datetime] = []
        for work_id in soak_ids:
            for event in self._work_events(work_id):
                try:
                    timestamps.append(datetime.fromisoformat(str(event["created_at"])))
                except ValueError:
                    pass
        proof_rows = [self.database.get_run(run_id) for run_id in proof_ids]
        duration_seconds = (
            (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) >= 2 else 0.0
        )
        requirements["twelve_hour_soak"] = bool(
            soak_ids
            and 1 <= len(proof_ids) <= 3
            and all(item is not None and item.status == "completed" for item in proof_rows)
            and duration_seconds >= 12 * 60 * 60
        )
        resolved["soak"] = {
            "work_item_ids": soak_ids,
            "proof_run_ids": proof_ids,
            "event_window_seconds": duration_seconds,
        }
        return requirements, resolved

    def run_workstation_certification(self, identifier: str) -> WorkstationCertification:
        value = self.get_workstation_certification(identifier)
        if value is None:
            raise KeyError(identifier)
        self.conn.execute(
            "UPDATE workstation_certifications SET status='running',stage='validating',error=NULL WHERE id=?",
            (identifier,),
        )
        self.conn.commit()
        requirements, resolved = self._validate_workstation_evidence(value)
        support_bundle_id = None
        try:
            from halo_forge.product_lab import ProductLabService

            support = ProductLabService(self.database, root=self.root.parent).create_support_bundle(
                enqueue=False
            )
            support_bundle_id = support.id
        except Exception:
            support_bundle_id = None
        requirements["privacy_safe_support_bundle"] = bool(support_bundle_id)
        resolved["support_bundle_id"] = support_bundle_id
        status = "beta_qualified" if all(requirements.values()) else "incomplete"
        family = self.runtime.get_profile(
            self.runtime.get_revision(value.runtime_revision_id).profile_id  # type: ignore[union-attr]
        ).accelerator_family  # type: ignore[union-attr]
        report = {
            "format_version": 1,
            "certification_id": identifier,
            "status": status,
            "platform_support": "local_verified" if family == "rocm" else "not_hardware_qualified",
            "requirements": requirements,
            "resolved_evidence": resolved,
            "runtime_qualification_id": value.runtime_qualification_id,
            "training_path_certification_id": value.instruction_path_certification_id,
        }
        parent = self.root / "workstations"
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{identifier}-", dir=parent))
        final = parent / identifier
        json_path = staging / "qualification.json"
        markdown_path = staging / "qualification.md"
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lines = [
            "# Halo Forge workstation qualification",
            "",
            f"Status: **{status.replace('_', ' ')}**",
            "",
            "## Evidence gates",
            "",
        ]
        lines.extend(
            f"- {'PASS' if passed else 'MISSING'} — {name.replace('_', ' ')}"
            for name, passed in requirements.items()
        )
        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest = {
            "format_version": 1,
            "files": {
                "qualification.json": _file_hash(json_path),
                "qualification.md": _file_hash(markdown_path),
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if final.exists():
            raise TrainingPathCertificationError("workstation certification bundle already exists")
        os.replace(staging, final)
        completed = _now()
        self.conn.execute(
            """UPDATE workstation_certifications
               SET status=?,stage='completed',evidence_json=?,qualification_hash=?,
                   report_path=?,support_bundle_id=?,progress_json=?,completed_at=?
               WHERE id=?""",
            (
                status,
                _json(report),
                _file_hash(final / "manifest.json"),
                str(final / "manifest.json"),
                support_bundle_id,
                _json({"requirements_complete": sum(requirements.values()), "requirements_total": len(requirements)}),
                completed,
                identifier,
            ),
        )
        self.conn.commit()
        return self.get_workstation_certification(identifier)  # type: ignore[return-value]

    # ---- read/retry/cancel -----------------------------------------

    def get_certification(self, identifier: str) -> Optional[TrainingPathCertification]:
        row = self.conn.execute("SELECT * FROM training_path_certifications WHERE id=?", (identifier,)).fetchone()
        return self._certification(row) if row else None

    def list_certifications(self, *, path_revision_id: Optional[str] = None) -> tuple[TrainingPathCertification, ...]:
        if path_revision_id:
            rows = self.conn.execute("SELECT * FROM training_path_certifications WHERE path_revision_id=? ORDER BY created_at DESC", (path_revision_id,)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM training_path_certifications ORDER BY created_at DESC").fetchall()
        return tuple(self._certification(row, include_steps=False) for row in rows)

    def cancel(self, identifier: str) -> TrainingPathCertification:
        value = self.get_certification(identifier)
        if value is None:
            raise KeyError(identifier)
        self.conn.execute("UPDATE training_path_certifications SET cancel_requested=1 WHERE id=?", (identifier,))
        self.conn.commit()
        if value.work_item_id and self.scheduler:
            self.scheduler.cancel(value.work_item_id)
        return self.get_certification(identifier)  # type: ignore[return-value]

    def retry(self, identifier: str, *, reason: str) -> TrainingPathCertification:
        value = self.get_certification(identifier)
        if value is None:
            raise KeyError(identifier)
        if value.status not in {"failed", "cancelled", "blocked", "stale", "waiting_for_accelerator"}:
            raise TrainingPathCertificationError("Only unfinished certifications can be retried")
        retained = str(reason or "").strip()
        if not retained:
            raise TrainingPathCertificationError("A retry reason is required")
        self.conn.execute("UPDATE training_path_certifications SET status='queued',stage='queued',error=NULL,cancel_requested=0,completed_at=NULL,progress_json=? WHERE id=?", (_json({"retry_reason": retained}), identifier))
        self.conn.commit()
        if value.work_item_id:
            self._scheduler().retry(value.work_item_id, reason=retained, force=True, sync_domain=False)
        return self.get_certification(identifier)  # type: ignore[return-value]

    def get_workstation_certification(self, identifier: str) -> Optional[WorkstationCertification]:
        row = self.conn.execute("SELECT * FROM workstation_certifications WHERE id=?", (identifier,)).fetchone()
        if not row:
            return None
        return WorkstationCertification(str(row["id"]), str(row["runtime_revision_id"]), str(row["runtime_qualification_id"]), str(row["instruction_path_revision_id"]), row["instruction_path_certification_id"], str(row["status"]), str(row["stage"]), str(row["host_identity_hash"]), str(row["device_identity_hash"]), _loads(row["evidence_json"], {}), row["qualification_hash"], row["report_path"], row["support_bundle_id"], _loads(row["progress_json"], {}), _loads(row["resume_cursor_json"], {}), row["work_item_id"], row["error"], str(row["created_at"]), row["completed_at"])

    def verify_workstation_certification(self, identifier: str) -> dict[str, Any]:
        value = self.get_workstation_certification(identifier)
        if value is None:
            raise KeyError(identifier)
        issues: list[str] = []
        if value.status not in {"beta_qualified", "incomplete"}:
            issues.append("workstation certification is not complete")
        if not self.runtime.verify(value.runtime_qualification_id)["valid"]:
            issues.append("runtime core qualification is stale or corrupt")
        if not value.instruction_path_certification_id:
            issues.append("instruction training-path certification is missing")
        elif not self.verify(value.instruction_path_certification_id)["valid"]:
            issues.append("instruction training-path certification is stale or corrupt")
        manifest_path = Path(str(value.report_path or ""))
        if not manifest_path.is_file():
            issues.append("workstation report manifest is missing")
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if value.qualification_hash != _file_hash(manifest_path):
                    issues.append("workstation report manifest checksum changed")
                for name, expected in dict(manifest.get("files") or {}).items():
                    path = manifest_path.parent / str(name)
                    if not path.is_file() or _file_hash(path) != str(expected):
                        issues.append(f"workstation report file failed integrity: {name}")
            except (OSError, ValueError, json.JSONDecodeError):
                issues.append("workstation report manifest is unreadable")
        requirements = dict(value.evidence.get("requirements") or {})
        if value.status == "beta_qualified" and not requirements:
            issues.append("beta qualification has no resolved evidence gates")
        elif value.status == "beta_qualified" and not all(requirements.values()):
            issues.append("beta qualification contains an incomplete evidence gate")
        return {
            "valid": not issues,
            "stale": any("stale" in issue or "changed" in issue for issue in issues),
            "issues": issues,
            "certification": value.to_dict(),
        }

    def execute_work_item(self, item: Any) -> Mapping[str, Any]:
        operation = str(item.launch_spec.get("operation") or "")
        if operation == "certify":
            return self.run_certification(str(item.domain_id)).to_dict()
        if operation == "workstation_certify":
            return self.run_workstation_certification(str(item.domain_id)).to_dict()
        raise TrainingPathCertificationError(f"Unknown certification operation: {operation}")

    def _scheduler(self) -> Any:
        if self.scheduler is None:
            from halo_forge.workstation_jobs import WorkstationScheduler

            self.scheduler = WorkstationScheduler(self.database)
        return self.scheduler

    @staticmethod
    def _revision(row: Any) -> TrainingPathProfileRevision:
        return TrainingPathProfileRevision(str(row["id"]), str(row["profile_id"]), int(row["revision_number"]), str(row["content_hash"]), str(row["runtime_family"]), str(row["backend"]), row["scenario_revision_id"], str(row["trainer_mode"]), str(row["model_id"]), str(row["model_revision"]), str(row["tokenizer_processor_hash"]), str(row["fixture_id"]), str(row["fixture_hash"]), str(row["trainer_adapter_version"]), str(row["capacity_adapter_version"]), _loads(row["configuration_json"], {}), tuple(_loads(row["expected_artifacts_json"], [])), str(row["created_at"]))

    def _certification(self, row: Any, *, include_steps: bool = True) -> TrainingPathCertification:
        steps: tuple[TrainingPathCertificationStep, ...] = ()
        if include_steps:
            values = self.conn.execute("SELECT * FROM training_path_certification_steps WHERE certification_id=? ORDER BY ordinal", (row["id"],)).fetchall()
            steps = tuple(TrainingPathCertificationStep(str(value["certification_id"]), int(value["ordinal"]), str(value["step_id"]), str(value["label"]), str(value["status"]), value["input_hash"], _loads(value["result_json"], {}), value["evidence_hash"], value["log_path"], value["started_at"], value["completed_at"]) for value in values)
        return TrainingPathCertification(str(row["id"]), str(row["path_revision_id"]), str(row["runtime_revision_id"]), str(row["runtime_qualification_id"]), str(row["status"]), str(row["stage"]), str(row["host_identity_hash"]), str(row["device_identity_hash"]), str(row["runtime_identity_hash"]), str(row["source_identity_hash"]), row["certification_hash"], row["evidence_path"], _loads(row["progress_json"], {}), _loads(row["resume_cursor_json"], {}), row["work_item_id"], row["error"], str(row["created_at"]), row["completed_at"], steps)
