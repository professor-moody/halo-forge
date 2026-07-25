"""Standalone public API app for the user-facing halo-forge frontend."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .service import PublicApiService
from halo_forge.version import DISPLAY_VERSION

LOGGER = logging.getLogger(__name__)

FASTAPI_IMPORT_ERROR: Exception | None = None

try:
    from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
except ImportError as exc:  # pragma: no cover - exercised in environments without FastAPI
    FASTAPI_IMPORT_ERROR = exc
    APIRouter = Depends = FastAPI = HTTPException = Query = Request = StreamingResponse = None  # type: ignore[assignment]
    CORSMiddleware = None  # type: ignore[assignment]
    FileResponse = JSONResponse = None  # type: ignore[assignment]


_DOMAIN_ERROR_PACKAGE = "halo_forge"
_INTERNAL_ERROR_DETAIL = "Internal server error"


def is_domain_error(exc: BaseException) -> bool:
    """Return True when ``exc`` belongs to halo-forge's own exception hierarchy.

    Each lab service declares its own exception types (``ReviewLabError``,
    ``ProductLabError``, ``DatasetLabError``, ``FutureLabError``, ...). They share no
    single base class, but they are all defined inside the ``halo_forge`` package, and
    only they describe a condition the caller can act on — so only they may be mapped
    onto a 4xx response carrying their own message. Everything else (``AttributeError``,
    ``RuntimeError``, ``OSError``, ``MemoryError``, ``sqlite3.OperationalError``, ...)
    is an internal fault that must surface as a 500 with no internal detail.
    """
    for klass in type(exc).__mro__:
        module = getattr(klass, "__module__", "") or ""
        if module == _DOMAIN_ERROR_PACKAGE or module.startswith(_DOMAIN_ERROR_PACKAGE + "."):
            return True
    return False


def find_frontend_dist(frontend_dist: str | Path | None = None) -> Path | None:
    """Return the built dashboard asset directory when it is available."""
    candidates: list[Path] = []
    if frontend_dist is not None:
        candidates.append(Path(frontend_dist))
    if env_dist := os.environ.get("HALO_FORGE_FRONTEND_DIST"):
        candidates.append(Path(env_dist))

    public_api_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            public_api_dir / "frontend",
            public_api_dir.parent.parent / "public_app" / "dist",
        ]
    )
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate.resolve()
    return None


def create_app(
    *,
    frontend_dist: str | Path | None = None,
    serve_frontend: bool = True,
) -> "FastAPI":
    """Create the standalone public API app."""
    if FastAPI is None:
        raise RuntimeError(
            "FastAPI is required for the public API. Install halo-forge with FastAPI/uvicorn dependencies."
        ) from FASTAPI_IMPORT_ERROR

    service = PublicApiService()
    api = FastAPI(
        title="halo-forge public API",
        version=DISPLAY_VERSION,
        docs_url="/api/public/docs",
        openapi_url="/api/public/openapi.json",
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Track P0 — any exception that is not a recognized domain error is an internal
    # fault: log it with its traceback and return an opaque 500 rather than echoing
    # the message (and a misleading 4xx) back to the caller.
    @api.exception_handler(Exception)
    async def _handle_internal_error(request: Request, exc: Exception) -> JSONResponse:
        """Log the traceback server-side and return an opaque 500 to the caller."""

        LOGGER.exception(
            "Unhandled error serving %s %s", request.method, request.url.path, exc_info=exc
        )
        return JSONResponse(status_code=500, content={"detail": _INTERNAL_ERROR_DETAIL})

    managed_supervisor = None

    @api.on_event("startup")
    async def start_managed_worker() -> None:
        nonlocal managed_supervisor
        disabled = str(os.environ.get("HALOFORGE_DISABLE_AUTO_WORKER") or "").lower()
        if disabled in {"1", "true", "yes", "on"}:
            return
        from halo_forge.workstation_jobs.supervisor import get_worker_supervisor

        managed_supervisor = get_worker_supervisor(service._dataset_database()).start()

    @api.on_event("shutdown")
    async def stop_managed_worker() -> None:
        if managed_supervisor is not None:
            managed_supervisor.stop()

    # Track P1 — bearer-token auth gate. Loopback requests pass through;
    # non-loopback requests need an Authorization: Bearer <token> header
    # validated against the token store (or HALOFORGE_API_TOKEN).
    from halo_forge.auth.dependency import require_token

    async def _auth_gate(request: Request) -> str:
        return require_token(request)

    router = APIRouter(
        prefix="/api/public",
        tags=["public"],
        dependencies=[Depends(_auth_gate)],
    )

    def _raise_review_error(exc: Exception) -> None:
        """Map Review Studio domain failures without coupling FastAPI to its internals."""

        name = exc.__class__.__name__
        if isinstance(exc, KeyError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if name in {"ReviewConflictError", "ReviewStateError", "ReviewIntegrityError"}:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if name == "ReviewEligibilityError":
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if isinstance(exc, (TypeError, ValueError)) or name.endswith("ValidationError"):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if is_domain_error(exc):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise exc

    def _raise_verifier_error(exc: Exception) -> None:
        """Map verifier identity, eligibility, and lifecycle failures."""

        if isinstance(exc, KeyError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        name = exc.__class__.__name__
        if "Protected" in name or "Eligibility" in name:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if isinstance(exc, (TypeError, ValueError)):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if is_domain_error(exc):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise exc

    def _raise_future_lab_error(exc: Exception) -> None:
        """Map the V11-V15 research workflow failures consistently."""

        if isinstance(exc, KeyError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        name = exc.__class__.__name__
        if "Eligibility" in name or "Protected" in name:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if isinstance(exc, (TypeError, ValueError)):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if is_domain_error(exc):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise exc

    def _raise_product_lab_error(exc: Exception) -> None:
        """Map V17 validation, lifecycle, and integrity failures."""

        if isinstance(exc, KeyError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if isinstance(exc, (TypeError, ValueError)):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if is_domain_error(exc):
            detail = str(exc)
            if detail.startswith("Unknown "):
                raise HTTPException(status_code=404, detail=detail) from exc
            raise HTTPException(status_code=409, detail=detail) from exc
        raise exc

    @router.get("/health")
    async def health() -> Dict[str, Any]:
        return {"ok": True}

    @router.get("/version")
    async def version() -> Dict[str, Any]:
        return service.get_version_info()

    @router.get("/backend")
    async def backend_info() -> Dict[str, Any]:
        return service.get_backend_info()

    @router.get("/workspace")
    async def workspace_info() -> Dict[str, Any]:
        return service.get_workspace_info()

    @router.get("/spec-descriptors/{kind}")
    async def list_spec_descriptors(kind: str) -> Dict[str, Any]:
        try:
            return service.list_spec_descriptors(kind)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/spec-descriptors/{kind}/{descriptor_id}/validate")
    async def validate_spec_descriptor(
        kind: str, descriptor_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.validate_spec_descriptor(kind, descriptor_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Labs V11-V15 share one transport-neutral implementation. These routes
    # intentionally use the same bounded list shape as the rest of the product.

    @router.get("/lab-capabilities")
    async def future_lab_capabilities() -> Dict[str, Any]:
        return service.get_future_lab_capabilities()

    @router.get("/guidance/{context_kind}/{context_id}")
    async def actionable_guidance(
        context_kind: str, context_id: str
    ) -> Dict[str, Any]:
        try:
            return service.get_actionable_guidance(context_kind, context_id)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.get("/outcome/profiles")
    async def list_outcome_profiles(
        scenario_revision_id: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_outcome_profiles(
            scenario_revision_id=scenario_revision_id,
            limit=limit,
            offset=offset,
        )

    @router.get("/outcome/assessments")
    async def list_outcome_assessments(
        proof_run_id: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_training_outcomes(
            proof_run_id=proof_run_id, limit=limit, offset=offset
        )

    @router.post("/outcome/assessments/{proof_run_id}", status_code=201)
    async def create_outcome_assessment(
        proof_run_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.assess_training_outcome(proof_run_id, payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post("/outcome/runs/{proof_run_id}/prepare", status_code=202)
    async def prepare_outcome_assessment(
        proof_run_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.prepare_training_outcome(proof_run_id, payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.get("/outcome/assessments/{assessment_id}")
    async def get_outcome_assessment(assessment_id: str) -> Dict[str, Any]:
        value = service.get_training_outcome(assessment_id)
        if value is None:
            raise HTTPException(status_code=404, detail="outcome assessment not found")
        return value

    @router.get("/outcome/assessments/{assessment_id}/findings")
    async def list_outcome_findings(
        assessment_id: str,
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_training_outcome_findings(
            assessment_id, limit=limit, offset=offset
        )

    @router.post("/outcome/runs/{proof_run_id}/review", status_code=201)
    async def review_outcome(
        proof_run_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.review_training_outcome(proof_run_id, payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.get("/outcome/runs/{proof_run_id}/full-run-context")
    async def outcome_full_run_context(
        proof_run_id: str,
        assessment_id: Optional[str] = Query(None),
        override_reason: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        try:
            return service.get_full_run_context(
                proof_run_id,
                assessment_id=assessment_id,
                override_reason=override_reason,
            )
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.get("/adaptation-studies")
    async def list_adaptation_studies(
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_adaptation_studies(limit=limit, offset=offset)

    @router.post("/adaptation-studies", status_code=201)
    async def create_adaptation_study(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_adaptation_study(payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.get("/adaptation-studies/{study_id}")
    async def get_adaptation_study(study_id: str) -> Dict[str, Any]:
        value = service.get_adaptation_study(study_id)
        if value is None:
            raise HTTPException(status_code=404, detail="adaptation study not found")
        return value

    @router.post("/adaptation-studies/protocols/validate")
    async def validate_adaptation_protocol(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.validate_adaptation_study_protocol(payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post("/adaptation-studies/{study_id}/protocols", status_code=201)
    async def create_adaptation_protocol(
        study_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.create_adaptation_study_protocol(study_id, payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.get("/adaptation-study-protocols/{revision_id}")
    async def get_adaptation_protocol(revision_id: str) -> Dict[str, Any]:
        value = service.get_adaptation_study_protocol(revision_id)
        if value is None:
            raise HTTPException(status_code=404, detail="study protocol not found")
        return value

    @router.get("/adaptation-study-protocols/{revision_id}/launch-plan")
    async def adaptation_study_launch_plan(revision_id: str) -> Dict[str, Any]:
        try:
            return service.get_adaptation_study_launch_plan(revision_id)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post(
        "/adaptation-study-protocols/{revision_id}/launch",
        status_code=202,
    )
    async def launch_adaptation_study(
        revision_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.launch_adaptation_study(revision_id, payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post("/adaptation-study-protocols/{revision_id}/materialize")
    async def materialize_adaptation_protocol(revision_id: str) -> Dict[str, Any]:
        try:
            return service.materialize_adaptation_study(revision_id)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post("/adaptation-study-assignments/{assignment_id}/run")
    async def attach_adaptation_run(
        assignment_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.attach_adaptation_study_run(assignment_id, payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post("/adaptation-study-protocols/{revision_id}/analyses", status_code=202)
    async def analyze_adaptation_protocol(
        revision_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.analyze_adaptation_study(revision_id, payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post("/adaptation-study-protocols/{revision_id}/deviations", status_code=201)
    async def record_adaptation_deviation(
        revision_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.record_adaptation_study_deviation(revision_id, payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post("/adaptation-study-protocols/{revision_id}/decisions", status_code=201)
    async def record_adaptation_decision(
        revision_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.record_adaptation_study_decision(revision_id, payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.get("/grounding/profiles")
    async def list_grounding_profiles(
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_grounding_profiles(limit=limit, offset=offset)

    @router.post("/grounding/profiles", status_code=201)
    async def create_grounding_profile(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_grounding_profile(payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post("/grounding/profiles/{profile_id}/revisions", status_code=201)
    async def create_grounding_profile_revision(
        profile_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.create_grounding_profile_revision(profile_id, payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.get("/grounding/batches")
    async def list_grounding_batches(
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_grounded_generations(limit=limit, offset=offset)

    @router.post(
        "/grounding/profile-revisions/{revision_id}/preview"
    )
    async def preview_grounding_batch(
        revision_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.preview_grounded_generation(revision_id, payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post("/grounding/profile-revisions/{revision_id}/batches", status_code=202)
    async def launch_grounding_batch(
        revision_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.launch_grounded_generation(revision_id, payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.get("/grounding/batches/{batch_id}")
    async def get_grounding_batch(batch_id: str) -> Dict[str, Any]:
        value = service.get_grounded_generation(batch_id)
        if value is None:
            raise HTTPException(status_code=404, detail="grounded generation not found")
        return value

    @router.get("/grounding/batches/{batch_id}/candidates")
    async def list_grounding_candidates(
        batch_id: str,
        status: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_grounded_candidates(
            batch_id, status=status, limit=limit, offset=offset
        )

    @router.post("/grounding/batches/{batch_id}/review-proposal", status_code=201)
    async def create_grounding_review_proposal(batch_id: str) -> Dict[str, Any]:
        try:
            return service.create_grounding_review_proposal(batch_id)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.get("/specialized-tasks")
    async def list_specialized_tasks() -> Dict[str, Any]:
        return service.list_specialized_tasks()

    @router.post("/specialized-tasks/readiness")
    async def specialized_task_readiness(
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.get_specialized_task_readiness(payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post("/specialized-task-artifacts/verify")
    async def verify_specialized_task_artifact(
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.verify_specialized_task_artifact(
                str(payload.get("artifact_path") or ""),
                payload,
            )
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post("/task-label-schemas", status_code=201)
    async def create_task_label_schema(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_task_label_schema(payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post("/task-label-schemas/{schema_id}/revisions", status_code=201)
    async def create_task_label_schema_revision(
        schema_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.create_task_label_schema_revision(schema_id, payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post("/specialized-tasks/classification/metrics")
    async def specialized_classification_metrics(
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.compute_classification_metrics(payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post("/specialized-tasks/retrieval/metrics")
    async def specialized_retrieval_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.compute_retrieval_metrics(payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.get("/environments")
    async def list_agent_environments(
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_agent_environments(limit=limit, offset=offset)

    @router.post("/environments", status_code=201)
    async def create_agent_environment(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_agent_environment(payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.get("/environments/{environment_id}")
    async def get_agent_environment(environment_id: str) -> Dict[str, Any]:
        value = service.get_agent_environment(environment_id)
        if value is None:
            raise HTTPException(status_code=404, detail="environment not found")
        return value

    @router.post("/environments/{environment_id}/revisions", status_code=201)
    async def create_agent_environment_revision(
        environment_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.create_agent_environment_revision(environment_id, payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post(
        "/environment-revisions/{environment_revision_id}/suites",
        status_code=201,
    )
    async def create_agent_episode_suite(
        environment_revision_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.create_episode_suite(environment_revision_id, payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post("/environment-suites/{suite_id}/revisions", status_code=201)
    async def create_agent_episode_suite_revision(
        suite_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.create_episode_suite_revision(suite_id, payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.get("/environment-episodes")
    async def list_agent_episodes(
        environment_revision_id: Optional[str] = Query(None),
        suite_revision_id: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_agent_episodes(
            environment_revision_id=environment_revision_id,
            suite_revision_id=suite_revision_id,
            limit=limit,
            offset=offset,
        )

    @router.post(
        "/environment-suite-revisions/{suite_revision_id}/episodes",
        status_code=202,
    )
    async def launch_agent_episode(
        suite_revision_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.launch_agent_episode(suite_revision_id, payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.get(
        "/environment-revisions/{environment_revision_id}/permissions"
    )
    async def environment_permission_preview(
        environment_revision_id: str,
    ) -> Dict[str, Any]:
        try:
            return service.get_environment_permission_preview(
                environment_revision_id
            )
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.get("/environment-episodes/{episode_id}")
    async def get_agent_episode(episode_id: str) -> Dict[str, Any]:
        value = service.get_agent_episode(episode_id)
        if value is None:
            raise HTTPException(status_code=404, detail="episode not found")
        return value

    @router.get("/environment-episodes/{episode_id}/steps")
    async def list_agent_episode_steps(
        episode_id: str,
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_agent_episode_steps(
            episode_id, limit=limit, offset=offset
        )

    @router.post("/environment-episodes/{episode_id}/replay")
    async def replay_agent_episode(episode_id: str) -> Dict[str, Any]:
        try:
            return service.replay_agent_episode(episode_id)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post("/environment-episodes/{episode_id}/rerun", status_code=202)
    async def rerun_agent_episode(
        episode_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.rerun_agent_episode(episode_id, payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post("/environment-episodes/compare")
    async def compare_agent_episodes(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.compare_agent_environments(payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.post("/environment-trajectories", status_code=202)
    async def publish_agent_trajectories(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.publish_agent_trajectories(payload)
        except Exception as exc:
            _raise_future_lab_error(exc)
            raise

    @router.get("/huggingface/status")
    async def huggingface_status() -> Dict[str, Any]:
        return service.huggingface_status()

    @router.post("/huggingface/token")
    async def huggingface_save_token(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.huggingface_save_token(payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Guided own-data capabilities, import, inspection, and mapping.
    @router.get("/interface-capabilities")
    async def interface_capabilities() -> Dict[str, Any]:
        return service.list_interface_capabilities()

    @router.get("/training-scenarios")
    async def training_scenarios(
        include_unavailable: bool = Query(False),
        modality: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_training_scenarios(
            include_unavailable=include_unavailable,
            modality=modality,
            limit=limit,
            offset=offset,
        )

    @router.post("/training-scenarios/advise")
    async def advise_training_scenario(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.advise_training_scenario(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/training-scenario-examples")
    async def guided_training_examples() -> Dict[str, Any]:
        return service.list_guided_training_examples()

    @router.get("/training-scenarios/{scenario_id}/examples")
    async def training_scenario_examples(scenario_id: str) -> Dict[str, Any]:
        try:
            return service.list_training_scenario_examples(scenario_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="training scenario not found") from exc

    @router.get("/training-scenarios/{scenario_id}/template")
    async def training_scenario_template(
        scenario_id: str, example_id: Optional[str] = Query(None)
    ) -> Dict[str, Any]:
        try:
            return service.get_training_scenario_template(
                scenario_id, example_id=example_id
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="training scenario template not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/training-scenarios/{scenario_id}")
    async def training_scenario(scenario_id: str) -> Dict[str, Any]:
        try:
            return service.get_training_scenario(scenario_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="training scenario not found") from exc

    @router.get("/dataset-imports")
    async def dataset_imports(
        status: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_dataset_imports(status=status, limit=limit, offset=offset)

    @router.post("/dataset-imports", status_code=201)
    async def create_dataset_import(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_dataset_import(payload)
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            status = 409 if exc.__class__.__name__ == "InsufficientDiskCapacityError" else 400
            detail: Any = str(exc)
            if status == 409 and getattr(exc, "forecast", None):
                detail = {"message": str(exc), "disk_forecast": dict(exc.forecast)}
            raise HTTPException(status_code=status, detail=detail) from exc

    @router.get("/dataset-imports/huggingface/options")
    async def huggingface_dataset_options(
        repo_id: str = Query(..., min_length=1),
        revision: str = Query(..., min_length=1),
    ) -> Dict[str, Any]:
        try:
            return service.get_huggingface_dataset_options(
                repo_id=repo_id, revision=revision
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/dataset-imports/{import_id}")
    async def get_dataset_import(import_id: str) -> Dict[str, Any]:
        item = service.get_dataset_import(import_id)
        if item is None:
            raise HTTPException(status_code=404, detail="dataset import not found")
        return item

    @router.post("/dataset-imports/{import_id}/cancel")
    async def cancel_dataset_import(import_id: str) -> Dict[str, Any]:
        try:
            return service.cancel_dataset_import(import_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset import not found") from exc
        except ValueError as exc:
            detail: Any = str(exc)
            if getattr(exc, "forecast", None):
                detail = {"message": str(exc), "disk_forecast": dict(exc.forecast)}
            raise HTTPException(status_code=409, detail=detail) from exc

    @router.post("/dataset-imports/{import_id}/retry", status_code=202)
    async def retry_dataset_import(import_id: str) -> Dict[str, Any]:
        try:
            return service.retry_dataset_import(import_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset import not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/dataset-imports/{import_id}/files", status_code=201)
    async def create_dataset_import_file(
        import_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.create_dataset_import_file(import_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset import not found") from exc
        except ValueError as exc:
            detail: Any = str(exc)
            if getattr(exc, "forecast", None):
                detail = {"message": str(exc), "disk_forecast": dict(exc.forecast)}
            raise HTTPException(status_code=409, detail=detail) from exc

    @router.put("/dataset-imports/{import_id}/files/{file_id}/content")
    async def upload_dataset_import_file_content(
        import_id: str, file_id: str, request: Request
    ) -> Dict[str, Any]:
        content_range = request.headers.get("content-range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range.strip())
        if match is None:
            raise HTTPException(
                status_code=400,
                detail="Content-Range must use bytes START-END/TOTAL",
            )
        content = await request.body()
        try:
            return service.upload_dataset_import_chunk(
                import_id,
                file_id,
                content,
                start=int(match.group(1)),
                end=int(match.group(2)),
                total=int(match.group(3)),
                chunk_sha256=request.headers.get("x-content-sha256"),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset import file not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/dataset-imports/{import_id}/inspect", status_code=202)
    async def inspect_dataset_import(
        import_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.inspect_dataset_import(import_id, payload or {})
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset import not found") from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/dataset-inspections/{inspection_id}")
    async def get_dataset_inspection(inspection_id: str) -> Dict[str, Any]:
        item = service.get_dataset_source_inspection(inspection_id)
        if item is None:
            raise HTTPException(status_code=404, detail="dataset inspection not found")
        return item

    @router.post("/dataset-inspections/{inspection_id}/cancel")
    async def cancel_dataset_inspection(inspection_id: str) -> Dict[str, Any]:
        try:
            item = service.cancel_dataset_source_inspection(inspection_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="dataset inspection not found")
        return item

    @router.post("/dataset-inspections/{inspection_id}/retry", status_code=202)
    async def retry_dataset_inspection(inspection_id: str) -> Dict[str, Any]:
        try:
            return service.retry_dataset_source_inspection(inspection_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset inspection not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/dataset-inspections/{inspection_id}/mapping-preview")
    async def dataset_mapping_preview(
        inspection_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.preview_dataset_mapping(inspection_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset inspection not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/dataset-inspections/{inspection_id}/semantic-preview")
    async def dataset_semantic_preview(
        inspection_id: str,
        payload: Dict[str, Any],
        limit: int = Query(50, ge=1, le=200),
    ) -> Dict[str, Any]:
        try:
            return service.preview_dataset_semantics(
                inspection_id, payload, limit=limit
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset inspection not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/dataset-inspections/{inspection_id}/readiness")
    async def dataset_inspection_readiness(
        inspection_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.get_dataset_inspection_readiness(
                inspection_id, payload
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset inspection not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/dataset-inspections/{inspection_id}/preparation-preview")
    async def dataset_preparation_preview(
        inspection_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.preview_dataset_preparation(inspection_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset inspection not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/dataset-inspections/{inspection_id}/register", status_code=202)
    async def register_dataset_inspection(
        inspection_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.register_inspected_dataset(inspection_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset inspection not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/dataset-imports/cleanup")
    async def cleanup_dataset_imports(
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            return service.cleanup_dataset_imports(payload)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.delete("/huggingface/token")
    async def huggingface_clear_token() -> Dict[str, Any]:
        return service.huggingface_clear_token()

    @router.post("/huggingface/check-model")
    async def huggingface_check_model(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.huggingface_check_model(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/telemetry")
    async def telemetry() -> Dict[str, Any]:
        return service.get_telemetry()

    @router.get("/telemetry/stream")
    async def stream_telemetry(
        interval: float = Query(default=2.0, ge=0.5, le=30.0),
    ) -> "StreamingResponse":
        return StreamingResponse(
            service.stream_telemetry(interval_seconds=interval),
            media_type="text/event-stream",
        )

    @router.get("/dashboard")
    async def dashboard_summary() -> Dict[str, Any]:
        return service.get_dashboard_summary()

    @router.get("/train/presets")
    async def list_training_presets() -> Dict[str, Any]:
        return {"items": service.list_training_presets()}

    @router.get("/train/datasets")
    async def list_training_datasets() -> Dict[str, Any]:
        return {"items": service.list_training_datasets()}

    @router.get("/train/dataset-versions")
    async def list_training_dataset_versions(
        mode: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        return {"items": service.list_training_dataset_versions(mode=mode)}

    # Human Feedback and Active Data Studio.
    @router.get("/review-capabilities")
    async def review_capabilities() -> Dict[str, Any]:
        return service.get_review_capabilities()

    @router.get("/annotation-schemas")
    async def list_annotation_schemas(
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_annotation_schemas(limit=limit, offset=offset)

    @router.post("/annotation-schemas/validate")
    async def validate_annotation_schema(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.validate_annotation_schema(payload)
        except Exception as exc:
            _raise_review_error(exc)

    @router.post("/annotation-schemas", status_code=201)
    async def create_annotation_schema(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_annotation_schema(payload)
        except Exception as exc:
            _raise_review_error(exc)

    @router.get("/annotation-schemas/{schema_id}")
    async def get_annotation_schema(schema_id: str) -> Dict[str, Any]:
        item = service.get_annotation_schema(schema_id)
        if item is None:
            raise HTTPException(status_code=404, detail="annotation schema not found")
        return item

    @router.get("/annotation-schemas/{schema_id}/revisions")
    async def list_annotation_schema_revisions(
        schema_id: str,
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            return service.list_annotation_schema_revisions(
                schema_id, limit=limit, offset=offset
            )
        except Exception as exc:
            _raise_review_error(exc)

    @router.post("/annotation-schemas/{schema_id}/revisions", status_code=201)
    async def revise_annotation_schema(
        schema_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.revise_annotation_schema(schema_id, payload)
        except Exception as exc:
            _raise_review_error(exc)

    @router.get("/annotation-schema-revisions/{revision_id}")
    async def get_annotation_schema_revision(revision_id: str) -> Dict[str, Any]:
        item = service.get_annotation_schema_revision(revision_id)
        if item is None:
            raise HTTPException(status_code=404, detail="annotation schema revision not found")
        return item

    @router.get("/acquisition-batches")
    async def list_acquisition_batches(
        status: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_acquisition_batches(status=status, limit=limit, offset=offset)

    @router.post("/acquisition-batches", status_code=202)
    async def create_acquisition_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_acquisition_batch(payload)
        except Exception as exc:
            _raise_review_error(exc)

    @router.get("/acquisition-batches/{batch_id}")
    async def get_acquisition_batch(batch_id: str) -> Dict[str, Any]:
        item = service.get_acquisition_batch(batch_id)
        if item is None:
            raise HTTPException(status_code=404, detail="acquisition batch not found")
        return item

    @router.get("/acquisition-batches/{batch_id}/candidates")
    async def list_acquisition_candidates(
        batch_id: str,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            return service.list_acquisition_candidates(batch_id, limit=limit, offset=offset)
        except Exception as exc:
            _raise_review_error(exc)

    @router.post("/acquisition-batches/{batch_id}/cancel")
    async def cancel_acquisition_batch(batch_id: str) -> Dict[str, Any]:
        try:
            return service.cancel_acquisition_batch(batch_id)
        except Exception as exc:
            _raise_review_error(exc)

    @router.post("/acquisition-batches/{batch_id}/retry", status_code=202)
    async def retry_acquisition_batch(batch_id: str) -> Dict[str, Any]:
        try:
            return service.retry_acquisition_batch(batch_id)
        except Exception as exc:
            _raise_review_error(exc)

    @router.get("/review-queues")
    async def list_review_queues(
        status: Optional[str] = Query(None),
        q: Optional[str] = Query(None, max_length=200),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_review_queues(
            status=status, query=q, limit=limit, offset=offset
        )

    @router.post("/review-queues", status_code=201)
    async def create_review_queue(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_review_queue(payload)
        except Exception as exc:
            _raise_review_error(exc)

    @router.get("/review-queues/summaries")
    async def list_review_queue_summaries(
        status: Optional[str] = Query(None),
        q: Optional[str] = Query(None, max_length=200),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_review_queue_summaries(
            status=status,
            query=q,
            limit=limit,
            offset=offset,
        )

    @router.get("/review-queues/{queue_id}")
    async def get_review_queue(queue_id: str) -> Dict[str, Any]:
        item = service.get_review_queue(queue_id)
        if item is None:
            raise HTTPException(status_code=404, detail="review queue not found")
        return item

    @router.post("/review-queues/{queue_id}/clone", status_code=201)
    async def clone_review_queue(queue_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.clone_review_queue(queue_id, payload)
        except Exception as exc:
            _raise_review_error(exc)

    @router.get("/review-queues/{queue_id}/statistics")
    async def review_queue_statistics(queue_id: str) -> Dict[str, Any]:
        try:
            return service.get_review_queue_statistics(queue_id)
        except Exception as exc:
            _raise_review_error(exc)

    @router.get("/review-queues/{queue_id}/items")
    async def list_review_items(
        queue_id: str,
        status: Optional[str] = Query(None),
        pass_number: Optional[int] = Query(None, ge=1, le=2),
        q: Optional[str] = Query(None, max_length=200),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            return service.list_review_items(
                queue_id,
                status=status,
                pass_number=pass_number,
                query=q,
                limit=limit,
                offset=offset,
            )
        except Exception as exc:
            _raise_review_error(exc)

    @router.post("/review-queues/{queue_id}/event-batches", status_code=201)
    async def submit_review_event_batch(
        queue_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.submit_review_event_batch(queue_id, payload)
        except Exception as exc:
            _raise_review_error(exc)

    @router.post("/review-queues/{queue_id}/label-set-revisions", status_code=202)
    async def publish_review_label_set(
        queue_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.publish_label_set(queue_id, payload or {})
        except Exception as exc:
            _raise_review_error(exc)

    @router.post("/review-queues/{queue_id}/{action}")
    async def update_review_queue_state(
        queue_id: str, action: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if action not in {"pause", "resume", "archive", "start-second-pass"}:
            raise HTTPException(status_code=404, detail="unknown review queue action")
        try:
            return service.update_review_queue_state(
                queue_id,
                action=action,
                reason=str((payload or {}).get("reason") or "") or None,
            )
        except Exception as exc:
            _raise_review_error(exc)

    @router.get("/review-items/{item_id}")
    async def get_review_item(item_id: str) -> Dict[str, Any]:
        item = service.get_review_item(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="review item not found")
        return item

    @router.get("/review-items/{item_id}/neighbors")
    async def get_review_item_neighbors(
        item_id: str,
        status: Optional[str] = Query(None),
        pass_number: Optional[int] = Query(None, ge=1, le=2),
        q: Optional[str] = Query(None, max_length=200),
    ) -> Dict[str, Any]:
        try:
            return service.get_review_item_neighbors(
                item_id, status=status, pass_number=pass_number, query=q
            )
        except Exception as exc:
            _raise_review_error(exc)

    @router.post("/review-items/{item_id}/events", status_code=201)
    async def submit_review_event(item_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if str(payload.get("event_type") or "") in {"adjudicate", "adjudication"}:
                return service.adjudicate_review_item(item_id, payload)
            return service.submit_review_event(item_id, payload)
        except Exception as exc:
            _raise_review_error(exc)

    @router.get("/review-items/{item_id}/events")
    async def list_review_events(
        item_id: str,
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            return service.list_review_events(item_id, limit=limit, offset=offset)
        except Exception as exc:
            _raise_review_error(exc)

    @router.get("/review-items/{item_id}/suggestions")
    async def list_review_suggestions(
        item_id: str,
        pass_number: Optional[int] = Query(None, ge=1, le=2),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            return service.list_review_suggestions(
                item_id, pass_number=pass_number, limit=limit, offset=offset
            )
        except Exception as exc:
            _raise_review_error(exc)

    @router.post("/review-items/{item_id}/suggestions", status_code=202)
    async def generate_review_suggestion(
        item_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.generate_review_suggestions(item_id, payload)
        except Exception as exc:
            _raise_review_error(exc)

    @router.get("/label-sets")
    async def list_label_sets(
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_label_sets(limit=limit, offset=offset)

    @router.get("/label-sets/{label_set_id}")
    async def get_label_set(label_set_id: str) -> Dict[str, Any]:
        item = service.get_label_set(label_set_id)
        if item is None:
            raise HTTPException(status_code=404, detail="label set not found")
        return item

    @router.get("/label-set-revisions/{revision_id}")
    async def get_label_set_revision(revision_id: str) -> Dict[str, Any]:
        item = service.get_label_set_revision(revision_id)
        if item is None:
            raise HTTPException(status_code=404, detail="label-set revision not found")
        return item

    @router.get("/label-set-revisions/{revision_id}/items")
    async def list_label_set_items(
        revision_id: str,
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            return service.list_label_set_items(revision_id, limit=limit, offset=offset)
        except Exception as exc:
            _raise_review_error(exc)

    @router.post("/label-set-revisions/{revision_id}/verify")
    async def verify_label_set_revision(revision_id: str) -> Dict[str, Any]:
        try:
            return service.verify_label_set_revision(revision_id)
        except Exception as exc:
            _raise_review_error(exc)

    @router.post("/label-set-revisions/{revision_id}/dataset-preview")
    async def preview_label_set_dataset(
        revision_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.preview_label_set_dataset(revision_id, payload or {})
        except Exception as exc:
            _raise_review_error(exc)

    @router.post("/label-set-revisions/{revision_id}/dataset-build", status_code=202)
    async def build_label_set_dataset(
        revision_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.build_label_set_dataset(revision_id, payload)
        except Exception as exc:
            _raise_review_error(exc)

    # Dataset Lab — source registration, immutable versions, and persistent jobs.
    @router.get("/datasets")
    async def list_datasets(
        modality: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_datasets(modality=modality, limit=limit, offset=offset)

    @router.post("/datasets")
    async def create_dataset(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_dataset(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            if is_domain_error(exc):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise

    @router.get("/datasets/{dataset_id}")
    async def get_dataset(dataset_id: str) -> Dict[str, Any]:
        item = service.get_dataset(dataset_id)
        if item is None:
            raise HTTPException(status_code=404, detail="dataset not found")
        return item

    @router.post("/dataset-sources/{source_id}/refresh", status_code=202)
    async def refresh_dataset_source_by_id(source_id: str) -> Dict[str, Any]:
        try:
            return service.refresh_dataset_source_by_id(source_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset source not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/datasets/{dataset_id}/preview")
    async def preview_dataset(
        dataset_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=500),
    ) -> Dict[str, Any]:
        try:
            return service.preview_dataset(dataset_id, offset=offset, limit=limit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/datasets/{dataset_id}/sources")
    async def list_dataset_sources(dataset_id: str) -> Dict[str, Any]:
        try:
            return service.list_dataset_sources(dataset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset not found") from exc

    @router.post("/datasets/{dataset_id}/refresh", status_code=202)
    async def refresh_dataset_source(dataset_id: str) -> Dict[str, Any]:
        try:
            return service.request_dataset_source_refresh(dataset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/dataset-sources/{source_id}")
    async def get_dataset_source(source_id: str) -> Dict[str, Any]:
        item = service.get_dataset_source(source_id)
        if item is None:
            raise HTTPException(status_code=404, detail="dataset source not found")
        return item

    @router.get("/dataset-source-assets/{source_id}")
    async def serve_dataset_source_asset(source_id: str, reference: str) -> "FileResponse":
        try:
            return FileResponse(service.dataset_source_asset_path(source_id, reference))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset asset not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/dataset-version-assets/{version_id}")
    async def serve_dataset_version_asset(version_id: str, path: str) -> "FileResponse":
        try:
            return FileResponse(service.dataset_version_asset_path(version_id, path))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset asset not found") from exc

    @router.get("/datasets/{dataset_id}/statistics")
    async def dataset_statistics(dataset_id: str) -> Dict[str, Any]:
        try:
            return service.dataset_statistics(dataset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/datasets/{dataset_id}/profile", status_code=202)
    async def profile_dataset(dataset_id: str) -> Dict[str, Any]:
        try:
            return service.profile_dataset(dataset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/document-extractors")
    async def document_extractors() -> Dict[str, Any]:
        return service.list_document_extractors()

    @router.get("/document-extractions")
    async def list_document_extractions(
        import_id: Optional[str] = Query(None),
        source_id: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_document_extractions(
            import_id=import_id,
            source_id=source_id,
            status=status,
            limit=limit,
            offset=offset,
        )

    @router.post("/document-extractions", status_code=202)
    async def create_document_extraction(
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            return service.create_document_extraction(payload)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="document extraction source not found"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/document-extractions/{extraction_id}")
    async def get_document_extraction(extraction_id: str) -> Dict[str, Any]:
        try:
            return service.get_document_extraction(extraction_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="document extraction not found"
            ) from exc

    @router.post("/document-extractions/{extraction_id}/preview")
    async def preview_document_extraction(
        extraction_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        values = payload or {}
        try:
            return service.preview_document_extraction(
                extraction_id,
                limit=int(values.get("limit", 20)),
                offset=int(values.get("offset", 0)),
                include_text=bool(values.get("include_text", True)),
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="document extraction not found"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/document-extractions/{extraction_id}/verify")
    async def verify_document_extraction(extraction_id: str) -> Dict[str, Any]:
        try:
            return service.verify_document_extraction(extraction_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="document extraction not found"
            ) from exc

    @router.post("/document-extractions/{extraction_id}/cancel")
    async def cancel_document_extraction(extraction_id: str) -> Dict[str, Any]:
        try:
            return service.cancel_document_extraction(extraction_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="document extraction not found"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/document-extractions/{extraction_id}/retry", status_code=202)
    async def retry_document_extraction(extraction_id: str) -> Dict[str, Any]:
        try:
            return service.retry_document_extraction(extraction_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="document extraction not found"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/datasets/{dataset_id}/versions")
    async def list_dataset_versions(dataset_id: str) -> Dict[str, Any]:
        try:
            return service.list_dataset_versions(dataset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset not found") from exc

    @router.post("/datasets/{dataset_id}/build", status_code=202)
    async def build_dataset(dataset_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.build_dataset(dataset_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            if is_domain_error(exc):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise

    @router.post("/dataset-recipes/validate")
    async def validate_dataset_recipe(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from halo_forge.data_lab import Recipe

            recipe = Recipe.from_value(payload.get("recipe", payload))
            return {"valid": True, "recipe": recipe.to_dict(), "recipe_hash": recipe.fingerprint}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/dataset-versions/{version_id}")
    async def get_dataset_version(version_id: str) -> Dict[str, Any]:
        item = service.get_dataset_version(version_id)
        if item is None:
            raise HTTPException(status_code=404, detail="dataset version not found")
        return item

    @router.get("/dataset-versions/{version_id}/preview")
    async def preview_dataset_version(
        version_id: str,
        split: str = Query("train"),
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=500),
    ) -> Dict[str, Any]:
        try:
            return service.preview_dataset_version(
                version_id, split=split, offset=offset, limit=limit
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset version not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/dataset-versions/{version_id}/statistics")
    async def dataset_version_statistics(version_id: str) -> Dict[str, Any]:
        try:
            return service.dataset_version_statistics(version_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset version not found") from exc

    @router.get("/dataset-versions/{version_id}/corpus-profile")
    async def dataset_version_corpus_profile(version_id: str) -> Dict[str, Any]:
        try:
            return service.corpus_profile(version_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset version not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/dataset-versions/{version_id}/packing-plan")
    async def dataset_version_corpus_packing_plan(
        version_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            result = service.corpus_packing_plan(version_id, payload)
            if result.get("status") == "preparing_dataset":
                return JSONResponse(status_code=202, content=result)
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset version not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/dataset-versions/{version_id}/readiness")
    async def dataset_version_readiness(
        version_id: str,
        trainer_mode: Optional[str] = Query(None),
        model: Optional[str] = Query(None),
        verifier_profile_revision_id: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        try:
            return service.get_dataset_version_readiness(
                version_id,
                trainer_mode=trainer_mode,
                model=model,
                verifier_profile_revision_id=verifier_profile_revision_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset version not found") from exc

    @router.post("/dataset-versions/{version_id}/proof-run")
    async def launch_dataset_proof_run(
        version_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            result = await service.launch_dataset_proof_run(version_id, payload)
            if result.get("status") == "preparing_dataset":
                return JSONResponse(status_code=202, content=result)
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset version not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/cpt/preflight")
    async def cpt_preflight(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = service.preflight_cpt(payload)
            if result.get("status") == "preparing_dataset":
                return JSONResponse(status_code=202, content=result)
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset version not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/cpt/launch")
    async def cpt_launch(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = await service.launch_cpt(payload)
            if result.get("status") == "preparing_dataset":
                return JSONResponse(status_code=202, content=result)
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset version not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/dataset-versions/{version_id}/export")
    async def export_dataset_version(
        version_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.export_dataset_version(version_id, payload or {})
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset version not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/dataset-versions/{version_id}/materialize", status_code=202)
    async def materialize_dataset_version(
        version_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.materialize_dataset_version(version_id, payload or {})
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset version not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/dataset-versions/{version_id}/clone-recipe")
    async def clone_dataset_recipe(version_id: str) -> Dict[str, Any]:
        item = service.get_dataset_version(version_id)
        if item is None:
            raise HTTPException(status_code=404, detail="dataset version not found")
        return {"recipe": item["recipe"], "parent_version_id": version_id}

    @router.get("/dataset-versions/{version_id}/training-artifacts")
    async def list_training_dataset_artifacts(version_id: str) -> Dict[str, Any]:
        try:
            return service.list_training_dataset_artifacts(version_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset version not found") from exc

    @router.post("/dataset-versions/{version_id}/training-artifacts", status_code=202)
    async def create_training_dataset_artifact(
        version_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.create_training_dataset_artifact(version_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset version not found") from exc
        except Exception as exc:
            if isinstance(exc, ValueError) or is_domain_error(exc):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise

    @router.get("/training-artifacts/{artifact_id}")
    async def get_training_dataset_artifact(artifact_id: str) -> Dict[str, Any]:
        item = service.get_training_dataset_artifact(artifact_id)
        if item is None:
            raise HTTPException(status_code=404, detail="training artifact not found")
        return item

    @router.get("/dataset-versions/{version_id}/runs")
    async def list_dataset_version_runs(version_id: str) -> Dict[str, Any]:
        try:
            return service.list_dataset_version_runs(version_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset version not found") from exc

    @router.get("/dataset-versions/{version_id}/compare")
    async def compare_dataset_versions(
        version_id: str, other_version_id: str = Query(...)
    ) -> Dict[str, Any]:
        try:
            return service.compare_dataset_versions(version_id, other_version_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset version not found") from exc
        except Exception as exc:
            if isinstance(exc, ValueError) or is_domain_error(exc):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise

    @router.get("/dataset-jobs")
    async def list_dataset_jobs(
        dataset_id: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=500),
    ) -> Dict[str, Any]:
        return service.list_dataset_jobs(dataset_id=dataset_id, status=status, limit=limit)

    @router.get("/dataset-jobs/{job_id}")
    async def get_dataset_job(job_id: str) -> Dict[str, Any]:
        item = service.get_dataset_job(job_id)
        if item is None:
            raise HTTPException(status_code=404, detail="dataset job not found")
        return item

    @router.post("/dataset-jobs/{job_id}/cancel")
    async def cancel_dataset_job(job_id: str) -> Dict[str, Any]:
        try:
            item = service.cancel_dataset_job(job_id)
        except Exception as exc:
            if is_domain_error(exc):
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise
        if item is None:
            raise HTTPException(status_code=404, detail="dataset job not found")
        return item

    @router.post("/dataset-jobs/{job_id}/retry", status_code=202)
    async def retry_dataset_job(job_id: str) -> Dict[str, Any]:
        try:
            item = service.retry_dataset_job(job_id)
        except Exception as exc:
            if isinstance(exc, ValueError) or is_domain_error(exc):
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise
        if item is None:
            raise HTTPException(status_code=404, detail="dataset job not found")
        return item

    # Lab v5 adaptive checkpoint policies and immutable research evidence.
    @router.post("/checkpoint-policies/resolve")
    async def resolve_checkpoint_policy(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.resolve_checkpoint_policy(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/checkpoint-policies")
    async def list_checkpoint_policies(
        trainer_mode: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_checkpoint_policies(
            trainer_mode=trainer_mode, limit=limit, offset=offset
        )

    @router.post("/checkpoint-policies", status_code=201)
    async def create_checkpoint_policy(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_checkpoint_policy(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/checkpoint-policies/{policy_id}/revisions")
    async def list_checkpoint_policy_revisions(
        policy_id: str,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            return service.list_checkpoint_policy_revisions(
                policy_id, limit=limit, offset=offset
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="checkpoint policy not found") from exc

    @router.post("/checkpoint-policies/{policy_id}/revisions", status_code=201)
    async def create_checkpoint_policy_revision(
        policy_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.create_checkpoint_policy_revision(policy_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="checkpoint policy not found") from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/checkpoint-policies/{policy_id}")
    async def get_checkpoint_policy(policy_id: str) -> Dict[str, Any]:
        item = service.get_checkpoint_policy(policy_id)
        if item is None:
            raise HTTPException(status_code=404, detail="checkpoint policy not found")
        return item

    # Reproducible repeat/sweep operations over the durable workstation queue.
    @router.get("/run-groups")
    async def list_run_groups(
        status: Optional[str] = Query(None),
        kind: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
    ) -> Dict[str, Any]:
        return service.list_run_groups(status=status, kind=kind, limit=limit)

    @router.post("/run-groups", status_code=202)
    async def create_run_group(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_run_group(payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/run-groups/{run_group_id}")
    async def get_run_group(run_group_id: str) -> Dict[str, Any]:
        try:
            return service.get_run_group(run_group_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/run-groups/{run_group_id}/trajectory")
    async def get_run_group_trajectory(run_group_id: str) -> Dict[str, Any]:
        try:
            return service.get_run_group_trajectory(run_group_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/run-groups/{run_group_id}/analyses")
    async def list_run_group_analyses(
        run_group_id: str,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            return service.list_run_group_analyses(
                run_group_id, limit=limit, offset=offset
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/run-groups/{run_group_id}/analyses", status_code=201)
    async def create_run_group_analysis(
        run_group_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.create_run_group_analysis(run_group_id, payload or {})
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/run-groups/{run_group_id}/cancel")
    async def cancel_run_group(run_group_id: str) -> Dict[str, Any]:
        try:
            return service.cancel_run_group(run_group_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/run-groups/{run_group_id}/resume", status_code=202)
    async def resume_run_group(
        run_group_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.resume_run_group(
                run_group_id, reason=(payload or {}).get("reason")
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/run-groups/{run_group_id}/advance", status_code=202)
    async def advance_run_group(run_group_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.advance_run_group(run_group_id, int(payload.get("rung_index", 0)))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/run-groups/{run_group_id}/compare")
    async def compare_run_group(
        run_group_id: str,
        other_run_group_id: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        try:
            return service.compare_run_group(run_group_id, other_run_group_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/run-groups/{run_group_id}/fork-best", status_code=202)
    async def fork_best_run_group(
        run_group_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.fork_best_run_group(run_group_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/gate-decisions/{decision_id}/review")
    async def review_checkpoint_gate(
        decision_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.review_checkpoint_gate(decision_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="gate decision not found") from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/research-decisions")
    async def list_research_decisions(
        run_group_id: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
    ) -> Dict[str, Any]:
        return service.list_research_decisions(run_group_id=run_group_id, limit=limit)

    @router.post("/research-decisions", status_code=201)
    async def create_research_decision(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_research_decision(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/research-decisions/{decision_id}")
    async def get_research_decision(decision_id: str) -> Dict[str, Any]:
        item = service.get_research_decision(decision_id)
        if item is None:
            raise HTTPException(status_code=404, detail="research decision not found")
        return item

    @router.get("/evidence-bundles")
    async def list_evidence_bundles(
        run_group_id: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
    ) -> Dict[str, Any]:
        return service.list_evidence_bundles(run_group_id=run_group_id, limit=limit)

    @router.post("/evidence-bundles", status_code=202)
    async def create_evidence_bundle(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_evidence_bundle(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/evidence-bundles/{bundle_id}")
    async def get_evidence_bundle(bundle_id: str) -> Dict[str, Any]:
        item = service.get_evidence_bundle(bundle_id)
        if item is None:
            raise HTTPException(status_code=404, detail="evidence bundle not found")
        return item

    @router.get("/workspace-drafts/{surface}/{draft_key}")
    async def get_workspace_draft(surface: str, draft_key: str) -> Dict[str, Any]:
        item = service.get_workspace_draft(surface, draft_key)
        if item is None:
            raise HTTPException(status_code=404, detail="workspace draft not found")
        return item

    @router.put("/workspace-drafts/{surface}/{draft_key}")
    async def save_workspace_draft(
        surface: str, draft_key: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.save_workspace_draft(surface, draft_key, payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.delete("/workspace-drafts/{surface}/{draft_key}")
    async def delete_workspace_draft(surface: str, draft_key: str) -> Dict[str, Any]:
        return {
            "deleted": service.delete_workspace_draft(surface, draft_key),
            "surface": surface,
            "key": draft_key,
        }

    @router.get("/search")
    async def global_search(
        q: str = Query(..., min_length=1),
        types: Optional[str] = Query(None),
        limit: int = Query(30, ge=1, le=100),
    ) -> Dict[str, Any]:
        requested_types = [value.strip() for value in (types or "").split(",") if value.strip()]
        return service.global_search(q, types=requested_types or None, limit=limit)

    @router.get("/trainer-execution-capabilities")
    async def list_trainer_execution_capabilities() -> Dict[str, Any]:
        return service.list_trainer_execution_capabilities()

    # One durable workstation queue shared by managed heavy operations.
    @router.get("/activity")
    async def activity_center(
        after_sequence: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
    ) -> Dict[str, Any]:
        return service.get_activity(after_sequence=after_sequence, limit=limit)

    @router.get("/activity/events")
    async def stream_activity_events(
        after_sequence: int = Query(0, ge=0),
    ) -> "StreamingResponse":
        return StreamingResponse(
            service.stream_activity(after_sequence=after_sequence),
            media_type="text/event-stream",
        )

    @router.get("/workers")
    async def list_workers(
        limit: int = Query(100, ge=1, le=1000),
    ) -> Dict[str, Any]:
        return service.list_workers(limit=limit)

    @router.get("/storage")
    async def storage_status() -> Dict[str, Any]:
        return service.storage_status()

    @router.post("/storage/cleanup")
    async def storage_cleanup(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            return service.storage_cleanup(payload or {"preview": True})
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="cleanup plan not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/work-items")
    async def list_work_items(
        status: Optional[str] = Query(None),
        kind: Optional[str] = Query(None),
        limit: int = Query(200, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_work_items(status=status, kind=kind, limit=limit, offset=offset)

    @router.get("/work-items/{work_item_id}")
    async def get_work_item(work_item_id: str) -> Dict[str, Any]:
        item = service.get_work_item(work_item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="work item not found")
        return item

    @router.post("/work-items/{work_item_id}/cancel")
    async def cancel_work_item(work_item_id: str) -> Dict[str, Any]:
        try:
            return service.cancel_work_item(work_item_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="work item not found") from exc

    @router.post("/work-items/{work_item_id}/retry", status_code=202)
    async def retry_work_item(
        work_item_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.retry_work_item(work_item_id, reason=(payload or {}).get("reason"))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="work item not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/model-artifacts")
    async def list_model_artifacts(
        run_id: Optional[str] = Query(None),
        run_group_id: Optional[str] = Query(None),
        artifact_kind: Optional[str] = Query(None),
        kind: Optional[str] = Query(None),
        query: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_model_artifacts(
            run_id=run_id,
            run_group_id=run_group_id,
            artifact_kind=kind or artifact_kind,
            query=query,
            limit=limit,
            offset=offset,
        )

    @router.post("/model-artifacts/import", status_code=201)
    async def import_model_artifact(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.import_model_artifact(payload)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/model-artifacts/{artifact_id}")
    async def get_model_artifact(artifact_id: str) -> Dict[str, Any]:
        item = service.get_model_artifact(artifact_id)
        if item is None:
            raise HTTPException(status_code=404, detail="model artifact not found")
        return item

    @router.get("/model-artifacts/{artifact_id}/lineage")
    async def get_model_artifact_lineage(artifact_id: str) -> Dict[str, Any]:
        try:
            return service.get_model_artifact_lineage(artifact_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model artifact not found") from exc

    @router.post("/model-artifacts/{artifact_id}/verify")
    async def verify_model_artifact(artifact_id: str) -> Dict[str, Any]:
        try:
            return service.verify_model_artifact(artifact_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model artifact not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/model-artifacts/{artifact_id}/pin")
    async def pin_model_artifact(artifact_id: str) -> Dict[str, Any]:
        try:
            return service.pin_model_artifact(artifact_id, pinned=True)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model artifact not found") from exc

    @router.delete("/model-artifacts/{artifact_id}/pin")
    async def unpin_model_artifact(artifact_id: str) -> Dict[str, Any]:
        try:
            return service.pin_model_artifact(artifact_id, pinned=False)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model artifact not found") from exc

    @router.post("/model-artifacts/{artifact_id}/tags")
    async def tag_model_artifact(artifact_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.tag_model_artifact(artifact_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model artifact not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/model-artifacts/{artifact_id}/promote")
    async def promote_model_artifact(artifact_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.promote_model_artifact(artifact_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model artifact not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            if exc.__class__.__name__ == "PromotionBlocked":
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise

    @router.post("/model-artifacts/{artifact_id}/serve", status_code=202)
    async def serve_model_artifact(
        artifact_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.serve_model_artifact(artifact_id, payload or {})
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model artifact not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/artifact-operations")
    async def list_artifact_operations(
        status: Optional[str] = Query(None),
        kind: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_artifact_operations(
            status=status, kind=kind, limit=limit, offset=offset
        )

    @router.post("/artifact-operations", status_code=202)
    async def create_artifact_operation(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_artifact_operation(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            if exc.__class__.__name__ in {"ArtifactStudioError", "UnsupportedArtifactCapability"}:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise

    @router.get("/artifact-operations/{operation_id}")
    async def get_artifact_operation(operation_id: str) -> Dict[str, Any]:
        try:
            return service.get_artifact_operation(operation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/qualifications")
    async def list_qualifications(
        artifact_id: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_qualifications(
            occurrence_id=artifact_id, status=status, limit=limit, offset=offset
        )

    @router.get("/qualifications/compare")
    async def compare_qualifications(
        base_id: str = Query(...), candidate_id: str = Query(...)
    ) -> Dict[str, Any]:
        try:
            return service.compare_qualifications(base_id, candidate_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/qualifications/{qualification_id}")
    async def get_qualification(qualification_id: str) -> Dict[str, Any]:
        value = service.get_qualification(qualification_id)
        if value is None:
            raise HTTPException(status_code=404, detail="qualification not found")
        return value

    @router.post("/qualifications", status_code=202)
    async def create_qualification(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_qualification(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/qualification-profiles")
    async def list_qualification_profiles(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_qualification_profiles(limit=limit, offset=offset)

    @router.get("/qualification-profiles/{revision_id}")
    async def get_qualification_profile(revision_id: str) -> Dict[str, Any]:
        value = service.get_qualification_profile(revision_id)
        if value is None:
            raise HTTPException(status_code=404, detail="qualification profile revision not found")
        return value

    @router.post("/qualification-profiles")
    async def create_qualification_profile(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_qualification_profile(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/train/verifiers")
    async def list_training_verifiers() -> Dict[str, Any]:
        return {"items": service.list_training_verifiers()}

    @router.get("/train/templates")
    async def list_training_templates() -> Dict[str, Any]:
        """Intent-driven training templates (gallery surface).

        Distinct from `/train/presets` (which is the mode-level
        configurator pre-fill source) — templates are user-facing
        starting points like "train Python coding" or "fine-tune
        whisper for podcasts", and bind a modality + model + dataset
        + verifier + hyperparams in a single object."""
        from halo_forge.training import list_categories, list_templates

        return {
            "categories": list_categories(),
            "items": list_templates(),
        }

    @router.get("/train/templates/{template_id}")
    async def get_training_template(template_id: str) -> Dict[str, Any]:
        from halo_forge.training import cli_invocation, get_template

        tpl = get_template(template_id)
        if tpl is None:
            raise HTTPException(status_code=404, detail=f"Unknown template: {template_id}")
        tpl["cli"] = cli_invocation(template_id)
        return tpl

    @router.get("/verifiers")
    async def list_verifier_catalog() -> Dict[str, Any]:
        """Full registry inventory (Track F-O). Distinct from
        `/train/verifiers` which is the curated code-execution catalog —
        this surface includes LLM-judge / schema / metric / user-plugin
        verifiers too."""
        return service.list_verifier_catalog()

    @router.get("/verifier-reliability/capabilities")
    async def verifier_reliability_capabilities() -> Dict[str, Any]:
        return service.get_verifier_reliability_capabilities()

    @router.get("/verifier-profiles")
    async def list_verifier_profiles(
        q: Optional[str] = Query(None, max_length=200),
        family: Optional[str] = Query(None),
        modality: Optional[str] = Query(None),
        task_type: Optional[str] = Query(None),
        qualified_only: bool = Query(False),
        include_overridden: bool = Query(False),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_verifier_profiles(
            query=q,
            family=family,
            modality=modality,
            task_type=task_type,
            qualified_only=qualified_only,
            include_overridden=include_overridden,
            limit=limit,
            offset=offset,
        )

    @router.post("/verifier-profiles", status_code=201)
    async def create_verifier_profile(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_verifier_profile(payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.post("/verifier-profiles/validate")
    async def validate_verifier_profile(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.validate_verifier_profile(payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/verifier-profiles/{profile_id}")
    async def get_verifier_profile(profile_id: str) -> Dict[str, Any]:
        value = service.get_verifier_profile(profile_id)
        if value is None:
            raise HTTPException(status_code=404, detail="verifier profile not found")
        return value

    @router.post("/verifier-profiles/{profile_id}/revisions", status_code=201)
    async def revise_verifier_profile(
        profile_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.revise_verifier_profile(profile_id, payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/verifier-profiles/{profile_id}/revisions")
    async def list_verifier_profile_revisions(
        profile_id: str,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            return service.list_verifier_profile_revisions(
                profile_id, limit=limit, offset=offset
            )
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/verifier-profile-revisions/{revision_id}")
    async def get_verifier_profile_revision(revision_id: str) -> Dict[str, Any]:
        value = service.get_verifier_profile(revision_id)
        if value is None:
            raise HTTPException(status_code=404, detail="verifier revision not found")
        return value

    @router.post("/verifier-profile-revisions/{revision_id}/promote")
    async def promote_verifier_revision(
        revision_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.promote_verifier_revision(revision_id, payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/verifier-profile-revisions/{revision_id}/usage")
    async def verifier_revision_usage(
        revision_id: str,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            return service.verifier_revision_usage(
                revision_id, limit=limit, offset=offset
            )
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/verifier-profile-revisions/{revision_id}/runtime-compatibility")
    async def verifier_runtime_compatibility(revision_id: str) -> Dict[str, Any]:
        try:
            return service.verifier_runtime_compatibility(revision_id)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/verifier-profiles/{profile_id}/aliases")
    async def verifier_alias_history(
        profile_id: str,
        alias: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            return service.list_verifier_alias_history(
                profile_id,
                alias=alias,
                limit=limit,
                offset=offset,
            )
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/verifier-qualifications")
    async def list_verifier_qualification_decisions(
        verifier_profile_revision_id: Optional[str] = Query(None),
        calibration_id: Optional[str] = Query(None),
        decision: Optional[str] = Query(None),
        scope: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            return service.list_verifier_qualification_decisions(
                verifier_revision_id=verifier_profile_revision_id,
                calibration_id=calibration_id,
                decision=decision,
                scope=scope,
                limit=limit,
                offset=offset,
            )
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/verifier-calibration-protocols")
    async def list_verifier_protocols(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_verifier_protocols(limit=limit, offset=offset)

    @router.post("/verifier-calibration-protocols", status_code=201)
    async def create_verifier_protocol(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_verifier_protocol(payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/verifier-calibration-protocols/{protocol_id}")
    async def get_verifier_protocol(protocol_id: str) -> Dict[str, Any]:
        value = service.get_verifier_protocol(protocol_id)
        if value is None:
            raise HTTPException(status_code=404, detail="calibration protocol not found")
        return value

    @router.get("/verifier-calibration-protocol-revisions/{revision_id}")
    async def get_verifier_protocol_revision(revision_id: str) -> Dict[str, Any]:
        value = service.get_verifier_protocol(revision_id)
        if value is None:
            raise HTTPException(status_code=404, detail="calibration protocol revision not found")
        return value

    @router.post("/verifier-calibration-protocols/{protocol_id}/revisions", status_code=201)
    async def revise_verifier_protocol(
        protocol_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.revise_verifier_protocol(protocol_id, payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/verifier-qualification-profiles")
    async def list_verifier_qualification_profiles(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_verifier_qualification_profiles(limit=limit, offset=offset)

    @router.post("/verifier-qualification-profiles", status_code=201)
    async def create_verifier_qualification_profile(
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            return service.create_verifier_qualification_profile(payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/verifier-qualification-profiles/{profile_id}")
    async def get_verifier_qualification_profile(profile_id: str) -> Dict[str, Any]:
        value = service.get_verifier_qualification_profile(profile_id)
        if value is None:
            raise HTTPException(status_code=404, detail="qualification profile not found")
        return value

    @router.get("/verifier-qualification-profile-revisions/{revision_id}")
    async def get_verifier_qualification_profile_revision(
        revision_id: str,
    ) -> Dict[str, Any]:
        value = service.get_verifier_qualification_profile(revision_id)
        if value is None:
            raise HTTPException(
                status_code=404,
                detail="qualification profile revision not found",
            )
        return value

    @router.post(
        "/verifier-qualification-profiles/{profile_id}/revisions", status_code=201
    )
    async def revise_verifier_qualification_profile(
        profile_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.revise_verifier_qualification_profile(profile_id, payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/verifier-calibrations/compare")
    async def compare_verifier_calibrations(
        base_id: str = Query(...),
        candidate_id: str = Query(...),
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
    ) -> Dict[str, Any]:
        try:
            return service.compare_verifier_calibrations(
                base_id, candidate_id, offset=offset, limit=limit
            )
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/verifier-calibrations")
    async def list_verifier_calibrations(
        verifier_profile_revision_id: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_verifier_calibrations(
            verifier_revision_id=verifier_profile_revision_id,
            status=status,
            limit=limit,
            offset=offset,
        )

    @router.post("/verifier-calibrations", status_code=202)
    async def launch_verifier_calibration(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.launch_verifier_calibration(payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/verifier-calibrations/{calibration_id}")
    async def get_verifier_calibration(calibration_id: str) -> Dict[str, Any]:
        value = service.get_verifier_calibration(calibration_id)
        if value is None:
            raise HTTPException(status_code=404, detail="verifier calibration not found")
        return value

    @router.get("/verifier-calibrations/{calibration_id}/samples")
    async def list_verifier_calibration_samples(
        calibration_id: str,
        partition: Optional[str] = Query(None),
        outcome: Optional[str] = Query(None),
        perturbation: Optional[str] = Query(None),
        q: Optional[str] = Query(None, max_length=200),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            return service.list_verifier_calibration_samples(
                calibration_id,
                partition=partition,
                outcome=outcome,
                perturbation=perturbation,
                query=q,
                limit=limit,
                offset=offset,
            )
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/verifier-calibrations/{calibration_id}/metrics")
    async def list_verifier_calibration_metrics(
        calibration_id: str,
        partition: Optional[str] = Query(None),
        subgroup: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            return service.list_verifier_calibration_metrics(
                calibration_id,
                partition=partition,
                subgroup=subgroup,
                limit=limit,
                offset=offset,
            )
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.post("/verifier-calibrations/{calibration_id}/cancel")
    async def cancel_verifier_calibration(calibration_id: str) -> Dict[str, Any]:
        try:
            return service.cancel_verifier_calibration(calibration_id)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.post("/verifier-calibrations/{calibration_id}/retry", status_code=202)
    async def retry_verifier_calibration(calibration_id: str) -> Dict[str, Any]:
        try:
            return service.retry_verifier_calibration(calibration_id)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.post("/verifier-calibrations/{calibration_id}/verify")
    async def verify_verifier_calibration(calibration_id: str) -> Dict[str, Any]:
        try:
            return service.verify_verifier_calibration(calibration_id)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.post("/verifier-calibrations/{calibration_id}/qualify", status_code=201)
    async def qualify_verifier_calibration(
        calibration_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.qualify_verifier_calibration(calibration_id, payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    # Halo Forge Lab v8 — Reward Integrity and Training Signal Studio.

    @router.get(
        "/reward-integrity-capabilities",
        operation_id="reward_integrity_capabilities_legacy",
    )
    @router.get(
        "/reward-integrity/capabilities",
        operation_id="reward_integrity_capabilities",
    )
    async def reward_integrity_capabilities() -> Dict[str, Any]:
        try:
            return service.get_reward_integrity_capabilities()
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.post("/reward-integrity-bindings/resolve")
    async def resolve_reward_integrity_binding(
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.resolve_reward_integrity_binding(payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/reward-systems")
    async def list_reward_systems(
        q: Optional[str] = Query(None, max_length=200),
        modality: Optional[str] = Query(None),
        task_type: Optional[str] = Query(None),
        trainer_mode: Optional[str] = Query(None),
        backend_family: str = Query("hf"),
        qualified_only: bool = Query(False),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_reward_systems(
            query=q,
            modality=modality,
            task_type=task_type,
            trainer_mode=trainer_mode,
            backend_family=backend_family,
            qualified_only=qualified_only,
            limit=limit,
            offset=offset,
        )

    @router.post("/reward-systems", status_code=201)
    async def create_reward_system(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_reward_system(payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.post("/reward-systems/validate")
    async def validate_reward_system(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.validate_reward_system(payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/reward-systems/{system_id}")
    async def get_reward_system(system_id: str) -> Dict[str, Any]:
        value = service.get_reward_system(system_id)
        if value is None:
            raise HTTPException(status_code=404, detail="reward system not found")
        return value

    @router.post("/reward-systems/{system_id}/revisions", status_code=201)
    async def revise_reward_system(
        system_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.revise_reward_system(system_id, payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/reward-system-revisions/{revision_id}/usage")
    async def list_reward_system_usage(
        revision_id: str,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_reward_system_usage(
            revision_id, limit=limit, offset=offset
        )

    @router.get("/reward-system-revisions/{revision_id}")
    async def get_reward_system_revision(revision_id: str) -> Dict[str, Any]:
        value = service.get_reward_system(revision_id)
        if value is None:
            raise HTTPException(status_code=404, detail="reward system revision not found")
        return value

    @router.get("/reward-audit-protocols")
    async def list_reward_protocols(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_reward_protocols(limit=limit, offset=offset)

    @router.post("/reward-audit-protocols", status_code=201)
    async def create_reward_protocol(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_reward_protocol(payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/reward-audit-protocols/{protocol_id}")
    async def get_reward_protocol(protocol_id: str) -> Dict[str, Any]:
        value = service.get_reward_protocol(protocol_id)
        if value is None:
            raise HTTPException(status_code=404, detail="reward audit protocol not found")
        return value

    @router.post("/reward-audit-protocols/{protocol_id}/revisions", status_code=201)
    async def revise_reward_protocol(
        protocol_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.revise_reward_protocol(protocol_id, payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/reward-audit-protocol-revisions/{revision_id}")
    async def get_reward_protocol_revision(revision_id: str) -> Dict[str, Any]:
        value = service.get_reward_protocol(revision_id)
        if value is None:
            raise HTTPException(status_code=404, detail="reward protocol revision not found")
        return value

    @router.get("/reward-integrity-profiles")
    async def list_reward_integrity_profiles(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_reward_integrity_profiles(limit=limit, offset=offset)

    @router.post("/reward-integrity-profiles", status_code=201)
    async def create_reward_integrity_profile(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_reward_integrity_profile(payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/reward-integrity-profiles/{profile_id}")
    async def get_reward_integrity_profile(profile_id: str) -> Dict[str, Any]:
        value = service.get_reward_integrity_profile(profile_id)
        if value is None:
            raise HTTPException(status_code=404, detail="reward integrity profile not found")
        return value

    @router.post("/reward-integrity-profiles/{profile_id}/revisions", status_code=201)
    async def revise_reward_integrity_profile(
        profile_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.revise_reward_integrity_profile(profile_id, payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/reward-integrity-profile-revisions/{revision_id}")
    async def get_reward_integrity_profile_revision(revision_id: str) -> Dict[str, Any]:
        value = service.get_reward_integrity_profile(revision_id)
        if value is None:
            raise HTTPException(status_code=404, detail="integrity profile revision not found")
        return value

    @router.get("/training-signals")
    async def list_training_signals(
        run_id: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_training_signal_shards(
            run_id=run_id, limit=limit, offset=offset
        )

    @router.get("/runs/{run_id}/training-signals")
    async def list_run_training_signals(
        run_id: str,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_training_signal_shards(
            run_id=run_id, limit=limit, offset=offset
        )

    @router.get("/training-signals/{shard_id}")
    async def get_training_signal(shard_id: str) -> Dict[str, Any]:
        value = service.get_training_signal_shard(shard_id)
        if value is None:
            raise HTTPException(status_code=404, detail="training signal shard not found")
        return value

    @router.post("/training-signals/{shard_id}/verify")
    async def verify_training_signal(shard_id: str) -> Dict[str, Any]:
        try:
            return service.verify_training_signal_shard(shard_id)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/reward-integrity-audits/compare")
    async def compare_reward_integrity_audits(
        base_id: str = Query(...),
        candidate_id: str = Query(...),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            return service.compare_reward_integrity_audits(
                base_id, candidate_id, limit=limit, offset=offset
            )
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/reward-integrity-audits")
    async def list_reward_integrity_audits(
        run_id: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_reward_integrity_audits(
            run_id=run_id, status=status, limit=limit, offset=offset
        )

    @router.get("/runs/{run_id}/reward-integrity-audits")
    async def list_run_reward_integrity_audits(
        run_id: str,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_reward_integrity_audits(
            run_id=run_id, limit=limit, offset=offset
        )

    @router.post("/reward-integrity-audits", status_code=202)
    async def launch_reward_integrity_audit(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.launch_reward_integrity_audit(payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/reward-integrity-audits/{audit_id}")
    async def get_reward_integrity_audit(audit_id: str) -> Dict[str, Any]:
        value = service.get_reward_integrity_audit(audit_id)
        if value is None:
            raise HTTPException(status_code=404, detail="reward integrity audit not found")
        return value

    @router.get("/reward-integrity-audits/{audit_id}/samples")
    async def list_reward_integrity_samples(
        audit_id: str,
        population: Optional[str] = Query(None),
        outcome: Optional[str] = Query(None),
        q: Optional[str] = Query(None, max_length=200),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            return service.list_reward_integrity_samples(
                audit_id,
                population=population,
                outcome=outcome,
                query=q,
                limit=limit,
                offset=offset,
            )
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/reward-integrity-audits/{audit_id}/metrics")
    async def list_reward_integrity_metrics(
        audit_id: str,
        subgroup: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            return service.list_reward_integrity_metrics(
                audit_id, subgroup=subgroup, limit=limit, offset=offset
            )
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.post("/reward-integrity-audits/{audit_id}/verify")
    async def verify_reward_integrity_audit(audit_id: str) -> Dict[str, Any]:
        try:
            return service.verify_reward_integrity_audit(audit_id)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.post("/reward-integrity-audits/{audit_id}/cancel")
    async def cancel_reward_integrity_audit(audit_id: str) -> Dict[str, Any]:
        try:
            return service.cancel_reward_integrity_audit(audit_id)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.post("/reward-integrity-audits/{audit_id}/retry", status_code=202)
    async def retry_reward_integrity_audit(
        audit_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.retry_reward_integrity_audit(audit_id, payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.post("/reward-integrity-audits/{audit_id}/review")
    async def review_reward_integrity_audit(
        audit_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.review_reward_integrity_audit(audit_id, payload)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/reward-integrity-audits/{audit_id}/fork-context")
    async def get_reward_integrity_fork_context(audit_id: str) -> Dict[str, Any]:
        try:
            return service.get_reward_integrity_fork_context(audit_id)
        except Exception as exc:
            _raise_verifier_error(exc)

    @router.get("/models")
    async def list_model_catalog(
        mode: Optional[str] = Query(None),
        backend: Optional[str] = Query(None),
        modality: Optional[str] = Query(None),
        provider: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        memory_tier: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        return service.list_model_catalog(
            mode=mode,
            backend=backend,
            modality=modality,
            provider=provider,
            status=status,
            memory_tier=memory_tier,
        )

    @router.get("/train/models")
    async def list_suggested_models(
        mode: Optional[str] = Query(None),
        modality: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        return {"items": service.list_suggested_models(mode=mode, modality=modality)}

    @router.get("/training-plan-capabilities")
    async def training_plan_capabilities() -> Dict[str, Any]:
        return service.training_plan_capabilities()

    @router.get("/runtime-capabilities")
    async def managed_runtime_capabilities() -> Dict[str, Any]:
        return service.managed_runtime_capabilities()

    @router.get("/runtime/paths")
    async def training_path_capabilities(
        family: str = Query(..., pattern="^(rocm|cuda)$"),
    ) -> Dict[str, Any]:
        try:
            return service.training_path_capabilities(family)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/training-path-revisions/{revision_id}/certify", status_code=202)
    async def launch_training_path_certification(
        revision_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.launch_training_path_certification(revision_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Training path not found") from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/training-path-revisions/{revision_id}/certification-preview")
    async def preview_training_path_certification(
        revision_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.preview_training_path_certification(revision_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Training path or runtime not found") from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/training-path-certifications/{certification_id}")
    async def get_training_path_certification(certification_id: str) -> Dict[str, Any]:
        try:
            return service.get_training_path_certification(certification_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Certification not found") from exc

    @router.get("/training-path-certifications/{certification_id}/steps")
    async def list_training_path_certification_steps(
        certification_id: str,
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            return service.list_training_path_certification_steps(
                certification_id, limit=limit, offset=offset
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Certification not found") from exc

    @router.post("/training-path-certifications/{certification_id}/verify")
    async def verify_training_path_certification(certification_id: str) -> Dict[str, Any]:
        try:
            return service.verify_training_path_certification(certification_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Certification not found") from exc

    @router.post("/training-path-certifications/{certification_id}/cancel")
    async def cancel_training_path_certification(certification_id: str) -> Dict[str, Any]:
        try:
            return service.cancel_training_path_certification(certification_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Certification not found") from exc

    @router.post("/training-path-certifications/{certification_id}/retry", status_code=202)
    async def retry_training_path_certification(
        certification_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.retry_training_path_certification(certification_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Certification not found") from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/training-path-certifications/{certification_id}/resume", status_code=202)
    async def resume_training_path_certification(
        certification_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.retry_training_path_certification(
                certification_id,
                {"reason": str((payload or {}).get("reason") or "Resume from the last verified step")},
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Certification not found") from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/training-path-certifications/{certification_id}/evidence")
    async def training_path_certification_evidence(certification_id: str) -> Dict[str, Any]:
        try:
            return service.training_path_certification_evidence(certification_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Certification not found") from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/release/workstation-certify", status_code=202)
    async def workstation_certify(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.workstation_certify(payload)
        except (KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/release/workstation-certifications/{certification_id}")
    async def workstation_certification_report(certification_id: str) -> Dict[str, Any]:
        try:
            return service.workstation_certification_report(certification_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Workstation certification not found") from exc

    @router.get("/release/workstation-certifications/{certification_id}/verify")
    async def verify_workstation_certification(certification_id: str) -> Dict[str, Any]:
        try:
            return service.verify_workstation_certification(certification_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Workstation certification not found") from exc

    @router.get("/runtimes")
    async def list_managed_runtimes() -> Dict[str, Any]:
        return service.list_managed_runtimes()

    @router.get("/runtimes/{identifier}")
    async def get_managed_runtime(identifier: str) -> Dict[str, Any]:
        try:
            return service.get_managed_runtime(identifier)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Managed runtime not found") from exc

    @router.post("/runtime-revisions/{revision_id}/prepare", status_code=202)
    async def prepare_managed_runtime(
        revision_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.prepare_managed_runtime(revision_id, payload or {})
        except (KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/runtime-revisions/{revision_id}/qualify", status_code=202)
    async def qualify_managed_runtime(revision_id: str) -> Dict[str, Any]:
        try:
            return service.qualify_managed_runtime(revision_id)
        except (KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/runtime-preparations/{preparation_id}")
    async def get_runtime_preparation(preparation_id: str) -> Dict[str, Any]:
        try:
            return service.get_runtime_preparation(preparation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Runtime preparation not found") from exc

    @router.post("/runtime-preparations/{preparation_id}/cancel")
    async def cancel_runtime_preparation(preparation_id: str) -> Dict[str, Any]:
        try:
            return service.cancel_runtime_work("preparation", preparation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Runtime preparation not found") from exc

    @router.post("/runtime-preparations/{preparation_id}/retry", status_code=202)
    async def retry_runtime_preparation(
        preparation_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.retry_runtime_work("preparation", preparation_id, payload or {})
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Runtime preparation not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/runtime-qualifications/{qualification_id}")
    async def get_runtime_qualification(qualification_id: str) -> Dict[str, Any]:
        try:
            return service.get_runtime_qualification(qualification_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Runtime qualification not found") from exc

    @router.post("/runtime-qualifications/{qualification_id}/verify")
    async def verify_runtime_qualification(qualification_id: str) -> Dict[str, Any]:
        try:
            return service.verify_runtime_qualification(qualification_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Runtime qualification not found") from exc

    @router.post("/runtime-qualifications/{qualification_id}/cancel")
    async def cancel_runtime_qualification(qualification_id: str) -> Dict[str, Any]:
        try:
            return service.cancel_runtime_work("qualification", qualification_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Runtime qualification not found") from exc

    @router.post("/runtime-qualifications/{qualification_id}/retry", status_code=202)
    async def retry_runtime_qualification(
        qualification_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.retry_runtime_work("qualification", qualification_id, payload or {})
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Runtime qualification not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/accelerator/availability")
    async def accelerator_availability(
        family: str = Query(..., pattern="^(rocm|cuda)$"),
    ) -> Dict[str, Any]:
        try:
            return service.accelerator_availability(family)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/training-plans")
    async def list_training_plans(
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_training_plans(limit=limit, offset=offset)

    @router.post("/training-plans/recommend", status_code=201)
    async def recommend_training_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.recommend_training_plan(payload)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/training-plans/{plan_id}")
    async def get_training_plan(plan_id: str) -> Dict[str, Any]:
        try:
            return service.get_training_plan(plan_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Training plan not found") from exc

    @router.get("/training-plan-revisions/{revision_id}")
    async def get_training_plan_revision(revision_id: str) -> Dict[str, Any]:
        try:
            return service.get_training_plan_revision(revision_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Training plan revision not found") from exc

    @router.get("/training-plan-revisions/{revision_id}/alternatives")
    async def get_training_plan_alternatives(revision_id: str) -> Dict[str, Any]:
        try:
            return service.get_training_plan_alternatives(revision_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Training plan revision not found") from exc

    @router.post("/training-plan-revisions/{revision_id}/alternatives", status_code=201)
    async def choose_training_plan_alternative(
        revision_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.choose_training_plan_alternative(revision_id, payload)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/training-plan-revisions/{revision_id}/confirm", status_code=201)
    async def confirm_training_plan(
        revision_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.confirm_training_plan(revision_id, payload or {})
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/training-plan-revisions/{revision_id}/prepare", status_code=202)
    async def prepare_training_plan_model(
        revision_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.prepare_training_plan_model(revision_id, payload or {})
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/model-preparations/{preparation_id}")
    async def get_model_preparation(preparation_id: str) -> Dict[str, Any]:
        try:
            return service.get_model_preparation(preparation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Model preparation not found") from exc

    @router.post("/model-preparations/{preparation_id}/cancel")
    async def cancel_model_preparation(preparation_id: str) -> Dict[str, Any]:
        try:
            return service.cancel_training_plan_work("model_preparation", preparation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Model preparation not found") from exc

    @router.post("/model-preparations/{preparation_id}/retry", status_code=202)
    async def retry_model_preparation(
        preparation_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.retry_training_plan_work(
                "model_preparation", preparation_id, payload or {}
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Model preparation not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/training-plan-revisions/{revision_id}/capacity-check", status_code=202)
    async def create_training_capacity_check(revision_id: str) -> Dict[str, Any]:
        try:
            return service.create_training_capacity_check(revision_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/training-capacity-checks/{check_id}")
    async def get_training_capacity_check(check_id: str) -> Dict[str, Any]:
        try:
            return service.get_training_capacity_check(check_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Capacity check not found") from exc

    @router.get("/training-capacity-checks/{check_id}/attempts")
    async def list_training_capacity_attempts(check_id: str) -> Dict[str, Any]:
        try:
            return service.list_training_capacity_attempts(check_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Capacity check not found") from exc

    @router.post("/training-capacity-checks/{check_id}/cancel")
    async def cancel_training_capacity_check(check_id: str) -> Dict[str, Any]:
        try:
            return service.cancel_training_plan_work("training_capacity_check", check_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Capacity check not found") from exc

    @router.post("/training-capacity-checks/{check_id}/retry", status_code=202)
    async def retry_training_capacity_check(
        check_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return service.retry_training_plan_work(
                "training_capacity_check", check_id, payload or {}
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Capacity check not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/training-plan-revisions/{revision_id}/readiness")
    async def get_training_plan_readiness(revision_id: str) -> Dict[str, Any]:
        try:
            return service.get_training_plan_readiness(revision_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Training plan revision not found") from exc

    @router.post("/training-plan-revisions/{revision_id}/proof", status_code=202)
    async def launch_training_plan_proof(
        revision_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            return await service.launch_training_plan_proof(revision_id, payload or {})
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/train/preflight")
    async def preflight_training(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = service.preflight_training(payload)
            if result.get("status") == "preparing_dataset":
                return JSONResponse(status_code=202, content=result)
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/train/launch")
    async def launch_training(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = await service.launch_training(payload)
            if result.get("status") == "preparing_dataset":
                return JSONResponse(status_code=202, content=result)
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/runs/{run_id}/full-run")
    async def launch_full_run_from_proof(
        run_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            result = await service.launch_full_run_from_proof(run_id, payload or {})
            if result.get("status") == "preparing_dataset":
                return JSONResponse(status_code=202, content=result)
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="training run not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/runs")
    async def list_runs(
        include_completed: bool = True,
        active_only: bool = False,
        include_research: bool = False,
    ) -> Dict[str, Any]:
        return service.list_runs(
            include_completed=include_completed,
            active_only=active_only,
            include_research=include_research,
        )

    @router.get("/runs/search")
    async def search_runs(
        modality: Optional[List[str]] = Query(None),
        status: Optional[List[str]] = Query(None),
        model: Optional[str] = Query(None),
        since: Optional[str] = Query(None),
        until: Optional[str] = Query(None),
        has_eval: Optional[bool] = Query(None),
        weights_updated: Optional[bool] = Query(None),
        sort_by: str = Query("timestamp"),
        sort_dir: str = Query("desc"),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        """DB-backed run search (Track F-G commit 2).

        Filter / sort / paginate the run index. Repeating ``modality=``
        or ``status=`` ANDs an IN-list (e.g. ``?modality=sft&modality=dpo``
        matches both). ``model=`` is a case-sensitive substring match
        against ``model_name``. The response includes ``facets``: the
        distinct modality and model values currently in the index, so
        a filter-chip UI can render without a second round trip.
        """
        return service.search_runs(
            modalities=modality,
            statuses=status,
            model_substring=model,
            since_iso=since,
            until_iso=until,
            has_eval=has_eval,
            weights_updated=weights_updated,
            sort_by=sort_by,
            sort_dir=sort_dir,
            limit=limit,
            offset=offset,
        )

    @router.get("/runs/{run_id}/lineage")
    async def get_run_lineage(run_id: str) -> Dict[str, Any]:
        """Return ancestors + descendants for a run (Track F-Q).

        Walks the lineage table BFS up + down. The response shape is
        ``{run_id, ancestors: [...], descendants: [...]}`` where each
        edge entry has the parent/child id, ``forked_at_cycle``, ``notes``,
        and a ``depth`` indicating distance from the queried run.
        """
        return service.get_run_lineage(run_id)

    @router.post("/runs/{run_id}/lineage")
    async def record_run_fork(run_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Record that ``run_id`` (the child) forked from a parent run.

        Body: ``{parent_run_id, forked_at_cycle?, notes?}``. Idempotent
        on the (child, parent) pair — re-recording the same edge updates
        the cycle/notes columns rather than failing.
        """
        parent = (payload.get("parent_run_id") or "").strip()
        if not parent:
            raise HTTPException(status_code=400, detail="parent_run_id is required")
        try:
            return service.record_run_fork(
                child_run_id=run_id,
                parent_run_id=parent,
                forked_at_cycle=payload.get("forked_at_cycle"),
                notes=payload.get("notes"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.delete("/runs/{run_id}/lineage/{parent_run_id}")
    async def delete_run_fork(run_id: str, parent_run_id: str) -> Dict[str, Any]:
        ok = service.remove_run_fork(
            child_run_id=run_id,
            parent_run_id=parent_run_id,
        )
        if not ok:
            raise HTTPException(status_code=404, detail="lineage edge not found")
        return {"deleted": True, "child_run_id": run_id, "parent_run_id": parent_run_id}

    @router.post("/playground/chat")
    async def playground_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Forward a chat request to a `halo-forge serve` endpoint (Track F-S).

        Body:
          ``{messages, model?, max_tokens?, temperature?, top_p?, stop?,
             serve_url?, api_key?}``

        Returns the upstream OpenAI-shaped response verbatim, or
        ``{upstream_error: true, status, detail}`` when the serve
        endpoint returns a non-2xx so the UI can render the actual
        problem instead of a generic 500.
        """
        messages = payload.get("messages") or []
        if not isinstance(messages, list) or not messages:
            raise HTTPException(
                status_code=400,
                detail="messages must be a non-empty list",
            )
        return service.playground_chat(
            messages=messages,
            model=payload.get("model"),
            max_tokens=int(256 if payload.get("max_tokens") is None else payload.get("max_tokens")),
            temperature=float(
                0.7 if payload.get("temperature") is None else payload.get("temperature")
            ),
            top_p=float(1.0 if payload.get("top_p") is None else payload.get("top_p")),
            stop=payload.get("stop"),
            serve_url=payload.get("serve_url"),
            api_key=payload.get("api_key"),
        )

    @router.get("/playground/sessions")
    async def list_playground_sessions(
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
        include_archived: bool = Query(False),
    ) -> Dict[str, Any]:
        return service.list_playground_sessions(
            limit=limit, offset=offset, include_archived=include_archived
        )

    @router.post("/playground/sessions", status_code=201)
    async def create_playground_session(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_playground_session(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model artifact not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/playground/sessions/{session_id}")
    async def get_playground_session(session_id: str) -> Dict[str, Any]:
        value = service.get_playground_session(session_id)
        if value is None:
            raise HTTPException(status_code=404, detail="playground session not found")
        return value

    @router.patch("/playground/sessions/{session_id}")
    async def update_playground_session(session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.update_playground_session(session_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.delete("/playground/sessions/{session_id}")
    async def archive_playground_session(session_id: str) -> Dict[str, Any]:
        try:
            return service.update_playground_session(session_id, {"archived": True})
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="playground session not found") from exc

    @router.post("/playground/sessions/{session_id}/messages", status_code=201)
    async def append_playground_message(session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.append_playground_message(session_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/playground/sessions/{session_id}/review", status_code=201)
    async def review_playground_session(session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.review_playground_session(session_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="playground session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/serve/status")
    async def serve_status() -> Dict[str, Any]:
        return service.serve_status()

    @router.post("/serve/start")
    async def serve_start(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.serve_start(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model artifact not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/serve/stop")
    async def serve_stop() -> Dict[str, Any]:
        return service.serve_stop()

    @router.get("/serve/logs")
    async def serve_logs(tail: int = Query(default=200, ge=1, le=5000)) -> Dict[str, Any]:
        return service.serve_logs(tail=tail)

    @router.get("/serve/health")
    async def serve_health() -> Dict[str, Any]:
        return service.serve_health()

    @router.get("/serving-profiles")
    async def list_serving_profiles(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        return service.list_serving_profiles(limit=limit, offset=offset)

    @router.delete("/serving/{serving_id}")
    async def release_artifact_serving(serving_id: str) -> Dict[str, Any]:
        return service.release_artifact_serving(serving_id)

    @router.get("/registry")
    async def list_registry() -> Dict[str, Any]:
        """List every model-registry entry (Track F-J).

        Each entry is a named bundle of run_ids the user wants to
        compare / promote / share as a unit.
        """
        return {"items": service.list_registry_entries()}

    @router.post("/registry")
    async def create_registry_entry(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new registry entry. Required field: ``name``."""
        try:
            return service.create_registry_entry(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/registry/{entry_id}")
    async def get_registry_entry(entry_id: int) -> Dict[str, Any]:
        entry = service.get_registry_entry(entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="registry entry not found")
        return entry

    @router.patch("/registry/{entry_id}")
    async def update_registry_entry(entry_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Patch a registry entry. Missing keys leave their fields alone;
        explicit nulls *replace* (so `description: null` clears it)."""
        entry = service.update_registry_entry(entry_id, payload)
        if entry is None:
            raise HTTPException(status_code=404, detail="registry entry not found")
        return entry

    @router.delete("/registry/{entry_id}")
    async def delete_registry_entry(entry_id: int) -> Dict[str, Any]:
        ok = service.delete_registry_entry(entry_id)
        if not ok:
            raise HTTPException(status_code=404, detail="registry entry not found")
        return {"deleted": True, "id": entry_id}

    # Dataset Lab v2 benchmark suites and persistent evaluations.
    @router.get("/benchmark-suites")
    async def list_benchmark_suites() -> Dict[str, Any]:
        return service.list_benchmark_suites()

    @router.post("/benchmark-suites", status_code=201)
    async def create_benchmark_suite(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_benchmark_suite(payload)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/benchmark-suites/{suite_id}")
    async def get_benchmark_suite(suite_id: str) -> Dict[str, Any]:
        item = service.get_benchmark_suite(suite_id)
        if item is None:
            raise HTTPException(status_code=404, detail="benchmark suite not found")
        return item

    @router.get("/benchmark-suite-revisions/{revision_id}")
    async def get_benchmark_suite_revision(revision_id: str) -> Dict[str, Any]:
        item = service.get_benchmark_suite_revision(revision_id)
        if item is None:
            raise HTTPException(status_code=404, detail="benchmark suite revision not found")
        return item

    @router.post("/benchmark-suites/{suite_id}/revisions", status_code=201)
    async def create_benchmark_suite_revision(
        suite_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.create_benchmark_suite_revision(suite_id, payload)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/evaluations/compare")
    async def compare_evaluations(
        base_id: str = Query(...),
        candidate_id: str = Query(...),
        offset: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
    ) -> Dict[str, Any]:
        try:
            return service.compare_evaluations(base_id, candidate_id, offset=offset, limit=limit)
        except Exception as exc:
            if isinstance(exc, (ValueError, KeyError)) or is_domain_error(exc):
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise

    @router.get("/evaluations/history")
    async def evaluation_history(
        subject_ref: Optional[str] = Query(None),
        suite_revision_id: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=500),
    ) -> Dict[str, Any]:
        return service.evaluation_history(
            subject_ref=subject_ref,
            suite_revision_id=suite_revision_id,
            limit=limit,
        )

    @router.get("/evaluations/drift")
    async def evaluation_drift(
        base_id: Optional[str] = Query(None),
        candidate_id: Optional[str] = Query(None),
        subject_ref: Optional[str] = Query(None),
        suite_revision_id: Optional[str] = Query(None),
        practical_delta: float = Query(0.0, ge=0.0),
    ) -> Dict[str, Any]:
        try:
            return service.evaluation_drift(
                base_id=base_id,
                candidate_id=candidate_id,
                subject_ref=subject_ref,
                suite_revision_id=suite_revision_id,
                practical_delta=practical_delta,
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/evaluations")
    async def list_evaluations(
        run_id: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=500),
    ) -> Dict[str, Any]:
        return service.list_evaluations(run_id=run_id, status=status, limit=limit)

    @router.post("/evaluations", status_code=202)
    async def launch_evaluation(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.launch_evaluation(payload)
        except Exception as exc:
            if isinstance(exc, (ValueError, TypeError)) or is_domain_error(exc):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise

    @router.get("/evaluation-jobs")
    async def list_evaluation_jobs() -> Dict[str, Any]:
        return service.list_evaluation_jobs()

    @router.get("/evaluations/{evaluation_id}")
    async def get_evaluation(evaluation_id: str) -> Dict[str, Any]:
        item = service.get_evaluation(evaluation_id)
        if item is None:
            raise HTTPException(status_code=404, detail="evaluation not found")
        return item

    @router.get("/evaluations/{evaluation_id}/samples")
    async def get_evaluation_samples(
        evaluation_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
    ) -> Dict[str, Any]:
        try:
            return service.get_evaluation_samples(evaluation_id, offset=offset, limit=limit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="evaluation not found") from exc

    @router.post("/evaluations/{evaluation_id}/cancel")
    async def cancel_evaluation(evaluation_id: str) -> Dict[str, Any]:
        try:
            return service.cancel_evaluation(evaluation_id)
        except Exception as exc:
            if is_domain_error(exc):
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise

    @router.post("/evaluations/{evaluation_id}/retry", status_code=202)
    async def retry_evaluation(evaluation_id: str) -> Dict[str, Any]:
        try:
            return service.retry_evaluation(evaluation_id)
        except Exception as exc:
            if is_domain_error(exc):
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise

    @router.post("/evaluation-mining/preview")
    async def preview_failure_mining(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.preview_failure_mining(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="evaluation not found") from exc
        except Exception as exc:
            if isinstance(exc, (ValueError, TypeError)) or is_domain_error(exc):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise

    @router.post("/evaluation-mining/build", status_code=202)
    async def build_failure_mined_dataset(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.build_failure_mined_dataset(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset or evaluation not found") from exc
        except Exception as exc:
            if isinstance(exc, (ValueError, TypeError)) or is_domain_error(exc):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise

    @router.get("/runs/{run_id}/launch-config")
    async def get_resolved_run_launch_config(run_id: str) -> Dict[str, Any]:
        try:
            return service.get_resolved_run_launch_config(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/eval/cohort")
    async def eval_cohort(
        run_ids: List[str] = Query(..., min_length=1),
    ) -> Dict[str, Any]:
        """Cohort eval table across N runs (Track F-K).

        Returns ``{runs, tasks, cells, best_per_task_higher_is_better}``
        — the runs × tasks grid the dashboard renders. Missing eval
        summaries surface as `available: False` on the run entry; the
        UI shows em-dashes for those rows.
        """
        return service.get_eval_cohort(list(run_ids))

    @router.get("/runs/{run_id}/eval")
    async def get_run_eval(run_id: str) -> Dict[str, Any]:
        """Per-run eval summary if `lm_eval_summary.json` exists in the
        run's output_dir; honest unavailable shape on miss."""
        return service.get_run_eval(run_id)

    @router.get("/runs/{run_id}")
    async def get_run_detail(
        run_id: str,
        include_research: bool = True,
        include_internal: bool = False,
    ) -> Dict[str, Any]:
        try:
            return service.get_run_detail(
                run_id,
                include_research=include_research,
                include_internal=include_internal,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/runs/{run_id}/cancel")
    async def cancel_run(run_id: str) -> Dict[str, Any]:
        return await service.cancel_run(run_id)

    @router.get("/runs/{run_id}/logs")
    async def get_run_logs(
        run_id: str,
        tail: int = Query(default=200, ge=1, le=5000),
    ) -> Dict[str, Any]:
        return service.get_run_logs(run_id, tail=tail)

    @router.get("/runs/{run_id}/logs/stream")
    async def stream_run_logs(
        run_id: str,
        tail: int = Query(default=200, ge=1, le=5000),
    ) -> "StreamingResponse":
        return StreamingResponse(
            service.stream_run_logs(run_id, initial_tail=tail),
            media_type="text/event-stream",
        )

    @router.get("/runs/{run_id}/samples")
    async def get_run_samples(
        run_id: str,
        cycle: Optional[int] = Query(default=None, ge=0),
        kind: str = Query(default="samples"),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> Dict[str, Any]:
        return service.get_run_samples(run_id, cycle=cycle, kind=kind, limit=limit)

    @router.get("/runs/{run_id}/live")
    async def get_run_live(
        run_id: str,
        include_research: bool = True,
    ) -> Dict[str, Any]:
        try:
            return service.get_run_live(run_id, include_research=include_research)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/runs/{run_id}/events")
    async def stream_run_events(
        run_id: str,
        include_research: bool = Query(default=True),
    ) -> "StreamingResponse":
        try:
            service.get_run_detail(run_id, include_research=False, include_internal=False)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return StreamingResponse(
            service.stream_run(run_id, include_research=include_research),
            media_type="text/event-stream",
        )

    @router.post("/runs/{run_id}/guided-recovery")
    async def apply_guided_recovery(
        run_id: str,
        payload: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        payload = payload or {}
        try:
            return await service.apply_guided_recovery(
                run_id,
                resume_latest=bool(payload.get("resume_latest", False)),
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/evaluation-batches", status_code=202)
    async def launch_evaluation_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.launch_evaluation_batch(payload)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/evaluation-batches/{batch_id}/comparison-samples")
    async def get_evaluation_batch_comparison_samples(
        batch_id: str,
        candidate_id: Optional[str] = Query(None),
        classification: Optional[str] = Query(None),
        q: Optional[str] = Query(None, max_length=200),
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
    ) -> Dict[str, Any]:
        try:
            return service.get_evaluation_batch_comparison_samples(
                batch_id,
                candidate_id=candidate_id,
                classification=classification,
                query=q,
                offset=offset,
                limit=limit,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="evaluation batch not found") from exc

    @router.get("/results/training")
    async def list_training_results(
        include_research: bool = False,
    ) -> Dict[str, Any]:
        return service.list_training_results(include_research=include_research)

    @router.get("/readiness")
    async def list_readiness() -> Dict[str, Any]:
        return service.list_readiness()

    @router.get("/docs")
    async def list_docs() -> Dict[str, Any]:
        return service.list_docs_capabilities()

    # Diagnostics — orphan launches + logs/ inventory. The /runs page
    # only shows runs that produced training_summary.json; runs that
    # aborted before that point were invisible. These endpoints back
    # the /diagnostics route in public_app.
    @router.get("/diagnostics/summary")
    async def diagnostics_summary() -> Dict[str, Any]:
        from halo_forge.public_api import diagnostics

        return diagnostics.summary(service.base_path)

    @router.get("/diagnostics/launches")
    async def diagnostics_launches() -> Dict[str, Any]:
        from halo_forge.public_api import diagnostics

        return {"items": diagnostics.inventory_launches(service.base_path)}

    @router.get("/diagnostics/logs")
    async def diagnostics_logs() -> Dict[str, Any]:
        from halo_forge.public_api import diagnostics

        return {"items": diagnostics.inventory_logs(service.base_path)}

    @router.get("/diagnostics/log")
    async def diagnostics_log_tail(path: str, tail: int = 200) -> Dict[str, Any]:
        from halo_forge.public_api import diagnostics

        return diagnostics.tail_log(
            base_path=service.base_path,
            requested_path=path,
            tail=int(tail),
        )

    # Lab v17 — cross-platform readiness, immutable data repairs, and
    # privacy-safe support artifacts. Heavy operations use the same durable
    # workstation queue as data builds and evaluation work.
    @router.get("/setup/readiness")
    async def workstation_readiness() -> Dict[str, Any]:
        return service.get_workstation_readiness()

    @router.post("/setup/remediations/{action}")
    async def apply_setup_remediation(action: str) -> Dict[str, Any]:
        try:
            return service.apply_setup_remediation(action)
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.get("/release/capability")
    async def distribution_capability() -> Dict[str, Any]:
        return service.get_distribution_capability()

    @router.get("/release/status")
    async def release_status() -> Dict[str, Any]:
        return service.get_release_status()

    @router.post("/release/qualifications", status_code=202)
    async def qualify_distribution(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.qualify_distribution(payload)
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.get("/release/qualifications")
    async def list_distribution_qualifications(
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
    ) -> Dict[str, Any]:
        return service.list_distribution_qualifications(limit=limit, offset=offset)

    @router.get("/release/qualifications/{qualification_id}")
    async def get_distribution_qualification(qualification_id: str) -> Dict[str, Any]:
        try:
            return service.get_distribution_qualification(qualification_id)
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.post("/release/qualifications/{qualification_id}/cancel")
    async def cancel_distribution_qualification(qualification_id: str) -> Dict[str, Any]:
        try:
            return service.cancel_distribution_qualification(qualification_id)
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.get("/dataset-repairs")
    async def list_dataset_repairs(
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
    ) -> Dict[str, Any]:
        return service.list_dataset_repairs(limit=limit, offset=offset)

    @router.post("/dataset-repairs", status_code=202)
    async def create_dataset_repair(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_dataset_repair(payload)
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.get("/dataset-repairs/{session_id}")
    async def get_dataset_repair(session_id: str) -> Dict[str, Any]:
        try:
            return service.get_dataset_repair(session_id)
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.get("/dataset-repairs/{session_id}/issues")
    async def list_dataset_repair_issues(
        session_id: str,
        category: Optional[str] = Query(None),
        severity: Optional[str] = Query(None),
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
    ) -> Dict[str, Any]:
        try:
            return service.list_dataset_repair_issues(
                session_id,
                category=category,
                severity=severity,
                offset=offset,
                limit=limit,
            )
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.post("/dataset-repairs/{session_id}/plans", status_code=201)
    async def create_dataset_repair_plan(
        session_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.create_dataset_repair_plan(session_id, payload)
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.get("/dataset-repair-plans/{revision_id}")
    async def get_dataset_repair_plan(revision_id: str) -> Dict[str, Any]:
        try:
            return service.get_dataset_repair_plan(revision_id)
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.post("/dataset-repairs/{session_id}/previews", status_code=202)
    async def create_dataset_repair_preview(
        session_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            return service.create_dataset_repair_preview(session_id, payload)
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.get("/dataset-repair-previews/{preview_id}")
    async def get_dataset_repair_preview(preview_id: str) -> Dict[str, Any]:
        try:
            return service.get_dataset_repair_preview(preview_id)
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.post("/dataset-repair-previews/{preview_id}/publish", status_code=201)
    async def publish_dataset_repair(preview_id: str) -> Dict[str, Any]:
        try:
            return service.publish_dataset_repair(preview_id)
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.get("/dataset-repair-revisions/{revision_id}")
    async def get_dataset_repair_revision(revision_id: str) -> Dict[str, Any]:
        try:
            return service.get_dataset_repair_revision(revision_id)
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.post("/dataset-repairs/{session_id}/rebase", status_code=201)
    async def rebase_dataset_repair(session_id: str) -> Dict[str, Any]:
        try:
            return service.rebase_dataset_repair(session_id)
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.post("/dataset-repairs/{session_id}/cancel")
    async def cancel_dataset_repair(session_id: str) -> Dict[str, Any]:
        try:
            return service.cancel_dataset_repair(session_id)
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.post("/support-bundles/preview")
    async def preview_support_bundle(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.preview_support_bundle(payload.get("categories"))
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.post("/support-bundles", status_code=202)
    async def create_support_bundle(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.create_support_bundle(payload.get("categories"))
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.get("/support-bundles/{bundle_id}")
    async def get_support_bundle(bundle_id: str) -> Dict[str, Any]:
        try:
            return service.get_support_bundle(bundle_id)
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.post("/support-bundles/{bundle_id}/verify")
    async def verify_support_bundle(bundle_id: str) -> Dict[str, Any]:
        try:
            return service.verify_support_bundle(bundle_id)
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    @router.delete("/support-bundles/{bundle_id}")
    async def delete_support_bundle(bundle_id: str) -> Dict[str, Any]:
        try:
            return service.delete_support_bundle(bundle_id)
        except Exception as exc:
            _raise_product_lab_error(exc)
            raise

    api.include_router(router)

    # Track P3 — Prometheus exposition. Lives at the root (`/metrics`)
    # because that's where scrapers look by convention. Auth bypass
    # mirrors the public_api router's loopback rule: scraping from a
    # remote Prometheus needs a token; scraping the local sidecar
    # doesn't.
    from fastapi.responses import PlainTextResponse

    @api.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics(request: Request) -> str:
        require_token(request)  # honors the loopback bypass
        from halo_forge.metrics import render_metrics

        try:
            telemetry = service.get_telemetry()
        except Exception:
            telemetry = None
        try:
            backend = service.get_backend_info()
        except Exception:
            backend = None
        try:
            run_stats = service.get_run_stats()
        except Exception:
            run_stats = None

        return render_metrics(
            telemetry=telemetry,
            run_stats=run_stats,
            backend_info=backend,
        )

    if serve_frontend:
        dist = find_frontend_dist(frontend_dist)
        if dist is not None:
            _mount_frontend(api, dist)

    return api


def _mount_frontend(api: "FastAPI", frontend_dist: Path) -> None:
    """Serve the built React app with SPA fallback routes."""
    assets_dir = frontend_dist / "assets"
    index_file = frontend_dist / "index.html"
    index_headers = {"Cache-Control": "no-store, max-age=0"}
    asset_headers = {"Cache-Control": "public, max-age=31536000, immutable"}

    @api.get("/assets/{path:path}", include_in_schema=False)
    async def serve_frontend_asset(path: str) -> "FileResponse":
        requested = (assets_dir / path).resolve()
        try:
            requested.relative_to(assets_dir)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="not found") from exc
        if not requested.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(requested, headers=asset_headers)

    @api.get("/", include_in_schema=False)
    async def serve_frontend_index() -> "FileResponse":
        return FileResponse(index_file, headers=index_headers)

    @api.get("/{path:path}", include_in_schema=False)
    async def serve_frontend_path(path: str) -> "FileResponse":
        if path.startswith("api/") or path == "metrics" or path.startswith("metrics/"):
            raise HTTPException(status_code=404, detail="not found")

        requested = (frontend_dist / path).resolve()
        try:
            requested.relative_to(frontend_dist)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="not found") from exc
        if requested.is_file():
            return FileResponse(requested, headers=asset_headers)
        return FileResponse(index_file, headers=index_headers)


app = create_app() if FastAPI is not None else None
