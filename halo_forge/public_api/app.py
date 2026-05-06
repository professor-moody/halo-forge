"""Standalone public API app for the user-facing halo-forge frontend."""

from __future__ import annotations

from typing import Any, Dict

from .service import PublicApiService

FASTAPI_IMPORT_ERROR: Exception | None = None

try:
    from fastapi import APIRouter, FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
except ImportError as exc:  # pragma: no cover - exercised in environments without FastAPI
    FASTAPI_IMPORT_ERROR = exc
    APIRouter = FastAPI = HTTPException = Query = StreamingResponse = None  # type: ignore[assignment]
    CORSMiddleware = None  # type: ignore[assignment]


def create_app() -> "FastAPI":
    """Create the standalone public API app."""
    if FastAPI is None:
        raise RuntimeError(
            "FastAPI is required for the public API. Install halo-forge with FastAPI/uvicorn dependencies."
        ) from FASTAPI_IMPORT_ERROR

    service = PublicApiService()
    api = FastAPI(
        title="halo-forge public API",
        version="0.1.0",
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
    router = APIRouter(prefix="/api/public", tags=["public"])

    @router.get("/health")
    async def health() -> Dict[str, Any]:
        return {"ok": True}

    @router.get("/backend")
    async def backend_info() -> Dict[str, Any]:
        return service.get_backend_info()

    @router.get("/telemetry")
    async def telemetry() -> Dict[str, Any]:
        return service.get_telemetry()

    @router.get("/dashboard")
    async def dashboard_summary() -> Dict[str, Any]:
        return service.get_dashboard_summary()

    @router.get("/train/presets")
    async def list_training_presets() -> Dict[str, Any]:
        return {"items": service.list_training_presets()}

    @router.post("/train/preflight")
    async def preflight_training(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return service.preflight_training(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/train/launch")
    async def launch_training(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return await service.launch_training(payload)
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

    api.include_router(router)
    return api


app = create_app() if FastAPI is not None else None
