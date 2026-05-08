"""Standalone public API app for the user-facing halo-forge frontend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .service import PublicApiService

FASTAPI_IMPORT_ERROR: Exception | None = None

try:
    from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
except ImportError as exc:  # pragma: no cover - exercised in environments without FastAPI
    FASTAPI_IMPORT_ERROR = exc
    APIRouter = Depends = FastAPI = HTTPException = Query = Request = StreamingResponse = None  # type: ignore[assignment]
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

    @router.get("/health")
    async def health() -> Dict[str, Any]:
        return {"ok": True}

    @router.get("/backend")
    async def backend_info() -> Dict[str, Any]:
        return service.get_backend_info()

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

    @router.get("/train/models")
    async def list_suggested_models() -> Dict[str, Any]:
        return {"items": service.list_suggested_models()}

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
            child_run_id=run_id, parent_run_id=parent_run_id,
        )
        if not ok:
            raise HTTPException(
                status_code=404, detail="lineage edge not found"
            )
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
                status_code=400, detail="messages must be a non-empty list",
            )
        return service.playground_chat(
            messages=messages,
            model=payload.get("model"),
            max_tokens=int(payload.get("max_tokens") or 256),
            temperature=float(payload.get("temperature") or 0.7),
            top_p=float(payload.get("top_p") or 1.0),
            stop=payload.get("stop"),
            serve_url=payload.get("serve_url"),
            api_key=payload.get("api_key"),
        )

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
    async def update_registry_entry(
        entry_id: int, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
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

    return api


app = create_app() if FastAPI is not None else None
