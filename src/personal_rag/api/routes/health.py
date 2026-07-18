"""Liveness, readiness, version, and metrics endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request, Response

from personal_rag import __version__
from personal_rag.observability import metrics_response

router = APIRouter(tags=["health"])


@router.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "alive", "version": __version__}


@router.get("/health/ready")
def readiness(request: Request, response: Response) -> dict[str, Any]:
    container = request.app.state.container
    checks: dict[str, str] = {}
    stats: dict[str, int] | None = None

    try:
        stats = container.repository.get_statistics()
        checks["metadata"] = "ready"
    except Exception:
        checks["metadata"] = "unavailable"

    if container.vector_store is None:
        checks["qdrant"] = "unavailable"
        checks["vector_inventory"] = "unavailable"
    else:
        try:
            container.vector_store.heartbeat()
            checks["qdrant"] = "ready"
            checks["vector_inventory"] = (
                "ready"
                if stats is not None and container.vector_store.count() == stats["chunk_count"]
                else "degraded"
            )
        except Exception:
            checks["qdrant"] = "unavailable"
            checks["vector_inventory"] = "unavailable"

    provider_errors = container.settings.provider_configuration_errors
    checks["providers"] = "ready" if not provider_errors else "not_configured"

    last_seen = container.repository.read_worker_heartbeat()
    if last_seen is None:
        checks["worker"] = "unavailable"
    else:
        if isinstance(last_seen, str):
            last_seen = datetime.fromisoformat(last_seen)
        if last_seen.tzinfo is None:
            last_seen = last_seen.replace(tzinfo=UTC)
        age = (datetime.now(UTC) - last_seen).total_seconds()
        checks["worker"] = (
            "ready" if age <= container.settings.worker_stale_seconds else "unavailable"
        )

    ready = all(value == "ready" for value in checks.values())
    if not ready:
        response.status_code = 503
    return {"status": "ready" if ready else "not_ready", "checks": checks}


@router.get("/version")
def version() -> dict[str, str]:
    return {"name": "personal-rag-system", "version": __version__}


@router.get("/metrics", include_in_schema=False)
def metrics(request: Request) -> Response:
    if not request.app.state.container.settings.metrics_enabled:
        return Response(status_code=404)
    return metrics_response()
