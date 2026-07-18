"""Sanitized dependency and collection status."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request

from personal_rag.models import DependencyState, SystemStatus

router = APIRouter(prefix="/status", tags=["status"])


def _worker_state(container: Any) -> tuple[DependencyState, datetime | None]:
    last_seen = container.repository.read_worker_heartbeat()
    if last_seen is None:
        return DependencyState(name="worker", status="unavailable", detail="No heartbeat"), None
    if isinstance(last_seen, str):
        last_seen = datetime.fromisoformat(last_seen)
    if last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=UTC)
    age = (datetime.now(UTC) - last_seen).total_seconds()
    if age > container.settings.worker_stale_seconds:
        return (
            DependencyState(name="worker", status="unavailable", detail="Heartbeat is stale"),
            last_seen,
        )
    return DependencyState(name="worker", status="ready"), last_seen


@router.get("", response_model=SystemStatus)
def system_status(request: Request) -> SystemStatus:
    container = request.app.state.container
    stats = container.repository.get_statistics()
    dependencies: list[DependencyState] = [DependencyState(name="metadata", status="ready")]

    if container.vector_store is None:
        dependencies.append(DependencyState(name="qdrant", status="unavailable"))
        dependencies.append(DependencyState(name="vector_inventory", status="unavailable"))
    else:
        try:
            container.vector_store.heartbeat()
            dependencies.append(DependencyState(name="qdrant", status="ready"))
            actual_chunks = container.vector_store.count()
            expected_chunks = stats["chunk_count"]
            if actual_chunks == expected_chunks:
                dependencies.append(DependencyState(name="vector_inventory", status="ready"))
            else:
                dependencies.append(
                    DependencyState(
                        name="vector_inventory",
                        status="degraded",
                        detail=(
                            f"Metadata expects {expected_chunks} chunks; "
                            f"Qdrant contains {actual_chunks}"
                        ),
                    )
                )
        except Exception:
            dependencies.append(DependencyState(name="qdrant", status="unavailable"))
            dependencies.append(DependencyState(name="vector_inventory", status="unavailable"))

    provider_errors = container.settings.provider_configuration_errors
    dependencies.append(
        DependencyState(
            name="providers",
            status="ready" if not provider_errors else "not_configured",
            detail="; ".join(provider_errors) if provider_errors else None,
        )
    )
    worker, worker_last_seen = _worker_state(container)
    dependencies.append(worker)

    if provider_errors:
        overall = "needs_setup"
    elif any(item.status in {"degraded", "unavailable"} for item in dependencies):
        overall = "degraded"
    else:
        overall = "ready"

    return SystemStatus(
        status=overall,
        collection=container.settings.qdrant_collection,
        document_count=stats["document_count"],
        ready_document_count=stats["ready_document_count"],
        chunk_count=stats["chunk_count"],
        queued_job_count=stats["queued_job_count"],
        embedding_provider=container.settings.embedding_provider,
        embedding_model=container.settings.embedding_model,
        embedding_dimensions=container.settings.embedding_dimensions,
        chat_provider=container.settings.chat_provider,
        chat_model=container.settings.chat_model,
        dependencies=dependencies,
        worker_last_seen_at=worker_last_seen,
    )
