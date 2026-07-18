"""FastAPI dependency accessors for the application container."""

from __future__ import annotations

from typing import Any

from fastapi import Request

from personal_rag.errors import ConfigurationError, RagError


def get_container(request: Request) -> Any:
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise RagError(
            "service_unavailable",
            "The application is still starting.",
            status_code=503,
            retryable=True,
        )
    return container


def get_repository(request: Request) -> Any:
    return get_container(request).repository


def get_rag_service(request: Request) -> Any:
    container = get_container(request)
    rag_service = container.rag_service
    if rag_service is None:
        details = getattr(container, "startup_errors", [])
        suffix = f" ({details[0]})" if details else ""
        raise ConfigurationError(f"The RAG providers are not ready{suffix}.")
    return rag_service
