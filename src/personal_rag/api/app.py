"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from personal_rag import __version__
from personal_rag.api.routes import chat, conversations, documents, health, jobs, status
from personal_rag.config import Settings, get_settings
from personal_rag.container import AppContainer, build_container
from personal_rag.errors import RagError
from personal_rag.models import ErrorDetail, ErrorEnvelope
from personal_rag.observability import (
    MetricsMiddleware,
    RequestBodyLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
    configure_logging,
    current_request_id,
    get_logger,
)
from personal_rag.security import require_api_key

ContainerFactory = Callable[[Settings], AppContainer]


def _error_response(
    *, code: str, message: str, status_code: int, retryable: bool = False
) -> JSONResponse:
    envelope = ErrorEnvelope(
        error=ErrorDetail(
            code=code,
            message=message,
            retryable=retryable,
            request_id=current_request_id(),
        )
    )
    headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else None
    return JSONResponse(
        status_code=status_code,
        content=envelope.model_dump(mode="json"),
        headers=headers,
    )


def create_app(
    settings: Settings | None = None,
    *,
    container: AppContainer | Any | None = None,
    container_factory: ContainerFactory = build_container,
) -> FastAPI:
    """Create an independently testable FastAPI application."""

    resolved_settings = settings or get_settings()
    configure_logging(level=resolved_settings.log_level, json_logs=resolved_settings.json_logs)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        owned = container is None
        application.state.container = (
            container if container is not None else container_factory(resolved_settings)
        )
        try:
            yield
        finally:
            if owned:
                application.state.container.close()

    docs_enabled = resolved_settings.environment != "production"
    application = FastAPI(
        title="Personal RAG API",
        description="Durable document ingestion and grounded chat with verifiable citations.",
        version=__version__,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
        lifespan=lifespan,
    )
    application.state.container = container

    application.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_upload_bytes=resolved_settings.upload_max_bytes,
    )
    application.add_middleware(SecurityHeadersMiddleware)
    if resolved_settings.metrics_enabled:
        application.add_middleware(MetricsMiddleware)
    application.add_middleware(RequestContextMiddleware)

    @application.exception_handler(RagError)
    async def handle_rag_error(_request: Request, error: RagError) -> JSONResponse:
        return _error_response(
            code=error.code,
            message=error.message,
            status_code=error.status_code,
            retryable=error.retryable,
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            code="validation_error",
            message="The request did not match the API contract.",
            status_code=422,
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, error: Exception) -> JSONResponse:
        logger = get_logger("api")
        details = {
            "error_type": type(error).__name__,
            "request_id": current_request_id(),
        }
        if resolved_settings.environment == "production":
            logger.error("unhandled_error", **details)
        else:
            logger.exception("unhandled_error", **details)
        return _error_response(
            code="internal_error",
            message="An unexpected error occurred.",
            status_code=500,
            retryable=False,
        )

    application.include_router(health.router)
    api = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])
    api.include_router(status.router)
    api.include_router(documents.router)
    api.include_router(jobs.router)
    api.include_router(conversations.router)
    api.include_router(chat.router)
    application.include_router(api)

    @application.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "name": "Personal RAG API",
            "version": __version__,
            "health": "/health/live",
        }

    return application


app = create_app()
