"""Content-safe logging, request IDs, security headers, and Prometheus metrics."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextvars import ContextVar
from typing import Any, cast

import structlog
from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)
SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")

REQUESTS = Counter(
    "personal_rag_http_requests_total",
    "HTTP requests handled by the API",
    ("method", "route", "status"),
)
REQUEST_DURATION = Histogram(
    "personal_rag_http_request_duration_seconds",
    "HTTP request latency",
    ("method", "route"),
)


def configure_logging(*, level: str, json_logs: bool) -> None:
    """Configure stdlib and structlog without logging request or document bodies."""

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    logging.basicConfig(level=getattr(logging, level), format="%(message)s", force=True)
    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))


def current_request_id() -> str | None:
    return request_id_context.get()


class _RequestBodyTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
    """Reject oversized multipart bodies before framework-level form parsing."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_upload_bytes: int,
        multipart_overhead_bytes: int = 1024 * 1024,
    ) -> None:
        self.app = app
        self.limit = max_upload_bytes + multipart_overhead_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/api/v1/documents"
        ):
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = 0
            if declared > self.limit:
                await self._send_rejection(send)
                return

        consumed = 0

        async def limited_receive() -> Message:
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > self.limit:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await self._send_rejection(send)

    @staticmethod
    async def _send_rejection(send: Send) -> None:
        payload = json.dumps(
            {
                "error": {
                    "code": "request_too_large",
                    "message": "The upload request exceeds the configured size limit.",
                    "retryable": False,
                    "request_id": current_request_id(),
                }
            },
            separators=(",", ":"),
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a safe correlation ID and structured request summary."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if SAFE_REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex
        token = request_id_context.set(request_id)
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            status_code = response.status_code if response is not None else 500
            route_object = request.scope.get("route")
            route = getattr(route_object, "path", "unmatched")
            get_logger("http").info(
                "request_complete",
                service="api",
                method=request.method,
                route=route,
                status_code=status_code,
                duration_ms=duration_ms,
            )
            structlog.contextvars.clear_contextvars()
            request_id_context.reset(token)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Apply conservative headers to API responses."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")
        if request.url.path in {"/docs", "/redoc"}:
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data: https://fastapi.tiangolo.com; frame-ancestors 'none'",
            )
        else:
            response.headers.setdefault(
                "Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'"
            )
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect low-cardinality route metrics without content or identifiers."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        route_object = request.scope.get("route")
        route = getattr(route_object, "path", "unmatched")
        REQUESTS.labels(request.method, route, str(response.status_code)).inc()
        REQUEST_DURATION.labels(request.method, route).observe(time.perf_counter() - started)
        return response


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
