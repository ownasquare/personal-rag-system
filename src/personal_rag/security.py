"""Authentication and security helpers."""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from personal_rag.errors import RagError

bearer_scheme = HTTPBearer(auto_error=False)
SAFE_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")


async def require_api_key(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)] = None,
) -> None:
    """Require the configured local bearer token on protected routes."""

    settings = request.app.state.container.settings
    if not settings.auth_enabled:
        return

    configured = settings.api_key
    supplied = credentials.credentials if credentials is not None else ""
    expected = configured.get_secret_value() if configured is not None else ""
    valid_scheme = credentials is not None and credentials.scheme.lower() == "bearer"
    if not valid_scheme or not expected or not hmac.compare_digest(supplied, expected):
        raise RagError(
            "unauthorized",
            "A valid bearer token is required.",
            status_code=401,
            retryable=False,
        )


def validate_idempotency_key(value: str | None) -> str | None:
    """Validate a retry key without storing or logging its raw value unnecessarily."""

    if value is None:
        return None
    if not SAFE_IDEMPOTENCY_KEY.fullmatch(value):
        raise RagError(
            "invalid_idempotency_key",
            "Idempotency-Key must be 8-128 URL-safe characters.",
            status_code=400,
        )
    return value


def idempotency_key_digest(value: str | None) -> str | None:
    """Create a one-way storage key for an accepted idempotency token."""

    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
