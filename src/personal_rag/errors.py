"""Stable, safe application errors."""

from __future__ import annotations


class RagError(Exception):
    """An expected application error safe to expose through the API."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


class ConfigurationError(RagError):
    """A configuration error that prevents one or more operations."""

    def __init__(self, message: str) -> None:
        super().__init__("configuration_error", message, status_code=503, retryable=False)


class ProviderError(RagError):
    """A normalized provider failure."""

    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        super().__init__(code, message, status_code=503, retryable=retryable)
