"""Typed, secret-safe HTTP client used by the Streamlit server process."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import TypeVar
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from personal_rag.config import Settings
from personal_rag.models import (
    ChatRequest,
    ChatResponse,
    DocumentList,
    DocumentPublic,
    ErrorEnvelope,
    JobRecord,
    JobStatus,
    SystemStatus,
    UploadReceipt,
)

DEFAULT_HTTP_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=90.0,
    write=30.0,
    pool=5.0,
)
TERMINAL_JOB_STATUSES = {JobStatus.SUCCEEDED, JobStatus.FAILED}

ModelT = TypeVar("ModelT", bound=BaseModel)


class HealthCheck(BaseModel):
    """Minimal health contract; dependency internals are intentionally ignored."""

    model_config = ConfigDict(extra="ignore")

    status: str = Field(min_length=1, max_length=64)


class ApiClientError(Exception):
    """A normalized API failure whose message is safe to show to the user."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int | None = None,
        retryable: bool = False,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.request_id = request_id


class RagApiClient:
    """Synchronous FastAPI client intended only for Streamlit's server runtime."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout: httpx.Timeout | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        normalized_url = base_url.rstrip("/")
        if not normalized_url:
            raise ValueError("base_url must not be empty")

        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.Client(
            base_url=normalized_url,
            headers=headers,
            timeout=timeout or DEFAULT_HTTP_TIMEOUT,
            transport=transport,
            follow_redirects=False,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> RagApiClient:
        """Construct the server-side client without exposing the bearer token to widgets."""

        api_key = settings.api_key.get_secret_value() if settings.api_key is not None else None
        return cls(base_url=settings.api_url, api_key=api_key)

    def __enter__(self) -> RagApiClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def health_live(self) -> HealthCheck:
        data = self._request_json("GET", "/health/live")
        return self._parse_model(data, HealthCheck)

    def health_ready(self) -> HealthCheck:
        data = self._request_json("GET", "/health/ready")
        return self._parse_model(data, HealthCheck)

    def get_status(self) -> SystemStatus:
        data = self._request_json("GET", "/api/v1/status")
        return self._parse_model(data, SystemStatus)

    def list_documents(self, *, limit: int = 100, offset: int = 0) -> DocumentList:
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("offset must not be negative")
        data = self._request_json(
            "GET",
            "/api/v1/documents",
            params={"limit": limit, "offset": offset},
        )
        return self._parse_model(data, DocumentList)

    def list_all_documents(self, *, max_documents: int = 2000) -> list[DocumentPublic]:
        """Read the bounded library through paginated API calls."""

        if max_documents < 1 or max_documents > 10_000:
            raise ValueError("max_documents must be between 1 and 10000")

        documents: list[DocumentPublic] = []
        page_size = min(100, max_documents)
        while len(documents) < max_documents:
            page = self.list_documents(limit=page_size, offset=len(documents))
            documents.extend(page.items)
            if not page.items or len(documents) >= page.total:
                break
        return documents[:max_documents]

    def get_document(self, document_id: str) -> DocumentPublic:
        data = self._request_json("GET", f"/api/v1/documents/{self._path_identifier(document_id)}")
        return self._parse_model(data, DocumentPublic)

    def upload_document(
        self,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> UploadReceipt:
        data = self._request_json(
            "POST",
            "/api/v1/documents",
            files={"file": (filename, content, content_type)},
        )
        return self._parse_model(data, UploadReceipt)

    def delete_document(self, document_id: str) -> JobRecord:
        data = self._request_json(
            "DELETE", f"/api/v1/documents/{self._path_identifier(document_id)}"
        )
        return self._parse_job(data)

    def reindex_document(self, document_id: str) -> JobRecord:
        data = self._request_json(
            "POST",
            f"/api/v1/documents/{self._path_identifier(document_id)}/reindex",
        )
        return self._parse_job(data)

    def get_job(self, job_id: str) -> JobRecord:
        data = self._request_json("GET", f"/api/v1/jobs/{self._path_identifier(job_id)}")
        return self._parse_model(data, JobRecord)

    def chat(self, request: ChatRequest) -> ChatResponse:
        data = self._request_json(
            "POST",
            "/api/v1/chat",
            json_body=request.model_dump(mode="json", exclude_none=True),
        )
        return self._parse_model(data, ChatResponse)

    def wait_for_job(
        self,
        job_id: str,
        *,
        timeout_seconds: float,
        poll_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> JobRecord:
        """Poll durable server truth until terminal state or a bounded timeout."""

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")

        deadline = clock() + timeout_seconds
        while True:
            job = self.get_job(job_id)
            if job.status in TERMINAL_JOB_STATUSES:
                return job
            now = clock()
            if now >= deadline:
                raise ApiClientError(
                    code="job_poll_timeout",
                    message="Processing is still running. Its progress is saved; refresh shortly.",
                    status_code=None,
                    retryable=True,
                )
            sleeper(min(poll_seconds, max(0.1, deadline - now)))

    @staticmethod
    def _path_identifier(value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("identifier must not be empty")
        return quote(cleaned, safe="")

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, int] | None = None,
        json_body: object | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> object:
        try:
            response = self._client.request(
                method,
                path,
                params=params,
                json=json_body,
                files=files,
            )
        except httpx.TimeoutException as exc:
            raise ApiClientError(
                code="api_timeout",
                message="The knowledge service took too long to respond. Please retry.",
                retryable=True,
            ) from exc
        except httpx.RequestError as exc:
            raise ApiClientError(
                code="api_unavailable",
                message="The knowledge service is unavailable. Check that it is running and retry.",
                retryable=True,
            ) from exc

        if response.is_error:
            raise self._error_from_response(response)
        if response.status_code == httpx.codes.NO_CONTENT:
            return {}
        try:
            data: object = response.json()
        except ValueError as exc:
            raise self._invalid_response_error() from exc
        return data

    @staticmethod
    def _parse_model(data: object, model_type: type[ModelT]) -> ModelT:
        try:
            return model_type.model_validate(data)
        except (ValidationError, TypeError, ValueError) as exc:
            raise RagApiClient._invalid_response_error() from exc

    @classmethod
    def _parse_job(cls, data: object) -> JobRecord:
        payload = data
        if isinstance(data, Mapping) and "job" in data:
            payload = data["job"]
        return cls._parse_model(payload, JobRecord)

    @staticmethod
    def _invalid_response_error() -> ApiClientError:
        return ApiClientError(
            code="invalid_response",
            message="The knowledge service returned an unexpected response. Please retry.",
            status_code=502,
            retryable=True,
        )

    @staticmethod
    def _error_from_response(response: httpx.Response) -> ApiClientError:
        if response.status_code == httpx.codes.UNAUTHORIZED:
            return ApiClientError(
                code="authentication_failed",
                message="The knowledge service rejected its server-side credentials.",
                status_code=response.status_code,
                retryable=False,
            )

        try:
            envelope = ErrorEnvelope.model_validate(response.json())
        except (ValidationError, TypeError, ValueError):
            fallback_message = (
                "The knowledge service is temporarily unavailable. Please retry."
                if response.status_code >= 500
                else "The knowledge service could not complete that request."
            )
            return ApiClientError(
                code="api_error",
                message=fallback_message,
                status_code=response.status_code,
                retryable=response.status_code >= 500,
            )

        detail = envelope.error
        return ApiClientError(
            code=detail.code,
            message=detail.message,
            status_code=response.status_code,
            retryable=detail.retryable,
            request_id=detail.request_id,
        )
