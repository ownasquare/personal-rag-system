from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import personal_rag.api.app as app_module
from personal_rag.api.app import create_app
from personal_rag.config import Settings
from personal_rag.errors import ProviderError
from personal_rag.models import (
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentPublic,
    DocumentRecord,
    DocumentStatus,
    JobKind,
    JobRecord,
    JobStage,
    JobStatus,
    UploadReceipt,
)


class FakeRepository:
    def __init__(self) -> None:
        self.documents: dict[str, DocumentRecord] = {}
        self.jobs: dict[str, JobRecord] = {}

    def get_statistics(self) -> dict[str, int]:
        active = [
            item for item in self.documents.values() if item.status is not DocumentStatus.DELETED
        ]
        ready = [item for item in active if item.status is DocumentStatus.READY]
        queued = [item for item in self.jobs.values() if item.status is JobStatus.QUEUED]
        return {
            "document_count": len(active),
            "ready_document_count": len(ready),
            "chunk_count": sum(item.chunk_count for item in ready),
            "queued_job_count": len(queued),
        }

    def read_worker_heartbeat(self) -> datetime:
        return datetime.now(UTC)

    def create_document_with_job(self, **values: Any) -> UploadReceipt:
        now = datetime.now(UTC)
        document = DocumentRecord(
            id=values["document_id"],
            display_name=values["display_name"],
            stored_path=values["stored_path"],
            content_type=values["content_type"],
            extension=values["extension"],
            content_sha256=values["content_sha256"],
            size_bytes=values["size_bytes"],
            status=DocumentStatus.QUEUED,
            embedding_fingerprint=values["embedding_fingerprint"],
            created_at=now,
            updated_at=now,
        )
        job = JobRecord(
            id=uuid4().hex,
            document_id=document.id,
            kind=JobKind.INGEST,
            status=JobStatus.QUEUED,
            stage=JobStage.QUEUED,
            created_at=now,
            updated_at=now,
        )
        self.documents[document.id] = document
        self.jobs[job.id] = job
        return UploadReceipt(
            document=DocumentPublic.from_record(document), job=job, duplicate=False
        )

    def list_documents(
        self, *, limit: int, offset: int, status: DocumentStatus | None = None
    ) -> list[DocumentRecord]:
        items = list(self.documents.values())
        if status is not None:
            items = [item for item in items if item.status is status]
        return items[offset : offset + limit]

    def count_documents(self, *, status: DocumentStatus | None = None) -> int:
        return len(self.list_documents(limit=1000, offset=0, status=status))

    def get_document(self, document_id: str) -> DocumentRecord | None:
        return self.documents.get(document_id)

    def get_job(self, job_id: str) -> JobRecord | None:
        return self.jobs.get(job_id)

    def request_reindex(self, document_id: str) -> JobRecord | None:
        return self._lifecycle_job(document_id, JobKind.REINDEX)

    def request_delete(self, document_id: str) -> JobRecord | None:
        return self._lifecycle_job(document_id, JobKind.DELETE)

    def _lifecycle_job(self, document_id: str, kind: JobKind) -> JobRecord | None:
        if document_id not in self.documents:
            return None
        now = datetime.now(UTC)
        job = JobRecord(
            id=uuid4().hex,
            document_id=document_id,
            kind=kind,
            status=JobStatus.QUEUED,
            stage=JobStage.QUEUED,
            created_at=now,
            updated_at=now,
        )
        self.jobs[job.id] = job
        return job


class FakeVectorStore:
    def __init__(self) -> None:
        self.point_count = 0

    def heartbeat(self) -> int:
        return 1

    def count(self) -> int:
        return self.point_count


class FakeRagService:
    def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            answer="The launch key was cobalt blue [S1].",
            citations=[
                Citation(
                    label="S1",
                    document_id="doc-1",
                    chunk_id="chunk-1",
                    document_name="notes.md",
                    section="Launch",
                    snippet="The launch key was cobalt blue.",
                    score=0.9,
                )
            ],
            no_answer=False,
        )


class FailingRagService:
    def __init__(self, *, retryable: bool) -> None:
        self.retryable = retryable

    def chat(self, request: ChatRequest) -> ChatResponse:
        del request
        code = "vector_store_unavailable" if self.retryable else "vector_query_rejected"
        message = (
            "The vector index is temporarily unavailable."
            if self.retryable
            else "The vector index could not process the query."
        )
        raise ProviderError(code, message, retryable=self.retryable) from RuntimeError(
            "private Qdrant response must not escape"
        )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        auth_enabled=True,
        api_key=SecretStr("test-api-key-which-is-long"),
        openai_api_key=SecretStr("test-provider-key"),
        qdrant_mode="memory",
        data_dir=tmp_path,
        json_logs=False,
    )


@pytest.fixture
def repository() -> FakeRepository:
    return FakeRepository()


@pytest.fixture
def client(settings: Settings, repository: FakeRepository) -> TestClient:
    container = SimpleNamespace(
        settings=settings,
        repository=repository,
        vector_store=FakeVectorStore(),
        rag_service=FakeRagService(),
        startup_errors=[],
        close=lambda: None,
    )
    app = create_app(settings, container=container)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-api-key-which-is-long"}


def test_liveness_is_public_and_has_request_id(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"
    assert len(response.headers["X-Request-ID"]) >= 8


def test_readiness_fails_closed_when_vector_inventory_diverges(client: TestClient) -> None:
    client.app.state.container.vector_store.point_count = 1

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["qdrant"] == "ready"
    assert response.json()["checks"]["vector_inventory"] == "degraded"


def test_sanitized_status_reports_ready_dependency_inventory(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.get("/api/v1/status", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    dependencies = {item["name"]: item for item in response.json()["dependencies"]}
    assert dependencies["metadata"]["status"] == "ready"
    assert dependencies["qdrant"]["status"] == "ready"
    assert dependencies["vector_inventory"]["status"] == "ready"
    assert dependencies["providers"]["status"] == "ready"
    assert dependencies["worker"]["status"] == "ready"
    assert "api_key" not in response.text


def test_sanitized_status_explains_vector_inventory_divergence(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    client.app.state.container.vector_store.point_count = 2

    response = client.get("/api/v1/status", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    dependencies = {item["name"]: item for item in response.json()["dependencies"]}
    assert dependencies["vector_inventory"]["status"] == "degraded"
    assert "expects 0 chunks" in dependencies["vector_inventory"]["detail"]


def test_data_routes_require_valid_bearer_token(client: TestClient) -> None:
    missing = client.get("/api/v1/documents")
    wrong = client.get("/api/v1/documents", headers={"Authorization": "Bearer definitely-wrong"})
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert missing.json()["error"]["code"] == "unauthorized"
    assert missing.headers["WWW-Authenticate"] == "Bearer"


def test_upload_returns_durable_job(
    client: TestClient,
    auth_headers: dict[str, str],
    repository: FakeRepository,
    settings: Settings,
) -> None:
    response = client.post(
        "/api/v1/documents",
        headers={**auth_headers, "Idempotency-Key": "upload-notes-001"},
        files={"file": ("notes.md", b"The launch key is cobalt blue.", "text/markdown")},
    )
    assert response.status_code == 202
    payload = response.json()
    assert payload["document"]["display_name"] == "notes.md"
    assert payload["job"]["status"] == "queued"
    assert payload["duplicate"] is False
    record = next(iter(repository.documents.values()))
    assert Path(record.stored_path).is_absolute() is False
    assert (settings.uploads_dir / record.stored_path).is_file()
    assert "stored_path" not in payload["document"]
    assert "content_sha256" not in payload["document"]


def test_upload_rejects_unsupported_and_empty_files(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    unsupported = client.post(
        "/api/v1/documents",
        headers=auth_headers,
        files={"file": ("archive.zip", b"PK", "application/zip")},
    )
    empty = client.post(
        "/api/v1/documents",
        headers=auth_headers,
        files={"file": ("notes.txt", b"", "text/plain")},
    )
    assert unsupported.status_code == 415
    assert unsupported.json()["error"]["code"] == "unsupported_file_type"
    assert empty.status_code == 422
    assert empty.json()["error"]["code"] == "empty_file"


def test_upload_rejects_oversized_declared_body_before_multipart_parsing(
    client: TestClient, auth_headers: dict[str, str], settings: Settings
) -> None:
    response = client.post(
        "/api/v1/documents",
        headers={
            **auth_headers,
            "Content-Type": "multipart/form-data; boundary=unused",
            "Content-Length": str(settings.upload_max_bytes + 2 * 1024 * 1024),
        },
        content=b"unused",
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_document_list_and_job_readback(client: TestClient, auth_headers: dict[str, str]) -> None:
    uploaded = client.post(
        "/api/v1/documents",
        headers=auth_headers,
        files={"file": ("notes.txt", b"grounded data", "text/plain")},
    ).json()
    listed = client.get("/api/v1/documents", headers=auth_headers)
    job = client.get(f"/api/v1/jobs/{uploaded['job']['id']}", headers=auth_headers)
    assert listed.status_code == 200
    assert listed.json()["total"] == 1
    assert job.status_code == 200
    assert job.json()["document_id"] == uploaded["document"]["id"]


def test_chat_returns_backend_citations(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": "What color was the launch key?", "top_k": 5},
    )
    assert response.status_code == 200
    assert response.json()["answer"].endswith("[S1].")
    assert response.json()["citations"][0]["chunk_id"] == "chunk-1"
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


@pytest.mark.parametrize(
    ("retryable", "expected_code"),
    [
        (True, "vector_store_unavailable"),
        (False, "vector_query_rejected"),
    ],
)
def test_chat_returns_safe_vector_query_error_contract(
    client: TestClient,
    auth_headers: dict[str, str],
    retryable: bool,
    expected_code: str,
) -> None:
    client.app.state.container.rag_service = FailingRagService(retryable=retryable)

    response = client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": "What color was the launch key?"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == expected_code
    assert response.json()["error"]["retryable"] is retryable
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]
    assert "private Qdrant response" not in response.text


def test_validation_error_does_not_echo_invalid_input(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    secret_like_input = "private-text-that-must-not-be-echoed"
    response = client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": "", "history": [{"role": "tool", "content": secret_like_input}]},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert secret_like_input not in response.text


def test_chat_history_has_a_schema_level_item_limit(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={
            "message": "question",
            "history": [{"role": "user", "content": "item"}] * 101,
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_production_unexpected_error_logging_never_receives_exception_text(
    settings: Settings,
    repository: FakeRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    class Recorder:
        def error(self, event: str, **values: object) -> None:
            events.append((event, values))

        def exception(self, event: str, **values: object) -> None:
            raise AssertionError(f"production must not log a traceback: {event} {values}")

    monkeypatch.setattr(app_module, "get_logger", lambda _name: Recorder())
    production = settings.model_copy(
        update={
            "environment": "production",
            "qdrant_api_key": SecretStr("test-qdrant-key-different-and-long"),
        }
    )
    container = SimpleNamespace(
        settings=production,
        repository=repository,
        vector_store=FakeVectorStore(),
        rag_service=FakeRagService(),
        startup_errors=[],
        close=lambda: None,
    )
    application = create_app(production, container=container)
    secret_text = "provider-secret-that-must-not-enter-logs"

    @application.get("/explode", include_in_schema=False)
    def explode() -> None:
        raise RuntimeError(secret_text)

    with TestClient(application, raise_server_exceptions=False) as test_client:
        response = test_client.get("/explode")

    assert response.status_code == 500
    assert secret_text not in response.text
    assert secret_text not in repr(events)
    assert events[0][0] == "unhandled_error"
    assert events[0][1]["error_type"] == "RuntimeError"
