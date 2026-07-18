from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
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
from personal_rag.errors import ProviderError, RagError
from personal_rag.models import (
    ChatHistoryMessage,
    ChatRequest,
    ChatResponse,
    Citation,
    ConversationSummary,
    ConversationTurn,
    ConversationTurnReservation,
    ConversationTurnStatus,
    DocumentPublic,
    DocumentRecord,
    DocumentSort,
    DocumentStatus,
    JobKind,
    JobRecord,
    JobStage,
    JobStatus,
    SortOrder,
    UploadReceipt,
)


class FakeRepository:
    def __init__(self) -> None:
        self.documents: dict[str, DocumentRecord] = {}
        self.jobs: dict[str, JobRecord] = {}
        self.conversations: dict[str, ConversationSummary] = {}
        self.turns: dict[str, ConversationTurn] = {}
        self.turn_fingerprints: dict[str, str] = {}
        self.turn_expiries: dict[str, datetime] = {}
        self.turn_tokens: dict[str, str] = {}

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
        self,
        *,
        limit: int,
        offset: int,
        query: str | None = None,
        statuses: Sequence[DocumentStatus] | None = None,
        sort: DocumentSort = DocumentSort.CREATED,
        order: SortOrder = SortOrder.DESC,
    ) -> list[DocumentRecord]:
        items = [
            item for item in self.documents.values() if item.status is not DocumentStatus.DELETED
        ]
        if statuses:
            selected_statuses = {DocumentStatus(value) for value in statuses}
            items = [item for item in items if item.status in selected_statuses]
        if query:
            normalized_query = unicodedata.normalize("NFC", query).casefold()
            items = [
                item
                for item in items
                if normalized_query
                in unicodedata.normalize("NFC", f"{item.display_name} {item.extension}").casefold()
            ]
        sort_value = DocumentSort(sort)
        reverse = SortOrder(order) is SortOrder.DESC
        if sort_value is DocumentSort.CREATED:
            items.sort(key=lambda item: (item.created_at, item.id), reverse=reverse)
        elif sort_value is DocumentSort.UPDATED:
            items.sort(key=lambda item: (item.updated_at, item.id), reverse=reverse)
        else:
            items.sort(
                key=lambda item: (
                    unicodedata.normalize("NFC", item.display_name).casefold(),
                    item.id,
                ),
                reverse=reverse,
            )
        return items[offset : offset + limit]

    def count_documents(
        self,
        *,
        query: str | None = None,
        statuses: Sequence[DocumentStatus] | None = None,
    ) -> int:
        return len(
            self.list_documents(
                limit=10_000,
                offset=0,
                query=query,
                statuses=statuses,
            )
        )

    def get_document(self, document_id: str) -> DocumentRecord | None:
        return self.documents.get(document_id)

    def get_job(self, job_id: str) -> JobRecord | None:
        return self.jobs.get(job_id)

    def list_jobs(
        self,
        *,
        limit: int,
        offset: int,
        status: JobStatus | None = None,
        document_id: str | None = None,
    ) -> list[JobRecord]:
        items = list(reversed(self.jobs.values()))
        if status is not None:
            items = [item for item in items if item.status is status]
        if document_id is not None:
            items = [item for item in items if item.document_id == document_id]
        return items[offset : offset + limit]

    def count_jobs(
        self,
        *,
        status: JobStatus | None = None,
        document_id: str | None = None,
    ) -> int:
        return len(
            self.list_jobs(
                limit=10_000,
                offset=0,
                status=status,
                document_id=document_id,
            )
        )

    def create_conversation(self, title: str | None = None) -> ConversationSummary:
        now = datetime.now(UTC)
        conversation = ConversationSummary(
            id=uuid4().hex,
            title=title or "New conversation",
            turn_count=0,
            created_at=now,
            updated_at=now,
        )
        self.conversations[conversation.id] = conversation
        return conversation

    def get_conversation(self, conversation_id: str) -> ConversationSummary | None:
        return self.conversations.get(conversation_id)

    def list_conversations(self, *, limit: int, offset: int) -> list[ConversationSummary]:
        items = sorted(
            self.conversations.values(),
            key=lambda item: (item.updated_at, item.id),
            reverse=True,
        )
        return items[offset : offset + limit]

    def count_conversations(self) -> int:
        return len(self.conversations)

    def delete_conversation(self, conversation_id: str) -> bool:
        conversation = self.conversations.pop(conversation_id, None)
        if conversation is None:
            return False
        for turn_id in [
            turn.id for turn in self.turns.values() if turn.conversation_id == conversation_id
        ]:
            self.turns.pop(turn_id, None)
            self.turn_fingerprints.pop(turn_id, None)
            self.turn_expiries.pop(turn_id, None)
            self.turn_tokens.pop(turn_id, None)
        return True

    def reserve_conversation_turn(
        self,
        conversation_id: str,
        *,
        client_turn_id: str,
        question: str,
        top_k: int | None,
        document_ids: list[str] | None,
        request_fingerprint: str,
        reservation_seconds: int = 120,
    ) -> ConversationTurnReservation:
        if conversation_id not in self.conversations:
            raise RagError(
                "conversation_not_found",
                "The requested conversation does not exist.",
                status_code=404,
            )
        existing = next(
            (
                turn
                for turn in self.turns.values()
                if turn.conversation_id == conversation_id and turn.client_turn_id == client_turn_id
            ),
            None,
        )
        now = datetime.now(UTC)
        if existing is not None:
            if self.turn_fingerprints[existing.id] != request_fingerprint:
                raise RagError(
                    "idempotency_conflict",
                    "This client turn identifier was already used for a different question.",
                    status_code=409,
                )
            if existing.status is ConversationTurnStatus.COMPLETED or (
                existing.status is ConversationTurnStatus.FAILED and not existing.retryable
            ):
                return ConversationTurnReservation(
                    turn=existing,
                    created=False,
                    cached_turn=existing,
                )
            if existing.status is ConversationTurnStatus.FAILED and existing.retryable:
                reservation_token = uuid4().hex
                recovered = existing.model_copy(
                    update={
                        "status": ConversationTurnStatus.PENDING,
                        "error_code": None,
                        "retryable": False,
                        "updated_at": now,
                    }
                )
                self.turns[existing.id] = recovered
                self.turn_expiries[existing.id] = now + timedelta(seconds=reservation_seconds)
                self.turn_tokens[existing.id] = reservation_token
                return ConversationTurnReservation(
                    turn=recovered,
                    created=True,
                    reservation_token=reservation_token,
                )
            if self.turn_expiries[existing.id] > now:
                raise RagError(
                    "conversation_turn_in_progress",
                    "This question is already being answered.",
                    status_code=409,
                    retryable=True,
                )
            reservation_token = uuid4().hex
            recovered = existing.model_copy(update={"updated_at": now})
            self.turns[existing.id] = recovered
            self.turn_expiries[existing.id] = now + timedelta(seconds=reservation_seconds)
            self.turn_tokens[existing.id] = reservation_token
            return ConversationTurnReservation(
                turn=recovered,
                created=True,
                reservation_token=reservation_token,
            )

        turn = ConversationTurn(
            id=uuid4().hex,
            conversation_id=conversation_id,
            client_turn_id=client_turn_id,
            status=ConversationTurnStatus.PENDING,
            question=question,
            top_k=top_k,
            document_ids=document_ids,
            created_at=now,
            updated_at=now,
        )
        self.turns[turn.id] = turn
        self.turn_fingerprints[turn.id] = request_fingerprint
        self.turn_expiries[turn.id] = now + timedelta(seconds=reservation_seconds)
        reservation_token = uuid4().hex
        self.turn_tokens[turn.id] = reservation_token
        return ConversationTurnReservation(
            turn=turn,
            created=True,
            reservation_token=reservation_token,
        )

    def renew_conversation_turn_reservation(
        self,
        turn_id: str,
        *,
        reservation_token: str,
        reservation_seconds: int = 120,
    ) -> bool:
        turn = self.turns[turn_id]
        if (
            turn.status is not ConversationTurnStatus.PENDING
            or self.turn_tokens.get(turn_id) != reservation_token
        ):
            return False
        self.turn_expiries[turn_id] = datetime.now(UTC) + timedelta(seconds=reservation_seconds)
        return True

    def complete_conversation_turn(
        self,
        turn_id: str,
        *,
        reservation_token: str,
        answer: str,
        citations: list[Citation],
        no_answer: bool,
        request_id: str | None,
    ) -> ConversationTurn:
        existing = self.turns[turn_id]
        if existing.status is ConversationTurnStatus.COMPLETED:
            return existing
        if self.turn_tokens.get(turn_id) != reservation_token:
            raise RagError(
                "conversation_turn_lease_lost",
                "This question is being completed by a newer request.",
                status_code=409,
                retryable=True,
            )
        now = datetime.now(UTC)
        turn = existing.model_copy(
            update={
                "status": ConversationTurnStatus.COMPLETED,
                "answer": answer,
                "citations": citations,
                "no_answer": no_answer,
                "request_id": request_id,
                "updated_at": now,
            }
        )
        self.turns[turn_id] = turn
        self.turn_tokens.pop(turn_id, None)
        conversation = self.conversations[turn.conversation_id]
        title = conversation.title
        if conversation.turn_count == 0 and title == "New conversation":
            title = turn.question if len(turn.question) <= 72 else f"{turn.question[:71]}…"
        self.conversations[turn.conversation_id] = conversation.model_copy(
            update={
                "title": title,
                "turn_count": conversation.turn_count + 1,
                "updated_at": now,
            }
        )
        return turn

    def fail_conversation_turn(
        self,
        turn_id: str,
        *,
        reservation_token: str,
        error_code: str,
        retryable: bool,
    ) -> ConversationTurn:
        existing = self.turns[turn_id]
        if (
            existing.status is not ConversationTurnStatus.PENDING
            or self.turn_tokens.get(turn_id) != reservation_token
        ):
            return existing
        turn = existing.model_copy(
            update={
                "status": ConversationTurnStatus.FAILED,
                "error_code": error_code,
                "retryable": retryable,
                "updated_at": datetime.now(UTC),
            }
        )
        self.turns[turn_id] = turn
        self.turn_tokens.pop(turn_id, None)
        return turn

    def conversation_history(self, conversation_id: str, *, limit: int) -> list[ChatHistoryMessage]:
        completed = [
            turn
            for turn in self.turns.values()
            if turn.conversation_id == conversation_id
            and turn.status is ConversationTurnStatus.COMPLETED
        ]
        messages = [
            message
            for turn in completed
            for message in (
                ChatHistoryMessage(role="user", content=turn.question),
                ChatHistoryMessage(role="assistant", content=turn.answer or ""),
            )
        ]
        return messages[-limit:] if limit else []

    def list_conversation_turns(
        self,
        conversation_id: str,
        *,
        limit: int,
        offset: int,
        include_incomplete: bool = False,
    ) -> list[ConversationTurn]:
        items = [
            turn
            for turn in self.turns.values()
            if turn.conversation_id == conversation_id
            and (include_incomplete or turn.status is ConversationTurnStatus.COMPLETED)
        ]
        return items[offset : offset + limit]

    def count_conversation_turns(
        self, conversation_id: str, *, include_incomplete: bool = False
    ) -> int:
        return len(
            self.list_conversation_turns(
                conversation_id,
                limit=10_000,
                offset=0,
                include_incomplete=include_incomplete,
            )
        )

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
    def __init__(self) -> None:
        self.call_count = 0
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.call_count += 1
        self.requests.append(request)
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


def _store_document(
    repository: FakeRepository,
    *,
    document_id: str,
    display_name: str,
    status: DocumentStatus = DocumentStatus.READY,
    created_at: datetime,
    updated_at: datetime,
) -> None:
    extension = Path(display_name).suffix.lower() or ".txt"
    repository.documents[document_id] = DocumentRecord(
        id=document_id,
        display_name=display_name,
        stored_path=f"{document_id}{extension}",
        content_type="text/markdown" if extension == ".md" else "text/plain",
        extension=extension,
        content_sha256=hashlib.sha256(document_id.encode()).hexdigest(),
        size_bytes=128,
        status=status,
        embedding_fingerprint="a" * 64,
        active_version=1 if status is DocumentStatus.READY else 0,
        chunk_count=1 if status is DocumentStatus.READY else 0,
        created_at=created_at,
        updated_at=updated_at,
        deleted_at=updated_at if status is DocumentStatus.DELETED else None,
    )


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
    conversation = client.get("/api/v1/conversations")
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert conversation.status_code == 401
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


def test_upload_accepts_long_markdown_suffix_with_canonical_content_type(
    client: TestClient,
    auth_headers: dict[str, str],
    repository: FakeRepository,
    settings: Settings,
) -> None:
    response = client.post(
        "/api/v1/documents",
        headers=auth_headers,
        files={"file": ("field-notes.markdown", b"# Field notes", "application/octet-stream")},
    )

    assert response.status_code == 202
    payload = response.json()["document"]
    assert payload["display_name"] == "field-notes.markdown"
    assert payload["extension"] == ".markdown"
    assert payload["content_type"] == "text/markdown"
    record = next(iter(repository.documents.values()))
    assert record.content_type == "text/markdown"
    assert (settings.uploads_dir / record.stored_path).suffix == ".markdown"


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


def test_document_list_keeps_default_order_and_single_status_compatibility(
    client: TestClient,
    auth_headers: dict[str, str],
    repository: FakeRepository,
) -> None:
    start = datetime(2026, 7, 18, 8, 0, tzinfo=UTC)
    _store_document(
        repository,
        document_id="doc-old-ready",
        display_name="old-ready.md",
        created_at=start,
        updated_at=start + timedelta(hours=2),
    )
    _store_document(
        repository,
        document_id="doc-new-failed",
        display_name="new-failed.md",
        status=DocumentStatus.FAILED,
        created_at=start + timedelta(days=1),
        updated_at=start + timedelta(days=1),
    )
    _store_document(
        repository,
        document_id="doc-newest-ready",
        display_name="newest-ready.md",
        created_at=start + timedelta(days=2),
        updated_at=start + timedelta(days=2),
    )

    default = client.get("/api/v1/documents", headers=auth_headers)
    ready = client.get(
        "/api/v1/documents",
        params={"status": "ready"},
        headers=auth_headers,
    )

    assert default.status_code == 200
    assert [item["id"] for item in default.json()["items"]] == [
        "doc-newest-ready",
        "doc-new-failed",
        "doc-old-ready",
    ]
    assert default.json()["total"] == 3
    assert {item["id"] for item in ready.json()["items"]} == {
        "doc-old-ready",
        "doc-newest-ready",
    }
    assert ready.json()["total"] == 2


def test_document_list_applies_literal_query_repeated_statuses_and_filtered_total(
    client: TestClient,
    auth_headers: dict[str, str],
    repository: FakeRepository,
) -> None:
    now = datetime(2026, 7, 18, 8, 0, tzinfo=UTC)
    _store_document(
        repository,
        document_id="doc-failed",
        display_name="Plan%_ Résumé.md",
        status=DocumentStatus.FAILED,
        created_at=now,
        updated_at=now,
    )
    _store_document(
        repository,
        document_id="doc-deletion-failed",
        display_name="Plan%_ deletion.txt",
        status=DocumentStatus.DELETION_FAILED,
        created_at=now + timedelta(minutes=1),
        updated_at=now + timedelta(minutes=1),
    )
    _store_document(
        repository,
        document_id="doc-ready",
        display_name="Plan%_ ready.md",
        created_at=now + timedelta(minutes=2),
        updated_at=now + timedelta(minutes=2),
    )
    _store_document(
        repository,
        document_id="doc-deleted",
        display_name="Plan%_ deleted.md",
        status=DocumentStatus.DELETED,
        created_at=now + timedelta(minutes=3),
        updated_at=now + timedelta(minutes=3),
    )

    response = client.get(
        "/api/v1/documents",
        params=[
            ("status", "failed"),
            ("status", "deletion_failed"),
            ("q", " PLAN%_ "),
            ("sort", "name"),
            ("order", "asc"),
            ("limit", "1"),
            ("offset", "0"),
        ],
        headers=auth_headers,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert payload["limit"] == 1
    assert payload["offset"] == 0
    assert [item["id"] for item in payload["items"]] == ["doc-deletion-failed"]


@pytest.mark.parametrize(
    ("sort", "order", "expected_ids"),
    [
        ("created", "asc", ["doc-zulu", "doc-alpha", "doc-middle"]),
        ("created", "desc", ["doc-middle", "doc-alpha", "doc-zulu"]),
        ("updated", "asc", ["doc-alpha", "doc-middle", "doc-zulu"]),
        ("updated", "desc", ["doc-zulu", "doc-middle", "doc-alpha"]),
        ("name", "asc", ["doc-alpha", "doc-middle", "doc-zulu"]),
        ("name", "desc", ["doc-zulu", "doc-middle", "doc-alpha"]),
    ],
)
def test_document_list_supports_each_fixed_sort_and_order(
    client: TestClient,
    auth_headers: dict[str, str],
    repository: FakeRepository,
    sort: str,
    order: str,
    expected_ids: list[str],
) -> None:
    start = datetime(2026, 7, 18, 8, 0, tzinfo=UTC)
    _store_document(
        repository,
        document_id="doc-zulu",
        display_name="Zulu.md",
        created_at=start,
        updated_at=start + timedelta(days=3),
    )
    _store_document(
        repository,
        document_id="doc-alpha",
        display_name="Alpha.md",
        created_at=start + timedelta(days=1),
        updated_at=start + timedelta(days=1),
    )
    _store_document(
        repository,
        document_id="doc-middle",
        display_name="Middle.md",
        created_at=start + timedelta(days=2),
        updated_at=start + timedelta(days=2),
    )

    response = client.get(
        "/api/v1/documents",
        params={"sort": sort, "order": order},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["items"]] == expected_ids


@pytest.mark.parametrize(
    "query",
    [
        "notes\x00private",
        "notes\u200bprivate",
        "\nprivate",
        "\ue000private",
    ],
)
def test_document_list_rejects_control_or_format_queries_without_echoing_them(
    client: TestClient,
    auth_headers: dict[str, str],
    query: str,
) -> None:
    response = client.get(
        "/api/v1/documents",
        params={"q": query},
        headers=auth_headers,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_document_query"
    assert "private" not in response.text


def test_document_list_rejects_overlong_query_with_sanitized_validation(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.get(
        "/api/v1/documents",
        params={"q": "x" * 201},
        headers=auth_headers,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert "x" * 32 not in response.text


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("status", "unknown"), ("sort", "score"), ("order", "sideways")],
)
def test_document_list_rejects_invalid_enums_with_sanitized_validation(
    client: TestClient,
    auth_headers: dict[str, str],
    parameter: str,
    value: str,
) -> None:
    response = client.get(
        "/api/v1/documents",
        params={parameter: value},
        headers=auth_headers,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert value not in response.text


def test_recent_jobs_are_paginated_and_filterable(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    first = client.post(
        "/api/v1/documents",
        headers=auth_headers,
        files={"file": ("first.txt", b"first", "text/plain")},
    ).json()
    client.post(
        "/api/v1/documents",
        headers=auth_headers,
        files={"file": ("second.txt", b"second", "text/plain")},
    )

    page = client.get("/api/v1/jobs?limit=1&offset=0", headers=auth_headers)
    filtered = client.get(
        f"/api/v1/jobs?document_id={first['document']['id']}",
        headers=auth_headers,
    )

    assert page.status_code == 200
    assert page.json()["total"] == 2
    assert len(page.json()["items"]) == 1
    assert filtered.json()["total"] == 1
    assert filtered.json()["items"][0]["document_id"] == first["document"]["id"]


def test_conversation_create_list_get_and_hard_delete(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    created = client.post(
        "/api/v1/conversations",
        headers=auth_headers,
        json={"title": "Atlas notes"},
    )
    conversation_id = created.json()["id"]
    listed = client.get("/api/v1/conversations", headers=auth_headers)
    fetched = client.get(f"/api/v1/conversations/{conversation_id}", headers=auth_headers)
    deleted = client.delete(f"/api/v1/conversations/{conversation_id}", headers=auth_headers)
    missing = client.get(f"/api/v1/conversations/{conversation_id}", headers=auth_headers)

    assert created.status_code == 201
    assert created.json()["title"] == "Atlas notes"
    assert listed.json()["total"] == 1
    assert fetched.json() == created.json()
    assert deleted.status_code == 204
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "conversation_not_found"


def test_completed_turn_is_cached_and_persisted_with_authoritative_citations(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    conversation = client.post("/api/v1/conversations", headers=auth_headers, json={}).json()
    body = {
        "client_turn_id": "client-turn-001",
        "message": "What color was the launch key?",
        "top_k": 4,
        "document_ids": ["doc-1"],
    }

    first = client.post(
        f"/api/v1/conversations/{conversation['id']}/turns",
        headers=auth_headers,
        json=body,
    )
    second = client.post(
        f"/api/v1/conversations/{conversation['id']}/turns",
        headers=auth_headers,
        json=body,
    )
    turns = client.get(
        f"/api/v1/conversations/{conversation['id']}/turns",
        headers=auth_headers,
    )
    service = client.app.state.container.rag_service

    assert first.status_code == 200
    assert first.json() == second.json()
    assert first.json()["status"] == "completed"
    assert first.json()["citations"][0]["chunk_id"] == "chunk-1"
    assert first.json()["request_id"] == first.headers["X-Request-ID"]
    assert turns.json()["total"] == 1
    assert turns.json()["items"] == [first.json()]
    assert service.call_count == 1

    follow_up = client.post(
        f"/api/v1/conversations/{conversation['id']}/turns",
        headers=auth_headers,
        json={
            "client_turn_id": "client-turn-002",
            "message": "What was it used for?",
        },
    )
    assert follow_up.status_code == 200
    assert [(item.role, item.content) for item in service.requests[-1].history] == [
        ("user", "What color was the launch key?"),
        ("assistant", "The launch key was cobalt blue [S1]."),
    ]


def test_active_turn_conflicts_and_expired_reservation_recovers(
    client: TestClient,
    auth_headers: dict[str, str],
    repository: FakeRepository,
) -> None:
    conversation = repository.create_conversation("Atlas")
    body = {
        "client_turn_id": "client-turn-001",
        "message": "What is the key?",
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "document_ids": None,
                "message": body["message"],
                "top_k": None,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    reserved = repository.reserve_conversation_turn(
        conversation.id,
        client_turn_id=body["client_turn_id"],
        question=body["message"],
        top_k=None,
        document_ids=None,
        request_fingerprint=fingerprint,
    )

    active = client.post(
        f"/api/v1/conversations/{conversation.id}/turns",
        headers=auth_headers,
        json=body,
    )
    assert active.status_code == 409
    assert active.json()["error"]["code"] == "conversation_turn_in_progress"
    assert active.json()["error"]["retryable"] is True

    repository.turn_expiries[reserved.turn.id] = datetime.now(UTC) - timedelta(seconds=1)
    recovered = client.post(
        f"/api/v1/conversations/{conversation.id}/turns",
        headers=auth_headers,
        json=body,
    )
    assert recovered.status_code == 200
    assert recovered.json()["id"] == reserved.turn.id
    assert client.app.state.container.rag_service.call_count == 1


def test_conversation_provider_failure_persists_only_safe_metadata(
    client: TestClient,
    auth_headers: dict[str, str],
    repository: FakeRepository,
) -> None:
    conversation = repository.create_conversation("Atlas")
    client.app.state.container.rag_service = FailingRagService(retryable=True)

    response = client.post(
        f"/api/v1/conversations/{conversation.id}/turns",
        headers=auth_headers,
        json={
            "client_turn_id": "client-turn-001",
            "message": "What is the key?",
        },
    )
    persisted = next(iter(repository.turns.values()))

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "vector_store_unavailable"
    assert persisted.status is ConversationTurnStatus.FAILED
    assert persisted.error_code == "vector_store_unavailable"
    assert persisted.retryable is True
    assert "private Qdrant response" not in response.text
    refreshed = client.get(
        f"/api/v1/conversations/{conversation.id}/turns",
        headers=auth_headers,
    )
    assert refreshed.json()["total"] == 1
    assert refreshed.json()["items"][0]["status"] == "failed"
    assert refreshed.json()["items"][0]["question"] == "What is the key?"

    client.app.state.container.rag_service = FakeRagService()
    retried = client.post(
        f"/api/v1/conversations/{conversation.id}/turns",
        headers=auth_headers,
        json={
            "client_turn_id": "client-turn-001",
            "message": "What is the key?",
        },
    )
    assert retried.status_code == 200
    assert retried.json()["id"] == persisted.id
    assert retried.json()["status"] == "completed"


def test_nonretryable_duplicate_returns_cached_failed_turn_without_provider_call(
    client: TestClient,
    auth_headers: dict[str, str],
    repository: FakeRepository,
) -> None:
    conversation = repository.create_conversation("Atlas")
    body = {
        "client_turn_id": "client-turn-001",
        "message": "What is the key?",
    }
    client.app.state.container.rag_service = FailingRagService(retryable=False)
    failed = client.post(
        f"/api/v1/conversations/{conversation.id}/turns",
        headers=auth_headers,
        json=body,
    )
    replacement = FakeRagService()
    client.app.state.container.rag_service = replacement

    duplicate = client.post(
        f"/api/v1/conversations/{conversation.id}/turns",
        headers=auth_headers,
        json=body,
    )

    assert failed.status_code == 503
    assert duplicate.status_code == 200
    assert duplicate.json()["status"] == "failed"
    assert duplicate.json()["error_code"] == "vector_query_rejected"
    assert duplicate.json()["retryable"] is False
    assert replacement.call_count == 0


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
