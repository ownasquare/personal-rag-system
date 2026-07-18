"""Contract tests for the server-side FastAPI client."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest

from personal_rag.models import ChatRequest, ConversationTurnCreate, JobStatus
from personal_rag.ui.client import ApiClientError, RagApiClient

NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC).isoformat()


def _document_payload() -> dict[str, object]:
    return {
        "id": "doc-1",
        "display_name": "notes.md",
        "content_type": "text/markdown",
        "extension": ".md",
        "size_bytes": 100,
        "status": "ready",
        "active_version": 1,
        "chunk_count": 2,
        "error_code": None,
        "error_message": None,
        "created_at": NOW,
        "updated_at": NOW,
    }


def _job_payload(status: str = "queued") -> dict[str, object]:
    terminal = status in {"succeeded", "failed"}
    return {
        "id": "job-1",
        "document_id": "doc-1",
        "kind": "ingest",
        "status": status,
        "stage": "complete" if status == "succeeded" else "queued",
        "progress": 1.0 if status == "succeeded" else 0.0,
        "attempts": 1,
        "max_attempts": 3,
        "lease_owner": None,
        "lease_expires_at": None,
        "error_code": None,
        "error_message": None,
        "created_at": NOW,
        "updated_at": NOW,
        "finished_at": NOW if terminal else None,
    }


def _conversation_payload() -> dict[str, object]:
    return {
        "id": "conversation-1",
        "title": "Atlas notes",
        "turn_count": 1,
        "created_at": NOW,
        "updated_at": NOW,
    }


def _turn_payload() -> dict[str, object]:
    return {
        "id": "turn-1",
        "conversation_id": "conversation-1",
        "client_turn_id": "client-turn-001",
        "status": "completed",
        "question": "What is the launch key?",
        "answer": "The launch key is cobalt [S1].",
        "citations": [
            {
                "label": "S1",
                "document_id": "doc-1",
                "chunk_id": "doc-1:1",
                "document_name": "notes.md",
                "page_number": None,
                "section": "Launch",
                "snippet": "The key is cobalt.",
                "score": 0.9,
            }
        ],
        "no_answer": False,
        "request_id": "request-1",
        "error_code": None,
        "retryable": False,
        "created_at": NOW,
        "updated_at": NOW,
    }


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> RagApiClient:
    return RagApiClient(
        base_url="http://rag.test",
        api_key="server-secret",
        transport=httpx.MockTransport(handler),
    )


def test_client_sends_bearer_token_and_parses_typed_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer server-secret"
        assert request.url.path == "/api/v1/status"
        return httpx.Response(
            200,
            json={
                "status": "ready",
                "collection": "personal_knowledge",
                "document_count": 1,
                "ready_document_count": 1,
                "chunk_count": 2,
                "queued_job_count": 0,
                "embedding_provider": "openai",
                "embedding_model": "text-embedding-3-large",
                "embedding_dimensions": 3072,
                "chat_provider": "openai",
                "chat_model": "gpt-4.1-mini",
                "dependencies": [{"name": "providers", "status": "ready"}],
                "worker_last_seen_at": NOW,
            },
        )

    with _client(handler) as client:
        status = client.get_status()

    assert status.ready_document_count == 1
    assert status.embedding_model == "text-embedding-3-large"


def test_client_uploads_file_once_and_validates_receipt() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/documents"
        assert request.method == "POST"
        assert b'filename="notes.md"' in request.content
        return httpx.Response(
            202,
            json={
                "document": _document_payload(),
                "job": _job_payload(),
                "duplicate": False,
            },
        )

    with _client(handler) as client:
        receipt = client.upload_document("notes.md", b"knowledge", "text/markdown")

    assert receipt.document.display_name == "notes.md"
    assert receipt.job.id == "job-1"


def test_library_pagination_respects_the_api_page_limit() -> None:
    requested_offsets: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        limit = int(request.url.params["limit"])
        offset = int(request.url.params["offset"])
        requested_offsets.append(offset)
        assert limit == 100
        count = 100 if offset == 0 else 1
        items = []
        for index in range(offset, offset + count):
            document = _document_payload()
            document["id"] = f"doc-{index}"
            document["display_name"] = f"notes-{index}.md"
            items.append(document)
        return httpx.Response(
            200,
            json={"items": items, "total": 101, "limit": limit, "offset": offset},
        )

    with _client(handler) as client:
        documents = client.list_all_documents()

    assert len(documents) == 101
    assert requested_offsets == [0, 100]


def test_client_maps_error_envelope_without_exposing_raw_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            503,
            json={
                "error": {
                    "code": "provider_unavailable",
                    "message": "The answer provider is temporarily unavailable.",
                    "retryable": True,
                    "request_id": "request-7",
                },
                "internal_trace": "must never reach the UI",
            },
        )

    with _client(handler) as client, pytest.raises(ApiClientError) as caught:
        client.get_status()

    assert caught.value.code == "provider_unavailable"
    assert caught.value.retryable is True
    assert caught.value.request_id == "request-7"
    assert "internal_trace" not in str(caught.value)


def test_client_maps_transport_and_invalid_payload_failures_to_safe_errors() -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("socket detail must stay private", request=request)

    with _client(unavailable) as client, pytest.raises(ApiClientError) as transport_error:
        client.health_live()

    assert transport_error.value.code == "api_unavailable"
    assert "socket detail" not in str(transport_error.value)

    with (
        _client(lambda request: httpx.Response(200, json={"unexpected": True})) as client,
        pytest.raises(ApiClientError) as payload_error,
    ):
        client.get_status()

    assert payload_error.value.code == "invalid_response"
    assert payload_error.value.retryable is True


def test_chat_request_and_backend_citations_remain_typed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/chat"
        assert request.method == "POST"
        return httpx.Response(
            200,
            json={
                "answer": "Cobalt [S1].",
                "citations": [
                    {
                        "label": "S1",
                        "document_id": "doc-1",
                        "chunk_id": "doc-1:1",
                        "document_name": "notes.md",
                        "page_number": None,
                        "section": "Launch",
                        "snippet": "The key is cobalt.",
                        "score": 0.9,
                    }
                ],
                "no_answer": False,
                "request_id": "request-1",
            },
        )

    with _client(handler) as client:
        response = client.chat(ChatRequest(message="What is the key?", top_k=3))

    assert response.citations[0].label == "S1"
    assert response.answer == "Cobalt [S1]."


def test_conversation_create_list_get_and_delete_use_exact_contracts() -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        assert request.headers["Authorization"] == "Bearer server-secret"
        if request.method == "POST":
            assert request.url.path == "/api/v1/conversations"
            assert json.loads(request.content) == {"title": "Atlas notes"}
            return httpx.Response(201, json=_conversation_payload())
        if request.url.path == "/api/v1/conversations":
            assert request.method == "GET"
            assert dict(request.url.params) == {"limit": "20", "offset": "10"}
            return httpx.Response(
                200,
                json={
                    "items": [_conversation_payload()],
                    "total": 1,
                    "limit": 20,
                    "offset": 10,
                },
            )
        assert request.url.path == "/api/v1/conversations/conversation-1"
        if request.method == "GET":
            return httpx.Response(200, json=_conversation_payload())
        assert request.method == "DELETE"
        return httpx.Response(204)

    with _client(handler) as client:
        created = client.create_conversation("Atlas notes")
        conversations = client.list_conversations(limit=20, offset=10)
        fetched = client.get_conversation("conversation-1")
        deleted = client.delete_conversation("conversation-1")

    assert created.title == "Atlas notes"
    assert conversations.items[0].id == "conversation-1"
    assert fetched.id == "conversation-1"
    assert deleted is None
    assert requests == [
        ("POST", "/api/v1/conversations"),
        ("GET", "/api/v1/conversations"),
        ("GET", "/api/v1/conversations/conversation-1"),
        ("DELETE", "/api/v1/conversations/conversation-1"),
    ]


def test_conversation_turn_list_and_create_preserve_typed_citations() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer server-secret"
        assert request.url.path == "/api/v1/conversations/conversation-1/turns"
        if request.method == "GET":
            assert dict(request.url.params) == {"limit": "25", "offset": "5"}
            return httpx.Response(
                200,
                json={"items": [_turn_payload()], "total": 1, "limit": 25, "offset": 5},
            )
        assert request.method == "POST"
        assert json.loads(request.content) == {
            "client_turn_id": "client-turn-001",
            "message": "What is the launch key?",
            "top_k": 3,
            "document_ids": ["doc-1"],
        }
        return httpx.Response(201, json=_turn_payload())

    turn_create = ConversationTurnCreate(
        client_turn_id="client-turn-001",
        message="What is the launch key?",
        top_k=3,
        document_ids=["doc-1"],
    )
    with _client(handler) as client:
        turns = client.list_conversation_turns("conversation-1", limit=25, offset=5)
        created = client.create_conversation_turn("conversation-1", turn_create)

    assert turns.items[0].answer == "The launch key is cobalt [S1]."
    assert created.citations[0].document_name == "notes.md"


def test_recent_jobs_use_bounded_filters_and_typed_list_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer server-secret"
        assert request.method == "GET"
        assert request.url.path == "/api/v1/jobs"
        assert dict(request.url.params) == {
            "limit": "40",
            "offset": "2",
            "status": "failed",
            "document_id": "doc-1",
        }
        return httpx.Response(
            200,
            json={"items": [_job_payload("failed")], "total": 1, "limit": 40, "offset": 2},
        )

    with _client(handler) as client:
        jobs = client.list_jobs(
            limit=40,
            offset=2,
            status=JobStatus.FAILED,
            document_id=" doc-1 ",
        )

    assert jobs.items[0].status == JobStatus.FAILED
    assert jobs.total == 1


def test_workspace_client_rejects_invalid_bounds_and_blank_identifiers_without_a_request() -> None:
    def unexpected_request(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"unexpected request: {request.method} {request.url}")

    with _client(unexpected_request) as client:
        with pytest.raises(ValueError, match="limit must be between"):
            client.list_conversations(limit=0)
        with pytest.raises(ValueError, match="offset must not be negative"):
            client.list_conversation_turns("conversation-1", offset=-1)
        with pytest.raises(ValueError, match="identifier must not be empty"):
            client.get_conversation("   ")
        with pytest.raises(ValueError, match="identifier must not be empty"):
            client.list_jobs(document_id="   ")


def test_job_polling_is_bounded_and_returns_terminal_truth() -> None:
    responses = iter([_job_payload("queued"), _job_payload("succeeded")])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/jobs/job-1"
        return httpx.Response(200, json=next(responses))

    clock_values = iter([0.0, 0.0, 0.1])
    sleeps: list[float] = []
    with _client(handler) as client:
        job = client.wait_for_job(
            "job-1",
            timeout_seconds=2.0,
            poll_seconds=0.1,
            clock=lambda: next(clock_values),
            sleeper=sleeps.append,
        )

    assert job.status == JobStatus.SUCCEEDED
    assert sleeps == [0.1]


def test_job_poll_timeout_is_retryable_and_safe() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json=_job_payload("queued"))

    clock_values = iter([0.0, 0.0, 2.0])
    with _client(handler) as client, pytest.raises(ApiClientError) as caught:
        client.wait_for_job(
            "job-1",
            timeout_seconds=1.0,
            poll_seconds=0.1,
            clock=lambda: next(clock_values),
            sleeper=lambda _: None,
        )

    assert caught.value.code == "job_poll_timeout"
    assert caught.value.retryable is True
