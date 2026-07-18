"""Contract checks for the deterministic rendered-browser fixture."""

from __future__ import annotations

import importlib
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from personal_rag.models import DocumentStatus, JobStatus
from tests.browser import fake_api


@pytest.fixture
def client() -> Iterator[TestClient]:
    module = importlib.reload(fake_api)
    with TestClient(module.app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {fake_api.TOKEN}"}


def test_document_library_query_is_literal_filtered_sorted_and_paginated(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    literal = client.get(
        "/api/v1/documents",
        params={"q": "RÉSUMÉ 100%_"},
        headers=auth_headers,
    )
    assert literal.status_code == 200
    assert literal.json()["total"] == 1
    assert literal.json()["items"][0]["display_name"] == "Résumé 100%_plan.md"

    decomposed = client.get(
        "/api/v1/documents",
        params={"q": "RE\u0301SUME\u0301"},
        headers=auth_headers,
    )
    assert decomposed.status_code == 200
    assert decomposed.json()["total"] == 1

    cross_field = client.get(
        "/api/v1/documents",
        params={"q": "md .md"},
        headers=auth_headers,
    )
    assert cross_field.status_code == 200
    assert cross_field.json()["total"] == 0

    filters = [
        ("status", DocumentStatus.FAILED.value),
        ("status", DocumentStatus.DELETION_FAILED.value),
        ("sort", "name"),
        ("order", "asc"),
        ("limit", "3"),
        ("offset", "0"),
    ]
    first = client.get("/api/v1/documents", params=filters, headers=auth_headers)
    assert first.status_code == 200
    payload = first.json()
    assert payload["total"] > len(payload["items"])
    assert [item["display_name"] for item in payload["items"]] == sorted(
        item["display_name"] for item in payload["items"]
    )
    assert {item["status"] for item in payload["items"]} <= {
        DocumentStatus.FAILED.value,
        DocumentStatus.DELETION_FAILED.value,
    }

    second = client.get(
        "/api/v1/documents",
        params=[*filters[:-1], ("offset", "3")],
        headers=auth_headers,
    )
    assert second.status_code == 200
    assert {item["id"] for item in payload["items"]}.isdisjoint(
        item["id"] for item in second.json()["items"]
    )


def test_demo_upload_uses_the_shared_markdown_allowlist(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(
        "/api/v1/documents",
        files={"file": ("field-notes.markdown", b"# Notes\nA useful detail.", "text/markdown")},
        headers=auth_headers,
    )

    assert response.status_code == 202
    assert response.json()["document"]["display_name"] == "field-notes.markdown"
    assert response.json()["document"]["extension"] == ".markdown"


def test_demo_unknown_question_is_honestly_bounded(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    conversation = client.post(
        "/api/v1/conversations",
        json={"title": None},
        headers=auth_headers,
    )
    turn = client.post(
        f"/api/v1/conversations/{conversation.json()['id']}/turns",
        json={
            "client_turn_id": "demo-unknown-question",
            "message": "Summarize the file I just uploaded.",
            "top_k": 5,
            "document_ids": None,
        },
        headers=auth_headers,
    )

    assert turn.status_code == 200
    assert turn.json()["no_answer"] is True
    assert turn.json()["citations"] == []
    assert "fixed sample answers" in turn.json()["answer"]


def test_reindex_progresses_to_ready_across_activity_reads(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    queued = client.post(
        "/api/v1/documents/browser-doc-failed/reindex",
        headers=auth_headers,
    )
    assert queued.status_code == 202
    job_id = queued.json()["id"]

    first_activity = client.get("/api/v1/jobs", headers=auth_headers)
    assert first_activity.status_code == 200
    first_job = next(item for item in first_activity.json()["items"] if item["id"] == job_id)
    assert first_job["status"] == JobStatus.RUNNING

    second_activity = client.get("/api/v1/jobs", headers=auth_headers)
    second_job = next(item for item in second_activity.json()["items"] if item["id"] == job_id)
    assert second_job["status"] == JobStatus.SUCCEEDED
    document = client.get(
        "/api/v1/documents/browser-doc-failed",
        headers=auth_headers,
    )
    assert document.json()["status"] == DocumentStatus.READY


def test_delete_supersedes_queued_reindex_and_purges_cited_history(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    reindex = client.post(
        "/api/v1/documents/browser-doc-1/reindex",
        headers=auth_headers,
    )
    assert reindex.status_code == 202

    deletion = client.delete(
        "/api/v1/documents/browser-doc-1",
        headers=auth_headers,
    )
    assert deletion.status_code == 202
    jobs = client.get("/api/v1/jobs", headers=auth_headers).json()["items"]
    superseded = next(item for item in jobs if item["id"] == reindex.json()["id"])
    assert superseded["status"] == JobStatus.FAILED
    assert superseded["error_code"] == "deletion_requested"

    client.get("/api/v1/jobs", headers=auth_headers)
    assert client.get("/api/v1/documents/browser-doc-1", headers=auth_headers).status_code == 404
    assert (
        client.get(
            "/api/v1/conversations/browser-conversation-1",
            headers=auth_headers,
        ).status_code
        == 404
    )


def test_delete_rejects_a_running_non_delete_job(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    reindex = client.post(
        "/api/v1/documents/browser-doc-1/reindex",
        headers=auth_headers,
    )
    assert reindex.status_code == 202
    client.get("/api/v1/jobs", headers=auth_headers)

    deletion = client.delete(
        "/api/v1/documents/browser-doc-1",
        headers=auth_headers,
    )
    assert deletion.status_code == 409
