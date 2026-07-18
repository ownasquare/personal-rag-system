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
